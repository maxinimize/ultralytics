# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""FGD and STN joint defense training pipeline, keeping the YOLO detector frozen."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.attacks.attack_bridge import build_attacker, run_attack_on_batch
from ultralytics.attacks.spatial_worst_of_k_stn import SpatialWorstOfKWithMetadata
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.defenses.fgd_stn_pipeline import (
    FGDSTNPipeline,
    PreprocessDetectorWrapper,
    canonicalize_mode,
    transform_batch_bboxes_for_stn,
)
from ultralytics.defenses.hgd_fgd import ResidualDenoiser, load_fgd_denoiser
from ultralytics.defenses.spatial_transformer import (
    SpatialTransformer,
    compute_control_points_loss,
    compute_masked_recon_loss,
    compute_validity_mask,
    load_stn_checkpoint,
)
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.train_adv_adaptive import (
    YOLODatasetAdvAdaptive,
    parse_list_arg,
)
from ultralytics.models.yolo.detect.val_fgd_stn import UnifiedFGDSTNValidator
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import DEFAULT_CFG, LOGGER, YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    unwrap_model,
)


def export_joint_checkpoint(
    joint_checkpoint: str | Path | dict,
    output_dir: str | Path | None = None,
    prefix: str = "",
) -> tuple[Path, Path]:
    """Split and export joint training checkpoint into separate FGD and STN checkpoints."""
    if isinstance(joint_checkpoint, (str, Path)):
        ckpt = torch_load(Path(joint_checkpoint), map_location="cpu")
        if output_dir is None:
            output_dir = Path(joint_checkpoint).parent
    else:
        ckpt = joint_checkpoint
        if output_dir is None:
            output_dir = Path(".")

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # 1. Export standalone FGD checkpoint
    fgd_state = ckpt.get("fgd_state_dict", ckpt.get("denoiser"))
    fgd_ckpt = {
        "epoch": ckpt.get("epoch", 0),
        "denoiser": fgd_state,
        "guidance_model": ckpt.get("guidance_model_reference", ckpt.get("guidance_model")),
        "fgd_config": ckpt.get("joint_training_config", {}).get("fgd", {}),
        "train_args": ckpt.get("train_args", {}),
    }
    fgd_path = out_p / f"{prefix}fgd.pt"
    torch.save(fgd_ckpt, fgd_path)

    # 2. Export standalone STN checkpoint
    stn_state = ckpt.get("stn_state_dict", ckpt.get("stn"))
    stn_ckpt = {
        "epoch": ckpt.get("epoch", 0),
        "stn": stn_state,
        "stn_config": ckpt.get("joint_training_config", {}).get("stn", {}),
        "stn_architecture": ckpt.get("stn_architecture", {}),
        "guidance_model": ckpt.get("guidance_model_reference", ckpt.get("guidance_model")),
        "best_fitness": ckpt.get("best_fitness", 0.0),
        "train_args": ckpt.get("train_args", {}),
    }
    stn_path = out_p / f"{prefix}stn.pt"
    torch.save(stn_ckpt, stn_path)

    return fgd_path, stn_path


class JointFGDSTNModel(nn.Module):
    """Joint training model container wrapping FGD denoiser, STN spatial transformer, and frozen YOLO detector."""

    def __init__(
        self,
        denoiser: nn.Module,
        stn: nn.Module,
        guidance_model: nn.Module,
        preprocess_mode: str = "stn_fgd",
        lambda_fgd: float = 1.0,
        lambda_geom: float = 1.0,
        lambda_recon: float = 1.0,
        lambda_det: float = 0.0,
        lambda_identity: float = 0.1,
    ) -> None:
        """Initialize joint model container."""
        super().__init__()
        self.denoiser = denoiser
        self.stn = stn
        self.guidance_model = guidance_model
        # Keep guidance detector frozen in eval mode
        self.guidance_model.eval().requires_grad_(False)

        self.pipeline = FGDSTNPipeline(denoiser=self.denoiser, stn=self.stn, freeze_modules=False)
        self.preprocess_mode = canonicalize_mode(preprocess_mode)

        self.lambda_fgd = float(lambda_fgd)
        self.lambda_geom = float(lambda_geom)
        self.lambda_recon = float(lambda_recon)
        self.lambda_det = float(lambda_det)
        self.lambda_identity = float(lambda_identity)

        self.stride = getattr(guidance_model, "stride", torch.tensor([32.0]))
        self.names = getattr(guidance_model, "names", {})
        self.criterion = None
        self.pt_path = getattr(guidance_model, "pt_path", "")
        self.yaml = getattr(guidance_model, "yaml", {})

    def set_head_attr(self, **kwargs) -> None:
        """Delegate setting guidance detector head attributes."""
        if hasattr(unwrap_model(self.guidance_model), "set_head_attr"):
            unwrap_model(self.guidance_model).set_head_attr(**kwargs)

    def train(self, mode: bool = True):
        """Set train mode: only updates defense modules, detector always remains in eval mode."""
        super().train(mode)
        self.denoiser.train(mode)
        self.stn.train(mode)
        self.guidance_model.eval()
        return self

    def features(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Extract P3, P4, P5 feature maps from frozen detector."""
        raw = self.guidance_model(images)[1]
        if "one2many" in raw:
            raw = raw["one2many"]
        feats = raw["feats"]
        if len(feats) != 3:
            raise ValueError(f"Detector must provide 3 feature scales P3, P4, P5, got {len(feats)}")
        return feats

    def forward(self, batch: torch.Tensor | dict) -> Any:
        """Forward pass: performs defense inference on tensor input or computes joint loss on batch dict."""
        if torch.is_tensor(batch):
            return self.pipeline(batch, mode=self.preprocess_mode)

        images = batch["img"]
        clean_images = batch["clean_img"]
        gt_matrices = batch["gt_matrix"]
        sample_types = batch.get("sample_type", [])
        b, _, h, w = images.shape
        device = images.device

        # 1. Run defense forward pass, ensuring gradients propagate to FGD and STN
        if self.preprocess_mode == "stn_fgd":
            rect_images, pred_matrices = self.stn(images)
            processed_images = self.denoiser(rect_images)
        elif self.preprocess_mode == "fgd_stn":
            denoised_images = self.denoiser(images)
            processed_images, pred_matrices = self.stn(denoised_images)
        else:
            raise ValueError(f"Unsupported preprocess mode: {self.preprocess_mode}")

        # 2. Build validity mask (exclude cropped/padded invalid regions from geometric attacks)
        masks = []
        for i in range(b):
            st = sample_types[i] if i < len(sample_types) else "raw"
            if st == "worstk":
                m_i = compute_validity_mask(gt_matrices[i : i + 1], pred_matrices[i : i + 1], height=h, width=w)
                masks.append(m_i)
            else:
                masks.append(torch.ones((1, 1, h, w), device=device, dtype=images.dtype))
        mask = torch.cat(masks, dim=0)

        # 3. FGD feature guidance loss (L_fgd)
        if self.lambda_fgd > 0.0:
            with torch.no_grad():
                clean_feats = [x.detach() for x in self.features(clean_images)]
            proc_feats = self.features(processed_images)
            scale_losses = []
            for s in range(3):
                mask_s = F.interpolate(mask, size=proc_feats[s].shape[-2:], mode="nearest")
                diff_s = (proc_feats[s] - clean_feats[s]).abs() * mask_s
                valid_counts = mask_s.sum(dim=[-2, -1]).clamp(min=1.0) * float(proc_feats[s].shape[1])
                scale_loss = (diff_s.sum(dim=[-2, -1]) / valid_counts).mean(dim=1)
                scale_losses.append(scale_loss)
            per_image_fgd = torch.stack(scale_losses, dim=1)
            loss_fgd = per_image_fgd.mean()
        else:
            loss_fgd = torch.tensor(0.0, device=device)

        # 4. STN geometric registration loss (L_geom)
        if self.lambda_geom > 0.0:
            loss_geom = compute_control_points_loss(pred_matrices, gt_matrices, height=h, width=w)
        else:
            loss_geom = torch.tensor(0.0, device=device)

        # 5. Masked image reconstruction loss (L_recon)
        if self.lambda_recon > 0.0:
            loss_recon = compute_masked_recon_loss(processed_images, clean_images, mask)
        else:
            loss_recon = torch.tensor(0.0, device=device)

        # 6. Identity preservation loss (L_identity) on clean samples
        if self.lambda_identity > 0.0:
            clean_indices = [i for i, st in enumerate(sample_types) if st in {"clean", "raw"}]
            if clean_indices:
                idx_t = torch.tensor(clean_indices, device=device)
                loss_recon_clean = (processed_images[idx_t] - clean_images[idx_t]).abs().mean()
                eye_clean = (
                    torch.eye(3, device=device, dtype=pred_matrices.dtype).unsqueeze(0).expand(len(clean_indices), 3, 3)
                )
                loss_geom_clean = compute_control_points_loss(pred_matrices[idx_t], eye_clean, height=h, width=w)
                loss_identity = loss_recon_clean + loss_geom_clean
            else:
                loss_identity = torch.tensor(0.0, device=device)
        else:
            loss_identity = torch.tensor(0.0, device=device)

        # 7. Optional detector loss (L_det)
        if self.lambda_det > 0.0:
            batch_det = {k: v.clone() if torch.is_tensor(v) else copy.copy(v) for k, v in batch.items()}
            batch_det["img"] = processed_images
            batch_det = transform_batch_bboxes_for_stn(batch_det, pred_matrices, height=h, width=w)
            det_preds = self.guidance_model(processed_images, augment=False)
            det_loss_items, _ = unwrap_model(self.guidance_model).loss(batch_det, det_preds)
            loss_det = det_loss_items.sum()
        else:
            loss_det = torch.tensor(0.0, device=device)

        # Weighted joint objective
        total_loss = (
            self.lambda_fgd * loss_fgd
            + self.lambda_geom * loss_geom
            + self.lambda_recon * loss_recon
            + self.lambda_det * loss_det
            + self.lambda_identity * loss_identity
        )

        loss_items = {
            "total_loss": total_loss.detach(),
            "fgd_loss": loss_fgd.detach(),
            "geom_loss": loss_geom.detach(),
            "recon_loss": loss_recon.detach(),
            "det_loss": loss_det.detach(),
            "identity_loss": loss_identity.detach(),
        }
        return total_loss, loss_items


class JointFGDSTNTrainer(DetectionTrainer):
    """Joint FGD and STN optimizer trainer, keeping guidance detector frozen."""

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict | None = None,
        _callbacks: dict | None = None,
    ) -> None:
        """Initialize joint defense trainer and parse research hyperparameters."""
        overrides = dict(overrides or {})
        defaults = {
            "preprocess_mode": "stn_fgd",
            "init": "pretrained",
            "fgd_weights": None,
            "stn_weights": None,
            "attack_weights": None,
            "attack_through_preprocess": True,
            "attack_name": ["pgd", "bim", "mim", "worstk"],
            "attack_ratio": [0.10, 0.10, 0.40, 0.15],
            "p_clean": 0.25,
            "worstk_k": 10,
            "degrees": 30.0,
            "translate": 0.10,
            "lambda_fgd": 1.0,
            "lambda_geom": 1.0,
            "lambda_recon": 1.0,
            "lambda_det": 0.0,
            "lambda_identity": 0.1,
            "lr_fgd": 1e-5,
            "lr_stn": 1e-4,
            "val_interval": 5,
            "eval_weights": None,
        }
        self.joint_cfg = {key: overrides.pop(key, value) for key, value in defaults.items()}

        baseline = {
            "task": "detect",
            "optimizer": "Adam",
            "weight_decay": 1e-5,
            "warmup_epochs": 0.0,
            "project": "runs/train_fgd_stn_joint",
            "name": "exp",
            "val": True,
            "plots": False,
        }
        super().__init__(cfg, {**baseline, **overrides}, _callbacks)

        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError(f"Joint training only supports CPU or single CUDA device, got {self.device.type}")

        # Parse and normalize configuration
        if not self.resume:
            self.preprocess_mode = canonicalize_mode(self.joint_cfg["preprocess_mode"])
            self.init_mode = str(self.joint_cfg["init"]).strip().lower()
            self.fgd_weights = self.joint_cfg["fgd_weights"]
            self.stn_weights = self.joint_cfg["stn_weights"]
            self.attack_through_preprocess = bool(self.joint_cfg["attack_through_preprocess"])
            self.attack_weights = self.joint_cfg["attack_weights"]
            self.attack_names = parse_list_arg(self.joint_cfg["attack_name"])
            ratios = self.joint_cfg["attack_ratio"]
            self.attack_ratios = [float(x) for x in parse_list_arg(ratios, float)] if ratios is not None else []
            self.p_clean = float(self.joint_cfg["p_clean"])
            self.worstk_k = int(self.joint_cfg["worstk_k"])
            self.degrees = float(self.joint_cfg["degrees"])
            self.translate = float(self.joint_cfg["translate"])
            self.lambda_fgd = float(self.joint_cfg["lambda_fgd"])
            self.lambda_geom = float(self.joint_cfg["lambda_geom"])
            self.lambda_recon = float(self.joint_cfg["lambda_recon"])
            self.lambda_det = float(self.joint_cfg["lambda_det"])
            self.lambda_identity = float(self.joint_cfg["lambda_identity"])
            self.lr_fgd = float(self.joint_cfg["lr_fgd"])
            self.lr_stn = float(self.joint_cfg["lr_stn"])
            self.val_interval = int(self.joint_cfg["val_interval"])
        else:
            j_cfg = self.joint_checkpoint.get("joint_training_config", {})
            self.preprocess_mode = canonicalize_mode(j_cfg.get("preprocess_mode", "stn_fgd"))
            self.init_mode = str(j_cfg.get("init", "pretrained")).strip().lower()
            self.fgd_weights = j_cfg.get("fgd_weights")
            self.stn_weights = j_cfg.get("stn_weights")
            atk_cfg = self.joint_checkpoint.get("attack_configuration", {})
            self.attack_through_preprocess = bool(atk_cfg.get("attack_through_preprocess", True))
            self.attack_weights = atk_cfg.get("attack_weights")
            self.attack_names = atk_cfg.get("attack_name", ["pgd", "bim", "mim", "worstk"])
            self.attack_ratios = [float(x) for x in atk_cfg.get("attack_ratio", [0.10, 0.10, 0.40, 0.15])]
            self.p_clean = float(atk_cfg.get("p_clean", 0.25))
            stn_c = j_cfg.get("stn", {})
            self.worstk_k = int(stn_c.get("worstk_k", 10))
            self.degrees = float(stn_c.get("degrees", 30.0))
            self.translate = float(stn_c.get("translate", 0.10))
            self.lambda_fgd = float(j_cfg.get("lambda_fgd", 1.0))
            self.lambda_geom = float(j_cfg.get("lambda_geom", 1.0))
            self.lambda_recon = float(j_cfg.get("lambda_recon", 1.0))
            self.lambda_det = float(j_cfg.get("lambda_det", 0.0))
            self.lambda_identity = float(j_cfg.get("lambda_identity", 0.1))
            self.lr_fgd = float(j_cfg.get("lr_fgd", 1e-5))
            self.lr_stn = float(j_cfg.get("lr_stn", 1e-4))
            self.val_interval = int(j_cfg.get("val_interval", 5))

        if self.preprocess_mode not in {"stn_fgd", "fgd_stn"}:
            raise ValueError(f"Joint training only supports 'stn_fgd' or 'fgd_stn', got: {self.preprocess_mode}")

        if self.init_mode not in {"pretrained", "scratch"}:
            raise ValueError(f"--init must be 'pretrained' or 'scratch', got: {self.init_mode}")

        if not self.resume and self.init_mode == "pretrained":
            if not self.fgd_weights or not Path(self.fgd_weights).is_file():
                raise FileNotFoundError(f"Pretrained fine-tuning mode requires valid --fgd_weights path, got: {self.fgd_weights}")
            if not self.stn_weights or not Path(self.stn_weights).is_file():
                raise FileNotFoundError(f"Pretrained fine-tuning mode requires valid --stn_weights path, got: {self.stn_weights}")

        total_ratio = sum(self.attack_ratios) + self.p_clean
        if abs(total_ratio - 1.0) > 1e-3:
            raise ValueError(
                f"Sum of attack ratios plus clean ratio p_clean must equal 1.0, got: sum(ratios)={sum(self.attack_ratios):.4f}, "
                f"p_clean={self.p_clean:.4f}, total={total_ratio:.4f}"
            )

        self.worstk_k = int(self.joint_cfg["worstk_k"])
        self.degrees = float(self.joint_cfg["degrees"])
        self.translate = float(self.joint_cfg["translate"])

        self.lambda_fgd = float(self.joint_cfg["lambda_fgd"])
        self.lambda_geom = float(self.joint_cfg["lambda_geom"])
        self.lambda_recon = float(self.joint_cfg["lambda_recon"])
        self.lambda_det = float(self.joint_cfg["lambda_det"])
        self.lambda_identity = float(self.joint_cfg["lambda_identity"])

        self.lr_fgd = float(self.joint_cfg["lr_fgd"])
        self.lr_stn = float(self.joint_cfg["lr_stn"])
        self.val_interval = int(self.joint_cfg["val_interval"])

        # Validation evaluation weights (default: equal weights across 5 scenarios, or custom weights)
        default_eval = [0.20, 0.20, 0.20, 0.20, 0.20]
        raw_ew = self.joint_cfg.get("eval_weights")
        if raw_ew is not None:
            self.eval_weights = [float(x) for x in parse_list_arg(raw_ew, float)]
        else:
            self.eval_weights = default_eval

        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        YAML.save(self.save_dir / "joint_config.yaml", self.joint_cfg)

    def check_resume(self, overrides: dict) -> None:
        """Resume optimizer, scheduler, and defense hyperparameters from joint checkpoint."""
        self.resume = bool(self.args.resume)
        self.joint_checkpoint = None
        if self.resume:
            if not isinstance(self.args.resume, (str, Path)):
                raise ValueError("Joint training resume requires a valid checkpoint path (e.g. weights/last.pt)")
            resume_path = Path(self.args.resume)
            if not resume_path.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

            self.joint_checkpoint = torch_load(resume_path, map_location="cpu")
            saved = self.joint_checkpoint.get("train_args", {}).copy()
            saved.update({k: overrides[k] for k in ("device", "workers", "epochs", "batch") if k in overrides})
            saved.update(resume=str(resume_path), exist_ok=True)
            self.args = get_cfg(DEFAULT_CFG, saved)

    def _setup_train(self) -> None:
        """Prepare frozen guidance model, initialize defense modules, build attackers and optimizer."""
        # 1. Load guidance YOLO detector and freeze completely
        self.guidance_model, _ = load_checkpoint(self.args.model, device=self.device)
        self.args.model = str(Path(self.guidance_model.pt_path).resolve())
        self.guidance_model.eval().requires_grad_(False)
        if hasattr(self.guidance_model, "args") and isinstance(self.guidance_model.args, dict):
            self.guidance_model.args = IterableSimpleNamespace(**self.guidance_model.args)

        if self.guidance_model.names != self.data["names"]:
            raise ValueError("Guidance model class names do not match dataset class names; please use matching weights.")

        self.stride = max(int(self.guidance_model.stride.max()), 32)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=self.stride, floor=self.stride, max_dim=1)

        # 2. Load or initialize FGD and STN according to init_mode
        if self.init_mode == "pretrained" and self.joint_checkpoint is None:
            LOGGER.info(f"Loading FGD ({self.fgd_weights}) and STN ({self.stn_weights}) in pretrained mode...")
            self.denoiser, _fgd_meta = load_fgd_denoiser(self.fgd_weights, device=self.device)
            self.stn, _stn_meta = load_stn_checkpoint(self.stn_weights, device=self.device)
        else:
            LOGGER.info("Initializing FGD and STN from scratch...")
            self.denoiser = ResidualDenoiser().to(self.device)
            self.stn = SpatialTransformer(
                max_degrees=self.degrees * 1.5,
                max_translate=self.translate * 1.5,
                fill=114.0 / 255.0,
            ).to(self.device)

        # Activate training state and gradient computation for defense modules
        self.denoiser.train().requires_grad_(True)
        self.stn.train().requires_grad_(True)

        # 3. Construct joint model container
        self.joint_model = JointFGDSTNModel(
            denoiser=self.denoiser,
            stn=self.stn,
            guidance_model=self.guidance_model,
            preprocess_mode=self.preprocess_mode,
            lambda_fgd=self.lambda_fgd,
            lambda_geom=self.lambda_geom,
            lambda_recon=self.lambda_recon,
            lambda_det=self.lambda_det,
            lambda_identity=self.lambda_identity,
        )
        self.model = self.joint_model

        # 4. Setup attack generation model and online attackers
        attack_weights = self.attack_weights
        if attack_weights is None or str(attack_weights).strip().lower() in {"", "none", "current"}:
            self.attack_model = copy.deepcopy(unwrap_model(self.guidance_model)).eval()
            self.joint_cfg["attack_weights"] = self.args.model
        else:
            self.attack_model, _ = load_checkpoint(attack_weights, device=self.device)
            self.joint_cfg["attack_weights"] = str(Path(self.attack_model.pt_path).resolve())
            self.attack_model.eval()

        for p in self.attack_model.parameters():
            p.requires_grad_(True)

        if self.attack_through_preprocess:
            proxy_model = PreprocessDetectorWrapper(
                pipeline=self.joint_model.pipeline,
                detector=self.attack_model,
                mode=self.preprocess_mode,
            )
        else:
            proxy_model = self.attack_model

        self.attackers = {}
        for name in self.attack_names:
            if name == "worstk":
                self.attackers["worstk"] = SpatialWorstOfKWithMetadata(
                    model=proxy_model,
                    k=self.worstk_k,
                    degrees=self.degrees,
                    translate=self.translate,
                    img_size=self.args.imgsz,
                    seed=int(self.args.seed) if self.args.seed is not None else None,
                )
            elif name in {"pgd", "bim", "mim"}:
                self.attackers[name] = build_attacker(
                    name,
                    model=proxy_model,
                    img_size=self.args.imgsz,
                )

        # 5. Build data pipeline and dataloader
        self.args.nbs = self.batch_size
        self.denoiser.train().requires_grad_(True)
        self.stn.train().requires_grad_(True)
        self._build_train_pipeline()

        # 6. Setup training environment
        self.freeze_layer_names = []
        self.amp = bool(self.args.amp not in {False, "fp32", "False"})
        if self.device.type == "cpu":
            self.scaler = torch.amp.GradScaler("cpu", enabled=False)
        else:
            self.scaler = (
                torch.amp.GradScaler(
                    self.device.type if self.device.type == "xpu" else "cuda",
                    enabled=self.amp and self.args.amp != "bf16",
                )
                if TORCH_2_4
                else torch.cuda.amp.GradScaler(enabled=self.amp and self.args.amp != "bf16")
            )

        self.ema = ModelEMA(self.model)
        self.ema.enabled = False
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False

        # 7. Validator configuration
        self.validator = self.get_validator()
        self.loss_names = ("total_loss", "fgd_loss", "geom_loss", "recon_loss", "det_loss", "identity_loss")
        self.metrics = {}
        self.best_fitness = 0.0

        # 8. Load checkpoint state if resuming
        if self.joint_checkpoint is not None:
            ckpt = self.joint_checkpoint
            fgd_st = ckpt.get("fgd_state_dict", ckpt.get("denoiser"))
            if fgd_st:
                self.denoiser.load_state_dict(fgd_st)
            stn_st = ckpt.get("stn_state_dict", ckpt.get("stn"))
            if stn_st:
                self.stn.load_state_dict(stn_st)

            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            elif "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])

            if "scheduler_state_dict" in ckpt and hasattr(self, "scheduler") and self.scheduler:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

            if "scaler_state_dict" in ckpt and hasattr(self, "scaler") and self.scaler:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])

            if "random_state" in ckpt:
                rnd = ckpt["random_state"]
                if "torch" in rnd:
                    torch.set_rng_state(rnd["torch"])
                if torch.cuda.is_available() and rnd.get("cuda") is not None:
                    torch.cuda.set_rng_state_all(rnd["cuda"])
                if "numpy" in rnd:
                    np.random.set_state(rnd["numpy"])
                if "python" in rnd:
                    random.setstate(rnd["python"])

            self.start_epoch = ckpt["epoch"] + 1
            self.best_fitness = float(ckpt.get("best_fitness", 0.0))
            if self.start_epoch >= self.epochs:
                raise ValueError(f"--epochs ({self.epochs}) must be greater than completed checkpoint epochs ({self.start_epoch}).")
            self.joint_checkpoint = None

        if hasattr(self, "scheduler") and hasattr(self.scheduler, "last_epoch"):
            self.scheduler.last_epoch = self.start_epoch - 1

        # Ensure defense modules remain trainable after resuming
        self.denoiser.train().requires_grad_(True)
        self.stn.train().requires_grad_(True)
        self.guidance_model.eval().requires_grad_(False)

        YAML.save(self.save_dir / "args.yaml", vars(self.args))
        YAML.save(self.save_dir / "joint_config.yaml", self.joint_cfg)
        LOGGER.info(
            f"FGD+STN joint training pipeline initialized (mode: {self.preprocess_mode}, init: {self.init_mode}, "
            f"LR_FGD: {self.lr_fgd}, LR_STN: {self.lr_stn})"
        )
        self.run_callbacks("on_pretrain_routine_end")

    def _build_train_pipeline(self) -> None:
        """Build data loader pipeline and configure guidance model detector attributes."""
        self.args.nbs = self.batch_size
        BaseTrainer._build_train_pipeline(self)
        if hasattr(unwrap_model(self.guidance_model), "set_head_attr"):
            datasets = {"train": self.train_loader.dataset, "val": self.test_loader.dataset}
            from ultralytics.models.yolo.detect.val import DetectionValidator

            DetectionValidator._check_max_det(self.args, datasets)
            unwrap_model(self.guidance_model).set_head_attr(max_det=self.args.max_det)

    def set_class_weights(self) -> None:
        """Joint defense optimization does not require updating class weights."""

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build dataset with fixed training attack distribution."""
        gs = max(int(unwrap_model(self.guidance_model).stride.max()), 32)
        if mode != "train":
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=True, stride=gs)

        return YOLODatasetAdvAdaptive(
            cfg=self.args,
            img_path=img_path,
            batch=batch,
            data=self.data,
            mode=mode,
            rect=False,
            stride=gs,
            attack_names=self.attack_names,
            attack_ratios=self.attack_ratios,
            p_clean=self.p_clean,
        )

    def build_optimizer(self, model: nn.Module, **kwargs) -> torch.optim.Optimizer:
        """Build parameter group optimizer: separately update FGD and STN parameters without guidance model."""
        self.denoiser.requires_grad_(True)
        self.stn.requires_grad_(True)
        fgd_params = [p for p in self.denoiser.parameters() if p.requires_grad]
        stn_params = [p for p in self.stn.parameters() if p.requires_grad]
        if not fgd_params:
            raise RuntimeError("FGD denoiser parameter list is empty! Ensure defense modules are not frozen.")
        if not stn_params:
            raise RuntimeError("STN parameter list is empty! Ensure defense modules are not frozen.")
        param_groups = [
            {
                "params": fgd_params,
                "lr": float(self.lr_fgd),
                "weight_decay": float(self.args.weight_decay),
                "name": "fgd",
            },
            {
                "params": stn_params,
                "lr": float(self.lr_stn),
                "weight_decay": float(self.args.weight_decay),
                "name": "stn",
            },
        ]
        LOGGER.info(
            f"Joint optimizer initialized: FGD={len(fgd_params)} params (lr={self.lr_fgd}), "
            f"STN={len(stn_params)} params (lr={self.lr_stn})"
        )
        return torch.optim.Adam(param_groups)

    def preprocess_batch(self, batch: dict) -> dict:
        """Generate adversarial samples online while preserving clean samples and ground-truth geometric matrices."""
        batch = super().preprocess_batch(batch)
        b = batch["img"].shape[0]
        device = batch["img"].device

        # Preserve original clean images and clean bounding boxes
        batch["clean_img"] = batch["img"].clone()
        batch["clean_batch_idx"] = batch["batch_idx"].clone() if "batch_idx" in batch else None
        batch["clean_cls"] = batch["cls"].clone() if "cls" in batch else None
        batch["clean_bboxes"] = batch["bboxes"].clone() if "bboxes" in batch else None

        sample_types = list(batch.get("sample_type", ["raw"] * b))
        batch["sample_type"] = sample_types

        # Default ground-truth spatial transformation matrix is identity
        gt_matrices = torch.eye(3, device=device, dtype=batch["img"].dtype).unsqueeze(0).repeat(b, 1, 1)

        # Split sub-batches and generate targeted adversarial samples
        sub_batches = self.split_batch_by_type(batch)

        # Suspend defense parameter gradients during attack generation
        den_states = [p.requires_grad for p in self.denoiser.parameters()]
        stn_states = [p.requires_grad for p in self.stn.parameters()]
        for p in self.denoiser.parameters():
            p.requires_grad = False
        for p in self.stn.parameters():
            p.requires_grad = False

        try:
            for att_name, attacker in self.attackers.items():
                if att_name not in sub_batches:
                    continue
                sub_b = sub_batches[att_name]
                indices = [i for i, st in enumerate(sample_types) if st == att_name]
                if not indices:
                    continue

                if att_name == "worstk" and callable(getattr(attacker, "forward_batch", None)):
                    attacker.forward_batch(sub_b)
                    adv_imgs = sub_b["img"]
                    if "gt_matrix" in sub_b:
                        gt_matrices[indices] = sub_b["gt_matrix"].to(device=device, dtype=gt_matrices.dtype)
                else:
                    adv_imgs = run_attack_on_batch(attacker, sub_b, label_policy="largest_box")

                if adv_imgs is not None:
                    sub_b["img"] = adv_imgs.detach()
                    batch["img"][indices] = adv_imgs.detach()
        finally:
            # Restore defense module parameter gradients for joint loss backpropagation
            for p, state in zip(self.denoiser.parameters(), den_states):
                p.requires_grad = state
            for p, state in zip(self.stn.parameters(), stn_states):
                p.requires_grad = state

        # Recombine bounding boxes including Worst-of-K transforms
        if "worstk" in sub_batches and "worstk" in self.attackers:
            new_batch_idx = []
            new_cls = []
            new_bboxes = []
            orig_bi_dim = batch["batch_idx"].dim() if "batch_idx" in batch and batch["batch_idx"] is not None else 1
            orig_cls_dim = batch["cls"].dim() if "cls" in batch and batch["cls"] is not None else 1

            for t, sub_b in sub_batches.items():
                indices = [idx for idx, st in enumerate(sample_types) if st == t]
                if not indices:
                    continue
                sub_bi = sub_b.get("batch_idx")
                if sub_bi is not None and sub_bi.numel() > 0:
                    sub_bi = sub_bi.view(-1).long()
                    indices_tensor = torch.tensor(indices, device=sub_bi.device, dtype=torch.long)
                    global_bi = indices_tensor[sub_bi]
                    new_batch_idx.append(global_bi)
                    new_cls.append(sub_b["cls"].view(-1, 1) if sub_b["cls"].ndim == 1 else sub_b["cls"])
                    new_bboxes.append(sub_b["bboxes"].view(-1, 4) if sub_b["bboxes"].ndim != 2 else sub_b["bboxes"])

            if new_batch_idx:
                cat_bi = torch.cat(new_batch_idx, dim=0)
                cat_cls = torch.cat(new_cls, dim=0)
                cat_box = torch.cat(new_bboxes, dim=0)

                sort_idx = torch.argsort(cat_bi.view(-1))
                cat_bi = cat_bi[sort_idx]
                cat_cls = cat_cls[sort_idx]
                cat_box = cat_box[sort_idx]

                if orig_bi_dim > 1 and cat_bi.dim() == 1:
                    cat_bi = cat_bi.unsqueeze(-1)
                batch["batch_idx"] = cat_bi

                if orig_cls_dim == 1 and cat_cls.dim() > 1:
                    cat_cls = cat_cls.squeeze(-1)
                elif orig_cls_dim > 1 and cat_cls.dim() == 1:
                    cat_cls = cat_cls.unsqueeze(-1)
                batch["cls"] = cat_cls
                batch["bboxes"] = cat_box
            else:
                batch["batch_idx"] = torch.zeros(0, dtype=torch.long, device=device)
                batch["cls"] = torch.zeros((0, 1) if orig_cls_dim > 1 else (0,), dtype=torch.float32, device=device)
                batch["bboxes"] = torch.zeros((0, 4), dtype=torch.float32, device=device)

        batch["gt_matrix"] = gt_matrices
        # Ensure adversarial images are detached from previous computation graph
        batch["img"] = batch["img"].detach()
        return batch

    def split_batch_by_type(self, batch: dict) -> dict[str, dict]:
        """Split full batch into sub-batches according to sample_type field."""
        sample_types = batch["sample_type"]
        unique_types = set(sample_types)
        sub_batches = {}

        for t in unique_types:
            indices = [i for i, st in enumerate(sample_types) if st == t]
            sub_b = {}
            sub_b["img"] = batch["img"][indices]

            for k in ["im_file", "sample_type", "ori_shape", "ratio_pad"]:
                if k in batch:
                    sub_b[k] = [batch[k][i] for i in indices]

            if "batch_idx" in batch and batch["batch_idx"] is not None:
                b_idx = batch["batch_idx"].view(-1)
                mask = torch.zeros_like(b_idx, dtype=torch.bool)
                for idx in indices:
                    mask |= b_idx == idx

                sub_b["cls"] = batch["cls"][mask]
                sub_b["bboxes"] = batch["bboxes"][mask]

                idx_map = {old_i: new_i for new_i, old_i in enumerate(indices)}
                old_bi = b_idx[mask]
                new_bi = torch.tensor(
                    [idx_map[int(x)] for x in old_bi],
                    dtype=old_bi.dtype,
                    device=old_bi.device,
                )
                sub_b["batch_idx"] = new_bi

            sub_batches[t] = sub_b

        return sub_batches

    def optimizer_step(self) -> None:
        """Check that defense module gradients are finite before optimizer step."""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise FloatingPointError(f"Non-finite gradients detected in optimizer group [{group.get('name')}].")
        super().optimizer_step()

    def get_validator(self) -> UnifiedFGDSTNValidator:
        """Configure joint validator covering clean and 4 adversarial attacks."""
        val_args = copy.copy(self.args)
        validator = UnifiedFGDSTNValidator(
            dataloader=self.test_loader,
            save_dir=self.save_dir,
            args=val_args,
            _callbacks=self.callbacks,
        )
        validator.device = self.device
        validator.data = self.data
        validator.pipeline = self.joint_model.pipeline.to(self.device)
        validator.guidance_model = self.guidance_model.to(self.device)

        val_attack_model = copy.deepcopy(unwrap_model(self.guidance_model)).to(self.device).eval()
        for p in val_attack_model.parameters():
            p.requires_grad_(True)

        validator.attackers = {
            "worstk": SpatialWorstOfKWithMetadata(
                model=val_attack_model,
                k=self.worstk_k,
                degrees=self.degrees,
                translate=self.translate,
                img_size=self.args.imgsz,
                seed=int(getattr(self.args, "seed", 0)),
            ),
            "pgd": build_attacker("pgd", model=val_attack_model, img_size=self.args.imgsz),
            "bim": build_attacker("bim", model=val_attack_model, img_size=self.args.imgsz),
            "mim": build_attacker("mim", model=val_attack_model, img_size=self.args.imgsz),
        }
        return validator

    def validate(self) -> tuple[dict, float]:
        """Run periodic validation and compute robust fitness from weighted mAP50-95."""
        if (self.epoch + 1) % self.val_interval != 0 and (self.epoch + 1) < self.epochs:
            return {}, 0.0

        # Temporarily switch defense modules to eval mode
        self.denoiser.eval().requires_grad_(False)
        self.stn.eval().requires_grad_(False)
        self.joint_model.pipeline.eval().requires_grad_(False)

        self.validator.device = self.device
        self.validator.data = self.data
        self.validator.dataloader = self.test_loader
        self.validator.pipeline = self.joint_model.pipeline.to(self.device)
        self.validator.guidance_model = self.guidance_model.to(self.device)

        try:
            summary = self.validator.evaluate_all(
                modes=[self.preprocess_mode],
                attacks=["clean", "pgd", "bim", "mim", "worstk"],
                seed=int(getattr(self.args, "seed", 0)),
            )
            raw_res = summary.get("raw_results", {})

            metrics = {}
            attack_order = ["clean", "pgd", "bim", "mim", "worstk"]
            fitness = 0.0
            total_w = 0.0

            for i, att in enumerate(attack_order):
                m_dict = raw_res.get(att, {}).get(self.preprocess_mode, {})
                map50 = float(m_dict.get("mAP50", 0.0))
                map95 = float(m_dict.get("mAP50-95", 0.0))
                metrics[f"{att}/mAP50"] = map50
                metrics[f"{att}/mAP50-95"] = map95

                w = self.eval_weights[i] if i < len(self.eval_weights) else (1.0 / len(attack_order))
                fitness += w * map95
                total_w += w

            if total_w > 0:
                fitness /= total_w
            metrics["fitness"] = fitness

            LOGGER.info(
                f"[Validation Epoch {self.epoch + 1}] Robust weighted mAP50-95 fitness: {fitness:.4f} "
                f"(Clean: {metrics.get('clean/mAP50-95', 0.0):.4f}, "
                f"PGD: {metrics.get('pgd/mAP50-95', 0.0):.4f}, "
                f"BIM: {metrics.get('bim/mAP50-95', 0.0):.4f}, "
                f"MIM: {metrics.get('mim/mAP50-95', 0.0):.4f}, "
                f"WorstK: {metrics.get('worstk/mAP50-95', 0.0):.4f})"
            )
            return metrics, fitness
        finally:
            # Restore training state and gradient computation to prevent permanent freezing of defense modules
            self.denoiser.train().requires_grad_(True)
            self.stn.train().requires_grad_(True)
            self.joint_model.pipeline.train().requires_grad_(True)

    def save_model(self) -> bool:
        """Save joint checkpoint and standalone FGD and STN checkpoints."""
        checkpoint = {
            "epoch": self.epoch,
            "fgd_state_dict": self.denoiser.state_dict(),
            "denoiser": self.denoiser.state_dict(),
            "stn_state_dict": self.stn.state_dict(),
            "stn": self.stn.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if hasattr(self, "scheduler") and self.scheduler
            else {},
            "scaler_state_dict": self.scaler.state_dict() if hasattr(self, "scaler") and self.scaler else {},
            "joint_training_config": {
                "preprocess_mode": self.preprocess_mode,
                "init": self.init_mode,
                "lambda_fgd": self.lambda_fgd,
                "lambda_geom": self.lambda_geom,
                "lambda_recon": self.lambda_recon,
                "lambda_det": self.lambda_det,
                "lambda_identity": self.lambda_identity,
                "lr_fgd": self.lr_fgd,
                "lr_stn": self.lr_stn,
                "val_interval": self.val_interval,
                "fgd": {
                    "lr": self.lr_fgd,
                },
                "stn": {
                    "lr": self.lr_stn,
                    "degrees": self.degrees,
                    "translate": self.translate,
                    "worstk_k": self.worstk_k,
                },
            },
            "stn_architecture": {
                "max_degrees": getattr(self.stn, "max_degrees", self.degrees * 1.5),
                "max_translate": getattr(self.stn, "max_translate", self.translate * 1.5),
                "fill": getattr(self.stn, "fill", 114.0 / 255.0),
            },
            "guidance_model_reference": str(self.args.model),
            "guidance_model": str(self.args.model),
            "attack_configuration": {
                "attack_name": self.attack_names,
                "attack_ratio": self.attack_ratios,
                "p_clean": self.p_clean,
                "attack_through_preprocess": self.attack_through_preprocess,
                "attack_weights": self.attack_weights,
            },
            "validation_metrics": getattr(self, "metrics", {}),
            "best_fitness": self.best_fitness,
            "train_args": vars(self.args),
            "random_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }

        # Save last.pt
        last_path = self.wdir / "last.pt"
        torch.save(checkpoint, last_path)

        # Export standalone last checkpoints
        export_joint_checkpoint(checkpoint, output_dir=self.wdir, prefix="last_")

        # Update best.pt if fitness improved
        current_fitness = float(self.fitness) if self.fitness is not None else 0.0
        if current_fitness > self.best_fitness:
            self.best_fitness = current_fitness
            checkpoint["best_fitness"] = self.best_fitness
            best_path = self.wdir / "best.pt"
            torch.save(checkpoint, best_path)
            export_joint_checkpoint(checkpoint, output_dir=self.wdir, prefix="best_")
            LOGGER.info(f"Saving new best joint checkpoint: robust fitness={self.best_fitness:.4f}")

        # Periodic checkpoint save
        if self.save_period > 0 and (self.epoch + 1) % self.save_period == 0:
            torch.save(checkpoint, self.wdir / f"epoch{self.epoch + 1}.pt")

        return True

    def final_eval(self) -> None:
        """Print final evaluation summary after training completes."""
        best_path = self.wdir / "best.pt"
        LOGGER.info(
            f"FGD+STN joint training completed successfully. Best joint checkpoint: {best_path} "
            f"(robust weighted mAP50-95={self.best_fitness:.4f})"
        )
