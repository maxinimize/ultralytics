# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""CLI runner for Unified FGD + STN Defense Validation across E1-E5 preprocessing modes and five attack types."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ultralytics.attacks.attack_bridge import build_attacker
from ultralytics.attacks.attack_utils import setup_attack_model
from ultralytics.attacks.spatial_worst_of_k_stn import SpatialWorstOfKWithMetadata
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.defenses.fgd_stn_pipeline import (
    MODE_DESCRIPTIONS,
    VALID_MODES,
    build_fgd_stn_pipeline,
    canonicalize_mode,
)
from ultralytics.models.yolo.detect.val_fgd_stn import (
    SUPPORTED_ATTACKS,
    UnifiedFGDSTNValidator,
    canonicalize_attack,
)
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import DEFAULT_CFG, LOGGER, IterableSimpleNamespace
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.torch_utils import select_device, unwrap_model


def parse_args():
    """Parse CLI options for Unified FGD + STN Defense validation."""
    parser = argparse.ArgumentParser(
        description="Unified FGD + STN Defense Validation across E1-E5 preprocessing modes and adversarial attacks."
    )

    # Model and Defense Checkpoints
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to YOLO detector weights (e.g. yolo12l.pt). If unset, attempts to read guidance model from FGD checkpoint.",
    )
    parser.add_argument(
        "--fgd_weights",
        default=None,
        help="Path to trained FGD checkpoint (weights/best.pt). If unset or 'none', evaluates without FGD denoiser.",
    )
    parser.add_argument(
        "--stn_weights",
        default=None,
        help="Path to trained STN checkpoint (weights/best.pt). If unset or 'none', evaluates with identity STN.",
    )
    parser.add_argument(
        "--attack_weights",
        default=None,
        help="Optional separate attack model checkpoint for black-box/transfer attacks. Default uses evaluated detector.",
    )

    # Dataset Settings
    parser.add_argument(
        "--data",
        required=True,
        help="Dataset YAML file path (e.g. coco_train_traffic5k_test.yaml).",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Dataset evaluation split (default: val).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=None,
        help="Filter dataset by specific class IDs (e.g. --classes 0 1 2 3).",
    )

    # Preprocessing Modes (E1 - E5)
    parser.add_argument(
        "--modes",
        "--mode",
        nargs="+",
        default=["all"],
        help=(
            "Preprocessing mode(s) to evaluate: 'all' (E1-E5), or any subset of "
            "['none', 'fgd', 'stn', 'fgd_stn', 'stn_fgd']. Also supports aliases E1-E5."
        ),
    )

    # Attack Selection
    parser.add_argument(
        "--attacks",
        "--attack",
        nargs="+",
        default=["all"],
        help=(
            "Adversarial attack(s) to evaluate: 'all' (all 5 attacks), or any subset of "
            "['clean', 'pgd', 'bim', 'mim', 'worstk']."
        ),
    )

    # Worst-of-K Attack Parameters
    parser.add_argument(
        "--worstk_k",
        type=int,
        default=10,
        help="Worst-of-k candidate sample count (default: 10).",
    )
    parser.add_argument(
        "--degrees",
        type=float,
        default=30.0,
        help="Worst-of-k maximum rotation degrees (default: 30.0).",
    )
    parser.add_argument(
        "--translate",
        type=float,
        default=0.10,
        help="Worst-of-k maximum translation ratio (default: 0.10).",
    )

    # PGD Attack Parameters
    parser.add_argument(
        "--pgd_eps",
        type=float,
        default=8.0 / 255.0,
        help="PGD attack epsilon bound in [0, 1] (default: 8/255).",
    )
    parser.add_argument(
        "--pgd_lr",
        type=float,
        default=2.0 / 255.0,
        help="PGD attack step size (default: 2/255).",
    )
    parser.add_argument(
        "--pgd_iters",
        type=int,
        default=10,
        help="PGD attack iteration count (default: 10).",
    )

    # BIM Attack Parameters
    parser.add_argument(
        "--bim_eps",
        type=float,
        default=0.05,
        help="BIM attack epsilon bound (default: 0.05).",
    )
    parser.add_argument(
        "--bim_lr",
        type=float,
        default=0.02,
        help="BIM attack step size (default: 0.02).",
    )
    parser.add_argument(
        "--bim_iters",
        type=int,
        default=5,
        help="BIM attack iteration count (default: 5).",
    )

    # MIM Attack Parameters
    parser.add_argument(
        "--mim_eps",
        type=float,
        default=8.0 / 255.0,
        help="MIM attack epsilon bound (default: 8/255).",
    )
    parser.add_argument(
        "--mim_lr",
        type=float,
        default=2.0 / 255.0,
        help="MIM attack step size (default: 2/255).",
    )
    parser.add_argument(
        "--mim_iters",
        type=int,
        default=5,
        help="MIM attack iteration count (default: 5).",
    )
    parser.add_argument(
        "--mim_decay",
        type=float,
        default=1.0,
        help="MIM momentum decay factor (default: 1.0).",
    )

    # Inference and Evaluation Settings
    parser.add_argument("--batch", type=int, default=8, help="Validation batch size (default: 8).")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size in pixels (default: 640).")
    parser.add_argument("--device", default="0", help="CUDA device index (e.g. 0) or 'cpu'.")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (default: 4).")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold (default: 0.001).")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold (default: 0.6).")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Reproducibility random seed for data sampling and attacks (default: 0).",
    )

    # Output and Logging Settings
    parser.add_argument(
        "--project",
        default="runs/val_fgd_stn",
        help="Project directory for saving validation results (default: runs/val_fgd_stn).",
    )
    parser.add_argument(
        "--name",
        default="exp",
        help="Experiment name (default: exp).",
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        default=False,
        help="Save raw prediction detections to predictions.json.",
    )

    return parser.parse_args()


def resolve_models_and_pipeline(args, device: torch.device):
    """Load detector, attack model, and unified defense pipeline."""
    # 1. Initialize Unified Defense Pipeline
    pipeline, meta = build_fgd_stn_pipeline(
        fgd_weights=args.fgd_weights,
        stn_weights=args.stn_weights,
        device=device,
        allow_identity_fallback=True,
        max_degrees=args.degrees * 1.5,
        max_translate=args.translate * 1.5,
    )

    # 2. Determine YOLO guidance weights
    weights = args.weights
    if not weights or str(weights).strip().lower() in {"", "none"}:
        guidance_path = meta.get("fgd_meta", {}).get("guidance_model")
        if guidance_path and Path(guidance_path).exists():
            weights = str(Path(guidance_path).resolve())
            LOGGER.info(f"Using guidance model from FGD checkpoint: {weights}")
        else:
            weights = "yolo12l.pt"
            LOGGER.info(f"Defaulting detector weights to: {weights}")

    # Load guidance YOLO detector
    guidance_model, _ = load_checkpoint(weights, device=device)
    guidance_model.eval().requires_grad_(False)
    if hasattr(guidance_model, "args") and isinstance(guidance_model.args, dict):
        guidance_model.args = IterableSimpleNamespace(**guidance_model.args)

    # 3. Setup attack model for black-box/transfer or white-box attack generation
    has_custom_attack_weights = (
        args.attack_weights
        and str(args.attack_weights).strip().lower() not in {"", "none", "current"}
        and Path(args.attack_weights).is_file()
    )

    if has_custom_attack_weights:
        LOGGER.info(f"Loading separate black-box attack model from {args.attack_weights}...")
        attack_model, _ = load_checkpoint(args.attack_weights, device=device)
        attack_model.eval().requires_grad_(False)
        if hasattr(attack_model, "args") and isinstance(attack_model.args, dict):
            attack_model.args = IterableSimpleNamespace(**attack_model.args)
        for p in attack_model.parameters():
            p.requires_grad_(True)
    else:
        # Separate clone of guidance detector for attack gradient backpropagation
        LOGGER.info("Reusing guidance detector architecture for attack generation.")
        attack_model = copy.deepcopy(unwrap_model(guidance_model)).eval()
        for p in attack_model.parameters():
            p.requires_grad_(True)

    return guidance_model, attack_model, pipeline


def build_attackers_map(args, attack_model: torch.nn.Module, imgsz: int, seed: int) -> Dict[str, Any]:
    """Instantiate attacker objects for requested attacks."""
    attackers = {}

    # Normalize requested attacks
    raw_attacks = args.attacks
    if isinstance(raw_attacks, str):
        requested = list(SUPPORTED_ATTACKS) if raw_attacks.lower() == "all" else [canonicalize_attack(raw_attacks)]
    else:
        if "all" in [str(a).lower() for a in raw_attacks]:
            requested = list(SUPPORTED_ATTACKS)
        else:
            requested = [canonicalize_attack(a) for a in raw_attacks]

    for att in set(requested):
        if att == "clean":
            continue
        elif att == "worstk":
            attackers["worstk"] = SpatialWorstOfKWithMetadata(
                model=attack_model,
                k=args.worstk_k,
                degrees=args.degrees,
                translate=args.translate,
                img_size=imgsz,
                seed=seed,
            )
            LOGGER.info(
                f"Initialized SpatialWorstOfK (k={args.worstk_k}, deg={args.degrees}, trans={args.translate})"
            )
        elif att == "pgd":
            attackers["pgd"] = build_attacker(
                "pgd",
                model=attack_model,
                img_size=imgsz,
                epsilon=args.pgd_eps,
                lr=args.pgd_lr,
                epoch=args.pgd_iters,
            )
            LOGGER.info(f"Initialized ARTPGD (eps={args.pgd_eps:.5f}, lr={args.pgd_lr:.5f}, iters={args.pgd_iters})")
        elif att == "bim":
            attackers["bim"] = build_attacker(
                "bim",
                model=attack_model,
                img_size=imgsz,
                epsilon=args.bim_eps,
                lr=args.bim_lr,
                epoch=args.bim_iters,
            )
            LOGGER.info(f"Initialized ARTBIM (eps={args.bim_eps:.5f}, lr={args.bim_lr:.5f}, iters={args.bim_iters})")
        elif att == "mim":
            attackers["mim"] = build_attacker(
                "mim",
                model=attack_model,
                img_size=imgsz,
                epsilon=args.mim_eps,
                lr=args.mim_lr,
                epoch=args.mim_iters,
                decay=args.mim_decay,
            )
            LOGGER.info(
                f"Initialized ARTMIM (eps={args.mim_eps:.5f}, lr={args.mim_lr:.5f}, iters={args.mim_iters}, decay={args.mim_decay})"
            )

    return attackers


def main():
    """Execute Unified FGD + STN Defense validation."""
    args = parse_args()
    device = select_device(args.device)

    # 1. Load models and defenses
    guidance_model, attack_model, pipeline = resolve_models_and_pipeline(args, device)

    # 2. Build dataset and dataloader
    data_dict = check_det_dataset(args.data)
    stride = max(int(guidance_model.stride.max()) if hasattr(guidance_model, "stride") else 32, 32)
    imgsz = check_imgsz(args.imgsz, stride=stride)

    dataset_cfg = copy.copy(DEFAULT_CFG)
    dataset_cfg.task = "detect"
    dataset_cfg.imgsz = imgsz
    dataset_cfg.classes = args.classes

    img_path = data_dict.get(args.split)
    if not img_path:
        raise ValueError(f"Split '{args.split}' not found in dataset definition {args.data}")

    dataset = build_yolo_dataset(
        cfg=dataset_cfg,
        img_path=img_path,
        batch=args.batch,
        data=data_dict,
        mode="val",
        rect=False,
        stride=stride,
    )
    dataloader = build_dataloader(dataset, batch=args.batch, workers=args.workers, shuffle=False)

    # 3. Build attackers
    attackers = build_attackers_map(args, attack_model, imgsz, seed=args.seed)

    # 4. Prepare save directory
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 5. Initialize validator
    validator_args = dict(
        task="detect",
        conf=args.conf,
        iou=args.iou,
        classes=args.classes,
        data=args.data,
        save_json=args.save_json,
    )
    validator = UnifiedFGDSTNValidator(
        dataloader=dataloader,
        save_dir=save_dir,
        args=validator_args,
    )
    validator.data = data_dict
    validator.device = device
    validator.guidance_model = guidance_model
    validator.attack_model = attack_model
    validator.pipeline = pipeline
    validator.attackers = attackers

    # 6. Run comprehensive evaluation
    results = validator.evaluate_all(modes=args.modes, attacks=args.attacks, seed=args.seed)
    return results


if __name__ == "__main__":
    main()
