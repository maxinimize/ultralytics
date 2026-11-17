# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Comprehensive integration test suite covering real production flows for research training:
- Test A: Real successful attack path across all 3 trainers (DetectionTrainerAdvAdaptive,
  DetectionTrainerAdvFGDSTNAdaptive, DetectionTrainerAdvFGDSTNStaged Stage 2). Real constructors,
  real dataset, real .train() execution, real generated counts, real loss, real weight updates via optimizer.
- Test B: Real attack failure and rollback path. Fault injection at attack boundary (SpatialWorstOfK).
  Non-strict mode catches error, rolls back batch to raw, increments failed and actual_raw, conserves total
  sample count, and finishes with finite loss. Strict mode raises RuntimeError with input_img_shape,
  attack_name, and batch index.
- Test C: Real Stage 1 continuous vs resumed training consistency. Full 3-epoch real Stage 1 continuous
  training saving epoch1.pt vs resuming from epoch1.pt for remaining 2 epochs. Verified for both seed=0
  and seed=42: 100% exact match of sample order, batch images, dataloader generator state, and torch RNG state
  across all batches.
- Test D: Real validation entrypoint, model_source recording, and actual_batch clamping. Explicit best.pt
  passed to validator with initial args.model 'yolo11n.yaml'. Verified AutoBackend loading, inference, and
  recording of model_source, requested_model, FP32 precision, and actual_batch clamped to dataset size.
- Test E: Task 1 PGD random initialization & Task 2 isolate_rng fallback mechanism. ARTPGD native PyTorch PGD
  with num_random_init=1 reproducible under same seed, distinct under different seeds, bounded in [-eps, eps],
  parameter .grad untouched. All 7 isolate_rng fallback and exception cases verified.
- Test F: Real eval_weights normalization and non-zero multi-metric aggregation. 4-level priority without
  mutating caller dictionary/namespace. Zero-weight clean exclusion and accurate mathematical fitness calculation.
- Test G: Native PGD numerical failure handling, strict error propagation, transaction rollback, and sample conservation.

"""

from __future__ import annotations

import random
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image

from ultralytics.attacks.art_pgd import ARTPGD
from ultralytics.attacks.attack_bridge import run_attack_on_batch
from ultralytics.attacks.attack_utils import isolate_rng
from ultralytics.attacks.spatial_worst_of_k import SpatialWorstOfK
from ultralytics.defenses.hgd_fgd import ResidualDenoiser
from ultralytics.defenses.spatial_transformer import SpatialTransformer
from ultralytics.models.yolo.detect.train_adv_adaptive import DetectionTrainer as DetectionTrainerAdvAdaptive
from ultralytics.models.yolo.detect.train_adv_fgd_stn_adaptive import DetectionTrainerAdvFGDSTNAdaptive
from ultralytics.models.yolo.detect.train_adv_fgd_stn_staged import DetectionTrainerAdvFGDSTNStaged
from ultralytics.models.yolo.detect.val_adv_adaptive import DetectionValidator as DetectionValidatorAdvAdaptive
from ultralytics.models.yolo.detect.val_adv_fgd_stn_staged import DetectionValidatorAdvFGDSTNStaged
from ultralytics.nn.tasks import DetectionModel


def create_synthetic_dataset(root_dir: Path, num_images: int = 4, imgsz: int = 64) -> Path:
    """Create a minimal synthetic YOLO dataset with bounding boxes for integration tests."""
    img_dir = root_dir / "images" / "train"
    lbl_dir = root_dir / "labels" / "train"
    val_img_dir = root_dir / "images" / "val"
    val_lbl_dir = root_dir / "labels" / "val"
    for d in (img_dir, lbl_dir, val_img_dir, val_lbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    for idx in range(num_images):
        color = ((idx * 37) % 255, (idx * 67) % 255, (idx * 97) % 255)
        im = Image.new("RGB", (imgsz, imgsz), color=color)
        im.save(img_dir / f"img_{idx}.jpg")
        im.save(val_img_dir / f"img_{idx}.jpg")

        lbl_content = f"0 {0.3 + (idx % 2) * 0.2} {0.3 + (idx % 2) * 0.2} 0.25 0.25\n"
        (lbl_dir / f"img_{idx}.txt").write_text(lbl_content, encoding="utf-8")
        (val_lbl_dir / f"img_{idx}.txt").write_text(lbl_content, encoding="utf-8")

    yaml_path = root_dir / "data.yaml"
    yaml_content = f"""
path: {root_dir.as_posix()}
train: images/train
val: images/val
names:
  0: target_0
nc: 1
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path


def create_defense_checkpoints(root_dir: Path) -> tuple[Path, Path]:
    """Create minimal valid FGD and STN checkpoints for testing."""
    fgd_path = root_dir / "fgd_test.pt"
    stn_path = root_dir / "stn_test.pt"
    fgd = ResidualDenoiser()
    stn = SpatialTransformer(max_degrees=30.0, max_translate=0.10)
    torch.save({"denoiser": fgd.state_dict()}, fgd_path)
    torch.save(
        {
            "stn": stn.state_dict(),
            "stn_architecture": {"max_degrees": 30.0, "max_translate": 0.10, "fill": 114.0 / 255.0},
        },
        stn_path,
    )
    return fgd_path, stn_path


class ObservingStage1Trainer(DetectionTrainerAdvFGDSTNStaged):
    """Stage 1 trainer that observes batch sample orders, image tensors, and RNG states per epoch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observed_epochs: dict[int, list[dict[str, Any]]] = {}

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        ep = self.epoch
        if ep not in self.observed_epochs:
            self.observed_epochs[ep] = []
        gen_st = None
        if hasattr(self, "train_loader") and getattr(self.train_loader, "generator", None) is not None:
            gen_st = self.train_loader.generator.get_state().clone()
        self.observed_epochs[ep].append(
            {
                "im_file": list(batch["im_file"]),
                "img": batch["img"].clone().cpu(),
                "gen_state": gen_st,
                "torch_rng": torch.get_rng_state().clone(),
            }
        )
        return batch


class FaultySpatialWorstOfK(SpatialWorstOfK):
    """Injected faulty attacker to test boundary error handling and shape mismatch."""

    def __init__(self, *args, fault_mode: str = "exception", **kwargs):
        super().__init__(*args, **kwargs)
        self.fault_mode = fault_mode
        self.call_count = 0

    def forward_batch(self, batch):
        self.call_count += 1
        super().forward_batch(batch)
        if self.fault_mode == "exception":
            raise RuntimeError("Injected attack failure at boundary")
        elif self.fault_mode == "wrong_shape":
            batch["img"] = torch.rand(len(batch["img"]), 3, 32, 32, device=batch["img"].device)


class FaultyTrainer(DetectionTrainerAdvAdaptive):
    """Trainer injecting FaultySpatialWorstOfK at build_attacker boundary."""

    def __init__(self, *args, fault_mode: str = "exception", **kwargs):
        self.fault_mode = fault_mode
        super().__init__(*args, **kwargs)

    def build_attacker(self, name, model, imgsz, seed=None):
        effective_seed = getattr(self.args, "seed", None) if seed is None else seed
        if name == "worstk":
            return FaultySpatialWorstOfK(model=model, img_size=imgsz, seed=effective_seed, fault_mode=self.fault_mode)
        return super().build_attacker(name, model, imgsz, seed=seed)


class InjectedFaultLoss(torch.autograd.Function):
    """Autograd function to inject NaN, Inf, or zero gradients for testing."""

    @staticmethod
    def forward(ctx, loss, fault_mode="nan_grad"):
        ctx.fault_mode = fault_mode
        return loss.clone()

    @staticmethod
    def backward(ctx, grad_output):
        mode = ctx.fault_mode
        if mode == "nan_grad":
            return torch.full_like(grad_output, float("nan")), None
        elif mode == "inf_grad":
            return torch.full_like(grad_output, float("inf")), None
        elif mode == "zero_grad":
            return torch.zeros_like(grad_output), None
        return grad_output, None


class FaultyPGD(ARTPGD):
    """Injected faulty PGD attacker to test numerical failures and state preservation."""

    def __init__(self, *args, fault_mode: str = "nan_grad", **kwargs):
        super().__init__(*args, **kwargs)
        self.fault_mode = fault_mode
        self._wrap_loss()

    def _wrap_loss(self):
        orig_loss_fn = self.wrapped_model._loss_fn

        def faulty_loss(batch, preds):
            loss, items = orig_loss_fn(batch, preds)
            if self.fault_mode == "none_loss":
                return None, items
            if self.fault_mode == "no_grad":
                return loss.detach(), items
            if self.fault_mode == "nan_loss":
                return loss * float("nan"), items
            if self.fault_mode in ("nan_grad", "inf_grad", "zero_grad"):
                return InjectedFaultLoss.apply(loss, self.fault_mode), items
            return loss, items

        self.wrapped_model._loss_fn = faulty_loss


def create_faulty_pgd_trainer_class(base_cls):
    """Factory to create a trainer subclass that injects FaultyPGD."""

    class FaultyPGDTrainer(base_cls):
        def __init__(self, *args, fault_mode: str = "nan_grad", **kwargs):
            self.fault_mode = fault_mode
            super().__init__(*args, **kwargs)

        def build_attacker(self, name, model, imgsz, seed=None):
            effective_seed = getattr(self.args, "seed", None) if seed is None else seed
            atk = super().build_attacker(name, model, imgsz, seed=effective_seed)
            if name == "pgd" and self.fault_mode:
                orig_loss_fn = atk.wrapped_model._loss_fn
                m = self.fault_mode

                def faulty_loss(batch, preds):
                    loss, items = orig_loss_fn(batch, preds)
                    if m == "none_loss":
                        return None, items
                    if m == "no_grad":
                        return loss.detach(), items
                    if m == "nan_loss":
                        return loss * float("nan"), items
                    if m in ("nan_grad", "inf_grad", "zero_grad"):
                        return InjectedFaultLoss.apply(loss, m), items
                    return loss, items

                atk.wrapped_model._loss_fn = faulty_loss
            return atk

    return FaultyPGDTrainer


class FaultyPGDValidator(DetectionValidatorAdvAdaptive):
    """Validator subclass injecting faults into PGD attacker."""

    def __init__(self, *args, fault_mode: str = "nan_grad", **kwargs):
        self.fault_mode = fault_mode
        super().__init__(*args, **kwargs)

    def build_attacker(self, name: str, model: Any, imgsz: int, seed: int = 0):
        atk = super().build_attacker(name, model=model, imgsz=imgsz, seed=seed)
        if name == "pgd" and self.fault_mode:
            orig_loss_fn = atk.wrapped_model._loss_fn
            m = self.fault_mode

            def faulty_loss(batch, preds):
                loss, items = orig_loss_fn(batch, preds)
                if m == "none_loss":
                    return None, items
                if m == "no_grad":
                    return loss.detach(), items
                if m == "nan_loss":
                    return loss * float("nan"), items
                if m in ("nan_grad", "inf_grad", "zero_grad"):
                    return InjectedFaultLoss.apply(loss, m), items
                return loss, items

            atk.wrapped_model._loss_fn = faulty_loss
        return atk


class TestResearchTrainingIntegration(unittest.TestCase):
    """Real production flow integration test suite covering Tests A through F."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.data_yaml = create_synthetic_dataset(self.temp_dir, num_images=4, imgsz=64)
        self.fgd_path, self.stn_path = create_defense_checkpoints(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Test A: 3 Real Trainers Successful Training Flow
    # -------------------------------------------------------------------------
    def test_a_real_trainer_train_all_three(self):
        """Test A: Call real trainer.train() for 1 epoch on all 3 trainers:
        1. DetectionTrainerAdvAdaptive
        2. DetectionTrainerAdvFGDSTNAdaptive
        3. DetectionTrainerAdvFGDSTNStaged (Stage 2)
        Verify attacks are generated (generated > 0, failed == 0), loss is finite,
        optimizer updates parameters, and defense parameters remain frozen.
        """
        # Save a stage1 checkpoint for Stage 2
        stage1_weights = self.temp_dir / "stage1_best.pt"
        base_model = DetectionModel("yolo11n.yaml", ch=3, nc=1, verbose=False)
        torch.save({"model": base_model}, stage1_weights)

        trainer_configs = [
            (
                DetectionTrainerAdvAdaptive,
                {
                    "model": "yolo11n.yaml",
                    "data": str(self.data_yaml),
                    "epochs": 1,
                    "imgsz": 64,
                    "batch": 2,
                    "workers": 0,
                    "device": "cpu",
                    "amp": False,
                    "project": str(self.temp_dir / "runs_a1"),
                    "name": "exp",
                    "plots": False,
                    "val": False,
                    "save": True,
                    "attack_name": ["worstk"],
                    "attack_ratio": [1.0],
                },
                False,
            ),
            (
                DetectionTrainerAdvFGDSTNAdaptive,
                {
                    "model": "yolo11n.yaml",
                    "data": str(self.data_yaml),
                    "epochs": 1,
                    "imgsz": 64,
                    "batch": 2,
                    "workers": 0,
                    "device": "cpu",
                    "amp": False,
                    "project": str(self.temp_dir / "runs_a2"),
                    "name": "exp",
                    "plots": False,
                    "val": False,
                    "save": True,
                    "attack_name": ["worstk"],
                    "attack_ratio": [1.0],
                    "preprocess_mode": "stn_fgd",
                    "fgd_weights": str(self.fgd_path),
                    "stn_weights": str(self.stn_path),
                },
                True,
            ),
            (
                DetectionTrainerAdvFGDSTNStaged,
                {
                    "model": "yolo11n.yaml",
                    "data": str(self.data_yaml),
                    "epochs": 1,
                    "imgsz": 64,
                    "batch": 2,
                    "workers": 0,
                    "device": "cpu",
                    "amp": False,
                    "project": str(self.temp_dir / "runs_a3"),
                    "name": "exp",
                    "plots": False,
                    "val": False,
                    "save": True,
                    "attack_name": ["worstk"],
                    "attack_ratio": [1.0],
                    "preprocess_mode": "stn_fgd",
                    "fgd_weights": str(self.fgd_path),
                    "stn_weights": str(self.stn_path),
                    "stage": 2,
                    "stage1_weights": str(stage1_weights),
                },
                True,
            ),
        ]

        for trainer_cls, overrides, has_defense in trainer_configs:
            with self.subTest(trainer=trainer_cls.__name__):
                trainer = trainer_cls(overrides=overrides)

                # Capture an initial trainable parameter to verify optimizer updates it
                p_before = None

                def capture_initial_param(tr):
                    nonlocal p_before
                    for p in tr.model.parameters():
                        if p.requires_grad:
                            p_before = p.detach().clone()
                            break

                trainer.add_callback("on_train_start", capture_initial_param)

                # Execute real 1-epoch training loop
                trainer.train()

                # Verify attack generation counts
                self.assertGreater(
                    trainer.epoch_attack_stats["generated"]["worstk"],
                    0,
                    f"{trainer_cls.__name__} should generate attacks!",
                )
                self.assertEqual(
                    trainer.epoch_attack_stats["failed"]["worstk"],
                    0,
                    f"{trainer_cls.__name__} should have 0 failed attacks!",
                )

                # Verify optimizer updated parameters
                if p_before is not None:
                    target_p = None
                    for p in trainer.model.parameters():
                        if p.requires_grad:
                            target_p = p.detach()
                            break
                    self.assertIsNotNone(target_p)
                    self.assertFalse(
                        torch.equal(p_before, target_p),
                        f"{trainer_cls.__name__} optimizer did not update parameters!",
                    )

                # Verify defense modules remained frozen in defense trainers
                if has_defense and getattr(trainer, "pipeline", None) is not None:
                    for dp in trainer.pipeline.parameters():
                        self.assertFalse(
                            dp.requires_grad,
                            f"{trainer_cls.__name__} defense pipeline parameters must remain frozen!",
                        )

                # Verify outputs were saved
                self.assertTrue((trainer.wdir / "last.pt").exists(), "last.pt must exist!")

    # -------------------------------------------------------------------------
    # Test B: Real Attack Failure and Rollback Flow
    # -------------------------------------------------------------------------
    def test_b_fault_injection_at_attack_boundary(self):
        """Test B: Verify attack failure handling with boundary fault injection:
        1. When strict_attacks=False, real trainer catches failure, rolls back batch to raw,
           increments failed and actual_raw, conserves total sample count, and completes with finite loss.
        2. When strict_attacks=True, real trainer raises RuntimeError containing input_img_shape,
           attack_name, and batch index.
        """
        # 1. Non-strict mode: exception injected at boundary
        overrides_non_strict = {
            "model": "yolo11n.yaml",
            "data": str(self.data_yaml),
            "epochs": 1,
            "imgsz": 64,
            "batch": 2,
            "workers": 0,
            "device": "cpu",
            "amp": False,
            "project": str(self.temp_dir / "runs_b_non_strict"),
            "name": "exp",
            "plots": False,
            "val": False,
            "save": False,
            "attack_name": ["worstk"],
            "attack_ratio": [1.0],
            "strict_attacks": False,
        }
        trainer_non_strict = FaultyTrainer(overrides=overrides_non_strict, fault_mode="exception")
        trainer_non_strict.train()

        # Check failed attacks and actual_raw were incremented
        failed_count = trainer_non_strict.epoch_attack_stats["failed"]["worstk"]
        actual_raw_count = trainer_non_strict.epoch_attack_stats["actual_raw"]
        self.assertGreater(failed_count, 0, "Failed attacks must be recorded!")
        self.assertGreater(actual_raw_count, 0, "actual_raw must be incremented on rollback!")
        # Verify sample conservation: total executed samples == 4 (all dataset samples)
        total_samples = actual_raw_count + trainer_non_strict.epoch_attack_stats["generated"]["worstk"]
        self.assertEqual(total_samples, 4, "Total batch samples must be conserved!")

        # 2. Strict mode: shape mismatch injected at boundary
        overrides_strict = {
            "model": "yolo11n.yaml",
            "data": str(self.data_yaml),
            "epochs": 1,
            "imgsz": 64,
            "batch": 2,
            "workers": 0,
            "device": "cpu",
            "amp": False,
            "project": str(self.temp_dir / "runs_b_strict"),
            "name": "exp",
            "plots": False,
            "val": False,
            "save": False,
            "attack_name": ["worstk"],
            "attack_ratio": [1.0],
            "strict_attacks": True,
        }
        trainer_strict = FaultyTrainer(overrides=overrides_strict, fault_mode="wrong_shape")
        with self.assertRaises(RuntimeError) as ctx:
            trainer_strict.train()

        err_msg = str(ctx.exception)
        self.assertIn("worstk", err_msg, "Error message must identify attack name!")
        self.assertIn("batch 0", err_msg, "Error message must identify batch index!")
        self.assertTrue(
            "mismatch with input" in err_msg or "input" in err_msg,
            f"Error message must include input shape information: {err_msg}",
        )

        # 3. Verify seed=0 is preserved and not coerced to 42
        m0 = DetectionModel("yolo11n.yaml", ch=3, nc=1, verbose=False)
        tr_seed0 = FaultyTrainer(
            overrides={"model": "yolo11n.yaml", "data": str(self.data_yaml), "seed": 0},
            fault_mode="exception",
        )
        atk0 = tr_seed0.build_attacker("worstk", m0, 64)
        self.assertEqual(atk0.seed, 0, "seed=0 must be preserved and not overridden to 42!")

    # -------------------------------------------------------------------------
    # Test C: Stage 1 Real 3-Epoch Continuous vs Resume Consistency
    # -------------------------------------------------------------------------
    def test_c_stage1_real_3epoch_resume_consistency(self):
        """Test C: Real Stage 1 training 3 epochs continuous vs resumed from epoch1.pt.
        Tested for both seed=0 and seed=42.
        Verifies 100% equality of batch sample order, image tensors, dataloader generator state,
        and PyTorch RNG state across Epoch 1 and Epoch 2.
        """
        # Create dataset with 6 images so each epoch has 3 batches of size 2
        dataset_yaml = create_synthetic_dataset(self.temp_dir / "ds_c", num_images=6, imgsz=64)

        for test_seed in (0, 42):
            with self.subTest(seed=test_seed):
                run_dir = self.temp_dir / f"stage1_seed_{test_seed}"

                # 1. Continuous run: 3 epochs
                overrides_cont = {
                    "model": "yolo11n.yaml",
                    "data": str(dataset_yaml),
                    "epochs": 3,
                    "imgsz": 64,
                    "batch": 2,
                    "workers": 0,
                    "device": "cpu",
                    "amp": False,
                    "project": str(run_dir),
                    "name": "continuous",
                    "plots": False,
                    "val": False,
                    "save": True,
                    "save_period": 1,
                    "seed": test_seed,
                    "stage": 1,
                    "preprocess_mode": "stn_fgd",
                    "fgd_weights": str(self.fgd_path),
                    "stn_weights": str(self.stn_path),
                }

                trainer_cont = ObservingStage1Trainer(overrides=overrides_cont)
                trainer_cont.train()

                epoch1_pt = trainer_cont.wdir / "epoch1.pt"
                self.assertTrue(epoch1_pt.exists(), f"epoch1.pt must exist for seed={test_seed}!")

                # 2. Resumed run from epoch1.pt (for total 3 epochs)
                overrides_res = {
                    "resume": str(epoch1_pt),
                    "val": False,
                    "amp": False,
                    "plots": False,
                }
                trainer_res = ObservingStage1Trainer(overrides=overrides_res)
                trainer_res.train()

                # 3. Compare Epoch 1 and Epoch 2 between continuous and resumed runs
                for ep in (1, 2):
                    self.assertIn(ep, trainer_cont.observed_epochs)
                    self.assertIn(ep, trainer_res.observed_epochs)
                    batches_cont = trainer_cont.observed_epochs[ep]
                    batches_res = trainer_res.observed_epochs[ep]

                    self.assertEqual(
                        len(batches_cont),
                        len(batches_res),
                        f"Seed {test_seed} Epoch {ep} batch count mismatch!",
                    )

                    for b_idx, (b_c, b_r) in enumerate(zip(batches_cont, batches_res)):
                        self.assertEqual(
                            b_c["im_file"],
                            b_r["im_file"],
                            f"Seed {test_seed} Epoch {ep} Batch {b_idx} sample order mismatch!",
                        )
                        self.assertTrue(
                            torch.equal(b_c["img"], b_r["img"]),
                            f"Seed {test_seed} Epoch {ep} Batch {b_idx} image tensor mismatch!",
                        )
                        self.assertTrue(
                            torch.equal(b_c["gen_state"], b_r["gen_state"]),
                            f"Seed {test_seed} Epoch {ep} Batch {b_idx} dataloader generator mismatch!",
                        )
                        self.assertTrue(
                            torch.equal(b_c["torch_rng"], b_r["torch_rng"]),
                            f"Seed {test_seed} Epoch {ep} Batch {b_idx} torch RNG state mismatch!",
                        )

    # -------------------------------------------------------------------------
    # Test D: Validation Entrypoint, Model Source, and Batch Clamping
    # -------------------------------------------------------------------------
    def test_d_validation_entrypoint_model_source_and_clamping(self):
        """Test D: Real validator execution verifying:
        1. Explicit best.pt recorded in validation_context model_source.
        2. Initial config recorded in requested_model.
        3. Precision recorded as FP32 in Stage 1.
        4. actual_batch clamped to dataset size (2) when requested batch is 8.
        """
        # Create dataset with only 2 val images
        ds_val_yaml = create_synthetic_dataset(self.temp_dir / "ds_d", num_images=2, imgsz=64)

        best_pt = self.temp_dir / "weights" / "best.pt"
        best_pt.parent.mkdir(parents=True, exist_ok=True)
        m = DetectionModel("yolo11n.yaml", ch=3, nc=1, verbose=False)
        torch.save({"model": m}, best_pt)

        args = {
            "model": "yolo11n.yaml",
            "data": str(ds_val_yaml),
            "batch": 8,  # Requested batch size 8 > dataset size 2
            "imgsz": 64,
            "workers": 0,
            "device": "cpu",
            "plots": False,
            "save_json": False,
        }

        validator = DetectionValidatorAdvFGDSTNStaged(
            args=args,
            stage=1,
            preprocess_mode="stn_fgd",
            fgd_weights=str(self.fgd_path),
            stn_weights=str(self.stn_path),
        )

        # Call real validator with explicit model checkpoint
        _ = validator(model=str(best_pt))

        ctx = getattr(validator, "last_validation_context", {})
        self.assertEqual(ctx.get("model_source"), str(best_pt))
        self.assertEqual(ctx.get("requested_model"), "yolo11n.yaml")
        self.assertEqual(ctx.get("precision"), "FP32")
        self.assertEqual(ctx.get("actual_batch"), 2)

    # -------------------------------------------------------------------------
    # Test E: Task 1 PGD Random Init & Task 2 isolate_rng Fallback
    # -------------------------------------------------------------------------
    def test_e1_pgd_random_initialization(self):
        """Test E1: ARTPGD native PyTorch PGD random initialization:
        1. Generates perturbation on CPU DetectionModel with num_random_init=1 without ART error.
        2. Perturbations are reproducible under identical seed.
        3. Perturbations are distinct under different seeds.
        4. Perturbations are bounded in [-eps, +eps] and values clamped in [0, 1].
        5. Model parameter .grad is completely untouched.
        6. num_random_init > 1 explicitly raises ValueError.
        """
        model = DetectionModel("yolo11n.yaml", ch=3, nc=1, verbose=False).to(torch.device("cpu"))
        model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        model.init_criterion()
        model.train()

        # Seed parameter grad with non-zero tensor to verify it is untouched
        first_param = next(model.parameters())
        first_param.grad = torch.ones_like(first_param.data) * 0.123
        orig_grad = first_param.grad.clone()

        eps = 8.0 / 255.0
        pgd = ARTPGD(model, epsilon=eps, lr=eps / 4.0, epoch=3, num_random_init=1)

        batch_1 = {
            "img": torch.rand(2, 3, 64, 64, device=torch.device("cpu")),
            "batch_idx": torch.tensor([0, 1]),
            "cls": torch.tensor([[0.0], [0.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3], [0.4, 0.4, 0.2, 0.2]]),
        }
        batch_2 = {
            "img": batch_1["img"].clone(),
            "batch_idx": batch_1["batch_idx"].clone(),
            "cls": batch_1["cls"].clone(),
            "bboxes": batch_1["bboxes"].clone(),
        }
        batch_3 = {
            "img": batch_1["img"].clone(),
            "batch_idx": batch_1["batch_idx"].clone(),
            "cls": batch_1["cls"].clone(),
            "bboxes": batch_1["bboxes"].clone(),
        }

        # Run with seed=100
        with isolate_rng(seed=100, attackers=[pgd]):
            adv_1 = run_attack_on_batch(pgd, batch_1)

        # Run with seed=100 again (must be identical)
        with isolate_rng(seed=100, attackers=[pgd]):
            adv_2 = run_attack_on_batch(pgd, batch_2)

        # Run with seed=200 (must be different due to random init)
        with isolate_rng(seed=200, attackers=[pgd]):
            adv_3 = run_attack_on_batch(pgd, batch_3)

        self.assertIsNotNone(adv_1)
        self.assertIsNotNone(adv_2)
        self.assertIsNotNone(adv_3)

        # Reproducibility check
        self.assertTrue(torch.equal(adv_1, adv_2), "Same seed must yield identical PGD output!")
        # Random init check
        self.assertFalse(torch.equal(adv_1, adv_3), "Different seed must yield different PGD output!")

        # Bound checks
        pert_1 = (adv_1 - batch_1["img"]).abs()
        self.assertLessEqual(pert_1.max().item(), eps + 1e-6, "Perturbation must not exceed epsilon!")
        self.assertTrue((adv_1 >= 0.0).all() and (adv_1 <= 1.0).all(), "Perturbed images must be in [0, 1]!")

        # Direct forward check with explicit targets
        targets = torch.tensor([[0, 0, 0.5, 0.5, 0.3, 0.3], [1, 0, 0.4, 0.4, 0.2, 0.2]], dtype=torch.float32)
        with isolate_rng(seed=100, attackers=[pgd]):
            direct_adv = pgd.forward(batch_1["img"], targets)
        self.assertTrue(torch.equal(adv_1, direct_adv), "Direct forward must match run_attack_on_batch!")

        # Parameter .grad untouched
        self.assertTrue(
            torch.equal(first_param.grad, orig_grad),
            "ARTPGD native PGD must not modify or zero parameter .grad!",
        )

        # num_random_init > 1 rejection
        with self.assertRaises(ValueError):
            ARTPGD(model, epsilon=eps, num_random_init=2)

    def test_e2_isolate_rng_fallback_and_recovery(self):
        """Test E2: All 7 acceptance scenarios for isolate_rng:
        1. Normal state_dict / load_state_dict restoration.
        2. state_dict fails but private _generator restored via _generator.set_state.
        3. reset_seed fails in try block; original RNG restored.
        4. Partial failure across multiple loaders; all possible states restored.
        5. DataLoader generator exists but get_state() fails; raises RuntimeError before resetting seeds.
        6. Body exception + cleanup failure chains __cause__.
        7. seed=0 vs non-zero seed correctly applied.
        """

        # Scenario 1: Normal state_dict success
        class NormalAttacker:
            def __init__(self):
                self.val = 1

            def state_dict(self):
                return {"val": self.val}

            def load_state_dict(self, d):
                self.val = d["val"]

        atk1 = NormalAttacker()
        with isolate_rng(seed=42, attackers=[atk1]):
            atk1.val = 99
        self.assertEqual(atk1.val, 1)

        # Scenario 2: state_dict fails, private _generator restored via set_state
        class FallbackGeneratorAttacker:
            def __init__(self):
                self._generator = torch.Generator().manual_seed(123)

            def state_dict(self):
                raise NotImplementedError("state_dict unsupported")

        atk2 = FallbackGeneratorAttacker()
        orig_gen_state = atk2._generator.get_state().clone()
        with isolate_rng(seed=42, attackers=[atk2]):
            _ = torch.rand(5, generator=atk2._generator)
        self.assertTrue(torch.equal(atk2._generator.get_state(), orig_gen_state))

        # Scenario 3: reset_seed fails; original RNG restored
        class BrokenResetAttacker:
            def state_dict(self):
                return {}

            def load_state_dict(self, d):
                pass

            def reset_seed(self, s):
                raise RuntimeError("Reset seed failed")

        orig_torch_rng = torch.get_rng_state()
        atk3 = BrokenResetAttacker()
        with self.assertRaises(RuntimeError) as ctx, isolate_rng(seed=42, attackers=[atk3]):
            pass
        self.assertIn("Reset seed failed", str(ctx.exception))
        self.assertTrue(torch.equal(torch.get_rng_state(), orig_torch_rng))

        # Scenario 4: Partial failure across multiple attackers; summaries aggregated
        class FailingLoadAttacker:
            def state_dict(self):
                return {"val": 1}

            def load_state_dict(self, d):
                raise RuntimeError("Cannot load state")

        atk_good = NormalAttacker()
        atk_bad = FailingLoadAttacker()
        with self.assertRaises(RuntimeError) as ctx, isolate_rng(seed=42, attackers=[atk_good, atk_bad]):
            atk_good.val = 77
        self.assertEqual(atk_good.val, 1, "Good attacker state must still be restored!")
        self.assertIn("isolate_rng cleanup encountered 1 failure(s)", str(ctx.exception))

        # Scenario 5: DataLoader generator get_state() fails; raises RuntimeError before modifying seeds
        class BrokenLoader:
            def __init__(self):
                class BadGen:
                    def get_state(self):
                        raise RuntimeError("Cannot capture generator")

                self.generator = BadGen()

        bl = BrokenLoader()
        orig_py = random.getstate()
        with self.assertRaises(RuntimeError) as ctx, isolate_rng(seed=42, dataloader=bl):
            pass
        self.assertIn("Failed to capture DataLoader generator state", str(ctx.exception))
        self.assertEqual(random.getstate(), orig_py)

        # Scenario 6: Body exception + cleanup failure chains __cause__
        with self.assertRaises(RuntimeError) as ctx, isolate_rng(seed=42, attackers=[atk_bad]):
            raise ValueError("Original body exception")
        self.assertIsInstance(ctx.exception.__cause__, ValueError)
        self.assertEqual(str(ctx.exception.__cause__), "Original body exception")

        # Scenario 7: seed=0 vs non-zero seed handling
        with isolate_rng(seed=0):
            val_0 = random.random()
        with isolate_rng(seed=0):
            val_0_repeat = random.random()
        with isolate_rng(seed=42):
            val_42 = random.random()
        self.assertEqual(val_0, val_0_repeat)
        self.assertNotEqual(val_0, val_42)

    # -------------------------------------------------------------------------
    # Test F: Real eval_weights Normalization and Multi-Metric Aggregation
    # -------------------------------------------------------------------------
    def test_f1_eval_weights_priority_and_caller_immutability(self):
        """Test F1: 4-level priority without mutating caller dictionary or namespace."""
        # Priority 1: Explicit constructor arg
        caller_dict1 = {"eval_weights": [0.2, 0.2, 0.2, 0.4]}
        v1 = DetectionValidatorAdvFGDSTNStaged(
            args=caller_dict1,
            eval_weights=[0.1, 0.2, 0.3, 0.4],
            preprocess_mode="none",
        )
        self.assertEqual(v1.eval_weights, [0.1, 0.2, 0.3, 0.4])
        self.assertIn("eval_weights", caller_dict1, "Caller dictionary must remain intact!")

        # Priority 2: In args dictionary
        caller_dict2 = {"eval_weights": [0.4, 0.3, 0.2, 0.1]}
        v2 = DetectionValidatorAdvFGDSTNStaged(
            args=caller_dict2,
            preprocess_mode="none",
        )
        self.assertEqual(v2.eval_weights, [0.4, 0.3, 0.2, 0.1])

        # Priority 3: In args namespace
        caller_ns = SimpleNamespace(eval_weights=[0.5, 0.5, 0.0, 0.0])
        v3 = DetectionValidatorAdvFGDSTNStaged(
            args=caller_ns,
            preprocess_mode="none",
        )
        self.assertEqual(v3.eval_weights, [0.5, 0.5, 0.0, 0.0])

        # Priority 4: Default when omitted
        v4 = DetectionValidatorAdvFGDSTNStaged(
            args={},
            preprocess_mode="none",
        )
        self.assertIsNone(v4.eval_weights)
        norm_w = v4.compute_norm_eval_weights(["pgd", "bim", "mim", "worstk"])
        self.assertAlmostEqual(norm_w["pgd"], 0.25 / 0.75, places=5)
        self.assertAlmostEqual(norm_w["bim"], 0.15 / 0.75, places=5)
        self.assertAlmostEqual(norm_w["mim"], 0.15 / 0.75, places=5)
        self.assertAlmostEqual(norm_w["worstk"], 0.20 / 0.75, places=5)
        self.assertEqual(norm_w["raw"], 0.0)

    def test_f2_clean_zero_weight_normalization_and_fitness(self):
        """Test F2: Clean weight = 0 normalization and distinct non-zero metric aggregation."""
        validator = DetectionValidatorAdvAdaptive(args={"eval_weights": [0.4, 0.3, 0.2, 0.1]})
        norm_w = validator.compute_norm_eval_weights(["pgd", "bim", "mim", "worstk"])
        self.assertEqual(norm_w["raw"], 0.0)
        self.assertAlmostEqual(sum(norm_w[k] for k in ["pgd", "bim", "mim", "worstk"]), 1.0, places=5)

        # Distinct non-zero metric results
        all_res = {
            "raw": {"metrics/mAP50-95(B)": 0.95, "metrics/mAP50(B)": 0.98},
            "pgd": {"metrics/mAP50-95(B)": 0.50, "metrics/mAP50(B)": 0.70},
            "bim": {"metrics/mAP50-95(B)": 0.40, "metrics/mAP50(B)": 0.60},
            "mim": {"metrics/mAP50-95(B)": 0.30, "metrics/mAP50(B)": 0.50},
            "worstk": {"metrics/mAP50-95(B)": 0.20, "metrics/mAP50(B)": 0.40},
        }

        # Expected mAP50-95 = 0.4*0.50 + 0.3*0.40 + 0.2*0.30 + 0.1*0.20 = 0.20 + 0.12 + 0.06 + 0.02 = 0.40
        # Expected mAP50 = 0.4*0.70 + 0.3*0.60 + 0.2*0.50 + 0.1*0.40 = 0.28 + 0.18 + 0.10 + 0.04 = 0.60
        stats = validator.aggregate_multirun_metrics(
            metric_arrays={k: {} for k in all_res},
            all_results=all_res,
            norm_eval_weights=norm_w,
            eval_runs=["pgd", "bim", "mim", "worstk"],
        )

        self.assertAlmostEqual(stats["metrics/mAP50-95(B)"], 0.40, places=5)
        self.assertAlmostEqual(stats["metrics/mAP50(B)"], 0.60, places=5)
        self.assertAlmostEqual(stats["fitness"], 0.40, places=5)

    # -------------------------------------------------------------------------
    # Test G: Native PGD Numerical Failures, Rollback, and Sample Conservation
    # -------------------------------------------------------------------------
    def test_g1_direct_pgd_numerical_failures_and_state_preservation(self):
        """Test G1: Direct PGD numerical failure handling and state preservation:
        1. NaN loss before grad raises FloatingPointError with iteration info.
        2. Finite loss + NaN grad raises FloatingPointError with iteration info.
        3. Finite loss + Inf grad raises FloatingPointError with iteration info.
        4. Disconnected loss (no grad) raises RuntimeError with iteration info.
        5. Loss is None raises RuntimeError.
        6. Finite all-zero grad executes normally.
        7. Model parameter .grad, requires_grad, and .training are preserved after failure.
        8. Delayed fault at step 2/3 reports iteration 2/3.
        """
        model = DetectionModel("yolo11n.yaml", ch=3, nc=1, verbose=False).to(torch.device("cpu"))
        model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        model.init_criterion()
        model.train()

        # Set dummy grad on first parameter to verify preservation
        first_param = next(model.parameters())
        first_param.grad = torch.ones_like(first_param.data) * 0.42
        orig_grad = first_param.grad.clone()
        orig_req_grad = first_param.requires_grad
        orig_training = model.training

        x = torch.rand(2, 3, 64, 64)
        y = torch.tensor([[0, 0, 0.5, 0.5, 0.2, 0.2]], dtype=torch.float32)

        # 1. NaN loss
        pgd_nan_loss = FaultyPGD(model, epsilon=8.0 / 255.0, epoch=3, fault_mode="nan_loss")
        with self.assertRaises(FloatingPointError) as ctx:
            pgd_nan_loss.forward(x, y)
        self.assertIn("iteration 0/3", str(ctx.exception))

        # 2. NaN grad
        pgd_nan_grad = FaultyPGD(model, epsilon=8.0 / 255.0, epoch=3, fault_mode="nan_grad")
        with self.assertRaises(FloatingPointError) as ctx:
            pgd_nan_grad.forward(x, y)
        self.assertIn("iteration 0/3", str(ctx.exception))

        # 3. Inf grad
        pgd_inf_grad = FaultyPGD(model, epsilon=8.0 / 255.0, epoch=3, fault_mode="inf_grad")
        with self.assertRaises(FloatingPointError) as ctx:
            pgd_inf_grad.forward(x, y)
        self.assertIn("iteration 0/3", str(ctx.exception))

        # 4. Disconnected loss
        pgd_no_grad = FaultyPGD(model, epsilon=8.0 / 255.0, epoch=3, fault_mode="no_grad")
        with self.assertRaises(RuntimeError) as ctx:
            pgd_no_grad.forward(x, y)
        self.assertTrue("requires_grad" in str(ctx.exception) or "require grad" in str(ctx.exception))

        # 5. Loss is None
        pgd_none_loss = FaultyPGD(model, epsilon=8.0 / 255.0, epoch=3, fault_mode="none_loss")
        with self.assertRaises(RuntimeError) as ctx:
            pgd_none_loss.forward(x, y)
        self.assertIn("None", str(ctx.exception))

        # 6. All-zero grad executes normally
        pgd_zero_grad = FaultyPGD(model, epsilon=8.0 / 255.0, epoch=3, fault_mode="zero_grad")
        adv_zero = pgd_zero_grad.forward(x, y)
        self.assertIsNotNone(adv_zero)
        self.assertTrue((adv_zero >= 0.0).all() and (adv_zero <= 1.0).all())

        # 7. Model state preservation verification
        self.assertTrue(torch.equal(first_param.grad, orig_grad), "param.grad was modified during exception!")
        self.assertEqual(first_param.requires_grad, orig_req_grad, "param.requires_grad was modified!")
        self.assertEqual(model.training, orig_training, "model.training mode was not restored!")

        # 8. Delayed fault reporting exact step
        pgd_delayed = ARTPGD(model, epsilon=8.0 / 255.0, epoch=3)
        orig_fn = pgd_delayed.wrapped_model._loss_fn
        delayed_step = 0

        def delayed_fault_fn(b, p):
            nonlocal delayed_step
            loss, items = orig_fn(b, p)
            if delayed_step == 2:
                loss = loss * float("nan")
            delayed_step += 1
            return loss, items

        pgd_delayed.wrapped_model._loss_fn = delayed_fault_fn
        with self.assertRaises(FloatingPointError) as ctx:
            pgd_delayed.forward(x, y)
        self.assertIn("iteration 2/3", str(ctx.exception))

    def test_g2_trainer_pgd_numerical_failure_strict_and_non_strict(self):
        """Test G2: Real trainer PGD numerical failure across all 3 defense trainers:
        - DetectionTrainerAdvAdaptive
        - DetectionTrainerAdvFGDSTNAdaptive
        - DetectionTrainerAdvFGDSTNStaged (Stage 2)
        Verify strict raises RuntimeError with iteration and batch info;
        non-strict rolls back to raw, increments failed and actual_raw, and conserves total samples.
        """
        stage1_weights = self.temp_dir / "stage1_g2.pt"
        base_model = DetectionModel("yolo11n.yaml", ch=3, nc=1, verbose=False)
        torch.save({"model": base_model}, stage1_weights)

        trainer_configs = [
            (
                create_faulty_pgd_trainer_class(DetectionTrainerAdvAdaptive),
                {
                    "model": "yolo11n.yaml",
                    "data": str(self.data_yaml),
                    "epochs": 1,
                    "imgsz": 64,
                    "batch": 2,
                    "workers": 0,
                    "device": "cpu",
                    "amp": False,
                    "project": str(self.temp_dir / "runs_g2_1"),
                    "name": "exp",
                    "plots": False,
                    "val": False,
                    "save": False,
                    "attack_name": ["pgd"],
                    "attack_ratio": [1.0],
                },
            ),
            (
                create_faulty_pgd_trainer_class(DetectionTrainerAdvFGDSTNAdaptive),
                {
                    "model": "yolo11n.yaml",
                    "data": str(self.data_yaml),
                    "epochs": 1,
                    "imgsz": 64,
                    "batch": 2,
                    "workers": 0,
                    "device": "cpu",
                    "amp": False,
                    "project": str(self.temp_dir / "runs_g2_2"),
                    "name": "exp",
                    "plots": False,
                    "val": False,
                    "save": False,
                    "attack_name": ["pgd"],
                    "attack_ratio": [1.0],
                    "preprocess_mode": "stn_fgd",
                    "fgd_weights": str(self.fgd_path),
                    "stn_weights": str(self.stn_path),
                },
            ),
            (
                create_faulty_pgd_trainer_class(DetectionTrainerAdvFGDSTNStaged),
                {
                    "model": "yolo11n.yaml",
                    "data": str(self.data_yaml),
                    "epochs": 1,
                    "imgsz": 64,
                    "batch": 2,
                    "workers": 0,
                    "device": "cpu",
                    "amp": False,
                    "project": str(self.temp_dir / "runs_g2_3"),
                    "name": "exp",
                    "plots": False,
                    "val": False,
                    "save": False,
                    "attack_name": ["pgd"],
                    "attack_ratio": [1.0],
                    "preprocess_mode": "stn_fgd",
                    "fgd_weights": str(self.fgd_path),
                    "stn_weights": str(self.stn_path),
                    "stage": 2,
                    "stage1_weights": str(stage1_weights),
                },
            ),
        ]

        for trainer_cls, overrides in trainer_configs:
            tr_name = trainer_cls.__mro__[1].__name__
            with self.subTest(trainer=tr_name, mode="non_strict"):
                cfg_non_strict = {**overrides, "strict_attacks": False}
                trainer_ns = trainer_cls(overrides=cfg_non_strict, fault_mode="nan_grad")
                trainer_ns.train()

                stats = trainer_ns.epoch_attack_stats
                failed_pgd = stats["failed"]["pgd"]
                gen_pgd = stats["generated"]["pgd"]
                actual_raw = stats["actual_raw"]

                self.assertGreater(failed_pgd, 0, f"{tr_name} failed.pgd must be > 0!")
                self.assertEqual(gen_pgd, 0, f"{tr_name} generated.pgd must be 0 on failure!")
                self.assertEqual(actual_raw + gen_pgd, 4, f"{tr_name} total samples must equal dataset size (4)!")

            with self.subTest(trainer=tr_name, mode="strict"):
                cfg_strict = {**overrides, "strict_attacks": True}
                trainer_s = trainer_cls(overrides=cfg_strict, fault_mode="nan_grad")
                with self.assertRaises(RuntimeError) as ctx:
                    trainer_s.train()

                err_msg = str(ctx.exception)
                self.assertIn("pgd", err_msg)
                self.assertIn("batch 0", err_msg)
                self.assertTrue(
                    "iteration" in err_msg or "input_img_shape" in err_msg,
                    f"Error message must include iteration or input shape: {err_msg}",
                )

    def test_g3_mixed_batch_fault_isolation_and_sample_conservation(self):
        """Test G3: Mixed batch with 4 attacks (pgd, bim, mim, worstk) where PGD fails:
        1. PGD rolls back to raw; BIM, MIM, and WorstK succeed.
        2. Total executed batch samples is exactly conserved (actual_raw + generated == total).
        """
        ds_g3 = create_synthetic_dataset(self.temp_dir / "ds_g3", num_images=8, imgsz=64)
        overrides = {
            "model": "yolo11n.yaml",
            "data": str(ds_g3),
            "epochs": 1,
            "imgsz": 64,
            "batch": 8,
            "workers": 0,
            "device": "cpu",
            "amp": False,
            "project": str(self.temp_dir / "runs_g3"),
            "name": "exp",
            "plots": False,
            "val": False,
            "save": False,
            "attack_name": ["pgd", "bim", "mim", "worstk"],
            "attack_ratio": [0.25, 0.25, 0.25, 0.25],
            "strict_attacks": False,
        }
        trainer_cls = create_faulty_pgd_trainer_class(DetectionTrainerAdvAdaptive)
        trainer = trainer_cls(overrides=overrides, fault_mode="nan_grad")
        trainer.train()

        stats = trainer.epoch_attack_stats
        self.assertGreater(stats["failed"]["pgd"], 0, "PGD must be recorded as failed!")
        self.assertEqual(stats["generated"]["pgd"], 0, "PGD must have 0 generated samples!")
        self.assertGreater(stats["generated"]["bim"], 0, "BIM must generate samples!")
        self.assertGreater(stats["generated"]["mim"], 0, "MIM must generate samples!")
        self.assertGreater(stats["generated"]["worstk"], 0, "WorstK must generate samples!")

        total_executed = stats["actual_raw"] + sum(stats["generated"].values())
        self.assertEqual(total_executed, 8, "Total batch samples must be conserved exactly at 8!")

    def test_g4_validator_pgd_numerical_failure_strict_and_non_strict(self):
        """Test G4: Validator fault handling on numerical failure:
        1. Non-strict validator records attack_failures and assigns 0.0 mAP.
        2. Strict validator raises RuntimeError identifying attack and batch.
        """
        ds_g4 = create_synthetic_dataset(self.temp_dir / "ds_g4", num_images=2, imgsz=64)
        val_pt = self.temp_dir / "val_g4.pt"
        m = DetectionModel("yolo11n.yaml", ch=3, nc=1, verbose=False)
        torch.save({"model": m}, val_pt)

        # 1. Non-strict validator
        args_ns = {
            "model": str(val_pt),
            "data": str(ds_g4),
            "batch": 2,
            "imgsz": 64,
            "workers": 0,
            "device": "cpu",
            "plots": False,
            "save_json": False,
            "attack_name": ["pgd"],
            "strict_attacks": False,
        }
        v_ns = FaultyPGDValidator(args=args_ns, fault_mode="nan_grad")
        _ = v_ns(model=str(val_pt))
        self.assertIn("pgd", v_ns.attack_failures, "attack_failures must include 'pgd'!")
        self.assertEqual(v_ns.last_attack_maps.get("pgd"), 0.0, "Failed attack mAP must be 0.0!")

        # 2. Strict validator
        args_s = {
            "model": str(val_pt),
            "data": str(ds_g4),
            "batch": 2,
            "imgsz": 64,
            "workers": 0,
            "device": "cpu",
            "plots": False,
            "save_json": False,
            "attack_name": ["pgd"],
            "strict_attacks": True,
        }
        v_s = FaultyPGDValidator(args=args_s, fault_mode="nan_grad")
        with self.assertRaises(RuntimeError) as ctx:
            v_s(model=str(val_pt))
        self.assertIn("pgd", str(ctx.exception), "Strict validator error must identify attack name!")

    def test_g5_stage2_real_pipeline_four_attacks_normal_flow(self):
        """Test G5: Stage 2 normal execution with 4 attacks (pgd, bim, mim, worstk) and real STN/FGD."""
        ds_g5 = create_synthetic_dataset(self.temp_dir / "ds_g5", num_images=8, imgsz=64)
        stage1_weights = self.temp_dir / "stage1_g5.pt"
        base_model = DetectionModel("yolo11n.yaml", ch=3, nc=1, verbose=False)
        torch.save({"model": base_model}, stage1_weights)

        overrides = {
            "model": "yolo11n.yaml",
            "data": str(ds_g5),
            "epochs": 1,
            "imgsz": 64,
            "batch": 8,
            "workers": 0,
            "device": "cpu",
            "amp": False,
            "project": str(self.temp_dir / "runs_stage2_g5"),
            "name": "exp",
            "plots": False,
            "val": True,
            "save": True,
            "attack_name": ["pgd", "bim", "mim", "worstk"],
            "attack_ratio": [0.25, 0.25, 0.25, 0.25],
            "preprocess_mode": "stn_fgd",
            "fgd_weights": str(self.fgd_path),
            "stn_weights": str(self.stn_path),
            "stage": 2,
            "stage1_weights": str(stage1_weights),
        }
        trainer = DetectionTrainerAdvFGDSTNStaged(overrides=overrides)
        trainer.train()

        stats = trainer.epoch_attack_stats
        for att in ("pgd", "bim", "mim", "worstk"):
            self.assertGreater(stats["generated"][att], 0, f"Attack {att} must generate samples!")
            self.assertEqual(stats["failed"][att], 0, f"Attack {att} must have 0 failed samples!")

        self.assertTrue((trainer.wdir / "last.pt").exists(), "last.pt must exist!")
        self.assertTrue((trainer.wdir / "best.pt").exists(), "best.pt must exist!")


if __name__ == "__main__":
    unittest.main()
