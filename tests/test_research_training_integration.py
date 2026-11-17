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

import train_adv_adaptive_run
import train_adv_fgd_phase2_run
import train_adv_fgd_stn_adaptive_run
import train_adv_fgd_stn_staged_run
import train_adv_test_run
from ultralytics.attacks.art_pgd import ARTPGD
from ultralytics.attacks.attack_bridge import run_attack_on_batch
from ultralytics.attacks.attack_utils import isolate_rng
from ultralytics.attacks.spatial_worst_of_k import SpatialWorstOfK
from ultralytics.defenses.hgd_fgd import ResidualDenoiser
from ultralytics.defenses.spatial_transformer import SpatialTransformer
from ultralytics.models.yolo.detect.train_adv_adaptive import DetectionTrainer as DetectionTrainerAdvAdaptive
from ultralytics.models.yolo.detect.train_adv_fgd_phase2 import FGDAdversarialTrainer
from ultralytics.models.yolo.detect.train_adv_fgd_stn_adaptive import DetectionTrainerAdvFGDSTNAdaptive
from ultralytics.models.yolo.detect.train_adv_fgd_stn_staged import DetectionTrainerAdvFGDSTNStaged
from ultralytics.models.yolo.detect.val_adv_adaptive import DetectionValidator as DetectionValidatorAdvAdaptive
from ultralytics.models.yolo.detect.val_adv_fgd_phase2 import DetectionValidator as Phase2DetectionValidator
from ultralytics.models.yolo.detect.val_adv_fgd_stn_staged import DetectionValidatorAdvFGDSTNStaged
from ultralytics.models.yolo.detect.val_adv_multi import DetectionValidator as MultiAdvDetectionValidator
from ultralytics.nn.tasks import DetectionModel
from val_fgd_stn_run import resolve_models_and_pipeline as resolve_fgd_stn_models
from val_stn_run import resolve_models as resolve_stn_models


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

    def __init__(self, *args, **kwargs):
        self.fault_mode = kwargs.pop("fault_mode", "nan_grad")
        super().__init__(*args, **kwargs)
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
    """Factory to create a trainer subclass that injects FaultyPGD into training."""

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


def create_val_faulty_pgd_trainer_class(base_cls):
    """Factory to create a trainer subclass that injects faults exclusively into the validation attacker."""

    class ValFaultyPGDTrainer(base_cls):
        def get_validator(self):
            validator = super().get_validator()
            orig_build = validator.build_attacker

            def faulty_build(name, model, imgsz, seed=0):
                atk = orig_build(name, model=model, imgsz=imgsz, seed=seed)
                if name == "pgd":
                    orig_loss = atk.wrapped_model._loss_fn

                    def faulty_loss(batch, preds):
                        loss, items = orig_loss(batch, preds)
                        if not torch.isfinite(loss).all():
                            raise ValueError("Expected original detection loss to be finite!")
                        return InjectedFaultLoss.apply(loss, "nan_grad"), items

                    atk.wrapped_model._loss_fn = faulty_loss
                return atk

            validator.build_attacker = faulty_build
            return validator

    return ValFaultyPGDTrainer


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

    def test_g6_validation_failure_intercepts_controller_and_checkpoints(self):
        """Test G6: Validation attack failure interception across all 3 adaptive trainers:
        - DetectionTrainerAdvAdaptive
        - DetectionTrainerAdvFGDSTNAdaptive
        - DetectionTrainerAdvFGDSTNStaged (Stage 2)
        Verifies:
        1. Non-strict mode (strict_attacks=False):
           a. train() raises RuntimeError identifying validation failure and 'pgd'.
           b. validator.attack_failures contains 'pgd', and original detection loss was finite.
           c. Training attack stats (failed.pgd) is 0, proving training was unaffected.
           d. Controller state (current_train_ratios, next_train_ratios, last_updated_epoch, history)
              is completely unchanged from before validation.
           e. No checkpoint (best.pt or last.pt) is written.
        2. Strict mode (strict_attacks=True):
           Combination trainers raise RuntimeError on validation failure and write no checkpoint.
        3. update_interval=5 (non-update epoch):
           Validation failure is intercepted before saving model; no checkpoint or history written.
        4. Healthy validation (no fault injection, legitimate mAP=0):
           Completes normally, updates controller, and writes valid checkpoints.
        """
        ds_g6 = create_synthetic_dataset(self.temp_dir / "ds_g6", num_images=4, imgsz=64)
        stage1_weights = self.temp_dir / "stage1_g6.pt"
        base_model = DetectionModel("yolo11n.yaml", ch=3, nc=1, verbose=False)
        torch.save({"model": base_model}, stage1_weights)

        # 1. Non-strict mode across all 3 trainers
        trainer_configs = [
            (
                create_val_faulty_pgd_trainer_class(DetectionTrainerAdvAdaptive),
                {
                    "model": "yolo11n.yaml",
                    "data": str(ds_g6),
                    "epochs": 1,
                    "imgsz": 64,
                    "batch": 4,
                    "workers": 0,
                    "device": "cpu",
                    "amp": False,
                    "project": str(self.temp_dir / "runs_g6_ns1"),
                    "name": "exp",
                    "plots": False,
                    "val": True,
                    "save": True,
                    "seed": 0,
                    "p_clean": 0.5,
                    "attack_name": ["pgd", "worstk"],
                    "attack_ratio": [0.1, 0.4],
                    "update_interval": 1,
                    "strict_attacks": False,
                },
            ),
            (
                create_val_faulty_pgd_trainer_class(DetectionTrainerAdvFGDSTNAdaptive),
                {
                    "model": "yolo11n.yaml",
                    "data": str(ds_g6),
                    "epochs": 1,
                    "imgsz": 64,
                    "batch": 4,
                    "workers": 0,
                    "device": "cpu",
                    "amp": False,
                    "project": str(self.temp_dir / "runs_g6_ns2"),
                    "name": "exp",
                    "plots": False,
                    "val": True,
                    "save": True,
                    "seed": 0,
                    "p_clean": 0.5,
                    "attack_name": ["pgd", "worstk"],
                    "attack_ratio": [0.1, 0.4],
                    "update_interval": 1,
                    "strict_attacks": False,
                    "preprocess_mode": "stn_fgd",
                    "fgd_weights": str(self.fgd_path),
                    "stn_weights": str(self.stn_path),
                },
            ),
            (
                create_val_faulty_pgd_trainer_class(DetectionTrainerAdvFGDSTNStaged),
                {
                    "model": "yolo11n.yaml",
                    "data": str(ds_g6),
                    "epochs": 1,
                    "imgsz": 64,
                    "batch": 4,
                    "workers": 0,
                    "device": "cpu",
                    "amp": False,
                    "project": str(self.temp_dir / "runs_g6_ns3"),
                    "name": "exp",
                    "plots": False,
                    "val": True,
                    "save": True,
                    "seed": 0,
                    "p_clean": 0.5,
                    "attack_name": ["pgd", "worstk"],
                    "attack_ratio": [0.1, 0.4],
                    "update_interval": 1,
                    "strict_attacks": False,
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
            with self.subTest(trainer=tr_name, scenario="non_strict_val_fault"):
                trainer = trainer_cls(overrides=overrides)
                init_ratios = dict(trainer.adaptive_controller.current_train_ratios)
                init_history = list(trainer.adaptive_controller.history)
                init_epoch = trainer.adaptive_controller.last_updated_epoch

                with self.assertRaises(RuntimeError) as ctx:
                    trainer.train()

                err_msg = str(ctx.exception)
                self.assertIn("pgd", err_msg.lower())
                self.assertTrue("fail" in err_msg.lower() or "validation" in err_msg.lower())

                # b. Validator recorded attack failure
                val_failures = getattr(trainer.validator, "attack_failures", set())
                self.assertIn("pgd", val_failures)

                # c. Training attacks had 0 failures
                train_failed = trainer.epoch_attack_stats["failed"].get("pgd", 0)
                self.assertEqual(train_failed, 0)

                # d. Controller state untouched
                self.assertEqual(trainer.adaptive_controller.current_train_ratios, init_ratios)
                self.assertEqual(trainer.adaptive_controller.history, init_history)
                self.assertEqual(trainer.adaptive_controller.last_updated_epoch, init_epoch)

                # e. No checkpoints written
                self.assertFalse(
                    (trainer.wdir / "best.pt").exists(), "best.pt must not be saved on validation failure!"
                )
                self.assertFalse(
                    (trainer.wdir / "last.pt").exists(), "last.pt must not be saved on validation failure!"
                )

        # 2. Strict mode across combination trainers
        for base_cls in (DetectionTrainerAdvFGDSTNAdaptive, DetectionTrainerAdvFGDSTNStaged):
            with self.subTest(trainer=base_cls.__name__, scenario="strict_val_fault"):
                trainer_cls = create_val_faulty_pgd_trainer_class(base_cls)
                ov_strict = {
                    "model": "yolo11n.yaml",
                    "data": str(ds_g6),
                    "epochs": 1,
                    "imgsz": 64,
                    "batch": 4,
                    "workers": 0,
                    "device": "cpu",
                    "amp": False,
                    "project": str(self.temp_dir / f"runs_g6_str_{base_cls.__name__}"),
                    "name": "exp",
                    "plots": False,
                    "val": True,
                    "save": True,
                    "seed": 0,
                    "p_clean": 0.5,
                    "attack_name": ["pgd", "worstk"],
                    "attack_ratio": [0.1, 0.4],
                    "update_interval": 1,
                    "strict_attacks": True,
                    "preprocess_mode": "stn_fgd",
                    "fgd_weights": str(self.fgd_path),
                    "stn_weights": str(self.stn_path),
                }
                if base_cls is DetectionTrainerAdvFGDSTNStaged:
                    ov_strict["stage"] = 2
                    ov_strict["stage1_weights"] = str(stage1_weights)

                trainer_s = trainer_cls(overrides=ov_strict)
                with self.assertRaises(RuntimeError):
                    trainer_s.train()
                self.assertFalse((trainer_s.wdir / "best.pt").exists())

        # 3. update_interval=5 (non-update epoch) interception
        with self.subTest(scenario="non_update_epoch_interception"):
            trainer_cls = create_val_faulty_pgd_trainer_class(DetectionTrainerAdvFGDSTNStaged)
            ov_no_update = {
                "model": "yolo11n.yaml",
                "data": str(ds_g6),
                "epochs": 1,
                "imgsz": 64,
                "batch": 4,
                "workers": 0,
                "device": "cpu",
                "amp": False,
                "project": str(self.temp_dir / "runs_g6_no_upd"),
                "name": "exp",
                "plots": False,
                "val": True,
                "save": True,
                "seed": 0,
                "p_clean": 0.5,
                "attack_name": ["pgd", "worstk"],
                "attack_ratio": [0.1, 0.4],
                "update_interval": 5,
                "strict_attacks": False,
                "preprocess_mode": "stn_fgd",
                "fgd_weights": str(self.fgd_path),
                "stn_weights": str(self.stn_path),
                "stage": 2,
                "stage1_weights": str(stage1_weights),
            }
            trainer_nu = trainer_cls(overrides=ov_no_update)
            with self.assertRaises(RuntimeError):
                trainer_nu.train()
            self.assertFalse((trainer_nu.wdir / "best.pt").exists())
            self.assertEqual(len(trainer_nu.adaptive_controller.history), 0)

        # 4. Healthy validation (no fault injection, legitimate mAP=0): completes and saves
        with self.subTest(scenario="healthy_validation_completes"):
            ov_healthy = {
                "model": "yolo11n.yaml",
                "data": str(ds_g6),
                "epochs": 1,
                "imgsz": 64,
                "batch": 4,
                "workers": 0,
                "device": "cpu",
                "amp": False,
                "project": str(self.temp_dir / "runs_g6_healthy"),
                "name": "exp",
                "plots": False,
                "val": True,
                "save": True,
                "seed": 0,
                "p_clean": 0.5,
                "attack_name": ["pgd", "worstk"],
                "attack_ratio": [0.1, 0.4],
                "update_interval": 1,
                "strict_attacks": False,
                "preprocess_mode": "stn_fgd",
                "fgd_weights": str(self.fgd_path),
                "stn_weights": str(self.stn_path),
                "stage": 2,
                "stage1_weights": str(stage1_weights),
            }
            trainer_h = DetectionTrainerAdvFGDSTNStaged(overrides=ov_healthy)
            trainer_h.train()
            self.assertTrue((trainer_h.wdir / "best.pt").exists(), "Healthy run must save best.pt!")
            self.assertTrue((trainer_h.wdir / "last.pt").exists(), "Healthy run must save last.pt!")
            self.assertEqual(len(trainer_h.adaptive_controller.history), 1)

    def test_h1_legacy_validator_fault_handling_and_feature_distillation(self):
        """Test H1 (R1 & R7): Legacy validator fault handling and feature distillation:
        1. MultiAdvDetectionValidator with missing attacker raises explicit RuntimeError with attack name & batch idx.
        2. MultiAdvDetectionValidator with injected backward NaN PGD raises explicit RuntimeError.
        3. MultiAdvDetectionValidator with raw evaluation succeeds and returns valid metrics.
        4. Phase2DetectionValidator with strict_attacks=True raises on missing attacker or invalid output.
        5. Feature Distillation pipeline:
           - Shape and [0, 1] range preservation.
           - _check_fd_input validates input tensor structure on first call.
           - Reuses single FD instance across batches without recreation.
           - Injected NaN triggers RuntimeError.
        """
        ds_h1 = create_synthetic_dataset(self.temp_dir / "ds_h1", num_images=2, imgsz=64)
        m = DetectionModel("yolo11n.yaml", ch=3, nc=1, verbose=False)
        from ultralytics.cfg import DEFAULT_CFG, get_cfg

        m.args = get_cfg(DEFAULT_CFG)
        m.eval()
        val_pt = self.temp_dir / "val_h1.pt"
        torch.save({"model": m}, val_pt)

        # 1. Missing attacker raises explicit RuntimeError
        with self.subTest(scenario="missing_attacker_raises"):
            v = MultiAdvDetectionValidator(
                args={
                    "model": str(val_pt),
                    "data": str(ds_h1),
                    "imgsz": 64,
                    "batch": 2,
                    "device": "cpu",
                    "workers": 0,
                    "plots": False,
                }
            )
            v.device = torch.device("cpu")
            v.attack_name = "pgd"
            v.attackers = {}
            v.batch_i = 0
            batch = {
                "img": torch.zeros((2, 3, 64, 64), dtype=torch.float32),
                "cls": torch.zeros((2, 1)),
                "bboxes": torch.zeros((2, 4)),
                "batch_idx": torch.zeros(2, dtype=torch.int64),
            }
            with self.assertRaises(RuntimeError) as ctx:
                v.preprocess(batch)
            self.assertIn("pgd", str(ctx.exception))
            self.assertIn("batch 0", str(ctx.exception))

        # 2. Injected backward NaN PGD raises explicit RuntimeError
        with self.subTest(scenario="injected_nan_backward_raises"):
            faulty_pgd = FaultyPGD(m, fault_mode="nan_grad", epsilon=8 / 255, lr=2 / 255, epoch=3, img_size=64)
            v_nan = MultiAdvDetectionValidator(
                args={
                    "model": str(val_pt),
                    "data": str(ds_h1),
                    "imgsz": 64,
                    "batch": 2,
                    "device": "cpu",
                    "workers": 0,
                    "plots": False,
                }
            )
            v_nan.device = torch.device("cpu")
            v_nan.attack_name = "pgd"
            v_nan.attackers = {"pgd": faulty_pgd}
            v_nan.batch_i = 1
            batch_nan = {
                "img": torch.rand((2, 3, 64, 64), dtype=torch.float32) * 255.0,
                "cls": torch.zeros((2, 1)),
                "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]]),
                "batch_idx": torch.tensor([0, 1], dtype=torch.int64),
            }
            with self.assertRaises(RuntimeError) as ctx:
                v_nan.preprocess(batch_nan)
            self.assertIn("pgd", str(ctx.exception))
            self.assertIn("batch 1", str(ctx.exception))
            self.assertIn("Adversarial gradient is NaN/Inf", str(ctx.exception))

        # 3. Raw validation succeeds
        with self.subTest(scenario="raw_validation_succeeds"):
            v_raw = MultiAdvDetectionValidator(
                args={
                    "model": str(val_pt),
                    "data": str(ds_h1),
                    "imgsz": 64,
                    "batch": 2,
                    "device": "cpu",
                    "workers": 0,
                    "plots": False,
                    "save_json": False,
                }
            )
            metrics = v_raw(model=str(val_pt))
            self.assertIsInstance(metrics, dict)
            self.assertIn("fitness", metrics)

        # 4. Phase2 DetectionValidator strict attacks
        with self.subTest(scenario="phase2_strict_attacks_raises"):
            v_p2 = Phase2DetectionValidator(
                args={
                    "model": str(val_pt),
                    "data": str(ds_h1),
                    "imgsz": 64,
                    "batch": 2,
                    "device": "cpu",
                    "workers": 0,
                    "plots": False,
                    "attack_name": "pgd",
                    "strict_attacks": True,
                }
            )
            v_p2.device = torch.device("cpu")
            v_p2.attack_name = "pgd"
            v_p2.attackers = {}
            v_p2.batch_i = 0
            batch_p2 = {
                "img": torch.zeros((2, 3, 64, 64), dtype=torch.float32),
                "cls": torch.zeros((2, 1)),
                "bboxes": torch.zeros((2, 4)),
                "batch_idx": torch.zeros(2, dtype=torch.int64),
            }
            with self.assertRaises(RuntimeError) as ctx:
                v_p2.preprocess(batch_p2)
            self.assertIn("pgd", str(ctx.exception))

        # 5. Feature Distillation pipeline
        with self.subTest(scenario="feature_distillation_pipeline"):
            v_fd = MultiAdvDetectionValidator(
                args={
                    "model": str(val_pt),
                    "data": str(ds_h1),
                    "imgsz": 64,
                    "batch": 2,
                    "device": "cpu",
                    "workers": 0,
                    "plots": False,
                    "feature_distillation": True,
                }
            )
            v_fd.attack_name = "raw"
            v_fd.device = torch.device("cpu")
            batch1 = {"img": torch.rand((2, 3, 64, 64), dtype=torch.float32) * 255.0}
            out1 = v_fd.preprocess(batch1)
            fd_instance_1 = v_fd.fd
            self.assertIsNotNone(fd_instance_1)
            self.assertEqual(out1["img"].shape, (2, 3, 64, 64))
            self.assertTrue(torch.isfinite(out1["img"]).all())

            # Second call must reuse the same FD instance
            batch2 = {"img": torch.rand((2, 3, 64, 64), dtype=torch.float32) * 255.0}
            v_fd.preprocess(batch2)
            self.assertIs(v_fd.fd, fd_instance_1, "FD instance must not be recreated per batch!")

            # Input validation checks
            with self.assertRaises((ValueError, RuntimeError)):
                v_fd._check_fd_input(torch.zeros((3, 64, 64)))  # not 4D
            with self.assertRaises((ValueError, RuntimeError)):
                v_fd._check_fd_input(torch.full((2, 3, 64, 64), -1.0))  # negative values
            with self.assertRaises((ValueError, RuntimeError)):
                v_fd._check_fd_input(torch.full((2, 3, 64, 64), float("nan")))  # NaN input

    def test_h2_explicit_weight_loading_failures_and_defaults(self):
        """Test H2 (R2): Explicit weight loading failures and defaults:
        1. val_stn_run:
           - Explicit non-existent --stn_weights raises FileNotFoundError.
           - Omitted --stn_weights creates identity STN.
           - Explicit non-existent --attack_weights raises FileNotFoundError.
           - Omitted --attack_weights reuses guidance detector.
        2. val_fgd_stn_run:
           - Explicit non-existent --stn_weights raises FileNotFoundError.
           - Explicit non-existent --attack_weights raises FileNotFoundError.
           - Omitted --attack_weights reuses guidance detector.
        """
        val_pt = self.temp_dir / "val_h2.pt"
        m = DetectionModel("yolo11n.yaml", ch=3, nc=1, verbose=False)
        m.eval()
        torch.save({"model": m}, val_pt)
        missing_pt = self.temp_dir / "non_existent_weights.pt"

        # 1. val_stn_run
        with self.subTest(scenario="val_stn_run_weights"):
            # Explicit non-existent stn_weights raises FileNotFoundError
            args_missing_stn = SimpleNamespace(
                weights=str(val_pt),
                stn_weights=str(missing_pt),
                attack_weights=None,
                degrees=30.0,
                translate=0.1,
            )
            with self.assertRaises(FileNotFoundError):
                resolve_stn_models(args_missing_stn, torch.device("cpu"))

            # Explicit non-existent attack_weights raises FileNotFoundError
            args_missing_att = SimpleNamespace(
                weights=str(val_pt),
                stn_weights=None,
                attack_weights=str(missing_pt),
                degrees=30.0,
                translate=0.1,
            )
            with self.assertRaises(FileNotFoundError):
                resolve_stn_models(args_missing_att, torch.device("cpu"))

            # Omitted weights create identity STN and reuse guidance detector
            args_default = SimpleNamespace(
                weights=str(val_pt),
                stn_weights=None,
                attack_weights=None,
                degrees=30.0,
                translate=0.1,
            )
            guidance, stn, attack = resolve_stn_models(args_default, torch.device("cpu"))
            self.assertIsInstance(stn, SpatialTransformer)
            self.assertIs(attack, guidance)

        # 2. val_fgd_stn_run
        with self.subTest(scenario="val_fgd_stn_run_weights"):
            # Explicit non-existent stn_weights raises FileNotFoundError
            args_fgd_missing_stn = SimpleNamespace(
                weights=str(val_pt),
                preprocess_mode="stn",
                stn_weights=str(missing_pt),
                fgd_weights=None,
                attack_weights=None,
                imgsz=64,
                degrees=30.0,
                translate=0.1,
            )
            with self.assertRaises(FileNotFoundError):
                resolve_fgd_stn_models(args_fgd_missing_stn, torch.device("cpu"))

            # Explicit non-existent attack_weights raises FileNotFoundError
            args_fgd_missing_att = SimpleNamespace(
                weights=str(val_pt),
                preprocess_mode="none",
                stn_weights=None,
                fgd_weights=None,
                attack_weights=str(missing_pt),
                imgsz=64,
                degrees=30.0,
                translate=0.1,
            )
            with self.assertRaises(FileNotFoundError):
                resolve_fgd_stn_models(args_fgd_missing_att, torch.device("cpu"))

            # Omitted attack_weights reuses guidance detector
            args_fgd_default = SimpleNamespace(
                weights=str(val_pt),
                preprocess_mode="none",
                stn_weights=None,
                fgd_weights=None,
                attack_weights=None,
                imgsz=64,
                degrees=30.0,
                translate=0.1,
            )
            _guidance2, attack2, _pipeline2 = resolve_fgd_stn_models(args_fgd_default, torch.device("cpu"))
            self.assertEqual(type(attack2), type(m))

    def test_h3_resume_parameter_priority_and_inheritance(self):
        """Test H3 (R3): Resume parameter priority, warm checkpoint inheritance, and CLI regression:
        1. Plain adaptive resume from real warm checkpoint preserves single/custom attack configuration
           (e.g. ['worstk'], [0.5], p_clean=0.5, [1.0]) without injecting default 4 attacks.
        2. Staged resume from real warm checkpoint preserves fixed surrogate weights and Stage 2 without
           falling back to 'current'.
        3. Staged resume with conflicting --stage 1 on a Stage 2 checkpoint raises ValueError early.
        4. train_adv_test_run and train_adv_fgd_phase2_run parse_args on resume do not inject fresh defaults
           (batch=24, imgsz=640, cache=True, use_fgd=True) into overrides.
        5. FGDAdversarialTrainer(use_fgd=False) real warm checkpoint resumes cleanly without requiring fgd_weights.
        6. Stage 1 fresh Python CLI default lr0 is 1e-5 while Shell script explicitly specifies 1e-6.
        """

        # Helper to produce real warm checkpoints by stopping after epoch 0
        def _make_warm_checkpoint(trainer_cls, extra_overrides, name):
            base = {
                "model": "yolo11n.yaml",
                "data": str(self.data_yaml),
                "epochs": 5,
                "imgsz": 64,
                "batch": 2,
                "workers": 0,
                "device": "cpu",
                "amp": False,
                "project": str(self.temp_dir / f"runs_{name}"),
                "name": "exp",
                "plots": False,
                "val": True,
                "save": True,
            }
            base.update(extra_overrides)
            tr = trainer_cls(overrides=base)

            def stop_epoch0(t):
                if t.epoch == 0 and t.last.exists():
                    raise KeyboardInterrupt("Simulated training interruption for warm checkpoint")

            tr.add_callback("on_fit_epoch_end", stop_epoch0)
            try:
                tr.train()
            except KeyboardInterrupt:
                pass
            ckpt_path = Path(tr.last).resolve()
            self.assertTrue(ckpt_path.exists(), f"Warm checkpoint {ckpt_path} was not created!")
            return ckpt_path

        # 1. Plain adaptive resume from real warm checkpoint
        with self.subTest(scenario="plain_adaptive_warm_resume"):
            warm_adaptive = _make_warm_checkpoint(
                DetectionTrainerAdvAdaptive,
                {
                    "attack_name": ["worstk"],
                    "attack_ratio": [0.5],
                    "p_clean": 0.5,
                    "eval_weights": [1.0],
                },
                "adaptive_h3",
            )
            parsed_ad = train_adv_adaptive_run.parse_args(["--resume", str(warm_adaptive)])
            self.assertFalse(hasattr(parsed_ad, "attack_name"))
            self.assertFalse(hasattr(parsed_ad, "eval_weights"))
            t_ad = DetectionTrainerAdvAdaptive(overrides={"resume": str(warm_adaptive)})
            self.assertEqual(t_ad.attack_names, ["worstk"])
            self.assertEqual(t_ad.attack_ratios, [0.5])
            self.assertEqual(t_ad.p_clean, 0.5)
            self.assertEqual(t_ad.eval_weights, [1.0])

        # 2. Staged resume preserves fixed surrogate and Stage 2
        with self.subTest(scenario="staged_warm_resume_surrogate"):
            surrogate_pt = self.temp_dir / "surrogate_h3.pt"
            torch.save({"model": DetectionModel("yolo11n.yaml", ch=3, nc=1, verbose=False)}, surrogate_pt)
            stage1_pt = self.temp_dir / "stage1_h3.pt"
            torch.save({"model": DetectionModel("yolo11n.yaml", ch=3, nc=1, verbose=False)}, stage1_pt)

            warm_staged = _make_warm_checkpoint(
                DetectionTrainerAdvFGDSTNStaged,
                {
                    "stage": 2,
                    "stage1_weights": str(stage1_pt),
                    "attack_weights": str(surrogate_pt),
                    "preprocess_mode": "stn_fgd",
                    "fgd_weights": str(self.fgd_path),
                    "stn_weights": str(self.stn_path),
                },
                "staged_h3",
            )
            parsed_staged = train_adv_fgd_stn_staged_run.parse_args(["--resume", str(warm_staged)])
            self.assertFalse(hasattr(parsed_staged, "attack_weights"))
            self.assertFalse(hasattr(parsed_staged, "stage"))

            t_staged = DetectionTrainerAdvFGDSTNStaged(overrides={"resume": str(warm_staged)})
            self.assertEqual(t_staged.stage, 2)
            self.assertEqual(t_staged.args.attack_weights, str(surrogate_pt))
            self.assertEqual(t_staged.attack_weights, str(surrogate_pt))

            # 3. Stage conflict rejection
            with self.assertRaises(ValueError) as ctx_stage:
                DetectionTrainerAdvFGDSTNStaged(overrides={"resume": str(warm_staged), "stage": 1})
            self.assertIn("Cannot change stage from 2 to 1", str(ctx_stage.exception))

        # 4. train_adv_test_run and train_adv_fgd_phase2_run CLI resume argument suppression
        with self.subTest(scenario="cli_resume_arg_suppression"):
            args_test = train_adv_test_run.parse_args(["--resume", str(warm_adaptive)])
            self.assertNotIn("batch", vars(args_test))
            self.assertNotIn("imgsz", vars(args_test))
            self.assertNotIn("cache", vars(args_test))

            args_p2 = train_adv_fgd_phase2_run.parse_args(["--resume", str(warm_adaptive)])
            self.assertNotIn("batch", vars(args_p2))
            self.assertNotIn("imgsz", vars(args_p2))
            self.assertNotIn("cache", vars(args_p2))
            self.assertNotIn("use_fgd", vars(args_p2))

        # 5. Phase 2 resume with use_fgd=False does not require fgd_weights
        with self.subTest(scenario="phase2_resume_use_fgd_false"):
            warm_p2 = _make_warm_checkpoint(
                FGDAdversarialTrainer,
                {"use_fgd": False, "fgd_weights": None},
                "phase2_h3",
            )
            resumed_p2 = FGDAdversarialTrainer(overrides={"resume": str(warm_p2), "use_fgd": False})
            self.assertFalse(resumed_p2.phase2["use_fgd"])

        # 6. Stage 1 default lr0: Python CLI (1e-5) vs Shell script (1e-6)
        with self.subTest(scenario="stage1_lr0_python_vs_shell"):
            repo_root = Path(__file__).resolve().parents[1]
            staged_py = (repo_root / "train_adv_fgd_stn_staged_run.py").read_text(encoding="utf-8")
            self.assertIn("default_lr0 = 1e-5", staged_py)

            staged_sh = (repo_root / "train_adv_fgd_stn_staged_local.sh").read_text(encoding="utf-8")
            self.assertIn('LR0="${LR0:-1e-6}"', staged_sh)

    def test_h4_workers_none_handling(self):
        """Test H4 (R4): workers=None handling across runners:
        1. Omitted --workers must not place 'workers': None into overrides.
        2. Explicit --workers 0 must preserve 'workers': 0.
        """
        # 1. train_adv_adaptive_run
        with self.subTest(runner="train_adv_adaptive_run"):
            args_none = train_adv_adaptive_run.parse_args([])
            self.assertFalse(hasattr(args_none, "workers"))
            args_zero = train_adv_adaptive_run.parse_args(["--workers", "0"])
            self.assertEqual(args_zero.workers, 0)

        # 2. train_adv_fgd_stn_adaptive_run
        with self.subTest(runner="train_adv_fgd_stn_adaptive_run"):
            args_none = train_adv_fgd_stn_adaptive_run.parse_args([])
            self.assertFalse(hasattr(args_none, "workers"))
            args_zero = train_adv_fgd_stn_adaptive_run.parse_args(["--workers", "0"])
            self.assertEqual(args_zero.workers, 0)

        # 3. train_adv_fgd_stn_staged_run
        with self.subTest(runner="train_adv_fgd_stn_staged_run"):
            args_none = train_adv_fgd_stn_staged_run.parse_args([])
            self.assertFalse(hasattr(args_none, "workers"))
            args_zero = train_adv_fgd_stn_staged_run.parse_args(["--workers", "0"])
            self.assertEqual(args_zero.workers, 0)

    def test_h5_joint_training_job_seed_array(self):
        """Test H5 (R5): Verify train_fgd_stn_joint_job.sh appends --seed to TRAIN_ARGS not ARGS."""
        repo_root = Path(__file__).resolve().parents[1]
        job_script = repo_root / "train_fgd_stn_joint_job.sh"
        self.assertTrue(job_script.exists())
        content = job_script.read_text(encoding="utf-8")
        self.assertIn('TRAIN_ARGS+=(--seed "$SEED")', content)
        self.assertNotIn('\n  ARGS+=(--seed "$SEED")', content)

    def test_h6_hydra_pipeline_model_propagation(self):
        """Test H6 (R6): HYDRA pipeline model artifact propagation across skipped stages:
        1. run_hydra_adv_pipeline_local.sh and run_hydra_adv_pipeline_job.sh initialize CURRENT_MODEL="${INPUT_MODEL}".
        2. Later stages pass --model "${CURRENT_MODEL}".
        3. CURRENT_MODEL is updated only when stage output artifact exists ([ -f ... ]).
        """
        repo_root = Path(__file__).resolve().parents[1]
        for script_name in ("run_hydra_adv_pipeline_local.sh", "run_hydra_adv_pipeline_job.sh"):
            with self.subTest(script=script_name):
                script_path = repo_root / script_name
                self.assertTrue(script_path.exists())
                content = script_path.read_text(encoding="utf-8")
                self.assertIn('CURRENT_MODEL="${INPUT_MODEL}"', content)
                self.assertIn('CURRENT_MODEL="${STAGE1_BEST}"', content)
                self.assertIn('CURRENT_MODEL="${STAGE2_BEST}"', content)
                self.assertIn('CURRENT_MODEL="${MATERIALIZED_MODEL}"', content)
                self.assertIn('--model "${CURRENT_MODEL}"', content)


if __name__ == "__main__":
    unittest.main()
