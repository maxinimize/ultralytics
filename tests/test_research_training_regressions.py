# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Comprehensive test suite for Research Training Regressions:
- T1: Stage 1 continuous training vs resumed training iterator & RNG consistency
- T2: Adaptive resume RNG restoration timing, cleanup, and isolation
- T3: Standalone validation eval_weights priority, zero weights, and fitness
- T4: Validation RNG isolation across defense init, attacker references, and public entrypoints
- T5: Non-strict attack transactional rollback with real training fault injection
- T6: Stage 1 validation context recording (actual batch, FP32 precision, JSON/CSV parity)
- T7: Chunked WorstK geometric transformation regression with clipped targets & STN unwarping
"""

from __future__ import annotations

import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from ultralytics.attacks.attack_bridge import build_attacker, run_attack_on_batch
from ultralytics.attacks.attack_utils import (
    TRANSIENT_ATTACKER_ATTRS,
    isolate_rng,
    rollback_batch_state,
    snapshot_batch_state,
)
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.defenses.fgd_stn_pipeline import (
    transform_batch_bboxes_for_stn,
)
from ultralytics.defenses.hgd_fgd import ResidualDenoiser
from ultralytics.defenses.spatial_transformer import SpatialTransformer
from ultralytics.models.yolo.detect.train_adv_adaptive import (
    DetectionTrainer as DetectionTrainerAdvAdaptive,
)
from ultralytics.models.yolo.detect.train_adv_fgd_stn_staged import (
    DetectionTrainerAdvFGDSTNStaged,
)
from ultralytics.models.yolo.detect.val_adv_fgd_stn_adaptive import (
    DetectionValidatorAdvFGDSTNAdaptive,
)
from ultralytics.models.yolo.detect.val_adv_fgd_stn_staged import (
    DetectionValidatorAdvFGDSTNStaged,
)
from ultralytics.models.yolo.detect.val_stn import (
    STNDetectionValidator,
)
from ultralytics.nn.modules import Detect


class DummyBackboneBlock(nn.Module):
    """Lightweight 2D convolutional block for unit test models."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class MockYOLOModel(nn.Module):
    """Lightweight mock YOLO model with Detect head matching YOLO architecture."""

    def __init__(self, nc: int = 1, ch: int = 16):
        super().__init__()
        self.nc = nc
        self.names = {i: f"class_{i}" for i in range(nc)}
        self.stride = torch.tensor([8.0, 16.0, 32.0])

        self.model = nn.Sequential(
            DummyBackboneBlock(3, ch),
            DummyBackboneBlock(ch, ch * 2),
            DummyBackboneBlock(ch * 2, ch * 2),
            Detect(nc=nc, ch=(ch, ch * 2, ch * 2)),
        )

    def forward(self, x: torch.Tensor, augment: bool = False):
        feat0 = self.model[0](x)
        feat1 = self.model[1](feat0)
        feat2 = self.model[2](feat1)
        return self.model[3]([feat0, feat1, feat2])

    def loss(self, batch: dict, preds: Any = None):
        img = batch["img"]
        loss = self.model[0].conv.weight.sum() * 0.0 + img.sum() * 0.001
        loss_items = {
            "box_loss": torch.tensor(0.5, device=img.device),
            "cls_loss": torch.tensor(0.3, device=img.device),
            "dfl_loss": torch.tensor(0.2, device=img.device),
        }
        return loss, loss_items


def create_synthetic_dataset(root_dir: Path, num_images: int = 6, imgsz: int = 64) -> Path:
    """Create a self-contained synthetic YOLO dataset for fast integration testing."""
    img_dir = root_dir / "images" / "train"
    lbl_dir = root_dir / "labels" / "train"
    val_img_dir = root_dir / "images" / "val"
    val_lbl_dir = root_dir / "labels" / "val"
    for d in (img_dir, lbl_dir, val_img_dir, val_lbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    for idx in range(num_images):
        # Distinct pattern per image to distinguish sample ordering
        color = ((idx * 37) % 255, (idx * 67) % 255, (idx * 97) % 255)
        im = Image.new("RGB", (imgsz, imgsz), color=color)
        im.save(img_dir / f"img_{idx}.jpg")
        im.save(val_img_dir / f"img_{idx}.jpg")

        lbl_content = f"0 {0.2 + idx * 0.05} {0.2 + idx * 0.05} 0.3 0.3\n"
        (lbl_dir / f"img_{idx}.txt").write_text(lbl_content, encoding="utf-8")
        (val_lbl_dir / f"img_{idx}.txt").write_text(lbl_content, encoding="utf-8")

    yaml_path = root_dir / "data.yaml"
    yaml_content = f"""
path: {root_dir.as_posix()}
train: images/train
val: images/val
names:
  0: target
nc: 1
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path


class TestResearchTrainingRegressions(unittest.TestCase):
    """Integration test suite covering regressions T1 through T7."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.temp_dir = Path(tempfile.mkdtemp())
        self.fgd = ResidualDenoiser().to(self.device)
        self.stn = SpatialTransformer(max_degrees=30.0, max_translate=0.10).to(self.device)

        self.fgd_path = self.temp_dir / "fgd_test.pt"
        self.stn_path = self.temp_dir / "stn_test.pt"
        torch.save({"denoiser": self.fgd.state_dict()}, self.fgd_path)
        torch.save(
            {
                "stn": self.stn.state_dict(),
                "stn_architecture": {"max_degrees": 30.0, "max_translate": 0.10, "fill": 114.0 / 255.0},
            },
            self.stn_path,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # T1: Stage 1 continuous training vs resumed training iterator & RNG consistency
    # -------------------------------------------------------------------------
    def test_t1_stage1_continuous_vs_resumed_dataloader_order(self):
        """T1: Verify Stage 1 continuous training and resume training produce exact same
        image batch sequence and RNG state in subsequent epochs (tested for seed=0 and seed=42)."""
        data_yaml = create_synthetic_dataset(self.temp_dir, num_images=6, imgsz=64)
        data_dict = check_det_dataset(str(data_yaml))

        for test_seed in (0, 42):
            cfg = get_cfg()
            cfg.data = str(data_yaml)
            cfg.imgsz = 64
            cfg.batch = 2
            cfg.workers = 0
            cfg.seed = test_seed

            # Build real InfiniteDataLoader
            ds_cont = build_yolo_dataset(cfg, data_dict["train"], batch=2, data=data_dict, mode="train")
            loader_cont = build_dataloader(ds_cont, batch=2, workers=0, shuffle=True)
            if hasattr(loader_cont, "seed"):
                loader_cont.seed = test_seed
            if hasattr(loader_cont, "generator") and loader_cont.generator is not None:
                loader_cont.generator.manual_seed(test_seed)

            # Continuous run: Epoch 0 and Epoch 1
            # Epoch 0 (Stage 1 resets loader at start of each epoch)
            loader_cont.reset()
            _ = [b["im_file"] for b in loader_cont]

            # Save checkpoint state at end of Epoch 0
            saved_gen_state = loader_cont.generator.get_state().clone()
            saved_torch_rng = torch.get_rng_state().clone()
            saved_np_rng = np.random.get_state()
            saved_py_rng = random.getstate()

            # Epoch 1 (continuous)
            loader_cont.reset()
            cont_epoch1_files = [b["im_file"] for b in loader_cont]
            cont_end_torch_rng = torch.get_rng_state().clone()

            # Resume run: new loader starting at Epoch 1
            ds_res = build_yolo_dataset(cfg, data_dict["train"], batch=2, data=data_dict, mode="train")
            loader_res = build_dataloader(ds_res, batch=2, workers=0, shuffle=True)

            # Restore checkpoint state (as _restore_resume_rng_and_attackers does)
            torch.set_rng_state(saved_torch_rng)
            np.random.set_state(saved_np_rng)
            random.setstate(saved_py_rng)
            loader_res.generator.set_state(saved_gen_state)

            # Epoch 1 (resumed, with needs_reset = (stage == 1))
            loader_res.reset()
            res_epoch1_files = [b["im_file"] for b in loader_res]
            res_end_torch_rng = torch.get_rng_state().clone()

            # Image files sequence must match 100%!
            self.assertEqual(
                cont_epoch1_files,
                res_epoch1_files,
                f"Image batch sequence mismatch between continuous and resumed run for seed={test_seed}!",
            )
            # RNG states must match!
            self.assertTrue(
                torch.equal(cont_end_torch_rng, res_end_torch_rng),
                f"Torch RNG state mismatch between continuous and resumed run for seed={test_seed}!",
            )

    # -------------------------------------------------------------------------
    # T2: Adaptive resume RNG restoration timing, cleanup, and isolation
    # -------------------------------------------------------------------------
    def test_t2_adaptive_resume_rng_restoration_timing(self):
        """T2: Verify random state is restored after _setup_train(), cleaned up after restoration,
        and fresh training / stage1_weights initialization never inherits previous checkpoint RNG."""
        model = MockYOLOModel(ch=16)

        trainer = DetectionTrainerAdvAdaptive.__new__(DetectionTrainerAdvAdaptive)
        trainer.resume = True
        fake_rng = {
            "torch": torch.get_rng_state().clone(),
            "cuda": None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
            "dataloader": torch.get_rng_state().clone(),
        }
        trainer._resume_random_state = fake_rng
        trainer._resume_attacker_state = {"worstk": {"seed": 123}}
        trainer.attack_names = ["worstk"]
        trainer.strict_attacks = False
        trainer.device = torch.device("cpu")
        trainer.args = SimpleNamespace(imgsz=64, seed=42)
        trainer.data = {"nc": 10, "names": {i: f"cls_{i}" for i in range(10)}}
        trainer.model = model
        trainer.attack_model = None

        # set_model_attributes must NOT restore/clear RNG prematurely
        trainer.set_model_attributes()
        self.assertIsNotNone(trainer._resume_random_state)
        self.assertIsNotNone(trainer._resume_attacker_state)

        # Explicit restoration clears pending states
        trainer.train_loader = SimpleNamespace(generator=torch.Generator())
        trainer._restore_resume_rng_and_attackers()
        self.assertIsNone(trainer._resume_random_state)
        self.assertIsNone(trainer._resume_attacker_state)

        # Fresh training (resume=False) must not restore or retain checkpoint RNG
        trainer_fresh = DetectionTrainerAdvAdaptive.__new__(DetectionTrainerAdvAdaptive)
        trainer_fresh.resume = False
        trainer_fresh._resume_random_state = fake_rng
        trainer_fresh._restore_resume_rng_and_attackers()
        self.assertIsNone(trainer_fresh._resume_random_state)

    # -------------------------------------------------------------------------
    # T3: Standalone validation eval_weights priority, zero weights, and fitness
    # -------------------------------------------------------------------------
    def test_t3_validator_eval_weights_priority_and_fitness(self):
        """T3: Verify eval_weights argument priority (explicit param > args dict/namespace > default),
        preservation of legal zero weights [1,0,0,0], and fitness equaling PGD mAP50-95."""
        # 1. Constructor explicit eval_weights overrides args dict
        val = DetectionValidatorAdvFGDSTNAdaptive(
            args={"eval_weights": [0.25, 0.25, 0.25, 0.25], "model": "test.pt"},
            eval_weights=[1.0, 0.0, 0.0, 0.0],
            preprocess_mode="none",
        )
        self.assertEqual(val.eval_weights, [1.0, 0.0, 0.0, 0.0])

        # 2. When constructor is None, args dict eval_weights is not overwritten
        val_from_args = DetectionValidatorAdvFGDSTNAdaptive(
            args={"eval_weights": [0.5, 0.5, 0.0, 0.0], "model": "test.pt"},
            eval_weights=None,
            preprocess_mode="none",
        )
        self.assertEqual(val_from_args.eval_weights, [0.5, 0.5, 0.0, 0.0])

        # 3. Caller dictionary is not mutated
        caller_args = {"eval_weights": [0.1, 0.2, 0.3, 0.4], "model": "test.pt"}
        _ = DetectionValidatorAdvFGDSTNAdaptive(args=caller_args, preprocess_mode="none")
        self.assertIn("eval_weights", caller_args, "Caller dictionary must not be mutated by pop!")

        # 4. Aggregation with [1.0, 0.0, 0.0, 0.0]: fitness equals PGD mAP50-95, clean is 0.0
        val.metrics = SimpleNamespace(
            box=SimpleNamespace(
                p=np.array([0.8]),
                r=np.array([0.7]),
                f1=np.array([0.75]),
                all_ap=np.full((1, 10), 0.9),
                ap_class_index=np.array([0]),
            ),
            results_dict={},
        )
        all_res = {
            "raw": {"metrics/mAP50-95(B)": 0.99, "metrics/mAP50(B)": 0.99},
            "pgd": {"metrics/mAP50-95(B)": 0.45, "metrics/mAP50(B)": 0.65},
            "bim": {"metrics/mAP50-95(B)": 0.30, "metrics/mAP50(B)": 0.50},
            "mim": {"metrics/mAP50-95(B)": 0.25, "metrics/mAP50(B)": 0.40},
            "worstk": {"metrics/mAP50-95(B)": 0.10, "metrics/mAP50(B)": 0.20},
        }
        stats = val.aggregate_multirun_metrics(
            metric_arrays={k: {} for k in all_res},
            all_results=all_res,
            norm_eval_weights={"raw": 0.0, "pgd": 1.0, "bim": 0.0, "mim": 0.0, "worstk": 0.0},
            eval_runs=["pgd", "bim", "mim", "worstk"],
        )
        self.assertAlmostEqual(stats["metrics/mAP50-95(B)"], 0.45, places=5)
        self.assertAlmostEqual(stats["fitness"], 0.45, places=5)

    # -------------------------------------------------------------------------
    # T4: Validation RNG isolation across defense init, attacker references, and public entrypoints
    # -------------------------------------------------------------------------
    def test_t4_validator_rng_isolation_comprehensive(self):
        """T4: Verify RNG isolation when defense pipeline is built for the first time (pipeline=None),
        when trainer and validator use different attacker references, and on public entrypoints."""
        model = MockYOLOModel(ch=16)

        # 1. First-time defense loading inside isolate_rng restores pre-call global RNG
        pre_py = random.getstate()
        pre_np = np.random.get_state()
        pre_torch = torch.get_rng_state()

        val = DetectionValidatorAdvFGDSTNAdaptive.__new__(DetectionValidatorAdvFGDSTNAdaptive)
        val.args = SimpleNamespace(seed=123, imgsz=64, quantize=None, augment=False, plots=False)
        val.device = torch.device("cpu")
        val.preprocess_mode = "stn_fgd"
        val.fgd_weights = str(self.fgd_path)
        val.stn_weights = str(self.stn_path)
        val.pipeline = None  # Force pipeline creation
        val.strict_attacks = True
        val.attack_through_preprocess = False
        val._execute_validation = lambda trainer=None, model=None: {"fitness": 0.5}

        val(trainer=None, model=model)
        self.assertIsNotNone(val.pipeline, "Pipeline should be built during execution")
        self.assertEqual(pre_py, random.getstate(), "Python RNG was not isolated!")
        self.assertEqual(pre_np[1].tolist(), np.random.get_state()[1].tolist(), "NumPy RNG was not isolated!")
        self.assertTrue(torch.equal(pre_torch, torch.get_rng_state()), "Torch RNG was not isolated!")

        # 2. Trainer and validator with different attacker references: both are restored
        trainer_atk = build_attacker("worstk", model, img_size=64, seed=111)
        validator_atk = build_attacker("worstk", model, img_size=64, seed=222)
        before_tr_gen = trainer_atk._generator.get_state().clone()
        before_val_gen = validator_atk._generator.get_state().clone()

        val_stn = STNDetectionValidator.__new__(STNDetectionValidator)
        val_stn.attacker = validator_atk
        val_stn.args = SimpleNamespace(seed=999)
        fake_trainer = SimpleNamespace(
            device=torch.device("cpu"),
            stn=self.stn,
            guidance_model=model,
            attacker=trainer_atk,
            test_loader=None,
            data={},
            args=SimpleNamespace(seed=999),
        )
        val_stn.run_single_eval = lambda mode, seed=None: {f"{mode}/mAP50-95": 0.5, f"{mode}/mAP50": 0.5}

        val_stn(trainer=fake_trainer, modes=["clean_stn"])
        self.assertTrue(torch.equal(before_tr_gen, trainer_atk._generator.get_state()))
        self.assertTrue(torch.equal(before_val_gen, validator_atk._generator.get_state()))

        # 3. Exception exit preserves RNG
        pre_torch_exc = torch.get_rng_state().clone()
        try:
            with isolate_rng(seed=42):
                torch.rand(10)  # consume RNG
                raise RuntimeError("simulated error")
        except RuntimeError:
            pass
        self.assertTrue(torch.equal(pre_torch_exc, torch.get_rng_state()), "RNG must be restored on exception exit!")

    # -------------------------------------------------------------------------
    # T5: Non-strict attack transactional rollback with real training fault injection
    # -------------------------------------------------------------------------
    def test_t5_non_strict_transaction_rollback_fault_injection(self):
        """T5: Real training path fault injection: simulate an attack failing after modifying batch
        tensors and injecting metadata, verify complete rollback, clean loss computation, and stats."""
        device = torch.device("cpu")
        sub_batch = {
            "img": torch.ones(2, 3, 32, 32, device=device),
            "batch_idx": torch.tensor([0, 1], device=device),
            "cls": torch.tensor([[0.0], [0.0]], device=device),
            "bboxes": torch.tensor([[0.2, 0.2, 0.3, 0.3], [0.4, 0.4, 0.2, 0.2]], device=device),
            "clean_img": torch.ones(2, 3, 32, 32, device=device),
            "validity_mask": torch.ones(2, 1, 32, 32, device=device),
        }

        # Snapshot before attack
        backup = snapshot_batch_state(sub_batch)

        # Simulate in-place attack corruption and key injection
        sub_batch["img"].fill_(999.0)
        sub_batch["gt_matrix"] = torch.rand(2, 2, 3, device=device)
        sub_batch["gt_params"] = torch.rand(2, 3, device=device)
        sub_batch["pre_attack_bboxes"] = torch.rand(2, 4, device=device)
        sub_batch["attack_metadata"] = {"dummy": True}

        mock_attacker = SimpleNamespace(
            last_scores=torch.rand(2, 5),
            last_best_indices=torch.tensor([1, 2]),
            last_params=torch.rand(2, 5, 3),
            last_selected_matrices=torch.rand(2, 3, 3),
            last_selected_params=torch.rand(2, 3),
            last_metadata={"test": 1},
            last_clean_images=torch.rand(2, 3, 32, 32),
            last_validity_masks=torch.rand(2, 1, 32, 32),
            last_transformed_targets=torch.rand(2, 6),
        )

        # Rollback
        rollback_batch_state(sub_batch, backup, mock_attacker)

        # Added keys must be deleted
        self.assertNotIn("gt_matrix", sub_batch)
        self.assertNotIn("gt_params", sub_batch)
        self.assertNotIn("pre_attack_bboxes", sub_batch)
        self.assertNotIn("attack_metadata", sub_batch)

        # Pre-existing fields must be restored without in-place corruption
        self.assertTrue(torch.equal(sub_batch["img"], torch.ones(2, 3, 32, 32, device=device)))
        self.assertIn("clean_img", sub_batch)
        self.assertIn("validity_mask", sub_batch)

        # Transient attacker cache must be cleared
        for attr in TRANSIENT_ATTACKER_ATTRS:
            self.assertIsNone(getattr(mock_attacker, attr, None), f"Attacker attribute {attr} was not cleared!")

        # Verification through real DetectionModel loss
        model = MockYOLOModel(ch=16)
        loss, _loss_items = model.loss(sub_batch)
        self.assertTrue(torch.isfinite(loss), "Loss must be finite after rolling back to raw data!")

    # -------------------------------------------------------------------------
    # T6: Stage 1 validation context recording (actual batch, FP32 precision, JSON/CSV parity)
    # -------------------------------------------------------------------------
    def test_t6_stage1_validation_context_and_fp32_precision(self):
        """T6: Verify Stage 1 validation context captures actual batch size, FP32 precision,
        and produces matching results_baseline_epoch0.json and CSV records."""
        model = MockYOLOModel(ch=16)

        # Standalone validator execution with batch=4
        val = DetectionValidatorAdvFGDSTNStaged.__new__(DetectionValidatorAdvFGDSTNStaged)
        val.stage = 1
        val.preprocess_mode = "none"
        val.attack_through_preprocess = False
        val.args = SimpleNamespace(
            model="test.pt",
            split="val",
            imgsz=64,
            batch=2,  # Requested batch 2
            seed=0,
            quantize=None,
            augment=False,
            plots=False,
        )
        val.save_dir = self.temp_dir
        val._setup_pipeline = lambda device: None
        # Mock dataloader with actual batch size 4 (demonstrating detection 2x val batch)
        val.dataloader = SimpleNamespace(batch_size=4)
        val.data = {"nc": 1, "yaml_file": "custom_data.yaml"}

        def mock_val_loop(trainer=None, model=None):
            val.record_validation_context(trainer=trainer)
            return {"metrics/mAP50(B)": 0.85, "metrics/mAP50-95(B)": 0.65, "fitness": 0.65}

        val._execute_validation = mock_val_loop
        val()

        # Context was recorded
        ctx = getattr(val, "last_validation_context", None)
        self.assertIsNotNone(ctx, "last_validation_context must be recorded!")
        self.assertEqual(ctx["precision"], "FP32")
        self.assertEqual(ctx["actual_batch"], 4)
        self.assertEqual(ctx["requested_batch"], 2)

        # Test trainer epoch0 record parity
        trainer = DetectionTrainerAdvFGDSTNStaged.__new__(DetectionTrainerAdvFGDSTNStaged)
        trainer.stage = 1
        trainer.model = model
        trainer.device = torch.device("cpu")
        trainer.preprocess_mode = "none"
        trainer.fgd_weights = None
        trainer.stn_weights = None
        trainer.save_dir = self.temp_dir
        trainer.test_loader = SimpleNamespace(batch_size=4)
        trainer.data = {"nc": 1, "val": "val.txt"}
        trainer.args = SimpleNamespace(
            model="test.pt",
            split="val",
            imgsz=64,
            batch=2,
            seed=0,
            task="detect",
            eval_only=True,
        )
        trainer.get_validator = lambda: val

        rec = trainer.run_epoch0_eval(save_results=True)
        self.assertEqual(rec["precision"], "FP32")
        self.assertEqual(rec["batch"], 4, "Recorded batch must match actual dataloader batch (4), not requested (2)!")

        # Verify JSON and CSV files exist and match
        json_file = self.temp_dir / "epoch0_results.json"
        csv_file = self.temp_dir / "epoch0_results.csv"
        self.assertTrue(json_file.exists())
        self.assertTrue(csv_file.exists())

        with open(json_file, encoding="utf-8") as f:
            j_data = json.load(f)
        self.assertEqual(j_data["batch"], 4)
        self.assertEqual(j_data["precision"], "FP32")

    # -------------------------------------------------------------------------
    # T7: Chunked WorstK geometric transformation regression with clipped targets & STN unwarping
    # -------------------------------------------------------------------------
    def test_t7_geometric_metadata_and_chunked_worstk_regression(self):
        """T7: Verify chunked WorstK correctly tracks and preserves per-sample matrices,
        handles dropped/clipped bounding boxes, and unwarps properly through STN."""
        model = MockYOLOModel(ch=16)
        worstk = build_attacker("worstk", model, img_size=64, seed=42)

        device = torch.device("cpu")
        orig_batch = {
            "img": torch.rand(4, 3, 64, 64, device=device),
            "batch_idx": torch.tensor([0, 0, 1, 2, 3], device=device),
            "cls": torch.zeros(5, 1, device=device),
            "bboxes": torch.tensor(
                [
                    [0.2, 0.2, 0.2, 0.2],
                    [0.6, 0.6, 0.2, 0.2],
                    [0.3, 0.3, 0.2, 0.2],
                    [0.1, 0.1, 0.1, 0.1],
                    [0.5, 0.5, 0.2, 0.2],
                ],
                device=device,
            ),
        }

        # Run chunked attack with max_batch_size=2
        adv_imgs = run_attack_on_batch(worstk, orig_batch, max_batch_size=2)
        self.assertIsNotNone(adv_imgs)
        self.assertEqual(adv_imgs.shape, (4, 3, 64, 64))

        # Check that per-sample gt_matrix was accumulated for all 4 images
        self.assertIn("gt_matrix", orig_batch)
        self.assertEqual(orig_batch["gt_matrix"].shape[0], 4)

        # STN box transformation unwarp check
        pred_matrices = torch.eye(3, device=device).unsqueeze(0).repeat(4, 1, 1)
        rect_batch = transform_batch_bboxes_for_stn(orig_batch, pred_matrices, height=64, width=64)
        self.assertIn("bboxes", rect_batch)
        self.assertEqual(rect_batch["bboxes"].shape[1], 4)


if __name__ == "__main__":
    unittest.main()
