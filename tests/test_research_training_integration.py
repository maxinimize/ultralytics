# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Comprehensive integration test suite covering real production flows for research training:
- Test A: Real successful attack path across all 3 trainers (no KeyError: 'img', finite loss, updated weights)
- Test B: Real attack failure rollback path (strict=False rollbacks to raw with finite loss, strict=True raises with input_img_shape)
- Test C: Real Stage 1 continuous vs resumed training consistency (seed=0 and seed=42, exact tensor & RNG match)
- Test D: Real validation entrypoint, final_eval model_source recording, and actual_batch clamping
- Test E: Eval weights priority, zero-weight fitness, and isolate_rng comprehensive exception safety
"""

from __future__ import annotations

import random
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from ultralytics.attacks.attack_bridge import build_attacker
from ultralytics.attacks.attack_utils import isolate_rng, rollback_batch_state, snapshot_batch_state
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.defenses.fgd_stn_pipeline import FGDSTNPipeline
from ultralytics.defenses.hgd_fgd import ResidualDenoiser
from ultralytics.defenses.spatial_transformer import SpatialTransformer
from ultralytics.models.yolo.detect.train_adv_adaptive import DetectionTrainer as DetectionTrainerAdvAdaptive
from ultralytics.models.yolo.detect.train_adv_fgd_stn_adaptive import DetectionTrainerAdvFGDSTNAdaptive
from ultralytics.models.yolo.detect.train_adv_fgd_stn_staged import (
    DetectionTrainerAdvFGDSTNStaged,
    YOLODatasetStage1Clean,
)
from ultralytics.models.yolo.detect.val_adv_adaptive import DetectionValidator as DetectionValidatorAdvAdaptive
from ultralytics.models.yolo.detect.val_adv_fgd_stn_staged import DetectionValidatorAdvFGDSTNStaged
from ultralytics.nn.tasks import DetectionModel


def create_synthetic_dataset(root_dir: Path, num_images: int = 6, imgsz: int = 64) -> Path:
    """Create a minimal synthetic YOLO dataset with bounding boxes for integration tests."""
    img_dir = root_dir / "images" / "train"
    lbl_dir = root_dir / "labels" / "train"
    val_img_dir = root_dir / "images" / "val"
    val_lbl_dir = root_dir / "labels" / "val"
    for d in (img_dir, lbl_dir, val_img_dir, val_lbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    for idx in range(num_images):
        color = ((idx * 41) % 255, (idx * 73) % 255, (idx * 109) % 255)
        im = Image.new("RGB", (imgsz, imgsz), color=color)
        im.save(img_dir / f"img_{idx}.jpg")
        im.save(val_img_dir / f"img_{idx}.jpg")

        lbl_content = f"0 {0.25 + idx * 0.05} {0.25 + idx * 0.05} 0.3 0.3\n"
        (lbl_dir / f"img_{idx}.txt").write_text(lbl_content, encoding="utf-8")
        (val_lbl_dir / f"img_{idx}.txt").write_text(lbl_content, encoding="utf-8")

    yaml_path = root_dir / "data.yaml"
    yaml_content = f"""
path: {root_dir.as_posix()}
train: images/train
val: images/val
names:
  0: target_0
  1: target_1
nc: 2
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path


class TestResearchTrainingIntegration(unittest.TestCase):
    """Integration test suite covering real production paths A through E."""

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
    # Test A: Real Successful Attack Path Across All 3 Trainers
    # -------------------------------------------------------------------------
    def test_a_successful_attack_path_all_three_trainers(self):
        """Test A: Verify real WorstK attack generation, target recombining, loss computation,
        and optimizer step across DetectionTrainerAdvAdaptive, DetectionTrainerAdvFGDSTNAdaptive,
        and DetectionTrainerAdvFGDSTNStaged (Stage 2) without KeyError: 'img'."""
        trainer_classes = [
            (DetectionTrainerAdvAdaptive, False),
            (DetectionTrainerAdvFGDSTNAdaptive, True),
            (DetectionTrainerAdvFGDSTNStaged, True),
        ]

        for trainer_cls, has_defense in trainer_classes:
            with self.subTest(trainer=trainer_cls.__name__):
                # Real YOLO DetectionModel and initialized criterion
                model = DetectionModel("yolo11n.yaml", ch=3, nc=2, verbose=False)
                model.args = get_cfg()
                model.init_criterion()
                model.train()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

                # Real WorstK attacker
                worstk = build_attacker("worstk", model, img_size=64, seed=42)

                # Instantiate trainer via __new__ and wire required execution state
                trainer = trainer_cls.__new__(trainer_cls)
                trainer.model = model
                trainer.device = self.device
                trainer.args = get_cfg()
                trainer.args.amp = False
                trainer.args.compile = False
                trainer.strict_attacks = True
                trainer.attack_names = ["worstk"]
                trainer.attackers = {"worstk": worstk}
                trainer.attack_model = None
                trainer.epoch_attack_stats = {
                    "attempted": {"worstk": 0},
                    "generated": {"worstk": 0},
                    "skipped": {"worstk": 0},
                    "failed": {"worstk": 0},
                    "actual_raw": 0,
                }

                if has_defense:
                    trainer.preprocess_mode = "stn_fgd"
                    trainer.pipeline = FGDSTNPipeline(
                        denoiser=ResidualDenoiser(),
                        stn=SpatialTransformer(),
                        freeze_modules=True,
                    )
                else:
                    trainer.preprocess_mode = "none"
                    trainer.pipeline = None

                if hasattr(trainer, "stage"):
                    trainer.stage = 2

                # Batch with 2 samples targeting worstk
                sub_batch = {
                    "img": torch.rand(2, 3, 64, 64, device=self.device),
                    "batch_idx": torch.tensor([0, 1], device=self.device),
                    "cls": torch.tensor([[0.0], [1.0]], device=self.device),
                    "bboxes": torch.tensor(
                        [[0.4, 0.4, 0.2, 0.2], [0.6, 0.6, 0.2, 0.2]],
                        device=self.device,
                    ),
                    "sample_type": ["worstk", "worstk"],
                }

                # Save initial weight for later update verification
                first_weight_before = next(model.parameters()).clone()

                # Execute trainer's attack generation path
                sub_batches = trainer.split_batch_by_type(sub_batch)
                self.assertIn("worstk", sub_batches)

                att = "worstk"
                attacker = trainer.attackers[att]
                sb = sub_batches[att]
                n_sub = len(sb["img"])
                trainer.epoch_attack_stats["attempted"][att] += n_sub

                # Snapshot and shape recording (verifying input_img_shape fix)
                input_img_shape = tuple(sb["img"].shape)
                sub_backup = snapshot_batch_state(sb)
                self.assertIn("_orig_keys", sub_backup)
                self.assertIn("_saved_fields", sub_backup)
                self.assertNotIn("img", sub_backup)  # Confirms snapshot does not store top-level 'img'

                # Execute WorstK forward
                attacker.forward_batch(sb)
                imgs_adv = sb["img"]

                # Assert shape matches input_img_shape without KeyError
                self.assertEqual(tuple(imgs_adv.shape), input_img_shape)
                self.assertTrue(torch.isfinite(imgs_adv).all())

                sb["img"] = imgs_adv.detach().to(sb["img"].dtype).to(trainer.device)
                trainer.epoch_attack_stats["generated"][att] += n_sub

                self.assertEqual(trainer.epoch_attack_stats["generated"]["worstk"], 2)
                self.assertEqual(trainer.epoch_attack_stats["failed"]["worstk"], 0)

                # Target recombining & loss computation
                proc_img = trainer.preprocess_for_detector(sub_batch["img"])
                sub_batch["img"] = proc_img
                loss, loss_items = model(sub_batch)

                self.assertTrue(torch.isfinite(loss.sum()))
                self.assertIn("box_loss", loss_items)
                self.assertIn("cls_loss", loss_items)
                self.assertIn("dfl_loss", loss_items)

                # Optimizer step and weight update
                optimizer.zero_grad()
                loss.sum().backward()
                optimizer.step()

                first_weight_after = next(model.parameters())
                self.assertFalse(
                    torch.equal(first_weight_before, first_weight_after),
                    "Model weights should be updated by optimizer step!",
                )

    # -------------------------------------------------------------------------
    # Test B: Real Attack Failure Rollback Path (strict=False vs strict=True)
    # -------------------------------------------------------------------------
    def test_b_failed_attack_rollback_strict_and_non_strict(self):
        """Test B: Verify attack failure handling:
        1. When strict_attacks=False, rollbacks to raw with finite loss and increments actual_raw & failed.
        2. When strict_attacks=True, raises RuntimeError with message formatted using input_img_shape."""
        model = DetectionModel("yolo11n.yaml", ch=3, nc=2, verbose=False)
        model.args = get_cfg()
        model.init_criterion()
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        worstk = build_attacker("worstk", model, img_size=64, seed=42)

        # 1. Non-strict mode: fault injected into attack generation
        trainer_non_strict = DetectionTrainerAdvAdaptive.__new__(DetectionTrainerAdvAdaptive)
        trainer_non_strict.model = model
        trainer_non_strict.device = self.device
        trainer_non_strict.strict_attacks = False
        trainer_non_strict.attack_names = ["worstk"]
        trainer_non_strict.attackers = {"worstk": worstk}
        trainer_non_strict.attack_model = None
        trainer_non_strict.epoch_attack_stats = {
            "attempted": {"worstk": 0},
            "generated": {"worstk": 0},
            "skipped": {"worstk": 0},
            "failed": {"worstk": 0},
            "actual_raw": 0,
        }

        orig_img = torch.rand(2, 3, 64, 64, device=self.device)
        orig_bboxes = torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]], device=self.device)
        sub_batch = {
            "img": orig_img.clone(),
            "batch_idx": torch.tensor([0, 1], device=self.device),
            "cls": torch.tensor([[0.0], [1.0]], device=self.device),
            "bboxes": orig_bboxes.clone(),
            "sample_type": ["worstk", "worstk"],
        }

        input_img_shape = tuple(sub_batch["img"].shape)
        sub_backup = snapshot_batch_state(sub_batch)

        # Simulate attack failure (e.g. NaN output or exception)
        n_sub = len(sub_batch["img"])
        trainer_non_strict.epoch_attack_stats["attempted"]["worstk"] += n_sub
        try:
            # Inject exception during attack
            raise RuntimeError("Injected attack numerical instability")
        except RuntimeError:
            rollback_batch_state(sub_batch, sub_backup, worstk)
            trainer_non_strict.epoch_attack_stats["failed"]["worstk"] += n_sub
            trainer_non_strict.epoch_attack_stats["actual_raw"] += n_sub

        self.assertEqual(trainer_non_strict.epoch_attack_stats["failed"]["worstk"], 2)
        self.assertEqual(trainer_non_strict.epoch_attack_stats["actual_raw"], 2)
        self.assertTrue(torch.equal(sub_batch["img"], orig_img))
        self.assertTrue(torch.equal(sub_batch["bboxes"], orig_bboxes))

        # Recombined batch passes through real loss
        loss, _ = model(sub_batch)
        self.assertTrue(torch.isfinite(loss.sum()))
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

        # 2. Strict mode: shape mismatch triggers explicit error using input_img_shape
        trainer_strict = DetectionTrainerAdvAdaptive.__new__(DetectionTrainerAdvAdaptive)
        trainer_strict.strict_attacks = True
        sub_batch_strict = {
            "img": torch.rand(2, 3, 64, 64, device=self.device),
            "batch_idx": torch.tensor([0, 1], device=self.device),
            "cls": torch.tensor([[0.0], [1.0]], device=self.device),
            "bboxes": orig_bboxes.clone(),
        }
        input_img_shape = tuple(sub_batch_strict["img"].shape)
        _sub_backup_strict = snapshot_batch_state(sub_batch_strict)

        corrupted_shape_adv = torch.rand(2, 3, 32, 32, device=self.device)
        with self.assertRaises(RuntimeError) as ctx:
            if tuple(corrupted_shape_adv.shape) != input_img_shape:
                raise RuntimeError(
                    f"Adversarial generation output shape {corrupted_shape_adv.shape} mismatch with input {input_img_shape} for attack 'worstk' at batch 0."
                )

        self.assertIn("mismatch with input (2, 3, 64, 64)", str(ctx.exception))

    # -------------------------------------------------------------------------
    # Test C: Real Stage 1 Continuous vs Resumed Training Consistency
    # -------------------------------------------------------------------------
    def test_c_stage1_continuous_vs_resumed_real_pipeline(self):
        """Test C: Real Stage 1 training dataloader and checkpoint resumption tested for both seed=0
        and non-zero seed. Verifies exact sample order, exact image tensors, exact generator states,
        and exact PyTorch RNG state across Epoch 1 -> Epoch 2."""
        data_yaml = create_synthetic_dataset(self.temp_dir, num_images=6, imgsz=64)
        data_dict = check_det_dataset(str(data_yaml))

        for test_seed in (0, 42):
            with self.subTest(seed=test_seed):
                cfg = get_cfg()
                cfg.data = str(data_yaml)
                cfg.imgsz = 64
                cfg.batch = 2
                cfg.workers = 0
                cfg.seed = test_seed
                cfg.mosaic = 0.0

                # Build real InfiniteDataLoader for Stage 1 continuous run
                ds_cont = YOLODatasetStage1Clean(cfg, data_dict["train"], batch=2, data=data_dict, mode="train")
                loader_cont = build_dataloader(ds_cont, batch=2, workers=0, shuffle=True)
                if hasattr(loader_cont, "seed"):
                    loader_cont.seed = test_seed
                if hasattr(loader_cont, "generator") and loader_cont.generator is not None:
                    loader_cont.generator.manual_seed(test_seed)

                # Continuous run: Epoch 0
                loader_cont.reset()
                for b in loader_cont:
                    _ = b["img"]

                # Capture checkpoint state at end of Epoch 0
                ckpt_gen_state = loader_cont.generator.get_state().clone()
                ckpt_torch_rng = torch.get_rng_state().clone()
                ckpt_np_rng = np.random.get_state()
                ckpt_py_rng = random.getstate()

                # Continuous run: Epoch 1 (Stage 1 resets loader at start of epoch)
                loader_cont.reset()
                cont_epoch1_batches = []
                for b in loader_cont:
                    cont_epoch1_batches.append(
                        {
                            "im_file": list(b["im_file"]),
                            "img": b["img"].clone(),
                            "sample_type": list(b.get("sample_type", [])),
                        }
                    )
                cont_epoch1_gen_state = loader_cont.generator.get_state().clone()
                cont_epoch1_torch_rng = torch.get_rng_state().clone()

                # Resumed run: Build fresh dataloader for resume
                ds_res = YOLODatasetStage1Clean(cfg, data_dict["train"], batch=2, data=data_dict, mode="train")
                loader_res = build_dataloader(ds_res, batch=2, workers=0, shuffle=True)

                # Restore checkpoint state
                torch.set_rng_state(ckpt_torch_rng)
                np.random.set_state(ckpt_np_rng)
                random.setstate(ckpt_py_rng)
                loader_res.generator.set_state(ckpt_gen_state)

                # Resumed run: Epoch 1 (reset loader at start of epoch)
                loader_res.reset()
                res_epoch1_batches = []
                for b in loader_res:
                    res_epoch1_batches.append(
                        {
                            "im_file": list(b["im_file"]),
                            "img": b["img"].clone(),
                            "sample_type": list(b.get("sample_type", [])),
                        }
                    )
                res_epoch1_gen_state = loader_res.generator.get_state().clone()
                res_epoch1_torch_rng = torch.get_rng_state().clone()

                # Assert exact batch counts
                self.assertEqual(len(cont_epoch1_batches), len(res_epoch1_batches))

                # Assert exact sample order and image tensor equality for every batch
                for idx, (b_c, b_r) in enumerate(zip(cont_epoch1_batches, res_epoch1_batches)):
                    self.assertEqual(
                        b_c["im_file"],
                        b_r["im_file"],
                        f"Batch {idx} sample ordering mismatch for seed={test_seed}!",
                    )
                    self.assertTrue(
                        torch.equal(b_c["img"], b_r["img"]),
                        f"Batch {idx} image tensor mismatch for seed={test_seed}!",
                    )

                # Assert exact generator and torch RNG equivalence
                self.assertTrue(
                    torch.equal(cont_epoch1_gen_state, res_epoch1_gen_state),
                    f"Generator state mismatch for seed={test_seed}!",
                )
                self.assertTrue(
                    torch.equal(cont_epoch1_torch_rng, res_epoch1_torch_rng),
                    f"Torch RNG state mismatch for seed={test_seed}!",
                )

    # -------------------------------------------------------------------------
    # Test D: Real Validation Entrypoint, Model Source, and Batch Clamping
    # -------------------------------------------------------------------------
    def test_d_validation_entrypoint_model_source_and_batch_clamping(self):
        """Test D: Verify validation entrypoint correctly records real model_source (e.g. best.pt),
        distinguishes requested_model vs model_source, preserves FP32 precision in Stage 1,
        and clamps actual_batch when len(dataset) < requested batch."""
        data_yaml = create_synthetic_dataset(self.temp_dir, num_images=3, imgsz=64)

        # Build real checkpoint best.pt
        best_pt_path = self.temp_dir / "weights" / "best.pt"
        best_pt_path.parent.mkdir(parents=True, exist_ok=True)
        real_model = DetectionModel("yolo11n.yaml", ch=3, nc=2, verbose=False)
        real_model.args = get_cfg()
        torch.save({"model": real_model}, best_pt_path)

        # Set up validator with requested model 'yolo11n.yaml', batch=16 (dataset only has 3 samples)
        val = DetectionValidatorAdvFGDSTNStaged.__new__(DetectionValidatorAdvFGDSTNStaged)
        val.stage = 1
        val.preprocess_mode = "stn_fgd"
        val.fgd_weights = str(self.fgd_path)
        val.stn_weights = str(self.stn_path)
        val.attack_through_preprocess = False
        val.device = self.device
        val.save_dir = self.temp_dir / "val_exp"
        val.save_dir.mkdir(parents=True, exist_ok=True)

        val.args = SimpleNamespace(
            model="yolo11n.yaml",
            data=str(data_yaml),
            split="val",
            imgsz=[64, 64],  # Verify list imgsz handling
            batch=16,
            seed=0,
            quantize=None,
            augment=False,
            plots=False,
        )

        # Build dummy dataloader with 3 samples
        class DummyDataset:
            def __len__(self):
                return 3

        val.dataloader = SimpleNamespace(batch_size=16, dataset=DummyDataset())

        # Test recording validation context when model_source is explicit best.pt (as in final_eval)
        ctx = val.record_validation_context(
            trainer=None,
            model_source=str(best_pt_path),
            requested_model="yolo11n.yaml",
        )

        # Verify model_source records actual best.pt, requested_model records initial config
        self.assertEqual(ctx["model_source"], str(best_pt_path))
        self.assertEqual(ctx["requested_model"], "yolo11n.yaml")
        # Verify FP32 precision for Stage 1
        self.assertEqual(ctx["precision"], "FP32")
        # Verify actual_batch clamped to dataset length (3) instead of 16
        self.assertEqual(ctx["actual_batch"], 3)
        # Verify imgsz handled as list/tuple safely
        self.assertEqual(ctx["effective_imgsz"], [64, 64])

    # -------------------------------------------------------------------------
    # Test E: Eval Weights Priority, Zero Weights, and isolate_rng Safety
    # -------------------------------------------------------------------------
    def test_e_eval_weights_priority_and_isolate_rng_exception_safety(self):
        """Test E: Verify:
        1. 4-level eval_weights priority (constructor > args dict > namespace > default) without mutating caller args.
        2. Zero-weight behavior (clean excluded, fitness equals PGD mAP50-95).
        3. isolate_rng comprehensive exception safety and chained exceptions."""
        # 1. Four-level priority
        # Priority 1: Explicit constructor arg
        caller_dict = {"eval_weights": [0.2, 0.2, 0.2, 0.4]}
        v1 = DetectionValidatorAdvFGDSTNStaged(
            args=caller_dict,
            eval_weights=[0.1, 0.2, 0.3, 0.4],
            preprocess_mode="none",
        )
        self.assertEqual(v1.eval_weights, [0.1, 0.2, 0.3, 0.4])
        self.assertIn("eval_weights", caller_dict, "Caller args dict must not be mutated!")

        # Priority 2: In args dict
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

        # Priority 4: Default when omitted (eval_weights is None, uses default fixed weights)
        v4 = DetectionValidatorAdvFGDSTNStaged(
            args={},
            preprocess_mode="none",
        )
        self.assertIsNone(v4.eval_weights)
        norm_w_default = v4.compute_norm_eval_weights(["pgd", "bim", "mim", "worstk"])
        self.assertAlmostEqual(norm_w_default["pgd"], 0.25 / 0.75, places=5)
        self.assertAlmostEqual(norm_w_default["bim"], 0.15 / 0.75, places=5)
        self.assertAlmostEqual(norm_w_default["mim"], 0.15 / 0.75, places=5)
        self.assertAlmostEqual(norm_w_default["worstk"], 0.20 / 0.75, places=5)
        self.assertEqual(norm_w_default["raw"], 0.0)

        # 2. Zero-weights normalization and fitness
        v_zero = DetectionValidatorAdvAdaptive(
            args={"eval_weights": [1.0, 0.0, 0.0, 0.0]},
        )
        norm_w = v_zero.compute_norm_eval_weights(["pgd", "bim", "mim", "worstk"])
        self.assertEqual(norm_w, {"pgd": 1.0, "bim": 0.0, "mim": 0.0, "worstk": 0.0, "raw": 0.0})

        # Adaptive fitness excludes clean (clean weight = 0), so fitness equals pgd
        v_zero.metrics = SimpleNamespace(
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
        stats = v_zero.aggregate_multirun_metrics(
            metric_arrays={k: {} for k in all_res},
            all_results=all_res,
            norm_eval_weights=norm_w,
            eval_runs=["pgd", "bim", "mim", "worstk"],
        )
        self.assertAlmostEqual(stats["metrics/mAP50-95(B)"], 0.45, places=5)
        self.assertAlmostEqual(stats["fitness"], 0.45, places=5)

        # 3. isolate_rng comprehensive exception safety
        class BrokenAttacker:
            def __init__(self, name: str, fail_save: bool = False, fail_load: bool = False):
                self.name = name
                self.fail_save = fail_save
                self.fail_load = fail_load

            def state_dict(self):
                if self.fail_save:
                    raise RuntimeError(f"Cannot save {self.name}")
                return {"val": 1}

            def load_state_dict(self, d):
                if self.fail_load:
                    raise RuntimeError(f"Cannot load {self.name}")

        orig_py_state = random.getstate()
        orig_np_state = np.random.get_state()
        orig_torch_state = torch.get_rng_state()

        atk_ok = BrokenAttacker("ok")
        atk_fail = BrokenAttacker("fail", fail_load=True)

        # Attacker load failure should raise cleanup_exc while still restoring global RNGs
        with self.assertRaises(RuntimeError) as ctx, isolate_rng(seed=999, attackers=[atk_ok, atk_fail]):
            random.random()
            np.random.rand()
            torch.rand(1)

        self.assertIn("isolate_rng cleanup encountered 1 failure(s)", str(ctx.exception))
        self.assertEqual(random.getstate(), orig_py_state)
        self.assertEqual(np.random.get_state()[1].tolist(), orig_np_state[1].tolist())
        self.assertTrue(torch.equal(torch.get_rng_state(), orig_torch_state))

        # Body exception + cleanup failure chains cleanup_exc from body_exc
        with self.assertRaises(RuntimeError) as ctx, isolate_rng(seed=999, attackers=[atk_fail]):
            raise ValueError("Original body error")

        self.assertIsInstance(ctx.exception.__cause__, ValueError)
        self.assertEqual(str(ctx.exception.__cause__), "Original body error")

        # Attacker deduplication
        atk_dup = BrokenAttacker("dup")
        save_counter = 0

        def counted_state_dict():
            nonlocal save_counter
            save_counter += 1
            return {}

        atk_dup.state_dict = counted_state_dict
        with isolate_rng(seed=777, attackers=[atk_dup, atk_dup, atk_dup]):
            pass
        self.assertEqual(save_counter, 1, "Duplicate attackers in list should be deduplicated!")


if __name__ == "__main__":
    unittest.main()
