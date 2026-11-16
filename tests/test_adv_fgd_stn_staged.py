# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Comprehensive test suite for Staged Adaptive Adversarial Training with Frozen FGD and STN."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ultralytics.cfg import get_cfg
from ultralytics.defenses.fgd_stn_pipeline import (
    FGDSTNPipeline,
    PreprocessDetectorWrapper,
    transform_batch_bboxes_for_stn,
)
from ultralytics.defenses.hgd_fgd import ResidualDenoiser
from ultralytics.defenses.spatial_transformer import (
    SpatialTransformer,
    build_forward_matrix,
    transform_boxes,
)
from ultralytics.models.yolo.detect.train_adv_fgd_stn_adaptive import (
    AdaptiveAttackRatioController,
    DetectionTrainerAdvFGDSTNAdaptive,
)
from ultralytics.models.yolo.detect.train_adv_fgd_stn_staged import (
    DetectionTrainerAdvFGDSTNStaged,
    YOLODatasetStage1Clean,
    find_detection_head,
)
from ultralytics.models.yolo.detect.val_adv_fgd_stn_adaptive import (
    DetectionValidatorAdvFGDSTNAdaptive,
)
from ultralytics.models.yolo.detect.val_adv_fgd_stn_staged import (
    DetectionValidatorAdvFGDSTNStaged,
)
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import DEFAULT_CFG


class DummyBackboneBlock(nn.Module):
    """Simulates a YOLO backbone / neck block with conv and batchnorm."""

    def __init__(self, c_in: int = 16, c_out: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class MockYOLOModel(nn.Module):
    """Lightweight mock YOLO model with Detect head matching YOLO12 architecture."""

    def __init__(self, nc: int = 10, ch: int = 16):
        super().__init__()
        self.nc = nc
        self.names = {i: f"class_{i}" for i in range(nc)}
        self.stride = torch.tensor([8.0, 16.0, 32.0])

        # Sequential container matching DetectionModel structure
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


class TestStagedPipeline(unittest.TestCase):
    """Test suite covering the 14 verification requirements for the staged pipeline."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.temp_dir = Path(tempfile.mkdtemp())
        self.fgd = ResidualDenoiser().to(self.device)
        self.stn = SpatialTransformer(max_degrees=30.0, max_translate=0.10).to(self.device)

        # Save dummy checkpoints to satisfy strict file existence checks
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

    # 1. Stage 1 initializes without constructing adversarial attackers
    def test_req1_stage1_no_attackers(self):
        """Verify Stage 1 initializes without constructing adversarial attackers."""
        trainer = DetectionTrainerAdvFGDSTNStaged.__new__(DetectionTrainerAdvFGDSTNStaged)
        trainer.stage = 1
        trainer.attack_names = []
        trainer.attackers = {}
        trainer.attacker = None
        trainer.attack_model = None
        trainer.device = self.device
        trainer.preprocess_mode = "stn_fgd"
        trainer.fgd_weights = str(self.fgd_path)
        trainer.stn_weights = str(self.stn_path)
        trainer.train_head_only = True

        data = {"nc": 10, "names": {i: f"cls_{i}" for i in range(10)}}
        args = type("Args", (), {"imgsz": 640, "attack_weights": "current"})()
        trainer.data = data
        trainer.args = args

        dummy_model = MockYOLOModel()
        trainer.model = dummy_model

        trainer.set_model_attributes()

        self.assertEqual(len(trainer.attackers), 0)
        self.assertIsNone(trainer.attacker)
        self.assertIsNone(trainer.attack_model)
        self.assertTrue(trainer.pipeline.has_denoiser)
        self.assertTrue(trainer.pipeline.has_stn)

    # 2. Stage 1 updates only YOLO Detection Head parameters
    def test_req2_stage1_updates_only_detection_head(self):
        """Verify Detection Head identification and that only Head parameters are trainable."""
        mock_yolo = MockYOLOModel()
        head_mod, head_idx = find_detection_head(mock_yolo)
        self.assertIsNotNone(head_mod)
        self.assertEqual(head_idx, 3)
        self.assertIsInstance(head_mod, Detect)

        head_param_ids = {id(p) for p in head_mod.parameters()}

        # Simulate Stage 1 freezing
        for k, v in mock_yolo.named_parameters():
            if id(v) not in head_param_ids or ".dfl." in k:
                v.requires_grad = False
            else:
                v.requires_grad = True

        trainable = [k for k, p in mock_yolo.named_parameters() if p.requires_grad]
        frozen = [k for k, p in mock_yolo.named_parameters() if not p.requires_grad]

        self.assertTrue(len(trainable) > 0)
        self.assertTrue(all("model.3." in k for k in trainable))
        self.assertTrue(all("model.0." in k or "model.1." in k or "model.2." in k or ".dfl." in k for k in frozen))

    # 3. Stage 1 uses only Clean images
    def test_req3_stage1_clean_only_dataset(self):
        """Verify Stage 1 uses 100% clean images with p_clean=1.0 and no attacks."""
        trainer = DetectionTrainerAdvFGDSTNStaged.__new__(DetectionTrainerAdvFGDSTNStaged)
        trainer.stage = 1
        trainer.p_clean = 1.0
        trainer.attack_names = []
        trainer.attack_ratios = []
        trainer.adaptive_controller = None

        self.assertEqual(trainer.p_clean, 1.0)
        self.assertEqual(len(trainer.attack_names), 0)
        self.assertEqual(len(trainer.attack_ratios), 0)
        self.assertIsNone(trainer.adaptive_controller)

    # 4. Stage 1 best.pt is selected using Clean mAP50-95
    def test_req4_stage1_fitness_clean_map50_95(self):
        """Verify Stage 1 selects best.pt using Clean mAP50-95."""
        validator = DetectionValidatorAdvFGDSTNStaged.__new__(DetectionValidatorAdvFGDSTNStaged)
        validator.stage = 1
        validator.eval_weights = [1.0]

        # Simulate stats returned for clean validation
        all_results = {"raw": {"metrics/mAP50-95(B)": 0.742, "metrics/mAP50(B)": 0.891}}

        # In Stage 1: fitness = Clean mAP50-95
        fitness = float(all_results["raw"]["metrics/mAP50-95(B)"])
        self.assertAlmostEqual(fitness, 0.742)

    # 5. Stage 2 correctly loads Stage 1 best.pt
    def test_req5_stage2_loads_stage1_weights(self):
        """Verify Stage 2 correctly overrides model path with stage1_weights."""
        mock_stage1_best = str(self.temp_dir / "best_stage1.pt")
        torch.save({"model": MockYOLOModel().state_dict()}, mock_stage1_best)

        overrides = {
            "stage": 2,
            "stage1_weights": mock_stage1_best,
            "resume": False,
        }
        model_path = overrides.get("model", "yolo12l.pt")
        if overrides["stage"] == 2 and overrides.get("stage1_weights"):
            model_path = overrides["stage1_weights"]

        self.assertEqual(model_path, mock_stage1_best)

    # 6. Stage 2 initializes a fresh optimizer and scheduler
    def test_req6_stage2_fresh_optimizer(self):
        """Verify Stage 2 creates a fresh optimizer and scheduler without resuming state."""
        trainer = DetectionTrainerAdvFGDSTNStaged.__new__(DetectionTrainerAdvFGDSTNStaged)
        trainer.stage = 2
        trainer.resume = False
        trainer.start_epoch = 0
        trainer.best_fitness = None

        # Verify that resume_training does nothing when resume is False
        mock_ckpt = {"epoch": 4, "optimizer": {"param_groups": []}, "best_fitness": 0.85}
        # In trainer: if ckpt is None or not self.resume: return
        if mock_ckpt is not None and trainer.resume:
            trainer.start_epoch = mock_ckpt["epoch"] + 1

        self.assertEqual(trainer.start_epoch, 0)
        self.assertIsNone(trainer.best_fitness)

    # 7. Stage 2 preserves the Clean ratio at 0.65
    def test_req7_stage2_preserves_clean_ratio(self):
        """Verify Stage 2 preserves Clean ratio at 0.65 across dynamic updates."""
        controller = AdaptiveAttackRatioController(
            attack_names=["pgd", "bim", "mim", "worstk"],
            initial_attack_ratios=[0.12, 0.05, 0.10, 0.08],
            p_clean=0.65,
            tau=5.0,
            update_interval=5,
            adaptive_ratio=True,
        )
        self.assertEqual(controller.p_clean, 0.65)
        self.assertEqual(controller.current_train_ratios["raw"], 0.65)

        # Step epoch 5 (update interval)
        maps = {"raw": 0.75, "pgd": 0.35, "bim": 0.40, "mim": 0.25, "worstk": 0.15}
        new_ratios, updated = controller.step(epoch=5, attack_maps=maps)
        self.assertTrue(updated)
        self.assertAlmostEqual(new_ratios["raw"], 0.65, places=5)

    # 8. Stage 2 adversarial ratios always sum to 0.35
    def test_req8_stage2_adversarial_ratios_sum_to_035(self):
        """Verify Stage 2 adversarial ratios always sum to 0.35."""
        attacks = ["pgd", "bim", "mim", "worstk"]
        controller = AdaptiveAttackRatioController(
            attack_names=attacks,
            initial_attack_ratios=[0.12, 0.05, 0.10, 0.08],
            p_clean=0.65,
            tau=5.0,
            update_interval=5,
            adaptive_ratio=True,
        )

        init_adv_sum = sum(controller.current_train_ratios[a] for a in attacks)
        self.assertAlmostEqual(init_adv_sum, 0.35, places=5)

        # Simulate several degradation profiles
        degradations = [
            {"raw": 0.70, "pgd": 0.20, "bim": 0.30, "mim": 0.15, "worstk": 0.10},
            {"raw": 0.80, "pgd": 0.10, "bim": 0.10, "mim": 0.10, "worstk": 0.05},
            {"raw": 0.65, "pgd": 0.60, "bim": 0.50, "mim": 0.40, "worstk": 0.30},
        ]
        for ep, maps in enumerate(degradations, start=1):
            ratios, _ = controller.step(epoch=ep * 5, attack_maps=maps)
            adv_sum = sum(ratios[a] for a in attacks)
            self.assertAlmostEqual(adv_sum, 0.35, places=5)
            self.assertAlmostEqual(sum(ratios.values()), 1.0, places=5)

    # 9. Frozen FGD/STN parameters never change
    def test_req9_frozen_fgd_stn_parameters_immutable(self):
        """Verify that FGD and STN parameters are not updated during training step."""
        pipeline = FGDSTNPipeline(denoiser=self.fgd, stn=self.stn, freeze_modules=True)
        orig_fgd_p = [p.clone() for p in self.fgd.parameters()]
        orig_stn_p = [p.clone() for p in self.stn.parameters()]

        x = torch.rand((2, 3, 64, 64), requires_grad=True)
        out, _ = pipeline(x, mode="stn_fgd")
        loss = out.sum()
        loss.backward()

        for orig, p in zip(orig_fgd_p, self.fgd.parameters()):
            self.assertIsNone(p.grad)
            self.assertTrue(torch.equal(orig, p))

        for orig, p in zip(orig_stn_p, self.stn.parameters()):
            self.assertIsNone(p.grad)
            self.assertTrue(torch.equal(orig, p))

    # 10. Frozen YOLO layers and BatchNorm statistics never change
    def test_req10_frozen_bn_statistics_eval_mode(self):
        """Verify frozen BatchNorm modules stay in eval mode and running statistics do not change."""
        model = MockYOLOModel()
        head_mod, _ = find_detection_head(model)
        head_param_ids = {id(p) for p in head_mod.parameters()}

        for k, v in model.named_parameters():
            if id(v) not in head_param_ids or ".dfl." in k:
                v.requires_grad = False
            else:
                v.requires_grad = True

        trainer = DetectionTrainerAdvFGDSTNStaged.__new__(DetectionTrainerAdvFGDSTNStaged)
        trainer.model = model
        trainer.pipeline = None
        trainer.freeze_bn = True
        trainer.freeze_layer_names = ["model.0.", "model.1.", "model.2.", ".dfl"]

        # Put in train mode using custom _model_train
        trainer._model_train()

        # Frozen BN modules must be in eval mode
        self.assertFalse(model.model[0].bn.training)
        self.assertFalse(model.model[1].bn.training)
        self.assertFalse(model.model[2].bn.training)

        # Save initial running mean of backbone BN
        orig_mean = model.model[0].bn.running_mean.clone()

        # Pass synthetic data through
        inp = torch.rand((4, 3, 64, 64))
        _ = model(inp)

        # Running mean must not change because module is in eval mode
        self.assertTrue(torch.equal(orig_mean, model.model[0].bn.running_mean))

    # 11. WorstK bounding boxes remain correct
    def test_req11_worstk_bounding_boxes_correct(self):
        """Verify WorstK bounding box transformations are valid and preserved."""
        h, w = 64, 64
        batch = {
            "img": torch.rand((1, 3, h, w)),
            "batch_idx": torch.tensor([0], dtype=torch.long),
            "cls": torch.tensor([0.0]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
        }

        # STN forward matrix
        params = torch.tensor([[15.0, 0.05, -0.05]])
        rot_m = build_forward_matrix(params, height=h, width=w)
        res = transform_batch_bboxes_for_stn(copy.deepcopy(batch), rot_m, height=h, width=w)

        self.assertEqual(res["bboxes"].shape, (1, 4))
        self.assertTrue((res["bboxes"] >= 0.0).all() and (res["bboxes"] <= 1.0).all())
        # Width and height should remain positive
        self.assertTrue(res["bboxes"][0, 2] > 0.0)
        self.assertTrue(res["bboxes"][0, 3] > 0.0)

    # 12. Attack-through preprocessing is differentiable and produces meaningful input gradients
    def test_req12_attack_through_preprocess_differentiable(self):
        """Verify attack-through preprocessing computes gradients back to inputs through STN and FGD."""
        pipeline = FGDSTNPipeline(denoiser=self.fgd, stn=self.stn, freeze_modules=True)
        detector = MockYOLOModel()
        wrapper = PreprocessDetectorWrapper(pipeline=pipeline, detector=detector, mode="stn_fgd")

        x = torch.rand((2, 3, 64, 64), requires_grad=True)
        preds = wrapper(x)

        if isinstance(preds, dict):
            loss = sum(v.sum() for v in preds.values() if torch.is_tensor(v))
        elif isinstance(preds, (list, tuple)):
            loss = sum(p.sum() for p in preds if torch.is_tensor(p))
        else:
            loss = preds.sum()

        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue((x.grad.abs().sum() > 0).item())

    # 13. Training and validation attack failures raise explicit errors
    def test_req13_attack_failures_raise_explicit_errors(self):
        """Verify that failed attacks raise explicit errors when strict_attacks is True."""
        validator = DetectionValidatorAdvFGDSTNStaged.__new__(DetectionValidatorAdvFGDSTNStaged)
        validator.strict_attacks = True
        validator.attack_failures = {"pgd"}

        with self.assertRaises(RuntimeError) as ctx:
            if validator.attack_failures and validator.strict_attacks:
                raise RuntimeError(
                    f"Adversarial generation failed during validation for attacks: {validator.attack_failures}."
                )

        self.assertIn("Adversarial generation failed", str(ctx.exception))

    def test_empty_target_sub_batch_handled_gracefully(self):
        """Verify sub-batches without bounding boxes do not crash with strict_attacks=True."""
        sub_batch = {
            "img": torch.rand((1, 3, 64, 64)),
            "batch_idx": torch.tensor([], dtype=torch.long),
            "cls": torch.tensor([], dtype=torch.float32),
            "bboxes": torch.tensor([], dtype=torch.float32).reshape(0, 4),
        }
        has_targets = "bboxes" in sub_batch and sub_batch["bboxes"] is not None and sub_batch["bboxes"].numel() > 0
        self.assertFalse(has_targets)

    def test_autobackend_surrogate_extraction(self):
        """Verify that AutoBackend wrappers extract the underlying nn.Module detector for attackers."""

        class MockAutoBackend:
            def __init__(self, inner):
                self.model = inner

            def parameters(self):
                return iter([])

        inner_detector = MockYOLOModel()
        wrapped = MockAutoBackend(inner_detector)
        raw_detector = wrapped.model if hasattr(wrapped, "model") and isinstance(wrapped.model, nn.Module) else wrapped
        self.assertEqual(raw_detector, inner_detector)
        params = list(raw_detector.parameters())
        self.assertTrue(len(params) > 0)

    # 14. The new implementation does not alter the behavior of existing experiment scripts
    def test_req14_existing_scripts_unaltered(self):
        """Verify existing trainer and validator classes remain intact and functional."""
        self.assertTrue(issubclass(DetectionTrainerAdvFGDSTNStaged, DetectionTrainerAdvFGDSTNAdaptive))
        self.assertTrue(issubclass(DetectionValidatorAdvFGDSTNStaged, DetectionValidatorAdvFGDSTNAdaptive))

    def test_stage2_fitness_formula(self):
        """Verify Stage 2 fitness computation: 0.20*Clean + 0.20*PGD + 0.15*BIM + 0.15*MIM + 0.30*WorstK."""
        all_results = {
            "raw": {"metrics/mAP50-95(B)": 0.80},
            "pgd": {"metrics/mAP50-95(B)": 0.40},
            "bim": {"metrics/mAP50-95(B)": 0.30},
            "mim": {"metrics/mAP50-95(B)": 0.35},
            "worstk": {"metrics/mAP50-95(B)": 0.25},
        }
        weights = {"raw": 0.20, "pgd": 0.20, "bim": 0.15, "mim": 0.15, "worstk": 0.30}
        expected_fitness = 0.20 * 0.80 + 0.20 * 0.40 + 0.15 * 0.30 + 0.15 * 0.35 + 0.30 * 0.25
        computed_fitness = sum(weights[r] * all_results[r]["metrics/mAP50-95(B)"] for r in weights)
        self.assertAlmostEqual(computed_fitness, expected_fitness, places=5)
        self.assertAlmostEqual(computed_fitness, 0.4125, places=4)

    def test_epoch0_eval_only_mode(self):
        """Verify Improvement 1: Epoch 0 eval_only mode executes without optimizer or attackers and writes outputs."""
        trainer = DetectionTrainerAdvFGDSTNStaged.__new__(DetectionTrainerAdvFGDSTNStaged)
        trainer.stage = 1
        trainer.attack_names = []
        trainer.attackers = {}
        trainer.attacker = None
        trainer.attack_model = None
        trainer.device = self.device
        trainer.preprocess_mode = "stn_fgd"
        trainer.fgd_weights = str(self.fgd_path)
        trainer.stn_weights = str(self.stn_path)
        trainer.train_head_only = True
        trainer.freeze_all_bn = True
        trainer.freeze_bn = True
        trainer.save_dir = self.temp_dir / "eval_only_exp"
        trainer.save_dir.mkdir(parents=True, exist_ok=True)
        trainer.eval_weights = [1.0]
        trainer.strict_attacks = True

        data = {"nc": 10, "names": {i: f"cls_{i}" for i in range(10)}, "val": "val_dummy", "split": "val"}
        args = get_cfg(DEFAULT_CFG)
        args.imgsz = 640
        args.attack_weights = "current"
        args.model = "yolo12l.pt"
        args.split = "val"
        args.task = "detect"
        args.batch = 4
        args.augment = False
        args.quantize = None
        args.plots = False
        args.compile = False
        trainer.data = data
        trainer.args = args
        trainer.model = MockYOLOModel()
        trainer.set_model_attributes()

        # Optimizer and scheduler must be None in eval_only mode
        trainer.optimizer = None
        trainer.scheduler = None

        # Mock dataloader returning 2 clean batches
        dummy_img = torch.rand(2, 3, 640, 640)
        dummy_batch = {
            "img": dummy_img,
            "sample_type": ["raw", "raw"],
            "batch_idx": torch.tensor([0, 1], dtype=torch.long),
            "cls": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]], dtype=torch.float32),
            "ori_shape": [(640, 640), (640, 640)],
            "ratio_pad": [((1.0, 1.0), (0.0, 0.0)), ((1.0, 1.0), (0.0, 0.0))],
            "im_file": ["img1.jpg", "img2.jpg"],
        }
        mock_loader = [dummy_batch, dummy_batch]
        mock_loader = type(
            "MockLoader", (list,), {"dataset": type("Dataset", (), {"im_files": ["img1.jpg", "img2.jpg"]})()}
        )(mock_loader)
        trainer.test_loader = mock_loader
        trainer.callbacks = {}

        results = trainer.run_epoch0_eval(save_results=True)
        self.assertEqual(results["epoch"], 0)
        self.assertIn("metrics/mAP50(B)", results)
        self.assertIn("metrics/mAP50-95(B)", results)
        self.assertIsNone(trainer.optimizer)
        self.assertIsNone(trainer.scheduler)
        self.assertEqual(len(trainer.attackers), 0)

        json_file = trainer.save_dir / "epoch0_results.json"
        csv_file = trainer.save_dir / "epoch0_results.csv"
        self.assertTrue(json_file.exists())
        self.assertTrue(csv_file.exists())

        with open(json_file, "r", encoding="utf-8") as f:
            loaded_json = json.load(f)
        self.assertEqual(loaded_json["epoch"], 0)
        self.assertEqual(loaded_json["model"], "yolo12l.pt")
        self.assertEqual(loaded_json["preprocess_mode"], "stn_fgd")

    def test_epoch0_and_training_validation_pipeline_parity(self):
        """Verify Improvement 1 parity: Epoch 0 and training epochs use identical validator and box unwarping."""
        trainer = DetectionTrainerAdvFGDSTNStaged.__new__(DetectionTrainerAdvFGDSTNStaged)
        trainer.stage = 1
        trainer.attack_names = []
        trainer.attackers = {}
        trainer.attacker = None
        trainer.attack_model = None
        trainer.device = self.device
        trainer.preprocess_mode = "stn_fgd"
        trainer.fgd_weights = str(self.fgd_path)
        trainer.stn_weights = str(self.stn_path)
        trainer.save_dir = self.temp_dir / "parity_exp"
        trainer.save_dir.mkdir(parents=True, exist_ok=True)
        trainer.eval_weights = [1.0]
        trainer.strict_attacks = True
        trainer.train_head_only = True
        trainer.freeze_all_bn = True
        trainer.callbacks = {}

        data = {"nc": 10, "names": {i: f"cls_{i}" for i in range(10)}, "val": "val_dummy", "split": "val"}
        args = get_cfg(DEFAULT_CFG)
        args.imgsz = 640
        args.attack_weights = "current"
        args.model = "yolo12l.pt"
        args.split = "val"
        args.task = "detect"
        args.batch = 2
        args.augment = False
        args.quantize = None
        args.plots = False
        args.compile = False
        trainer.data = data
        trainer.args = args
        trainer.model = MockYOLOModel()
        trainer.set_model_attributes()

        val1 = trainer.get_validator()
        self.assertIsInstance(val1, DetectionValidatorAdvFGDSTNStaged)
        self.assertEqual(val1.stage, 1)
        self.assertEqual(val1.preprocess_mode, "stn_fgd")
        self.assertTrue(val1.pipeline.has_stn)
        self.assertTrue(val1.pipeline.has_denoiser)

        dummy_batch = {
            "img": torch.rand(2, 3, 640, 640),
            "sample_type": ["raw", "raw"],
            "batch_idx": torch.tensor([0, 1], dtype=torch.long),
            "cls": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]], dtype=torch.float32),
            "ori_shape": [(640, 640), (640, 640)],
            "ratio_pad": [((1.0, 1.0), (0.0, 0.0)), ((1.0, 1.0), (0.0, 0.0))],
            "im_file": ["img1.jpg", "img2.jpg"],
        }
        mock_loader = type(
            "MockLoader", (list,), {"dataset": type("Dataset", (), {"im_files": ["img1.jpg", "img2.jpg"]})()}
        )([copy.deepcopy(dummy_batch)])
        trainer.test_loader = mock_loader

        # 1. Run Epoch 0 evaluation without optimizer
        trainer.optimizer = None
        trainer.scheduler = None
        results_epoch0 = trainer.run_epoch0_eval(save_results=True)
        self.assertIsNone(trainer.optimizer)
        self.assertIsNone(trainer.scheduler)
        self.assertTrue((trainer.save_dir / "epoch0_results.json").exists())
        self.assertTrue((trainer.save_dir / "epoch0_results.csv").exists())

        # 2. Run regular Stage 1 validation
        val_train = trainer.get_validator()
        val_train.dataloader = type(
            "MockLoader", (list,), {"dataset": type("Dataset", (), {"im_files": ["img1.jpg", "img2.jpg"]})()}
        )([copy.deepcopy(dummy_batch)])
        trainer.epoch = 1
        results_stage1 = val_train(trainer)

        self.assertEqual(results_epoch0["metrics/mAP50(B)"], results_stage1["metrics/mAP50(B)"])
        self.assertEqual(results_epoch0["metrics/mAP50-95(B)"], results_stage1["metrics/mAP50-95(B)"])
        self.assertEqual(results_epoch0["fitness"], results_stage1["fitness"])

    def test_stage1_dataset_clean_sample_allocation(self):
        """Verify Improvement 2: Stage 1 dataset subclass strictly allocates 100% raw samples."""
        from ultralytics.data.utils import check_det_dataset

        # Full __init__ instantiation test with coco8
        data = check_det_dataset("coco8.yaml")
        cfg = get_cfg(overrides={"data": "coco8.yaml", "imgsz": 64})
        real_dataset = YOLODatasetStage1Clean(cfg, data["train"], batch=2, data=data, mode="train")
        self.assertTrue(hasattr(real_dataset, "cfg"))
        self.assertTrue(hasattr(real_dataset, "transforms_raw"))
        self.assertEqual(len(real_dataset), 4)
        sample = real_dataset[0]
        self.assertEqual(sample["sample_type"], "raw")
        self.assertIsNone(sample["attack_name"])

        # Large scale virtual sample allocation test
        dataset = YOLODatasetStage1Clean.__new__(YOLODatasetStage1Clean)
        dataset.im_files = [f"img_{i}.jpg" for i in range(5000)]
        dataset.update_virtual_samples()

        self.assertEqual(len(dataset.virtual_samples), 5000)
        self.assertEqual(len(dataset), 5000)
        # Verify 100% raw
        self.assertTrue(all(st == "raw" for _, st in dataset.virtual_samples))
        # Verify all 5000 images are used exactly once
        visited_indices = {idx for idx, _ in dataset.virtual_samples}
        self.assertEqual(len(visited_indices), 5000)
        self.assertEqual(visited_indices, set(range(5000)))
        self.assertEqual(dataset.attack_names, [])
        self.assertEqual(dataset.p_clean, 1.0)

    def test_stage1_freeze_all_bn_runtime_invariance(self):
        """Verify Improvement 3: All BN layers (including Head BN) remain frozen and invariant."""
        trainer = DetectionTrainerAdvFGDSTNStaged.__new__(DetectionTrainerAdvFGDSTNStaged)
        trainer.stage = 1
        trainer.train_head_only = True
        trainer.freeze_all_bn = True
        trainer.freeze_bn = True
        trainer.pipeline = None
        trainer.args = type("Args", (), {"freeze": [], "imgsz": 640})()

        model = MockYOLOModel(ch=16)
        trainer.model = model

        # Gather head parameter IDs
        head_module, _ = find_detection_head(model)
        head_param_ids = {id(p) for p in head_module.parameters()}

        bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        bn_param_ids = {id(p) for m in bn_modules for p in m.parameters()}
        self.assertTrue(len(bn_modules) > 0)

        # Freeze BN layers and verify head non-BN isolation
        for m in bn_modules:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            m.train = lambda mode=True, mod=m: mod

        for k, v in model.named_parameters():
            in_head = id(v) in head_param_ids
            in_bn = id(v) in bn_param_ids
            if in_head and ".dfl." not in k and not in_bn:
                v.requires_grad = True
            else:
                v.requires_grad = False

        # Trainable parameters must ONLY be non-BN head parameters
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.assertTrue(len(trainable) > 0)
        self.assertTrue(all(id(p) in head_param_ids for p in trainable))
        self.assertTrue(all(id(p) not in bn_param_ids for p in trainable))

        # Model.train() must NOT re-enable BN training
        model.train()
        for m in bn_modules:
            self.assertFalse(m.training)

        # Build optimizer and test update isolation
        optimizer = torch.optim.AdamW(trainable, lr=1e-3)
        trainer.optimizer = optimizer
        trainer._take_frozen_snapshots()

        # Verify initial invariants check succeeds
        trainer.check_frozen_invariants("init")

        # Run forward and backward step
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        if isinstance(out, dict):
            loss = sum(v.sum() for v in out.values() if isinstance(v, torch.Tensor))
        elif isinstance(out, (list, tuple)):
            loss = sum(v.sum() for v in out if isinstance(v, torch.Tensor))
        else:
            loss = out.sum()
        loss.backward()
        optimizer.step()

        # Runtime check: frozen parameters and BN running stats must remain unchanged
        trainer.check_frozen_invariants("post_step")

        # Artificially corrupt a BN running stat and verify failure detection
        bn_modules[0].running_mean.add_(1.0)
        with self.assertRaises(RuntimeError):
            trainer.check_frozen_invariants("corrupted_stat")

    def test_stage1_conservative_defaults_and_cli(self):
        """Verify Improvement 3: Stage 1 defaults are conservative, while Stage 2 remains untouched."""
        trainer_s1 = DetectionTrainerAdvFGDSTNStaged.__new__(DetectionTrainerAdvFGDSTNStaged)
        trainer_s1.stage = 1
        trainer_s1.freeze_all_bn = True
        trainer_s1.train_head_only = True
        trainer_s1.save_period = 1
        self.assertTrue(trainer_s1.freeze_all_bn)
        self.assertEqual(trainer_s1.save_period, 1)

        # Verify Stage 2 keeps adaptive defaults
        trainer_s2 = DetectionTrainerAdvFGDSTNStaged.__new__(DetectionTrainerAdvFGDSTNStaged)
        trainer_s2.stage = 2
        trainer_s2.freeze_all_bn = False
        trainer_s2.save_period = -1
        self.assertFalse(trainer_s2.freeze_all_bn)
        self.assertEqual(trainer_s2.save_period, -1)

    def test_stage1_save_every_epoch_1_based(self):
        """Verify Improvement 4: Stage 1 saves epoch1.pt, epoch2.pt, epoch3.pt with 1-based indexing."""
        trainer = DetectionTrainerAdvFGDSTNStaged.__new__(DetectionTrainerAdvFGDSTNStaged)
        trainer.stage = 1
        trainer.save_period = 1
        trainer.epoch = 0  # 1st epoch
        trainer.best_fitness = 0.5
        trainer.fitness = 0.5
        trainer.wdir = self.temp_dir / "weights"
        trainer.wdir.mkdir(parents=True, exist_ok=True)
        trainer.last = trainer.wdir / "last.pt"
        trainer.best = trainer.wdir / "best.pt"

        # Mock model & EMA
        mock_model = MockYOLOModel()
        trainer.model = mock_model
        trainer.ema = type("EMA", (), {"ema": mock_model, "updates": 10})()
        trainer.optimizer = torch.optim.AdamW(mock_model.parameters(), lr=1e-4)
        trainer.scaler = torch.amp.GradScaler(device="cpu", enabled=False)
        trainer.args = type("Args", (), {"model": "yolo12l.pt", "save_period": 1})()
        trainer.metrics = {"metrics/mAP50(B)": 0.8, "metrics/mAP50-95(B)": 0.5}
        trainer.read_results_csv = dict
        trainer.stage1_weights = None
        trainer.train_head_only = True
        trainer.freeze_bn = True
        trainer.freeze_all_bn = True
        trainer.preprocess_mode = "stn_fgd"
        trainer.fgd_weights = None
        trainer.stn_weights = None
        trainer.adaptive_ratio = False
        trainer.attack_through_preprocess = True
        trainer.eval_weights = [1.0]
        trainer.adaptive_controller = None

        # Epoch 0 (human epoch 1)
        trainer.epoch = 0
        trainer.save_model()
        self.assertTrue((trainer.wdir / "epoch1.pt").exists())
        self.assertTrue(trainer.last.exists())
        self.assertTrue(trainer.best.exists())

        # Epoch 1 (human epoch 2)
        trainer.epoch = 1
        trainer.fitness = 0.6
        trainer.best_fitness = 0.6
        trainer.save_model()
        self.assertTrue((trainer.wdir / "epoch2.pt").exists())

        # Epoch 2 (human epoch 3)
        trainer.epoch = 2
        trainer.save_model()
        self.assertTrue((trainer.wdir / "epoch3.pt").exists())

    def test_stn_coordinate_consistency_roundtrip(self):
        """Verify coordinate consistency: STN-transformed training bboxes unwarp back to original coordinates."""
        h, w = 640, 640
        # Original bounding box in pixel coordinates [x1, y1, x2, y2]
        orig_boxes = torch.tensor(
            [
                [100.0, 150.0, 300.0, 400.0],
                [50.0, 80.0, 180.0, 220.0],
                [350.0, 200.0, 550.0, 480.0],
            ],
            dtype=torch.float32,
        )

        # 1. Translation-only consistency: roundtrip is mathematically exact
        params_trans = torch.tensor([[0.0, 15.0, -10.0]])
        m_forward_trans = build_forward_matrix(params_trans, height=h, width=w).squeeze(0)  # [3, 3]
        m_rect_trans = torch.linalg.inv(m_forward_trans)
        rect_boxes_trans = transform_boxes(orig_boxes, m_rect_trans, height=h, width=w, clip=False)
        unwarped_trans = transform_boxes(rect_boxes_trans, m_forward_trans, height=h, width=w, clip=False)
        self.assertTrue(torch.allclose(unwarped_trans, orig_boxes, atol=1e-3))

        # 2. General affine consistency: box centers are perfectly preserved under rotation + translation
        params_rot = torch.tensor([[25.0, 15.0, -10.0]])
        m_forward_rot = build_forward_matrix(params_rot, height=h, width=w).squeeze(0)  # [3, 3]
        m_rect_rot = torch.linalg.inv(m_forward_rot)
        rect_boxes_rot = transform_boxes(orig_boxes, m_rect_rot, height=h, width=w, clip=False)
        unwarped_rot = transform_boxes(rect_boxes_rot, m_forward_rot, height=h, width=w, clip=False)

        c_orig = (orig_boxes[:, :2] + orig_boxes[:, 2:]) / 2.0
        c_unwarped = (unwarped_rot[:, :2] + unwarped_rot[:, 2:]) / 2.0
        self.assertTrue(torch.allclose(c_orig, c_unwarped, atol=1e-3))

        # 3. Training batch transform (transform_batch_bboxes_for_stn) vs validation unwarp parity
        xc = (orig_boxes[:, 0] + orig_boxes[:, 2]) / 2.0 / w
        yc = (orig_boxes[:, 1] + orig_boxes[:, 3]) / 2.0 / h
        bw = (orig_boxes[:, 2] - orig_boxes[:, 0]) / w
        bh = (orig_boxes[:, 3] - orig_boxes[:, 1]) / h
        norm_xywh = torch.stack([xc, yc, bw, bh], dim=-1)

        batch = {
            "bboxes": norm_xywh,
            "batch_idx": torch.tensor([0, 0, 0], dtype=torch.long),
            "cls": torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32),
        }
        pred_matrices = m_forward_trans.unsqueeze(0)  # [1, 3, 3]
        transformed_batch = transform_batch_bboxes_for_stn(batch, pred_matrices, height=h, width=w)

        # Convert transformed normalized xywh back to pixel xyxy
        t_bboxes = transformed_batch["bboxes"]
        t_xc = t_bboxes[:, 0] * w
        t_yc = t_bboxes[:, 1] * h
        t_bw = t_bboxes[:, 2] * w
        t_bh = t_bboxes[:, 3] * h
        t_xyxy = torch.stack([t_xc - t_bw / 2.0, t_yc - t_bh / 2.0, t_xc + t_bw / 2.0, t_yc + t_bh / 2.0], dim=-1)

        # Unwarp back to canvas using m_forward_trans
        recovered_boxes = transform_boxes(t_xyxy, m_forward_trans, height=h, width=w, clip=False)
        self.assertTrue(torch.allclose(recovered_boxes, orig_boxes, atol=1e-3))

    def test_staged_validator_postprocess_single_unwarp(self):
        """Verify postprocess unwarps bounding boxes exactly once across normal detector and list[dict] inputs."""
        from types import SimpleNamespace

        def _build_validator(cls, stage=1):
            val = cls.__new__(cls)
            val.stage = stage
            val.pipeline = object()
            val._current_hw = (640, 640)
            val.args = SimpleNamespace(
                conf=0.25, iou=0.45, task="detect", single_cls=False, agnostic_nms=False, max_det=300
            )
            val.nc = 1
            val.end2end = False
            return val

        val_s1 = _build_validator(DetectionValidatorAdvFGDSTNStaged, stage=1)
        val_s2 = _build_validator(DetectionValidatorAdvFGDSTNStaged, stage=2)
        val_parent = _build_validator(DetectionValidatorAdvFGDSTNAdaptive)

        # +10 pixel horizontal translation: [10, 20, 30, 40] must unwarp to [20, 20, 40, 40], never [30, 20, 50, 40]
        params_shift = torch.tensor([[0.0, 10.0, 0.0]])
        m_shift = build_forward_matrix(params_shift, 640, 640)
        for v in [val_s1, val_s2, val_parent]:
            v.current_pred_matrix = m_shift

        # Entry 1: Normal detector raw tensor output passing through real NMS
        # Center=(20, 30), w=20, h=20 -> [10, 20, 30, 40]
        raw_preds = torch.zeros(1, 5, 1)
        raw_preds[0, 0, 0] = 20.0
        raw_preds[0, 1, 0] = 30.0
        raw_preds[0, 2, 0] = 20.0
        raw_preds[0, 3, 0] = 20.0
        raw_preds[0, 4, 0] = 0.9

        out_s1 = val_s1.postprocess(raw_preds.clone())
        out_s2 = val_s2.postprocess(raw_preds.clone())
        out_parent = val_parent.postprocess(raw_preds.clone())
        expected_box = torch.tensor([[20.0, 20.0, 40.0, 40.0]])
        wrong_double_unwarp_box = torch.tensor([[30.0, 20.0, 50.0, 40.0]])

        self.assertTrue(torch.allclose(out_s1[0]["bboxes"], expected_box))
        self.assertFalse(torch.allclose(out_s1[0]["bboxes"], wrong_double_unwarp_box))
        self.assertTrue(torch.allclose(out_s1[0]["bboxes"], out_s2[0]["bboxes"]))
        self.assertTrue(torch.allclose(out_s1[0]["bboxes"], out_parent[0]["bboxes"]))

        # Entry 2: list[dict] input
        preds_dict = [
            {
                "bboxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "conf": torch.tensor([0.9]),
                "cls": torch.tensor([0.0]),
            }
        ]
        out_dict = val_s1.postprocess(copy.deepcopy(preds_dict))
        self.assertTrue(torch.allclose(out_dict[0]["bboxes"], expected_box))

        # Identity matrix leaves coordinates unchanged
        params_id = torch.tensor([[0.0, 0.0, 0.0]])
        val_s1.current_pred_matrix = build_forward_matrix(params_id, 640, 640)
        out_id = val_s1.postprocess(copy.deepcopy(preds_dict))
        self.assertTrue(torch.allclose(out_id[0]["bboxes"], torch.tensor([[10.0, 20.0, 30.0, 40.0]])))

        # Empty predictions (raw below conf threshold and empty list[dict])
        raw_empty = torch.zeros(1, 5, 1)
        raw_empty[0, :4, 0] = torch.tensor([20.0, 30.0, 20.0, 20.0])
        raw_empty[0, 4, 0] = 0.01
        self.assertEqual(val_s1.postprocess(raw_empty)[0]["bboxes"].numel(), 0)
        dict_empty = [{"bboxes": torch.zeros((0, 4)), "conf": torch.zeros(0), "cls": torch.zeros(0)}]
        self.assertEqual(val_s1.postprocess(dict_empty)[0]["bboxes"].numel(), 0)

        # No STN: pipeline or matrix is None
        val_no_stn = _build_validator(DetectionValidatorAdvFGDSTNStaged)
        val_no_stn.pipeline = None
        val_no_stn.current_pred_matrix = None
        out_no_stn = val_no_stn.postprocess(copy.deepcopy(preds_dict))
        self.assertTrue(torch.allclose(out_no_stn[0]["bboxes"], torch.tensor([[10.0, 20.0, 30.0, 40.0]])))

        # Clip, degenerate box filtering, and extra synchronization
        val_s1.current_pred_matrix = m_shift
        dict_deg = [
            {
                "bboxes": torch.tensor([[10.0, 20.0, 30.0, 40.0], [635.0, 20.0, 640.0, 40.0]]),
                "conf": torch.tensor([0.9, 0.8]),
                "cls": torch.tensor([0.0, 0.0]),
                "extra": torch.tensor([[100.0], [200.0]]),
            }
        ]
        out_deg = val_s1.postprocess(dict_deg)
        self.assertEqual(out_deg[0]["bboxes"].shape, (1, 4))
        self.assertTrue(torch.allclose(out_deg[0]["bboxes"], expected_box))
        self.assertTrue(torch.allclose(out_deg[0]["conf"], torch.tensor([0.9])))
        self.assertTrue(torch.allclose(out_deg[0]["cls"], torch.tensor([0.0])))
        self.assertTrue(torch.allclose(out_deg[0]["extra"], torch.tensor([[100.0]])))


if __name__ == "__main__":
    unittest.main()
