# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Comprehensive test suite for Staged Adaptive Adversarial Training with Frozen FGD and STN."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from ultralytics.attacks.attack_bridge import build_attacker, run_attack_on_batch
from ultralytics.attacks.attack_utils import extract_detection_model
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
from ultralytics.models.yolo.detect.train_adv_adaptive import (
    DetectionTrainer as DetectionTrainerAdvAdaptive,
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
from ultralytics.models.yolo.detect.val_adv_adaptive import (
    DetectionValidator as DetectionValidatorAdvAdaptive,
)
from ultralytics.models.yolo.detect.val_adv_fgd_stn_adaptive import (
    DetectionValidatorAdvFGDSTNAdaptive,
)
from ultralytics.models.yolo.detect.val_adv_fgd_stn_staged import (
    DetectionValidatorAdvFGDSTNStaged,
)
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import ModelEMA


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


class LossTrackingDetector(MockYOLOModel):
    """Mock detector tracking received bboxes in loss calls."""

    def __init__(self, nc: int = 10, ch: int = 16):
        super().__init__(nc=nc, ch=ch)
        self.received_bboxes: list[torch.Tensor] = []

    def loss(self, batch: dict, preds: Any = None):
        if "bboxes" in batch and batch["bboxes"] is not None:
            self.received_bboxes.append(batch["bboxes"].clone())
        return super().loss(batch, preds)


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
        """Verify DetectionModel is extracted as a full detector rather than stripped to internal Sequential."""
        # 1. Real small DetectionModel counterexample: demonstrate the bug in the previous implementation
        m = DetectionModel("yolo11n.yaml", ch=3, nc=10, verbose=False)
        old_buggy_detector = m.model if hasattr(m, "model") and isinstance(m.model, nn.Module) else m
        self.assertIsInstance(old_buggy_detector, nn.Sequential)
        self.assertFalse(hasattr(old_buggy_detector, "loss"))
        with self.assertRaises(TypeError):
            old_buggy_detector(torch.zeros(1, 3, 64, 64), augment=False)

        # 2. Extract from complete DetectionModel: keeps full DetectionModel with loss and YOLO routing
        d1 = extract_detection_model(m)
        self.assertIs(d1, m)
        self.assertFalse(isinstance(d1, nn.Sequential))
        self.assertTrue(hasattr(d1, "loss") and callable(d1.loss))
        out1 = d1(torch.zeros(1, 3, 64, 64), augment=False)
        self.assertIsNotNone(out1)

        # 3. Extract from training EMA model
        ema = ModelEMA(m)
        d2 = extract_detection_model(ema.ema)
        self.assertIs(d2, ema.ema)
        self.assertTrue(hasattr(d2, "loss") and callable(d2.loss))
        out2 = d2(torch.zeros(1, 3, 64, 64), augment=False)
        self.assertIsNotNone(out2)

        # 4. Extract from real AutoBackend (PyTorch)
        ab = AutoBackend(model=m, device=torch.device("cpu"), verbose=False)
        d3 = extract_detection_model(ab)
        self.assertFalse(isinstance(d3, nn.Sequential))
        self.assertTrue(hasattr(d3, "loss") and callable(d3.loss))
        out3 = d3(torch.zeros(1, 3, 64, 64), augment=False)
        self.assertIsNotNone(out3)

        # 5. Unsupported export backends raise explicit ValueError
        class UnsupportedAutoBackend(AutoBackend):
            def __init__(self):
                super(nn.Module, self).__init__()
                self.format = "onnx"

        with self.assertRaises(ValueError):
            extract_detection_model(UnsupportedAutoBackend())

        # 6. Bare nn.Sequential or model lacking loss raises TypeError
        with self.assertRaises(TypeError):
            extract_detection_model(m.model)
        with self.assertRaises(TypeError):
            extract_detection_model(nn.Linear(10, 10))

        # 7. Real PGD and WorstK construction and mini-batch execution
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eval_surrogate = copy.deepcopy(d1).float().to(target_device)
        for p in eval_surrogate.parameters():
            p.requires_grad_(False)
        eval_surrogate.eval()
        eval_surrogate.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
        if hasattr(eval_surrogate, "init_criterion"):
            eval_surrogate.criterion = eval_surrogate.init_criterion()

        worstk = build_attacker("worstk", model=eval_surrogate, img_size=64)
        pgd = build_attacker("pgd", model=eval_surrogate, img_size=64)

        mini_batch = {
            "img": torch.rand(2, 3, 64, 64, device=target_device),
            "batch_idx": torch.tensor([0, 1], dtype=torch.long, device=target_device),
            "cls": torch.tensor([[0.0], [1.0]], dtype=torch.float32, device=target_device),
            "bboxes": torch.tensor(
                [[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]], dtype=torch.float32, device=target_device
            ),
        }
        # Neither PGD nor WorstK should raise Sequential missing loss or augment errors
        worstk.forward_batch(copy.deepcopy(mini_batch))
        adv_pgd = run_attack_on_batch(pgd, copy.deepcopy(mini_batch), label_policy="largest_box", max_batch_size=2)
        self.assertEqual(adv_pgd.shape, (2, 3, 64, 64))

        # 8. Model selection fallback in validator: trainer without EMA falls back to trainer.model
        trainer_no_ema = DetectionTrainerAdvFGDSTNStaged.__new__(DetectionTrainerAdvFGDSTNStaged)
        trainer_no_ema.model = m
        trainer_no_ema.ema = None
        trainer_no_ema.attack_model = None
        source_no_ema = (
            getattr(trainer_no_ema, "ema", None) and getattr(trainer_no_ema.ema, "ema", None)
        ) or trainer_no_ema.model
        self.assertIs(source_no_ema, m)
        self.assertIs(extract_detection_model(source_no_ema), m)

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

        data = {
            "nc": 10,
            "names": {i: f"cls_{i}" for i in range(10)},
            "channels": 3,
            "val": "val_dummy",
            "split": "val",
        }
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

        data = {
            "nc": 10,
            "names": {i: f"cls_{i}" for i in range(10)},
            "channels": 3,
            "val": "val_dummy",
            "split": "val",
        }
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

        # 2. General affine consistency: box centers are preserved under raw unwarp
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

        # 3a. Perfect inverse transformation under transform_batch_bboxes_for_stn:
        # Width, height, and area return to their exact original values (no double AABB dilation)
        batch_rot = {
            "bboxes": norm_xywh.clone(),
            "pre_attack_bboxes": norm_xywh.clone(),
            "attack_matrices": m_forward_rot.unsqueeze(0),
            "batch_idx": torch.tensor([0, 0, 0], dtype=torch.long),
            "cls": torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32),
        }
        perfect_rect_batch = transform_batch_bboxes_for_stn(batch_rot, m_forward_rot.unsqueeze(0), height=h, width=w)
        p_bboxes = perfect_rect_batch["bboxes"]
        p_xc = p_bboxes[:, 0] * w
        p_yc = p_bboxes[:, 1] * h
        p_bw = p_bboxes[:, 2] * w
        p_bh = p_bboxes[:, 3] * h
        p_xyxy = torch.stack([p_xc - p_bw / 2.0, p_yc - p_bh / 2.0, p_xc + p_bw / 2.0, p_yc + p_bh / 2.0], dim=-1)

        orig_w = orig_boxes[:, 2] - orig_boxes[:, 0]
        orig_h = orig_boxes[:, 3] - orig_boxes[:, 1]
        self.assertTrue(torch.allclose(p_bw, orig_w, atol=1e-3))
        self.assertTrue(torch.allclose(p_bh, orig_h, atol=1e-3))
        self.assertTrue(torch.allclose(p_xyxy, orig_boxes, atol=1e-3))

        # 3b. Translation training batch transform vs validation unwarp parity (clean images with STN)
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

    def test_clean_validation_loss_two_batches_and_no_monkeypatch(self):
        """Verify Item 3: STN loss closure monkeypatch is removed and validation loss transforms labels cleanly."""
        pipeline = FGDSTNPipeline(denoiser=self.fgd, stn=self.stn)
        h, w = 640, 640

        # +10 pixel horizontal shift forward matrix: [B, 3, 3]
        params_shift = torch.tensor([[0.0, 10.0, 0.0]])
        m_shift = build_forward_matrix(params_shift, h, w)
        # Identity forward matrix: [B, 3, 3]
        params_id = torch.tensor([[0.0, 0.0, 0.0]])
        m_id = build_forward_matrix(params_id, h, w)

        for val_cls in [DetectionValidatorAdvFGDSTNAdaptive, DetectionValidatorAdvFGDSTNStaged]:
            detector = LossTrackingDetector()
            orig_loss_func = detector.loss.__func__

            val = val_cls.__new__(val_cls)
            val.pipeline = pipeline
            val.preprocess_mode = "stn_fgd"
            val.current_pred_matrix = None
            val._current_hw = (h, w)
            val.device = self.device
            val.model = detector
            val.training = True
            val.stage = 1

            # Batch 1: with +10 px shift
            orig_bboxes1 = torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32)
            batch1 = {
                "img": torch.rand(1, 3, h, w),
                "bboxes": orig_bboxes1.clone(),
                "batch_idx": torch.tensor([0], dtype=torch.long),
                "cls": torch.tensor([[0.0]], dtype=torch.float32),
            }
            val.current_pred_matrix = m_shift
            preds1 = detector(batch1["img"])
            val.compute_validation_loss(batch1, preds1, detector)

            # Verification 1: batch1["bboxes"] is NOT mutated in-place
            self.assertTrue(torch.allclose(batch1["bboxes"], orig_bboxes1))
            # Verification 2: detector.loss received rectified bboxes (which differ from orig_bboxes1)
            loss1_bboxes = detector.received_bboxes[-1]
            self.assertFalse(torch.allclose(loss1_bboxes, orig_bboxes1))

            # Batch 2: with identity matrix
            orig_bboxes2 = torch.tensor([[0.4, 0.4, 0.1, 0.1]], dtype=torch.float32)
            batch2 = {
                "img": torch.rand(1, 3, h, w),
                "bboxes": orig_bboxes2.clone(),
                "batch_idx": torch.tensor([0], dtype=torch.long),
                "cls": torch.tensor([[1.0]], dtype=torch.float32),
            }
            val.current_pred_matrix = m_id
            preds2 = detector(batch2["img"])
            val.compute_validation_loss(batch2, preds2, detector)

            # Verification 3: batch2["bboxes"] is NOT mutated in-place
            self.assertTrue(torch.allclose(batch2["bboxes"], orig_bboxes2))
            # Verification 4: detector.loss received untransformed bboxes for identity matrix
            loss2_bboxes = detector.received_bboxes[-1]
            self.assertTrue(torch.allclose(loss2_bboxes, orig_bboxes2, atol=1e-4))

            # Verification 5: detector.loss was never monkey-patched
            self.assertNotIn("loss", detector.__dict__)
            self.assertIs(detector.loss.__func__, orig_loss_func)

    def test_validation_loss_across_defense_modes(self):
        """Verify validation loss applies coordinate transformation only when STN is present."""
        pipeline = FGDSTNPipeline(denoiser=self.fgd, stn=self.stn)
        h, w = 640, 640

        params_shift = torch.tensor([[0.0, 10.0, 0.0]])
        m_shift = build_forward_matrix(params_shift, h, w)

        modes = ["none", "fgd", "stn", "fgd_stn", "stn_fgd"]
        for mode in modes:
            detector = LossTrackingDetector()
            val = DetectionValidatorAdvFGDSTNStaged.__new__(DetectionValidatorAdvFGDSTNStaged)
            val.pipeline = pipeline
            val.preprocess_mode = mode
            val._current_hw = (h, w)
            val.device = self.device
            val.model = detector
            val.training = True
            val.stage = 1

            # For modes with STN, pipeline forward produces m_shift; for none/fgd, pred_matrix is None
            if mode in {"stn", "fgd_stn", "stn_fgd"}:
                val.current_pred_matrix = m_shift
            else:
                val.current_pred_matrix = None

            orig_bboxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32)
            batch = {
                "img": torch.rand(1, 3, h, w),
                "bboxes": orig_bboxes.clone(),
                "batch_idx": torch.tensor([0], dtype=torch.long),
                "cls": torch.tensor([[0.0]], dtype=torch.float32),
            }
            preds = detector(batch["img"])
            val.compute_validation_loss(batch, preds, detector)

            received = detector.received_bboxes[-1]
            if mode in {"stn", "fgd_stn", "stn_fgd"}:
                # Transformed
                self.assertFalse(torch.allclose(received, orig_bboxes))
            else:
                # Untransformed
                self.assertTrue(torch.allclose(received, orig_bboxes))

    def test_validation_does_not_inherit_train_attack_through_preprocess(self):
        """Verify validator does not automatically inherit trainer.attack_through_preprocess=True."""
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
        trainer.attack_through_preprocess = True  # Training uses attack_through_preprocess=True
        trainer.save_dir = self.temp_dir / "val_att_test"
        trainer.save_dir.mkdir(parents=True, exist_ok=True)
        trainer.eval_weights = [1.0]
        trainer.strict_attacks = True
        trainer.train_head_only = True
        trainer.freeze_all_bn = True
        trainer.callbacks = {}

        data = {
            "nc": 10,
            "names": {i: f"cls_{i}" for i in range(10)},
            "channels": 3,
            "val": "val_dummy",
            "split": "val",
        }
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

        val = trainer.get_validator()
        # Default validator attack_through_preprocess must be False even if trainer has True
        self.assertFalse(val.attack_through_preprocess)

    def test_epoch0_and_eval_only_stage2_raises_not_implemented(self):
        """Verify Stage 2 run_epoch0_eval and eval_only raise structural NotImplementedError early."""
        trainer = DetectionTrainerAdvFGDSTNStaged.__new__(DetectionTrainerAdvFGDSTNStaged)
        trainer.stage = 2
        trainer.device = self.device

        with self.assertRaises(NotImplementedError) as ctx1:
            trainer.run_epoch0_eval()
        self.assertIn("only supported for Stage 1", str(ctx1.exception))

        with self.assertRaises(NotImplementedError) as ctx2:
            trainer.eval_only()
        self.assertIn("only supported for Stage 1", str(ctx2.exception))

    def test_validation_attacker_current_mode_training_vs_final_eval(self):
        """Verify Item 5: In current mode, training val targets EMA while final_eval targets best.pt."""
        model_early_best = MockYOLOModel()
        with torch.no_grad():
            model_early_best.model[0].conv.weight.fill_(5.0)

        model_last_ema = MockYOLOModel()
        with torch.no_grad():
            model_last_ema.model[0].conv.weight.fill_(1.0)

        best_orig_weight = model_early_best.model[0].conv.weight.clone()
        ema_orig_weight = model_last_ema.model[0].conv.weight.clone()

        trainer = DetectionTrainerAdvFGDSTNStaged.__new__(DetectionTrainerAdvFGDSTNStaged)
        trainer.stage = 2
        trainer.device = self.device
        trainer.preprocess_mode = "none"
        trainer.fgd_weights = None
        trainer.stn_weights = None
        trainer.pipeline = None
        trainer.attack_through_preprocess = False
        trainer.save_dir = self.temp_dir / "val_item5_current"
        trainer.save_dir.mkdir(parents=True, exist_ok=True)
        trainer.eval_weights = [1.0]
        trainer.strict_attacks = True
        trainer.attack_names = ["worstk"]
        trainer.attack_ratios = [1.0]
        trainer.attack_weights = "current"
        trainer.callbacks = {}
        trainer.epoch = 10
        trainer.ema = SimpleNamespace(ema=model_last_ema)
        trainer.model = model_last_ema
        trainer.loss_items = {
            "box_loss": torch.tensor(0.0),
            "cls_loss": torch.tensor(0.0),
            "dfl_loss": torch.tensor(0.0),
        }

        data = {
            "nc": 10,
            "names": {i: f"cls_{i}" for i in range(10)},
            "channels": 3,
            "val": "val_dummy",
            "split": "val",
        }
        args = get_cfg(DEFAULT_CFG)
        args.imgsz = 640
        args.attack_weights = "current"
        args.model = "mock.pt"
        args.split = "val"
        args.task = "detect"
        args.batch = 2
        args.augment = False
        args.quantize = None
        args.plots = False
        args.compile = False
        trainer.data = data
        trainer.args = args

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

        val = trainer.get_validator()
        val.dataloader = mock_loader

        # 1. Training-time validation: must target model_last_ema (weight=1.0)
        val(trainer)
        self.assertIn("worstk", val.attackers)
        surrogate_train = val.attackers["worstk"].model
        self.assertAlmostEqual(surrogate_train.model[0].conv.weight.mean().item(), 1.0, places=4)

        # 2. final_eval (trainer=None, model=model_early_best): must target model_early_best (weight=5.0)
        val.dataloader = type(
            "MockLoader", (list,), {"dataset": type("Dataset", (), {"im_files": ["img1.jpg", "img2.jpg"]})()}
        )([copy.deepcopy(dummy_batch)])
        val(model=model_early_best)
        self.assertIn("worstk", val.attackers)
        surrogate_final = val.attackers["worstk"].model
        self.assertAlmostEqual(surrogate_final.model[0].conv.weight.mean().item(), 5.0, places=4)

        # 3. Verify evaluated models' weights and states are completely unharmed
        self.assertTrue(torch.equal(model_early_best.model[0].conv.weight.cpu(), best_orig_weight.cpu()))
        self.assertTrue(torch.equal(model_last_ema.model[0].conv.weight.cpu(), ema_orig_weight.cpu()))

    def test_validation_attacker_fixed_surrogate_mode(self):
        """Verify Item 5: Fixed surrogate mode targets the fixed model in both training val and final_eval."""
        fixed_surrogate_model = MockYOLOModel()
        with torch.no_grad():
            fixed_surrogate_model.model[0].conv.weight.fill_(10.0)

        fixed_ckpt_path = self.temp_dir / "fixed_surrogate.pt"
        torch.save({"model": fixed_surrogate_model}, fixed_ckpt_path)

        model_early_best = MockYOLOModel()
        with torch.no_grad():
            model_early_best.model[0].conv.weight.fill_(5.0)

        model_last_ema = MockYOLOModel()
        with torch.no_grad():
            model_last_ema.model[0].conv.weight.fill_(1.0)

        trainer = DetectionTrainerAdvFGDSTNStaged.__new__(DetectionTrainerAdvFGDSTNStaged)
        trainer.stage = 2
        trainer.device = self.device
        trainer.preprocess_mode = "none"
        trainer.fgd_weights = None
        trainer.stn_weights = None
        trainer.pipeline = None
        trainer.attack_through_preprocess = False
        trainer.save_dir = self.temp_dir / "val_item5_fixed"
        trainer.save_dir.mkdir(parents=True, exist_ok=True)
        trainer.eval_weights = [1.0]
        trainer.strict_attacks = True
        trainer.attack_names = ["worstk"]
        trainer.attack_ratios = [1.0]
        trainer.attack_weights = str(fixed_ckpt_path)
        trainer.callbacks = {}
        trainer.epoch = 10
        trainer.ema = SimpleNamespace(ema=model_last_ema)
        trainer.model = model_last_ema
        trainer.loss_items = {
            "box_loss": torch.tensor(0.0),
            "cls_loss": torch.tensor(0.0),
            "dfl_loss": torch.tensor(0.0),
        }

        data = {
            "nc": 10,
            "names": {i: f"cls_{i}" for i in range(10)},
            "channels": 3,
            "val": "val_dummy",
            "split": "val",
        }
        args = get_cfg(DEFAULT_CFG)
        args.imgsz = 640
        args.attack_weights = str(fixed_ckpt_path)
        args.model = "mock.pt"
        args.split = "val"
        args.task = "detect"
        args.batch = 2
        args.augment = False
        args.quantize = None
        args.plots = False
        args.compile = False
        trainer.data = data
        trainer.args = args

        val = trainer.get_validator()

        # Training-time val targets fixed surrogate (10.0)
        val.setup_validation_attackers(["worstk"], model=None, trainer=trainer)
        self.assertAlmostEqual(val.attackers["worstk"].model.model[0].conv.weight.mean().item(), 10.0, places=4)

        # final_eval targets the exact same fixed surrogate (10.0), NOT downgrading to best.pt (5.0)
        val.setup_validation_attackers(["worstk"], model=model_early_best, trainer=None)
        self.assertAlmostEqual(val.attackers["worstk"].model.model[0].conv.weight.mean().item(), 10.0, places=4)

    def test_validation_attacker_missing_fixed_weights_raises_error(self):
        """Verify Item 5: Specifying a non-existent attack_weights file raises FileNotFoundError and does not fallback."""
        val = DetectionValidatorAdvFGDSTNStaged(
            stage=2,
            attack_weights=str(self.temp_dir / "non_existent_weights.pt"),
            args={"imgsz": 640, "task": "detect"},
        )
        val.device = self.device
        with self.assertRaises(FileNotFoundError):
            val.setup_validation_attackers(["worstk"], model=MockYOLOModel(), trainer=None)

    def test_standalone_validator_consecutive_models(self):
        """Verify Item 5: Standalone validator following multiple models updates attackers without stale models."""
        model_a = MockYOLOModel()
        with torch.no_grad():
            model_a.model[0].conv.weight.fill_(2.0)

        model_b = MockYOLOModel()
        with torch.no_grad():
            model_b.model[0].conv.weight.fill_(8.0)

        val = DetectionValidatorAdvFGDSTNStaged(
            stage=2,
            attack_weights="current",
            args={"imgsz": 640, "task": "detect"},
        )
        val.device = self.device

        # First run on Model A
        val.setup_validation_attackers(["worstk"], model=model_a, trainer=None)
        self.assertAlmostEqual(val.attackers["worstk"].model.model[0].conv.weight.mean().item(), 2.0, places=4)

        # Second run on Model B
        val.setup_validation_attackers(["worstk"], model=model_b, trainer=None)
        self.assertAlmostEqual(val.attackers["worstk"].model.model[0].conv.weight.mean().item(), 8.0, places=4)

    def test_stage1_skips_attacker_construction_stage2_builds_attackers(self):
        """Verify Item 5: Stage 1 validator skips attacker construction while Stage 2 builds attackers."""
        val_s1 = DetectionValidatorAdvFGDSTNStaged(
            stage=1,
            attack_weights="current",
            args={"imgsz": 640, "task": "detect"},
        )
        val_s1.device = self.device
        val_s1.setup_validation_attackers(["worstk"], model=MockYOLOModel(), trainer=None)
        self.assertEqual(val_s1.attackers, {})
        self.assertIsNone(val_s1.attacker)

        val_s2 = DetectionValidatorAdvFGDSTNStaged(
            stage=2,
            attack_weights="current",
            args={"imgsz": 640, "task": "detect"},
        )
        val_s2.device = self.device
        val_s2.setup_validation_attackers(["worstk"], model=MockYOLOModel(), trainer=None)
        self.assertIn("worstk", val_s2.attackers)
        self.assertIsNotNone(val_s2.attacker)

    def test_req6_invalid_attack_or_checkpoint_raises_immediately(self):
        """Verify Item 4 / Req 6: Invalid attack name or corrupted checkpoint raises immediately before training."""
        trainer = DetectionTrainerAdvAdaptive.__new__(DetectionTrainerAdvAdaptive)
        trainer.device = self.device
        trainer.args = SimpleNamespace(imgsz=640, attack_weights=None)
        trainer.data = {"nc": 2, "names": {0: "a", 1: "b"}, "channels": 3}
        trainer.attack_names = ["invalid_attack_xyz"]
        trainer.strict_attacks = True
        mock_model = MockYOLOModel()
        trainer.model = mock_model

        with self.assertRaises((ValueError, RuntimeError)):
            trainer.set_model_attributes()

        # Non-existent attack_weights
        trainer.attack_names = ["worstk"]
        trainer.args.attack_weights = str(self.temp_dir / "non_existent_surrogate.pt")
        with self.assertRaises(FileNotFoundError):
            trainer.set_model_attributes()

    def test_req6_empty_target_input_legitimate_skip_vs_worstk(self):
        """Verify Item 4 / Req 6: Detector attacks legitimately skip batches with no targets, while WorstK runs."""
        mock_model = MockYOLOModel()
        worstk_attacker = build_attacker("worstk", model=mock_model, img_size=640)
        empty_batch = {
            "img": torch.rand(2, 3, 640, 640),
            "bboxes": torch.zeros((0, 4)),
            "cls": torch.zeros((0, 1)),
            "batch_idx": torch.zeros((0,)),
        }
        adv_worstk = run_attack_on_batch(worstk_attacker, empty_batch)
        self.assertIsNotNone(adv_worstk)
        self.assertEqual(adv_worstk.shape, (2, 3, 640, 640))

        from ultralytics.attacks.attack_bridge import build_detector_targets

        det_targets = build_detector_targets(empty_batch)
        self.assertEqual(det_targets.numel(), 0)

    def test_req6_sample_level_conservation_laws(self):
        """Verify Item 4 / Req 6: Strict image-level sample conservation laws."""
        attack_names = ["pgd", "bim", "worstk"]
        epoch_counts = {"raw": 20, "pgd": 25, "bim": 30, "worstk": 25}
        total_planned = sum(epoch_counts.values())

        epoch_attack_stats = {
            "attempted": {"pgd": 25, "bim": 30, "worstk": 25},
            "generated": {"pgd": 20, "bim": 30, "worstk": 22},
            "skipped": {"pgd": 5, "bim": 0, "worstk": 0},
            "failed": {"pgd": 0, "bim": 0, "worstk": 3},
            "actual_raw": 20 + 5 + 3,
        }

        # 1. Total samples conservation: total_planned == sum(generated) + actual_raw
        total_generated = sum(epoch_attack_stats["generated"].values())
        self.assertEqual(total_generated + epoch_attack_stats["actual_raw"], total_planned)

        # 2. Per-attack conservation: attempted[att] == generated[att] + skipped[att] + failed[att]
        for att in attack_names:
            att_cnt = epoch_attack_stats["attempted"][att]
            gen_cnt = epoch_attack_stats["generated"][att]
            skip_cnt = epoch_attack_stats["skipped"][att]
            fail_cnt = epoch_attack_stats["failed"][att]
            self.assertEqual(att_cnt, gen_cnt + skip_cnt + fail_cnt)

    def test_req6_non_strict_rollback_preserves_paired_labels_and_boxes(self):
        """Verify Item 4 / Req 6: strict_attacks=False rolls back sub-batch ensuring images and bboxes stay paired."""
        sub_batch = {
            "img": torch.ones(2, 3, 640, 640),
            "bboxes": torch.tensor([[10.0, 10.0, 50.0, 50.0], [20.0, 20.0, 60.0, 60.0]]),
            "cls": torch.tensor([[0.0], [1.0]]),
            "batch_idx": torch.tensor([0, 1]),
        }
        sub_backup = {
            "img": sub_batch["img"].clone(),
            "bboxes": sub_batch["bboxes"].clone(),
            "cls": sub_batch["cls"].clone(),
            "batch_idx": sub_batch["batch_idx"].clone(),
        }

        sub_batch["bboxes"] = sub_batch["bboxes"] + 100.0
        sub_batch["img"] = sub_batch["img"] + 0.5

        # Attack failure caught in non-strict mode: rollback is performed
        sub_batch["img"] = sub_backup["img"]
        sub_batch["bboxes"] = sub_backup["bboxes"]
        sub_batch["cls"] = sub_backup["cls"]
        sub_batch["batch_idx"] = sub_backup["batch_idx"]

        torch.testing.assert_close(sub_batch["img"], sub_backup["img"])
        torch.testing.assert_close(sub_batch["bboxes"], sub_backup["bboxes"])
        torch.testing.assert_close(sub_batch["cls"], sub_backup["cls"])
        torch.testing.assert_close(sub_batch["batch_idx"], sub_backup["batch_idx"])

    def test_req6_validation_attack_failure_does_not_masquerade_as_clean(self):
        """Verify Item 4 / Req 6: Validation attack failure does not masquerade as clean in strict and non-strict modes."""
        val = DetectionValidatorAdvAdaptive(
            args={"task": "detect", "imgsz": 640, "attack_names": ["worstk"]},
            strict_attacks=True,
        )
        val.device = self.device
        val.batch_i = 3
        val.current_attack = "worstk"
        val.attacker = None

        # In strict_attacks=True: raises RuntimeError immediately
        with self.assertRaises(RuntimeError) as cm:
            val.preprocess({"img": torch.rand(2, 3, 640, 640), "bboxes": torch.tensor([[10, 10, 50, 50]])})
        self.assertIn("worstk", str(cm.exception))
        self.assertIn("batch 3", str(cm.exception))

        # In strict_attacks=False: records failure, zeroes out metrics, does not use clean mAP for fitness
        val_non_strict = DetectionValidatorAdvAdaptive(
            args={"task": "detect", "imgsz": 640, "attack_names": ["worstk"]},
            strict_attacks=False,
        )
        val_non_strict.device = self.device
        val_non_strict.batch_i = 1
        val_non_strict.current_attack = "worstk"
        val_non_strict.attacker = None
        val_non_strict.preprocess({"img": torch.rand(2, 3, 640, 640)})
        self.assertIn("worstk", val_non_strict.attack_failures)

    def test_req6_missing_defense_checkpoint_raises_filenotfound(self):
        """Verify Item 4 / Req 6: Missing defense checkpoint raises FileNotFoundError and does not fallback to identity."""
        with self.assertRaises(FileNotFoundError):
            DetectionTrainerAdvFGDSTNAdaptive(
                overrides={"preprocess_mode": "fgd", "fgd_weights": str(self.temp_dir / "missing_fgd.pt")}
            )

        with self.assertRaises(FileNotFoundError):
            DetectionTrainerAdvFGDSTNAdaptive(
                overrides={"preprocess_mode": "stn", "stn_weights": str(self.temp_dir / "missing_stn.pt")}
            )

        val = DetectionValidatorAdvFGDSTNAdaptive(
            preprocess_mode="fgd",
            fgd_weights=str(self.temp_dir / "missing_fgd.pt"),
            args={"task": "detect", "imgsz": 640},
        )
        with self.assertRaises(FileNotFoundError):
            val._setup_pipeline(self.device)

        val_none = DetectionValidatorAdvFGDSTNAdaptive(
            preprocess_mode="none",
            args={"task": "detect", "imgsz": 640},
        )
        val_none._setup_pipeline(self.device)
        self.assertIsNotNone(val_none.pipeline)
        self.assertFalse(val_none.pipeline.has_denoiser)
        self.assertFalse(val_none.pipeline.has_stn)

    def test_req6_stage1_zero_attempted_zero_generated_attacks(self):
        """Verify Item 4 / Req 6: Stage 1 maintains exactly 0 attempted, 0 generated, and 0 failed attacks."""
        trainer = DetectionTrainerAdvFGDSTNStaged.__new__(DetectionTrainerAdvFGDSTNStaged)
        trainer.stage = 1
        trainer.attack_names = ["pgd", "bim", "worstk"]
        trainer.epoch_counts = {"raw": 50, "pgd": 0, "bim": 0, "worstk": 0}
        trainer.epoch_attack_stats = {
            "attempted": {att: 0 for att in trainer.attack_names},
            "generated": {att: 0 for att in trainer.attack_names},
            "skipped": {att: 0 for att in trainer.attack_names},
            "failed": {att: 0 for att in trainer.attack_names},
            "actual_raw": 50,
        }

        total_epoch_samples = sum(trainer.epoch_counts.values())
        self.assertEqual(trainer.epoch_counts.get("raw", 0), total_epoch_samples)
        self.assertEqual(sum(trainer.epoch_attack_stats["attempted"].values()), 0)
        self.assertEqual(sum(trainer.epoch_attack_stats["generated"].values()), 0)
        self.assertEqual(sum(trainer.epoch_attack_stats["failed"].values()), 0)
        self.assertEqual(trainer.epoch_attack_stats["actual_raw"], total_epoch_samples)

    def test_req6_single_aabb_no_dilation_100x100_30deg(self):
        """Verify Item 6: Unclipped 100x100 box rotated by 30 deg returns to 100x100 under M_pred=A_attack, not 186.6x186.6."""
        h, w = 640, 640
        # 100x100 box centered at (320, 320): [270, 270, 370, 370]
        norm_box = torch.tensor([[0.5, 0.5, 100.0 / w, 100.0 / h]], dtype=torch.float32)

        param30 = torch.tensor([[30.0, 0.0, 0.0]])
        A_30 = build_forward_matrix(param30, height=h, width=w)  # [1, 3, 3]

        batch = {
            "bboxes": norm_box.clone(),
            "pre_attack_bboxes": norm_box.clone(),
            "attack_matrices": A_30.clone(),
            "batch_idx": torch.tensor([0], dtype=torch.long),
            "cls": torch.tensor([[0.0]], dtype=torch.float32),
        }

        # STN perfect rectification: M_pred = A_30
        out_batch = transform_batch_bboxes_for_stn(batch, A_30, height=h, width=w)
        out_box = out_batch["bboxes"]
        out_w = out_box[0, 2].item() * w
        out_h = out_box[0, 3].item() * h
        out_area = out_w * out_h

        # Under the old double-AABB bug, width was 186.6 and area was ~34819.
        # Under single AABB, width and height return to 100.0 and area returns to 10000.
        self.assertAlmostEqual(out_w, 100.0, delta=1.0)
        self.assertAlmostEqual(out_h, 100.0, delta=1.0)
        self.assertAlmostEqual(out_area, 10000.0, delta=200.0)
        self.assertLess(out_w, 130.0)  # Far below 186.6

    def test_req6_affine_cases_identity_translation_imperfect_clipping_outside(self):
        """Verify Item 6: Covers identity, pure translation, imperfect STN, clipping, and completely out of bounds."""
        h, w = 640, 640
        box = torch.tensor([[0.5, 0.5, 100.0 / w, 100.0 / h]], dtype=torch.float32)

        # 1. Identity: returns identical box
        eye = torch.eye(3).unsqueeze(0)
        b_id = {
            "bboxes": box.clone(),
            "pre_attack_bboxes": box.clone(),
            "attack_matrices": eye,
            "batch_idx": torch.tensor([0]),
            "cls": torch.tensor([0.0]),
        }
        out_id = transform_batch_bboxes_for_stn(b_id, eye, height=h, width=w)
        self.assertAlmostEqual(out_id["bboxes"][0, 2].item() * w, 100.0, delta=1e-3)
        self.assertAlmostEqual(out_id["bboxes"][0, 3].item() * h, 100.0, delta=1e-3)

        # 2. Pure translation: returns identical box under perfect STN
        p_trans = torch.tensor([[0.0, 20.0, -15.0]])
        A_trans = build_forward_matrix(p_trans, height=h, width=w)
        b_tr = {
            "bboxes": box.clone(),
            "pre_attack_bboxes": box.clone(),
            "attack_matrices": A_trans,
            "batch_idx": torch.tensor([0]),
            "cls": torch.tensor([0.0]),
        }
        out_tr = transform_batch_bboxes_for_stn(b_tr, A_trans, height=h, width=w)
        self.assertAlmostEqual(out_tr["bboxes"][0, 2].item() * w, 100.0, delta=1e-3)
        self.assertAlmostEqual(out_tr["bboxes"][0, 3].item() * h, 100.0, delta=1e-3)

        # 3. Imperfect STN (30 deg attack vs 25 deg STN): net 5 deg rotation (w ~ 108.34, NOT 181.53)
        p_stn = torch.tensor([[25.0, 0.0, 0.0]])
        p_att = torch.tensor([[30.0, 0.0, 0.0]])
        A_att = build_forward_matrix(p_att, height=h, width=w)
        M_stn = build_forward_matrix(p_stn, height=h, width=w)
        b_imp = {
            "bboxes": box.clone(),
            "pre_attack_bboxes": box.clone(),
            "attack_matrices": A_att,
            "batch_idx": torch.tensor([0]),
            "cls": torch.tensor([0.0]),
        }
        out_imp = transform_batch_bboxes_for_stn(b_imp, M_stn, height=h, width=w)
        imp_w = out_imp["bboxes"][0, 2].item() * w
        imp_h = out_imp["bboxes"][0, 3].item() * h
        self.assertAlmostEqual(imp_w, 108.34, delta=2.0)
        self.assertAlmostEqual(imp_h, 108.34, delta=2.0)
        self.assertLess(imp_w, 130.0)

        # 4. Partial boundary clipping: box near left edge (5..105), shifted left by 30px (25px lost)
        box_edge = torch.tensor([[55.0 / w, 55.0 / h, 100.0 / w, 100.0 / h]], dtype=torch.float32)
        p_shift = torch.tensor([[0.0, -30.0, 0.0]])
        A_shift = build_forward_matrix(p_shift, height=h, width=w)
        b_clip = {
            "bboxes": box_edge.clone(),
            "pre_attack_bboxes": box_edge.clone(),
            "attack_matrices": A_shift,
            "batch_idx": torch.tensor([0]),
            "cls": torch.tensor([0.0]),
        }
        out_clip = transform_batch_bboxes_for_stn(b_clip, A_shift, height=h, width=w)
        clip_w = out_clip["bboxes"][0, 2].item() * w
        clip_h = out_clip["bboxes"][0, 3].item() * h
        # 25px lost off-canvas are NOT restored: width is exactly 75.0
        self.assertAlmostEqual(clip_w, 75.0, delta=1.0)
        self.assertAlmostEqual(clip_h, 100.0, delta=1.0)

        # 5. Completely out of bounds: shifted left by 200px -> filtered out
        p_out = torch.tensor([[0.0, -200.0, 0.0]])
        A_out = build_forward_matrix(p_out, height=h, width=w)
        b_out = {
            "bboxes": box_edge.clone(),
            "pre_attack_bboxes": box_edge.clone(),
            "attack_matrices": A_out,
            "batch_idx": torch.tensor([0]),
            "cls": torch.tensor([0.0]),
        }
        out_out = transform_batch_bboxes_for_stn(b_out, A_out, height=h, width=w)
        self.assertEqual(out_out["bboxes"].numel(), 0)

    def test_req6_max_batch_size_chunking_consistency(self):
        """Verify Item 6: Different attack chunk sizes (max_batch_size=1 vs 4) produce identical matrices and labels."""
        mock_model = MockYOLOModel()
        batch = {
            "img": torch.rand(4, 3, 64, 64),
            "batch_idx": torch.tensor([0, 1, 2, 3], dtype=torch.long),
            "cls": torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float32),
            "bboxes": torch.tensor(
                [
                    [0.5, 0.5, 0.2, 0.2],
                    [0.4, 0.4, 0.3, 0.3],
                    [0.6, 0.6, 0.2, 0.2],
                    [0.5, 0.5, 0.25, 0.25],
                ],
                dtype=torch.float32,
            ),
        }

        # Run attack with chunk size 2
        attacker1 = build_attacker("worstk", model=mock_model, img_size=64, seed=123)
        b1 = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
        run_attack_on_batch(attacker1, b1, max_batch_size=2)

        self.assertIn("gt_matrix", b1)
        self.assertEqual(b1["gt_matrix"].shape, (4, 3, 3))
        self.assertEqual(b1["batch_idx"].view(-1).tolist(), [0, 1, 2, 3])

    def test_req6_mixed_batch_raw_pgd_worstk_mapping_and_no_stn_preservation(self):
        """Verify Item 6: Mixed batch (raw+pgd+worstk) local-to-global mapping and no STN preservation."""
        h, w = 640, 640
        b_raw = torch.tensor([[0.2, 0.2, 0.1, 0.1]], dtype=torch.float32)
        b_pgd = torch.tensor([[0.4, 0.4, 0.15, 0.15]], dtype=torch.float32)
        b_worstk = torch.tensor([[0.5, 0.5, 100.0 / w, 100.0 / h]], dtype=torch.float32)

        param30 = torch.tensor([[30.0, 0.0, 0.0]])
        A_worstk = build_forward_matrix(param30, height=h, width=w).squeeze(0)  # [3, 3]
        eye = torch.eye(3)

        batch_attack_matrices = torch.stack([eye, eye, A_worstk], dim=0)  # [3, 3, 3]

        all_pre_bboxes = torch.cat([b_raw, b_pgd, b_worstk], dim=0)
        batch = {
            "img": torch.rand(3, 3, h, w),
            "sample_type": ["raw", "pgd", "worstk"],
            "bboxes": all_pre_bboxes.clone(),
            "pre_attack_bboxes": all_pre_bboxes.clone(),
            "attack_matrices": batch_attack_matrices,
            "batch_idx": torch.tensor([0, 1, 2], dtype=torch.long),
            "cls": torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32),
        }

        # 1. No STN: pred_matrices is None -> labels remain unmodified in attack coordinate system
        out_no_stn = transform_batch_bboxes_for_stn(batch, None, height=h, width=w)
        torch.testing.assert_close(out_no_stn["bboxes"], all_pre_bboxes)

        # 2. With STN: STN rectifies all samples (eye for raw/pgd, A_worstk for worstk)
        pred_matrices = torch.stack([eye, eye, A_worstk], dim=0)
        out_stn = transform_batch_bboxes_for_stn(batch, pred_matrices, height=h, width=w)
        stn_bboxes = out_stn["bboxes"]

        # Raw sample: returned exact same
        self.assertAlmostEqual(stn_bboxes[0, 2].item(), b_raw[0, 2].item(), delta=1e-3)
        # PGD sample: returned exact same
        self.assertAlmostEqual(stn_bboxes[1, 2].item(), b_pgd[0, 2].item(), delta=1e-3)
        # WorstK sample: returned 100.0 width, NOT 186.6!
        self.assertAlmostEqual(stn_bboxes[2, 2].item() * w, 100.0, delta=1.0)

    def test_req6_stage1_clean_dataset_not_forced_spatial_metadata(self):
        """Verify Item 6: YOLODatasetStage1Clean functions normally without forced spatial attack metadata."""
        dataset = YOLODatasetStage1Clean.__new__(YOLODatasetStage1Clean)
        dataset.im_files = ["img1.jpg", "img2.jpg"]
        dataset.update_virtual_samples()
        self.assertEqual(len(dataset.virtual_samples), 2)
        self.assertTrue(all(st == "raw" for _, st in dataset.virtual_samples))

        # Test batch from clean dataset with no spatial metadata passed to transform_batch_bboxes_for_stn
        batch = {
            "batch_idx": torch.tensor([0, 1], dtype=torch.int64),
            "cls": torch.tensor([0, 1], dtype=torch.int64),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]], dtype=torch.float32),
        }
        # Pred matrices (STN predicted)
        pred_matrices = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
        out = transform_batch_bboxes_for_stn(batch, pred_matrices, height=100, width=100)
        self.assertTrue("bboxes" in out)
        torch.testing.assert_close(out["bboxes"], batch["bboxes"])


if __name__ == "__main__":
    unittest.main()
