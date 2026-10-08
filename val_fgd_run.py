# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Evaluate robust object detection with a trained FGD residual denoiser."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

from ultralytics.attacks.attack_bridge import build_attacker
from ultralytics.attacks.attack_utils import setup_attack_model
from ultralytics.defenses.hgd_fgd import FGDDetectorWrapper, load_fgd_denoiser
from ultralytics.models.yolo.detect.train_adv_test import DetectionTrainer
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model


def parse_args():
    """Parse CLI arguments for FGD defense validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fgd_weights",
        default=None,
        help="Path to trained FGD checkpoint (weights/last.pt). Pass 'none' or leave unset to evaluate without denoiser.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to YOLO detector weights (e.g. yolo12l.pt). If unset, attempts to read guidance_model from --fgd_weights.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Dataset YAML file path (e.g. coco_train_traffic5k_test.yaml).",
    )
    parser.add_argument(
        "--attack_name",
        default="raw",
        help="Attack to evaluate: 'raw' (clean), 'pgd', 'bim', 'mim', or multiple space-separated attacks.",
    )
    parser.add_argument(
        "--attack_weights",
        default="",
        help="Optional separate attack checkpoint for transfer / black-box attacks. Empty or 'current' uses evaluated detector.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=None,
        help="Filter dataset by specific class IDs (e.g. --classes 0 1 2 3 5 6 7 9 11 12).",
    )
    parser.add_argument(
        "--attack_through_fgd",
        choices=["true", "false"],
        default="false",
        help="Adaptive white-box attack through FGD; false preserves Phase 1 evaluation.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Dataset evaluation split.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu'.")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument(
        "--quantize",
        type=int,
        choices=[8, 16],
        default=None,
        help="Inference precision (16 for FP16, None for FP32).",
    )
    parser.add_argument("--project", default="runs/val_fgd", help="Results project directory.")
    parser.add_argument("--name", default="exp", help="Results run name.")
    parser.add_argument("--save_json", action="store_true", default=False, help="Save evaluation results to JSON.")
    return parser.parse_args()


def main():
    """Execute FGD validation on the specified dataset and attack."""
    args = parse_args()

    device_str = str(args.device)
    device = torch.device(f"cuda:{device_str}" if device_str.isdigit() else device_str)

    denoiser, meta = load_fgd_denoiser(args.fgd_weights, device=device)

    detector_weights = args.weights
    if not detector_weights or str(detector_weights).strip().lower() in {"", "none"}:
        guidance = meta.get("guidance_model")
        if guidance and Path(guidance).exists():
            detector_weights = str(Path(guidance).resolve())
            LOGGER.info(f"Using guidance model from FGD checkpoint: {detector_weights}")
        else:
            detector_weights = "yolo12l.pt"
            LOGGER.info(f"Defaulting detector weights to: {detector_weights}")

    overrides = dict(
        model=detector_weights,
        data=args.data,
        imgsz=args.imgsz,
        split=args.split,
        batch=args.batch,
        device=args.device,
        attack_name=args.attack_name,
        project=args.project,
        name=args.name,
        classes=args.classes,
        quantize=args.quantize,
        save_json=args.save_json,
        workers=args.workers,
    )

    trainer = DetectionTrainer(overrides=overrides, attack_weights=args.attack_weights)
    trainer._setup_train()

    eval_model = trainer.model.eval()
    for p in eval_model.parameters():
        p.requires_grad = False

    is_raw = str(args.attack_name).strip().lower() in {"raw", "none", "clean"}
    if not is_raw:
        attack_weights = args.attack_weights
        use_current_model = (attack_weights is None) or (str(attack_weights).strip().lower() in {"", "none", "current"})
        if use_current_model:
            attack_model = copy.deepcopy(unwrap_model(eval_model)).eval()
            for p in attack_model.parameters():
                p.requires_grad_(True)
        elif attack_weights:
            attack_model = setup_attack_model(
                attack_weights,
                device=trainer.device,
                nc=trainer.data["nc"],
                training=False,
                imgsz=args.imgsz,
            )
            attack_model.eval()
            for p in attack_model.parameters():
                p.requires_grad_(True)
        else:
            attack_model = None

        if attack_model is not None:
            if args.attack_through_fgd == "true" and denoiser is not None:
                attack_model = FGDDetectorWrapper(denoiser, attack_model)
            attacker = build_attacker(args.attack_name, model=attack_model, img_size=args.imgsz)
            trainer.attacker = attacker
            trainer.attackers = {args.attack_name: attacker}

    validator = trainer.get_validator()
    if args.attack_through_fgd == "true" and denoiser is not None:
        from ultralytics.models.yolo.detect.val_adv_fgd_phase2 import DetectionValidator

        validator = DetectionValidator(
            validator.dataloader, save_dir=validator.save_dir, args=validator.args, _callbacks=trainer.callbacks
        )
    validator.model = eval_model
    validator.attacker = getattr(trainer, "attacker", None)
    validator.attackers = getattr(trainer, "attackers", {})
    validator.attack_name = args.attack_name
    validator.denoiser = denoiser
    validator.strict_attacks = args.attack_through_fgd == "true" and denoiser is not None

    LOGGER.info(
        f"Starting validation: attack='{args.attack_name}', denoiser={'enabled' if denoiser is not None else 'disabled'}"
    )
    stats = validator(model=validator.model)
    LOGGER.info(f"Validation finished. Results: {stats}")
    return stats


if __name__ == "__main__":
    main()
