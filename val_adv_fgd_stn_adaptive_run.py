# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""CLI runner for Standalone Adaptive Adversarial Validation with FGD + STN Defense."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ultralytics.data.utils import check_det_dataset
from ultralytics.defenses.fgd_stn_pipeline import (
    MODE_DESCRIPTIONS,
    VALID_MODES,
    build_fgd_stn_pipeline,
    canonicalize_mode,
)
from ultralytics.models.yolo.detect.val_adv_fgd_stn_adaptive import (
    DetectionValidatorAdvFGDSTNAdaptive,
)


def parse_bool(value: Any) -> bool:
    """Accept explicit true/false/1/0 toggles."""
    if isinstance(value, bool):
        return value
    val = str(value).strip().lower()
    if val in {"true", "1", "yes", "y"}:
        return True
    if val in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}")


def main():
    """Parse CLI options and execute standalone validation."""
    parser = argparse.ArgumentParser(
        description="Standalone Adaptive Adversarial Validation with Unified FGD + STN Defenses."
    )
    parser.add_argument(
        "--model",
        "--weights",
        default="yolo12l.pt",
        help="Path to YOLO detector weights (e.g. yolo12l.pt or best.pt).",
    )
    parser.add_argument(
        "--preprocess_mode",
        "--mode",
        default="fgd_stn",
        help="Preprocessing mode: none, fgd, stn, fgd_stn, stn_fgd.",
    )
    parser.add_argument(
        "--fgd_weights",
        default=None,
        help="Path to trained FGD denoiser checkpoint.",
    )
    parser.add_argument(
        "--stn_weights",
        default=None,
        help="Path to trained STN checkpoint.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Dataset YAML path (e.g. coco_train_traffic5k_test.yaml).",
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
        help="Filter dataset by specific class IDs.",
    )
    parser.add_argument(
        "--attack_name",
        nargs="+",
        default=None,
        help="Attacks to evaluate (default: pgd bim mim worstk).",
    )
    parser.add_argument(
        "--eval_weights",
        nargs="+",
        default=None,
        help="Fixed evaluation weights for fitness (default: 0.25 0.15 0.15 0.20).",
    )
    parser.add_argument(
        "--attack_through_preprocess",
        type=parse_bool,
        default=True,
        help="Whether attacks pass through defense preprocessing pipeline (default: true).",
    )
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8).")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640).")
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu'.")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold.")
    parser.add_argument(
        "--project",
        default="runs/val_adv_fgd_stn_adaptive",
        help="Project directory for results.",
    )
    parser.add_argument("--name", default="exp", help="Experiment name.")

    args = parser.parse_args()
    mode = canonicalize_mode(args.preprocess_mode)

    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    validator_args = dict(
        task="detect",
        model=args.model,
        data=args.data,
        split=args.split,
        classes=args.classes,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        preprocess_mode=mode,
        fgd_weights=args.fgd_weights,
        stn_weights=args.stn_weights,
        attack_through_preprocess=args.attack_through_preprocess,
        attack_name=args.attack_name,
        eval_weights=args.eval_weights,
    )

    validator = DetectionValidatorAdvFGDSTNAdaptive(
        save_dir=save_dir,
        args=validator_args,
        preprocess_mode=mode,
        fgd_weights=args.fgd_weights,
        stn_weights=args.stn_weights,
        attack_through_preprocess=args.attack_through_preprocess,
    )
    validator()


if __name__ == "__main__":
    main()
