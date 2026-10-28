# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""CLI runner for FGD + STN Defense with Adaptive Adversarial Training."""

from __future__ import annotations

import argparse
import os
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from ultralytics.defenses.fgd_stn_pipeline import canonicalize_mode
from ultralytics.models.yolo.detect.train_adv_fgd_stn_adaptive import (
    DetectionTrainerAdvFGDSTNAdaptive,
)


def parse_bool(value: Any) -> bool:
    """Accept explicit true/false/1/0 toggles without silently treating typos as false."""
    if isinstance(value, bool):
        return value
    val = str(value).strip().lower()
    if val in {"true", "1", "yes", "y"}:
        return True
    if val in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}")


def main():
    """Parse CLI options and execute Combined FGD + STN Adaptive Adversarial Training."""
    parser = argparse.ArgumentParser(
        description="FGD + STN Defense with Adaptive Adversarial Training across E1-E5 Preprocessing Modes."
    )

    # Preprocessing and Defense Modules
    parser.add_argument(
        "--preprocess_mode",
        "--mode",
        default="none",
        help=(
            "Preprocessing defense mode: 'none' (YOLO only), 'fgd' (FGD -> YOLO), "
            "'stn' (STN -> YOLO), 'fgd_stn' (FGD -> STN -> YOLO), 'stn_fgd' (STN -> FGD -> YOLO)."
        ),
    )
    parser.add_argument(
        "--fgd_weights",
        default=None,
        help="Path to trained FGD checkpoint (weights/best.pt). Required for modes 'fgd', 'fgd_stn', 'stn_fgd'.",
    )
    parser.add_argument(
        "--stn_weights",
        default=None,
        help="Path to trained STN checkpoint (weights/best.pt). Required for modes 'stn', 'fgd_stn', 'stn_fgd'.",
    )

    # Adaptive Ratio Settings
    parser.add_argument(
        "--adaptive_ratio",
        type=parse_bool,
        default=True,
        help="Enable dynamic attack ratio adjustment every update_interval epochs (default: true).",
    )
    parser.add_argument(
        "--p_clean",
        type=float,
        default=0.25,
        help="Fixed Clean training ratio (default: 0.25).",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=5.0,
        help="Softmax temperature tau for dynamic ratio updates (default: 5.0).",
    )
    parser.add_argument(
        "--update_interval",
        type=int,
        default=5,
        help="Epoch interval for recalculating training ratios (default: 5).",
    )
    parser.add_argument(
        "--eval_weights",
        nargs="+",
        default=None,
        help="Fixed validation weights for checkpoint selection (default: 0.25 0.15 0.15 0.20 for PGD BIM MIM WorstK).",
    )

    # Attack Configuration
    parser.add_argument(
        "--attack_name",
        nargs="+",
        default=None,
        help="Attack names to train against (default: pgd bim mim worstk).",
    )
    parser.add_argument(
        "--attack_ratio",
        nargs="+",
        default=None,
        help="Initial training ratios for attacks (default: 0.10 0.10 0.40 0.15).",
    )
    parser.add_argument(
        "--attack_weights",
        default=None,
        help="Optional external attack model weights. Default uses current YOLO detector (white-box).",
    )
    parser.add_argument(
        "--attack_through_preprocess",
        type=parse_bool,
        default=True,
        help="Whether attack generation passes through the frozen defense pipeline (default: true).",
    )

    # Standard Training Settings
    parser.add_argument("--model", default="yolo12l.pt", help="Base YOLO detector checkpoint or YAML.")
    parser.add_argument("--data", default=None, help="Dataset YAML path (e.g. coco_train_traffic5k.yaml).")
    parser.add_argument("--epochs", type=int, default=5, help="Total training epochs (default: 5).")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--device", default="0", help="CUDA device index (e.g. 0 or 0,1) or 'cpu'.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size in pixels (default: 640).")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader worker processes.")
    parser.add_argument(
        "--project",
        default="runs/train_adv_fgd_stn_adaptive",
        help="Project directory for logging and checkpoints.",
    )
    parser.add_argument("--name", default="exp", help="Experiment name (default: exp).")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="Resume training from checkpoint (e.g. --resume runs/.../last.pt).",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Filter dataset by specific class indices.",
    )
    parser.add_argument(
        "--cache",
        nargs="?",
        const=True,
        default=True,
        help="Cache dataset images in RAM (True/ram, disk, False).",
    )
    parser.add_argument(
        "--amp",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Precision mode (default: bf16).",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of dataset to use for short smoke runs.",
    )

    args = parser.parse_args()

    cache_val = args.cache
    if isinstance(cache_val, str):
        if cache_val.lower() in ("true", "1", "ram"):
            cache_val = "ram"
        elif cache_val.lower() in ("false", "0", "none"):
            cache_val = False

    mode = canonicalize_mode(args.preprocess_mode)

    overrides = {
        "model": args.model,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": args.device,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "resume": args.resume,
        "classes": args.classes,
        "cache": cache_val,
        "amp": args.amp,
        "fraction": args.fraction,
        "preprocess_mode": mode,
        "fgd_weights": args.fgd_weights,
        "stn_weights": args.stn_weights,
        "adaptive_ratio": args.adaptive_ratio,
        "attack_through_preprocess": args.attack_through_preprocess,
        "p_clean": args.p_clean,
        "tau": args.tau,
        "update_interval": args.update_interval,
    }

    if args.data is not None:
        overrides["data"] = args.data
    elif not args.resume:
        overrides["data"] = "coco128.yaml"

    if args.attack_weights is not None:
        overrides["attack_weights"] = args.attack_weights

    overrides["attack_name"] = args.attack_name if args.attack_name is not None else ["pgd", "bim", "mim", "worstk"]
    overrides["attack_ratio"] = args.attack_ratio if args.attack_ratio is not None else [0.10, 0.10, 0.40, 0.15]
    overrides["eval_weights"] = args.eval_weights if args.eval_weights is not None else [0.25, 0.15, 0.15, 0.20]

    trainer = DetectionTrainerAdvFGDSTNAdaptive(overrides=overrides, attack_weights=args.attack_weights)
    trainer.train()


if __name__ == "__main__":
    main()
