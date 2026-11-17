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


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for Combined FGD + STN Adaptive Adversarial Training."""
    parser = argparse.ArgumentParser(
        description="FGD + STN Defense with Adaptive Adversarial Training across E1-E5 Preprocessing Modes.",
        argument_default=argparse.SUPPRESS,
    )

    # Preprocessing and Defense Modules
    parser.add_argument(
        "--preprocess_mode",
        "--mode",
        help=(
            "Preprocessing defense mode: 'none' (YOLO only), 'fgd' (FGD -> YOLO), "
            "'stn' (STN -> YOLO), 'fgd_stn' (FGD -> STN -> YOLO), 'stn_fgd' (STN -> FGD -> YOLO)."
        ),
    )
    parser.add_argument(
        "--fgd_weights",
        help="Path to trained FGD checkpoint (weights/best.pt). Required for modes 'fgd', 'fgd_stn', 'stn_fgd'.",
    )
    parser.add_argument(
        "--stn_weights",
        help="Path to trained STN checkpoint (weights/best.pt). Required for modes 'stn', 'fgd_stn', 'stn_fgd'.",
    )

    # Adaptive Ratio Settings
    parser.add_argument(
        "--adaptive_ratio",
        type=parse_bool,
        help="Enable dynamic attack ratio adjustment every update_interval epochs (default: true).",
    )
    parser.add_argument(
        "--p_clean",
        type=float,
        help="Fixed Clean training ratio (default: 0.25).",
    )
    parser.add_argument(
        "--tau",
        type=float,
        help="Softmax temperature tau for dynamic ratio updates (default: 5.0).",
    )
    parser.add_argument(
        "--update_interval",
        type=int,
        help="Epoch interval for recalculating training ratios (default: 5).",
    )
    parser.add_argument(
        "--eval_weights",
        nargs="+",
        help="Fixed validation weights for checkpoint selection (default: 0.25 0.15 0.15 0.20 for PGD BIM MIM WorstK).",
    )

    # Attack Configuration
    parser.add_argument(
        "--attack_name",
        nargs="+",
        help="Attack names to train against (default: pgd bim mim worstk).",
    )
    parser.add_argument(
        "--attack_ratio",
        nargs="+",
        help="Initial training ratios for attacks (default: 0.10 0.10 0.40 0.15).",
    )
    parser.add_argument(
        "--attack_weights",
        help="Optional external attack model weights. Default uses current YOLO detector (white-box).",
    )
    parser.add_argument(
        "--attack_through_preprocess",
        type=parse_bool,
        help="Whether attack generation passes through the frozen defense pipeline (default: true).",
    )

    # Standard Training Settings
    parser.add_argument("--model", help="Base YOLO detector checkpoint or YAML (default: yolo12l.pt).")
    parser.add_argument("--data", help="Dataset YAML path (e.g. coco_train_traffic5k.yaml).")
    parser.add_argument("--epochs", type=int, help="Total training epochs (default: 5).")
    parser.add_argument("--batch", type=int, help="Batch size (default: 16).")
    parser.add_argument("--device", help="CUDA device index (e.g. 0 or 0,1) or 'cpu' (default: 0).")
    parser.add_argument("--imgsz", type=int, help="Image size in pixels (default: 640).")
    parser.add_argument("--workers", type=int, help="DataLoader worker processes.")
    parser.add_argument(
        "--project",
        help="Project directory for logging and checkpoints (default: runs/train_adv_fgd_stn_adaptive).",
    )
    parser.add_argument("--name", help="Experiment name (default: exp).")
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
        help="Filter dataset by specific class indices.",
    )
    parser.add_argument(
        "--cache",
        nargs="?",
        const=True,
        help="Cache dataset images in RAM (True/ram, disk, False).",
    )
    parser.add_argument(
        "--amp",
        choices=["bf16", "fp16", "fp32"],
        help="Precision mode (default: bf16).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility (default: None, inherits default).",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        help="Fraction of dataset to use for short smoke runs (default: 1.0).",
    )
    return parser


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    return build_parser().parse_args(args)


def main():
    """Parse CLI options and execute Combined FGD + STN Adaptive Adversarial Training."""
    args = parse_args()

    cache_val = getattr(args, "cache", True)
    if isinstance(cache_val, str):
        if cache_val.lower() in ("true", "1", "ram"):
            cache_val = "ram"
        elif cache_val.lower() in ("false", "0", "none"):
            cache_val = False

    is_resume = bool(getattr(args, "resume", False))
    if is_resume:
        overrides = dict(resume=args.resume)
        if hasattr(args, "model"):
            overrides["model"] = args.model
        if hasattr(args, "data"):
            overrides["data"] = args.data
        if hasattr(args, "epochs"):
            overrides["epochs"] = args.epochs
        if hasattr(args, "batch"):
            overrides["batch"] = args.batch
        if hasattr(args, "device"):
            overrides["device"] = args.device
        if hasattr(args, "imgsz"):
            overrides["imgsz"] = args.imgsz
        if hasattr(args, "project"):
            overrides["project"] = args.project
        if hasattr(args, "name"):
            overrides["name"] = args.name
        if hasattr(args, "classes"):
            overrides["classes"] = args.classes
        if hasattr(args, "cache"):
            overrides["cache"] = cache_val
        if hasattr(args, "amp"):
            overrides["amp"] = args.amp
        if hasattr(args, "fraction"):
            overrides["fraction"] = args.fraction
        if hasattr(args, "preprocess_mode"):
            overrides["preprocess_mode"] = canonicalize_mode(args.preprocess_mode)
        if hasattr(args, "fgd_weights"):
            overrides["fgd_weights"] = args.fgd_weights
        if hasattr(args, "stn_weights"):
            overrides["stn_weights"] = args.stn_weights
        if hasattr(args, "adaptive_ratio"):
            overrides["adaptive_ratio"] = args.adaptive_ratio
        if hasattr(args, "attack_through_preprocess"):
            overrides["attack_through_preprocess"] = args.attack_through_preprocess
        if hasattr(args, "p_clean"):
            overrides["p_clean"] = args.p_clean
        if hasattr(args, "tau"):
            overrides["tau"] = args.tau
        if hasattr(args, "update_interval"):
            overrides["update_interval"] = args.update_interval
        if hasattr(args, "attack_name"):
            overrides["attack_name"] = args.attack_name
        if hasattr(args, "attack_ratio"):
            overrides["attack_ratio"] = args.attack_ratio
        if hasattr(args, "eval_weights"):
            overrides["eval_weights"] = args.eval_weights
    else:
        mode = canonicalize_mode(getattr(args, "preprocess_mode", "none"))
        overrides = {
            "model": getattr(args, "model", "yolo12l.pt"),
            "data": getattr(args, "data", "coco128.yaml"),
            "epochs": getattr(args, "epochs", 5),
            "batch": getattr(args, "batch", 16),
            "device": getattr(args, "device", "0"),
            "imgsz": getattr(args, "imgsz", 640),
            "project": getattr(args, "project", "runs/train_adv_fgd_stn_adaptive"),
            "name": getattr(args, "name", "exp"),
            "resume": False,
            "classes": getattr(args, "classes", None),
            "cache": cache_val,
            "amp": getattr(args, "amp", "bf16"),
            "fraction": getattr(args, "fraction", 1.0),
            "preprocess_mode": mode,
            "fgd_weights": getattr(args, "fgd_weights", None),
            "stn_weights": getattr(args, "stn_weights", None),
            "adaptive_ratio": getattr(args, "adaptive_ratio", True),
            "attack_through_preprocess": getattr(args, "attack_through_preprocess", True),
            "p_clean": getattr(args, "p_clean", 0.25),
            "tau": getattr(args, "tau", 5.0),
            "update_interval": getattr(args, "update_interval", 5),
            "attack_name": getattr(args, "attack_name", ["pgd", "bim", "mim", "worstk"]),
            "attack_ratio": getattr(args, "attack_ratio", [0.10, 0.10, 0.40, 0.15]),
            "eval_weights": getattr(args, "eval_weights", [0.25, 0.15, 0.15, 0.20]),
        }

    if hasattr(args, "workers") and args.workers is not None:
        overrides["workers"] = args.workers
    if hasattr(args, "seed") and args.seed is not None:
        overrides["seed"] = int(args.seed)
    if hasattr(args, "attack_weights") and args.attack_weights is not None:
        overrides["attack_weights"] = args.attack_weights

    trainer = DetectionTrainerAdvFGDSTNAdaptive(
        overrides=overrides, attack_weights=getattr(args, "attack_weights", None)
    )
    trainer.train()


if __name__ == "__main__":
    main()
