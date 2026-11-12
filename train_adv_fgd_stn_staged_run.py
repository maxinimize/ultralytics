# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""CLI runner for Staged Adaptive Adversarial Training with Frozen FGD and STN for YOLO."""

from __future__ import annotations

import argparse
import os
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from ultralytics.defenses.fgd_stn_pipeline import canonicalize_mode
from ultralytics.models.yolo.detect.train_adv_fgd_stn_staged import (
    DetectionTrainerAdvFGDSTNStaged,
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
    """Parse CLI arguments and run Staged Adaptive Adversarial Training."""
    parser = argparse.ArgumentParser(
        description="Staged Adaptive Adversarial Training with Frozen FGD and STN for YOLO12l."
    )

    # Staged Pipeline Controls
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        default=1,
        help="Training stage: 1 (Clean Fine-tuning) or 2 (Adaptive Adversarial Training).",
    )
    parser.add_argument(
        "--stage1_weights",
        default=None,
        help="Path to Stage 1 best.pt checkpoint for initializing Stage 2.",
    )
    parser.add_argument(
        "--train_head_only",
        type=parse_bool,
        default=True,
        help="Whether to train only the YOLO Detection Head while freezing Backbone and Neck (default: true).",
    )
    parser.add_argument(
        "--freeze_bn",
        type=parse_bool,
        default=True,
        help="Whether to keep BatchNorm statistics in eval mode for all frozen layers (default: true).",
    )

    # Preprocessing and Defense Modules
    parser.add_argument(
        "--preprocess_mode",
        "--mode",
        default="stn_fgd",
        help="Preprocessing defense mode: 'none', 'fgd', 'stn', 'fgd_stn', 'stn_fgd' (default: stn_fgd).",
    )
    parser.add_argument(
        "--fgd_weights",
        default="fgd_yolo12l.pt",
        help="Path to trained FGD checkpoint (weights/best.pt). Required for modes 'fgd', 'fgd_stn', 'stn_fgd'.",
    )
    parser.add_argument(
        "--stn_weights",
        default="runs/detect/runs/train_stn/stn_train_yolo12l_coco_train_traffic5k_ep50_b8_k10_deg30.0_tr0.10/weights/best.pt",
        help="Path to trained STN checkpoint (weights/best.pt). Required for modes 'stn', 'fgd_stn', 'stn_fgd'.",
    )

    # Adaptive Ratio Settings
    parser.add_argument(
        "--adaptive_ratio",
        type=parse_bool,
        default=None,
        help="Enable dynamic attack ratio adjustment every update_interval epochs (default: false for stage 1, true for stage 2).",
    )
    parser.add_argument(
        "--p_clean",
        type=float,
        default=None,
        help="Fixed Clean training ratio (default: 1.0 for stage 1, 0.65 for stage 2).",
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
        help="Validation weights for checkpoint fitness (Stage 1: 1.0; Stage 2 default: 0.20 0.20 0.15 0.15 0.30 for Clean PGD BIM MIM WorstK).",
    )

    # Attack Configuration
    parser.add_argument(
        "--attack_name",
        nargs="+",
        default=None,
        help="Attack names to train against (default: empty for stage 1, pgd bim mim worstk for stage 2).",
    )
    parser.add_argument(
        "--attack_ratio",
        nargs="+",
        default=None,
        help="Initial training ratios for attacks (default: empty for stage 1, 0.12 0.05 0.10 0.08 for stage 2).",
    )
    parser.add_argument(
        "--attack_weights",
        default="current",
        help="Surrogate model weights for attack generation: 'current' (white-box AT) or path to external surrogate.",
    )
    parser.add_argument(
        "--attack_through_preprocess",
        type=parse_bool,
        default=True,
        help="Whether attack generation passes through the frozen defense pipeline (default: true).",
    )
    parser.add_argument(
        "--strict_attacks",
        type=parse_bool,
        default=True,
        help="Raise explicit errors if adversarial generation fails, prohibiting silent fallback (default: true).",
    )

    # Standard Training Settings
    parser.add_argument(
        "--model",
        default="yolo12l.pt",
        help="Base YOLO detector checkpoint (default: yolo12l.pt). Overridden by --stage1_weights in Stage 2.",
    )
    parser.add_argument(
        "--data",
        default="coco_train_traffic5k.yaml",
        help="Dataset YAML path (default: coco_train_traffic5k.yaml).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Total training epochs (default: 5 for stage 1, 20 for stage 2).",
    )
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size in pixels (default: 640).")
    parser.add_argument("--optimizer", default="AdamW", help="Optimizer name (default: AdamW).")
    parser.add_argument("--lr0", type=float, default=1e-5, help="Initial learning rate (default: 1e-5).")
    parser.add_argument("--warmup_epochs", type=float, default=0.0, help="Warmup epochs (default: 0.0).")
    parser.add_argument("--warmup_bias_lr", type=float, default=0.0, help="Warmup bias learning rate (default: 0.0).")
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu' (default: 0).")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader worker processes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")
    parser.add_argument(
        "--project",
        default="runs/train_adv_fgd_stn_staged",
        help="Project directory for logging and checkpoints.",
    )
    parser.add_argument("--name", default=None, help="Experiment name.")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="Resume interrupted training run (e.g. --resume runs/.../last.pt).",
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

    # Normalize cache argument
    cache_val = args.cache
    if isinstance(cache_val, str):
        if cache_val.lower() in ("true", "1", "ram"):
            cache_val = "ram"
        elif cache_val.lower() in ("false", "0", "none"):
            cache_val = False

    mode = canonicalize_mode(args.preprocess_mode)

    # Stage defaults
    if args.stage == 1:
        default_name = "stage1_clean_finetune"
        default_epochs = 5
        default_p_clean = 1.0
        default_adaptive = False
        default_att_names = []
        default_att_ratios = []
        default_eval_w = [1.0]
    else:
        default_name = "stage2_adaptive_at"
        default_epochs = 20
        default_p_clean = 0.65
        default_adaptive = True
        default_att_names = ["pgd", "bim", "mim", "worstk"]
        default_att_ratios = [0.12, 0.05, 0.10, 0.08]
        default_eval_w = [0.20, 0.20, 0.15, 0.15, 0.30]

    epochs = args.epochs if args.epochs is not None else default_epochs
    p_clean = args.p_clean if args.p_clean is not None else default_p_clean
    adaptive_ratio = args.adaptive_ratio if args.adaptive_ratio is not None else default_adaptive
    attack_name = args.attack_name if args.attack_name is not None else default_att_names
    attack_ratio = args.attack_ratio if args.attack_ratio is not None else default_att_ratios
    eval_weights = args.eval_weights if args.eval_weights is not None else default_eval_w
    exp_name = args.name if args.name is not None else default_name

    # Check model selection for Stage 2
    model_path = args.model
    if args.stage == 2 and args.stage1_weights:
        model_path = args.stage1_weights

    overrides = {
        "stage": args.stage,
        "stage1_weights": args.stage1_weights,
        "train_head_only": args.train_head_only,
        "freeze_bn": args.freeze_bn,
        "model": model_path,
        "data": args.data,
        "epochs": epochs,
        "batch": args.batch,
        "device": args.device,
        "imgsz": args.imgsz,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "warmup_epochs": args.warmup_epochs,
        "warmup_bias_lr": args.warmup_bias_lr,
        "seed": args.seed,
        "project": args.project,
        "name": exp_name,
        "resume": args.resume,
        "cache": cache_val,
        "amp": args.amp,
        "fraction": args.fraction,
        "preprocess_mode": mode,
        "fgd_weights": args.fgd_weights,
        "stn_weights": args.stn_weights,
        "adaptive_ratio": adaptive_ratio,
        "attack_through_preprocess": args.attack_through_preprocess,
        "p_clean": p_clean,
        "tau": args.tau,
        "update_interval": args.update_interval,
        "eval_weights": eval_weights,
        "attack_name": attack_name,
        "attack_ratio": attack_ratio,
        "attack_weights": args.attack_weights,
        "strict_attacks": args.strict_attacks,
    }

    if args.workers is not None:
        overrides["workers"] = args.workers
    if args.classes is not None:
        overrides["classes"] = args.classes

    trainer = DetectionTrainerAdvFGDSTNStaged(overrides=overrides, attack_weights=args.attack_weights)
    trainer.train()


if __name__ == "__main__":
    main()
