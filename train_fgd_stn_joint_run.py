# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""FGD and STN joint defense training CLI launch script."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from ultralytics.defenses.fgd_stn_pipeline import canonicalize_mode
from ultralytics.models.yolo.detect.train_fgd_stn_joint import JointFGDSTNTrainer


def parse_bool(value: Any) -> bool:
    """Parse boolean CLI argument."""
    if isinstance(value, bool):
        return value
    val = str(value).strip().lower()
    if val in {"true", "1", "yes", "y"}:
        return True
    if val in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value (true/false), got: {value!r}")


def main():
    """Parse command-line arguments and launch FGD and STN joint training."""
    parser = argparse.ArgumentParser(description="FGD and STN joint defense training pipeline")

    # Model and dataset configuration
    parser.add_argument("--model", default="yolo12l.pt", help="Guidance detector model weights path (default: yolo12l.pt)")
    parser.add_argument("--data", default="coco128.yaml", help="Dataset configuration YAML path")
    parser.add_argument("--classes", nargs="+", type=int, default=None, help="Filter dataset by specified class IDs")

    # Defense pipeline and initialization settings
    parser.add_argument(
        "--preprocess_mode",
        "--mode",
        choices=["stn_fgd", "fgd_stn"],
        default="stn_fgd",
        help="Joint defense execution order: 'stn_fgd' (STN -> FGD) or 'fgd_stn' (FGD -> STN)",
    )
    parser.add_argument(
        "--init",
        choices=["pretrained", "scratch"],
        default="pretrained",
        help="Defense initialization mode: 'pretrained' (fine-tune loaded checkpoints) or 'scratch' (train from identity)",
    )
    parser.add_argument("--fgd_weights", default=None, help="Pretrained FGD checkpoint path")
    parser.add_argument("--stn_weights", default=None, help="Pretrained STN checkpoint path")

    # Attack generation and penetration settings
    parser.add_argument("--attack_weights", default=None, help="Optional separate attack model weights path, defaults to detector itself")
    parser.add_argument(
        "--attack_through_preprocess",
        type=parse_bool,
        default=True,
        help="Whether attack generation penetrates the full defense pipeline (default: true)",
    )
    parser.add_argument(
        "--attack_name",
        nargs="+",
        default=["pgd", "bim", "mim", "worstk"],
        help="Adversarial attack types generated during training (default: pgd bim mim worstk)",
    )
    parser.add_argument(
        "--attack_ratio",
        nargs="+",
        type=float,
        default=[0.10, 0.10, 0.40, 0.15],
        help="Initial training sample ratios for each attack type (default: 0.10 0.10 0.40 0.15)",
    )
    parser.add_argument("--p_clean", type=float, default=0.25, help="Fixed ratio of clean samples (default: 0.25)")
    parser.add_argument("--worstk_k", type=int, default=10, help="Worst-of-k candidate transform samples (default: 10)")
    parser.add_argument("--degrees", type=float, default=30.0, help="Worst-of-k maximum rotation degrees (default: 30.0)")
    parser.add_argument("--translate", type=float, default=0.10, help="Worst-of-k maximum translation fraction (default: 0.10)")

    # Joint loss weight configuration
    parser.add_argument("--lambda_fgd", type=float, default=1.0, help="FGD feature guidance loss weight (default: 1.0)")
    parser.add_argument("--lambda_geom", type=float, default=1.0, help="STN geometric control point loss weight (default: 1.0)")
    parser.add_argument("--lambda_recon", type=float, default=1.0, help="Masked image reconstruction loss weight (default: 1.0)")
    parser.add_argument("--lambda_det", type=float, default=0.0, help="Frozen YOLO detection loss weight (default: 0.0)")
    parser.add_argument("--lambda_identity", type=float, default=0.1, help="Clean sample identity preservation loss weight (default: 0.1)")

    # Optimizer and training hyperparameters
    parser.add_argument("--lr_fgd", type=float, default=1e-5, help="FGD denoiser initial learning rate (default: 1e-5)")
    parser.add_argument("--lr_stn", type=float, default=1e-4, help="STN spatial transformer initial learning rate (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Adam optimizer weight decay (default: 1e-5)")
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=4, help="Training batch size (default: 4)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution size (default: 640)")
    parser.add_argument("--device", default="0", help="Training device (e.g. 0 or cpu)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker processes (default: 4)")
    parser.add_argument("--cache", default="True", help="Dataset cache mode (ram, disk, True, False)")
    parser.add_argument(
        "--amp", default="bf16", choices=["bf16", "fp16", "fp32", "True", "False"], help="Mixed precision mode (default: bf16)"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")

    # Validation and checkpoint settings
    parser.add_argument("--val_interval", type=int, default=5, help="Validation interval in epochs (default: 5)")
    parser.add_argument(
        "--eval_weights",
        nargs="+",
        type=float,
        default=None,
        help="Validation evaluation weighting ratios (default: equal weights for Clean/PGD/BIM/MIM/Worst-of-K)",
    )
    parser.add_argument("--project", default="runs/train_fgd_stn_joint", help="Training output directory")
    parser.add_argument("--name", default="exp", help="Experiment name")
    parser.add_argument("--save_period", type=int, default=-1, help="Periodic checkpoint save interval (default: -1)")
    parser.add_argument("--resume", default=False, help="Resume training checkpoint path (e.g. weights/last.pt)")

    args = parser.parse_args()

    # Normalization and early validation
    mode = canonicalize_mode(args.preprocess_mode)
    if mode not in {"stn_fgd", "fgd_stn"}:
        raise ValueError(f"Joint training defense mode must be 'stn_fgd' or 'fgd_stn', got: {args.preprocess_mode}")

    init_mode = str(args.init).strip().lower()
    if init_mode not in {"pretrained", "scratch"}:
        raise ValueError(f"--init must be 'pretrained' or 'scratch', got: {args.init}")

    if not args.resume and init_mode == "pretrained":
        if not args.fgd_weights or not Path(args.fgd_weights).is_file():
            raise FileNotFoundError(f"Pretrained fine-tuning requires valid --fgd_weights path, got: {args.fgd_weights}")
        if not args.stn_weights or not Path(args.stn_weights).is_file():
            raise FileNotFoundError(f"Pretrained fine-tuning requires valid --stn_weights path, got: {args.stn_weights}")

    # Verify sum of ratios
    attack_ratios = [float(x) for x in args.attack_ratio]
    total_ratio = sum(attack_ratios) + float(args.p_clean)
    if abs(total_ratio - 1.0) > 1e-3:
        raise ValueError(
            f"Sum of attack ratios and clean sample ratio must equal 1.0, got: sum(attack_ratio)={sum(attack_ratios):.4f}, "
            f"p_clean={args.p_clean:.4f}, total={total_ratio:.4f}"
        )

    # Verify non-negative loss weights
    for w_name, w_val in [
        ("lambda_fgd", args.lambda_fgd),
        ("lambda_geom", args.lambda_geom),
        ("lambda_recon", args.lambda_recon),
        ("lambda_det", args.lambda_det),
        ("lambda_identity", args.lambda_identity),
    ]:
        if float(w_val) < 0.0:
            raise ValueError(f"Loss weight {w_name} must be non-negative, got: {w_val}")

    # Verify learning rates > 0
    if float(args.lr_fgd) <= 0.0:
        raise ValueError(f"FGD learning rate --lr_fgd must be > 0, got: {args.lr_fgd}")
    if float(args.lr_stn) <= 0.0:
        raise ValueError(f"STN learning rate --lr_stn must be > 0, got: {args.lr_stn}")

    # Cache parsing
    cache_val = args.cache
    if isinstance(cache_val, str):
        if cache_val.lower() in ("true", "1", "ram"):
            cache_val = "ram"
        elif cache_val.lower() in ("false", "0", "none"):
            cache_val = False

    overrides = {
        "model": args.model,
        "data": args.data,
        "classes": args.classes,
        "preprocess_mode": mode,
        "init": init_mode,
        "fgd_weights": args.fgd_weights,
        "stn_weights": args.stn_weights,
        "attack_weights": args.attack_weights,
        "attack_through_preprocess": args.attack_through_preprocess,
        "attack_name": args.attack_name,
        "attack_ratio": attack_ratios,
        "p_clean": float(args.p_clean),
        "worstk_k": int(args.worstk_k),
        "degrees": float(args.degrees),
        "translate": float(args.translate),
        "lambda_fgd": float(args.lambda_fgd),
        "lambda_geom": float(args.lambda_geom),
        "lambda_recon": float(args.lambda_recon),
        "lambda_det": float(args.lambda_det),
        "lambda_identity": float(args.lambda_identity),
        "lr_fgd": float(args.lr_fgd),
        "lr_stn": float(args.lr_stn),
        "weight_decay": float(args.weight_decay),
        "epochs": int(args.epochs),
        "batch": int(args.batch),
        "imgsz": int(args.imgsz),
        "device": str(args.device),
        "workers": int(args.workers),
        "cache": cache_val,
        "amp": args.amp,
        "seed": int(args.seed),
        "val_interval": int(args.val_interval),
        "eval_weights": args.eval_weights,
        "project": args.project,
        "name": args.name,
        "save_period": int(args.save_period),
        "resume": args.resume,
    }

    trainer = JointFGDSTNTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
