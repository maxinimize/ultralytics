"""Entry point for HYDRA fixed-mask fine-tuning (Milestones 2 and 3).

Supports both:
    - Clean fixed-mask fine-tuning (HydraDetectionTrainer)
    - Robust adversarial fixed-mask fine-tuning (HydraAdversarialDetectionTrainer)

Expected workflow:
    1. Load a HYDRA checkpoint containing optimized scores or fixed masks (from hydra_prune.py).
    2. Materialize/freeze the requested layer-wise top-k mask.
    3. Freeze score parameters and unfreeze retained convolution weights.
    4. Fine-tune retained weights using the standard or robust adversarial trainer.
    5. Enforce zero values for pruned connections after every optimizer step.
    6. Validate and save normal Ultralytics metrics and checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics.utils import LOGGER


def parse_args():
    """Parse options for fixed-mask fine-tuning."""
    parser = argparse.ArgumentParser(description="HYDRA Fixed-Mask Fine-Tuning Tool")
    parser.add_argument("--model", required=True, help="HYDRA score checkpoint (.pt) from score search.")
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML file.")
    parser.add_argument("--keep-ratio", type=float, required=True, help="Fraction of retained connections (must match score search).")
    parser.add_argument("--epochs", type=int, default=30, help="Number of fine-tuning epochs.")
    parser.add_argument("--device", default="0", help="Target device (e.g. '0', '0,1', or 'cpu').")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader worker processes.")
    parser.add_argument("--project", default="runs/hydra_finetune", help="Project output directory.")
    parser.add_argument("--name", default=None, help="Experiment run name.")
    parser.add_argument("--classes", type=int, nargs="+", default=None, help="Optional class filtering indices.")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate for fine-tuning.")

    # Adversarial parameters (Milestone 3)
    parser.add_argument(
        "--attack_name",
        nargs="+",
        default=None,
        help="Adversarial attack names (e.g. pgd, bim, mim, worstk). If specified, robust fine-tuning is used.",
    )
    parser.add_argument(
        "--attack_ratio",
        nargs="+",
        default=None,
        help="Virtual sample ratios for adversarial attacks (e.g. 0.5 or 0.4 0.5).",
    )
    parser.add_argument(
        "--attack_weights",
        default="",
        help="Optional surrogate model path for attacks. Default '' attacks the live model.",
    )
    parser.add_argument(
        "--attack_num",
        type=int,
        default=None,
        help="Number of attack types. Automatically derived if omitted.",
    )
    return parser.parse_args()


def main():
    """Launch fixed-mask fine-tuning."""
    args = parse_args()
    overrides = {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "device": args.device,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "project": args.project,
        "hydra_stage": "finetune",
        "hydra_keep_ratio": args.keep_ratio,
    }
    if args.name is not None:
        overrides["name"] = args.name
    if args.classes is not None:
        overrides["classes"] = args.classes
    if args.lr0 is not None:
        overrides["lr0"] = args.lr0

    # Determine whether to use robust or clean trainer
    is_adversarial = bool(
        args.attack_name and any(a.lower() not in {"", "none", "clean"} for a in args.attack_name)
    )

    if is_adversarial:
        from ultralytics.models.yolo.detect.hydra_adversarial import HydraAdversarialDetectionTrainer

        overrides["attack_name"] = args.attack_name
        if args.attack_ratio is not None:
            overrides["attack_ratio"] = [float(r) for r in args.attack_ratio]
        else:
            overrides["attack_ratio"] = [0.5] * len(args.attack_name)

        attack_num = args.attack_num if args.attack_num is not None else len(args.attack_name)
        overrides["attack_num"] = attack_num

        LOGGER.info(
            f"Starting ROBUST fixed-mask fine-tuning with attacks: {overrides['attack_name']} "
            f"(ratios={overrides['attack_ratio']})"
        )
        trainer = HydraAdversarialDetectionTrainer(overrides=overrides, attack_weights=args.attack_weights or "")
    else:
        from ultralytics.models.yolo.detect.hydra_train import HydraDetectionTrainer

        LOGGER.info("Starting CLEAN fixed-mask fine-tuning (HydraDetectionTrainer)")
        trainer = HydraDetectionTrainer(overrides=overrides)

    trainer.train()

    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("HYDRA FIXED-MASK FINE-TUNING COMPLETE")
    LOGGER.info(f"Retained keep ratio: {args.keep_ratio:.4f}")
    best_path = getattr(trainer, "best", None) or (trainer.wdir / "best.pt")
    last_path = getattr(trainer, "last", None) or (trainer.wdir / "last.pt")
    LOGGER.info(f"Best fine-tuned checkpoint: {best_path}")
    LOGGER.info(f"Last fine-tuned checkpoint: {last_path}")
    LOGGER.info("=" * 60 + "\n")


if __name__ == "__main__":
    main()

