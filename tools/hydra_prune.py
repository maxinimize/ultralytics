"""Entry point for HYDRA score search (Milestones 1, 2, and 3).

Supports both:
    - Clean score search (HydraDetectionTrainer)
    - Robust adversarial score search (HydraAdversarialDetectionTrainer)

Expected workflow:
    1. Load existing dense or adversarially-trained Ultralytics checkpoint.
    2. Convert selected Conv2d leaves to HydraConv2d.
    3. Verify keep_ratio=1 equivalence (preflight check).
    4. Initialize scaled importance scores.
    5. Freeze dense weights and optimize popup_scores only (under clean or robust objective).
    6. Save optimized-score checkpoint and metadata.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.nn.pruning.hydra_convert import convert_to_hydra, iter_hydra_layers
from ultralytics.nn.pruning.hydra_layer import HydraMode
from ultralytics.nn.pruning.hydra_score import initialize_hydra_scores
from ultralytics.utils import LOGGER


def parse_args():
    """Parse orchestration options for score search."""
    parser = argparse.ArgumentParser(description="HYDRA Score Search Pruning Tool")
    parser.add_argument("--model", required=True, help="Input dense or adversarially-trained checkpoint (.pt).")
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML file.")
    parser.add_argument("--keep-ratio", type=float, default=0.5, help="Fraction of retained connections (e.g. 0.5 for 50% sparsity).")
    parser.add_argument("--epochs", type=int, default=20, help="Number of score search epochs.")
    parser.add_argument("--device", default="0", help="Target device (e.g. '0', '0,1', or 'cpu').")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader worker processes.")
    parser.add_argument("--project", default="runs/hydra_prune", help="Project output directory.")
    parser.add_argument("--name", default=None, help="Experiment run name.")
    parser.add_argument("--classes", type=int, nargs="+", default=None, help="Optional class filtering indices.")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate for score optimizer.")
    parser.add_argument("--preflight", action="store_true", default=True, help="Run keep_ratio=1 equivalence preflight check.")
    parser.add_argument("--no-preflight", action="store_false", dest="preflight", help="Skip preflight check.")

    # Adversarial parameters (Milestone 3)
    parser.add_argument(
        "--attack_name",
        nargs="+",
        default=None,
        help="Adversarial attack names (e.g. pgd, bim, mim, worstk). If specified, robust score search is used.",
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


def run_preflight_equivalence_check(model_path: str):
    """Verify that a converted model with keep_ratio=1.0 matches the original dense model."""
    yolo = YOLO(model_path)
    dense_model = copy.deepcopy(yolo.model).to("cpu").eval()
    hydra_model = copy.deepcopy(yolo.model).to("cpu").eval()

    convert_to_hydra(hydra_model, keep_ratio=1.0)
    initialize_hydra_scores(hydra_model)

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        dense_out = dense_model(x)
        for _, layer in iter_hydra_layers(hydra_model):
            layer.set_mode(HydraMode.SCORE)
        hydra_out = hydra_model(x)

    def diff(o1, o2):
        if isinstance(o1, torch.Tensor):
            return (o1 - o2).abs().max().item()
        elif isinstance(o1, (list, tuple)):
            return max(diff(a, b) for a, b in zip(o1, o2))
        elif isinstance(o1, dict):
            return max(diff(o1[k], o2[k]) for k in o1)
        return 0.0

    max_diff = diff(dense_out, hydra_out)
    if max_diff > 1e-4:
        raise ValueError(f"Preflight check failed! Max difference: {max_diff}")
    LOGGER.info(f"Preflight k=1.0 equivalence check passed (max diff = {max_diff:.2e}).")


def main():
    """Launch HYDRA score search."""
    args = parse_args()
    if args.preflight:
        run_preflight_equivalence_check(args.model)

    overrides = {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "device": args.device,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "project": args.project,
        "hydra_stage": "score",
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
            f"Starting ROBUST score search with attacks: {overrides['attack_name']} "
            f"(ratios={overrides['attack_ratio']})"
        )
        trainer = HydraAdversarialDetectionTrainer(overrides=overrides, attack_weights=args.attack_weights or "")
    else:
        from ultralytics.models.yolo.detect.hydra_train import HydraDetectionTrainer

        LOGGER.info("Starting CLEAN score search (HydraDetectionTrainer)")
        trainer = HydraDetectionTrainer(overrides=overrides)

    trainer.train()
    report = trainer.finalize_score_search()

    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("HYDRA SCORE SEARCH COMPLETE")
    LOGGER.info(f"Target keep ratio: {args.keep_ratio:.4f}")
    LOGGER.info(
        f"Final mask sparsity: {report.sparsity:.4f} "
        f"({report.kept_mask_elements}/{report.total_mask_elements} connections retained)"
    )
    best_path = getattr(trainer, "best", None) or (trainer.wdir / "best.pt")
    last_path = getattr(trainer, "last", None) or (trainer.wdir / "last.pt")
    LOGGER.info(f"Best score checkpoint: {best_path}")
    LOGGER.info(f"Last score checkpoint: {last_path}")
    LOGGER.info("=" * 60 + "\n")


if __name__ == "__main__":
    main()

