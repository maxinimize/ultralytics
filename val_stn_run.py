# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""CLI runner for standalone STN defense validation across baseline and defended configurations."""

from __future__ import annotations

import argparse
from copy import copy
from pathlib import Path

import torch

from ultralytics.attacks.spatial_worst_of_k_stn import SpatialWorstOfKWithMetadata
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.defenses.spatial_transformer import SpatialTransformer, load_stn_checkpoint
from ultralytics.models.yolo.detect.val_stn import STNDetectionValidator
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import DEFAULT_CFG, LOGGER, IterableSimpleNamespace
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.torch_utils import select_device


def parse_args(args=None):
    """Parse STN validation CLI options."""
    parser = argparse.ArgumentParser(description="Evaluate STN Defense under Worst-of-k Attacks")
    parser.add_argument("--weights", default="yolo12l.pt", help="YOLO detector weights")
    parser.add_argument(
        "--stn_weights",
        default=None,
        help="Trained STN checkpoint (weights/best.pt). If None, uses identity STN for stn modes",
    )
    parser.add_argument("--data", required=True, help="Dataset YAML file")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Dataset split to evaluate")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["clean_baseline", "worstk_baseline", "clean_stn", "worstk_stn"],
        help="Modes to evaluate: clean_baseline, worstk_baseline, clean_stn, worstk_stn",
    )
    parser.add_argument("--worstk_k", type=int, default=10, help="Worst-of-k candidate count (default: 10)")
    parser.add_argument("--degrees", type=float, default=30.0, help="Worst-of-k max rotation degrees (default: 30.0)")
    parser.add_argument(
        "--translate", type=float, default=0.10, help="Worst-of-k max translation ratio (default: 0.10)"
    )
    parser.add_argument(
        "--attack_weights",
        default=None,
        help="Optional separate attack model weights; default uses evaluated detector",
    )
    parser.add_argument("--batch", type=int, default=8, help="Validation batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", default="0", help="CUDA device (e.g. 0) or 'cpu'")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--classes", nargs="+", type=int, default=None, help="Filter dataset by class indices")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproducibility seed for attack candidate generation (default: inherited from cfg/default.yaml)",
    )
    parser.add_argument("--project", default="runs/val_stn", help="Output project directory")
    parser.add_argument("--name", default="exp", help="Experiment name")
    return parser.parse_args(args)


def resolve_models(args, device: torch.device):
    """Load guidance detector, STN defense, and attack model according to CLI args."""
    # 1. Load guidance detector
    guidance_model, _ = load_checkpoint(args.weights, device=device)
    guidance_model.eval().requires_grad_(False)
    if hasattr(guidance_model, "args") and isinstance(guidance_model.args, dict):
        guidance_model.args = IterableSimpleNamespace(**guidance_model.args)

    # 2. Load or instantiate STN defense
    stn_weights_str = str(args.stn_weights).strip() if args.stn_weights is not None else ""
    if stn_weights_str and stn_weights_str.lower() not in {"", "none"}:
        stn, _ = load_stn_checkpoint(args.stn_weights, device=device)
    else:
        LOGGER.info("No stn_weights provided; using identity-initialized STN module.")
        stn = SpatialTransformer(max_degrees=args.degrees * 1.5, max_translate=args.translate * 1.5).to(device)
    stn.eval().requires_grad_(False)

    # 3. Load or reuse attack model
    attack_weights_str = str(args.attack_weights).strip() if args.attack_weights is not None else ""
    if attack_weights_str and attack_weights_str.lower() not in {"", "none", "current", "baseline"}:
        attack_model, _ = load_checkpoint(args.attack_weights, device=device)
        attack_model.eval().requires_grad_(False)
        if hasattr(attack_model, "args") and isinstance(attack_model.args, dict):
            attack_model.args = IterableSimpleNamespace(**attack_model.args)
    else:
        attack_model = guidance_model

    return guidance_model, stn, attack_model


def main():
    """Run STN validation comparison across specified modes."""
    args = parse_args()
    device = select_device(args.device)

    guidance_model, stn, attack_model = resolve_models(args, device)

    # 4. Instantiate Worst-of-k attacker
    effective_seed = int(DEFAULT_CFG.seed) if args.seed is None else int(args.seed)
    attacker = SpatialWorstOfKWithMetadata(
        model=attack_model,
        k=args.worstk_k,
        degrees=args.degrees,
        translate=args.translate,
        img_size=args.imgsz,
        seed=effective_seed,
    )

    # 5. Build validation dataloader
    data_dict = check_det_dataset(args.data)
    stride = max(int(guidance_model.stride.max()), 32)
    imgsz = check_imgsz(args.imgsz, stride=stride)
    dataset_cfg = copy(DEFAULT_CFG)
    dataset_cfg.task = "detect"
    dataset_cfg.imgsz = imgsz
    dataset_cfg.classes = args.classes

    dataset = build_yolo_dataset(
        cfg=dataset_cfg,
        img_path=data_dict.get(args.split),
        batch=args.batch,
        data=data_dict,
        mode="val",
        rect=False,
        stride=stride,
    )
    dataloader = build_dataloader(dataset, batch=args.batch, workers=args.workers, shuffle=False)

    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    validator = STNDetectionValidator(
        dataloader=dataloader,
        save_dir=save_dir,
        args=dict(task="detect", conf=0.001, iou=0.6, classes=args.classes, data=args.data),
    )
    validator.data = data_dict
    validator.device = device
    validator.stn = stn
    validator.guidance_model = guidance_model
    validator.attacker = attacker

    LOGGER.info(f"Starting STN validation across modes: {args.modes}")
    results = validator(modes=args.modes, seed=effective_seed)

    LOGGER.info("=" * 60)
    LOGGER.info("STN Defense Validation Summary:")
    for mode in args.modes:
        m50 = results.get(f"{mode}/mAP50", 0.0)
        m95 = results.get(f"{mode}/mAP50-95", 0.0)
        extra = ""
        if f"{mode}/geom_error" in results:
            extra += f", Geom Error: {results[f'{mode}/geom_error']:.4f}"
        if f"{mode}/masked_recon" in results:
            extra += f", Masked Recon: {results[f'{mode}/masked_recon']:.4f}"
        LOGGER.info(f"  {mode:<18}: mAP50={m50:.4f}, mAP50-95={m95:.4f}{extra}")
    LOGGER.info("=" * 60)
    return results


if __name__ == "__main__":
    main()
