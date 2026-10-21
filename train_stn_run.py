# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""CLI script for independent Spatial Transformer Network (STN) training."""

from __future__ import annotations

import argparse
from ultralytics.models.yolo.detect.train_stn import STNTrainer


def main():
    """Parse STN training arguments and launch trainer."""
    parser = argparse.ArgumentParser(description="Spatial Transformer Network (STN) Defense Training")
    parser.add_argument("--model", default="yolo12l.pt", help="Frozen guidance detector weights (default: yolo12l.pt)")
    parser.add_argument("--data", default="coco128.yaml", help="Dataset configuration YAML")
    parser.add_argument(
        "--attack_weights",
        default=None,
        help="Optional separate weights for Worst-of-k attack; default uses guidance weights",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Total epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--classes", nargs="+", type=int, default=None, help="Filter dataset by class indices")
    parser.add_argument("--cache", choices=["ram", "disk", "True", "False"], default=None, help="Dataset image cache")
    parser.add_argument("--device", default="0", help="CUDA device index (e.g. 0) or cpu")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--lr0", type=float, default=1e-3, help="Initial Adam learning rate (default: 1e-3)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (default: 1e-5)")
    parser.add_argument("--degrees", type=float, default=30.0, help="Worst-of-k max rotation degrees (default: 30.0)")
    parser.add_argument("--translate", type=float, default=0.10, help="Worst-of-k max translation ratio (default: 0.10)")
    parser.add_argument("--worstk_k", type=int, default=10, help="Number of spatial candidate transforms (default: 10)")
    parser.add_argument("--loss_geom_weight", type=float, default=1.0, help="Weight for geometry loss (default: 1.0)")
    parser.add_argument("--loss_recon_weight", type=float, default=1.0, help="Weight for masked recon loss (default: 1.0)")
    parser.add_argument("--loss_det_weight", type=float, default=0.0, help="Weight for frozen-YOLO det loss (default: 0.0)")
    parser.add_argument("--val_interval", type=int, default=1, help="Validation interval in epochs (default: 1)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility (default: 0)")
    parser.add_argument("--project", default="runs/train_stn", help="Results directory (default: runs/train_stn)")
    parser.add_argument("--name", default="exp", help="Experiment name (default: exp)")
    parser.add_argument("--save_period", type=int, default=-1, help="Periodic checkpoint save interval")
    parser.add_argument("--resume", default=False, help="Resume from checkpoint path (weights/last.pt)")

    args = parser.parse_args()

    cache_val = args.cache
    if isinstance(cache_val, str):
        if cache_val.lower() in ("true", "1", "ram"):
            cache_val = "ram"
        elif cache_val.lower() in ("false", "0", "none"):
            cache_val = False

    overrides = vars(args)
    overrides["cache"] = cache_val

    STNTrainer(overrides=overrides).train()


if __name__ == "__main__":
    main()
