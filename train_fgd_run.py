"""Train a residual FGD denoiser with online PGD/BIM/MIM on augmented YOLO batches."""

import argparse

from ultralytics.models.yolo.detect.train_fgd import FGDTrainer


def main():
    """Parse the independent research CLI without extending global YOLO arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo12l.pt", help="Frozen guidance checkpoint")
    parser.add_argument("--data", default="coco128.yaml")
    parser.add_argument(
        "--attack_weights", default=None, help="Independent attack checkpoint; default: guidance weights"
    )
    parser.add_argument("--attack_name", nargs="+", choices=["pgd", "bim", "mim"], default=["pgd", "bim", "mim"])
    parser.add_argument(
        "--attack_ratio", nargs="+", type=float, default=None, help="Default: equal disjoint partitions"
    )
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS, help="Total epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--classes", nargs="+", type=int, default=None, help="Keep only these dataset class IDs")
    parser.add_argument("--cache", choices=["ram", "disk"], default=None, help="Optional dataset image cache")
    parser.add_argument("--device", default="0", help="Single CUDA device or cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multi_scale", type=float, default=0.0)
    parser.add_argument("--project", default="runs/fgd")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--save_period", type=int, default=-1)
    parser.add_argument("--resume", default=False, help="FGD weights/last.pt; restores saved experiment settings")
    FGDTrainer(overrides=vars(parser.parse_args())).train()


if __name__ == "__main__":
    main()
