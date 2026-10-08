"""Phase 2 entry point copied from train_adv_test_run.py; optimize YOLO behind frozen FGD."""

import argparse

from ultralytics.models.yolo.detect.train_adv_fgd_phase2 import FGDAdversarialTrainer


def parse_bool(value):
    """Accept explicit true/false toggles without silently treating typos as false."""
    if value.lower() not in {"true", "false"}:
        raise argparse.ArgumentTypeError("Expected true or false")
    return value.lower() == "true"


def main():
    """Parse the copied AT entry point and select frozen-FGD training or the original trainer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo12l.pt")
    parser.add_argument("--data", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--attack_weights", default=None, help="Path to attack model weights")
    parser.add_argument("--attack_name", nargs="+", default=None, help="Attack name or list of attack names")
    parser.add_argument("--attack_ratio", nargs="+", default=None, help="Attack ratio or list of attack ratios")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--project", default="runs/train_adv_fgd_phase2", help="Project directory")
    parser.add_argument("--name", default="exp", help="Experiment name")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="Resume training from last checkpoint")
    parser.add_argument(
        "--classes", type=int, nargs="+", default=None, help="Filter dataset by class indices, e.g. --classes 1 2 3 5 7"
    )
    parser.add_argument(
        "--cache", nargs="?", const=True, default=True, help="Cache dataset images in RAM (True/ram, disk, False)"
    )
    parser.add_argument("--use_fgd", type=parse_bool, default=True)
    parser.add_argument(
        "--fgd_weights", default=argparse.SUPPRESS, help="Phase 1 FGD checkpoint; required when enabled"
    )
    parser.add_argument("--attack_through_fgd", type=parse_bool, default=argparse.SUPPRESS)
    parser.add_argument("--attack_epsilon", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--attack_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--attack_step_size", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--amp", choices=["bf16", "fp16", "fp32"], default=argparse.SUPPRESS)
    parser.add_argument("--fraction", type=float, default=1.0, help="Dataset fraction for a short smoke run")
    args = parser.parse_args()

    cache_val = args.cache
    if isinstance(cache_val, str):
        if cache_val.lower() in ("true", "1", "ram"):
            cache_val = "ram"
        elif cache_val.lower() in ("false", "0", "none"):
            cache_val = False

    overrides = dict(
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        imgsz=args.imgsz,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        classes=args.classes,
        cache=cache_val,
    )
    if args.data is not None:
        overrides["data"] = args.data
    elif not args.resume:
        overrides["data"] = "coco128.yaml"
    if args.attack_weights is not None:
        overrides["attack_weights"] = args.attack_weights
    if args.attack_name is not None:
        overrides["attack_name"] = args.attack_name
    if args.attack_ratio is not None:
        overrides["attack_ratio"] = args.attack_ratio

    overrides["fraction"] = args.fraction
    if args.use_fgd:
        for key in FGDAdversarialTrainer.defaults:
            if hasattr(args, key):
                overrides[key] = getattr(args, key)
        trainer_class = FGDAdversarialTrainer
    else:
        from ultralytics.models.yolo.detect.train_adv_test import DetectionTrainer

        trainer_class = DetectionTrainer
    if hasattr(args, "amp"):
        overrides["amp"] = args.amp

    trainer = trainer_class(overrides=overrides, attack_weights=args.attack_weights)
    trainer.train()


if __name__ == "__main__":
    main()
