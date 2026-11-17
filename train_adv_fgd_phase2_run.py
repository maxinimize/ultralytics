"""Phase 2 entry point copied from train_adv_test_run.py; optimize YOLO behind frozen FGD."""

import argparse

from ultralytics.models.yolo.detect.train_adv_adaptive import _load_saved_args
from ultralytics.models.yolo.detect.train_adv_fgd_phase2 import FGDAdversarialTrainer


def parse_bool(value):
    """Accept explicit true/false toggles without silently treating typos as false."""
    if isinstance(value, bool):
        return value
    if value.lower() not in {"true", "false", "1", "0"}:
        raise argparse.ArgumentTypeError("Expected true or false")
    return value.lower() in ("true", "1")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for Phase 2 entry point."""
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--model", help="Base model checkpoint or yaml")
    parser.add_argument("--data", default=None, help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch", type=int, help="Batch size")
    parser.add_argument("--attack_weights", default=None, help="Path to attack model weights")
    parser.add_argument("--attack_name", nargs="+", default=None, help="Attack name or list of attack names")
    parser.add_argument("--attack_ratio", nargs="+", default=None, help="Attack ratio or list of attack ratios")
    parser.add_argument("--device", help="Device string")
    parser.add_argument("--imgsz", type=int, help="Image size")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--project", help="Project directory")
    parser.add_argument("--name", help="Experiment name")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="Resume training from last checkpoint")
    parser.add_argument(
        "--classes", type=int, nargs="+", default=None, help="Filter dataset by class indices, e.g. --classes 1 2 3 5 7"
    )
    parser.add_argument("--cache", nargs="?", const=True, help="Cache dataset images in RAM (True/ram, disk, False)")
    parser.add_argument("--use_fgd", type=parse_bool)
    parser.add_argument("--fgd_weights", help="Phase 1 FGD checkpoint; required when enabled")
    parser.add_argument("--attack_through_fgd", type=parse_bool)
    parser.add_argument("--attack_epsilon", type=float)
    parser.add_argument("--attack_steps", type=int)
    parser.add_argument("--attack_step_size", type=float)
    parser.add_argument("--amp", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--fraction", type=float, help="Dataset fraction for a short smoke run")
    return parser


def parse_args(args=None):
    """Parse CLI arguments."""
    return build_parser().parse_args(args)


def main():
    """Parse the copied AT entry point and select frozen-FGD training or the original trainer."""
    args = parse_args()

    cache_val = getattr(args, "cache", None)
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
        if hasattr(args, "classes") and args.classes is not None:
            overrides["classes"] = args.classes
        if hasattr(args, "cache"):
            overrides["cache"] = cache_val
        if hasattr(args, "workers") and args.workers is not None:
            overrides["workers"] = args.workers
        if hasattr(args, "data") and args.data is not None:
            overrides["data"] = args.data
        if hasattr(args, "attack_weights") and args.attack_weights is not None:
            overrides["attack_weights"] = args.attack_weights
        if hasattr(args, "attack_name") and args.attack_name is not None:
            overrides["attack_name"] = args.attack_name
        if hasattr(args, "attack_ratio") and args.attack_ratio is not None:
            overrides["attack_ratio"] = args.attack_ratio
        if hasattr(args, "fraction"):
            overrides["fraction"] = args.fraction
        if hasattr(args, "amp"):
            overrides["amp"] = args.amp
    else:
        overrides = dict(
            model=getattr(args, "model", "yolo12l.pt"),
            epochs=getattr(args, "epochs", 5),
            batch=getattr(args, "batch", 24),
            device=getattr(args, "device", "0"),
            imgsz=getattr(args, "imgsz", 640),
            project=getattr(args, "project", "runs/train_adv_fgd_phase2"),
            name=getattr(args, "name", "exp"),
            resume=False,
            classes=getattr(args, "classes", None),
            cache=cache_val if cache_val is not None else True,
            fraction=getattr(args, "fraction", 1.0),
        )
        if hasattr(args, "workers") and args.workers is not None:
            overrides["workers"] = args.workers
        if hasattr(args, "data") and args.data is not None:
            overrides["data"] = args.data
        else:
            overrides["data"] = "coco128.yaml"
        if hasattr(args, "attack_weights") and args.attack_weights is not None:
            overrides["attack_weights"] = args.attack_weights
        if hasattr(args, "attack_name") and args.attack_name is not None:
            overrides["attack_name"] = args.attack_name
        if hasattr(args, "attack_ratio") and args.attack_ratio is not None:
            overrides["attack_ratio"] = args.attack_ratio
        if hasattr(args, "amp"):
            overrides["amp"] = args.amp

    if hasattr(args, "use_fgd"):
        use_fgd = args.use_fgd
        overrides["use_fgd"] = use_fgd
    elif is_resume:
        saved_args = _load_saved_args(args.resume)
        use_fgd = saved_args.get("use_fgd", True)
        if isinstance(use_fgd, str):
            use_fgd = use_fgd.lower() in ("true", "1", "yes")
        overrides["use_fgd"] = use_fgd
    else:
        use_fgd = True
        overrides["use_fgd"] = True

    for key in FGDAdversarialTrainer.defaults:
        if hasattr(args, key):
            overrides[key] = getattr(args, key)

    trainer = FGDAdversarialTrainer(overrides=overrides, attack_weights=getattr(args, "attack_weights", None))
    trainer.train()


if __name__ == "__main__":
    main()
