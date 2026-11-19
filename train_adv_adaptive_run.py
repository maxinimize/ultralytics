import argparse

from ultralytics.models.yolo.detect.train_adv_adaptive import DetectionTrainer


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for Dynamic Attack Ratio Adversarial Training."""
    parser = argparse.ArgumentParser(
        description="Dynamic Attack Ratio Adversarial Training",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument("--model", help="Base model checkpoint or yaml (default: yolov8x.pt)")
    parser.add_argument("--data", help="Dataset YAML path (default: coco128.yaml)")
    parser.add_argument("--epochs", type=int, help="Number of epochs (default: 5)")
    parser.add_argument("--batch", type=int, help="Batch size (default: 24)")
    parser.add_argument("--attack_weights", help="Path to attack model weights")
    parser.add_argument("--attack_name", nargs="+", help="Attack names (default: pgd bim mim worstk)")
    parser.add_argument("--attack_ratio", nargs="+", help="Initial attack ratios (default: 0.10 0.10 0.40 0.15)")
    parser.add_argument("--p_clean", type=float, help="Clean training ratio (default: 0.25)")
    parser.add_argument("--tau", type=float, help="Softmax temperature tau for dynamic updates (default: 5.0)")
    parser.add_argument("--update_interval", type=int, help="Epoch interval for ratio update (default: 5)")
    parser.add_argument("--eval_weights", nargs="+", help="Fixed evaluation weights (default: 0.25 0.25 0.25 0.25)")
    parser.add_argument("--device", help="Device string (e.g. 0 or 0,1, default: 0,1)")
    parser.add_argument("--imgsz", type=int, help="Image size (default: 640)")
    parser.add_argument("--conf", type=float, help="Validation confidence threshold (default: 0.001)")
    parser.add_argument("--iou", type=float, help="Validation NMS IoU threshold (default: 0.60)")
    parser.add_argument("--max_det", type=int, help="Validation maximum detections per image (default: 300)")
    parser.add_argument("--workers", type=int, help="DataLoader workers")
    parser.add_argument("--project", help="Project directory (default: runs/train_adv)")
    parser.add_argument("--name", help="Experiment name (default: exp)")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="Resume training from last checkpoint")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility (default: None, inherits default)")
    parser.add_argument("--classes", type=int, nargs="+", help="Filter dataset by class indices")
    parser.add_argument("--cache", nargs="?", const=True, help="Cache dataset images in RAM (True/ram, disk, False)")
    return parser


def parse_args(args=None):
    """Parse CLI arguments."""
    return build_parser().parse_args(args)


def main():
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
        if hasattr(args, "conf"):
            overrides["conf"] = args.conf
        if hasattr(args, "iou"):
            overrides["iou"] = args.iou
        if hasattr(args, "max_det"):
            overrides["max_det"] = args.max_det
    else:
        overrides = dict(
            model=getattr(args, "model", "yolov8x.pt"),
            data=getattr(args, "data", "coco128.yaml"),
            epochs=getattr(args, "epochs", 5),
            batch=getattr(args, "batch", 24),
            device=getattr(args, "device", "0,1"),
            imgsz=getattr(args, "imgsz", 640),
            conf=getattr(args, "conf", 0.001),
            iou=getattr(args, "iou", 0.60),
            max_det=getattr(args, "max_det", 300),
            project=getattr(args, "project", "runs/train_adv"),
            name=getattr(args, "name", "exp"),
            resume=False,
            classes=getattr(args, "classes", None),
            cache=cache_val,
            p_clean=getattr(args, "p_clean", 0.25),
            tau=getattr(args, "tau", 5.0),
            update_interval=getattr(args, "update_interval", 5),
            attack_name=getattr(args, "attack_name", ["pgd", "bim", "mim", "worstk"]),
            attack_ratio=getattr(args, "attack_ratio", [0.10, 0.10, 0.40, 0.15]),
            eval_weights=getattr(args, "eval_weights", [0.25, 0.25, 0.25, 0.25]),
        )

    if hasattr(args, "workers") and args.workers is not None:
        overrides["workers"] = args.workers
    if hasattr(args, "seed") and args.seed is not None:
        overrides["seed"] = int(args.seed)
    if hasattr(args, "attack_weights") and args.attack_weights is not None:
        overrides["attack_weights"] = args.attack_weights

    trainer = DetectionTrainer(overrides=overrides, attack_weights=getattr(args, "attack_weights", None))
    trainer.train()


if __name__ == "__main__":
    main()
