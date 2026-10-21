import argparse
from ultralytics.models.yolo.detect.train_adv_adaptive import DetectionTrainer


def main():
    parser = argparse.ArgumentParser(description="Dynamic Attack Ratio Adversarial Training")
    parser.add_argument('--model', default='yolov8x.pt', help='Base model checkpoint or yaml')
    parser.add_argument('--data', default=None, help='Dataset YAML path')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=24, help='Batch size')
    parser.add_argument('--attack_weights', default=None, help='Path to attack model weights')
    parser.add_argument('--attack_name', nargs='+', default=None, help='Attack names (default: pgd bim mim worstk)')
    parser.add_argument('--attack_ratio', nargs='+', default=None, help='Initial attack ratios (default: 0.10 0.10 0.40 0.15)')
    parser.add_argument('--p_clean', type=float, default=0.25, help='Clean training ratio (default: 0.25)')
    parser.add_argument('--tau', type=float, default=5.0, help='Softmax temperature tau for dynamic updates (default: 5.0)')
    parser.add_argument('--update_interval', type=int, default=5, help='Epoch interval for ratio update (default: 5)')
    parser.add_argument('--eval_weights', nargs='+', default=None, help='Fixed evaluation weights (default: 0.25 0.15 0.15 0.20)')
    parser.add_argument('--device', default='0,1', help='Device string (e.g. 0 or 0,1)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--workers', type=int, default=None, help='DataLoader workers')
    parser.add_argument('--project', default='runs/train_adv', help='Project directory')
    parser.add_argument('--name', default='exp', help='Experiment name')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='Resume training from last checkpoint')
    parser.add_argument('--classes', type=int, nargs='+', default=None, help='Filter dataset by class indices')
    parser.add_argument('--cache', nargs='?', const=True, default=True, help='Cache dataset images in RAM (True/ram, disk, False)')
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
        p_clean=args.p_clean,
        tau=args.tau,
        update_interval=args.update_interval,
    )
    if args.data is not None:
        overrides["data"] = args.data
    elif not args.resume:
        overrides["data"] = "coco128.yaml"
    if args.attack_weights is not None:
        overrides["attack_weights"] = args.attack_weights

    overrides["attack_name"] = args.attack_name if args.attack_name is not None else ["pgd", "bim", "mim", "worstk"]
    overrides["attack_ratio"] = args.attack_ratio if args.attack_ratio is not None else [0.10, 0.10, 0.40, 0.15]
    overrides["eval_weights"] = args.eval_weights if args.eval_weights is not None else [0.25, 0.15, 0.15, 0.20]

    trainer = DetectionTrainer(overrides=overrides, attack_weights=args.attack_weights)
    trainer.train()


if __name__ == '__main__':
    main()
