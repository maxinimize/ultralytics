import argparse
from ultralytics.models.yolo.detect.train_test import DetectionTrainerAdvRatio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8x.pt')
    parser.add_argument('--data', default='coco128.yaml')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch', type=int, default=24)
    parser.add_argument('--attack_weights', default=None, help='Path to validation attack model weights')
    parser.add_argument('--attack_num', type=int, default=None, help='Number of attack types')
    parser.add_argument('--attack_name', nargs='+', default=['pgd'], help='Attack name mapping to pre-generated images (e.g. pgd, jsma, cw, fgsm)')
    parser.add_argument('--ratio', nargs='+', default=['0.5'], help='Adversarial image ratio (fraction of adversarial samples, float 0.0 to 1.0)')
    parser.add_argument('--device', default='0')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--project', default='runs/train_test', help='Project directory')
    parser.add_argument('--name', default='exp', help='Experiment name')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='Resume training from last checkpoint')
    args = parser.parse_args()

    overrides = dict(
        model=args.model,
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        imgsz=args.imgsz,
        workers=args.workers,
        attack_name=args.attack_name,
        attack_num=args.attack_num,
        ratio=args.ratio,
        attack_weights=args.attack_weights,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )

    trainer = DetectionTrainerAdvRatio(overrides=overrides)
    trainer.train()


if __name__ == '__main__':
    main()
