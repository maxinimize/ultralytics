import argparse
from ultralytics.models.yolo.detect.train_adv import DetectionTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8x.pt')
    parser.add_argument('--data', default='coco128.yaml')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch', type=int, default=24)
    parser.add_argument('--attack_weights', default=None, help='Path to attack model weights')
    parser.add_argument('--attack_name', default='cw', help='Attack name passed to the shared factory, e.g. fgsm, pgd, mim, cw, bim, deepfool, jsma, uap, autoattack')
    parser.add_argument('--device', default='0,1')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--project', default='runs/train_adv', help='Project directory')
    parser.add_argument('--name', default='exp', help='Experiment name')
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
        project=args.project,
        name=args.name,
    )
    trainer = DetectionTrainer(overrides=overrides, attack_weights=args.attack_weights)
    trainer.train()


if __name__ == '__main__':
    main()
