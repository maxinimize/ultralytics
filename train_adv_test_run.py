import argparse
from ultralytics.models.yolo.detect.train_adv_test import DetectionTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8x.pt')
    parser.add_argument('--data', default='coco128.yaml')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch', type=int, default=24)
    parser.add_argument('--attack_weights', default=None, help='Path to attack model weights')
    parser.add_argument('--attack_num', type=int, default=1, help='Number of attack types')
    parser.add_argument('--attack_name', nargs='+', default=['pgd'], help='Attack name or list of attack names')
    parser.add_argument('--attack_ratio', nargs='+', default=['0.5'], help='Attack ratio or list of attack ratios')
    parser.add_argument('--use_pregenerated_adv', action='store_true', help='Load pre-generated adversarial images from disk')
    parser.add_argument('--device', default='0,1')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--project', default='runs/train_adv', help='Project directory')
    parser.add_argument('--name', default='exp', help='Experiment name')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='Resume training from last checkpoint')
    parser.add_argument('--no_train_aug', action='store_true', help='Disable YOLO training augmentations for adversarial training.')
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
        attack_ratio=args.attack_ratio,
        use_pregenerated_adv=args.use_pregenerated_adv,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )

    if args.no_train_aug:
        overrides.update({
        # image mixing augmentations
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,

        # geometric augmentations
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,

        # flipping / channel changes
        "flipud": 0.0,
        "fliplr": 0.0,
        "bgr": 0.0,

        # color augmentations
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,

        # training-time image size variation
        "multi_scale": 0.0,

        # classification-specific, but safe to disable explicitly
        "auto_augment": None,
        "erasing": 0.0,
    })

    trainer = DetectionTrainer(overrides=overrides, attack_weights=args.attack_weights)
    trainer.train()


if __name__ == '__main__':
    main()
