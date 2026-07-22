import argparse
from ultralytics.models.yolo.detect.train_adv_dynamic import DetectionTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8x.pt')
    parser.add_argument('--data', default='coco128.yaml')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch', type=int, default=24)
    parser.add_argument('--attack_name', default='cw', help='Attack name passed to the shared factory, e.g. fgsm, pgd, mim, cw, bim, deepfool, jsma, uap, autoattack')
    parser.add_argument('--use_pregenerated_adv', action='store_true', help='Load pre-generated adversarial images from disk')
    parser.add_argument('--device', default='0,1')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--project', default='runs/train_adv', help='Project directory')
    parser.add_argument('--name', default='exp', help='Experiment name')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='Resume training from last checkpoint')
    
    # Dynamic pool arguments
    parser.add_argument('--attack_mix_ratio', type=float, default=0.5, help='Fraction of attacked images in each batch')
    parser.add_argument('--num_aug', type=int, default=5, help='Number of clean/attacked augmented images generated per original image')
    parser.add_argument('--pool_update_period', type=int, default=5, help='Epoch update frequency for regenerating dynamic pools')
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
        use_pregenerated_adv=args.use_pregenerated_adv,
        project=args.project,
        name=args.name,
        resume=args.resume,
        attack_mix_ratio=args.attack_mix_ratio,
        num_aug=args.num_aug,
        pool_update_period=args.pool_update_period,
    )

    trainer = DetectionTrainer(overrides=overrides)
    trainer.train()


if __name__ == '__main__':
    main()
