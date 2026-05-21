import argparse
from ultralytics.models.yolo.detect.train_adv import DetectionTrainer
from ultralytics.attacks.attack_utils import setup_attack_model
from ultralytics.attacks.attack_bridge import build_attacker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Model weights for the detector being evaluated")
    parser.add_argument("--data", required=True, help="Dataset yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--attack_weights", default="", help="Optional weights used to build a separate attack model")
    parser.add_argument("--attack_name", default="cw", help="Attack name used by the shared factory, e.g. cw, bim, deepfool, jsma, uap, autoattack, pgd, mim")
    parser.add_argument("--use_whitebox", action="store_true", help="Use the evaluated model itself as the attack model")
    parser.add_argument("--attack_override", action="store_true", help="Force rebuilding the attacker in this validation script even if trainer._setup_train() already created one")
    parser.add_argument('--project', default='runs/val_adv', help='Project directory')
    parser.add_argument('--name', default='exp', help='Experiment name')
    args = parser.parse_args()

    overrides = dict(
        model=args.weights,
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        attack_name=args.attack_name,
        project=args.project,
        name=args.name,
    )

    trainer = DetectionTrainer(overrides=overrides, attack_weights=args.attack_weights)
    trainer._setup_train()

    # Keep train/val on the same factory path. Rebuild only when explicitly requested,
    # when using white-box mode, or when setup_train did not create an attacker.
    need_rebuild = args.use_whitebox or args.attack_override or getattr(trainer, "attacker", None) is None

    if need_rebuild:
        if args.use_whitebox:
            attack_model = trainer.model
        elif args.attack_weights:
            attack_model = setup_attack_model(
                args.attack_weights,
                device=trainer.device,
                nc=trainer.data["nc"],
                training=False,
                imgsz=args.imgsz,
            )
        else:
            attack_model = None

        if attack_model is not None:
            attack_model.eval()
            for p in attack_model.parameters():
                p.requires_grad = True
            trainer.attacker = build_attacker(args.attack_name, model=attack_model, img_size=args.imgsz)

    validator = trainer.get_validator()
    validator.model = trainer.model
    validator.attacker = trainer.attacker
    validator.attack_name = args.attack_name

    stats = validator(model=validator.model)
    print("Validation results:", stats)


if __name__ == "__main__":
    main()
