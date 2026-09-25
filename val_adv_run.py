import argparse
from ultralytics.models.yolo.detect.train_adv_test import DetectionTrainer
from ultralytics.attacks.attack_utils import setup_attack_model
from ultralytics.attacks.attack_bridge import build_attacker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Model weights for the detector being evaluated")
    parser.add_argument("--data", required=True, help="Dataset yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--attack_weights", default="", help="Optional weights for separate attack model. Empty or 'current' uses evaluated model as white-box")
    parser.add_argument("--attack_name", default="cw", help="Attack name used by the shared factory, e.g. cw, bim, deepfool, jsma, uap, autoattack, pgd, mim")
    parser.add_argument("--worstk_k", type=int, default=None, help="Number of spatial candidate transforms for worstk attack (default: 10)")
    parser.add_argument('--project', default='runs/val_adv', help='Project directory')
    parser.add_argument('--name', default='exp', help='Experiment name')
    parser.add_argument('--classes', type=int, nargs='+', default=None, help='Filter dataset by class indices, e.g. --classes 1 2 3 5 7')
    parser.add_argument("--feature_distillation", action="store_true", default=False, help="Enable Feature Distillation defense during validation")
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
        classes=args.classes,
    )

    trainer = DetectionTrainer(overrides=overrides, attack_weights=args.attack_weights)
    trainer._setup_train()

    # Fallback to build attacker only if setup_train did not create one
    if getattr(trainer, "attacker", None) is None:
        attack_weights = args.attack_weights
        use_current_model = (attack_weights is None) or (str(attack_weights).strip().lower() in {"", "none", "current"})
        if use_current_model:
            attack_model = trainer.model
        elif attack_weights:
            attack_model = setup_attack_model(
                attack_weights,
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
            attacker_kwargs = {}
            if args.worstk_k is not None and args.attack_name.lower().strip() == "worstk":
                attacker_kwargs["k"] = args.worstk_k
            trainer.attacker = build_attacker(args.attack_name, model=attack_model, img_size=args.imgsz, **attacker_kwargs)
    elif args.worstk_k is not None and hasattr(trainer.attacker, "k"):
        trainer.attacker.k = args.worstk_k

    validator = trainer.get_validator()
    validator.model = trainer.model
    validator.attacker = trainer.attacker
    validator.attack_name = args.attack_name
    validator.feature_distillation = args.feature_distillation

    stats = validator(model=validator.model)
    print("Validation results:", stats)


if __name__ == "__main__":
    main()
