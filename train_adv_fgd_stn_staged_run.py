# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""CLI runner for Staged Adaptive Adversarial Training with Frozen FGD and STN for YOLO."""

from __future__ import annotations

import argparse
import os
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from ultralytics.defenses.fgd_stn_pipeline import canonicalize_mode
from ultralytics.models.yolo.detect.train_adv_fgd_stn_staged import (
    DetectionTrainerAdvFGDSTNStaged,
)


def parse_bool(value: Any) -> bool:
    """Accept explicit true/false/1/0 toggles without silently treating typos as false."""
    if isinstance(value, bool):
        return value
    val = str(value).strip().lower()
    if val in {"true", "1", "yes", "y"}:
        return True
    if val in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for Staged Adaptive Adversarial Training."""
    parser = argparse.ArgumentParser(
        description="Staged Adaptive Adversarial Training with Frozen FGD and STN for YOLO12l.",
        argument_default=argparse.SUPPRESS,
    )

    # Staged Pipeline Controls
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        help="Training stage: 1 (Clean Fine-tuning) or 2 (Adaptive Adversarial Training).",
    )
    parser.add_argument(
        "--stage1_weights",
        help="Path to Stage 1 best.pt checkpoint for initializing Stage 2.",
    )
    parser.add_argument(
        "--train_head_only",
        type=parse_bool,
        help="Whether to train only the YOLO Detection Head while freezing Backbone and Neck (default: true).",
    )
    parser.add_argument(
        "--freeze_bn",
        type=parse_bool,
        help="Whether to keep BatchNorm statistics in eval mode for all frozen layers (default: true).",
    )
    parser.add_argument(
        "--freeze_all_bn",
        type=parse_bool,
        help="Freeze all YOLO BatchNorm layers in eval mode and freeze affine parameters (default: true for stage 1, false for stage 2).",
    )
    parser.add_argument(
        "--eval_only",
        nargs="?",
        const=True,
        type=parse_bool,
        help="Evaluate baseline Clean mAP50 and mAP50-95 without training.",
    )
    parser.add_argument(
        "--eval_epoch0",
        nargs="?",
        const=True,
        type=parse_bool,
        help="Evaluate Epoch 0 baseline before starting Stage 1 training.",
    )

    # Preprocessing and Defense Modules
    parser.add_argument(
        "--preprocess_mode",
        "--mode",
        help="Preprocessing defense mode: 'none', 'fgd', 'stn', 'fgd_stn', 'stn_fgd' (default: stn_fgd).",
    )
    parser.add_argument(
        "--fgd_weights",
        help="Path to trained FGD checkpoint (weights/best.pt). Required for modes 'fgd', 'fgd_stn', 'stn_fgd'.",
    )
    parser.add_argument(
        "--stn_weights",
        help="Path to trained STN checkpoint (weights/best.pt). Required for modes 'stn', 'fgd_stn', 'stn_fgd'.",
    )

    # Adaptive Ratio Settings
    parser.add_argument(
        "--adaptive_ratio",
        type=parse_bool,
        help="Enable dynamic attack ratio adjustment every update_interval epochs (default: false for stage 1, true for stage 2).",
    )
    parser.add_argument(
        "--p_clean",
        type=float,
        help="Fixed Clean training ratio (default: 1.0 for stage 1, 0.65 for stage 2).",
    )
    parser.add_argument(
        "--tau",
        type=float,
        help="Softmax temperature tau for dynamic ratio updates (default: 5.0).",
    )
    parser.add_argument(
        "--update_interval",
        type=int,
        help="Epoch interval for recalculating training ratios (default: 5).",
    )
    parser.add_argument(
        "--eval_weights",
        nargs="+",
        help="Validation weights for checkpoint fitness (Stage 1: 1.0; Stage 2 default: 0.20 0.20 0.15 0.15 0.30 for Clean PGD BIM MIM WorstK).",
    )

    # Attack Configuration
    parser.add_argument(
        "--attack_name",
        nargs="+",
        help="Attack names to train against (default: empty for stage 1, pgd bim mim worstk for stage 2).",
    )
    parser.add_argument(
        "--attack_ratio",
        nargs="+",
        help="Initial training ratios for attacks (default: empty for stage 1, 0.12 0.05 0.10 0.08 for stage 2).",
    )
    parser.add_argument(
        "--attack_weights",
        help="Surrogate model weights for attack generation: 'current' (white-box AT) or path to external surrogate.",
    )
    parser.add_argument(
        "--attack_through_preprocess",
        type=parse_bool,
        help="Whether attack generation passes through the frozen defense pipeline (default: true).",
    )
    parser.add_argument(
        "--strict_attacks",
        type=parse_bool,
        help="Raise explicit errors if adversarial generation fails, prohibiting silent fallback (default: true).",
    )

    # Standard Training Settings
    parser.add_argument(
        "--model",
        help="Base YOLO detector checkpoint (default: yolo12l.pt). Overridden by --stage1_weights in Stage 2.",
    )
    parser.add_argument(
        "--data",
        help="Dataset YAML path (default: coco_train_traffic5k.yaml).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Total training epochs (default: 5 for stage 1, 20 for stage 2).",
    )
    parser.add_argument("--batch", type=int, help="Batch size (default: 16).")
    parser.add_argument("--imgsz", type=int, help="Image size in pixels (default: 640).")
    parser.add_argument("--optimizer", help="Optimizer name (default: AdamW).")
    parser.add_argument("--lr0", type=float, help="Initial learning rate (default: 1e-5).")
    parser.add_argument("--warmup_epochs", type=float, help="Warmup epochs (default: 0.0).")
    parser.add_argument("--warmup_bias_lr", type=float, help="Warmup bias learning rate (default: 0.0).")
    parser.add_argument("--device", help="CUDA device index or 'cpu' (default: 0).")
    parser.add_argument("--workers", type=int, help="DataLoader worker processes.")
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility (default: inherited from cfg/default.yaml).",
    )
    parser.add_argument(
        "--project",
        help="Project directory for logging and checkpoints.",
    )
    parser.add_argument("--name", help="Experiment name.")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="Resume interrupted training run (e.g. --resume runs/.../last.pt).",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        help="Filter dataset by specific class indices.",
    )
    parser.add_argument(
        "--cache",
        nargs="?",
        const=True,
        help="Cache dataset images in RAM (True/ram, disk, False).",
    )
    parser.add_argument(
        "--amp",
        choices=["bf16", "fp16", "fp32"],
        help="Precision mode (default: bf16).",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        help="Fraction of dataset to use for short smoke runs.",
    )
    parser.add_argument("--lrf", type=float, help="Final learning rate fraction.")
    parser.add_argument("--mosaic", type=float, help="Mosaic augmentation probability.")
    parser.add_argument("--scale", type=float, help="Scale augmentation gain.")
    parser.add_argument("--translate", type=float, help="Translation augmentation fraction.")
    parser.add_argument("--fliplr", type=float, help="Horizontal flip probability.")
    parser.add_argument("--flipud", type=float, help="Vertical flip probability.")
    parser.add_argument("--hsv_h", type=float, help="HSV hue augmentation fraction.")
    parser.add_argument("--hsv_s", type=float, help="HSV saturation augmentation fraction.")
    parser.add_argument("--hsv_v", type=float, help="HSV value augmentation fraction.")
    parser.add_argument("--mixup", type=float, help="MixUp augmentation probability.")
    parser.add_argument("--cutmix", type=float, help="CutMix augmentation probability.")
    parser.add_argument("--copy_paste", type=float, help="Copy-paste augmentation fraction.")
    parser.add_argument("--close_mosaic", type=int, help="Disable mosaic for final N epochs.")
    parser.add_argument(
        "--save_period",
        type=int,
        help="Save checkpoint every X epochs (default: 1 for stage 1, -1 for stage 2).",
    )
    return parser


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    return build_parser().parse_args(args)


def main():
    """Parse CLI arguments and run Staged Adaptive Adversarial Training."""
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
        if hasattr(args, "stage"):
            overrides["stage"] = args.stage
        if hasattr(args, "stage1_weights"):
            overrides["stage1_weights"] = args.stage1_weights
        if hasattr(args, "train_head_only"):
            overrides["train_head_only"] = args.train_head_only
        if hasattr(args, "freeze_bn"):
            overrides["freeze_bn"] = args.freeze_bn
        if hasattr(args, "freeze_all_bn"):
            overrides["freeze_all_bn"] = args.freeze_all_bn
        if hasattr(args, "eval_only"):
            overrides["eval_only"] = args.eval_only
        if hasattr(args, "eval_epoch0"):
            overrides["eval_epoch0"] = args.eval_epoch0
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
        if hasattr(args, "optimizer"):
            overrides["optimizer"] = args.optimizer
        if hasattr(args, "lr0"):
            overrides["lr0"] = args.lr0
        if hasattr(args, "lrf"):
            overrides["lrf"] = args.lrf
        if hasattr(args, "warmup_epochs"):
            overrides["warmup_epochs"] = args.warmup_epochs
        if hasattr(args, "warmup_bias_lr"):
            overrides["warmup_bias_lr"] = args.warmup_bias_lr
        if hasattr(args, "mosaic"):
            overrides["mosaic"] = args.mosaic
        if hasattr(args, "scale"):
            overrides["scale"] = args.scale
        if hasattr(args, "translate"):
            overrides["translate"] = args.translate
        if hasattr(args, "fliplr"):
            overrides["fliplr"] = args.fliplr
        if hasattr(args, "flipud"):
            overrides["flipud"] = args.flipud
        if hasattr(args, "hsv_h"):
            overrides["hsv_h"] = args.hsv_h
        if hasattr(args, "hsv_s"):
            overrides["hsv_s"] = args.hsv_s
        if hasattr(args, "hsv_v"):
            overrides["hsv_v"] = args.hsv_v
        if hasattr(args, "mixup"):
            overrides["mixup"] = args.mixup
        if hasattr(args, "cutmix"):
            overrides["cutmix"] = args.cutmix
        if hasattr(args, "copy_paste"):
            overrides["copy_paste"] = args.copy_paste
        if hasattr(args, "close_mosaic"):
            overrides["close_mosaic"] = args.close_mosaic
        if hasattr(args, "save_period"):
            overrides["save_period"] = args.save_period
        if hasattr(args, "project"):
            overrides["project"] = args.project
        if hasattr(args, "name"):
            overrides["name"] = args.name
        if hasattr(args, "cache"):
            overrides["cache"] = cache_val
        if hasattr(args, "amp"):
            overrides["amp"] = args.amp
        if hasattr(args, "fraction"):
            overrides["fraction"] = args.fraction
        if hasattr(args, "preprocess_mode"):
            overrides["preprocess_mode"] = canonicalize_mode(args.preprocess_mode)
        if hasattr(args, "fgd_weights"):
            overrides["fgd_weights"] = args.fgd_weights
        if hasattr(args, "stn_weights"):
            overrides["stn_weights"] = args.stn_weights
        if hasattr(args, "adaptive_ratio"):
            overrides["adaptive_ratio"] = args.adaptive_ratio
        if hasattr(args, "attack_through_preprocess"):
            overrides["attack_through_preprocess"] = args.attack_through_preprocess
        if hasattr(args, "p_clean"):
            overrides["p_clean"] = args.p_clean
        if hasattr(args, "tau"):
            overrides["tau"] = args.tau
        if hasattr(args, "update_interval"):
            overrides["update_interval"] = args.update_interval
        if hasattr(args, "eval_weights"):
            overrides["eval_weights"] = args.eval_weights
        if hasattr(args, "attack_name"):
            overrides["attack_name"] = args.attack_name
        if hasattr(args, "attack_ratio"):
            overrides["attack_ratio"] = args.attack_ratio
        if hasattr(args, "attack_weights"):
            overrides["attack_weights"] = args.attack_weights
        if hasattr(args, "strict_attacks"):
            overrides["strict_attacks"] = args.strict_attacks
    else:
        stage = getattr(args, "stage", 1)
        mode = canonicalize_mode(getattr(args, "preprocess_mode", "stn_fgd"))
        if stage == 1:
            default_name = (
                f"stage1_baseline_eval_{mode}" if getattr(args, "eval_only", False) else "stage1_clean_finetune"
            )
            default_epochs = 3
            default_lr0 = 1e-6
            default_lrf = 1.0
            default_warmup_epochs = 0.0
            default_warmup_bias_lr = 0.0
            default_freeze_all_bn = True
            default_mosaic = 0.0
            default_scale = 0.0
            default_translate = 0.0
            default_fliplr = 0.0
            default_flipud = 0.0
            default_hsv_h = 0.0
            default_hsv_s = 0.0
            default_hsv_v = 0.0
            default_mixup = 0.0
            default_cutmix = 0.0
            default_copy_paste = 0.0
            default_close_mosaic = 0
            default_save_period = 1
            default_p_clean = 1.0
            default_adaptive = False
            default_att_names = []
            default_att_ratios = []
            default_eval_w = [1.0]
        else:
            default_name = "stage2_adaptive_at"
            default_epochs = 20
            default_lr0 = 1e-5
            default_lrf = 0.01
            default_warmup_epochs = 0.0
            default_warmup_bias_lr = 0.0
            default_freeze_all_bn = False
            default_mosaic = 1.0
            default_scale = 0.5
            default_translate = 0.1
            default_fliplr = 0.5
            default_flipud = 0.0
            default_hsv_h = 0.015
            default_hsv_s = 0.7
            default_hsv_v = 0.4
            default_mixup = 0.0
            default_cutmix = 0.0
            default_copy_paste = 0.0
            default_close_mosaic = 10
            default_save_period = -1
            default_p_clean = 0.65
            default_adaptive = True
            default_att_names = ["pgd", "bim", "mim", "worstk"]
            default_att_ratios = [0.12, 0.05, 0.10, 0.08]
            default_eval_w = [0.20, 0.20, 0.15, 0.15, 0.30]

        stage1_w = getattr(args, "stage1_weights", None)
        model_path = getattr(args, "model", "yolo12l.pt")
        if stage == 2 and stage1_w:
            model_path = stage1_w

        epochs = getattr(args, "epochs", default_epochs)
        lr0 = getattr(args, "lr0", default_lr0)
        lrf = getattr(args, "lrf", default_lrf)
        warmup_epochs = getattr(args, "warmup_epochs", default_warmup_epochs)
        warmup_bias_lr = getattr(args, "warmup_bias_lr", default_warmup_bias_lr)
        freeze_all_bn = getattr(args, "freeze_all_bn", default_freeze_all_bn)
        mosaic = getattr(args, "mosaic", default_mosaic)
        scale = getattr(args, "scale", default_scale)
        translate = getattr(args, "translate", default_translate)
        fliplr = getattr(args, "fliplr", default_fliplr)
        flipud = getattr(args, "flipud", default_flipud)
        hsv_h = getattr(args, "hsv_h", default_hsv_h)
        hsv_s = getattr(args, "hsv_s", default_hsv_s)
        hsv_v = getattr(args, "hsv_v", default_hsv_v)
        mixup = getattr(args, "mixup", default_mixup)
        cutmix = getattr(args, "cutmix", default_cutmix)
        copy_paste = getattr(args, "copy_paste", default_copy_paste)
        close_mosaic = getattr(args, "close_mosaic", default_close_mosaic)
        save_period = getattr(args, "save_period", default_save_period)
        p_clean = getattr(args, "p_clean", default_p_clean)
        adaptive_ratio = getattr(args, "adaptive_ratio", default_adaptive)
        attack_name = getattr(args, "attack_name", default_att_names)
        attack_ratio = getattr(args, "attack_ratio", default_att_ratios)
        eval_weights = getattr(args, "eval_weights", default_eval_w)
        exp_name = getattr(args, "name", default_name)

        overrides = {
            "stage": stage,
            "stage1_weights": stage1_w,
            "train_head_only": getattr(args, "train_head_only", True),
            "freeze_bn": getattr(args, "freeze_bn", True),
            "freeze_all_bn": freeze_all_bn,
            "eval_only": getattr(args, "eval_only", False),
            "eval_epoch0": getattr(args, "eval_epoch0", False),
            "model": model_path,
            "data": getattr(args, "data", "coco_train_traffic5k.yaml"),
            "epochs": epochs,
            "batch": getattr(args, "batch", 16),
            "device": getattr(args, "device", "0"),
            "imgsz": getattr(args, "imgsz", 640),
            "optimizer": getattr(args, "optimizer", "AdamW"),
            "lr0": lr0,
            "lrf": lrf,
            "warmup_epochs": warmup_epochs,
            "warmup_bias_lr": warmup_bias_lr,
            "mosaic": mosaic,
            "scale": scale,
            "translate": translate,
            "fliplr": fliplr,
            "flipud": flipud,
            "hsv_h": hsv_h,
            "hsv_s": hsv_s,
            "hsv_v": hsv_v,
            "mixup": mixup,
            "cutmix": cutmix,
            "copy_paste": copy_paste,
            "close_mosaic": close_mosaic,
            "save_period": save_period,
            "project": getattr(args, "project", "runs/train_adv_fgd_stn_staged"),
            "name": exp_name,
            "resume": False,
            "cache": cache_val,
            "amp": getattr(args, "amp", "bf16"),
            "fraction": getattr(args, "fraction", 1.0),
            "preprocess_mode": mode,
            "fgd_weights": getattr(args, "fgd_weights", "fgd_yolo12l.pt"),
            "stn_weights": getattr(
                args,
                "stn_weights",
                "runs/detect/runs/train_stn/stn_train_yolo12l_coco_train_traffic5k_ep50_b8_k10_deg30.0_tr0.10/weights/best.pt",
            ),
            "adaptive_ratio": adaptive_ratio,
            "attack_through_preprocess": getattr(args, "attack_through_preprocess", True),
            "p_clean": p_clean,
            "tau": getattr(args, "tau", 5.0),
            "update_interval": getattr(args, "update_interval", 5),
            "eval_weights": eval_weights,
            "attack_name": attack_name,
            "attack_ratio": attack_ratio,
            "attack_weights": getattr(args, "attack_weights", "current"),
            "strict_attacks": getattr(args, "strict_attacks", True),
        }

    if hasattr(args, "workers") and args.workers is not None:
        overrides["workers"] = args.workers
    if hasattr(args, "classes") and args.classes is not None:
        overrides["classes"] = args.classes
    if hasattr(args, "seed") and args.seed is not None:
        overrides["seed"] = int(args.seed)

    trainer = DetectionTrainerAdvFGDSTNStaged(
        overrides=overrides, attack_weights=getattr(args, "attack_weights", "current")
    )
    if getattr(args, "eval_only", False):
        trainer.eval_only()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
