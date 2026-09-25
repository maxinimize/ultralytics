"""Evaluate dense and HYDRA-pruned models with clean and adversarial metrics.

Reuses the project's existing clean and adversarial validators from
`ultralytics.models.yolo.detect.val_adv_multi`.

Outputs:
    - requested keep/prune ratio and actual mask sparsity (if HYDRA model)
    - clean precision / recall / mAP50 / mAP50-95
    - PGD mAP50 / mAP50-95
    - BIM mAP50 / mAP50-95
    - MIM mAP50 / mAP50-95
    - Worst-of-k mAP50 / mAP50-95 (when requested)
"""

from __future__ import annotations

import argparse
import copy
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from ultralytics import YOLO
from ultralytics.attacks.attack_bridge import build_attacker
from ultralytics.attacks.attack_utils import setup_attack_model
from ultralytics.models.yolo.detect.val_adv_multi import DetectionValidator as DetectionValidatorAdv
from ultralytics.nn.pruning.hydra_convert import iter_hydra_layers
from ultralytics.utils import LOGGER, IterableSimpleNamespace
from ultralytics.utils.torch_utils import select_device, unwrap_model


def parse_args():
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="HYDRA Clean and Adversarial Evaluator")
    parser.add_argument("--model", required=True, help="Path to model weights (.pt).")
    parser.add_argument("--data", required=True, help="Path to dataset YAML file.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for validation.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--device", default="cpu", help="Device to use ('cpu' or CUDA device index).")
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["clean", "pgd", "bim", "mim"],
        help="List of evaluation modes to run (e.g. clean, pgd, bim, mim, worstk).",
    )
    parser.add_argument(
        "--attack_weights",
        default=None,
        help="Optional surrogate model path for black-box/transfer evaluation. If omitted, uses white-box evaluation.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Optional class filtering indices (e.g. 0 1 2 3 5 6 7 9 11 12).",
    )
    return parser.parse_args()


def get_model_sparsity_info(model: torch.nn.Module) -> dict[str, Any]:
    """Inspect model for HYDRA layers and return sparsity statistics."""
    hydra_layers = list(iter_hydra_layers(model))
    if not hydra_layers:
        return {"is_hydra": False, "sparsity": 0.0, "kept_ratio": 1.0}

    total_elems = 0
    kept_elems = 0
    for _, layer in hydra_layers:
        numel = layer.weight.numel()
        total_elems += numel
        if hasattr(layer, "fixed_mask"):
            kept_elems += int(layer.fixed_mask.sum().item())
        else:
            kept_elems += int(round(layer.keep_ratio * numel))

    sparsity = 1.0 - (kept_elems / total_elems) if total_elems > 0 else 0.0
    return {
        "is_hydra": True,
        "total_elements": total_elems,
        "kept_elements": kept_elems,
        "sparsity": sparsity,
        "kept_ratio": kept_elems / total_elems if total_elems > 0 else 1.0,
    }


def _ensure_namespace_compatibility(model: torch.nn.Module) -> None:
    """Ensure model.args and criterion.hyp are IterableSimpleNamespace."""
    if hasattr(model, "args") and isinstance(model.args, dict):
        model.args = IterableSimpleNamespace(**model.args)
    if hasattr(model, "criterion") and hasattr(model.criterion, "hyp") and isinstance(model.criterion.hyp, dict):
        model.criterion.hyp = IterableSimpleNamespace(**model.criterion.hyp)


def evaluate_clean_and_adversarial(
    model_path: str,
    data_path: str,
    attacks: list[str],
    imgsz: int = 640,
    batch: int = 16,
    device: str = "cpu",
    attack_weights: str | None = None,
    classes: list[int] | None = None,
) -> dict[str, dict[str, float]]:
    """Run clean and adversarial validation on the target model.

    Returns:
        Dictionary mapping attack names ('clean', 'pgd', etc.) to metric dicts.
    """
    dev = select_device(device, verbose=False)
    yolo = YOLO(model_path)
    model = yolo.model.to(dev)
    _ensure_namespace_compatibility(model)
    model.eval()

    sparsity_info = get_model_sparsity_info(model)
    if sparsity_info["is_hydra"]:
        LOGGER.info(
            f"HYDRA Model detected: sparsity={sparsity_info['sparsity']:.4f}, "
            f"kept_ratio={sparsity_info['kept_ratio']:.4f} "
            f"({sparsity_info['kept_elements']}/{sparsity_info['total_elements']} weights)"
        )

    results = {}

    # 1. Clean evaluation
    if "clean" in attacks:
        LOGGER.info("Evaluating clean performance...")
        val_kwargs = {
            "data": data_path,
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
            "plots": False,
            "verbose": False,
        }
        if classes is not None:
            val_kwargs["classes"] = classes
        clean_metrics = yolo.val(**val_kwargs)
        results["clean"] = {
            "precision": float(clean_metrics.box.mp),
            "recall": float(clean_metrics.box.mr),
            "mAP50": float(clean_metrics.box.map50),
            "mAP50-95": float(clean_metrics.box.map),
        }

    # 2. Adversarial attacks
    adv_attacks = [a for a in attacks if a != "clean"]
    if adv_attacks:
        for att_name in adv_attacks:
            LOGGER.info(f"Evaluating adversarial robustness against: {att_name}...")

            # Fresh load to prevent inference_mode tensor contamination from clean validation
            adv_yolo = YOLO(model_path)
            eval_model = adv_yolo.model.to(dev).eval()
            _ensure_namespace_compatibility(eval_model)

            use_current_model = (attack_weights is None) or (str(attack_weights).strip().lower() in {"", "none", "current"})
            if use_current_model:
                LOGGER.info("Adversarial evaluation: WHITE-BOX mode (attacking evaluated model itself).")
                surrogate_model = copy.deepcopy(unwrap_model(eval_model)).to(dev).eval()
                for p in surrogate_model.parameters():
                    p.requires_grad_(True)
                _ensure_namespace_compatibility(surrogate_model)
            else:
                LOGGER.info(f"Adversarial evaluation: FIXED-SURROGATE mode from {attack_weights}.")
                surrogate_model = setup_attack_model(
                    attack_weights,
                    device=dev,
                    nc=len(adv_yolo.names),
                    training=False,
                    imgsz=imgsz,
                )
                surrogate_model.eval()
                for p in surrogate_model.parameters():
                    p.requires_grad_(True)
                _ensure_namespace_compatibility(surrogate_model)

            attacker = build_attacker(att_name, model=surrogate_model, img_size=imgsz)

            validator_args = deepcopy(adv_yolo.overrides)
            validator_args["data"] = data_path
            validator_args["imgsz"] = imgsz
            validator_args["batch"] = batch
            validator_args["device"] = device
            validator_args["plots"] = False
            validator_args["save_json"] = False
            if classes is not None:
                validator_args["classes"] = classes

            # Instantiate existing adversarial validator
            validator = DetectionValidatorAdv(args=validator_args)
            validator.attacker = attacker
            validator.attackers = {att_name: attacker}
            validator.attack_names = [att_name]
            validator.attack_ratios = [1.0]

            val_res = validator(model=eval_model)

            mAP50 = float(val_res.get("metrics/mAP50(B)", 0.0))
            mAP50_95 = float(val_res.get("metrics/mAP50-95(B)", 0.0))
            results[att_name] = {
                "mAP50": mAP50,
                "mAP50-95": mAP50_95,
            }

    return results


def main():
    """CLI entry point for hydra_eval."""
    args = parse_args()
    results = evaluate_clean_and_adversarial(
        model_path=args.model,
        data_path=args.data,
        attacks=args.attacks,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        attack_weights=args.attack_weights,
        classes=args.classes,
    )

    print("\n" + "=" * 60)
    print("HYDRA EVALUATION SUMMARY")
    print("=" * 60)
    for name, metrics in results.items():
        if name == "clean":
            print(
                f"  {name.upper():<10} | P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f} | "
                f"mAP50: {metrics['mAP50']:.4f} | mAP50-95: {metrics['mAP50-95']:.4f}"
            )
        else:
            print(
                f"  {name.upper():<10} | mAP50: {metrics['mAP50']:.4f} | mAP50-95: {metrics['mAP50-95']:.4f}"
            )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
