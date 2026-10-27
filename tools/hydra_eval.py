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
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.attacks.attack_bridge import build_attacker
from ultralytics.attacks.attack_utils import setup_attack_model
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.yolo.detect.val_adv_multi import DetectionValidator as DetectionValidatorAdv
from ultralytics.nn.pruning.hydra_convert import iter_hydra_layers
from ultralytics.utils import LOGGER, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops, select_device, unwrap_model


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
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=True,
        help="Run model efficiency benchmark (latency, FPS, GFLOPs, parameters).",
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_false",
        dest="benchmark",
        help="Skip model efficiency benchmark.",
    )
    parser.add_argument("--benchmark_runs", type=int, default=50, help="Number of timing repetitions for benchmark.")
    parser.add_argument("--benchmark_warmup", type=int, default=10, help="Number of warmup iterations for benchmark.")
    return parser.parse_args()


def get_model_sparsity_info(model: torch.nn.Module) -> dict[str, Any]:
    """Inspect model for HYDRA layers or zeroed weights and return comprehensive sparsity statistics."""
    unwrapped = unwrap_model(model)
    hydra_layers = list(iter_hydra_layers(unwrapped))

    total_dense_params = sum(p.numel() for name, p in unwrapped.named_parameters() if "popup_scores" not in name)

    if hydra_layers:
        total_elems = 0
        kept_elems = 0
        total_kernels = 0
        kept_kernels = 0
        pruning_mode = str(getattr(hydra_layers[0][1], "pruning_mode", "unstructured"))

        for _, layer in hydra_layers:
            numel = layer.weight.numel()
            total_elems += numel
            mask = getattr(layer, "fixed_mask", None)
            if mask is None:
                mask = layer.dynamic_mask()

            kept_elems += int((mask > 0.5).sum().item())
            k_count = mask.shape[0] * mask.shape[1]
            total_kernels += k_count
            k_active = (mask > 0.5).any(dim=-1).any(dim=-1)
            kept_kernels += int(k_active.sum().item())

        pruned_kernels = total_kernels - kept_kernels
        kernel_sparsity = 1.0 - (kept_kernels / total_kernels) if total_kernels > 0 else 0.0
        element_sparsity = 1.0 - (kept_elems / total_elems) if total_elems > 0 else 0.0
        effective_nonzero = total_dense_params - (total_elems - kept_elems)

        return {
            "is_hydra": True,
            "pruning_mode": pruning_mode,
            "eligible_layers": len(hydra_layers),
            "total_kernels": total_kernels,
            "kept_kernels": kept_kernels,
            "pruned_kernels": pruned_kernels,
            "kernel_sparsity": kernel_sparsity,
            "total_elements": total_elems,
            "kept_elements": kept_elems,
            "element_sparsity": element_sparsity,
            "sparsity": element_sparsity,
            "kept_ratio": kept_elems / total_elems if total_elems > 0 else 1.0,
            "total_dense_params": total_dense_params,
            "effective_nonzero_params": effective_nonzero,
            "physical_compression": 1.0,
        }

    # Materialized model (plain nn.Conv2d layers with zeros in pruned weights)
    conv_layers = [m for m in unwrapped.modules() if isinstance(m, torch.nn.Conv2d)]
    total_elems = 0
    kept_elems = 0
    total_kernels = 0
    kept_kernels = 0

    for m in conv_layers:
        w = m.weight.data
        total_elems += w.numel()
        nonzero_mask = (w != 0)
        kept_elems += int(nonzero_mask.sum().item())

        k_count = w.shape[0] * w.shape[1]
        total_kernels += k_count
        k_active = nonzero_mask.any(dim=-1).any(dim=-1)
        k_kept = int(k_active.sum().item())
        kept_kernels += k_kept

    pruned_kernels = total_kernels - kept_kernels
    kernel_sparsity = 1.0 - (kept_kernels / total_kernels) if total_kernels > 0 else 0.0
    element_sparsity = 1.0 - (kept_elems / total_elems) if total_elems > 0 else 0.0
    effective_nonzero = sum(int((p != 0).sum().item()) for p in unwrapped.parameters())

    return {
        "is_hydra": False,
        "is_materialized": element_sparsity > 0.0,
        "eligible_layers": len(conv_layers),
        "total_kernels": total_kernels,
        "kept_kernels": kept_kernels,
        "pruned_kernels": pruned_kernels,
        "kernel_sparsity": kernel_sparsity,
        "total_elements": total_elems,
        "kept_elements": kept_elems,
        "element_sparsity": element_sparsity,
        "sparsity": element_sparsity,
        "kept_ratio": kept_elems / total_elems if total_elems > 0 else 1.0,
        "total_dense_params": total_dense_params,
        "effective_nonzero_params": effective_nonzero,
        "physical_compression": 1.0,
    }


def benchmark_model_efficiency(
    model: torch.nn.Module,
    model_path: str | None = None,
    imgsz: int = 640,
    batch_size: int = 1,
    device: str = "cpu",
    half: bool = False,
    num_warmup: int = 10,
    num_runs: int = 50,
) -> dict[str, Any]:
    """Measure inference latency, FPS, GFLOPs, dense & effective parameter count, and file size."""
    dev = select_device(device, verbose=False)
    unwrapped = copy.deepcopy(unwrap_model(model)).to(dev).eval()

    if half and dev.type != "cpu":
        unwrapped = unwrapped.half()
        dtype = torch.float16
    else:
        unwrapped = unwrapped.float()
        dtype = torch.float32

    # 1. Parameter counts
    dense_params = sum(p.numel() for name, p in unwrapped.named_parameters() if "popup_scores" not in name)
    effective_nonzeros = sum(int((p != 0).sum().item()) for name, p in unwrapped.named_parameters() if "popup_scores" not in name)

    # 2. Model file size
    file_size_mb = 0.0
    if model_path is not None and Path(model_path).exists():
        file_size_mb = Path(model_path).stat().st_size / (1024 * 1024)

    # 3. GFLOPs
    try:
        gflops = float(get_flops(unwrapped, imgsz))
    except Exception as e:
        LOGGER.warning(f"Failed to calculate FLOPs: {e}")
        gflops = 0.0

    # 4. Latency measurement (batch_size=1 for latency per image)
    dummy = torch.randn(batch_size, 3, imgsz, imgsz, device=dev, dtype=dtype)
    times = []

    with torch.no_grad():
        # Warmup
        for _ in range(num_warmup):
            _ = unwrapped(dummy)
        if dev.type == "cuda":
            torch.cuda.synchronize()

        # Timing
        for _ in range(num_runs):
            if dev.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = unwrapped(dummy)
                end_event.record()
                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
            else:
                t0 = time.perf_counter()
                _ = unwrapped(dummy)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
            times.append(elapsed_ms / batch_size)

    mean_lat = float(np.mean(times))
    median_lat = float(np.median(times))
    fps = 1000.0 / median_lat if median_lat > 0 else 0.0

    return {
        "device": str(dev),
        "imgsz": imgsz,
        "batch_size": batch_size,
        "precision": "FP16" if dtype == torch.float16 else "FP32",
        "dense_params": dense_params,
        "effective_nonzeros": effective_nonzeros,
        "file_size_mb": file_size_mb,
        "gflops": gflops,
        "mean_latency_ms": mean_lat,
        "median_latency_ms": median_lat,
        "fps": fps,
        "num_runs": num_runs,
    }


def _ensure_namespace_compatibility(model: torch.nn.Module) -> None:
    """Ensure model.args and criterion.hyp are IterableSimpleNamespace with complete hyperparameters."""
    base_args = copy.deepcopy(DEFAULT_CFG_DICT)
    if hasattr(model, "args") and model.args is not None:
        if isinstance(model.args, IterableSimpleNamespace):
            base_args.update(vars(model.args))
        elif isinstance(model.args, dict):
            base_args.update(model.args)
    model.args = IterableSimpleNamespace(**base_args)

    if hasattr(model, "criterion") and hasattr(model.criterion, "hyp"):
        base_hyp = copy.deepcopy(DEFAULT_CFG_DICT)
        if isinstance(model.criterion.hyp, IterableSimpleNamespace):
            base_hyp.update(vars(model.criterion.hyp))
        elif isinstance(model.criterion.hyp, dict):
            base_hyp.update(model.criterion.hyp)
        model.criterion.hyp = IterableSimpleNamespace(**base_hyp)


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

    # Inspect model sparsity
    yolo_model = YOLO(args.model)
    sparsity_info = get_model_sparsity_info(yolo_model.model)

    # Clean and adversarial evaluation
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

    # Efficiency benchmark
    bench_results = None
    if args.benchmark:
        LOGGER.info("Running efficiency benchmark (latency, FPS, GFLOPs)...")
        bench_results = benchmark_model_efficiency(
            model=yolo_model.model,
            model_path=args.model,
            imgsz=args.imgsz,
            batch_size=1,
            device=args.device,
            half=False,
            num_warmup=args.benchmark_warmup,
            num_runs=args.benchmark_runs,
        )

    print("\n" + "=" * 70)
    print("HYDRA COMPREHENSIVE EVALUATION & BENCHMARK REPORT")
    print("=" * 70)

    # 1. Sparsity & Model Statistics
    print("\n[Model & Sparsity Statistics]")
    print(f"  Model Path:              {args.model}")
    print(f"  Eligible Conv2d Layers:  {sparsity_info['eligible_layers']}")
    print(f"  Total Eligible Kernels:  {sparsity_info['total_kernels']}")
    print(f"  Retained Kernels:        {sparsity_info['kept_kernels']}")
    print(f"  Pruned Kernels:          {sparsity_info['pruned_kernels']}")
    print(f"  Actual Kernel Sparsity:  {sparsity_info['kernel_sparsity']:.4f} ({sparsity_info['kernel_sparsity']*100:.2f}%)")
    print(f"  Actual Element Sparsity: {sparsity_info['element_sparsity']:.4f} ({sparsity_info['element_sparsity']*100:.2f}%)")
    print(f"  Total Dense Parameters:  {sparsity_info['total_dense_params']:,}")
    print(f"  Effective Nonzero Params:{sparsity_info['effective_nonzero_params']:,}")
    print(f"  Physical Compression:    {sparsity_info['physical_compression']:.2f}x (Plan A: mask-based)")

    # 2. Clean & Adversarial Robustness Metrics
    print("\n[Detection Performance & Adversarial Robustness]")
    print(f"  {'Attack Mode':<12} | {'Precision':<10} | {'Recall':<10} | {'mAP50':<10} | {'mAP50-95':<10}")
    print("  " + "-" * 58)
    for name, metrics in results.items():
        p_str = f"{metrics.get('precision', 0.0):.4f}" if 'precision' in metrics else "-"
        r_str = f"{metrics.get('recall', 0.0):.4f}" if 'recall' in metrics else "-"
        m50_str = f"{metrics.get('mAP50', 0.0):.4f}"
        m_str = f"{metrics.get('mAP50-95', 0.0):.4f}"
        print(f"  {name.upper():<12} | {p_str:<10} | {r_str:<10} | {m50_str:<10} | {m_str:<10}")

    # 3. Efficiency Benchmark Metrics
    if bench_results is not None:
        print("\n[Efficiency Benchmark]")
        print(f"  Device:                  {bench_results['device']}")
        print(f"  Image Size:              {bench_results['imgsz']}x{bench_results['imgsz']}")
        print(f"  Precision:               {bench_results['precision']}")
        print(f"  Model File Size:         {bench_results['file_size_mb']:.2f} MB")
        print(f"  GFLOPs:                  {bench_results['gflops']:.2f}")
        print(f"  Median Latency:          {bench_results['median_latency_ms']:.2f} ms/image")
        print(f"  Mean Latency:            {bench_results['mean_latency_ms']:.2f} ms/image")
        print(f"  Inference FPS:           {bench_results['fps']:.2f} frames/sec")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
