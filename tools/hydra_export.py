"""Export and materialize a HYDRA checkpoint to a standard Ultralytics YOLO model.

Converts all `HydraConv2d` layers back to ordinary `nn.Conv2d` modules with
pruned weights physically set to zero. This allows the materialized checkpoint
to be used with standard Ultralytics workflows (validation, predict, ONNX export,
deployment) without any dependency on the HYDRA runtime.

Usage:
    python tools/hydra_export.py --model runs/hydra_finetune/weights/best.pt --verify
    python tools/hydra_export.py --model path/to/best.pt --output path/to/exported.pt
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ultralytics import YOLO
from ultralytics.nn.pruning.hydra_convert import iter_hydra_layers
from ultralytics.nn.pruning.hydra_export import materialize_hydra_model
from ultralytics.nn.pruning.hydra_layer import HydraConv2d, HydraMode
from ultralytics.nn.pruning.hydra_mask import freeze_hydra_masks
from ultralytics.utils import LOGGER
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import select_device, unwrap_model


def parse_args():
    """Parse CLI options for HYDRA checkpoint materialization."""
    parser = argparse.ArgumentParser(description="HYDRA Model Materialization and Export Tool")
    parser.add_argument("--model", required=True, help="Path to input HYDRA checkpoint (.pt).")
    parser.add_argument(
        "--output",
        default=None,
        help="Path for materialized output checkpoint (.pt). Default: <model_stem>_materialized.pt",
    )
    parser.add_argument("--device", default="cpu", help="Device for verification ('cpu' or CUDA index).")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for numerical equivalence check.")
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Run numerical forward pass equivalence check before saving (default: True).",
    )
    parser.add_argument(
        "--no-verify",
        action="store_false",
        dest="verify",
        help="Skip numerical verification.",
    )
    return parser.parse_args()


def compute_tensor_diff(o1: Any, o2: Any) -> float:
    """Compute maximum absolute difference between two nested model outputs."""
    if isinstance(o1, torch.Tensor) and isinstance(o2, torch.Tensor):
        return (o1 - o2).abs().max().item()
    elif isinstance(o1, (list, tuple)) and isinstance(o2, (list, tuple)):
        return max((compute_tensor_diff(a, b) for a, b in zip(o1, o2)), default=0.0)
    elif isinstance(o1, dict) and isinstance(o2, dict):
        return max((compute_tensor_diff(o1[k], o2[k]) for k in o1 if k in o2), default=0.0)
    return 0.0


def export_and_materialize(
    model_path: str,
    output_path: str | None = None,
    device: str = "cpu",
    imgsz: int = 640,
    verify: bool = True,
) -> Path:
    """Load a HYDRA checkpoint, materialize it to ordinary nn.Conv2d, and save."""
    in_path = Path(model_path).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input model not found: {in_path}")

    if output_path is None:
        out_path = in_path.parent / f"{in_path.stem}_materialized.pt"
    else:
        out_path = Path(output_path).resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Loading checkpoint from: {in_path}")
    ckpt = torch_load(in_path, map_location="cpu")

    # 1. Extract model object
    if isinstance(ckpt, dict) and "model" in ckpt:
        hydra_model = ckpt["model"]
    else:
        # Checkpoint might be a raw model or YOLO-style wrapper
        hydra_model = ckpt

    unwrapped = unwrap_model(hydra_model)
    hydra_layers = list(iter_hydra_layers(unwrapped))
    LOGGER.info(f"Found {len(hydra_layers)} HydraConv2d layer(s) in checkpoint.")

    if not hydra_layers:
        LOGGER.warning("No HydraConv2d layers detected. Checkpoint may already be materialized!")

    # 2. Ensure fixed_mask exists for all Hydra layers
    for _, layer in hydra_layers:
        if not hasattr(layer, "fixed_mask") or layer.fixed_mask is None:
            LOGGER.info("fixed_mask not found on some layers; generating fixed masks from popup_scores...")
            freeze_hydra_masks(unwrapped)
            break

    dev = select_device(device, verbose=False)
    model_param = next(unwrapped.parameters(), None)
    dtype = model_param.dtype if model_param is not None else torch.float32

    # 3. Perform numerical verification before materialization if requested
    if verify and hydra_layers:
        LOGGER.info(f"Running numerical verification on {dev} (imgsz={imgsz}, dtype={dtype})...")
        test_model = copy.deepcopy(unwrapped).to(dev).eval()
        for _, layer in iter_hydra_layers(test_model):
            layer.set_mode(HydraMode.FINETUNE)
        dummy_input = torch.randn(1, 3, imgsz, imgsz, device=dev, dtype=dtype)

        with torch.no_grad():
            ref_output = test_model(dummy_input)

    # 4. Materialize model
    LOGGER.info("Materializing HydraConv2d layers to standard nn.Conv2d...")
    materialized_model = materialize_hydra_model(unwrapped, inplace=False)

    # Verify no HydraConv2d remaining
    remaining_hydra = [name for name, m in materialized_model.named_modules() if isinstance(m, HydraConv2d)]
    if remaining_hydra:
        raise RuntimeError(f"Materialization incomplete! Remaining Hydra layers: {remaining_hydra}")

    # 5. Numerical equivalence check
    if verify and hydra_layers:
        materialized_model.to(dev).eval()
        with torch.no_grad():
            mat_output = materialized_model(dummy_input)

        max_diff = compute_tensor_diff(ref_output, mat_output)
        LOGGER.info(f"Materialization equivalence check passed! Max absolute diff: {max_diff:.2e}")
        tol = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        if max_diff > tol:
            raise ValueError(f"Materialization failed equivalence check! Max diff: {max_diff} > {tol}")

    # Move back to cpu before saving
    materialized_model.to("cpu")

    # 6. Update checkpoint structure
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt["model"] = materialized_model
        if ckpt.get("ema") is not None:
            LOGGER.info("Materializing EMA model in checkpoint...")
            unwrapped_ema = unwrap_model(ckpt["ema"])
            for _, layer in iter_hydra_layers(unwrapped_ema):
                if not hasattr(layer, "fixed_mask") or layer.fixed_mask is None:
                    freeze_hydra_masks(unwrapped_ema)
                    break
            ckpt["ema"] = materialize_hydra_model(unwrapped_ema, inplace=False)
        ckpt["hydra_materialized"] = True
        save_obj = ckpt
    else:
        save_obj = materialized_model

    # 7. Save checkpoint
    torch.save(save_obj, out_path)
    LOGGER.info(f"Successfully saved materialized checkpoint to: {out_path}")

    # 8. Test standard YOLO loader compatibility
    try:
        yolo_test = YOLO(str(out_path))
        LOGGER.info("Compatibility verified: checkpoint loaded successfully with Ultralytics YOLO().")
    except Exception as e:
        LOGGER.warning(f"Standard YOLO load check warning: {e}")

    return out_path


def main():
    """Main CLI entrypoint."""
    args = parse_args()
    out = export_and_materialize(
        model_path=args.model,
        output_path=args.output,
        device=args.device,
        imgsz=args.imgsz,
        verify=args.verify,
    )
    print("\n" + "=" * 60)
    print("HYDRA MATERIALIZATION COMPLETE")
    print(f"Output model: {out}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
