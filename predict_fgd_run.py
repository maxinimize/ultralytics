"""Run Phase 2 inference with external frozen FGD and the standard YOLO prediction pipeline."""

import argparse
from pathlib import Path

from ultralytics.defenses.hgd_fgd import apply_fgd, load_fgd_denoiser
from ultralytics.models.yolo.detect.predict import DetectionPredictor


class FGDPredictor(DetectionPredictor):
    """Apply Phase 1 preprocessing after standard YOLO letterboxing and RGB normalization."""

    def __init__(self, overrides, fgd_weights=None):
        """Keep FGD outside the prediction detector and load it on the selected device."""
        super().__init__(overrides=overrides)
        self.fgd_weights = fgd_weights
        self.denoiser = None

    def setup_model(self, *args, **kwargs):
        """Reuse detector setup, then load the optional external frozen denoiser."""
        super().setup_model(*args, **kwargs)
        if self.fgd_weights is not None:
            self.denoiser, _ = load_fgd_denoiser(self.fgd_weights, self.device)

    def preprocess(self, images):
        """Denoise all clean/attacked inputs before detection without changing image geometry."""
        images = super().preprocess(images)
        return apply_fgd(self.denoiser, images) if self.denoiser is not None else images


def main():
    """Parse detector/FGD weights and reuse the existing inference and output behavior."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--fgd_weights")
    parser.add_argument("--use_fgd", choices=["true", "false"], default="true")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    if args.use_fgd == "true" and (not args.fgd_weights or not Path(args.fgd_weights).is_file()):
        parser.error(f"--use_fgd=true requires an existing FGD checkpoint: {args.fgd_weights!r}")
    options = vars(args).copy()
    options.pop("use_fgd")
    options.pop("fgd_weights")
    FGDPredictor(options, args.fgd_weights if args.use_fgd == "true" else None).predict_cli()


if __name__ == "__main__":
    main()
