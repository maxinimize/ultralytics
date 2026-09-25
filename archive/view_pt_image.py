import torch
import torchvision
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Convert an adversarial .pt cache file to an image.")
    parser.add_argument("pt_file", type=str, help="Path to the .pt file (e.g. img_dp.pt)")
    parser.add_argument("--out", type=str, default=None, help="Output image path (default: same directory, .jpg extension)")
    args = parser.parse_args()

    pt_path = Path(args.pt_file)
    if not pt_path.exists():
        print(f"File not found: {pt_path}")
        return

    # Load the tensor
    tensor = torch.load(pt_path, map_location="cpu")
    
    # YOLO validation tensors are usually (C, H, W) and normalized to [0, 1]
    if tensor.dim() == 4:
        # If batch dimension exists, take the first image
        tensor = tensor[0]
        
    if tensor.dim() != 3:
        print(f"Expected 3D tensor (C, H, W), got {tensor.dim()}D tensor.")
        return
        
    out_path = args.out
    if out_path is None:
        out_path = pt_path.with_suffix('.jpg')
        
    # Save the tensor as an image
    torchvision.utils.save_image(tensor, out_path)
    print(f"Successfully converted {pt_path.name} to {Path(out_path).name}")

if __name__ == "__main__":
    main()
