import argparse
import math
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch

from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.models.yolo.detect.train_adv import setup_attack_model
from ultralytics.attacks.attack_bridge import build_attacker, run_attack_on_batch
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER, IterableSimpleNamespace
from ultralytics.data.utils import check_det_dataset


def format_time(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# Custom subset wrapper to delegate attributes to the underlying dataset
class YOLOSubset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def main():
    parser = argparse.ArgumentParser(description="Pre-generate adversarial images for YOLO training dataset")
    parser.add_argument('--model', required=True, help='Path to target/attack model weights (e.g. yolo12l.pt)')
    parser.add_argument('--data', required=True, help='Path to dataset yaml file (e.g. coco_train.yaml)')
    parser.add_argument('--attack_name', default='pgd', help='Attack name (e.g. pgd, dp, mim, cw)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', default='0', help='GPU device ID or cpu')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for adversarial image generation')
    parser.add_argument('--split', default='train', choices=['train', 'val'], help='Dataset split to attack (train or val)')
    args = parser.parse_args()

    # 1. Device setup
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != 'cpu' else "cpu")
    LOGGER.info(f"Using device: {device}")

    # 2. Parse dataset configuration
    data_dict = check_det_dataset(args.data)
    split_path = data_dict.get(args.split)
    if not split_path:
        raise ValueError(f"No {args.split} dataset found in {args.data}")

    # 3. Load model and attacker
    LOGGER.info(f"Loading attack model from: {args.model}")
    model = setup_attack_model(
        args.model,
        device=device,
        nc=len(data_dict['names']),
        training=False,
        imgsz=args.imgsz
    )
    
    # Enable gradients for parameters
    for p in model.parameters():
        p.requires_grad = True

    # Build attacker
    attacker = build_attacker(args.attack_name, model=model, img_size=args.imgsz)
    LOGGER.info(f"Initialized attacker: {args.attack_name}")

    # 4. Build Dataset
    # We use validation mode to load images individually with standard LetterBox resizing
    # and no random augmentations (like Mosaic, MixUp, flips, etc.)
    cfg = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    cfg.imgsz = args.imgsz
    cfg.rect = False
    cfg.cache = False
    cfg.single_cls = False
    cfg.task = 'detect'
    cfg.classes = None
    cfg.fraction = 1.0

    dataset = build_yolo_dataset(
        cfg=cfg,
        img_path=split_path,
        batch=args.batch_size,
        data=data_dict,
        mode='val',  # 'val' disables training augmentations
        stride=int(model.stride.max())
    )
    total_images = len(dataset)
    LOGGER.info(f"Loaded {args.split} dataset from {split_path} with {total_images} images.")

    # 5. Filter out already processed images
    existing_indices = []
    to_process_indices = []
    for i, im_file in enumerate(dataset.im_files):
        p = Path(im_file)
        save_path = p.parent / f"{p.stem}_{args.attack_name}{p.suffix}"
        if save_path.exists():
            existing_indices.append(i)
        else:
            to_process_indices.append(i)

    skipped_count = len(existing_indices)
    if skipped_count > 0:
        LOGGER.info(f"Found {skipped_count} existing adversarial images. They will be skipped.")

    if len(to_process_indices) == 0:
        LOGGER.info("All images have existing adversarial counterparts. Nothing to process.")
        print()
        LOGGER.info("=== Adversarial Generation Summary ===")
        LOGGER.info(f"Total Images: {total_images}")
        LOGGER.info(f"Success: 0")
        LOGGER.info(f"Skipped: {skipped_count}")
        LOGGER.info(f"Failed: 0")
        LOGGER.info(f"Time Elapsed: 00:00:00")
        LOGGER.info("======================================")
        return

    # Wrap the remaining dataset to process via dataloader
    process_dataset = YOLOSubset(dataset, to_process_indices)

    # 6. Build Dataloader
    dataloader = build_dataloader(
        process_dataset,
        batch=args.batch_size,
        workers=4,
        shuffle=False,
        rank=-1
    )

    # 7. Generation loop
    start_time = time.time()
    success_count = 0
    failed_count = 0
    completed_count = skipped_count
    failed_paths = []

    def print_progress():
        elapsed = time.time() - start_time
        processed = success_count + failed_count
        avg_time = elapsed / processed if processed > 0 else 0.0
        remaining = total_images - completed_count
        eta = remaining * avg_time
        
        log_str = (
            f"\r[{args.attack_name.upper()}] {completed_count}/{total_images} completed | "
            f"success={success_count} | skipped={skipped_count} | failed={failed_count} | "
            f"elapsed={format_time(elapsed)} | avg={avg_time:.2f}s/img | ETA={format_time(eta)}"
        )
        sys.stdout.write(log_str)
        sys.stdout.flush()

    # Initial print
    print_progress()

    for batch in dataloader:
        batch_size = len(batch['im_file'])
        
        try:
            # Move tensors to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
            batch['img'] = batch['img'].float() / 255.0  # scale to [0, 1]

            # Run attacker
            with torch.amp.autocast(device_type="cuda", enabled=False):
                imgs_adv = run_attack_on_batch(attacker, batch, label_policy="largest_box")
            
            if imgs_adv is None:
                raise RuntimeError("Attacker returned None (likely no targets in batch)")

            # Process each image in the batch
            for j in range(batch_size):
                im_file = batch['im_file'][j]
                img_adv = imgs_adv[j]
                
                # Get shape metadata and handles potential tensor collation
                resized_shape = batch['resized_shape'][j]
                if isinstance(resized_shape, torch.Tensor):
                    resized_shape = resized_shape.cpu().tolist()
                h, w = resized_shape
                
                ori_shape = batch['ori_shape'][j]
                if isinstance(ori_shape, torch.Tensor):
                    ori_shape = ori_shape.cpu().tolist()
                h0, w0 = ori_shape
                
                # Crop padding
                top_pad = (args.imgsz - h) // 2
                left_pad = (args.imgsz - w) // 2
                img_adv_cropped = img_adv[:, top_pad : top_pad + h, left_pad : left_pad + w]
                
                # Convert to numpy uint8 RGB image
                img_adv_np = (img_adv_cropped.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                
                # Resize back
                img_adv_resized = cv2.resize(img_adv_np, (w0, h0), interpolation=cv2.INTER_LINEAR)
                
                # BGR convert for cv2
                img_adv_bgr = cv2.cvtColor(img_adv_resized, cv2.COLOR_RGB2BGR)
                
                # Save path
                p = Path(im_file)
                save_path = p.parent / f"{p.stem}_{args.attack_name}{p.suffix}"
                
                # Save
                cv2.imwrite(str(save_path), img_adv_bgr)
                success_count += 1
                completed_count += 1

        except Exception as e:
            # If batch processing fails, mark all files in the batch as failed
            for im_file in batch['im_file']:
                failed_count += 1
                completed_count += 1
                failed_paths.append(im_file)
            
            # Print traceback cleanly to stderr to avoid breaking the inline progress line
            sys.stdout.write("\r" + " " * 100 + "\r")
            sys.stdout.flush()
            print(f"Error processing batch: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            
        print_progress()

    # Final newline and summary printing
    print()
    LOGGER.info("=== Adversarial Generation Summary ===")
    LOGGER.info(f"Total Images: {total_images}")
    LOGGER.info(f"Success: {success_count}")
    LOGGER.info(f"Skipped: {skipped_count}")
    LOGGER.info(f"Failed: {failed_count}")
    LOGGER.info(f"Time Elapsed: {format_time(time.time() - start_time)}")
    LOGGER.info("======================================")

    # Write failed image paths if any
    if failed_paths:
        failed_file = Path(f"failed_{args.attack_name}.txt")
        try:
            with open(failed_file, "w") as f:
                for fp in failed_paths:
                    f.write(f"{fp}\n")
            LOGGER.info(f"Failed image paths written to {failed_file}")
        except Exception as e:
            LOGGER.warning(f"Could not write failed image paths to file: {e}")


if __name__ == '__main__':
    main()
