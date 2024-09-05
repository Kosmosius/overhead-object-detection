# scripts/preprocess_data.py

import os
from PIL import Image
import numpy as np
from src.data.augmentation import DataAugmentor
from src.utils.file_utils import ensure_dir_exists
from transformers import DetrFeatureExtractor
import argparse
import json


def chip_image(image, chip_size, overlap):
    """
    Chip a large image into smaller sections of specified size with optional overlap.
    
    Args:
        image (PIL.Image): The input image to be chipped.
        chip_size (tuple): The (width, height) of each chip.
        overlap (float): The overlap ratio between chips (0 <= overlap < 1).

    Returns:
        List[PIL.Image]: List of image chips.
        List[dict]: Metadata about each chip (e.g., original position in the source image).
    """
    width, height = image.size
    chip_w, chip_h = chip_size
    stride_w = int(chip_w * (1 - overlap))
    stride_h = int(chip_h * (1 - overlap))

    chips = []
    chip_metadata = []

    for y in range(0, height - chip_h + 1, stride_h):
        for x in range(0, width - chip_w + 1, stride_w):
            chip = image.crop((x, y, x + chip_w, y + chip_h))
            chips.append(chip)
            chip_metadata.append({
                "original_x": x,
                "original_y": y,
                "chip_width": chip_w,
                "chip_height": chip_h
            })

    return chips, chip_metadata


def adjust_gsd(image, original_gsd, target_gsd):
    """
    Adjust the image's resolution based on Ground Sample Distance (GSD).

    Args:
        image (PIL.Image): The input image to be resized.
        original_gsd (float): Original GSD of the image (meters/pixel).
        target_gsd (float): Target GSD for resizing (meters/pixel).

    Returns:
        PIL.Image: Resized image according to the new GSD.
    """
    if original_gsd == target_gsd:
        return image

    scale_factor = original_gsd / target_gsd
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)

    return image.resize((new_width, new_height), Image.BICUBIC)


def preprocess_dataset(data_dir, output_dir, chip_size=(512, 512), overlap=0.2, augment=False, original_gsd=None, target_gsd=None):
    """
    Preprocess the dataset by chipping large images and applying optional GSD adjustment and augmentations.

    Args:
        data_dir (str): Path to the directory containing raw images.
        output_dir (str): Path to save the processed images.
        chip_size (tuple): The size of each chip (width, height).
        overlap (float): The overlap ratio between chips (0 <= overlap < 1).
        augment (bool): Whether to apply data augmentations.
        original_gsd (float): The original GSD (in meters/pixel) of the dataset. Optional.
        target_gsd (float): The target GSD (in meters/pixel) for resizing. Optional.
    """
    # Ensure the output directory exists
    ensure_dir_exists(output_dir)

    augmentor = DataAugmentor(apply_geometric=augment, apply_photometric=augment)

    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
            continue

        # Load image
        image = Image.open(img_path)

        # Adjust image for GSD if specified
        if original_gsd and target_gsd:
            image = adjust_gsd(image, original_gsd, target_gsd)

        # Chip the image
        chips, chip_metadata = chip_image(image, chip_size, overlap)

        # Apply augmentations if required and save the chips
        for idx, chip in enumerate(chips):
            if augment:
                chip = augmentor.apply_augmentation(chip)

            # Save chip
            chip_output_path = os.path.join(output_dir, f"{img_name.split('.')[0]}_chip_{idx}.png")
            chip.save(chip_output_path)

            # Save metadata for each chip (location in original image)
            metadata_output_path = os.path.join(output_dir, f"{img_name.split('.')[0]}_chip_{idx}_metadata.json")
            with open(metadata_output_path, 'w') as meta_file:
                json.dump(chip_metadata[idx], meta_file, indent=4)

            print(f"Processed and saved chip: {chip_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset for object detection, including chipping and GSD adjustment.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing raw images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed images.")
    parser.add_argument("--chip_size", type=int, nargs=2, default=[512, 512], help="Width and height of each image chip.")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap ratio between image chips (0 <= overlap < 1).")
    parser.add_argument("--augment", action="store_true", help="Whether to apply data augmentations.")
    parser.add_argument("--original_gsd", type=float, help="Original GSD of the dataset (in meters/pixel).")
    parser.add_argument("--target_gsd", type=float, help="Target GSD for resizing (in meters/pixel).")

    args = parser.parse_args()

    preprocess_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        chip_size=tuple(args.chip_size),
        overlap=args.overlap,
        augment=args.augment,
        original_gsd=args.original_gsd,
        target_gsd=args.target_gsd
    )

"""
python scripts/preprocess_data.py --data_dir /path/to/raw/images --output_dir /path/to/processed/images --chip_size 512 512 --overlap 0.2 --augment --original_gsd 1.5 --target_gsd 1.0
"""
