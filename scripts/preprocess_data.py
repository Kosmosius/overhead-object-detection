# scripts/preprocess_data.py

import os
from PIL import Image
from src.data.augmentation import apply_augmentation

def preprocess_images(image_dir, output_dir, resize=(800, 800), augment=False):
    """
    Preprocess images by resizing and optionally applying augmentations.

    Args:
        image_dir (str): Directory containing the raw images.
        output_dir (str): Directory to save the processed images.
        resize (tuple): Size to which images should be resized (width, height).
        augment (bool): Whether to apply augmentations.
    """
    os.makedirs(output_dir, exist_ok=True)
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        with Image.open(img_path) as img:
            img_resized = img.resize(resize)
            if augment:
                img_resized = apply_augmentation(img_resized)
            img_resized.save(os.path.join(output_dir, img_name))
            print(f"Processed and saved: {img_name}")

if __name__ == "__main__":
    IMAGE_DIR = "/path/to/raw/images"  # Replace with actual path
    OUTPUT_DIR = "/path/to/processed/images"  # Replace with actual path
    
    preprocess_images(IMAGE_DIR, OUTPUT_DIR, augment=True)
