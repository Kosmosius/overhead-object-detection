# scripts/preprocess_data.py

import os
from PIL import Image
from src.data.augmentation import DataAugmentor

def preprocess_images(image_dir, output_dir, resize=(800, 800), augment=False):
    """
    Preprocess images by resizing and optionally applying augmentations.

    Args:
        image_dir (str): Directory containing the raw images.
        output_dir (str): Directory to save the processed images.
        resize (tuple): Size to which images should be resized (width, height).
        augment (bool): Whether to apply augmentations.
    """
    # Initialize the DataAugmentor for applying augmentation
    augmentor = DataAugmentor(apply_geometric=augment, apply_photometric=augment)

    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        with Image.open(img_path) as img:
            # Resize the image
            img_resized = img.resize(resize)

            # Apply augmentation if specified
            if augment:
                img_resized = augmentor.apply_augmentation(img_resized)

            # Save the processed image
            img_resized.save(os.path.join(output_dir, img_name))
            print(f"Processed and saved: {img_name}")

if __name__ == "__main__":
    IMAGE_DIR = "/path/to/raw/images"  # Replace with actual path
    OUTPUT_DIR = "/path/to/processed/images"  # Replace with actual path
    RESIZE = (800, 800)  # Set your desired resize dimensions
    AUGMENT = True  # Set to True if you want to apply augmentations

    preprocess_images(IMAGE_DIR, OUTPUT_DIR, resize=RESIZE, augment=AUGMENT)
