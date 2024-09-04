# src/data/augmentation.py

import random
from PIL import Image, ImageOps, ImageEnhance
import torchvision.transforms as T

class DataAugmentor:
    """
    Class to apply data augmentation for object detection tasks.
    This includes geometric and photometric transformations.
    """

    def __init__(self, apply_geometric=True, apply_photometric=True):
        self.apply_geometric = apply_geometric
        self.apply_photometric = apply_photometric

    def geometric_transforms(self, image):
        """
        Apply random geometric transformations to the image.
        Args:
            image (PIL.Image): The input image.

        Returns:
            PIL.Image: Transformed image.
        """
        # Define random geometric transforms (rotation, flipping, resizing, etc.)
        transforms = []

        # Random horizontal flip with a probability of 50%
        if random.random() > 0.5:
            transforms.append(T.RandomHorizontalFlip(p=1))

        # Random rotation between -10 and 10 degrees
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            transforms.append(T.RandomRotation((angle, angle)))

        # Random resizing
        if random.random() > 0.5:
            size = random.uniform(0.8, 1.2)  # Scale between 80% and 120%
            transforms.append(T.Resize(int(image.size[1] * size)))

        # Apply the transformations sequentially
        transform_pipeline = T.Compose(transforms)
        return transform_pipeline(image)

    def photometric_transforms(self, image):
        """
        Apply random photometric transformations to the image.
        Args:
            image (PIL.Image): The input image.

        Returns:
            PIL.Image: Transformed image.
        """
        # Define random photometric transformations (brightness, contrast, saturation, etc.)
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(0.7, 1.3)  # Adjust brightness
            image = enhancer.enhance(factor)

        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(0.7, 1.3)  # Adjust contrast
            image = enhancer.enhance(factor)

        if random.random() > 0.5:
            enhancer = ImageEnhance.Saturation(image)
            factor = random.uniform(0.7, 1.3)  # Adjust saturation
            image = enhancer.enhance(factor)

        return image

    def apply_augmentation(self, image):
        """
        Apply augmentation to the image using geometric and photometric transformations.

        Args:
            image (PIL.Image): The input image.

        Returns:
            PIL.Image: Augmented image.
        """
        # Apply geometric transforms
        if self.apply_geometric:
            image = self.geometric_transforms(image)

        # Apply photometric transforms
        if self.apply_photometric:
            image = self.photometric_transforms(image)

        return image
