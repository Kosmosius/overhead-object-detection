# src/data/augmentation.py

import random
from PIL import Image, ImageOps, ImageEnhance
import torchvision.transforms as T
import torch

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
            image (PIL.Image or torch.Tensor): The input image.

        Returns:
            PIL.Image or torch.Tensor: Transformed image.
        """
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

        # Apply the transformations
        transform_pipeline = T.Compose(transforms)
        return transform_pipeline(image)

    def photometric_transforms(self, image):
        """
        Apply random photometric transformations to the image.

        Args:
            image (PIL.Image or torch.Tensor): The input image.

        Returns:
            PIL.Image or torch.Tensor: Transformed image.
        """
        if isinstance(image, Image.Image):
            # PIL-based photometric transforms
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
        else:
            # Tensor-based photometric transforms (for future support of Vision Transformers, etc.)
            if random.random() > 0.5:
                image = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)(image)

        return image

    def apply_augmentation(self, image):
        """
        Apply augmentation to the image using geometric and photometric transformations.

        Args:
            image (PIL.Image or torch.Tensor): The input image.

        Returns:
            PIL.Image or torch.Tensor: Augmented image.
        """
        if self.apply_geometric:
            image = self.geometric_transforms(image)

        if self.apply_photometric:
            image = self.photometric_transforms(image)

        return image
