# src/data/augmentation.py

import random
from PIL import Image, ImageEnhance
import torchvision.transforms as T
import torch

class DataAugmentor:
    """
    Class to apply data augmentation for object detection tasks.
    This includes geometric and photometric transformations.
    """

    def __init__(self, apply_geometric=True, apply_photometric=True, seed=None):
        """
        Initialize the DataAugmentor class with options to apply geometric and photometric transforms.

        Args:
            apply_geometric (bool): Whether to apply geometric transformations.
            apply_photometric (bool): Whether to apply photometric transformations.
            seed (int, optional): Seed for reproducibility.
        """
        self.apply_geometric = apply_geometric
        self.apply_photometric = apply_photometric
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        self.transform_pipeline = None

    def _get_geometric_transforms(self):
        """
        Create the geometric transformation pipeline.

        Returns:
            T.Compose: Composed geometric transformations.
        """
        transforms = []

        # Random horizontal flip with a probability of 50%
        if random.random() > 0.5:
            transforms.append(T.RandomHorizontalFlip(p=1))

        # Random rotation between -10 and 10 degrees
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            transforms.append(T.RandomRotation((angle, angle)))

        # Random resizing while maintaining the aspect ratio
        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.2)  # Scale between 80% and 120%
            transforms.append(T.Resize((int(scale * 224), int(scale * 224))))  # Example with fixed size, adjust as needed

        return T.Compose(transforms)

    def _get_photometric_transforms(self, image):
        """
        Apply random photometric transformations to the image.

        Args:
            image (PIL.Image or torch.Tensor): The input image.

        Returns:
            PIL.Image or torch.Tensor: Transformed image.
        """
        if isinstance(image, Image.Image):
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

        elif isinstance(image, torch.Tensor):
            if random.random() > 0.5:
                color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
                image = color_jitter(image)

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
            image = self._get_geometric_transforms()(image)

        if self.apply_photometric:
            image = self._get_photometric_transforms(image)

        return image
