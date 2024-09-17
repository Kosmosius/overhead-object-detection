# src/data/augmentation.py

import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Any

class DataAugmentor:
    """
    Class to apply data augmentation for object detection tasks.
    This includes geometric and photometric transformations applied
    to both images and bounding boxes.
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
        self.seed = seed

        transforms = []
        if self.apply_geometric:
            transforms.append(
                A.Affine(scale=(0.8, 1.2), translate_percent=(0.1, 0.1))
            )
        if self.apply_photometric:
            transforms.append(
                A.RandomBrightnessContrast()
            )
        if transforms:
            self.transform = A.Compose(
                transforms,
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'])
            )
        else:
            self.transform = None

    def _get_transforms(self):
        """
        Create the transformation pipeline using Albumentations.

        Returns:
            A.Compose: Composed transformations.
        """
        transforms = []

        if self.apply_geometric:
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=10,
                    interpolation=1,
                    border_mode=0,
                    value=(0, 0, 0),
                    p=0.5
                ),
            ])

        if self.apply_photometric:
            transforms.extend([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
            ])

        # Always convert images to tensor
        transforms.append(ToTensorV2())

        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='pascal_voc',  # or 'coco' depending on your bbox format
                label_fields=['category_ids'],
                min_area=0,
                min_visibility=0.5
            )
        )

    def apply_augmentation(self, image, bboxes, category_ids):
        """
        Apply augmentation to the image and bounding boxes.

        Args:
            image (numpy.ndarray): The input image as a NumPy array.
            bboxes (List[List[float]]): List of bounding boxes [x_min, y_min, x_max, y_max].
            category_ids (List[int]): List of category IDs corresponding to each bounding box.

        Returns:
            Dict[str, Any]: Dictionary containing the augmented image, bounding boxes, and category IDs.
        """
        # Set random seed for reproducibility if seed is provided
        if self.seed is not None:
            random.seed(self.seed)

        augmented = self.transform(
            image=image,
            bboxes=bboxes,
            category_ids=category_ids
        )

        return {
            'image': augmented['image'],
            'bboxes': augmented['bboxes'],
            'category_ids': augmented['category_ids']
        }
