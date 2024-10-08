# src/data/augmentation.py

import numpy as np
import logging
from typing import Optional, Dict, Any, List

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from transformers import AutoImageProcessor
from datasets import Dataset

logger = logging.getLogger(__name__)


class DataAugmentation:
    """
    Class to apply data augmentation for object detection tasks using Albumentations.
    Integrates with HuggingFace's AutoImageProcessor and datasets library, optimized for rare object detection.
    """

    def __init__(
        self,
        model_name_or_path: str,
        bbox_format: str = 'coco',  # 'coco' or 'pascal_voc'
        min_visibility: float = 0.5,
        augmentation_transforms: Optional[A.Compose] = None,
    ):
        """
        Initializes the DataAugmentation class.

        Args:
            model_name_or_path (str): Path to the pretrained model or model identifier from HuggingFace.
            bbox_format (str): Format of the bounding boxes ('coco' or 'pascal_voc').
            min_visibility (float): Minimum visibility of bounding boxes after augmentation.
            augmentation_transforms (A.Compose, optional): Custom augmentation transforms to apply.
                If None, default transforms optimized for rare object detection will be used.
        """
        # Load the processor corresponding to the model
        self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        self.bbox_format = bbox_format
        self.min_visibility = min_visibility

        # Define default augmentation transforms if none provided
        if augmentation_transforms is None:
            self.augmentation_transforms = self.default_transforms()
        else:
            self.augmentation_transforms = augmentation_transforms

        logger.info(f"DataAugmentation initialized for model '{model_name_or_path}' with bbox format '{bbox_format}'.")

    def default_transforms(self) -> A.Compose:
        """
        Defines the default augmentation transforms using Albumentations.
        These transforms are optimized for rare object detection by focusing on augmentations
        that preserve rare object instances and enhance their visibility.

        Returns:
            A.Compose: Composed Albumentations transforms.
        """
        return A.Compose(
            [
                # Geometric Transformations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.RandomRotate90(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=0,
                ),
                A.RandomResizedCrop(
                    height=512,
                    width=512,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    p=0.5,
                ),
                # Photometric Transformations
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5,
                ),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.MotionBlur(blur_limit=3, p=0.2),
                A.ToGray(p=0.1),
                # Noise Injection
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                # Cutout
                A.Cutout(
                    num_holes=8,
                    max_h_size=32,
                    max_w_size=32,
                    fill_value=0,
                    p=0.5,
                ),
                # Convert image to tensor
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format=self.bbox_format,
                label_fields=['category_ids'],
                min_visibility=self.min_visibility,
                filter_lost_elements=True,
            ),
        )

    def __call__(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies augmentation to a batch of examples.

        Args:
            examples (Dict[str, Any]): A batch of examples containing 'image' and 'annotations'.

        Returns:
            Dict[str, Any]: The augmented batch of examples.
        """
        images = examples['image']
        annotations = examples.get('annotations', [])

        augmented_images = []
        augmented_annotations = []

        for idx, (image, annotation) in enumerate(zip(images, annotations)):
            image_np = np.array(image.convert("RGB"))

            # Extract bounding boxes and category IDs
            bboxes = [obj['bbox'] for obj in annotation['objects']]
            category_ids = [obj['category_id'] for obj in annotation['objects']]

            # Handle cases where there are no bounding boxes
            if not bboxes:
                logger.warning(f"No bounding boxes found for image at index {idx}. Skipping augmentation.")
                augmented_images.append(image_np)
                augmented_annotations.append({'objects': []})
                continue

            # Apply augmentation
            try:
                augmented = self.augmentation_transforms(
                    image=image_np,
                    bboxes=bboxes,
                    category_ids=category_ids,
                )
            except Exception as e:
                logger.error(f"Augmentation failed for image at index {idx}: {e}")
                augmented_images.append(image_np)
                augmented_annotations.append(annotation)
                continue

            # Check if any bounding boxes remain after augmentation
            if not augmented['bboxes']:
                logger.warning(f"All bounding boxes are lost after augmentation for image at index {idx}.")
                # Optionally, skip this image or keep the original
                augmented_images.append(image_np)
                augmented_annotations.append(annotation)
                continue

            augmented_images.append(augmented['image'])

            # Reconstruct annotations after augmentation
            augmented_objects = [
                {'bbox': bbox, 'category_id': category_id}
                for bbox, category_id in zip(augmented['bboxes'], augmented['category_ids'])
            ]
            augmented_annotations.append({'objects': augmented_objects})

        # Use the processor to prepare the inputs for the model
        inputs = self.processor(
            images=augmented_images,
            annotations=augmented_annotations,
            return_tensors="pt",
        )

        # Update examples with processed inputs
        examples.update(inputs)

        return examples
