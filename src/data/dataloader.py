# src/data/dataloader.py

import os
import logging
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
import cv2
import torch
from src.data.augmentation import DataAugmentor
from typing import Callable, Optional, Tuple, List, Dict
from albumentations.pytorch import ToTensorV2

class CocoDataset(Dataset):
    """
    Custom Dataset class for the COCO dataset.
    Returns images and annotations in a format compatible with HuggingFace Transformers,
    and applies data augmentation if provided.
    """

    def __init__(
        self,
        img_dir: str,
        ann_file: str,
        transforms: Optional[DataAugmentor] = None,
        feature_extractor: Optional[Callable] = None,
    ):
        """
        Initialize the CocoDataset.

        Args:
            img_dir (str): Directory with all the images.
            ann_file (str): Path to the annotation file.
            transforms (DataAugmentor, optional): DataAugmentor instance for applying augmentations.
            feature_extractor (Callable, optional): Feature extractor to preprocess images.
        """
        self.img_dir = img_dir
        self.ann_file = ann_file  # **Added this line**
        self.coco = COCO(ann_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transforms = transforms
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        image_id = self.image_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        # Load image
        img_info = self.coco.imgs[image_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get bounding boxes and category IDs
        bboxes = [ann['bbox'] for ann in anns]
        category_ids = [ann['category_id'] for ann in anns]

        # Convert bounding boxes to [x_min, y_min, x_max, y_max] format
        bboxes = self._convert_bbox_format(bboxes)

        # Apply augmentations if provided
        if self.transforms:
            # **Changed to use keyword arguments and pass image as numpy array**
            augmented = self.transforms.apply_augmentation(
                image=image,
                bboxes=bboxes,
                category_ids=category_ids
            )
            image = augmented['image']
            bboxes = augmented['bboxes']
            category_ids = augmented['category_ids']
        else:
            # Convert image to tensor
            image = ToTensorV2()(image=image)['image']

        # **Added feature extractor invocation with images as list**
        if self.feature_extractor is not None:
            # Pass image as a list (expected by feature_extractor)
            image = self.feature_extractor(images=[image], return_tensors="pt")['pixel_values'].squeeze()
        else:
            # Convert image to tensor and permute dimensions
            image = torch.tensor(image).permute(2, 0, 1)

        # Prepare target
        target = {}
        target['boxes'] = torch.tensor(bboxes, dtype=torch.float32)
        target['labels'] = torch.tensor(category_ids, dtype=torch.int64)
        target['image_id'] = torch.tensor([image_id])

        return image, target

    def __len__(self) -> int:
        return len(self.image_ids)

    def _convert_bbox_format(self, bboxes: List[List[float]]) -> List[List[float]]:
        """
        Convert bounding boxes from COCO format [x, y, width, height]
        to [x_min, y_min, x_max, y_max] format.

        Args:
            bboxes (List[List[float]]): List of bounding boxes in COCO format.

        Returns:
            List[List[float]]: List of bounding boxes in [x_min, y_min, x_max, y_max] format.
        """
        converted_bboxes = []
        for bbox in bboxes:
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = bbox[0] + bbox[2]
            y_max = bbox[1] + bbox[3]
            converted_bboxes.append([x_min, y_min, x_max, y_max])
        return converted_bboxes


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[List[torch.Tensor], List[Dict]]:
    """
    Custom collate function to handle batching of images and annotations.
    Ensures that each batch is compatible with HuggingFace models.

    Args:
        batch (List[Tuple[torch.Tensor, Dict]]): List of tuples containing images and targets.

    Returns:
        Tuple[List[torch.Tensor], List[Dict]]: Batched images and targets.
    """
    images, targets = zip(*batch)
    images = list(images)
    targets = list(targets)
    return images, targets


def validate_data_paths(data_dir: str, ann_file: str, img_dir: str) -> None:
    """
    Validates that the required data paths exist.

    Args:
        data_dir (str): Path to the dataset directory.
        ann_file (str): Path to the annotation file.
        img_dir (str): Path to the images directory.

    Raises:
        FileNotFoundError: If any of the data paths do not exist.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory '{data_dir}' not found.")
    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"Annotation file '{ann_file}' not found.")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory '{img_dir}' not found.")
    logging.info(f"Validated data paths for '{img_dir}' and '{ann_file}'.")


def get_dataset(
    dataset_type: str,
    data_dir: str,
    mode: str,
    transforms: Optional[DataAugmentor] = None,
    feature_extractor: Optional[Callable] = None,
    skip_empty_check: bool = False  # **Added this parameter**
) -> Dataset:
    """
    Returns the appropriate dataset based on the dataset type.

    Args:
        dataset_type (str): The type of dataset (e.g., 'coco').
        data_dir (str): Path to the dataset directory.
        mode (str): Mode of the dataset ('train' or 'val').
        transforms (DataAugmentor, optional): DataAugmentor instance for applying augmentations.
        feature_extractor (Callable, optional): Feature extractor to preprocess images.
        skip_empty_check (bool): Whether to skip checking if the dataset is empty.

    Returns:
        Dataset: A dataset object.
    """
    if dataset_type == 'coco':
        ann_file = os.path.join(data_dir, f'annotations/instances_{mode}2017.json')
        img_dir = os.path.join(data_dir, f'{mode}2017')

        # Validate paths
        validate_data_paths(data_dir, ann_file, img_dir)

        dataset = CocoDataset(
            img_dir=img_dir,
            ann_file=ann_file,
            transforms=transforms,
            feature_extractor=feature_extractor
        )

        # **Added check for empty dataset**
        if not skip_empty_check and len(dataset) == 0:
            raise ValueError("Dataset is empty.")

        return dataset
    else:
        raise ValueError(f"Dataset type '{dataset_type}' is not supported.")


def get_dataloader(
    data_dir: str,
    batch_size: int,
    mode: str = 'train',
    feature_extractor: Optional[Callable] = None,
    dataset_type: str = 'coco',
    num_workers: int = 4,
    pin_memory: bool = True,
    skip_empty_check: bool = False  # **Added this parameter**
) -> DataLoader:
    """
    Returns a DataLoader for the specified dataset.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Number of samples per batch.
        mode (str): Mode of the dataset, 'train' or 'val'.
        feature_extractor (Callable, optional): Feature extractor to preprocess images.
        dataset_type (str): The type of dataset (default is 'coco').
        num_workers (int): Number of worker processes for data loading.
        pin_memory (bool): Whether to pin memory in data loader.
        skip_empty_check (bool): Whether to skip checking if the dataset is empty.

    Returns:
        DataLoader: DataLoader for the specified dataset.
    """
    assert mode in ['train', 'val'], "Mode should be either 'train' or 'val'."

    # Initialize DataAugmentor for training mode
    transforms = None
    if mode == 'train':
        transforms = DataAugmentor(apply_geometric=True, apply_photometric=True, seed=42)

    # Get dataset based on the type
    dataset = get_dataset(
        dataset_type=dataset_type,
        data_dir=data_dir,
        mode=mode,
        transforms=transforms,
        feature_extractor=feature_extractor,
        skip_empty_check=skip_empty_check  # **Pass the parameter**
    )

    if len(dataset) == 0:
        raise ValueError("Cannot create DataLoader with an empty dataset.")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if mode == 'train' else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
