# src/data/dataloader.py

# src/data/dataloader.py

import os
from torch.utils.data import DataLoader
from transformers import DataCollator
from torchvision.datasets import CocoDetection
import logging

class BaseDataset:
    """
    Abstract base dataset class for supporting different datasets.
    All datasets must override the __getitem__ and __len__ methods.
    """
    def __getitem__(self, idx):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def __len__(self):
        raise NotImplementedError("This method should be overridden by subclasses.")


class CocoDataset(CocoDetection, BaseDataset):
    """
    Custom CocoDetection class to return images and annotations in a format
    compatible with HuggingFace Transformers.
    """
    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        annotations = {"boxes": [], "labels": []}

        for obj in target:
            annotations["boxes"].append(obj["bbox"])
            annotations["labels"].append(obj["category_id"])
        
        return image, annotations


def collate_fn(batch):
    """
    Custom collate function to handle batching of images and annotations.
    Ensures that each batch is compatible with HuggingFace models.
    """
    images, annotations = zip(*batch)
    return list(images), list(annotations)


def validate_data_paths(data_dir, ann_file, img_dir):
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


def get_dataset(dataset_type, data_dir, mode, feature_extractor=None):
    """
    Returns the appropriate dataset based on the dataset type.

    Args:
        dataset_type (str): The type of dataset (e.g., 'coco').
        data_dir (str): Path to the dataset directory.
        mode (str): Mode of the dataset ('train' or 'val').
        feature_extractor (optional): HuggingFace FeatureExtractor to preprocess images.

    Returns:
        dataset: A dataset object.
    """
    if dataset_type == 'coco':
        ann_file = os.path.join(data_dir, f'annotations/instances_{mode}2017.json')
        img_dir = os.path.join(data_dir, f'{mode}2017')

        # Validate paths
        validate_data_paths(data_dir, ann_file, img_dir)

        return CocoDataset(
            root=img_dir,
            annFile=ann_file,
            transform=lambda x: feature_extractor(images=x, return_tensors="pt").pixel_values[0] if feature_extractor else x
        )
    else:
        raise ValueError(f"Dataset type '{dataset_type}' is not supported.")


def get_dataloader(data_dir, batch_size, mode='train', feature_extractor=None, dataset_type='coco'):
    """
    Returns a DataLoader for the specified dataset.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Number of samples per batch.
        mode (str): Mode of the dataset, 'train' or 'val'.
        feature_extractor (optional): HuggingFace FeatureExtractor to preprocess images.
        dataset_type (str): The type of dataset (default is 'coco').

    Returns:
        DataLoader: DataLoader for the specified dataset.
    """
    assert mode in ['train', 'val'], "Mode should be either 'train' or 'val'."

    # Get dataset based on the type
    dataset = get_dataset(
        dataset_type=dataset_type,
        data_dir=data_dir,
        mode=mode,
        feature_extractor=feature_extractor
    )

    # Data collator for handling batches
    data_collator = DataCollator(feature_extractor=feature_extractor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if mode == 'train' else False,
        collate_fn=collate_fn
    )
