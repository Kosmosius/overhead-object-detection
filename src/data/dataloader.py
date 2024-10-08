# src/data/dataloader.py

import logging
from typing import Optional, Dict, Any

from datasets import load_dataset, DatasetDict
from transformers import AutoImageProcessor
from torch.utils.data import DataLoader
from src.data.augmentation import DataAugmentation
import torch

logger = logging.getLogger(__name__)


def get_datasets(
    model_name_or_path: str,
    dataset_name: str,
    train_split: str = "train",
    val_split: str = "validation",
    cache_dir: Optional[str] = None,
    augmentation: Optional[DataAugmentation] = None,
    bbox_format: str = "coco",
    remove_columns: Optional[list] = None,
) -> DatasetDict:
    """
    Loads and preprocesses the datasets using HuggingFace's datasets library.

    Args:
        model_name_or_path (str): Path to the pretrained model or model identifier from HuggingFace.
        dataset_name (str): Name of the dataset to load.
        train_split (str): Name of the training split.
        val_split (str): Name of the validation split.
        cache_dir (str, optional): Directory to cache the dataset.
        augmentation (DataAugmentation, optional): Data augmentation to apply.
        bbox_format (str): Format of the bounding boxes ('coco' or 'pascal_voc').
        remove_columns (list, optional): List of columns to remove from the dataset.

    Returns:
        DatasetDict: A dictionary containing the training and validation datasets.
    """
    # Load the datasets
    datasets = load_dataset(
        dataset_name,
        split={"train": train_split, "validation": val_split},
        cache_dir=cache_dir,
    )

    # Load the processor
    processor = AutoImageProcessor.from_pretrained(model_name_or_path)

    def preprocess_function(examples):
        images = [image.convert("RGB") for image in examples["image"]]
        annotations = examples.get("annotations", None)

        # Apply augmentation if provided
        if augmentation:
            augmented = augmentation(
                {
                    "image": images,
                    "annotations": annotations,
                }
            )
            images = augmented["image"]
            annotations = augmented["annotations"]

        # Use the processor to prepare the inputs
        inputs = processor(
            images=images,
            annotations=annotations,
            return_tensors="pt",
        )

        # Prepare the labels
        labels = inputs.pop("labels")
        inputs["labels"] = [{k: v for k, v in label.items()} for label in labels]

        return inputs

    # Apply the preprocessing function to both training and validation datasets
    for split in datasets.keys():
        datasets[split] = datasets[split].map(
            preprocess_function,
            batched=True,
            remove_columns=remove_columns or datasets[split].column_names,
        )

    return datasets


def collate_fn(batch):
    """
    Custom collate function to handle batching of images and annotations.

    Args:
        batch: A list of examples returned by the dataset.

    Returns:
        Dict[str, Any]: A batch of data ready for the model.
    """
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    labels = [example["labels"] for example in batch]
    return {"pixel_values": pixel_values, "labels": labels}


def get_dataloaders(
    datasets: DatasetDict,
    batch_size: int,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """
    Creates DataLoaders for the training and validation datasets.

    Args:
        datasets (DatasetDict): The datasets to create DataLoaders for.
        batch_size (int): Batch size for data loading.
        num_workers (int, optional): Number of worker processes.

    Returns:
        Dict[str, DataLoader]: A dictionary containing the DataLoaders for training and validation.
    """
    dataloaders = {}
    for split in ["train", "validation"]:
        dataloaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
    return dataloaders
