# src/data/dataset_factory.py

import logging
from typing import Optional, Dict, Any

from datasets import load_dataset, DatasetDict
from transformers import AutoImageProcessor
from torch.utils.data import DataLoader
from src.data.augmentation import DataAugmentation
import torch

logger = logging.getLogger(__name__)


class DatasetFactory:
    """
    Factory class to load datasets and create DataLoaders.
    Supports multiple datasets and integrates with HuggingFace standards.
    """

    def __init__(
        self,
        model_name_or_path: str,
        dataset_name_or_path: str,
        train_split: str = "train",
        val_split: str = "validation",
        test_split: Optional[str] = None,
        cache_dir: Optional[str] = None,
        augmentation: Optional[DataAugmentation] = None,
        bbox_format: str = "coco",
        remove_columns: Optional[list] = None,
    ):
        """
        Initializes the DatasetFactory.

        Args:
            model_name_or_path (str): Path to the pretrained model or model identifier from HuggingFace.
            dataset_name_or_path (str): Name or path of the dataset to load.
            train_split (str): Name of the training split.
            val_split (str): Name of the validation split.
            test_split (str, optional): Name of the test split.
            cache_dir (str, optional): Directory to cache the dataset.
            augmentation (DataAugmentation, optional): Data augmentation to apply.
            bbox_format (str): Format of the bounding boxes ('coco' or 'pascal_voc').
            remove_columns (list, optional): List of columns to remove from the dataset.
        """
        self.model_name_or_path = model_name_or_path
        self.dataset_name_or_path = dataset_name_or_path
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.cache_dir = cache_dir
        self.augmentation = augmentation
        self.bbox_format = bbox_format
        self.remove_columns = remove_columns

        # Load the processor
        self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        logger.info(f"Loaded processor for model '{model_name_or_path}'.")

        # Load and preprocess datasets
        self.datasets = self._load_datasets()

    def _load_datasets(self) -> DatasetDict:
        """
        Loads and preprocesses the datasets.

        Returns:
            DatasetDict: A dictionary containing the datasets.
        """
        splits = {"train": self.train_split, "validation": self.val_split}
        if self.test_split:
            splits["test"] = self.test_split

        # Load the datasets
        datasets = load_dataset(
            path=self.dataset_name_or_path,
            split=splits,
            cache_dir=self.cache_dir,
        )
        logger.info(f"Loaded dataset '{self.dataset_name_or_path}' with splits {list(datasets.keys())}.")

        def preprocess_function(examples):
            images = [image.convert("RGB") for image in examples["image"]]
            annotations = examples.get("annotations", None)

            # Apply augmentation if provided
            if self.augmentation:
                augmented = self.augmentation(
                    {
                        "image": images,
                        "annotations": annotations,
                    }
                )
                images = augmented["image"]
                annotations = augmented["annotations"]

            # Use the processor to prepare the inputs
            inputs = self.processor(
                images=images,
                annotations=annotations,
                return_tensors="pt",
            )

            # Prepare the labels
            labels = inputs.pop("labels")
            inputs["labels"] = [{k: v for k, v in label.items()} for label in labels]

            return inputs

        # Apply the preprocessing function to the datasets
        for split in datasets.keys():
            datasets[split] = datasets[split].map(
                preprocess_function,
                batched=True,
                remove_columns=self.remove_columns or datasets[split].column_names,
                num_proc=4,  # Utilize multiprocessing for faster preprocessing
            )
            logger.info(f"Preprocessed '{split}' split.")

        return datasets

    def get_dataloaders(
        self,
        batch_size: int,
        num_workers: int = 4,
    ) -> Dict[str, DataLoader]:
        """
        Creates DataLoaders for the datasets.

        Args:
            batch_size (int): Batch size for data loading.
            num_workers (int, optional): Number of worker processes.

        Returns:
            Dict[str, DataLoader]: A dictionary containing the DataLoaders.
        """
        dataloaders = {}
        for split, dataset in self.datasets.items():
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                collate_fn=self.collate_fn,
            )
            logger.info(f"Created DataLoader for '{split}' split with batch size {batch_size}.")

        return dataloaders

    @staticmethod
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
