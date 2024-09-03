# src/data/dataset_factory.py

from src.data.dataloader import get_dataloader
from transformers import DetrFeatureExtractor

class DatasetFactory:
    def __init__(self, data_dir, batch_size, dataset_type='coco', mode='train'):
        """
        Initialize the DatasetFactory.

        Args:
            data_dir (str): Path to the dataset directory.
            batch_size (int): Number of samples per batch.
            dataset_type (str): Type of dataset ('coco' for now).
            mode (str): Mode of the dataset, 'train' or 'val'.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dataset_type = dataset_type
        self.mode = mode
        self.feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

    def get_dataloader(self):
        """
        Get the DataLoader for the specified dataset.
        """
        if self.dataset_type == 'coco':
            return get_dataloader(
                data_dir=self.data_dir,
                batch_size=self.batch_size,
                feature_extractor=self.feature_extractor,
                mode=self.mode
            )
        else:
            raise ValueError(f"Dataset type {self.dataset_type} not supported.")
