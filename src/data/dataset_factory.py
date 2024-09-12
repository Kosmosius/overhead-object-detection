# src/data/dataset_factory.py

import logging
from src.data.dataloader import get_dataloader
from transformers import AutoFeatureExtractor

class DatasetNotSupportedError(Exception):
    """Custom exception for unsupported datasets."""
    pass

class BaseDatasetFactory:
    """
    Base class for dataset factories. 
    This can be extended for additional datasets beyond COCO.
    """
    def __init__(self, data_dir, batch_size, mode='train', feature_extractor_name='facebook/detr-resnet-50'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.mode = mode
        self.feature_extractor_name = feature_extractor_name
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_name)
    
    def get_dataloader(self):
        """Abstract method to be implemented in subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def log_dataset_info(self):
        """Log basic dataset information."""
        logging.info(f"Dataset: {self.__class__.__name__}")
        logging.info(f"Data Directory: {self.data_dir}")
        logging.info(f"Batch Size: {self.batch_size}")
        logging.info(f"Mode: {self.mode}")
        logging.info(f"Feature Extractor: {self.feature_extractor_name}")


class CocoDatasetFactory(BaseDatasetFactory):
    """
    Dataset factory for COCO dataset.
    Extends the BaseDatasetFactory and implements the get_dataloader method for COCO.
    """
    def get_dataloader(self):
        self.log_dataset_info()  # Log dataset information for debugging
        
        return get_dataloader(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            mode=self.mode,
            feature_extractor=self.feature_extractor,
            dataset_type='coco'
        )


class DatasetFactory:
    """
    Factory class to select the correct dataset factory based on the dataset type.
    """
    def __init__(self, data_dir, batch_size, dataset_type='coco', mode='train', feature_extractor_name='facebook/detr-resnet-50'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dataset_type = dataset_type
        self.mode = mode
        self.feature_extractor_name = feature_extractor_name
        self.dataset_factory = self._select_factory()

    def _select_factory(self):
        """
        Select the appropriate factory based on the dataset type.
        """
        if self.dataset_type == 'coco':
            return CocoDatasetFactory(
                data_dir=self.data_dir,
                batch_size=self.batch_size,
                mode=self.mode,
                feature_extractor_name=self.feature_extractor_name
            )
        else:
            raise DatasetNotSupportedError(f"Dataset type '{self.dataset_type}' is not supported.")

    def get_dataloader(self):
        """
        Get the dataloader by calling the appropriate factory.
        """
        return self.dataset_factory.get_dataloader()

