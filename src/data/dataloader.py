# src/data/dataloader.py

from torch.utils.data import DataLoader
from transformers import DataCollator
from torchvision.datasets import CocoDetection

class BaseDataset:
    """
    Abstract base dataset class for supporting different datasets.
    """
    def __getitem__(self, idx):
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
    This ensures each batch is compatible with HuggingFace models.
    """
    images, annotations = zip(*batch)
    return list(images), list(annotations)

def get_dataloader(data_dir, batch_size, mode='train', feature_extractor=None, dataset_type='coco'):
    """
    Returns a DataLoader for the specified dataset.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Number of samples per batch.
        mode (str): Mode of the dataset, 'train' or 'val'.
        feature_extractor: HuggingFace FeatureExtractor to preprocess images.
        dataset_type (str): The type of dataset (default is 'coco').

    Returns:
        DataLoader: DataLoader for the specified dataset.
    """
    assert mode in ['train', 'val'], "Mode should be either 'train' or 'val'."

    if dataset_type == 'coco':
        ann_file = f'annotations/instances_{mode}2017.json'
        img_dir = f'{mode}2017'
        dataset = CocoDataset(
            root=f"{data_dir}/{img_dir}",
            annFile=f"{data_dir}/{ann_file}",
            transform=lambda x: feature_extractor(images=x, return_tensors="pt").pixel_values[0] if feature_extractor else x
        )
    else:
        raise ValueError(f"Dataset type '{dataset_type}' is not supported.")

    data_collator = DataCollator(feature_extractor=feature_extractor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if mode == 'train' else False,
        collate_fn=collate_fn
    )
