# src/data/dataloader.py

from torch.utils.data import DataLoader
from transformers import DetrFeatureExtractor, DataCollator
from torchvision.datasets import CocoDetection

class CocoDataset(CocoDetection):
    """
    Custom CocoDetection class to return images and annotations in a format
    compatible with HuggingFace Transformers.
    """

    def __getitem__(self, idx):
        """
        Overrides the default __getitem__ method to return image and target
        as a dictionary suitable for HuggingFace Transformers.
        """
        image, target = super().__getitem__(idx)
        # Extract the bounding boxes and labels
        annotations = {"boxes": [], "labels": []}
        for obj in target:
            annotations["boxes"].append(obj["bbox"])  # COCO format [x_min, y_min, width, height]
            annotations["labels"].append(obj["category_id"])  # COCO class category id
        return image, annotations


def collate_fn(batch):
    """
    Custom collate function to handle batching of images and annotations.
    This ensures each batch is compatible with HuggingFace models.
    """
    images, annotations = zip(*batch)
    return list(images), list(annotations)


def get_dataloader(data_dir, batch_size, mode='train', feature_extractor=None):
    """
    Returns a DataLoader for the COCO dataset.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Number of samples per batch.
        mode (str): Mode of the dataset, 'train' or 'val'.
        feature_extractor (DetrFeatureExtractor, optional): Feature extractor for preprocessing images.

    Returns:
        DataLoader: DataLoader for the specified dataset.
    """
    assert mode in ['train', 'val'], "Mode should be either 'train' or 'val'."

    ann_file = f'annotations/instances_{mode}2017.json'
    img_dir = f'{mode}2017'

    # Dynamically determine dataset class (supporting COCO and others)
    if dataset_type == 'coco':
        dataset = CocoDataset(
            root=f"{data_dir}/{img_dir}",
            annFile=f"{data_dir}/{ann_file}",
            transform=lambda x: feature_extractor(images=x, return_tensors="pt").pixel_values[0] if feature_extractor else x
        )
    # Extend for future datasets...

    # Use the DataCollator to handle padding for batch inputs
    data_collator = DataCollator(feature_extractor=feature_extractor)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if mode == 'train' else False,
        collate_fn=collate_fn
    )

    return dataloader
