# src/data/dataloader.py

from torch.utils.data import DataLoader
from transformers import DetrFeatureExtractor
from torchvision.datasets import CocoDetection

def collate_fn(batch):
    """
    Custom collate function to handle batches with varying sizes.
    """
    return tuple(zip(*batch))

def get_dataloader(data_dir, batch_size, feature_extractor, mode='train'):
    """
    Returns a DataLoader for the COCO dataset.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Number of samples per batch.
        feature_extractor (DetrFeatureExtractor): Feature extractor for preprocessing images.
        mode (str): Mode of the dataset, 'train' or 'val'.

    Returns:
        DataLoader: DataLoader for the specified dataset.
    """
    assert mode in ['train', 'val'], "Mode should be either 'train' or 'val'."

    ann_file = f'annotations/instances_{mode}2017.json'
    img_dir = f'{mode}2017'

    dataset = CocoDetection(
        root=f"{data_dir}/{img_dir}",
        annFile=f"{data_dir}/{ann_file}",
        transform=lambda x: feature_extractor(images=x, return_tensors="pt").pixel_values[0]
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if mode == 'train' else False,
        collate_fn=collate_fn
    )

    return dataloader
