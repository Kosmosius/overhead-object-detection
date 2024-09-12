# tests/unit/src/data/test_augmentation.py

import pytest
from src.data.augmentation import DataAugmentor
from PIL import Image
import torch

@pytest.fixture
def sample_image():
    """Fixture for a sample image in PIL format."""
    return Image.new('RGB', (224, 224))

@pytest.fixture
def sample_tensor():
    """Fixture for a sample image in tensor format."""
    return torch.rand((3, 224, 224))

def test_geometric_transform_applied(sample_image):
    augmentor = DataAugmentor(apply_geometric=True, apply_photometric=False, seed=42)
    augmented_image = augmentor.apply_augmentation(sample_image)

    assert augmented_image.size == (224, 224)  # Ensure the image dimensions remain the same

def test_photometric_transform_applied(sample_image):
    augmentor = DataAugmentor(apply_geometric=False, apply_photometric=True, seed=42)
    augmented_image = augmentor.apply_augmentation(sample_image)

    assert isinstance(augmented_image, Image.Image)  # Ensure it's still a PIL image

def test_tensor_augmentation(sample_tensor):
    augmentor = DataAugmentor(apply_geometric=False, apply_photometric=True, seed=42)
    augmented_image = augmentor.apply_augmentation(sample_tensor)

    assert augmented_image.shape == sample_tensor.shape  # Ensure shape consistency with tensors

def test_combined_augmentation(sample_image):
    augmentor = DataAugmentor(apply_geometric=True, apply_photometric=True, seed=42)
    augmented_image = augmentor.apply_augmentation(sample_image)

    assert augmented_image.size == (224, 224)  # Ensure size stays the same with combined augmentation
