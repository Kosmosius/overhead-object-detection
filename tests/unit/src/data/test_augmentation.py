# tests/unit/src/data/test_augmentation.py

import pytest
from unittest.mock import patch, MagicMock
from src.data.augmentation import DataAugmentor
import numpy as np
import torch

@pytest.fixture
def sample_image():
    """Fixture for a sample image as a NumPy array."""
    # Create a dummy image (224x224 RGB)
    return np.zeros((224, 224, 3), dtype=np.uint8)

@pytest.fixture
def sample_bboxes():
    """Fixture for sample bounding boxes."""
    # Two bounding boxes: [x_min, y_min, x_max, y_max]
    return [
        [50, 50, 150, 150],
        [30, 30, 100, 100]
    ]

@pytest.fixture
def sample_category_ids():
    """Fixture for sample category IDs."""
    return [1, 2]

@pytest.mark.parametrize(
    "apply_geometric, apply_photometric",
    [
        (True, False),
        (False, True),
        (True, True),
        (False, False)
    ]
)
def test_apply_augmentation_transformations(
    sample_image,
    sample_bboxes,
    sample_category_ids,
    apply_geometric,
    apply_photometric
):
    """
    Test that the apply_augmentation method correctly applies geometric and/or photometric transformations.
    """
    # Initialize DataAugmentor with given configuration and a fixed seed
    seed = 42
    augmentor = DataAugmentor(
        apply_geometric=apply_geometric,
        apply_photometric=apply_photometric,
        seed=seed
    )

    with patch('albumentations.Compose') as mock_compose:
        # Mock the transformation pipeline
        mock_transform = MagicMock()
        # Define the return value of the transform pipeline
        augmented_image = sample_image.copy()
        augmented_bboxes = [bbox.copy() for bbox in sample_bboxes]
        augmented_category_ids = sample_category_ids.copy()

        # Correctly set the return value on the mock_transform
        mock_transform.return_value = {
            'image': augmented_image,
            'bboxes': augmented_bboxes,
            'category_ids': augmented_category_ids
        }
        mock_compose.return_value = mock_transform

        # Apply augmentation
        result = augmentor.apply_augmentation(
            image=sample_image,
            bboxes=sample_bboxes,
            category_ids=sample_category_ids
        )

        # Assertions
        assert 'image' in result, "Result should contain 'image'."
        assert 'bboxes' in result, "Result should contain 'bboxes'."
        assert 'category_ids' in result, "Result should contain 'category_ids'."

        # Verify image dimensions remain the same (C, H, W)
        expected_shape = (3, sample_image.shape[0], sample_image.shape[1])
        assert result['image'].shape == expected_shape, "Augmented image dimensions should match the expected shape (C, H, W)."

        # Verify bounding boxes are lists of the same length
        assert len(result['bboxes']) == len(sample_bboxes), "Number of bounding boxes should remain the same."
        for orig_bbox, aug_bbox in zip(sample_bboxes, result['bboxes']):
            assert aug_bbox == orig_bbox, "Bounding boxes should be correctly transformed."

        # Verify category IDs remain unchanged
        assert result['category_ids'] == sample_category_ids, "Category IDs should remain unchanged."

        # Ensure the transformation pipeline was called with correct parameters
        mock_compose.assert_called_once()
        mock_transform.assert_called_once_with(
            image=sample_image,
            bboxes=sample_bboxes,
            category_ids=sample_category_ids
        )

def test_apply_augmentation_reproducibility(sample_image, sample_bboxes, sample_category_ids):
    """
    Test that applying augmentation with the same seed results in reproducible outputs.
    """
    seed = 123
    augmentor1 = DataAugmentor(apply_geometric=True, apply_photometric=True, seed=seed)
    augmentor2 = DataAugmentor(apply_geometric=True, apply_photometric=True, seed=seed)

    with patch('albumentations.Compose') as mock_compose1, \
         patch('albumentations.Compose') as mock_compose2:

        # Setup first augmentor's transform
        mock_transform1 = MagicMock()
        mock_transform1.return_value = {
            'image': sample_image.copy(),
            'bboxes': [bbox.copy() for bbox in sample_bboxes],
            'category_ids': sample_category_ids.copy()
        }
        mock_compose1.return_value = mock_transform1

        # Setup second augmentor's transform
        mock_transform2 = MagicMock()
        mock_transform2.return_value = {
            'image': sample_image.copy(),
            'bboxes': [bbox.copy() for bbox in sample_bboxes],
            'category_ids': sample_category_ids.copy()
        }
        mock_compose2.return_value = mock_transform2

        # Apply augmentations
        result1 = augmentor1.apply_augmentation(
            image=sample_image,
            bboxes=sample_bboxes,
            category_ids=sample_category_ids
        )
        result2 = augmentor2.apply_augmentation(
            image=sample_image,
            bboxes=sample_bboxes,
            category_ids=sample_category_ids
        )

        # Assertions to ensure reproducibility
        assert torch.equal(result1['image'], result2['image']), "Images should be identical with the same seed."
        assert result1['bboxes'] == result2['bboxes'], "Bounding boxes should be identical with the same seed."
        assert result1['category_ids'] == result2['category_ids'], "Category IDs should be identical with the same seed."

def test_apply_augmentation_correctness(sample_image, sample_bboxes, sample_category_ids):
    """
    Test the correctness of the augmented outputs by using real Albumentations transforms.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    # Define a real DataAugmentor without mocking
    augmentor = DataAugmentor(apply_geometric=True, apply_photometric=True, seed=42)

    # Apply augmentation
    result = augmentor.apply_augmentation(
        image=sample_image,
        bboxes=sample_bboxes,
        category_ids=sample_category_ids
    )

    # Assertions
    assert 'image' in result, "Result should contain 'image'."
    assert 'bboxes' in result, "Result should contain 'bboxes'."
    assert 'category_ids' in result, "Result should contain 'category_ids'."

    # Check image type and shape
    assert isinstance(result['image'], torch.Tensor), "Augmented image should be a torch.Tensor."
    assert result['image'].shape == (3, 224, 224), "Augmented image dimensions should match the original."

    # Check bounding boxes
    assert isinstance(result['bboxes'], list), "Bounding boxes should be a list."
    assert len(result['bboxes']) == len(sample_bboxes), "Number of bounding boxes should remain the same."
    for bbox in result['bboxes']:
        assert len(bbox) == 4, "Each bounding box should have four coordinates."

    # Check category IDs
    assert result['category_ids'] == sample_category_ids, "Category IDs should remain unchanged."

def test_apply_augmentation_no_transform(sample_image, sample_bboxes, sample_category_ids):
    """
    Test that no transformations are applied when both geometric and photometric are disabled.
    """
    augmentor = DataAugmentor(apply_geometric=False, apply_photometric=False, seed=42)

    with patch('albumentations.Compose') as mock_compose:
        # Mock the transformation pipeline to return the input as-is
        mock_transform = MagicMock()
        mock_transform.return_value = {
            'image': sample_image.copy(),
            'bboxes': [bbox.copy() for bbox in sample_bboxes],
            'category_ids': sample_category_ids.copy()
        }
        mock_compose.return_value = mock_transform

        # Apply augmentation
        result = augmentor.apply_augmentation(
            image=sample_image,
            bboxes=sample_bboxes,
            category_ids=sample_category_ids
        )

        # Assertions
        expected_image = torch.from_numpy(sample_image).permute(2, 0, 1)
        assert torch.equal(result['image'], expected_image), "Image should remain unchanged when no transforms are applied."
        assert result['bboxes'] == sample_bboxes, "Bounding boxes should remain unchanged when no transforms are applied."
        assert result['category_ids'] == sample_category_ids, "Category IDs should remain unchanged when no transforms are applied."

        # Ensure the transformation pipeline was called correctly
        mock_compose.assert_called_once()
        mock_transform.assert_called_once_with(
            image=sample_image,
            bboxes=sample_bboxes,
            category_ids=sample_category_ids
        )

def test_apply_augmentation_empty_bboxes(sample_image, sample_category_ids):
    """
    Test that the augmentation method handles empty bounding boxes gracefully.
    """
    augmentor = DataAugmentor(apply_geometric=True, apply_photometric=True, seed=42)
    empty_bboxes = []
    empty_category_ids = []

    with patch('albumentations.Compose') as mock_compose:
        # Mock the transformation pipeline to handle empty bboxes
        mock_transform = MagicMock()
        mock_transform.return_value = {
            'image': sample_image.copy(),
            'bboxes': empty_bboxes,
            'category_ids': empty_category_ids
        }
        mock_compose.return_value = mock_transform

        # Apply augmentation
        result = augmentor.apply_augmentation(
            image=sample_image,
            bboxes=empty_bboxes,
            category_ids=empty_category_ids
        )

        # Assertions
        expected_image = torch.from_numpy(sample_image).permute(2, 0, 1)
        assert torch.equal(result['image'], expected_image), "Image should remain unchanged."
        assert result['bboxes'] == empty_bboxes, "Bounding boxes should remain empty."
        assert result['category_ids'] == empty_category_ids, "Category IDs should remain empty."

def test_apply_augmentation_invalid_bbox_format(sample_image, sample_bboxes, sample_category_ids):
    """
    Test that the augmentation method raises an error when bounding boxes are in an invalid format.
    """
    augmentor = DataAugmentor(apply_geometric=True, apply_photometric=True, seed=42)
    invalid_bboxes = ["invalid_bbox_format"]

    with pytest.raises(ValueError):
        augmentor.apply_augmentation(
            image=sample_image,
            bboxes=invalid_bboxes,
            category_ids=sample_category_ids
        )
