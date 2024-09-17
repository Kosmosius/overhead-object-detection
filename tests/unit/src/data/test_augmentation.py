# tests/unit/src/data/test_augmentation.py

import pytest
from src.data.augmentation import DataAugmentor
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
    assert result['image'].shape == (3, sample_image.shape[0], sample_image.shape[1]), "Augmented image dimensions should match the expected shape (C, H, W)."

    # Check bounding boxes
    assert isinstance(result['bboxes'], list), "Bounding boxes should be a list."
    assert len(result['bboxes']) == len(sample_bboxes), "Number of bounding boxes should remain the same."
    for orig_bbox, aug_bbox in zip(sample_bboxes, result['bboxes']):
        if apply_geometric:
            # Bounding boxes may change when geometric transformations are applied
            assert aug_bbox != orig_bbox, "Bounding boxes should be transformed when geometric transformations are applied."
        else:
            # Bounding boxes should remain the same when no geometric transformations are applied
            assert aug_bbox == orig_bbox, "Bounding boxes should remain unchanged when no geometric transformations are applied."

    # Check category IDs
    assert result['category_ids'] == sample_category_ids, "Category IDs should remain unchanged."

def test_apply_augmentation_reproducibility(sample_image, sample_bboxes, sample_category_ids):
    """
    Test that applying augmentation with the same seed results in reproducible outputs.
    """
    seed = 123
    augmentor1 = DataAugmentor(apply_geometric=True, apply_photometric=True, seed=seed)
    augmentor2 = DataAugmentor(apply_geometric=True, apply_photometric=True, seed=seed)

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

def test_apply_augmentation_no_transform(sample_image, sample_bboxes, sample_category_ids):
    """
    Test that no transformations are applied when both geometric and photometric are disabled.
    """
    augmentor = DataAugmentor(apply_geometric=False, apply_photometric=False, seed=42)

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

def test_apply_augmentation_empty_bboxes(sample_image):
    """
    Test that the augmentation method handles empty bounding boxes gracefully.
    """
    augmentor = DataAugmentor(apply_geometric=True, apply_photometric=True, seed=42)
    empty_bboxes = []
    empty_category_ids = []

    # Apply augmentation
    result = augmentor.apply_augmentation(
        image=sample_image,
        bboxes=empty_bboxes,
        category_ids=empty_category_ids
    )

    # Assertions
    assert isinstance(result['image'], torch.Tensor), "Augmented image should be a torch.Tensor."
    assert result['bboxes'] == empty_bboxes, "Bounding boxes should remain empty."
    assert result['category_ids'] == empty_category_ids, "Category IDs should remain empty."

def test_apply_augmentation_invalid_bbox_format(sample_image, sample_category_ids):
    """
    Test that the augmentation method raises an error when bounding boxes are in an invalid format.
    """
    augmentor = DataAugmentor(apply_geometric=True, apply_photometric=True, seed=42)
    invalid_bboxes = ["invalid_bbox_format"]

    with pytest.raises(Exception):
        augmentor.apply_augmentation(
            image=sample_image,
            bboxes=invalid_bboxes,
            category_ids=sample_category_ids
        )

def test_apply_augmentation_correctness(sample_image, sample_bboxes, sample_category_ids):
    """
    Test the correctness of the augmented outputs by using real Albumentations transforms.
    """
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
    assert result['image'].shape == (3, sample_image.shape[0], sample_image.shape[1]), "Augmented image dimensions should match the expected shape (C, H, W)."

    # Check bounding boxes
    assert isinstance(result['bboxes'], list), "Bounding boxes should be a list."
    assert len(result['bboxes']) == len(sample_bboxes), "Number of bounding boxes should remain the same."
    for bbox in result['bboxes']:
        assert len(bbox) == 4, "Each bounding box should have four coordinates."

    # Check category IDs
    assert result['category_ids'] == sample_category_ids, "Category IDs should remain unchanged."

def test_apply_augmentation_with_photometric_only(sample_image, sample_bboxes, sample_category_ids):
    """
    Test that photometric transformations are applied correctly without affecting bounding boxes.
    """
    augmentor = DataAugmentor(apply_geometric=False, apply_photometric=True, seed=42)

    # Apply augmentation
    result = augmentor.apply_augmentation(
        image=sample_image,
        bboxes=sample_bboxes,
        category_ids=sample_category_ids
    )

    # Image should have changed due to photometric transformations
    expected_image = torch.from_numpy(sample_image).permute(2, 0, 1)
    assert not torch.equal(result['image'], expected_image), "Image should be transformed when photometric transformations are applied."

    # Bounding boxes should remain unchanged
    assert result['bboxes'] == sample_bboxes, "Bounding boxes should remain unchanged when only photometric transformations are applied."

def test_apply_augmentation_with_geometric_only(sample_image, sample_bboxes, sample_category_ids):
    """
    Test that geometric transformations are applied and bounding boxes are adjusted accordingly.
    """
    augmentor = DataAugmentor(apply_geometric=True, apply_photometric=False, seed=42)

    # Apply augmentation
    result = augmentor.apply_augmentation(
        image=sample_image,
        bboxes=sample_bboxes,
        category_ids=sample_category_ids
    )

    # Image should have changed due to geometric transformations
    expected_image = torch.from_numpy(sample_image).permute(2, 0, 1)
    assert not torch.equal(result['image'], expected_image), "Image should be transformed when geometric transformations are applied."

    # Bounding boxes should have changed
    for orig_bbox, aug_bbox in zip(sample_bboxes, result['bboxes']):
        assert aug_bbox != orig_bbox, "Bounding boxes should be transformed when geometric transformations are applied."

def test_data_augmentor_without_seed(sample_image, sample_bboxes, sample_category_ids):
    """
    Test that the augmentations are different when no seed is provided.
    """
    augmentor1 = DataAugmentor(apply_geometric=True, apply_photometric=True)
    augmentor2 = DataAugmentor(apply_geometric=True, apply_photometric=True)

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

    # With no seed, the results should differ
    assert not torch.equal(result1['image'], result2['image']), "Images should differ when no seed is set."
    assert result1['bboxes'] != result2['bboxes'], "Bounding boxes should differ when no seed is set."
