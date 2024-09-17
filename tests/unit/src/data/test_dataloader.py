# tests/unit/src/data/test_dataloader.py

import pytest
import os
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from src.data.dataloader import CocoDataset, collate_fn, get_dataloader, validate_data_paths
from src.data.augmentation import DataAugmentor
from pycocotools.coco import COCO
from PIL import Image

# Fixtures for sample data
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

@pytest.fixture
def sample_target():
    """Fixture for a sample target dictionary."""
    return {
        'boxes': torch.tensor([[50, 50, 150, 150], [30, 30, 100, 100]], dtype=torch.float32),
        'labels': torch.tensor([1, 2], dtype=torch.int64),
        'image_id': torch.tensor([1], dtype=torch.int64)
    }

@pytest.fixture
def mock_coco():
    """Fixture to mock the COCO API."""
    with patch('src.data.dataloader.COCO') as mock_coco_class:
        mock_coco_instance = MagicMock()
        mock_coco_instance.imgs = {
            1: {'file_name': 'image1.jpg'},
            2: {'file_name': 'image2.jpg'}
        }
        mock_coco_instance.getAnnIds.side_effect = lambda imgIds: [1, 2] if imgIds == [1] else []
        mock_coco_instance.loadAnns.return_value = [
            {'bbox': [50, 50, 100, 100], 'category_id': 1},
            {'bbox': [30, 30, 70, 70], 'category_id': 2}
        ]
        mock_coco_class.return_value = mock_coco_instance
        yield mock_coco_instance

@pytest.fixture
def mock_cv2_imread():
    """Fixture to mock cv2.imread."""
    with patch('src.data.dataloader.cv2.imread') as mock_imread:
        mock_imread.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        yield mock_imread

@pytest.fixture
def mock_cv2_cvtColor():
    """Fixture to mock cv2.cvtColor."""
    with patch('src.data.dataloader.cv2.cvtColor') as mock_cvtColor:
        mock_cvtColor.side_effect = lambda img, code: img  # Return image unchanged
        yield mock_cvtColor

@pytest.fixture
def mock_augmentor():
    """Fixture to mock DataAugmentor."""
    with patch('src.data.dataloader.DataAugmentor') as MockAugmentor:
        mock_instance = MockAugmentor.return_value
        mock_instance.apply_augmentation.return_value = {
            'image': torch.zeros((3, 224, 224), dtype=torch.float32),  # Return torch.Tensor
            'bboxes': [
                [50, 50, 150, 150],
                [30, 30, 100, 100]
            ],
            'category_ids': [1, 2]
        }
        yield mock_instance

@pytest.fixture
def sample_data_dir(tmp_path):
    """Fixture for a temporary data directory."""
    return tmp_path / "data"

# Test for validate_data_paths
def test_validate_data_paths_valid(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ann_file = data_dir / "annotations" / "instances_train2017.json"
    ann_file.parent.mkdir(parents=True, exist_ok=True)
    ann_file.touch()
    img_dir = data_dir / "train2017"
    img_dir.mkdir()
    
    # Should not raise an exception
    try:
        from src.data.dataloader import validate_data_paths
        validate_data_paths(str(data_dir), str(ann_file), str(img_dir))
    except FileNotFoundError:
        pytest.fail("validate_data_paths raised FileNotFoundError unexpectedly!")

@pytest.mark.parametrize("missing_path", ["data_dir", "ann_file", "img_dir"])
def test_validate_data_paths_invalid(tmp_path, missing_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ann_file = data_dir / "annotations" / "instances_train2017.json"
    ann_file.parent.mkdir(parents=True, exist_ok=True)
    ann_file.touch()
    img_dir = data_dir / "train2017"
    img_dir.mkdir()
    
    if missing_path == "data_dir":
        # Pass a non-existent data directory
        with pytest.raises(FileNotFoundError):
            from src.data.dataloader import validate_data_paths
            validate_data_paths(str(data_dir.parent / "nonexistent_data_dir"), str(ann_file), str(img_dir))
    elif missing_path == "ann_file":
        # Pass a non-existent annotation file
        with pytest.raises(FileNotFoundError):
            from src.data.dataloader import validate_data_paths
            validate_data_paths(str(data_dir), str(data_dir / "annotations" / "nonexistent.json"), str(img_dir))
    elif missing_path == "img_dir":
        # Pass a non-existent image directory
        with pytest.raises(FileNotFoundError):
            from src.data.dataloader import validate_data_paths
            validate_data_paths(str(data_dir), str(ann_file), str(data_dir / "nonexistent_train2017"))

# Test CocoDataset initialization
def test_coco_dataset_initialization(sample_data_dir, mock_coco, mock_cv2_imread, mock_cv2_cvtColor):
    ann_file = os.path.join(sample_data_dir, "annotations", "instances_train2017.json")
    img_dir = os.path.join(sample_data_dir, "train2017")
    os.makedirs(os.path.dirname(ann_file), exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    with open(ann_file, 'w') as f:
        f.write("{}")  # Mock empty JSON
    
    dataset = CocoDataset(
        img_dir=img_dir,
        ann_file=ann_file,
        transforms=None,
        feature_extractor=None
    )

    assert dataset.img_dir == img_dir
    assert dataset.ann_file == ann_file
    assert len(dataset) == 2  # Based on mock_coco.imgs
    mock_coco.getAnnIds.assert_called_with(imgIds=[1])  # Assuming dataset starts from index 0 with image_id=1
    mock_coco.loadAnns.assert_called_with([1, 2])

# Test CocoDataset __getitem__
def test_coco_dataset_getitem(sample_image, sample_bboxes, sample_category_ids, mock_coco, mock_cv2_imread, mock_cv2_cvtColor, mock_augmentor):
    dataset = CocoDataset(
        img_dir="dummy_dir",
        ann_file="dummy_ann.json",
        transforms=mock_augmentor,
        feature_extractor=None
    )

    # Mock the dataset's internal data to return sample data
    # Assuming dataset internally uses image index 0 which corresponds to image_id=1
    # Mock the COCO dataset to return image_id=1 and the corresponding annotations
    # Here, the actual image loading is mocked by cv2.imread and cvtColor

    image, target = dataset[0]

    # Assertions on image
    assert isinstance(image, torch.Tensor), "Image should be a torch.Tensor."
    assert image.shape == (3, 224, 224), "Image tensor shape mismatch."

    # Assertions on target
    assert 'boxes' in target, "Target should contain 'boxes'."
    assert 'labels' in target, "Target should contain 'labels'."
    assert 'image_id' in target, "Target should contain 'image_id'."
    assert target['boxes'].shape == (2, 4), "Boxes tensor shape mismatch."
    assert target['labels'].shape == (2,), "Labels tensor shape mismatch."
    assert target['image_id'].item() == 1, "Image ID mismatch."

    # Ensure augmentation was applied
    mock_augmentor.apply_augmentation.assert_called_once_with(
        image=sample_image,
        bboxes=sample_bboxes,
        category_ids=sample_category_ids
    )

# Test CocoDataset __getitem__ without augmentations
def test_coco_dataset_getitem_no_augment(sample_image, sample_bboxes, sample_category_ids, mock_coco, mock_cv2_imread, mock_cv2_cvtColor):
    with patch('src.data.dataloader.ToTensorV2') as mock_to_tensor:
        mock_to_tensor_instance = mock_to_tensor.return_value
        mock_to_tensor_instance.return_value = {'image': torch.rand(3, 224, 224)}
    
        dataset = CocoDataset(
            img_dir="dummy_dir",
            ann_file="dummy_ann.json",
            transforms=None,
            feature_extractor=None
        )

        # Mock internal data
        image, target = dataset[0]

        # Assertions on image
        assert isinstance(image, torch.Tensor), "Image should be a torch.Tensor."
        assert image.shape == (3, 224, 224), "Image tensor shape mismatch."

        # Assertions on target
        assert 'boxes' in target
        assert 'labels' in target
        assert 'image_id' in target
        assert target['boxes'].shape == (2, 4)
        assert target['labels'].shape == (2,)
        assert target['image_id'].item() == 1

        # Ensure ToTensorV2 was called
        mock_to_tensor_instance.assert_called_once()

# Test CocoDataset with feature extractor
def test_coco_dataset_with_feature_extractor(sample_image, sample_bboxes, sample_category_ids, mock_coco, mock_cv2_imread, mock_cv2_cvtColor):
    mock_feature_extractor = MagicMock()
    mock_feature_extractor.return_value = {'pixel_values': torch.rand(3, 224, 224)}
    
    dataset = CocoDataset(
        img_dir="dummy_dir",
        ann_file="dummy_ann.json",
        transforms=None,
        feature_extractor=mock_feature_extractor
    )

    image, target = dataset[0]

    # Assertions on image
    mock_feature_extractor.assert_called_once_with(image=sample_image, return_tensors="pt")
    assert isinstance(image, torch.Tensor), "Image should be a torch.Tensor."
    assert image.shape == (3, 224, 224), "Image tensor shape mismatch."

    # Assertions on target
    assert 'boxes' in target
    assert 'labels' in target
    assert 'image_id' in target
    assert target['boxes'].shape == (2, 4)
    assert target['labels'].shape == (2,)
    assert target['image_id'].item() == 1

# Test collate_fn
def test_collate_fn(sample_image, sample_bboxes, sample_category_ids):
    image_tensor = torch.rand(3, 224, 224)
    target = {
        'boxes': torch.tensor(sample_bboxes, dtype=torch.float32),
        'labels': torch.tensor(sample_category_ids, dtype=torch.int64),
        'image_id': torch.tensor([1], dtype=torch.int64)
    }

    batch = [(image_tensor, target), (image_tensor, target)]
    images, targets = collate_fn(batch)

    # Assertions on images
    assert isinstance(images, list) or isinstance(images, tuple), "Images should be a list or tuple."
    assert len(images) == 2, "Number of images in batch mismatch."
    assert all(isinstance(img, torch.Tensor) for img in images), "All images should be torch.Tensors."

    # Assertions on targets
    assert isinstance(targets, list) or isinstance(targets, tuple), "Targets should be a list or tuple."
    assert len(targets) == 2, "Number of targets in batch mismatch."
    for tgt in targets:
        assert isinstance(tgt, dict), "Each target should be a dictionary."
        assert 'boxes' in tgt and 'labels' in tgt and 'image_id' in tgt, "Target keys missing."

# Test get_dataloader
@pytest.mark.parametrize(
    "mode, apply_geometric, apply_photometric",
    [
        ('train', True, True),
        ('train', False, True),
        ('train', True, False),
        ('val', False, False)
    ]
)
def test_get_dataloader(
    tmp_path,
    mode,
    apply_geometric,
    apply_photometric,
    mock_coco,
    mock_cv2_imread,
    mock_cv2_cvtColor
):
    # Setup mock dataset
    data_dir = tmp_path / "data"
    ann_file = data_dir / "annotations" / f"instances_{mode}2017.json"
    img_dir = data_dir / f"{mode}2017"
    ann_file.parent.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    with open(ann_file, 'w') as f:
        f.write("{}")  # Mock empty JSON

    # Mock augmentation if applicable
    with patch('src.data.dataloader.DataAugmentor') as MockAugmentor:
        mock_augmentor = MockAugmentor.return_value
        if mode == 'train' and (apply_geometric or apply_photometric):
            mock_augmentor.apply_augmentation.return_value = {
                'image': torch.zeros((3, 224, 224), dtype=torch.float32),  # Return torch.Tensor
                'bboxes': [
                    [50, 50, 150, 150],
                    [30, 30, 100, 100]
                ],
                'category_ids': [1, 2]
            }
        else:
            mock_augmentor = None

        # Create DataLoader
        dataloader = get_dataloader(
            data_dir=str(data_dir),
            batch_size=2,
            mode=mode,
            feature_extractor=None,
            dataset_type='coco',
            num_workers=0,  # Set to 0 for testing
            pin_memory=False
        )

        # Assertions on DataLoader
        assert isinstance(dataloader, torch.utils.data.DataLoader), "Should return a DataLoader instance."
        assert dataloader.batch_size == 2, "Batch size mismatch."
        # Removed the incorrect assertion on 'shuffle' attribute

        # Iterate over DataLoader and assert batch contents
        for batch in dataloader:
            images, targets = batch
            assert isinstance(images, list), "Images should be a list."
            assert isinstance(targets, list), "Targets should be a list."
            assert len(images) == 2, "Batch size mismatch in images."
            assert len(targets) == 2, "Batch size mismatch in targets."

            for img, tgt in zip(images, targets):
                assert isinstance(img, torch.Tensor), "Image should be a torch.Tensor."
                assert img.shape == (3, 224, 224), "Image tensor shape mismatch."
                assert isinstance(tgt, dict), "Target should be a dictionary."
                assert 'boxes' in tgt and 'labels' in tgt and 'image_id' in tgt, "Target keys missing."
                assert tgt['boxes'].shape == (2, 4), "Boxes tensor shape mismatch."
                assert tgt['labels'].shape == (2,), "Labels tensor shape mismatch."
                assert isinstance(tgt['image_id'], torch.Tensor), "Image ID should be a torch.Tensor."

# Test CocoDataset with missing image
def test_coco_dataset_missing_image(sample_data_dir, mock_coco, mock_cv2_imread, mock_cv2_cvtColor):
    ann_file = os.path.join(sample_data_dir, "annotations", "instances_train2017.json")
    img_dir = os.path.join(sample_data_dir, "train2017")
    os.makedirs(os.path.dirname(ann_file), exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    with open(ann_file, 'w') as f:
        f.write("{}")  # Mock empty JSON

    # Mock cv2.imread to return None (simulate missing image)
    with patch('src.data.dataloader.cv2.imread', return_value=None):
        dataset = CocoDataset(
            img_dir=img_dir,
            ann_file=ann_file,
            transforms=None,
            feature_extractor=None
        )
        with pytest.raises(FileNotFoundError):
            _ = dataset[0]

# Test CocoDataset with invalid bounding box format
def test_coco_dataset_invalid_bbox_format(sample_data_dir, mock_coco, mock_cv2_imread, mock_cv2_cvtColor, mock_augmentor):
    ann_file = os.path.join(sample_data_dir, "annotations", "instances_train2017.json")
    img_dir = os.path.join(sample_data_dir, "train2017")
    os.makedirs(os.path.dirname(ann_file), exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    with open(ann_file, 'w') as f:
        f.write("{}")  # Mock empty JSON

    # Mock augmentation to return invalid bbox format
    mock_augmentor.apply_augmentation.return_value = {
        'image': torch.zeros((3, 224, 224), dtype=torch.float32),
        'bboxes': ["invalid_bbox"],
        'category_ids': [1]
    }

    dataset = CocoDataset(
        img_dir=img_dir,
        ann_file=ann_file,
        transforms=mock_augmentor,
        feature_extractor=None
    )

    with pytest.raises(ValueError):
        _ = dataset[0]

# Test get_dataloader with unsupported dataset type
def test_get_dataloader_unsupported_dataset():
    with pytest.raises(ValueError):
        from src.data.dataloader import get_dataloader
        get_dataloader(
            data_dir="dummy_dir",
            batch_size=2,
            mode='train',
            feature_extractor=None,
            dataset_type='unsupported',
            num_workers=0,
            pin_memory=False
        )

# Test get_dataloader with empty dataset
def test_get_dataloader_empty_dataset(sample_data_dir, mock_coco, mock_cv2_imread, mock_cv2_cvtColor, mock_augmentor):
    ann_file = os.path.join(sample_data_dir, "annotations", "instances_train2017.json")
    img_dir = os.path.join(sample_data_dir, "train2017")
    os.makedirs(os.path.dirname(ann_file), exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    with open(ann_file, 'w') as f:
        f.write("{}")  # Mock empty JSON

    # Mock COCO to have no images
    mock_coco.imgs = {}
    mock_coco.getAnnIds.return_value = []
    mock_coco.loadAnns.return_value = []

    dataloader = get_dataloader(
        data_dir=str(sample_data_dir),
        batch_size=2,
        mode='train',
        feature_extractor=None,
        dataset_type='coco',
        num_workers=0,
        pin_memory=False
    )

    assert len(dataloader) == 0, "Dataloader should be empty when dataset has no images."

# Test get_dataloader reproducibility with seed
def test_get_dataloader_reproducibility(sample_data_dir, mock_coco, mock_cv2_imread, mock_cv2_cvtColor):
    ann_file = os.path.join(sample_data_dir, "annotations", "instances_train2017.json")
    img_dir = os.path.join(sample_data_dir, "train2017")
    os.makedirs(os.path.dirname(ann_file), exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    with open(ann_file, 'w') as f:
        f.write("{}")  # Mock empty JSON

    # Initialize DataAugmentor with a fixed seed
    with patch('src.data.dataloader.DataAugmentor') as MockAugmentor:
        mock_augmentor = MockAugmentor.return_value
        mock_augmentor.apply_augmentation.side_effect = lambda image, bboxes, category_ids: {
            'image': image.clone(),  # Assuming image is a torch.Tensor
            'bboxes': bboxes.copy(),
            'category_ids': category_ids.copy()
        }

        dataloader1 = get_dataloader(
            data_dir=str(sample_data_dir),
            batch_size=2,
            mode='train',
            feature_extractor=None,
            dataset_type='coco',
            num_workers=0,
            pin_memory=False
        )
        dataloader2 = get_dataloader(
            data_dir=str(sample_data_dir),
            batch_size=2,
            mode='train',
            feature_extractor=None,
            dataset_type='coco',
            num_workers=0,
            pin_memory=False
        )

        # Since augmentations are mocked to return copies, the outputs should be identical
        for batch1, batch2 in zip(dataloader1, dataloader2):
            images1, targets1 = batch1
            images2, targets2 = batch2
            assert torch.equal(images1, images2), "Images should be identical across dataloaders with the same seed."
            assert targets1 == targets2, "Targets should be identical across dataloaders with the same seed."
