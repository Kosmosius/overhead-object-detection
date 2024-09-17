# tests/unit/src/data/test_dataset_factory.py

import pytest
from unittest.mock import patch, MagicMock
from src.data.dataset_factory import (
    DatasetFactory,
    CocoDatasetFactory,
    DatasetNotSupportedError,
    BaseDatasetFactory
)
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor
import logging  # Added import for logging


@pytest.fixture
def mock_feature_extractor():
    """Fixture to mock AutoFeatureExtractor.from_pretrained."""
    with patch('src.data.dataset_factory.AutoFeatureExtractor.from_pretrained') as mock_extractor:
        mock_instance = MagicMock(spec=AutoFeatureExtractor)
        mock_extractor.return_value = mock_instance
        yield mock_extractor, mock_instance


@pytest.fixture
def mock_get_dataloader():
    """Fixture to mock get_dataloader function."""
    with patch('src.data.dataset_factory.get_dataloader') as mock_dl:
        mock_data_loader = MagicMock(spec=DataLoader)
        mock_dl.return_value = mock_data_loader
        yield mock_dl, mock_data_loader


@pytest.fixture
def sample_data_dir(tmp_path):
    """Fixture for a temporary data directory."""
    return tmp_path / "data"


def test_coco_dataset_factory_initialization(sample_data_dir, mock_feature_extractor):
    """
    Test that CocoDatasetFactory initializes correctly with given parameters.
    """
    _, mock_extractor_instance = mock_feature_extractor

    factory = CocoDatasetFactory(
        data_dir=str(sample_data_dir),
        batch_size=4,
        mode='train',
        feature_extractor_name='facebook/detr-resnet-50'
    )

    assert factory.data_dir == str(sample_data_dir), "Data directory mismatch."
    assert factory.batch_size == 4, "Batch size mismatch."
    assert factory.mode == 'train', "Mode mismatch."
    assert factory.feature_extractor_name == 'facebook/detr-resnet-50', "Feature extractor name mismatch."
    mock_feature_extractor[0].assert_called_once_with('facebook/detr-resnet-50')
    assert factory.feature_extractor == mock_feature_extractor[1], "Feature extractor instance mismatch."


def test_dataset_factory_creation_coco(sample_data_dir, mock_feature_extractor, mock_get_dataloader):
    """
    Test that DatasetFactory creates a CocoDatasetFactory and retrieves a DataLoader correctly.
    """
    mock_get_dataloader_func, mock_data_loader = mock_get_dataloader

    factory = DatasetFactory(
        data_dir=str(sample_data_dir),
        batch_size=8,
        dataset_type='coco',
        mode='train',
        feature_extractor_name='facebook/detr-resnet-50'
    )

    assert isinstance(factory.dataset_factory, CocoDatasetFactory), "Factory type mismatch."

    dataloader = factory.get_dataloader()

    mock_feature_extractor[0].assert_called_once_with('facebook/detr-resnet-50')
    mock_get_dataloader_func.assert_called_once_with(
        data_dir=str(sample_data_dir),
        batch_size=8,
        mode='train',
        feature_extractor=mock_feature_extractor[1],
        dataset_type='coco'
    )
    assert dataloader == mock_data_loader, "Returned DataLoader does not match the mock."


def test_dataset_factory_unsupported_type(sample_data_dir):
    """
    Test that DatasetFactory raises DatasetNotSupportedError for unsupported dataset types.
    """
    with pytest.raises(DatasetNotSupportedError) as exc_info:
        DatasetFactory(
            data_dir=str(sample_data_dir),
            batch_size=8,
            dataset_type='unsupported_dataset',
            mode='train',
            feature_extractor_name='facebook/detr-resnet-50'
        )
    
    assert "Dataset type 'unsupported_dataset' is not supported." in str(exc_info.value), "Error message mismatch."


def test_coco_dataset_factory_logging(sample_data_dir, mock_feature_extractor, mock_get_dataloader, caplog):
    """
    Test that CocoDatasetFactory logs dataset information correctly.
    """
    _, mock_extractor_instance = mock_feature_extractor
    mock_get_dataloader_func, mock_data_loader = mock_get_dataloader

    factory = CocoDatasetFactory(
        data_dir=str(sample_data_dir),
        batch_size=16,
        mode='val',
        feature_extractor_name='facebook/detr-resnet-50'
    )

    # Assuming that CocoDatasetFactory logs dataset info during initialization or when get_dataloader is called
    # Adjust based on actual implementation
    with caplog.at_level(logging.INFO):
        dataloader = factory.get_dataloader()

    # Check that log messages are present
    assert "Dataset: CocoDatasetFactory" in caplog.text
    assert f"Data Directory: {str(sample_data_dir)}" in caplog.text
    assert "Batch Size: 16" in caplog.text
    assert "Mode: val" in caplog.text
    assert "Feature Extractor: facebook/detr-resnet-50" in caplog.text


def test_coco_dataset_factory_get_dataloader(sample_data_dir, mock_feature_extractor, mock_get_dataloader):
    """
    Test that CocoDatasetFactory.get_dataloader returns the expected DataLoader.
    """
    factory = CocoDatasetFactory(
        data_dir=str(sample_data_dir),
        batch_size=32,
        mode='train',
        feature_extractor_name='facebook/detr-resnet-50'
    )

    dataloader = factory.get_dataloader()

    assert isinstance(dataloader, DataLoader), "Returned object is not a DataLoader."
    assert dataloader == mock_get_dataloader[1], "Returned DataLoader does not match the mock."


def test_base_dataset_factory_not_implemented(sample_data_dir, mock_feature_extractor):
    """
    Test that BaseDatasetFactory.get_dataloader raises NotImplementedError.
    """
    from src.data.dataset_factory import BaseDatasetFactory

    factory = BaseDatasetFactory(
        data_dir=str(sample_data_dir),
        batch_size=4,
        mode='train',
        feature_extractor_name='facebook/detr-resnet-50'
    )

    with pytest.raises(NotImplementedError) as exc_info:
        factory.get_dataloader()
    
    assert "This method should be implemented by subclasses." in str(exc_info.value), "Error message mismatch."


@pytest.mark.parametrize(
    "dataset_type, expected_factory",
    [
        ('coco', CocoDatasetFactory),
        # Future support for other datasets can be added here
    ]
)
def test_dataset_factory_selection(sample_data_dir, mock_feature_extractor, mock_get_dataloader, dataset_type, expected_factory):
    """
    Test that DatasetFactory selects the correct factory based on dataset_type.
    """
    factory = DatasetFactory(
        data_dir=str(sample_data_dir),
        batch_size=16,
        dataset_type=dataset_type,
        mode='train',
        feature_extractor_name='facebook/detr-resnet-50'
    )

    assert isinstance(factory.dataset_factory, expected_factory), f"Factory should be an instance of {expected_factory.__name__}."


def test_dataset_factory_feature_extractor(sample_data_dir, mock_feature_extractor, mock_get_dataloader):
    """
    Test that DatasetFactory initializes the feature extractor correctly.
    """
    factory = DatasetFactory(
        data_dir=str(sample_data_dir),
        batch_size=8,
        dataset_type='coco',
        mode='train',
        feature_extractor_name='facebook/detr-resnet-50'
    )

    # The feature extractor should have been initialized
    mock_feature_extractor[0].assert_called_once_with('facebook/detr-resnet-50')
    assert factory.dataset_factory.feature_extractor == mock_feature_extractor[1], "Feature extractor instance mismatch."


def test_dataset_factory_multiple_calls(sample_data_dir, mock_feature_extractor, mock_get_dataloader):
    """
    Test that multiple calls to get_dataloader work correctly and use the same factory.
    """
    factory = DatasetFactory(
        data_dir=str(sample_data_dir),
        batch_size=5,
        dataset_type='coco',
        mode='train',
        feature_extractor_name='facebook/detr-resnet-50'
    )

    dataloader1 = factory.get_dataloader()
    dataloader2 = factory.get_dataloader()

    mock_get_dataloader_func, mock_data_loader = mock_get_dataloader
    # Since get_dataloader is called twice
    assert mock_get_dataloader_func.call_count == 2, "get_dataloader should be called twice."
    assert dataloader1 == mock_data_loader, "First DataLoader call does not match the mock."
    assert dataloader2 == mock_data_loader, "Second DataLoader call does not match the mock."


def test_dataset_factory_invalid_mode(sample_data_dir, mock_feature_extractor, mock_get_dataloader):
    """
    Test that DatasetFactory handles invalid mode values appropriately.
    """
    with pytest.raises(AssertionError) as exc_info:
        DatasetFactory(
            data_dir=str(sample_data_dir),
            batch_size=8,
            dataset_type='coco',
            mode='invalid_mode',  # Assuming mode should be 'train' or 'val'
            feature_extractor_name='facebook/detr-resnet-50'
        )
    
    assert "Mode should be either 'train' or 'val'." in str(exc_info.value), "Error message mismatch."


def test_dataset_factory_default_parameters(sample_data_dir, mock_feature_extractor, mock_get_dataloader):
    """
    Test that DatasetFactory uses default parameters when not explicitly provided.
    """
    # Assuming DatasetFactory has default values for dataset_type, mode, feature_extractor_name
    factory = DatasetFactory(
        data_dir=str(sample_data_dir),
        batch_size=4
        # Not specifying dataset_type, mode, feature_extractor_name
    )

    assert factory.dataset_type == 'coco', "Default dataset_type should be 'coco'."
    assert factory.mode == 'train', "Default mode should be 'train'."
    assert factory.feature_extractor_name == 'facebook/detr-resnet-50', "Default feature extractor name mismatch."

    assert isinstance(factory.dataset_factory, CocoDatasetFactory), "Default factory should be CocoDatasetFactory."

    mock_feature_extractor[0].assert_called_once_with('facebook/detr-resnet-50')
    mock_get_dataloader_func, mock_data_loader = mock_get_dataloader
    mock_get_dataloader_func.assert_called_once_with(
        data_dir=str(sample_data_dir),
        batch_size=4,
        mode='train',
        feature_extractor=mock_feature_extractor[1],
        dataset_type='coco'
    )
    assert factory.get_dataloader() == mock_data_loader, "DataLoader returned does not match the mock."


def test_dataset_factory_logger(sample_data_dir, mock_feature_extractor, mock_get_dataloader, caplog):
    """
    Test that DatasetFactory logs dataset information correctly.
    """
    factory = DatasetFactory(
        data_dir=str(sample_data_dir),
        batch_size=12,
        dataset_type='coco',
        mode='val',
        feature_extractor_name='facebook/detr-resnet-50'
    )

    with caplog.at_level(logging.INFO):
        dataloader = factory.get_dataloader()

    # Check that log messages are present
    assert "Dataset: CocoDatasetFactory" in caplog.text
    assert f"Data Directory: {str(sample_data_dir)}" in caplog.text
    assert "Batch Size: 12" in caplog.text
    assert "Mode: val" in caplog.text
    assert "Feature Extractor: facebook/detr-resnet-50" in caplog.text
