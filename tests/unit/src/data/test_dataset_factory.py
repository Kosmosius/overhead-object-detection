# tests/unit/src/data/test_dataset_factory.py

import pytest
from unittest.mock import patch, MagicMock
from src.data.dataset_factory import DatasetFactory, CocoDatasetFactory, DatasetNotSupportedError

@pytest.fixture
def mock_feature_extractor():
    """Mock feature extractor for testing."""
    return MagicMock()

@pytest.fixture
def sample_data_dir(tmpdir):
    """Fixture for a temporary data directory."""
    return tmpdir.mkdir("data")

def test_coco_dataset_factory_initialization(sample_data_dir, mock_feature_extractor):
    factory = CocoDatasetFactory(data_dir=str(sample_data_dir), 
                                 batch_size=2, 
                                 mode='train', 
                                 feature_extractor_name='facebook/detr-resnet-50')
    assert isinstance(factory, CocoDatasetFactory)

def test_dataset_factory_creation(sample_data_dir, mock_feature_extractor):
    with patch('src.data.dataloader.CocoDataset', autospec=True):
        factory = DatasetFactory(data_dir=str(sample_data_dir), 
                                 batch_size=2, 
                                 dataset_type='coco', 
                                 mode='train')
        dataloader = factory.get_dataloader()
        assert isinstance(dataloader, torch.utils.data.DataLoader)

def test_unsupported_dataset_type(sample_data_dir):
    with pytest.raises(DatasetNotSupportedError):
        DatasetFactory(data_dir=str(sample_data_dir), batch_size=2, dataset_type='unsupported')
