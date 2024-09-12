# tests/unit/src/data/test_dataloader.py

import pytest
from unittest.mock import MagicMock, patch
from src.data.dataloader import get_dataloader, collate_fn
import os
import torch

@pytest.fixture
def mock_feature_extractor():
    """Mock feature extractor for testing."""
    return MagicMock()

@pytest.fixture
def sample_data_dir(tmpdir):
    """Fixture for a temporary data directory."""
    return tmpdir.mkdir("data")

def test_dataloader_initialization(sample_data_dir, mock_feature_extractor):
    with patch('src.data.dataloader.CocoDataset', autospec=True) as MockCocoDataset:
        mock_dataset = MockCocoDataset.return_value
        mock_dataset.__len__.return_value = 10

        dataloader = get_dataloader(data_dir=str(sample_data_dir), 
                                    batch_size=2, 
                                    mode='train', 
                                    feature_extractor=mock_feature_extractor)
        assert isinstance(dataloader, torch.utils.data.DataLoader)
        assert len(dataloader) > 0

def test_collate_fn():
    mock_image = torch.rand((3, 224, 224))
    mock_annotation = {"boxes": torch.rand((2, 4)), "labels": torch.tensor([1, 2])}

    batch = [(mock_image, mock_annotation), (mock_image, mock_annotation)]
    images, annotations = collate_fn(batch)

    assert len(images) == 2
    assert len(annotations) == 2
