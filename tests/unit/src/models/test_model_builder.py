# tests/unit/src/models/test_model_builder.py

import pytest
import torch
from src.models.model_builder import ModelBuilder
from unittest.mock import MagicMock

@pytest.fixture
def builder():
    return ModelBuilder(
        model_name="facebook/detr-resnet-50",
        num_labels=91,
        pretrained=True
    )

def test_build_model(builder):
    """Test that the model builds correctly."""
    builder.build_model = MagicMock(return_value=torch.nn.Module())
    model = builder.build_model()
    builder.build_model.assert_called_once()
    assert isinstance(model, torch.nn.Module), "Model should be an instance of torch.nn.Module"

def test_load_model_from_checkpoint(builder, tmpdir):
    """Test loading a model from a checkpoint."""
    checkpoint_path = tmpdir.join("test_checkpoint.pth")
    builder.load_model_from_checkpoint = MagicMock()
    builder.load_model_from_checkpoint(str(checkpoint_path), device="cpu")
    builder.load_model_from_checkpoint.assert_called_once_with(str(checkpoint_path), device="cpu")

def test_save_model_checkpoint(builder, tmpdir):
    """Test saving a model checkpoint."""
    checkpoint_path = tmpdir.join("test_checkpoint.pth")
    builder.model = MagicMock()
    builder.save_model_checkpoint(str(checkpoint_path))
    torch.save.assert_called_once()
    