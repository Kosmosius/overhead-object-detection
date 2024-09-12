# tests/unit/src/models/test_foundation_model.py

import pytest
import torch
from src.models.foundation_model import HuggingFaceObjectDetectionModel
from unittest.mock import MagicMock

@pytest.fixture
def mock_model():
    return HuggingFaceObjectDetectionModel(
        model_name="facebook/detr-resnet-50",
        num_classes=91,
        pretrained=False,
        device="cpu"
    )

def test_model_initialization(mock_model):
    """Test that the HuggingFace object detection model initializes correctly."""
    assert isinstance(mock_model.model, torch.nn.Module), "Model should be a torch.nn.Module"
    assert mock_model.model_name == "facebook/detr-resnet-50"
    assert mock_model.num_classes == 91

def test_model_forward_pass(mock_model):
    """Test the forward pass of the model with mock input."""
    mock_input = torch.rand(2, 3, 224, 224)  # 2 images of 224x224 with 3 channels
    mock_model.model = MagicMock(return_value={"logits": torch.rand(2, 91), "pred_boxes": torch.rand(2, 4)})
    output = mock_model.forward(pixel_values=mock_input)
    
    assert "logits" in output, "Model output should contain logits"
    assert "pred_boxes" in output, "Model output should contain predicted boxes"

def test_save_model(mock_model, tmpdir):
    """Test that the model can be saved to a specified path."""
    save_path = tmpdir.mkdir("models").join("test_model")
    mock_model.model = MagicMock()
    mock_model.save(str(save_path))
    mock_model.model.save_pretrained.assert_called_once_with(str(save_path))

def test_load_model(mock_model, tmpdir):
    """Test loading a model from a specified path."""
    load_path = tmpdir.mkdir("models").join("test_model")
    mock_model.model = MagicMock()
    mock_model.load(str(load_path))
    mock_model.model.from_pretrained.assert_called_once_with(str(load_path))
