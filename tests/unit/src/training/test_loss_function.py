# tests/unit/src/training/test_loss_function.py

import torch
import pytest
from src.training.loss_functions import CustomLossFunction, compute_loss

@pytest.fixture
def mock_outputs():
    return {
        'logits': torch.rand((2, 10)),  # 2 samples, 10 classes
        'pred_boxes': torch.rand((2, 4))  # 2 samples, 4 bounding box coordinates (x, y, w, h)
    }

@pytest.fixture
def mock_targets():
    return [
        {'labels': torch.tensor([1, 2]), 'boxes': torch.rand((2, 4))},  # Sample 1
        {'labels': torch.tensor([3, 4]), 'boxes': torch.rand((2, 4))}   # Sample 2
    ]

@pytest.fixture
def mock_config():
    return {
        "class_weight": 1.0,
        "bbox_weight": 1.0,
        "giou_weight": 1.0
    }

def test_custom_loss_function(mock_outputs, mock_targets, mock_config):
    model = torch.nn.Module()  # Mock model (not used directly in this case)
    loss_fn = CustomLossFunction(model, mock_config)
    
    loss = loss_fn(mock_outputs, mock_targets)
    
    assert "total_loss" in loss
    assert "classification_loss" in loss
    assert "bbox_loss" in loss
    assert "giou_loss" in loss
    assert loss['total_loss'] > 0, "Total loss should be greater than 0."

def test_compute_loss(mock_outputs, mock_targets, mock_config):
    model = torch.nn.Module()  # Mock model
    loss = compute_loss(mock_outputs, mock_targets, model, mock_config)

    assert isinstance(loss, dict)
    assert "total_loss" in loss
    assert loss["total_loss"].item() >= 0  # Ensure loss is a valid number
