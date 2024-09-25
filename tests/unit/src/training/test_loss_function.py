# tests/unit/src/training/test_loss_functions.py

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch
import logging
from src.training.loss_functions import LossFunctionFactory, compute_loss


@pytest.fixture
def mock_outputs():
    """Fixture for mock model outputs."""
    return {
        'logits': torch.tensor([[0.2, 0.8], [0.6, 0.4]], dtype=torch.float32, requires_grad=True),  # 2 samples, 2 classes
        'pred_boxes': torch.tensor(
            [[[50.0, 50.0, 150.0, 150.0]], [[30.0, 30.0, 100.0, 100.0]]],
            dtype=torch.float32,
            requires_grad=True
        )  # 2 samples, 1 bbox each
    }

@pytest.fixture
def mock_targets():
    """Fixture for mock targets."""
    return [
        {'labels': torch.tensor([1], dtype=torch.long), 'boxes': torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)},
        {'labels': torch.tensor([0], dtype=torch.long), 'boxes': torch.tensor([[30, 30, 100, 100]], dtype=torch.float32)}
    ]

@pytest.fixture
def default_loss_config():
    """Fixture for default loss configuration."""
    return {
        "loss_type": "cross_entropy",
        "loss_weights": {
            "classification_loss": 1.0,
            "bbox_loss": 1.0
        },
        "loss_kwargs": {}
    }

@pytest.fixture
def custom_loss_config():
    """Fixture for custom loss configuration."""
    return {
        "loss_type": "mse",
        "loss_weights": {
            "classification_loss": 0.7,
            "bbox_loss": 0.3
        },
        "loss_kwargs": {
            "reduction": "sum"
        }
    }


@pytest.mark.parametrize(
    "loss_type, expected_class, kwargs, description",
    [
        ("cross_entropy", nn.CrossEntropyLoss, {}, "Default CrossEntropyLoss"),
        ("cross_entropy", nn.CrossEntropyLoss, {"weight": torch.tensor([1.0, 2.0])}, "CrossEntropyLoss with weight"),
        ("mse", nn.MSELoss, {"reduction": "sum"}, "MSELoss with reduction sum"),
        ("l1", nn.L1Loss, {"reduction": "mean"}, "L1Loss with reduction mean"),
        ("smooth_l1", nn.SmoothL1Loss, {"beta": 1.0}, "SmoothL1Loss with custom beta"),
    ]
)
def test_loss_function_factory_supported(loss_type, expected_class, kwargs, description):
    """Test that LossFunctionFactory returns the correct loss function class."""
    criterion = LossFunctionFactory.get_loss_function(loss_type, **kwargs)
    assert isinstance(criterion, expected_class), f"{description} should return instance of {expected_class.__name__}."

def test_loss_function_factory_unsupported():
    """Test that requesting an unsupported loss function raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        LossFunctionFactory.get_loss_function("unsupported_loss")
    assert "Unsupported loss type: unsupported_loss" in str(exc_info.value), "Did not raise expected ValueError for unsupported loss."

def test_loss_function_factory_kwargs():
    """Test that LossFunctionFactory correctly passes kwargs to the loss function."""
    weight = torch.tensor([1.0, 2.0])
    criterion = LossFunctionFactory.get_loss_function("cross_entropy", weight=weight)
    assert isinstance(criterion, nn.CrossEntropyLoss), "Criterion should be CrossEntropyLoss."
    assert torch.equal(criterion.weight, weight), "Weight parameter not correctly passed to CrossEntropyLoss."

# 2. Tests for compute_loss

def test_compute_loss_classification_and_bbox(mock_outputs, mock_targets, default_loss_config):
    """Test compute_loss with both classification and bbox losses."""
    loss_config = {
        "loss_types": {
            "classification_loss": "cross_entropy",
            "bbox_loss": "smooth_l1"
        },
        "loss_weights": {
            "classification_loss": 1.0,
            "bbox_loss": 1.0
        },
        "loss_kwargs": {
            "classification_loss": {},
            "bbox_loss": {}
        }
    }

    loss = compute_loss(mock_outputs, mock_targets, model=None, loss_config=loss_config)
    
    assert "classification_loss" in loss, "classification_loss should be present in loss dictionary."
    assert "bbox_loss" in loss, "bbox_loss should be present in loss dictionary."
    assert "total_loss" in loss, "total_loss should be present in loss dictionary."
    assert loss["total_loss"] == loss["classification_loss"] + loss["bbox_loss"], "total_loss should be the sum of individual losses."
    
    # Check that losses are scalar tensors
    assert loss["classification_loss"].dim() == 0, "classification_loss should be a scalar tensor."
    assert loss["bbox_loss"].dim() == 0, "bbox_loss should be a scalar tensor."
    assert loss["total_loss"].dim() == 0, "total_loss should be a scalar tensor."

def test_compute_loss_only_classification(mock_outputs, mock_targets, default_loss_config):
    """Test compute_loss with only classification loss (no bbox loss)."""
    outputs = {'logits': mock_outputs['logits']}
    targets = [{'labels': t['labels']} for t in mock_targets]
    
    loss_config = {
        "loss_type": "cross_entropy",
        "loss_weights": {
            "classification_loss": 1.0
            # bbox_loss is not provided
        },
        "loss_kwargs": {}
    }
    
    loss = compute_loss(outputs, targets, model=None, loss_config=loss_config)
    
    assert "classification_loss" in loss, "classification_loss should be present in loss dictionary."
    assert "bbox_loss" not in loss, "bbox_loss should not be present in loss dictionary."
    assert "total_loss" in loss, "total_loss should be present in loss dictionary."
    assert loss["total_loss"] == loss["classification_loss"], "total_loss should equal classification_loss."
    
    # Check that losses are scalar tensors
    assert loss["classification_loss"].dim() == 0, "classification_loss should be a scalar tensor."
    assert loss["total_loss"].dim() == 0, "total_loss should be a scalar tensor."

def test_compute_loss_only_bbox(mock_outputs, mock_targets, default_loss_config):
    """Test compute_loss with only bbox loss (no classification loss)."""
    outputs = {'pred_boxes': mock_outputs['pred_boxes']}
    targets = [{'boxes': t['boxes']} for t in mock_targets]
    
    loss_config = {
        "loss_type": "smooth_l1",
        "loss_weights": {
            "bbox_loss": 1.5
            # classification_loss is not provided
        },
        "loss_kwargs": {}
    }
    
    loss = compute_loss(outputs, targets, model=None, loss_config=loss_config)
    
    assert "classification_loss" not in loss, "classification_loss should not be present in loss dictionary."
    assert "bbox_loss" in loss, "bbox_loss should be present in loss dictionary."
    assert "total_loss" in loss, "total_loss should be present in loss dictionary."
    assert loss["total_loss"] == loss["bbox_loss"], "total_loss should equal bbox_loss."
    
    # Check that losses are scalar tensors
    assert loss["bbox_loss"].dim() == 0, "bbox_loss should be a scalar tensor."
    assert loss["total_loss"].dim() == 0, "total_loss should be a scalar tensor."

def test_compute_loss_custom_loss_type():
    """Test compute_loss with a custom loss type (e.g., MSELoss)."""
    loss_config = {
        "loss_types": {
            "classification_loss": "mse",
            "bbox_loss": "smooth_l1"
        },
        "loss_weights": {
            "classification_loss": 0.7,
            "bbox_loss": 0.3
        },
        "loss_kwargs": {
            "classification_loss": {"reduction": "sum"},
            "bbox_loss": {}
        }
    }

    outputs = {
        'logits': torch.randn(4, 10, requires_grad=True),  # 4 predictions, 10 classes
        'pred_boxes': torch.randn(4, 4, requires_grad=True)  # 4 predictions
    }
    targets = [
        {'labels': torch.randn(1, 10), 'boxes': torch.randn(1, 4)},
        {'labels': torch.randn(1, 10), 'boxes': torch.randn(1, 4)},
        {'labels': torch.randn(1, 10), 'boxes': torch.randn(1, 4)},
        {'labels': torch.randn(1, 10), 'boxes': torch.randn(1, 4)}
    ]

    loss = compute_loss(outputs, targets, model=None, loss_config=loss_config)

    # Compute expected individual losses
    concatenated_labels = torch.cat([t['labels'] for t in targets]).float()
    classification_loss = nn.MSELoss(reduction="sum")(outputs['logits'], concatenated_labels).item()
    bbox_loss = nn.SmoothL1Loss()(outputs['pred_boxes'], torch.cat([t['boxes'] for t in targets])).item()

    # Expected weighted losses
    expected_classification_loss = classification_loss * 0.7
    expected_bbox_loss = bbox_loss * 0.3
    expected_total_loss = expected_classification_loss + expected_bbox_loss

    assert torch.isclose(torch.tensor(loss["classification_loss"].item()), torch.tensor(expected_classification_loss), atol=1e-6), "classification_loss weighting mismatch."
    assert torch.isclose(torch.tensor(loss["bbox_loss"].item()), torch.tensor(expected_bbox_loss), atol=1e-6), "bbox_loss weighting mismatch."
    assert torch.isclose(torch.tensor(loss["total_loss"].item()), torch.tensor(expected_total_loss), atol=1e-6), "total_loss aggregation mismatch."

def test_compute_loss_unsupported_loss_type(mock_outputs, mock_targets):
    """Test compute_loss with an unsupported loss type."""
    loss_config = {
        "loss_types": {
            "classification_loss": "unsupported_loss",
            "bbox_loss": "unsupported_loss"
        },
        "loss_weights": {
            "classification_loss": 1.0,
            "bbox_loss": 1.0
        },
        "loss_kwargs": {}
    }
    
    with pytest.raises(ValueError) as exc_info:
        compute_loss(mock_outputs, mock_targets, model=None, loss_config=loss_config)
    
    assert "Unsupported classification loss type: unsupported_loss." in str(exc_info.value) or \
           "Unsupported bbox loss type: unsupported_loss." in str(exc_info.value), \
           "Did not raise expected ValueError for unsupported loss type."

def test_compute_loss_missing_classification_fields(mock_outputs, mock_targets):
    """Test compute_loss when classification fields are missing."""
    # Remove classification fields from outputs and targets
    outputs = {'pred_boxes': mock_outputs['pred_boxes']}
    targets = [{'boxes': t['boxes']} for t in mock_targets]

    loss_config = {
        "loss_types": {
            "bbox_loss": "smooth_l1"
            # 'classification_loss' is intentionally omitted
        },
        "loss_weights": {
            "bbox_loss": 1.0
            # 'classification_loss' is intentionally omitted
        },
        "loss_kwargs": {
            "bbox_loss": {}
            # 'classification_loss' is intentionally omitted
        }
    }

    loss = compute_loss(outputs, targets, model=None, loss_config=loss_config)

    assert "classification_loss" not in loss, "classification_loss should not be present when classification fields are missing."
    assert "bbox_loss" in loss, "bbox_loss should be present in loss dictionary."
    assert "total_loss" in loss, "total_loss should be present in loss dictionary."
    assert loss["total_loss"] == loss["bbox_loss"], "total_loss should equal bbox_loss when classification loss is not computed."

    # Check that losses are scalar tensors
    assert loss["bbox_loss"].dim() == 0, "bbox_loss should be a scalar tensor."
    assert loss["total_loss"].dim() == 0, "total_loss should be a scalar tensor."

def test_compute_loss_missing_bbox_fields(mock_outputs, mock_targets, default_loss_config):
    """Test compute_loss when bbox fields are missing."""
    outputs = {'logits': mock_outputs['logits']}
    targets = [{'labels': t['labels']} for t in mock_targets]
    
    loss_config = {
        "loss_type": "cross_entropy",
        "loss_weights": {
            "classification_loss": 1.0
        },
        "loss_kwargs": {}
    }
    
    loss = compute_loss(outputs, targets, model=None, loss_config=loss_config)
    
    assert "classification_loss" in loss, "classification_loss should be present."
    assert "bbox_loss" not in loss, "bbox_loss should not be present when bbox fields are missing."
    assert "total_loss" in loss, "total_loss should be present."
    assert loss["total_loss"] == loss["classification_loss"], "total_loss should equal classification_loss when bbox loss is not computed."

def test_compute_loss_no_losses(mock_outputs, mock_targets, default_loss_config):
    """Test compute_loss when neither classification nor bbox fields are present."""
    outputs = {}
    targets = [{} for _ in mock_targets]
    
    loss_config = {
        "loss_type": "cross_entropy",
        "loss_weights": {},
        "loss_kwargs": {}
    }
    
    loss = compute_loss(outputs, targets, model=None, loss_config=loss_config)
    
    assert "classification_loss" not in loss, "classification_loss should not be present."
    assert "bbox_loss" not in loss, "bbox_loss should not be present."
    assert "total_loss" in loss, "total_loss should be present."
    assert loss["total_loss"].item() == 0.0, "total_loss should be zero when no individual losses are computed."

def test_compute_loss_empty_targets(mock_outputs, default_loss_config):
    """Test compute_loss with empty targets."""
    outputs = mock_outputs
    targets = []

    loss = compute_loss(outputs, targets, model=None, loss_config=default_loss_config)

    assert "total_loss" in loss, "total_loss should be present."
    assert loss["total_loss"].item() == 0.0, "total_loss should be zero when targets are empty."

def test_compute_loss_custom_loss_weights(mock_outputs, mock_targets):
    """Test compute_loss with custom loss weights."""
    loss_config = {
        "loss_types": {
            "classification_loss": "mse",
            "bbox_loss": "smooth_l1"
        },
        "loss_weights": {
            "classification_loss": 0.7,
            "bbox_loss": 0.3
        },
        "loss_kwargs": {
            "classification_loss": {"reduction": "sum"},
            "bbox_loss": {}
        }
    }

    # Adjust logits and targets for MSELoss
    outputs = {
        'logits': torch.randn(4, 5, requires_grad=True),  # 4 predictions, 5 classes
        'pred_boxes': torch.randn(4, 4, requires_grad=True)
    }
    targets = [
        {'labels': torch.randint(0, 5, (2,), dtype=torch.long), 'boxes': torch.randn(2, 4)},
        {'labels': torch.randint(0, 5, (2,), dtype=torch.long), 'boxes': torch.randn(2, 4)}
    ]

    loss = compute_loss(outputs, targets, model=None, loss_config=loss_config)

    # One-hot encode labels for MSELoss
    num_classes = outputs['logits'].size(1)
    labels_one_hot = torch.nn.functional.one_hot(torch.cat([t['labels'] for t in targets]), num_classes=num_classes).float()

    # Compute expected individual losses
    classification_loss = nn.MSELoss(reduction="sum")(outputs['logits'], labels_one_hot).item()
    bbox_loss = nn.SmoothL1Loss()(outputs['pred_boxes'], torch.cat([t['boxes'] for t in targets])).item()

    # Apply weights
    expected_classification_loss = classification_loss * 0.7
    expected_bbox_loss = bbox_loss * 0.3
    expected_total_loss = expected_classification_loss + expected_bbox_loss

    # Use torch.isclose for comparison
    assert torch.isclose(torch.tensor(loss["classification_loss"].item()), torch.tensor(expected_classification_loss), atol=1e-6), "classification_loss weighting mismatch."
    assert torch.isclose(torch.tensor(loss["bbox_loss"].item()), torch.tensor(expected_bbox_loss), atol=1e-6), "bbox_loss weighting mismatch."
    assert torch.isclose(torch.tensor(loss["total_loss"].item()), torch.tensor(expected_total_loss), atol=1e-6), "total_loss aggregation mismatch."

def test_compute_loss_custom_loss_weights_with_custom_config(mock_outputs, mock_targets, custom_loss_config):
    """Test compute_loss with custom loss weights using a custom loss configuration."""
    loss_config = {
        "loss_types": {
            "classification_loss": "mse",
            "bbox_loss": "smooth_l1"
        },
        "loss_weights": {
            "classification_loss": 0.7,
            "bbox_loss": 0.3
        },
        "loss_kwargs": {
            "classification_loss": {"reduction": "sum"},
            "bbox_loss": {}
        }
    }

    loss = compute_loss(mock_outputs, mock_targets, model=None, loss_config=loss_config)

    # One-hot encode labels for MSELoss
    num_classes = outputs['logits'].size(1)
    labels_one_hot = torch.nn.functional.one_hot(torch.cat([t['labels'] for t in mock_targets]), num_classes=num_classes).float()

    # Compute expected individual losses
    classification_loss = nn.MSELoss(reduction="sum")(mock_outputs['logits'], labels_one_hot).item()
    bbox_loss = nn.SmoothL1Loss()(mock_outputs['pred_boxes'], torch.cat([t['boxes'] for t in mock_targets])).item()

    # Apply weights
    expected_classification_loss = classification_loss * 0.7
    expected_bbox_loss = bbox_loss * 0.3
    expected_total_loss = expected_classification_loss + expected_bbox_loss

    # Use torch.isclose for comparison
    assert torch.isclose(torch.tensor(loss["classification_loss"].item()), torch.tensor(expected_classification_loss), atol=1e-6), "classification_loss weighting mismatch."
    assert torch.isclose(torch.tensor(loss["bbox_loss"].item()), torch.tensor(expected_bbox_loss), atol=1e-6), "bbox_loss weighting mismatch."
    assert torch.isclose(torch.tensor(loss["total_loss"].item()), torch.tensor(expected_total_loss), atol=1e-6), "total_loss aggregation mismatch."

def test_compute_loss_logging(mock_outputs, mock_targets, default_loss_config, caplog):
    """Test that compute_loss logs the computed losses."""
    loss_config = {
        "loss_types": {
            "classification_loss": "cross_entropy",
            "bbox_loss": "smooth_l1"
        },
        "loss_weights": {
            "classification_loss": 1.0,
            "bbox_loss": 1.0
        },
        "loss_kwargs": {
            "classification_loss": {},
            "bbox_loss": {}
        }
    }

    with caplog.at_level(logging.DEBUG):
        compute_loss(mock_outputs, mock_targets, model=None, loss_config=loss_config)
    
    assert "Classification loss computed" in caplog.text, "Did not log classification loss."
    assert "Bounding box loss computed" in caplog.text, "Did not log bounding box loss."
    assert "Total loss computed" in caplog.text, "Did not log total loss."


def test_compute_loss_all_predictions_below_threshold(default_loss_config):
    """Test compute_loss when all predictions are below the confidence threshold."""
    outputs = {
        'logits': torch.randn(2, 3, requires_grad=True),
        'pred_boxes': torch.randn(2, 4, requires_grad=True)
    }
    targets = [
        {'labels': torch.tensor([1, 2], dtype=torch.long), 'boxes': torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=torch.float32)},
        {'labels': torch.tensor([3, 4], dtype=torch.long), 'boxes': torch.tensor([[40, 40, 50, 50], [60, 60, 70, 70]], dtype=torch.float32)}
    ]
    
    # Set high confidence threshold to filter out all predictions
    loss_config = {
        "loss_type": "cross_entropy",
        "loss_weights": {
            "classification_loss": 1.0,
            "bbox_loss": 1.0
        },
        "loss_kwargs": {}
    }
    
    # Mock LossFunctionFactory to return a dummy loss function that returns zero
    with patch.object(LossFunctionFactory, 'get_loss_function', return_value=nn.CrossEntropyLoss()):
        # Mock the loss functions to simulate all predictions being filtered out
        with patch.object(nn.CrossEntropyLoss, '__call__', return_value=torch.tensor(0.0, requires_grad=True)), \
             patch.object(nn.SmoothL1Loss, '__call__', return_value=torch.tensor(0.0, requires_grad=True)):
            loss = compute_loss(outputs, targets, model=None, loss_config=loss_config)
    
    assert "classification_loss" in loss, "classification_loss should be present."
    assert "bbox_loss" in loss, "bbox_loss should be present."
    assert "total_loss" in loss, "total_loss should be present."
    assert loss["classification_loss"].item() == 0.0, "classification_loss should be zero."
    assert loss["bbox_loss"].item() == 0.0, "bbox_loss should be zero."
    assert loss["total_loss"].item() == 0.0, "total_loss should be zero."

def test_compute_loss_incompatible_shapes(default_loss_config):
    """Test compute_loss with incompatible shapes between outputs and targets."""
    outputs = {
        'logits': torch.randn(2, 3, requires_grad=True),
        'pred_boxes': torch.randn(2, 4, requires_grad=True)
    }
    targets = [
        {'labels': torch.tensor([1], dtype=torch.long), 'boxes': torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)},  # 1 label
        {'labels': torch.tensor([2, 3, 4], dtype=torch.long), 'boxes': torch.tensor([[20.0, 20.0, 30.0, 30.0],
                                                                                      [40.0, 40.0, 50.0, 50.0],
                                                                                      [60.0, 60.0, 70.0, 70.0]], dtype=torch.float32)}  # 3 labels
    ]

    with pytest.raises(ValueError) as exc_info:
        compute_loss(outputs, targets, model=None, loss_config=default_loss_config)
    assert "Expected input batch_size" in str(exc_info.value), "Expected ValueError due to batch size mismatch."

def test_compute_loss_non_tensor_targets(default_loss_config):
    """Test compute_loss with non-tensor targets."""
    outputs = {
        'logits': torch.randn(2, 3, dtype=torch.float32, requires_grad=True),
        'pred_boxes': torch.randn(2, 4, dtype=torch.float32, requires_grad=True)
    }
    targets = [
        {'labels': [1, 2], 'boxes': [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]},  # Lists instead of tensors
        {'labels': [3, 4], 'boxes': [[40.0, 40.0, 50.0, 50.0], [60.0, 60.0, 70.0, 70.0]]}
    ]

    with pytest.raises(TypeError) as exc_info:
        compute_loss(outputs, targets, model=None, loss_config=default_loss_config)
    assert "expected Tensor as element" in str(exc_info.value), "Expected TypeError due to non-tensor targets."


def test_compute_loss_large_batch(default_loss_config):
    """Test compute_loss with a large batch size."""
    batch_size = 1000
    num_classes = 10
    outputs = {
        'logits': torch.randn(batch_size, num_classes, dtype=torch.float32, requires_grad=True),
        'pred_boxes': torch.randn(batch_size, 4, dtype=torch.float32, requires_grad=True)
    }
    targets = [
        {'labels': torch.randint(0, num_classes, (1,), dtype=torch.long), 'boxes': torch.randn(1, 4)}
        for _ in range(batch_size)
    ]

    loss = compute_loss(outputs, targets, model=None, loss_config=default_loss_config)

    assert "classification_loss" in loss, "classification_loss should be present."
    assert "bbox_loss" in loss, "bbox_loss should be present."
    assert "total_loss" in loss, "total_loss should be present."
    assert loss["total_loss"].item() >= 0, "total_loss should be non-negative."

def test_compute_loss_gradient_computation(mock_outputs, mock_targets, default_loss_config):
    """Test that gradients are correctly computed for the loss."""
    loss = compute_loss(mock_outputs, mock_targets, model=None, loss_config=default_loss_config)
    loss["total_loss"].backward()
    
    assert mock_outputs['logits'].grad is not None, "Gradients for logits should be computed."
    assert mock_outputs['pred_boxes'].grad is not None, "Gradients for pred_boxes should be computed."
    assert mock_outputs['logits'].grad.shape == mock_outputs['logits'].shape, "Gradients shape mismatch for logits."
    assert mock_outputs['pred_boxes'].grad.shape == mock_outputs['pred_boxes'].shape, "Gradients shape mismatch for pred_boxes."

def test_compute_loss_loss_kwargs(default_loss_config, mock_outputs, mock_targets):
    """Test that LossFunctionFactory correctly passes kwargs to the loss function."""
    # Assuming logits have 2 classes
    weight = torch.tensor([1.0, 2.0], dtype=torch.float32)  # Correct size matching number of classes (2)
    loss_config = {
        "loss_types": {
            "classification_loss": "cross_entropy",
            "bbox_loss": "smooth_l1"
        },
        "loss_weights": {
            "classification_loss": 2.0,
            "bbox_loss": 0.5
        },
        "loss_kwargs": {
            "classification_loss": {"weight": weight},
            "bbox_loss": {}
        }
    }

    loss = compute_loss(mock_outputs, mock_targets, model=None, loss_config=loss_config)

    # Compute expected individual losses with weights
    expected_classification_loss = nn.CrossEntropyLoss(weight=weight)(mock_outputs['logits'], torch.cat([t['labels'] for t in mock_targets])).item()
    expected_bbox_loss = nn.SmoothL1Loss()(mock_outputs['pred_boxes'], torch.cat([t['boxes'] for t in mock_targets])).item()

    # Apply weights
    expected_classification_loss *= 2.0
    expected_bbox_loss *= 0.5
    expected_total_loss = expected_classification_loss + expected_bbox_loss

    # Use torch.isclose for comparison
    assert "classification_loss" in loss, "classification_loss should be present."
    assert "bbox_loss" in loss, "bbox_loss should be present."
    assert "total_loss" in loss, "total_loss should be present."

    assert torch.isclose(torch.tensor(loss["classification_loss"].item()), torch.tensor(expected_classification_loss), atol=1e-6), "classification_loss weighting mismatch."
    assert torch.isclose(torch.tensor(loss["bbox_loss"].item()), torch.tensor(expected_bbox_loss), atol=1e-6), "bbox_loss weighting mismatch."
    assert torch.isclose(torch.tensor(loss["total_loss"].item()), torch.tensor(expected_total_loss), atol=1e-6), "total_loss aggregation mismatch."

def test_compute_loss_full_integration(default_loss_config, mock_outputs, mock_targets):
    """Integration test for compute_loss with realistic inputs."""
    loss_config = {
        "loss_types": {
            "classification_loss": "cross_entropy",
            "bbox_loss": "smooth_l1"
        },
        "loss_weights": {
            "classification_loss": 1.0,
            "bbox_loss": 1.0
        },
        "loss_kwargs": {
            "classification_loss": {},
            "bbox_loss": {}
        }
    }

    loss = compute_loss(mock_outputs, mock_targets, model=None, loss_config=loss_config)

    assert "classification_loss" in loss, "classification_loss should be present."
    assert "bbox_loss" in loss, "bbox_loss should be present."
    assert "total_loss" in loss, "total_loss should be present."
    assert loss["total_loss"] >= 0, "total_loss should be non-negative."

    # Ensure that loss tensors require gradients
    assert loss["classification_loss"].requires_grad, "classification_loss should require gradients."
    assert loss["bbox_loss"].requires_grad, "bbox_loss should require gradients."
    assert loss["total_loss"].requires_grad, "total_loss should require gradients."
    

def test_compute_loss_negative_weights(mock_outputs, mock_targets):
    """Test compute_loss with negative loss weights."""
    loss_config = {
        "loss_types": {
            "classification_loss": "cross_entropy",
            "bbox_loss": "smooth_l1"
        },
        "loss_weights": {
            "classification_loss": -1.0,  # Negative weight
            "bbox_loss": 1.0
        },
        "loss_kwargs": {}
    }

    loss = compute_loss(mock_outputs, mock_targets, model=None, loss_config=loss_config)

    # Compute expected individual losses
    expected_classification_loss = nn.CrossEntropyLoss()(mock_outputs['logits'], torch.cat([t['labels'] for t in mock_targets])).item()
    expected_bbox_loss = nn.SmoothL1Loss()(mock_outputs['pred_boxes'], torch.cat([t['boxes'] for t in mock_targets])).item()

    # Apply weights
    expected_classification_loss *= -1.0
    expected_bbox_loss *= 1.0
    expected_total_loss = expected_classification_loss + expected_bbox_loss

    # Use torch.isclose for comparison
    assert "classification_loss" in loss, "classification_loss should be present."
    assert "bbox_loss" in loss, "bbox_loss should be present."
    assert "total_loss" in loss, "total_loss should be present."

    assert torch.isclose(torch.tensor(loss["classification_loss"].item()), torch.tensor(expected_classification_loss), atol=1e-6), "classification_loss weighting mismatch with negative weight."
    assert torch.isclose(torch.tensor(loss["bbox_loss"].item()), torch.tensor(expected_bbox_loss), atol=1e-6), "bbox_loss weighting mismatch."
    assert torch.isclose(torch.tensor(loss["total_loss"].item()), torch.tensor(expected_total_loss), atol=1e-6), "total_loss aggregation mismatch with negative weight."

def test_compute_loss_non_standard_loss_type():
    """Test compute_loss with a loss type that doesn't fit standard patterns."""
    loss_config = {
        "loss_types": {
            "classification_loss": "l1",
            "bbox_loss": "mse"
        },
        "loss_weights": {
            "classification_loss": 1.0,
            "bbox_loss": 2.0
        },
        "loss_kwargs": {}
    }

    outputs = {
        'logits': torch.randn(4, 5, requires_grad=True),  # 4 predictions, 5 classes
        'pred_boxes': torch.randn(4, 4, requires_grad=True)  # 4 predictions
    }
    targets = [
        {'labels': torch.randn(1, 5), 'boxes': torch.randn(1, 4)},
        {'labels': torch.randn(1, 5), 'boxes': torch.randn(1, 4)},
        {'labels': torch.randn(1, 5), 'boxes': torch.randn(1, 4)},
        {'labels': torch.randn(1, 5), 'boxes': torch.randn(1, 4)}
    ]

    loss = compute_loss(outputs, targets, model=None, loss_config=loss_config)

    # Prepare labels as L1Loss expects matching shapes
    labels = torch.cat([t['labels'] for t in targets]).to(outputs['logits'].device).float()

    # Compute expected individual losses
    classification_loss = nn.L1Loss()(outputs['logits'], labels).item()
    bbox_loss = nn.MSELoss()(outputs['pred_boxes'], torch.cat([t['boxes'] for t in targets])).item()

    # Apply weights
    expected_classification_loss = classification_loss * 1.0
    expected_bbox_loss = bbox_loss * 2.0
    expected_total_loss = expected_classification_loss + expected_bbox_loss

    # Use torch.isclose for comparison
    assert "classification_loss" in loss, "classification_loss should be present."
    assert "bbox_loss" in loss, "bbox_loss should be present."
    assert "total_loss" in loss, "total_loss should be present."

    assert torch.isclose(torch.tensor(loss["classification_loss"].item()), torch.tensor(expected_classification_loss), atol=1e-6), "classification_loss weighting mismatch."
    assert torch.isclose(torch.tensor(loss["bbox_loss"].item()), torch.tensor(expected_bbox_loss), atol=1e-6), "bbox_loss weighting mismatch."
    assert torch.isclose(torch.tensor(loss["total_loss"].item()), torch.tensor(expected_total_loss), atol=1e-6), "total_loss aggregation mismatch."

def test_compute_loss_incorrect_loss_kwargs():
    """Test compute_loss with incorrect loss_kwargs that should raise an error."""
    loss_config = {
        "loss_types": {
            "classification_loss": "cross_entropy",
            "bbox_loss": "smooth_l1"
        },
        "loss_weights": {
            "classification_loss": 1.0,
            "bbox_loss": 1.0
        },
        "loss_kwargs": {
            "classification_loss": {"invalid_param": True},  # Invalid parameter for CrossEntropyLoss
            "bbox_loss": {}
        }
    }

    with pytest.raises(TypeError) as exc_info:
        compute_loss(
            outputs={'logits': torch.randn(2, 3, requires_grad=True)},
            targets=[
                {'labels': torch.tensor([1, 2], dtype=torch.long)}
            ],
            model=None,
            loss_config=loss_config
        )
    
    # Optionally, verify the exception message contains information about the invalid parameter
    assert "invalid_param" in str(exc_info.value), "TypeError should mention the invalid parameter."

def test_compute_loss_non_tensor_inputs():
    """Test compute_loss with non-tensor inputs to ensure proper error handling."""
    outputs = {
        'logits': [[0.1, 0.9], [0.8, 0.2]],  # Lists instead of tensors
        'pred_boxes': [[[10, 10, 50, 50]], [[20, 20, 60, 60]]]  # Lists instead of tensors
    }
    targets = [
        {'labels': [1, 2], 'boxes': [[10, 10, 50, 50]]},  # Lists instead of tensors
        {'labels': [3, 4], 'boxes': [[20, 20, 60, 60]]}
    ]
    
    loss_config = {
        "loss_type": "cross_entropy",
        "loss_weights": {
            "classification_loss": 1.0,
            "bbox_loss": 1.0
        },
        "loss_kwargs": {}
    }
    
    with pytest.raises(AttributeError):
        compute_loss(outputs, targets, model=None, loss_config=loss_config)


def test_compute_loss_logging(mock_outputs, mock_targets, default_loss_config, caplog):
    """Test that compute_loss logs the computed losses."""
    with caplog.at_level(logging.DEBUG):
        compute_loss(mock_outputs, mock_targets, model=None, loss_config=default_loss_config)
    
    assert "Classification loss computed" in caplog.text, "Did not log classification loss."
    assert "Bounding box loss computed" in caplog.text, "Did not log bounding box loss."


def test_compute_loss_reproducibility():
    """Test that compute_loss produces consistent results with the same inputs."""
    loss_config = {
        "loss_types": {
            "classification_loss": "cross_entropy",
            "bbox_loss": "smooth_l1"
        },
        "loss_weights": {
            "classification_loss": 1.0,
            "bbox_loss": 1.0
        },
        "loss_kwargs": {}
    }

    outputs = {
        'logits': torch.tensor([[0.2, 0.8], [0.6, 0.4]], dtype=torch.float32, requires_grad=True),
        'pred_boxes': torch.tensor(
            [[[50.0, 50.0, 150.0, 150.0]], [[30.0, 30.0, 100.0, 100.0]]],
            dtype=torch.float32,
            requires_grad=True
        )
    }
    targets = [
        {'labels': torch.tensor([1], dtype=torch.long), 'boxes': torch.tensor([[50.0, 50.0, 150.0, 150.0]], dtype=torch.float32)},
        {'labels': torch.tensor([0], dtype=torch.long), 'boxes': torch.tensor([[30.0, 30.0, 100.0, 100.0]], dtype=torch.float32)}
    ]
    
    loss1 = compute_loss(outputs, targets, model=None, loss_config=loss_config)
    loss2 = compute_loss(outputs, targets, model=None, loss_config=loss_config)

    assert loss1["classification_loss"].item() == loss2["classification_loss"].item(), "Classification losses should be identical."
    assert loss1["bbox_loss"].item() == loss2["bbox_loss"].item(), "Bounding box losses should be identical."
    assert loss1["total_loss"].item() == loss2["total_loss"].item(), "Total losses should be identical."


def test_compute_loss_multiple_predictions_per_sample(default_loss_config):
    """Test compute_loss with multiple predictions per sample."""
    loss_config = {
        "loss_types": {
            "classification_loss": "cross_entropy",
            "bbox_loss": "smooth_l1"
        },
        "loss_weights": {
            "classification_loss": 1.0,
            "bbox_loss": 1.0
        },
        "loss_kwargs": {
            "classification_loss": {},
            "bbox_loss": {}
        }
    }

    outputs = {
        'logits': torch.tensor([
            [0.1, 0.9, 0.0],
            [0.3, 0.7, 0.0],
            [0.5, 0.5, 0.0],
            [0.2, 0.8, 0.0]
        ], dtype=torch.float32, requires_grad=True),  # 4 predictions, 3 classes
        'pred_boxes': torch.tensor([
            [10.0, 10.0, 50.0, 50.0],
            [20.0, 20.0, 60.0, 60.0],
            [30.0, 30.0, 70.0, 70.0],
            [40.0, 40.0, 80.0, 80.0]
        ], dtype=torch.float32, requires_grad=True)  # 4 predictions
    }
    targets = [
        {'labels': torch.tensor([1, 2], dtype=torch.long), 'boxes': torch.tensor([[10.0, 10.0, 50.0, 50.0],
                                                                                   [20.0, 20.0, 60.0, 60.0]], dtype=torch.float32)},
        {'labels': torch.tensor([0, 1], dtype=torch.long), 'boxes': torch.tensor([[30.0, 30.0, 70.0, 70.0],
                                                                                   [40.0, 40.0, 80.0, 80.0]], dtype=torch.float32)}
    ]

    loss = compute_loss(outputs, targets, model=None, loss_config=loss_config)

    # Compute expected individual losses
    classification_loss = nn.CrossEntropyLoss()(outputs['logits'], torch.cat([t['labels'] for t in targets])).item()
    bbox_loss = nn.SmoothL1Loss()(outputs['pred_boxes'], torch.cat([t['boxes'] for t in targets])).item()

    # Apply weights
    expected_classification_loss = classification_loss * 1.0
    expected_bbox_loss = bbox_loss * 1.0
    expected_total_loss = expected_classification_loss + expected_bbox_loss

    # Use torch.isclose for comparison
    assert "classification_loss" in loss, "classification_loss should be present."
    assert "bbox_loss" in loss, "bbox_loss should be present."
    assert "total_loss" in loss, "total_loss should be present."

    assert torch.isclose(torch.tensor(loss["classification_loss"].item()), torch.tensor(expected_classification_loss), atol=1e-6), "classification_loss weighting mismatch."
    assert torch.isclose(torch.tensor(loss["bbox_loss"].item()), torch.tensor(expected_bbox_loss), atol=1e-6), "bbox_loss weighting mismatch."
    assert torch.isclose(torch.tensor(loss["total_loss"].item()), torch.tensor(expected_total_loss), atol=1e-6), "total_loss should be the sum of individual losses."

def test_compute_loss_incorrect_device(mock_outputs, mock_targets, default_loss_config):
    """Test compute_loss when outputs are on a different device."""
    # Move outputs to CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    outputs = {
        'logits': mock_outputs['logits'].to(device),
        'pred_boxes': mock_outputs['pred_boxes'].to(device)
    }
    targets = [
        {'labels': t['labels'].to(device), 'boxes': t['boxes'].to(device)}
        for t in mock_targets
    ]
    
    loss = compute_loss(outputs, targets, model=None, loss_config=default_loss_config)
    
    assert "classification_loss" in loss, "classification_loss should be present."
    assert "bbox_loss" in loss, "bbox_loss should be present."
    assert "total_loss" in loss, "total_loss should be present."
    assert loss["total_loss"].device.type == device, "total_loss should be on the correct device."

