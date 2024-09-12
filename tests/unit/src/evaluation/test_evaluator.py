# tests/unit/src/evaluation/test_evaluator.py

import pytest
import torch
from unittest.mock import MagicMock
from src.evaluation.evaluator import Evaluator
from torch.utils.data import DataLoader

@pytest.fixture
def mock_model():
    """Fixture to return a mock HuggingFace model."""
    model = MagicMock()
    model.eval = MagicMock(return_value=None)  # Ensure model.eval() is callable
    model.return_value = {
        "pred_boxes": torch.rand(4, 4),  # Random bounding boxes
        "logits": torch.rand(4, 91)  # Random logits for 91 classes
    }
    return model

@pytest.fixture
def mock_dataloader():
    """Fixture to return a mock DataLoader."""
    images = [torch.rand(3, 224, 224) for _ in range(4)]  # Random 224x224 images
    targets = [{"boxes": torch.rand(3, 4), "labels": torch.randint(0, 91, (3,))} for _ in range(4)]  # Random targets
    return DataLoader(list(zip(images, targets)), batch_size=2)

def test_process_batch(mock_model, mock_dataloader):
    """Test the _process_batch function to ensure predictions and ground truths are correctly handled."""
    evaluator = Evaluator(mock_model, device='cpu', confidence_threshold=0.5)

    # Simulate batch processing
    for batch in mock_dataloader:
        images, targets = batch
        outputs = mock_model(pixel_values=images)  # Call the mock model

        predictions, ground_truths = evaluator._process_batch(outputs, targets)

        # Ensure that predictions and ground truths are correctly structured
        assert len(predictions) == len(ground_truths)
        assert "boxes" in predictions[0]
        assert "scores" in predictions[0]
        assert "labels" in ground_truths[0]

def test_evaluate(mock_model, mock_dataloader):
    """Test the evaluate function to ensure the full evaluation pipeline works."""
    evaluator = Evaluator(mock_model, device='cpu', confidence_threshold=0.5)

    # Call the evaluation method and check the output metrics
    metrics = evaluator.evaluate(mock_dataloader)
    assert isinstance(metrics, dict)
    assert "mAP" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
