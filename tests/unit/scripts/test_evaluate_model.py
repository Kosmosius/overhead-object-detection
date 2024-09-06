# tests/unit/scripts/test_evaluate_model.py

import pytest
import torch
from unittest import mock
from scripts.evaluate_model import evaluate_model
from src.models.model_builder import load_model_from_checkpoint
from src.data.dataloader import get_dataloader
from src.evaluation.evaluator import Evaluator
from src.utils.config_parser import ConfigParser
from src.utils.system_utils import check_device
from src.utils.logging import setup_logging

# Mock logging setup for the evaluation
@mock.patch("src.utils.logging.setup_logging")
def test_evaluate_model_logging(mock_logging):
    """Test that logging is set up correctly during evaluation."""
    model_checkpoint = "fake_checkpoint.pt"
    data_dir = "/path/to/dataset"
    config_path = "/path/to/config"
    batch_size = 4
    device = "cpu"
    iou_threshold = 0.5

    # Call the evaluate_model function
    evaluate_model(
        model_checkpoint=model_checkpoint,
        data_dir=data_dir,
        config_path=config_path,
        batch_size=batch_size,
        device=device,
        iou_threshold=iou_threshold,
    )

    # Assert that logging was set up once
    mock_logging.assert_called_once()

# Mock model loading
@mock.patch("src.models.model_builder.load_model_from_checkpoint")
@mock.patch("src.data.dataloader.get_dataloader")
@mock.patch("src.evaluation.evaluator.Evaluator.evaluate")
def test_evaluate_model_core_functions(mock_evaluate, mock_dataloader, mock_load_model):
    """Test that model loading, dataloader, and evaluation functions are called."""
    mock_model = mock.Mock()  # Mock the model object
    mock_load_model.return_value = mock_model
    mock_evaluate.return_value = {"mAP": 0.8}  # Mock the evaluation result

    model_checkpoint = "fake_checkpoint.pt"
    data_dir = "/path/to/dataset"
    config_path = "/path/to/config"
    batch_size = 4
    device = "cpu"
    iou_threshold = 0.5

    # Call the evaluate_model function
    evaluate_model(
        model_checkpoint=model_checkpoint,
        data_dir=data_dir,
        config_path=config_path,
        batch_size=batch_size,
        device=device,
        iou_threshold=iou_threshold,
    )

    # Assert that the model was loaded correctly
    mock_load_model.assert_called_once_with(model_checkpoint, mock.ANY, num_labels=mock.ANY, device=device)

    # Assert that the dataloader was called correctly
    mock_dataloader.assert_called_once_with(data_dir=data_dir, batch_size=batch_size, mode='val', feature_extractor=mock.ANY)

    # Assert that the evaluation was run
    mock_evaluate.assert_called_once()

# Edge case: Incomplete evaluation data (e.g., corrupted or missing images)
@mock.patch("src.models.model_builder.load_model_from_checkpoint")
@mock.patch("src.data.dataloader.get_dataloader")
@mock.patch("src.evaluation.evaluator.Evaluator.evaluate")
def test_evaluate_model_incomplete_data(mock_evaluate, mock_dataloader, mock_load_model):
    """Test evaluation with incomplete or corrupted evaluation data."""
    mock_model = mock.Mock()
    mock_load_model.return_value = mock_model
    mock_evaluate.side_effect = ValueError("Corrupted or missing data")

    model_checkpoint = "fake_checkpoint.pt"
    data_dir = "/path/to/corrupted/dataset"
    config_path = "/path/to/config"
    batch_size = 4
    device = "cpu"
    iou_threshold = 0.5

    # Expecting ValueError due to corrupted data
    with pytest.raises(ValueError, match="Corrupted or missing data"):
        evaluate_model(
            model_checkpoint=model_checkpoint,
            data_dir=data_dir,
            config_path=config_path,
            batch_size=batch_size,
            device=device,
            iou_threshold=iou_threshold,
        )

# Edge case: Model with no predictions
@mock.patch("src.models.model_builder.load_model_from_checkpoint")
@mock.patch("src.data.dataloader.get_dataloader")
@mock.patch("src.evaluation.evaluator.Evaluator.evaluate")
def test_evaluate_model_no_predictions(mock_evaluate, mock_dataloader, mock_load_model):
    """Test evaluation with a model that outputs no predictions."""
    mock_model = mock.Mock()
    mock_load_model.return_value = mock_model
    mock_evaluate.return_value = {"mAP": 0.0}  # No predictions

    model_checkpoint = "fake_checkpoint.pt"
    data_dir = "/path/to/dataset"
    config_path = "/path/to/config"
    batch_size = 4
    device = "cpu"
    iou_threshold = 0.5

    result = evaluate_model(
        model_checkpoint=model_checkpoint,
        data_dir=data_dir,
        config_path=config_path,
        batch_size=batch_size,
        device=device,
        iou_threshold=iou_threshold,
    )

    assert result["mAP"] == 0.0, "Expected mAP to be 0.0 for models with no predictions"

# Edge case: Mismatched input shapes between model and dataset
@mock.patch("src.models.model_builder.load_model_from_checkpoint")
@mock.patch("src.data.dataloader.get_dataloader")
@mock.patch("src.evaluation.evaluator.Evaluator.evaluate")
def test_evaluate_model_mismatched_input_shapes(mock_evaluate, mock_dataloader, mock_load_model):
    """Test evaluation with mismatched input shapes between model and dataset."""
    mock_model = mock.Mock()
    mock_load_model.return_value = mock_model
    mock_evaluate.side_effect = RuntimeError("Input shape mismatch")

    model_checkpoint = "fake_checkpoint.pt"
    data_dir = "/path/to/dataset"
    config_path = "/path/to/config"
    batch_size = 4
    device = "cpu"
    iou_threshold = 0.5

    # Expecting RuntimeError due to input shape mismatch
    with pytest.raises(RuntimeError, match="Input shape mismatch"):
        evaluate_model(
            model_checkpoint=model_checkpoint,
            data_dir=data_dir,
            config_path=config_path,
            batch_size=batch_size,
            device=device,
            iou_threshold=iou_threshold,
        )

# Performance test: Large dataset evaluation
@mock.patch("src.models.model_builder.load_model_from_checkpoint")
@mock.patch("src.data.dataloader.get_dataloader")
@mock.patch("src.evaluation.evaluator.Evaluator.evaluate")
def test_evaluate_model_performance(mock_evaluate, mock_dataloader, mock_load_model):
    """Test evaluation performance with a large dataset."""
    mock_model = mock.Mock()
    mock_load_model.return_value = mock_model
    large_dataset_size = 1000000  # Simulate a large dataset size
    mock_evaluate.return_value = {"mAP": 0.75}

    model_checkpoint = "fake_checkpoint.pt"
    data_dir = "/path/to/large/dataset"
    config_path = "/path/to/config"
    batch_size = 64  # Use a larger batch size for performance
    device = "cuda"  # Assume using GPU for large-scale evaluation
    iou_threshold = 0.5

    # Call the evaluate_model function
    result = evaluate_model(
        model_checkpoint=model_checkpoint,
        data_dir=data_dir,
        config_path=config_path,
        batch_size=batch_size,
        device=device,
        iou_threshold=iou_threshold,
    )

    assert result["mAP"] == 0.75, "Expected mAP to match the mock return value"
    mock_evaluate.assert_called_once()

