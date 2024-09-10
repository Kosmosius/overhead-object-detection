# tests/unit/scripts/test_monitor_model.py

import time
import pytest
import torch
from unittest import mock
from prometheus_client import Gauge
from scripts.monitor_model import monitor_model_performance, main
from src.evaluation.evaluator import Evaluator

# Core test: Monitor model performance with mock data
@mock.patch("src.evaluation.evaluator.Evaluator.evaluate")
@mock.patch("prometheus_client.Gauge.set")
@mock.patch("time.sleep", return_value=None)  # Prevent sleep during tests
def test_monitor_model_performance(mock_sleep, mock_set, mock_evaluate):
    """Test that model performance metrics are correctly monitored and updated."""
    mock_model = mock.Mock()
    mock_dataloader = mock.Mock()
    device = torch.device("cpu")

    # Mock evaluation output
    mock_evaluate.return_value = {
        "total_loss": 0.5,
        "precision": 0.8,
        "recall": 0.75,
        "map": 0.82
    }

    # Run monitoring for a single iteration
    with mock.patch("time.time", side_effect=[0, 1]):  # Simulate time intervals
        monitor_model_performance(mock_model, mock_dataloader, device, metrics_interval=30)

    # Check that Prometheus metrics are updated correctly
    assert mock_set.call_count == 4  # Called for each metric (loss, precision, recall, map)
    mock_set.assert_any_call(0.5)  # Check for the loss metric
    mock_set.assert_any_call(0.8)  # Check for the precision metric
    mock_set.assert_any_call(0.75)  # Check for the recall metric
    mock_set.assert_any_call(0.82)  # Check for the mAP metric

# Edge case: Handling missing metrics
@mock.patch("src.evaluation.evaluator.Evaluator.evaluate")
@mock.patch("prometheus_client.Gauge.set")
@mock.patch("time.sleep", return_value=None)
def test_monitor_model_performance_missing_metrics(mock_sleep, mock_set, mock_evaluate):
    """Test how the function handles missing metrics."""
    mock_model = mock.Mock()
    mock_dataloader = mock.Mock()
    device = torch.device("cpu")

    # Mock evaluation output with missing metrics
    mock_evaluate.return_value = {
        "total_loss": None,  # Missing loss
        "precision": None,  # Missing precision
        "recall": 0.75,  # Valid recall
        "map": None  # Missing mAP
    }

    monitor_model_performance(mock_model, mock_dataloader, device, metrics_interval=30)

    # Ensure Prometheus metrics are updated with default values (0) for missing metrics
    mock_set.assert_any_call(0)  # For missing total_loss and precision
    mock_set.assert_any_call(0.75)  # For valid recall

# Edge case: Handling uninitialized model
@mock.patch("time.sleep", return_value=None)
def test_monitor_model_performance_uninitialized_model(mock_sleep):
    """Test handling of uninitialized model."""
    mock_model = None  # Simulating an uninitialized model
    mock_dataloader = mock.Mock()
    device = torch.device("cpu")

    with pytest.raises(AttributeError):
        monitor_model_performance(mock_model, mock_dataloader, device, metrics_interval=30)

# Edge case: Connection issues with Prometheus
@mock.patch("prometheus_client.start_http_server")
def test_main_prometheus_connection_failure(mock_start_http_server):
    """Test handling Prometheus connection failure."""
    # Simulate a failure in starting Prometheus server
    mock_start_http_server.side_effect = OSError("Failed to start Prometheus server")

    with pytest.raises(OSError, match="Failed to start Prometheus server"):
        main()

# Mocking test: Check Prometheus server start
@mock.patch("prometheus_client.start_http_server")
@mock.patch("scripts.monitor_model.monitor_model_performance")
@mock.patch("src.utils.config_parser.ConfigParser")
@mock.patch("src.models.foundation_model.HuggingFaceObjectDetectionModel.load")
def test_main_prometheus_server_start(mock_model_load, mock_config_parser, mock_monitor_performance, mock_start_http_server):
    """Test that the Prometheus server starts correctly."""
    mock_model = mock.Mock()
    mock_model_load.return_value = mock_model

    mock_config = mock.Mock()
    mock_config_parser.return_value = mock_config

    # Set required parameters for the mock configuration
    mock_config.get.return_value = "fake_model"

    # Run the main function (entry point)
    main()

    # Ensure Prometheus HTTP server started
    mock_start_http_server.assert_called_once_with(8000)

    # Ensure model performance monitoring started
    mock_monitor_performance.assert_called_once()

# Edge case: Invalid model checkpoint
@mock.patch("src.models.foundation_model.HuggingFaceObjectDetectionModel.load")
def test_main_invalid_model_checkpoint(mock_model_load):
    """Test handling of invalid model checkpoint."""
    # Simulate an invalid model checkpoint
    mock_model_load.side_effect = FileNotFoundError("Model checkpoint not found")

    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        main()

# Edge case: Invalid device handling
@mock.patch("scripts.monitor_model.monitor_model_performance")
@mock.patch("src.utils.config_parser.ConfigParser")
@mock.patch("src.models.foundation_model.HuggingFaceObjectDetectionModel.load")
def test_main_invalid_device(mock_model_load, mock_config_parser, mock_monitor_performance):
    """Test handling of invalid device."""
    mock_model = mock.Mock()
    mock_model_load.return_value = mock_model

    mock_config = mock.Mock()
    mock_config_parser.return_value = mock_config

    with pytest.raises(ValueError, match="Invalid device"):
        main()

# Performance test: Large-scale monitoring
@mock.patch("src.evaluation.evaluator.Evaluator.evaluate")
@mock.patch("prometheus_client.Gauge.set")
@mock.patch("time.sleep", return_value=None)
def test_monitor_performance_large_scale(mock_sleep, mock_set, mock_evaluate, benchmark):
    """Test performance when monitoring model on large datasets."""
    mock_model = mock.Mock()
    mock_dataloader = mock.Mock()
    device = torch.device("cpu")

    # Mock evaluation metrics for large datasets
    mock_evaluate.return_value = {
        "total_loss": 0.7,
        "precision": 0.85,
        "recall": 0.80,
        "map": 0.88
    }

    # Benchmark performance of monitoring
    benchmark(monitor_model_performance, mock_model, mock_dataloader, device, metrics_interval=1)
