# tests/unit/scripts/test_export_model.py

import os
import pytest
import torch
from unittest import mock
from scripts.export_model import export_model
from transformers import DetrForObjectDetection
from src.utils.config_parser import ConfigParser
from src.models.foundation_model import HuggingFaceObjectDetectionModel
from src.utils.system_utils import check_device

# Core function test: TorchScript export
@mock.patch("scripts.export_model.HuggingFaceObjectDetectionModel.load")
@mock.patch("torch.jit.script")
@mock.patch("os.makedirs")
def test_export_model_torchscript(mock_makedirs, mock_jit_script, mock_model_load):
    """Test successful export to TorchScript format."""
    mock_model = mock.Mock()
    mock_model_load.return_value = mock_model
    mock_jit_script.return_value = mock.Mock()  # Mock the scripted model

    model_checkpoint = "fake_checkpoint.pt"
    output_path = "output/model_scripted.pt"
    config_path = "fake_config.yml"
    device = "cpu"

    # Call the export_model function
    export_model(
        model_checkpoint=model_checkpoint,
        output_path=output_path,
        config_path=config_path,
        device=device
    )

    # Ensure necessary directory is created
    mock_makedirs.assert_called_once_with(os.path.dirname(output_path), exist_ok=True)

    # Ensure the model is loaded and scripted
    mock_model_load.assert_called_once_with(model_checkpoint)
    mock_jit_script.assert_called_once_with(mock_model.model)

    # Ensure the scripted model is saved
    mock_jit_script().save.assert_called_once_with(output_path)


# Edge case: Missing model checkpoint
@mock.patch("scripts.export_model.HuggingFaceObjectDetectionModel.load")
def test_export_model_missing_checkpoint(mock_model_load):
    """Test handling when the model checkpoint is missing."""
    mock_model_load.side_effect = FileNotFoundError("Checkpoint not found")

    model_checkpoint = "missing_checkpoint.pt"
    output_path = "output/model_scripted.pt"
    config_path = "fake_config.yml"
    device = "cpu"

    # Expecting FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        export_model(
            model_checkpoint=model_checkpoint,
            output_path=output_path,
            config_path=config_path,
            device=device
        )


# Edge case: Unsupported device (e.g., typo in device name)
@mock.patch("scripts.export_model.HuggingFaceObjectDetectionModel.load")
def test_export_model_invalid_device(mock_model_load):
    """Test handling of invalid device."""
    mock_model = mock.Mock()
    mock_model_load.return_value = mock_model

    model_checkpoint = "fake_checkpoint.pt"
    output_path = "output/model_scripted.pt"
    config_path = "fake_config.yml"
    device = "invalid_device"  # Typo or unsupported device

    # Expecting RuntimeError due to invalid device
    with pytest.raises(RuntimeError, match="Invalid device"):
        export_model(
            model_checkpoint=model_checkpoint,
            output_path=output_path,
            config_path=config_path,
            device=device
        )


# Edge case: Permission error when saving the exported model
@mock.patch("scripts.export_model.HuggingFaceObjectDetectionModel.load")
@mock.patch("torch.jit.script")
@mock.patch("os.makedirs")
def test_export_model_permission_error(mock_makedirs, mock_jit_script, mock_model_load):
    """Test permission error when trying to save the exported model."""
    mock_model = mock.Mock()
    mock_model_load.return_value = mock_model
    mock_jit_script.return_value = mock.Mock()

    model_checkpoint = "fake_checkpoint.pt"
    output_path = "/restricted_dir/model_scripted.pt"  # Path with no write permissions
    config_path = "fake_config.yml"
    device = "cpu"

    # Mock an IOError when saving the model
    mock_jit_script().save.side_effect = PermissionError("Permission denied")

    # Expecting PermissionError
    with pytest.raises(PermissionError, match="Permission denied"):
        export_model(
            model_checkpoint=model_checkpoint,
            output_path=output_path,
            config_path=config_path,
            device=device
        )


# Mocking external dependencies: ConfigParser and filesystem
@mock.patch("scripts.export_model.HuggingFaceObjectDetectionModel.load")
@mock.patch("torch.jit.script")
@mock.patch("os.makedirs")
@mock.patch("src.utils.config_parser.ConfigParser")
def test_export_model_with_config(mock_config_parser, mock_makedirs, mock_jit_script, mock_model_load):
    """Test export with mocked ConfigParser."""
    mock_model = mock.Mock()
    mock_model_load.return_value = mock_model
    mock_config_parser().get.return_value = "detr_model"  # Mock model name

    model_checkpoint = "fake_checkpoint.pt"
    output_path = "output/model_scripted.pt"
    config_path = "fake_config.yml"
    device = "cpu"

    # Call the export_model function
    export_model(
        model_checkpoint=model_checkpoint,
        output_path=output_path,
        config_path=config_path,
        device=device
    )

    # Ensure the config parser is used correctly
    mock_config_parser.assert_called_once_with(config_path)
    mock_config_parser().get.assert_called_with("model_name")


# Performance test: Export on CPU vs GPU
@mock.patch("scripts.export_model.HuggingFaceObjectDetectionModel.load")
@mock.patch("torch.jit.script")
@mock.patch("os.makedirs")
def test_export_model_cpu_vs_gpu(mock_makedirs, mock_jit_script, mock_model_load):
    """Test export performance on both CPU and GPU."""
    mock_model = mock.Mock()
    mock_model_load.return_value = mock_model

    model_checkpoint = "fake_checkpoint.pt"
    output_path = "output/model_scripted.pt"
    config_path = "fake_config.yml"

    # Export on CPU
    export_model(
        model_checkpoint=model_checkpoint,
        output_path=output_path,
        config_path=config_path,
        device="cpu"
    )
    mock_model.to.assert_called_with("cpu")

    # Export on GPU
    export_model(
        model_checkpoint=model_checkpoint,
        output_path=output_path,
        config_path=config_path,
        device="cuda"
    )
    mock_model.to.assert_called_with("cuda")


# Edge case: Missing config file
@mock.patch("scripts.export_model.HuggingFaceObjectDetectionModel.load")
@mock.patch("os.makedirs")
def test_export_model_missing_config(mock_makedirs, mock_model_load):
    """Test handling when the configuration file is missing."""
    mock_model = mock.Mock()
    mock_model_load.return_value = mock_model

    model_checkpoint = "fake_checkpoint.pt"
    output_path = "output/model_scripted.pt"
    config_path = "missing_config.yml"  # Non-existent config file
    device = "cpu"

    # Expecting FileNotFoundError due to missing config
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        export_model(
            model_checkpoint=model_checkpoint,
            output_path=output_path,
            config_path=config_path,
            device=device
        )
