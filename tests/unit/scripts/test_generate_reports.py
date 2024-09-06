# tests/unit/scripts/test_generate_reports.py

import os
import json
import pytest
from unittest import mock
from scripts.generate_reports import generate_evaluation_report
from src.models.foundation_model import HuggingFaceObjectDetectionModel
from src.utils.config_parser import ConfigParser
from src.evaluation.evaluator import Evaluator
from src.data.dataloader import get_dataloader

# Core function test: Generate evaluation report
@mock.patch("scripts.generate_reports.HuggingFaceObjectDetectionModel.load")
@mock.patch("os.makedirs")
@mock.patch("json.dump")
@mock.patch("builtins.open", new_callable=mock.mock_open)
@mock.patch("src.evaluation.evaluator.Evaluator.evaluate")
@mock.patch("src.data.dataloader.get_dataloader")
@mock.patch("src.utils.config_parser.ConfigParser")
def test_generate_evaluation_report(mock_config_parser, mock_dataloader, mock_evaluate, mock_open, mock_json_dump, mock_makedirs, mock_model_load):
    """Test generating the evaluation report and saving it to the output directory."""
    mock_model = mock.Mock()
    mock_model_load.return_value = mock_model
    mock_dataloader.return_value = mock.Mock()  # Mock dataloader
    mock_evaluate.return_value = {"mAP": 0.75}  # Mock evaluation metrics
    mock_config_parser().get.side_effect = ["detr_model", 91, 8]  # Mock config values

    # Call generate_evaluation_report
    generate_evaluation_report(
        model_checkpoint="fake_checkpoint.pt",
        config_path="fake_config.yml",
        data_dir="fake_data_dir",
        output_dir="output/reports",
        device="cuda"
    )

    # Ensure the dataloader, model loading, and evaluation were called
    mock_model_load.assert_called_once_with("fake_checkpoint.pt")
    mock_dataloader.assert_called_once_with(data_dir="fake_data_dir", batch_size=8, mode="val", feature_extractor=None)
    mock_evaluate.assert_called_once()

    # Ensure the output directory is created
    mock_makedirs.assert_called_once_with("output/reports", exist_ok=True)

    # Ensure the evaluation report is saved
    mock_open.assert_called_once_with("output/reports/evaluation_report.json", "w")
    mock_json_dump.assert_called_once_with({"mAP": 0.75}, mock_open(), indent=4)


# Edge case: Missing model checkpoint
@mock.patch("scripts.generate_reports.HuggingFaceObjectDetectionModel.load")
def test_generate_evaluation_report_missing_checkpoint(mock_model_load):
    """Test handling when the model checkpoint is missing."""
    mock_model_load.side_effect = FileNotFoundError("Checkpoint not found")

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        generate_evaluation_report(
            model_checkpoint="missing_checkpoint.pt",
            config_path="fake_config.yml",
            data_dir="fake_data_dir",
            output_dir="output/reports",
            device="cuda"
        )


# Edge case: Empty dataset
@mock.patch("scripts.generate_reports.HuggingFaceObjectDetectionModel.load")
@mock.patch("src.data.dataloader.get_dataloader")
def test_generate_evaluation_report_empty_dataset(mock_dataloader, mock_model_load):
    """Test handling when the dataset is empty."""
    mock_model_load.return_value = mock.Mock()
    mock_dataloader.return_value = []  # Mock empty dataloader

    with pytest.raises(ValueError, match="Empty dataset provided"):
        generate_evaluation_report(
            model_checkpoint="fake_checkpoint.pt",
            config_path="fake_config.yml",
            data_dir="fake_data_dir",
            output_dir="output/reports",
            device="cuda"
        )


# Edge case: Invalid configuration file
@mock.patch("scripts.generate_reports.HuggingFaceObjectDetectionModel.load")
@mock.patch("src.utils.config_parser.ConfigParser")
def test_generate_evaluation_report_invalid_config(mock_config_parser, mock_model_load):
    """Test handling invalid configuration files."""
    mock_model_load.return_value = mock.Mock()
    mock_config_parser.side_effect = FileNotFoundError("Configuration file not found")

    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        generate_evaluation_report(
            model_checkpoint="fake_checkpoint.pt",
            config_path="invalid_config.yml",
            data_dir="fake_data_dir",
            output_dir="output/reports",
            device="cuda"
        )


# Mocking device usage: CPU vs. CUDA
@mock.patch("scripts.generate_reports.HuggingFaceObjectDetectionModel.load")
@mock.patch("src.evaluation.evaluator.Evaluator.evaluate")
def test_generate_evaluation_report_device(mock_evaluate, mock_model_load):
    """Test that the evaluation runs correctly on CPU vs. CUDA."""
    mock_model = mock.Mock()
    mock_model_load.return_value = mock_model
    mock_evaluate.return_value = {"mAP": 0.75}

    # Run on CPU
    generate_evaluation_report(
        model_checkpoint="fake_checkpoint.pt",
        config_path="fake_config.yml",
        data_dir="fake_data_dir",
        output_dir="output/reports",
        device="cpu"
    )
    mock_model.to.assert_called_with("cpu")

    # Run on CUDA
    generate_evaluation_report(
        model_checkpoint="fake_checkpoint.pt",
        config_path="fake_config.yml",
        data_dir="fake_data_dir",
        output_dir="output/reports",
        device="cuda"
    )
    mock_model.to.assert_called_with("cuda")


# Edge case: Permission error when saving report
@mock.patch("scripts.generate_reports.HuggingFaceObjectDetectionModel.load")
@mock.patch("os.makedirs")
@mock.patch("builtins.open", new_callable=mock.mock_open)
@mock.patch("json.dump")
def test_generate_evaluation_report_permission_error(mock_json_dump, mock_open, mock_makedirs, mock_model_load):
    """Test handling permission errors when saving the report."""
    mock_model = mock.Mock()
    mock_model_load.return_value = mock_model

    # Mock permission error when saving the report
    mock_open.side_effect = PermissionError("Permission denied")

    with pytest.raises(PermissionError, match="Permission denied"):
        generate_evaluation_report(
            model_checkpoint="fake_checkpoint.pt",
            config_path="fake_config.yml",
            data_dir="fake_data_dir",
            output_dir="/restricted_dir/reports",
            device="cuda"
        )

