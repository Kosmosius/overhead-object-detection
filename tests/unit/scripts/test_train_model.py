# tests/unit/scripts/test_train_model.py

import os
import pytest
import torch
from unittest import mock
from scripts.train_model import (
    parse_arguments, load_config, save_checkpoint, load_checkpoint, auto_lr_finder
)
from transformers import Trainer, TrainingArguments
from peft import PeftConfig
from src.training.peft_finetune import setup_peft_model


# Unit test for parse_arguments
def test_parse_arguments():
    """Test argument parsing with standard inputs."""
    test_args = [
        "--config", "config.yaml",
        "--resume", "checkpoint.pth",
        "--validate_every", "1",
        "--use_trainer",
        "--early_stopping", "3",
        "--gradient_accumulation_steps", "2",
        "--lr_finder",
        "--distributed"
    ]
    with mock.patch("sys.argv", ["train_model.py"] + test_args):
        args = parse_arguments()
        assert args.config == "config.yaml"
        assert args.resume == "checkpoint.pth"
        assert args.validate_every == 1
        assert args.use_trainer is True
        assert args.early_stopping == 3
        assert args.gradient_accumulation_steps == 2
        assert args.lr_finder is True
        assert args.distributed is True


# Unit test for load_config
@mock.patch('scripts.train_model.ConfigParser')  # Patch the correct import path from train_model.py
@mock.patch('scripts.train_model.os.path.exists', return_value=True)  # Patch os.path.exists in the correct module
@mock.patch('src.utils.config_parser.ConfigParser._load_yaml', return_value={"model_name": "detr", "num_classes": 5})  # Mock internal _load_yaml
def test_load_config(mock_load_yaml, mock_exists, mock_config_parser):
    """Test loading of configuration file."""
    # Mock the configuration returned by ConfigParser
    mock_config = {"model_name": "detr", "num_classes": 5}
    mock_config_parser.return_value.config = mock_config

    # Call the function being tested
    config = load_config("config.yaml")

    # Verify that os.path.exists and ConfigParser were called correctly
    mock_exists.assert_called_once_with("config.yaml")
    mock_config_parser.assert_called_once_with("config.yaml")
    mock_load_yaml.assert_called_once_with("config.yaml")

    # Assert that the config returned matches the mocked config
    assert config == mock_config


# Unit test for save_checkpoint
@mock.patch('torch.save')
@mock.patch('os.makedirs')
def test_save_checkpoint(mock_makedirs, mock_torch_save):
    """Test saving model checkpoint."""
    model = mock.MagicMock()
    optimizer = mock.MagicMock()
    scheduler = mock.MagicMock()

    save_checkpoint(model, optimizer, scheduler, epoch=5, checkpoint_dir="checkpoints")

    mock_makedirs.assert_called_once_with("checkpoints", exist_ok=True)
    mock_torch_save.assert_called_once()


# Unit test for load_checkpoint
@mock.patch('torch.load')
@mock.patch('os.path.exists', return_value=True)
def test_load_checkpoint(mock_path_exists, mock_torch_load):
    """Test loading model checkpoint."""
    model = mock.MagicMock()
    optimizer = mock.MagicMock()
    scheduler = mock.MagicMock()

    mock_torch_load.return_value = {
        "model_state_dict": mock.MagicMock(),
        "optimizer_state_dict": mock.MagicMock(),
        "scheduler_state_dict": mock.MagicMock(),
        "epoch": 5
    }

    start_epoch = load_checkpoint("checkpoint.pth", model, optimizer, scheduler)

    mock_path_exists.assert_called_once_with("checkpoint.pth")
    mock_torch_load.assert_called_once_with("checkpoint.pth")
    assert start_epoch == 6


# Unit test for handling a missing config file
@mock.patch('scripts.train_model.ConfigParser', side_effect=FileNotFoundError)
@mock.patch('scripts.train_model.os.path.exists', return_value=False)  # Patch os.path.exists
def test_load_config_missing_file(mock_exists, mock_config_parser):
    """Test loading a missing configuration file."""
    with pytest.raises(FileNotFoundError):
        load_config("invalid_config.yaml")

    # Ensure os.path.exists was called
    mock_exists.assert_called_once_with("invalid_config.yaml")
    mock_config_parser.assert_called_once_with("invalid_config.yaml")


# Unit test for auto_lr_finder
def test_auto_lr_finder():
    """Test automatic learning rate finder."""
    optimizer = mock.MagicMock()
    model = mock.MagicMock()
    dataloader = mock.MagicMock()
    config = {}

    auto_lr_finder(optimizer, model, dataloader, config)

    assert config["learning_rate"] == 1e-3


# Edge case test for invalid configuration in load_config
@mock.patch('os.path.exists', return_value=False)
@mock.patch('src.utils.config_parser.ConfigParser', side_effect=FileNotFoundError)
def test_load_config_missing_file(mock_exists, mock_config_parser):
    """Test loading a missing configuration file."""
    with pytest.raises(FileNotFoundError):
        load_config("invalid_config.yaml")
    mock_exists.assert_called_once_with("invalid_config.yaml")
    mock_config_parser.assert_called_once_with("invalid_config.yaml")


# Edge case test for unsupported PEFT configuration
@mock.patch('transformers.DetrForObjectDetection.from_pretrained', side_effect=OSError("model_name is not a valid model identifier"))
def test_unsupported_peft_model(mock_from_pretrained):
    """Test handling of unsupported PEFT model during training."""
    with pytest.raises(OSError, match="model_name is not a valid model identifier"):
        setup_peft_model("model_name", 10, PeftConfig())


# Mocking HuggingFace Trainer API
@mock.patch('transformers.Trainer.train')
@mock.patch('mlflow.utils.validation', create=True)
def test_trainer_api_integration(mock_trainer_train, mock_mlflow_validation):
    """Test integration with HuggingFace Trainer API."""
    mock_mlflow_validation.MAX_PARAM_VAL_LENGTH = 250
    model = mock.MagicMock()
    args = TrainingArguments(output_dir="output")
    trainer = Trainer(model=model, args=args)

    trainer.train()
    mock_trainer_train.assert_called_once()


# Edge case test for resuming from checkpoint
@mock.patch('torch.load')
@mock.patch('os.path.exists', return_value=True)
def test_resume_training_from_checkpoint(mock_path_exists, mock_torch_load):
    """Test resuming training from a checkpoint."""
    model = mock.MagicMock()
    optimizer = mock.MagicMock()
    scheduler = mock.MagicMock()

    mock_torch_load.return_value = {
        "model_state_dict": mock.MagicMock(),
        "optimizer_state_dict": mock.MagicMock(),
        "scheduler_state_dict": mock.MagicMock(),
        "epoch": 5
    }

    start_epoch = load_checkpoint("checkpoint.pth", model, optimizer, scheduler)
    assert start_epoch == 6


# Performance test: distributed training setup (mocked)
@mock.patch('sys.argv', ['train_model.py', '--config', 'config.yaml', '--distributed'])
@mock.patch('torch.cuda.is_available', return_value=True)
@mock.patch('torch.distributed.init_process_group')
def test_distributed_training_setup(mock_init_process_group, mock_is_available):
    """Test distributed training setup."""
    args = parse_arguments()
    assert args.config == "config.yaml"
    assert args.distributed is True

    device = torch.device("cuda" if mock_is_available() else "cpu")
    assert device == torch.device("cuda")
    mock_init_process_group.assert_called_once()

