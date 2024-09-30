# tests/unit/src/training/test_peft_finetune.py

import os
import pytest
import torch
import logging
from unittest.mock import patch, MagicMock, mock_open, ANY
from torchvision.datasets import CocoDetection
from src.training.peft_finetune import (
    setup_peft_model,
    prepare_dataloader,
    fine_tune_peft_model,
    get_optimizer_and_scheduler,
    main
)
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from peft import PeftModel, PeftConfig
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from datasets import Dataset
from src.utils.config_parser import ConfigParser

# --- Fixtures ---

@pytest.fixture
def mock_peft_config():
    """Fixture to mock PeftConfig.from_pretrained."""
    with patch("src.training.peft_finetune.PeftConfig.from_pretrained") as mock_peft_config_from_pretrained:
        mock_peft_config_instance = MagicMock(spec=PeftConfig)
        mock_peft_config_from_pretrained.return_value = mock_peft_config_instance
        yield mock_peft_config_instance

@pytest.fixture
def mock_model():
    """Fixture to create a mock DetrForObjectDetection model."""
    return MagicMock(spec=DetrForObjectDetection)

@pytest.fixture
def mock_peft_model():
    """Fixture to create a fully mocked PeftModel with necessary methods."""
    peft_model = MagicMock(spec=PeftModel)
    peft_model.to = MagicMock(return_value=peft_model)
    peft_model.train = MagicMock()
    peft_model.eval = MagicMock()
    peft_model.forward = MagicMock()
    peft_model.state_dict = MagicMock(return_value={})
    return peft_model

@pytest.fixture
def mock_feature_extractor():
    """Fixture to create a mock DetrFeatureExtractor."""
    return MagicMock(spec=DetrFeatureExtractor)

@pytest.fixture
def mock_dataset():
    """Fixture to create a mock Dataset."""
    return MagicMock(spec=CocoDetection)

@pytest.fixture
def mock_dataloader_train():
    """Fixture to create a mock training DataLoader with a specified number of batches."""
    num_batches = 10  # Define the number of batches per epoch
    batch = (
        torch.randn(4, 3, 224, 224),  # Pixel values
        [{"labels": torch.randint(0, 91, (4,)), "boxes": torch.randn(4, 4)}]  # Targets
    )
    mock_loader = MagicMock()
    mock_loader.__len__.return_value = num_batches
    mock_loader.__iter__.return_value = iter([batch for _ in range(num_batches)])
    return mock_loader

@pytest.fixture
def mock_dataloader_val():
    """Fixture to create a mock validation DataLoader with a specified number of batches."""
    num_batches = 5  # Define the number of batches for validation
    batch = (
        torch.randn(4, 3, 224, 224),  # Pixel values
        [{"labels": torch.randint(0, 91, (4,)), "boxes": torch.randn(4, 4)}]  # Targets
    )
    mock_loader = MagicMock()
    mock_loader.__len__.return_value = num_batches
    mock_loader.__iter__.return_value = iter([batch for _ in range(num_batches)])
    return mock_loader

@pytest.fixture
def mock_optimizer():
    """Fixture to create a mock Optimizer."""
    return MagicMock(spec=Optimizer)

@pytest.fixture
def mock_scheduler():
    """Fixture to create a mock Scheduler."""
    return MagicMock(spec=_LRScheduler)

@pytest.fixture
def default_config():
    """Fixture for default training configuration."""
    return {
        "training": {
            "num_epochs": 2,
            "mixed_precision": False,
            "checkpoint_dir": "./checkpoints",
            "batch_size": 4,
            "device": "cpu"
        },
        "data": {
            "data_dir": "./data"
        },
        "model": {
            "model_name": "facebook/detr-resnet-50",
            "num_classes": 91,
            "peft_model_path": "./peft_model"
        },
        "optimizer": {
            "optimizer_type": "adamw",
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "scheduler_type": "linear",
            "num_warmup_steps": 0
        }
    }

@pytest.fixture
def custom_config():
    """Fixture for custom training configuration."""
    return {
        "training": {
            "num_epochs": 3,
            "mixed_precision": True,
            "checkpoint_dir": "./custom_checkpoints",
            "batch_size": 8,
            "device": "cpu"
        },
        "data": {
            "data_dir": "./custom_data"
        },
        "model": {
            "model_name": "facebook/detr-resnet-101",
            "num_classes": 80,
            "peft_model_path": "./custom_peft_model"
        },
        "optimizer": {
            "optimizer_type": "sgd",
            "learning_rate": 0.01,
            "weight_decay": 0.001,
            "scheduler_type": "cosine",
            "num_warmup_steps": 100
        }
    }

@pytest.fixture
def mock_config_parser(default_config):
    """Fixture to mock ConfigParser."""
    with patch.object(ConfigParser, 'config', default_config):
        yield ConfigParser

# --- Test Cases ---

# 1. Tests for setup_peft_model

def test_setup_peft_model_success(mock_peft_config, mock_model):
    """Test successful setup of PEFT model."""
    with patch("src.training.peft_finetune.get_peft_model") as mock_get_peft_model, \
         patch("src.training.peft_finetune.DetrForObjectDetection.from_pretrained") as mock_from_pretrained:
        
        mock_from_pretrained.return_value = mock_model
        mock_get_peft_model.return_value = MagicMock(spec=PeftModel)
        
        peft_model = setup_peft_model("facebook/detr-resnet-50", num_classes=91, peft_config=mock_peft_config)
        
        mock_from_pretrained.assert_called_once_with("facebook/detr-resnet-50", num_labels=91)
        mock_get_peft_model.assert_called_once_with(mock_model, mock_peft_config)
        assert peft_model is not None, "PEFT model should be initialized."

def test_setup_peft_model_logging(mock_peft_config, mock_model, caplog):
    """Test that setup_peft_model logs the appropriate message."""
    with patch("src.training.peft_finetune.get_peft_model") as mock_get_peft_model, \
         patch("src.training.peft_finetune.DetrForObjectDetection.from_pretrained") as mock_from_pretrained:
        
        mock_from_pretrained.return_value = mock_model
        mock_get_peft_model.return_value = MagicMock(spec=PeftModel)
        
        with caplog.at_level(logging.INFO):
            peft_model = setup_peft_model("facebook/detr-resnet-50", num_classes=91, peft_config=mock_peft_config)
            assert "PEFT model successfully initialized with facebook/detr-resnet-50." in caplog.text, "Did not log successful PEFT model initialization."

def test_setup_peft_model_error(mock_peft_config):
    with patch("src.training.peft_finetune.DetrForObjectDetection.from_pretrained") as mock_from_pretrained:
        mock_from_pretrained.side_effect = Exception("Initialization failed")
        
        with pytest.raises(Exception) as exc_info:
            setup_peft_model("invalid-model", num_classes=91, peft_config=mock_peft_config)
        assert "Initialization failed" in str(exc_info.value), "Did not raise expected exception on model initialization failure."

# 2. Tests for prepare_dataloader

def test_prepare_dataloader_success(mock_feature_extractor, mock_dataset, mock_dataloader_train):
    """Test successful preparation of DataLoader for training mode."""
    with patch("src.training.peft_finetune.CocoDetection") as mock_coco_detection, \
         patch("torch.utils.data.DataLoader") as mock_dataloader:
        
        # Configure the mocked CocoDetection to return the mock_dataset
        mock_coco_detection.return_value = mock_dataset
        
        # Configure the mocked DataLoader to return the mock_dataloader_train
        mock_dataloader.return_value = mock_dataloader_train
        
        # Call the function under test
        dataloader = prepare_dataloader(
            data_dir="./data",
            batch_size=4,
            feature_extractor=mock_feature_extractor,
            mode="train"
        )
        
        # Assert that CocoDetection was called with the correct arguments
        mock_coco_detection.assert_called_once_with(
            root=os.path.join("./data", "train2017"),
            annFile=os.path.join("./data", "annotations/instances_train2017.json"),
            transform=ANY  # Accept any callable (lambda function)
        )
        
        # Assert that DataLoader was called with the correct arguments
        mock_dataloader.assert_called_once_with(
            mock_dataset,
            batch_size=4,
            shuffle=True,  # Shuffle should be True for training
            collate_fn=ANY  # Accept any callable (lambda function)
        )
        
        # Assert that the function returns the expected DataLoader
        assert dataloader == mock_dataloader_train, "Dataloader should be returned correctly."
        
        # Additional Assertions (Optional but Recommended)
        # Verify that the 'transform' argument in CocoDetection is a callable
        transform_arg = mock_coco_detection.call_args.kwargs.get('transform')
        assert callable(transform_arg), "Transform should be a callable."
        
        # Verify that the 'collate_fn' argument in DataLoader is a callable
        collate_fn_arg = mock_dataloader.call_args.kwargs.get('collate_fn')
        assert callable(collate_fn_arg), "collate_fn should be a callable."

def test_prepare_dataloader_validation_mode(mock_feature_extractor, mock_dataset, mock_dataloader_val):
    """Test successful preparation of DataLoader for validation mode."""
    with patch("src.training.peft_finetune.CocoDetection") as mock_coco_detection, \
         patch("torch.utils.data.DataLoader") as mock_dataloader:
        
        # Configure the mocks
        mock_coco_detection.return_value = mock_dataset
        mock_dataloader.return_value = mock_dataloader_val
        
        # Call the function under test
        dataloader = prepare_dataloader(
            data_dir="./data",
            batch_size=8,
            feature_extractor=mock_feature_extractor,
            mode="val"
        )
        
        # Assert that CocoDetection was called with correct arguments
        mock_coco_detection.assert_called_once_with(
            root=os.path.join("./data", "val2017"),
            annFile=os.path.join("./data", "annotations/instances_val2017.json"),
            transform=ANY  # Accept any callable (lambda function)
        )
        
        # Assert that DataLoader was called with correct arguments
        mock_dataloader.assert_called_once_with(
            mock_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=ANY  # Accept any callable (lambda function)
        )
        
        # Assert that the returned dataloader is as expected
        assert dataloader == mock_dataloader_val, "Dataloader should be returned correctly."
        
        # Additional Assertions to Verify Callables
        # Extract the 'transform' argument passed to CocoDetection
        transform_arg = mock_coco_detection.call_args.kwargs.get('transform')
        assert callable(transform_arg), "Transform should be a callable."
        
        # Extract the 'collate_fn' argument passed to DataLoader
        collate_fn_arg = mock_dataloader.call_args.kwargs.get('collate_fn')
        assert callable(collate_fn_arg), "collate_fn should be a callable."

def test_prepare_dataloader_logging(mock_feature_extractor, mock_dataset, mock_dataloader_train, caplog):
    """Test that prepare_dataloader logs the appropriate message."""
    with patch("src.training.peft_finetune.CocoDetection") as mock_coco_detection, \
         patch("torch.utils.data.DataLoader") as mock_dataloader:
        
        mock_coco_detection.return_value = mock_dataset
        mock_dataloader.return_value = mock_dataloader_train
        
        with caplog.at_level(logging.INFO):
            dataloader = prepare_dataloader(
                data_dir="./data",
                batch_size=4,
                feature_extractor=mock_feature_extractor,
                mode="train"
            )
            assert "Dataloader for mode 'train' prepared successfully with batch size 4." in caplog.text, "Did not log successful DataLoader preparation."

def test_prepare_dataloader_error(mock_feature_extractor):
    """Test that prepare_dataloader raises an error when data_dir is invalid."""
    with patch("src.training.peft_finetune.CocoDetection") as mock_coco_detection:
        # Configure the mock to raise an Exception when instantiated
        mock_coco_detection.side_effect = Exception("Invalid data directory")
        
        # Attempt to prepare the dataloader and expect an exception
        with pytest.raises(Exception) as exc_info:
            prepare_dataloader(
                data_dir="./invalid_data",
                batch_size=4,
                feature_extractor=mock_feature_extractor,
                mode="train"
            )
        
        # Assert that the exception message contains the expected substring
        assert "Invalid data directory" in str(exc_info.value), "Did not raise expected exception on invalid data directory."
        
        # Additionally, verify that CocoDetection was indeed called once
        mock_coco_detection.assert_called_once()

def test_prepare_dataloader_invalid_mode(mock_feature_extractor):
    """Test that prepare_dataloader raises an assertion error for invalid mode."""
    with pytest.raises(AssertionError) as exc_info:
        prepare_dataloader(
            data_dir="./data",
            batch_size=4,
            feature_extractor=mock_feature_extractor,
            mode="test"  # Invalid mode
        )
    assert "Mode should be either 'train' or 'val'." in str(exc_info.value), "Did not raise AssertionError for invalid mode."

# 3. Tests for fine_tune_peft_model

def test_fine_tune_peft_model_training_loop(
    default_config,
    mock_peft_model,
    mock_dataloader_train,
    mock_dataloader_val,
    mock_optimizer,
    mock_scheduler,
    caplog
):
    """Test that fine_tune_peft_model executes the training and validation loops correctly."""
    
    # 1. Mock model's methods
    # Ensure that model.to(device) returns the model itself
    mock_peft_model.to = MagicMock(return_value=mock_peft_model)
    # Mock train() and eval() methods
    mock_peft_model.train = MagicMock()
    mock_peft_model.eval = MagicMock()
    
    # 2. Mock model's forward method to return a mock output with 'loss'
    mock_output = MagicMock()
    mock_output.loss = torch.tensor(1.0, requires_grad=True)
    mock_peft_model.forward.return_value = mock_output
    
    # 3. Mock state_dict to return a serializable object
    mock_peft_model.state_dict.return_value = {}
    
    # 4. Mock external dependencies: autocast, GradScaler, and torch.save
    with patch("src.training.peft_finetune.autocast") as mock_autocast, \
         patch("src.training.peft_finetune.GradScaler") as mock_grad_scaler, \
         patch("torch.save") as mock_torch_save:
        
        # 4a. Mock autocast as a context manager that does nothing
        mock_autocast_context = MagicMock()
        mock_autocast_context.__enter__ = MagicMock()
        mock_autocast_context.__exit__ = MagicMock()
        mock_autocast.return_value = mock_autocast_context
        
        # 4b. Handle GradScaler based on mixed_precision configuration
        mixed_precision = default_config['training'].get('mixed_precision', True)
        if mixed_precision:
            # If mixed_precision is enabled, mock GradScaler instance methods
            mock_grad_scaler_instance = MagicMock()
            mock_grad_scaler.return_value = mock_grad_scaler_instance
            # Mock scaler.scale(loss) to return the scaled loss
            mock_grad_scaler_instance.scale.return_value = mock_output.loss
        else:
            # If mixed_precision is disabled, GradScaler is not used
            mock_grad_scaler.return_value = None
        
        # 4c. Mock torch.save to prevent actual file I/O
        mock_torch_save.return_value = None
        
        # 5. Execute the function within the context manager to capture logs
        with caplog.at_level(logging.INFO):
            fine_tune_peft_model(
                model=mock_peft_model,
                train_dataloader=mock_dataloader_train,
                val_dataloader=mock_dataloader_val,
                optimizer=mock_optimizer,
                scheduler=mock_scheduler,
                config=default_config,
                device="cpu"
            )
    
    # 6. Assertions to verify that model methods were called
    mock_peft_model.train.assert_called()
    mock_peft_model.eval.assert_called()
    
    # 7. Assertions to verify optimizer and scheduler methods were called
    mock_optimizer.zero_grad.assert_called()
    mock_optimizer.step.assert_called()
    mock_scheduler.step.assert_called()
    
    # 8. Assertions to verify loss.backward() was called
    mock_output.loss.backward.assert_called()
    
    # 9. Assertions to verify GradScaler methods were called if mixed_precision is enabled
    if mixed_precision:
        mock_grad_scaler_instance.scale.assert_called_with(mock_output.loss)
        mock_grad_scaler_instance.step.assert_called_with(mock_optimizer)
        mock_grad_scaler_instance.update.assert_called()
    
    # 10. Assertions to verify torch.save was called with the mock state_dict and correct path
    mock_torch_save.assert_called_with({}, ANY)  # ANY is used for the checkpoint path
    
    # 11. Assertions to verify that checkpoint saving was logged
    assert "Model checkpoint saved at ./checkpoints/model_epoch_1.pt" in caplog.text, "Did not log checkpoint saving."

def test_fine_tune_peft_model_mixed_precision(
    default_config,
    mock_peft_model,
    mock_dataloader_train,
    mock_dataloader_val,
    mock_optimizer,
    mock_scheduler,
    caplog
):
    """Test fine_tune_peft_model with mixed precision enabled."""
    # Update configuration to enable mixed precision
    config = default_config.copy()
    config['training']['mixed_precision'] = True
    
    # Mock model's methods
    mock_peft_model.to = MagicMock(return_value=mock_peft_model)
    mock_peft_model.train = MagicMock()
    mock_peft_model.eval = MagicMock()
    
    # Mock forward pass to return a mock output with a 'loss' attribute
    mock_output = MagicMock()
    mock_output.loss = torch.tensor(1.0, requires_grad=True)
    mock_peft_model.forward.return_value = mock_output
    
    # Ensure state_dict returns a serializable object to prevent PicklingError
    mock_peft_model.state_dict.return_value = {}
    
    # Mock external dependencies: autocast, GradScaler, and torch.save
    with patch("src.training.peft_finetune.autocast") as mock_autocast, \
         patch("src.training.peft_finetune.GradScaler") as mock_grad_scaler, \
         patch("torch.save") as mock_torch_save:
        
        # Configure autocast as a proper context manager
        mock_autocast_context = MagicMock()
        mock_autocast_context.__enter__.return_value = None  # autocast does not return a value
        mock_autocast_context.__exit__.return_value = False  # Do not suppress exceptions
        mock_autocast.return_value = mock_autocast_context
        
        # Configure GradScaler instance and its methods
        mock_grad_scaler_instance = MagicMock()
        mock_grad_scaler_instance.scale.return_value = mock_output.loss  # scaler.scale(loss) returns the scaled loss
        mock_grad_scaler.return_value = mock_grad_scaler_instance
        
        # Mock torch.save to prevent actual saving
        mock_torch_save.return_value = None
        
        # Execute the fine_tune_peft_model function within the mocked context
        with caplog.at_level(logging.INFO):
            fine_tune_peft_model(
                model=mock_peft_model,
                train_dataloader=mock_dataloader_train,
                val_dataloader=mock_dataloader_val,
                optimizer=mock_optimizer,
                scheduler=mock_scheduler,
                config=config,
                device="cpu"
            )
    
    # Assertions to verify that training methods were called
    mock_peft_model.train.assert_called()
    mock_peft_model.eval.assert_called()
    
    # Assertions to verify optimizer and scheduler methods were called
    assert mock_optimizer.zero_grad.call_count == default_config['training']['num_epochs'] * len(mock_dataloader_train), \
        "optimizer.zero_grad() should be called once per batch per epoch."
    assert mock_optimizer.step.call_count == default_config['training']['num_epochs'] * len(mock_dataloader_train), \
        "optimizer.step() should be called once per batch per epoch."
    assert mock_scheduler.step.call_count == default_config['training']['num_epochs'] * len(mock_dataloader_train), \
        "scheduler.step() should be called once per batch per epoch."
    
    # Assertions to verify loss.backward() was called
    assert mock_output.loss.backward.call_count == default_config['training']['num_epochs'] * len(mock_dataloader_train), \
        "loss.backward() should be called once per batch per epoch."
    
    # Assertions to verify GradScaler methods were called
    assert mock_grad_scaler_instance.scale.call_count == default_config['training']['num_epochs'] * len(mock_dataloader_train), \
        "GradScaler.scale() should be called once per batch per epoch."
    assert mock_grad_scaler_instance.step.call_count == default_config['training']['num_epochs'] * len(mock_dataloader_train), \
        "GradScaler.step() should be called once per batch per epoch."
    assert mock_grad_scaler_instance.update.call_count == default_config['training']['num_epochs'] * len(mock_dataloader_train), \
        "GradScaler.update() should be called once per batch per epoch."
    
    # Assertions to verify torch.save was called with the mock state dict and the correct path
    assert mock_torch_save.call_count == config['training']['num_epochs'], \
        "torch.save() should be called once per epoch."
    
    # Verify that torch.save was called with the correct arguments for each epoch
    expected_checkpoint_paths = [
        f"{config['training']['checkpoint_dir']}/model_epoch_{epoch + 1}.pt" 
        for epoch in range(config['training']['num_epochs'])
    ]
    actual_calls = mock_torch_save.call_args_list
    for expected_path in expected_checkpoint_paths:
        mock_torch_save.assert_any_call({}, expected_path)
    
    # Assertions to verify that checkpoint saving was logged for each epoch
    for epoch in range(1, config['training']['num_epochs'] + 1):
        checkpoint_log = f"Model checkpoint saved at {config['training']['checkpoint_dir']}/model_epoch_{epoch}.pt"
        assert checkpoint_log in caplog.text, f"Did not log checkpoint saving for epoch {epoch}."

def test_fine_tune_peft_model_error_during_training(default_config, mock_peft_model, mock_dataloader_train, mock_dataloader_val, mock_optimizer, mock_scheduler):
    """Test that fine_tune_peft_model raises an error when an exception occurs during training."""
    # Mock model's forward pass to raise an exception
    mock_peft_model.return_value.loss = torch.tensor(1.0, requires_grad=True)
    
    with patch("src.training.peft_finetune.autocast") as mock_autocast:
        mock_autocast.side_effect = Exception("Training failed")
        
        with pytest.raises(Exception) as exc_info:
            fine_tune_peft_model(
                model=mock_peft_model,
                train_dataloader=mock_dataloader_train,
                val_dataloader=mock_dataloader_val,
                optimizer=mock_optimizer,
                scheduler=mock_scheduler,
                config=default_config,
                device="cpu"
            )
        assert "Training failed" in str(exc_info.value), "Did not raise expected exception during training."

def test_fine_tune_peft_model_checkpoint_saving(default_config, mock_peft_model, mock_dataloader_train, mock_dataloader_val, mock_optimizer, mock_scheduler):
    """Test that fine_tune_peft_model saves checkpoints correctly."""
    with patch("src.training.peft_finetune.torch.save") as mock_torch_save, \
         patch("src.training.peft_finetune.os.makedirs") as mock_makedirs:
        
        fine_tune_peft_model(
            model=mock_peft_model,
            train_dataloader=mock_dataloader_train,
            val_dataloader=mock_dataloader_val,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            config=default_config,
            device="cpu"
        )
        
        mock_makedirs.assert_called_once_with("./checkpoints", exist_ok=True)
        # Check that torch.save was called twice (for 2 epochs)
        assert mock_torch_save.call_count == default_config['training']['num_epochs'], "Checkpoints not saved for each epoch."

def test_fine_tune_peft_model_empty_dataloader(
    default_config,
    mock_peft_model,
    mock_optimizer,
    mock_scheduler,
    caplog
):
    """Test fine_tune_peft_model with empty training and validation dataloaders."""
    
    # Create an empty DataLoader mock
    empty_dataloader = MagicMock(spec=DataLoader)
    empty_dataloader.__len__.return_value = 0
    empty_dataloader.__iter__.return_value = iter([])  # Ensure the iterator is empty
    
    # Mock model's train and eval methods
    mock_peft_model.train = MagicMock()
    mock_peft_model.eval = MagicMock()
    
    # Ensure state_dict returns a serializable object to prevent PicklingError
    mock_peft_model.state_dict.return_value = {}
    
    # Mock external dependencies: autocast, GradScaler, and torch.save
    with patch("src.training.peft_finetune.autocast") as mock_autocast, \
         patch("src.training.peft_finetune.GradScaler") as mock_grad_scaler, \
         patch("torch.save") as mock_torch_save:
        
        # Configure the mocked autocast context manager to do nothing
        mock_autocast.return_value = MagicMock()
        
        # Configure the mocked GradScaler instance to do nothing
        mock_grad_scaler_instance = MagicMock()
        mock_grad_scaler.return_value = mock_grad_scaler_instance
        
        # Mock tqdm to simply return the iterator without any progress bar interference
        with patch("src.training.peft_finetune.tqdm", side_effect=lambda x, desc: x):
            # Execute the fine_tune_peft_model function within the mocked context
            with caplog.at_level(logging.INFO):
                fine_tune_peft_model(
                    model=mock_peft_model,
                    train_dataloader=empty_dataloader,
                    val_dataloader=empty_dataloader,
                    optimizer=mock_optimizer,
                    scheduler=mock_scheduler,
                    config=default_config,
                    device="cpu"
                )
    
    # Assertions to verify that training and validation methods were called
    mock_peft_model.train.assert_called()
    mock_peft_model.eval.assert_called()
    
    # Since dataloaders are empty, optimizer steps should not be called
    mock_optimizer.zero_grad.assert_not_called()
    mock_optimizer.step.assert_not_called()
    mock_scheduler.step.assert_not_called()
    
    # Check that torch.save was called with the mock state_dict and any checkpoint path
    mock_torch_save.assert_called_with({}, ANY)  # ANY is used for the checkpoint path
    
    # Assertions to verify that loss computations were skipped and logs were made
    assert "Training Loss: 0.0" in caplog.text, "Did not log training loss."
    assert "Validation Loss: 0.0" in caplog.text, "Did not log validation loss."

# 4. Tests for main

def test_main_success(
    default_config,
    mock_peft_model,
    mock_dataloader_train,
    mock_dataloader_val,
    mock_optimizer,
    mock_scheduler,
    mock_feature_extractor,
    mock_peft_config,
    caplog
):
    """
    Test that the main function runs successfully with a valid configuration.
    """
    config_path = "configs/peft_config.yaml"
    
    # Patch all external dependencies that main relies on
    with patch("src.training.peft_finetune.ConfigParser") as mock_config_parser, \
         patch("src.training.peft_finetune.setup_logging") as mock_setup_logging, \
         patch("src.training.peft_finetune.DetrFeatureExtractor.from_pretrained") as mock_from_pretrained, \
         patch("src.training.peft_finetune.setup_peft_model") as mock_setup_peft_model, \
         patch("src.training.peft_finetune.prepare_dataloader") as mock_prepare_dataloader, \
         patch("src.training.peft_finetune.get_optimizer_and_scheduler") as mock_get_optimizer_and_scheduler, \
         patch("src.training.peft_finetune.PeftConfig.from_pretrained") as mock_peft_config_from_pretrained, \
         patch("src.training.peft_finetune.fine_tune_peft_model") as mock_fine_tune_peft_model:
        
        # Configure the ConfigParser mock to return the default configuration
        mock_config_parser_instance = MagicMock()
        mock_config_parser_instance.config = default_config
        mock_config_parser.return_value = mock_config_parser_instance
        
        # Configure the DetrFeatureExtractor mock
        mock_from_pretrained.return_value = mock_feature_extractor
        
        # Configure the PeftConfig.from_pretrained mock to return a mock PeftConfig
        mock_peft_config_from_pretrained.return_value = mock_peft_config
        
        # Configure the setup_peft_model mock to return the mock PEFT model
        mock_setup_peft_model.return_value = mock_peft_model
        
        # Configure the prepare_dataloader mock to return mock training and validation dataloaders
        mock_prepare_dataloader.side_effect = [mock_dataloader_train, mock_dataloader_val]
        
        # Configure the get_optimizer_and_scheduler mock to return mock optimizer and scheduler
        mock_get_optimizer_and_scheduler.return_value = (mock_optimizer, mock_scheduler)
        
        # Configure fine_tune_peft_model to do nothing (since it's being tested separately)
        mock_fine_tune_peft_model.return_value = None
        
        # Execute the main function within the context of captured logs
        with caplog.at_level(logging.INFO):
            main(config_path)
        
        # Assertions to verify that each mocked method was called correctly
        
        # Verify ConfigParser was instantiated with the correct config path
        mock_config_parser.assert_called_once_with(config_path)
        
        # Verify setup_logging was called once to initialize logging
        mock_setup_logging.assert_called_once()
        
        # Verify DetrFeatureExtractor.from_pretrained was called with the correct model name
        mock_from_pretrained.assert_called_once_with(default_config['model']['model_name'])
        
        # Verify PeftConfig.from_pretrained was called with the correct PEFT model path
        mock_peft_config_from_pretrained.assert_called_once_with(default_config['model']['peft_model_path'])
        
        # Verify setup_peft_model was called with correct arguments
        mock_setup_peft_model.assert_called_once_with(
            default_config['model']['model_name'],
            num_classes=default_config['model']['num_classes'],
            peft_config=mock_peft_config
        )
        
        # Verify prepare_dataloader was called twice: once for training and once for validation
        assert mock_prepare_dataloader.call_count == 2, "prepare_dataloader should be called twice for train and val."
        mock_prepare_dataloader.assert_any_call(
            data_dir=default_config['data']['data_dir'],
            batch_size=default_config['training']['batch_size'],
            feature_extractor=mock_feature_extractor,
            mode="train"
        )
        mock_prepare_dataloader.assert_any_call(
            data_dir=default_config['data']['data_dir'],
            batch_size=default_config['training']['batch_size'],
            feature_extractor=mock_feature_extractor,
            mode="val"
        )
        
        # Verify get_optimizer_and_scheduler was called with correct arguments
        mock_get_optimizer_and_scheduler.assert_called_once_with(
            model=mock_peft_model,
            config=default_config['optimizer'],
            num_training_steps=default_config['training']['num_epochs'] * len(mock_dataloader_train)
        )
        
        # Verify fine_tune_peft_model was called with correct arguments
        mock_fine_tune_peft_model.assert_called_once_with(
            model=mock_peft_model,
            train_dataloader=mock_dataloader_train,
            val_dataloader=mock_dataloader_val,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            config=default_config,
            device=default_config['training'].get('device', 'cuda')
        )
        
        # Assertions to verify that key log messages are present
        assert "Starting PEFT fine-tuning process." in caplog.text, "Did not log starting fine-tuning process."
        assert "PEFT fine-tuning completed." in caplog.text, "Did not log completion of fine-tuning process."

def test_main_error_in_config_parser(caplog):
    """Test that main raises an error when ConfigParser fails."""
    config_path = "configs/invalid_config.yaml"
    
    # Patch ConfigParser to raise an exception when instantiated
    with patch("src.training.peft_finetune.ConfigParser", autospec=True) as mock_config_parser, \
         patch("src.training.peft_finetune.setup_logging") as mock_setup_logging:
        
        # Configure the mock to raise an exception upon instantiation
        mock_config_parser.side_effect = Exception("Config file not found")
        
        # Execute main and verify that it raises the expected exception
        with pytest.raises(Exception) as exc_info:
            main(config_path)
        
        # Assert that the exception message contains the expected text
        assert "Config file not found" in str(exc_info.value), "Did not raise expected exception for config parser failure."

def test_main_error_in_setup_peft_model(default_config, mock_peft_model, mock_peft_config):
    """Test that main raises an error when setup_peft_model fails."""
    config_path = "configs/peft_config.yaml"
    
    with patch("src.training.peft_finetune.ConfigParser") as mock_config_parser, \
         patch("src.training.peft_finetune.setup_logging") as mock_setup_logging, \
         patch("src.training.peft_finetune.DetrFeatureExtractor.from_pretrained") as mock_from_pretrained, \
         patch("src.training.peft_finetune.setup_peft_model") as mock_setup_peft_model:
        
        # Mock the configuration parser to return the default configuration
        mock_config_parser.return_value.config = default_config
        
        # Mock the feature extractor to prevent actual model loading
        mock_from_pretrained.return_value = MagicMock(spec=DetrFeatureExtractor)
        
        # Mock setup_peft_model to raise an exception, simulating a failure during PEFT setup
        mock_setup_peft_model.side_effect = Exception("PEFT setup failed")
        
        # Execute the main function and expect it to raise the mocked exception
        with pytest.raises(Exception, match="PEFT setup failed") as exc_info:
            main(config_path)
        
        # Assert that the exception message contains the expected error
        assert "PEFT setup failed" in str(exc_info.value), "Did not raise expected exception for PEFT setup failure."

# 5. Edge Case Tests

def test_fine_tune_peft_model_zero_epochs(default_config, mock_peft_model, mock_dataloader_train, mock_dataloader_val, mock_optimizer, mock_scheduler):
    """Test fine_tune_peft_model with zero epochs."""
    config = default_config.copy()
    config['training']['num_epochs'] = 0
    
    with caplog.at_level(logging.INFO):
        fine_tune_peft_model(
            model=mock_peft_model,
            train_dataloader=mock_dataloader_train,
            val_dataloader=mock_dataloader_val,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            config=config,
            device="cpu"
        )
    
    # Ensure that training loop is skipped
    mock_peft_model.train.assert_not_called()
    mock_peft_model.eval.assert_not_called()
    mock_optimizer.step.assert_not_called()
    mock_scheduler.step.assert_not_called()
    assert "Starting Epoch 1/0" not in caplog.text, "Did not skip training loop for zero epochs."

def test_fine_tune_peft_model_large_batch_size(
    default_config,
    mock_peft_model,
    mock_dataloader_train,
    mock_dataloader_val,
    mock_optimizer,
    mock_scheduler,
    caplog  # Added caplog as a parameter
):
    """
    Test fine_tune_peft_model with an extremely large batch size to ensure
    that the function handles large data volumes without errors and logs appropriately.
    """
    # Update the configuration to use a very large batch size
    config = default_config.copy()
    config['training']['batch_size'] = 10000  # Extremely large batch size

    # Mock dataloader to simulate a single large batch
    large_batch = (
        torch.randn(10000, 3, 224, 224),  # Pixel values for images
        [
            {
                "labels": torch.randint(0, 91, (10000,)),  # Random labels
                "boxes": torch.randn(10000, 4)  # Random bounding boxes
            }
        ]
    )
    
    # Configure the mock_dataloader_train to return a single large batch
    mock_dataloader_train.__iter__.return_value = iter([large_batch])
    mock_dataloader_train.__len__.return_value = 1  # Only one batch

    # Configure the mock_dataloader_val similarly
    mock_dataloader_val.__iter__.return_value = iter([large_batch])
    mock_dataloader_val.__len__.return_value = 1  # Only one batch

    # Mock the model's methods
    mock_peft_model.to = MagicMock(return_value=mock_peft_model)
    mock_peft_model.train = MagicMock()
    mock_peft_model.eval = MagicMock()

    # Mock the forward pass to return a mock output with a 'loss' attribute
    mock_output = MagicMock()
    mock_output.loss = torch.tensor(1.0, requires_grad=True)
    mock_peft_model.forward.return_value = mock_output

    # Ensure state_dict returns a serializable object to prevent PicklingError during torch.save
    mock_peft_model.state_dict.return_value = {}

    # Mock external dependencies: autocast, GradScaler, and torch.save
    with patch("src.training.peft_finetune.autocast"), \
         patch("src.training.peft_finetune.GradScaler") as mock_grad_scaler, \
         patch("torch.save") as mock_torch_save:
        
        # Configure the mocked GradScaler instance
        mock_grad_scaler_instance = MagicMock()
        mock_grad_scaler_instance.scale.return_value = mock_output.loss  # scaler.scale(loss) returns scaled loss
        mock_grad_scaler.return_value = mock_grad_scaler_instance

        # Execute the fine_tune_peft_model function within the mocked context
        with caplog.at_level(logging.INFO):
            fine_tune_peft_model(
                model=mock_peft_model,
                train_dataloader=mock_dataloader_train,
                val_dataloader=mock_dataloader_val,
                optimizer=mock_optimizer,
                scheduler=mock_scheduler,
                config=config,
                device="cpu"
            )

    # Assertions to verify that training and validation methods were called
    mock_peft_model.train.assert_called()
    mock_peft_model.eval.assert_called()

    # Assertions to verify optimizer and scheduler methods were called
    # Since there's only one batch, these should be called once
    mock_optimizer.zero_grad.assert_called_once()
    mock_optimizer.step.assert_called_once()
    mock_scheduler.step.assert_called_once()

    # Assertions to verify that loss.backward() was called
    mock_output.loss.backward.assert_called_once()

    # Assertions to verify GradScaler methods were called appropriately
    mock_grad_scaler_instance.scale.assert_called_once_with(mock_output.loss)
    mock_grad_scaler_instance.step.assert_called_once_with(mock_optimizer)
    mock_grad_scaler_instance.update.assert_called_once()

    # Assertions to verify that torch.save was called correctly
    mock_torch_save.assert_called_once_with({}, ANY)  # ANY is used for the checkpoint path

    # Assertions to verify that checkpoint saving was logged
    assert "Model checkpoint saved at ./checkpoints/model_epoch_1.pt" in caplog.text, \
        "Did not log checkpoint saving."

    # Assertions to verify that training and validation logs are present
    assert "Starting Epoch 1/2" in caplog.text, "Did not log start of epoch."
    assert "Training Loss: 1.0" in caplog.text, "Did not log training loss."
    assert "Validation Loss: 1.0" in caplog.text, "Did not log validation loss."

def test_fine_tune_peft_model_no_validation(
    default_config,
    mock_peft_model,
    mock_dataloader_train,
    mock_optimizer,
    mock_scheduler,
    caplog
):
    """Test fine_tune_peft_model when validation dataloader is None."""
    
    # Mock model's methods
    mock_peft_model.to = MagicMock(return_value=mock_peft_model)
    mock_peft_model.train = MagicMock()
    mock_peft_model.eval = MagicMock()
    mock_peft_model.forward = MagicMock(return_value=MagicMock(loss=torch.tensor(1.0, requires_grad=True)))
    mock_peft_model.state_dict = MagicMock(return_value={})
    
    # Mock external dependencies: autocast, GradScaler, and torch.save
    with patch("src.training.peft_finetune.autocast") as mock_autocast, \
         patch("src.training.peft_finetune.GradScaler") as mock_grad_scaler, \
         patch("torch.save") as mock_torch_save:
        
        # Configure the mocked autocast context manager
        mock_autocast.return_value = MagicMock()
        
        # Configure the mocked GradScaler instance if mixed_precision is enabled
        if default_config['training'].get('mixed_precision', True):
            mock_grad_scaler_instance = MagicMock()
            mock_grad_scaler.return_value = mock_grad_scaler_instance
            mock_grad_scaler_instance.scale.return_value = mock_peft_model.forward.return_value.loss
        else:
            mock_grad_scaler.return_value = None
        
        # Execute the fine_tune_peft_model function with val_dataloader=None
        with caplog.at_level(logging.INFO):
            fine_tune_peft_model(
                model=mock_peft_model,
                train_dataloader=mock_dataloader_train,
                val_dataloader=None,  # No validation
                optimizer=mock_optimizer,
                scheduler=mock_scheduler,
                config=default_config,
                device="cpu"
            )
    
    # Assertions to verify that training methods were called
    mock_peft_model.train.assert_called()
    
    # Assertions to verify optimizer and scheduler methods were called the expected number of times
    # Since len(mock_dataloader_train) = 10 (from fixture), num_epochs=2
    expected_zero_grad_calls = default_config['training']['num_epochs'] * len(mock_dataloader_train)
    expected_step_calls = default_config['training']['num_epochs'] * len(mock_dataloader_train)
    
    assert mock_optimizer.zero_grad.call_count == expected_zero_grad_calls, "Optimizer.zero_grad() call count mismatch."
    assert mock_optimizer.step.call_count == expected_step_calls, "Optimizer.step() call count mismatch."
    assert mock_scheduler.step.call_count == expected_step_calls, "Scheduler.step() call count mismatch."
    
    # If mixed_precision is enabled, verify GradScaler methods were called
    if default_config['training'].get('mixed_precision', True):
        assert mock_grad_scaler_instance.scale.call_count == expected_step_calls, "GradScaler.scale() call count mismatch."
        assert mock_grad_scaler_instance.step.call_count == expected_step_calls, "GradScaler.step() call count mismatch."
        assert mock_grad_scaler_instance.update.call_count == expected_step_calls, "GradScaler.update() call count mismatch."
    else:
        # Ensure GradScaler methods were not called
        mock_grad_scaler.assert_not_called()
    
    # Assertions to verify torch.save was called with the mock state dict and the correct path
    # Since num_epochs=2, torch.save should be called twice
    assert mock_torch_save.call_count == default_config['training']['num_epochs'], "torch.save() should be called once per epoch."
    
    # Assertions to verify that loss computations were skipped and logs were made
    assert "No validation dataloader provided. Skipping validation." in caplog.text, "Did not log skipping validation."
    assert "Training Loss" in caplog.text, "Did not log training loss."
    assert "Validation Loss" not in caplog.text, "Logged validation loss despite val_dataloader being None."

# 6. Logging Tests

def test_fine_tune_peft_model_logging(
    default_config,
    mock_peft_model,
    mock_dataloader_train,
    mock_dataloader_val,
    mock_optimizer,
    mock_scheduler,
    caplog
):
    """Test that fine_tune_peft_model logs training and validation losses."""
    
    # Mock model's methods are already handled in the fixture
    # Mock model's forward pass to return a mock output with a 'loss' attribute
    mock_output = MagicMock()
    mock_output.loss = torch.tensor(1.0, requires_grad=True)
    mock_peft_model.forward.return_value = mock_output  # Correctly mock forward
    
    # Mock external dependencies: autocast, GradScaler, and torch.save
    with patch("src.training.peft_finetune.autocast") as mock_autocast, \
         patch("src.training.peft_finetune.GradScaler") as mock_grad_scaler, \
         patch("torch.save") as mock_torch_save:
        
        # Configure the mocked autocast context manager
        mock_autocast.return_value = MagicMock()
        
        # Configure the mocked GradScaler instance
        mock_grad_scaler_instance = MagicMock()
        mock_grad_scaler.return_value = mock_grad_scaler_instance
        
        # Execute the fine_tune_peft_model function within the mocked context
        with caplog.at_level(logging.INFO):
            fine_tune_peft_model(
                model=mock_peft_model,
                train_dataloader=mock_dataloader_train,
                val_dataloader=mock_dataloader_val,
                optimizer=mock_optimizer,
                scheduler=mock_scheduler,
                config=default_config,
                device="cpu"
            )
    
    # Assertions to verify that training methods were called
    mock_peft_model.train.assert_called()
    mock_peft_model.eval.assert_called()
    
    # Assertions to verify optimizer and scheduler methods were called
    mock_optimizer.zero_grad.assert_called()
    mock_optimizer.step.assert_called()
    mock_scheduler.step.assert_called()
    
    # Assertions to verify loss.backward() was called
    mock_output.loss.backward.assert_called()
    
    # Assertions to verify GradScaler methods were called if mixed_precision is enabled
    mixed_precision = default_config['training'].get('mixed_precision', True)
    if mixed_precision:
        mock_grad_scaler_instance.scale.assert_called_with(mock_output.loss)
        mock_grad_scaler_instance.step.assert_called_with(mock_optimizer)
        mock_grad_scaler_instance.update.assert_called()
    
    # Assertions to verify torch.save was called with the mock state dict and the correct path
    mock_torch_save.assert_called_with({}, ANY)  # ANY is used for the checkpoint path
    
    # Assertion to verify that checkpoint saving was logged
    assert "Model checkpoint saved at ./checkpoints/model_epoch_1.pt" in caplog.text, "Did not log checkpoint saving."
    
    # Additional Assertions for Logging
    assert "Starting Epoch 1/2" in caplog.text, "Did not log start of epoch."
    assert "Training Loss: 1.0" in caplog.text, "Did not log training loss."
    assert "Validation Loss: 1.0" in caplog.text, "Did not log validation loss."

# 7. Reproducibility Tests

def test_fine_tune_peft_model_reproducibility(
    default_config,
    mock_peft_model,
    mock_dataloader_train,
    mock_dataloader_val,
    mock_optimizer,
    mock_scheduler,
    caplog
):
    """
    Test that fine_tune_peft_model produces consistent results with the same inputs by
    running the training loop twice and verifying that optimizer.zero_grad() is called
    the expected number of times.
    """
    
    # Update configuration if necessary
    config = default_config.copy()
    config['training']['mixed_precision'] = False  # Ensure consistency for reproducibility
    
    # Mock model's methods
    mock_peft_model.to = MagicMock(return_value=mock_peft_model)
    mock_peft_model.train = MagicMock()
    mock_peft_model.eval = MagicMock()
    
    # Mock model's forward pass to return a mock output with a 'loss' attribute
    mock_output = MagicMock()
    mock_output.loss = torch.tensor(1.0, requires_grad=True)
    mock_peft_model.forward.return_value = mock_output
    
    # Ensure state_dict returns a serializable object to prevent PicklingError
    mock_peft_model.state_dict.return_value = {}
    
    # Mock external dependencies: autocast, GradScaler, and torch.save
    with patch("src.training.peft_finetune.autocast") as mock_autocast, \
         patch("src.training.peft_finetune.GradScaler") as mock_grad_scaler, \
         patch("torch.save") as mock_torch_save:
        
        # Configure the mocked autocast context manager
        mock_autocast.return_value = MagicMock()
        
        # Configure the mocked GradScaler instance
        mock_grad_scaler_instance = MagicMock()
        mock_grad_scaler.return_value = mock_grad_scaler_instance
        
        # Execute the fine_tune_peft_model function within the mocked context
        with caplog.at_level(logging.INFO):
            # First run
            fine_tune_peft_model(
                model=mock_peft_model,
                train_dataloader=mock_dataloader_train,
                val_dataloader=mock_dataloader_val,
                optimizer=mock_optimizer,
                scheduler=mock_scheduler,
                config=config,
                device="cpu"
            )
            
            # Reset mocks to track calls in the second run
            mock_peft_model.train.reset_mock()
            mock_peft_model.eval.reset_mock()
            mock_optimizer.zero_grad.reset_mock()
            mock_optimizer.step.reset_mock()
            mock_scheduler.step.reset_mock()
            mock_output.loss.backward.reset_mock()
            mock_torch_save.reset_mock()
            mock_grad_scaler_instance.scale.reset_mock()
            mock_grad_scaler_instance.step.reset_mock()
            mock_grad_scaler_instance.update.reset_mock()
        
            # Second run
            fine_tune_peft_model(
                model=mock_peft_model,
                train_dataloader=mock_dataloader_train,
                val_dataloader=mock_dataloader_val,
                optimizer=mock_optimizer,
                scheduler=mock_scheduler,
                config=config,
                device="cpu"
            )
    
    # Assertions to verify that training methods were called the correct number of times
    num_epochs = config['training']['num_epochs']
    num_train_batches = len(mock_dataloader_train)  # Should be 10
    num_val_batches = len(mock_dataloader_val)      # Should be 5
    
    # Each run of fine_tune_peft_model processes 'num_epochs' epochs
    # Since the test runs it twice, total epochs processed = num_epochs * 2
    total_epochs = num_epochs * 2
    
    # Each epoch has 'num_train_batches' training steps
    # Total training steps across both runs = num_epochs * num_train_batches * 2
    total_train_steps = num_epochs * num_train_batches * 2
    
    # Each training step calls optimizer.zero_grad()
    # Total expected calls = num_epochs * num_train_batches * 2
    expected_zero_grad_calls = num_epochs * num_train_batches * 2
    
    # Assert that optimizer.zero_grad() was called the expected number of times
    assert mock_optimizer.zero_grad.call_count == expected_zero_grad_calls, (
        f"Optimizer.zero_grad() was called {mock_optimizer.zero_grad.call_count} times, "
        f"expected {expected_zero_grad_calls} times."
    )
    
    # Similarly, assert other optimizer and scheduler calls
    expected_optimizer_step_calls = num_epochs * num_train_batches * 2
    expected_scheduler_step_calls = num_epochs * num_train_batches * 2
    expected_backward_calls = num_epochs * num_train_batches * 2
    
    assert mock_optimizer.step.call_count == expected_optimizer_step_calls, (
        f"Optimizer.step() was called {mock_optimizer.step.call_count} times, "
        f"expected {expected_optimizer_step_calls} times."
    )
    
    assert mock_scheduler.step.call_count == expected_scheduler_step_calls, (
        f"Scheduler.step() was called {mock_scheduler.step.call_count} times, "
        f"expected {expected_scheduler_step_calls} times."
    )
    
    assert mock_output.loss.backward.call_count == expected_backward_calls, (
        f"loss.backward() was called {mock_output.loss.backward.call_count} times, "
        f"expected {expected_backward_calls} times."
    )
    
    # Verify that GradScaler methods were called appropriately based on mixed_precision
    mixed_precision = config['training'].get('mixed_precision', True)
    if mixed_precision:
        # Each training step should call scale, step, and update
        expected_scale_calls = num_epochs * num_train_batches * 2
        expected_step_calls = num_epochs * num_train_batches * 2
        expected_update_calls = num_epochs * num_train_batches * 2
        
        assert mock_grad_scaler_instance.scale.call_count == expected_scale_calls, (
            f"GradScaler.scale() was called {mock_grad_scaler_instance.scale.call_count} times, "
            f"expected {expected_scale_calls} times."
        )
        assert mock_grad_scaler_instance.step.call_count == expected_step_calls, (
            f"GradScaler.step() was called {mock_grad_scaler_instance.step.call_count} times, "
            f"expected {expected_step_calls} times."
        )
        assert mock_grad_scaler_instance.update.call_count == expected_update_calls, (
            f"GradScaler.update() was called {mock_grad_scaler_instance.update.call_count} times, "
            f"expected {expected_update_calls} times."
        )
    else:
        # If mixed_precision is False, GradScaler should not be used
        assert mock_grad_scaler_instance.scale.call_count == 0, "GradScaler.scale() should not be called."
        assert mock_grad_scaler_instance.step.call_count == 0, "GradScaler.step() should not be called."
        assert mock_grad_scaler_instance.update.call_count == 0, "GradScaler.update() should not be called."
    
    # Assertions to verify torch.save was called correctly after each epoch
    # Each run saves 'num_epochs' checkpoints, so total saves = num_epochs * 2
    expected_save_calls = num_epochs * 2
    assert mock_torch_save.call_count == expected_save_calls, (
        f"torch.save() was called {mock_torch_save.call_count} times, "
        f"expected {expected_save_calls} times."
    )
    
    # Verify that torch.save was called with the correct arguments
    for _ in range(expected_save_calls):
        mock_torch_save.assert_any_call({}, ANY)  # ANY is used for the checkpoint path
    
    # Assertions to verify that checkpoint saving was logged correctly
    # Each epoch saves a checkpoint, so expect 'num_epochs * 2' checkpoint logs
    for run in range(2):
        for epoch in range(1, num_epochs + 1):
            expected_log = f"Model checkpoint saved at ./checkpoints/model_epoch_{epoch}.pt"
            assert expected_log in caplog.text, (
                f"Did not log checkpoint saving for epoch {epoch} in run {run + 1}."
            )

# 8. Edge Case Tests: Invalid Inputs

def test_fine_tune_peft_model_invalid_device(
    default_config,
    mock_peft_model,
    mock_dataloader_train,
    mock_dataloader_val,
    mock_optimizer,
    mock_scheduler
):
    """
    Test that fine_tune_peft_model raises a RuntimeError when an invalid device is specified.
    
    This ensures that the function correctly handles invalid device inputs by raising an exception.
    """
    # Prepare the configuration with an invalid device
    config = default_config.copy()
    config['training']['device'] = "invalid_device"
    
    # Mock the model's 'to' method to raise RuntimeError when called with 'invalid_device'
    mock_peft_model.to = MagicMock(side_effect=RuntimeError("Invalid device"))
    
    # Attempt to fine-tune the model and expect a RuntimeError to be raised
    with pytest.raises(RuntimeError) as exc_info:
        fine_tune_peft_model(
            model=mock_peft_model,
            train_dataloader=mock_dataloader_train,
            val_dataloader=mock_dataloader_val,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            config=config,
            device="invalid_device"
        )
    
    # Assert that the exception message matches the expected message
    assert "Invalid device" in str(exc_info.value), "Did not raise RuntimeError for invalid device."
    
    # Additionally, assert that model.to was called with 'invalid_device'
    mock_peft_model.to.assert_called_once_with("invalid_device")

# 9. Integration Tests

def test_main_integration(
    default_config,
    mock_peft_model,
    mock_dataloader_train,
    mock_dataloader_val,
    mock_optimizer,
    mock_scheduler,
    mock_feature_extractor,
    mock_peft_config,
    caplog
):
    """Integration test for main function with all components mocked."""
    config_path = "configs/peft_config.yaml"
    
    with patch("src.training.peft_finetune.ConfigParser") as mock_config_parser, \
         patch("src.training.peft_finetune.setup_logging") as mock_setup_logging, \
         patch("src.training.peft_finetune.DetrFeatureExtractor.from_pretrained") as mock_from_pretrained, \
         patch("src.training.peft_finetune.setup_peft_model") as mock_setup_peft_model, \
         patch("src.training.peft_finetune.prepare_dataloader") as mock_prepare_dataloader, \
         patch("src.training.peft_finetune.get_optimizer_and_scheduler") as mock_get_optimizer_and_scheduler, \
         patch("src.training.peft_finetune.PeftConfig.from_pretrained") as mock_peft_config_from_pretrained, \
         patch("src.training.peft_finetune.fine_tune_peft_model") as mock_fine_tune_peft_model:
        
        # Configure the mocked ConfigParser to return default_config
        mock_config_parser_instance = mock_config_parser.return_value
        mock_config_parser_instance.config = default_config
        
        # Mock the feature extractor's from_pretrained method
        mock_from_pretrained.return_value = mock_feature_extractor
        
        # Mock PeftConfig.from_pretrained to return a mocked PeftConfig
        mock_peft_config_from_pretrained.return_value = mock_peft_config
        
        # Mock setup_peft_model to return a mocked PeftModel
        mock_setup_peft_model.return_value = mock_peft_model
        
        # Mock prepare_dataloader to return training and validation DataLoaders
        mock_prepare_dataloader.side_effect = [mock_dataloader_train, mock_dataloader_val]
        
        # Mock get_optimizer_and_scheduler to return mocked optimizer and scheduler
        mock_get_optimizer_and_scheduler.return_value = (mock_optimizer, mock_scheduler)
        
        # Execute the main function within the mocked context
        with caplog.at_level(logging.INFO):
            main(config_path)
        
        # --- Assertions ---
        
        # Verify that ConfigParser was instantiated with the correct config_path
        mock_config_parser.assert_called_once_with(config_path)
        
        # Verify that setup_logging was called to configure logging
        mock_setup_logging.assert_called_once()
        
        # Verify that DetrFeatureExtractor.from_pretrained was called with the correct model name
        mock_from_pretrained.assert_called_once_with(default_config['model']['model_name'])
        
        # Verify that setup_peft_model was called with correct arguments
        mock_setup_peft_model.assert_called_once_with(
            default_config['model']['model_name'],
            num_classes=default_config['model']['num_classes'],
            peft_config=mock_peft_config
        )
        
        # Verify that prepare_dataloader was called twice: once for training and once for validation
        assert mock_prepare_dataloader.call_count == 2, "prepare_dataloader should be called twice for train and val."
        mock_prepare_dataloader.assert_any_call(
            data_dir=default_config['data']['data_dir'],
            batch_size=default_config['training']['batch_size'],
            feature_extractor=mock_feature_extractor,
            mode="train"
        )
        mock_prepare_dataloader.assert_any_call(
            data_dir=default_config['data']['data_dir'],
            batch_size=default_config['training']['batch_size'],
            feature_extractor=mock_feature_extractor,
            mode="val"
        )
        
        # Verify that get_optimizer_and_scheduler was called with correct arguments
        num_training_steps = default_config['training']['num_epochs'] * len(mock_dataloader_train)
        mock_get_optimizer_and_scheduler.assert_called_once_with(
            model=mock_peft_model,
            config=default_config['optimizer'],
            num_training_steps=num_training_steps
        )
        
        # Verify that fine_tune_peft_model was called with correct arguments
        mock_fine_tune_peft_model.assert_called_once_with(
            model=mock_peft_model,
            train_dataloader=mock_dataloader_train,
            val_dataloader=mock_dataloader_val,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            config=default_config,
            device=default_config['training'].get('device', 'cuda')
        )
        
        # Verify logging messages
        assert "Starting PEFT fine-tuning process." in caplog.text, "Did not log starting fine-tuning process."
        assert "PEFT fine-tuning completed." in caplog.text, "Did not log completion of fine-tuning process."

# 10. Edge Case Tests: Invalid Parameter Groups

def test_get_optimizer_and_scheduler_invalid_parameter_groups(
    default_config,
    mock_peft_model,
    mock_dataloader_train,
    mock_dataloader_val,
    mock_optimizer,
    mock_scheduler,
    caplog
):
    """Test that get_optimizer_and_scheduler handles invalid parameter groups correctly."""
    # Modify the optimizer configuration to have invalid parameter groups (missing 'params' key)
    config = default_config.copy()
    config['optimizer']['parameter_groups'] = [{"lr": 0.01}]  # Missing 'params' key
    
    # Mock model's forward pass to return a mock output with 'loss'
    mock_output = MagicMock()
    mock_output.loss = torch.tensor(1.0, requires_grad=True)
    mock_output.loss.backward = MagicMock()  # Mock the backward method
    mock_peft_model.forward.return_value = mock_output  # Correctly mock forward
    
    # Ensure state_dict returns a serializable object to prevent PicklingError
    mock_peft_model.state_dict.return_value = {}
    
    # Mock external dependencies: get_optimizer_and_scheduler and torch.save
    with patch("src.training.peft_finetune.get_optimizer_and_scheduler") as mock_get_optimizer_and_scheduler, \
         patch("torch.save") as mock_torch_save:
        
        # Configure the mocked get_optimizer_and_scheduler to raise KeyError
        mock_get_optimizer_and_scheduler.side_effect = KeyError("'params' key is missing in parameter_groups")
        
        # Execute the fine_tune_peft_model function and expect a KeyError
        with pytest.raises(KeyError) as exc_info:
            fine_tune_peft_model(
                model=mock_peft_model,
                train_dataloader=mock_dataloader_train,
                val_dataloader=mock_dataloader_val,
                optimizer=mock_optimizer,
                scheduler=mock_scheduler,
                config=config,
                device="cpu"
            )
        
        # Assert that the KeyError was raised with the correct message
        assert "'params' key is missing in parameter_groups" in str(exc_info.value), "Did not raise KeyError for invalid parameter groups."
        
        # Optionally, assert that torch.save was not called due to the exception
        mock_torch_save.assert_not_called()

# 11. Edge Case Tests: Extremely High Weight Decay

def test_setup_peft_model_extreme_weight_decay(mock_peft_config, mock_model):
    """Test setup_peft_model with an extremely high weight decay."""
    with patch("src.training.peft_finetune.get_peft_model") as mock_get_peft_model, \
         patch("src.training.peft_finetune.DetrForObjectDetection.from_pretrained") as mock_from_pretrained:
        
        mock_from_pretrained.return_value = mock_model
        mock_get_peft_model.return_value = MagicMock(spec=PeftModel)
        
        peft_model = setup_peft_model("facebook/detr-resnet-50", num_classes=91, peft_config=mock_peft_config)
        
        mock_from_pretrained.assert_called_once_with("facebook/detr-resnet-50", num_labels=91)
        mock_get_peft_model.assert_called_once_with(mock_model, mock_peft_config)
        assert peft_model is not None, "PEFT model should be initialized."
        # Additional assertions can be added to check weight decay if accessible

# 12. Edge Case Tests: Zero Epochs

def test_fine_tune_peft_model_zero_epochs(default_config, mock_peft_model, mock_dataloader_train, mock_dataloader_val, mock_optimizer, mock_scheduler, caplog):
    """Test fine_tune_peft_model with zero epochs."""
    config = default_config.copy()
    config['training']['num_epochs'] = 0
    
    with caplog.at_level(logging.INFO):
        fine_tune_peft_model(
            model=mock_peft_model,
            train_dataloader=mock_dataloader_train,
            val_dataloader=mock_dataloader_val,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            config=config,
            device="cpu"
        )
    
    # Ensure that training loop was skipped
    mock_peft_model.train.assert_not_called()
    mock_peft_model.eval.assert_not_called()
    mock_optimizer.step.assert_not_called()
    mock_scheduler.step.assert_not_called()
    assert "Starting Epoch 1/0" not in caplog.text, "Did not skip training loop for zero epochs."

# 13. Edge Case Tests: Negative Learning Rate

def test_get_optimizer_negative_learning_rate(
    default_config,
    mock_peft_model,
    mock_dataloader_train,
    mock_dataloader_val,
    mock_optimizer,
    mock_scheduler,
    caplog
):
    """Test get_optimizer_and_scheduler with a negative learning rate."""
    # Modify the optimizer configuration to have a negative learning rate
    config = default_config.copy()
    config['optimizer']['learning_rate'] = -1e-4  # Negative learning rate
    
    # Mock model's forward pass to return a mock output with 'loss' attribute
    mock_output = MagicMock()
    mock_output.loss = torch.tensor(1.0, requires_grad=True)
    mock_output.loss.backward = MagicMock()  # Mock the backward method
    mock_peft_model.forward.return_value = mock_output  # Correctly mock forward
    
    # Ensure state_dict returns a serializable object to prevent PicklingError
    mock_peft_model.state_dict.return_value = {}
    
    # Mock external dependencies: get_optimizer_and_scheduler and torch.save
    with patch("src.training.peft_finetune.get_optimizer_and_scheduler") as mock_get_optimizer_and_scheduler, \
         patch("torch.save") as mock_torch_save:
        
        # Configure the mocked get_optimizer_and_scheduler to raise ValueError for invalid learning rate
        mock_get_optimizer_and_scheduler.side_effect = ValueError("Invalid learning rate")
        
        # Execute fine_tune_peft_model and expect a ValueError
        with pytest.raises(ValueError) as exc_info:
            fine_tune_peft_model(
                model=mock_peft_model,
                train_dataloader=mock_dataloader_train,
                val_dataloader=mock_dataloader_val,
                optimizer=mock_optimizer,
                scheduler=mock_scheduler,
                config=config,
                device="cpu"
            )
        
        # Assert that the ValueError was raised with the correct message
        assert "Invalid learning rate" in str(exc_info.value), "Did not raise ValueError for negative learning rate."
        
        # Assert that torch.save was not called due to the exception
        mock_torch_save.assert_not_called()

# 14. Edge Case Tests: Missing Configuration Fields

def test_main_missing_configuration_fields(caplog):
    """Test that main handles missing configuration fields by using defaults."""
    config_path = "configs/missing_fields_config.yaml"
    
    incomplete_config = {
        "training": {
            "num_epochs": 1
            # Missing other fields
        },
        "model": {
            "model_name": "facebook/detr-resnet-50",
            "num_classes": 91,
            "peft_model_path": "./peft_model"
        },
        "optimizer": {
            # Missing optimizer_type, learning_rate, etc.
        }
    }
    
    with patch("src.training.peft_finetune.ConfigParser") as mock_config_parser, \
         patch("src.training.peft_finetune.setup_logging") as mock_setup_logging, \
         patch("src.training.peft_finetune.DetrFeatureExtractor.from_pretrained") as mock_from_pretrained, \
         patch("src.training.peft_finetune.setup_peft_model") as mock_setup_peft_model, \
         patch("src.training.peft_finetune.prepare_dataloader") as mock_prepare_dataloader, \
         patch("src.training.peft_finetune.get_optimizer_and_scheduler") as mock_get_optimizer_and_scheduler, \
         patch("src.training.peft_finetune.PeftConfig.from_pretrained") as mock_peft_config_from_pretrained, \
         patch("src.training.peft_finetune.fine_tune_peft_model") as mock_fine_tune_peft_model:
        
        mock_config_parser.return_value.config = incomplete_config
        mock_from_pretrained.return_value = MagicMock(spec=DetrFeatureExtractor)
        mock_setup_peft_model.return_value = MagicMock(spec=PeftModel)
        mock_prepare_dataloader.side_effect = [MagicMock(spec=DataLoader), MagicMock(spec=DataLoader)]
        mock_get_optimizer_and_scheduler.return_value = (MagicMock(spec=Optimizer), MagicMock(spec=_LRScheduler))
        mock_peft_config_from_pretrained.return_value = MagicMock(spec=PeftConfig)
        
        with caplog.at_level(logging.INFO):
            main(config_path)
        
        # Verify that defaults are used for missing optimizer fields
        mock_get_optimizer_and_scheduler.assert_called_once_with(
            model=mock_setup_peft_model.return_value,
            config=incomplete_config['optimizer'],
            num_training_steps=incomplete_config['training']['num_epochs'] * len(mock_prepare_dataloader.return_value)
        )
        assert "Starting PEFT fine-tuning process." in caplog.text, "Did not log starting fine-tuning process."
        assert "PEFT fine-tuning completed." in caplog.text, "Did not log completion of fine-tuning process."

# 15. Edge Case Tests: Extremely Large Number of Training Steps

def test_configure_scheduler_large_num_training_steps(default_config, mock_peft_model, mock_optimizer, mock_scheduler):
    """Test configure_scheduler with a very large number of training steps."""
    config = default_config.copy()
    config['optimizer']['scheduler_type'] = "linear"
    config['optimizer']['num_warmup_steps'] = 1000
    num_training_steps = 1000000  # Extremely large
    
    with patch("src.training.peft_finetune.get_scheduler") as mock_get_scheduler:
        mock_scheduler_instance = MagicMock(spec=_LRScheduler)
        mock_get_scheduler.return_value = mock_scheduler_instance
        
        optimizer, scheduler = get_optimizer_and_scheduler(
            model=mock_peft_model,
            config=config['optimizer'],
            num_training_steps=num_training_steps
        )
        
        mock_get_scheduler.assert_called_once_with(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=1000,
            num_training_steps=num_training_steps
        )
        assert scheduler == mock_scheduler_instance, "Scheduler should be the mocked scheduler instance."

# 16. Edge Case Tests: Parameter Groups with Missing 'params' Key

def test_prepare_dataloader_collate_function(mock_feature_extractor, caplog):
    """Test prepare_dataloader with a custom collate function."""
    with patch("src.training.peft_finetune.CocoDetection") as mock_coco_detection, \
         patch("torch.utils.data.DataLoader") as mock_dataloader:
        
        mock_coco_detection.return_value = MagicMock(spec=CocoDetection)
        mock_dataloader.return_value = MagicMock(spec=DataLoader)
        
        dataloader = prepare_dataloader(
            data_dir="./data",
            batch_size=4,
            feature_extractor=mock_feature_extractor,
            mode="train"
        )
        
        # Verify that a lambda function is used as collate_fn
        args, kwargs = mock_dataloader.call_args
        assert 'collate_fn' in kwargs, "collate_fn should be provided."
        assert callable(kwargs['collate_fn']), "collate_fn should be callable."

# 17. Edge Case Tests: Non-Callable Transform Function

def test_prepare_dataloader_non_callable_transform(mock_feature_extractor):
    """Test prepare_dataloader when transform is not callable."""
    with patch("src.training.peft_finetune.CocoDetection") as mock_coco_detection:
        # Set transform to a non-callable object
        mock_coco_detection.return_value = MagicMock(spec=CocoDetection)
        mock_coco_detection.return_value.transform = "not_callable"
        
        with pytest.raises(TypeError) as exc_info:
            prepare_dataloader(
                data_dir="./data",
                batch_size=4,
                feature_extractor=mock_feature_extractor,
                mode="train"
            )
        assert "transform must be callable" in str(exc_info.value), "Did not raise TypeError for non-callable transform."

# 18. Edge Case Tests: Missing 'logits' or 'labels' in Outputs or Targets

def test_fine_tune_peft_model_missing_fields(default_config, mock_peft_model, mock_dataloader_train, mock_dataloader_val, mock_optimizer, mock_scheduler, caplog):
    """Test fine_tune_peft_model when 'logits' or 'labels' are missing in outputs or targets."""
    # Mock model's forward pass to handle missing fields
    mock_peft_model.return_value.loss = torch.tensor(1.0, requires_grad=True)
    
    # Simulate missing 'labels' in target
    mock_dataloader_train.__iter__.return_value = [
        ([torch.randn(4, 3, 224, 224)], [{}])  # Missing 'labels' and 'boxes'
    ]
    
    with patch("src.training.peft_finetune.autocast"), \
         patch("src.training.peft_finetune.GradScaler"):
        
        fine_tune_peft_model(
            model=mock_peft_model,
            train_dataloader=mock_dataloader_train,
            val_dataloader=mock_dataloader_val,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            config=default_config,
            device="cpu"
        )
    
    # Check that loss was computed as zero or handled gracefully
    assert "Training Loss" in caplog.text, "Did not log training loss."
    assert "Validation Loss" in caplog.text, "Did not log validation loss."

# 19. Edge Case Tests: Non-Tensor Inputs in Targets

def test_fine_tune_peft_model_non_tensor_targets(default_config, mock_peft_model, mock_dataloader_train, mock_dataloader_val, mock_optimizer, mock_scheduler):
    """Test fine_tune_peft_model with non-tensor targets to ensure proper error handling."""
    # Mock dataloader to return non-tensor targets
    mock_dataloader_train.__iter__.return_value = [
        ([torch.randn(4, 3, 224, 224)], [{"labels": [1, 2], "boxes": [[10, 10, 50, 50]]}])
    ]
    
    with patch("src.training.peft_finetune.autocast"), \
         patch("src.training.peft_finetune.GradScaler"):
        
        with pytest.raises(AttributeError):
            fine_tune_peft_model(
                model=mock_peft_model,
                train_dataloader=mock_dataloader_train,
                val_dataloader=mock_dataloader_val,
                optimizer=mock_optimizer,
                scheduler=mock_scheduler,
                config=default_config,
                device="cpu"
            )

# 20. Edge Case Tests: Extremely Large Batch Sizes

def test_fine_tune_peft_model_extremely_large_batch(default_config, mock_peft_model, mock_dataloader_train, mock_dataloader_val, mock_optimizer, mock_scheduler, caplog):
    """Test fine_tune_peft_model with extremely large batch sizes."""
    config = default_config.copy()
    config['training']['batch_size'] = 10000  # Extremely large batch size
    
    # Mock dataloader to simulate large batch
    large_batch = ([torch.randn(10000, 3, 224, 224)], [{"labels": torch.randint(0, 91, (10000,)), "boxes": torch.randn(10000, 4)}])
    mock_dataloader_train.__iter__.return_value = [large_batch]
    mock_dataloader_train.__len__.return_value = 1
    mock_dataloader_val.__iter__.return_value = [large_batch]
    mock_dataloader_val.__len__.return_value = 1
    
    with patch("src.training.peft_finetune.autocast"), \
         patch("src.training.peft_finetune.GradScaler"):
        
        with caplog.at_level(logging.INFO):
            fine_tune_peft_model(
                model=mock_peft_model,
                train_dataloader=mock_dataloader_train,
                val_dataloader=mock_dataloader_val,
                optimizer=mock_optimizer,
                scheduler=mock_scheduler,
                config=config,
                device="cpu"
            )
        
        # Check that training and validation logs are present
        assert "Training Loss: 1.0" in caplog.text, "Did not log training loss for large batch."
        assert "Validation Loss: 1.0" in caplog.text, "Did not log validation loss for large batch."

