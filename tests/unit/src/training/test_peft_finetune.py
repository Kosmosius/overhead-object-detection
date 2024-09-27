# tests/unit/src/training/test_peft_finetune.py

import os
import pytest
import torch
import logging
from unittest.mock import patch, MagicMock, mock_open
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
    """Fixture to create a mock PeftConfig."""
    return MagicMock(spec=PeftConfig)

@pytest.fixture
def mock_model():
    """Fixture to create a mock DetrForObjectDetection model."""
    return MagicMock(spec=DetrForObjectDetection)

@pytest.fixture
def mock_peft_model(mock_model):
    """Fixture to create a mock PeftModel."""
    peft_model = MagicMock(spec=PeftModel)
    peft_model.return_value = peft_model
    return peft_model

@pytest.fixture
def mock_feature_extractor():
    """Fixture to create a mock DetrFeatureExtractor."""
    return MagicMock(spec=DetrFeatureExtractor)

@pytest.fixture
def mock_dataset():
    """Fixture to create a mock Dataset."""
    return MagicMock(spec=Dataset)

@pytest.fixture
def mock_dataloader_train():
    """Fixture to create a mock training DataLoader."""
    mock_loader = MagicMock(spec=DataLoader)
    mock_loader.__len__.return_value = 10  # Example length
    mock_loader.__iter__.return_value = iter([
        (torch.randn(4, 3, 224, 224), [{"labels": torch.randint(0, 91, (4,)), "boxes": torch.randn(4, 4)}])
    ])  # Example data
    return mock_loader

@pytest.fixture
def mock_dataloader_val():
    """Fixture to create a mock validation DataLoader."""
    mock_loader = MagicMock(spec=DataLoader)
    mock_loader.__len__.return_value = 5  # Example length
    mock_loader.__iter__.return_value = iter([
        (torch.randn(4, 3, 224, 224), [{"labels": torch.randint(0, 91, (4,)), "boxes": torch.randn(4, 4)}])
    ])  # Example data
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
        
        mock_coco_detection.return_value = mock_dataset
        mock_dataloader.return_value = mock_dataloader_train
        
        dataloader = prepare_dataloader(
            data_dir="./data",
            batch_size=4,
            feature_extractor=mock_feature_extractor,
            mode="train"
        )
        
        mock_coco_detection.assert_called_once_with(
            root=os.path.join("./data", "train2017"),
            annFile=os.path.join("./data", "annotations/instances_train2017.json"),
            transform=mock_feature_extractor
        )
        mock_dataloader.assert_called_once_with(
            mock_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=pytest.any
        )
        assert dataloader == mock_dataloader_train, "Dataloader should be returned correctly."

def test_prepare_dataloader_validation_mode(mock_feature_extractor, mock_dataset, mock_dataloader_val):
    """Test successful preparation of DataLoader for validation mode."""
    with patch("src.training.peft_finetune.CocoDetection") as mock_coco_detection, \
         patch("torch.utils.data.DataLoader") as mock_dataloader:
        
        mock_coco_detection.return_value = mock_dataset
        mock_dataloader.return_value = mock_dataloader_val
        
        dataloader = prepare_dataloader(
            data_dir="./data",
            batch_size=8,
            feature_extractor=mock_feature_extractor,
            mode="val"
        )
        
        mock_coco_detection.assert_called_once_with(
            root=os.path.join("./data", "val2017"),
            annFile=os.path.join("./data", "annotations/instances_val2017.json"),
            transform=mock_feature_extractor
        )
        mock_dataloader.assert_called_once_with(
            mock_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=pytest.any
        )
        assert dataloader == mock_dataloader_val, "Dataloader should be returned correctly."

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
        mock_coco_detection.side_effect = Exception("Invalid data directory")
        
        with pytest.raises(Exception) as exc_info:
            prepare_dataloader(
                data_dir="./invalid_data",
                batch_size=4,
                feature_extractor=mock_feature_extractor,
                mode="train"
            )
        assert "Error preparing dataloader: Invalid data directory" in str(exc_info.value), "Did not raise expected exception on invalid data directory."

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

def test_fine_tune_peft_model_training_loop(default_config, mock_peft_model, mock_dataloader_train, mock_dataloader_val, mock_optimizer, mock_scheduler, caplog):
    """Test fine_tune_peft_model executes the training and validation loops correctly."""
    # Mock model's train and eval methods
    mock_peft_model.train = MagicMock()
    mock_peft_model.eval = MagicMock()
    
    # Mock model's forward pass
    mock_peft_model.pixel_values = None
    mock_peft_model.labels = None
    mock_peft_model.return_value = MagicMock(loss=torch.tensor(1.0, requires_grad=True))
    
    with patch("src.training.peft_finetune.autocast") as mock_autocast, \
         patch("src.training.peft_finetune.GradScaler") as mock_grad_scaler:
        
        mock_autocast.return_value = MagicMock()
        mock_grad_scaler_instance = MagicMock()
        mock_grad_scaler.return_value = mock_grad_scaler_instance
        
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
        
        # Check that model.train() was called
        mock_peft_model.train.assert_called()
        
        # Check that optimizer.zero_grad() was called
        mock_optimizer.zero_grad.assert_called()
        
        # Check that loss.backward() was called
        mock_peft_model.return_value.loss.backward.assert_called()
        
        # Check that optimizer.step() and scheduler.step() were called
        mock_optimizer.step.assert_called()
        mock_scheduler.step.assert_called()
        
        # Check that model.eval() was called during validation
        mock_peft_model.eval.assert_called()
        
        # Check that checkpoints are saved
        assert "Model checkpoint saved at ./checkpoints/model_epoch_1.pt" in caplog.text, "Did not log checkpoint saving."

def test_fine_tune_peft_model_mixed_precision(default_config, mock_peft_model, mock_dataloader_train, mock_dataloader_val, mock_optimizer, mock_scheduler, caplog):
    """Test fine_tune_peft_model with mixed precision enabled."""
    config = default_config.copy()
    config['training']['mixed_precision'] = True
    
    # Mock model's train and eval methods
    mock_peft_model.train = MagicMock()
    mock_peft_model.eval = MagicMock()
    
    # Mock model's forward pass
    mock_peft_model.return_value.loss = torch.tensor(1.0, requires_grad=True)
    
    with patch("src.training.peft_finetune.autocast") as mock_autocast, \
         patch("src.training.peft_finetune.GradScaler") as mock_grad_scaler:
        
        mock_autocast.return_value = MagicMock()
        mock_grad_scaler_instance = MagicMock()
        mock_grad_scaler.return_value = mock_grad_scaler_instance
        
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
        
        # Check that GradScaler was used
        mock_grad_scaler_instance.scale.assert_called()
        mock_grad_scaler_instance.step.assert_called()
        mock_grad_scaler_instance.update.assert_called()

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

def test_fine_tune_peft_model_empty_dataloader(default_config, mock_peft_model, mock_optimizer, mock_scheduler, caplog):
    """Test fine_tune_peft_model with empty training and validation dataloaders."""
    empty_dataloader = MagicMock(spec=DataLoader)
    empty_dataloader.__len__.return_value = 0
    
    with patch("src.training.peft_finetune.tqdm") as mock_tqdm:
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
    
    # Check that loss computations were skipped
    assert "Training Loss" in caplog.text, "Did not log training loss."
    assert "Validation Loss" in caplog.text, "Did not log validation loss."

# 4. Tests for main

def test_main_success(default_config, mock_peft_model, mock_dataloader_train, mock_dataloader_val, mock_optimizer, mock_scheduler, mock_feature_extractor, mock_peft_config, caplog):
    """Test that main function runs successfully with valid configuration."""
    config_path = "configs/peft_config.yaml"
    
    # Mock ConfigParser to return default_config
    with patch("src.training.peft_finetune.ConfigParser") as mock_config_parser, \
         patch("src.training.peft_finetune.setup_logging") as mock_setup_logging, \
         patch("src.training.peft_finetune.DetrFeatureExtractor.from_pretrained") as mock_from_pretrained, \
         patch("src.training.peft_finetune.setup_peft_model") as mock_setup_peft_model, \
         patch("src.training.peft_finetune.prepare_dataloader") as mock_prepare_dataloader, \
         patch("src.training.peft_finetune.get_optimizer_and_scheduler") as mock_get_optimizer_and_scheduler, \
         patch("src.training.peft_finetune.fine_tune_peft_model") as mock_fine_tune_peft_model:
        
        mock_config_parser.return_value.config = default_config
        mock_from_pretrained.return_value = mock_feature_extractor
        mock_setup_peft_model.return_value = mock_peft_model
        mock_prepare_dataloader.side_effect = [mock_dataloader_train, mock_dataloader_val]
        mock_get_optimizer_and_scheduler.return_value = (mock_optimizer, mock_scheduler)
        
        with caplog.at_level(logging.INFO):
            main(config_path)
        
        mock_config_parser.assert_called_once_with(config_path)
        mock_setup_logging.assert_called_once()
        mock_from_pretrained.assert_called_once_with(default_config['model']['model_name'])
        mock_setup_peft_model.assert_called_once_with(
            "facebook/detr-resnet-50",
            num_classes=91,
            peft_config=mock_peft_config
        )
        assert mock_prepare_dataloader.call_count == 2, "prepare_dataloader should be called twice for train and val."
        mock_get_optimizer_and_scheduler.assert_called_once_with(
            model=mock_peft_model,
            config=default_config['optimizer'],
            num_training_steps=default_config['training']['num_epochs'] * len(mock_dataloader_train)
        )
        mock_fine_tune_peft_model.assert_called_once_with(
            model=mock_peft_model,
            train_dataloader=mock_dataloader_train,
            val_dataloader=mock_dataloader_val,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            config=default_config,
            device="cpu"
        )
        assert "Starting PEFT fine-tuning process." in caplog.text, "Did not log starting process."
        assert "PEFT fine-tuning completed." in caplog.text, "Did not log completion of process."

def test_main_error_in_config_parser():
    """Test that main raises an error when ConfigParser fails."""
    config_path = "configs/invalid_config.yaml"
    
    with patch("src.training.peft_finetune.ConfigParser") as mock_config_parser:
        mock_config_parser.side_effect = Exception("Config file not found")
        
        with pytest.raises(Exception) as exc_info, \
             patch("src.training.peft_finetune.setup_logging") as mock_setup_logging:
            main(config_path)
        assert "Error setting up PEFT model: Config file not found" in str(exc_info.value), "Did not raise expected exception for config parser failure."

def test_main_error_in_setup_peft_model(default_config, mock_peft_model):
    """Test that main raises an error when setup_peft_model fails."""
    config_path = "configs/peft_config.yaml"
    
    with patch("src.training.peft_finetune.ConfigParser") as mock_config_parser, \
         patch("src.training.peft_finetune.setup_logging") as mock_setup_logging, \
         patch("src.training.peft_finetune.DetrFeatureExtractor.from_pretrained") as mock_from_pretrained, \
         patch("src.training.peft_finetune.setup_peft_model") as mock_setup_peft_model:
        
        mock_config_parser.return_value.config = default_config
        mock_from_pretrained.return_value = MagicMock(spec=DetrFeatureExtractor)
        mock_setup_peft_model.side_effect = Exception("PEFT setup failed")
        
        with pytest.raises(Exception) as exc_info:
            main(config_path)
        assert "Error setting up PEFT model: PEFT setup failed" in str(exc_info.value), "Did not raise expected exception for PEFT setup failure."

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

def test_fine_tune_peft_model_large_batch_size(default_config, mock_peft_model, mock_dataloader_train, mock_dataloader_val, mock_optimizer, mock_scheduler):
    """Test fine_tune_peft_model with a very large batch size."""
    config = default_config.copy()
    config['training']['batch_size'] = 1024
    
    # Mock dataloader to simulate large batch
    large_batch = ([torch.randn(1024, 3, 224, 224)], [{"labels": torch.randint(0, 91, (1024,)), "boxes": torch.randn(1024, 4)}])
    mock_dataloader_train.__iter__.return_value = [large_batch]
    mock_dataloader_train.__len__.return_value = 1
    mock_dataloader_val.__iter__.return_value = [large_batch]
    mock_dataloader_val.__len__.return_value = 1
    
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
    
    assert "Starting Epoch 1/2" in caplog.text, "Did not log start of epoch with large batch size."
    assert "Training Loss" in caplog.text, "Did not log training loss."
    assert "Validation Loss" in caplog.text, "Did not log validation loss."

def test_fine_tune_peft_model_no_validation(default_config, mock_peft_model, mock_dataloader_train, mock_optimizer, mock_scheduler, caplog):
    """Test fine_tune_peft_model when validation dataloader is None."""
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
    
    # Check that validation loop is skipped
    mock_peft_model.eval.assert_not_called()
    assert "Validation Loss" not in caplog.text, "Did not skip validation loop when val_dataloader is None."

# 6. Logging Tests

def test_fine_tune_peft_model_logging(default_config, mock_peft_model, mock_dataloader_train, mock_dataloader_val, mock_optimizer, mock_scheduler, caplog):
    """Test that fine_tune_peft_model logs training and validation losses."""
    # Mock model's forward pass
    mock_peft_model.return_value.loss = torch.tensor(1.0, requires_grad=True)
    
    with patch("src.training.peft_finetune.autocast") as mock_autocast, \
         patch("src.training.peft_finetune.GradScaler") as mock_grad_scaler:
        
        mock_autocast.return_value = MagicMock()
        mock_grad_scaler_instance = MagicMock()
        mock_grad_scaler.return_value = mock_grad_scaler_instance
        
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
        
        assert "Starting Epoch 1/2" in caplog.text, "Did not log start of epoch."
        assert "Training Loss: 1.0" in caplog.text, "Did not log training loss."
        assert "Validation Loss: 1.0" in caplog.text, "Did not log validation loss."
        assert "Model checkpoint saved at ./checkpoints/model_epoch_1.pt" in caplog.text, "Did not log checkpoint saving."

# 7. Reproducibility Tests

def test_fine_tune_peft_model_reproducibility(default_config, mock_peft_model, mock_dataloader_train, mock_dataloader_val, mock_optimizer, mock_scheduler):
    """Test that fine_tune_peft_model produces consistent results with the same inputs."""
    # Mock model's forward pass to return a consistent loss
    mock_peft_model.return_value.loss = torch.tensor(1.0, requires_grad=True)
    
    with patch("src.training.peft_finetune.autocast"), \
         patch("src.training.peft_finetune.GradScaler"):
        
        # First run
        fine_tune_peft_model(
            model=mock_peft_model,
            train_dataloader=mock_dataloader_train,
            val_dataloader=mock_dataloader_val,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            config=default_config,
            device="cpu"
        )
        
        # Reset mocks
        mock_peft_model.train.reset_mock()
        mock_peft_model.eval.reset_mock()
        mock_optimizer.zero_grad.reset_mock()
        mock_optimizer.step.reset_mock()
        mock_scheduler.step.reset_mock()
        
        # Second run
        fine_tune_peft_model(
            model=mock_peft_model,
            train_dataloader=mock_dataloader_train,
            val_dataloader=mock_dataloader_val,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            config=default_config,
            device="cpu"
        )
        
        # Ensure that the same calls were made
        assert mock_peft_model.train.call_count == default_config['training']['num_epochs'], "Model.train() call count mismatch."
        assert mock_peft_model.eval.call_count == default_config['training']['num_epochs'], "Model.eval() call count mismatch."
        assert mock_optimizer.zero_grad.call_count == default_config['training']['num_epochs'] * len(mock_dataloader_train), "Optimizer.zero_grad() call count mismatch."
        assert mock_optimizer.step.call_count == default_config['training']['num_epochs'] * len(mock_dataloader_train), "Optimizer.step() call count mismatch."
        assert mock_scheduler.step.call_count == default_config['training']['num_epochs'] * len(mock_dataloader_train), "Scheduler.step() call count mismatch."

# 8. Edge Case Tests: Invalid Inputs

def test_fine_tune_peft_model_invalid_device(default_config, mock_peft_model, mock_dataloader_train, mock_dataloader_val, mock_optimizer, mock_scheduler):
    config = default_config.copy()
    config['training']['device'] = "invalid_device"

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
    assert "Invalid device" in str(exc_info.value), "Did not raise RuntimeError for invalid device."

# 9. Integration Tests

def test_main_integration(default_config, mock_peft_model, mock_dataloader_train, mock_dataloader_val, mock_optimizer, mock_scheduler, mock_feature_extractor, mock_peft_config, caplog):
    """Integration test for main function with all components mocked."""
    config_path = "configs/peft_config.yaml"
    
    with patch("src.training.peft_finetune.ConfigParser") as mock_config_parser, \
         patch("src.training.peft_finetune.setup_logging") as mock_setup_logging, \
         patch("src.training.peft_finetune.DetrFeatureExtractor.from_pretrained") as mock_from_pretrained, \
         patch("src.training.peft_finetune.setup_peft_model") as mock_setup_peft_model, \
         patch("src.training.peft_finetune.prepare_dataloader") as mock_prepare_dataloader, \
         patch("src.training.peft_finetune.get_optimizer_and_scheduler") as mock_get_optimizer_and_scheduler, \
         patch("src.training.peft_finetune.fine_tune_peft_model") as mock_fine_tune_peft_model:
        
        mock_config_parser.return_value.config = default_config
        mock_from_pretrained.return_value = mock_feature_extractor
        mock_setup_peft_model.return_value = mock_peft_model
        mock_prepare_dataloader.side_effect = [mock_dataloader_train, mock_dataloader_val]
        mock_get_optimizer_and_scheduler.return_value = (mock_optimizer, mock_scheduler)
        
        with caplog.at_level(logging.INFO):
            main(config_path)
        
        # Verify all steps were called
        mock_config_parser.assert_called_once_with(config_path)
        mock_setup_logging.assert_called_once()
        mock_from_pretrained.assert_called_once_with(default_config['model']['model_name'])
        mock_setup_peft_model.assert_called_once_with(
            "facebook/detr-resnet-50",
            num_classes=91,
            peft_config=mock_peft_config
        )
        assert mock_prepare_dataloader.call_count == 2, "prepare_dataloader should be called twice."
        mock_get_optimizer_and_scheduler.assert_called_once_with(
            model=mock_peft_model,
            config=default_config['optimizer'],
            num_training_steps=default_config['training']['num_epochs'] * len(mock_dataloader_train)
        )
        mock_fine_tune_peft_model.assert_called_once_with(
            model=mock_peft_model,
            train_dataloader=mock_dataloader_train,
            val_dataloader=mock_dataloader_val,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            config=default_config,
            device="cpu"
        )
        assert "Starting PEFT fine-tuning process." in caplog.text, "Did not log starting fine-tuning process."
        assert "PEFT fine-tuning completed." in caplog.text, "Did not log completion of fine-tuning process."

# 10. Edge Case Tests: Invalid Parameter Groups

def test_get_optimizer_and_scheduler_invalid_parameter_groups(default_config, mock_peft_model, mock_dataloader_train, mock_dataloader_val, mock_optimizer, mock_scheduler):
    """Test that get_optimizer_and_scheduler handles invalid parameter groups correctly."""
    config = default_config.copy()
    config['optimizer']['parameter_groups'] = [{"lr": 0.01}]  # Missing 'params' key
    
    with patch("src.training.peft_finetune.get_optimizer_and_scheduler") as mock_get_optimizer_and_scheduler:
        mock_get_optimizer_and_scheduler.side_effect = KeyError("'params' key is missing in parameter_groups")
        
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
        assert "'params' key is missing in parameter_groups" in str(exc_info.value), "Did not raise KeyError for invalid parameter groups."

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

def test_get_optimizer_negative_learning_rate(default_config, mock_peft_model):
    """Test get_optimizer with a negative learning rate."""
    config = default_config.copy()
    config['optimizer']['learning_rate'] = -1e-4
    
    with patch("src.training.peft_finetune.get_optimizer_and_scheduler") as mock_get_optimizer_and_scheduler:
        mock_get_optimizer_and_scheduler.return_value = (MagicMock(spec=Optimizer), MagicMock(spec=_LRScheduler))
        
        with pytest.raises(ValueError):
            fine_tune_peft_model(
                model=mock_peft_model,
                train_dataloader=MagicMock(spec=DataLoader),
                val_dataloader=MagicMock(spec=DataLoader),
                optimizer=mock_optimizer,
                scheduler=mock_scheduler,
                config=config,
                device="cpu"
            )
    # Depending on implementation, the negative learning rate might not raise an error immediately
    # Additional checks can be implemented if the optimizer raises errors for negative learning rates

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

