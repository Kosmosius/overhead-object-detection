# tests/unit/src/training/test_trainer.py

import os
import pytest
import torch
import logging
from unittest.mock import patch, MagicMock, mock_open
from src.training.trainer import (
    save_checkpoint,
    train_epoch,
    validate_epoch,
    train_model,
    main
)
from transformers import get_scheduler
from src.models.model_factory import ModelFactory
from src.evaluation.evaluator import Evaluator
from src.training.loss_functions import compute_loss
from src.data.dataloader import get_dataloader
from src.utils.config_parser import ConfigParser
from src.utils.logging_utils import setup_logging
from src.training.optimizers import get_optimizer_and_scheduler

# --- Fixtures ---

@pytest.fixture
def mock_model_instance():
    """Fixture to create a mock model instance with a save method."""
    mock_instance = MagicMock()
    mock_instance.save = MagicMock()
    mock_instance.model = MagicMock(spec=torch.nn.Module)
    return mock_instance

@pytest.fixture
def mock_optimizer():
    """Fixture to create a mock Optimizer."""
    return MagicMock(spec=torch.optim.Optimizer)

@pytest.fixture
def mock_scheduler():
    """Fixture to create a mock Scheduler."""
    return MagicMock(spec=torch.optim.lr_scheduler._LRScheduler)

@pytest.fixture
def mock_dataloader():
    """Fixture to create a mock DataLoader."""
    mock_dl = MagicMock()
    mock_dl.__len__.return_value = 10
    mock_dl.__iter__.return_value = iter([([torch.randn(3, 224, 224)], [{"labels": torch.tensor([1, 2]), "boxes": torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]])}]) for _ in range(10)])
    return mock_dl

@pytest.fixture
def mock_evaluator():
    """Fixture to create a mock Evaluator."""
    evaluator = MagicMock(spec=Evaluator)
    evaluator.evaluate.return_value = {"AP": 0.75}
    return evaluator

@pytest.fixture
def default_config():
    """Fixture for default training configuration."""
    return {
        "training": {
            "num_epochs": 2,
            "gradient_clipping": 1.0,
            "early_stopping_patience": 1,
            "checkpoint_dir": "./checkpoints",
            "batch_size": 4,
            "device": "cpu"
        },
        "data": {
            "data_dir": "./data"
        },
        "model": {
            "model_type": "detr",
            "model_name": "facebook/detr-resnet-50",
            "num_classes": 91
        },
        "optimizer": {
            "optimizer_type": "adamw",
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "scheduler_type": "linear",
            "num_warmup_steps": 0
        },
        "loss": {
            "classification_loss_weight": 1.0,
            "bbox_loss_weight": 1.0
        }
    }

@pytest.fixture
def mock_config_parser(default_config):
    """Fixture to mock ConfigParser."""
    with patch.object(ConfigParser, 'config', default_config):
        yield ConfigParser

# --- Test Cases ---

# 1. Tests for save_checkpoint

def test_save_checkpoint_success(mock_model_instance, mock_optimizer, mock_scheduler, default_config, caplog):
    """Test successful saving of a checkpoint."""
    epoch = 0
    with caplog.at_level(logging.INFO):
        save_checkpoint(
            model_instance=mock_model_instance,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            epoch=epoch,
            checkpoint_dir=default_config['training']['checkpoint_dir'],
            best_model=False
        )
    
    # Verify that model_instance.save was called with correct path and metadata
    expected_path = os.path.join(default_config['training']['checkpoint_dir'], f"model_epoch_{epoch + 1}")
    mock_model_instance.save.assert_called_once_with(
        expected_path,
        metadata={
            "epoch": epoch + 1,
            "optimizer_state_dict": mock_optimizer.state_dict(),
            "scheduler_state_dict": mock_scheduler.state_dict()
        }
    )
    
    # Verify logging
    assert f"Model checkpoint saved at {expected_path}" in caplog.text, "Checkpoint saving log not found."

def test_save_checkpoint_best_model(mock_model_instance, mock_optimizer, mock_scheduler, default_config, caplog):
    """Test saving of the best model."""
    epoch = 1
    with caplog.at_level(logging.INFO):
        save_checkpoint(
            model_instance=mock_model_instance,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            epoch=epoch,
            checkpoint_dir=default_config['training']['checkpoint_dir'],
            best_model=True
        )
    
    # Verify that model_instance.save was called with 'best_model' path
    expected_path = os.path.join(default_config['training']['checkpoint_dir'], "best_model")
    mock_model_instance.save.assert_called_once_with(
        expected_path,
        metadata={
            "epoch": epoch + 1,
            "optimizer_state_dict": mock_optimizer.state_dict(),
            "scheduler_state_dict": mock_scheduler.state_dict()
        }
    )
    
    # Verify logging
    assert f"Model checkpoint saved at {expected_path}" in caplog.text, "Best model checkpoint saving log not found."

def test_save_checkpoint_error(mock_model_instance, mock_optimizer, mock_scheduler, default_config, caplog):
    """Test that save_checkpoint raises an error when model_instance.save fails."""
    mock_model_instance.save.side_effect = Exception("Save failed")
    
    epoch = 0
    with pytest.raises(Exception) as exc_info, caplog.at_level(logging.ERROR):
        save_checkpoint(
            model_instance=mock_model_instance,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            epoch=epoch,
            checkpoint_dir=default_config['training']['checkpoint_dir'],
            best_model=False
        )
    
    assert "Error saving checkpoint: Save failed" in caplog.text, "Error logging for checkpoint saving not found."
    assert "Save failed" in str(exc_info.value), "Exception message mismatch."

# 2. Tests for train_epoch

def test_train_epoch_success(mock_model_instance, mock_dataloader, mock_optimizer, mock_scheduler, default_config):
    """Test successful execution of train_epoch."""
    # Mock compute_loss to return a dictionary with 'total_loss'
    with patch("src.training.trainer.compute_loss") as mock_compute_loss:
        mock_compute_loss.return_value = {"total_loss": torch.tensor(1.0)}
        
        avg_loss = train_epoch(
            model_instance=mock_model_instance,
            dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            device=torch.device(default_config['training']['device']),
            gradient_clipping=default_config['training']['gradient_clipping'],
            loss_config=default_config['loss']
        )
        
        # Assertions
        assert avg_loss == 1.0, "Average loss should be 1.0"
        mock_model_instance.model.train.assert_called_once()
        mock_optimizer.zero_grad.assert_called()
        assert mock_compute_loss.call_count == len(mock_dataloader), "compute_loss should be called for each batch"
        mock_optimizer.step.assert_called()
        mock_scheduler.step.assert_called()
        torch.nn.utils.clip_grad_norm_.assert_called_with(
            mock_model_instance.model.parameters(),
            default_config['training']['gradient_clipping']
        )

def test_train_epoch_gradient_clipping(mock_model_instance, mock_dataloader, mock_optimizer, mock_scheduler, default_config):
    """Test that gradient clipping is applied when specified."""
    # Mock compute_loss to return a dictionary with 'total_loss'
    with patch("src.training.trainer.compute_loss") as mock_compute_loss, \
         patch("torch.nn.utils.clip_grad_norm_") as mock_clip_grad:
        mock_compute_loss.return_value = {"total_loss": torch.tensor(1.0)}
        
        train_epoch(
            model_instance=mock_model_instance,
            dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            device=torch.device(default_config['training']['device']),
            gradient_clipping=default_config['training']['gradient_clipping'],
            loss_config=default_config['loss']
        )
        
        # Verify that gradient clipping was called
        mock_clip_grad.assert_called_with(
            mock_model_instance.model.parameters(),
            default_config['training']['gradient_clipping']
        )

def test_train_epoch_no_gradient_clipping(mock_model_instance, mock_dataloader, mock_optimizer, mock_scheduler, default_config):
    """Test that gradient clipping is not applied when not specified."""
    # Mock compute_loss to return a dictionary with 'total_loss'
    with patch("src.training.trainer.compute_loss") as mock_compute_loss, \
         patch("torch.nn.utils.clip_grad_norm_") as mock_clip_grad:
        mock_compute_loss.return_value = {"total_loss": torch.tensor(1.0)}
        
        train_epoch(
            model_instance=mock_model_instance,
            dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            device=torch.device(default_config['training']['device']),
            gradient_clipping=None,
            loss_config=default_config['loss']
        )
        
        # Verify that gradient clipping was not called
        mock_clip_grad.assert_not_called()

def test_train_epoch_error_during_training(mock_model_instance, mock_dataloader, mock_optimizer, mock_scheduler, default_config):
    """Test that train_epoch raises an error when compute_loss fails."""
    with patch("src.training.trainer.compute_loss") as mock_compute_loss:
        mock_compute_loss.side_effect = Exception("Loss computation failed")
        
        with pytest.raises(Exception) as exc_info:
            train_epoch(
                model_instance=mock_model_instance,
                dataloader=mock_dataloader,
                optimizer=mock_optimizer,
                scheduler=mock_scheduler,
                device=torch.device(default_config['training']['device']),
                gradient_clipping=default_config['training']['gradient_clipping'],
                loss_config=default_config['loss']
            )
        
        assert "Loss computation failed" in str(exc_info.value), "Exception message mismatch."

def test_train_epoch_empty_dataloader(mock_model_instance, mock_optimizer, mock_scheduler, default_config):
    """Test train_epoch with an empty dataloader."""
    empty_dataloader = MagicMock()
    empty_dataloader.__len__.return_value = 0
    empty_dataloader.__iter__.return_value = iter([])
    
    with patch("src.training.trainer.compute_loss") as mock_compute_loss:
        avg_loss = train_epoch(
            model_instance=mock_model_instance,
            dataloader=empty_dataloader,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            device=torch.device(default_config['training']['device']),
            gradient_clipping=default_config['training']['gradient_clipping'],
            loss_config=default_config['loss']
        )
        
        # Assertions
        assert avg_loss == 0.0, "Average loss should be 0.0 for empty dataloader"
        mock_compute_loss.assert_not_called()
        mock_optimizer.step.assert_not_called()
        mock_scheduler.step.assert_not_called()

# 3. Tests for validate_epoch

def test_validate_epoch_success(mock_model_instance, mock_dataloader, mock_evaluator, default_config):
    """Test successful execution of validate_epoch."""
    with patch("src.training.trainer.Evaluator", return_value=mock_evaluator) as mock_evaluator_class:
        metrics = validate_epoch(
            model_instance=mock_model_instance,
            dataloader=mock_dataloader,
            device=torch.device(default_config['training']['device'])
        )
        
        # Assertions
        mock_model_instance.model.eval.assert_called_once()
        mock_evaluator.evaluate.assert_called_once_with(mock_dataloader)
        assert metrics == {"AP": 0.75}, "Metrics should match the mocked evaluation results."

def test_validate_epoch_error_during_evaluation(mock_model_instance, mock_dataloader, default_config):
    """Test that validate_epoch raises an error when evaluation fails."""
    with patch("src.training.trainer.Evaluator") as mock_evaluator_class:
        mock_evaluator = MagicMock(spec=Evaluator)
        mock_evaluator.evaluate.side_effect = Exception("Evaluation failed")
        mock_evaluator_class.return_value = mock_evaluator
        
        with pytest.raises(Exception) as exc_info:
            validate_epoch(
                model_instance=mock_model_instance,
                dataloader=mock_dataloader,
                device=torch.device(default_config['training']['device'])
            )
        
        assert "Evaluation failed" in str(exc_info.value), "Exception message mismatch."

def test_validate_epoch_empty_dataloader(mock_model_instance, default_config):
    """Test validate_epoch with an empty dataloader."""
    empty_dataloader = MagicMock()
    empty_dataloader.__len__.return_value = 0
    empty_dataloader.__iter__.return_value = iter([])
    
    with patch("src.training.trainer.Evaluator") as mock_evaluator_class:
        mock_evaluator = MagicMock(spec=Evaluator)
        mock_evaluator.evaluate.return_value = {}
        mock_evaluator_class.return_value = mock_evaluator
        
        metrics = validate_epoch(
            model_instance=mock_model_instance,
            dataloader=empty_dataloader,
            device=torch.device(default_config['training']['device'])
        )
        
        # Assertions
        mock_evaluator.evaluate.assert_called_once_with(empty_dataloader)
        assert metrics == {}, "Metrics should be empty for empty dataloader."

# 4. Tests for train_model

def test_train_model_success(mock_model_instance, mock_dataloader, mock_optimizer, mock_scheduler, default_config, mock_evaluator, caplog):
    """Test successful execution of train_model."""
    with patch("src.training.trainer.train_epoch", return_value=1.0) as mock_train_epoch, \
         patch("src.training.trainer.validate_epoch", return_value={"AP": 0.75}) as mock_validate_epoch, \
         patch("src.training.trainer.save_checkpoint") as mock_save_checkpoint, \
         patch("src.training.trainer.Evaluator", return_value=mock_evaluator):
        
        with caplog.at_level(logging.INFO):
            train_model(
                model_instance=mock_model_instance,
                train_dataloader=mock_dataloader,
                val_dataloader=mock_dataloader,
                optimizer=mock_optimizer,
                scheduler=mock_scheduler,
                config=default_config,
                device=torch.device(default_config['training']['device']),
                checkpoint_dir=default_config['training']['checkpoint_dir']
            )
        
        # Assertions
        assert mock_train_epoch.call_count == default_config['training']['num_epochs'], "train_epoch should be called once per epoch."
        assert mock_validate_epoch.call_count == default_config['training']['num_epochs'], "validate_epoch should be called once per epoch."
        assert mock_save_checkpoint.call_count == default_config['training']['num_epochs'] + 1, "save_checkpoint should be called for each epoch and best model."
        
        # Check logging
        for epoch in range(default_config['training']['num_epochs']):
            assert f"Starting Epoch {epoch + 1}/{default_config['training']['num_epochs']}" in caplog.text, "Epoch start log missing."
            assert f"Epoch {epoch + 1} - Training Loss: 1.0" in caplog.text, "Training loss log missing."
            assert f"Epoch {epoch + 1} - Validation Metrics: {{'AP': 0.75}}" in caplog.text, "Validation metrics log missing."
            assert f"Model checkpoint saved at {default_config['training']['checkpoint_dir']}/model_epoch_{epoch + 1}" in caplog.text, "Checkpoint saving log missing."
        
        # Check best model saving
        assert "New best model saved at Epoch 1" in caplog.text, "Best model saving log missing."

def test_train_model_early_stopping(mock_model_instance, mock_dataloader, mock_optimizer, mock_scheduler, default_config, mock_evaluator, caplog):
    """Test that train_model triggers early stopping when no improvement is observed."""
    config = default_config.copy()
    config['training']['num_epochs'] = 3
    config['training']['early_stopping_patience'] = 1
    
    # Simulate decreasing AP metrics
    ap_metrics = [0.75, 0.70, 0.65]
    
    with patch("src.training.trainer.train_epoch", side_effect=[1.0, 1.0, 1.0]) as mock_train_epoch, \
         patch("src.training.trainer.validate_epoch", side_effect=[{"AP": ap_metrics[0]}, {"AP": ap_metrics[1]}, {"AP": ap_metrics[2]}]) as mock_validate_epoch, \
         patch("src.training.trainer.save_checkpoint") as mock_save_checkpoint, \
         patch("src.training.trainer.Evaluator", return_value=mock_evaluator):
        
        with caplog.at_level(logging.INFO):
            train_model(
                model_instance=mock_model_instance,
                train_dataloader=mock_dataloader,
                val_dataloader=mock_dataloader,
                optimizer=mock_optimizer,
                scheduler=mock_scheduler,
                config=config,
                device=torch.device(config['training']['device']),
                checkpoint_dir=config['training']['checkpoint_dir']
            )
        
        # Assertions
        assert mock_train_epoch.call_count == 3, "train_epoch should be called three times."
        assert mock_validate_epoch.call_count == 3, "validate_epoch should be called three times."
        assert mock_save_checkpoint.call_count == 4, "save_checkpoint should be called for each epoch and best model."
        
        # Check logging for early stopping
        assert "Early stopping triggered after 3 epochs with no improvement." in caplog.text, "Early stopping log missing."

def test_train_model_best_model_saving(mock_model_instance, mock_dataloader, mock_optimizer, mock_scheduler, default_config, mock_evaluator, caplog):
    """Test that the best model is saved when a new best validation metric is achieved."""
    ap_metrics = [0.75, 0.80, 0.78]
    
    with patch("src.training.trainer.train_epoch", side_effect=[1.0, 1.0, 1.0]) as mock_train_epoch, \
         patch("src.training.trainer.validate_epoch", side_effect=[{"AP": ap_metrics[0]}, {"AP": ap_metrics[1]}, {"AP": ap_metrics[2]}]) as mock_validate_epoch, \
         patch("src.training.trainer.save_checkpoint") as mock_save_checkpoint, \
         patch("src.training.trainer.Evaluator", return_value=mock_evaluator):
        
        with caplog.at_level(logging.INFO):
            train_model(
                model_instance=mock_model_instance,
                train_dataloader=mock_dataloader,
                val_dataloader=mock_dataloader,
                optimizer=mock_optimizer,
                scheduler=mock_scheduler,
                config=default_config,
                device=torch.device(default_config['training']['device']),
                checkpoint_dir=default_config['training']['checkpoint_dir']
            )
        
        # Verify that best model was saved when AP improved
        best_model_calls = [call for call in mock_save_checkpoint.call_args_list if call.kwargs.get('best_model')]
        assert len(best_model_calls) == 1, "Best model should be saved once."
        best_model_call = best_model_calls[0]
        expected_path = os.path.join(default_config['training']['checkpoint_dir'], "best_model")
        best_model_call.kwargs['checkpoint_dir'] == default_config['training']['checkpoint_dir']
        best_model_call.kwargs['best_model'] is True

        # Check logging for best model
        assert "New best model saved at Epoch 2" in caplog.text, "Best model saving log missing."

def test_train_model_error_during_training(mock_model_instance, mock_dataloader, mock_optimizer, mock_scheduler, default_config):
    """Test that train_model raises an error when an exception occurs during training."""
    with patch("src.training.trainer.train_epoch", side_effect=Exception("Training failed")) as mock_train_epoch, \
         patch("src.training.trainer.validate_epoch") as mock_validate_epoch, \
         patch("src.training.trainer.save_checkpoint") as mock_save_checkpoint, \
         patch("src.training.trainer.Evaluator", return_value=MagicMock(spec=Evaluator)):
        
        with pytest.raises(Exception) as exc_info:
            train_model(
                model_instance=mock_model_instance,
                train_dataloader=mock_dataloader,
                val_dataloader=mock_dataloader,
                optimizer=mock_optimizer,
                scheduler=mock_scheduler,
                config=default_config,
                device=torch.device(default_config['training']['device']),
                checkpoint_dir=default_config['training']['checkpoint_dir']
            )
        
        assert "Training failed" in str(exc_info.value), "Exception message mismatch."
        mock_save_checkpoint.assert_called_once_with(
            model_instance=mock_model_instance,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            epoch=0,
            checkpoint_dir=default_config['training']['checkpoint_dir']
        )

def test_train_model_zero_epochs(mock_model_instance, mock_dataloader, mock_optimizer, mock_scheduler, default_config, caplog):
    """Test that train_model handles zero epochs gracefully."""
    config = default_config.copy()
    config['training']['num_epochs'] = 0
    
    with caplog.at_level(logging.INFO):
        train_model(
            model_instance=mock_model_instance,
            train_dataloader=mock_dataloader,
            val_dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            config=config,
            device=torch.device(config['training']['device']),
            checkpoint_dir=config['training']['checkpoint_dir']
        )
    
    # Verify that training and validation loops were not executed
    mock_model_instance.model.train.assert_not_called()
    mock_model_instance.model.eval.assert_not_called()
    mock_optimizer.zero_grad.assert_not_called()
    mock_optimizer.step.assert_not_called()
    mock_scheduler.step.assert_not_called()
    
    # Verify logging
    assert "Starting Epoch 1/0" not in caplog.text, "Epoch start log should not be present for zero epochs."
    assert "Training completed." in caplog.text, "Training completion log missing."

def test_train_model_extremely_high_weight_decay(mock_model_instance, mock_dataloader, mock_optimizer, mock_scheduler, default_config):
    """Test that train_model handles extremely high weight decay values."""
    config = default_config.copy()
    config['optimizer']['weight_decay'] = 100.0  # Extremely high
    
    with patch("src.training.trainer.train_epoch", return_value=1.0), \
         patch("src.training.trainer.validate_epoch", return_value={"AP": 0.75}), \
         patch("src.training.trainer.save_checkpoint") as mock_save_checkpoint, \
         patch("src.training.trainer.Evaluator", return_value=MagicMock(spec=Evaluator)):
        
        train_model(
            model_instance=mock_model_instance,
            train_dataloader=mock_dataloader,
            val_dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            config=config,
            device=torch.device(config['training']['device']),
            checkpoint_dir=config['training']['checkpoint_dir']
        )
        
        # Verify that optimizer was set with high weight decay
        mock_optimizer.step.assert_called()
        mock_optimizer.zero_grad.assert_called()

# 5. Tests for main

def test_main_success(mock_model_instance, mock_dataloader, mock_optimizer, mock_scheduler, default_config, caplog):
    """Test successful execution of the main function."""
    config_path = "configs/train_config.yaml"
    
    with patch("src.training.trainer.ConfigParser") as mock_config_parser, \
         patch("src.training.trainer.setup_logging") as mock_setup_logging, \
         patch("src.training.trainer.ModelFactory.create_model", return_value=mock_model_instance) as mock_create_model, \
         patch("src.training.trainer.get_dataloader", return_value=mock_dataloader) as mock_get_dataloader, \
         patch("src.training.trainer.get_optimizer_and_scheduler", return_value=(mock_optimizer, mock_scheduler)) as mock_get_optimizer_scheduler, \
         patch("src.training.trainer.train_model") as mock_train_model, \
         patch("src.training.trainer.Evaluator"):
        
        # Mock ConfigParser to return default_config
        mock_config_parser_instance = MagicMock()
        mock_config_parser_instance.config = default_config
        mock_config_parser.return_value = mock_config_parser_instance
        
        # Mock feature extractor
        with patch("src.training.trainer.DetrFeatureExtractor.from_pretrained") as mock_feature_extractor:
            mock_feature_extractor.return_value = MagicMock()
            
            # Mock torch.save for final model saving
            with patch("src.training.trainer.os.path.join", side_effect=lambda *args: "/".join(args)), \
                 patch("src.training.trainer.ModelFactory.create_model", return_value=mock_model_instance), \
                 patch("src.training.trainer.ModelFactory") as mock_model_factory:
                
                # Mock model_instance.save
                mock_model_instance.save = MagicMock()
                
                main(config_path)
                
                # Assertions
                mock_config_parser.assert_called_once_with(config_path)
                mock_setup_logging.assert_called_once()
                mock_create_model.assert_called_once_with(
                    model_type=default_config['model']['model_type'],
                    model_name=default_config['model']['model_name'],
                    num_labels=default_config['model']['num_classes']
                )
                mock_get_dataloader.assert_any_call(
                    data_dir=default_config['data']['data_dir'],
                    batch_size=default_config['training']['batch_size'],
                    mode="train",
                    feature_extractor=MagicMock(),
                    num_workers=default_config['training'].get('num_workers', 4),
                    pin_memory=default_config['training'].get('pin_memory', True)
                )
                mock_get_dataloader.assert_any_call(
                    data_dir=default_config['data']['data_dir'],
                    batch_size=default_config['training']['batch_size'],
                    mode="val",
                    feature_extractor=MagicMock(),
                    num_workers=default_config['training'].get('num_workers', 4),
                    pin_memory=default_config['training'].get('pin_memory', True)
                )
                mock_get_optimizer_scheduler.assert_called_once_with(
                    model=mock_model_instance.model,
                    config={**default_config['optimizer'], **default_config['optimizer']},
                    num_training_steps=default_config['training']['num_epochs'] * len(mock_dataloader)
                )
                mock_train_model.assert_called_once_with(
                    model_instance=mock_model_instance,
                    train_dataloader=mock_dataloader,
                    val_dataloader=mock_dataloader,
                    optimizer=mock_optimizer,
                    scheduler=mock_scheduler,
                    config=default_config,
                    device=torch.device(default_config['training']['device']),
                    checkpoint_dir=default_config['training']['checkpoint_dir']
                )
                mock_model_instance.save.assert_called_once_with(
                    os.path.join(default_config['training']['output_dir'], 'final_model'),
                    metadata=mock_model_instance.state_dict()
                )
                
                # Check logging
                assert "Feature extractor 'facebook/detr-resnet-50' loaded successfully." in caplog.text, "Feature extractor loading log missing."
                assert f"Model '{default_config['model']['model_name']}' initialized and moved to device '{default_config['training']['device']}'." in caplog.text, "Model initialization log missing."
                assert "Training completed." in caplog.text, "Training completion log missing."

def test_main_error_in_config_parser():
    """Test that main raises an error when ConfigParser fails."""
    config_path = "configs/invalid_train_config.yaml"
    
    with patch("src.training.trainer.ConfigParser") as mock_config_parser, \
         patch("src.training.trainer.setup_logging") as mock_setup_logging:
        mock_config_parser.side_effect = Exception("Configuration file not found")
        
        with pytest.raises(Exception) as exc_info:
            main(config_path)
        
        assert "Configuration file not found" in str(exc_info.value), "Exception message mismatch."

def test_main_error_in_feature_extractor(mock_model_instance, default_config):
    """Test that main raises an error when feature extractor loading fails."""
    config_path = "configs/train_config.yaml"
    
    with patch("src.training.trainer.ConfigParser") as mock_config_parser, \
         patch("src.training.trainer.setup_logging") as mock_setup_logging, \
         patch("src.training.trainer.DetrFeatureExtractor.from_pretrained") as mock_feature_extractor:
        
        # Mock ConfigParser to return default_config
        mock_config_parser_instance = MagicMock()
        mock_config_parser_instance.config = default_config
        mock_config_parser.return_value = mock_config_parser_instance
        
        # Simulate feature extractor loading failure
        mock_feature_extractor.side_effect = Exception("Feature extractor loading failed")
        
        with pytest.raises(Exception) as exc_info, caplog.at_level(logging.ERROR):
            main(config_path)
        
        assert "Error loading feature extractor: Feature extractor loading failed" in caplog.text, "Error logging for feature extractor not found."
        assert "Feature extractor loading failed" in str(exc_info.value), "Exception message mismatch."

def test_main_error_in_model_initialization(mock_dataloader, default_config):
    """Test that main raises an error when model initialization fails."""
    config_path = "configs/train_config.yaml"
    
    with patch("src.training.trainer.ConfigParser") as mock_config_parser, \
         patch("src.training.trainer.setup_logging") as mock_setup_logging, \
         patch("src.training.trainer.ModelFactory.create_model") as mock_create_model, \
         patch("src.training.trainer.get_dataloader", return_value=mock_dataloader):
        
        # Mock ConfigParser to return default_config
        mock_config_parser_instance = MagicMock()
        mock_config_parser_instance.config = default_config
        mock_config_parser.return_value = mock_config_parser_instance
        
        # Simulate model creation failure
        mock_create_model.side_effect = Exception("Model initialization failed")
        
        with pytest.raises(Exception) as exc_info, caplog.at_level(logging.ERROR):
            main(config_path)
        
        assert "Error initializing model: Model initialization failed" in caplog.text, "Error logging for model initialization not found."
        assert "Model initialization failed" in str(exc_info.value), "Exception message mismatch."

def test_main_error_in_dataloader_preparation(mock_model_instance, default_config):
    """Test that main raises an error when dataloader preparation fails."""
    config_path = "configs/train_config.yaml"
    
    with patch("src.training.trainer.ConfigParser") as mock_config_parser, \
         patch("src.training.trainer.setup_logging") as mock_setup_logging, \
         patch("src.training.trainer.DetrFeatureExtractor.from_pretrained") as mock_feature_extractor, \
         patch("src.training.trainer.ModelFactory.create_model", return_value=mock_model_instance), \
         patch("src.training.trainer.get_dataloader") as mock_get_dataloader:
        
        # Mock ConfigParser to return default_config
        mock_config_parser_instance = MagicMock()
        mock_config_parser_instance.config = default_config
        mock_config_parser.return_value = mock_config_parser_instance
        
        # Simulate dataloader preparation failure
        mock_get_dataloader.side_effect = Exception("Dataloader preparation failed")
        
        with pytest.raises(Exception) as exc_info, caplog.at_level(logging.ERROR):
            main(config_path)
        
        assert "Error preparing dataloaders: Dataloader preparation failed" in caplog.text, "Error logging for dataloader preparation not found."
        assert "Dataloader preparation failed" in str(exc_info.value), "Exception message mismatch."

def test_main_error_in_optimizer_scheduler_setup(mock_model_instance, mock_dataloader, default_config):
    """Test that main raises an error when optimizer and scheduler setup fails."""
    config_path = "configs/train_config.yaml"
    
    with patch("src.training.trainer.ConfigParser") as mock_config_parser, \
         patch("src.training.trainer.setup_logging") as mock_setup_logging, \
         patch("src.training.trainer.DetrFeatureExtractor.from_pretrained") as mock_feature_extractor, \
         patch("src.training.trainer.ModelFactory.create_model", return_value=mock_model_instance), \
         patch("src.training.trainer.get_dataloader", return_value=mock_dataloader), \
         patch("src.training.trainer.get_optimizer_and_scheduler") as mock_get_optimizer_scheduler:
        
        # Mock ConfigParser to return default_config
        mock_config_parser_instance = MagicMock()
        mock_config_parser_instance.config = default_config
        mock_config_parser.return_value = mock_config_parser_instance
        
        # Simulate optimizer and scheduler setup failure
        mock_get_optimizer_scheduler.side_effect = Exception("Optimizer setup failed")
        
        with pytest.raises(Exception) as exc_info, caplog.at_level(logging.ERROR):
            main(config_path)
        
        assert "Error setting up optimizer and scheduler: Optimizer setup failed" in caplog.text, "Error logging for optimizer and scheduler setup not found."
        assert "Optimizer setup failed" in str(exc_info.value), "Exception message mismatch."

def test_main_final_model_saving(mock_model_instance, mock_dataloader, mock_optimizer, mock_scheduler, default_config, caplog):
    """Test that the final model is saved correctly after training."""
    config_path = "configs/train_config.yaml"
    
    with patch("src.training.trainer.ConfigParser") as mock_config_parser, \
         patch("src.training.trainer.setup_logging") as mock_setup_logging, \
         patch("src.training.trainer.DetrFeatureExtractor.from_pretrained") as mock_feature_extractor, \
         patch("src.training.trainer.ModelFactory.create_model", return_value=mock_model_instance), \
         patch("src.training.trainer.get_dataloader", return_value=mock_dataloader), \
         patch("src.training.trainer.get_optimizer_and_scheduler", return_value=(mock_optimizer, mock_scheduler)), \
         patch("src.training.trainer.train_model") as mock_train_model, \
         patch("src.training.trainer.ModelFactory"):
        
        # Mock ConfigParser to return default_config
        mock_config_parser_instance = MagicMock()
        mock_config_parser_instance.config = default_config
        mock_config_parser.return_value = mock_config_parser_instance
        
        # Mock feature extractor
        mock_feature_extractor.return_value = MagicMock()
        
        # Mock model_instance.save
        mock_model_instance.save = MagicMock()
        
        with caplog.at_level(logging.INFO):
            main(config_path)
        
        # Verify that the final model was saved
        final_model_path = os.path.join(default_config['training']['output_dir'], 'final_model')
        mock_model_instance.save.assert_called_once_with(final_model_path)
        assert f"Final model saved at {final_model_path}" in caplog.text, "Final model saving log missing."

# 6. Edge Case Tests

def test_train_model_zero_epochs(mock_model_instance, mock_dataloader, mock_optimizer, mock_scheduler, default_config, caplog):
    """Test that train_model handles zero epochs gracefully."""
    config = default_config.copy()
    config['training']['num_epochs'] = 0
    
    with patch("src.training.trainer.train_epoch"), \
         patch("src.training.trainer.validate_epoch"), \
         patch("src.training.trainer.save_checkpoint") as mock_save_checkpoint, \
         patch("src.training.trainer.Evaluator", return_value=MagicMock(spec=Evaluator)):
        
        with caplog.at_level(logging.INFO):
            train_model(
                model_instance=mock_model_instance,
                train_dataloader=mock_dataloader,
                val_dataloader=mock_dataloader,
                optimizer=mock_optimizer,
                scheduler=mock_scheduler,
                config=config,
                device=torch.device(config['training']['device']),
                checkpoint_dir=config['training']['checkpoint_dir']
            )
        
        # Verify that no training or validation was performed
        mock_save_checkpoint.assert_not_called()
        assert "Starting Epoch 1/0" not in caplog.text, "Epoch start log should not be present for zero epochs."
        assert "Training completed." in caplog.text, "Training completion log missing."

def test_train_model_large_weight_decay(mock_model_instance, mock_dataloader, mock_optimizer, mock_scheduler, default_config):
    """Test that train_model handles extremely high weight decay values."""
    config = default_config.copy()
    config['optimizer']['weight_decay'] = 100.0  # Extremely high weight decay
    
    with patch("src.training.trainer.train_epoch", return_value=1.0), \
         patch("src.training.trainer.validate_epoch", return_value={"AP": 0.75}), \
         patch("src.training.trainer.save_checkpoint"), \
         patch("src.training.trainer.Evaluator", return_value=MagicMock(spec=Evaluator)):
        
        train_model(
            model_instance=mock_model_instance,
            train_dataloader=mock_dataloader,
            val_dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            config=config,
            device=torch.device(config['training']['device']),
            checkpoint_dir=config['training']['checkpoint_dir']
        )
        
        # Verify that optimizer was set with high weight decay
        mock_optimizer.step.assert_called()

def test_train_epoch_zero_loss(mock_model_instance, mock_dataloader, mock_optimizer, mock_scheduler, default_config):
    """Test train_epoch when all losses are zero."""
    with patch("src.training.trainer.compute_loss") as mock_compute_loss:
        mock_compute_loss.return_value = {"total_loss": torch.tensor(0.0)}
        
        avg_loss = train_epoch(
            model_instance=mock_model_instance,
            dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            device=torch.device(default_config['training']['device']),
            gradient_clipping=default_config['training']['gradient_clipping'],
            loss_config=default_config['loss']
        )
        
        # Assertions
        assert avg_loss == 0.0, "Average loss should be 0.0 when all losses are zero"
        mock_optimizer.step.assert_called()
        mock_scheduler.step.assert_called()

def test_validate_epoch_zero_metrics(mock_model_instance, mock_dataloader, default_config):
    """Test validate_epoch when evaluator returns zero metrics."""
    with patch("src.training.trainer.Evaluator") as mock_evaluator_class:
        mock_evaluator = MagicMock(spec=Evaluator)
        mock_evaluator.evaluate.return_value = {"AP": 0.0}
        mock_evaluator_class.return_value = mock_evaluator
        
        metrics = validate_epoch(
            model_instance=mock_model_instance,
            dataloader=mock_dataloader,
            device=torch.device(default_config['training']['device'])
        )
        
        # Assertions
        mock_evaluator.evaluate.assert_called_once_with(mock_dataloader)
        assert metrics == {"AP": 0.0}, "Metrics should be zero when evaluator returns zero."

# 7. Logging Tests

def test_train_model_logging(mock_model_instance, mock_dataloader, mock_optimizer, mock_scheduler, default_config, mock_evaluator, caplog):
    """Test that train_model logs training and validation losses and checkpoint saving."""
    with patch("src.training.trainer.train_epoch", return_value=1.0) as mock_train_epoch, \
         patch("src.training.trainer.validate_epoch", return_value={"AP": 0.75}) as mock_validate_epoch, \
         patch("src.training.trainer.save_checkpoint") as mock_save_checkpoint, \
         patch("src.training.trainer.Evaluator", return_value=mock_evaluator):
        
        with caplog.at_level(logging.INFO):
            train_model(
                model_instance=mock_model_instance,
                train_dataloader=mock_dataloader,
                val_dataloader=mock_dataloader,
                optimizer=mock_optimizer,
                scheduler=mock_scheduler,
                config=default_config,
                device=torch.device(default_config['training']['device']),
                checkpoint_dir=default_config['training']['checkpoint_dir']
            )
        
        # Check that logs contain training and validation loss
        for epoch in range(default_config['training']['num_epochs']):
            assert f"Starting Epoch {epoch + 1}/{default_config['training']['num_epochs']}" in caplog.text, "Epoch start log missing."
            assert f"Epoch {epoch + 1} - Training Loss: 1.0" in caplog.text, "Training loss log missing."
            assert f"Epoch {epoch + 1} - Validation Metrics: {{'AP': 0.75}}" in caplog.text, "Validation metrics log missing."
            assert f"Model checkpoint saved at {default_config['training']['checkpoint_dir']}/model_epoch_{epoch + 1}" in caplog.text, "Checkpoint saving log missing."

# 8. Reproducibility Tests

def test_train_model_reproducibility(mock_model_instance, mock_dataloader, mock_optimizer, mock_scheduler, default_config, mock_evaluator):
    """Test that train_model produces consistent results with the same inputs across multiple runs."""
    with patch("src.training.trainer.train_epoch", return_value=1.0) as mock_train_epoch, \
         patch("src.training.trainer.validate_epoch", return_value={"AP": 0.75}) as mock_validate_epoch, \
         patch("src.training.trainer.save_checkpoint") as mock_save_checkpoint, \
         patch("src.training.trainer.Evaluator", return_value=mock_evaluator):
        
        # First run
        train_model(
            model_instance=mock_model_instance,
            train_dataloader=mock_dataloader,
            val_dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            config=default_config,
            device=torch.device(default_config['training']['device']),
            checkpoint_dir=default_config['training']['checkpoint_dir']
        )
        
        # Second run
        train_model(
            model_instance=mock_model_instance,
            train_dataloader=mock_dataloader,
            val_dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            config=default_config,
            device=torch.device(default_config['training']['device']),
            checkpoint_dir=default_config['training']['checkpoint_dir']
        )
        
        # Verify that train_epoch and validate_epoch were called twice per epoch
        assert mock_train_epoch.call_count == default_config['training']['num_epochs'] * 2, "train_epoch call count mismatch."
        assert mock_validate_epoch.call_count == default_config['training']['num_epochs'] * 2, "validate_epoch call count mismatch."
        assert mock_save_checkpoint.call_count == default_config['training']['num_epochs'] * 2, "save_checkpoint call count mismatch."

# 9. Edge Case Tests: Invalid Parameter Groups

def test_get_optimizer_and_scheduler_invalid_parameter_groups(default_config, mock_model_instance, mock_optimizer, mock_scheduler):
    """Test that get_optimizer_and_scheduler handles invalid parameter groups correctly."""
    config = default_config.copy()
    config['optimizer']['parameter_groups'] = [{"lr": 0.01}]  # Missing 'params' key
    
    with patch("src.training.trainer.get_optimizer_and_scheduler") as mock_get_optimizer_and_scheduler:
        mock_get_optimizer_and_scheduler.side_effect = KeyError("'params' key is missing in parameter_groups")
        
        with pytest.raises(KeyError) as exc_info:
            train_model(
                model_instance=mock_model_instance,
                train_dataloader=MagicMock(spec=torch.utils.data.DataLoader),
                val_dataloader=MagicMock(spec=torch.utils.data.DataLoader),
                optimizer=mock_optimizer,
                scheduler=mock_scheduler,
                config=config,
                device=torch.device(config['training']['device']),
                checkpoint_dir=config['training']['checkpoint_dir']
            )
        
        assert "'params' key is missing in parameter_groups" in str(exc_info.value), "Exception message mismatch."

# 10. Edge Case Tests: Extremely High Weight Decay

def test_train_model_extremely_high_weight_decay(mock_model_instance, mock_dataloader, mock_optimizer, mock_scheduler, default_config):
    """Test that train_model handles extremely high weight decay values."""
    config = default_config.copy()
    config['optimizer']['weight_decay'] = 1000.0  # Extremely high weight decay
    
    with patch("src.training.trainer.train_epoch", return_value=1.0), \
         patch("src.training.trainer.validate_epoch", return_value={"AP": 0.75}), \
         patch("src.training.trainer.save_checkpoint") as mock_save_checkpoint, \
         patch("src.training.trainer.Evaluator", return_value=MagicMock(spec=Evaluator)):
        
        train_model(
            model_instance=mock_model_instance,
            train_dataloader=mock_dataloader,
            val_dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            config=config,
            device=torch.device(config['training']['device']),
            checkpoint_dir=config['training']['checkpoint_dir']
        )
        
        # Verify that optimizer was set with high weight decay
        mock_optimizer.step.assert_called()

# 11. Edge Case Tests: Zero Loss

def test_train_epoch_zero_loss(mock_model_instance, mock_dataloader, mock_optimizer, mock_scheduler, default_config):
    """Test train_epoch when all losses are zero."""
    with patch("src.training.trainer.compute_loss") as mock_compute_loss:
        mock_compute_loss.return_value = {"total_loss": torch.tensor(0.0)}
        
        avg_loss = train_epoch(
            model_instance=mock_model_instance,
            dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            device=torch.device(default_config['training']['device']),
            gradient_clipping=default_config['training']['gradient_clipping'],
            loss_config=default_config['loss']
        )
        
        # Assertions
        assert avg_loss == 0.0, "Average loss should be 0.0 when all losses are zero"
        mock_optimizer.step.assert_called()
        mock_scheduler.step.assert_called()

# 12. Edge Case Tests: Missing Configuration Fields

def test_main_missing_configuration_fields(mock_model_instance, mock_dataloader, mock_optimizer, mock_scheduler, caplog):
    """Test that main handles missing configuration fields by using defaults."""
    config_path = "configs/missing_fields_train_config.yaml"
    
    incomplete_config = {
        "training": {
            "num_epochs": 1
            # Missing other fields
        },
        "data": {
            "data_dir": "./data"
        },
        "model": {
            "model_type": "detr",
            "model_name": "facebook/detr-resnet-50",
            "num_classes": 91
        },
        "optimizer": {
            # Missing optimizer_type, learning_rate, etc.
        },
        "loss": {
            # Missing loss weights
        }
    }
    
    with patch("src.training.trainer.ConfigParser") as mock_config_parser, \
         patch("src.training.trainer.setup_logging") as mock_setup_logging, \
         patch("src.training.trainer.DetrFeatureExtractor.from_pretrained") as mock_feature_extractor, \
         patch("src.training.trainer.ModelFactory.create_model", return_value=mock_model_instance), \
         patch("src.training.trainer.get_dataloader", return_value=mock_dataloader), \
         patch("src.training.trainer.get_optimizer_and_scheduler", return_value=(mock_optimizer, mock_scheduler)), \
         patch("src.training.trainer.train_model") as mock_train_model:
        
        # Mock ConfigParser to return incomplete_config
        mock_config_parser_instance = MagicMock()
        mock_config_parser_instance.config = incomplete_config
        mock_config_parser.return_value = mock_config_parser_instance
        
        # Mock feature extractor
        mock_feature_extractor.return_value = MagicMock()
        
        # Mock model_instance.save
        mock_model_instance.save = MagicMock()
        
        with caplog.at_level(logging.INFO):
            main(config_path)
        
        # Verify that defaults are used for missing optimizer and loss fields
        mock_get_optimizer_scheduler.assert_called_once_with(
            model=mock_model_instance.model,
            config=incomplete_config['optimizer'],
            num_training_steps=incomplete_config['training']['num_epochs'] * len(mock_dataloader)
        )
        mock_train_model.assert_called_once_with(
            model_instance=mock_model_instance,
            train_dataloader=mock_dataloader,
            val_dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            config=incomplete_config,
            device=torch.device(incomplete_config['training']['device']),
            checkpoint_dir=incomplete_config['training']['checkpoint_dir']
        )
        
        # Check logging
        assert "Training completed." in caplog.text, "Training completion log missing."

# 13. Edge Case Tests: Negative Learning Rate

def test_train_model_negative_learning_rate(mock_model_instance, mock_dataloader, mock_optimizer, mock_scheduler, default_config):
    """Test that train_model handles negative learning rate values."""
    config = default_config.copy()
    config['optimizer']['learning_rate'] = -1e-4  # Negative learning rate
    
    with patch("src.training.trainer.train_epoch", return_value=1.0), \
         patch("src.training.trainer.validate_epoch", return_value={"AP": 0.75}), \
         patch("src.training.trainer.save_checkpoint") as mock_save_checkpoint, \
         patch("src.training.trainer.Evaluator", return_value=MagicMock(spec=Evaluator)):
        
        # Assume that the optimizer does not raise an error for negative learning rates
        train_model(
            model_instance=mock_model_instance,
            train_dataloader=mock_dataloader,
            val_dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            config=config,
            device=torch.device(config['training']['device']),
            checkpoint_dir=config['training']['checkpoint_dir']
        )
        
        # Verify that optimizer was called despite negative learning rate
        mock_optimizer.step.assert_called()

# 14. Edge Case Tests: Extremely Large Number of Training Steps

def test_main_large_num_training_steps(mock_model_instance, mock_dataloader, mock_optimizer, mock_scheduler, default_config, caplog):
    """Test that main can handle an extremely large number of training steps."""
    config_path = "configs/large_steps_train_config.yaml"
    
    config = default_config.copy()
    config['training']['num_epochs'] = 1000  # Extremely large number of epochs
    
    with patch("src.training.trainer.ConfigParser") as mock_config_parser, \
         patch("src.training.trainer.setup_logging") as mock_setup_logging, \
         patch("src.training.trainer.DetrFeatureExtractor.from_pretrained") as mock_feature_extractor, \
         patch("src.training.trainer.ModelFactory.create_model", return_value=mock_model_instance), \
         patch("src.training.trainer.get_dataloader", return_value=mock_dataloader), \
         patch("src.training.trainer.get_optimizer_and_scheduler", return_value=(mock_optimizer, mock_scheduler)), \
         patch("src.training.trainer.train_model") as mock_train_model:
        
        # Mock ConfigParser to return config
        mock_config_parser_instance = MagicMock()
        mock_config_parser_instance.config = config
        mock_config_parser.return_value = mock_config_parser_instance
        
        # Mock feature extractor
        mock_feature_extractor.return_value = MagicMock()
        
        # Mock model_instance.save
        mock_model_instance.save = MagicMock()
        
        with caplog.at_level(logging.INFO):
            main(config_path)
        
        # Verify that train_model was called with correct number of training steps
        mock_train_model.assert_called_once_with(
            model_instance=mock_model_instance,
            train_dataloader=mock_dataloader,
            val_dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            config=config,
            device=torch.device(config['training']['device']),
            checkpoint_dir=config['training']['checkpoint_dir']
        )
        
        # Check logging
        assert "Training completed." in caplog.text, "Training completion log missing."

# 15. Edge Case Tests: Non-Tensor Inputs in Targets

def test_train_model_non_tensor_targets(mock_model_instance, mock_dataloader, mock_optimizer, mock_scheduler, default_config):
    """Test that train_model raises an error when targets contain non-tensor data."""
    with patch("src.training.trainer.train_epoch") as mock_train_epoch:
        mock_train_epoch.side_effect = AttributeError("Targets must be tensors")
        
        with pytest.raises(AttributeError) as exc_info:
            train_model(
                model_instance=mock_model_instance,
                train_dataloader=mock_dataloader,
                val_dataloader=mock_dataloader,
                optimizer=mock_optimizer,
                scheduler=mock_scheduler,
                config=default_config,
                device=torch.device(default_config['training']['device']),
                checkpoint_dir=default_config['training']['checkpoint_dir']
            )
        
        assert "Targets must be tensors" in str(exc_info.value), "Exception message mismatch."

# 16. Edge Case Tests: Extremely Large Batch Sizes

def test_train_model_extremely_large_batch_size(mock_model_instance, mock_optimizer, mock_scheduler, default_config, caplog):
    """Test that train_model can handle extremely large batch sizes."""
    config = default_config.copy()
    config['training']['batch_size'] = 10000  # Extremely large batch size
    
    # Mock dataloader to return large batches
    def large
