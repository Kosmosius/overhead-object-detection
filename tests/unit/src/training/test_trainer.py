# tests/unit/src/training/test_trainer.py

import torch
from unittest.mock import MagicMock, patch
from src.training.trainer import train_epoch, validate_epoch, train_model
from src.evaluation.evaluator import Evaluator
from transformers import AdamW

@pytest.fixture
def mock_dataloader():
    return MagicMock()  # Mock dataloader

@pytest.fixture
def mock_model():
    return MagicMock(spec=torch.nn.Module)  # Mock model

@pytest.fixture
def mock_optimizer():
    return MagicMock(spec=AdamW)

def test_train_epoch(mock_model, mock_dataloader, mock_optimizer):
    mock_model.return_value = {"total_loss": torch.tensor(1.0)}
    scheduler = MagicMock()
    
    loss = train_epoch(mock_model, mock_dataloader, mock_optimizer, scheduler, device="cpu")
    
    assert loss >= 0, "Loss should be a non-negative number"
    mock_optimizer.zero_grad.assert_called()
    mock_optimizer.step.assert_called()

def test_validate_epoch(mock_model, mock_dataloader):
    with patch("src.training.trainer.Evaluator.evaluate") as mock_evaluate:
        mock_evaluate.return_value = {"precision": 0.8, "recall": 0.7}
        metrics = validate_epoch(mock_model, mock_dataloader, device="cpu")
        
        assert "precision" in metrics
        assert "recall" in metrics

def test_train_model(mock_model, mock_dataloader, mock_optimizer):
    scheduler = MagicMock()
    with patch("src.training.trainer.train_epoch") as mock_train_epoch, \
         patch("src.training.trainer.validate_epoch") as mock_validate_epoch, \
         patch("src.training.trainer.save_checkpoint") as mock_save_checkpoint:
        
        mock_train_epoch.return_value = 1.0  # Training loss
        mock_validate_epoch.return_value = {"total_loss": 0.9}  # Validation metrics
        
        train_model(mock_model, mock_dataloader, mock_dataloader, mock_optimizer, scheduler, 
                    num_epochs=1, device="cpu", checkpoint_dir="checkpoints")
        
        mock_train_epoch.assert_called()
        mock_validate_epoch.assert_called()
        mock_save_checkpoint.assert_called()
