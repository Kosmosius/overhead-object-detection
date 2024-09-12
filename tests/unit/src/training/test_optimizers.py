# tests/unit/src/training/test_optimizers.py

import torch
from transformers import AdamW
from src.training.optimizers import get_optimizer, configure_optimizer, configure_scheduler

@pytest.fixture
def mock_model():
    return torch.nn.Linear(10, 2)  # Mock simple model

@pytest.fixture
def mock_config():
    return {
        "optimizer_type": "adamw",
        "learning_rate": 5e-5,
        "weight_decay": 0.01
    }

def test_get_optimizer(mock_model):
    optimizer = get_optimizer(mock_model, optimizer_type="adamw", learning_rate=5e-5)
    assert isinstance(optimizer, AdamW)

def test_configure_optimizer(mock_model, mock_config):
    optimizer = configure_optimizer(mock_model, mock_config)
    assert isinstance(optimizer, AdamW)

def test_configure_scheduler(mock_model, mock_config):
    optimizer = configure_optimizer(mock_model, mock_config)
    num_training_steps = 100
    scheduler = configure_scheduler(optimizer, num_training_steps, mock_config)
    
    assert scheduler.get_lr()[0] == mock_config["learning_rate"], "Learning rate mismatch"
