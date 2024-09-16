# tests/unit/src/training/test_optimizers.py

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
from src.training.optimizers import (
    get_optimizer,
    configure_optimizer,
    configure_scheduler,
    get_optimizer_and_scheduler
)
from transformers import SchedulerType, get_scheduler as actual_get_scheduler

# --- Fixtures ---

@pytest.fixture
def mock_model():
    """Fixture to create a mock model."""
    return nn.Linear(10, 2)  # Simple linear model for testing

@pytest.fixture
def default_config():
    """Fixture for default optimizer and scheduler configuration."""
    return {
        "optimizer_type": "adamw",
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "scheduler_type": "linear",
        "num_warmup_steps": 0
    }

@pytest.fixture
def custom_config():
    """Fixture for custom optimizer and scheduler configuration."""
    return {
        "optimizer_type": "sgd",
        "learning_rate": 0.1,
        "weight_decay": 0.001,
        "scheduler_type": "cosine",
        "num_warmup_steps": 100,
        "parameter_groups": [
            {"params": "group1", "lr": 0.05},
            {"params": "group2", "lr": 0.1, "weight_decay": 0.0001}
        ]
    }

# --- Test Cases ---

# 1. Tests for get_optimizer

@pytest.mark.parametrize(
    "optimizer_type, expected_class, kwargs, description",
    [
        ("adamw", torch.optim.AdamW, {"lr": 5e-5, "weight_decay": 0.01}, "Default AdamW"),
        ("adam", torch.optim.Adam, {"lr": 1e-3, "weight_decay": 0.001}, "Adam with custom parameters"),
        ("sgd", torch.optim.SGD, {"lr": 0.1, "weight_decay": 0.0005, "momentum": 0.9}, "SGD with momentum"),
        ("rmsprop", torch.optim.RMSprop, {"lr": 0.01, "weight_decay": 0.0001, "momentum": 0.9}, "RMSprop with momentum"),
        ("nadam", torch.optim.NAdam, {"lr": 0.002, "weight_decay": 0.0002}, "Nadam with custom parameters"),
    ]
)
def test_get_optimizer_supported_types(optimizer_type, expected_class, kwargs, description, mock_model):
    """Test that get_optimizer returns the correct optimizer instance for supported types."""
    optimizer = get_optimizer(mock_model, optimizer_type=optimizer_type, **kwargs)
    assert isinstance(optimizer, expected_class), f"{description} should be an instance of {expected_class.__name__}."

def test_get_optimizer_unsupported_type(mock_model):
    """Test that get_optimizer raises ValueError for unsupported optimizer types."""
    with pytest.raises(ValueError) as exc_info:
        get_optimizer(mock_model, optimizer_type="unsupported_optimizer")
    assert "Loss function 'unsupported_optimizer' is not supported." in str(exc_info.value), "Did not raise ValueError for unsupported optimizer type."

def test_get_optimizer_with_parameter_groups(mock_model):
    """Test that get_optimizer correctly handles parameter groups."""
    parameter_groups = [
        {"params": mock_model.parameters(), "lr": 0.01, "weight_decay": 0.001},
    ]
    optimizer = get_optimizer(
        mock_model,
        optimizer_type="adamw",
        learning_rate=5e-5,
        weight_decay=0.01,
        parameter_groups=parameter_groups
    )
    assert optimizer.param_groups[0]['lr'] == 0.01, "Parameter group learning rate mismatch."
    assert optimizer.param_groups[0]['weight_decay'] == 0.001, "Parameter group weight decay mismatch."

# 2. Tests for configure_optimizer

def test_configure_optimizer_default(default_config, mock_model):
    """Test that configure_optimizer returns the correct optimizer based on default configuration."""
    optimizer = configure_optimizer(mock_model, default_config)
    assert isinstance(optimizer, torch.optim.AdamW), "Default optimizer should be AdamW."
    assert optimizer.param_groups[0]['lr'] == default_config["learning_rate"], "Learning rate mismatch."
    assert optimizer.param_groups[0]['weight_decay'] == default_config["weight_decay"], "Weight decay mismatch."

def test_configure_optimizer_custom(custom_config, mock_model):
    """Test that configure_optimizer returns the correct optimizer based on custom configuration."""
    optimizer = configure_optimizer(mock_model, custom_config)
    assert isinstance(optimizer, torch.optim.SGD), "Custom optimizer should be SGD."
    assert optimizer.param_groups[0]['lr'] == 0.05, "Parameter group 1 learning rate mismatch."
    assert optimizer.param_groups[1]['lr'] == 0.1, "Parameter group 2 learning rate mismatch."
    assert optimizer.param_groups[1]['weight_decay'] == 0.0001, "Parameter group 2 weight decay mismatch."

def test_configure_optimizer_missing_fields(default_config, mock_model):
    """Test that configure_optimizer uses default values when some config fields are missing."""
    partial_config = {
        "optimizer_type": "adam",
        # Missing learning_rate and weight_decay
    }
    optimizer = configure_optimizer(mock_model, partial_config)
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer should be Adam."
    assert optimizer.param_groups[0]['lr'] == 5e-5, "Default learning rate mismatch."
    assert optimizer.param_groups[0]['weight_decay'] == 0.01, "Default weight decay mismatch."

def test_configure_optimizer_invalid_type(mock_model):
    """Test that configure_optimizer raises ValueError for invalid optimizer types in config."""
    invalid_config = {
        "optimizer_type": "invalid_optimizer",
        "learning_rate": 1e-3,
        "weight_decay": 0.01
    }
    with pytest.raises(ValueError) as exc_info:
        configure_optimizer(mock_model, invalid_config)
    assert "Loss function 'invalid_optimizer' is not supported." in str(exc_info.value), "Did not raise ValueError for invalid optimizer type."

# 3. Tests for configure_scheduler

@pytest.mark.parametrize(
    "scheduler_type, expected_scheduler_type, kwargs, description",
    [
        ("linear", SchedulerType.LINEAR, {"num_warmup_steps": 0, "num_training_steps": 1000}, "Linear scheduler with default parameters"),
        ("cosine", SchedulerType.COSINE, {"num_warmup_steps": 100, "num_training_steps": 1000}, "Cosine scheduler with custom parameters"),
        ("cosine_with_restarts", SchedulerType.COSINE_WITH_RESTARTS, {"num_warmup_steps": 100, "num_training_steps": 1000, "num_cycles": 0.5}, "Cosine with restarts scheduler"),
        ("polynomial", SchedulerType.POLYNOMIAL, {"num_warmup_steps": 100, "num_training_steps": 1000, "power": 1.0}, "Polynomial scheduler"),
    ]
)
def test_configure_scheduler_supported_types(scheduler_type, expected_scheduler_type, kwargs, description, mock_model, default_config):
    """Test that configure_scheduler returns the correct scheduler instance for supported types."""
    optimizer = configure_optimizer(mock_model, default_config)
    with patch('src.training.optimizers.get_scheduler') as mock_get_scheduler:
        mock_scheduler = MagicMock()
        mock_get_scheduler.return_value = mock_scheduler
        scheduler = configure_scheduler(optimizer, num_training_steps=kwargs["num_training_steps"], config={"scheduler_type": scheduler_type, **kwargs})
        mock_get_scheduler.assert_called_once_with(
            name=scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=kwargs["num_warmup_steps"],
            num_training_steps=kwargs["num_training_steps"],
            **({k: v for k, v in kwargs.items() if k not in ["num_warmup_steps", "num_training_steps"]})
        )
        assert scheduler == mock_scheduler, f"{description} should return the mocked scheduler instance."

def test_configure_scheduler_unsupported_type(mock_model, default_config):
    """Test that configure_scheduler raises ValueError for unsupported scheduler types."""
    optimizer = configure_optimizer(mock_model, default_config)
    invalid_config = {
        "scheduler_type": "unsupported_scheduler",
        "num_warmup_steps": 100,
        "num_training_steps": 1000
    }
    with pytest.raises(ValueError) as exc_info:
        configure_scheduler(optimizer, num_training_steps=1000, config=invalid_config)
    assert "Unsupported scheduler type" in str(exc_info.value), "Did not raise ValueError for unsupported scheduler type."

def test_configure_scheduler_missing_fields(default_config, mock_model):
    """Test that configure_scheduler uses default values when some config fields are missing."""
    optimizer = configure_optimizer(mock_model, default_config)
    config = {
        "scheduler_type": "linear",
        # Missing num_warmup_steps
        # Missing num_training_steps
    }
    with patch('src.training.optimizers.get_scheduler') as mock_get_scheduler:
        mock_scheduler = MagicMock()
        mock_get_scheduler.return_value = mock_scheduler
        scheduler = configure_scheduler(optimizer, num_training_steps=1000, config=config)
        mock_get_scheduler.assert_called_once_with(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=1000
        )
        assert scheduler == mock_scheduler, "Scheduler should return the mocked scheduler instance."

# 4. Tests for get_optimizer_and_scheduler

def test_get_optimizer_and_scheduler_default(default_config, mock_model):
    """Test that get_optimizer_and_scheduler returns correct optimizer and scheduler based on default configuration."""
    with patch('src.training.optimizers.get_scheduler') as mock_get_scheduler:
        mock_scheduler = MagicMock()
        mock_get_scheduler.return_value = mock_scheduler
        
        optimizer, scheduler = get_optimizer_and_scheduler(mock_model, default_config, num_training_steps=1000)
        
        # Check optimizer
        assert isinstance(optimizer, torch.optim.AdamW), "Optimizer should be AdamW."
        assert optimizer.param_groups[0]['lr'] == default_config["learning_rate"], "Learning rate mismatch."
        assert optimizer.param_groups[0]['weight_decay'] == default_config["weight_decay"], "Weight decay mismatch."
        
        # Check scheduler
        mock_get_scheduler.assert_called_once_with(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=1000
        )
        assert scheduler == mock_scheduler, "Scheduler should be the mocked scheduler instance."

def test_get_optimizer_and_scheduler_custom(custom_config, mock_model):
    """Test that get_optimizer_and_scheduler returns correct optimizer and scheduler based on custom configuration."""
    with patch('src.training.optimizers.get_scheduler') as mock_get_scheduler:
        mock_scheduler = MagicMock()
        mock_get_scheduler.return_value = mock_scheduler
        
        optimizer, scheduler = get_optimizer_and_scheduler(mock_model, custom_config, num_training_steps=2000)
        
        # Check optimizer
        assert isinstance(optimizer, torch.optim.SGD), "Optimizer should be SGD."
        # Since parameter_groups are provided, check param_groups
        assert len(optimizer.param_groups) == 2, "There should be two parameter groups."
        assert optimizer.param_groups[0]['lr'] == 0.05, "Parameter group 1 learning rate mismatch."
        assert optimizer.param_groups[1]['lr'] == 0.1, "Parameter group 2 learning rate mismatch."
        assert optimizer.param_groups[1]['weight_decay'] == 0.0001, "Parameter group 2 weight decay mismatch."
        
        # Check scheduler
        mock_get_scheduler.assert_called_once_with(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=2000
        )
        assert scheduler == mock_scheduler, "Scheduler should be the mocked scheduler instance."

def test_get_optimizer_and_scheduler_invalid_optimizer(custom_config, mock_model):
    """Test that get_optimizer_and_scheduler raises ValueError for invalid optimizer types in configuration."""
    invalid_config = custom_config.copy()
    invalid_config["optimizer_type"] = "invalid_optimizer"
    
    with pytest.raises(ValueError) as exc_info:
        get_optimizer_and_scheduler(mock_model, invalid_config, num_training_steps=2000)
    assert "Loss function 'invalid_optimizer' is not supported." in str(exc_info.value), "Did not raise ValueError for invalid optimizer type."

def test_get_optimizer_and_scheduler_invalid_scheduler(custom_config, mock_model):
    """Test that get_optimizer_and_scheduler raises ValueError for invalid scheduler types in configuration."""
    invalid_config = custom_config.copy()
    invalid_config["scheduler_type"] = "invalid_scheduler"
    
    with patch('src.training.optimizers.get_scheduler') as mock_get_scheduler:
        mock_get_scheduler.side_effect = ValueError("Unsupported scheduler type 'invalid_scheduler'.")
        with pytest.raises(ValueError) as exc_info:
            get_optimizer_and_scheduler(mock_model, invalid_config, num_training_steps=2000)
        assert "Unsupported scheduler type 'invalid_scheduler'." in str(exc_info.value), "Did not raise ValueError for invalid scheduler type."

# 5. Edge Case Tests

def test_get_optimizer_extreme_learning_rate(mock_model):
    """Test get_optimizer with an extremely high learning rate."""
    optimizer = get_optimizer(mock_model, optimizer_type="adamw", learning_rate=1.0, weight_decay=0.01)
    assert optimizer.param_groups[0]['lr'] == 1.0, "Extreme learning rate mismatch."

def test_get_optimizer_zero_weight_decay(mock_model):
    """Test get_optimizer with zero weight decay."""
    optimizer = get_optimizer(mock_model, optimizer_type="adamw", learning_rate=5e-5, weight_decay=0.0)
    assert optimizer.param_groups[0]['weight_decay'] == 0.0, "Weight decay should be zero."

def test_get_optimizer_parameter_groups_empty(mock_model):
    """Test get_optimizer with empty parameter_groups."""
    parameter_groups = []
    with pytest.raises(ValueError):
        get_optimizer(mock_model, optimizer_type="adamw", parameter_groups=parameter_groups)

def test_get_optimizer_negative_learning_rate(mock_model):
    """Test get_optimizer with a negative learning rate."""
    optimizer = get_optimizer(mock_model, optimizer_type="adamw", learning_rate=-1e-5, weight_decay=0.01)
    assert optimizer.param_groups[0]['lr'] == -1e-5, "Negative learning rate mismatch."

def test_configure_scheduler_large_num_training_steps(default_config, mock_model):
    """Test configure_scheduler with a very large number of training steps."""
    optimizer = configure_optimizer(mock_model, default_config)
    config = {
        "scheduler_type": "linear",
        "num_warmup_steps": 100,
        "num_training_steps": 100000
    }
    with patch('src.training.optimizers.get_scheduler') as mock_get_scheduler:
        mock_scheduler = MagicMock()
        mock_get_scheduler.return_value = mock_scheduler
        scheduler = configure_scheduler(optimizer, num_training_steps=100000, config=config)
        mock_get_scheduler.assert_called_once_with(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=100000
        )
        assert scheduler == mock_scheduler, "Scheduler should be the mocked scheduler instance."

# 6. Parameter Groups with Different Optimizer Parameters

def test_get_optimizer_parameter_groups_different_params(mock_model):
    """Test get_optimizer with parameter groups having different learning rates and weight decays."""
    parameter_groups = [
        {"params": mock_model.parameters(), "lr": 0.01, "weight_decay": 0.001},
        {"params": mock_model.parameters(), "lr": 0.02, "weight_decay": 0.0001}
    ]
    optimizer = get_optimizer(
        mock_model,
        optimizer_type="adamw",
        learning_rate=5e-5,  # Should be overridden by parameter groups
        weight_decay=0.01,    # Should be overridden by parameter groups
        parameter_groups=parameter_groups
    )
    assert len(optimizer.param_groups) == 2, "There should be two parameter groups."
    assert optimizer.param_groups[0]['lr'] == 0.01, "First parameter group learning rate mismatch."
    assert optimizer.param_groups[0]['weight_decay'] == 0.001, "First parameter group weight decay mismatch."
    assert optimizer.param_groups[1]['lr'] == 0.02, "Second parameter group learning rate mismatch."
    assert optimizer.param_groups[1]['weight_decay'] == 0.0001, "Second parameter group weight decay mismatch."

# 7. Mocking External Dependencies

def test_configure_scheduler_mocked_scheduler(default_config, mock_model):
    """Test configure_scheduler with a mocked scheduler."""
    optimizer = configure_optimizer(mock_model, default_config)
    with patch('src.training.optimizers.get_scheduler') as mock_get_scheduler:
        mock_scheduler = MagicMock()
        mock_get_scheduler.return_value = mock_scheduler
        scheduler = configure_scheduler(optimizer, num_training_steps=1000, config=default_config)
        mock_get_scheduler.assert_called_once_with(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=1000
        )
        assert scheduler == mock_scheduler, "Scheduler should be the mocked scheduler instance."

# 8. Comprehensive Integration Test

def test_get_optimizer_and_scheduler_integration(default_config, mock_model):
    """Integration test for get_optimizer_and_scheduler with default configuration."""
    with patch('src.training.optimizers.get_scheduler') as mock_get_scheduler:
        mock_scheduler = MagicMock()
        mock_get_scheduler.return_value = mock_scheduler
        
        optimizer, scheduler = get_optimizer_and_scheduler(mock_model, default_config, num_training_steps=1000)
        
        # Verify optimizer
        assert isinstance(optimizer, torch.optim.AdamW), "Optimizer should be AdamW."
        assert optimizer.param_groups[0]['lr'] == default_config["learning_rate"], "Learning rate mismatch."
        assert optimizer.param_groups[0]['weight_decay'] == default_config["weight_decay"], "Weight decay mismatch."
        
        # Verify scheduler
        mock_get_scheduler.assert_called_once_with(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=1000
        )
        assert scheduler == mock_scheduler, "Scheduler should be the mocked scheduler instance."

# 9. Edge Case: Empty Parameter Groups

def test_get_optimizer_empty_parameter_groups(mock_model):
    """Test get_optimizer with empty parameter_groups list."""
    parameter_groups = []
    with pytest.raises(ValueError) as exc_info:
        get_optimizer(
            mock_model,
            optimizer_type="adamw",
            learning_rate=5e-5,
            weight_decay=0.01,
            parameter_groups=parameter_groups
        )
    assert "optimizer_type" in str(exc_info.value), "Unexpected error message for empty parameter_groups."

# 10. Edge Case: Missing Optimizer Parameters

def test_get_optimizer_missing_parameters(mock_model):
    """Test get_optimizer when essential parameters are missing."""
    # Missing learning_rate and weight_decay should default to provided values
    optimizer = get_optimizer(
        mock_model,
        optimizer_type="adamw"
    )
    assert isinstance(optimizer, torch.optim.AdamW), "Optimizer should be AdamW."
    # Check if defaults are used
    assert optimizer.param_groups[0]['lr'] == 5e-5, "Default learning rate mismatch."
    assert optimizer.param_groups[0]['weight_decay'] == 0.01, "Default weight decay mismatch."

# 11. Edge Case: Invalid Parameter Group Structure

def test_get_optimizer_invalid_parameter_group_structure(mock_model):
    """Test get_optimizer with incorrectly structured parameter_groups."""
    # parameter_groups should be a list of dicts with 'params' key
    parameter_groups = [
        {"lr": 0.01, "weight_decay": 0.001},  # Missing 'params' key
    ]
    with pytest.raises(KeyError):
        get_optimizer(
            mock_model,
            optimizer_type="adamw",
            learning_rate=5e-5,
            weight_decay=0.01,
            parameter_groups=parameter_groups
        )

# 12. Edge Case: Large Number of Parameter Groups

def test_get_optimizer_large_number_of_parameter_groups(mock_model):
    """Test get_optimizer with a large number of parameter groups."""
    parameter_groups = [{"params": mock_model.parameters(), "lr": 0.001} for _ in range(100)]
    optimizer = get_optimizer(
        mock_model,
        optimizer_type="adamw",
        learning_rate=5e-5,
        weight_decay=0.01,
        parameter_groups=parameter_groups
    )
    assert len(optimizer.param_groups) == 100, "Number of parameter groups mismatch."
    for pg in optimizer.param_groups:
        assert pg['lr'] == 0.001, "Parameter group learning rate mismatch."
        assert pg['weight_decay'] == 0.01, "Parameter group weight decay mismatch."

# 13. Edge Case: Very High Weight Decay

def test_get_optimizer_very_high_weight_decay(mock_model):
    """Test get_optimizer with a very high weight decay."""
    optimizer = get_optimizer(mock_model, optimizer_type="adamw", learning_rate=5e-5, weight_decay=10.0)
    assert optimizer.param_groups[0]['weight_decay'] == 10.0, "Weight decay should be set to 10.0."

# 14. Edge Case: Learning Rate Zero

def test_get_optimizer_learning_rate_zero(mock_model):
    """Test get_optimizer with a learning rate of zero."""
    optimizer = get_optimizer(mock_model, optimizer_type="adamw", learning_rate=0.0, weight_decay=0.01)
    assert optimizer.param_groups[0]['lr'] == 0.0, "Learning rate should be zero."

# 15. Scheduler Configuration with Additional Arguments

def test_configure_scheduler_with_additional_kwargs(default_config, mock_model):
    """Test configure_scheduler with additional scheduler-specific keyword arguments."""
    config = {
        "scheduler_type": "linear",
        "num_warmup_steps": 100,
        "num_training_steps": 1000,
        "additional_arg": "value"  # Non-standard argument
    }
    optimizer = configure_optimizer(mock_model, default_config)
    with patch('src.training.optimizers.get_scheduler') as mock_get_scheduler:
        mock_scheduler = MagicMock()
        mock_get_scheduler.return_value = mock_scheduler
        scheduler = configure_scheduler(optimizer, num_training_steps=1000, config=config)
        # 'additional_arg' should be ignored or passed depending on implementation
        mock_get_scheduler.assert_called_once_with(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=1000,
            additional_arg="value"
        )
        assert scheduler == mock_scheduler, "Scheduler should be the mocked scheduler instance."

# 16. Parameter Groups with Mixed Optimizer Parameters

def test_get_optimizer_parameter_groups_mixed_params(mock_model):
    """Test get_optimizer with parameter groups having mixed optimizer parameters."""
    parameter_groups = [
        {"params": mock_model.parameters(), "lr": 0.01, "weight_decay": 0.001},
        {"params": mock_model.parameters(), "lr": 0.02},  # Missing weight_decay, should use default
    ]
    optimizer = get_optimizer(
        mock_model,
        optimizer_type="adamw",
        learning_rate=5e-5,  # Default learning rate
        weight_decay=0.01,    # Default weight decay
        parameter_groups=parameter_groups
    )
    assert len(optimizer.param_groups) == 2, "There should be two parameter groups."
    assert optimizer.param_groups[0]['lr'] == 0.01, "First parameter group learning rate mismatch."
    assert optimizer.param_groups[0]['weight_decay'] == 0.001, "First parameter group weight decay mismatch."
    assert optimizer.param_groups[1]['lr'] == 0.02, "Second parameter group learning rate mismatch."
    assert optimizer.param_groups[1]['weight_decay'] == 0.01, "Second parameter group weight decay should use default."

# 17. Scheduler Configuration Missing Required Fields

def test_configure_scheduler_missing_required_fields(default_config, mock_model):
    """Test configure_scheduler when required scheduler fields are missing."""
    optimizer = configure_optimizer(mock_model, default_config)
    config = {
        "scheduler_type": "linear",
        # Missing num_training_steps
    }
    with pytest.raises(TypeError):
        configure_scheduler(optimizer, num_training_steps=None, config=config)

# 18. Test get_optimizer_and_scheduler with Scheduler Return None

def test_get_optimizer_and_scheduler_scheduler_none(default_config, mock_model):
    """Test get_optimizer_and_scheduler when scheduler is None."""
    with patch('src.training.optimizers.get_scheduler') as mock_get_scheduler:
        mock_get_scheduler.return_value = None
        optimizer, scheduler = get_optimizer_and_scheduler(mock_model, default_config, num_training_steps=1000)
        mock_get_scheduler.assert_called_once_with(
            name="linear",
            optimizer=configure_optimizer(mock_model, default_config),
            num_warmup_steps=0,
            num_training_steps=1000
        )
        assert scheduler is None, "Scheduler should be None as mocked."

# 19. Test get_optimizer_and_scheduler with Scheduler Type Dependent on Optimizer

def test_get_optimizer_and_scheduler_scheduler_dependent_on_optimizer(custom_config, mock_model):
    """Test get_optimizer_and_scheduler where scheduler type depends on optimizer type."""
    # For example, using a cosine scheduler with SGD optimizer
    with patch('src.training.optimizers.get_scheduler') as mock_get_scheduler:
        mock_scheduler = MagicMock()
        mock_get_scheduler.return_value = mock_scheduler
        
        optimizer, scheduler = get_optimizer_and_scheduler(mock_model, custom_config, num_training_steps=2000)
        
        assert isinstance(optimizer, torch.optim.SGD), "Optimizer should be SGD."
        mock_get_scheduler.assert_called_once_with(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=2000
        )
        assert scheduler == mock_scheduler, "Scheduler should be the mocked scheduler instance."

# 20. Test get_optimizer_and_scheduler with Parameter Groups

def test_get_optimizer_and_scheduler_with_parameter_groups(custom_config, mock_model):
    """Test get_optimizer_and_scheduler with parameter groups in configuration."""
    with patch('src.training.optimizers.get_scheduler') as mock_get_scheduler:
        mock_scheduler = MagicMock()
        mock_get_scheduler.return_value = mock_scheduler
        
        optimizer, scheduler = get_optimizer_and_scheduler(mock_model, custom_config, num_training_steps=2000)
        
        assert isinstance(optimizer, torch.optim.SGD), "Optimizer should be SGD."
        assert len(optimizer.param_groups) == 2, "There should be two parameter groups."
        assert optimizer.param_groups[0]['lr'] == 0.05, "First parameter group learning rate mismatch."
        assert optimizer.param_groups[1]['lr'] == 0.1, "Second parameter group learning rate mismatch."
        mock_get_scheduler.assert_called_once_with(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=2000
        )
        assert scheduler == mock_scheduler, "Scheduler should be the mocked scheduler instance."

