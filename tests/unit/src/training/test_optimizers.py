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
        "lr": 5e-5,
        "weight_decay": 0.01,
        "scheduler_type": "linear",
        "num_warmup_steps": 0
        # 'parameter_groups' is intentionally omitted to test its optionality
    }

@pytest.fixture
def custom_config():
    """Fixture for custom optimizer and scheduler configuration."""
    return {
        "optimizer_type": "sgd",
        "lr": 0.1,
        "weight_decay": 0.001,
        "scheduler_type": "cosine",
        "num_warmup_steps": 100,
        "parameter_groups": [
            {"params": "group1", "lr": 0.05},  # Will be updated in tests to use actual parameters
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
    assert "Unsupported optimizer type 'unsupported_optimizer'" in str(exc_info.value), "Did not raise ValueError for unsupported optimizer type."

def test_get_optimizer_with_parameter_groups(mock_model):
    """Test that get_optimizer correctly handles parameter groups."""
    parameter_groups = [
        {"params": list(mock_model.parameters()), "lr": 0.01, "weight_decay": 0.001},
    ]
    optimizer = get_optimizer(
        mock_model,
        optimizer_type="adamw",
        lr=5e-5,
        weight_decay=0.01,
        parameter_groups=parameter_groups
    )
    assert optimizer.param_groups[0]['lr'] == 0.01, "Parameter group learning rate mismatch."
    assert optimizer.param_groups[0]['weight_decay'] == 0.001, "Parameter group weight decay mismatch."

def test_get_optimizer_parameter_groups_empty(mock_model):
    """Test get_optimizer with empty parameter_groups list."""
    with pytest.raises(ValueError) as exc_info:
        get_optimizer(
            mock_model,
            optimizer_type="adamw",
            lr=5e-5,
            weight_decay=0.01,
            parameter_groups=[]
        )
    assert "parameter_groups cannot be an empty list." in str(exc_info.value), "Unexpected error message for empty parameter_groups."

# 2. Tests for configure_optimizer

def test_configure_optimizer_default(default_config, mock_model):
    """Test that configure_optimizer returns the correct optimizer based on default configuration."""
    optimizer = configure_optimizer(mock_model, default_config)
    assert isinstance(optimizer, torch.optim.AdamW), "Default optimizer should be AdamW."
    assert optimizer.param_groups[0]['lr'] == default_config["lr"], "Learning rate mismatch."
    assert optimizer.param_groups[0]['weight_decay'] == default_config["weight_decay"], "Weight decay mismatch."

def test_configure_optimizer_custom(custom_config, mock_model):
    """Test that configure_optimizer returns the correct optimizer based on custom configuration."""
    # Update parameter_groups to use actual tensor parameters and include weight_decay
    param1 = mock_model.weight
    param2 = mock_model.bias
    custom_config["parameter_groups"] = [
        {"params": [param1], "lr": 0.05, "weight_decay": 0.001},
        {"params": [param2], "lr": 0.1, "weight_decay": 0.0001}
    ]

    optimizer = configure_optimizer(mock_model, custom_config)
    assert isinstance(optimizer, torch.optim.SGD), "Optimizer should be SGD."
    assert len(optimizer.param_groups) == 2, "There should be two parameter groups."
    assert optimizer.param_groups[0]['lr'] == 0.05, "Parameter group 1 learning rate mismatch."
    assert optimizer.param_groups[0]['weight_decay'] == 0.001, "Parameter group 1 weight decay mismatch."
    assert optimizer.param_groups[1]['lr'] == 0.1, "Parameter group 2 learning rate mismatch."
    assert optimizer.param_groups[1]['weight_decay'] == 0.0001, "Parameter group 2 weight decay mismatch."

def test_configure_optimizer_missing_fields(default_config, mock_model):
    """Test that configure_optimizer uses default values when some config fields are missing."""
    partial_config = {
        "optimizer_type": "adam",
        # Missing 'lr' and 'weight_decay'
    }
    optimizer = configure_optimizer(mock_model, partial_config)
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer should be Adam."
    assert optimizer.param_groups[0]['lr'] == 5e-5, "Default learning rate mismatch."
    assert optimizer.param_groups[0]['weight_decay'] == 0.01, "Default weight decay mismatch."

def test_configure_optimizer_invalid_type(mock_model):
    """Test that configure_optimizer raises ValueError for invalid optimizer types in config."""
    invalid_config = {
        "optimizer_type": "invalid_optimizer",
        "lr": 1e-3,
        "weight_decay": 0.01
    }
    with pytest.raises(ValueError) as exc_info:
        configure_optimizer(mock_model, invalid_config)
    assert "Unsupported optimizer type 'invalid_optimizer'" in str(exc_info.value), "Did not raise ValueError for invalid optimizer type."

# 3. Tests for configure_scheduler

@pytest.mark.parametrize(
    "scheduler_type, scheduler_specific_kwargs, description",
    [
        (
            "linear",
            {"num_warmup_steps": 0, "num_training_steps": 1000},
            "Linear scheduler with default parameters"
        ),
        (
            "cosine",
            {"num_warmup_steps": 100, "num_training_steps": 1000},
            "Cosine scheduler with custom parameters"
        ),
        (
            "cosine_with_restarts",
            {"num_warmup_steps": 100, "num_training_steps": 1000, "num_cycles": 0.5},
            "Cosine with restarts scheduler"
        ),
        (
            "polynomial",
            {"num_warmup_steps": 100, "num_training_steps": 1000, "power": 1.0},
            "Polynomial scheduler"
        ),
    ]
)
def test_configure_scheduler_supported_types(scheduler_type, scheduler_specific_kwargs, description):
    """
    Test configure_scheduler with various supported scheduler types and additional arguments.

    Args:
        scheduler_type (str): The type of scheduler to test.
        scheduler_specific_kwargs (dict): Additional scheduler-specific keyword arguments.
        description (str): Description of the scheduler type.
    """
    # Initialize optimizer
    optimizer = torch.optim.AdamW(nn.Linear(10, 2).parameters(), lr=5e-5, weight_decay=0.01)

    # Create config dictionary with scheduler type and specific kwargs
    config = {
        "scheduler_type": scheduler_type,
        "num_warmup_steps": scheduler_specific_kwargs.get("num_warmup_steps", 0),
        "num_training_steps": scheduler_specific_kwargs.get("num_training_steps", 1000),
    }
    # Add additional scheduler-specific arguments, excluding 'num_training_steps'
    config.update({k: v for k, v in scheduler_specific_kwargs.items() if k not in ["num_warmup_steps", "num_training_steps"]})

    with patch('src.training.optimizers.get_scheduler') as mock_get_scheduler:
        mock_scheduler = MagicMock()
        mock_get_scheduler.return_value = mock_scheduler

        # Configure scheduler
        scheduler = configure_scheduler(optimizer, config["num_training_steps"], config)

        # Prepare expected call arguments
        expected_call_args = {
            "name": scheduler_type,
            "optimizer": optimizer,
            "num_warmup_steps": config["num_warmup_steps"],
            "num_training_steps": config["num_training_steps"],
        }
        expected_call_args.update({k: v for k, v in scheduler_specific_kwargs.items() if k not in ["num_warmup_steps", "num_training_steps"]})

        # Assert that get_scheduler was called with the expected arguments
        mock_get_scheduler.assert_called_once_with(**expected_call_args)

        # Assert that the returned scheduler is the mocked instance
        assert scheduler == mock_scheduler, f"Scheduler should be the mocked scheduler instance for {description}."

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
    assert "Unsupported scheduler type 'unsupported_scheduler'." in str(exc_info.value), "Did not raise ValueError for unsupported scheduler type."

def test_configure_scheduler_missing_fields(default_config, mock_model):
    """Test that configure_scheduler raises ValueError when required scheduler fields are missing."""
    optimizer = configure_optimizer(mock_model, default_config)
    config = {
        "scheduler_type": "linear",
        # Missing 'num_warmup_steps' and 'num_training_steps'
    }
    with pytest.raises(ValueError) as exc_info:
        configure_scheduler(optimizer, num_training_steps=None, config=config)
    assert "linear requires `num_training_steps`" in str(exc_info.value), "Did not raise ValueError for missing num_training_steps."

# 4. Tests for get_optimizer_and_scheduler

def test_get_optimizer_and_scheduler_default(default_config, mock_model):
    """Test that get_optimizer_and_scheduler returns correct optimizer and scheduler based on default configuration."""
    with patch('src.training.optimizers.get_scheduler') as mock_get_scheduler:
        mock_scheduler = MagicMock()
        mock_get_scheduler.return_value = mock_scheduler

        optimizer, scheduler = get_optimizer_and_scheduler(mock_model, default_config, num_training_steps=1000)

        # Check optimizer
        assert isinstance(optimizer, torch.optim.AdamW), "Optimizer should be AdamW."
        assert optimizer.param_groups[0]['lr'] == default_config["lr"], "Learning rate mismatch."
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
    # Update parameter_groups to use actual tensor parameters and include weight_decay
    param1 = mock_model.weight
    param2 = mock_model.bias
    custom_config["parameter_groups"] = [
        {"params": [param1], "lr": 0.05, "weight_decay": 0.001},
        {"params": [param2], "lr": 0.1, "weight_decay": 0.0001}
    ]

    with patch('src.training.optimizers.get_scheduler') as mock_get_scheduler:
        mock_scheduler = MagicMock()
        mock_get_scheduler.return_value = mock_scheduler

        optimizer, scheduler = get_optimizer_and_scheduler(mock_model, custom_config, num_training_steps=2000)

        # Check optimizer
        assert isinstance(optimizer, torch.optim.SGD), "Optimizer should be SGD."
        assert len(optimizer.param_groups) == 2, "There should be two parameter groups."
        assert optimizer.param_groups[0]['lr'] == 0.05, "Parameter group 1 learning rate mismatch."
        assert optimizer.param_groups[0]['weight_decay'] == 0.001, "Parameter group 1 weight decay mismatch."
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
    assert "Unsupported optimizer type 'invalid_optimizer'" in str(exc_info.value), "Did not raise ValueError for invalid optimizer type."

def test_get_optimizer_and_scheduler_invalid_scheduler(custom_config, mock_model):
    """Test that get_optimizer_and_scheduler raises ValueError for invalid scheduler types in configuration."""
    # Update parameter_groups to use actual tensor parameters and include weight_decay
    param1 = mock_model.weight
    param2 = mock_model.bias
    custom_config["parameter_groups"] = [
        {"params": [param1], "lr": 0.05, "weight_decay": 0.001},
        {"params": [param2], "lr": 0.1, "weight_decay": 0.0001}
    ]
    # Set an invalid scheduler type
    custom_config["scheduler_type"] = "invalid_scheduler"

    with pytest.raises(ValueError) as exc_info:
        get_optimizer_and_scheduler(mock_model, custom_config, num_training_steps=2000)
    assert "Unsupported scheduler type 'invalid_scheduler'." in str(exc_info.value), "Did not raise ValueError for invalid scheduler type."

# 5. Edge Case Tests

def test_get_optimizer_extreme_learning_rate(mock_model):
    """Test get_optimizer with an extremely high learning rate."""
    optimizer = get_optimizer(mock_model, optimizer_type="adamw", lr=1.0, weight_decay=0.01)
    assert optimizer.param_groups[0]['lr'] == 1.0, "Extreme learning rate mismatch."

def test_get_optimizer_zero_weight_decay(mock_model):
    """Test get_optimizer with zero weight decay."""
    optimizer = get_optimizer(mock_model, optimizer_type="adamw", lr=5e-5, weight_decay=0.0)
    assert optimizer.param_groups[0]['weight_decay'] == 0.0, "Weight decay should be zero."

def test_get_optimizer_negative_learning_rate(mock_model):
    """Test get_optimizer with a negative learning rate."""
    with pytest.raises(ValueError) as exc_info:
        get_optimizer(
            mock_model,
            optimizer_type="adamw",
            lr=-1e-5,
            weight_decay=0.01
        )
    assert "Invalid learning rate" in str(exc_info.value), "Did not raise ValueError for negative learning rate."

def test_get_optimizer_learning_rate_zero(mock_model):
    """Test get_optimizer with a learning rate of zero."""
    with pytest.raises(ValueError) as exc_info:
        get_optimizer(
            mock_model,
            optimizer_type="adamw",
            lr=0.0,
            weight_decay=0.01
        )
    assert "Invalid learning rate" in str(exc_info.value), "Did not raise ValueError for zero learning rate."

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
    """Test get_optimizer with parameter groups having overlapping parameters."""
    param1 = mock_model.weight
    param2 = mock_model.bias
    # Intentional overlap: param1 is in both groups
    parameter_groups = [
        {"params": [param1], "lr": 0.01, "weight_decay": 0.001},
        {"params": [param1, param2], "lr": 0.02, "weight_decay": 0.0001}
    ]
    with pytest.raises(ValueError) as exc_info:
        get_optimizer(
            mock_model,
            optimizer_type="adamw",
            lr=5e-5,
            weight_decay=0.01,
            parameter_groups=parameter_groups
        )
    assert "Some parameters appear in more than one parameter group" in str(exc_info.value), "Did not raise ValueError for overlapping parameters."

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
        assert optimizer.param_groups[0]['lr'] == default_config["lr"], "Learning rate mismatch."
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
    with pytest.raises(ValueError) as exc_info:
        get_optimizer(
            mock_model,
            optimizer_type="adamw",
            lr=5e-5,
            weight_decay=0.01,
            parameter_groups=[]
        )
    assert "parameter_groups cannot be an empty list." in str(exc_info.value), "Unexpected error message for empty parameter_groups."

# 10. Edge Case: Missing Optimizer Parameters

def test_get_optimizer_missing_parameters(mock_model):
    """Test get_optimizer when essential parameters are missing."""
    # Missing 'lr' and 'weight_decay' should default to provided values
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
    with pytest.raises(ValueError) as exc_info:
        get_optimizer(
            mock_model,
            optimizer_type="adamw",
            lr=5e-5,
            weight_decay=0.01,
            parameter_groups=parameter_groups
        )
    assert "Each parameter group must have a 'params' list." in str(exc_info.value), "Did not raise ValueError for missing 'params' in parameter group."

# 12. Edge Case: Large Number of Parameter Groups

def test_get_optimizer_large_number_of_parameter_groups(mock_model):
    """Test get_optimizer with a large number of parameter groups."""
    param1 = mock_model.weight
    param2 = mock_model.bias
    # Intentional overlap: param1 is in multiple groups
    parameter_groups = [{"params": [param1], "lr": 0.001} for _ in range(50)] + [{"params": [param1, param2], "lr": 0.002} for _ in range(50)]
    with pytest.raises(ValueError) as exc_info:
        get_optimizer(
            mock_model,
            optimizer_type="adamw",
            lr=5e-5,
            weight_decay=0.01,
            parameter_groups=parameter_groups
        )
    assert "Some parameters appear in more than one parameter group" in str(exc_info.value), "Did not raise ValueError for overlapping parameters."

# 13. Edge Case: Very High Weight Decay

def test_get_optimizer_very_high_weight_decay(mock_model):
    """Test get_optimizer with a very high weight decay."""
    optimizer = get_optimizer(mock_model, optimizer_type="adamw", lr=5e-5, weight_decay=10.0)
    assert optimizer.param_groups[0]['weight_decay'] == 10.0, "Weight decay should be set to 10.0."

# 14. Edge Case: Learning Rate Zero

def test_get_optimizer_learning_rate_zero(mock_model):
    """Test get_optimizer with a learning rate of zero."""
    with pytest.raises(ValueError) as exc_info:
        get_optimizer(
            mock_model,
            optimizer_type="adamw",
            lr=0.0,
            weight_decay=0.01
        )
    assert "Invalid learning rate" in str(exc_info.value), "Did not raise ValueError for zero learning rate."

# 15. Scheduler Configuration with Additional Arguments

def test_configure_scheduler_with_additional_kwargs(default_config, mock_model):
    """Test configure_scheduler with additional scheduler-specific keyword arguments."""
    optimizer = configure_optimizer(mock_model, default_config)
    config = {
        "scheduler_type": "linear",
        "num_warmup_steps": 100,
        "num_training_steps": 1000,
        "additional_arg": "value"  # Non-standard argument
    }
    with patch('src.training.optimizers.get_scheduler') as mock_get_scheduler:
        mock_scheduler = MagicMock()
        mock_get_scheduler.return_value = mock_scheduler
        scheduler = configure_scheduler(optimizer, num_training_steps=1000, config=config)
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
    """Test get_optimizer with parameter groups having overlapping parameters."""
    param1 = mock_model.weight
    param2 = mock_model.bias
    # Intentional overlap: param1 is in both groups
    parameter_groups = [
        {"params": [param1], "lr": 0.01, "weight_decay": 0.001},
        {"params": [param1, param2], "lr": 0.02, "weight_decay": 0.0001}
    ]
    with pytest.raises(ValueError) as exc_info:
        get_optimizer(
            mock_model,
            optimizer_type="adamw",
            lr=5e-5,
            weight_decay=0.01,
            parameter_groups=parameter_groups
        )
    assert "Some parameters appear in more than one parameter group" in str(exc_info.value), "Did not raise ValueError for overlapping parameters."

# 17. Scheduler Configuration Missing Required Fields

def test_configure_scheduler_missing_required_fields(default_config, mock_model):
    """Test configure_scheduler when required scheduler fields are missing."""
    optimizer = configure_optimizer(mock_model, default_config)
    config = {
        "scheduler_type": "linear",
        # Missing 'num_training_steps'
    }
    with pytest.raises(ValueError) as exc_info:
        configure_scheduler(optimizer, num_training_steps=None, config=config)
    assert "linear requires `num_training_steps`" in str(exc_info.value), "Did not raise ValueError for missing num_training_steps."

# 18. Test get_optimizer_and_scheduler with Scheduler Return None

from unittest.mock import ANY

def test_get_optimizer_and_scheduler_scheduler_none(default_config, mock_model):
    """Test get_optimizer_and_scheduler when scheduler is None."""
    with patch('src.training.optimizers.get_scheduler') as mock_get_scheduler:
        mock_get_scheduler.return_value = None
        optimizer, scheduler = get_optimizer_and_scheduler(mock_model, default_config, num_training_steps=1000)
        mock_get_scheduler.assert_called_once_with(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=1000
        )
        assert scheduler is None, "Scheduler should be None as mocked."

# 19. Test get_optimizer_and_scheduler with Scheduler Type Dependent on Optimizer

def test_get_optimizer_and_scheduler_scheduler_dependent_on_optimizer(custom_config, mock_model):
    """Test get_optimizer_and_scheduler where scheduler type depends on optimizer type."""
    # Update parameter_groups to use actual tensor parameters and include weight_decay
    param1 = mock_model.weight
    param2 = mock_model.bias
    custom_config["parameter_groups"] = [
        {"params": [param1], "lr": 0.05, "weight_decay": 0.001},
        {"params": [param2], "lr": 0.1, "weight_decay": 0.0001}
    ]
    # Set scheduler type to something specific, e.g., cosine
    custom_config["scheduler_type"] = "cosine"

    with patch('src.training.optimizers.get_scheduler') as mock_get_scheduler:
        mock_scheduler = MagicMock()
        mock_get_scheduler.return_value = mock_scheduler

        optimizer, scheduler = get_optimizer_and_scheduler(mock_model, custom_config, num_training_steps=2000)

        # Check optimizer
        assert isinstance(optimizer, torch.optim.SGD), "Optimizer should be SGD."
        assert len(optimizer.param_groups) == 2, "There should be two parameter groups."
        assert optimizer.param_groups[0]['lr'] == 0.05, "First parameter group learning rate mismatch."
        assert optimizer.param_groups[0]['weight_decay'] == 0.001, "First parameter group weight decay mismatch."
        assert optimizer.param_groups[1]['lr'] == 0.1, "Second parameter group learning rate mismatch."
        assert optimizer.param_groups[1]['weight_decay'] == 0.0001, "Second parameter group weight decay mismatch."

        # Check scheduler
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
    # Update parameter_groups to use actual tensor parameters and include weight_decay
    param1 = mock_model.weight
    param2 = mock_model.bias
    custom_config["parameter_groups"] = [
        {"params": [param1], "lr": 0.05, "weight_decay": 0.001},
        {"params": [param2], "lr": 0.1, "weight_decay": 0.0001}
    ]

    with patch('src.training.optimizers.get_scheduler') as mock_get_scheduler:
        mock_scheduler = MagicMock()
        mock_get_scheduler.return_value = mock_scheduler

        optimizer, scheduler = get_optimizer_and_scheduler(mock_model, custom_config, num_training_steps=2000)

        # Check optimizer
        assert isinstance(optimizer, torch.optim.SGD), "Optimizer should be SGD."
        assert len(optimizer.param_groups) == 2, "There should be two parameter groups."
        assert optimizer.param_groups[0]['lr'] == 0.05, "First parameter group learning rate mismatch."
        assert optimizer.param_groups[0]['weight_decay'] == 0.001, "First parameter group weight decay mismatch."
        assert optimizer.param_groups[1]['lr'] == 0.1, "Second parameter group learning rate mismatch."
        assert optimizer.param_groups[1]['weight_decay'] == 0.0001, "Second parameter group weight decay mismatch."

        # Check scheduler
        mock_get_scheduler.assert_called_once_with(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=2000
        )
        assert scheduler == mock_scheduler, "Scheduler should be the mocked scheduler instance."
