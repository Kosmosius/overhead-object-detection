# src/training/optimizers.py

import torch
from torch.optim import SGD, Adam, RMSprop, AdamW
from transformers import get_scheduler
from typing import Optional, List, Dict, Any, Tuple, Union

def get_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    parameter_groups: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Return the appropriate optimizer for the given model.

    Args:
        model (torch.nn.Module): The model being trained.
        optimizer_type (str): The type of optimizer to use ('adamw', 'adam', 'sgd', or 'rmsprop').
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 penalty) for the optimizer.
        parameter_groups (List[Dict[str, Any]], optional): 
            Optional list of parameter groups with specific learning rates or optimizations.
        **kwargs: Additional keyword arguments for optimizer initialization.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.

    Raises:
        ValueError: If the optimizer type is unsupported or parameter_groups are invalid.
    """
    optimizer_type = optimizer_type.lower()

    # Define supported optimizers
    optimizers = {
        "adamw": AdamW,
        "adam": Adam,
        "sgd": SGD,
        "rmsprop": RMSprop,
    }

    if optimizer_type not in optimizers:
        raise ValueError(
            f"Unsupported optimizer type '{optimizer_type}'. Supported types: {list(optimizers.keys())}"
        )

    optimizer_class = optimizers[optimizer_type]

    if parameter_groups:
        if not isinstance(parameter_groups, list) or not all(isinstance(pg, dict) for pg in parameter_groups):
            raise ValueError("parameter_groups must be a list of dictionaries.")
        params = parameter_groups
    else:
        params = model.parameters()

    try:
        optimizer = optimizer_class(params, lr=learning_rate, weight_decay=weight_decay, **kwargs)
    except TypeError as e:
        raise TypeError(f"Error initializing optimizer '{optimizer_type}': {e}")

    return optimizer

def configure_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Configure the optimizer from a configuration file.

    Args:
        model (torch.nn.Module): The model being optimized.
        config (dict): Configuration for optimizer parameters.

    Returns:
        torch.optim.Optimizer: Configured optimizer based on the config file.
    """
    return get_optimizer(
        model=model,
        optimizer_type=config.get("optimizer_type", "adamw"),
        learning_rate=config.get("learning_rate", 5e-5),
        weight_decay=config.get("weight_decay", 0.01),
        parameter_groups=config.get("parameter_groups", None)
    )

def configure_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    config: Dict[str, Any]
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Configure the learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer being used.
        num_training_steps (int): Total number of training steps.
        config (dict): Configuration for scheduler settings.

    Returns:
        torch.optim.lr_scheduler.LRScheduler: Configured learning rate scheduler.
    """
    scheduler_type = config.get("scheduler_type", "linear")
    num_warmup_steps = config.get("num_warmup_steps", 0)

    return get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

def get_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: Dict[str, Any],
    num_training_steps: int
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """
    Return both optimizer and scheduler based on the provided configuration.

    Args:
        model (torch.nn.Module): The model being trained.
        config (dict): Configuration containing optimizer and scheduler details.
        num_training_steps (int): The number of training steps.

    Returns:
        tuple: Optimizer and scheduler objects.
    """
    optimizer = configure_optimizer(model, config)
    scheduler = configure_scheduler(optimizer, num_training_steps, config)

    return optimizer, scheduler
