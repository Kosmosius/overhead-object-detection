# src/training/optimizers.py

import torch
from torch.optim import SGD, Adam, RMSprop, AdamW
from transformers import get_scheduler, SchedulerType
from typing import Optional, List, Dict, Any, Tuple
from torch.optim.lr_scheduler import _LRScheduler
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Define the mapping from SchedulerType to scheduler functions or string identifiers
TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.LINEAR: "linear",
    SchedulerType.COSINE: "cosine",
    SchedulerType.COSINE_WITH_RESTARTS: "cosine_with_restarts",
    SchedulerType.POLYNOMIAL: "polynomial",
    SchedulerType.CONSTANT_WITH_WARMUP: "constant_with_warmup",
    SchedulerType.INVERSE_SQRT: "inverse_sqrt",
    SchedulerType.WARMUP_STABLE_DECAY: "warmup_stable_decay",
    # Add other scheduler types as needed
}

def get_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = "adamw",
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    parameter_groups: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Return the appropriate optimizer for the given model.

    Args:
        model (torch.nn.Module): The model being trained.
        optimizer_type (str): The type of optimizer to use ('adamw', 'adam', 'sgd', or 'rmsprop').
        lr (float): The learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 penalty) for the optimizer.
        parameter_groups (List[Dict[str, Any]], optional):
            Optional list of parameter groups with specific learning rates or optimizations.
        **kwargs: Additional keyword arguments for optimizer initialization.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.

    Raises:
        ValueError: If the optimizer type is unsupported or parameter_groups are invalid.
        TypeError: If a parameter group contains non-Parameter elements.
    """
    optimizer_type = optimizer_type.lower()
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

    # Exclude 'lr' and 'weight_decay' from **kwargs to prevent duplication
    excluded_keys = ['lr', 'weight_decay']
    optimizer_kwargs = {k: v for k, v in kwargs.items() if k not in excluded_keys}

    # Always set 'lr' and 'weight_decay' in optimizer_kwargs
    optimizer_kwargs['lr'] = lr
    optimizer_kwargs['weight_decay'] = weight_decay

    if parameter_groups is not None:
        if not isinstance(parameter_groups, list) or not all(isinstance(pg, dict) for pg in parameter_groups):
            raise ValueError("parameter_groups must be a list of dictionaries.")
        if len(parameter_groups) == 0:
            raise ValueError("parameter_groups cannot be an empty list.")

        # Ensure no overlapping parameters across groups
        seen_params = set()
        for group in parameter_groups:
            if "params" not in group:
                raise ValueError("Each parameter group must have a 'params' list.")
            params = group["params"]
            if not isinstance(params, list):
                raise ValueError("Each parameter group must have a 'params' list.")
            for param in params:
                if param in seen_params:
                    raise ValueError("Some parameters appear in more than one parameter group.")
                if not isinstance(param, torch.nn.Parameter):
                    raise TypeError(
                        "optimizer can only optimize Tensors, "
                        f"but one of the params is {type(param).__name__}"
                    )
                seen_params.add(param)

        params = parameter_groups
    else:
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        params = model.parameters()

    try:
        optimizer = optimizer_class(params, **optimizer_kwargs)
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
    optimizer_type = config.get('optimizer_type', 'AdamW')
    learning_rate = config.get('learning_rate', 1e-4)

    if learning_rate <= 0:
        raise ValueError("Learning rate must be a positive value.")

    if optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, **config.get('optimizer_kwargs', {}))
    
    return get_optimizer(
        model=model,
        optimizer_type=config.get("optimizer_type", "adamw"),
        lr=config.get("lr", 5e-5),
        weight_decay=config.get("weight_decay", 0.01),
        parameter_groups=config.get("parameter_groups", None)
    )

def configure_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: Optional[int],
    config: Dict[str, Any]
) -> Optional[_LRScheduler]:
    """
    Configure the scheduler from a configuration file.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which the scheduler is being configured.
        num_training_steps (int): The total number of training steps.
        config (dict): Configuration for scheduler parameters.

    Returns:
        Optional[_LRScheduler]: Configured scheduler or None if not applicable.
    """
    scheduler_type = config.get("scheduler_type", "linear")
    num_warmup_steps = config.get("num_warmup_steps", 0)
    # Exclude scheduler-related and optimizer-related keys
    excluded_keys = [
        "scheduler_type",
        "num_warmup_steps",
        "num_training_steps",
        "optimizer_type",
        "lr",
        "weight_decay",
        "parameter_groups"
    ]
    scheduler_specific_kwargs = {
        k: v for k, v in config.items() if k not in excluded_keys
    }

    try:
        schedule_type = SchedulerType(scheduler_type)
    except ValueError:
        raise ValueError(f"Unsupported scheduler type '{scheduler_type}'.")

    schedulers_requiring_steps = {
        SchedulerType.LINEAR,
        SchedulerType.COSINE,
        SchedulerType.POLYNOMIAL,
        SchedulerType.CONSTANT_WITH_WARMUP,
        SchedulerType.INVERSE_SQRT,
        SchedulerType.WARMUP_STABLE_DECAY,
        SchedulerType.COSINE_WITH_RESTARTS,
    }

    if schedule_type in schedulers_requiring_steps and num_training_steps is None:
        raise ValueError(f"{scheduler_type} requires `num_training_steps`, please provide that argument.")

    schedule_func = TYPE_TO_SCHEDULER_FUNCTION.get(schedule_type, None)
    if schedule_func is None:
        raise ValueError(f"Unsupported scheduler type '{scheduler_type}'.")

    return get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **scheduler_specific_kwargs
    )

def get_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: Dict[str, Any],
    num_training_steps: int
) -> Tuple[torch.optim.Optimizer, Optional[_LRScheduler]]:
    """
    Return both optimizer and scheduler based on the provided configuration.

    Args:
        model (torch.nn.Module): The model being trained.
        config (dict): Configuration containing optimizer and scheduler details.
        num_training_steps (int): The number of training steps.

    Returns:
        Tuple[torch.optim.Optimizer, Optional[_LRScheduler]]: 
            Optimizer and scheduler objects.
    """
    try:
        parameter_groups = config['parameter_groups']
    except KeyError:
        logger.error("'params' key is missing in parameter_groups")
        raise KeyError("'params' key is missing in parameter_groups")
    
    optimizer = configure_optimizer(model, config)
    scheduler = configure_scheduler(optimizer, num_training_steps, config)

    return optimizer, scheduler
