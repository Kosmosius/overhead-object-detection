# src/training/optimizers.py

import torch
from torch.optim import SGD, Adam, RMSprop, AdamW, Nadam
from transformers import get_scheduler

def get_optimizer(model, optimizer_type="adamw", learning_rate=5e-5, weight_decay=0.01, parameter_groups=None):
    """
    Return the appropriate optimizer for the given model.
    
    Args:
        model (torch.nn.Module): The model being trained.
        optimizer_type (str): The type of optimizer to use ('adamw', 'sgd', 'adam', 'nadam', or 'rmsprop').
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 penalty) for the optimizer.
        parameter_groups (list, optional): Optional list of parameter groups with specific learning rates or optimizations.
        
    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """
    if parameter_groups:
        params = parameter_groups
    else:
        params = model.parameters()
    
    optimizers = {
        "adamw": AdamW(params, lr=learning_rate, weight_decay=weight_decay),
        "adam": Adam(params, lr=learning_rate, weight_decay=weight_decay),
        "sgd": SGD(params, lr=learning_rate, weight_decay=weight_decay, momentum=0.9),
        "rmsprop": RMSprop(params, lr=learning_rate, weight_decay=weight_decay, momentum=0.9),
        "nadam": Nadam(params, lr=learning_rate, weight_decay=weight_decay)
    }

    if optimizer_type.lower() not in optimizers:
        raise ValueError(f"Unsupported optimizer type '{optimizer_type}'. Supported types: {list(optimizers.keys())}")

    return optimizers[optimizer_type.lower()]

def configure_optimizer(model, config):
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

def configure_scheduler(optimizer, num_training_steps, config):
    """
    Configure the learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer being used.
        num_training_steps (int): Total number of training steps.
        config (dict): Configuration for scheduler settings.

    Returns:
        torch.optim.lr_scheduler: Configured learning rate scheduler.
    """
    return get_scheduler(
        name=config.get("scheduler_type", "linear"),
        optimizer=optimizer,
        num_warmup_steps=config.get("num_warmup_steps", 0),
        num_training_steps=num_training_steps
    )

def get_optimizer_and_scheduler(model, config, num_training_steps):
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
