# src/training/optimizers.py

from transformers import AdamW, get_scheduler, Adafactor
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def get_optimizer(
    model,
    optimizer_name: str = "adamw",
    learning_rate: float = 5e-5,
    weight_decay: float = 0.0,
    **kwargs
):
    """
    Creates an optimizer for the given model parameters using HuggingFace's optimizers.

    Args:
        model (torch.nn.Module): The model to optimize.
        optimizer_name (str): Name of the optimizer ('adamw', 'adafactor').
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay coefficient.
        **kwargs: Additional keyword arguments for the optimizer.

    Returns:
        torch.optim.Optimizer: The optimizer instance.
    """
    optimizer_name = optimizer_name.lower()
    optimizer = None

    if optimizer_name == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
        logger.info("Using AdamW optimizer.")
    elif optimizer_name == "adafactor":
        optimizer = Adafactor(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            scale_parameter=False,
            relative_step=False,
            **kwargs
        )
        logger.info("Using Adafactor optimizer.")
    else:
        raise ValueError(f"Unsupported optimizer name '{optimizer_name}'.")

    return optimizer


def get_scheduler(
    optimizer,
    scheduler_name: str,
    num_warmup_steps: int,
    num_training_steps: int,
    **kwargs
):
    """
    Creates a learning rate scheduler using HuggingFace's scheduler functions.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        scheduler_name (str): Name of the scheduler.
        num_warmup_steps (int): Number of warmup steps.
        num_training_steps (int): Total number of training steps.
        **kwargs: Additional keyword arguments for the scheduler.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: The scheduler instance.
    """
    scheduler = get_scheduler(
        name=scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **kwargs
    )
    logger.info(f"Using {scheduler_name} scheduler.")
    return scheduler


def get_optimizer_and_scheduler(
    model,
    optimizer_name: str,
    scheduler_name: str,
    learning_rate: float,
    weight_decay: float,
    num_warmup_steps: int,
    num_training_steps: int,
    **kwargs
) -> Tuple:
    """
    Creates both optimizer and scheduler.

    Args:
        model (torch.nn.Module): The model to optimize.
        optimizer_name (str): Name of the optimizer.
        scheduler_name (str): Name of the scheduler.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay coefficient.
        num_warmup_steps (int): Number of warmup steps for the scheduler.
        num_training_steps (int): Total number of training steps.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]: The optimizer and scheduler instances.
    """
    optimizer = get_optimizer(
        model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        **kwargs
    )

    scheduler = get_scheduler(
        optimizer,
        scheduler_name=scheduler_name,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **kwargs
    )

    return optimizer, scheduler
