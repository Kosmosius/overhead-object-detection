# src/training/optimizers.py

from transformers import AdamW, get_scheduler
from torch.optim import SGD, RMSprop

def get_optimizer(model, optimizer_type="adamw", learning_rate=5e-5, weight_decay=0.01):
    """
    Return the appropriate optimizer for the given model.

    Args:
        model (nn.Module): The model being trained.
        optimizer_type (str): The type of optimizer to use ('adamw', 'sgd', or 'rmsprop').
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 penalty) for the optimizer.

    Returns:
        Optimizer: The initialized optimizer.
    """
    params = model.parameters()

    if optimizer_type == "adamw":
        return AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        return SGD(params, lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_type == "rmsprop":
        return RMSprop(params, lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer type {optimizer_type}")

def configure_optimizer(model, config):
    """
    Configure the optimizer from a configuration file.

    Args:
        model (nn.Module): The model being optimized.
        config (dict): Configuration for optimizer parameters.

    Returns:
        Optimizer: Configured optimizer based on the config file.
    """
    return get_optimizer(
        model=model,
        optimizer_type=config.get("optimizer_type", "adamw"),
        learning_rate=config.get("learning_rate", 5e-5),
        weight_decay=config.get("weight_decay", 0.01)
    )

def configure_scheduler(optimizer, num_training_steps, config):
    """
    Configure the learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer being used.
        num_training_steps (int): Total number of training steps.
        config (dict): Configuration for scheduler settings.

    Returns:
        Scheduler: Configured learning rate scheduler.
    """
    return get_scheduler(
        name=config.get("scheduler_type", "linear"),
        optimizer=optimizer,
        num_warmup_steps=config.get("num_warmup_steps", 0),
        num_training_steps=num_training_steps
    )
