# src/training/optimizers.py

from transformers import AdamW
from torch.optim import SGD, RMSprop

def get_optimizer(model, optimizer_type="adamw", learning_rate=5e-5, weight_decay=0.01):
    """
    Returns the optimizer for training the model.

    Args:
        model (nn.Module): The model whose parameters are being optimized.
        optimizer_type (str): The type of optimizer to use. Options: ['adamw', 'sgd', 'rmsprop'].
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay to apply (L2 regularization).

    Returns:
        torch.optim.Optimizer: Initialized optimizer for the model parameters.
    """
    # Get the model parameters
    model_params = model.parameters()

    if optimizer_type.lower() == "adamw":
        # HuggingFace's recommended optimizer for Transformers models
        return AdamW(model_params, lr=learning_rate, weight_decay=weight_decay)
    
    elif optimizer_type.lower() == "sgd":
        # Option for Stochastic Gradient Descent (SGD)
        return SGD(model_params, lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    
    elif optimizer_type.lower() == "rmsprop":
        # Option for RMSprop optimizer
        return RMSprop(model_params, lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    
    else:
        raise ValueError(f"Optimizer type '{optimizer_type}' not supported. Choose from ['adamw', 'sgd', 'rmsprop'].")

def configure_optimizer(model, optimizer_type="adamw", learning_rate=5e-5, weight_decay=0.01):
    """
    Configures the optimizer with the model and returns it for training.

    Args:
        model (nn.Module): The model to be optimized.
        optimizer_type (str): The type of optimizer to use (default is 'adamw').
        learning_rate (float): The learning rate to use for the optimizer.
        weight_decay (float): The weight decay (L2 penalty) to apply to the optimizer.

    Returns:
        Optimizer: Configured optimizer ready for model training.
    """
    return get_optimizer(
        model=model,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
