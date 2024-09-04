# src/utils/logging.py

import logging
import os
from transformers.utils.logging import get_logger

def setup_logging(log_file=None, log_level=logging.INFO):
    """
    Setup logging for the project. Logs to both the console and optionally to a file.

    Args:
        log_file (str, optional): Path to the log file. If None, logs will not be written to a file.
        log_level (int, optional): Logging level. Defaults to logging.INFO.
    """
    # Set up the basic configuration
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)

    # Use HuggingFace's logger for consistent logging
    transformers_logger = get_logger("transformers")
    transformers_logger.setLevel(log_level)

    logging.info(f"Logging setup complete. Log level: {log_level}")
    if log_file:
        logging.info(f"Logging to file: {log_file}")

def get_logger_for_module(module_name):
    """
    Get a logger for a specific module in the project.

    Args:
        module_name (str): Name of the module.

    Returns:
        logging.Logger: Logger instance for the module.
    """
    logger = logging.getLogger(module_name)
    return logger

def log_model_info(model, logger=None):
    """
    Log model details like architecture and parameters count.

    Args:
        model (PreTrainedModel): HuggingFace model instance.
        logger (logging.Logger, optional): Logger to use. If None, will use the default logger.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    model_name = model.__class__.__name__
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {model_name}, Total parameters: {param_count:,}")

def log_training_metrics(metrics, epoch, logger=None):
    """
    Log training metrics for a given epoch.

    Args:
        metrics (dict): Dictionary of metrics to log.
        epoch (int): Epoch number.
        logger (logging.Logger, optional): Logger to use. If None, will use the default logger.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Epoch {epoch} - Training Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")

def log_validation_metrics(metrics, logger=None):
    """
    Log validation metrics.

    Args:
        metrics (dict): Dictionary of validation metrics to log.
        logger (logging.Logger, optional): Logger to use. If None, will use the default logger.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Validation Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")
