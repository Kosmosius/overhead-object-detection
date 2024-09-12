# src/utils/logging.py

import logging
import os
from logging.handlers import RotatingFileHandler
from transformers.utils.logging import get_logger as hf_get_logger


def setup_logging(log_file="logs/evaluation.log", log_level=logging.INFO, max_bytes=10_485_760, backup_count=5):
    """
    Set up logging for the project, logging to both the console and optionally to a rotating file.
    
    Args:
        log_file (str, optional): Path to the log file. If None, logs will not be written to a file.
        log_level (int, optional): Logging level (default is logging.INFO).
        max_bytes (int, optional): Maximum size in bytes for the rotating log file before a new one is created (default: 10MB).
        backup_count (int, optional): Number of backup log files to keep (default: 5).
    """
    # Create log directory if needed
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    # Set up log format and handlers
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]

    # Add rotating file handler if log_file is provided
    if log_file:
        rotating_file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        handlers.append(rotating_file_handler)

    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)

    # Use HuggingFace's logger for consistency
    hf_logger = hf_get_logger("transformers")
    hf_logger.setLevel(log_level)

    # Log initialization
    logging.info(f"Logging setup complete. Log level: {logging.getLevelName(log_level)}")
    if log_file:
        logging.info(f"Logging to file: {log_file} (max {max_bytes / (1024 ** 2)}MB, {backup_count} backups)")


def get_logger_for_module(module_name):
    """
    Get a logger for a specific module, using a consistent naming convention.
    
    Args:
        module_name (str): Name of the module.

    Returns:
        logging.Logger: Configured logger for the module.
    """
    return logging.getLogger(module_name)


def log_model_info(model, logger=None):
    """
    Log model details such as architecture and parameter count.
    
    Args:
        model (PreTrainedModel): HuggingFace model instance.
        logger (logging.Logger, optional): Logger to use. If None, the default logger is used.
    """
    logger = logger or logging.getLogger(__name__)
    model_name = model.__class__.__name__
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {model_name}, Total trainable parameters: {param_count:,}")


def log_metrics(metrics, epoch=None, phase="Training", logger=None):
    """
    Log metrics such as precision, recall, mAP, etc., for a specific phase (training, validation).
    
    Args:
        metrics (dict): Dictionary of metrics to log.
        epoch (int, optional): Epoch number. If provided, it is included in the log.
        phase (str): Indicates whether the metrics are for "Training" or "Validation".
        logger (logging.Logger, optional): Logger to use. If None, the default logger is used.
    """
    logger = logger or logging.getLogger(__name__)
    
    if epoch is not None:
        logger.info(f"Epoch {epoch} - {phase} Metrics:")
    else:
        logger.info(f"{phase} Metrics:")
    
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")


def dynamic_log_level(new_level):
    """
    Dynamically change the log level for all loggers.
    
    Args:
        new_level (str): New logging level as a string (e.g., 'DEBUG', 'INFO', 'WARNING', etc.).
    """
    level = getattr(logging, new_level.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {new_level}")

    for handler in logging.root.handlers:
        handler.setLevel(level)
    
    logging.getLogger().setLevel(level)
    logging.info(f"Log level changed to {new_level}")


# Example of using dynamic_log_level
# dynamic_log_level('DEBUG')
