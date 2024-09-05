# src/utils/logging.py

import logging
import os
from logging.handlers import RotatingFileHandler
from transformers.utils.logging import get_logger

def setup_logging(log_file=None, log_level=logging.INFO, max_bytes=10_485_760, backup_count=5):
    """
    Set up logging for the project. Logs to both the console and optionally to a rotating file.
    
    Args:
        log_file (str, optional): Path to the log file. If None, logs will not be written to a file.
        log_level (int, optional): Logging level. Defaults to logging.INFO.
        max_bytes (int, optional): Maximum size in bytes for the rotating log file before a new one is created. Defaults to 10MB.
        backup_count (int, optional): Number of backup log files to keep. Defaults to 5.
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        rotating_file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        handlers.append(rotating_file_handler)

    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)

    # Use HuggingFace's logger for consistent logging
    transformers_logger = get_logger("transformers")
    transformers_logger.setLevel(log_level)

    logging.info(f"Logging setup complete. Log level: {logging.getLevelName(log_level)}")
    if log_file:
        logging.info(f"Logging to file: {log_file} (max {max_bytes / (1024 ** 2)}MB, {backup_count} backups)")

def get_logger_for_module(module_name):
    """
    Get a logger for a specific module in the project.

    Args:
        module_name (str): Name of the module.

    Returns:
        logging.Logger: Logger instance for the module.
    """
    return logging.getLogger(module_name)

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
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
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

