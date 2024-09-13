# src/utils/logging_utils.py

import logging
import os
from typing import Optional, Dict, Any
import yaml
import logging.config

def setup_logging(
    default_path: str = 'configs/logging_config.yaml',
    default_level: int = logging.INFO,
    env_key: str = 'LOG_CFG'
) -> None:
    """
    Set up logging configuration.

    Args:
        default_path (str): Path to the default logging configuration file.
        default_level (int): Default logging level.
        env_key (str): Environment variable key for logging configuration path.
    """
    path = os.getenv(env_key, default_path)
    if os.path.exists(path):
        with open(path, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
            except Exception as e:
                print(f"Error in Logging Configuration: {e}")
                logging.basicConfig(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        print(f"Failed to load configuration file. Using default configs")

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name (str, optional): Name of the logger. If None, get the root logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)

def log_model_info(model: Any, logger: Optional[logging.Logger] = None) -> None:
    """
    Log model details such as architecture and parameter count.

    Args:
        model (Any): The model instance.
        logger (logging.Logger, optional): Logger to use. If None, use the root logger.
    """
    logger = logger or logging.getLogger(__name__)
    model_name = model.__class__.__name__
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {model_name}, Total trainable parameters: {param_count:,}")

def log_metrics(
    metrics: Dict[str, Any],
    epoch: Optional[int] = None,
    phase: str = "Training",
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log metrics for a specific phase (training, validation).

    Args:
        metrics (Dict[str, Any]): Dictionary of metrics to log.
        epoch (int, optional): Epoch number. If provided, include it in the log.
        phase (str): Indicates whether the metrics are for 'Training' or 'Validation'.
        logger (logging.Logger, optional): Logger to use. If None, use the root logger.
    """
    logger = logger or logging.getLogger(__name__)
    if epoch is not None:
        logger.info(f"Epoch {epoch} - {phase} Metrics:")
    else:
        logger.info(f"{phase} Metrics:")

    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")

def dynamic_log_level(new_level: str) -> None:
    """
    Dynamically change the log level for all loggers.

    Args:
        new_level (str): New logging level as a string (e.g., 'DEBUG', 'INFO', 'WARNING').
    """
    level = getattr(logging, new_level.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {new_level}")

    logging.getLogger().setLevel(level)
    for handler in logging.root.handlers:
        handler.setLevel(level)
    logging.info(f"Log level changed to {new_level}")
