# src/utils/system_utils.py

import torch
import platform
import psutil
import logging

def check_device():
    """
    Check if a CUDA-capable GPU is available and return the appropriate device.

    Returns:
        torch.device: 'cuda' if a GPU is available, else 'cpu'.
    """
    if torch.cuda.is_available():
        logging.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device('cuda')
    else:
        logging.info("CUDA is not available. Using CPU.")
        return torch.device('cpu')

def check_system_info():
    """
    Log basic system information including platform, processor, and memory.

    Returns:
        dict: Dictionary containing system information (OS, CPU, memory).
    """
    system_info = {
        "Platform": platform.system(),
        "Platform Version": platform.version(),
        "Architecture": platform.architecture()[0],
        "Processor": platform.processor(),
        "CPU Cores": psutil.cpu_count(logical=False),
        "Logical Cores": psutil.cpu_count(logical=True),
        "Total RAM (GB)": round(psutil.virtual_memory().total / (1024 ** 3), 2),
    }

    for key, value in system_info.items():
        logging.info(f"{key}: {value}")

    return system_info

def check_memory_requirements(min_memory_gb):
    """
    Check if the system has enough RAM to meet the minimum memory requirements.

    Args:
        min_memory_gb (int): The minimum required memory in GB.

    Returns:
        bool: True if the system meets the memory requirements, False otherwise.
    """
    total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
    if total_memory_gb < min_memory_gb:
        logging.warning(f"Insufficient memory: {total_memory_gb:.2f} GB available, {min_memory_gb} GB required.")
        return False
    logging.info(f"Memory check passed: {total_memory_gb:.2f} GB available.")
    return True

def check_python_version(min_version=(3, 8)):
    """
    Check if the current Python version meets the minimum version requirement.

    Args:
        min_version (tuple): Minimum required Python version as a tuple (major, minor).

    Returns:
        bool: True if the current Python version meets the minimum requirement, False otherwise.
    """
    current_version = platform.python_version_tuple()
    current_version = tuple(map(int, current_version[:2]))

    if current_version < min_version:
        logging.warning(f"Python {min_version[0]}.{min_version[1]} or higher is required, but found Python {current_version[0]}.{current_version[1]}.")
        return False
    logging.info(f"Python version check passed: {platform.python_version()}")
    return True

