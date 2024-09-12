# tests/unit/src/utils/test_system_utils.py

import torch
import pytest
from src.utils.system_utils import check_device, check_system_info, check_memory_requirements, check_python_version, check_disk_space

def test_check_device():
    device = check_device()
    assert isinstance(device, torch.device)

def test_check_system_info():
    system_info = check_system_info()
    assert "Platform" in system_info
    assert "Total RAM (GB)" in system_info

def test_check_memory_requirements():
    assert check_memory_requirements(min_memory_gb=1)

def test_check_python_version():
    assert check_python_version(min_version=(3, 6))

def test_check_disk_space():
    assert check_disk_space(min_disk_gb=1)
