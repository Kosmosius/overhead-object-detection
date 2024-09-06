# tests/test_setup_environment.py

import os
import pytest
import subprocess
import sys
from unittest import mock
from scripts.setup_environment import (
    install_requirements, setup_huggingface_cache, install_nvidia_drivers, configure_environment
)
from src.utils.system_utils import check_python_version, check_memory_requirements, check_disk_space, check_hf_library_installation

# Unit test for install_requirements
@mock.patch('subprocess.check_call')
def test_install_requirements(mock_check_call):
    """Test normal installation of requirements using pip."""
    install_requirements("requirements.txt")
    mock_check_call.assert_called_once_with([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

@mock.patch('subprocess.check_call', side_effect=subprocess.CalledProcessError(1, 'pip install'))
def test_install_requirements_failure(mock_check_call):
    """Test failure in installing requirements."""
    with pytest.raises(SystemExit):
        install_requirements("requirements.txt")
    mock_check_call.assert_called_once()

# Unit test for setup_huggingface_cache
@mock.patch('os.makedirs')
@mock.patch.dict(os.environ, {}, clear=True)
def test_setup_huggingface_cache(mock_makedirs):
    """Test setting up the HuggingFace cache directory."""
    setup_huggingface_cache(".cache/huggingface")
    
    # Assert that environment variable was set and directory was created
    assert os.environ["HF_HOME"] == ".cache/huggingface"
    mock_makedirs.assert_called_once_with(".cache/huggingface", exist_ok=True)

@mock.patch('os.makedirs', side_effect=PermissionError)
def test_setup_huggingface_cache_permission_error(mock_makedirs):
    """Test handling of permission errors when creating the HuggingFace cache directory."""
    with pytest.raises(PermissionError):
        setup_huggingface_cache(".cache/huggingface")
    mock_makedirs.assert_called_once()

# Unit test for install_nvidia_drivers (Linux-specific)
@mock.patch('sys.platform', 'linux')
@mock.patch('subprocess.run')
def test_install_nvidia_drivers_linux(mock_run):
    """Test NVIDIA driver installation on Linux."""
    install_nvidia_drivers()
    mock_run.assert_called_once_with(["sudo", "apt-get", "install", "-y", "nvidia-driver-470"], check=True)

@mock.patch('sys.platform', 'linux')
@mock.patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'apt-get install'))
def test_install_nvidia_drivers_linux_failure(mock_run):
    """Test failure in installing NVIDIA drivers on Linux."""
    with pytest.raises(SystemExit):
        install_nvidia_drivers()
    mock_run.assert_called_once()

# Test NVIDIA driver installation on non-Linux platform
@mock.patch('sys.platform', 'darwin')
def test_install_nvidia_drivers_non_linux():
    """Test that NVIDIA drivers are not installed on non-Linux platforms."""
    with mock.patch('subprocess.run') as mock_run:
        install_nvidia_drivers()
        mock_run.assert_not_called()

# Unit test for configure_environment
@mock.patch('scripts.setup_environment.install_requirements')
@mock.patch('scripts.setup_environment.setup_huggingface_cache')
@mock.patch('scripts.setup_environment.install_nvidia_drivers')
@mock.patch('src.utils.system_utils.check_python_version', return_value=True)
@mock.patch('src.utils.system_utils.check_memory_requirements', return_value=True)
@mock.patch('src.utils.system_utils.check_disk_space', return_value=True)
@mock.patch('src.utils.system_utils.check_hf_library_installation')
def test_configure_environment_normal(
    mock_hf_check, mock_disk_check, mock_memory_check, mock_python_check,
    mock_nvidia_install, mock_cache_setup, mock_requirements_install
):
    """Test normal environment configuration."""
    configure_environment()
    
    # Ensure all system checks and setup steps are called
    mock_python_check.assert_called_once_with(min_version=(3, 8))
    mock_memory_check.assert_called_once_with(min_memory_gb=16)
    mock_disk_check.assert_called_once_with(min_disk_gb=20)
    mock_hf_check.assert_called_once()
    mock_requirements_install.assert_called_once()
    mock_cache_setup.assert_called_once()
    mock_nvidia_install.assert_called_once()

# Edge case test: Insufficient memory
@mock.patch('src.utils.system_utils.check_python_version', return_value=True)
@mock.patch('src.utils.system_utils.check_memory_requirements', return_value=False)
@mock.patch('src.utils.system_utils.check_disk_space', return_value=True)
def test_configure_environment_insufficient_memory(mock_disk_check, mock_memory_check, mock_python_check):
    """Test environment configuration fails due to insufficient memory."""
    with pytest.raises(SystemExit):
        configure_environment()
    
    # Ensure memory check fails and halts configuration
    mock_memory_check.assert_called_once()

# Edge case test: Insufficient disk space
@mock.patch('src.utils.system_utils.check_python_version', return_value=True)
@mock.patch('src.utils.system_utils.check_memory_requirements', return_value=True)
@mock.patch('src.utils.system_utils.check_disk_space', return_value=False)
def test_configure_environment_insufficient_disk_space(mock_disk_check, mock_memory_check, mock_python_check):
    """Test environment configuration fails due to insufficient disk space."""
    with pytest.raises(SystemExit):
        configure_environment()
    
    # Ensure disk space check fails and halts configuration
    mock_disk_check.assert_called_once()

# Edge case test: Incorrect Python version
@mock.patch('src.utils.system_utils.check_python_version', return_value=False)
def test_configure_environment_incorrect_python(mock_python_check):
    """Test environment configuration fails due to incorrect Python version."""
    with pytest.raises(SystemExit):
        configure_environment()
    
    # Ensure Python version check fails and halts configuration
    mock_python_check.assert_called_once()

