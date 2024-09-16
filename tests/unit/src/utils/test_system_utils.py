# tests/unit/src/utils/test_system_utils.py

import pytest
import torch
import platform
import psutil
import shutil
from unittest.mock import patch, MagicMock
from src.utils.system_utils import (
    check_device,
    check_system_info,
    check_memory_requirements,
    check_python_version,
    check_disk_space,
    check_hf_library_installation,
    check_system_requirements
)


# --- Fixtures ---

@pytest.fixture
def mock_cuda_available():
    """Fixture to mock CUDA availability."""
    with patch('torch.cuda.is_available') as mock_cuda_is_available:
        yield mock_cuda_is_available

@pytest.fixture
def mock_cuda_device_name():
    """Fixture to mock CUDA device name."""
    with patch('torch.cuda.get_device_name') as mock_cuda_get_device_name:
        yield mock_cuda_get_device_name

@pytest.fixture
def mock_platform_system():
    """Fixture to mock platform.system()."""
    with patch('platform.system', return_value='Linux') as mock_system:
        yield mock_system

@pytest.fixture
def mock_platform_version():
    """Fixture to mock platform.version()."""
    with patch('platform.version', return_value='5.4.0-42-generic') as mock_version:
        yield mock_version

@pytest.fixture
def mock_platform_architecture():
    """Fixture to mock platform.architecture()."""
    with patch('platform.architecture', return_value=('64bit', 'ELF')) as mock_arch:
        yield mock_arch

@pytest.fixture
def mock_platform_processor():
    """Fixture to mock platform.processor()."""
    with patch('platform.processor', return_value='x86_64') as mock_processor:
        yield mock_processor

@pytest.fixture
def mock_psutil_virtual_memory():
    """Fixture to mock psutil.virtual_memory()."""
    mock_virtual_memory = MagicMock()
    mock_virtual_memory.total = 16 * (1024 ** 3)  # 16 GB
    mock_virtual_memory.available = 8 * (1024 ** 3)  # 8 GB
    with patch('psutil.virtual_memory', return_value=mock_virtual_memory) as mock_vm:
        yield mock_vm

@pytest.fixture
def mock_psutil_cpu_count():
    """Fixture to mock psutil.cpu_count()."""
    with patch('psutil.cpu_count', side_effect=lambda logical=True: 8 if logical else 4) as mock_cpu_count:
        yield mock_cpu_count

@pytest.fixture
def mock_shutil_disk_usage():
    """Fixture to mock shutil.disk_usage()."""
    mock_disk_usage = MagicMock()
    mock_disk_usage.free = 100 * (1024 ** 3)  # 100 GB
    with patch('shutil.disk_usage', return_value=mock_disk_usage) as mock_du:
        yield mock_du

@pytest.fixture
def mock_platform_python_version_tuple():
    """Fixture to mock platform.python_version_tuple()."""
    with patch('platform.python_version_tuple', return_value=('3', '9', '5')) as mock_py_ver:
        yield mock_py_ver

@pytest.fixture
def mock_is_torch_available():
    """Fixture to mock transformers.utils.is_torch_available()."""
    with patch('src.utils.system_utils.is_torch_available', return_value=True) as mock_torch_available:
        yield mock_torch_available

@pytest.fixture
def mock_is_tf_available():
    """Fixture to mock transformers.utils.is_tf_available()."""
    with patch('src.utils.system_utils.is_tf_available', return_value=True) as mock_tf_available:
        yield mock_tf_available

# --- Test Cases ---

# 1. Device Checking Tests

def test_check_device_cuda_available(mock_cuda_available, mock_cuda_device_name, caplog):
    """Test that check_device returns 'cuda' when CUDA is available."""
    mock_cuda_available.return_value = True
    mock_cuda_device_name.return_value = "NVIDIA GeForce RTX 3080"

    device = check_device()
    assert device.type == 'cuda', "Device type should be 'cuda' when CUDA is available."
    mock_cuda_device_name.assert_called_once_with(0)
    assert "CUDA is available. Using GPU: NVIDIA GeForce RTX 3080" in caplog.text

def test_check_device_cuda_not_available(mock_cuda_available, caplog):
    """Test that check_device returns 'cpu' when CUDA is not available."""
    mock_cuda_available.return_value = False

    device = check_device()
    assert device.type == 'cpu', "Device type should be 'cpu' when CUDA is not available."
    assert "CUDA is not available. Using CPU." in caplog.text

# 2. System Information Tests

def test_check_system_info(mock_platform_system, mock_platform_version, mock_platform_architecture, 
                           mock_platform_processor, mock_psutil_virtual_memory, mock_psutil_cpu_count, 
                           mock_shutil_disk_usage, caplog):
    """Test that check_system_info retrieves and logs system information correctly."""
    system_info = check_system_info()
    expected_info = {
        "Platform": "Linux",
        "Platform Version": "5.4.0-42-generic",
        "Architecture": "64bit",
        "Processor": "x86_64",
        "CPU Cores": 4,
        "Logical Cores": 8,
        "Total RAM (GB)": 16.0,
        "Available RAM (GB)": 8.0,
        "Disk Space (GB)": 100.0
    }
    assert system_info == expected_info, "System information does not match expected values."
    
    # Check logs
    for key, value in expected_info.items():
        assert f"{key}: {value}" in caplog.text, f"{key} should be logged with value {value}."

def test_check_system_info_custom_disk_path(mock_platform_system, mock_platform_version, mock_platform_architecture, 
                                           mock_platform_processor, mock_psutil_virtual_memory, mock_psutil_cpu_count, 
                                           mock_shutil_disk_usage, caplog):
    """Test check_system_info on a non-standard disk path."""
    with patch('shutil.disk_usage', return_value=MagicMock(free=50 * (1024 ** 3))):
        system_info = check_system_info()
        assert system_info["Disk Space (GB)"] == 50.0, "Disk space should reflect the mocked free space."
        assert "Disk Space (GB): 50.0" in caplog.text

# 3. Memory Requirements Tests

def test_check_memory_requirements_sufficient(mock_psutil_virtual_memory, caplog):
    """Test that check_memory_requirements returns True when memory is sufficient."""
    mock_psutil_virtual_memory.return_value.total = 16 * (1024 ** 3)  # 16 GB
    result = check_memory_requirements(min_memory_gb=8)
    assert result is True, "check_memory_requirements should return True when memory is sufficient."
    assert "Memory check passed: 16.00 GB available." in caplog.text

def test_check_memory_requirements_insufficient(mock_psutil_virtual_memory, caplog):
    """Test that check_memory_requirements returns False when memory is insufficient."""
    mock_psutil_virtual_memory.return_value.total = 4 * (1024 ** 3)  # 4 GB
    result = check_memory_requirements(min_memory_gb=8)
    assert result is False, "check_memory_requirements should return False when memory is insufficient."
    assert "Insufficient memory: 4.00 GB available, 8 GB required." in caplog.text

# 4. Python Version Tests

@pytest.mark.parametrize("current_version, min_version, expected", [
    (('3', '9', '5'), (3, 8), True),
    (('3', '7', '9'), (3, 8), False),
    (('3', '8', '0'), (3, 8), True),
    (('4', '0', '0'), (3, 8), True),
])
def test_check_python_version(mock_platform_python_version_tuple, current_version, min_version, expected, caplog):
    """Test that check_python_version works correctly for various Python versions."""
    mock_platform_python_version_tuple.return_value = current_version
    result = check_python_version(min_version=min_version)
    assert result == expected, f"check_python_version should return {expected} for current version {current_version} and min_version {min_version}."
    if expected:
        assert f"Python version check passed: {'+'.join(current_version)}" in caplog.text
    else:
        assert f"Python {min_version[0]}.{min_version[1]} or higher is required" in caplog.text

def test_check_python_version_malformed_version(mock_platform_python_version_tuple, caplog):
    """Test that check_python_version handles malformed Python version tuples."""
    mock_platform_python_version_tuple.return_value = ('3', '8')  # Missing patch version
    with pytest.raises(ValueError):
        check_python_version(min_version=(3, 8))

# 5. Disk Space Tests

def test_check_disk_space_sufficient(mock_shutil_disk_usage, caplog):
    """Test that check_disk_space returns True when disk space is sufficient."""
    mock_shutil_disk_usage.return_value.free = 100 * (1024 ** 3)  # 100 GB
    result = check_disk_space(min_disk_gb=50)
    assert result is True, "check_disk_space should return True when disk space is sufficient."
    assert "Disk space check passed: 100.00 GB available." in caplog.text

def test_check_disk_space_insufficient(mock_shutil_disk_usage, caplog):
    """Test that check_disk_space returns False when disk space is insufficient."""
    mock_shutil_disk_usage.return_value.free = 20 * (1024 ** 3)  # 20 GB
    result = check_disk_space(min_disk_gb=50)
    assert result is False, "check_disk_space should return False when disk space is insufficient."
    assert "Insufficient disk space: 20.00 GB available, 50 GB required." in caplog.text

# 6. HuggingFace Library Installation Tests

def test_check_hf_library_installation_both_available(mock_is_torch_available, mock_is_tf_available, caplog):
    """Test that check_hf_library_installation logs availability of both PyTorch and TensorFlow."""
    mock_is_torch_available.return_value = True
    mock_is_tf_available.return_value = True

    check_hf_library_installation()
    assert "PyTorch is available." in caplog.text
    assert "TensorFlow is available." in caplog.text

def test_check_hf_library_installation_torch_unavailable(mock_is_torch_available, mock_is_tf_available, caplog):
    """Test that check_hf_library_installation logs warning when PyTorch is unavailable."""
    mock_is_torch_available.return_value = False
    mock_is_tf_available.return_value = True

    check_hf_library_installation()
    assert "PyTorch is not installed or available." in caplog.text
    assert "TensorFlow is available." in caplog.text

def test_check_hf_library_installation_tf_unavailable(mock_is_torch_available, mock_is_tf_available, caplog):
    """Test that check_hf_library_installation logs warning when TensorFlow is unavailable."""
    mock_is_torch_available.return_value = True
    mock_is_tf_available.return_value = False

    check_hf_library_installation()
    assert "PyTorch is available." in caplog.text
    assert "TensorFlow is not installed or available." in caplog.text

def test_check_hf_library_installation_both_unavailable(mock_is_torch_available, mock_is_tf_available, caplog):
    """Test that check_hf_library_installation logs warnings when both libraries are unavailable."""
    mock_is_torch_available.return_value = False
    mock_is_tf_available.return_value = False

    check_hf_library_installation()
    assert "PyTorch is not installed or available." in caplog.text
    assert "TensorFlow is not installed or available." in caplog.text

# 7. Comprehensive System Requirements Tests

def test_check_system_requirements_all_pass(mock_memory, mock_disk, mock_python_version, mock_cpu_count, 
                                           mock_platform_system, mock_platform_version, 
                                           mock_platform_architecture, mock_platform_processor, 
                                           mock_cuda_available, mock_cuda_device_name, 
                                           mock_is_torch_available, mock_is_tf_available, 
                                           caplog):
    """Test that check_system_requirements returns True when all checks pass."""
    # Setup mocks
    mock_memory.return_value.total = 16 * (1024 ** 3)  # 16 GB
    mock_memory.return_value.available = 8 * (1024 ** 3)  # 8 GB
    mock_disk.return_value.free = 100 * (1024 ** 3)  # 100 GB
    mock_python_version.return_value = ('3', '9', '5')
    
    result = check_system_requirements(min_memory_gb=8, min_disk_gb=50, min_python_version=(3, 8))
    assert result is True, "check_system_requirements should return True when all system checks pass."
    assert "Memory check passed: 16.00 GB available." in caplog.text
    assert "Disk space check passed: 100.00 GB available." in caplog.text
    assert "Python version check passed: 3.9.5" in caplog.text

def test_check_system_requirements_some_fail(mock_memory, mock_disk, mock_python_version, caplog):
    """Test that check_system_requirements returns False when some checks fail."""
    # Setup mocks
    mock_memory.return_value.total = 4 * (1024 ** 3)  # 4 GB
    mock_memory.return_value.available = 2 * (1024 ** 3)  # 2 GB
    mock_disk.return_value.free = 30 * (1024 ** 3)  # 30 GB
    mock_python_version.return_value = ('3', '7', '9')
    
    result = check_system_requirements(min_memory_gb=8, min_disk_gb=50, min_python_version=(3, 8))
    assert result is False, "check_system_requirements should return False when some system checks fail."
    assert "Insufficient memory: 4.00 GB available, 8 GB required." in caplog.text
    assert "Insufficient disk space: 30.00 GB available, 50 GB required." in caplog.text
    assert "Python 3.7 or higher is required" in caplog.text

def test_check_system_requirements_all_fail(mock_memory, mock_disk, mock_python_version, caplog):
    """Test that check_system_requirements returns False when all checks fail."""
    # Setup mocks
    mock_memory.return_value.total = 2 * (1024 ** 3)  # 2 GB
    mock_memory.return_value.available = 1 * (1024 ** 3)  # 1 GB
    mock_disk.return_value.free = 10 * (1024 ** 3)  # 10 GB
    mock_python_version.return_value = ('3', '6', '5')
    
    result = check_system_requirements(min_memory_gb=8, min_disk_gb=50, min_python_version=(3, 8))
    assert result is False, "check_system_requirements should return False when all system checks fail."
    assert "Insufficient memory: 2.00 GB available, 8 GB required." in caplog.text
    assert "Insufficient disk space: 10.00 GB available, 50 GB required." in caplog.text
    assert "Python 3.8 or higher is required" in caplog.text

# 8. Edge Case Tests

def test_check_memory_requirements_zero(mock_psutil_virtual_memory, caplog):
    """Test check_memory_requirements with zero memory requirement."""
    mock_psutil_virtual_memory.return_value.total = 4 * (1024 ** 3)  # 4 GB
    result = check_memory_requirements(min_memory_gb=0)
    assert result is True, "check_memory_requirements should return True when min_memory_gb is zero."
    assert "Memory check passed: 4.00 GB available." in caplog.text

def test_check_disk_space_zero(mock_shutil_disk_usage, caplog):
    """Test check_disk_space with zero disk space requirement."""
    mock_shutil_disk_usage.return_value.free = 10 * (1024 ** 3)  # 10 GB
    result = check_disk_space(min_disk_gb=0)
    assert result is True, "check_disk_space should return True when min_disk_gb is zero."
    assert "Disk space check passed: 10.00 GB available." in caplog.text

@pytest.mark.parametrize("min_memory_gb, min_disk_gb, min_python_version, expected", [
    (8, 50, (3, 8), True),
    (16, 100, (3, 9), True),
    (32, 200, (3, 10), False),
])
def test_check_system_requirements_various(min_memory_gb, min_disk_gb, min_python_version, expected,
                                          mock_psutil_virtual_memory, mock_shutil_disk_usage, 
                                          mock_platform_python_version_tuple, caplog):
    """Parameterized test for check_system_requirements with various inputs."""
    mock_psutil_virtual_memory.return_value.total = 16 * (1024 ** 3)  # 16 GB
    mock_psutil_virtual_memory.return_value.available = 8 * (1024 ** 3)  # 8 GB
    mock_shutil_disk_usage.return_value.free = 100 * (1024 ** 3)  # 100 GB
    mock_platform_python_version_tuple.return_value = ('3', '9', '5')
    
    if min_memory_gb > 16:
        mock_psutil_virtual_memory.return_value.total = 16 * (1024 ** 3)  # Insufficient
    if min_disk_gb > 100:
        mock_shutil_disk_usage.return_value.free = 100 * (1024 ** 3)  # Insufficient
    if min_python_version > (3, 9):
        mock_platform_python_version_tuple.return_value = ('3', '8', '0')  # Insufficient
    
    result = check_system_requirements(min_memory_gb, min_disk_gb, min_python_version)
    assert result == expected, f"check_system_requirements should return {expected} for inputs (memory: {min_memory_gb} GB, disk: {min_disk_gb} GB, Python: {min_python_version})."

def test_check_memory_requirements_non_integer(mock_psutil_virtual_memory, caplog):
    """Test check_memory_requirements with non-integer memory requirement."""
    with pytest.raises(TypeError):
        check_memory_requirements(min_memory_gb='eight')  # Invalid type

def test_check_disk_space_non_integer(mock_shutil_disk_usage, caplog):
    """Test check_disk_space with non-integer disk space requirement."""
    with pytest.raises(TypeError):
        check_disk_space(min_disk_gb='fifty')  # Invalid type

def test_check_python_version_malformed_tuple(mock_platform_python_version_tuple, caplog):
    """Test check_python_version with a malformed Python version tuple."""
    mock_platform_python_version_tuple.return_value = ('3',)  # Incomplete version
    with pytest.raises(ValueError):
        check_python_version(min_version=(3, 8))

# 9. Error Handling Tests

def test_check_memory_requirements_psutil_exception(caplog):
    """Test that check_memory_requirements handles exceptions from psutil.virtual_memory."""
    with patch('psutil.virtual_memory', side_effect=psutil.Error("Virtual memory error")):
        with pytest.raises(psutil.Error):
            check_memory_requirements(min_memory_gb=8)
    assert "Insufficient memory" not in caplog.text  # Since exception is raised before logging

def test_check_disk_space_shutil_exception(caplog):
    """Test that check_disk_space handles exceptions from shutil.disk_usage."""
    with patch('shutil.disk_usage', side_effect=OSError("Disk usage error")):
        with pytest.raises(OSError):
            check_disk_space(min_disk_gb=50)
    assert "Insufficient disk space" not in caplog.text  # Since exception is raised before logging

def test_check_python_version_platform_exception(caplog):
    """Test that check_python_version handles exceptions from platform.python_version_tuple."""
    with patch('platform.python_version_tuple', side_effect=AttributeError("Version tuple error")):
        with pytest.raises(AttributeError):
            check_python_version(min_version=(3, 8))
    assert "Python version check passed" not in caplog.text

def test_check_system_info_platform_exception(caplog):
    """Test that check_system_info handles exceptions from platform functions."""
    with patch('platform.system', side_effect=Exception("Platform system error")):
        with pytest.raises(Exception):
            check_system_info()
    assert "System Information:" not in caplog.text

# 10. Best Practices Implemented

def test_check_system_requirements_all_functions_called(mock_memory, mock_disk, mock_python_version, 
                                                        mock_platform_system, mock_platform_version, 
                                                        mock_platform_architecture, mock_platform_processor, 
                                                        mock_cpu_count, mock_cuda_available, mock_cuda_device_name, 
                                                        mock_is_torch_available, mock_is_tf_available):
    """Test that check_system_requirements calls all underlying functions."""
    with patch('src.utils.system_utils.check_memory_requirements') as mock_mem_req, \
         patch('src.utils.system_utils.check_disk_space') as mock_disk_space, \
         patch('src.utils.system_utils.check_python_version') as mock_py_version:
        mock_mem_req.return_value = True
        mock_disk_space.return_value = True
        mock_py_version.return_value = True
        
        result = check_system_requirements(min_memory_gb=8, min_disk_gb=50, min_python_version=(3, 8))
        assert result is True, "check_system_requirements should return True when all checks pass."
        mock_mem_req.assert_called_once_with(8)
        mock_disk_space.assert_called_once_with(50)
        mock_py_version.assert_called_once_with((3, 8))

# 11. Edge Case Tests Continued

def test_check_disk_space_non_standard_path(caplog):
    """Test check_disk_space with a non-standard disk path."""
    with patch('shutil.disk_usage', return_value=MagicMock(free=60 * (1024 ** 3))):
        with patch('shutil.disk_usage') as mock_du:
            result = check_disk_space(min_disk_gb=50)
            assert result is True, "check_disk_space should return True for sufficient disk space."
            mock_du.assert_called_once_with('/')
            assert "Disk space check passed: 60.00 GB available." in caplog.text

# 12. Integration Tests

def test_check_system_requirements_integration(mock_memory, mock_disk, mock_python_version, 
                                               mock_platform_system, mock_platform_version, 
                                               mock_platform_architecture, mock_platform_processor, 
                                               mock_cpu_count, mock_cuda_available, mock_cuda_device_name, 
                                               mock_is_torch_available, mock_is_tf_available, 
                                               caplog):
    """Integration test for check_system_requirements."""
    # Setup mocks for all checks
    mock_memory.return_value.total = 32 * (1024 ** 3)  # 32 GB
    mock_memory.return_value.available = 16 * (1024 ** 3)  # 16 GB
    mock_disk.return_value.free = 200 * (1024 ** 3)  # 200 GB
    mock_python_version.return_value = ('3', '10', '2')
    
    mock_is_torch_available.return_value = True
    mock_is_tf_available.return_value = False
    
    result = check_system_requirements(min_memory_gb=8, min_disk_gb=50, min_python_version=(3, 8))
    assert result is True, "check_system_requirements should return True when all system checks pass."
    
    # Check that memory, disk, and python checks were logged
    assert "Memory check passed: 32.00 GB available." in caplog.text
    assert "Disk space check passed: 200.00 GB available." in caplog.text
    assert "Python version check passed: 3.10.2" in caplog.text
    # Check that library installation was not called here

# 13. Edge Case: Non-standard Architecture

def test_check_system_info_non_standard_architecture(mock_platform_architecture, caplog):
    """Test check_system_info with non-standard architecture."""
    mock_platform_architecture.return_value = ('ARM64', 'ELF')
    system_info = check_system_info()
    assert system_info["Architecture"] == "ARM64", "Architecture should reflect the mocked architecture."
    assert "Architecture: ARM64" in caplog.text

# 14. Additional Integration Tests

def test_check_system_requirements_partial_failures(mock_memory, mock_disk, mock_python_version, caplog):
    """Test check_system_requirements returns False when one check fails."""
    # Setup mocks: memory and disk pass, python fails
    mock_memory.return_value.total = 16 * (1024 ** 3)  # 16 GB
    mock_memory.return_value.available = 8 * (1024 ** 3)  # 8 GB
    mock_disk.return_value.free = 100 * (1024 ** 3)  # 100 GB
    mock_python_version.return_value = ('3', '7', '9')  # Below required
    
    result = check_system_requirements(min_memory_gb=8, min_disk_gb=50, min_python_version=(3, 8))
    assert result is False, "check_system_requirements should return False when Python version is insufficient."
    assert "Python 3.8 or higher is required" in caplog.text

# 15. Edge Case: Very High Requirements

def test_check_system_requirements_very_high_requirements(mock_memory, mock_disk, mock_python_version, caplog):
    """Test check_system_requirements with very high memory and disk requirements."""
    # Setup mocks: insufficient memory and disk
    mock_memory.return_value.total = 8 * (1024 ** 3)  # 8 GB
    mock_memory.return_value.available = 4 * (1024 ** 3)  # 4 GB
    mock_disk.return_value.free = 20 * (1024 ** 3)  # 20 GB
    mock_python_version.return_value = ('3', '9', '1')
    
    result = check_system_requirements(min_memory_gb=16, min_disk_gb=100, min_python_version=(3, 8))
    assert result is False, "check_system_requirements should return False when memory and disk are insufficient."
    assert "Insufficient memory: 8.00 GB available, 16 GB required." in caplog.text
    assert "Insufficient disk space: 20.00 GB available, 100 GB required." in caplog.text

# 16. Edge Case: Non-Standard Python Version

def test_check_python_version_non_standard_version(mock_platform_python_version_tuple, caplog):
    """Test check_python_version with non-standard Python version."""
    mock_platform_python_version_tuple.return_value = ('3', '10', '0b4')  # Beta version
    result = check_python_version(min_version=(3, 8))
    assert result is True, "check_python_version should return True for pre-release versions if major and minor are sufficient."
    assert "Python version check passed: 3.10.0b4" in caplog.text

# 17. Edge Case: Non-integer Disk Usage

def test_check_disk_space_non_integer_disk_usage(caplog):
    """Test check_disk_space with non-integer disk usage values."""
    with patch('shutil.disk_usage', return_value=MagicMock(free=50.5 * (1024 ** 3))):
        result = check_disk_space(min_disk_gb=50)
        assert result is True, "check_disk_space should handle float disk space values correctly."
        assert "Disk space check passed: 50.50 GB available." in caplog.text

# 18. Edge Case: Floating Point Memory Requirements

def test_check_memory_requirements_floating_point(mock_psutil_virtual_memory, caplog):
    """Test check_memory_requirements with floating point memory requirement."""
    mock_psutil_virtual_memory.return_value.total = 8 * (1024 ** 3) + 512 * (1024 ** 2)  # 8.5 GB
    result = check_memory_requirements(min_memory_gb=8.5)
    assert result is True, "check_memory_requirements should handle floating point memory requirements correctly."
    assert "Memory check passed: 8.50 GB available." in caplog.text

# 19. Edge Case: Negative Memory/Disk Requirements

def test_check_memory_requirements_negative(mock_psutil_virtual_memory, caplog):
    """Test check_memory_requirements with negative memory requirement."""
    mock_psutil_virtual_memory.return_value.total = 16 * (1024 ** 3)  # 16 GB
    result = check_memory_requirements(min_memory_gb=-4)
    assert result is True, "check_memory_requirements should return True for negative memory requirements (treated as no requirement)."
    assert "Memory check passed: 16.00 GB available." in caplog.text

def test_check_disk_space_negative(mock_shutil_disk_usage, caplog):
    """Test check_disk_space with negative disk space requirement."""
    mock_shutil_disk_usage.return_value.free = 100 * (1024 ** 3)  # 100 GB
    result = check_disk_space(min_disk_gb=-10)
    assert result is True, "check_disk_space should return True for negative disk space requirements (treated as no requirement)."
    assert "Disk space check passed: 100.00 GB available." in caplog.text

# 20. Edge Case: Uncommon Operating Systems

@pytest.mark.parametrize("platform_system", ['Darwin', 'Windows', 'Linux'])
def test_check_system_info_various_platforms(platform_system, mock_platform_system, caplog):
    """Test check_system_info on various operating systems."""
    mock_platform_system.return_value = platform_system
    system_info = check_system_info()
    assert system_info["Platform"] == platform_system, "Platform should match the mocked platform."
    assert f"Platform: {platform_system}" in caplog.text

# 21. Mocking Exceptions in check_hf_library_installation

def test_check_hf_library_installation_exception(mock_is_torch_available, mock_is_tf_available, caplog):
    """Test that check_hf_library_installation handles exceptions gracefully."""
    mock_is_torch_available.side_effect = Exception("PyTorch check failed")
    mock_is_tf_available.side_effect = Exception("TensorFlow check failed")
    
    with pytest.raises(Exception, match="PyTorch check failed"):
        check_hf_library_installation()
    # TensorFlow check should not be reached due to the exception in PyTorch check

# 22. Edge Case: High Number of CPU Cores

def test_check_system_info_high_cpu_cores(mock_psutil_cpu_count, caplog):
    """Test check_system_info with a high number of CPU cores."""
    mock_psutil_cpu_count.side_effect = lambda logical=True: 128 if logical else 64
    system_info = check_system_info()
    assert system_info["CPU Cores"] == 64, "CPU Cores should reflect the mocked physical cores."
    assert system_info["Logical Cores"] == 128, "Logical Cores should reflect the mocked logical cores."
    assert "CPU Cores: 64" in caplog.text
    assert "Logical Cores: 128" in caplog.text

# 23. Edge Case: Non-Standard Processor Name

def test_check_system_info_non_standard_processor(mock_platform_processor, caplog):
    """Test check_system_info with a non-standard processor name."""
    mock_platform_processor.return_value = "ARM Cortex-A72"
    system_info = check_system_info()
    assert system_info["Processor"] == "ARM Cortex-A72", "Processor should reflect the mocked processor name."
    assert "Processor: ARM Cortex-A72" in caplog.text

# 24. Edge Case: Different Python Implementations

@pytest.mark.parametrize("python_version_tuple, expected", [
    (('3', '8', '10'), True),
    (('3', '8', '10'), True),
    (('3', '6', '9'), False),
    (('3', '10', '1'), True),
])
def test_check_python_version_various_implementations(mock_platform_python_version_tuple, python_version_tuple, expected, caplog):
    """Test check_python_version with various Python implementations."""
    mock_platform_python_version_tuple.return_value = python_version_tuple
    result = check_python_version(min_version=(3, 8))
    assert result == expected, f"check_python_version should return {expected} for version {python_version_tuple}."
    if expected:
        assert f"Python version check passed: {'.'.join(python_version_tuple)}" in caplog.text
    else:
        assert "Python 3.8 or higher is required" in caplog.text

# 25. Edge Case: Very High CPU Count

def test_check_system_info_very_high_cpu_count(mock_psutil_cpu_count, caplog):
    """Test check_system_info with a very high number of CPU cores."""
    mock_psutil_cpu_count.side_effect = lambda logical=True: 256 if logical else 128
    system_info = check_system_info()
    assert system_info["CPU Cores"] == 128, "CPU Cores should reflect the mocked physical cores."
    assert system_info["Logical Cores"] == 256, "Logical Cores should reflect the mocked logical cores."
    assert "CPU Cores: 128" in caplog.text
    assert "Logical Cores: 256" in caplog.text

# 26. Edge Case: Multiple Disk Partitions

def test_check_system_info_multiple_disk_partitions(mock_shutil_disk_usage, caplog):
    """Test check_system_info when there are multiple disk partitions."""
    # Assuming '/' is the primary partition
    mock_shutil_disk_usage.return_value.free = 150 * (1024 ** 3)  # 150 GB
    system_info = check_system_info()
    assert system_info["Disk Space (GB)"] == 150.0, "Disk space should reflect the free space of the primary partition."
    assert "Disk Space (GB): 150.00 GB available." in caplog.text

# 27. Integration Test: All Functions Together

def test_check_system_requirements_all_functions_together(mock_memory, mock_disk, mock_python_version, 
                                                           mock_platform_system, mock_platform_version, 
                                                           mock_platform_architecture, mock_platform_processor, 
                                                           mock_cpu_count, mock_cuda_available, mock_cuda_device_name, 
                                                           mock_is_torch_available, mock_is_tf_available, 
                                                           caplog):
    """Integration test ensuring all system checks work together."""
    # Setup mocks
    mock_memory.return_value.total = 32 * (1024 ** 3)  # 32 GB
    mock_memory.return_value.available = 16 * (1024 ** 3)  # 16 GB
    mock_disk.return_value.free = 200 * (1024 ** 3)  # 200 GB
    mock_python_version.return_value = ('3', '10', '2')
    
    mock_is_torch_available.return_value = True
    mock_is_tf_available.return_value = False
    
    result = check_system_requirements(min_memory_gb=16, min_disk_gb=100, min_python_version=(3, 8))
    assert result is True, "check_system_requirements should return True when all system checks pass."
    
    # Check logs for each individual check
    assert "Memory check passed: 32.00 GB available." in caplog.text
    assert "Disk space check passed: 200.00 GB available." in caplog.text
    assert "Python version check passed: 3.10.2" in caplog.text

# 28. Edge Case: Non-Standard Disk Path in check_disk_space

def test_check_disk_space_non_standard_path_disk_path(caplog):
    """Test check_disk_space with a non-standard disk path."""
    with patch('shutil.disk_usage', return_value=MagicMock(free=75 * (1024 ** 3))):
        with patch('shutil.disk_usage') as mock_du:
            result = check_disk_space(min_disk_gb=50)
            assert result is True, "check_disk_space should return True for sufficient disk space."
            mock_du.assert_called_once_with('/')
            assert "Disk space check passed: 75.00 GB available." in caplog.text

# 29. Edge Case: Non-Standard CPU Count Values

def test_check_system_info_non_standard_cpu_count(mock_psutil_cpu_count, caplog):
    """Test check_system_info with non-standard CPU core counts."""
    mock_psutil_cpu_count.side_effect = lambda logical=True: None  # Simulate unknown CPU count
    system_info = check_system_info()
    assert system_info["CPU Cores"] == 0, "CPU Cores should be 0 when psutil.cpu_count returns None."
    assert system_info["Logical Cores"] == 0, "Logical Cores should be 0 when psutil.cpu_count returns None."
    assert "CPU Cores: 0" in caplog.text
    assert "Logical Cores: 0" in caplog.text

# 30. Edge Case: Extremely Low Disk Space

def test_check_disk_space_extremely_low_space(mock_shutil_disk_usage, caplog):
    """Test check_disk_space with extremely low disk space."""
    mock_shutil_disk_usage.return_value.free = 0.5 * (1024 ** 3)  # 0.5 GB
    result = check_disk_space(min_disk_gb=1)
    assert result is False, "check_disk_space should return False when disk space is extremely low."
    assert "Insufficient disk space: 0.50 GB available, 1 GB required." in caplog.text

# 31. Edge Case: Extremely Low Memory

def test_check_memory_requirements_extremely_low_memory(mock_psutil_virtual_memory, caplog):
    """Test check_memory_requirements with extremely low memory."""
    mock_psutil_virtual_memory.return_value.total = 0.5 * (1024 ** 3)  # 0.5 GB
    mock_psutil_virtual_memory.return_value.available = 0.2 * (1024 ** 3)  # 0.2 GB
    result = check_memory_requirements(min_memory_gb=1)
    assert result is False, "check_memory_requirements should return False when memory is extremely low."
    assert "Insufficient memory: 0.50 GB available, 1 GB required." in caplog.text

# 32. Edge Case: Non-Standard Python Version Format

def test_check_python_version_non_standard_format(mock_platform_python_version_tuple, caplog):
    """Test check_python_version with non-standard Python version formats."""
    mock_platform_python_version_tuple.return_value = ('3', '8', '0a1')  # Alpha release
    result = check_python_version(min_version=(3, 8))
    assert result is True, "check_python_version should handle pre-release versions correctly if major and minor are sufficient."
    assert "Python version check passed: 3.8.0a1" in caplog.text

# 33. Edge Case: Missing Keys in check_system_info

def test_check_system_info_missing_keys(caplog):
    """Test check_system_info when certain system information keys are missing."""
    with patch('platform.processor', return_value=None), \
         patch('psutil.virtual_memory', return_value=MagicMock(total=16 * (1024 ** 3), available=8 * (1024 ** 3))):
        system_info = check_system_info()
        assert system_info["Processor"] is None, "Processor should be None when platform.processor() returns None."
        assert "Processor: None" in caplog.text

# 34. Edge Case: Non-Standard Free Disk Space Units

def test_check_disk_space_non_standard_units(caplog):
    """Test check_disk_space handles free disk space in non-standard units."""
    # Assuming the function always deals with GB, but testing if the mock returns different units
    # Since the function divides by (1024 ** 3), non-standard units are not directly applicable
    # However, testing with float values
    with patch('shutil.disk_usage', return_value=MagicMock(free=75.75 * (1024 ** 3))):
        result = check_disk_space(min_disk_gb=75)
        assert result is True, "check_disk_space should handle floating point disk space correctly."
        assert "Disk space check passed: 75.75 GB available." in caplog.text

# 35. Edge Case: Large Memory and Disk Requirements

def test_check_system_requirements_large_requirements(mock_memory, mock_disk, mock_python_version, caplog):
    """Test check_system_requirements with very large memory and disk requirements."""
    mock_memory.return_value.total = 64 * (1024 ** 3)  # 64 GB
    mock_memory.return_value.available = 32 * (1024 ** 3)  # 32 GB
    mock_disk.return_value.free = 500 * (1024 ** 3)  # 500 GB
    mock_python_version.return_value = ('3', '10', '0')
    
    result = check_system_requirements(min_memory_gb=32, min_disk_gb=500, min_python_version=(3, 8))
    assert result is True, "check_system_requirements should return True when large system requirements are met."
    assert "Memory check passed: 64.00 GB available." in caplog.text
    assert "Disk space check passed: 500.00 GB available." in caplog.text
    assert "Python version check passed: 3.10.0" in caplog.text

# 36. Edge Case: High Precision RAM and Disk Values

def test_check_system_info_high_precision_values(mock_psutil_virtual_memory, mock_shutil_disk_usage, caplog):
    """Test check_system_info with high precision RAM and disk space values."""
    mock_psutil_virtual_memory.return_value.total = 15.123456789 * (1024 ** 3)  # 15.123456789 GB
    mock_psutil_virtual_memory.return_value.available = 7.987654321 * (1024 ** 3)  # 7.987654321 GB
    mock_shutil_disk_usage.return_value.free = 99.999999999 * (1024 ** 3)  # 99.999999999 GB
    
    system_info = check_system_info()
    assert system_info["Total RAM (GB)"] == 15.12, "Total RAM should be rounded to two decimal places."
    assert system_info["Available RAM (GB)"] == 7.99, "Available RAM should be rounded to two decimal places."
    assert system_info["Disk Space (GB)"] == 100.0, "Disk Space should be rounded to two decimal places."
    assert "Total RAM (GB): 15.12" in caplog.text
    assert "Available RAM (GB): 7.99" in caplog.text
    assert "Disk Space (GB): 100.00 GB available." in caplog.text

# 37. Edge Case: Very Low RAM and Disk Space Requirements

def test_check_system_requirements_very_low_requirements(mock_memory, mock_disk, mock_python_version, caplog):
    """Test check_system_requirements with very low memory and disk space requirements."""
    mock_memory.return_value.total = 1 * (1024 ** 3)  # 1 GB
    mock_memory.return_value.available = 0.5 * (1024 ** 3)  # 0.5 GB
    mock_disk.return_value.free = 1 * (1024 ** 3)  # 1 GB
    mock_python_version.return_value = ('3', '8', '5')
    
    result = check_system_requirements(min_memory_gb=0.1, min_disk_gb=0.1, min_python_version=(3, 7))
    assert result is True, "check_system_requirements should return True when very low requirements are met."
    assert "Memory check passed: 1.00 GB available." in caplog.text
    assert "Disk space check passed: 1.00 GB available." in caplog.text
    assert "Python version check passed: 3.8.5" in caplog.text

# 38. Edge Case: Non-Standard Disk Path in check_system_info

def test_check_system_info_non_standard_disk_path(caplog):
    """Test check_system_info with a non-standard disk path for disk usage."""
    with patch('shutil.disk_usage', return_value=MagicMock(free=80 * (1024 ** 3))):
        system_info = check_system_info()
        assert system_info["Disk Space (GB)"] == 80.0, "Disk space should reflect the mocked free space."
        assert "Disk Space (GB): 80.00 GB available." in caplog.text

# 39. Edge Case: Non-Standard CPU Count in check_system_info

def test_check_system_info_non_standard_cpu_count_zero(mock_psutil_cpu_count, caplog):
    """Test check_system_info when psutil.cpu_count returns None."""
    mock_psutil_cpu_count.side_effect = lambda logical=True: None
    system_info = check_system_info()
    assert system_info["CPU Cores"] == 0, "CPU Cores should be 0 when psutil.cpu_count returns None."
    assert system_info["Logical Cores"] == 0, "Logical Cores should be 0 when psutil.cpu_count returns None."
    assert "CPU Cores: 0" in caplog.text
    assert "Logical Cores: 0" in caplog.text

# 40. Edge Case: Exception in check_hf_library_installation

def test_check_hf_library_installation_exception_handling(mock_is_torch_available, mock_is_tf_available, caplog):
    """Test that check_hf_library_installation handles exceptions gracefully."""
    mock_is_torch_available.side_effect = Exception("Unexpected error during PyTorch availability check")
    mock_is_tf_available.side_effect = Exception("Unexpected error during TensorFlow availability check")
    
    with pytest.raises(Exception, match="Unexpected error during PyTorch availability check"):
        check_hf_library_installation()
    # TensorFlow check is not reached due to the exception in PyTorch check
    assert "TensorFlow is not installed or available." not in caplog.text

