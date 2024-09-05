# scripts/setup_environment.py

import os
import subprocess
import sys
from transformers import is_torch_available, is_tf_available
from src.utils.system_utils import check_python_version, check_memory_requirements, check_disk_space, check_hf_library_installation

def install_requirements(requirements_file="requirements.txt"):
    """
    Install packages from a requirements file using pip.
    
    Args:
        requirements_file (str): Path to the requirements.txt file.
    """
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print(f"Successfully installed packages from {requirements_file}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install packages: {e}")
        sys.exit(1)

def setup_huggingface_cache(cache_dir=".cache/huggingface"):
    """
    Set up HuggingFace's cache directory.
    
    Args:
        cache_dir (str): Path to the cache directory for HuggingFace.
    """
    os.environ["HF_HOME"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    print(f"HuggingFace cache directory set at {cache_dir}")

def install_nvidia_drivers():
    """
    Install NVIDIA drivers for GPU usage (if necessary).
    """
    if sys.platform == "linux":
        try:
            subprocess.run(["sudo", "apt-get", "install", "-y", "nvidia-driver-470"], check=True)
            print("NVIDIA drivers installed.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install NVIDIA drivers: {e}")
            sys.exit(1)
    else:
        print("NVIDIA driver installation is supported only on Linux. Please manually install on other platforms.")

def configure_environment():
    """
    Set up the environment by installing necessary libraries, checking system requirements,
    and configuring HuggingFace cache.
    """
    print("Checking Python version...")
    if not check_python_version(min_version=(3, 8)):
        sys.exit("Python 3.8 or higher is required. Please upgrade your Python version.")
    
    print("Checking memory requirements...")
    if not check_memory_requirements(min_memory_gb=16):
        sys.exit("Insufficient memory for this project. Please ensure you have at least 16 GB of RAM.")
    
    print("Checking disk space requirements...")
    if not check_disk_space(min_disk_gb=20):
        sys.exit("Insufficient disk space. Please ensure you have at least 20 GB of free space.")
    
    print("Checking for HuggingFace library installations...")
    check_hf_library_installation()

    print("Installing required Python packages...")
    install_requirements()

    print("Setting up HuggingFace cache directory...")
    setup_huggingface_cache()

    print("Environment setup complete!")

if __name__ == "__main__":
    configure_environment()
