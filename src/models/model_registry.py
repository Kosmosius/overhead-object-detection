# src/models/model_registry.py

import os
import logging
from transformers import AutoModelForObjectDetection
from pathlib import Path


class ModelRegistryError(Exception):
    """Custom exception class for ModelRegistry-related errors."""
    pass


class ModelRegistry:
    """
    Model Registry to save, load, and manage multiple versions of HuggingFace object detection models.
    """

    def __init__(self, registry_dir="model_registry"):
        """
        Initialize the model registry.

        Args:
            registry_dir (str): Directory to store registered models.
        """
        self.registry_dir = Path(registry_dir)
        self._ensure_directory_exists()

    def _ensure_directory_exists(self):
        """Ensure that the registry directory exists, creating it if necessary."""
        try:
            self.registry_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logging.error(f"Failed to create model registry directory: {e}")
            raise ModelRegistryError(f"Failed to create model registry directory: {e}")

    def _get_model_path(self, version: str) -> Path:
        """
        Get the file path for a specific model version.

        Args:
            version (str): Version identifier for the model.

        Returns:
            Path: Path to the model directory.
        """
        return self.registry_dir / f"model_v{version}"

    def save_model(self, model, version: str):
        """
        Save a HuggingFace model to the registry.

        Args:
            model: HuggingFace object detection model to save.
            version (str): Version identifier for the model.

        Raises:
            ModelRegistryError: If the model cannot be saved.
        """
        model_path = self._get_model_path(version)
        try:
            model.save_pretrained(model_path)
            logging.info(f"Model version {version} saved at {model_path}")
        except Exception as e:
            logging.error(f"Error saving model version {version}: {e}")
            raise ModelRegistryError(f"Failed to save model version {version}: {e}")

    def load_model(self, model_name: str, version: str, device: str = "cuda"):
        """
        Load a HuggingFace model from the registry.

        Args:
            model_name (str): HuggingFace model name or path (e.g., "facebook/detr-resnet-50").
            version (str): Version identifier for the model.
            device (str): Device to load the model onto.

        Returns:
            model: Loaded HuggingFace model.

        Raises:
            ModelRegistryError: If the model cannot be loaded.
        """
        model_path = self._get_model_path(version)
        if not model_path.exists():
            logging.error(f"Model version {version} not found in {model_path}")
            raise ModelRegistryError(f"Model version {version} not found")

        try:
            model = AutoModelForObjectDetection.from_pretrained(model_path)
            model.to(device)
            logging.info(f"Model version {version} loaded from {model_path}")
            return model
        except Exception as e:
            logging.error(f"Error loading model version {version}: {e}")
            raise ModelRegistryError(f"Failed to load model version {version}: {e}")

    def list_available_models(self):
        """
        List all available model versions in the registry.

        Returns:
            list: List of available model versions.
        """
        try:
            models = [
                f.name.split("_v")[-1] for f in self.registry_dir.iterdir()
                if f.is_dir() and f.name.startswith("model_v")
            ]
            logging.info(f"Available models: {models}")
            return models
        except Exception as e:
            logging.error(f"Error listing available models: {e}")
            raise ModelRegistryError(f"Failed to list available models: {e}")

    def delete_model(self, version: str):
        """
        Delete a model version from the registry.

        Args:
            version (str): Version identifier for the model to delete.

        Raises:
            ModelRegistryError: If the model cannot be deleted.
        """
        model_path = self._get_model_path(version)
        if not model_path.exists():
            logging.error(f"Model version {version} not found for deletion.")
            raise ModelRegistryError(f"Model version {version} not found for deletion")

        try:
            for item in model_path.iterdir():
                if item.is_dir():
                    item.rmdir()
                else:
                    item.unlink()
            model_path.rmdir()
            logging.info(f"Model version {version} deleted.")
        except Exception as e:
            logging.error(f"Error deleting model version {version}: {e}")
            raise ModelRegistryError(f"Failed to delete model version {version}: {e}")
