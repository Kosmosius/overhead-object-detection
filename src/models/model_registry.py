# src/models/model_registry.py

import os
import torch
from transformers import AutoModelForObjectDetection, PretrainedConfig

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
        self.registry_dir = registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)

    def save_model(self, model, version: str):
        """
        Save a HuggingFace model to the registry.

        Args:
            model: HuggingFace object detection model to save.
            version (str): Version identifier for the model.
        """
        model_path = os.path.join(self.registry_dir, f"model_v{version}")
        model.save_pretrained(model_path)
        print(f"Model version {version} saved at {model_path}")

    def load_model(self, model_name: str, version: str, device: str = "cuda"):
        """
        Load a HuggingFace model from the registry.

        Args:
            model_name (str): HuggingFace model name or path (e.g., "facebook/detr-resnet-50").
            version (str): Version identifier for the model.
            device (str): Device to load the model onto.

        Returns:
            model: Loaded HuggingFace model.
        """
        model_path = os.path.join(self.registry_dir, f"model_v{version}")
        model = AutoModelForObjectDetection.from_pretrained(model_path)
        model.to(device)
        print(f"Model version {version} loaded from {model_path}")
        return model

    def list_available_models(self):
        """
        List all available model versions in the registry.

        Returns:
            list: List of available model versions.
        """
        return [f.split("_v")[-1] for f in os.listdir(self.registry_dir) if os.path.isdir(os.path.join(self.registry_dir, f))]

    def delete_model(self, version: str):
        """
        Delete a model version from the registry.

        Args:
            version (str): Version identifier for the model to delete.
        """
        model_path = os.path.join(self.registry_dir, f"model_v{version}")
        if os.path.exists(model_path):
            os.rmdir(model_path)
            print(f"Model version {version} deleted.")
        else:
            print(f"Model version {version} not found.")

