# src/models/model_registry.py

import os
import torch
from transformers import DetrForObjectDetection
from src.models.model_builder import build_detr_model

class ModelRegistry:
    def __init__(self, registry_dir="model_registry"):
        """
        Initialize the model registry.
        
        Args:
            registry_dir (str): Directory to store registered models.
        """
        self.registry_dir = registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)
    
    def save_model(self, model: DetrForObjectDetection, version: str):
        """
        Save a model to the registry.
        
        Args:
            model (DetrForObjectDetection): The model to be saved.
            version (str): Version identifier for the model.
        """
        model_path = os.path.join(self.registry_dir, f"detr_model_v{version}.bin")
        model.save_pretrained(model_path)
        print(f"Model version {version} saved at {model_path}")
    
    def load_model(self, version: str, num_labels: int = 91, device: str = "cuda"):
        """
        Load a model from the registry.
        
        Args:
            version (str): Version identifier for the model to be loaded.
            num_labels (int): Number of object classes (default: 91).
            device (str): Device to load the model onto (default: "cuda").
        
        Returns:
            DetrForObjectDetection: Loaded model.
        """
        model_path = os.path.join(self.registry_dir, f"detr_model_v{version}.bin")
        model = build_detr_model(num_labels=num_labels, pretrained=False)
        model.from_pretrained(model_path)
        model.to(device)
        print(f"Model version {version} loaded from {model_path}")
        return model

    def list_available_models(self):
        """
        List all saved model versions in the registry.

        Returns:
            list: List of available model versions.
        """
        return [f.split("_v")[-1].split(".")[0] for f in os.listdir(self.registry_dir) if f.endswith(".bin")]

    def delete_model(self, version: str):
        """
        Delete a model from the registry.

        Args:
            version (str): Version identifier of the model to delete.
        """
        model_path = os.path.join(self.registry_dir, f"detr_model_v{version}.bin")
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"Model version {version} deleted.")
        else:
            print(f"Model version {version} not found.")
