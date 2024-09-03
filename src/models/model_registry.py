# src/models/model_registry.py

import os
import torch
from src.models.foundation_model import DetrObjectDetectionModel

class ModelRegistry:
    def __init__(self, registry_dir="model_registry"):
        """
        Initialize the model registry.
        
        Args:
            registry_dir (str): Directory to store registered models.
        """
        self.registry_dir = registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)
    
    def save_model(self, model, version):
        """
        Save a model to the registry.
        
        Args:
            model (DetrObjectDetectionModel): The model to be saved.
            version (str): Version identifier for the model.
        """
        model_path = os.path.join(self.registry_dir, f"detr_model_v{version}.pt")
        model.save(model_path)
        print(f"Model version {version} saved at {model_path}")
    
    def load_model(self, version):
        """
        Load a model from the registry.
        
        Args:
            version (str): Version identifier for the model to be loaded.
        
        Returns:
            DetrObjectDetectionModel: Loaded model.
        """
        model_path = os.path.join(self.registry_dir, f"detr_model_v{version}.pt")
        model = DetrObjectDetectionModel(num_classes=91)
        model.load(model_path)
        print(f"Model version {version} loaded from {model_path}")
        return model
