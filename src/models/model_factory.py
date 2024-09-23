# src/models/model_factory.py

import logging
import os
import json
from typing import Type, Dict, Any, Optional
from abc import ABC, abstractmethod
from transformers import (
    DetrForObjectDetection,
    DetrConfig,
    AutoModelForObjectDetection,
    AutoConfig,
    PreTrainedModel,
)
import torch

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for object detection models.
    """

    def __init__(self, model_name: str, num_labels: int, **kwargs):
        """
        Initialize the model with the given parameters.

        Args:
            model_name (str): Name of the pre-trained model.
            num_labels (int): Number of object classes.
            **kwargs: Additional keyword arguments.
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = self._build_model(**kwargs)

    @abstractmethod
    def _build_model(self, **kwargs) -> PreTrainedModel:
        """
        Build and return the model instance.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            PreTrainedModel: The initialized model.
        """
        pass

    def save(self, save_directory: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save the model to the specified directory, along with optional metadata.

        Args:
            save_directory (str): Directory to save the model.
            metadata (Dict[str, Any], optional): Additional metadata to save with the model.
        """
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)
        logger.info(f"Model saved to {save_directory}")

        if metadata:
            metadata_path = os.path.join(save_directory, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Metadata saved to {metadata_path}")

    @classmethod
    def load(cls, load_directory: str) -> 'BaseModel':
        """
        Load the model from the specified directory, along with metadata if available.
    
        Args:
            load_directory (str): Directory to load the model from.
    
        Returns:
            BaseModel: An instance of the loaded model.
        """
        try:
            config = AutoConfig.from_pretrained(load_directory)
            model_class = MODEL_REGISTRY.get(config.model_type)
            if not model_class:
                raise ValueError(f"Model type '{config.model_type}' is not registered.")
    
            model = model_class(
                model_name=load_directory,
                num_labels=config.num_labels
            )
            logger.info(f"Model loaded from {load_directory}")
    
            metadata_path = os.path.join(load_directory, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Metadata loaded from {metadata_path}")
                model.metadata = metadata
    
            return model
        except Exception as e:
            logger.error(f"Error loading model from {load_directory}: {e}")
            raise Exception(f"Error loading model from {load_directory}: {e}")


    def freeze_backbone(self) -> None:
        """
        Freeze the backbone parameters to prevent them from being updated during training.
        """
        for name, param in self.model.named_parameters():
            if any(backbone_layer in name for backbone_layer in ['backbone', 'body']):
                param.requires_grad = False
        logger.info("Backbone parameters have been frozen.")

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze the backbone parameters to allow them to be updated during training.
        """
        for name, param in self.model.named_parameters():
            if any(backbone_layer in name for backbone_layer in ['backbone', 'body']):
                param.requires_grad = True
        logger.info("Backbone parameters have been unfrozen.")

    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get the state dictionary of the model.

        Returns:
            Dict[str, Any]: The state dictionary.
        """
        return self.model.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load a state dictionary into the model.

        Args:
            state_dict (Dict[str, Any]): The state dictionary to load.
        """
        self.model.load_state_dict(state_dict)
        logger.info("Model state dictionary loaded.")

# Registry for available models
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}

def register_model(name: str):
    """
    Decorator to register a new model class in the MODEL_REGISTRY.

    Args:
        name (str): Name of the model.
    """
    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

@register_model('detr')
class DetrModel(BaseModel):
    """
    DETR (Detection Transformer) model implementation.
    """

    def _build_model(self, **kwargs) -> DetrForObjectDetection:
        """
        Build and return a DetrForObjectDetection model.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            DetrForObjectDetection: The initialized DETR model.
        """
        try:
            config = DetrConfig.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                **kwargs
            )
            model = DetrForObjectDetection.from_pretrained(
                self.model_name,
                config=config
            )
            logger.info(f"DETR model initialized with {self.num_labels} labels.")
            return model
        except Exception as e:
            logger.error(f"Error initializing DETR model: {e}")
            raise

# Additional models can be registered here using the @register_model decorator.

class ModelVersioning:
    """
    Class to handle model versioning, saving, loading, and registry.
    """

    def __init__(self, model_dir: str = 'models'):
        """
        Initialize the ModelVersioning with a directory to store models.

        Args:
            model_dir (str): Directory to store model versions.
        """
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.registry_file = os.path.join(self.model_dir, 'model_registry.json')
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """
        Load the model registry from a JSON file.

        Returns:
            Dict[str, Any]: The model registry.
        """
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                registry = json.load(f)
            logger.info("Model registry loaded.")
        else:
            registry = {}
            logger.info("Model registry initialized.")
        return registry

    def _save_registry(self) -> None:
        """
        Save the model registry to a JSON file.
        """
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=4)
        logger.info("Model registry saved.")

    def register_model(
        self,
        model_name: str,
        model_instance: BaseModel,
        version: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a new model version.

        Args:
            model_name (str): Name of the model type.
            model_instance (BaseModel): The model instance to save.
            version (str): Version identifier (e.g., semantic versioning).
            metadata (Dict[str, Any], optional): Additional metadata.
        """
        model_path = os.path.join(self.model_dir, f"{model_name}_{version}")
        model_instance.save(model_path, metadata=metadata)

        # Update registry
        self.registry.setdefault(model_name, {})
        self.registry[model_name][version] = {
            'path': model_path,
            'metadata': metadata or {}
        }
        self._save_registry()
        logger.info(f"Model '{model_name}' version '{version}' registered.")

    def load_model(self, model_name: str, version: str) -> BaseModel:
        """
        Load a specific version of a model.

        Args:
            model_name (str): Name of the model type.
            version (str): Version identifier.

        Returns:
            BaseModel: An instance of the loaded model.
        """
        model_info = self.registry.get(model_name, {}).get(version)
        if not model_info:
            raise ValueError(f"Model '{model_name}' version '{version}' not found in registry.")

        model_path = model_info['path']
        model_class = MODEL_REGISTRY.get(model_name)
        if not model_class:
            raise ValueError(f"Model type '{model_name}' is not registered.")

        model_instance = model_class.load(model_path)
        logger.info(f"Model '{model_name}' version '{version}' loaded.")
        return model_instance

    def delete_model(self, model_name: str, version: str) -> None:
        """
        Delete a specific version of a model.

        Args:
            model_name (str): Name of the model type.
            version (str): Version identifier.
        """
        model_info = self.registry.get(model_name, {}).pop(version, None)
        if not model_info:
            raise ValueError(f"Model '{model_name}' version '{version}' not found in registry.")

        # Delete model files
        model_path = model_info['path']
        if os.path.exists(model_path):
            import shutil
            shutil.rmtree(model_path)
            logger.info(f"Model files at '{model_path}' deleted.")

        self._save_registry()
        logger.info(f"Model '{model_name}' version '{version}' deleted from registry.")

    def list_models(self) -> Dict[str, Any]:
        """
        List all registered models and their versions.

        Returns:
            Dict[str, Any]: Dictionary of models and versions.
        """
        return self.registry

class ModelFactory:
    """
    Factory class to build, manage, and version models.
    """

    @staticmethod
    def create_model(
        model_type: str,
        model_name: str,
        num_labels: int,
        **kwargs
    ) -> BaseModel:
        """
        Create a model of the specified type.

        Args:
            model_type (str): Type of the model to create.
            model_name (str): Name of the pre-trained model.
            num_labels (int): Number of object classes.
            **kwargs: Additional keyword arguments.

        Returns:
            BaseModel: An instance of a model subclass.
        """
        model_class = MODEL_REGISTRY.get(model_type)
        if not model_class:
            raise ValueError(f"Model type '{model_type}' is not registered.")
        model = model_class(model_name=model_name, num_labels=num_labels, **kwargs)
        return model

    @staticmethod
    def get_available_models() -> Dict[str, Type[BaseModel]]:
        """
        Get a list of available registered model types.

        Returns:
            Dict[str, Type[BaseModel]]: Dictionary of registered model types.
        """
        return MODEL_REGISTRY

