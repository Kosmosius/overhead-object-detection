# src/models/zoo/base_model.py

import logging
from typing import Optional, Any, Dict, Union

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForObjectDetection,
    AutoModelForImageClassification,
    AutoModelForSemanticSegmentation,
    AutoImageProcessor,
    AutoFeatureExtractor,
    PreTrainedModel,
)

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    """
    A base class for models, providing common utilities and leveraging HuggingFace Transformers to the maximum extent.
    """

    def __init__(
        self,
        model_name_or_path: str,
        task_type: str,
        num_labels: Optional[int] = None,
        label2id: Optional[Dict[str, int]] = None,
        id2label: Optional[Dict[int, str]] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initializes the BaseModel with a HuggingFace PreTrainedModel.

        Args:
            model_name_or_path (str): Path to the pretrained model or model identifier from HuggingFace.
            task_type (str): Type of task ('object_detection', 'image_classification', 'semantic_segmentation').
            num_labels (int, optional): Number of labels/classes for the task.
            label2id (Dict[str, int], optional): Mapping from label names to IDs.
            id2label (Dict[int, str], optional): Mapping from IDs to label names.
            config_kwargs (Dict[str, Any], optional): Additional keyword arguments for configuration.
            model_kwargs (Dict[str, Any], optional): Additional keyword arguments for model instantiation.
            device (Union[str, torch.device], optional): Device to run the model on.
        """
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_type = task_type
        self.num_labels = num_labels
        self.label2id = label2id
        self.id2label = id2label
        self.config_kwargs = config_kwargs or {}
        self.model_kwargs = model_kwargs or {}
        self.device = torch.device(device if device else 'cpu')

        # Load configuration
        self.config = AutoConfig.from_pretrained(
            self.model_name_or_path,
            num_labels=self.num_labels,
            label2id=self.label2id,
            id2label=self.id2label,
            **self.config_kwargs,
        )
        logger.info(f"Configuration loaded for model '{self.model_name_or_path}'.")

        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        logger.info(f"Model '{self.model_name_or_path}' loaded and moved to device '{self.device}'.")

        # Load processor
        self.processor = self._load_processor()

    def _load_model(self) -> PreTrainedModel:
        """
        Loads the appropriate model based on the task type.

        Returns:
            PreTrainedModel: The loaded model.
        """
        try:
            if self.task_type == 'object_detection':
                model = AutoModelForObjectDetection.from_pretrained(
                    self.model_name_or_path,
                    config=self.config,
                    **self.model_kwargs,
                )
                logger.info(f"Loaded AutoModelForObjectDetection for '{self.model_name_or_path}'.")
            elif self.task_type == 'image_classification':
                model = AutoModelForImageClassification.from_pretrained(
                    self.model_name_or_path,
                    config=self.config,
                    **self.model_kwargs,
                )
                logger.info(f"Loaded AutoModelForImageClassification for '{self.model_name_or_path}'.")
            elif self.task_type == 'semantic_segmentation':
                model = AutoModelForSemanticSegmentation.from_pretrained(
                    self.model_name_or_path,
                    config=self.config,
                    **self.model_kwargs,
                )
                logger.info(f"Loaded AutoModelForSemanticSegmentation for '{self.model_name_or_path}'.")
            else:
                logger.error(f"Unsupported task type '{self.task_type}'.")
                raise ValueError(f"Unsupported task type '{self.task_type}'.")
            return model
        except Exception as e:
            logger.error(f"Failed to load model '{self.model_name_or_path}': {e}")
            raise

    def _load_processor(self):
        """
        Loads the appropriate image processor or feature extractor.

        Returns:
            AutoImageProcessor or AutoFeatureExtractor: The loaded processor.
        """
        try:
            processor = AutoImageProcessor.from_pretrained(self.model_name_or_path)
            logger.info(f"Loaded AutoImageProcessor for '{self.model_name_or_path}'.")
        except Exception as e_image_processor:
            logger.debug(f"AutoImageProcessor not found: {e_image_processor}")
            try:
                processor = AutoFeatureExtractor.from_pretrained(self.model_name_or_path)
                logger.info(f"Loaded AutoFeatureExtractor for '{self.model_name_or_path}'.")
            except Exception as e_feature_extractor:
                logger.error(f"Failed to create processor: {e_feature_extractor}")
                raise RuntimeError(
                    f"Could not load processor for '{self.model_name_or_path}'. "
                    f"Errors: {e_image_processor}, {e_feature_extractor}"
                )
        return processor

    def forward(self, *args, **kwargs):
        """
        Forward pass of the model.

        Args:
            *args: Positional arguments for the model's forward method.
            **kwargs: Keyword arguments for the model's forward method.

        Returns:
            Any: The output of the model's forward method.
        """
        return self.model(*args, **kwargs)

    def save(self, save_directory: str):
        """
        Saves the model and processor to the specified directory.

        Args:
            save_directory (str): Directory to save the model and processor.
        """
        self.model.save_pretrained(save_directory)
        self.processor.save_pretrained(save_directory)
        logger.info(f"Model and processor saved to {save_directory}.")

    @classmethod
    def load(cls, load_directory: str, task_type: str, device: Optional[Union[str, torch.device]] = None):
        """
        Loads the model and processor from the specified directory.

        Args:
            load_directory (str): Directory from which to load the model and processor.
            task_type (str): Type of task ('object_detection', 'image_classification', 'semantic_segmentation').
            device (Union[str, torch.device], optional): Device to run the model on.

        Returns:
            BaseModel: An instance of the model.
        """
        return cls(
            model_name_or_path=load_directory,
            task_type=task_type,
            device=device,
        )

    def to(self, device: Union[str, torch.device]):
        """
        Moves the model to the specified device.

        Args:
            device (Union[str, torch.device]): The device to move the model to.
        """
        self.device = torch.device(device)
        self.model.to(self.device)
        logger.info(f"Model moved to device '{self.device}'.")

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Returns the model's parameters.

        Returns:
            Dict[str, torch.Tensor]: The model's parameters.
        """
        return self.model.state_dict()

    def load_parameters(self, state_dict: Dict[str, torch.Tensor]):
        """
        Loads parameters into the model.

        Args:
            state_dict (Dict[str, torch.Tensor]): The state dictionary to load.
        """
        self.model.load_state_dict(state_dict)
        logger.info("Model parameters loaded.")

    def freeze_backbone(self):
        """
        Freezes the backbone parameters to prevent them from being updated during training.
        """
        for name, param in self.model.named_parameters():
            if 'backbone' in name or 'body' in name:
                param.requires_grad = False
        logger.info("Backbone parameters have been frozen.")

    def unfreeze_backbone(self):
        """
        Unfreezes the backbone parameters to allow them to be updated during training.
        """
        for name, param in self.model.named_parameters():
            if 'backbone' in name or 'body' in name:
                param.requires_grad = True
        logger.info("Backbone parameters have been unfrozen.")
