# src/models/zoo/detr.py

import logging
from typing import Optional, Any, Dict, Union, List

import torch
from torch import nn
from transformers import DetrForObjectDetection, DetrConfig, AutoImageProcessor

logger = logging.getLogger(__name__)


class DETRModel(nn.Module):
    """
    DETR (Detection Transformer) Model for Object Detection.

    This class wraps HuggingFace's DetrForObjectDetection model and provides utility methods.
    """

    def __init__(
        self,
        model_name_or_path: str = "facebook/detr-resnet-50",
        num_labels: int = 91,
        label2id: Optional[Dict[str, int]] = None,
        id2label: Optional[Dict[int, str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        """
        Initializes the DETR model.

        Args:
            model_name_or_path (str): Path to the pretrained model or model identifier from HuggingFace.
            num_labels (int): Number of labels/classes for the task.
            label2id (Dict[str, int], optional): Mapping from label names to IDs.
            id2label (Dict[int, str], optional): Mapping from IDs to label names.
            device (Union[str, torch.device], optional): Device to run the model on.
            **kwargs: Additional keyword arguments for configuration.
        """
        super().__init__()
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))

        # Load configuration
        self.config = DetrConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            **kwargs,
        )
        logger.info(f"Loaded configuration for '{model_name_or_path}'.")

        # Load model
        self.model = DetrForObjectDetection.from_pretrained(
            model_name_or_path,
            config=self.config,
            **kwargs,
        )
        self.model.to(self.device)
        logger.info(f"Loaded model '{model_name_or_path}' to device '{self.device}'.")

        # Load processor
        self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        logger.info(f"Loaded processor for '{model_name_or_path}'.")

    def forward(
        self,
        images: List[Union[str, Any]],
        annotations: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Forward pass through the model.

        Args:
            images (List[Union[str, Any]]): List of input images.
            annotations (List[Dict[str, Any]], optional): Annotations for training.

        Returns:
            Model outputs.
        """
        # Prepare inputs
        inputs = self.processor(images=images, annotations=annotations, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs

    def save_pretrained(self, save_directory: str):
        """
        Save the model and processor to the specified directory.

        Args:
            save_directory (str): Directory to save the model and processor.
        """
        self.model.save_pretrained(save_directory)
        self.processor.save_pretrained(save_directory)
        logger.info(f"Model and processor saved to '{save_directory}'.")

    @classmethod
    def from_pretrained(cls, load_directory: str, device: Optional[Union[str, torch.device]] = None):
        """
        Load the model and processor from the specified directory.

        Args:
            load_directory (str): Directory from which to load the model and processor.
            device (Union[str, torch.device], optional): Device to run the model on.

        Returns:
            DETRModel instance.
        """
        device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        model = DetrForObjectDetection.from_pretrained(load_directory)
        model.to(device)
        processor = AutoImageProcessor.from_pretrained(load_directory)
        instance = cls.__new__(cls)
        instance.model = model
        instance.processor = processor
        instance.device = device
        logger.info(f"Loaded model and processor from '{load_directory}' to device '{device}'.")
        return instance

    def to(self, device: Union[str, torch.device]):
        """
        Move the model to the specified device.

        Args:
            device (Union[str, torch.device]): The device to move the model to.
        """
        self.device = torch.device(device)
        self.model.to(self.device)
        logger.info(f"Model moved to device '{self.device}'.")

    def eval(self):
        """
        Set the model to evaluation mode.
        """
        self.model.eval()
        logger.info("Model set to evaluation mode.")

    def train(self):
        """
        Set the model to training mode.
        """
        self.model.train()
        logger.info("Model set to training mode.")
