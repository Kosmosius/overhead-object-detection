# src/models/zoo/pvtv2.py

import os
import logging
from typing import Optional, List, Dict, Any, Union, Tuple, Type

import torch
from torch import nn
from transformers import (
    PvtV2Model,
    PvtV2Config,
    AutoImageProcessor,
    PreTrainedModel,
)
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class PVTv2Model(BaseModel):
    """
    PVTv2 Backbone Model for Object Detection.
    
    This class integrates the PVTv2 transformer as a backbone for object detection models.
    It inherits from the abstract BaseModel, implementing the required methods for training,
    evaluation, and inference.
    
    Args:
        model_name_or_path (str): Path or identifier of the pretrained PVTv2 model.
        pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
        linear_attention (bool, optional): Whether to use linear attention. Defaults to False.
        output_features (List[str], optional): List of feature map names to output. Defaults to ['stage1', 'stage2', 'stage3', 'stage4'].
        device (Union[str, torch.device], optional): Device to load the model on. Defaults to 'cpu'.
        num_labels (int, optional): Number of labels for the task. Relevant for classification tasks.
        **kwargs: Additional keyword arguments for model configuration.
    """

    def __init__(
        self,
        model_name_or_path: str = "OpenGVLab/pvt_v2_b0",
        pretrained: bool = True,
        linear_attention: bool = False,
        output_features: Optional[List[str]] = None,
        device: Union[str, torch.device] = "cpu",
        num_labels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            model_class=PvtV2Model,
            config=kwargs.get('config', None),
            num_labels=num_labels,
            **kwargs,
        )
        
        # Update configuration for linear attention if specified
        self.config.linear_attention = linear_attention

        # Load the PVTv2 model with or without pretrained weights
        try:
            self.model = PvtV2Model.from_pretrained(
                self.model_name_or_path,
                config=self.config,
                local_files_only=not pretrained
            )
            logger.info(f"PVTv2Model loaded from {self.model_name_or_path} with pretrained={pretrained}")
        except Exception as e:
            logger.error(f"Failed to load PVTv2Model: {e}")
            raise

        # Move model to the specified device
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode by default

        # Define output features
        if output_features is None:
            self.output_features = ['stage1', 'stage2', 'stage3', 'stage4']
        else:
            self.output_features = output_features
        logger.debug(f"Output features set to: {self.output_features}")

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the PVTv2 backbone.
        
        Args:
            pixel_values (torch.Tensor): Input images of shape (batch_size, num_channels, height, width).
        
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the output feature maps.
        """
        self.model.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                output_hidden_states=True
            )
        
        # Extract the hidden states corresponding to the output features
        feature_maps = {}
        # Assuming hidden_states[0] is the embeddings, hidden_states[1] to hidden_states[4] correspond to stages 1-4
        for idx, feature in enumerate(self.output_features, 1):
            try:
                feature_maps[feature] = outputs.hidden_states[idx].to(self.device)
                logger.debug(f"Extracted feature map '{feature}' with shape {outputs.hidden_states[idx].shape}")
            except IndexError:
                logger.error(f"Feature index {idx} out of range for hidden_states")
                raise
        
        return feature_maps

    def compute_loss(self, outputs: Any, targets: Any) -> torch.Tensor:
        """
        Computes the loss given model outputs and targets.
        
        As PVTv2 serves as a backbone, loss computation is task-specific and should be implemented
        by subclasses that integrate PVTv2 with task-specific heads.
        
        Args:
            outputs (Any): Outputs from the model.
            targets (Any): Ground truth targets.
        
        Returns:
            torch.Tensor: The computed loss.
        """
        raise NotImplementedError("compute_loss must be implemented in subclasses.")

    def compute_metrics(self, outputs: List[Any], targets: List[Any], image_ids: List[Any]) -> Dict[str, float]:
        """
        Computes evaluation metrics given the model outputs and targets.
        
        As PVTv2 serves as a backbone, metric computation is task-specific and should be implemented
        by subclasses that integrate PVTv2 with task-specific evaluation logic.
        
        Args:
            outputs (List[Any]): List of model outputs.
            targets (List[Any]): List of ground truth targets.
            image_ids (List[Any]): List of image IDs corresponding to the batches.
        
        Returns:
            Dict[str, float]: Dictionary of computed metrics.
        """
        raise NotImplementedError("compute_metrics must be implemented in subclasses.")

    def get_feature_maps(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Retrieves the feature maps from the PVTv2 backbone.
        
        Args:
            pixel_values (torch.Tensor): Input images.
        
        Returns:
            Dict[str, torch.Tensor]: Feature maps at specified stages.
        """
        return self.forward(pixel_values)

    def load_local_pretrained(self, local_path: str):
        """
        Load pretrained PVTv2 model weights from a local directory.
        
        Args:
            local_path (str): Path to the directory containing pretrained model weights.
        """
        try:
            self.model = PvtV2Model.from_pretrained(
                local_path,
                config=self.config,
                local_files_only=True
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded PVTv2Model from local path: {local_path}")
        except Exception as e:
            logger.error(f"Failed to load PVTv2Model from local path '{local_path}': {e}")
            raise

    def save_pretrained(self, save_path: str):
        """
        Save the PVTv2 model and feature extractor to the specified directory.
        
        Args:
            save_path (str): Directory path to save the model and feature extractor.
        """
        try:
            os.makedirs(save_path, exist_ok=True)
            self.model.save_pretrained(save_path)
            self.feature_extractor.save_pretrained(save_path)
            logger.info(f"Saved PVTv2Model and feature extractor to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save PVTv2Model to '{save_path}': {e}")
            raise

"""
# Example usage of PVTv2Model

from PIL import Image
import requests
import torch

from transformers import AutoImageProcessor
from src.models.zoo.pvtv2 import PVTv2Model

# Initialize the PVTv2 backbone model
pvtv2_backbone = PVTv2Model(
    model_name_or_path="OpenGVLab/pvt_v2_b0",
    pretrained=True,
    linear_attention=False,
    output_features=['stage1', 'stage2', 'stage3', 'stage4'],
    device="cpu",
)

# Load and preprocess an image
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
image_processor = AutoImageProcessor.from_pretrained("OpenGVLab/pvt_v2_b0")
processed = image_processor(images=image, return_tensors="pt")
pixel_values = processed["pixel_values"].to(pvtv2_backbone.device)

# Forward pass to extract feature maps
feature_maps = pvtv2_backbone.get_feature_maps(pixel_values)

# Display feature map shapes
for feature_name, feature_tensor in feature_maps.items():
    print(f"{feature_name}: {feature_tensor.shape}")
"""
