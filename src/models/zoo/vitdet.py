# src/models/zoo/vitdet.py

import torch
from torch import nn
from transformers import VitDetModel, VitDetConfig
from typing import Optional

class ViTDet(nn.Module):
    """
    ViTDet Model: A Vision Transformer backbone for object detection.

    This class wraps Hugging Face's VitDetModel to integrate with the project's model zoo.
    It provides methods for loading pretrained weights, performing forward passes,
    extracting features, and managing model parameters for fine-tuning or freezing.

    Reference:
    Exploring Plain Vision Transformer Backbones for Object Detection
    by Yanghao Li, Hanzi Mao, Ross Girshick, Kaiming He
    """

    def __init__(
        self,
        config: Optional[VitDetConfig] = None,
        pretrained: bool = True,
        model_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initializes the ViTDet model.

        Args:
            config (VitDetConfig, optional): Configuration for the ViTDet model.
                If None, the default configuration is used.
            pretrained (bool, optional): If True, loads pretrained weights.
                Requires `model_path` to be provided if in an air-gapped environment.
            model_path (str, optional): Path to the pretrained model directory.
                Required if `pretrained` is True and operating in an air-gapped environment.
            **kwargs: Additional keyword arguments for VitDetModel.from_pretrained.
        """
        super(ViTDet, self).__init__()
        if config is None:
            config = VitDetConfig()
        if pretrained:
            if model_path is None:
                raise ValueError(
                    "To load pretrained weights in an air-gapped environment, `model_path` must be provided."
                )
            self.vitdet = VitDetModel.from_pretrained(
                model_path, config=config, local_files_only=True, **kwargs
            )
        else:
            self.vitdet = VitDetModel(config)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config_path: Optional[str] = None,
        **kwargs
    ):
        """
        Class method to load a pretrained ViTDet model from a local path.

        Args:
            model_path (str): Path to the pretrained model directory.
            config_path (str, optional): Path to the model configuration file.
                If None, the configuration is loaded from `model_path`.
            **kwargs: Additional keyword arguments for VitDetModel.from_pretrained.

        Returns:
            ViTDet: An instance of the ViTDet model with pretrained weights.
        """
        if config_path:
            config = VitDetConfig.from_pretrained(config_path, **kwargs)
        else:
            config = VitDetConfig.from_pretrained(model_path, **kwargs)
        model = cls(config=config, pretrained=True, model_path=model_path, **kwargs)
        return model

    def forward(self, pixel_values: torch.Tensor, **kwargs):
        """
        Performs a forward pass through the ViTDet backbone.

        Args:
            pixel_values (torch.Tensor): Input image tensor of shape (batch_size, num_channels, height, width).
            **kwargs: Additional keyword arguments for VitDetModel.

        Returns:
            transformers.modeling_outputs.BaseModelOutput: The output of the VitDet backbone.
        """
        return self.vitdet(pixel_values=pixel_values, **kwargs)

    def get_features(self, pixel_values: torch.Tensor, **kwargs):
        """
        Extracts features from the input images using the VitDet backbone.

        Args:
            pixel_values (torch.Tensor): Input image tensor.
            **kwargs: Additional keyword arguments for VitDetModel.

        Returns:
            torch.Tensor: Extracted features from the backbone (last_hidden_state).
        """
        outputs = self.forward(pixel_values, **kwargs)
        return outputs.last_hidden_state

    def save_pretrained(self, save_directory: str):
        """
        Saves the VitDet model to the specified directory.

        Args:
            save_directory (str): Directory to save the model.
        """
        self.vitdet.save_pretrained(save_directory)

    def load_pretrained(self, load_directory: str):
        """
        Loads the VitDet model from the specified directory.

        Args:
            load_directory (str): Directory to load the model from.
        """
        self.vitdet = VitDetModel.from_pretrained(
            load_directory, config=self.vitdet.config, local_files_only=True
        )

    def freeze_backbone(self):
        """
        Freezes the backbone parameters to prevent them from being updated during training.
        """
        for param in self.vitdet.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """
        Unfreezes the backbone parameters to allow them to be updated during training.
        """
        for param in self.vitdet.parameters():
            param.requires_grad = True

