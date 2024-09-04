# src/models/foundation_model.py

import torch
from transformers import AutoModelForObjectDetection, PretrainedConfig

class HuggingFaceObjectDetectionModel:
    """
    Generalized HuggingFace object detection model loader to handle various transformer-based models.
    """
    def __init__(self, model_name: str, num_classes: int, pretrained=True):
        """
        Initialize the object detection model.

        Args:
            model_name (str): Pretrained model name or path (e.g., "facebook/detr-resnet-50").
            num_classes (int): Number of object detection classes.
            pretrained (bool): Whether to load pretrained weights.
        """
        self.model_name = model_name
        config = PretrainedConfig.from_pretrained(model_name)
        config.num_labels = num_classes
        self.model = AutoModelForObjectDetection.from_pretrained(model_name, config=config) if pretrained else AutoModelForObjectDetection(config=config)

    def forward(self, pixel_values, pixel_mask=None):
        """
        Forward pass through the model.

        Args:
            pixel_values (torch.Tensor): Input image tensor.
            pixel_mask (torch.Tensor): Optional pixel mask for images.

        Returns:
            dict: Model outputs (logits and bounding boxes).
        """
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def save(self, output_path: str):
        """
        Save the model using HuggingFace's save_pretrained method.

        Args:
            output_path (str): Path to save the model.
        """
        self.model.save_pretrained(output_path)

    def load(self, model_path: str):
        """
        Load a model from a saved path using HuggingFace's from_pretrained method.

        Args:
            model_path (str): Path to the saved model.
        """
        self.model = AutoModelForObjectDetection.from_pretrained(model_path)
