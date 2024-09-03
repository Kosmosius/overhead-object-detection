# src/models/foundation_model.py

import torch
from transformers import DetrForObjectDetection, DetrConfig

class DetrObjectDetectionModel:
    def __init__(self, num_classes):
        """
        Initialize the DETR model for object detection.
        Args:
            num_classes (int): Number of object classes (including background).
        """
        # Configuration for the DETR model
        self.config = DetrConfig.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=num_classes
        )
        # Load the DETR model with the specified configuration
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50", 
            config=self.config
        )

    def forward(self, pixel_values, pixel_mask=None):
        """
        Forward pass through the model.
        Args:
            pixel_values (torch.Tensor): The input images tensor.
            pixel_mask (torch.Tensor): Optional pixel mask for the images.
        Returns:
            dict: Output of the model containing logits and bounding boxes.
        """
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def save(self, output_path):
        """
        Save the model to the specified output path.
        Args:
            output_path (str): Path to save the model.
        """
        self.model.save_pretrained(output_path)

    def load(self, model_path):
        """
        Load the model from the specified path.
        Args:
            model_path (str): Path to the saved model.
        """
        self.model = DetrForObjectDetection.from_pretrained(model_path)

