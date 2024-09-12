# src/models/foundation_model.py

import torch
from transformers import AutoModelForObjectDetection, PretrainedConfig
import logging

class HuggingFaceObjectDetectionModel:
    """
    A generalized HuggingFace object detection model handler. This class allows
    loading and saving models, handling multiple model architectures, and
    performing forward passes with optional pixel masking.
    """

    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True, device: str = "cuda"):
        """
        Initialize the object detection model with optional pre-trained weights.

        Args:
            model_name (str): Pretrained model name or path (e.g., "facebook/detr-resnet-50").
            num_classes (int): Number of object detection classes (including background).
            pretrained (bool): Whether to load pretrained weights. Defaults to True.
            device (str): Device to run the model on ('cuda' or 'cpu'). Defaults to 'cuda'.
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.model = self._initialize_model(pretrained)
        logging.info(f"Initialized {model_name} with {num_classes} classes on {device}.")

    def _initialize_model(self, pretrained: bool) -> torch.nn.Module:
        """
        Internal method to initialize the model with or without pre-trained weights.

        Args:
            pretrained (bool): Whether to load pretrained weights.

        Returns:
            torch.nn.Module: Loaded object detection model.
        """
        try:
            config = PretrainedConfig.from_pretrained(self.model_name)
            config.num_labels = self.num_classes
            if pretrained:
                model = AutoModelForObjectDetection.from_pretrained(self.model_name, config=config)
                logging.info(f"Loaded pretrained model: {self.model_name}")
            else:
                model = AutoModelForObjectDetection(config=config)
                logging.info(f"Initialized model with custom configuration: {self.model_name}")
            return model.to(self.device)
        except Exception as e:
            logging.error(f"Error loading model {self.model_name}: {str(e)}")
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor = None) -> dict:
        """
        Perform a forward pass through the model.

        Args:
            pixel_values (torch.Tensor): Input image tensor.
            pixel_mask (torch.Tensor, optional): Optional pixel mask tensor.

        Returns:
            dict: Model outputs, including logits and bounding boxes.
        """
        try:
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            logging.debug(f"Forward pass completed for model {self.model_name}.")
            return outputs
        except Exception as e:
            logging.error(f"Error during forward pass: {str(e)}")
            raise RuntimeError(f"Model forward pass failed: {e}")

    def save(self, output_path: str):
        """
        Save the model using HuggingFace's save_pretrained method.

        Args:
            output_path (str): Path to save the model.
        """
        try:
            self.model.save_pretrained(output_path)
            logging.info(f"Model saved at {output_path}")
        except Exception as e:
            logging.error(f"Error saving model to {output_path}: {str(e)}")
            raise IOError(f"Failed to save model at {output_path}: {e}")

    def load(self, model_path: str):
        """
        Load a model from a saved path using HuggingFace's from_pretrained method.

        Args:
            model_path (str): Path to the saved model.
        """
        try:
            self.model = AutoModelForObjectDetection.from_pretrained(model_path).to(self.device)
            logging.info(f"Model loaded from {model_path}")
        except Exception as e:
            logging.error(f"Error loading model from {model_path}: {str(e)}")
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    def freeze_backbone(self):
        """
        Freeze the backbone layers of the model to fine-tune only the detection heads.
        """
        try:
            for param in self.model.base_model.parameters():
                param.requires_grad = False
            logging.info("Backbone layers have been frozen.")
        except AttributeError:
            logging.warning(f"Model {self.model_name} does not have a backbone that can be frozen.")

    def unfreeze_backbone(self):
        """
        Unfreeze the backbone layers of the model to allow full fine-tuning.
        """
        try:
            for param in self.model.base_model.parameters():
                param.requires_grad = True
            logging.info("Backbone layers have been unfrozen.")
        except AttributeError:
            logging.warning(f"Model {self.model_name} does not have a backbone that can be unfrozen.")
