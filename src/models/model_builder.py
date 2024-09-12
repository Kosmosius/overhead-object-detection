# src/models/model_builder.py

import torch
import logging
from transformers import AutoConfig, AutoModelForObjectDetection

class ModelBuilder:
    """
    Class to build and manage HuggingFace object detection models.
    This class encapsulates the logic for model initialization, loading from checkpoints, and saving.
    """

    def __init__(self, model_name: str, num_labels: int, pretrained: bool = True):
        """
        Initialize the ModelBuilder with a model name and number of labels.

        Args:
            model_name (str): The name or path of the HuggingFace model.
            num_labels (int): The number of object detection classes (including background).
            pretrained (bool): Whether to load pretrained weights. Default is True.
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.pretrained = pretrained
        self.model = None

    def build_model(self):
        """
        Build the object detection model using HuggingFace's AutoModelForObjectDetection.

        Returns:
            model: HuggingFace model instance.
        """
        try:
            logging.info(f"Building model: {self.model_name} with {self.num_labels} labels")
            config = AutoConfig.from_pretrained(self.model_name)
            config.num_labels = self.num_labels

            if self.pretrained:
                self.model = AutoModelForObjectDetection.from_pretrained(self.model_name, config=config)
                logging.info(f"Loaded pretrained model: {self.model_name}")
            else:
                self.model = AutoModelForObjectDetection(config=config)
                logging.info(f"Initialized model with random weights: {self.model_name}")

            return self.model

        except Exception as e:
            logging.error(f"Error in building model: {str(e)}")
            raise ValueError(f"Model creation failed for {self.model_name}: {str(e)}")

    def load_model_from_checkpoint(self, checkpoint_path: str, device: str = "cuda"):
        """
        Load a HuggingFace object detection model from a checkpoint.

        Args:
            checkpoint_path (str): Path to the saved model checkpoint.
            device (str): Device to load the model onto. Default is "cuda".

        Returns:
            model: Loaded HuggingFace model.
        """
        try:
            if not self.model:
                self.build_model()

            logging.info(f"Loading model from checkpoint: {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            self.model.to(device)
            self.model.eval()
            logging.info(f"Model loaded from checkpoint: {checkpoint_path}")

            return self.model

        except FileNotFoundError:
            logging.error(f"Checkpoint not found at {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        except Exception as e:
            logging.error(f"Error loading model from checkpoint: {str(e)}")
            raise RuntimeError(f"Error loading model from {checkpoint_path}: {str(e)}")

    def save_model_checkpoint(self, checkpoint_path: str):
        """
        Save a HuggingFace model checkpoint.

        Args:
            checkpoint_path (str): Path to save the model checkpoint.
        """
        try:
            logging.info(f"Saving model checkpoint to {checkpoint_path}")
            torch.save(self.model.state_dict(), checkpoint_path)
            logging.info(f"Model checkpoint saved at {checkpoint_path}")
        except Exception as e:
            logging.error(f"Error saving model checkpoint: {str(e)}")
            raise RuntimeError(f"Failed to save model checkpoint: {str(e)}")


def configure_model(model_name: str, num_labels: int, pretrained: bool = True):
    """
    Function to quickly configure and return a HuggingFace object detection model.

    Args:
        model_name (str): HuggingFace model name or path (e.g., "facebook/detr-resnet-50").
        num_labels (int): Number of object detection classes (including background).
        pretrained (bool): Whether to load pretrained weights. Default is True.

    Returns:
        model: HuggingFace object detection model.
    """
    builder = ModelBuilder(model_name, num_labels, pretrained)
    return builder.build_model()


def load_checkpoint(checkpoint_path: str, model_name: str, num_labels: int, device: str = "cuda"):
    """
    Load a HuggingFace object detection model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the saved model checkpoint.
        model_name (str): HuggingFace model name or path (e.g., "facebook/detr-resnet-50").
        num_labels (int): Number of object classes.
        device (str): Device to load the model onto. Default is "cuda".

    Returns:
        model: Loaded HuggingFace model.
    """
    builder = ModelBuilder(model_name, num_labels, pretrained=False)
    return builder.load_model_from_checkpoint(checkpoint_path, device)


def save_checkpoint(model, checkpoint_path: str):
    """
    Save a HuggingFace model checkpoint.

    Args:
        model: The HuggingFace model to save.
        checkpoint_path (str): Path to save the model checkpoint.
    """
    builder = ModelBuilder(model_name="", num_labels=0)  # model_name and num_labels are unused in save
    builder.model = model  # Assign the provided model
    builder.save_model_checkpoint(checkpoint_path)
