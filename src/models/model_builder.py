# src/models/model_builder.py

import torch
from transformers import AutoConfig, AutoModelForObjectDetection

def build_model(model_name: str, num_labels: int, pretrained: bool = True):
    """
    Build and return a HuggingFace object detection model.

    Args:
        model_name (str): HuggingFace model name or path (e.g., "facebook/detr-resnet-50").
        num_labels (int): Number of object detection classes (including background).
        pretrained (bool): Whether to load the pretrained weights.

    Returns:
        model: HuggingFace model instance.
    """
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels

    if pretrained:
        model = AutoModelForObjectDetection.from_pretrained(model_name, config=config)
    else:
        model = AutoModelForObjectDetection(config=config)

    return model

def load_model_from_checkpoint(checkpoint_path: str, model_name: str, num_labels: int, device: str = "cuda"):
    """
    Load a HuggingFace object detection model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the saved model checkpoint.
        model_name (str): HuggingFace model name or path (e.g., "facebook/detr-resnet-50").
        num_labels (int): Number of object classes.
        device (str): Device to load the model onto.

    Returns:
        model: Loaded HuggingFace model.
    """
    model = build_model(model_name, num_labels, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def save_model_checkpoint(model, checkpoint_path: str):
    """
    Save a HuggingFace model checkpoint.

    Args:
        model: The HuggingFace model to save.
        checkpoint_path (str): Path to save the model checkpoint.
    """
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved at {checkpoint_path}")
