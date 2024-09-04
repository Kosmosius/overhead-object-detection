# src/models/model_builder.py

import torch
from transformers import DetrConfig, DetrForObjectDetection

def build_detr_model(num_labels: int = 91, backbone: str = "facebook/detr-resnet-50", pretrained: bool = True):
    """
    Build and return a DETR (Detection Transformer) model.

    Args:
        num_labels (int): Number of object classes including background (default: 91 for COCO dataset).
        backbone (str): Pretrained backbone model (default: "facebook/detr-resnet-50").
        pretrained (bool): Whether to load the pretrained weights.

    Returns:
        model: HuggingFace DETR model.
    """
    config = DetrConfig.from_pretrained(backbone)
    config.num_labels = num_labels

    if pretrained:
        model = DetrForObjectDetection.from_pretrained(backbone, config=config)
    else:
        model = DetrForObjectDetection(config=config)

    return model

def load_model_from_checkpoint(checkpoint_path: str, num_labels: int = 91, device: str = "cuda"):
    """
    Load a DETR model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the saved model checkpoint.
        num_labels (int): Number of object classes including background (default: 91).
        device (str): Device to load the model onto (default: "cuda").

    Returns:
        model: The loaded DETR model.
    """
    model = build_detr_model(num_labels=num_labels, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def save_model_checkpoint(model, checkpoint_path: str):
    """
    Save the current state of a DETR model to a checkpoint.

    Args:
        model: The DETR model to save.
        checkpoint_path (str): Path to save the model checkpoint.
    """
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved at {checkpoint_path}")

def get_model_params(model):
    """
    Retrieve the number of trainable parameters in the model.

    Args:
        model: The model whose parameters are to be counted.

    Returns:
        int: The number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
