# src/training/loss_functions.py

import torch
import torch.nn as nn
from transformers import DetrForObjectDetection

class CustomLossFunction(nn.Module):
    """
    Generalized loss function for object detection tasks.
    Supports DETR and other models with configurable components.
    
    Args:
        model (nn.Module): The model that outputs logits and bounding boxes.
        class_weight (float): Weight for classification loss.
        bbox_weight (float): Weight for bounding box regression loss.
        giou_weight (float): Weight for GIoU loss.
    """
    def __init__(self, model, class_weight: float = 1.0, bbox_weight: float = 1.0, giou_weight: float = 1.0):
        super(CustomLossFunction, self).__init__()
        self.model = model
        self.class_weight = class_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight

        # Define classification loss
        self.classification_loss_fn = nn.CrossEntropyLoss()

        # Bounding box regression loss (L1 loss)
        self.bbox_loss_fn = nn.L1Loss()

        # GIoU loss (optionally switch to IoU, DIoU, or CIoU)
        self.giou_loss_fn = nn.SmoothL1Loss()  # Replace with a better GIoU loss if needed

    def forward(self, outputs, targets):
        """
        Compute the combined loss for object detection models.

        Args:
            outputs (dict): Model outputs (logits, bounding boxes).
            targets (list): Target bounding boxes and class labels.

        Returns:
            dict: Dictionary containing total loss and individual loss components.
        """
        pred_logits = outputs.logits
        pred_boxes = outputs.pred_boxes

        # Classification loss
        target_labels = torch.cat([t['labels'] for t in targets]).to(pred_logits.device)
        classification_loss = self.classification_loss_fn(pred_logits, target_labels)

        # Bounding box regression loss
        target_boxes = torch.cat([t['boxes'] for t in targets]).to(pred_boxes.device)
        bbox_loss = self.bbox_loss_fn(pred_boxes, target_boxes)

        # GIoU loss
        giou_loss = self.giou_loss_fn(pred_boxes, target_boxes)  # You can add GIoU here if preferred

        total_loss = (self.class_weight * classification_loss) + (self.bbox_weight * bbox_loss) + (self.giou_weight * giou_loss)

        return {
            "total_loss": total_loss,
            "classification_loss": classification_loss,
            "bbox_loss": bbox_loss,
            "giou_loss": giou_loss
        }

def get_loss_function(model, loss_type="custom", class_weight: float = 1.0, bbox_weight: float = 1.0, giou_weight: float = 1.0):
    """
    Retrieve the appropriate loss function based on the loss type.

    Args:
        model (nn.Module): The model for which to compute the loss.
        loss_type (str): Type of loss function ('custom' for now).
        class_weight (float): Weight for the classification loss.
        bbox_weight (float): Weight for the bounding box regression loss.
        giou_weight (float): Weight for the GIoU loss.

    Returns:
        nn.Module: The loss function to use.
    """
    if loss_type == "custom":
        return CustomLossFunction(model, class_weight, bbox_weight, giou_weight)
    else:
        raise ValueError(f"Loss type {loss_type} is not supported.")
