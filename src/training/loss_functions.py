# src/training/loss_functions.py

import torch
import torch.nn as nn
from transformers import DetrForObjectDetection

class DetrLossFunction(nn.Module):
    """
    Custom loss function for DETR (DEtection TRansformers).
    This loss function combines classification loss and bounding box regression loss.

    Args:
        model (DetrForObjectDetection): The DETR model that outputs logits and bounding boxes.
        class_weight (float): Weight for classification loss.
        bbox_weight (float): Weight for bounding box regression loss.
        giou_weight (float): Weight for Generalized IoU (GIoU) loss.
    """
    def __init__(self, model: DetrForObjectDetection, class_weight: float = 1.0, bbox_weight: float = 1.0, giou_weight: float = 1.0):
        super(DetrLossFunction, self).__init__()
        self.model = model
        self.class_weight = class_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight

        # Cross-entropy for classification loss
        self.classification_loss_fn = nn.CrossEntropyLoss()

        # L1 loss for bounding box regression
        self.bbox_loss_fn = nn.L1Loss()

        # GIoU loss for bounding boxes
        self.giou_loss_fn = nn.SmoothL1Loss()  # You could use GIoU from torchvision if needed

    def forward(self, outputs, targets):
        """
        Compute the combined loss for DETR model outputs.

        Args:
            outputs (dict): Dictionary containing model outputs (logits and bounding boxes).
            targets (list): List of dictionaries containing target bounding boxes and class labels.

        Returns:
            dict: A dictionary containing the total loss and individual components (classification, bbox, giou).
        """
        pred_logits = outputs.logits
        pred_boxes = outputs.pred_boxes

        # Classification targets and loss
        target_labels = torch.cat([t['labels'] for t in targets]).to(pred_logits.device)
        classification_loss = self.classification_loss_fn(pred_logits, target_labels)

        # Bounding box regression targets and loss
        target_boxes = torch.cat([t['boxes'] for t in targets]).to(pred_boxes.device)
        bbox_loss = self.bbox_loss_fn(pred_boxes, target_boxes)

        # Generalized IoU loss
        giou_loss = self.giou_loss_fn(pred_boxes, target_boxes)  # Optionally replace with a proper GIoU implementation

        # Total loss
        total_loss = (self.class_weight * classification_loss) + (self.bbox_weight * bbox_loss) + (self.giou_weight * giou_loss)

        return {
            "total_loss": total_loss,
            "classification_loss": classification_loss,
            "bbox_loss": bbox_loss,
            "giou_loss": giou_loss
        }

def get_loss_function(model: DetrForObjectDetection, class_weight: float = 1.0, bbox_weight: float = 1.0, giou_weight: float = 1.0):
    """
    Get a custom loss function for the DETR model.

    Args:
        model (DetrForObjectDetection): The DETR model to use for loss computation.
        class_weight (float): Weight for the classification loss.
        bbox_weight (float): Weight for the bounding box regression loss.
        giou_weight (float): Weight for the GIoU loss.

    Returns:
        DetrLossFunction: An instance of the custom DETR loss function.
    """
    return DetrLossFunction(model, class_weight, bbox_weight, giou_weight)
