# src/training/loss_functions.py

import torch
import torch.nn as nn
from torchvision.ops import generalized_box_iou_loss

class CustomLossFunction(nn.Module):
    """
    Generalized loss function for object detection tasks.
    Supports models like DETR with configurable components.

    Args:
        model (nn.Module): The object detection model that outputs logits and bounding boxes.
        config (dict): Configuration dict for weights and loss functions.
    """

    def __init__(self, model, config: dict):
        super(CustomLossFunction, self).__init__()
        self.model = model
        self.class_weight = config.get("class_weight", 1.0)
        self.bbox_weight = config.get("bbox_weight", 1.0)
        self.giou_weight = config.get("giou_weight", 1.0)
        
        # Allow different classification losses, with default to CrossEntropy
        self.classification_loss_fn = config.get("classification_loss_fn", nn.CrossEntropyLoss())
        
        # L1 loss for bounding box regression
        self.bbox_loss_fn = config.get("bbox_loss_fn", nn.L1Loss())

        # GIoU loss using torchvision's generalized_box_iou_loss
        self.giou_loss_fn = config.get("giou_loss_fn", generalized_box_iou_loss)

    def forward(self, outputs, targets):
        """
        Compute the combined loss for object detection models.

        Args:
            outputs (dict): Model outputs containing logits and bounding boxes.
            targets (list): List of dictionaries containing target bounding boxes and class labels.

        Returns:
            dict: Dictionary containing the total loss and individual loss components.
        """
        pred_logits = outputs['logits']
        pred_boxes = outputs['pred_boxes']

        # Concatenate all target labels and boxes
        target_labels = torch.cat([t['labels'] for t in targets]).to(pred_logits.device)
        target_boxes = torch.cat([t['boxes'] for t in targets]).to(pred_boxes.device)

        # Classification loss
        classification_loss = self.classification_loss_fn(pred_logits, target_labels)

        # Bounding box regression loss
        bbox_loss = self.bbox_loss_fn(pred_boxes, target_boxes)

        # GIoU loss
        giou_loss = self.giou_loss_fn(pred_boxes, target_boxes)

        # Combine the losses based on weights from config
        total_loss = (self.class_weight * classification_loss) + \
                     (self.bbox_weight * bbox_loss) + \
                     (self.giou_weight * giou_loss)

        return {
            "total_loss": total_loss,
            "classification_loss": classification_loss,
            "bbox_loss": bbox_loss,
            "giou_loss": giou_loss
        }

def compute_loss(outputs, targets, model, config: dict):
    """
    Wrapper function to compute the loss given the model's outputs and the target labels and boxes.
    
    Args:
        outputs (dict): Model outputs (logits and bounding boxes).
        targets (list): Target bounding boxes and labels.
        model (nn.Module): Object detection model being trained.
        config (dict): Configuration dictionary with loss weights and loss functions.

    Returns:
        dict: Dictionary containing the total loss and individual loss components.
    """
    loss_fn = CustomLossFunction(model, config)
    return loss_fn(outputs, targets)

def get_loss_function(model, config: dict):
    """
    Retrieve the appropriate loss function based on the specified loss configuration.
    
    Args:
        model (nn.Module): The object detection model for which the loss function is used.
        config (dict): Configuration dict for specifying loss types and weights.

    Returns:
        nn.Module: Instantiated loss function.
    """
    return CustomLossFunction(model, config)
