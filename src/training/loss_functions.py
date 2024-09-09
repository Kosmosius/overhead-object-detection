# src/training/loss_functions.py

import torch
import torch.nn as nn

class CustomLossFunction(nn.Module):
    """
    Generalized loss function for object detection tasks.
    Supports models like DETR with configurable components.
    
    Args:
        model (nn.Module): The object detection model that outputs logits and bounding boxes.
        class_weight (float): Weight for classification loss.
        bbox_weight (float): Weight for bounding box regression loss.
        giou_weight (float): Weight for Generalized IoU (GIoU) loss.
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

        # Placeholder for GIoU loss (can be replaced with a more advanced implementation)
        self.giou_loss_fn = nn.SmoothL1Loss()

    def forward(self, outputs, targets):
        """
        Compute the combined loss for object detection models.

        Args:
            outputs (dict): Model outputs containing logits and bounding boxes.
            targets (list): List of dictionaries containing target bounding boxes and class labels.

        Returns:
            dict: Dictionary containing the total loss and individual loss components.
        """
        pred_logits = outputs.logits
        pred_boxes = outputs.pred_boxes

        # Concatenate all target labels and boxes
        target_labels = torch.cat([t['labels'] for t in targets]).to(pred_logits.device)
        target_boxes = torch.cat([t['boxes'] for t in targets]).to(pred_boxes.device)

        # Compute classification loss
        classification_loss = self.classification_loss_fn(pred_logits, target_labels)

        # Compute bounding box regression loss
        bbox_loss = self.bbox_loss_fn(pred_boxes, target_boxes)

        # Compute GIoU loss (can be replaced with a proper GIoU loss function)
        giou_loss = self.giou_loss_fn(pred_boxes, target_boxes)

        # Combine the losses
        total_loss = (self.class_weight * classification_loss) + \
                     (self.bbox_weight * bbox_loss) + \
                     (self.giou_weight * giou_loss)

        return {
            "total_loss": total_loss,
            "classification_loss": classification_loss,
            "bbox_loss": bbox_loss,
            "giou_loss": giou_loss
        }


def compute_loss(outputs, targets, model, class_weight: float = 1.0, bbox_weight: float = 1.0, giou_weight: float = 1.0):
    """
    Wrapper function to compute the loss given the model's outputs and the target labels and boxes.
    
    Args:
        outputs (dict): Model outputs (logits and bounding boxes).
        targets (list): Target bounding boxes and labels.
        model (nn.Module): Object detection model being trained.
        class_weight (float): Weight for classification loss.
        bbox_weight (float): Weight for bounding box loss.
        giou_weight (float): Weight for Generalized IoU loss.

    Returns:
        dict: Dictionary containing the total loss and individual loss components.
    """
    loss_fn = CustomLossFunction(model, class_weight, bbox_weight, giou_weight)
    return loss_fn(outputs, targets)


def get_loss_function(model, loss_type="custom", class_weight: float = 1.0, bbox_weight: float = 1.0, giou_weight: float = 1.0):
    """
    Retrieve the appropriate loss function based on the specified loss type.
    
    Args:
        model (nn.Module): The object detection model for which the loss function is used.
        loss_type (str): The type of loss function to use ('custom').
        class_weight (float): Weight for classification loss.
        bbox_weight (float): Weight for bounding box regression loss.
        giou_weight (float): Weight for GIoU loss.

    Returns:
        nn.Module: Instantiated loss function.
    """
    if loss_type == "custom":
        return CustomLossFunction(model, class_weight, bbox_weight, giou_weight)
    else:
        raise ValueError(f"Loss type '{loss_type}' is not supported.")
