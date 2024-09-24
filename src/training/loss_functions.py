# src/training/loss_functions.py

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class LossFunctionFactory:
    """
    Factory class to create loss functions based on configuration.
    """

    @staticmethod
    def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
        """
        Retrieve a loss function based on the specified type.

        Args:
            loss_type (str): The type of loss function to use.
            **kwargs: Additional keyword arguments for loss function initialization.

        Returns:
            nn.Module: An instance of a PyTorch loss function.
        """
        loss_functions = {
            'cross_entropy': nn.CrossEntropyLoss,
            'mse': nn.MSELoss,
            'l1': nn.L1Loss,
            'smooth_l1': nn.SmoothL1Loss,
            # Add more loss functions as needed
        }

        loss_class = loss_functions.get(loss_type)
        if not loss_class:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        return loss_class(**kwargs)

def compute_loss(
    outputs: Dict[str, torch.Tensor],
    targets: List[Dict[str, torch.Tensor]],
    model: nn.Module,
    loss_config: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    """
    Compute the loss between model outputs and targets.

    Args:
        outputs (Dict[str, torch.Tensor]): Outputs from the model.
        targets (List[Dict[str, torch.Tensor]]): Ground truth annotations.
        model (nn.Module): The model being trained.
        loss_config (Dict[str, Any]): Configuration for the loss function.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing individual losses and the total loss.
    """
    loss_type = loss_config.get('loss_type', 'cross_entropy')
    loss_weights = loss_config.get('loss_weights', {})
    loss_kwargs = loss_config.get('loss_kwargs', {})

    loss_dict = {}
    
    # Initialize total_loss on the same device as outputs
    if outputs:
        device = next(iter(outputs.values())).device
    else:
        device = torch.device('cpu')  # Default to CPU if outputs are empty
    total_loss = torch.tensor(0.0, device=device)

    if not targets:
        # If targets are empty, return zero loss
        loss_dict['total_loss'] = total_loss
        return loss_dict

    # Compute classification loss if 'logits' and 'labels' are present
    if 'logits' in outputs and all('labels' in target for target in targets):
        logits = outputs['logits']
        try:
            labels = torch.cat([t['labels'] for t in targets])
        except RuntimeError as e:
            raise ValueError(f"Error concatenating labels: {e}")

        classification_loss_fn = LossFunctionFactory.get_loss_function(loss_type, **loss_kwargs)
        classification_loss = classification_loss_fn(logits, labels)
        weight = loss_weights.get('classification_loss', 1.0)
        loss_dict['classification_loss'] = classification_loss * weight
        total_loss = total_loss + loss_dict['classification_loss']

    # Compute bbox loss if 'pred_boxes' and 'boxes' are present
    if 'pred_boxes' in outputs and all('boxes' in target for target in targets):
        pred_boxes = outputs['pred_boxes']
        try:
            boxes = torch.cat([t['boxes'] for t in targets])
        except RuntimeError as e:
            raise ValueError(f"Error concatenating boxes: {e}")

        bbox_loss_fn = LossFunctionFactory.get_loss_function(loss_type, **loss_kwargs)
        bbox_loss = bbox_loss_fn(pred_boxes, boxes)
        weight = loss_weights.get('bbox_loss', 1.0)
        loss_dict['bbox_loss'] = bbox_loss * weight
        total_loss = total_loss + loss_dict['bbox_loss']

    loss_dict['total_loss'] = total_loss
    return loss_dict
