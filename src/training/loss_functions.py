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
    def get_loss_function(
        loss_type: str,
        **kwargs
    ) -> nn.Module:
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
            raise ValueError(f"Loss function '{loss_type}' is not supported.")

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

    # Instantiate the loss function
    criterion = LossFunctionFactory.get_loss_function(loss_type, **loss_kwargs)
    total_loss = torch.tensor(0.0, requires_grad=True).to(outputs['logits'].device)
    loss_dict = {}

    # Compute classification loss
    if 'logits' in outputs and 'labels' in targets[0]:
        logits = outputs['logits']
        labels = torch.cat([t['labels'] for t in targets])
        classification_loss = criterion(logits, labels)
        weight = loss_weights.get('classification_loss', 1.0)
        loss_dict['classification_loss'] = classification_loss * weight
        total_loss += loss_dict['classification_loss']
        logger.debug(f"Classification loss computed: {classification_loss.item()}")

    # Compute bounding box regression loss
    if 'pred_boxes' in outputs and 'boxes' in targets[0]:
        pred_boxes = outputs['pred_boxes']
        target_boxes = torch.cat([t['boxes'] for t in targets])
        bbox_loss_fn = nn.SmoothL1Loss()
        bbox_loss = bbox_loss_fn(pred_boxes, target_boxes)
        weight = loss_weights.get('bbox_loss', 1.0)
        loss_dict['bbox_loss'] = bbox_loss * weight
        total_loss += loss_dict['bbox_loss']
        logger.debug(f"Bounding box loss computed: {bbox_loss.item()}")

    # Add other losses as needed
    # For example, auxiliary losses, regularization terms, etc.

    loss_dict['total_loss'] = total_loss
    return loss_dict
