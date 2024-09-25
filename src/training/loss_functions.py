# src/training/loss_functions.py

from typing import Dict, List, Any
import torch
import torch.nn as nn
import logging 

class LossFunctionFactory:
    @staticmethod
    def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
        """Factory method to get the appropriate loss function based on loss_type."""
        if loss_type == "cross_entropy":
            return nn.CrossEntropyLoss(**kwargs)
        elif loss_type == "smooth_l1":
            return nn.SmoothL1Loss(**kwargs)
        elif loss_type == "mse":
            return nn.MSELoss(**kwargs)
        elif loss_type == "l1":
            return nn.L1Loss(**kwargs)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

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
        loss_config (Dict[str, Any]): Configuration for the loss functions.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing individual losses and the total loss.
    """
    loss_weights = loss_config.get('loss_weights', {})
    loss_kwargs = loss_config.get('loss_kwargs', {})
    loss_types = loss_config.get('loss_types', {
        'classification_loss': 'cross_entropy',
        'bbox_loss': 'smooth_l1'
    })

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

    # Compute classification loss
    if 'classification_loss' in loss_types and 'logits' in outputs and all('labels' in target for target in targets):
        logits = outputs['logits']
        try:
            labels = torch.cat([t['labels'] for t in targets]).to(logits.device)
            if classification_loss_type in ['cross_entropy', 'nll_loss']:
                labels = labels.long()
            elif classification_loss_type in ['mse', 'l1', 'smooth_l1']:
                labels = labels.float()
            else:
                # Default to float if loss type is unrecognized (optional)
                labels = labels.float()
        except RuntimeError as e:
            raise ValueError(f"Error concatenating labels: {e}")

        classification_loss_type = loss_types.get('classification_loss', 'cross_entropy')
        classification_loss_kwargs = loss_kwargs.get('classification_loss', {})
        try:
            classification_loss_fn = LossFunctionFactory.get_loss_function(classification_loss_type, **classification_loss_kwargs)
        except ValueError as e:
            raise ValueError(f"Unsupported classification loss type: {classification_loss_type}.") from e

        classification_loss = classification_loss_fn(logits, labels)
        weight = loss_weights.get('classification_loss', 1.0)
        loss_dict['classification_loss'] = classification_loss * weight
        total_loss = total_loss + loss_dict['classification_loss']

        logging.debug(f"Classification loss computed: {loss_dict['classification_loss'].item()}")

    # Compute bbox loss
    if 'bbox_loss' in loss_types and 'pred_boxes' in outputs and all('boxes' in target for target in targets):
        pred_boxes = outputs['pred_boxes']
        try:
            boxes = torch.cat([t['boxes'] for t in targets]).to(pred_boxes.device).float()
        except RuntimeError as e:
            raise ValueError(f"Error concatenating boxes: {e}")

        bbox_loss_type = loss_types.get('bbox_loss', 'smooth_l1')
        bbox_loss_kwargs = loss_kwargs.get('bbox_loss', {})
        try:
            bbox_loss_fn = LossFunctionFactory.get_loss_function(bbox_loss_type, **bbox_loss_kwargs)
        except ValueError as e:
            raise ValueError(f"Unsupported bbox loss type: {bbox_loss_type}.") from e

        bbox_loss = bbox_loss_fn(pred_boxes, boxes)
        weight = loss_weights.get('bbox_loss', 1.0)
        loss_dict['bbox_loss'] = bbox_loss * weight
        total_loss = total_loss + loss_dict['bbox_loss']

        logging.debug(f"Bounding box loss computed: {loss_dict['bbox_loss'].item()}")

    loss_dict['total_loss'] = total_loss

    logging.debug(f"Total loss computed: {loss_dict['total_loss'].item()}")

    return loss_dict
