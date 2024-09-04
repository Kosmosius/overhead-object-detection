# src/evaluation/metrics.py

import torch
from torchvision.ops import box_iou
from transformers import DetrForObjectDetection
from typing import List, Dict, Tuple

def compute_map(predictions: List[Dict[str, torch.Tensor]], ground_truths: List[Dict[str, torch.Tensor]], iou_threshold: float = 0.5) -> Tuple[float, float]:
    """
    Compute the Mean Average Precision (mAP) for the predictions.

    Args:
        predictions (list of dict): List of predictions from the model.
        ground_truths (list of dict): List of ground truth annotations.
        iou_threshold (float): IoU threshold for considering a prediction as correct.

    Returns:
        tuple: Precision and recall metrics.
    """
    true_positives, false_positives, total_ground_truths = 0, 0, 0

    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred['boxes']
        gt_boxes = gt['boxes']

        # Calculate IoU between predicted and ground truth boxes
        ious = box_iou(pred_boxes, gt_boxes)

        # Count total ground truths
        total_ground_truths += len(gt_boxes)

        # Count true positives and false positives
        true_positives += (ious.max(dim=1)[0] > iou_threshold).sum().item()
        false_positives += (ious.max(dim=1)[0] <= iou_threshold).sum().item()

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / total_ground_truths if total_ground_truths > 0 else 0

    return precision, recall

def evaluate_model(model: DetrForObjectDetection, dataloader: torch.utils.data.DataLoader, device: str = 'cuda') -> Tuple[float, float]:
    """
    Evaluate the model on the validation dataset.

    Args:
        model (DetrForObjectDetection): The trained DETR model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        device (str): Device to run the evaluation on ('cuda' or 'cpu').

    Returns:
        tuple: Precision and recall metrics for the evaluation.
    """
    model.to(device)
    model.eval()

    all_predictions, all_ground_truths = [], []

    with torch.no_grad():
        for images, targets in dataloader:
            pixel_values = torch.stack(images).to(device)

            # Forward pass
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            boxes = outputs.pred_boxes

            # Collect predictions and ground truths
            for i in range(len(images)):
                pred = {
                    'boxes': boxes[i].cpu(),
                    'scores': logits[i].cpu()
                }
                gt = {
                    'boxes': targets[i]['boxes'].cpu(),
                    'labels': targets[i]['labels'].cpu()
                }

                all_predictions.append(pred)
                all_ground_truths.append(gt)

    # Compute precision and recall
    precision, recall = compute_map(all_predictions, all_ground_truths)
    return precision, recall
