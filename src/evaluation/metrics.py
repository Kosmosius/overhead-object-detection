# src/evaluation/metrics.py

import torch
from torchvision.ops import box_iou
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Dict, Tuple, Union

def compute_iou(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between predicted and ground truth boxes.

    Args:
        pred_boxes (Tensor): Predicted bounding boxes (Nx4).
        gt_boxes (Tensor): Ground truth bounding boxes (Mx4).

    Returns:
        Tensor: IoU scores (NxM).
    """
    return box_iou(pred_boxes, gt_boxes)

def compute_map(
    predictions: List[Dict[str, torch.Tensor]],
    ground_truths: List[Dict[str, torch.Tensor]],
    iou_thresholds: List[float] = [0.5]
) -> Dict[str, float]:
    """
    Compute mAP, precision, recall, and F1-score for object detection models.

    Args:
        predictions (list of dict): List of predicted bounding boxes and scores.
        ground_truths (list of dict): List of ground truth bounding boxes and labels.
        iou_thresholds (list of float): List of IoU thresholds for mAP calculation (default: [0.5]).

    Returns:
        dict: Aggregated metrics such as precision, recall, F1-score, and mAP.
    """
    all_metrics = {}

    for iou_threshold in iou_thresholds:
        all_pred_labels, all_true_labels = [], []
        
        for pred, gt in zip(predictions, ground_truths):
            pred_boxes = pred['boxes']
            gt_boxes = gt['boxes']
            
            ious = compute_iou(pred_boxes, gt_boxes)
            iou_max, _ = ious.max(dim=1)
            
            pred_labels = (iou_max >= iou_threshold).float().tolist()
            gt_labels = [1.0] * len(gt_boxes)
            
            all_pred_labels.extend(pred_labels)
            all_true_labels.extend(gt_labels)
        
        precision = precision_score(all_true_labels, all_pred_labels, zero_division=0)
        recall = recall_score(all_true_labels, all_pred_labels, zero_division=0)
        f1 = f1_score(all_true_labels, all_pred_labels, zero_division=0)
        
        all_metrics[f'precision@{iou_threshold}'] = precision
        all_metrics[f'recall@{iou_threshold}'] = recall
        all_metrics[f'f1_score@{iou_threshold}'] = f1

    return all_metrics

def evaluate_model(
    model,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    iou_thresholds: List[float] = [0.5]
) -> Dict[str, float]:
    """
    Evaluate a model using IoU-based metrics such as mAP, precision, and recall.

    Args:
        model: HuggingFace object detection model (e.g., DETR, YOLO).
        dataloader: DataLoader for evaluation dataset.
        device: Device to run the evaluation on.
        iou_thresholds: List of IoU thresholds for mAP calculation.

    Returns:
        dict: Metrics such as precision, recall, F1-score, and mAP.
    """
    model.eval()
    model.to(device)

    all_predictions, all_ground_truths = [], []

    with torch.no_grad():
        for images, targets in dataloader:
            pixel_values = torch.stack(images).to(device)
            outputs = model(pixel_values=pixel_values)

            pred_boxes = outputs.pred_boxes.cpu()
            logits = outputs.logits.cpu()

            for i in range(len(images)):
                pred = {
                    "boxes": pred_boxes[i],
                    "scores": logits[i]
                }
                gt = {
                    "boxes": targets[i]["boxes"].cpu(),
                    "labels": targets[i]["labels"].cpu()
                }

                all_predictions.append(pred)
                all_ground_truths.append(gt)

    metrics = compute_map(all_predictions, all_ground_truths, iou_thresholds)
    return metrics

