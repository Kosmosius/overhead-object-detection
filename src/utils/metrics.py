# src/utils/metrics.py

import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from torchvision.ops import box_iou
from typing import List, Dict, Tuple, Optional
from datasets import load_metric


def compute_iou(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute Intersection over Union (IoU) between predicted and ground truth boxes.

    Args:
        pred_boxes (torch.Tensor): Predicted bounding boxes (Nx4).
        gt_boxes (torch.Tensor): Ground truth bounding boxes (Mx4).

    Returns:
        torch.Tensor: IoU scores (NxM).
    """
    return box_iou(pred_boxes, gt_boxes)


def compute_precision_recall_f1(pred_labels: List[float], gt_labels: List[float]) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score based on predicted and ground truth labels.

    Args:
        pred_labels (List[float]): Predicted labels (1 for correct, 0 for incorrect).
        gt_labels (List[float]): Ground truth labels (1 for true objects).

    Returns:
        Dict[str, float]: Dictionary containing precision, recall, and F1 score.
    """
    precision = precision_score(gt_labels, pred_labels, zero_division=0)
    recall = recall_score(gt_labels, pred_labels, zero_division=0)
    f1 = f1_score(gt_labels, pred_labels, zero_division=0)
    
    return {"precision": precision, "recall": recall, "f1_score": f1}


def compute_map(
    predictions: List[Dict[str, torch.Tensor]], 
    ground_truths: List[Dict[str, torch.Tensor]], 
    iou_thresholds: Optional[List[float]] = [0.5]
) -> Dict[str, Dict[str, float]]:
    """
    Compute mAP, precision, recall, and F1 score for object detection models at multiple IoU thresholds.

    Args:
        predictions (List[Dict[str, torch.Tensor]]): Predicted bounding boxes and scores.
        ground_truths (List[Dict[str, torch.Tensor]]): Ground truth bounding boxes and labels.
        iou_thresholds (List[float], optional): List of IoU thresholds to calculate metrics at. Defaults to [0.5].

    Returns:
        Dict[str, Dict[str, float]]: Dictionary containing precision, recall, F1 score, and mAP for each IoU threshold.
    """
    metrics = {}

    for iou_threshold in iou_thresholds:
        pred_labels, gt_labels = [], []

        for pred, gt in zip(predictions, ground_truths):
            pred_boxes = pred['boxes']
            gt_boxes = gt['boxes']

            ious = compute_iou(pred_boxes, gt_boxes)
            iou_max, _ = ious.max(dim=1)

            pred_labels.extend((iou_max >= iou_threshold).float().tolist())
            gt_labels.extend([1.0] * len(gt_boxes))

        threshold_metrics = compute_precision_recall_f1(pred_labels, gt_labels)
        threshold_metrics['mAP'] = sum(pred_labels) / len(pred_labels) if pred_labels else 0.0
        metrics[f"IoU@{iou_threshold}"] = threshold_metrics

    return metrics


def compute_auc_at_fp_per_km(
    predictions: List[Dict[str, torch.Tensor]], 
    ground_truths: List[Dict[str, torch.Tensor]], 
    area: float, 
    max_fp_per_km: float = 1.0
) -> float:
    """
    Compute AUC of ROC curve, limiting to a false-positive rate per square kilometer.

    Args:
        predictions (List[Dict[str, torch.Tensor]]): List of predicted bounding boxes and scores.
        ground_truths (List[Dict[str, torch.Tensor]]): List of ground truth bounding boxes and labels.
        area (float): Area in square kilometers for false positive normalization.
        max_fp_per_km (float, optional): Max false positives allowed per square km. Defaults to 1.0.

    Returns:
        float: AUC of the ROC curve limited to the given false-positive rate.
    """
    pred_scores, gt_labels = [], []

    for pred, gt in zip(predictions, ground_truths):
        pred_scores.extend(pred['scores'].tolist())
        gt_labels.extend([1] * len(gt['boxes']))  # Ground truths are positives

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(gt_labels, pred_scores)
    fpr_per_km = fpr / area

    # Find the threshold where the false-positive rate is <= max_fp_per_km
    idx = (fpr_per_km <= max_fp_per_km).nonzero(as_tuple=True)[0][-1] if (fpr_per_km <= max_fp_per_km).any() else -1
    roc_auc = auc(fpr[:idx + 1], tpr[:idx + 1]) if idx != -1 else auc(fpr, tpr)

    return roc_auc


def evaluate_model(
    model, 
    dataloader, 
    area: float, 
    device: str = 'cuda', 
    iou_thresholds: Optional[List[float]] = [0.5]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a model and compute key metrics on a validation dataset.

    Args:
        model: The trained object detection model (e.g., DetrForObjectDetection).
        dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        area (float): Area in square kilometers for false-positive normalization.
        device (str, optional): Device for model evaluation. Defaults to 'cuda'.
        iou_thresholds (List[float], optional): List of IoU thresholds for metrics calculation. Defaults to [0.5].

    Returns:
        Dict[str, Dict[str, float]]: Metrics including precision, recall, F1 score, mAP, and AUC at 1 FP/km².
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
                pred = {"boxes": pred_boxes[i], "scores": logits[i]}
                gt = {"boxes": targets[i]["boxes"].cpu(), "labels": targets[i]["labels"].cpu()}

                all_predictions.append(pred)
                all_ground_truths.append(gt)

    metrics = compute_map(all_predictions, all_ground_truths, iou_thresholds)
    metrics["AUC_fp_per_km"] = compute_auc_at_fp_per_km(all_predictions, all_ground_truths, area)

    return metrics
