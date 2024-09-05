# src/utils/metrics.py

import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from torchvision.ops import box_iou
from typing import List, Dict, Tuple
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


def compute_map(predictions: List[Dict[str, torch.Tensor]], 
                ground_truths: List[Dict[str, torch.Tensor]], 
                iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute the Mean Average Precision (mAP), precision, recall, and F1 score for object detection models.

    Args:
        predictions (list of dict): List of predicted bounding boxes and scores.
        ground_truths (list of dict): List of ground truth bounding boxes and labels.
        iou_threshold (float, optional): IoU threshold to consider a prediction correct. Defaults to 0.5.

    Returns:
        dict: A dictionary containing 'precision', 'recall', 'f1_score', and 'map' (Mean Average Precision).
    """
    all_pred_labels = []
    all_true_labels = []
    
    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred['boxes']
        gt_boxes = gt['boxes']
        
        # Compute IoU and identify which predictions are valid
        ious = compute_iou(pred_boxes, gt_boxes)
        iou_max, _ = ious.max(dim=1)

        pred_labels = (iou_max >= iou_threshold).float().tolist()
        gt_labels = [1.0] * len(gt_boxes)  # Ground truths are always considered positives
        
        all_pred_labels.extend(pred_labels)
        all_true_labels.extend(gt_labels)
    
    precision = precision_score(all_true_labels, all_pred_labels, zero_division=0)
    recall = recall_score(all_true_labels, all_pred_labels, zero_division=0)
    f1 = f1_score(all_true_labels, all_pred_labels, zero_division=0)

    # HuggingFace Metric for mAP
    metric = load_metric('mean_average_precision')
    map_score = metric.compute(predictions=all_pred_labels, references=all_true_labels)

    return {"precision": precision, "recall": recall, "f1_score": f1, "map": map_score['mean_average_precision']}


def compute_auc_at_fp_per_km(predictions: List[Dict[str, torch.Tensor]],
                             ground_truths: List[Dict[str, torch.Tensor]],
                             area: float,
                             max_fp_per_km: float = 1.0) -> float:
    """
    Compute the Area Under the Curve (AUC) of the Receiver Operator Curve (ROC) at 1 false-positive per square km.

    Args:
        predictions (list of dict): List of predicted bounding boxes and scores.
        ground_truths (list of dict): List of ground truth bounding boxes and labels.
        area (float): The area in square kilometers to normalize the false positives.
        max_fp_per_km (float, optional): Maximum allowed false positives per square km. Defaults to 1.0.

    Returns:
        float: The AUC of the ROC curve at the given constraint.
    """
    all_pred_scores = []
    all_true_labels = []
    
    for pred, gt in zip(predictions, ground_truths):
        pred_scores = pred['scores'].tolist()
        gt_labels = [1] * len(gt['boxes'])  # All ground truths are positives

        all_pred_scores.extend(pred_scores)
        all_true_labels.extend(gt_labels)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(all_true_labels, all_pred_scores)

    # Convert false positives to false positives per km²
    fpr_per_km = fpr / area

    # Find the point where the false-positive rate is closest to the max_fp_per_km constraint
    idx = (fpr_per_km <= max_fp_per_km).nonzero(as_tuple=True)[0][-1] if (fpr_per_km <= max_fp_per_km).any() else -1

    # Compute the AUC up to that point
    roc_auc = auc(fpr[:idx + 1], tpr[:idx + 1])

    return roc_auc


def evaluate_model(model, dataloader, area: float, device='cuda', iou_threshold=0.5) -> Dict[str, float]:
    """
    Evaluate the model and compute metrics on the validation dataset.

    Args:
        model: The trained object detection model (e.g., DetrForObjectDetection).
        dataloader (DataLoader): DataLoader for the validation dataset.
        area (float): Area in square kilometers for false-positive normalization.
        device (str, optional): Device to run the evaluation on ('cuda' or 'cpu'). Defaults to 'cuda'.
        iou_threshold (float, optional): IoU threshold for evaluating predictions. Defaults to 0.5.

    Returns:
        dict: A dictionary containing 'precision', 'recall', 'f1_score', 'map', and 'auc_fp_per_km'.
    """
    model.eval()
    model.to(device)

    all_predictions, all_ground_truths = [], []

    with torch.no_grad():
        for batch in dataloader:
            images, targets = batch
            pixel_values = torch.stack(images).to(device)

            # Run inference
            outputs = model(pixel_values=pixel_values)
            pred_boxes = outputs.pred_boxes
            logits = outputs.logits

            # Process each image in the batch
            for i in range(len(images)):
                pred = {
                    "boxes": pred_boxes[i].cpu(),
                    "scores": logits[i].cpu()
                }
                gt = {
                    "boxes": targets[i]["boxes"].cpu(),
                    "labels": targets[i]["labels"].cpu()
                }

                all_predictions.append(pred)
                all_ground_truths.append(gt)

    # Compute basic metrics (precision, recall, f1, map)
    metrics = compute_map(all_predictions, all_ground_truths, iou_threshold)

    # Compute AUC at 1 false positive per square km
    auc_fp_per_km = compute_auc_at_fp_per_km(all_predictions, all_ground_truths, area)
    metrics["auc_fp_per_km"] = auc_fp_per_km

    return metrics
