# src/utils/metrics.py

import torch
from transformers import DetrForObjectDetection
from sklearn.metrics import precision_score, recall_score, f1_score
from torchvision.ops import box_iou

def compute_iou(pred_boxes, gt_boxes):
    """
    Compute Intersection over Union (IoU) between predicted and ground truth boxes.

    Args:
        pred_boxes (Tensor): Predicted bounding boxes (Nx4).
        gt_boxes (Tensor): Ground truth bounding boxes (Mx4).

    Returns:
        Tensor: IoU scores (NxM).
    """
    return box_iou(pred_boxes, gt_boxes)

def compute_map(predictions, ground_truths, iou_threshold=0.5):
    """
    Compute the Mean Average Precision (mAP) for object detection models.

    Args:
        predictions (list of dict): List of predicted bounding boxes and scores.
        ground_truths (list of dict): List of ground truth bounding boxes and labels.
        iou_threshold (float, optional): IoU threshold to consider a prediction correct. Defaults to 0.5.

    Returns:
        dict: A dictionary containing 'precision', 'recall', and 'f1_score'.
    """
    all_pred_labels = []
    all_true_labels = []
    
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

    return {"precision": precision, "recall": recall, "f1_score": f1}

def evaluate_model(model, dataloader, device='cuda', iou_threshold=0.5):
    """
    Evaluate the model and compute metrics on the validation dataset.

    Args:
        model (DetrForObjectDetection): The trained DETR model.
        dataloader (DataLoader): DataLoader for the validation dataset.
        device (str, optional): Device to run the evaluation on ('cuda' or 'cpu'). Defaults to 'cuda'.
        iou_threshold (float, optional): IoU threshold for evaluating predictions. Defaults to 0.5.

    Returns:
        dict: A dictionary containing 'precision', 'recall', and 'f1_score'.
    """
    model.eval()
    model.to(device)

    all_predictions, all_ground_truths = [], []

    with torch.no_grad():
        for images, targets in dataloader:
            pixel_values = torch.stack(images).to(device)

            outputs = model(pixel_values=pixel_values)
            pred_boxes = outputs.pred_boxes
            logits = outputs.logits

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

    metrics = compute_map(all_predictions, all_ground_truths, iou_threshold)
    return metrics
