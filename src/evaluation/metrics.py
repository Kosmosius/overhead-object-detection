# src/evaluation/metrics.py

import torch
from transformers import DetrForObjectDetection
from torchvision.ops import box_iou

def compute_map(predictions, ground_truths, iou_threshold=0.5):
    """
    Compute the Mean Average Precision (mAP) for the predictions.

    Args:
        predictions (list of dict): List of predictions from the model.
        ground_truths (list of dict): List of ground truth annotations.
        iou_threshold (float): IoU threshold for considering a prediction as correct.

    Returns:
        float: Computed mAP score.
    """
    tp, fp, total_gt = 0, 0, 0

    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred['boxes']
        gt_boxes = gt['boxes']

        ious = box_iou(pred_boxes, gt_boxes)

        total_gt += len(gt_boxes)
        tp += (ious.max(dim=1)[0] > iou_threshold).sum().item()
        fp += (ious.max(dim=1)[0] <= iou_threshold).sum().item()

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / total_gt if total_gt > 0 else 0

    return precision, recall

def evaluate_model(model, dataloader, device='cuda'):
    """
    Evaluate the model on the validation dataset.

    Args:
        model (DetrForObjectDetection): The trained DETR model.
        dataloader (DataLoader): DataLoader for the validation dataset.
        device (str): Device to run the evaluation on ('cuda' or 'cpu').

    Returns:
        float: The mAP score for the model.
    """
    model.eval()
    model.to(device)

    all_predictions, all_ground_truths = [], []

    with torch.no_grad():
        for images, targets in dataloader:
            pixel_values = torch.stack(images).to(device)

            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            boxes = outputs.pred_boxes

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

    precision, recall = compute_map(all_predictions, all_ground_truths)
    return precision, recall
