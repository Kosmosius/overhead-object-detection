# src/utils/metrics.py

import torch
from typing import List, Dict, Optional
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import logging


def compute_coco_eval_metrics(
    predictions: List[Dict[str, torch.Tensor]],
    ground_truths: List[Dict[str, torch.Tensor]],
    iou_type: str = 'bbox',
    max_dets: List[int] = [1, 10, 100]
) -> Dict[str, float]:
    """
    Compute COCO evaluation metrics including mAP using pycocotools.

    Args:
        predictions (List[Dict[str, torch.Tensor]]): List of predictions with keys 'boxes', 'scores', and 'labels'.
        ground_truths (List[Dict[str, torch.Tensor]]): List of ground truths with keys 'boxes' and 'labels'.
        iou_type (str, optional): Type of IoU ('bbox' or 'segm'). Defaults to 'bbox'.
        max_dets (List[int], optional): Maximum detections per image for evaluation. Defaults to [1, 10, 100].

    Returns:
        Dict[str, float]: Dictionary containing COCO evaluation metrics.
    """
    # Prepare data in COCO format
    coco_gt = _prepare_coco_ground_truth(ground_truths)
    coco_dt = _prepare_coco_detections(predictions)

    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.params.maxDets = max_dets
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract metrics
    metrics = {
        'AP': coco_eval.stats[0],
        'AP50': coco_eval.stats[1],
        'AP75': coco_eval.stats[2],
        'AP_small': coco_eval.stats[3],
        'AP_medium': coco_eval.stats[4],
        'AP_large': coco_eval.stats[5],
        'AR1': coco_eval.stats[6],
        'AR10': coco_eval.stats[7],
        'AR100': coco_eval.stats[8],
        'AR_small': coco_eval.stats[9],
        'AR_medium': coco_eval.stats[10],
        'AR_large': coco_eval.stats[11],
    }

    return metrics


def _prepare_coco_ground_truth(ground_truths: List[Dict[str, torch.Tensor]]) -> COCO:
    """
    Prepare ground truth data in COCO format.

    Args:
        ground_truths (List[Dict[str, torch.Tensor]]): List of ground truths.

    Returns:
        COCO: COCO object containing ground truth annotations.
    """
    gt_annotations = []
    image_ids = []

    for idx, gt in enumerate(ground_truths):
        image_id = int(gt['image_id'].item()) if 'image_id' in gt else idx
        image_ids.append(image_id)
        boxes = gt['boxes']
        labels = gt['labels']

        for i in range(len(boxes)):
            bbox = boxes[i].tolist()
            category_id = int(labels[i].item())
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min

            annotation = {
                'id': len(gt_annotations) + 1,
                'image_id': image_id,
                'category_id': category_id,
                'bbox': [x_min, y_min, width, height],
                'area': width * height,
                'iscrowd': 0
            }
            gt_annotations.append(annotation)

    gt_coco_format = {
        'images': [{'id': img_id} for img_id in set(image_ids)],
        'annotations': gt_annotations,
        'categories': [{'id': i} for i in range(1, max(labels).item() + 1)]
    }

    coco_gt = COCO()
    coco_gt.dataset = gt_coco_format
    coco_gt.createIndex()

    return coco_gt


def _prepare_coco_detections(predictions: List[Dict[str, torch.Tensor]]) -> COCO:
    """
    Prepare detection results in COCO format.

    Args:
        predictions (List[Dict[str, torch.Tensor]]): List of predictions.

    Returns:
        COCO: COCO object containing detection results.
    """
    dt_annotations = []
    image_ids = []

    for idx, pred in enumerate(predictions):
        image_id = int(pred['image_id'].item()) if 'image_id' in pred else idx
        image_ids.append(image_id)
        boxes = pred['boxes']
        scores = pred['scores']
        labels = pred['labels']

        for i in range(len(boxes)):
            bbox = boxes[i].tolist()
            score = float(scores[i].item())
            category_id = int(labels[i].item())
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min

            annotation = {
                'image_id': image_id,
                'category_id': category_id,
                'bbox': [x_min, y_min, width, height],
                'score': score
            }
            dt_annotations.append(annotation)

    dt_coco_format = {
        'images': [{'id': img_id} for img_id in set(image_ids)],
        'annotations': dt_annotations,
        'categories': [{'id': i} for i in range(1, max(labels).item() + 1)]
    }

    coco_dt = coco_gt_load_results(dt_coco_format)

    return coco_dt


def coco_gt_load_results(coco_dt_dict: Dict) -> COCO:
    """
    Load detection results into a COCO object.

    Args:
        coco_dt_dict (Dict): Dictionary containing detection results.

    Returns:
        COCO: COCO object containing detection results.
    """
    coco_dt = COCO()
    coco_dt.dataset = coco_dt_dict
    coco_dt.createIndex()
    return coco_dt


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


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute the IoU between two sets of boxes.

    Args:
        boxes1 (torch.Tensor): Boxes in [x1, y1, x2, y2] format.
        boxes2 (torch.Tensor): Boxes in [x1, y1, x2, y2] format.

    Returns:
        torch.Tensor: IoU matrix of size (len(boxes1), len(boxes2)).
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_xmin = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_ymin = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_xmax = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_ymax = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    inter_area = (inter_xmax - inter_xmin).clamp(min=0) * (inter_ymax - inter_ymin).clamp(min=0)
    union_area = area1[:, None] + area2 - inter_area

    iou = inter_area / union_area
    return iou


def compute_precision_recall_f1(
    predictions: List[Dict[str, torch.Tensor]],
    ground_truths: List[Dict[str, torch.Tensor]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score for object detection.

    Args:
        predictions (List[Dict[str, torch.Tensor]]): List of predictions.
        ground_truths (List[Dict[str, torch.Tensor]]): List of ground truths.
        iou_threshold (float, optional): IoU threshold to consider a detection correct. Defaults to 0.5.

    Returns:
        Dict[str, float]: Dictionary containing precision, recall, and F1 score.
    """
    tp = 0
    fp = 0
    fn = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred['boxes']
        gt_boxes = gt['boxes']

        if len(pred_boxes) == 0 and len(gt_boxes) == 0:
            continue
        elif len(pred_boxes) == 0:
            fn += len(gt_boxes)
            continue
        elif len(gt_boxes) == 0:
            fp += len(pred_boxes)
            continue

        ious = compute_iou(pred_boxes, gt_boxes)
        max_ious, _ = ious.max(dim=1)

        tp += (max_ious >= iou_threshold).sum().item()
        fp += (max_ious < iou_threshold).sum().item()
        fn += max(0, len(gt_boxes) - tp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1_score': f1}


def compute_auc_at_fp_per_image(
    predictions: List[Dict[str, torch.Tensor]],
    ground_truths: List[Dict[str, torch.Tensor]],
    max_fp_per_image: float = 1.0
) -> float:
    """
    Compute the Area Under the Curve (AUC) of the ROC curve at a specified false-positive rate per image.

    Args:
        predictions (List[Dict[str, torch.Tensor]]): List of predictions.
        ground_truths (List[Dict[str, torch.Tensor]]): List of ground truths.
        max_fp_per_image (float, optional): Maximum allowed false positives per image. Defaults to 1.0.

    Returns:
        float: AUC of the ROC curve up to the specified false-positive rate.
    """
    from sklearn.metrics import roc_curve, auc

    all_scores = []
    all_labels = []

    for pred, gt in zip(predictions, ground_truths):
        scores = pred['scores'].tolist()
        labels = [1] * len(gt['boxes'])  # Assuming all ground truths are positives

        all_scores.extend(scores)
        all_labels.extend(labels)

    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    # Assuming one image, adjust if multiple images
    fpr_per_image = fpr  # In this context, fpr is already normalized per image

    idx = np.where(fpr_per_image <= max_fp_per_image)[0]
    if len(idx) == 0:
        logging.warning("No FPR values below the specified max_fp_per_image.")
        return 0.0

    roc_auc = auc(fpr_per_image[idx], tpr[idx])

    return roc_auc


def evaluate_model(
    predictions: List[Dict[str, torch.Tensor]],
    ground_truths: List[Dict[str, torch.Tensor]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate the model and compute metrics.

    Args:
        predictions (List[Dict[str, torch.Tensor]]): List of predictions.
        ground_truths (List[Dict[str, torch.Tensor]]): List of ground truths.
        iou_threshold (float, optional): IoU threshold for evaluation. Defaults to 0.5.

    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics.
    """
    metrics = compute_coco_eval_metrics(predictions, ground_truths)
    prf1 = compute_precision_recall_f1(predictions, ground_truths, iou_threshold)
    auc_score = compute_auc_at_fp_per_image(predictions, ground_truths)

    # Combine metrics
    metrics.update(prf1)
    metrics['AUC'] = auc_score

    return metrics
