# tests/unit/src/evaluation/test_metrics.py


import torch
import pytest
from src.utils.metrics import compute_iou, compute_precision_recall_f1, compute_map

def test_compute_iou():
    """Test IoU computation for simple bounding boxes."""
    pred_boxes = torch.tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
    gt_boxes = torch.tensor([[5, 5, 15, 15]])
    
    iou_scores = compute_iou(pred_boxes, gt_boxes)

    # Check that the IoU scores have the correct shape and values
    assert iou_scores.shape == (2, 1)
    assert iou_scores[0, 0].item() > 0  # IoU for overlapping boxes should be > 0
    assert iou_scores[1, 0].item() == 0  # No overlap for the second pair

def test_compute_precision_recall_f1():
    """Test precision, recall, and F1 computation."""
    pred_labels = [1, 0, 1, 0, 1]
    gt_labels = [1, 0, 0, 0, 1]

    metrics = compute_precision_recall_f1(pred_labels, gt_labels)

    # Assert the metrics return reasonable values
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.6666666666666666
    assert metrics["f1_score"] == pytest.approx(0.8, rel=1e-2)

def test_compute_map():
    """Test mean average precision (mAP) computation for simplified predictions."""
    predictions = [
        {"boxes": torch.tensor([[0, 0, 10, 10]]), "scores": torch.tensor([0.9])},
        {"boxes": torch.tensor([[10, 10, 20, 20]]), "scores": torch.tensor([0.8])}
    ]
    ground_truths = [
        {"boxes": torch.tensor([[5, 5, 15, 15]]), "labels": torch.tensor([1])},
        {"boxes": torch.tensor([[12, 12, 22, 22]]), "labels": torch.tensor([1])}
    ]

    # Test mAP calculation with IoU threshold of 0.5
    metrics = compute_map(predictions, ground_truths, iou_threshold=0.5)

    # Assert that mAP, precision, and recall are returned
    assert isinstance(metrics, dict)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "mAP" in metrics
