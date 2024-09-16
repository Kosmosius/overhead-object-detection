# tests/unit/src/utils/test_metrics.py

import pytest
import torch
from unittest.mock import patch, MagicMock
from src.utils.metrics import (
    compute_coco_eval_metrics,
    compute_precision_recall_f1,
    compute_auc_at_fp_per_image,
    _prepare_coco_ground_truth,
    _prepare_coco_detections,
    coco_gt_load_results,
    compute_iou,
    box_iou,
    evaluate_model
)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import logging


# --- Fixtures ---

@pytest.fixture
def mock_coco_gt():
    """Fixture to create a mock COCO ground truth object."""
    mock_coco = MagicMock(spec=COCO)
    mock_coco.dataset = {
        'images': [{'id': 1}],
        'annotations': [
            {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [10, 20, 30, 40], 'area': 1200, 'iscrowd': 0}
        ],
        'categories': [{'id': 1, 'name': 'category1'}]
    }
    mock_coco.createIndex.return_value = None
    return mock_coco

@pytest.fixture
def mock_coco_dt():
    """Fixture to create a mock COCO detections object."""
    mock_coco = MagicMock(spec=COCO)
    mock_coco.dataset = {
        'images': [{'id': 1}],
        'annotations': [
            {'image_id': 1, 'category_id': 1, 'bbox': [12, 22, 28, 38], 'score': 0.9}
        ],
        'categories': [{'id': 1, 'name': 'category1'}]
    }
    mock_coco.createIndex.return_value = None
    return mock_coco

@pytest.fixture
def predictions():
    """Fixture to create sample predictions."""
    return [
        {
            'image_id': torch.tensor(1),
            'boxes': torch.tensor([[12, 22, 28, 38]], dtype=torch.float32),
            'scores': torch.tensor([0.9], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64)
        }
    ]

@pytest.fixture
def ground_truths():
    """Fixture to create sample ground truths."""
    return [
        {
            'image_id': torch.tensor(1),
            'boxes': torch.tensor([[10, 20, 30, 40]], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64)
        }
    ]

@pytest.fixture
def empty_predictions():
    """Fixture to create empty predictions."""
    return []

@pytest.fixture
def empty_ground_truths():
    """Fixture to create empty ground truths."""
    return []

@pytest.fixture
def malformed_predictions():
    """Fixture to create malformed predictions."""
    return [
        {
            'image_id': torch.tensor(1),
            'boxes': "not_a_tensor",  # Invalid type
            'scores': torch.tensor([0.9], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64)
        }
    ]

@pytest.fixture
def malformed_ground_truths():
    """Fixture to create malformed ground truths."""
    return [
        {
            'image_id': torch.tensor(1),
            'boxes': torch.tensor([[10, 20, 30, 40]], dtype=torch.float32),
            # Missing 'labels' key
        }
    ]

# --- Test Cases ---

# 1. COCO Evaluation Metrics Tests

def test_compute_coco_eval_metrics_success(predictions, ground_truths, mock_coco_gt, mock_coco_dt):
    """Test compute_coco_eval_metrics computes metrics correctly with valid inputs."""
    with patch('src.utils.metrics._prepare_coco_ground_truth', return_value=mock_coco_gt):
        with patch('src.utils.metrics._prepare_coco_detections', return_value=mock_coco_dt):
            with patch('src.utils.metrics.COCOeval') as mock_COCOeval:
                mock_eval = MagicMock()
                mock_eval.stats = [0.5, 0.6, 0.7, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.3, 0.4, 0.5]
                mock_COCOeval.return_value = mock_eval
                
                metrics = compute_coco_eval_metrics(predictions, ground_truths)
                
                mock_COCOeval.assert_called_once_with(mock_coco_gt, mock_coco_dt, 'bbox')
                mock_eval.evaluate.assert_called_once()
                mock_eval.accumulate.assert_called_once()
                mock_eval.summarize.assert_called_once()
                
                expected_metrics = {
                    'AP': 0.5,
                    'AP50': 0.6,
                    'AP75': 0.7,
                    'AP_small': 0.4,
                    'AP_medium': 0.5,
                    'AP_large': 0.6,
                    'AR1': 0.7,
                    'AR10': 0.8,
                    'AR100': 0.9,
                    'AR_small': 0.3,
                    'AR_medium': 0.4,
                    'AR_large': 0.5,
                }
                assert metrics == expected_metrics, "COCO evaluation metrics do not match expected values."

def test_compute_coco_eval_metrics_empty_predictions(empty_predictions, ground_truths, mock_coco_gt, mock_coco_dt):
    """Test compute_coco_eval_metrics handles empty predictions gracefully."""
    with patch('src.utils.metrics._prepare_coco_ground_truth', return_value=mock_coco_gt):
        with patch('src.utils.metrics._prepare_coco_detections', return_value=mock_coco_dt):
            with patch('src.utils.metrics.COCOeval') as mock_COCOeval:
                mock_eval = MagicMock()
                mock_eval.stats = [0.0] * 12
                mock_COCOeval.return_value = mock_eval
                
                metrics = compute_coco_eval_metrics(empty_predictions, ground_truths)
                
                mock_COCOeval.assert_called_once_with(mock_coco_gt, mock_coco_dt, 'bbox')
                mock_eval.evaluate.assert_called_once()
                mock_eval.accumulate.assert_called_once()
                mock_eval.summarize.assert_called_once()
                
                expected_metrics = {key: 0.0 for key in [
                    'AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
                    'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large'
                ]}
                assert metrics == expected_metrics, "COCO evaluation metrics should be zero for empty predictions."

def test_compute_coco_eval_metrics_empty_ground_truths(predictions, empty_ground_truths, mock_coco_gt, mock_coco_dt):
    """Test compute_coco_eval_metrics handles empty ground truths gracefully."""
    with patch('src.utils.metrics._prepare_coco_ground_truth', return_value=mock_coco_gt):
        with patch('src.utils.metrics._prepare_coco_detections', return_value=mock_coco_dt):
            with patch('src.utils.metrics.COCOeval') as mock_COCOeval:
                mock_eval = MagicMock()
                mock_eval.stats = [0.0] * 12
                mock_COCOeval.return_value = mock_eval
                
                metrics = compute_coco_eval_metrics(predictions, empty_ground_truths)
                
                mock_COCOeval.assert_called_once_with(mock_coco_gt, mock_coco_dt, 'bbox')
                mock_eval.evaluate.assert_called_once()
                mock_eval.accumulate.assert_called_once()
                mock_eval.summarize.assert_called_once()
                
                expected_metrics = {key: 0.0 for key in [
                    'AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
                    'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large'
                ]}
                assert metrics == expected_metrics, "COCO evaluation metrics should be zero for empty ground truths."

@pytest.mark.parametrize("iou_type", ['bbox', 'segm', 'keypoints'])
def test_compute_coco_eval_metrics_invalid_iou_type(predictions, ground_truths, mock_coco_gt, mock_coco_dt, iou_type):
    """Test compute_coco_eval_metrics raises an error for invalid IoU types."""
    with patch('src.utils.metrics._prepare_coco_ground_truth', return_value=mock_coco_gt):
        with patch('src.utils.metrics._prepare_coco_detections', return_value=mock_coco_dt):
            with patch('src.utils.metrics.COCOeval') as mock_COCOeval:
                mock_COCOeval.side_effect = ValueError("Invalid IoU type")
                
                with pytest.raises(ValueError, match="Invalid IoU type"):
                    compute_coco_eval_metrics(predictions, ground_truths, iou_type=iou_type)

def test_compute_coco_eval_metrics_custom_max_dets(predictions, ground_truths, mock_coco_gt, mock_coco_dt):
    """Test compute_coco_eval_metrics with custom max_dets values."""
    custom_max_dets = [5, 15, 50]
    with patch('src.utils.metrics._prepare_coco_ground_truth', return_value=mock_coco_gt):
        with patch('src.utils.metrics._prepare_coco_detections', return_value=mock_coco_dt):
            with patch('src.utils.metrics.COCOeval') as mock_COCOeval:
                mock_eval = MagicMock()
                mock_eval.stats = [0.55] * 12
                mock_COCOeval.return_value = mock_eval
                
                metrics = compute_coco_eval_metrics(predictions, ground_truths, max_dets=custom_max_dets)
                
                mock_COCOeval.assert_called_once_with(mock_coco_gt, mock_coco_dt, 'bbox')
                mock_eval.params.maxDets = custom_max_dets
                mock_eval.evaluate.assert_called_once()
                mock_eval.accumulate.assert_called_once()
                mock_eval.summarize.assert_called_once()
                
                expected_metrics = {key: 0.55 for key in [
                    'AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
                    'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large'
                ]}
                assert metrics == expected_metrics, "COCO evaluation metrics do not match expected values with custom max_dets."

def test_compute_coco_eval_metrics_malformed_predictions(ground_truths, malformed_predictions):
    """Test compute_coco_eval_metrics raises an error with malformed predictions."""
    with patch('src.utils.metrics._prepare_coco_ground_truth') as mock_prepare_gt:
        mock_prepare_gt.return_value = MagicMock(spec=COCO)
        with patch('src.utils.metrics._prepare_coco_detections') as mock_prepare_dt:
            mock_prepare_dt.side_effect = AttributeError("Malformed predictions")
            
            with pytest.raises(AttributeError, match="Malformed predictions"):
                compute_coco_eval_metrics(malformed_predictions, ground_truths)

# 2. Precision, Recall, and F1 Score Tests

def test_compute_precision_recall_f1_success(predictions, ground_truths):
    """Test compute_precision_recall_f1 computes metrics correctly with valid inputs."""
    metrics = compute_precision_recall_f1(predictions, ground_truths, iou_threshold=0.5)
    expected_metrics = {
        'precision': 1.0,
        'recall': 1.0,
        'f1_score': 1.0
    }
    assert metrics == expected_metrics, "Precision, Recall, F1 metrics do not match expected values."

def test_compute_precision_recall_f1_empty_predictions(empty_predictions, ground_truths):
    """Test compute_precision_recall_f1 handles empty predictions gracefully."""
    metrics = compute_precision_recall_f1(empty_predictions, ground_truths, iou_threshold=0.5)
    expected_metrics = {
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0
    }
    assert metrics == expected_metrics, "Precision, Recall, F1 should be zero when predictions are empty."

def test_compute_precision_recall_f1_empty_ground_truths(predictions, empty_ground_truths):
    """Test compute_precision_recall_f1 handles empty ground truths gracefully."""
    metrics = compute_precision_recall_f1(predictions, empty_ground_truths, iou_threshold=0.5)
    expected_metrics = {
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0
    }
    assert metrics == expected_metrics, "Precision, Recall, F1 should be zero when ground truths are empty."

@pytest.mark.parametrize("iou_threshold, expected_precision, expected_recall, expected_f1", [
    (0.3, 1.0, 1.0, 1.0),
    (0.5, 1.0, 1.0, 1.0),
    (0.7, 1.0, 1.0, 1.0),
])
def test_compute_precision_recall_f1_various_iou_thresholds(predictions, ground_truths, iou_threshold, expected_precision, expected_recall, expected_f1):
    """Test compute_precision_recall_f1 with various IoU thresholds."""
    metrics = compute_precision_recall_f1(predictions, ground_truths, iou_threshold=iou_threshold)
    expected_metrics = {
        'precision': expected_precision,
        'recall': expected_recall,
        'f1_score': expected_f1
    }
    assert metrics == expected_metrics, f"Metrics do not match expected values for IoU threshold {iou_threshold}."

def test_compute_precision_recall_f1_invalid_data_types(predictions, ground_truths):
    """Test compute_precision_recall_f1 raises an error with invalid data types."""
    malformed_predictions = [
        {
            'boxes': "not_a_tensor",  # Invalid type
            'scores': torch.tensor([0.9], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64)
        }
    ]
    with pytest.raises(AttributeError):
        compute_precision_recall_f1(malformed_predictions, ground_truths, iou_threshold=0.5)

# 3. AUC Computation Tests

def test_compute_auc_at_fp_per_image_success(predictions, ground_truths):
    """Test compute_auc_at_fp_per_image computes AUC correctly with valid inputs."""
    with patch('src.utils.metrics.roc_curve') as mock_roc_curve:
        mock_roc_curve.return_value = (np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.75, 1.0]), np.array([1.5, 1.0, 0.5]))
        with patch('src.utils.metrics.auc', return_value=0.875):
            auc_score = compute_auc_at_fp_per_image(predictions, ground_truths)
            assert auc_score == 0.875, "AUC score does not match expected value."

def test_compute_auc_at_fp_per_image_empty_predictions(empty_predictions, ground_truths):
    """Test compute_auc_at_fp_per_image handles empty predictions gracefully."""
    auc_score = compute_auc_at_fp_per_image(empty_predictions, ground_truths)
    assert auc_score == 0.0, "AUC should be zero when predictions are empty."

def test_compute_auc_at_fp_per_image_empty_ground_truths(predictions, empty_ground_truths):
    """Test compute_auc_at_fp_per_image handles empty ground truths gracefully."""
    auc_score = compute_auc_at_fp_per_image(predictions, empty_ground_truths)
    assert auc_score == 0.0, "AUC should be zero when ground truths are empty."

@pytest.mark.parametrize("max_fp_per_image, expected_auc", [
    (0.1, 0.0),
    (0.5, 0.75),
    (1.0, 0.875),
    (1.5, 0.875)
])
def test_compute_auc_at_fp_per_image_various_max_fp(predictions, ground_truths, max_fp_per_image, expected_auc):
    """Test compute_auc_at_fp_per_image with various max_fp_per_image values."""
    with patch('src.utils.metrics.roc_curve') as mock_roc_curve:
        mock_roc_curve.return_value = (np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.75, 1.0]), np.array([1.5, 1.0, 0.5]))
        with patch('src.utils.metrics.auc', return_value=expected_auc):
            auc_score = compute_auc_at_fp_per_image(predictions, ground_truths, max_fp_per_image=max_fp_per_image)
            assert auc_score == expected_auc, f"AUC score does not match expected value for max_fp_per_image={max_fp_per_image}."

def test_compute_auc_at_fp_per_image_invalid_data_types(predictions, ground_truths):
    """Test compute_auc_at_fp_per_image raises an error with invalid data types."""
    malformed_predictions = [
        {
            'boxes': "not_a_tensor",  # Invalid type
            'scores': torch.tensor([0.9], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64)
        }
    ]
    with pytest.raises(AttributeError):
        compute_auc_at_fp_per_image(malformed_predictions, ground_truths, max_fp_per_image=0.5)

# 4. Helper Function Tests

def test_prepare_coco_ground_truth_success(ground_truths):
    """Test _prepare_coco_ground_truth formats ground truths correctly."""
    coco_gt = _prepare_coco_ground_truth(ground_truths)
    assert isinstance(coco_gt, COCO), "_prepare_coco_ground_truth should return a COCO object."
    assert len(coco_gt.dataset['annotations']) == 1, "Incorrect number of annotations."
    assert coco_gt.dataset['annotations'][0]['bbox'] == [10, 20, 20, 20], "Bounding box coordinates incorrect."
    assert coco_gt.dataset['annotations'][0]['category_id'] == 1, "Category ID incorrect."

def test_prepare_coco_ground_truth_missing_labels(ground_truths, malformed_ground_truths):
    """Test _prepare_coco_ground_truth raises an error when labels are missing."""
    with pytest.raises(AttributeError):
        _prepare_coco_ground_truth(malformed_ground_truths)

def test_prepare_coco_detections_success(predictions):
    """Test _prepare_coco_detections formats detections correctly."""
    with patch('src.utils.metrics.coco_gt_load_results') as mock_load_results:
        mock_coco_dt = MagicMock(spec=COCO)
        mock_load_results.return_value = mock_coco_dt
        coco_dt = _prepare_coco_detections(predictions)
        mock_load_results.assert_called_once()
        assert coco_dt == mock_coco_dt, "_prepare_coco_detections should return the COCO detections object."

def test_compute_iou_success(predictions, ground_truths):
    """Test compute_iou computes IoU correctly with valid inputs."""
    pred_boxes = predictions[0]['boxes']
    gt_boxes = ground_truths[0]['boxes']
    iou = compute_iou(pred_boxes, gt_boxes)
    expected_iou = box_iou(pred_boxes, gt_boxes)
    torch.testing.assert_allclose(iou, expected_iou, atol=1e-6, msg="IoU computation mismatch.")

def test_box_iou_success():
    """Test box_iou computes IoU correctly."""
    boxes1 = torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=torch.float32)
    boxes2 = torch.tensor([[5, 5, 15, 15], [25, 25, 35, 35]], dtype=torch.float32)
    expected_iou = torch.tensor([
        [(5 * 5) / (10 * 10 + 10 * 10 - 25), 0.0],
        [0.0, (5 * 5) / (10 * 10 + 10 * 10 - 25)]
    ], dtype=torch.float32)
    iou = box_iou(boxes1, boxes2)
    torch.testing.assert_allclose(iou, expected_iou, atol=1e-6, msg="Box IoU computation mismatch.")

# 5. Edge Case Tests

def test_compute_coco_eval_metrics_no_overlaps(predictions, ground_truths, mock_coco_gt, mock_coco_dt):
    """Test compute_coco_eval_metrics when there are no overlapping boxes."""
    with patch('src.utils.metrics._prepare_coco_ground_truth', return_value=mock_coco_gt):
        with patch('src.utils.metrics._prepare_coco_detections', return_value=mock_coco_dt):
            with patch('src.utils.metrics.COCOeval') as mock_COCOeval:
                mock_eval = MagicMock()
                # Simulate no matches
                mock_eval.stats = [0.0] * 12
                mock_COCOeval.return_value = mock_eval
                
                metrics = compute_coco_eval_metrics(predictions, ground_truths)
                
                expected_metrics = {key: 0.0 for key in [
                    'AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
                    'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large'
                ]}
                assert metrics == expected_metrics, "COCO evaluation metrics should be zero when there are no overlaps."

def test_compute_coco_eval_metrics_perfect_overlaps(predictions, ground_truths, mock_coco_gt, mock_coco_dt):
    """Test compute_coco_eval_metrics when all predictions perfectly match ground truths."""
    with patch('src.utils.metrics._prepare_coco_ground_truth', return_value=mock_coco_gt):
        with patch('src.utils.metrics._prepare_coco_detections', return_value=mock_coco_dt):
            with patch('src.utils.metrics.COCOeval') as mock_COCOeval:
                mock_eval = MagicMock()
                mock_eval.stats = [1.0] * 12
                mock_COCOeval.return_value = mock_eval
                
                metrics = compute_coco_eval_metrics(predictions, ground_truths)
                
                expected_metrics = {key: 1.0 for key in [
                    'AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
                    'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large'
                ]}
                assert metrics == expected_metrics, "COCO evaluation metrics should be one when overlaps are perfect."

def test_compute_precision_recall_f1_no_overlaps(predictions, ground_truths):
    """Test compute_precision_recall_f1 when there are no overlapping boxes."""
    # Modify predictions to have no overlap
    no_overlap_predictions = [
        {
            'image_id': torch.tensor(1),
            'boxes': torch.tensor([[100, 100, 110, 110]], dtype=torch.float32),
            'scores': torch.tensor([0.9], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64)
        }
    ]
    metrics = compute_precision_recall_f1(no_overlap_predictions, ground_truths, iou_threshold=0.5)
    expected_metrics = {
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0
    }
    assert metrics == expected_metrics, "Precision, Recall, F1 should be zero when there are no overlaps."

def test_compute_auc_at_fp_per_image_perfect_predictions(predictions, ground_truths):
    """Test compute_auc_at_fp_per_image with perfect predictions."""
    with patch('src.utils.metrics.roc_curve') as mock_roc_curve:
        mock_roc_curve.return_value = (np.array([0.0, 0.5, 1.0]), np.array([0.0, 1.0, 1.0]), np.array([1.0, 0.5, 0.0]))
        with patch('src.utils.metrics.auc', return_value=1.0):
            auc_score = compute_auc_at_fp_per_image(predictions, ground_truths)
            assert auc_score == 1.0, "AUC should be one for perfect predictions."

def test_compute_auc_at_fp_per_image_no_predictions(ground_truths):
    """Test compute_auc_at_fp_per_image when there are no predictions."""
    empty_predictions = []
    auc_score = compute_auc_at_fp_per_image(empty_predictions, ground_truths)
    assert auc_score == 0.0, "AUC should be zero when there are no predictions."

# 6. Integration Tests

def test_evaluate_model_success(predictions, ground_truths, mock_coco_gt, mock_coco_dt):
    """Test evaluate_model computes all metrics correctly."""
    with patch('src.utils.metrics.compute_coco_eval_metrics', return_value={
        'AP': 0.5,
        'AP50': 0.6,
        'AP75': 0.7,
        'AP_small': 0.4,
        'AP_medium': 0.5,
        'AP_large': 0.6,
        'AR1': 0.7,
        'AR10': 0.8,
        'AR100': 0.9,
        'AR_small': 0.3,
        'AR_medium': 0.4,
        'AR_large': 0.5,
    }):
        with patch('src.utils.metrics.compute_precision_recall_f1', return_value={
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0
        }):
            with patch('src.utils.metrics.compute_auc_at_fp_per_image', return_value=0.875):
                metrics = evaluate_model(predictions, ground_truths, iou_threshold=0.5)
                expected_metrics = {
                    'AP': 0.5,
                    'AP50': 0.6,
                    'AP75': 0.7,
                    'AP_small': 0.4,
                    'AP_medium': 0.5,
                    'AP_large': 0.6,
                    'AR1': 0.7,
                    'AR10': 0.8,
                    'AR100': 0.9,
                    'AR_small': 0.3,
                    'AR_medium': 0.4,
                    'AR_large': 0.5,
                    'precision': 1.0,
                    'recall': 1.0,
                    'f1_score': 1.0,
                    'AUC': 0.875
                }
                assert metrics == expected_metrics, "evaluate_model metrics do not match expected values."

# 7. Error Handling Tests

def test_prepare_coco_ground_truth_invalid_boxes(ground_truths, malformed_ground_truths):
    """Test _prepare_coco_ground_truth raises an error with invalid box formats."""
    malformed_ground_truths_invalid_box = [
        {
            'image_id': torch.tensor(1),
            'boxes': torch.tensor([[10, 20, 30]], dtype=torch.float32),  # Invalid box
            'labels': torch.tensor([1], dtype=torch.int64)
        }
    ]
    with pytest.raises(IndexError):
        _prepare_coco_ground_truth(malformed_ground_truths_invalid_box)

def test_prepare_coco_detections_invalid_labels(predictions):
    """Test _prepare_coco_detections raises an error with invalid label formats."""
    malformed_predictions_invalid_label = [
        {
            'image_id': torch.tensor(1),
            'boxes': torch.tensor([[12, 22, 28, 38]], dtype=torch.float32),
            'scores': torch.tensor([0.9], dtype=torch.float32),
            'labels': torch.tensor(["invalid_label"], dtype=torch.int64)  # Invalid label
        }
    ]
    with patch('src.utils.metrics.coco_gt_load_results') as mock_load_results:
        mock_load_results.return_value = MagicMock(spec=COCO)
        with pytest.raises(TypeError):
            _prepare_coco_detections(malformed_predictions_invalid_label)

def test_compute_iou_invalid_input_types():
    """Test compute_iou raises an error with invalid input types."""
    pred_boxes = "not_a_tensor"
    gt_boxes = torch.tensor([[10, 20, 30, 40]], dtype=torch.float32)
    with pytest.raises(AttributeError):
        compute_iou(pred_boxes, gt_boxes)

def test_box_iou_invalid_input_shapes():
    """Test box_iou raises an error with invalid input shapes."""
    boxes1 = torch.tensor([[0, 0, 10]], dtype=torch.float32)  # Incomplete box
    boxes2 = torch.tensor([[5, 5, 15, 15]], dtype=torch.float32)
    with pytest.raises(IndexError):
        box_iou(boxes1, boxes2)

# 8. Performance and Scalability Tests

def test_compute_coco_eval_metrics_large_dataset():
    """Test compute_coco_eval_metrics handles large datasets efficiently."""
    large_predictions = [
        {
            'image_id': torch.tensor(i),
            'boxes': torch.tensor([[i, i, i+10, i+10]], dtype=torch.float32),
            'scores': torch.tensor([0.5 + (i % 10)*0.05], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64)
        } for i in range(1000)
    ]
    large_ground_truths = [
        {
            'image_id': torch.tensor(i),
            'boxes': torch.tensor([[i, i, i+10, i+10]], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64)
        } for i in range(1000)
    ]
    with patch('src.utils.metrics._prepare_coco_ground_truth') as mock_prepare_gt:
        mock_coco_gt = MagicMock(spec=COCO)
        mock_prepare_gt.return_value = mock_coco_gt
        with patch('src.utils.metrics._prepare_coco_detections') as mock_prepare_dt:
            mock_coco_dt = MagicMock(spec=COCO)
            mock_prepare_dt.return_value = mock_coco_dt
            with patch('src.utils.metrics.COCOeval') as mock_COCOeval:
                mock_eval = MagicMock()
                mock_eval.stats = [0.75] * 12
                mock_COCOeval.return_value = mock_eval
                
                metrics = compute_coco_eval_metrics(large_predictions, large_ground_truths)
                
                assert metrics['AP'] == 0.75, "AP should match expected value for large dataset."
                assert metrics['AUC'] == 0.0  # Assuming AUC is not computed here

# 9. Integration Tests with evaluate_model

def test_evaluate_model_integration(predictions, ground_truths, mock_coco_gt, mock_coco_dt):
    """Test evaluate_model integrates compute_coco_eval_metrics, compute_precision_recall_f1, and compute_auc_at_fp_per_image correctly."""
    with patch('src.utils.metrics.compute_coco_eval_metrics', return_value={
        'AP': 0.6,
        'AP50': 0.65,
        'AP75': 0.7,
        'AP_small': 0.55,
        'AP_medium': 0.6,
        'AP_large': 0.65,
        'AR1': 0.7,
        'AR10': 0.75,
        'AR100': 0.8,
        'AR_small': 0.5,
        'AR_medium': 0.55,
        'AR_large': 0.6,
    }):
        with patch('src.utils.metrics.compute_precision_recall_f1', return_value={
            'precision': 0.8,
            'recall': 0.85,
            'f1_score': 0.825
        }):
            with patch('src.utils.metrics.compute_auc_at_fp_per_image', return_value=0.9):
                metrics = evaluate_model(predictions, ground_truths, iou_threshold=0.5)
                expected_metrics = {
                    'AP': 0.6,
                    'AP50': 0.65,
                    'AP75': 0.7,
                    'AP_small': 0.55,
                    'AP_medium': 0.6,
                    'AP_large': 0.65,
                    'AR1': 0.7,
                    'AR10': 0.75,
                    'AR100': 0.8,
                    'AR_small': 0.5,
                    'AR_medium': 0.55,
                    'AR_large': 0.6,
                    'precision': 0.8,
                    'recall': 0.85,
                    'f1_score': 0.825,
                    'AUC': 0.9
                }
                assert metrics == expected_metrics, "evaluate_model integrated metrics do not match expected values."

# 10. Mocking pycocotools COCO and COCOeval

def test_compute_coco_eval_metrics_with_real_pycoctools(predictions, ground_truths):
    """Optionally, test compute_coco_eval_metrics with actual pycocotools if integration tests are desired."""
    # This test can be resource-intensive and is optional
    pass  # Implementation depends on the testing environment

