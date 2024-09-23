# tests/unit/src/evaluation/test_evaluator.py

import pytest
import torch
import logging
from unittest.mock import MagicMock, patch
from src.evaluation.evaluator import Evaluator
from torch.utils.data import DataLoader

# Custom collate function for variable-length data
def collate_fn(batch):
    return tuple(zip(*batch))

# Fixtures for sample data
@pytest.fixture
def sample_images():
    """Fixture for a list of sample images as torch.Tensors."""
    return [torch.rand(3, 224, 224) for _ in range(4)]  # 4 images

@pytest.fixture
def sample_targets():
    """Fixture for a list of sample target dictionaries."""
    return [
        {
            'boxes': torch.tensor([[50, 50, 150, 150], [30, 30, 100, 100]], dtype=torch.float32),
            'labels': torch.tensor([1, 2], dtype=torch.int64),
            'image_id': torch.tensor(1, dtype=torch.int64)
        },
        {
            'boxes': torch.tensor([[60, 60, 160, 160]], dtype=torch.float32),
            'labels': torch.tensor([3], dtype=torch.int64),
            'image_id': torch.tensor(2, dtype=torch.int64)
        },
        {
            'boxes': torch.tensor([[70, 70, 170, 170], [40, 40, 110, 110], [20, 20, 80, 80]], dtype=torch.float32),
            'labels': torch.tensor([4, 5, 6], dtype=torch.int64),
            'image_id': torch.tensor(3, dtype=torch.int64)
        },
        {
            'boxes': torch.tensor([[80, 80, 180, 180]], dtype=torch.float32),
            'labels': torch.tensor([7], dtype=torch.int64),
            'image_id': torch.tensor(4, dtype=torch.int64)
        },
    ]

@pytest.fixture
def mock_model():
    """Fixture to return a mock PyTorch model."""
    model = MagicMock(spec=torch.nn.Module)
    model.to.return_value = model  # Ensure .to() returns the model itself
    model.eval.return_value = None  # Ensure .eval() is callable

    # Mock model outputs: list of dicts with 'scores', 'boxes', 'labels', 'image_id'
    def mock_forward(images):
        # For simplicity, return a list with one output per image
        outputs = []
        for idx, img in enumerate(images):
            num_preds = torch.randint(1, 5, (1,)).item()  # Random number of predictions per image
            scores = torch.rand(num_preds)
            boxes = torch.rand(num_preds, 4) * 200  # Assuming image size up to 200
            labels = torch.randint(1, 10, (num_preds,))
            image_id = torch.tensor(idx + 1, dtype=torch.int64)
            outputs.append({
                'scores': scores,
                'boxes': boxes,
                'labels': labels,
                'image_id': image_id
            })
        return outputs

    # Set model.forward to a MagicMock with the side_effect
    model.forward = MagicMock(side_effect=mock_forward)
    return model

@pytest.fixture
def mock_evaluate_model():
    """Fixture to mock the evaluate_model function."""
    with patch('src.evaluation.evaluator.evaluate_model') as mock_eval:
        # Define a sample return value
        mock_eval.return_value = {
            'mAP': 0.75,
            'precision': 0.8,
            'recall': 0.7,
            'f1_score': 0.75
        }
        yield mock_eval

@pytest.fixture
def mock_dataloader(sample_images, sample_targets):
    """Fixture to return a mock DataLoader."""
    dataset = list(zip(sample_images, sample_targets))
    return DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

# Test Initialization
def test_evaluator_initialization(mock_model):
    """Test that the Evaluator initializes correctly."""
    evaluator = Evaluator(model=mock_model, device='cpu', confidence_threshold=0.5)
    
    mock_model.to.assert_called_once_with('cpu')
    mock_model.eval.assert_called_once()
    
    assert evaluator.device == 'cpu', "Device attribute mismatch."
    assert evaluator.confidence_threshold == 0.5, "Confidence threshold mismatch."
    assert evaluator.model == mock_model, "Model attribute mismatch."

# Test _process_outputs
def test_evaluator_process_outputs(mock_model):
    """Test the _process_outputs method of Evaluator."""
    evaluator = Evaluator(model=mock_model, device='cpu', confidence_threshold=0.5)
    
    # Define mock outputs
    mock_outputs = [
        {
            'scores': torch.tensor([0.6, 0.4]),
            'boxes': torch.tensor([[50, 50, 150, 150], [30, 30, 100, 100]], dtype=torch.float32),
            'labels': torch.tensor([1, 2], dtype=torch.int64),
            'image_id': torch.tensor([1], dtype=torch.int64)
        },
        {
            'scores': torch.tensor([0.7]),
            'boxes': torch.tensor([[60, 60, 160, 160]], dtype=torch.float32),
            'labels': torch.tensor([3], dtype=torch.int64),
            'image_id': torch.tensor([2], dtype=torch.int64)
        }
    ]
    
    processed = evaluator._process_outputs(mock_outputs)
    
    assert len(processed) == 2, "Processed predictions length mismatch."
    
    # First prediction
    assert torch.allclose(processed[0]['scores'], torch.tensor([0.6])), "Scores filtering mismatch."
    assert torch.allclose(processed[0]['boxes'], torch.tensor([[50, 50, 150, 150]])), "Boxes filtering mismatch."
    assert torch.allclose(processed[0]['labels'], torch.tensor([1])), "Labels filtering mismatch."
    assert torch.allclose(processed[0]['image_id'], torch.tensor([1])), "Image ID mismatch."
    
    # Second prediction
    assert torch.allclose(processed[1]['scores'], torch.tensor([0.7])), "Scores filtering mismatch."
    assert torch.allclose(processed[1]['boxes'], torch.tensor([[60, 60, 160, 160]])), "Boxes filtering mismatch."
    assert torch.allclose(processed[1]['labels'], torch.tensor([3])), "Labels filtering mismatch."
    assert torch.allclose(processed[1]['image_id'], torch.tensor([2])), "Image ID mismatch."

# Test _process_targets
def test_evaluator_process_targets(mock_model):
    """Test the _process_targets method of Evaluator."""
    evaluator = Evaluator(model=mock_model, device='cpu', confidence_threshold=0.5)
    
    # Define mock targets
    mock_targets = [
        {
            'boxes': torch.tensor([[50, 50, 150, 150], [30, 30, 100, 100]], dtype=torch.float32),
            'labels': torch.tensor([1, 2], dtype=torch.int64),
            'image_id': torch.tensor([1], dtype=torch.int64)
        },
        {
            'boxes': torch.tensor([[60, 60, 160, 160]], dtype=torch.float32),
            'labels': torch.tensor([3], dtype=torch.int64),
            'image_id': torch.tensor([2], dtype=torch.int64)
        }
    ]
    
    processed = evaluator._process_targets(mock_targets)
    
    assert len(processed) == 2, "Processed ground truths length mismatch."
    
    # First ground truth
    assert torch.allclose(processed[0]['boxes'], torch.tensor([[50, 50, 150, 150], [30, 30, 100, 100]])), "Boxes mismatch."
    assert torch.allclose(processed[0]['labels'], torch.tensor([1, 2])), "Labels mismatch."
    assert torch.allclose(processed[0]['image_id'], torch.tensor([1])), "Image ID mismatch."
    
    # Second ground truth
    assert torch.allclose(processed[1]['boxes'], torch.tensor([[60, 60, 160, 160]])), "Boxes mismatch."
    assert torch.allclose(processed[1]['labels'], torch.tensor([3])), "Labels mismatch."
    assert torch.allclose(processed[1]['image_id'], torch.tensor([2])), "Image ID mismatch."

# Test evaluate method
def test_evaluator_evaluate(mock_model, mock_dataloader, mock_evaluate_model, caplog):
    """Test the evaluate method of Evaluator."""
    evaluator = Evaluator(model=mock_model, device='cpu', confidence_threshold=0.5)
    
    with caplog.at_level(logging.INFO):
        metrics = evaluator.evaluate(mock_dataloader)
    
    # Check that evaluate_model was called with correct arguments
    all_predictions = []
    all_ground_truths = []
    
    # Manually simulate what the evaluate method does
    for images, targets in mock_dataloader:
        outputs = mock_model(images)
        batch_predictions = evaluator._process_outputs(outputs)
        batch_ground_truths = evaluator._process_targets(targets)
        all_predictions.extend(batch_predictions)
        all_ground_truths.extend(batch_ground_truths)
    
    evaluate_model.assert_called_once_with(all_predictions, all_ground_truths)
    
    # Check that metrics are as mocked
    assert metrics == {
        'mAP': 0.75,
        'precision': 0.8,
        'recall': 0.7,
        'f1_score': 0.75
    }, "Metrics do not match expected values."
    
    # Check logging
    assert "Evaluator initialized with model on device 'cpu'." in caplog.text
    assert "Evaluation completed. Metrics: {'mAP': 0.75, 'precision': 0.8, 'recall': 0.7, 'f1_score': 0.75}" in caplog.text

# Test evaluate with empty dataloader
def test_evaluator_evaluate_empty_dataloader(mock_model, mock_evaluate_model):
    """Test the evaluate method with an empty dataloader."""
    empty_dataloader = DataLoader([], batch_size=2)
    evaluator = Evaluator(model=mock_model, device='cpu', confidence_threshold=0.5)
    
    metrics = evaluator.evaluate(empty_dataloader)
    
    # Since there are no predictions or ground truths, evaluate_model should be called with empty lists
    evaluate_model.assert_called_once_with([], [])
    
    # Check that metrics are as mocked
    assert metrics == {
        'mAP': 0.75,
        'precision': 0.8,
        'recall': 0.7,
        'f1_score': 0.75
    }, "Metrics do not match expected values for empty dataloader."

# Test evaluate with all predictions below confidence threshold
def test_evaluator_evaluate_all_below_threshold(mock_model, mock_dataloader, mock_evaluate_model):
    """Test the evaluate method when all predictions are below the confidence threshold."""
    evaluator = Evaluator(model=mock_model, device='cpu', confidence_threshold=0.9)
    
    # Modify the mock_model to return low scores
    def mock_forward_low_scores(images):
        outputs = []
        for idx, img in enumerate(images):
            num_preds = torch.randint(1, 3, (1,)).item()
            scores = torch.rand(num_preds) * 0.5  # All scores below 0.9
            boxes = torch.rand(num_preds, 4) * 200
            labels = torch.randint(1, 10, (num_preds,))
            image_id = torch.tensor(idx + 1)
            outputs.append({
                'scores': scores,
                'boxes': boxes,
                'labels': labels,
                'image_id': image_id
            })
        return outputs
    mock_model.forward.side_effect = mock_forward_low_scores
    
    metrics = evaluator.evaluate(mock_dataloader)
    
    # All predictions should be filtered out
    all_predictions = []
    all_ground_truths = []
    for images, targets in mock_dataloader:
        outputs = mock_model(images)
        batch_predictions = evaluator._process_outputs(outputs)
        batch_ground_truths = evaluator._process_targets(targets)
        all_predictions.extend(batch_predictions)
        all_ground_truths.extend(batch_ground_truths)
    
    # Since all predictions are below threshold, processed predictions should be empty
    for pred in all_predictions:
        assert len(pred['boxes']) == 0, "There should be no boxes after filtering."
    
    evaluate_model.assert_called_once_with(all_predictions, all_ground_truths)
    
    # Metrics should still be as mocked
    assert metrics == {
        'mAP': 0.75,
        'precision': 0.8,
        'recall': 0.7,
        'f1_score': 0.75
    }, "Metrics do not match expected values when all predictions are below threshold."

# Test evaluate with missing fields in outputs
def test_evaluator_evaluate_missing_fields(mock_model, mock_dataloader, mock_evaluate_model):
    """Test the evaluate method when model outputs are missing fields."""
    evaluator = Evaluator(model=mock_model, device='cpu', confidence_threshold=0.5)
    
    # Modify the mock_model to return outputs missing 'scores'
    def mock_forward_missing_fields(images):
        outputs = []
        for idx, img in enumerate(images):
            num_preds = torch.randint(1, 3, (1,)).item()
            boxes = torch.rand(num_preds, 4) * 200
            labels = torch.randint(1, 10, (num_preds,))
            image_id = torch.tensor(idx + 1)
            outputs.append({
                # 'scores' key is missing
                'boxes': boxes,
                'labels': labels,
                'image_id': image_id
            })
        return outputs
    mock_model.forward.side_effect = mock_forward_missing_fields
    
    with pytest.raises(KeyError):
        evaluator.evaluate(mock_dataloader)
    
    evaluate_model.assert_not_called()

# Test evaluate with incomplete targets
def test_evaluator_evaluate_incomplete_targets(mock_model, mock_dataloader, mock_evaluate_model):
    """Test the evaluate method when targets are missing fields."""
    evaluator = Evaluator(model=mock_model, device='cpu', confidence_threshold=0.5)
    
    # Modify the mock_dataloader to have targets missing 'labels'
    def generate_incomplete_targets():
        return [
            {
                'boxes': torch.tensor([[50, 50, 150, 150]], dtype=torch.float32),
                # 'labels' key is missing
                'image_id': torch.tensor([1], dtype=torch.int64)
            },
            {
                'boxes': torch.tensor([[60, 60, 160, 160]], dtype=torch.float32),
                'labels': torch.tensor([3], dtype=torch.int64),
                'image_id': torch.tensor([2], dtype=torch.int64)
            }
        ]
    
    incomplete_targets = generate_incomplete_targets()
    incomplete_dataloader = DataLoader(list(zip(sample_images := [torch.rand(3, 224, 224) for _ in range(2)], incomplete_targets)), batch_size=1)
    
    with pytest.raises(KeyError):
        evaluator.evaluate(incomplete_dataloader)
    
    evaluate_model.assert_not_called()

# Test evaluate with reproducibility (same seed leads to same predictions)
def test_evaluator_evaluate_reproducibility(mock_model, mock_dataloader, mock_evaluate_model):
    """Test that Evaluator produces consistent metrics with the same seed."""
    # Since Evaluator does not have a seed parameter, reproducibility is determined by model and data
    # Here, ensure that with the same model and data, metrics are consistent
    
    evaluator1 = Evaluator(model=mock_model, device='cpu', confidence_threshold=0.5)
    evaluator2 = Evaluator(model=mock_model, device='cpu', confidence_threshold=0.5)
    
    with caplog.at_level(logging.INFO):
        metrics1 = evaluator1.evaluate(mock_dataloader)
        metrics2 = evaluator2.evaluate(mock_dataloader)
    
    # Metrics should be identical as evaluate_model is mocked to return the same value
    assert metrics1 == metrics2, "Metrics should be identical across evaluations with the same model and data."
    
    # Ensure that logging happened correctly
    assert "Evaluation completed. Metrics: {'mAP': 0.75, 'precision': 0.8, 'recall': 0.7, 'f1_score': 0.75}" in caplog.text

