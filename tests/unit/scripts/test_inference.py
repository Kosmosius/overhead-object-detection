# tests/unit/scripts/test_inference.py

import os
import pytest
import torch
from unittest import mock
from scripts.inference import load_image, run_inference, visualize_predictions
from PIL import Image

# Core test: Load image
@mock.patch("PIL.Image.open")
def test_load_image(mock_open):
    """Test the image loading functionality."""
    # Mock an image being loaded
    mock_image = mock.Mock(spec=Image.Image)
    mock_open.return_value = mock_image

    image_path = "fake_image.jpg"
    loaded_image = load_image(image_path)

    # Ensure the image is opened and converted to RGB
    mock_open.assert_called_once_with(image_path)
    mock_image.convert.assert_called_once_with("RGB")
    assert loaded_image == mock_image.convert()

# Core test: Visualize predictions
@mock.patch("matplotlib.pyplot.subplots")
@mock.patch("matplotlib.pyplot.show")
def test_visualize_predictions(mock_show, mock_subplots):
    """Test visualization of model predictions."""
    mock_ax = mock.Mock()
    mock_subplots.return_value = (mock.Mock(), mock_ax)

    # Mock input image and predictions
    mock_image = mock.Mock(spec=Image.Image)
    predictions = {
        'boxes': [[100, 150, 200, 250]],
        'scores': [0.95]
    }

    # Call the visualize function
    visualize_predictions(mock_image, predictions, threshold=0.5)

    # Ensure correct drawing behavior
    mock_ax.imshow.assert_called_once_with(mock_image)
    mock_ax.add_patch.assert_called_once()
    mock_show.assert_called_once()

# Mock inference helper functions
@mock.patch("scripts.inference.HuggingFaceObjectDetectionModel.load")
@mock.patch("transformers.DetrFeatureExtractor.from_pretrained")
@mock.patch("scripts.inference.load_image")
@mock.patch("torch.no_grad")
def test_run_inference(mock_no_grad, mock_load_image, mock_feature_extractor, mock_model_load):
    """Test the run_inference functionality with mocked components."""
    mock_model = mock.Mock()
    mock_model_load.return_value = mock_model

    mock_image = mock.Mock()
    mock_load_image.return_value = mock_image

    mock_extractor = mock.Mock()
    mock_feature_extractor.return_value = mock_extractor

    # Mock pixel values for feature extractor output
    mock_extractor.return_tensors.return_value.pixel_values = torch.tensor([[[[1.0]]]])

    mock_model.return_value.logits = torch.tensor([[0.9]])
    mock_model.return_value.pred_boxes = torch.tensor([[[10, 20, 30, 40]]])

    # Run the inference
    run_inference(mock_model, "fake_image.jpg", mock_extractor, device="cpu")

    # Verify image was loaded
    mock_load_image.assert_called_once_with("fake_image.jpg")
    mock_extractor.assert_called_once_with(images=mock_image, return_tensors="pt")
    mock_model.to.assert_called_once_with("cpu")
    mock_model.eval.assert_called_once()

    # Ensure the predictions were processed correctly
    mock_no_grad.assert_called_once()

# Edge case: Invalid image path
@mock.patch("scripts.inference.HuggingFaceObjectDetectionModel.load")
@mock.patch("PIL.Image.open")
def test_run_inference_invalid_image(mock_image_open, mock_model_load):
    """Test handling of invalid image paths."""
    mock_model = mock.Mock()
    mock_model_load.return_value = mock_model

    # Raise a FileNotFoundError for the image path
    mock_image_open.side_effect = FileNotFoundError("Image not found")

    with pytest.raises(FileNotFoundError, match="Image not found"):
        run_inference(mock_model, "invalid_image.jpg", mock.Mock(), device="cpu")

# Edge case: Missing model checkpoint
@mock.patch("scripts.inference.HuggingFaceObjectDetectionModel.load")
def test_run_inference_missing_model_checkpoint(mock_model_load):
    """Test handling of missing model checkpoint."""
    mock_model_load.side_effect = FileNotFoundError("Model checkpoint not found")

    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        run_inference(mock.Mock(), "fake_image.jpg", mock.Mock(), device="cpu")

# Edge case: Invalid device name
@mock.patch("scripts.inference.HuggingFaceObjectDetectionModel.load")
def test_run_inference_invalid_device(mock_model_load):
    """Test handling of invalid device names."""
    mock_model = mock.Mock()
    mock_model_load.return_value = mock_model

    with pytest.raises(ValueError, match="Invalid device"):
        run_inference(mock_model, "fake_image.jpg", mock.Mock(), device="invalid_device")

# Performance test: Inference speed on large images
@mock.patch("scripts.inference.HuggingFaceObjectDetectionModel.load")
@mock.patch("scripts.inference.load_image")
@mock.patch("transformers.DetrFeatureExtractor.from_pretrained")
def test_run_inference_large_image(mock_feature_extractor, mock_load_image, mock_model_load, benchmark):
    """Test inference performance with a large image."""
    mock_model = mock.Mock()
    mock_model_load.return_value = mock_model

    # Mock a large image being loaded
    large_image = Image.new("RGB", (4000, 3000))
    mock_load_image.return_value = large_image

    # Mock pixel values for feature extractor
    mock_extractor = mock.Mock()
    mock_feature_extractor.return_value = mock_extractor
    mock_extractor.return_tensors.return_value.pixel_values = torch.rand((1, 3, 3000, 4000))

    # Benchmark the inference function
    result = benchmark(run_inference, mock_model, "fake_large_image.jpg", mock_extractor, device="cpu")
    assert result is None  # As run_inference returns None

    # Verify large image processing occurred
    mock_load_image.assert_called_once_with("fake_large_image.jpg")

# Device handling: CPU vs CUDA
@mock.patch("scripts.inference.HuggingFaceObjectDetectionModel.load")
@mock.patch("scripts.inference.load_image")
def test_run_inference_device_handling(mock_load_image, mock_model_load):
    """Test that inference runs correctly on CPU vs CUDA."""
    mock_model = mock.Mock()
    mock_model_load.return_value = mock_model

    mock_image = mock.Mock()
    mock_load_image.return_value = mock_image

    # Run on CPU
    run_inference(mock_model, "fake_image.jpg", mock.Mock(), device="cpu")
    mock_model.to.assert_called_with("cpu")

    # Run on CUDA
    run_inference(mock_model, "fake_image.jpg", mock.Mock(), device="cuda")
    mock_model.to.assert_called_with("cuda")
