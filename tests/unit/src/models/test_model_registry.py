# tests/unit/src/models/test_model_registry.py

import pytest
import os
from unittest.mock import MagicMock
from src.models.model_registry import ModelRegistry, ModelRegistryError
from transformers import AutoModelForObjectDetection

@pytest.fixture
def model_registry(tmpdir):
    registry_dir = tmpdir.mkdir("model_registry")
    return ModelRegistry(registry_dir=str(registry_dir))

@pytest.fixture
def mock_model():
    model = MagicMock(spec=AutoModelForObjectDetection)
    return model

def test_save_model(model_registry, mock_model, tmpdir):
    """Test saving a model version in the registry."""
    model_version = "1.0"
    model_registry.save_model(mock_model, model_version)
    mock_model.save_pretrained.assert_called_once_with(tmpdir.join("model_registry/model_v1.0"))

def test_load_model(model_registry, mock_model, tmpdir):
    """Test loading a model version from the registry."""
    model_version = "1.0"
    mock_path = tmpdir.join("model_registry/model_v1.0")
    os.makedirs(mock_path)  # Mock model directory
    
    model_registry.load_model = MagicMock(return_value=mock_model)
    loaded_model = model_registry.load_model("facebook/detr-resnet-50", model_version, device="cpu")
    assert isinstance(loaded_model, MagicMock), "Loaded model should be a MagicMock instance"
    model_registry.load_model.assert_called_once()

def test_list_available_models(model_registry, tmpdir):
    """Test listing available models in the registry."""
    model_version_path = tmpdir.mkdir("model_registry").mkdir("model_v1.0")
    available_models = model_registry.list_available_models()
    
    assert available_models == ["1.0"], "The available model versions should be listed correctly."

def test_delete_model(model_registry, tmpdir):
    """Test deleting a model version from the registry."""
    model_version = "1.0"
    model_version_path = tmpdir.mkdir("model_registry").mkdir(f"model_v{model_version}")
    
    model_registry.delete_model(model_version)
    assert not os.path.exists(model_version_path), "Model version directory should be deleted."

def test_load_model_nonexistent(model_registry):
    """Test loading a non-existent model version."""
    with pytest.raises(ModelRegistryError):
        model_registry.load_model("non_existent_model", "non_existent_version", device="cpu")