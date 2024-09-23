# tests/unit/src/models/test_model_factory.py

import pytest
from unittest.mock import patch, MagicMock, mock_open
from src.models.model_factory import (
    BaseModel,
    DetrModel,
    MODEL_REGISTRY,
    register_model,
    ModelVersioning,
    ModelFactory
)
from transformers import PreTrainedModel
import os
import json
import shutil
import torch


@pytest.fixture
def mock_pretrained_model():
    """Fixture to mock a PreTrainedModel instance."""
    with patch('src.models.model_factory.DetrForObjectDetection') as mock_detr:
        mock_model_instance = MagicMock(spec=PreTrainedModel)
        mock_detr.from_pretrained.return_value = mock_model_instance
        yield mock_detr, mock_model_instance

@pytest.fixture
def sample_metadata():
    """Fixture for sample metadata."""
    return {
        'training_config': {
            'learning_rate': 0.001,
            'epochs': 10
        },
        'dataset': 'coco'
    }

@pytest.fixture
def temporary_model_dir(tmp_path):
    """Fixture for a temporary model directory."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


def test_model_registration():
    """Test that models are correctly registered and prevent duplicate registrations."""
    initial_registry = MODEL_REGISTRY.copy()
    
    @register_model('test_model')
    class TestModel(BaseModel):
        def _build_model(self, **kwargs) -> PreTrainedModel:
            return MagicMock(spec=PreTrainedModel)
    
    assert 'test_model' in MODEL_REGISTRY, "Model 'test_model' should be registered."
    assert MODEL_REGISTRY['test_model'] == TestModel, "Registered model class mismatch."
    
    # Attempt to register the same model again should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        @register_model('test_model')
        class DuplicateTestModel(BaseModel):
            def _build_model(self, **kwargs) -> PreTrainedModel:
                return MagicMock(spec=PreTrainedModel)
    
    assert "Model 'test_model' is already registered." in str(exc_info.value), "Duplicate registration did not raise correct exception."
    
    # Restore original registry
    MODEL_REGISTRY.clear()
    MODEL_REGISTRY.update(initial_registry)


def test_base_model_save_load(mock_pretrained_model, temporary_model_dir, sample_metadata):
    """Test saving and loading of a BaseModel subclass."""
    mock_detr, mock_model_instance = mock_pretrained_model
    
    # Create a DetrModel instance
    model = DetrModel(
        model_name='facebook/detr-resnet-50',
        num_labels=91
    )
    
    # Mock the save_pretrained method
    with patch.object(model.model, 'save_pretrained') as mock_save_pretrained:
        # Mock file writing for metadata
        with patch('builtins.open', mock_open()) as mocked_file:
            model.save(str(temporary_model_dir), metadata=sample_metadata)
            
            # Check that save_pretrained was called with the correct directory
            mock_save_pretrained.assert_called_once_with(str(temporary_model_dir))
            
            # Check that metadata was written correctly
            mocked_file.assert_called_once_with(os.path.join(str(temporary_model_dir), 'metadata.json'), 'w')
            handle = mocked_file()
            handle.write.assert_called_once_with(json.dumps(sample_metadata, indent=4))
    
    # Mock loading the model
    with patch('src.models.model_factory.AutoConfig') as mock_auto_config, \
         patch('src.models.model_factory.AutoModelForObjectDetection.from_pretrained') as mock_from_pretrained, \
         patch('builtins.open', mock_open(read_data=json.dumps(sample_metadata))) as mocked_file_load:
        
        mock_auto_config.from_pretrained.return_value = MagicMock(model_type='detr')
        mock_from_pretrained.return_value = mock_model_instance
        
        loaded_model = DetrModel.load(str(temporary_model_dir))
        
        # Check that AutoConfig was called correctly
        mock_auto_config.from_pretrained.assert_called_once_with(str(temporary_model_dir))
        
        # Check that the correct model class was instantiated
        mock_from_pretrained.assert_called_once_with(str(temporary_model_dir), config=mock_auto_config.from_pretrained.return_value)
        
        # Check that metadata was loaded
        mocked_file_load.assert_called_once_with(os.path.join(str(temporary_model_dir), 'metadata.json'), 'r')
        assert hasattr(loaded_model, 'metadata'), "Loaded model should have 'metadata' attribute."
        assert loaded_model.metadata == sample_metadata, "Loaded metadata does not match."
    
    # Check that the loaded model is an instance of DetrModel
    assert isinstance(loaded_model, DetrModel), "Loaded model is not an instance of DetrModel."


def test_model_factory_creation(mock_pretrained_model, temporary_model_dir):
    """Test that ModelFactory creates models correctly."""
    mock_detr, mock_model_instance = mock_pretrained_model
    
    # Register DetrModel if not already registered
    if 'detr' not in MODEL_REGISTRY:
        MODEL_REGISTRY['detr'] = DetrModel
    
    # Create a model using ModelFactory
    model = ModelFactory.create_model(
        model_type='detr',
        model_name='facebook/detr-resnet-50',
        num_labels=91
    )
    
    assert isinstance(model, DetrModel), "ModelFactory did not create an instance of DetrModel."
    assert model.model_name == 'facebook/detr-resnet-50', "Model name mismatch."
    assert model.num_labels == 91, "Number of labels mismatch."
    mock_pretrained_model[0].from_pretrained.assert_called_once_with(
        'facebook/detr-resnet-50',
        num_labels=91
    )


def test_model_factory_unsupported_type():
    """Test that ModelFactory raises an error for unsupported model types."""
    with pytest.raises(ValueError) as exc_info:
        ModelFactory.create_model(
            model_type='unsupported_model',
            model_name='some/model',
            num_labels=10
        )
    
    assert "Model type 'unsupported_model' is not registered." in str(exc_info.value), "Incorrect error message for unsupported model type."


def test_model_versioning_register_load(mock_pretrained_model, temporary_model_dir, sample_metadata):
    """Test registering and loading models using ModelVersioning."""
    mock_detr, mock_model_instance = mock_pretrained_model
    
    # Initialize ModelVersioning
    versioning = ModelVersioning(model_dir=str(temporary_model_dir))
    
    # Create and register a model
    model = DetrModel(
        model_name='facebook/detr-resnet-50',
        num_labels=91
    )
    
    with patch.object(model, 'save') as mock_save:
        versioning.register_model(
            model_name='detr',
            model_instance=model,
            version='v1.0',
            metadata=sample_metadata
        )
        
        # Check that save was called with correct parameters
        mock_save.assert_called_once_with(os.path.join(str(temporary_model_dir), 'detr_v1.0'), metadata=sample_metadata)
    
    # Check that the registry was updated
    assert 'detr' in versioning.registry, "Model 'detr' should be in registry."
    assert 'v1.0' in versioning.registry['detr'], "Version 'v1.0' should be registered for model 'detr'."
    assert versioning.registry['detr']['v1.0']['metadata'] == sample_metadata, "Metadata mismatch in registry."
    
    # Mock loading the model
    with patch('src.models.model_factory.MODEL_REGISTRY.get', return_value=DetrModel), \
         patch.object(DetrModel, 'load', return_value=model):
        
        loaded_model = versioning.load_model('detr', 'v1.0')
        
        # Check that MODEL_REGISTRY.get was called correctly
        # and DetrModel.load was called with the correct path
        DetrModel.load.assert_called_once_with(os.path.join(str(temporary_model_dir), 'detr_v1.0'))
        
        assert loaded_model == model, "Loaded model does not match the original model."


def test_model_versioning_delete_model(mock_pretrained_model, temporary_model_dir):
    """Test deleting a model version using ModelVersioning."""
    mock_detr, mock_model_instance = mock_pretrained_model
    
    # Initialize ModelVersioning
    versioning = ModelVersioning(model_dir=str(temporary_model_dir))
    
    # Register a model version
    with patch.object(DetrModel, 'save') as mock_save:
        model = DetrModel(
            model_name='facebook/detr-resnet-50',
            num_labels=91
        )
        versioning.register_model(
            model_name='detr',
            model_instance=model,
            version='v1.0',
            metadata={}
        )
    
    # Ensure the model is registered
    assert 'detr' in versioning.registry
    assert 'v1.0' in versioning.registry['detr']
    
    # Mock shutil.rmtree to prevent actual file deletion
    with patch('shutil.rmtree') as mock_rmtree:
        versioning.delete_model('detr', 'v1.0')
        
        # Check that the model path was attempted to be deleted
        mock_rmtree.assert_called_once_with(os.path.join(str(temporary_model_dir), 'detr_v1.0'))
    
    # Check that the registry was updated
    assert 'v1.0' not in versioning.registry.get('detr', {}), "Model version 'v1.0' was not deleted from registry."


def test_model_versioning_load_nonexistent(mock_pretrained_model, temporary_model_dir):
    """Test loading a non-existent model version raises an error."""
    mock_detr, mock_model_instance = mock_pretrained_model
    
    # Initialize ModelVersioning
    versioning = ModelVersioning(model_dir=str(temporary_model_dir))
    
    with pytest.raises(ValueError) as exc_info:
        versioning.load_model('detr', 'nonexistent_version')
    
    assert "Model 'detr' version 'nonexistent_version' not found in registry." in str(exc_info.value), "Incorrect error message for non-existent model version."


def test_model_versioning_list_models(mock_pretrained_model, temporary_model_dir):
    """Test listing all registered models and their versions."""
    mock_detr, mock_model_instance = mock_pretrained_model
    
    # Initialize ModelVersioning
    versioning = ModelVersioning(model_dir=str(temporary_model_dir))
    
    # Register multiple model versions
    with patch.object(DetrModel, 'save'):
        model1 = DetrModel(
            model_name='facebook/detr-resnet-50',
            num_labels=91
        )
        model2 = DetrModel(
            model_name='facebook/detr-resnet-50',
            num_labels=91
        )
        versioning.register_model(
            model_name='detr',
            model_instance=model1,
            version='v1.0',
            metadata={}
        )
        versioning.register_model(
            model_name='detr',
            model_instance=model2,
            version='v1.1',
            metadata={}
        )
    
    # List models
    models = versioning.list_models()
    
    assert 'detr' in models, "Model 'detr' should be listed."
    assert 'v1.0' in models['detr'], "Version 'v1.0' should be listed under 'detr'."
    assert 'v1.1' in models['detr'], "Version 'v1.1' should be listed under 'detr'."
    
    assert models['detr']['v1.0']['path'] == os.path.join(str(temporary_model_dir), 'detr_v1.0'), "Model path mismatch for version 'v1.0'."
    assert models['detr']['v1.1']['path'] == os.path.join(str(temporary_model_dir), 'detr_v1.1'), "Model path mismatch for version 'v1.1'."


def test_freeze_unfreeze_backbone(mock_pretrained_model):
    """Test freezing and unfreezing backbone parameters in the model."""
    mock_detr, mock_model_instance = mock_pretrained_model
    
    # Mock named_parameters
    mock_model_instance.named_parameters.return_value = [
        ('backbone.layer1.weight', MagicMock()),
        ('backbone.layer1.bias', MagicMock()),
        ('classifier.weight', MagicMock()),
        ('classifier.bias', MagicMock())
    ]
    
    # Initialize DetrModel
    model = DetrModel(
        model_name='facebook/detr-resnet-50',
        num_labels=91
    )
    
    # Freeze backbone
    with patch('logging.Logger.info') as mock_logger_info:
        model.freeze_backbone()
        
        # Check that backbone parameters have requires_grad set to False
        for name, param in model.model.named_parameters():
            if 'backbone' in name:
                assert param.requires_grad == False, f"Parameter '{name}' should be frozen."
            else:
                assert param.requires_grad == True, f"Parameter '{name}' should not be frozen."
        
        # Check that logging was called
        mock_logger_info.assert_called_once_with("Backbone parameters have been frozen.")
    
    # Unfreeze backbone
    with patch('logging.Logger.info') as mock_logger_info:
        model.unfreeze_backbone()
        
        # Check that backbone parameters have requires_grad set to True
        for name, param in model.model.named_parameters():
            if 'backbone' in name:
                assert param.requires_grad == True, f"Parameter '{name}' should be unfrozen."
            else:
                assert param.requires_grad == True, f"Parameter '{name}' should not be unfrozen."
        
        # Check that logging was called
        mock_logger_info.assert_called_once_with("Backbone parameters have been unfrozen.")


def test_model_factory_get_available_models():
    """Test that ModelFactory can list available registered models."""
    initial_registry = MODEL_REGISTRY.copy()
    
    @register_model('test_model')
    class TestModel(BaseModel):
        def _build_model(self, **kwargs) -> PreTrainedModel:
            return MagicMock(spec=PreTrainedModel)
    
    available_models = ModelFactory.get_available_models()
    
    assert 'test_model' in available_models, "Model 'test_model' should be listed as available."
    assert available_models['test_model'] == TestModel, "Available model class mismatch."
    
    # Restore original registry
    MODEL_REGISTRY.clear()
    MODEL_REGISTRY.update(initial_registry)

# 11. Test ModelFactory Create and Register Multiple Models

def test_model_factory_create_multiple_models(mock_pretrained_model):
    """Test creating multiple different models using ModelFactory."""
    mock_detr, mock_model_instance = mock_pretrained_model
    
    @register_model('test_model_1')
    class TestModel1(BaseModel):
        def _build_model(self, **kwargs) -> PreTrainedModel:
            return MagicMock(spec=PreTrainedModel)
    
    @register_model('test_model_2')
    class TestModel2(BaseModel):
        def _build_model(self, **kwargs) -> PreTrainedModel:
            return MagicMock(spec=PreTrainedModel)
    
    # Create TestModel1
    model1 = ModelFactory.create_model(
        model_type='test_model_1',
        model_name='test/model1',
        num_labels=10
    )
    assert isinstance(model1, TestModel1), "ModelFactory did not create an instance of TestModel1."
    mock_pretrained_model[0].from_pretrained.assert_called_with('test/model1', num_labels=10)
    
    # Create TestModel2
    model2 = ModelFactory.create_model(
        model_type='test_model_2',
        model_name='test/model2',
        num_labels=20
    )
    assert isinstance(model2, TestModel2), "ModelFactory did not create an instance of TestModel2."
    mock_pretrained_model[0].from_pretrained.assert_called_with('test/model2', num_labels=20)

# 12. Test ModelFactory Create Model Without Registration

def test_model_factory_create_model_without_registration():
    """Test that ModelFactory raises an error when creating a model type that is not registered."""
    with pytest.raises(ValueError) as exc_info:
        ModelFactory.create_model(
            model_type='non_registered_model',
            model_name='some/model',
            num_labels=5
        )
    
    assert "Model type 'non_registered_model' is not registered." in str(exc_info.value), "Incorrect error message for non-registered model type."


def test_model_factory_save_load_versioning(mock_pretrained_model, temporary_model_dir, sample_metadata):
    """Test saving and loading model versions using ModelVersioning via ModelFactory."""
    mock_detr, mock_model_instance = mock_pretrained_model
    
    # Register DetrModel if not already registered
    if 'detr' not in MODEL_REGISTRY:
        MODEL_REGISTRY['detr'] = DetrModel
    
    # Initialize ModelVersioning
    versioning = ModelVersioning(model_dir=str(temporary_model_dir))
    
    # Create and register a model version
    model = ModelFactory.create_model(
        model_type='detr',
        model_name='facebook/detr-resnet-50',
        num_labels=91
    )
    
    with patch.object(model, 'save') as mock_save:
        versioning.register_model(
            model_name='detr',
            model_instance=model,
            version='v1.0',
            metadata=sample_metadata
        )
        
        # Check that save was called correctly
        mock_save.assert_called_once_with(os.path.join(str(temporary_model_dir), 'detr_v1.0'), metadata=sample_metadata)
    
    # Load the registered model
    with patch.object(DetrModel, 'load', return_value=model) as mock_load:
        loaded_model = versioning.load_model('detr', 'v1.0')
        
        mock_load.assert_called_once_with(os.path.join(str(temporary_model_dir), 'detr_v1.0'))
        assert loaded_model == model, "Loaded model does not match the original model."


def test_model_versioning_register_existing_version(mock_pretrained_model, temporary_model_dir):
    """Test that registering a model version that already exists raises an error."""
    mock_detr, mock_model_instance = mock_pretrained_model
    
    # Initialize ModelVersioning
    versioning = ModelVersioning(model_dir=str(temporary_model_dir))
    
    # Register a model version
    with patch.object(DetrModel, 'save'):
        model = DetrModel(
            model_name='facebook/detr-resnet-50',
            num_labels=91
        )
        versioning.register_model(
            model_name='detr',
            model_instance=model,
            version='v1.0',
            metadata={}
        )
    
    # Attempt to register the same version again should overwrite or raise an error based on implementation
    # Assuming it overwrites, we check that it updates the registry
    with patch.object(DetrModel, 'save'):
        versioning.register_model(
            model_name='detr',
            model_instance=model,
            version='v1.0',
            metadata={'updated': True}
        )
    
    assert versioning.registry['detr']['v1.0']['metadata'] == {'updated': True}, "Model version 'v1.0' should be updated with new metadata."

# 15. Test BaseModel Load with Unregistered Model Type

def test_base_model_load_unregistered_model(mock_pretrained_model, temporary_model_dir):
    """Test that loading a model with an unregistered model type raises an error."""
    mock_detr, mock_model_instance = mock_pretrained_model
    
    # Create a dummy config with an unregistered model_type
    with patch('src.models.model_factory.AutoConfig.from_pretrained') as mock_auto_config:
        mock_config = MagicMock()
        mock_config.model_type = 'unregistered_model_type'
        mock_auto_config.return_value = mock_config
        
        with pytest.raises(ValueError) as exc_info:
            BaseModel.load(str(temporary_model_dir))
        
        assert "Model type 'unregistered_model_type' is not registered." in str(exc_info.value), "Incorrect error message for unregistered model type."


def test_base_model_load_missing_metadata(mock_pretrained_model, temporary_model_dir):
    """Test loading a model without metadata."""
    mock_detr, mock_model_instance = mock_pretrained_model
    
    # Mock the presence of model files but absence of metadata
    with patch('src.models.model_factory.AutoConfig.from_pretrained') as mock_auto_config, \
         patch('src.models.model_factory.MODEL_REGISTRY.get', return_value=DetrModel), \
         patch.object(DetrModel, 'load') as mock_load:
        
        mock_config = MagicMock()
        mock_config.model_type = 'detr'
        mock_config.num_labels = 91
        mock_auto_config.from_pretrained.return_value = mock_config
        
        mock_load.return_value = DetrModel.load(str(temporary_model_dir))
        
        # Ensure metadata.json does not exist
        with patch('os.path.exists', return_value=False):
            model = BaseModel.load(str(temporary_model_dir))
        
        mock_load.assert_called_once_with(str(temporary_model_dir))
        assert not hasattr(model, 'metadata'), "Model should not have 'metadata' attribute when metadata.json is missing."


def test_base_model_load_corrupted_metadata(mock_pretrained_model, temporary_model_dir):
    """Test loading a model with corrupted metadata."""
    mock_detr, mock_model_instance = mock_pretrained_model
    
    # Mock the presence of model files and corrupted metadata
    with patch('src.models.model_factory.AutoConfig.from_pretrained') as mock_auto_config, \
         patch('src.models.model_factory.MODEL_REGISTRY.get', return_value=DetrModel), \
         patch.object(DetrModel, 'load') as mock_load, \
         patch('builtins.open', mock_open(read_data='corrupted json')):
        
        mock_config = MagicMock()
        mock_config.model_type = 'detr'
        mock_config.num_labels = 91
        mock_auto_config.from_pretrained.return_value = mock_config
        
        mock_load.return_value = DetrModel.load(str(temporary_model_dir))
        
        # Ensure metadata.json exists but is corrupted
        with patch('os.path.exists', return_value=True):
            with pytest.raises(json.JSONDecodeError):
                BaseModel.load(str(temporary_model_dir))
        
        mock_load.assert_called_once_with(str(temporary_model_dir))


def test_model_save_directory_creation(mock_pretrained_model, temporary_model_dir, sample_metadata):
    """Test that the save method creates the directory if it does not exist."""
    mock_detr, mock_model_instance = mock_pretrained_model
    
    model = DetrModel(
        model_name='facebook/detr-resnet-50',
        num_labels=91
    )
    
    # Define a nested save directory
    nested_save_dir = temporary_model_dir / "nested" / "detr_v1.0"
    
    with patch.object(model.model, 'save_pretrained') as mock_save_pretrained, \
         patch('builtins.open', mock_open()) as mocked_file, \
         patch('os.makedirs') as mock_makedirs:
        
        # Ensure os.makedirs is called with exist_ok=True
        mock_makedirs.return_value = None
        
        model.save(str(nested_save_dir), metadata=sample_metadata)
        
        mock_makedirs.assert_called_once_with(str(nested_save_dir), exist_ok=True)
        mock_save_pretrained.assert_called_once_with(str(nested_save_dir))
        mocked_file.assert_called_once_with(os.path.join(str(nested_save_dir), 'metadata.json'), 'w')


def test_model_versioning_load_when_registry_empty(temporary_model_dir):
    """Test loading a model when the registry is empty."""
    versioning = ModelVersioning(model_dir=str(temporary_model_dir))
    
    with pytest.raises(ValueError) as exc_info:
        versioning.load_model('detr', 'v1.0')
    
    assert "Model 'detr' version 'v1.0' not found in registry." in str(exc_info.value), "Incorrect error message when registry is empty."


def test_model_versioning_delete_nonexistent(temporary_model_dir):
    """Test deleting a non-existent model version raises an error."""
    versioning = ModelVersioning(model_dir=str(temporary_model_dir))
    
    with pytest.raises(ValueError) as exc_info:
        versioning.delete_model('detr', 'v2.0')
    
    assert "Model 'detr' version 'v2.0' not found in registry." in str(exc_info.value), "Incorrect error message for deleting non-existent model version."


def test_base_model_state_dict(mock_pretrained_model):
    """Test getting and loading the state dictionary of a model."""
    mock_detr, mock_model_instance = mock_pretrained_model
    
    model = DetrModel(
        model_name='facebook/detr-resnet-50',
        num_labels=91
    )
    
    # Mock state_dict
    mock_state_dict = {'layer1.weight': torch.randn(10, 10)}
    model.model.state_dict.return_value = mock_state_dict
    
    state_dict = model.get_state_dict()
    assert state_dict == mock_state_dict, "State dict does not match expected."
    
    # Mock load_state_dict
    with patch.object(model.model, 'load_state_dict') as mock_load_state_dict:
        model.load_state_dict(mock_state_dict)
        mock_load_state_dict.assert_called_once_with(mock_state_dict)


def test_base_model_load_error_handling(mock_pretrained_model, temporary_model_dir):
    """Test that the load method handles errors gracefully."""
    mock_detr, mock_model_instance = mock_pretrained_model
    
    # Mock AutoConfig to raise an exception
    with patch('src.models.model_factory.AutoConfig.from_pretrained', side_effect=Exception("Config loading failed")):
        with pytest.raises(Exception) as exc_info:
            BaseModel.load(str(temporary_model_dir))
        
        assert "Error loading model from" in str(exc_info.value), "Incorrect error message when AutoConfig fails."


def test_model_factory_create_model_with_kwargs(mock_pretrained_model):
    """Test creating a model with additional keyword arguments."""
    mock_detr, mock_model_instance = mock_pretrained_model
    
    # Assuming additional kwargs are passed to DetrConfig
    model = ModelFactory.create_model(
        model_type='detr',
        model_name='facebook/detr-resnet-50',
        num_labels=91,
        backbone={'depth': 50}
    )
    
    # Check that from_pretrained was called with additional kwargs
    mock_pretrained_model[0].from_pretrained.assert_called_once_with(
        'facebook/detr-resnet-50',
        num_labels=91,
        backbone={'depth': 50}
    )
    assert isinstance(model, DetrModel), "Created model is not an instance of DetrModel."


def test_model_versioning_registry_persistence(mock_pretrained_model, temporary_model_dir, sample_metadata):
    """Test that the model registry persists across different ModelVersioning instances."""
    mock_detr, mock_model_instance = mock_pretrained_model
    
    # Initialize first ModelVersioning instance and register a model
    versioning1 = ModelVersioning(model_dir=str(temporary_model_dir))
    
    with patch.object(DetrModel, 'save'):
        model = DetrModel(
            model_name='facebook/detr-resnet-50',
            num_labels=91
        )
        versioning1.register_model(
            model_name='detr',
            model_instance=model,
            version='v1.0',
            metadata=sample_metadata
        )
    
    # Initialize a new ModelVersioning instance and check registry
    versioning2 = ModelVersioning(model_dir=str(temporary_model_dir))
    
    assert 'detr' in versioning2.registry, "Model 'detr' should be present in the new registry instance."
    assert 'v1.0' in versioning2.registry['detr'], "Version 'v1.0' should be present in the new registry instance."
    assert versioning2.registry['detr']['v1.0']['metadata'] == sample_metadata, "Metadata mismatch in persisted registry."


def test_model_factory_get_available_models_after_registration():
    """Test that available models are listed correctly after registration."""
    initial_registry = MODEL_REGISTRY.copy()
    
    @register_model('new_model')
    class NewModel(BaseModel):
        def _build_model(self, **kwargs) -> PreTrainedModel:
            return MagicMock(spec=PreTrainedModel)
    
    available_models = ModelFactory.get_available_models()
    
    assert 'new_model' in available_models, "Newly registered model should be listed in available models."
    assert available_models['new_model'] == NewModel, "Available model class mismatch for 'new_model'."
    
    # Restore original registry
    MODEL_REGISTRY.clear()
    MODEL_REGISTRY.update(initial_registry)


def test_model_factory_save_load_multiple_models(mock_pretrained_model, temporary_model_dir, sample_metadata):
    """Test saving and loading multiple model versions using ModelVersioning."""
    mock_detr, mock_model_instance = mock_pretrained_model
    
    # Register DetrModel if not already registered
    if 'detr' not in MODEL_REGISTRY:
        MODEL_REGISTRY['detr'] = DetrModel
    
    # Initialize ModelVersioning
    versioning = ModelVersioning(model_dir=str(temporary_model_dir))
    
    # Create and register multiple model versions
    with patch.object(DetrModel, 'save'):
        model_v1 = DetrModel(
            model_name='facebook/detr-resnet-50',
            num_labels=91
        )
        versioning.register_model(
            model_name='detr',
            model_instance=model_v1,
            version='v1.0',
            metadata={'description': 'Initial version'}
        )
        
        model_v2 = DetrModel(
            model_name='facebook/detr-resnet-50',
            num_labels=91
        )
        versioning.register_model(
            model_name='detr',
            model_instance=model_v2,
            version='v2.0',
            metadata={'description': 'Second version'}
        )
    
    # Load both models
    with patch.object(DetrModel, 'load', side_effect=[model_v1, model_v2]):
        loaded_model_v1 = versioning.load_model('detr', 'v1.0')
        loaded_model_v2 = versioning.load_model('detr', 'v2.0')
    
        assert loaded_model_v1 == model_v1, "Loaded model v1.0 does not match."
        assert loaded_model_v2 == model_v2, "Loaded model v2.0 does not match."
    
    # Check registry
    models = versioning.list_models()
    assert 'detr' in models
    assert 'v1.0' in models['detr']
    assert 'v2.0' in models['detr']


def test_model_versioning_delete_all_versions(mock_pretrained_model, temporary_model_dir):
    """Test deleting all versions of a model."""
    mock_detr, mock_model_instance = mock_pretrained_model
    
    # Initialize ModelVersioning
    versioning = Model
