# tests/unit/src/utils/test_config_parser.py

import os
import pytest
import json
import yaml
from unittest.mock import patch, MagicMock
from src.utils.config_parser import ConfigParser
from jsonschema.exceptions import ValidationError

# --- Fixtures ---

@pytest.fixture
def tmp_yaml_config(tmp_path):
    """Fixture to create a temporary YAML configuration file."""
    config = {
        "model": {
            "model_type": "detr",
            "model_name": "facebook/detr-resnet-50",
            "num_classes": 91
        },
        "training": {
            "num_epochs": 10,
            "gradient_clipping": 1.0,
            "early_stopping_patience": 3,
            "checkpoint_dir": "./checkpoints",
            "output_dir": "./output"
        },
        "optimizer": {
            "optimizer_type": "adamw",
            "learning_rate": 5e-5,
            "weight_decay": 0.01
        },
        "scheduler": {
            "scheduler_type": "linear",
            "num_warmup_steps": 0
        },
        "loss": {
            "classification_loss_weight": 1.0,
            "bbox_loss_weight": 1.0
        }
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path

@pytest.fixture
def tmp_json_config(tmp_path):
    """Fixture to create a temporary JSON configuration file."""
    config = {
        "model": {
            "model_type": "detr",
            "model_name": "facebook/detr-resnet-50",
            "num_classes": 91
        },
        "training": {
            "num_epochs": 10,
            "gradient_clipping": 1.0,
            "early_stopping_patience": 3,
            "checkpoint_dir": "./checkpoints",
            "output_dir": "./output"
        },
        "optimizer": {
            "optimizer_type": "adamw",
            "learning_rate": 5e-5,
            "weight_decay": 0.01
        },
        "scheduler": {
            "scheduler_type": "linear",
            "num_warmup_steps": 0
        },
        "loss": {
            "classification_loss_weight": 1.0,
            "bbox_loss_weight": 1.0
        }
    }
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    return config_path

@pytest.fixture
def tmp_yaml_schema(tmp_path):
    """Fixture to create a temporary YAML schema file."""
    schema = {
        "type": "object",
        "properties": {
            "model": {
                "type": "object",
                "properties": {
                    "model_type": {"type": "string"},
                    "model_name": {"type": "string"},
                    "num_classes": {"type": "integer", "minimum": 1}
                },
                "required": ["model_type", "model_name", "num_classes"]
            },
            "training": {
                "type": "object",
                "properties": {
                    "num_epochs": {"type": "integer", "minimum": 1},
                    "gradient_clipping": {"type": "number", "minimum": 0},
                    "early_stopping_patience": {"type": "integer", "minimum": 0},
                    "checkpoint_dir": {"type": "string"},
                    "output_dir": {"type": "string"}
                },
                "required": ["num_epochs", "checkpoint_dir", "output_dir"]
            },
            "optimizer": {
                "type": "object",
                "properties": {
                    "optimizer_type": {"type": "string"},
                    "learning_rate": {"type": "number", "minimum": 0},
                    "weight_decay": {"type": "number", "minimum": 0}
                },
                "required": ["optimizer_type", "learning_rate", "weight_decay"]
            },
            "scheduler": {
                "type": "object",
                "properties": {
                    "scheduler_type": {"type": "string"},
                    "num_warmup_steps": {"type": "integer", "minimum": 0}
                },
                "required": ["scheduler_type", "num_warmup_steps"]
            },
            "loss": {
                "type": "object",
                "properties": {
                    "classification_loss_weight": {"type": "number", "minimum": 0},
                    "bbox_loss_weight": {"type": "number", "minimum": 0}
                },
                "required": ["classification_loss_weight", "bbox_loss_weight"]
            }
        },
        "required": ["model", "training", "optimizer", "scheduler", "loss"]
    }
    schema_path = tmp_path / "schema.yaml"
    with open(schema_path, "w") as f:
        yaml.dump(schema, f)
    return schema_path

@pytest.fixture
def tmp_json_schema(tmp_path):
    """Fixture to create a temporary JSON schema file."""
    schema = {
        "type": "object",
        "properties": {
            "model": {
                "type": "object",
                "properties": {
                    "model_type": {"type": "string"},
                    "model_name": {"type": "string"},
                    "num_classes": {"type": "integer", "minimum": 1}
                },
                "required": ["model_type", "model_name", "num_classes"]
            },
            "training": {
                "type": "object",
                "properties": {
                    "num_epochs": {"type": "integer", "minimum": 1},
                    "gradient_clipping": {"type": "number", "minimum": 0},
                    "early_stopping_patience": {"type": "integer", "minimum": 0},
                    "checkpoint_dir": {"type": "string"},
                    "output_dir": {"type": "string"}
                },
                "required": ["num_epochs", "checkpoint_dir", "output_dir"]
            },
            "optimizer": {
                "type": "object",
                "properties": {
                    "optimizer_type": {"type": "string"},
                    "learning_rate": {"type": "number", "minimum": 0},
                    "weight_decay": {"type": "number", "minimum": 0}
                },
                "required": ["optimizer_type", "learning_rate", "weight_decay"]
            },
            "scheduler": {
                "type": "object",
                "properties": {
                    "scheduler_type": {"type": "string"},
                    "num_warmup_steps": {"type": "integer", "minimum": 0}
                },
                "required": ["scheduler_type", "num_warmup_steps"]
            },
            "loss": {
                "type": "object",
                "properties": {
                    "classification_loss_weight": {"type": "number", "minimum": 0},
                    "bbox_loss_weight": {"type": "number", "minimum": 0}
                },
                "required": ["classification_loss_weight", "bbox_loss_weight"]
            }
        },
        "required": ["model", "training", "optimizer", "scheduler", "loss"]
    }
    schema_path = tmp_path / "schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=4)
    return schema_path

@pytest.fixture
def malformed_yaml_config(tmp_path):
    """Fixture to create a malformed YAML configuration file."""
    malformed_content = "model: {model_type: detr, model_name: detr-resnet-50, num_classes: 91"  # Missing closing }
    config_path = tmp_path / "malformed_config.yaml"
    with open(config_path, "w") as f:
        f.write(malformed_content)
    return config_path

@pytest.fixture
def malformed_json_config(tmp_path):
    """Fixture to create a malformed JSON configuration file."""
    malformed_content = '{"model": {"model_type": "detr", "model_name": "detr-resnet-50", "num_classes": 91}'  # Missing closing }
    config_path = tmp_path / "malformed_config.json"
    with open(config_path, "w") as f:
        f.write(malformed_content)
    return config_path

# --- Test Cases ---

# 1. Initialization Tests

@pytest.mark.parametrize("file_format,fixture", [
    (".yaml", "tmp_yaml_config"),
    (".json", "tmp_json_config")
])
def test_config_parser_initialization_success(file_format, fixture, request):
    """Test that ConfigParser initializes correctly with supported file formats."""
    config_path = request.getfixturevalue(fixture)
    parser = ConfigParser(config_path=config_path)
    assert isinstance(parser.config, dict), "Configuration should be a dictionary."
    assert parser.config["model"]["model_name"] == "facebook/detr-resnet-50", "Model name mismatch."

def test_config_parser_initialization_unsupported_format(tmp_path):
    """Test that ConfigParser raises an error with unsupported file formats."""
    unsupported_file = tmp_path / "config.txt"
    with open(unsupported_file, "w") as f:
        f.write("unsupported format")
    
    with pytest.raises(ValueError) as exc_info:
        ConfigParser(config_path=str(unsupported_file))
    
    assert "Unsupported configuration file format: .txt" in str(exc_info.value), "Unsupported file format error not raised."

def test_config_parser_initialization_nonexistent_file():
    """Test that ConfigParser raises FileNotFoundError for non-existent files."""
    nonexistent_file = "nonexistent_config.yaml"
    with pytest.raises(FileNotFoundError) as exc_info:
        ConfigParser(config_path=nonexistent_file)
    
    assert f"Configuration file not found: {nonexistent_file}" in str(exc_info.value), "Non-existent file error not raised."

# 2. Schema Validation Tests

def test_config_parser_schema_validation_success(tmp_yaml_config, tmp_yaml_schema):
    """Test that ConfigParser validates configuration against schema successfully."""
    parser = ConfigParser(config_path=tmp_yaml_config, schema_path=tmp_yaml_schema)
    assert parser.config["model"]["model_type"] == "detr", "Model type mismatch after schema validation."

def test_config_parser_schema_validation_failure(tmp_yaml_config, tmp_yaml_schema, tmp_path):
    """Test that ConfigParser raises ValidationError when configuration does not conform to schema."""
    # Modify the config to violate the schema (e.g., negative num_classes)
    with open(tmp_yaml_config, "r") as f:
        config = yaml.safe_load(f)
    config["model"]["num_classes"] = -5
    with open(tmp_yaml_config, "w") as f:
        yaml.dump(config, f)
    
    with pytest.raises(ValidationError) as exc_info:
        ConfigParser(config_path=tmp_yaml_config, schema_path=tmp_yaml_schema)
    
    assert "must have a minimum value of 1" in str(exc_info.value), "ValidationError for num_classes not raised."

def test_config_parser_schema_validation_nonexistent_schema(tmp_yaml_config, tmp_path):
    """Test that ConfigParser raises FileNotFoundError when schema file does not exist."""
    nonexistent_schema = tmp_path / "nonexistent_schema.yaml"
    with pytest.raises(FileNotFoundError) as exc_info:
        ConfigParser(config_path=tmp_yaml_config, schema_path=str(nonexistent_schema))
    
    assert f"Schema file not found: {nonexistent_schema}" in str(exc_info.value), "Non-existent schema file error not raised."

def test_config_parser_schema_validation_unsupported_schema_format(tmp_yaml_config, tmp_path):
    """Test that ConfigParser raises ValueError for unsupported schema file formats."""
    unsupported_schema = tmp_path / "schema.txt"
    with open(unsupported_schema, "w") as f:
        f.write("unsupported format")
    
    with pytest.raises(ValueError) as exc_info:
        ConfigParser(config_path=tmp_yaml_config, schema_path=str(unsupported_schema))
    
    assert "Unsupported schema file format: .txt" in str(exc_info.value), "Unsupported schema format error not raised."

# 3. Value Retrieval Tests

def test_config_parser_get_existing_key(tmp_yaml_config):
    """Test retrieving an existing top-level key."""
    parser = ConfigParser(config_path=tmp_yaml_config)
    model_name = parser.get("model.model_name")
    assert model_name == "facebook/detr-resnet-50", "Retrieved model name does not match."

def test_config_parser_get_nested_key(tmp_yaml_config):
    """Test retrieving a nested key using dot notation."""
    parser = ConfigParser(config_path=tmp_yaml_config)
    learning_rate = parser.get("optimizer.learning_rate")
    assert learning_rate == 5e-5, "Retrieved learning rate does not match."

def test_config_parser_get_nonexistent_key_with_default(tmp_yaml_config):
    """Test retrieving a non-existent key with a default value."""
    parser = ConfigParser(config_path=tmp_yaml_config)
    dropout_rate = parser.get("model.dropout_rate", default=0.5)
    assert dropout_rate == 0.5, "Default value for non-existent key not returned."

def test_config_parser_get_nonexistent_key_without_default(tmp_yaml_config, caplog):
    """Test retrieving a non-existent key without providing a default value."""
    parser = ConfigParser(config_path=tmp_yaml_config)
    result = parser.get("optimizer.momentum")
    assert result is None, "Result should be None for non-existent key without default."
    assert "Key 'optimizer.momentum' not found in the configuration. Returning default value." in caplog.text, "Warning log for missing key not found."

# 4. Configuration Update Tests

def test_config_parser_update_top_level_key(tmp_yaml_config):
    """Test updating a top-level key in the configuration."""
    parser = ConfigParser(config_path=tmp_yaml_config)
    parser.update({"training": {"num_epochs": 20}})
    assert parser.get("training.num_epochs") == 20, "Top-level key 'num_epochs' was not updated correctly."

def test_config_parser_update_nested_key(tmp_yaml_config):
    """Test updating a nested key in the configuration."""
    parser = ConfigParser(config_path=tmp_yaml_config)
    parser.update({"model": {"num_classes": 100}})
    assert parser.get("model.num_classes") == 100, "Nested key 'num_classes' was not updated correctly."

def test_config_parser_add_new_nested_key(tmp_yaml_config):
    """Test adding a new nested key to the configuration."""
    parser = ConfigParser(config_path=tmp_yaml_config)
    parser.update({"model": {"dropout_rate": 0.3}})
    assert parser.get("model.dropout_rate") == 0.3, "New nested key 'dropout_rate' was not added correctly."

# 5. Saving Configuration Tests

@pytest.mark.parametrize("file_format,fixture", [
    (".yaml", "tmp_yaml_config"),
    (".json", "tmp_json_config")
])
def test_config_parser_save_configuration(file_format, fixture, request, tmp_path):
    """Test saving the configuration in YAML and JSON formats."""
    config_path = request.getfixturevalue(fixture)
    parser = ConfigParser(config_path=config_path)
    output_path = tmp_path / f"saved_config{file_format}"
    parser.save(output_path=str(output_path))
    
    assert os.path.exists(output_path), "Saved configuration file does not exist."
    
    # Load the saved configuration and compare
    if file_format == ".yaml":
        with open(output_path, "r") as f:
            saved_config = yaml.safe_load(f)
    else:
        with open(output_path, "r") as f:
            saved_config = json.load(f)
    
    assert saved_config == parser.config, "Saved configuration does not match the original."

def test_config_parser_save_configuration_unsupported_format(tmp_yaml_config, tmp_path):
    """Test that saving configuration in an unsupported file format raises an error."""
    parser = ConfigParser(config_path=tmp_yaml_config)
    unsupported_save_path = tmp_path / "config.txt"
    
    with pytest.raises(ValueError) as exc_info:
        parser.save(output_path=str(unsupported_save_path))
    
    assert "Unsupported file format for saving: .txt" in str(exc_info.value), "Unsupported file format error not raised."

# 6. Utility Method Tests

def test_config_parser_display(tmp_yaml_config, capsys):
    """Test that the display method prints the configuration correctly."""
    parser = ConfigParser(config_path=tmp_yaml_config)
    parser.display()
    
    captured = capsys.readouterr()
    assert "model:" in captured.out, "Displayed configuration does not contain 'model'."
    assert "training:" in captured.out, "Displayed configuration does not contain 'training'."

def test_config_parser_to_dict(tmp_yaml_config):
    """Test that the to_dict method returns the correct configuration dictionary."""
    parser = ConfigParser(config_path=tmp_yaml_config)
    config_dict = parser.to_dict()
    assert isinstance(config_dict, dict), "to_dict should return a dictionary."
    assert config_dict["model"]["model_type"] == "detr", "to_dict returned incorrect data."

# 7. Edge Case Tests

def test_config_parser_empty_configuration_file(tmp_path):
    """Test that ConfigParser handles empty configuration files gracefully."""
    empty_config = tmp_path / "empty_config.yaml"
    open(empty_config, "w").close()  # Create an empty file
    
    with pytest.raises(Exception) as exc_info:
        ConfigParser(config_path=str(empty_config))
    
    assert "Error loading configuration file" in str(exc_info.value), "Empty configuration file should raise an error."

def test_config_parser_malformed_yaml(malformed_yaml_config):
    """Test that ConfigParser raises an error when loading malformed YAML."""
    with pytest.raises(Exception) as exc_info:
        ConfigParser(config_path=malformed_yaml_config)
    
    assert "Error loading configuration file" in str(exc_info.value), "Malformed YAML should raise an error."

def test_config_parser_malformed_json(malformed_json_config):
    """Test that ConfigParser raises an error when loading malformed JSON."""
    with pytest.raises(Exception) as exc_info:
        ConfigParser(config_path=malformed_json_config)
    
    assert "Error loading configuration file" in str(exc_info.value), "Malformed JSON should raise an error."

# 8. Additional Edge Case Tests

def test_config_parser_update_with_empty_dict(tmp_yaml_config):
    """Test updating the configuration with an empty dictionary does not alter the config."""
    parser = ConfigParser(config_path=tmp_yaml_config)
    original_config = parser.to_dict().copy()
    parser.update({})
    assert parser.to_dict() == original_config, "Updating with an empty dictionary should not alter the configuration."

def test_config_parser_save_after_update(tmp_yaml_config, tmp_path):
    """Test saving the configuration after performing updates."""
    parser = ConfigParser(config_path=tmp_yaml_config)
    parser.update({"training": {"num_epochs": 15}})
    output_path = tmp_path / "updated_config.yaml"
    parser.save(output_path=str(output_path))
    
    assert os.path.exists(output_path), "Updated configuration file does not exist."
    
    with open(output_path, "r") as f:
        saved_config = yaml.safe_load(f)
    
    assert saved_config["training"]["num_epochs"] == 15, "Updated 'num_epochs' was not saved correctly."

def test_config_parser_get_with_invalid_key_format(tmp_yaml_config):
    """Test that get method handles invalid key formats gracefully."""
    parser = ConfigParser(config_path=tmp_yaml_config)
    with pytest.raises(TypeError):
        parser.get(123)  # Non-string key

def test_config_parser_update_with_non_dict(tmp_yaml_config):
    """Test that updating the configuration with a non-dictionary raises an error."""
    parser = ConfigParser(config_path=tmp_yaml_config)
    with pytest.raises(AttributeError):
        parser.update("not_a_dict")  # Non-dictionary update

# 9. Schema Validation with Additional Properties

def test_config_parser_schema_validation_additional_properties(tmp_yaml_config, tmp_yaml_schema):
    """Test that ConfigParser handles additional properties in the configuration."""
    # Modify the config to include an additional property not defined in the schema
    with open(tmp_yaml_config, "r") as f:
        config = yaml.safe_load(f)
    config["extra_property"] = "extra_value"
    with open(tmp_yaml_config, "w") as f:
        yaml.dump(config, f)
    
    with pytest.raises(ValidationError) as exc_info:
        ConfigParser(config_path=tmp_yaml_config, schema_path=tmp_yaml_schema)
    
    assert "Additional properties are not allowed" in str(exc_info.value), "ValidationError for additional properties not raised."

# 10. Schema Validation Missing Required Fields

def test_config_parser_schema_validation_missing_required_fields(tmp_yaml_config, tmp_yaml_schema, tmp_path):
    """Test that ConfigParser raises ValidationError when required fields are missing."""
    # Remove a required field from the config
    with open(tmp_yaml_config, "r") as f:
        config = yaml.safe_load(f)
    del config["model"]["model_type"]
    with open(tmp_yaml_config, "w") as f:
        yaml.dump(config, f)
    
    with pytest.raises(ValidationError) as exc_info:
        ConfigParser(config_path=tmp_yaml_config, schema_path=tmp_yaml_schema)
    
    assert "is a required property" in str(exc_info.value), "ValidationError for missing required fields not raised."

# 11. Schema Validation Invalid Data Types

def test_config_parser_schema_validation_invalid_data_types(tmp_yaml_config, tmp_yaml_schema):
    """Test that ConfigParser raises ValidationError when data types do not match the schema."""
    # Change a field to an incorrect data type
    with open(tmp_yaml_config, "r") as f:
        config = yaml.safe_load(f)
    config["training"]["num_epochs"] = "ten"  # Should be integer
    with open(tmp_yaml_config, "w") as f:
        yaml.dump(config, f)
    
    with pytest.raises(ValidationError) as exc_info:
        ConfigParser(config_path=tmp_yaml_config, schema_path=tmp_yaml_schema)
    
    assert "is not of type 'integer'" in str(exc_info.value), "ValidationError for invalid data types not raised."

# 12. Schema Validation Optional Fields

def test_config_parser_schema_validation_optional_fields(tmp_yaml_config, tmp_yaml_schema):
    """Test that ConfigParser correctly handles optional fields."""
    # Assume 'early_stopping_patience' is optional and remove it
    with open(tmp_yaml_config, "r") as f:
        config = yaml.safe_load(f)
    del config["training"]["early_stopping_patience"]
    with open(tmp_yaml_config, "w") as f:
        yaml.dump(config, f)
    
    # Update schema to make 'early_stopping_patience' optional
    with patch("src.utils.config_parser.yaml.safe_load") as mock_yaml_load:
        # Assuming schema allows optional 'early_stopping_patience'
        parser = ConfigParser(config_path=tmp_yaml_config, schema_path=tmp_yaml_schema)
        assert parser.get("training.early_stopping_patience") is None, "Optional field 'early_stopping_patience' should be None when not present."

# 13. Configuration Conversion Tests

def test_config_parser_to_dict(tmp_yaml_config):
    """Test that to_dict method returns the correct configuration dictionary."""
    parser = ConfigParser(config_path=tmp_yaml_config)
    config_dict = parser.to_dict()
    assert isinstance(config_dict, dict), "to_dict should return a dictionary."
    assert config_dict["optimizer"]["optimizer_type"] == "adamw", "Optimizer type mismatch in to_dict."

# 14. Configuration Update with Nested Dictionaries

def test_config_parser_update_nested_dictionaries(tmp_yaml_config):
    """Test that updating nested dictionaries works correctly."""
    parser = ConfigParser(config_path=tmp_yaml_config)
    updates = {
        "optimizer": {
            "learning_rate": 3e-5,
            "weight_decay": 0.02
        },
        "loss": {
            "classification_loss_weight": 1.5
        }
    }
    parser.update(updates)
    assert parser.get("optimizer.learning_rate") == 3e-5, "Nested key 'optimizer.learning_rate' was not updated correctly."
    assert parser.get("optimizer.weight_decay") == 0.02, "Nested key 'optimizer.weight_decay' was not updated correctly."
    assert parser.get("loss.classification_loss_weight") == 1.5, "Nested key 'loss.classification_loss_weight' was not updated correctly."

# 15. Configuration Retrieval with Incorrect Key Formats

def test_config_parser_get_with_incorrect_key_format(tmp_yaml_config, caplog):
    """Test that get method handles incorrect key formats gracefully."""
    parser = ConfigParser(config_path=tmp_yaml_config)
    # Attempt to retrieve a key using list instead of dot notation string
    with pytest.raises(AttributeError):
        parser.get(["model", "model_name"])
    # Attempt to retrieve a key using integer
    with pytest.raises(TypeError):
        parser.get(123)

# 16. Configuration Saving with Overwrite Protection

def test_config_parser_save_overwrite(tmp_yaml_config, tmp_path):
    """Test that saving a configuration overwrites existing files."""
    parser = ConfigParser(config_path=tmp_yaml_config)
    output_path = tmp_path / "config.yaml"
    parser.save(output_path=str(output_path))
    
    # Modify the configuration and save again to the same path
    parser.update({"model": {"model_name": "new-model-name"}})
    parser.save(output_path=str(output_path))
    
    with open(output_path, "r") as f:
        saved_config = yaml.safe_load(f)
    
    assert saved_config["model"]["model_name"] == "new-model-name", "Configuration file was not overwritten correctly."

# 17. Schema Validation with Additional Nested Properties

def test_config_parser_schema_validation_additional_nested_properties(tmp_yaml_config, tmp_yaml_schema):
    """Test that ConfigParser handles additional nested properties in the configuration."""
    # Add an additional nested property
    with open(tmp_yaml_config, "r") as f:
        config = yaml.safe_load(f)
    config["model"]["additional_param"] = {"sub_param": "value"}
    with open(tmp_yaml_config, "w") as f:
        yaml.dump(config, f)
    
    with pytest.raises(ValidationError) as exc_info:
        ConfigParser(config_path=tmp_yaml_config, schema_path=tmp_yaml_schema)
    
    assert "Additional properties are not allowed" in str(exc_info.value), "ValidationError for additional nested properties not raised."

# 18. Schema Validation with Incorrect Data Types in Nested Properties

def test_config_parser_schema_validation_incorrect_nested_data_types(tmp_yaml_config, tmp_yaml_schema):
    """Test that ConfigParser raises ValidationError for incorrect data types in nested properties."""
    # Change a nested property's data type
    with open(tmp_yaml_config, "r") as f:
        config = yaml.safe_load(f)
    config["scheduler"]["num_warmup_steps"] = "ten"  # Should be integer
    with open(tmp_yaml_config, "w") as f:
        yaml.dump(config, f)
    
    with pytest.raises(ValidationError) as exc_info:
        ConfigParser(config_path=tmp_yaml_config, schema_path=tmp_yaml_schema)
    
    assert "is not of type 'integer'" in str(exc_info.value), "ValidationError for incorrect nested data types not raised."

# 19. Schema Validation with Missing Entire Sections

def test_config_parser_schema_validation_missing_sections(tmp_yaml_config, tmp_yaml_schema, tmp_path):
    """Test that ConfigParser raises ValidationError when entire sections are missing."""
    # Remove the 'optimizer' section
    with open(tmp_yaml_config, "r") as f:
        config = yaml.safe_load(f)
    del config["optimizer"]
    with open(tmp_yaml_config, "w") as f:
        yaml.dump(config, f)
    
    with pytest.raises(ValidationError) as exc_info:
        ConfigParser(config_path=tmp_yaml_config, schema_path=tmp_yaml_schema)
    
    assert "is a required property" in str(exc_info.value), "ValidationError for missing sections not raised."

# 20. Schema Validation with Incorrect Schema Structure

def test_config_parser_schema_validation_incorrect_schema_structure(tmp_yaml_config, tmp_path):
    """Test that ConfigParser raises an error when the schema itself is malformed."""
    # Create a malformed schema
    malformed_schema_path = tmp_path / "malformed_schema.yaml"
    with open(malformed_schema_path, "w") as f:
        f.write("type: object\nproperties: [")  # Invalid YAML
    
    with pytest.raises(Exception) as exc_info:
        ConfigParser(config_path=tmp_yaml_config, schema_path=str(malformed_schema_path))
    
    assert "Error loading schema file" in str(exc_info.value), "Malformed schema should raise an error."

