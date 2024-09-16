# tests/unit/src/utils/test_logging_utils.py

import os
import pytest
import logging
from unittest.mock import patch, MagicMock, mock_open
from src.utils.logging_utils import (
    setup_logging,
    get_logger,
    log_model_info,
    log_metrics,
    dynamic_log_level
)
from jsonschema.exceptions import ValidationError

# --- Fixtures ---

@pytest.fixture
def valid_logging_config(tmp_path):
    """Fixture to create a valid YAML logging configuration file."""
    config = {
        "version": 1,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": "DEBUG"
            },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "standard",
                "filename": str(tmp_path / "test.log"),
                "level": "INFO"
            }
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["console", "file"],
                "level": "DEBUG",
                "propagate": False
            },
            "specific_logger": {
                "handlers": ["console"],
                "level": "WARNING",
                "propagate": False
            }
        }
    }
    config_path = tmp_path / "logging_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path

@pytest.fixture
def malformed_logging_config(tmp_path):
    """Fixture to create a malformed YAML logging configuration file."""
    malformed_content = "version: 1\nhandlers:\n  console:\n    class: logging.StreamHandler\n    formatter: standard\n    level: DEBUG\n  file:\n    class: logging.FileHandler\n    formatter: standard\n    filename: test.log\n    level: INFO\nloggers:\n  '':\n    handlers: [console, file]\n    level: DEBUG\n    propagate: False\n  'specific_logger':\n    handlers: [console]\n    level: WARNING\n    propagate: False\n    extra_field"  # Missing value for 'extra_field'
    config_path = tmp_path / "malformed_logging_config.yaml"
    with open(config_path, "w") as f:
        f.write(malformed_content)
    return config_path

@pytest.fixture
def temp_log_file(tmp_path):
    """Fixture to create a temporary log file."""
    return tmp_path / "temp_test.log"

# --- Test Cases ---

# 1. Logging Setup Tests

def test_setup_logging_with_valid_config(valid_logging_config, temp_log_file, caplog):
    """Test that setup_logging configures logging correctly with a valid config file."""
    with patch("src.utils.logging_utils.Path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=yaml.dump({
            "version": 1,
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": "DEBUG"
                }
            },
            "formatters": {
                "standard": {
                    "format": "%(levelname)s:%(name)s:%(message)s"
                }
            },
            "root": {
                "handlers": ["console"],
                "level": "DEBUG"
            }
        }))) as mocked_file:
            setup_logging(default_path=str(valid_logging_config))
    
    logger = get_logger()
    with caplog.at_level(logging.DEBUG):
        logger.debug("Debug message")
        logger.info("Info message")
    
    assert "Debug message" in caplog.text
    assert "Info message" in caplog.text

def test_setup_logging_with_missing_config(temp_log_file, caplog):
    """Test that setup_logging uses default configuration when config file is missing."""
    with patch("src.utils.logging_utils.Path.exists", return_value=False):
        with patch("logging.basicConfig") as mock_basicConfig:
            setup_logging(default_path="nonexistent_config.yaml")
            mock_basicConfig.assert_called_once_with(level=logging.INFO)
    
    logger = get_logger()
    with caplog.at_level(logging.INFO):
        logger.info("Info message")
    
    assert "Failed to load configuration file. Using default configs" in caplog.text
    assert "Info message" in caplog.text

def test_setup_logging_with_invalid_config(malformed_logging_config, caplog):
    """Test that setup_logging handles malformed config files gracefully."""
    with patch("src.utils.logging_utils.Path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data="invalid_yaml: [unbalanced brackets")):
            with patch("logging.config.dictConfig", side_effect=Exception("YAML Load Error")) as mock_dictConfig:
                with patch("logging.basicConfig") as mock_basicConfig:
                    setup_logging(default_path=str(malformed_logging_config))
                    mock_dictConfig.assert_called_once()
                    mock_basicConfig.assert_called_once_with(level=logging.INFO)
    
    assert "Error in Logging Configuration: YAML Load Error" in caplog.text
    logger = get_logger()
    with caplog.at_level(logging.INFO):
        logger.info("Info after error")
    
    assert "Info after error" in caplog.text

def test_setup_logging_with_env_variable(valid_logging_config, tmp_path, caplog):
    """Test that setup_logging loads configuration from environment variable if set."""
    os.environ['LOG_CFG'] = str(valid_logging_config)
    try:
        with patch("src.utils.logging_utils.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=yaml.dump({
                "version": 1,
                "handlers": {
                    "console": {
                        "class": "logging.StreamHandler",
                        "formatter": "standard",
                        "level": "DEBUG"
                    }
                },
                "formatters": {
                    "standard": {
                        "format": "%(levelname)s:%(name)s:%(message)s"
                    }
                },
                "root": {
                    "handlers": ["console"],
                    "level": "DEBUG"
                }
            }))) as mocked_file:
                setup_logging()
        
        logger = get_logger()
        with caplog.at_level(logging.DEBUG):
            logger.debug("Debug message from env")
        
        assert "Debug message from env" in caplog.text
    finally:
        del os.environ['LOG_CFG']

# 2. Logger Retrieval Tests

def test_get_root_logger(caplog):
    """Test retrieving the root logger."""
    logger = get_logger()
    assert isinstance(logger, logging.Logger), "get_logger should return a Logger instance."
    
    with caplog.at_level(logging.INFO):
        logger.info("Root logger info message")
    
    assert "Root logger info message" in caplog.text

def test_get_named_logger(caplog):
    """Test retrieving a named logger."""
    logger_name = "test_logger"
    logger = get_logger(logger_name)
    assert logger.name == logger_name, "Logger name does not match."
    
    with caplog.at_level(logging.INFO):
        logger.info("Named logger info message")
    
    assert "Named logger info message" in caplog.text

# 3. Model Information Logging Tests

def test_log_model_info_success(caplog):
    """Test that log_model_info logs model details correctly."""
    class MockModel:
        def __init__(self):
            self.__class__.__name__ = "MockModel"
        
        def parameters(self):
            return [MagicMock(numel=100, requires_grad=True),
                    MagicMock(numel=200, requires_grad=True)]
    
    mock_model = MockModel()
    logger = get_logger("test_logger")
    
    with caplog.at_level(logging.INFO):
        log_model_info(mock_model, logger)
    
    assert "Model: MockModel, Total trainable parameters: 300" in caplog.text

def test_log_model_info_no_parameters(caplog):
    """Test that log_model_info handles models with no trainable parameters."""
    class MockModelNoParams:
        def __init__(self):
            self.__class__.__name__ = "MockModelNoParams"
        
        def parameters(self):
            return []
    
    mock_model = MockModelNoParams()
    logger = get_logger("test_logger_no_params")
    
    with caplog.at_level(logging.INFO):
        log_model_info(mock_model, logger)
    
    assert "Model: MockModelNoParams, Total trainable parameters: 0" in caplog.text

# 4. Metrics Logging Tests

def test_log_metrics_with_epoch(caplog):
    """Test that log_metrics logs metrics correctly with epoch information."""
    metrics = {"accuracy": 0.95, "loss": 0.05}
    epoch = 5
    phase = "Validation"
    logger = get_logger("metrics_logger")
    
    with caplog.at_level(logging.INFO):
        log_metrics(metrics, epoch=epoch, phase=phase, logger=logger)
    
    assert f"Epoch {epoch} - {phase} Metrics:" in caplog.text
    assert "accuracy: 0.95" in caplog.text
    assert "loss: 0.05" in caplog.text

def test_log_metrics_without_epoch(caplog):
    """Test that log_metrics logs metrics correctly without epoch information."""
    metrics = {"precision": 0.88, "recall": 0.92}
    phase = "Training"
    logger = get_logger("metrics_logger_no_epoch")
    
    with caplog.at_level(logging.INFO):
        log_metrics(metrics, phase=phase, logger=logger)
    
    assert f"{phase} Metrics:" in caplog.text
    assert "precision: 0.88" in caplog.text
    assert "recall: 0.92" in caplog.text

def test_log_metrics_empty_metrics(caplog):
    """Test that log_metrics handles empty metrics gracefully."""
    metrics = {}
    phase = "Testing"
    logger = get_logger("empty_metrics_logger")
    
    with caplog.at_level(logging.INFO):
        log_metrics(metrics, phase=phase, logger=logger)
    
    assert f"{phase} Metrics:" in caplog.text
    # No metrics should be logged
    for record in caplog.records:
        assert "Metrics:" in record.message
        assert len(record.message.strip().split("\n")) == 1, "No metrics should be logged for empty metrics."

# 5. Dynamic Log Level Tests

def test_dynamic_log_level_success(caplog):
    """Test that dynamic_log_level successfully changes the log level."""
    logger = get_logger("dynamic_logger")
    
    with caplog.at_level(logging.INFO):
        logger.info("Before log level change - INFO")
        logger.debug("Before log level change - DEBUG")
    
    # Change log level to DEBUG
    with caplog.at_level(logging.DEBUG):
        dynamic_log_level("DEBUG")
        logger.debug("After log level change - DEBUG")
        logger.info("After log level change - INFO")
    
    assert "Before log level change - INFO" in caplog.text
    assert "Before log level change - DEBUG" not in caplog.text
    assert "After log level change - DEBUG" in caplog.text
    assert "After log level change - INFO" in caplog.text

def test_dynamic_log_level_invalid_level(caplog):
    """Test that dynamic_log_level raises ValueError for invalid log levels."""
    with pytest.raises(ValueError) as exc_info:
        dynamic_log_level("INVALID_LEVEL")
    
    assert "Invalid log level: INVALID_LEVEL" in str(exc_info.value)

# 6. Edge Case Tests

def test_log_model_info_nested_parameters(caplog):
    """Test that log_model_info correctly counts nested parameters."""
    class NestedMockParam:
        def __init__(self, numel, requires_grad):
            self.numel = numel
            self.requires_grad = requires_grad
    
    class MockNestedModel:
        def __init__(self):
            self.__class__.__name__ = "MockNestedModel"
        
        def parameters(self):
            return [
                NestedMockParam(50, True),
                NestedMockParam(150, True),
                NestedMockParam(200, False)  # Should not be counted
            ]
    
    mock_model = MockNestedModel()
    logger = get_logger("nested_param_logger")
    
    with caplog.at_level(logging.INFO):
        log_model_info(mock_model, logger)
    
    assert "Model: MockNestedModel, Total trainable parameters: 200" in caplog.text

def test_log_metrics_various_data_types(caplog):
    """Test that log_metrics handles various data types correctly."""
    metrics = {
        "accuracy": 0.95,
        "loss": 0.05,
        "confusion_matrix": [[50, 2], [1, 47]],
        "class_report": {"precision": 0.96, "recall": 0.94}
    }
    phase = "Evaluation"
    logger = get_logger("various_data_types_logger")
    
    with caplog.at_level(logging.INFO):
        log_metrics(metrics, phase=phase, logger=logger)
    
    assert f"{phase} Metrics:" in caplog.text
    assert "accuracy: 0.95" in caplog.text
    assert "loss: 0.05" in caplog.text
    assert "confusion_matrix: [[50, 2], [1, 47]]" in caplog.text
    assert "class_report: {'precision': 0.96, 'recall': 0.94}" in caplog.text

def test_setup_logging_with_unsupported_file_format(tmp_path, caplog):
    """Test that setup_logging handles unsupported configuration file formats."""
    unsupported_config = tmp_path / "logging_config.txt"
    with open(unsupported_config, "w") as f:
        f.write("unsupported content")
    
    with patch("src.utils.logging_utils.Path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data="unsupported content")):
            with patch("logging.config.dictConfig") as mock_dictConfig:
                setup_logging(default_path=str(unsupported_config))
                # Since the content is unsupported, dictConfig might still be called, but in reality, it's more about the file extension.
                # However, in the provided `setup_logging`, it doesn't check for file extension, so it tries to load whatever is there.
                mock_dictConfig.assert_called()
    
    logger = get_logger()
    with caplog.at_level(logging.INFO):
        logger.info("Info message after unsupported config.")
    
    assert "Info message after unsupported config." in caplog.text

# 7. Additional Edge Case Tests

def test_log_metrics_with_non_string_keys(caplog):
    """Test that log_metrics handles non-string metric keys gracefully."""
    metrics = {
        1: "one",
        2: "two"
    }
    phase = "Numeric Keys Phase"
    logger = get_logger("numeric_keys_logger")
    
    with caplog.at_level(logging.INFO):
        log_metrics(metrics, phase=phase, logger=logger)
    
    assert f"{phase} Metrics:" in caplog.text
    assert "1: one" in caplog.text
    assert "2: two" in caplog.text

def test_log_model_info_with_inaccessible_parameters(caplog):
    """Test that log_model_info handles parameters that cannot be accessed."""
    class MockModelInaccessibleParams:
        def __init__(self):
            self.__class__.__name__ = "MockModelInaccessibleParams"
        
        def parameters(self):
            return [MagicMock()]
    
    mock_model = MockModelInaccessibleParams()
    mock_model.parameters()[0].numel = 100
    mock_model.parameters()[0].requires_grad = True
    mock_model.parameters()[0].__getattr__.side_effect = AttributeError("Inaccessible attribute")
    
    logger = get_logger("inaccessible_params_logger")
    
    with caplog.at_level(logging.INFO):
        # The parameter count should handle exceptions internally if any
        log_model_info(mock_model, logger)
    
    assert "Model: MockModelInaccessibleParams, Total trainable parameters: 100" in caplog.text

def test_dynamic_log_level_case_insensitive(caplog):
    """Test that dynamic_log_level handles log levels case-insensitively."""
    logger = get_logger("case_insensitive_logger")
    
    with caplog.at_level(logging.WARNING):
        logger.debug("This DEBUG message should not appear.")
        logger.warning("This WARNING message should appear.")
    
    assert "This WARNING message should appear." in caplog.text
    assert "This DEBUG message should not appear." not in caplog.text
    
    # Change log level to DEBUG using lowercase
    with caplog.at_level(logging.DEBUG):
        dynamic_log_level("debug")
        logger.debug("This DEBUG message should now appear.")
    
    assert "Log level changed to debug" in caplog.text
    assert "This DEBUG message should now appear." in caplog.text

def test_get_logger_without_name(caplog):
    """Test that get_logger without a name retrieves the root logger."""
    logger = get_logger()
    assert logger.name == "", "Default logger should be the root logger."
    
    with caplog.at_level(logging.INFO):
        logger.info("Root logger info message.")
    
    assert "Root logger info message." in caplog.text

def test_log_metrics_with_large_values(caplog):
    """Test that log_metrics handles large metric values correctly."""
    metrics = {
        "very_large_number": 1e18,
        "small_number": 1e-18
    }
    phase = "Large Numbers Phase"
    logger = get_logger("large_numbers_logger")
    
    with caplog.at_level(logging.INFO):
        log_metrics(metrics, phase=phase, logger=logger)
    
    assert f"{phase} Metrics:" in caplog.text
    assert "very_large_number: 1e+18" in caplog.text
    assert "small_number: 1e-18" in caplog.text

def test_dynamic_log_level_multiple_handlers(caplog):
    """Test that dynamic_log_level changes the log level across all handlers."""
    with patch("src.utils.logging_utils.Path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=yaml.dump({
            "version": 1,
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": "INFO"
                },
                "file": {
                    "class": "logging.FileHandler",
                    "formatter": "standard",
                    "filename": "test.log",
                    "level": "INFO"
                }
            },
            "formatters": {
                "standard": {
                    "format": "%(levelname)s:%(name)s:%(message)s"
                }
            },
            "root": {
                "handlers": ["console", "file"],
                "level": "INFO"
            }
        }))) as mocked_file:
            setup_logging(default_path="valid_logging_config.yaml")
    
    logger = get_logger("multi_handler_logger")
    with caplog.at_level(logging.INFO):
        logger.info("Info before level change.")
        logger.debug("Debug before level change.")
    
    assert "Info before level change." in caplog.text
    assert "Debug before level change." not in caplog.text
    
    # Change log level to DEBUG
    with caplog.at_level(logging.DEBUG):
        dynamic_log_level("DEBUG")
        logger.debug("Debug after level change.")
        logger.info("Info after level change.")
    
    assert "Debug after level change." in caplog.text
    assert "Info after level change." in caplog.text

# 8. Test Logging Integration with Multiple Loggers

def test_logging_with_multiple_loggers(caplog):
    """Test that multiple loggers operate independently based on their configurations."""
    logger1 = get_logger("logger1")
    logger2 = get_logger("logger2")
    
    with caplog.at_level(logging.INFO):
        logger1.info("Logger1 INFO message.")
        logger2.info("Logger2 INFO message.")
        logger1.debug("Logger1 DEBUG message.")
        logger2.debug("Logger2 DEBUG message.")
    
    # Assuming logger1 and logger2 are both root loggers or have the same level
    # Since setup_logging was called in previous tests, levels might be set accordingly
    # Adjust expectations based on actual logging configurations
    assert "Logger1 INFO message." in caplog.text
    assert "Logger2 INFO message." in caplog.text
    # DEBUG messages may or may not appear depending on the logger's level
    # For this test, assume default level is INFO, so DEBUG should not appear
    assert "Logger1 DEBUG message." not in caplog.text
    assert "Logger2 DEBUG message." not in caplog.text

# 9. Test Dynamic Log Level Persistence

def test_dynamic_log_level_persistence(caplog):
    """Test that dynamic_log_level changes persist across different logger instances."""
    logger1 = get_logger("persistent_logger1")
    logger2 = get_logger("persistent_logger2")
    
    # Change log level to ERROR
    with caplog.at_level(logging.ERROR):
        dynamic_log_level("ERROR")
        logger1.debug("Logger1 DEBUG after ERROR level.")
        logger2.error("Logger2 ERROR after ERROR level.")
    
    assert "Log level changed to ERROR" in caplog.text
    assert "Logger1 DEBUG after ERROR level." not in caplog.text
    assert "Logger2 ERROR after ERROR level." in caplog.text
    
    # Change log level back to INFO
    with caplog.at_level(logging.INFO):
        dynamic_log_level("INFO")
        logger1.info("Logger1 INFO after INFO level.")
        logger2.debug("Logger2 DEBUG after INFO level.")
    
    assert "Log level changed to INFO" in caplog.text
    assert "Logger1 INFO after INFO level." in caplog.text
    assert "Logger2 DEBUG after INFO level." not in caplog.text

# 10. Test Log Model Info with Complex Models

def test_log_model_info_complex_model(caplog):
    """Test log_model_info with a model that has nested modules and parameters."""
    class NestedMockParam:
        def __init__(self, numel, requires_grad):
            self.numel = numel
            self.requires_grad = requires_grad
    
    class MockSubModule:
        def parameters(self):
            return [NestedMockParam(50, True), NestedMockParam(150, True)]
    
    class MockComplexModel:
        def __init__(self):
            self.__class__.__name__ = "MockComplexModel"
            self.submodule = MockSubModule()
        
        def parameters(self):
            return self.submodule.parameters()
    
    mock_model = MockComplexModel()
    logger = get_logger("complex_model_logger")
    
    with caplog.at_level(logging.INFO):
        log_model_info(mock_model, logger)
    
    assert "Model: MockComplexModel, Total trainable parameters: 200" in caplog.text

# 11. Test Log Metrics with Non-String Values

def test_log_metrics_with_non_string_values(caplog):
    """Test that log_metrics handles non-string metric values correctly."""
    metrics = {
        "accuracy": 0.95,
        "model": MagicMock(),
        "timestamp": None
    }
    phase = "Test Phase"
    logger = get_logger("non_string_values_logger")
    
    with caplog.at_level(logging.INFO):
        log_metrics(metrics, phase=phase, logger=logger)
    
    assert f"{phase} Metrics:" in caplog.text
    assert "accuracy: 0.95" in caplog.text
    assert "model: <MagicMock name='mock()' id=" in caplog.text  # MagicMock representation
    assert "timestamp: None" in caplog.text

# 12. Test Log Metrics with Special Characters in Keys

def test_log_metrics_with_special_characters_in_keys(caplog):
    """Test that log_metrics handles metric keys with special characters."""
    metrics = {
        "accuracy@epoch#1": 0.95,
        "loss (validation)": 0.05
    }
    phase = "Special Characters Phase"
    logger = get_logger("special_chars_logger")
    
    with caplog.at_level(logging.INFO):
        log_metrics(metrics, phase=phase, logger=logger)
    
    assert f"{phase} Metrics:" in caplog.text
    assert "accuracy@epoch#1: 0.95" in caplog.text
    assert "loss (validation): 0.05" in caplog.text

# 13. Test Dynamic Log Level with Multiple Handlers

def test_dynamic_log_level_with_multiple_handlers(caplog, valid_logging_config, tmp_path):
    """Test that dynamic_log_level changes the log level across all handlers."""
    # Setup logging with multiple handlers
    with patch("src.utils.logging_utils.Path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=yaml.dump({
            "version": 1,
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": "INFO"
                },
                "file": {
                    "class": "logging.FileHandler",
                    "formatter": "standard",
                    "filename": str(tmp_path / "test_multi.log"),
                    "level": "INFO"
                }
            },
            "formatters": {
                "standard": {
                    "format": "%(levelname)s:%(name)s:%(message)s"
                }
            },
            "root": {
                "handlers": ["console", "file"],
                "level": "INFO"
            }
        }))) as mocked_file:
            setup_logging(default_path=str(valid_logging_config))
    
    logger = get_logger("multi_handler_logger")
    
    with caplog.at_level(logging.INFO):
        logger.info("Initial INFO message.")
        logger.debug("Initial DEBUG message.")
    
    assert "Initial INFO message." in caplog.text
    assert "Initial DEBUG message." not in caplog.text
    
    # Change log level to DEBUG
    with caplog.at_level(logging.DEBUG):
        dynamic_log_level("DEBUG")
        logger.debug("DEBUG after level change.")
        logger.info("INFO after level change.")
    
    assert "Log level changed to DEBUG" in caplog.text
    assert "DEBUG after level change." in caplog.text
    assert "INFO after level change." in caplog.text
    
    # Verify that the file handler also logs DEBUG messages
    file_log_path = tmp_path / "test_multi.log"
    with open(file_log_path, "r") as f:
        file_logs = f.read()
    
    assert "DEBUG after level change." in file_logs
    assert "INFO after level change." in file_logs

# 14. Test Log Metrics with Large Number of Metrics

def test_log_metrics_with_large_number_of_metrics(caplog):
    """Test that log_metrics can handle a large number of metrics."""
    metrics = {f"metric_{i}": i for i in range(1000)}
    phase = "Large Metrics Phase"
    logger = get_logger("large_metrics_logger")
    
    with caplog.at_level(logging.INFO):
        log_metrics(metrics, phase=phase, logger=logger)
    
    assert f"{phase} Metrics:" in caplog.text
    for i in range(1000):
        assert f"metric_{i}: {i}" in caplog.text

# 15. Test Log Metrics with Unicode Characters

def test_log_metrics_with_unicode_characters(caplog):
    """Test that log_metrics handles metric keys and values with Unicode characters."""
    metrics = {
        "准确率": 0.95,
        "损失": 0.05,
        "复现率": 0.92
    }
    phase = "Unicode Phase"
    logger = get_logger("unicode_metrics_logger")
    
    with caplog.at_level(logging.INFO):
        log_metrics(metrics, phase=phase, logger=logger)
    
    assert f"{phase} Metrics:" in caplog.text
    assert "准确率: 0.95" in caplog.text
    assert "损失: 0.05" in caplog.text
    assert "复现率: 0.92" in caplog.text

# 16. Test Log Model Info with Dynamic Attributes

def test_log_model_info_with_dynamic_attributes(caplog):
    """Test that log_model_info logs dynamic attributes of a model."""
    class MockDynamicModel:
        def __init__(self):
            self.__class__.__name__ = "MockDynamicModel"
            self.dynamic_param = "dynamic_value"
        
        def parameters(self):
            return [MagicMock(numel=100, requires_grad=True)]
    
    mock_model = MockDynamicModel()
    logger = get_logger("dynamic_attributes_logger")
    
    with caplog.at_level(logging.INFO):
        log_model_info(mock_model, logger)
    
    assert "Model: MockDynamicModel, Total trainable parameters: 100" in caplog.text

# 17. Test Log Metrics with Iterable Values

def test_log_metrics_with_iterable_values(caplog):
    """Test that log_metrics handles iterable metric values correctly."""
    metrics = {
        "confusion_matrix": [[50, 2], [1, 47]],
        "predictions": [0, 1, 1, 0, 1]
    }
    phase = "Iterable Metrics Phase"
    logger = get_logger("iterable_metrics_logger")
    
    with caplog.at_level(logging.INFO):
        log_metrics(metrics, phase=phase, logger=logger)
    
    assert f"{phase} Metrics:" in caplog.text
    assert "confusion_matrix: [[50, 2], [1, 47]]" in caplog.text
    assert "predictions: [0, 1, 1, 0, 1]" in caplog.text

# 18. Test Setup Logging with Empty Configuration File

def test_setup_logging_with_empty_config(tmp_path, caplog):
    """Test that setup_logging handles an empty configuration file gracefully."""
    empty_config = tmp_path / "empty_logging_config.yaml"
    open(empty_config, "w").close()  # Create an empty file
    
    with patch("src.utils.logging_utils.Path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data="")):
            with patch("logging.config.dictConfig", side_effect=Exception("Empty config")) as mock_dictConfig:
                with patch("logging.basicConfig") as mock_basicConfig:
                    setup_logging(default_path=str(empty_config))
                    mock_dictConfig.assert_called_once()
                    mock_basicConfig.assert_called_once_with(level=logging.INFO)
    
    logger = get_logger()
    with caplog.at_level(logging.INFO):
        logger.info("Info after empty config.")
    
    assert "Error in Logging Configuration: Empty config" in caplog.text
    assert "Info after empty config." in caplog.text

# 19. Test Log Metrics with Boolean Values

def test_log_metrics_with_boolean_values(caplog):
    """Test that log_metrics handles boolean metric values correctly."""
    metrics = {
        "is_converged": True,
        "has_overfitted": False
    }
    phase = "Boolean Metrics Phase"
    logger = get_logger("boolean_metrics_logger")
    
    with caplog.at_level(logging.INFO):
        log_metrics(metrics, phase=phase, logger=logger)
    
    assert f"{phase} Metrics:" in caplog.text
    assert "is_converged: True" in caplog.text
    assert "has_overfitted: False" in caplog.text

# 20. Test Log Metrics with Nested Dictionaries

def test_log_metrics_with_nested_dictionaries(caplog):
    """Test that log_metrics handles nested dictionaries correctly."""
    metrics = {
        "layer1": {"accuracy": 0.95, "loss": 0.05},
        "layer2": {"accuracy": 0.96, "loss": 0.04}
    }
    phase = "Nested Metrics Phase"
    logger = get_logger("nested_metrics_logger")
    
    with caplog.at_level(logging.INFO):
        log_metrics(metrics, phase=phase, logger=logger)
    
    assert f"{phase} Metrics:" in caplog.text
    assert "layer1: {'accuracy': 0.95, 'loss': 0.05}" in caplog.text
    assert "layer2: {'accuracy': 0.96, 'loss': 0.04}" in caplog.text

