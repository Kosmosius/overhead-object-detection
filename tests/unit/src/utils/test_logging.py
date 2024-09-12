# tests/unit/src/utils/test_logging.py

import logging
import os
from src.utils.logging import setup_logging, get_logger_for_module, log_model_info, log_metrics

def test_setup_logging(tmp_path):
    log_file = os.path.join(tmp_path, "test.log")
    setup_logging(log_file=log_file, log_level=logging.DEBUG)
    logger = logging.getLogger()
    logger.debug("Test debug message")
    
    assert os.path.exists(log_file)
    with open(log_file, "r") as f:
        content = f.read()
        assert "Test debug message" in content

def test_get_logger_for_module():
    logger = get_logger_for_module("test_module")
    assert isinstance(logger, logging.Logger)
    logger.info("Test message")

def test_log_model_info(caplog):
    logger = logging.getLogger("test_logger")
    class MockModel:
        def parameters(self):
            return [torch.ones(1, 1, requires_grad=True)]
    mock_model = MockModel()
    
    with caplog.at_level(logging.INFO):
        log_model_info(mock_model, logger)
    
    assert "Total trainable parameters" in caplog.text

def test_log_metrics(caplog):
    logger = logging.getLogger("test_logger")
    metrics = {"precision": 0.8, "recall": 0.7, "f1_score": 0.75}
    
    with caplog.at_level(logging.INFO):
        log_metrics(metrics, epoch=1, phase="Training", logger=logger)
    
    assert "Epoch 1 - Training Metrics" in caplog.text
    assert "precision: 0.8" in caplog.text
