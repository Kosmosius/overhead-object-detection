# tests/unit/src/utils/test_config_parser.py

import os
import pytest
import json
import yaml
from src.utils.config_parser import ConfigParser

@pytest.fixture
def yaml_config(tmp_path):
    config = {"model": "detr-resnet-50", "num_classes": 91}
    config_path = os.path.join(tmp_path, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path

@pytest.fixture
def json_config(tmp_path):
    config = {"model": "detr-resnet-50", "num_classes": 91}
    config_path = os.path.join(tmp_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)
    return config_path

def test_load_yaml_config(yaml_config):
    parser = ConfigParser(config_path=yaml_config)
    config = parser.load_config()
    assert config["model"] == "detr-resnet-50"

def test_load_json_config(json_config):
    parser = ConfigParser(config_path=json_config)
    config = parser.load_config()
    assert config["model"] == "detr-resnet-50"

def test_get_value(yaml_config):
    parser = ConfigParser(config_path=yaml_config)
    model = parser.get("model")
    assert model == "detr-resnet-50"

def test_save_config(tmp_path, yaml_config):
    parser = ConfigParser(config_path=yaml_config)
    output_path = os.path.join(tmp_path, "output.yaml")
    parser.save(output_path)
    assert os.path.exists(output_path)

def test_update_config(yaml_config):
    parser = ConfigParser(config_path=yaml_config)
    parser.update({"new_key": "new_value"})
    assert parser.get("new_key") == "new_value"

def test_to_huggingface_config(json_config):
    parser = ConfigParser(config_path=json_config)
    hf_config = parser.to_huggingface_config()
    assert hf_config.model == "detr-resnet-50"
