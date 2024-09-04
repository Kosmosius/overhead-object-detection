# src/utils/config_parser.py

import yaml
import json
import os
from transformers import PretrainedConfig

class ConfigParser:
    """
    Utility class for loading and parsing configuration files (YAML, JSON).
    Provides integration with HuggingFace PretrainedConfig.
    """

    def __init__(self, config_path):
        """
        Initialize the ConfigParser with the provided configuration file path.

        Args:
            config_path (str): Path to the configuration file (YAML or JSON).
        """
        self.config_path = config_path
        self.config = self.load_config(config_path)

    def load_config(self, config_path):
        """
        Load a YAML or JSON configuration file.

        Args:
            config_path (str): Path to the configuration file.

        Returns:
            dict: Parsed configuration data.
        """
        ext = os.path.splitext(config_path)[1]
        if ext == '.yaml' or ext == '.yml':
            return self._load_yaml(config_path)
        elif ext == '.json':
            return self._load_json(config_path)
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}")

    def _load_yaml(self, config_path):
        """
        Load a YAML configuration file.

        Args:
            config_path (str): Path to the YAML file.

        Returns:
            dict: Parsed YAML configuration.
        """
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def _load_json(self, config_path):
        """
        Load a JSON configuration file.

        Args:
            config_path (str): Path to the JSON file.

        Returns:
            dict: Parsed JSON configuration.
        """
        with open(config_path, 'r') as file:
            return json.load(file)

    def get(self, key, default=None):
        """
        Retrieve a value from the configuration.

        Args:
            key (str): The key to retrieve from the configuration.
            default (any): The default value if the key is not found.

        Returns:
            any: The value corresponding to the key, or the default value.
        """
        return self.config.get(key, default)

    def to_huggingface_config(self):
        """
        Convert the parsed configuration to a HuggingFace PretrainedConfig.

        Returns:
            PretrainedConfig: HuggingFace configuration object.
        """
        return PretrainedConfig(**self.config)

    def save(self, output_path):
        """
        Save the current configuration to a file (YAML or JSON).

        Args:
            output_path (str): Path to save the configuration file.
        """
        ext = os.path.splitext(output_path)[1]
        if ext == '.yaml' or ext == '.yml':
            self._save_yaml(output_path)
        elif ext == '.json':
            self._save_json(output_path)
        else:
            raise ValueError(f"Unsupported file format for saving: {ext}")

    def _save_yaml(self, output_path):
        """
        Save the configuration as a YAML file.

        Args:
            output_path (str): Path to save the YAML file.
        """
        with open(output_path, 'w') as file:
            yaml.safe_dump(self.config, file)

    def _save_json(self, output_path):
        """
        Save the configuration as a JSON file.

        Args:
            output_path (str): Path to save the JSON file.
        """
        with open(output_path, 'w') as file:
            json.dump(self.config, file, indent=4)
