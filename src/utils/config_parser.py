# src/utils/config_parser.py

import yaml
import json
import os
from transformers import PretrainedConfig
from typing import Any, Dict, Optional, Union
import jsonschema
from jsonschema.exceptions import ValidationError


class ConfigParser:
    """
    Utility class for loading and parsing configuration files (YAML, JSON).
    Provides integration with HuggingFace PretrainedConfig.
    Supports schema validation for correctness.
    """

    def __init__(self, config_path: str, schema_path: Optional[str] = None):
        """
        Initialize the ConfigParser with the provided configuration file path.

        Args:
            config_path (str): Path to the configuration file (YAML or JSON).
            schema_path (Optional[str]): Optional path to a JSON schema for validation.
        """
        self.config_path = config_path
        self.config = self.load_config(config_path)

        if schema_path:
            self.validate_schema(schema_path)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load a YAML or JSON configuration file.

        Args:
            config_path (str): Path to the configuration file.

        Returns:
            Dict[str, Any]: Parsed configuration data.

        Raises:
            ValueError: If the file extension is not supported.
            FileNotFoundError: If the file does not exist.
            Exception: For other I/O or parsing errors.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        ext = os.path.splitext(config_path)[1].lower()

        try:
            if ext in ['.yaml', '.yml']:
                return self._load_yaml(config_path)
            elif ext == '.json':
                return self._load_json(config_path)
            else:
                raise ValueError(f"Unsupported configuration file format: {ext}")
        except Exception as e:
            raise Exception(f"Error loading configuration file: {str(e)}")

    def _load_yaml(self, config_path: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.

        Args:
            config_path (str): Path to the YAML file.

        Returns:
            Dict[str, Any]: Parsed YAML configuration.
        """
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def _load_json(self, config_path: str) -> Dict[str, Any]:
        """
        Load a JSON configuration file.

        Args:
            config_path (str): Path to the JSON file.

        Returns:
            Dict[str, Any]: Parsed JSON configuration.
        """
        with open(config_path, 'r') as file:
            return json.load(file)

    def validate_schema(self, schema_path: str) -> None:
        """
        Validate the configuration against a provided JSON schema.

        Args:
            schema_path (str): Path to the JSON schema file.

        Raises:
            ValidationError: If the configuration does not conform to the schema.
            Exception: For any errors loading or validating the schema.
        """
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        try:
            with open(schema_path, 'r') as schema_file:
                schema = json.load(schema_file)

            jsonschema.validate(instance=self.config, schema=schema)
            print(f"Configuration at {self.config_path} is valid against the schema.")
        except ValidationError as e:
            raise ValidationError(f"Schema validation error: {e.message}")
        except Exception as e:
            raise Exception(f"Error loading schema file: {str(e)}")

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Retrieve a value from the configuration.

        Args:
            key (str): The key to retrieve from the configuration.
            default (Any, optional): The default value if the key is not found.

        Returns:
            Any: The value corresponding to the key, or the default value.
        """
        return self.config.get(key, default)

    def to_huggingface_config(self) -> PretrainedConfig:
        """
        Convert the parsed configuration to a HuggingFace PretrainedConfig.

        Returns:
            PretrainedConfig: HuggingFace configuration object.

        Raises:
            ValueError: If the configuration cannot be converted to PretrainedConfig.
        """
        try:
            return PretrainedConfig(**self.config)
        except Exception as e:
            raise ValueError(f"Error converting configuration to HuggingFace PretrainedConfig: {str(e)}")

    def save(self, output_path: str) -> None:
        """
        Save the current configuration to a file (YAML or JSON).

        Args:
            output_path (str): Path to save the configuration file.

        Raises:
            ValueError: If the file extension is unsupported.
            Exception: For I/O errors during saving.
        """
        ext = os.path.splitext(output_path)[1].lower()
        try:
            if ext in ['.yaml', '.yml']:
                self._save_yaml(output_path)
            elif ext == '.json':
                self._save_json(output_path)
            else:
                raise ValueError(f"Unsupported file format for saving: {ext}")
        except Exception as e:
            raise Exception(f"Error saving configuration file: {str(e)}")

    def _save_yaml(self, output_path: str) -> None:
        """
        Save the configuration as a YAML file.

        Args:
            output_path (str): Path to save the YAML file.
        """
        with open(output_path, 'w') as file:
            yaml.safe_dump(self.config, file)

    def _save_json(self, output_path: str) -> None:
        """
        Save the configuration as a JSON file.

        Args:
            output_path (str): Path to save the JSON file.
        """
        with open(output_path, 'w') as file:
            json.dump(self.config, file, indent=4)
