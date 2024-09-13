# src/utils/config_parser.py

import yaml
import json
import os
from typing import Any, Dict, Optional
import logging
import jsonschema
from jsonschema.exceptions import ValidationError

class ConfigParser:
    """
    Utility class for loading and parsing configuration files (YAML, JSON).
    Provides integration with schema validation for correctness.
    """

    def __init__(self, config_path: str, schema_path: Optional[str] = None):
        """
        Initialize the ConfigParser with the provided configuration file path.

        Args:
            config_path (str): Path to the configuration file (YAML or JSON).
            schema_path (Optional[str]): Optional path to a schema file (YAML or JSON) for validation.
        """
        self.config_path = config_path
        self.schema_path = schema_path
        self.config = self.load_config()
        if self.schema_path:
            self.validate_schema()

    def load_config(self) -> Dict[str, Any]:
        """
        Load a YAML or JSON configuration file.

        Returns:
            Dict[str, Any]: Parsed configuration data.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not supported.
            Exception: For other I/O or parsing errors.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        ext = os.path.splitext(self.config_path)[1].lower()

        try:
            if ext in ['.yaml', '.yml']:
                return self._load_yaml()
            elif ext == '.json':
                return self._load_json()
            else:
                raise ValueError(f"Unsupported configuration file format: {ext}")
        except Exception as e:
            logging.error(f"Error loading configuration file: {str(e)}")
            raise

    def _load_yaml(self) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def _load_json(self) -> Dict[str, Any]:
        """Load a JSON configuration file."""
        with open(self.config_path, 'r') as file:
            return json.load(file)

    def validate_schema(self) -> None:
        """
        Validate the configuration against a provided schema (YAML or JSON).

        Raises:
            ValidationError: If the configuration does not conform to the schema.
            Exception: For any errors loading or validating the schema.
        """
        if not os.path.exists(self.schema_path):
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

        ext = os.path.splitext(self.schema_path)[1].lower()

        try:
            if ext in ['.yaml', '.yml']:
                with open(self.schema_path, 'r') as schema_file:
                    schema = yaml.safe_load(schema_file)
            elif ext == '.json':
                with open(self.schema_path, 'r') as schema_file:
                    schema = json.load(schema_file)
            else:
                raise ValueError(f"Unsupported schema file format: {ext}")

            jsonschema.validate(instance=self.config, schema=schema)
            logging.info(f"Configuration at {self.config_path} is valid against the schema.")
        except ValidationError as e:
            logging.error(f"Schema validation error: {e.message}")
            raise ValidationError(f"Schema validation error: {e.message}")
        except Exception as e:
            logging.error(f"Error loading schema file: {str(e)}")
            raise

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Retrieve a value from the configuration using a dotted key path.

        Args:
            key (str): The key to retrieve from the configuration, using dot notation for nested keys.
            default (Any, optional): The default value if the key is not found.

        Returns:
            Any: The value corresponding to the key, or the default value.
        """
        keys = key.split(".")
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            logging.warning(f"Key '{key}' not found in the configuration. Returning default value.")
            return default

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update the configuration with new values.

        Args:
            updates (Dict[str, Any]): Dictionary of values to update in the configuration.
        """
        self._update_dict(self.config, updates)
        logging.info(f"Configuration updated with values: {updates}")

    def _update_dict(self, d: Dict[str, Any], u: Dict[str, Any]) -> None:
        """Recursively update a dictionary."""
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = d.get(k, {})
                self._update_dict(d[k], v)
            else:
                d[k] = v

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
                with open(output_path, 'w') as file:
                    yaml.safe_dump(self.config, file)
            elif ext == '.json':
                with open(output_path, 'w') as file:
                    json.dump(self.config, file, indent=4)
            else:
                raise ValueError(f"Unsupported file format for saving: {ext}")
            logging.info(f"Configuration saved to {output_path}")
        except Exception as e:
            logging.error(f"Error saving configuration file: {str(e)}")
            raise

    def display(self) -> None:
        """Print the current configuration in a readable format."""
        print(yaml.dump(self.config, default_flow_style=False))

    def to_dict(self) -> Dict[str, Any]:
        """
        Get the configuration as a dictionary.

        Returns:
            Dict[str, Any]: The configuration dictionary.
        """
        return self.config
