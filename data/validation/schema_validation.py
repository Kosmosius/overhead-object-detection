# src/data/validation/schema_validation.py

import jsonschema
import json

def validate_coco_schema(annotations_path, schema_path):
    """
    Validate COCO annotations against a schema.
    
    Args:
        annotations_path (str): Path to the COCO annotations file.
        schema_path (str): Path to the JSON schema file.
    
    Returns:
        bool: True if the annotations conform to the schema, False otherwise.
    """
    try:
        with open(schema_path, 'r') as schema_file:
            schema = json.load(schema_file)
        with open(annotations_path, 'r') as annotations_file:
            annotations = json.load(annotations_file)
        jsonschema.validate(instance=annotations, schema=schema)
        print(f"COCO annotations file {annotations_path} conforms to the schema.")
        return True
    except jsonschema.exceptions.ValidationError as e:
        print(f"Validation error: {e.message}")
        return False
    except Exception as e:
        print(f"Error loading files: {e}")
        return False

if __name__ == "__main__":
    ANNOTATIONS_PATH = "/path/to/coco/annotations.json"  # Replace with actual path
    SCHEMA_PATH = "/path/to/coco/schema.json"  # Replace with actual path
    
    schema_valid = validate_coco_schema(ANNOTATIONS_PATH, SCHEMA_PATH)
    
    if schema_valid:
        print("Schema validation passed.")
    else:
        print("Schema validation failed.")
