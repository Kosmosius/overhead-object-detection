# src/data/validation/geojson_validator.py

import json
from pycocotools.coco import COCO

def validate_geojson(geojson_path):
    """
    Validate a GeoJSON file.
    
    Args:
        geojson_path (str): Path to the GeoJSON file.
    
    Returns:
        bool: True if validation is successful, False otherwise.
    """
    try:
        with open(geojson_path, 'r') as f:
            data = json.load(f)
            # Basic GeoJSON structure validation
            if "type" in data and data["type"] == "FeatureCollection":
                print(f"GeoJSON file {geojson_path} is valid.")
                return True
            else:
                raise ValueError("Invalid GeoJSON structure.")
    except Exception as e:
        print(f"Invalid GeoJSON file {geojson_path}: {e}")
        return False

def validate_coco_annotations(annotations_path):
    """
    Validate COCO annotations (JSON format similar to GeoJSON).
    
    Args:
        annotations_path (str): Path to the COCO annotations file.
    
    Returns:
        bool: True if validation is successful, False otherwise.
    """
    try:
        coco = COCO(annotations_path)
        print(f"COCO annotations file {annotations_path} loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading COCO annotations file {annotations_path}: {e}")
        return False

if __name__ == "__main__":
    GEOJSON_PATH = "/path/to/geojson/file.geojson"  # Replace with actual path
    ANNOTATIONS_PATH = "/path/to/coco/annotations.json"  # Replace with actual path
    
    geojson_valid = validate_geojson(GEOJSON_PATH)
    coco_valid = validate_coco_annotations(ANNOTATIONS_PATH)
    
    if geojson_valid and coco_valid:
        print("GeoJSON and COCO annotation validation passed.")
    else:
        print("Validation failed.")
