# scripts/clean_data.py

import os
import json
import shutil
import logging
from pycocotools.coco import COCO
from geojson import load as load_geojson
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)

def ensure_dir_exists(directory: str):
    """Ensure that a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def clean_coco(annotations_path: str, output_path: str):
    """Clean a COCO dataset by ensuring that images referenced in annotations exist."""
    coco = COCO(annotations_path)
    ensure_dir_exists(output_path)

    image_ids = coco.getImgIds()
    for img_id in tqdm(image_ids, desc="Cleaning COCO dataset"):
        img_info = coco.loadImgs([img_id])[0]
        img_path = img_info['file_name']
        if not os.path.exists(img_path):
            logging.warning(f"Missing image: {img_path}")
        else:
            shutil.copy(img_path, output_path)

    logging.info("COCO dataset cleaning completed.")


def clean_geojson(geojson_path: str, output_path: str):
    """Clean a GeoJSON dataset by ensuring valid features and properties."""
    with open(geojson_path, 'r') as file:
        geo_data = load_geojson(file)
        
    ensure_dir_exists(output_path)

    features = geo_data.get("features", [])
    for feature in tqdm(features, desc="Cleaning GeoJSON dataset"):
        if "geometry" not in feature or "properties" not in feature:
            logging.warning(f"Invalid feature found: {feature}")
        else:
            # Write cleaned features back to the output GeoJSON
            with open(os.path.join(output_path, "cleaned_data.geojson"), 'w') as output_file:
                json.dump(geo_data, output_file)

    logging.info("GeoJSON dataset cleaning completed.")

def clean_overhead_dataset(dataset_name: str, annotations_path: str, image_dir: str, output_path: str):
    """
    Clean overhead-specific object detection datasets like DOTA, xView, etc.
    This involves checking the existence of images and ensuring annotations are valid.
    """
    ensure_dir_exists(output_path)

    if dataset_name.lower() in ['dota', 'dota2.0']:
        logging.info(f"Cleaning {dataset_name} dataset.")
        clean_dota_dataset(annotations_path, image_dir, output_path)
    elif dataset_name.lower() == 'xview':
        logging.info("Cleaning xView dataset.")
        clean_xview_dataset(annotations_path, image_dir, output_path)
    elif dataset_name.lower() in ['isaid', 'aid', 'dior', 'hrsc2016', 'landcover.ai', 'soda-a', 'fmars']:
        logging.info(f"Cleaning {dataset_name} dataset.")
        clean_standard_overhead_dataset(annotations_path, image_dir, output_path)
    else:
        logging.error(f"Dataset {dataset_name} is not supported.")


def clean_dota_dataset(annotations_path: str, image_dir: str, output_path: str):
    """
    Clean DOTA or DOTA 2.0 dataset.
    Validate the images and ensure annotation structure is correct.
    """
    with open(annotations_path, 'r') as file:
        annotations = json.load(file)
    
    for ann in tqdm(annotations, desc="Cleaning DOTA dataset"):
        img_path = os.path.join(image_dir, ann['image_name'])
        if not os.path.exists(img_path):
            logging.warning(f"Missing image: {ann['image_name']}")
        else:
            shutil.copy(img_path, output_path)

    logging.info(f"DOTA dataset cleaning completed.")


def clean_xview_dataset(annotations_path: str, image_dir: str, output_path: str):
    """
    Clean xView dataset.
    Ensure images and annotations are valid.
    """
    with open(annotations_path, 'r') as file:
        annotations = json.load(file)

    for img_info in tqdm(annotations['images'], desc="Cleaning xView dataset"):
        img_path = os.path.join(image_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            logging.warning(f"Missing image: {img_info['file_name']}")
        else:
            shutil.copy(img_path, output_path)

    logging.info("xView dataset cleaning completed.")


def clean_standard_overhead_dataset(annotations_path: str, image_dir: str, output_path: str):
    """
    Clean standard overhead datasets like iSAID, AID, DIOR, etc.
    Validate images and ensure annotations are consistent.
    """
    with open(annotations_path, 'r') as file:
        annotations = json.load(file)

    for img_info in tqdm(annotations['images'], desc=f"Cleaning {annotations_path.split('/')[-1]} dataset"):
        img_path = os.path.join(image_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            logging.warning(f"Missing image: {img_info['file_name']}")
        else:
            shutil.copy(img_path, output_path)

    logging.info(f"{annotations_path.split('/')[-1]} dataset cleaning completed.")


def main():
    # Example usage
    dataset_type = input("Enter the dataset type (COCO, GeoJSON, DOTA, xView, etc.): ").strip().lower()
    annotations_path = input("Enter the path to annotations: ").strip()
    image_dir = input("Enter the path to image directory: ").strip()
    output_dir = input("Enter the path to output directory: ").strip()

    if dataset_type == "coco":
        clean_coco(annotations_path, output_dir)
    elif dataset_type == "geojson":
        clean_geojson(annotations_path, output_dir)
    else:
        clean_overhead_dataset(dataset_type, annotations_path, image_dir, output_dir)


if __name__ == "__main__":
    main()
