# src/data/validation/quality_checks.py

import os
from pycocotools.coco import COCO

def validate_images(image_dir, annotations_path):
    """
    Validate that all images referenced in the COCO annotations exist in the directory.
    
    Args:
        image_dir (str): Path to the directory containing images.
        annotations_path (str): Path to the COCO annotations file.
    
    Returns:
        bool: True if all images exist, False otherwise.
    """
    coco = COCO(annotations_path)
    image_ids = coco.getImgIds()
    missing_images = []

    for img_id in image_ids:
        img_info = coco.loadImgs([img_id])[0]
        image_path = os.path.join(image_dir, img_info['file_name'])
        if not os.path.exists(image_path):
            missing_images.append(img_info['file_name'])
    
    if missing_images:
        print(f"Missing images: {missing_images}")
        return False
    print(f"All images in {image_dir} are valid.")
    return True

def check_annotation_completeness(annotations_path):
    """
    Check that all annotations in the COCO file are complete and well-formed.
    
    Args:
        annotations_path (str): Path to the COCO annotations file.
    
    Returns:
        bool: True if all annotations are complete, False otherwise.
    """
    coco = COCO(annotations_path)
    incomplete_annotations = []

    for ann_id in coco.getAnnIds():
        ann = coco.loadAnns([ann_id])[0]
        if not ann.get('bbox') or not ann.get('category_id'):
            incomplete_annotations.append(ann_id)
    
    if incomplete_annotations:
        print(f"Incomplete annotations: {incomplete_annotations}")
        return False
    print("All annotations are complete.")
    return True

if __name__ == "__main__":
    IMAGE_DIR = "/path/to/images"  # Replace with actual path
    ANNOTATIONS_PATH = "/path/to/coco/annotations.json"  # Replace with actual path
    
    images_valid = validate_images(IMAGE_DIR, ANNOTATIONS_PATH)
    annotations_complete = check_annotation_completeness(ANNOTATIONS_PATH)
    
    if images_valid and annotations_complete:
        print("Image and annotation quality checks passed.")
    else:
        print("Quality checks failed.")
