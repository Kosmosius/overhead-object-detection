# scripts/inference.py

import argparse
import torch
from PIL import Image
from transformers import DetrFeatureExtractor
from src.models.foundation_model import HuggingFaceObjectDetectionModel
from src.utils.config_parser import ConfigParser
from src.utils.logging import setup_logging
from src.utils.system_utils import check_device
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_image(image_path):
    """
    Load an image from the file system.

    Args:
        image_path (str): Path to the image file.

    Returns:
        PIL.Image: Loaded image in RGB format.
    """
    return Image.open(image_path).convert("RGB")


def visualize_predictions(image, predictions, threshold=0.5):
    """
    Visualize the predictions on the image.

    Args:
        image (PIL.Image): The input image.
        predictions (dict): Predictions from the model containing bounding boxes and scores.
        threshold (float): Score threshold for displaying boxes.
    """
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box, score in zip(predictions['boxes'], predictions['scores']):
        if score >= threshold:
            xmin, ymin, xmax, ymax = box
            width, height = xmax - xmin, ymax - ymin
            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    plt.show()


def run_inference(model, image_path, feature_extractor, device='cuda'):
    """
    Run inference on a single image and visualize the results.

    Args:
        model: The loaded object detection model.
        image_path (str): Path to the input image.
        feature_extractor: HuggingFace feature extractor for preprocessing.
        device (str): The device to run inference on ('cuda' or 'cpu').
    """
    # Load and preprocess the image
    image = load_image(image_path)
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)

    # Run inference
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
    
    logits = outputs.logits.cpu()
    boxes = outputs.pred_boxes.cpu()

    predictions = {'boxes': boxes[0], 'scores': logits[0]}

    # Visualize the predictions
    visualize_predictions(image, predictions)


def main():
    parser = argparse.ArgumentParser(description="Run inference on an image using a trained object detection model.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file for inference.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on ('cuda' or 'cpu').")

    args = parser.parse_args()

    # Set up logging
    setup_logging(log_file="inference.log")

    # Load configuration
    config_parser = ConfigParser(args.config_path)
    model_name = config_parser.get("model_name")
    num_classes = config_parser.get("num_classes")

    # Load the model
    model = HuggingFaceObjectDetectionModel(model_name=model_name, num_classes=num_classes)
    model.load(args.model_checkpoint)

    # Load the feature extractor
    feature_extractor = DetrFeatureExtractor.from_pretrained(model_name)

    # Check device
    device = check_device() if args.device == "auto" else torch.device(args.device)

    # Run inference
    run_inference(model, args.image_path, feature_extractor, device)


if __name__ == "__main__":
    main()

"""
python scripts/inference.py --model_checkpoint output/detr_model --config_path configs/training/default_training.yml --image_path data/samples/sample_image.jpg --device cuda
"""
