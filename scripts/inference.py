# src/scripts/inference.py

import torch
from transformers import DetrFeatureExtractor
from src.models.foundation_model import DetrObjectDetectionModel
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_image(image_path):
    """
    Load an image from the file system.
    """
    return Image.open(image_path).convert("RGB")

def visualize_predictions(image, predictions, threshold=0.5):
    """
    Visualize the predictions on the image.
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

def run_inference(model, image_path, device='cuda'):
    """
    Run inference on a single image and visualize the results.
    """
    # Load and preprocess image
    image = load_image(image_path)
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)

    # Run inference
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model.forward(pixel_values=pixel_values)
    
    logits = outputs.logits.cpu()
    boxes = outputs.pred_boxes.cpu()
    
    predictions = {'boxes': boxes[0], 'scores': logits[0]}

    # Visualize results
    visualize_predictions(image, predictions)

if __name__ == "__main__":
    # Define constants
    MODEL_PATH = "output/detr_model"  # Replace with your trained model path
    IMAGE_PATH = "path/to/image.jpg"  # Replace with your test image path
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the model
    model = DetrObjectDetectionModel(num_classes=91)  # 91 for COCO dataset
    model.load(MODEL_PATH)
    
    # Run inference
    run_inference(model, IMAGE_PATH, device=DEVICE)
