# scripts/export_model.py

import torch
from src.models.foundation_model import DetrObjectDetectionModel

def export_model(model_path, output_path):
    """
    Export the trained model to TorchScript format.
    
    Args:
        model_path (str): Path to the trained model.
        output_path (str): Path to save the exported model.
    """
    # Load the trained model
    model = DetrObjectDetectionModel(num_classes=91)  # Assuming COCO dataset with 91 classes
    model.load(model_path)
    model.model.eval()

    # Export to TorchScript
    scripted_model = torch.jit.script(model.model)
    scripted_model.save(output_path)
    print(f"Model successfully exported to {output_path}")

if __name__ == "__main__":
    MODEL_PATH = "output/best_detr_model"  # Replace with your trained model path
    OUTPUT_PATH = "output/detr_model_scripted.pt"  # Replace with desired output path
    
    export_model(MODEL_PATH, OUTPUT_PATH)
