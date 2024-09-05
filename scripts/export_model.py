# scripts/export_model.py

import argparse
import torch
from transformers import DetrForObjectDetection
from src.models.model_registry import ModelRegistry
from src.models.foundation_model import HuggingFaceObjectDetectionModel
from src.utils.config_parser import ConfigParser
from src.utils.logging import setup_logging
from src.utils.system_utils import check_device
import os


def export_model(model_checkpoint: str, output_path: str, config_path: str, device: str = "cuda"):
    """
    Export a trained model to TorchScript format or ONNX format for deployment.

    Args:
        model_checkpoint (str): Path to the trained model checkpoint.
        output_path (str): Path to save the exported model.
        config_path (str): Path to the configuration file.
        device (str): Device to use for exporting ('cuda' or 'cpu').
    """
    # Set up logging
    setup_logging(log_file="export_model.log")

    # Load the configuration
    config_parser = ConfigParser(config_path)
    model_name = config_parser.get("model_name")
    num_classes = config_parser.get("num_classes")

    # Load the model
    model = HuggingFaceObjectDetectionModel(model_name=model_name, num_classes=num_classes)
    model.load(model_checkpoint)

    # Move model to the specified device
    model.to(device)
    model.model.eval()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Export to TorchScript format
    try:
        scripted_model = torch.jit.script(model.model)
        scripted_model.save(output_path)
        print(f"Model successfully exported to {output_path}")
    except Exception as e:
        print(f"Error exporting model: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Export a trained object detection model to TorchScript.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the exported model.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for exporting ('cuda' or 'cpu').")

    args = parser.parse_args()

    # Check device
    device = check_device() if args.device == "auto" else torch.device(args.device)

    # Export the model
    export_model(
        model_checkpoint=args.model_checkpoint,
        output_path=args.output_path,
        config_path=args.config_path,
        device=device
    )


if __name__ == "__main__":
    main()

"""
python scripts/export_model.py --model_checkpoint output/detr_model --output_path output/detr_model_scripted.pt --config_path configs/training/default_training.yml --device cuda
"""
