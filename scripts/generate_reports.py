# scripts/generate_reports.py

import argparse
import json
import os
from src.evaluation.evaluator import Evaluator
from src.data.dataloader import get_dataloader
from src.models.foundation_model import HuggingFaceObjectDetectionModel
from src.utils.config_parser import ConfigParser
from src.utils.logging import setup_logging
from src.utils.system_utils import check_device
from src.utils.metrics import evaluate_model

def generate_evaluation_report(model_checkpoint, config_path, data_dir, output_dir, device="cuda"):
    """
    Generate a performance report by evaluating the trained model on the validation set.

    Args:
        model_checkpoint (str): Path to the trained model checkpoint.
        config_path (str): Path to the configuration file.
        data_dir (str): Path to the dataset directory.
        output_dir (str): Directory to save the evaluation report.
        device (str): Device to run the evaluation ('cuda' or 'cpu').
    """
    # Set up logging
    setup_logging(log_file="generate_reports.log")

    # Load configuration
    config_parser = ConfigParser(config_path)
    model_name = config_parser.get("model_name")
    num_classes = config_parser.get("num_classes")
    batch_size = config_parser.get("batch_size", 8)

    # Load the model
    model = HuggingFaceObjectDetectionModel(model_name=model_name, num_classes=num_classes)
    model.load(model_checkpoint)

    # Load the validation dataloader
    dataloader = get_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        mode="val",
        feature_extractor=None,  # Assume feature extractor is built into the model
    )

    # Initialize evaluator and run evaluation
    evaluator = Evaluator(model, device)
    metrics = evaluator.evaluate(dataloader)

    # Save the report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Evaluation report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate a performance report for a trained object detection model.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the evaluation report.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation ('cuda' or 'cpu').")

    args = parser.parse_args()

    # Check device
    device = check_device() if args.device == "auto" else torch.device(args.device)

    # Generate evaluation report
    generate_evaluation_report(
        model_checkpoint=args.model_checkpoint,
        config_path=args.config_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=device
    )


if __name__ == "__main__":
    main()

"""
python scripts/generate_reports.py --model_checkpoint output/detr_model --config_path configs/training/default_training.yml --data_dir data/coco --output_dir output/reports --device cuda
"""
