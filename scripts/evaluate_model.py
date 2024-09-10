# scripts/evaluate_model.py

import argparse
import logging
import torch
import os
from src.models.model_builder import load_model_from_checkpoint
from src.data.dataloader import get_dataloader
from src.evaluation.evaluator import Evaluator
from src.utils.config_parser import ConfigParser
from src.utils.system_utils import check_device, check_disk_space, check_memory_requirements
from src.utils.logging import setup_logging

# Set up logging
log_file = "logs/evaluation.log"
log_dir = os.path.dirname(log_file)

# Ensure the directory for the log file exists, if a directory is specified
if log_dir:
    os.makedirs(log_dir, exist_ok=True)

# Setup logging
setup_logging(log_file=log_file)

def evaluate_model(model_checkpoint: str, data_dir: str, config_path: str, batch_size: int, device: str, iou_threshold: float):
    """
    Load a model from checkpoint and evaluate it on the specified dataset.

    Args:
        model_checkpoint (str): Path to the model checkpoint.
        data_dir (str): Path to the dataset directory.
        config_path (str): Path to the configuration file.
        batch_size (int): Batch size for evaluation.
        device (str): Device to run evaluation on ('cuda' or 'cpu').
        iou_threshold (float): IoU threshold for evaluating predictions.
    """
    # Load the config
    config_parser = ConfigParser(config_path)
    model_name = config_parser.get("model_name")
    num_classes = config_parser.get("num_classes")

    # Load the model
    model = load_model_from_checkpoint(model_checkpoint, model_name, num_labels=num_classes, device=device)
    
    # Prepare the dataloader
    feature_extractor = config_parser.get("feature_extractor")
    dataloader = get_dataloader(data_dir=data_dir, batch_size=batch_size, mode='val', feature_extractor=feature_extractor)
    
    # Create an Evaluator and run evaluation
    evaluator = Evaluator(model, device=device)
    metrics = evaluator.evaluate(dataloader, iou_thresholds=[iou_threshold])

    # Log metrics
    logging.info(f"Evaluation Metrics: {metrics}")

    # Save metrics to a file
    evaluator.save_metrics(metrics, output_path="output/evaluation_metrics.txt")
    logging.info("Evaluation complete, metrics saved.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained object detection model.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the model checkpoint to evaluate.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation.")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for evaluating predictions.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on ('cuda' or 'cpu').")
    args = parser.parse_args()

    # System checks
    check_memory_requirements(min_memory_gb=8)
    check_disk_space(min_disk_gb=10)
    
    # Set device
    device = check_device() if args.device == "auto" else torch.device(args.device)
    
    # Run the evaluation
    evaluate_model(
        model_checkpoint=args.model_checkpoint,
        data_dir=args.data_dir,
        config_path=args.config_path,
        batch_size=args.batch_size,
        device=device,
        iou_threshold=args.iou_threshold
    )


if __name__ == "__main__":
    main()

"""
python scripts/evaluate_model.py --model_checkpoint output/detr_model --data_dir /path/to/validation/dataset --config_path configs/training/default_training.yml --batch_size 4 --iou_threshold 0.5 --device cuda
"""
