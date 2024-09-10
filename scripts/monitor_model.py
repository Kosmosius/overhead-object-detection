# scripts/monitor_model.py

import argparse
import time
import torch
from src.models.foundation_model import HuggingFaceObjectDetectionModel
from src.utils.config_parser import ConfigParser
from src.utils.logging import setup_logging
from src.evaluation.evaluator import Evaluator
from src.data.dataloader import get_dataloader
from src.utils.system_utils import check_device
from prometheus_client import Gauge, start_http_server

# Prometheus metrics
LOSS_METRIC = Gauge('training_loss', 'Current training loss')
PRECISION_METRIC = Gauge('validation_precision', 'Validation precision')
RECALL_METRIC = Gauge('validation_recall', 'Validation recall')
MAP_METRIC = Gauge('validation_map', 'Mean Average Precision')

def monitor_model_performance(model, dataloader, device, metrics_interval=30):
    """
    Monitor and report model performance metrics using Prometheus.

    Args:
        model: The model to monitor.
        dataloader: The DataLoader for validation data.
        device: The device to run evaluation on.
        metrics_interval (int): Time interval in seconds for reporting metrics.
    """
    while True:
        evaluator = Evaluator(model, dataloader, device)
        metrics = evaluator.evaluate()
        
        # Update Prometheus metrics
        LOSS_METRIC.set(metrics.get("total_loss", 0))
        PRECISION_METRIC.set(metrics.get("precision", 0))
        RECALL_METRIC.set(metrics.get("recall", 0))
        MAP_METRIC.set(metrics.get("map", 0))

        # Log metrics
        logging.info(f"Updated metrics: Precision={metrics.get('precision', 0)}, Recall={metrics.get('recall', 0)}, mAP={metrics.get('map', 0)}")
        
        # Wait for the next interval
        time.sleep(metrics_interval)


def main():
    parser = argparse.ArgumentParser(description="Monitor model performance metrics.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on ('cuda' or 'cpu').")
    parser.add_argument("--port", type=int, default=8000, help="Port to expose Prometheus metrics.")
    parser.add_argument("--metrics_interval", type=int, default=30, help="Time interval in seconds to update metrics.")

    args = parser.parse_args()

    # Set up logging
    setup_logging(log_file="monitoring.log")

    # Load configuration
    config_parser = ConfigParser(args.config_path)
    model_name = config_parser.get("model_name")
    num_classes = config_parser.get("num_classes")
    batch_size = config_parser.get("batch_size")

    # Load model
    model = HuggingFaceObjectDetectionModel(model_name=model_name, num_classes=num_classes)
    model.load(args.model_checkpoint)

    # Prepare dataloader
    feature_extractor = DetrFeatureExtractor.from_pretrained(model_name)
    val_loader = get_dataloader(data_dir=args.data_dir, batch_size=batch_size, mode="val", feature_extractor=feature_extractor)

    # Check device
    device = check_device() if args.device == "auto" else torch.device(args.device)

    # Start Prometheus HTTP server
    start_http_server(args.port)
    logging.info(f"Prometheus server started at port {args.port}")

    # Monitor model performance
    monitor_model_performance(model, val_loader, device, metrics_interval=args.metrics_interval)


if __name__ == "__main__":
    main()

"""
python scripts/monitor_model.py --model_checkpoint output/detr_model --config_path configs/training/default_training.yml --data_dir /path/to/data --port 8000 --metrics_interval 30 --device cuda
"""
