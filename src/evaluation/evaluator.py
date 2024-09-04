# src/evaluation/evaluator.py

import torch
from src.evaluation.metrics import compute_map

class Evaluator:
    def __init__(self, model, device: str = 'cuda'):
        """
        Initialize the evaluator with the model and device.

        Args:
            model: HuggingFace object detection model (e.g., DETR, YOLO).
            device (str): Device to run the evaluation on.
        """
        self.model = model.to(device)
        self.device = device

    def evaluate(self, dataloader, iou_thresholds=[0.5]):
        """
        Evaluate the model on the provided dataloader and compute precision, recall, and mAP.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader with validation or test data.
            iou_thresholds (list): IoU thresholds for mAP calculation.

        Returns:
            dict: Computed precision, recall, F1-score, and mAP.
        """
        self.model.eval()
        all_predictions = []
        all_ground_truths = []

        with torch.no_grad():
            for images, targets in dataloader:
                pixel_values = torch.stack(images).to(self.device)
                outputs = self.model(pixel_values=pixel_values)

                pred_boxes = outputs.pred_boxes.cpu()
                logits = outputs.logits.cpu()

                for i in range(len(images)):
                    pred = {
                        'boxes': pred_boxes[i],
                        'scores': logits[i]
                    }
                    gt = {
                        'boxes': targets[i]['boxes'].cpu(),
                        'labels': targets[i]['labels'].cpu()
                    }
                    all_predictions.append(pred)
                    all_ground_truths.append(gt)

        metrics = compute_map(all_predictions, all_ground_truths, iou_thresholds)
        return metrics

    def save_metrics(self, metrics: dict, output_path: str, output_format='txt'):
        """
        Save the evaluation metrics to a file.

        Args:
            metrics (dict): Computed metrics to save.
            output_path (str): Path to save the metrics file.
            output_format (str): Format of the output file ('txt' or 'json').
        """
        if output_format == 'json':
            import json
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=4)
        else:
            with open(output_path, 'w') as f:
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value}\n")

        print(f"Evaluation metrics saved to {output_path}")
