# src/evaluation/evaluator.py

import torch
from transformers import DetrForObjectDetection
from src.evaluation.metrics import compute_map

class Evaluator:
    def __init__(self, model: DetrForObjectDetection, device: str = 'cuda'):
        """
        Initialize the evaluator with the model and device.

        Args:
            model (DetrForObjectDetection): Pretrained DETR model for evaluation.
            device (str): Device to run the evaluation on, either 'cuda' or 'cpu'.
        """
        self.model = model.to(device)
        self.device = device

    def evaluate(self, dataloader):
        """
        Run evaluation on the provided dataloader and compute the precision and recall.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader with validation or test data.

        Returns:
            tuple: Precision and recall metrics.
        """
        self.model.eval()
        all_predictions = []
        all_ground_truths = []

        with torch.no_grad():
            for images, targets in dataloader:
                pixel_values = torch.stack(images).to(self.device)
                outputs = self.model(pixel_values=pixel_values)

                # Extract predictions and convert to CPU
                logits = outputs.logits.cpu()
                boxes = outputs.pred_boxes.cpu()

                for i in range(len(images)):
                    pred = {
                        'boxes': boxes[i],
                        'scores': logits[i]
                    }
                    gt = {
                        'boxes': targets[i]['boxes'].cpu(),
                        'labels': targets[i]['labels'].cpu()
                    }
                    all_predictions.append(pred)
                    all_ground_truths.append(gt)

        # Compute precision and recall using custom mAP calculation
        precision, recall = compute_map(all_predictions, all_ground_truths)
        return precision, recall

    def save_metrics(self, precision: float, recall: float, output_path: str):
        """
        Save the evaluation metrics to a file.

        Args:
            precision (float): Computed precision.
            recall (float): Computed recall.
            output_path (str): Path to save the metrics file.
        """
        with open(output_path, 'w') as f:
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
        print(f"Evaluation metrics saved to {output_path}")
