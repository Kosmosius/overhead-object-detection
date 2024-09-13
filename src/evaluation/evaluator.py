# src/evaluation/evaluator.py

import torch
import logging
from tqdm import tqdm
from src.utils.metrics import compute_map

class Evaluator:
    def __init__(self, model, device: str = 'cuda', confidence_threshold: float = 0.5):
        """
        Initialize the evaluator with the model and device.

        Args:
            model: HuggingFace object detection model (e.g., DETR, YOLO).
            device (str): Device to run the evaluation on ('cuda' or 'cpu').
            confidence_threshold (float): Minimum confidence score for considering a detection valid.
        """
        self.model = model.to(device)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model.eval()

    def _process_batch(self, outputs, targets):
        """
        Process a batch of model outputs and targets to extract predictions and ground truths.

        Args:
            outputs (dict): Model outputs containing predicted bounding boxes and logits.
            targets (list): List of dictionaries with ground truth bounding boxes and labels.

        Returns:
            List[dict]: List of predictions and ground truth dictionaries.
        """
        pred_boxes = outputs['pred_boxes']
        pred_logits = outputs['logits']

        predictions = []
        ground_truths = []

        for i in range(len(pred_boxes)):
            pred_scores = torch.softmax(pred_logits[i], dim=-1)
            scores, labels = pred_scores.max(dim=-1)
            
            # Apply confidence threshold
            valid = scores >= self.confidence_threshold
            pred_boxes_valid = pred_boxes[i][valid].detach().cpu()
            scores_valid = scores[valid].detach().cpu()
            labels_valid = labels[valid].detach().cpu()

            # Store predictions
            predictions.append({
                'boxes': pred_boxes_valid,
                'scores': scores_valid,
                'labels': labels_valid
            })

            # Store ground truth
            gt_boxes = targets[i]['boxes'].cpu()
            gt_labels = targets[i]['labels'].cpu()
            ground_truths.append({
                'boxes': gt_boxes,
                'labels': gt_labels
            })

        return predictions, ground_truths

    def evaluate(self, dataloader, iou_thresholds=[0.5]):
        """
        Evaluate the model on the provided dataloader and compute precision, recall, and mAP.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader with validation or test data.
            iou_thresholds (list): IoU thresholds for mAP calculation.

        Returns:
            dict: Computed precision, recall, F1-score, and mAP.
        """
        all_predictions = []
        all_ground_truths = []

        # Validation for input types
        assert isinstance(dataloader, torch.utils.data.DataLoader), "dataloader must be a valid DataLoader"
        assert hasattr(self.model, "eval"), "The model should implement an evaluation method"

        logging.info("Starting model evaluation...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                images, targets = batch
                images = torch.stack(images).to(self.device)

                # Mixed precision inference (optional, faster)
                with torch.cuda.amp.autocast():
                    outputs = self.model(pixel_values=images)

                # Process outputs and targets
                predictions, ground_truths = self._process_batch(outputs, targets)
                all_predictions.extend(predictions)
                all_ground_truths.extend(ground_truths)

        # Compute metrics
        metrics = compute_map(all_predictions, all_ground_truths, iou_thresholds)
        logging.info(f"Evaluation completed. Metrics: {metrics}")
        return metrics

    def save_metrics(self, metrics: dict, output_path: str, output_format='txt'):
        """
        Save the evaluation metrics to a file.

        Args:
            metrics (dict): Computed metrics to save.
            output_path (str): Path to save the metrics file.
            output_format (str): Format of the output file ('txt' or 'json').
        """
        assert output_format in ['txt', 'json'], "Output format must be 'txt' or 'json'"

        if output_format == 'json':
            import json
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=4)
        else:
            with open(output_path, 'w') as f:
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value}\n")

        logging.info(f"Evaluation metrics saved to {output_path}")
