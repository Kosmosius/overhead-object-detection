# src/evaluation/evaluator.py

import torch
import logging
from tqdm import tqdm
from typing import List, Dict, Any
from src.utils.metrics import evaluate_model

class Evaluator:
    """
    Class for evaluating object detection models.
    Computes metrics such as mAP, precision, recall, and F1 score using the consolidated metrics module.
    """

    def __init__(self, model: torch.nn.Module, device: str = 'cuda', confidence_threshold: float = 0.5):
        """
        Initialize the Evaluator.

        Args:
            model (torch.nn.Module): The object detection model to evaluate.
            device (str): Device to run the evaluation on ('cuda' or 'cpu').
            confidence_threshold (float): Confidence threshold to filter predictions.
        """
        self.model = model.to(device)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model.eval()
        logging.info(f"Evaluator initialized with model on device '{self.device}'.")

    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate the model using the provided dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.

        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics.
        """
        all_predictions = []
        all_ground_truths = []

        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc="Evaluating"):
                # Move images and targets to the device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Forward pass
                outputs = self.model(images)

                # Process outputs and targets
                batch_predictions = self._process_outputs(outputs)
                batch_ground_truths = self._process_targets(targets)

                # Append to lists
                all_predictions.extend(batch_predictions)
                all_ground_truths.extend(batch_ground_truths)

        # Compute metrics using the consolidated module
        metrics = evaluate_model(all_predictions, all_ground_truths)

        logging.info(f"Evaluation completed. Metrics: {metrics}")
        return metrics

    def _process_outputs(self, outputs: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        """
        Process model outputs to extract predictions.

        Args:
            outputs (List[Dict[str, Any]]): Raw outputs from the model.

        Returns:
            List[Dict[str, torch.Tensor]]: Processed predictions.
        """
        processed_predictions = []

        for output in outputs:
            # Apply confidence threshold
            scores = output['scores']
            keep = scores >= self.confidence_threshold

            # Extract boxes, labels, and scores
            boxes = output['boxes'][keep].detach().cpu()
            labels = output['labels'][keep].detach().cpu()
            scores = scores[keep].detach().cpu()

            prediction = {
                'boxes': boxes,
                'labels': labels,
                'scores': scores,
                # Include image_id if available
                'image_id': output.get('image_id', torch.tensor(-1)).detach().cpu()
            }

            processed_predictions.append(prediction)

        return processed_predictions

    def _process_targets(self, targets: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        """
        Process targets to extract ground truth annotations.

        Args:
            targets (List[Dict[str, Any]]): Targets from the dataloader.

        Returns:
            List[Dict[str, torch.Tensor]]: Processed ground truths.
        """
        processed_targets = []

        for target in targets:
            ground_truth = {
                'boxes': target['boxes'].detach().cpu(),
                'labels': target['labels'].detach().cpu(),
                # Include image_id if available
                'image_id': target.get('image_id', torch.tensor(-1)).detach().cpu()
            }

            processed_targets.append(ground_truth)

        return processed_targets
