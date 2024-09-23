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
        Initialize the Evaluator with a model and device.

        Args:
            model (torch.nn.Module): The model to evaluate.
            device (str): Device to run evaluation on ('cuda' or 'cpu').
            confidence_threshold (float): Confidence threshold for filtering predictions.
        """
        self.model = model.to(device)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model.eval()
        logger.info(f"Evaluator initialized with model on device '{device}'.")

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
                try:
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                except AttributeError as e:
                    logger.error(f"Error moving targets to device: {e}")
                    raise

                # Get model outputs
                outputs = self.model(images)

                # Process outputs and targets
                batch_predictions = self._process_outputs(outputs)
                batch_ground_truths = self._process_targets(targets)

                # Accumulate predictions and ground truths
                all_predictions.extend(batch_predictions)
                all_ground_truths.extend(batch_ground_truths)

        # Compute evaluation metrics
        metrics = evaluate_model(all_predictions, all_ground_truths)
        logger.info(f"Evaluation completed. Metrics: {metrics}")
        return metrics

    def _process_outputs(self, outputs: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """
        Process model outputs to filter based on confidence threshold.

        Args:
            outputs (List[Dict[str, torch.Tensor]]): Model outputs.

        Returns:
            List[Dict[str, torch.Tensor]]: Filtered predictions.
        """
        processed_outputs = []
        for output in outputs:
            required_keys = ['scores', 'boxes', 'labels', 'image_id']
            if not all(key in output for key in required_keys):
                logger.error("Model output missing required keys.")
                raise KeyError("Model output missing required keys.")

            scores = output['scores']
            keep = scores >= self.confidence_threshold
            filtered_output = {
                'boxes': output['boxes'][keep],
                'scores': scores[keep],
                'labels': output['labels'][keep],
                'image_id': output['image_id']
            }
            processed_outputs.append(filtered_output)
        return processed_outputs

    def _process_targets(self, targets: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """
        Process ground truth targets.

        Args:
            targets (List[Dict[str, torch.Tensor]]): Ground truth targets.

        Returns:
            List[Dict[str, torch.Tensor]]: Processed ground truths.
        """
        processed_targets = []
        for target in targets:
            required_keys = ['boxes', 'labels', 'image_id']
            if not all(key in target for key in required_keys):
                logger.error("Target missing required keys.")
                raise KeyError("Target missing required keys.")

            processed_target = {
                'boxes': target['boxes'],
                'labels': target['labels'],
                'image_id': target['image_id']
            }
            processed_targets.append(processed_target)
        return processed_targets
