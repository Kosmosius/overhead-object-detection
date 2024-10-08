# src/evaluation/evaluator.py

import logging
from typing import Dict, Any

import torch
from datasets import load_metric
from transformers import EvalPrediction

logger = logging.getLogger(__name__)


def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
    """
    Computes evaluation metrics using HuggingFace's evaluate library.

    Args:
        p (EvalPrediction): An EvalPrediction object with predictions and label_ids.

    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics.
    """
    # Load the COCO evaluation metric
    metric = load_metric("coco_evaluation")

    # Convert predictions and references to the expected format
    predictions = []
    references = []

    for idx, (pred, label) in enumerate(zip(p.predictions, p.label_ids)):
        # Process predictions
        processed_pred = {
            "image_id": label["image_id"],
            "category_id": pred["labels"],
            "bbox": pred["boxes"],
            "score": pred["scores"],
        }
        predictions.append(processed_pred)

        # Process references (ground truth)
        processed_label = {
            "image_id": label["image_id"],
            "category_id": label["labels"],
            "bbox": label["boxes"],
        }
        references.append(processed_label)

    # Compute metrics
    results = metric.compute(predictions=predictions, references=references)
    logger.info(f"Evaluation metrics: {results}")

    return results
