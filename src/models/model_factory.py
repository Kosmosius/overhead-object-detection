# src/models/model_factory.py

import logging
from typing import Optional, Any, Dict

from transformers import (
    AutoConfig,
    AutoModelForObjectDetection,
    AutoModelForImageClassification,
    AutoModelForSemanticSegmentation,
    PreTrainedModel,
    AutoFeatureExtractor,
    AutoImageProcessor,
)

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory class to create models for different tasks using HuggingFace's AutoModel classes.
    """

    @staticmethod
    def create_model(
        model_name_or_path: str,
        task_type: str,
        num_labels: Optional[int] = None,
        label2id: Optional[Dict[str, int]] = None,
        id2label: Optional[Dict[int, str]] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> PreTrainedModel:
        """
        Creates a model using HuggingFace's AutoModel classes.

        Args:
            model_name_or_path (str): Path to the pretrained model or model identifier from HuggingFace.
            task_type (str): Type of task ('object_detection', 'image_classification', 'semantic_segmentation').
            num_labels (int, optional): Number of labels/classes for the task.
            label2id (Dict[str, int], optional): Mapping from label names to IDs.
            id2label (Dict[int, str], optional): Mapping from IDs to label names.
            config_kwargs (Dict[str, Any], optional): Additional keyword arguments for configuration.
            model_kwargs (Dict[str, Any], optional): Additional keyword arguments for model instantiation.

        Returns:
            PreTrainedModel: A HuggingFace model instance.

        Raises:
            ValueError: If the task type is not supported.
        """
        logger.info(f"Loading model '{model_name_or_path}' for task '{task_type}'.")

        config_kwargs = config_kwargs or {}
        model_kwargs = model_kwargs or {}

        try:
            # Load configuration with specified parameters
            config = AutoConfig.from_pretrained(
                model_name_or_path,
                num_labels=num_labels,
                label2id=label2id,
                id2label=id2label,
                **config_kwargs,
            )
            logger.debug(f"Configuration loaded: {config}")

            # Select the appropriate AutoModel class based on task type
            if task_type == 'object_detection':
                model = AutoModelForObjectDetection.from_pretrained(
                    model_name_or_path,
                    config=config,
                    **model_kwargs,
                )
                logger.info(f"Loaded AutoModelForObjectDetection for '{model_name_or_path}'.")
            elif task_type == 'image_classification':
                model = AutoModelForImageClassification.from_pretrained(
                    model_name_or_path,
                    config=config,
                    **model_kwargs,
                )
                logger.info(f"Loaded AutoModelForImageClassification for '{model_name_or_path}'.")
            elif task_type == 'semantic_segmentation':
                model = AutoModelForSemanticSegmentation.from_pretrained(
                    model_name_or_path,
                    config=config,
                    **model_kwargs,
                )
                logger.info(f"Loaded AutoModelForSemanticSegmentation for '{model_name_or_path}'.")
            else:
                logger.error(f"Unsupported task type '{task_type}'.")
                raise ValueError(f"Unsupported task type '{task_type}'.")

            return model

        except Exception as e:
            logger.error(f"Failed to create model '{model_name_or_path}' for task '{task_type}': {e}")
            raise

    @staticmethod
    def create_processor(
        model_name_or_path: str,
    ) -> Any:
        """
        Creates an image processor or feature extractor using HuggingFace's Auto classes.

        Args:
            model_name_or_path (str): Path to the pretrained model or model identifier from HuggingFace.

        Returns:
            AutoImageProcessor or AutoFeatureExtractor: An image processor instance.
        """
        try:
            # Try to load AutoImageProcessor first
            processor = AutoImageProcessor.from_pretrained(model_name_or_path)
            logger.info(f"Loaded AutoImageProcessor for '{model_name_or_path}'.")
        except Exception as e_image_processor:
            logger.debug(f"AutoImageProcessor not found: {e_image_processor}")
            try:
                # Fallback to AutoFeatureExtractor
                processor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
                logger.info(f"Loaded AutoFeatureExtractor for '{model_name_or_path}'.")
            except Exception as e_feature_extractor:
                logger.error(f"Failed to create processor: {e_feature_extractor}")
                raise RuntimeError(
                    f"Could not load processor for '{model_name_or_path}'. "
                    f"Errors: {e_image_processor}, {e_feature_extractor}"
                )
        return processor
