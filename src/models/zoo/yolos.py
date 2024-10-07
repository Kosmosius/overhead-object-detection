# src/models/zoo/yolos.py

import torch
from torch import nn
from transformers import (
    YolosForObjectDetection,
    YolosConfig,
    YolosImageProcessor,
)
from typing import List, Dict, Tuple, Optional, Any, Type
from PIL import Image
import os

from src.models.zoo.base_model import BaseModel
from src.utils.logging_utils import get_logger


logger = get_logger(__name__)


class YOLOS(BaseModel):
    """
    YOLOS (You Only Look at One Sequence) Object Detection Model Wrapper.

    This class extends the BaseModel to provide functionalities specific to the YOLOS model from Hugging Face.
    It includes methods for initialization, forward pass, loss computation, metric computation, prediction,
    training, evaluation, saving, and loading.

    Reference:
    - You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection
      by Yuxin Fang, Bencheng Liao, Xinggang Wang, et al.
    - Hugging Face Transformers: https://github.com/huggingface/transformers
    """

    def __init__(
        self,
        model_name_or_path: str = "hustvl/yolos-base",
        model_class: Optional[Type[YolosForObjectDetection]] = None,
        config: Optional[Union[Dict[str, Any], str]] = None,
        num_labels: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        """
        Initializes the YOLOS model by extending the BaseModel.

        Args:
            model_name_or_path (str): Path to the pretrained YOLOS model or model identifier from Hugging Face.
            model_class (Optional[Type[YolosForObjectDetection]]): Specific Hugging Face YOLOS model class to load.
                If None, defaults to YolosForObjectDetection.
            config (Union[Dict[str, Any], str], optional): Configuration dictionary or path to config file.
            num_labels (int, optional): Number of labels for object detection.
            device (Optional[Union[str, torch.device]]): Device to run the model on ('cpu' or 'cuda').
                If None, defaults to CUDA if available.
            **kwargs: Additional keyword arguments for model configuration.
        """
        model_class = model_class or YolosForObjectDetection
        super().__init__(
            model_name_or_path=model_name_or_path,
            model_class=model_class,
            config=config,
            num_labels=num_labels,
            **kwargs,
        )
        self.logger = get_logger(f"{self.__class__.__name__}")
        self.logger.info("YOLOS model initialized successfully.")

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> Any:
        """
        Defines the forward pass of the YOLOS model.

        Args:
            pixel_values (torch.Tensor): Input image tensor of shape (batch_size, num_channels, height, width).
            **kwargs: Additional keyword arguments for the model.

        Returns:
            transformers.modeling_outputs.YolosObjectDetectionOutput: Model outputs containing logits and bounding boxes.
        """
        return self.model(pixel_values=pixel_values, **kwargs)

    def compute_loss(self, outputs: Any, targets: Any) -> torch.Tensor:
        """
        Computes the loss given model outputs and targets.

        Args:
            outputs (YolosObjectDetectionOutput): Outputs from the YOLOS model.
            targets (List[Dict[str, Any]]): Ground truth targets for object detection.

        Returns:
            torch.Tensor: The computed loss.
        """
        if targets is None:
            raise ValueError("Targets must be provided for loss computation.")

        loss = outputs.loss
        return loss

    def compute_metrics(
        self,
        outputs: List[Any],
        targets: List[Any],
        image_ids: List[Any]
    ) -> Dict[str, float]:
        """
        Computes evaluation metrics given the model outputs and targets.

        Args:
            outputs (List[YolosObjectDetectionOutput]): List of model outputs.
            targets (List[List[Dict[str, Any]]]): List of ground truth targets.
            image_ids (List[Any]): List of image IDs corresponding to the batches.

        Returns:
            Dict[str, float]: Dictionary of computed metrics (e.g., mAP).
        """
        # Placeholder for actual metric computation, e.g., mAP using COCO API
        # Implement custom metric computation or integrate with existing libraries
        # For demonstration, we'll return a dummy metric
        self.logger.info("Computing evaluation metrics.")
        metrics = {"mAP": 0.0}  # Replace with actual metric computation
        return metrics

    def predict(
        self,
        images: List[Image.Image],
        threshold: float = 0.5,
        top_k: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Performs object detection on a list of images.

        Args:
            images (List[PIL.Image.Image]): List of PIL Image objects.
            threshold (float, optional): Score threshold to filter detections.
                Defaults to 0.5.
            top_k (int, optional): Maximum number of predictions to retain per image.
                Defaults to 100.

        Returns:
            List[Dict[str, Any]]: List of detection results per image.
                Each dictionary contains 'scores', 'labels', and 'boxes'.
        """
        self.logger.info("Starting prediction.")
        self.model.eval()
        with torch.no_grad():
            # Preprocess images
            inputs = self.feature_extractor(images=images, return_tensors="pt")
            inputs = self.prepare_inputs(inputs)

            # Forward pass
            outputs = self.forward(**inputs)

            # Post-process outputs
            target_sizes = [image.size[::-1] for image in images]  # (height, width)
            results = self.post_process_predictions(
                outputs=outputs,
                threshold=threshold,
                top_k=top_k,
                target_sizes=target_sizes,
            )

        self.logger.info("Prediction completed.")
        return results

    def save_model(self, save_directory: str):
        """
        Saves the YOLOS model and image processor to the specified directory.

        Args:
            save_directory (str): Path to the directory where the model will be saved.
        """
        super().save(save_directory)
        self.logger.info(f"YOLOS model and image processor saved to {save_directory}.")

    @classmethod
    def load_from_directory(
        cls,
        load_directory: str,
        model_class: Optional[Type[YolosForObjectDetection]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        """
        Loads the YOLOS model and image processor from a specified directory.

        Args:
            load_directory (str): Path to the directory where the model is saved.
            model_class (Optional[Type[YolosForObjectDetection]]): Specific model class to instantiate.
                If None, defaults to YolosForObjectDetection.
            device (Optional[Union[str, torch.device]]): Device to run the model on ('cpu' or 'cuda').
                If None, defaults to CUDA if available.
            **kwargs: Additional keyword arguments for model configuration.

        Returns:
            YOLOS: An instance of the YOLOS class loaded from the directory.
        """
        instance = cls(
            model_name_or_path=load_directory,
            model_class=model_class,
            device=device,
            **kwargs,
        )
        instance.logger.info(f"YOLOS model loaded from {load_directory}.")
        return instance

    def train_model(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        num_warmup_steps: int = 0,
        log_interval: int = 10,
        use_amp: bool = False,
        **kwargs,
    ):
        """
        Trains the YOLOS model using the provided dataloaders.

        Args:
            train_dataloader (torch.utils.data.DataLoader): The training data loader.
            val_dataloader (torch.utils.data.DataLoader, optional): The validation data loader.
            epochs (int, optional): Number of training epochs. Defaults to 1.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
            weight_decay (float, optional): Weight decay coefficient. Defaults to 0.01.
            num_warmup_steps (int, optional): Number of warmup steps for the scheduler. Defaults to 0.
            log_interval (int, optional): Interval for logging progress. Defaults to 10.
            use_amp (bool, optional): Whether to use mixed precision training. Defaults to False.
            **kwargs: Additional keyword arguments for the training loop.
        """
        self.logger.info("Starting training process.")
        optimizer = self.get_optimizer(learning_rate=learning_rate, weight_decay=weight_decay)
        total_steps = epochs * len(train_dataloader)
        scheduler = self.get_scheduler(
            optimizer=optimizer,
            scheduler_class=get_linear_schedule_with_warmup,
            scheduler_params={
                "num_warmup_steps": num_warmup_steps,
                "num_training_steps": total_steps,
            },
        )

        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        for epoch in range(epochs):
            self.logger.info(f"Starting epoch {epoch + 1}/{epochs}")
            epoch_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                loss = self.training_step(batch, optimizer, scaler, use_amp)
                scheduler.step()
                epoch_loss += loss

                if (step + 1) % log_interval == 0:
                    avg_loss = epoch_loss / (step + 1)
                    self.logger.info(
                        f"Epoch [{epoch + 1}/{epochs}] Step [{step + 1}/{len(train_dataloader)}] Loss: {loss:.4f} Avg Loss: {avg_loss:.4f}"
                    )

            avg_epoch_loss = epoch_loss / len(train_dataloader)
            self.logger.info(f"Epoch [{epoch + 1}/{epochs}] completed. Average Loss: {avg_epoch_loss:.4f}")

            if val_dataloader:
                self.evaluate(val_dataloader)

        self.logger.info("Training process completed.")

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluates the YOLOS model on the provided dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The evaluation data loader.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        self.logger.info("Starting evaluation.")
        self.model.eval()
        all_outputs = []
        all_targets = []
        image_ids = []

        for batch in dataloader:
            outputs = self.evaluation_step(batch)
            all_outputs.append(outputs)
            all_targets.append(batch.get('labels'))
            image_ids.extend(batch.get('image_id', []))

        metrics = self.compute_metrics(all_outputs, all_targets, image_ids)
        self.logger.info(f"Evaluation metrics: {metrics}")
        return metrics

