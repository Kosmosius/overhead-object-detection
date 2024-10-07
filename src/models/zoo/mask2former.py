# src/models/zoo/mask2former.py

"""
Mask2Former Model Definition for Overhead Object Detection

This module provides a Mask2Former model tailored for overhead imagery analysis.
It leverages Hugging Face's Transformers library to utilize pre-trained Mask2Former
models and integrates advanced fine-tuning techniques such as LoRA and QLoRA for
efficient domain adaptation.

Contributors:
- Shivalika Singh
- Alara Dirik

Original Mask2Former implementation: [GitHub Repository](https://github.com/facebookresearch/Mask2Former)
"""

import os
import logging
from typing import Any, Dict, Optional, Type, List, Tuple, Union

import torch
from torch.optim import Optimizer
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerConfig,
    AutoImageProcessor,
    Trainer,
    TrainingArguments,
    AdapterConfig,
    LoRAConfig,
    PreTrainedModel,
)
from transformers.adapters import AdapterType

from src.models.zoo.base_model import BaseModel
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Mask2FormerModel(BaseModel):
    """
    Mask2Former Model for Universal Image Segmentation.

    This class extends the BaseModel to provide functionalities specific to the Mask2Former architecture,
    including loading pre-trained weights, fine-tuning with adapters, and handling various segmentation tasks.

    Attributes:
        model (Mask2FormerForUniversalSegmentation): The Mask2Former model.
        image_processor (AutoImageProcessor): The image processor for preprocessing and postprocessing.
    """

    def __init__(
        self,
        model_name_or_path: str = "facebook/mask2former-swin-small-coco-instance",
        adapter_type: Optional[str] = None,
        adapter_config: Optional[Dict[str, Any]] = None,
        use_pretrained: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        """
        Initializes the Mask2Former model with the given configuration.

        Args:
            model_name_or_path (str, optional): Path to the pretrained model or model identifier from Hugging Face.
                Defaults to "facebook/mask2former-swin-small-coco-instance".
            adapter_type (Optional[str], optional): Type of adapter to apply (e.g., 'lora', 'qlora'). Defaults to None.
            adapter_config (Optional[Dict[str, Any]], optional): Configuration parameters for the adapter. Defaults to None.
            use_pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
            device (Optional[Union[str, torch.device]], optional): Device to run the model on ('cpu' or 'cuda').
                If None, defaults to CUDA if available.
            **kwargs: Additional keyword arguments for model configuration.
        """
        super().__init__(
            model_name_or_path=model_name_or_path,
            model_class=Mask2FormerForUniversalSegmentation if use_pretrained else Mask2FormerForUniversalSegmentation,
            config=None,  # Config is loaded within BaseModel
            num_labels=None,  # Mask2Former handles multiple labels internally
            **kwargs,
        )

        # Replace the base model with Mask2FormerForUniversalSegmentation
        if use_pretrained:
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_name_or_path, **kwargs
            )
            logger.info(f"Loaded pretrained Mask2Former model from {model_name_or_path}")
        else:
            self.model = Mask2FormerForUniversalSegmentation(self.config)
            logger.info("Initialized Mask2Former model with random weights")

        # Initialize the image processor
        self.image_processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        logger.info("Initialized AutoImageProcessor")

        # Device management
        self.to_device(device)

        # Adapter integration
        if adapter_type:
            self.add_adapter(adapter_type, adapter_config)

    def add_adapter(self, adapter_type: str, adapter_config: Optional[Dict[str, Any]] = None):
        """
        Adds an adapter to the Mask2Former model for efficient fine-tuning.

        Args:
            adapter_type (str): Type of adapter to add ('lora', 'qlora').
            adapter_config (Optional[Dict[str, Any]], optional): Configuration parameters for the adapter.
                If None, default LoRA configuration is used.
        """
        if adapter_type.lower() == 'lora':
            config = adapter_config or {"r": 8, "lora_alpha": 32, "lora_dropout": 0.1}
            lora_config = LoRAConfig(**config)
            self.model.add_adapter("lora_adapter", config=lora_config, adapter_type=AdapterType.text_langchain)
            self.model.train_adapter("lora_adapter")
            logger.info("LoRA adapter added successfully.")
        elif adapter_type.lower() == 'qlora':
            # Placeholder for QLoRA integration
            logger.warning("QLoRA integration is not implemented yet.")
            raise NotImplementedError("QLoRA integration is not implemented yet.")
        else:
            logger.error(f"Adapter type '{adapter_type}' is not supported.")
            raise ValueError(f"Adapter type '{adapter_type}' is not supported.")

    def forward(self, pixel_values: torch.Tensor, pixel_mask: Optional[torch.Tensor] = None, **kwargs) -> Any:
        """
        Forward pass through the Mask2Former model.

        Args:
            pixel_values (torch.Tensor): Input images of shape (batch_size, num_channels, height, width).
            pixel_mask (Optional[torch.Tensor], optional): Mask tensor indicating valid pixels. Shape (batch_size, height, width).
            **kwargs: Additional keyword arguments for the model.

        Returns:
            Any: Model outputs.
        """
        inputs = {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask
        }
        inputs.update(kwargs)
        outputs = self.model(**inputs)
        return outputs

    def compute_loss(self, outputs: Any, targets: Any) -> torch.Tensor:
        """
        Computes the loss given model outputs and targets.

        Args:
            outputs (Any): Outputs from the model.
            targets (Any): Ground truth targets.

        Returns:
            torch.Tensor: The computed loss.
        """
        if hasattr(outputs, 'loss'):
            return outputs.loss
        else:
            logger.error("Outputs do not contain loss information.")
            raise AttributeError("Outputs do not contain loss information.")

    def compute_metrics(self, outputs: List[Any], targets: List[Any], image_ids: List[Any]) -> Dict[str, float]:
        """
        Computes evaluation metrics given the model outputs and targets.

        Args:
            outputs (List[Any]): List of model outputs.
            targets (List[Any]): List of ground truth targets.
            image_ids (List[Any]): List of image IDs corresponding to the batches.

        Returns:
            Dict[str, float]: Dictionary of computed metrics.
        """
        # Placeholder for metric computation (e.g., mAP, IoU)
        # Implement actual metric computation using COCO Evaluator or similar
        metrics = {"mAP": 0.0}  # Replace with actual computation
        logger.info(f"Computed metrics: {metrics}")
        return metrics

    def post_process_predictions(
        self,
        outputs: Any,
        threshold: float = 0.5,
        top_k: int = 100,
        target_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Applies post-processing to model outputs to obtain final predictions.

        Args:
            outputs (Any): Raw outputs from the model.
            threshold (float, optional): Score threshold to filter predictions.
            top_k (int, optional): Maximum number of predictions to retain per image.
            target_sizes (Optional[List[Tuple[int, int]]], optional): Target sizes for resizing boxes.

        Returns:
            List[Dict[str, Any]]: List of prediction dictionaries containing 'scores', 'labels', and 'boxes'.
        """
        results = self.image_processor.post_process_instance_segmentation(
            outputs, threshold=threshold, target_sizes=target_sizes, top_k=top_k
        )
        predictions = []
        for result in results:
            prediction = {
                "scores": result["scores"].cpu().numpy(),
                "labels": result["labels"].cpu().numpy(),
                "boxes": result["boxes"].cpu().numpy(),
                # Add more fields if needed (e.g., masks)
            }
            predictions.append(prediction)
        return predictions

    def fine_tune(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        num_warmup_steps: int = 0,
        log_interval: int = 100,
        use_amp: bool = False,
        **kwargs,
    ):
        """
        Fine-tunes the Mask2Former model on the provided dataset.

        Args:
            train_dataloader (torch.utils.data.DataLoader): The training dataset.
            eval_dataloader (Optional[torch.utils.data.DataLoader], optional): The evaluation dataset.
            epochs (int, optional): Number of training epochs. Defaults to 10.
            learning_rate (float, optional): Learning rate. Defaults to 1e-4.
            weight_decay (float, optional): Weight decay. Defaults to 1e-4.
            num_warmup_steps (int, optional): Number of warmup steps for the scheduler. Defaults to 0.
            log_interval (int, optional): Logging frequency in steps. Defaults to 100.
            use_amp (bool, optional): Whether to use mixed precision training. Defaults to False.
            **kwargs: Additional keyword arguments for Trainer.
        """
        optimizer = self.get_optimizer(learning_rate=learning_rate, weight_decay=weight_decay)
        total_steps = epochs * len(train_dataloader)
        scheduler = self.get_scheduler(
            optimizer=optimizer,
            scheduler_class=lambda opt, **s_params: get_linear_schedule_with_warmup(opt, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps),
            scheduler_params={},
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir="./results",
                num_train_epochs=epochs,
                per_device_train_batch_size=train_dataloader.batch_size,
                per_device_eval_batch_size=eval_dataloader.batch_size if eval_dataloader else train_dataloader.batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                logging_dir="./logs",
                logging_steps=log_interval,
                evaluation_strategy="steps" if eval_dataloader else "no",
                save_steps=log_interval,
                save_total_limit=3,
                load_best_model_at_end=True if eval_dataloader else False,
                fp16=use_amp,
                **kwargs,
            ),
            train_dataset=train_dataloader.dataset,
            eval_dataset=eval_dataloader.dataset if eval_dataloader else None,
            tokenizer=self.image_processor,
            data_collator=None,  # Define if necessary
            compute_metrics=self.compute_metrics,
        )

        # Start training
        logger.info("Starting fine-tuning process.")
        trainer.train()
        logger.info("Fine-tuning completed successfully.")

        # Save the best model
        trainer.save_model()
        logger.info("Best model saved successfully.")

    def save_model(self, save_directory: str):
        """
        Saves the Mask2Former model and adapters to the specified directory.

        Args:
            save_directory (str): Directory path to save the model.
        """
        super().save(save_directory)
        # Save adapters if any
        adapters = self.model.config.adapters
        if adapters:
            for adapter_name in adapters:
                self.model.save_adapter(save_directory, adapter_name)
                logger.info(f"Adapter '{adapter_name}' saved successfully.")
        logger.info(f"Mask2Former model and adapters saved to {save_directory}")

    @classmethod
    def load_model(
        cls,
        load_directory: str,
        adapter_type: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        """
        Loads the Mask2Former model and adapters from the specified directory.

        Args:
            load_directory (str): Directory path to load the model from.
            adapter_type (Optional[str], optional): Type of adapter to load ('lora', 'qlora'). Defaults to None.
            device (Optional[Union[str, torch.device]], optional): Device to run the model on ('cpu' or 'cuda').
                If None, defaults to CUDA if available.
            **kwargs: Additional keyword arguments for model configuration.

        Returns:
            Mask2FormerModel: An instance of the Mask2Former model.
        """
        model = cls(
            model_name_or_path=load_directory,
            adapter_type=None,  # Adapters are loaded separately
            use_pretrained=True,
            device=device,
            **kwargs,
        )
        if adapter_type:
            adapter_name = f"{adapter_type}_adapter"
            model.model.load_adapter(load_directory, adapter_name)
            model.model.set_active_adapters(adapter_name)
            logger.info(f"{adapter_type.upper()} adapter loaded and set as active.")
        return model

    def predict(
        self,
        images: List[torch.Tensor],
        task: str = 'instance',
        threshold: float = 0.5,
        top_k: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Performs object detection on a batch of images.

        Args:
            images (List[torch.Tensor]): List of images to perform detection on. Each image should be a tensor of shape (3, H, W).
            task (str, optional): Type of segmentation task ('instance', 'semantic', 'panoptic'). Defaults to 'instance'.
            threshold (float, optional): Confidence threshold to filter predictions. Defaults to 0.5.
            top_k (int, optional): Number of top predictions to return per image. Defaults to 100.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing 'scores', 'labels', and 'boxes' for each image.
        """
        self.model.eval()
        with torch.no_grad():
            inputs = self.image_processor(images=images, return_tensors="pt")
            inputs = self.prepare_inputs(inputs)
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.shape[-2:] for image in images])
        if task == 'instance':
            results = self.image_processor.post_process_instance_segmentation(
                outputs, threshold=threshold, target_sizes=target_sizes, top_k=top_k
            )
        elif task == 'semantic':
            results = self.image_processor.post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes
            )
        elif task == 'panoptic':
            results = self.image_processor.post_process_panoptic_segmentation(
                outputs, threshold=threshold, target_sizes=target_sizes
            )
        else:
            logger.error(f"Unsupported task type: {task}")
            raise ValueError(f"Unsupported task type: {task}")

        # Convert results to a standardized format
        predictions = []
        for result in results:
            prediction = {}
            if task in ['instance', 'panoptic']:
                prediction['scores'] = result['scores'].cpu().numpy()
                prediction['labels'] = result['labels'].cpu().numpy()
                prediction['boxes'] = result['boxes'].cpu().numpy()
                if task == 'panoptic':
                    prediction['segmentation'] = result.get('segmentation', None)
            elif task == 'semantic':
                prediction['segmentation'] = result['segmentation'].cpu().numpy()
            predictions.append(prediction)

        return predictions

    def inference(
        self,
        dataloader: torch.utils.data.DataLoader,
        task: str = 'instance',
        threshold: float = 0.5,
        top_k: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Runs inference on a dataset.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
            task (str, optional): Type of segmentation task ('instance', 'semantic', 'panoptic'). Defaults to 'instance'.
            threshold (float, optional): Confidence threshold to filter predictions. Defaults to 0.5.
            top_k (int, optional): Number of top predictions to return per image. Defaults to 100.

        Returns:
            List[Dict[str, Any]]: List of post-processed segmentation maps.
        """
        self.model.eval()
        all_predictions = []

        for batch in dataloader:
            images = batch['pixel_values']
            with torch.no_grad():
                outputs = self.model(pixel_values=images, pixel_mask=batch.get('pixel_mask', None))

            if task == 'instance':
                preds = self.image_processor.post_process_instance_segmentation(
                    outputs, threshold=threshold, target_sizes=batch['target_sizes'], top_k=top_k
                )
            elif task == 'semantic':
                preds = self.image_processor.post_process_semantic_segmentation(
                    outputs, target_sizes=batch['target_sizes']
                )
            elif task == 'panoptic':
                preds = self.image_processor.post_process_panoptic_segmentation(
                    outputs, threshold=threshold, target_sizes=batch['target_sizes']
                )
            else:
                logger.error(f"Unsupported task type: {task}")
                raise ValueError(f"Unsupported task type: {task}")

            for result in preds:
                prediction = {}
                if task in ['instance', 'panoptic']:
                    prediction['scores'] = result['scores'].cpu().numpy()
                    prediction['labels'] = result['labels'].cpu().numpy()
                    prediction['boxes'] = result['boxes'].cpu().numpy()
                    if task == 'panoptic':
                        prediction['segmentation'] = result.get('segmentation', None)
                elif task == 'semantic':
                    prediction['segmentation'] = result['segmentation'].cpu().numpy()
                all_predictions.append(prediction)

        return all_predictions


if __name__ == "__main__":
    # Example usage
    import argparse
    from src.data.dataloader import get_dataloader  # Assuming a dataloader module exists
    from transformers import TrainingArguments

    parser = argparse.ArgumentParser(description="Fine-tune Mask2Former Model")
    parser.add_argument('--config', type=str, required=True, help='Path to Mask2Former config')
    parser.add_argument('--adapter', type=str, choices=['lora', 'qlora'], help='Type of adapter to use')
    parser.add_argument('--adapter_config', type=dict, help='Configuration for the adapter')
    parser.add_argument('--train_dataset', type=str, required=True, help='Path to training dataset')
    parser.add_argument('--eval_dataset', type=str, required=True, help='Path to evaluation dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the model')
    parser.add_argument('--device', type=str, default=None, help='Device to run the model on')
    args = parser.parse_args()

    # Initialize Mask2Former model
    model = Mask2FormerModel(
        model_name_or_path=args.config,
        adapter_type=args.adapter,
        adapter_config=args.adapter_config,
        use_pretrained=True,
        device=args.device,
    )

    # Prepare datasets
    train_loader = get_dataloader(args.train_dataset, batch_size=8, shuffle=True)
    eval_loader = get_dataloader(args.eval_dataset, batch_size=8, shuffle=False)

    # Fine-tune the model
    model.fine_tune(
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        epochs=10,
        learning_rate=1e-4,
        weight_decay=1e-4,
        num_warmup_steps=0,
        log_interval=100,
        use_amp=False,
    )

    # Save the trained model
    model.save_model(args.output_dir)

"""
if __name__ == "__main__":
    # Example usage
    import argparse
    from src.data.dataloader import get_dataloader  # Assuming a dataloader module exists
    from transformers import TrainingArguments

    parser = argparse.ArgumentParser(description="Fine-tune Mask2Former Model")
    parser.add_argument('--config', type=str, required=True, help='Path to Mask2Former config')
    parser.add_argument('--adapter', type=str, choices=['lora', 'qlora'], help='Type of adapter to use')
    parser.add_argument('--adapter_config', type=dict, help='Configuration for the adapter')
    parser.add_argument('--train_dataset', type=str, required=True, help='Path to training dataset')
    parser.add_argument('--eval_dataset', type=str, required=True, help='Path to evaluation dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the model')
    parser.add_argument('--device', type=str, default=None, help='Device to run the model on')
    args = parser.parse_args()

    # Initialize Mask2Former model
    model = Mask2FormerModel(
        model_name_or_path=args.config,
        adapter_type=args.adapter,
        adapter_config=args.adapter_config,
        use_pretrained=True,
        device=args.device,
    )

    # Prepare datasets
    train_loader = get_dataloader(args.train_dataset, batch_size=8, shuffle=True)
    eval_loader = get_dataloader(args.eval_dataset, batch_size=8, shuffle=False)

    # Fine-tune the model
    model.fine_tune(
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        epochs=10,
        learning_rate=1e-4,
        weight_decay=1e-4,
        num_warmup_steps=0,
        log_interval=100,
        use_amp=False,
    )

    # Save the trained model
    model.save_model(args.output_dir)
"""
