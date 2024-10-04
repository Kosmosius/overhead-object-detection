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

import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig, AutoImageProcessor
from transformers.adapters import AdapterConfig, LoRAConfig
from src.models.adapters.lora_adapter import LoRAAdapter
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class Mask2FormerModel:
    """
    Wrapper class for Mask2Former model integrating Hugging Face Transformers and custom adapters.

    Attributes:
        model (Mask2FormerForUniversalSegmentation): The Mask2Former model.
        image_processor (AutoImageProcessor): The image processor for preprocessing and postprocessing.
    """

    def __init__(self, config_path: str, adapter_type: str = None, adapter_config: dict = None, use_pretrained: bool = True):
        """
        Initializes the Mask2Former model with the given configuration.

        Args:
            config_path (str): Path to the Mask2Former configuration file.
            adapter_type (str, optional): Type of adapter to apply (e.g., 'lora', 'qlora'). Defaults to None.
            adapter_config (dict, optional): Configuration parameters for the adapter. Defaults to None.
            use_pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
        """
        logger.info("Initializing Mask2Former model.")
        self.config = Mask2FormerConfig.from_pretrained(config_path)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(config_path) if use_pretrained else Mask2FormerForUniversalSegmentation(self.config)

        if adapter_type:
            self._add_adapter(adapter_type, adapter_config)

        self.image_processor = AutoImageProcessor.from_pretrained(config_path)
        logger.info("Mask2Former model initialized successfully.")

    def _add_adapter(self, adapter_type: str, adapter_config: dict):
        """
        Adds an adapter to the Mask2Former model.

        Args:
            adapter_type (str): Type of adapter to add ('lora' or 'qlora').
            adapter_config (dict): Configuration parameters for the adapter.
        """
        logger.info(f"Adding adapter of type: {adapter_type}")
        if adapter_type.lower() == 'lora':
            config = LoRAConfig(**adapter_config)
            self.model.add_adapter("lora_adapter", config=config)
            self.model.train_adapter("lora_adapter")
            logger.info("LoRA adapter added successfully.")
        elif adapter_type.lower() == 'qlora':
            config = LoRAConfig(**adapter_config)  # Assuming QLoRA uses similar config; adjust if different
            self.model.add_adapter("qlora_adapter", config=config)
            self.model.train_adapter("qlora_adapter")
            logger.info("QLoRA adapter added successfully.")
        else:
            logger.warning(f"Adapter type '{adapter_type}' is not supported.")

    def fine_tune(self, train_dataset, eval_dataset, training_args):
        """
        Fine-tunes the Mask2Former model on the provided dataset.

        Args:
            train_dataset (torch.utils.data.Dataset): The training dataset.
            eval_dataset (torch.utils.data.Dataset): The evaluation dataset.
            training_args (transformers.TrainingArguments): Training arguments for the Trainer.
        
        Returns:
            transformers.Trainer: The Trainer instance after training.
        """
        from transformers import Trainer

        logger.info("Starting fine-tuning process.")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            image_processor=self.image_processor,
        )

        trainer.train()
        logger.info("Fine-tuning completed successfully.")
        return trainer

    def save_model(self, save_directory: str):
        """
        Saves the trained model and adapters to the specified directory.

        Args:
            save_directory (str): Directory path to save the model.
        """
        logger.info(f"Saving Mask2Former model to {save_directory}")
        self.model.save_pretrained(save_directory)
        self.image_processor.save_pretrained(save_directory)
        if self.model.is_adapter_trained("lora_adapter"):
            self.model.save_adapter(save_directory, "lora_adapter")
            logger.info("LoRA adapter saved successfully.")
        if self.model.is_adapter_trained("qlora_adapter"):
            self.model.save_adapter(save_directory, "qlora_adapter")
            logger.info("QLoRA adapter saved successfully.")
        logger.info("Model and adapters saved successfully.")

    def load_model(self, load_directory: str, adapter_type: str = None):
        """
        Loads the Mask2Former model and adapters from the specified directory.

        Args:
            load_directory (str): Directory path to load the model from.
            adapter_type (str, optional): Type of adapter to load ('lora', 'qlora'). Defaults to None.
        """
        logger.info(f"Loading Mask2Former model from {load_directory}")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(load_directory)
        self.image_processor = AutoImageProcessor.from_pretrained(load_directory)

        if adapter_type:
            adapter_name = f"{adapter_type}_adapter"
            self.model.load_adapter(load_directory, adapter_name)
            self.model.set_active_adapters(adapter_name)
            logger.info(f"{adapter_type.upper()} adapter loaded and set as active.")

        logger.info("Model loaded successfully.")

    def predict(self, image: torch.Tensor, task: str = 'instance', threshold: float = 0.5):
        """
        Performs prediction on a single image.

        Args:
            image (torch.Tensor): Input image tensor.
            task (str, optional): Type of segmentation task ('instance', 'semantic', 'panoptic'). Defaults to 'instance'.
            threshold (float, optional): Threshold for prediction confidence. Defaults to 0.5.
        
        Returns:
            dict: Post-processed segmentation map.
        """
        self.model.eval()
        with torch.no_grad():
            inputs = self.image_processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)

        if task == 'instance':
            pred = self.image_processor.post_process_instance_segmentation(outputs, threshold=threshold, target_sizes=[image.shape[-2:]])[0]
        elif task == 'semantic':
            pred = self.image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.shape[-2:]])[0]
        elif task == 'panoptic':
            pred = self.image_processor.post_process_panoptic_segmentation(outputs, threshold=threshold, target_sizes=[image.shape[-2:]])[0]
        else:
            raise ValueError(f"Unsupported task type: {task}")

        return pred

    def inference(self, dataloader, task: str = 'instance', threshold: float = 0.5):
        """
        Runs inference on a dataset.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
            task (str, optional): Type of segmentation task ('instance', 'semantic', 'panoptic'). Defaults to 'instance'.
            threshold (float, optional): Threshold for prediction confidence. Defaults to 0.5.
        
        Returns:
            List[dict]: List of post-processed segmentation maps.
        """
        self.model.eval()
        results = []

        for batch in dataloader:
            images = batch['pixel_values']
            with torch.no_grad():
                outputs = self.model(**batch)

            if task == 'instance':
                preds = self.image_processor.post_process_instance_segmentation(outputs, threshold=threshold, target_sizes=batch['target_sizes'])[0]
            elif task == 'semantic':
                preds = self.image_processor.post_process_semantic_segmentation(outputs, target_sizes=batch['target_sizes'])[0]
            elif task == 'panoptic':
                preds = self.image_processor.post_process_panoptic_segmentation(outputs, threshold=threshold, target_sizes=batch['target_sizes'])[0]
            else:
                raise ValueError(f"Unsupported task type: {task}")

            results.append(preds)

        return results

if __name__ == "__main__":
    # Example usage
    import argparse
    from transformers import TrainingArguments
    from src.data.dataloader import get_dataloader  # Assuming a dataloader module exists
    from src.models.zoo.mask2former import Mask2FormerModel

    parser = argparse.ArgumentParser(description="Train Mask2Former Model")
    parser.add_argument('--config', type=str, required=True, help='Path to Mask2Former config')
    parser.add_argument('--adapter', type=str, choices=['lora', 'qlora'], help='Type of adapter to use')
    parser.add_argument('--adapter_config', type=dict, help='Configuration for the adapter')
    parser.add_argument('--train_dataset', type=str, required=True, help='Path to training dataset')
    parser.add_argument('--eval_dataset', type=str, required=True, help='Path to evaluation dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the model')
    args = parser.parse_args()

    # Initialize model
    model = Mask2FormerModel(config_path=args.config, adapter_type=args.adapter, adapter_config=args.adapter_config)

    # Prepare datasets
    train_loader = get_dataloader(args.train_dataset, batch_size=8, shuffle=True)
    eval_loader = get_dataloader(args.eval_dataset, batch_size=8, shuffle=False)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        load_best_model_at_end=True,
    )

    # Fine-tune the model
    trainer = model.fine_tune(train_dataset=train_loader.dataset, eval_dataset=eval_loader.dataset, training_args=training_args)

    # Save the trained model
    model.save_model(args.output_dir)
