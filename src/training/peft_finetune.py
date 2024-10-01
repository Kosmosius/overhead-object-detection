# src/training/peft_finetune.py

import os
import torch
import logging
from typing import Optional
from tqdm import tqdm
from datasets import load_dataset
from torch.cuda.amp import autocast, GradScaler
from torchvision.datasets import CocoDetection
from transformers import AdamW, DetrFeatureExtractor, get_scheduler, DetrForObjectDetection
from src.models.model_factory import ModelFactory
from src.utils.config_parser import ConfigParser
from src.utils.logging_utils import setup_logging
from src.training.optimizers import get_optimizer_and_scheduler
from peft import PeftModel, PeftConfig, get_peft_model

# Set up logging
logger = logging.getLogger(__name__)

def setup_peft_model(model_name: str, num_classes: int, peft_config: PeftConfig) -> PeftModel:
    """
    Set up a PEFT model by applying PEFT-specific configurations to a pre-trained model.

    Args:
        pretrained_model_name (str): HuggingFace model name or path (e.g., "facebook/detr-resnet-50").
        num_classes (int): Number of object detection classes.
        peft_config (PeftConfig): Configuration for PEFT fine-tuning.

    Returns:
        PeftModel: PEFT-configured model.
    """
    try:
        model = DetrForObjectDetection.from_pretrained(pretrained_model_name, num_labels=num_classes)
        peft_model = get_peft_model(model, peft_config)
        logger.info(f"PEFT model successfully initialized with {pretrained_model_name}.")
        return peft_model
    except Exception as e:
        error_message = f"Error setting up PEFT model: {e}"
        logger.error(error_message)
        raise Exception(error_message) from e

def prepare_dataloader(data_dir: str, batch_size: int, feature_extractor: DetrFeatureExtractor, mode="train") -> torch.utils.data.DataLoader:
    """
    Prepare a DataLoader for a dataset (supports both COCO-style and HuggingFace datasets).

    Args:
        data_dir (str): Path to the dataset directory (for custom datasets like COCO).
        batch_size (int): Batch size for loading data.
        feature_extractor (DetrFeatureExtractor): Feature extractor for preprocessing images.
        mode (str): Either "train" or "val" for training or validation mode.

    Returns:
        torch.utils.data.DataLoader: Prepared DataLoader for the specified dataset.
    """
    try:
        assert mode in ["train", "val"], "Mode should be either 'train' or 'val'."

        ann_file = os.path.join(data_dir, f'annotations/instances_{mode}2017.json')
        img_dir = os.path.join(data_dir, f'{mode}2017')

        dataset = CocoDetection(
            root=img_dir,
            annFile=ann_file,
            transform=lambda x: feature_extractor(images=x, return_tensors="pt").pixel_values[0]
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True if mode == "train" else False,
            collate_fn=lambda batch: tuple(zip(*batch))
        )

        logger.info(f"Dataloader for mode '{mode}' prepared successfully with batch size {batch_size}.")
        return dataloader

    except Exception as e:
        logger.error(f"Error preparing dataloader: {e}")
        raise

def fine_tune_peft_model(
    model: PeftModel,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: Optional[torch.utils.data.DataLoader],
    optimizer,
    scheduler,
    config: dict,
    device: str = "cuda"
) -> None:
    """
    Fine-tune a PEFT model with a training and validation loop.

    Args:
        model (PeftModel): The model to be fine-tuned.
        train_dataloader (torch.utils.data.DataLoader): Dataloader for training data.
        val_dataloader (Optional[torch.utils.data.DataLoader]): Dataloader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for fine-tuning.
        scheduler (torch.optim.lr_scheduler): Scheduler for learning rate adjustment.
        config (dict): Configuration settings for training (epochs, mixed_precision, etc.).
        device (str): Device for training ('cuda' or 'cpu').
    """
    model.to(device)
    num_epochs = config['training']['num_epochs']
    mixed_precision = config['training'].get('mixed_precision', True)
    scaler = GradScaler() if mixed_precision else None
    checkpoint_dir = config['training']['checkpoint_dir']

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        logger.info(f"Starting Epoch {epoch + 1}/{num_epochs}")

        # Training loop
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()

            pixel_values, target = batch

            # Validate and move pixel_values to device
            if isinstance(pixel_values, list):
                pixel_values = torch.stack(pixel_values).to(device)
            elif isinstance(pixel_values, torch.Tensor):
                pixel_values = pixel_values.to(device)
            else:
                raise TypeError("pixel_values must be a list or torch.Tensor")

            # Validate and move targets to device
            if isinstance(target, dict):
                required_fields = ['labels', 'boxes']
                for field in required_fields:
                    if field not in target:
                        raise KeyError(f"Missing '{field}' in target")
                    if not isinstance(target[field], torch.Tensor):
                        raise TypeError(f"'{field}' in target must be a torch.Tensor")
                target = {k: v.to(device) for k, v in target.items()}
            elif isinstance(target, list):
                target = [{k: v.to(device) for k, v in t.items()} for t in target]
            else:
                raise TypeError("target must be a dict or list of dicts")

            with autocast(enabled=mixed_precision):
                outputs = model(pixel_values, target)
                loss = outputs.loss

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}")

        # Validation loop
        if val_dataloader:
            model.eval()
            val_loss = 0.0
            logger.info(f"Starting Validation for Epoch {epoch + 1}/{num_epochs}")

            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc=f"Validating Epoch {epoch + 1}/{num_epochs}"):
                    pixel_values, target = batch

                    # Validate and move pixel_values to device
                    if isinstance(pixel_values, list):
                        pixel_values = torch.stack(pixel_values).to(device)
                    elif isinstance(pixel_values, torch.Tensor):
                        pixel_values = pixel_values.to(device)
                    else:
                        raise TypeError("pixel_values must be a list or torch.Tensor")

                    # Validate and move targets to device
                    if isinstance(target, dict):
                        required_fields = ['labels', 'boxes']
                        for field in required_fields:
                            if field not in target:
                                raise KeyError(f"Missing '{field}' in target")
                            if not isinstance(target[field], torch.Tensor):
                                raise TypeError(f"'{field}' in target must be a torch.Tensor")
                        target = {k: v.to(device) for k, v in target.items()}
                    elif isinstance(target, list):
                        target = [{k: v.to(device) for k, v in t.items()} for t in target]
                    else:
                        raise TypeError("target must be a dict or list of dicts")

                    with autocast(enabled=mixed_precision):
                        outputs = model(pixel_values, target)
                        loss = outputs.loss

                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_dataloader)
            logger.info(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, checkpoint_path)
        logger.info(f"Model checkpoint saved at {checkpoint_path}")

def main(config_path: str) -> None:
    """
    Main function for fine-tuning the PEFT model using the provided configuration file.

    Args:
        config_path (str): Path to the configuration file.
    """
    # Load configuration
    config_parser = ConfigParser(config_path)
    config = config_parser.config

    # Set up logging
    setup_logging()
    logger.info("Starting PEFT fine-tuning process.")

    # Provide default values for missing configuration sections
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    optimizer_config = config.get('optimizer', {})
    loss_config = config.get('loss', {})

    # Validate essential configuration fields
    required_data_fields = ['data_dir']
    for field in required_data_fields:
        if field not in data_config:
            raise KeyError(f"Missing required data configuration field: '{field}'")

    # Prepare feature extractor
    feature_extractor = DetrFeatureExtractor.from_pretrained(model_config['model_name'])

    # Prepare PEFT model
    peft_config = PeftConfig.from_pretrained(model_config['peft_model_path'])
    model = setup_peft_model(
        model_name=model_config['model_name'],
        num_classes=model_config['num_classes'],
        peft_config=peft_config
    )

    # Prepare dataloaders
    train_loader = prepare_dataloader(
        data_dir=data_config['data_dir'],
        batch_size=training_config['batch_size'],
        feature_extractor=feature_extractor,
        mode="train"
    )
    val_loader = prepare_dataloader(
        data_dir=data_config['data_dir'],
        batch_size=training_config['batch_size'],
        feature_extractor=feature_extractor,
        mode="val"
    )

    # Set up optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model,
        config=optimizer_config,
        num_training_steps=training_config['num_epochs'] * len(train_loader)
    )

    # Fine-tune the model
    fine_tune_peft_model(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=training_config.get('device', 'cpu')
    )

    # Save the final model
    final_model_path = os.path.join(training_config.get('output_dir', './output'), 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved at {final_model_path}")

if __name__ == "__main__":
    config_path = "configs/peft_config.yaml"
    main(config_path)
