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
            pixel_values = pixel_values.to(device)
            target = [{k: v.to(device) for k, v in t.items()} for t in target]

            with autocast(enabled=mixed_precision):
                try:
                    outputs = model(pixel_values=pixel_values, labels=target)
                    loss = outputs.loss
                except Exception as e:
                    logger.error(f"Error during forward pass: {e}")
                    raise

            if mixed_precision:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step()
            train_loss += loss.item()

        if len(train_dataloader) > 0:
            avg_train_loss = train_loss / len(train_dataloader)
        else:
            avg_train_loss = 0.0
            logger.warning("Training dataloader is empty. Setting average training loss to 0.0.")

        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss}")

        # Validation loop
        if val_dataloader is not None:
            model.eval()
            val_loss = 0.0
            logger.info(f"Starting Validation for Epoch {epoch + 1}/{num_epochs}")

            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc=f"Validating Epoch {epoch + 1}/{num_epochs}"):
                    try:
                        pixel_values, target = batch
                        pixel_values = pixel_values.to(device)
                        target = [{k: v.to(device) for k, v in t.items()} for t in target]

                        with autocast(enabled=mixed_precision):
                            outputs = model(pixel_values=pixel_values, labels=target)
                            loss = outputs.loss
                        
                        val_loss += loss.item()
                    except Exception as e:
                        logger.error(f"Error during validation: {e}")
                        raise

            if len(val_dataloader) > 0:
                avg_val_loss = val_loss / len(val_dataloader)
            else:
                avg_val_loss = 0.0
                logger.warning("Validation dataloader is empty. Setting average validation loss to 0.0.")

            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss}")
        else:
            logger.info("No validation dataloader provided. Skipping validation.")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pt")
        try:
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Model checkpoint saved at {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            raise

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

    # Prepare feature extractor
    feature_extractor = DetrFeatureExtractor.from_pretrained(config['model']['model_name'])

    # Prepare PEFT model
    peft_config = PeftConfig.from_pretrained(config['model']['peft_model_path'])
    model = setup_peft_model(
        model_name=config['model']['model_name'],
        num_classes=config['model']['num_classes'],
        peft_config=peft_config
    )

    # Prepare dataloaders
    train_loader = prepare_dataloader(
        data_dir=config['data']['data_dir'],
        batch_size=config['training']['batch_size'],
        feature_extractor=feature_extractor,
        mode="train"
    )

    val_loader = prepare_dataloader(
        data_dir=config['data']['data_dir'],
        batch_size=config['training']['batch_size'],
        feature_extractor=feature_extractor,
        mode="val"
    )

    # Set up optimizer and scheduler
    num_training_steps = config['training']['num_epochs'] * len(train_loader)
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model,
        config=config['optimizer'],
        num_training_steps=num_training_steps
    )

    # Fine-tune PEFT model
    fine_tune_peft_model(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=config['training'].get('device', 'cuda')
    )

    logger.info("PEFT fine-tuning completed.")

if __name__ == "__main__":
    config_path = "configs/peft_config.yaml"
    main(config_path)
