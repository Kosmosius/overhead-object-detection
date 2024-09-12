# src/training/peft_finetune.py

import torch
import logging
from transformers import DetrForObjectDetection, AdamW, get_scheduler, DetrFeatureExtractor
from peft import PeftModel, PeftConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from torch.cuda.amp import autocast, GradScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_peft_model(pretrained_model_name: str, num_classes: int, peft_config: PeftConfig) -> PeftModel:
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
        logging.info(f"PEFT model successfully initialized with {pretrained_model_name}.")
        return peft_model
    except Exception as e:
        logging.error(f"Error setting up PEFT model: {e}")
        raise

def prepare_dataloader(data_dir: str, batch_size: int, feature_extractor: DetrFeatureExtractor, mode="train") -> DataLoader:
    """
    Prepare a DataLoader for a dataset (supports both COCO-style and HuggingFace datasets).

    Args:
        data_dir (str): Path to the dataset directory (for custom datasets like COCO).
        batch_size (int): Batch size for loading data.
        feature_extractor (DetrFeatureExtractor): Feature extractor for preprocessing images.
        mode (str): Either "train" or "val" for training or validation mode.

    Returns:
        DataLoader: Prepared DataLoader for the specified dataset.
    """
    try:
        if data_dir:
            from torchvision.datasets import CocoDetection
            assert mode in ["train", "val"], "Mode should be either 'train' or 'val'."
            
            ann_file = f'annotations/instances_{mode}2017.json'
            img_dir = f'{mode}2017'

            dataset = CocoDetection(
                root=f"{data_dir}/{img_dir}",
                annFile=f"{data_dir}/{ann_file}",
                transform=lambda x: feature_extractor(images=x, return_tensors="pt").pixel_values[0]
            )

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True if mode == "train" else False,
                collate_fn=lambda batch: tuple(zip(*batch))
            )

        else:
            # Use HuggingFace's `datasets` library if no custom data directory is provided
            dataset = load_dataset('coco', split=mode)
            dataset.set_format(type='torch', columns=['pixel_values', 'labels'])

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True if mode == "train" else False
            )

        logging.info(f"Dataloader for mode '{mode}' prepared successfully with batch size {batch_size}.")
        return dataloader

    except Exception as e:
        logging.error(f"Error preparing dataloader: {e}")
        raise

def fine_tune_peft_model(
    model: PeftModel,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer,
    scheduler,
    num_epochs: int,
    device: str = "cuda",
    mixed_precision: bool = True,
    checkpoint_dir: str = "checkpoints"
):
    """
    Fine-tune a PEFT model with a training and validation loop.

    Args:
        model (PeftModel): The model to be fine-tuned.
        train_dataloader (DataLoader): Dataloader for training data.
        val_dataloader (DataLoader): Dataloader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for fine-tuning.
        scheduler (torch.optim.lr_scheduler): Scheduler for learning rate adjustment.
        num_epochs (int): Number of fine-tuning epochs.
        device (str): Device for training ('cuda' or 'cpu').
        mixed_precision (bool): Whether to use mixed precision for faster training.
        checkpoint_dir (str): Directory to save checkpoints.
    """
    model.to(device)
    scaler = GradScaler() if mixed_precision else None
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        logging.info(f"Starting Epoch {epoch + 1}/{num_epochs}")
        
        # Training loop
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()

            pixel_values, target = batch
            pixel_values = pixel_values.to(device)
            target = [{k: v.to(device) for k, v in t.items()} for t in target]

            with autocast(enabled=mixed_precision):
                outputs = model(pixel_values=pixel_values, labels=target)
                loss = outputs.loss

            if mixed_precision:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Validating Epoch {epoch + 1}/{num_epochs}"):
                pixel_values, target = batch
                pixel_values = pixel_values.to(device)
                target = [{k: v.to(device) for k, v in t.items()} for t in target]

                outputs = model(pixel_values=pixel_values, labels=target)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"Model checkpoint saved at {checkpoint_path}")
