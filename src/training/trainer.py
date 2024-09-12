# src/training.trainer.py

import os
import torch
import logging
from transformers import AdamW, get_scheduler, DetrFeatureExtractor
from src.models.foundation_model import HuggingFaceObjectDetectionModel
from src.evaluation.evaluator import Evaluator
from src.training.loss_functions import compute_loss
from src.training.peft_finetune import prepare_dataloader
from peft import PeftConfig

def setup_logging(log_file="training.log", log_level=logging.INFO):
    """
    Set up logging to a file and the console.
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir="checkpoints", best_model=False):
    """
    Save model checkpoint.

    Args:
        model: The trained model.
        optimizer: The optimizer used for training.
        scheduler: The learning rate scheduler.
        epoch: The current epoch number.
        checkpoint_dir: Directory to save model checkpoints.
        best_model: Boolean flag indicating if this is the best model so far.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pt")
    if best_model:
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Model checkpoint saved at {checkpoint_path}")

def train_epoch(model, dataloader, optimizer, scheduler, device, gradient_clipping=None):
    """
    Train the model for one epoch.

    Args:
        model: The object detection model.
        dataloader: DataLoader for the training data.
        optimizer: Optimizer used for training.
        scheduler: Learning rate scheduler.
        device: Device to run the training on ('cuda' or 'cpu').
        gradient_clipping: Value for gradient clipping (optional).

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()

        pixel_values = batch['pixel_values'].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]

        outputs = model(pixel_values=pixel_values)
        loss = compute_loss(outputs, labels)

        loss["total_loss"].backward()

        if gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

        optimizer.step()
        scheduler.step()

        total_loss += loss["total_loss"].item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def validate_epoch(model, dataloader, device):
    """
    Validate the model on the validation set.

    Args:
        model: The trained model.
        dataloader: DataLoader containing validation data.
        device: Device to run validation on.

    Returns:
        dict: Evaluation metrics.
    """
    evaluator = Evaluator(model, device=device)
    metrics = evaluator.evaluate(dataloader)
    return metrics

def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, num_epochs=10, device='cuda', 
                checkpoint_dir="checkpoints", gradient_clipping=None, early_stopping_patience=None):
    """
    Train and validate the object detection model.

    Args:
        model: The object detection model to train.
        train_dataloader: DataLoader for training data.
        val_dataloader: DataLoader for validation data.
        optimizer: Optimizer for model parameters.
        scheduler: Learning rate scheduler.
        num_epochs: Number of training epochs.
        device: Device to use for training ('cuda' or 'cpu').
        checkpoint_dir: Directory to save model checkpoints.
        gradient_clipping: Value for gradient clipping (optional).
        early_stopping_patience: Number of epochs with no improvement before stopping (optional).
    """
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        logging.info(f"Starting Epoch {epoch + 1}/{num_epochs}")

        # Train for one epoch
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device, gradient_clipping)
        logging.info(f"Epoch {epoch + 1} - Training Loss: {train_loss}")

        # Validate the model
        val_metrics = validate_epoch(model, val_dataloader, device)
        val_loss = val_metrics.get("total_loss", 0.0)
        logging.info(f"Epoch {epoch + 1} - Validation Loss: {val_loss}")
        logging.info(f"Epoch {epoch + 1} - Validation Metrics: {val_metrics}")

        # Save checkpoint for the current epoch
        save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir)

        # Early stopping based on validation loss
        if early_stopping_patience:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir, best_model=True)
                logging.info(f"New best model saved at Epoch {epoch + 1}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    logging.info(f"Early stopping triggered after {epoch + 1} epochs with no improvement.")
                    break
    logging.info("Training completed.")

def validate_model(model, dataloader, device):
    """
    Validate the model on the validation set.

    Args:
        model: The trained model.
        dataloader: DataLoader containing validation data.
        device: The device to run validation on.

    Returns:
        dict: Validation metrics.
    """
    return validate_epoch(model, dataloader, device)


if __name__ == "__main__":
    # Define constants
    DATA_DIR = "/path/to/coco/dataset"  # Replace with your dataset path
    BATCH_SIZE = 4
    NUM_CLASSES = 91  # For COCO dataset
    NUM_EPOCHS = 5
    LEARNING_RATE = 5e-5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CHECKPOINT_DIR = "output/checkpoints"
    MODEL_NAME = "facebook/detr-resnet-50"

    # PEFT Configuration
    peft_config = PeftConfig.from_pretrained(MODEL_NAME)

    # Set up logging
    setup_logging()

    # Initialize feature extractor, model, optimizer, and scheduler
    feature_extractor = DetrFeatureExtractor.from_pretrained(MODEL_NAME)
    model = HuggingFaceObjectDetectionModel(MODEL_NAME, num_classes=NUM_CLASSES)

    optimizer = AdamW(model.model.parameters(), lr=LEARNING_RATE)
    train_loader = prepare_dataloader(DATA_DIR, BATCH_SIZE, feature_extractor, mode="train")
    val_loader = prepare_dataloader(DATA_DIR, BATCH_SIZE, feature_extractor, mode="val")

    num_training_steps = NUM_EPOCHS * len(train_loader)
    scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Train the model with early stopping and gradient clipping
    train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        scheduler, 
        num_epochs=NUM_EPOCHS, 
        device=DEVICE, 
        checkpoint_dir=CHECKPOINT_DIR,
        gradient_clipping=1.0,  # Optional: enable gradient clipping
        early_stopping_patience=3  # Optional: enable early stopping
    )

    # Validate the model on the validation dataset
    validate_model(model, val_loader, device=DEVICE)

    # Save the final model
    model.save("output/detr_model")
    logging.info("Final model saved.")
