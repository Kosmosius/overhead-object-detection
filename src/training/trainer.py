# src/training.trainer.py

import os
import torch
import logging
from transformers import AdamW, get_scheduler, DetrFeatureExtractor
from src.models.foundation_model import HuggingFaceObjectDetectionModel
from src.evaluation.evaluator import Evaluator
from src.training.loss_functions import compute_loss
from src.training.peft_finetune import setup_peft_model, prepare_dataloader
from peft import PeftConfig

def setup_logging(log_file="training.log"):
    """
    Set up logging to a file and the console.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir="checkpoints"):
    """
    Save model checkpoint.
    
    Args:
        model: The trained model.
        optimizer: The optimizer used for training.
        scheduler: The learning rate scheduler.
        epoch: The current epoch number.
        checkpoint_dir: Directory to save model checkpoints.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pt")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Model checkpoint saved at {checkpoint_path}")


def train_model(model, dataloader, optimizer, scheduler, num_epochs=10, device='cuda', checkpoint_dir="checkpoints"):
    """
    Train the object detection model.
    
    Args:
        model (HuggingFaceObjectDetectionModel): The object detection model to train.
        dataloader (DataLoader): The training DataLoader.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        num_epochs (int): Number of training epochs.
        device (str): The device to use for training ('cuda' or 'cpu').
        checkpoint_dir (str): Directory to save model checkpoints.
    """
    model.train()
    model.to(device)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        logging.info(f"Starting Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # Move batch to device
            pixel_values = batch['pixel_values'].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]

            # Forward pass and loss computation
            outputs = model.forward(pixel_values=pixel_values)
            loss = compute_loss(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if batch_idx % 10 == 0:
                logging.info(f"  Batch {batch_idx}, Loss: {loss.item()}")

        epoch_loss = running_loss / len(dataloader)
        logging.info(f"Epoch {epoch + 1} completed with loss: {epoch_loss}")

        # Save checkpoint after each epoch
        save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir)

    logging.info("Training completed.")


def validate_model(model, dataloader, device):
    """
    Validate the model on the validation set.
    
    Args:
        model: The trained model.
        dataloader: The DataLoader containing validation data.
        device: The device to run validation on.
    """
    evaluator = Evaluator(model, device=device)
    metrics = evaluator.evaluate(dataloader)
    logging.info(f"Validation results: {metrics}")
    return metrics


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
    scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=NUM_EPOCHS * len(train_loader))

    # Train the model
    train_model(model, train_loader, optimizer, scheduler, num_epochs=NUM_EPOCHS, device=DEVICE, checkpoint_dir=CHECKPOINT_DIR)

    # Load validation dataloader and evaluate the model
    val_loader = prepare_dataloader(DATA_DIR, BATCH_SIZE, feature_extractor, mode="val")
    validate_model(model, val_loader, device=DEVICE)

    # Save the final model
    model.save("output/detr_model")
    logging.info("Final model saved.")
