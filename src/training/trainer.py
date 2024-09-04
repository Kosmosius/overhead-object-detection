# src/training/trainer.py

import torch
from torch.utils.data import DataLoader
from transformers import DetrFeatureExtractor, AdamW, get_scheduler
from src.models.foundation_model import DetrObjectDetectionModel
from src.evaluation.evaluator import evaluate_model
from src.training.loss_functions import compute_loss
from src.training.peft_finetune import setup_peft_model, prepare_dataloader
import os
import logging
from peft import PeftConfig

def setup_logging(log_file="training.log"):
    """
    Set up logging to a file and the console.
    !Check for compatibility with `src/utils/logging.py`.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def train(model, dataloader, optimizer, scheduler, num_epochs=10, device='cuda', checkpoint_dir="checkpoints"):
    """
    Train the object detection model (supports PEFT).

    Args:
        model (PeftModel/DetrObjectDetectionModel): The object detection model to train.
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
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # Move batch to device
            pixel_values = batch['pixel_values'].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]

            # Forward pass and loss computation
            outputs = model(pixel_values=pixel_values)
            loss = compute_loss(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if batch_idx % 10 == 0:
                logging.info(f"  Batch {batch_idx}, Loss: {loss.item()}")

        epoch_loss = running_loss / len(dataloader)
        logging.info(f"  Epoch Loss: {epoch_loss}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"detr_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"Model checkpoint saved at {checkpoint_path}")

    logging.info("Training complete.")

if __name__ == "__main__":
    # Define constants
    DATASET_NAME = "coco"
    DATA_DIR = "/path/to/coco/dataset"  # Replace with your dataset path
    BATCH_SIZE = 4
    NUM_CLASSES = 91  # For COCO dataset
    NUM_EPOCHS = 5
    LEARNING_RATE = 5e-5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CHECKPOINT_DIR = "output/checkpoints"

    # PEFT Configuration
    peft_config = PeftConfig.from_pretrained("facebook/detr-resnet-50")

    # Set up logging
    setup_logging()

    # Initialize feature extractor, model, optimizer, and scheduler
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    model = setup_peft_model("facebook/detr-resnet-50", num_classes=NUM_CLASSES, peft_config=peft_config)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Prepare the scheduler with the correct number of steps
    train_loader = prepare_dataloader(DATA_DIR, BATCH_SIZE, feature_extractor, mode="train")
    scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=NUM_EPOCHS * len(train_loader))

    # Train the model
    train(model, train_loader, optimizer, scheduler, num_epochs=NUM_EPOCHS, device=DEVICE, checkpoint_dir=CHECKPOINT_DIR)
    
    # Load validation dataloader and evaluate the model
    val_loader = prepare_dataloader(DATA_DIR, BATCH_SIZE, feature_extractor, mode="val")
    precision, recall = evaluate_model(model, val_loader, device=DEVICE)
    logging.info(f"Validation Precision: {precision}, Recall: {recall}")

    # Save the final trained model
    model.save_pretrained("output/detr_model")
    logging.info("Final model saved.")
