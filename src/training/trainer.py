# src/training/trainer.py

import os
import torch
import logging
from transformers import get_scheduler
from src.models.model_factory import ModelFactory
from src.evaluation.evaluator import Evaluator
from src.training.loss_functions import compute_loss
from src.data.dataloader import get_dataloader
from src.utils.config_parser import ConfigParser
from src.utils.logging_utils import setup_logging
from src.training.optimizers import get_optimizer_and_scheduler

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    checkpoint_dir: str = "checkpoints",
    best_model: bool = False
) -> None:
    """
    Save model checkpoint.

    Args:
        model (torch.nn.Module): The trained model.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        epoch (int): The current epoch number.
        checkpoint_dir (str): Directory to save model checkpoints.
        best_model (bool): Boolean flag indicating if this is the best model so far.
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

def train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    gradient_clipping: float = None
) -> float:
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The object detection model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        device (torch.device): Device to run the training on ('cuda' or 'cpu').
        gradient_clipping (float): Value for gradient clipping (optional).

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    for images, targets in dataloader:
        optimizer.zero_grad()

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)
        loss_dict = compute_loss(outputs, targets, model, {})
        loss = loss_dict["total_loss"]

        loss.backward()

        if gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def validate_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate the model on the validation set.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader containing validation data.
        device (torch.device): Device to run validation on.

    Returns:
        Dict[str, float]: Evaluation metrics.
    """
    evaluator = Evaluator(model, device=device)
    metrics = evaluator.evaluate(dataloader)
    return metrics

def train_model(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: Dict[str, Any],
    device: torch.device,
    checkpoint_dir: str = "checkpoints"
) -> None:
    """
    Train and validate the object detection model.

    Args:
        model (torch.nn.Module): The object detection model to train.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        config (Dict[str, Any]): Configuration parameters.
        device (torch.device): Device to use for training ('cuda' or 'cpu').
        checkpoint_dir (str): Directory to save model checkpoints.
    """
    num_epochs = config['training']['num_epochs']
    gradient_clipping = config['training'].get('gradient_clipping', None)
    early_stopping_patience = config['training'].get('early_stopping_patience', None)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        logging.info(f"Starting Epoch {epoch + 1}/{num_epochs}")

        # Train for one epoch
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device, gradient_clipping)
        logging.info(f"Epoch {epoch + 1} - Training Loss: {train_loss}")

        # Validate the model
        val_metrics = validate_epoch(model, val_dataloader, device)
        val_loss = val_metrics.get("AP", 0.0)  # Assuming 'AP' is the primary metric
        logging.info(f"Epoch {epoch + 1} - Validation Metrics: {val_metrics}")

        # Save checkpoint for the current epoch
        save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir)

        # Early stopping based on validation loss
        if early_stopping_patience:
            if val_loss > best_val_loss:
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

if __name__ == "__main__":
    # Load configuration
    config_path = "configs/train_config.yaml"
    config_parser = ConfigParser(config_path)
    config = config_parser.config

    # Set up logging
    setup_logging(log_file=config['logging']['log_file'], log_level=config['logging']['log_level'])

    # Define constants from configuration
    data_config = config['data']
    model_config = config['model']
    training_config = config['training']
    optimizer_config = config['optimizer']
    scheduler_config = config['scheduler']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize feature extractor and model
    try:
        from transformers import DetrFeatureExtractor
        feature_extractor = DetrFeatureExtractor.from_pretrained(model_config['model_name'])
    except Exception as e:
        logging.error(f"Error loading feature extractor: {e}")
        raise

    model_instance = ModelFactory.create_model(
        model_type=model_config.get('model_type', 'detr'),
        model_name=model_config['model_name'],
        num_labels=model_config['num_classes']
    )
    model = model_instance.model.to(device)

    # Prepare dataloaders
    train_loader = get_dataloader(
        data_dir=data_config['data_dir'],
        batch_size=training_config['batch_size'],
        mode="train",
        feature_extractor=feature_extractor,
        num_workers=training_config.get('num_workers', 4),
        pin_memory=training_config.get('pin_memory', True)
    )

    val_loader = get_dataloader(
        data_dir=data_config['data_dir'],
        batch_size=training_config['batch_size'],
        mode="val",
        feature_extractor=feature_extractor,
        num_workers=training_config.get('num_workers', 4),
        pin_memory=training_config.get('pin_memory', True)
    )

    # Set up optimizer and scheduler
    num_training_steps = training_config['num_epochs'] * len(train_loader)
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model,
        config={**optimizer_config, **scheduler_config},
        num_training_steps=num_training_steps
    )

    # Train the model
    train_model(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        checkpoint_dir=training_config['checkpoint_dir']
    )

    # Save the final model
    model_save_path = os.path.join(training_config['output_dir'], 'final_model')
    model.save_pretrained(model_save_path)
    logging.info(f"Final model saved at {model_save_path}")
