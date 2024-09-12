# scripts/train_model.py

# scripts/train_model.py

import os
import argparse
import logging
import torch
from transformers import AdamW, get_scheduler, DetrFeatureExtractor
from transformers import Trainer, TrainingArguments
from peft import PeftConfig, get_peft_model
from src.training.trainer import train_model
from src.training.peft_finetune import setup_peft_model, prepare_dataloader
from src.utils.config_parser import ConfigParser
from src.evaluation.evaluator import Evaluator
from src.utils.logging import setup_logging
from src.utils.system_utils import check_device
from torch.cuda.amp import GradScaler
import random

def parse_arguments():
    """
    Parse command line arguments for model training.
    """
    parser = argparse.ArgumentParser(description="Train an object detection model with HuggingFace and PEFT.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training configuration file (YAML or JSON).")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume training from.")
    parser.add_argument("--validate_every", type=int, default=1, help="Run validation every N epochs.")
    parser.add_argument("--use_trainer", action='store_true', help="Use HuggingFace Trainer API for training.")
    parser.add_argument("--early_stopping", type=int, default=0, help="Stop training if no improvement after N epochs (0 to disable).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients.")
    parser.add_argument("--lr_finder", action='store_true', help="Enable automatic learning rate finder.")
    parser.add_argument("--distributed", action='store_true', help="Enable distributed training.")
    return parser.parse_args()

def load_config(config_path):
    """
    Load configuration using the ConfigParser from src.utils.
    
    Args:
        config_path (str): Path to the configuration file.
    
    Returns:
        dict: Parsed configuration.
    """
    config_parser = ConfigParser(config_path)
    return config_parser.config

def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir):
    """
    Save model checkpoint.
    
    Args:
        model: The trained model.
        optimizer: The optimizer used for training.
        scheduler: The learning rate scheduler.
        epoch: The current epoch number.
        checkpoint_dir: Directory to save the checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch
    }
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint.
        model: The model to load the state into.
        optimizer: The optimizer to load the state into.
        scheduler: The scheduler to load the state into.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist.")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    logging.info(f"Resumed training from checkpoint {checkpoint_path} at epoch {start_epoch}")
    return start_epoch

def auto_lr_finder(optimizer, model, dataloader, config):
    """
    Automatically find the optimal learning rate.
    """
    logging.info("Starting learning rate finder...")
    best_lr = 1e-3  # Placeholder; need to implement logic
    config["learning_rate"] = best_lr
    logging.info(f"Found optimal learning rate: {best_lr}")

def validate_model(model, dataloader, device):
    """
    Validate the model using the Evaluator class.

    Args:
        model: The object detection model to validate.
        dataloader: The DataLoader containing validation data.
        device: The device to run validation on.
    
    Returns:
        dict: The evaluation metrics.
    """
    evaluator = Evaluator(model, device=device)
    metrics = evaluator.evaluate(dataloader)
    return metrics

def main():
    args = parse_arguments()

    # Load the configuration file
    config = load_config(args.config)

    # Set up logging
    setup_logging(config.get("log_file", "training.log"), log_level=logging.INFO)

    # Device setup
    device = check_device()

    # Load the feature extractor and dataset
    feature_extractor = DetrFeatureExtractor.from_pretrained(config["model_name"]) # Needs to be generalized
    train_loader = prepare_dataloader(config["data_dir"], config["batch_size"], feature_extractor, mode="train")
    val_loader = prepare_dataloader(config["data_dir"], config["batch_size"], feature_extractor, mode="val")

    # Set up PEFT configuration if provided
    if "peft_config" in config:
        logging.info("PEFT configuration detected. Setting up PEFT model...")
        peft_config = PeftConfig.from_pretrained(config["peft_config"])
        model = setup_peft_model(config["model_name"], config["num_classes"], peft_config)
    else:
        logging.info("No PEFT configuration provided. Using HuggingFaceObjectDetectionModel.")
        model = HuggingFaceObjectDetectionModel(config["model_name"], num_classes=config["num_classes"])

    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"]) # Add other optimizer options
    scheduler = get_scheduler(
        name=config.get("scheduler_type", "linear"),
        optimizer=optimizer,
        num_warmup_steps=config.get("num_warmup_steps", 0),
        num_training_steps=config["num_epochs"] * len(train_loader)
    )

    # Enable learning rate finder if requested
    if args.lr_finder:
        auto_lr_finder(optimizer, model, train_loader, config)

    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)

    # Automatic mixed precision (AMP)
    scaler = GradScaler()

    # If using HuggingFace Trainer API
    if args.use_trainer:
        training_args = TrainingArguments(
            output_dir=config["output_dir"],
            evaluation_strategy="epoch",
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            num_train_epochs=config["num_epochs"],
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            save_total_limit=2,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            report_to="tensorboard",
            push_to_hub=False,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_loader.dataset,
            eval_dataset=val_loader.dataset,
            tokenizer=feature_extractor,
            compute_metrics=None,  # Add custom compute_metrics if needed
        )

        if args.early_stopping > 0:
            trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=args.early_stopping))

        trainer.train()
    else:
        # Manual training loop with mixed precision
        for epoch in range(start_epoch, config["num_epochs"]):
            logging.info(f"Starting epoch {epoch + 1}/{config['num_epochs']}")
            train_model(model, train_loader, optimizer, scheduler, scaler, device=device)

            # Save checkpoint
            if (epoch + 1) % config.get("save_every", 1) == 0:
                save_checkpoint(model, optimizer, scheduler, epoch + 1, config["checkpoint_dir"])

            # Periodic validation
            if (epoch + 1) % args.validate_every == 0:
                logging.info("Evaluating the model on validation data...")
                metrics = validate_model(model, val_loader, device=device)
                logging.info(f"Validation metrics: {metrics}")

    # Save the final model
    output_model_path = os.path.join(config["output_dir"], "final_model")
    model.save(output_model_path)
    logging.info(f"Final model saved to {output_model_path}")

if __name__ == "__main__":
    main()

