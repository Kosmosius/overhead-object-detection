# scripts/optimize_hparams.py

import argparse
import os
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from transformers import TrainingArguments, Trainer, DetrForObjectDetection, DetrFeatureExtractor
from src.models.foundation_model import HuggingFaceObjectDetectionModel
from src.data.dataloader import get_dataloader
from src.utils.config_parser import ConfigParser
from src.evaluation.evaluator import Evaluator
from src.utils.logging import setup_logging

def hyperparameter_search(model, train_dataloader, val_dataloader, training_args, search_space):
    """
    Conduct hyperparameter search using Ray Tune with HuggingFace's Trainer.

    Args:
        model (HuggingFaceObjectDetectionModel): HuggingFace model instance.
        train_dataloader (DataLoader): Training DataLoader.
        val_dataloader (DataLoader): Validation DataLoader.
        training_args (TrainingArguments): HuggingFace TrainingArguments for the model.
        search_space (dict): Dictionary defining the hyperparameter search space.
    """
    def model_init():
        return model.model  # Use your HuggingFace model initialization here

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataloader.dataset,
        eval_dataset=val_dataloader.dataset,
        compute_metrics=Evaluator.compute_metrics,  # Custom evaluation function
    )

    # Setup Ray Tune integration with HuggingFace
    tune_config = search_space

    # Ray Tune scheduler
    scheduler = ASHAScheduler(metric="eval_loss", mode="min")

    # CLI Reporter for better feedback in the console
    reporter = CLIReporter(
        parameter_columns=["learning_rate", "num_train_epochs", "per_device_train_batch_size"],
        metric_columns=["eval_loss", "eval_accuracy", "epoch", "training_iteration"]
    )

    # Run the hyperparameter search
    trainer.hyperparameter_search(
        hp_space=lambda _: tune_config,
        backend="ray",
        n_trials=10,  # Adjust the number of trials
        resources_per_trial={"cpu": 2, "gpu": 1},  # Adjust for your setup
        scheduler=scheduler,
        keep_checkpoints_num=1,
        progress_reporter=reporter,
        local_dir="./ray_results",  # Directory to store results
        log_to_file=True,
    )

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for object detection models.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run training on ('cuda' or 'cpu').")
    
    args = parser.parse_args()

    # Set up logging
    setup_logging(log_file="hyperparameter_optimization.log")

    # Load configuration
    config_parser = ConfigParser(args.config_path)
    model_name = config_parser.get("model_name")
    num_classes = config_parser.get("num_classes")

    # Load model
    model = HuggingFaceObjectDetectionModel(model_name=model_name, num_classes=num_classes)
    model.load(args.model_checkpoint)

    # Prepare dataloaders
    feature_extractor = DetrFeatureExtractor.from_pretrained(model_name)
    train_loader = get_dataloader(data_dir=args.data_dir, batch_size=args.batch_size, mode="train", feature_extractor=feature_extractor)
    val_loader = get_dataloader(data_dir=args.data_dir, batch_size=args.batch_size, mode="val", feature_extractor=feature_extractor)

    # Define training arguments using HuggingFace's TrainingArguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    # Define the search space for hyperparameter tuning
    search_space = {
        "learning_rate": tune.loguniform(1e-5, 5e-4),
        "num_train_epochs": tune.choice([2, 3, 5]),
        "per_device_train_batch_size": tune.choice([4, 8, 16]),
        "weight_decay": tune.uniform(0.01, 0.1),
    }

    # Run hyperparameter optimization
    hyperparameter_search(model, train_loader, val_loader, training_args, search_space)

if __name__ == "__main__":
    main()


"""
python scripts/optimize_hparams.py --model_checkpoint output/detr_model --config_path configs/training/default_training.yml --data_dir /path/to/data --batch_size 4 --num_epochs 5 --device cuda
"""
