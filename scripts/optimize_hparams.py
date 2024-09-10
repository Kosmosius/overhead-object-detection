# scripts/optimize_hparams.py

import argparse
import torch
from transformers import TrainingArguments, Trainer, HfArgumentParser
from src.models.foundation_model import HuggingFaceObjectDetectionModel
from src.data.dataloader import get_dataloader
from src.utils.config_parser import ConfigParser
from src.training.loss_functions import get_loss_function
from src.evaluation.evaluator import Evaluator
from src.utils.logging import setup_logging
from transformers import HfArgumentParser
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.huggingface import HuggingFaceTrainer
from ray.tune.search.hyperopt import HyperOptSearch

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
    # Ray Tune search algorithm and scheduler
    hyperopt_search = HyperOptSearch()
    scheduler = ASHAScheduler(metric="eval_loss", mode="min")

    # Create a HuggingFaceTrainer instance for Ray Tune
    tune_trainer = HuggingFaceTrainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataloader.dataset,
        eval_dataset=val_dataloader.dataset,
        compute_metrics=lambda p: run_evaluation(model, val_dataloader, p),
        tokenizer=None,
    )

    # Run Ray Tune hyperparameter search
    analysis = tune.run(
        tune.with_parameters(tune_trainer),
        config=search_space,
        metric="eval_loss",
        mode="min",
        search_alg=hyperopt_search,
        scheduler=scheduler,
        num_samples=10,  # Number of hyperparameter configurations to try
    )

    best_trial = analysis.get_best_trial(metric="eval_loss", mode="min")
    print(f"Best hyperparameters: {best_trial.config}")

def run_evaluation(model, val_dataloader, predictions):
    """
    Evaluate model predictions using the Evaluator class.

    Args:
        model (HuggingFaceObjectDetectionModel): HuggingFace model instance.
        val_dataloader (DataLoader): Validation DataLoader.
        predictions (dict): Model predictions.

    Returns:
        dict: Evaluation metrics (e.g., mAP).
    """
    evaluator = Evaluator(model, val_dataloader, device="cuda")  # Assuming 'cuda' device
    return evaluator.evaluate(predictions)

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
    )

    # Define the search space for hyperparameter tuning
    search_space = {
        "learning_rate": tune.loguniform(1e-5, 5e-4),
        "num_train_epochs": tune.choice([3, 5, 10]),
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
