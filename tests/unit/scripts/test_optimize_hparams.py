# tests/unit/scripts/test_optimize_hparams.py

import pytest
from unittest import mock
import torch
from ray import tune
from transformers import TrainingArguments
from src.models.foundation_model import HuggingFaceObjectDetectionModel
from scripts.optimize_hparams import hyperparameter_search, main

# Mock configuration parser
@mock.patch("src.utils.config_parser.ConfigParser")
@mock.patch("src.models.foundation_model.HuggingFaceObjectDetectionModel.load")
def test_main_hyperparameter_optimization(mock_load_model, mock_config_parser):
    """Test the main function, ensuring the correct initialization of hyperparameter optimization."""

    # Mock configuration return values
    mock_config_parser.return_value.get.side_effect = lambda key: {
        "model_name": "facebook/detr-resnet-50",
        "num_classes": 80
    }.get(key, None)

    # Mock model loading
    mock_model = mock.Mock(spec=HuggingFaceObjectDetectionModel)
    mock_load_model.return_value = mock_model

    # Mock dataloader preparation
    with mock.patch("src.data.dataloader.get_dataloader") as mock_dataloader:
        # Mock dataloader objects
        train_loader = mock.Mock()
        val_loader = mock.Mock()
        mock_dataloader.side_effect = [train_loader, val_loader]

        # Run the main function
        main()

        # Assertions
        mock_config_parser.assert_called_once()  # Ensure config was loaded
        mock_load_model.assert_called_once()  # Ensure model was loaded
        mock_dataloader.assert_any_call(data_dir=mock.ANY, batch_size=mock.ANY, mode="train", feature_extractor=mock.ANY)
        mock_dataloader.assert_any_call(data_dir=mock.ANY, batch_size=mock.ANY, mode="val", feature_extractor=mock.ANY)


# Core test: Hyperparameter search with valid inputs
@mock.patch("ray.tune.run")
@mock.patch("ray.tune.integration.huggingface.HuggingFaceTrainer")
@mock.patch("src.evaluation.evaluator.evaluate_model")
def test_hyperparameter_search(mock_evaluate_model, mock_hf_trainer, mock_tune_run):
    """Test that hyperparameter search runs correctly with mocked Ray Tune and HuggingFace Trainer."""
    
    # Mock model, data, and training args
    mock_model = mock.Mock()
    mock_train_loader = mock.Mock()
    mock_val_loader = mock.Mock()
    training_args = TrainingArguments(output_dir="./results", evaluation_strategy="epoch")
    
    # Define a mock search space
    search_space = {
        "learning_rate": tune.loguniform(1e-5, 5e-4),
        "num_train_epochs": tune.choice([3, 5, 10]),
        "per_device_train_batch_size": tune.choice([4, 8, 16]),
        "weight_decay": tune.uniform(0.01, 0.1),
    }

    # Run hyperparameter search
    hyperparameter_search(mock_model, mock_train_loader, mock_val_loader, training_args, search_space)

    # Assertions: ensure HuggingFaceTrainer was called and search was executed
    mock_hf_trainer.assert_called_once()
    mock_tune_run.assert_called_once()
    mock_tune_run.assert_called_with(
        tune.with_parameters(mock_hf_trainer),
        config=search_space,
        metric="eval_loss",
        mode="min",
        search_alg=mock.ANY,
        scheduler=mock.ANY,
        num_samples=10,
    )


# Edge case: Invalid search space
@mock.patch("ray.tune.run")
def test_hyperparameter_search_invalid_search_space(mock_tune_run):
    """Test that hyperparameter search handles invalid search spaces properly."""
    
    # Mock model, data, and training args
    mock_model = mock.Mock()
    mock_train_loader = mock.Mock()
    mock_val_loader = mock.Mock()
    training_args = TrainingArguments(output_dir="./results", evaluation_strategy="epoch")

    # Define an invalid search space (non-existent hyperparameter)
    search_space = {
        "invalid_hyperparameter": tune.loguniform(1e-5, 5e-4),
    }

    # Expect ValueError when invalid hyperparameter is encountered
    with pytest.raises(ValueError, match="invalid_hyperparameter"):
        hyperparameter_search(mock_model, mock_train_loader, mock_val_loader, training_args, search_space)

    # Ensure Ray Tune wasn't called
    mock_tune_run.assert_not_called()


# Edge case: Failed trials
@mock.patch("ray.tune.run")
@mock.patch("ray.tune.integration.huggingface.HuggingFaceTrainer")
@mock.patch("src.evaluation.evaluator.evaluate_model")
def test_hyperparameter_search_failed_trials(mock_evaluate_model, mock_hf_trainer, mock_tune_run):
    """Test handling of failed trials during hyperparameter optimization."""
    
    # Mock model, data, and training args
    mock_model = mock.Mock()
    mock_train_loader = mock.Mock()
    mock_val_loader = mock.Mock()
    training_args = TrainingArguments(output_dir="./results", evaluation_strategy="epoch")

    # Mock failure in Ray Tune run
    mock_tune_run.side_effect = RuntimeError("Hyperparameter tuning failed")

    # Define a mock search space
    search_space = {
        "learning_rate": tune.loguniform(1e-5, 5e-4),
    }

    # Expect RuntimeError due to failed trials
    with pytest.raises(RuntimeError, match="Hyperparameter tuning failed"):
        hyperparameter_search(mock_model, mock_train_loader, mock_val_loader, training_args, search_space)


# Performance test: Parallel trials with Ray Tune
@mock.patch("ray.tune.run")
@mock.patch("ray.tune.integration.huggingface.HuggingFaceTrainer")
def test_hyperparameter_search_parallel_trials(mock_hf_trainer, mock_tune_run, benchmark):
    """Test performance of hyperparameter search with parallel trials."""
    
    # Mock model, data, and training args
    mock_model = mock.Mock()
    mock_train_loader = mock.Mock()
    mock_val_loader = mock.Mock()
    training_args = TrainingArguments(output_dir="./results", evaluation_strategy="epoch")

    # Define a search space
    search_space = {
        "learning_rate": tune.loguniform(1e-5, 5e-4),
        "num_train_epochs": tune.choice([3, 5, 10]),
    }

    # Benchmark performance for parallel trials
    benchmark(
        hyperparameter_search,
        mock_model,
        mock_train_loader,
        mock_val_loader,
        training_args,
        search_space,
    )

    # Ensure Ray Tune was called
    mock_tune_run.assert_called_once()
