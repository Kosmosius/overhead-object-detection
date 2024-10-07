# src/models/zoo/base_model.py

import os
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Type, List

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from transformers import (
    PreTrainedModel,
    AutoConfig,
    AutoModel,
    AutoImageProcessor,
    get_linear_schedule_with_warmup,
)

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class BaseModel(ABC, nn.Module):
    """
    An abstract base class for object detection models in the model zoo.
    This class defines common interfaces and utilities for initializing,
    training, and evaluating models using the Hugging Face Transformers library.
    """

    def __init__(
        self,
        model_name_or_path: str,
        model_class: Optional[Type[PreTrainedModel]] = None,
        config: Optional[Union[Dict[str, Any], str]] = None,
        num_labels: Optional[int] = None,
        **kwargs,
    ):
        """
        Initializes the base model with the given configuration.

        Args:
            model_name_or_path (str): Path to the pretrained model or model identifier from Hugging Face.
            model_class (Optional[Type[PreTrainedModel]]): Specific model class to instantiate.
            config (Union[Dict[str, Any], str], optional): Configuration dictionary or path to config file.
            num_labels (int, optional): Number of labels for object detection.
            **kwargs: Additional keyword arguments for model configuration.
        """
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.model_class = model_class
        self.num_labels = num_labels
        self.load_kwargs = kwargs

        # Load configuration
        if config is None:
            self.config = AutoConfig.from_pretrained(
                self.model_name_or_path,
                num_labels=self.num_labels,
                **kwargs,
            )
            self.logger = get_logger(f"{self.__class__.__name__}.config")
        elif isinstance(config, dict):
            self.config = AutoConfig.from_dict(config)
            self.logger = get_logger(f"{self.__class__.__name__}.config")
        else:
            self.config = AutoConfig.from_pretrained(
                config,
                num_labels=self.num_labels,
                **kwargs,
            )
            self.logger = get_logger(f"{self.__class__.__name__}.config")

        # Initialize the model
        self.model = self._load_model(self.model_class)

        # Initialize the feature extractor
        self.feature_extractor = self._load_feature_extractor()

    def _load_model(self, model_class: Optional[Type[PreTrainedModel]] = None) -> PreTrainedModel:
        """
        Loads a pretrained model from Hugging Face.

        Args:
            model_class (Optional[Type[PreTrainedModel]]): Specific model class to instantiate.

        Returns:
            PreTrainedModel: The loaded model.
        """
        if model_class is None:
            model_class = AutoModel
        try:
            model = model_class.from_pretrained(
                self.model_name_or_path,
                config=self.config,
                **self.load_kwargs,
            )
            self.logger.info(f"Loaded model from {self.model_name_or_path} using {model_class.__name__}")
            return model
        except EnvironmentError as e:
            self.logger.error(f"Environment error during model loading: {e}")
            raise
        except ValueError as e:
            self.logger.error(f"Value error during model loading: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during model loading: {e}")
            raise

    def _load_feature_extractor(self) -> Union[AutoImageProcessor, Any]:
        """
        Loads the appropriate feature extractor or image processor.

        Returns:
            AutoImageProcessor or AutoFeatureExtractor: The loaded processor.
        """
        try:
            processor = AutoImageProcessor.from_pretrained(self.model_name_or_path)
            self.logger.info("Loaded AutoImageProcessor")
        except ValueError:
            processor = AutoImageProcessor.from_pretrained(
                self.model_name_or_path, 
                trust_remote_code=True  # If necessary for custom processors
            )
            self.logger.info("Loaded AutoImageProcessor as fallback")
        return processor

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Defines the forward pass of the model.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def compute_loss(self, outputs, targets) -> torch.Tensor:
        """
        Computes the loss given model outputs and targets.
        Must be implemented by subclasses.

        Args:
            outputs: Outputs from the model.
            targets: Ground truth targets.

        Returns:
            torch.Tensor: The computed loss.
        """
        pass

    def save(self, save_directory: str):
        """
        Saves the model and configuration to the specified directory.

        Args:
            save_directory (str): Directory to save the model and config.
        """
        os.makedirs(save_directory, exist_ok=True)
        try:
            self.model.save_pretrained(save_directory)
            self.config.save_pretrained(save_directory)
            self.feature_extractor.save_pretrained(save_directory)
            self.logger.info(f"Model, config, and feature extractor saved to {save_directory}")
        except Exception as e:
            self.logger.error(f"Failed to save model components: {e}")
            raise

    @classmethod
    def load(cls, load_directory: str, model_class: Optional[Type[PreTrainedModel]] = None, **kwargs):
        """
        Loads the model and configuration from the specified directory.

        Args:
            load_directory (str): Directory from which to load the model and config.
            model_class (Optional[Type[PreTrainedModel]]): Specific model class to instantiate.

        Returns:
            BaseModel: An instance of the model.
        """
        return cls(
            model_name_or_path=load_directory,
            model_class=model_class,
            config=load_directory,
            **kwargs,
        )

    def to_device(self, device: Union[str, torch.device]):
        """
        Moves the model to the specified device.

        Args:
            device (Union[str, torch.device]): The device to move the model to.

        Returns:
            BaseModel: The model on the specified device.
        """
        self.model.to(device)
        self.feature_extractor.to(device)
        self.logger.info(f"Moved model and feature extractor to {device}")
        return self

    def train_model(self, mode: bool = True):
        """
        Sets the model to training or evaluation mode.

        Args:
            mode (bool): True for training mode, False for evaluation mode.

        Returns:
            BaseModel: The model in the specified mode.
        """
        self.model.train(mode)
        self.feature_extractor.train(mode)
        self.logger.info(f"Set model and feature extractor to {'train' if mode else 'eval'} mode")
        return self

    def get_optimizer(
        self,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        """
        Creates an optimizer for the model parameters.

        Args:
            optimizer_class (Type[Optimizer], optional): Optimizer class to use. Defaults to torch.optim.AdamW.
            optimizer_params (Optional[Dict[str, Any]], optional): Parameters for the optimizer. Defaults to None.

        Returns:
            Optimizer: An instance of the optimizer.
        """
        optimizer_params = optimizer_params or {"lr": 1e-4, "weight_decay": 0.01}
        optimizer = optimizer_class(self.parameters(), **optimizer_params)
        self.logger.info(f"Initialized optimizer: {optimizer_class.__name__} with params: {optimizer_params}")
        return optimizer

    def get_scheduler(
        self,
        optimizer: Optimizer,
        scheduler_class: Type[_LRScheduler] = get_linear_schedule_with_warmup,
        scheduler_params: Optional[Dict[str, Any]] = None
    ) -> _LRScheduler:
        """
        Creates a learning rate scheduler.

        Args:
            optimizer (Optimizer): The optimizer for which to schedule the learning rate.
            scheduler_class (Type[_LRScheduler], optional): Scheduler class to use. Defaults to get_linear_schedule_with_warmup.
            scheduler_params (Optional[Dict[str, Any]], optional): Parameters for the scheduler. Defaults to None.

        Returns:
            _LRScheduler: A learning rate scheduler instance.
        """
        scheduler_params = scheduler_params or {}
        scheduler = scheduler_class(optimizer, **scheduler_params)
        self.logger.info(f"Initialized scheduler: {scheduler_class.__name__} with params: {scheduler_params}")
        return scheduler

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Prepares inputs for the model, including preprocessing and moving to device.

        Args:
            inputs (Dict[str, Any]): A dictionary of inputs.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of processed inputs.
        """
        device = next(self.model.parameters()).device
        processed_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        self.logger.debug(f"Prepared inputs moved to {device}")
        return processed_inputs

    def training_step(
        self, 
        batch: Dict[str, Any], 
        optimizer: Optimizer, 
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        use_amp: bool = False
    ) -> float:
        """
        Performs a single training step.

        Args:
            batch (Dict[str, Any]): A batch of data.
            optimizer (Optimizer): The optimizer to use.
            scaler (Optional[torch.cuda.amp.GradScaler], optional): Gradient scaler for mixed precision. Defaults to None.
            use_amp (bool, optional): Whether to use mixed precision. Defaults to False.

        Returns:
            float: The loss value for this step.
        """
        self.train_model(True)
        inputs = self.prepare_inputs(batch)
        optimizer.zero_grad()
        
        if scaler and use_amp:
            with torch.cuda.amp.autocast():
                outputs = self(**inputs)
                loss = self.compute_loss(outputs, inputs.get('labels'))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = self(**inputs)
            loss = self.compute_loss(outputs, inputs.get('labels'))
            loss.backward()
            optimizer.step()
        
        loss_value = loss.item()
        self.logger.debug(f"Training step completed with loss: {loss_value}")
        return loss_value

    def evaluation_step(self, batch: Dict[str, Any]) -> Any:
        """
        Performs a single evaluation step.

        Args:
            batch (Dict[str, Any]): A batch of data.

        Returns:
            Any: The outputs from the model.
        """
        self.train_model(False)
        inputs = self.prepare_inputs(batch)
        with torch.no_grad():
            outputs = self(**inputs)
        self.logger.debug("Evaluation step completed")
        return outputs

    def fit(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 1,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params: Optional[Dict[str, Any]] = None,
        scheduler_class: Type[_LRScheduler] = get_linear_schedule_with_warmup,
        scheduler_params: Optional[Dict[str, Any]]
