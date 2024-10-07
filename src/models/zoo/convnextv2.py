# src/models/zoo/convnextv2.py

import os
import logging
from typing import Optional, List, Dict, Any

import torch
from torch import nn
from transformers import (
    ConvNeXTV2Model,
    ConvNeXTV2Config,
    PreTrainedModel,
    AutoImageProcessor,
)
from transformers.modeling_outputs import ObjectDetectionOutput
from transformers.utils import logging as hf_logging

# Set up logging
logger = logging.getLogger(__name__)
hf_logging.set_verbosity_info()


class ConvNeXTV2ForObjectDetection(PreTrainedModel):
    """
    ConvNeXTV2 Model for Object Detection.

    This class integrates ConvNeXTV2 as the backbone with object detection heads.
    It leverages Hugging Face's Transformers library for configuration and model loading.
    """

    config_class = ConvNeXTV2Config

    def __init__(self, config: ConvNeXTV2Config):
        """
        Initialize the ConvNeXTV2ForObjectDetection model.

        Args:
            config (ConvNeXTV2Config): Configuration class with all the parameters of the model.
        """
        super().__init__(config)
        self.convnext = ConvNeXTV2Model(config)

        # Define object detection heads
        # Classification head: Predicts class scores for each spatial location
        self.classification_head = nn.Linear(config.hidden_sizes[-1], config.num_labels)

        # Regression head: Predicts bounding box coordinates [x, y, width, height]
        self.regression_head = nn.Linear(config.hidden_sizes[-1], 4)

        # Initialize weights
        self.init_weights()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[List[Dict[str, Any]]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> ObjectDetectionOutput:
        """
        Forward pass for object detection.

        Args:
            pixel_values (torch.FloatTensor): Pixel values of the input images.
            labels (List[Dict[str, Any]], optional): Ground truth annotations for computing loss.
            output_hidden_states (bool, optional): Whether to output hidden states.
            return_dict (bool, optional): Whether to return a dict instead of a tuple.

        Returns:
            ObjectDetectionOutput: The output of the model containing losses and predictions.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Extract features using ConvNeXTV2 backbone
        outputs = self.convnext(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, channels, height, width)

        # Reshape for detection heads: flatten spatial dimensions
        batch_size, channels, height, width = last_hidden_state.shape
        features = last_hidden_state.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        # Shape: (batch_size, num_tokens, channels)

        # Classification logits for each token
        logits = self.classification_head(features)  # Shape: (batch_size, num_tokens, num_labels)

        # Bounding box predictions for each token
        bboxes = self.regression_head(features)  # Shape: (batch_size, num_tokens, 4)

        loss = None
        if labels is not None:
            # Compute losses (classification and bounding box)
            classification_loss_fct = nn.CrossEntropyLoss()
            bbox_loss_fct = nn.L1Loss()

            # Example: Flatten tensors for loss computation
            # In practice, implement a matching algorithm like Hungarian matching
            # to align predictions with ground truth objects
            classification_targets = torch.stack([torch.tensor(item['labels']) for item in labels]).to(logits.device)
            bbox_targets = torch.stack([torch.tensor(item['boxes']) for item in labels]).to(bboxes.device)

            # Flatten tensors
            logits_flat = logits.view(-1, self.config.num_labels)
            classification_targets_flat = classification_targets.view(-1)
            bboxes_flat = bboxes.view(-1, 4)
            bbox_targets_flat = bbox_targets.view(-1, 4)

            # Compute losses
            classification_loss = classification_loss_fct(logits_flat, classification_targets_flat)
            bbox_loss = bbox_loss_fct(bboxes_flat, bbox_targets_flat)

            loss = classification_loss + bbox_loss

        if not return_dict:
            output = (logits, bboxes) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ObjectDetectionOutput(
            loss=loss,
            logits=logits,
            bboxes=bboxes,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ):
        """
        Load a pretrained ConvNeXTV2ForObjectDetection model from Hugging Face or local path.

        Args:
            pretrained_model_name_or_path (str): Name or path of the pretrained model.
            *model_args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            ConvNeXTV2ForObjectDetection: The loaded model.
        """
        config = ConvNeXTV2Config.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = cls(config)
        model.convnext = ConvNeXTV2Model.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.classification_head = nn.Linear(config.hidden_sizes[-1], config.num_labels)
        model.regression_head = nn.Linear(config.hidden_sizes[-1], 4)
        model.init_weights()
        return model


def load_convnextv2_for_object_detection(
    model_name: str,
    local_model_dir: Optional[str] = None,
    device: str = 'cpu'
) -> ConvNeXTV2ForObjectDetection:
    """
    Load a ConvNeXTV2ForObjectDetection model.

    This function facilitates loading the model from either Hugging Face's hub or a local directory,
    ensuring compatibility with air-gapped deployment environments.

    Args:
        model_name (str): Name or path of the ConvNeXTV2 model.
        local_model_dir (str, optional): Local directory to load the model from. Defaults to None.
        device (str, optional): Device to load the model on ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        ConvNeXTV2ForObjectDetection: The loaded model.
    """
    try:
        model_path = local_model_dir if local_model_dir else model_name
        model = ConvNeXTV2ForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=model_path,
            local_files_only=bool(local_model_dir),
        )
        model.to(device)
        model.eval()
        logger.info(f"ConvNeXTV2ForObjectDetection model loaded successfully on {device}")
        return model
    except Exception as e:
        logger.error(f"Error loading ConvNeXTV2ForObjectDetection model: {e}")
        raise e


def get_image_processor(model_name: str, local_model_dir: Optional[str] = None) -> AutoImageProcessor:
    """
    Load the image processor associated with the ConvNeXTV2 model.

    This ensures that images are preprocessed in the same way as the model expects.

    Args:
        model_name (str): Name or path of the ConvNeXTV2 model.
        local_model_dir (str, optional): Local directory to load the image processor from. Defaults to None.

    Returns:
        AutoImageProcessor: The image processor instance.
    """
    try:
        processor_path = local_model_dir if local_model_dir else model_name
        image_processor = AutoImageProcessor.from_pretrained(
            pretrained_model_name_or_path=processor_path,
            local_files_only=bool(local_model_dir),
        )
        logger.info(f"Image processor loaded successfully from {processor_path}")
        return image_processor
    except Exception as e:
        logger.error(f"Error loading image processor: {e}")
        raise e


# Example usage (This should be moved to a separate script or notebook)

if __name__ == "__main__":
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser(description="ConvNeXTV2 Object Detection Inference")
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the ConvNeXTV2 model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--local_model_dir", type=str, default=None, help="Local directory to load the model from")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on ('cpu' or 'cuda')")

    args = parser.parse_args()

    # Load the model
    model = load_convnextv2_for_object_detection(
        model_name=args.model_name,
        local_model_dir=args.local_model_dir,
        device=args.device
    )

    # Load the image processor
    image_processor = get_image_processor(
        model_name=args.model_name,
        local_model_dir=args.local_model_dir
    )

    # Load and preprocess the image
    image = Image.open(args.image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract predictions
    logits = outputs.logits  # Shape: (batch_size, num_tokens, num_labels)
    bboxes = outputs.bboxes  # Shape: (batch_size, num_tokens, 4)

    # Example: Get the top prediction
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]

    print(f"Predicted label: {predicted_label} (Index: {predicted_class_idx})")
    print(f"Bounding boxes: {bboxes}")
