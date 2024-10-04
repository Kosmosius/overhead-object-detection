# src/models/zoo/pvtv2.py

import os
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
from transformers import PvtV2Config, PvtV2Model, AutoImageProcessor

class PvtV2Backbone(nn.Module):
    """
    PvtV2 Backbone for Object Detection.

    This class wraps the PvtV2Model from Hugging Face Transformers, allowing it to be used as a backbone in object detection architectures.
    """

    def __init__(
        self,
        model_name: str = "OpenGVLab/pvt_v2_b0",
        pretrained: bool = True,
        linear_attention: bool = False,
        output_features: Optional[List[str]] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Initializes the PvtV2Backbone.

        Args:
            model_name (str): Name or path of the pre-trained PVTv2 model.
            pretrained (bool): Whether to load pre-trained weights.
            linear_attention (bool): Whether to use linear attention.
            output_features (list, optional): List of feature maps to output. Defaults to ['stage1', 'stage2', 'stage3', 'stage4'].
            device (str or torch.device): Device to load the model on.
        """
        super(PvtV2Backbone, self).__init__()

        # Load configuration with the option for linear attention
        self.config = PvtV2Config.from_pretrained(
            model_name,
            linear_attention=linear_attention
        )

        # Load the PvtV2Model
        self.model = PvtV2Model.from_pretrained(
            model_name,
            config=self.config,
            local_files_only=not pretrained  # Ensures models are loaded locally for air-gapped environments
        )
        
        # Move model to the specified device
        self.model.to(device)
        self.model.eval()  # Set to evaluation mode

        # Define output features
        if output_features is None:
            # Default to all stages if not specified
            self.output_features = ['stage1', 'stage2', 'stage3', 'stage4']
        else:
            self.output_features = output_features

    def forward(self, pixel_values: torch.Tensor) -> dict:
        """
        Forward pass through the PvtV2 backbone.

        Args:
            pixel_values (torch.Tensor): Input images of shape (batch_size, num_channels, height, width).

        Returns:
            dict: A dictionary containing the output feature maps.
        """
        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                output_hidden_states=True
            )

        # Extract the hidden states corresponding to the output features
        feature_maps = {}
        # Assuming hidden_states[0] is the embeddings, hidden_states[1] to hidden_states[4] correspond to stages 1-4
        for idx, feature in enumerate(self.output_features, 1):
            feature_maps[feature] = outputs.hidden_states[idx]

        return feature_maps

def get_pvtv2_backbone(
    model_size: str = "b0",
    pretrained: bool = True,
    linear_attention: bool = False,
    output_features: Optional[List[str]] = None,
    device: Union[str, torch.device] = "cpu",
) -> PvtV2Backbone:
    """
    Utility function to get a pre-configured PvtV2Backbone.

    Args:
        model_size (str): Size of the PVTv2 model (e.g., 'b0', 'b1', 'b2', etc.).
        pretrained (bool): Whether to load pre-trained weights.
        linear_attention (bool): Whether to use linear attention.
        output_features (list, optional): List of feature maps to output.
        device (str or torch.device): Device to load the model on.

    Returns:
        PvtV2Backbone: An instance of the PvtV2Backbone.
    """
    model_map = {
        "b0": "OpenGVLab/pvt_v2_b0",
        "b1": "OpenGVLab/pvt_v2_b1",
        "b2": "OpenGVLab/pvt_v2_b2",
        "b3": "OpenGVLab/pvt_v2_b3",
        "b4": "OpenGVLab/pvt_v2_b4",
        "b5": "OpenGVLab/pvt_v2_b5",
    }

    model_name = model_map.get(model_size.lower())
    if model_name is None:
        raise ValueError(f"Invalid model_size '{model_size}'. Expected one of {list(model_map.keys())}.")

    backbone = PvtV2Backbone(
        model_name=model_name,
        pretrained=pretrained,
        linear_attention=linear_attention,
        output_features=output_features,
        device=device,
    )

    return backbone

if __name__ == "__main__":
    # Example usage
    from PIL import Image
    import requests

    # Initialize backbone
    backbone = get_pvtv2_backbone(
        model_size="b0",
        pretrained=True,
        linear_attention=False,
        output_features=['stage1', 'stage2', 'stage3', 'stage4'],
        device="cpu"
    )

    # Load and preprocess image
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    image_processor = AutoImageProcessor.from_pretrained("OpenGVLab/pvt_v2_b0")
    processed = image_processor(image, return_tensors="pt")

    # Forward pass
    features = backbone(processed["pixel_values"])

    # Print feature map shapes
    for name, feature in features.items():
        print(f"{name}: {feature.shape}")
