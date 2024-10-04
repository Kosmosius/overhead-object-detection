# src/models/zoo/swin_transformer_v2.py

import torch
import torch.nn as nn
from transformers import Swinv2Model, Swinv2Config

class SwinTransformerV2(nn.Module):
    """
    Swin Transformer V2 model for overhead object detection.

    This class encapsulates the Swin Transformer V2 architecture, leveraging Hugging Face's 
    Transformers library. It provides functionalities to initialize the model with or without 
    pretrained weights, perform forward passes, and manage model parameters for fine-tuning.

    Args:
        config (dict): Configuration dictionary for the Swin Transformer V2 model.
        pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
        model_name_or_path (str, optional): Path to the pretrained model or model identifier 
                                            from Hugging Face's model hub. Defaults to 
                                            'microsoft/swinv2-tiny-patch4-window8-256'.
    """
    def __init__(self, config: dict, pretrained: bool = True, model_name_or_path: str = 'microsoft/swinv2-tiny-patch4-window8-256'):
        super(SwinTransformerV2, self).__init__()
        self.config = Swinv2Config(**config)
        if pretrained:
            self.model = Swinv2Model.from_pretrained(model_name_or_path, config=self.config)
        else:
            self.model = Swinv2Model(self.config)

    def forward(self, pixel_values: torch.Tensor, **kwargs):
        """
        Forward pass of the Swin Transformer V2 model.

        Args:
            pixel_values (torch.Tensor): Input images tensor of shape (batch_size, num_channels, height, width).
            **kwargs: Additional keyword arguments for the Swinv2Model forward method.

        Returns:
            transformers.models.swinv2.modeling_swinv2.Swinv2ModelOutput: Output from Swinv2Model.
        """
        return self.model(pixel_values=pixel_values, **kwargs)

    def get_last_hidden_state(self, pixel_values: torch.Tensor):
        """
        Retrieve the last hidden state from the Swin Transformer V2 model.

        Args:
            pixel_values (torch.Tensor): Input images tensor.

        Returns:
            torch.Tensor: Last hidden state tensor of shape (batch_size, sequence_length, hidden_size).
        """
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state

    def get_pooler_output(self, pixel_values: torch.Tensor):
        """
        Retrieve the pooler output from the Swin Transformer V2 model.

        Args:
            pixel_values (torch.Tensor): Input images tensor.

        Returns:
            torch.Tensor: Pooler output tensor of shape (batch_size, hidden_size).
        """
        outputs = self.model(pixel_values=pixel_values)
        return outputs.pooler_output

    def freeze_model(self):
        """
        Freeze all parameters of the Swin Transformer V2 model to prevent them from being updated during training.
        """
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_model(self):
        """
        Unfreeze all parameters of the Swin Transformer V2 model to allow them to be updated during training.
        """
        for param in self.model.parameters():
            param.requires_grad = True

    def load_local_pretrained(self, local_path: str):
        """
        Load pretrained weights from a local directory.

        Args:
            local_path (str): Path to the directory containing pretrained model weights.
        """
        self.model = Swinv2Model.from_pretrained(local_path, config=self.config, local_files_only=True)

    def save_model(self, save_path: str):
        """
        Save the Swin Transformer V2 model to a specified path.

        Args:
            save_path (str): Directory where the model will be saved.
        """
        self.model.save_pretrained(save_path)
        self.config.save_pretrained(save_path)

    def resize_position_embeddings(self, new_image_size: int, new_patch_size: int):
        """
        Resize the position embeddings to accommodate a new image size or patch size.

        Args:
            new_image_size (int): New image resolution.
            new_patch_size (int): New patch resolution.
        """
        self.model.resize_position_embeddings(new_image_size=new_image_size, new_patch_size=new_patch_size)

# Example Usage:
if __name__ == "__main__":
    from transformers import AutoImageProcessor

    # Sample configuration for Swin Transformer V2
    config = {
        "image_size": 224,
        "patch_size": 4,
        "num_channels": 3,
        "embed_dim": 96,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "window_size": 7,
        "pretrained_window_sizes": [0, 0, 0, 0],
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "drop_path_rate": 0.1,
        "hidden_act": "gelu",
        "use_absolute_embeddings": False,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-05,
        "encoder_stride": 32,
        "out_features": None,
        "out_indices": None
    }

    # Initialize the Swin Transformer V2 model
    model = SwinTransformerV2(config=config, pretrained=True)

    # Initialize the image processor
    image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")

    # Load a sample image (replace with actual image loading in practice)
    from PIL import Image
    import requests

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # Preprocess the image
    inputs = image_processor(images=image, return_tensors="pt")

    # Perform a forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Access the last hidden states
    last_hidden_states = outputs.last_hidden_state
    print(last_hidden_states.shape)  # Example output: torch.Size([1, 64, 768])

    # Access the pooler output
    pooler_output = model.get_pooler_output(inputs['pixel_values'])
    print(pooler_output.shape)  # Example output: torch.Size([1, 768])
