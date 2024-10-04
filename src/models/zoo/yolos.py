# src/models/zoo/yolos.py

import torch
from transformers import YolosForObjectDetection, YolosConfig, YolosImageProcessor
from typing import List, Dict, Tuple, Optional
from PIL import Image
import os

class YOLOS:
    """
    YOLOS (You Only Look at One Sequence) Object Detection Model Wrapper.

    This class provides an interface to initialize, load, and perform inference
    using the YOLOS model from Hugging Face's transformers library.
    """

    def __init__(
        self,
        model_name: str = "hustvl/yolos-base",
        model_path: Optional[str] = None,
        image_processor_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation: Optional[str] = "sdpa",
        torch_dtype: torch.dtype = torch.float16,
    ):
        """
        Initializes the YOLOS model and image processor.

        Args:
            model_name (str): Hugging Face model identifier.
            model_path (Optional[str]): Local path to the pre-trained model.
                If provided, loads the model from the local directory.
            image_processor_path (Optional[str]): Local path to the image processor.
                If provided, loads the processor from the local directory.
            device (str): Device to run the model on ('cuda' or 'cpu').
            attn_implementation (Optional[str]): Attention implementation to use.
                Set to "sdpa" for Scaled Dot-Product Attention if available.
            torch_dtype (torch.dtype): Data type for the model weights.
                Use torch.float16 for half-precision.
        """
        self.device = device

        # Load YolosImageProcessor
        if image_processor_path and os.path.exists(image_processor_path):
            self.image_processor = YolosImageProcessor.from_pretrained(
                image_processor_path
            )
        else:
            self.image_processor = YolosImageProcessor.from_pretrained(model_name)

        # Load YolosForObjectDetection model
        if model_path and os.path.exists(model_path):
            self.model = YolosForObjectDetection.from_pretrained(
                model_path,
                attn_implementation=attn_implementation,
                torch_dtype=torch_dtype,
                local_files_only=True,
            )
        else:
            self.model = YolosForObjectDetection.from_pretrained(
                model_name,
                attn_implementation=attn_implementation,
                torch_dtype=torch_dtype,
            )

        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        images: List[Image.Image],
        threshold: float = 0.5,
        target_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Dict[str, List]]:
        """
        Performs object detection on a list of images.

        Args:
            images (List[PIL.Image.Image]): List of PIL Image objects.
            threshold (float, optional): Score threshold to filter detections.
                Defaults to 0.5.
            target_sizes (Optional[List[Tuple[int, int]]]): List of target sizes
                (height, width) for each image. If None, original sizes are used.

        Returns:
            List[Dict[str, List]]: List of detection results per image.
                Each dictionary contains 'scores', 'labels', and 'boxes'.
        """
        # Preprocess images
        inputs = self.image_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process outputs
        if target_sizes is None:
            target_sizes = [image.size[::-1] for image in images]  # (height, width)

        results = self.image_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )

        # Convert boxes to (xmin, ymin, xmax, ymax) format
        for result in results:
            boxes = result["boxes"].tolist()
            converted_boxes = [
                [
                    box[0],
                    box[1],
                    box[2],
                    box[3],
                ]
                for box in boxes
            ]
            result["boxes"] = converted_boxes

        return results

    def save_model(self, save_directory: str):
        """
        Saves the model and image processor to a specified directory.

        Args:
            save_directory (str): Path to the directory where the model will be saved.
        """
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)
        self.image_processor.save_pretrained(save_directory)

    @classmethod
    def load_from_directory(
        cls,
        save_directory: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation: Optional[str] = "sdpa",
        torch_dtype: torch.dtype = torch.float16,
    ):
        """
        Loads the YOLOS model and image processor from a specified directory.

        Args:
            save_directory (str): Path to the directory where the model is saved.
            device (str): Device to run the model on ('cuda' or 'cpu').
            attn_implementation (Optional[str]): Attention implementation to use.
                Set to "sdpa" for Scaled Dot-Product Attention if available.
            torch_dtype (torch.dtype): Data type for the model weights.
                Use torch.float16 for half-precision.

        Returns:
            YOLOS: An instance of the YOLOS class.
        """
        return cls(
            model_path=save_directory,
            image_processor_path=save_directory,
            device=device,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
