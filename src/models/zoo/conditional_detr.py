# src/models/zoo/conditional_detr.py

import torch
from torch import nn
from typing import Optional, List, Dict, Tuple, Union
from PIL import Image
import os

from transformers import (
    ConditionalDetrConfig,
    ConditionalDetrForObjectDetection,
    ConditionalDetrForSegmentation,
    AutoImageProcessor,
)
from transformers.modeling_outputs import ConditionalDetrObjectDetectionOutput, ConditionalDetrSegmentationOutput

from src.models.zoo.base_model import BaseModel
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ConditionalDetrModel(BaseModel):
    """
    Wrapper for HuggingFace's Conditional DETR models for Object Detection and Segmentation.

    This class extends the BaseModel to provide functionalities specific to Conditional DETR,
    including initialization, training, evaluation, and inference for both object detection
    and segmentation tasks.
    """

    def __init__(
        self,
        model_name_or_path: str,
        task: str = "object_detection",
        config: Optional[ConditionalDetrConfig] = None,
        use_pretrained_backbone: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        """
        Initializes the Conditional DETR model for the specified task.

        Args:
            model_name_or_path (str): Path to the pretrained model or model identifier from HuggingFace hub.
            task (str, optional): Task type - "object_detection" or "segmentation". Defaults to "object_detection".
            config (Optional[ConditionalDetrConfig], optional): Model configuration. If None, uses default configuration.
            use_pretrained_backbone (bool, optional): Whether to use a pretrained backbone. Defaults to True.
            device (Optional[Union[str, torch.device]], optional): Device to run the model on ('cpu' or 'cuda'). If None, defaults to CUDA if available.
            **kwargs: Additional keyword arguments for model configuration.
        """
        self.task = task.lower()
        if self.task not in ["object_detection", "segmentation"]:
            raise ValueError("Task must be either 'object_detection' or 'segmentation'.")

        # Select the appropriate model class based on the task
        if self.task == "object_detection":
            model_class = ConditionalDetrForObjectDetection
        else:
            model_class = ConditionalDetrForSegmentation

        super().__init__(
            model_name_or_path=model_name_or_path,
            model_class=model_class,
            config=config,
            num_labels=None,  # Specify if needed
            **kwargs,
        )

        # Update configuration with task-specific parameters if necessary
        if config is not None:
            self.config = config
        else:
            self.config = ConditionalDetrConfig.from_pretrained(model_name_or_path, **kwargs)

        # Initialize the model with task-specific parameters
        if self.task == "object_detection":
            self.model = ConditionalDetrForObjectDetection.from_pretrained(
                model_name_or_path,
                config=self.config,
                use_pretrained_backbone=use_pretrained_backbone,
                **kwargs,
            )
        else:
            self.model = ConditionalDetrForSegmentation.from_pretrained(
                model_name_or_path,
                config=self.config,
                use_pretrained_backbone=use_pretrained_backbone,
                **kwargs,
            )

        # Move model to the specified device
        self.to_device(device)

        logger.info(f"Initialized ConditionalDetrModel for task '{self.task}' with model '{model_name_or_path}'.")

    def forward(self, **inputs):
        """
        Defines the forward pass of the model.

        Args:
            **inputs: Arbitrary keyword arguments corresponding to the model's forward method.

        Returns:
            ConditionalDetrObjectDetectionOutput or ConditionalDetrSegmentationOutput:
                Depending on the initialized task.
        """
        return self.model(**inputs)

    def compute_loss(self, outputs, targets) -> torch.Tensor:
        """
        Computes the loss given model outputs and targets.

        Args:
            outputs: Outputs from the model.
            targets: Ground truth targets.

        Returns:
            torch.Tensor: The computed loss.
        """
        if self.task == "object_detection":
            loss = outputs.loss
        elif self.task == "segmentation":
            loss = outputs.loss
        else:
            raise ValueError("Unsupported task type.")
        return loss

    def compute_metrics(
        self,
        outputs: List[Union[ConditionalDetrObjectDetectionOutput, ConditionalDetrSegmentationOutput]],
        targets: List[Any],
        image_ids: List[Any],
    ) -> Dict[str, float]:
        """
        Computes evaluation metrics given the model outputs and targets.

        Args:
            outputs (List[ConditionalDetrObjectDetectionOutput or ConditionalDetrSegmentationOutput]): List of model outputs.
            targets (List[Any]): List of ground truth targets.
            image_ids (List[Any]): List of image IDs corresponding to the outputs.

        Returns:
            Dict[str, float]: Dictionary of computed metrics.
        """
        # Placeholder for actual metric computation
        # Implement COCO mAP or other relevant metrics here
        # This example assumes object_detection task and uses COCO API

        if self.task != "object_detection":
            logger.warning("compute_metrics is currently implemented for object_detection only.")
            return {}

        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        # Convert targets and predictions to COCO format
        coco_gt = self._convert_targets_to_coco(targets)
        coco_dt = self._convert_predictions_to_coco(outputs, image_ids)

        # Initialize COCO API for ground truth and detections
        coco = COCO()
        coco.dataset = coco_gt
        coco.createIndex()

        coco_dt = coco.loadRes(coco_dt)
        coco_eval = COCOeval(coco, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics = {
            "mAP": coco_eval.stats[0],
            "mAP_50": coco_eval.stats[1],
            "mAP_75": coco_eval.stats[2],
            # Add more metrics as needed
        }

        return metrics

    def _convert_targets_to_coco(self, targets: List[Dict]) -> Dict:
        """
        Converts ground truth targets to COCO format.

        Args:
            targets (List[Dict]): List of ground truth targets.

        Returns:
            Dict: Ground truth data in COCO format.
        """
        coco_format = {
            "images": [],
            "annotations": [],
            "categories": self._get_coco_categories(),
        }

        for target in targets:
            image_id = target["image_id"]
            coco_format["images"].append({
                "id": image_id,
                "file_name": target.get("file_name", f"image_{image_id}.jpg"),
                # Add other image metadata if available
            })
            for anno in target["annotations"]:
                coco_format["annotations"].append({
                    "id": len(coco_format["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": anno["category_id"],
                    "bbox": anno["bbox"],
                    "area": anno["bbox"][2] * anno["bbox"][3],
                    "iscrowd": 0,
                })

        return coco_format

    def _convert_predictions_to_coco(
        self,
        outputs: List[Union[ConditionalDetrObjectDetectionOutput, ConditionalDetrSegmentationOutput]],
        image_ids: List[Any],
    ) -> List[Dict]:
        """
        Converts model predictions to COCO format.

        Args:
            outputs (List[ConditionalDetrObjectDetectionOutput or ConditionalDetrSegmentationOutput]): List of model outputs.
            image_ids (List[Any]): List of image IDs corresponding to the outputs.

        Returns:
            List[Dict]: List of prediction annotations in COCO format.
        """
        coco_results = []
        for output, img_id in zip(outputs, image_ids):
            if self.task == "object_detection":
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                boxes = output["boxes"].cpu().numpy()  # (xmin, ymin, xmax, ymax)
                for score, label, box in zip(scores, labels, boxes):
                    coco_results.append({
                        "image_id": img_id,
                        "category_id": label,
                        "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                        "score": float(score),
                    })
            elif self.task == "segmentation":
                # Implement segmentation metrics if needed
                pass  # Placeholder
        return coco_results

    def _get_coco_categories(self) -> List[Dict]:
        """
        Retrieves COCO category mappings from the model configuration.

        Returns:
            List[Dict]: List of category dictionaries with 'id' and 'name'.
        """
        categories = []
        for id, label in self.model.config.id2label.items():
            if id == 0:
                continue  # Typically, 0 is reserved for 'no-object'
            categories.append({
                "id": id,
                "name": label,
            })
        return categories

    def predict(
        self,
        images: Union[Image.Image, List[Image.Image]],
        threshold: float = 0.5,
        top_k: int = 100,
    ) -> Union[Dict, List[Dict]]:
        """
        Performs inference on a single image or a batch of images.

        Args:
            images (Union[Image.Image, List[Image.Image]]): PIL Image or list of PIL Images for inference.
            threshold (float, optional): Score threshold to filter predictions. Defaults to 0.5.
            top_k (int, optional): Maximum number of predictions to retain per image. Defaults to 100.

        Returns:
            Union[Dict, List[Dict]]: Prediction dictionary or list of prediction dictionaries containing 'scores', 'labels', and 'boxes' or 'pred_masks' based on the task.
        """
        is_single = False
        if isinstance(images, Image.Image):
            images = [images]
            is_single = True

        inputs = self.feature_extractor(images=images, return_tensors="pt")
        inputs = self.prepare_inputs(inputs)

        self.model.eval()
        with torch.no_grad():
            outputs = self(**inputs)

        target_sizes = torch.tensor([img.size[::-1] for img in images]).to(self.device)

        if self.task == "object_detection":
            results = self.feature_extractor.post_process_object_detection(
                outputs, threshold=threshold, target_sizes=target_sizes, top_k=top_k
            )
            predictions = []
            for result in results:
                prediction = {
                    "scores": result["scores"].cpu().numpy(),
                    "labels": result["labels"].cpu().numpy(),
                    "boxes": result["boxes"].cpu().numpy(),
                }
                predictions.append(prediction)
        elif self.task == "segmentation":
            results = self.feature_extractor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                target_sizes=target_sizes,
                mask_threshold=threshold,
                overlap_mask_area_threshold=0.8
            )
            predictions = []
            for result in results:
                prediction = {
                    "segmentation": result["segmentation"].cpu(),
                    "segments_info": result["segments_info"],
                }
                predictions.append(prediction)
        else:
            raise ValueError("Unsupported task type.")

        if is_single:
            return predictions[0]
        return predictions

    # Optionally, override or extend other BaseModel methods if needed
    # For example, you might want to implement custom training loops or additional utilities

"""
# Example Usage
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup
    from PIL import Image
    import requests

    # Initialize the model
    model_name = "microsoft/conditional-detr-resnet-50"
    task = "object_detection"  # or "segmentation"
    conditional_detr = ConditionalDetrModel(model_name_or_path=model_name, task=task)

    # Prepare a sample image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # Perform prediction
    predictions = conditional_detr.predict(image)
    print(predictions)

    # Assume you have train_dataloader and eval_dataloader
    # train_dataloader = DataLoader(...)  # Define your training DataLoader
    # eval_dataloader = DataLoader(...)   # Define your evaluation DataLoader

    # Example of training
    optimizer = conditional_detr.get_optimizer(learning_rate=1e-4, weight_decay=0.01)
    scheduler = conditional_detr.get_scheduler(
        optimizer=optimizer,
        scheduler_class=get_linear_schedule_with_warmup,
        scheduler_params={
            "num_warmup_steps": 100,
            "num_training_steps": 1000,
        },
    )

    conditional_detr.fit(
        train_dataloader=train_dataloader,
        val_dataloader=eval_dataloader,
        epochs=10,
        optimizer_class=torch.optim.AdamW,
        optimizer_params={"lr": 1e-4, "weight_decay": 0.01},
        scheduler_class=get_linear_schedule_with_warmup,
        scheduler_params={"num_warmup_steps": 100, "num_training_steps": 1000},
    )

    # Example of saving the model
    # conditional_detr.save("path/to/save/directory")

    # Example of loading the model
    # loaded_model = ConditionalDetrModel.load("path/to/save/directory", task=task)
    """
