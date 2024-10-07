# src/models/zoo/conditional_detr.py

import torch
from transformers import (
    ConditionalDetrConfig,
    ConditionalDetrForObjectDetection,
    ConditionalDetrForSegmentation,
    AutoImageProcessor,
)
from typing import Optional, List, Dict, Tuple, Union
from PIL import Image
import os

class ConditionalDetrModel:
    """
    Wrapper for HuggingFace's Conditional DETR models for Object Detection and Segmentation.

    This class provides an interface to initialize, train, evaluate, and perform inference
    using the Conditional DETR model, leveraging HuggingFace's transformers library for
    seamless integration and maximal functionality.
    """

    def __init__(
        self,
        model_name_or_path: str,
        task: str = "object_detection",
        config: Optional[ConditionalDetrConfig] = None,
        use_pretrained_backbone: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Initializes the Conditional DETR model for the specified task.

        Args:
            model_name_or_path (str): Path to the pretrained model or model identifier from HuggingFace hub.
            task (str, optional): Task type - "object_detection" or "segmentation". Defaults to "object_detection".
            config (Optional[ConditionalDetrConfig], optional): Model configuration. If None, uses default configuration.
            use_pretrained_backbone (bool, optional): Whether to use a pretrained backbone. Defaults to True.
            device (Optional[torch.device], optional): Device to run the model on ('cpu' or 'cuda'). If None, defaults to CUDA if available.
        """
        self.task = task.lower()
        if self.task not in ["object_detection", "segmentation"]:
            raise ValueError("Task must be either 'object_detection' or 'segmentation'.")

        self.config = config if config is not None else ConditionalDetrConfig()
        self.image_processor = AutoImageProcessor.from_pretrained(model_name_or_path)

        if self.task == "object_detection":
            self.model = ConditionalDetrForObjectDetection.from_pretrained(
                model_name_or_path,
                config=self.config,
                use_pretrained_backbone=use_pretrained_backbone,
            )
        else:
            self.model = ConditionalDetrForSegmentation.from_pretrained(
                model_name_or_path,
                config=self.config,
                use_pretrained_backbone=use_pretrained_backbone,
            )

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def save_model(self, save_directory: str):
        """
        Saves the model and image processor to the specified directory.

        Args:
            save_directory (str): Directory path to save the model and processor.
        """
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)
        self.image_processor.save_pretrained(save_directory)
        print(f"Model and image processor saved to {save_directory}")

    def load_model(
        self,
        model_name_or_path: str,
        task: Optional[str] = None,
        config: Optional[ConditionalDetrConfig] = None,
        use_pretrained_backbone: bool = True,
    ):
        """
        Loads the model and image processor from a specified path or identifier.

        Args:
            model_name_or_path (str): Path to the pretrained model or model identifier from HuggingFace hub.
            task (Optional[str], optional): Task type - "object_detection" or "segmentation". If None, retains existing task.
            config (Optional[ConditionalDetrConfig], optional): Model configuration. If None, uses default configuration.
            use_pretrained_backbone (bool, optional): Whether to use a pretrained backbone. Defaults to True.
        """
        if task:
            task = task.lower()
            if task not in ["object_detection", "segmentation"]:
                raise ValueError("Task must be either 'object_detection' or 'segmentation'.")
            self.task = task

        self.config = config if config is not None else ConditionalDetrConfig()
        self.image_processor = AutoImageProcessor.from_pretrained(model_name_or_path)

        if self.task == "object_detection":
            self.model = ConditionalDetrForObjectDetection.from_pretrained(
                model_name_or_path,
                config=self.config,
                use_pretrained_backbone=use_pretrained_backbone,
            )
        else:
            self.model = ConditionalDetrForSegmentation.from_pretrained(
                model_name_or_path,
                config=self.config,
                use_pretrained_backbone=use_pretrained_backbone,
            )

        self.model.to(self.device)
        print(f"Model loaded from {model_name_or_path} for task '{self.task}'.")

    def predict(
        self,
        images: Union[Image.Image, List[Image.Image]],
        threshold: float = 0.5,
        top_k: int = 100,
    ) -> List[Dict]:
        """
        Performs inference on a single image or a batch of images.

        Args:
            images (Union[Image.Image, List[Image.Image]]): PIL Image or list of PIL Images for inference.
            threshold (float, optional): Score threshold to filter predictions. Defaults to 0.5.
            top_k (int, optional): Maximum number of predictions to retain per image. Defaults to 100.

        Returns:
            List[Dict]: List of prediction dictionaries containing 'scores', 'labels', and 'boxes' or 'pred_masks' based on the task.
        """
        is_single = False
        if isinstance(images, Image.Image):
            images = [images]
            is_single = True

        inputs = self.image_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([img.size[::-1] for img in images]).to(self.device)

        if self.task == "object_detection":
            results = self.image_processor.post_process_object_detection(
                outputs, threshold=threshold, target_sizes=target_sizes, top_k=top_k
            )
        elif self.task == "segmentation":
            results = self.image_processor.post_process_instance_segmentation(
                outputs, threshold=threshold, target_sizes=target_sizes, mask_threshold=threshold, overlap_mask_area_threshold=0.8
            )
        else:
            raise ValueError("Unsupported task type.")

        # Convert results to CPU and detach tensors
        predictions = []
        for result in results:
            if self.task == "object_detection":
                prediction = {
                    "scores": result["scores"].cpu().numpy(),
                    "labels": result["labels"].cpu().numpy(),
                    "boxes": result["boxes"].cpu().numpy(),
                }
            elif self.task == "segmentation":
                prediction = {
                    "segmentation": result["segmentation"].cpu(),
                    "segments_info": result["segments_info"],
                }
            predictions.append(prediction)

        if is_single:
            return predictions[0]
        return predictions

    def train_model(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 10,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        gradient_accumulation_steps: int = 1,
        device: Optional[torch.device] = None,
    ):
        """
        Trains the Conditional DETR model using a custom training loop.

        Args:
            train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            num_epochs (int, optional): Number of training epochs. Defaults to 10.
            scheduler (Optional[torch.optim.lr_scheduler._LRScheduler], optional): Learning rate scheduler. Defaults to None.
            gradient_accumulation_steps (int, optional): Steps to accumulate gradients before updating. Defaults to 1.
            device (Optional[torch.device], optional): Device to train the model on. If None, uses initialized device.
        """
        device = device if device else self.device
        self.model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                images = batch["images"]
                labels = batch["labels"]

                inputs = self.image_processor(images=images, return_tensors="pt").to(device)
                if self.task == "object_detection":
                    outputs = self.model(**inputs, labels=labels)
                elif self.task == "segmentation":
                    outputs = self.model(**inputs, labels=labels)
                else:
                    raise ValueError("Unsupported task type.")

                loss = outputs.loss
                loss = loss / gradient_accumulation_steps
                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * gradient_accumulation_steps

            avg_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    def evaluate_model(
        self,
        eval_dataloader: torch.utils.data.DataLoader,
        threshold: float = 0.5,
        top_k: int = 100,
    ) -> Dict:
        """
        Evaluates the Conditional DETR model using COCO evaluation metrics.

        Args:
            eval_dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
            threshold (float, optional): Score threshold to filter predictions. Defaults to 0.5.
            top_k (int, optional): Maximum number of predictions to retain per image. Defaults to 100.

        Returns:
            Dict: Evaluation metrics such as mAP.
        """
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        self.model.eval()
        device = self.device
        all_predictions = []
        all_targets = []
        image_ids = []

        with torch.no_grad():
            for batch in eval_dataloader:
                images = batch["images"]
                labels = batch["labels"]
                img_ids = batch.get("image_id", None)

                inputs = self.image_processor(images=images, return_tensors="pt").to(device)
                outputs = self.model(**inputs)

                target_sizes = torch.tensor([img.size[::-1] for img in images]).to(device)

                if self.task == "object_detection":
                    results = self.image_processor.post_process_object_detection(
                        outputs, threshold=threshold, target_sizes=target_sizes, top_k=top_k
                    )
                elif self.task == "segmentation":
                    results = self.image_processor.post_process_instance_segmentation(
                        outputs, threshold=threshold, target_sizes=target_sizes, mask_threshold=threshold, overlap_mask_area_threshold=0.8
                    )
                else:
                    raise ValueError("Unsupported task type.")

                for result, target in zip(results, labels):
                    if self.task == "object_detection":
                        preds = {
                            "scores": result["scores"].cpu().numpy(),
                            "labels": result["labels"].cpu().numpy(),
                            "boxes": result["boxes"].cpu().numpy(),
                        }
                        all_predictions.append(preds)
                        all_targets.append(target)
                        image_ids.append(target["image_id"])
                    elif self.task == "segmentation":
                        preds = {
                            "segmentation": result["segmentation"].cpu(),
                            "segments_info": result["segments_info"],
                        }
                        all_predictions.append(preds)
                        all_targets.append(target)
                        image_ids.append(target["image_id"])

        # Convert to COCO format
        coco_gt = self._convert_targets_to_coco(all_targets)
        coco_dt = self._convert_predictions_to_coco(all_predictions, image_ids)

        # Initialize COCO API
        coco = COCO()
        coco.dataset = coco_gt
        coco.createIndex()

        coco_results = coco_dt
        coco_dt = coco.loadRes(coco_results)
        coco_eval = COCOeval(coco, coco_dt, 'bbox' if self.task == "object_detection" else 'segm')
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

        for idx, target in enumerate(targets, 1):
            coco_format["images"].append({
                "id": target["image_id"],
                "file_name": target.get("file_name", f"image_{target['image_id']}.jpg"),
                # Add other image metadata if available
            })
            for anno in target["annotations"]:
                coco_format["annotations"].append({
                    "id": len(coco_format["annotations"]) + 1,
                    "image_id": target["image_id"],
                    "category_id": anno["category_id"],
                    "bbox": anno["bbox"],
                    "area": anno["bbox"][2] * anno["bbox"][3],
                    "iscrowd": 0,
                })

        return coco_format

    def _convert_predictions_to_coco(self, predictions: List[Dict], image_ids: List[int]) -> List[Dict]:
        """
        Converts model predictions to COCO format.

        Args:
            predictions (List[Dict]): List of model predictions.
            image_ids (List[int]): Corresponding image IDs for each prediction.

        Returns:
            List[Dict]: List of prediction annotations in COCO format.
        """
        coco_results = []
        for pred, img_id in zip(predictions, image_ids):
            for score, label, box in zip(pred["scores"], pred["labels"], pred["boxes"]):
                xmin, ymin, xmax, ymax = box.tolist()
                width = xmax - xmin
                height = ymax - ymin
                coco_results.append({
                    "image_id": img_id,
                    "category_id": label,
                    "bbox": [xmin, ymin, width, height],
                    "score": score,
                })
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

    def post_process(
        self,
        outputs: Union[ConditionalDetrForObjectDetection.Output, ConditionalDetrForSegmentation.Output],
        threshold: float = 0.5,
        target_sizes: Optional[torch.Tensor] = None,
        top_k: int = 100,
    ) -> List[Dict]:
        """
        Applies post-processing to model outputs.

        Args:
            outputs (Union[ConditionalDetrForObjectDetection.Output, ConditionalDetrForSegmentation.Output]): Raw model outputs.
            threshold (float, optional): Score threshold to filter predictions. Defaults to 0.5.
            target_sizes (Optional[torch.Tensor], optional): Target sizes for resizing predictions. Defaults to None.
            top_k (int, optional): Maximum number of predictions to retain per image. Defaults to 100.

        Returns:
            List[Dict]: List of post-processed predictions.
        """
        if self.task == "object_detection":
            results = self.image_processor.post_process_object_detection(
                outputs, threshold=threshold, target_sizes=target_sizes, top_k=top_k
            )
        elif self.task == "segmentation":
            results = self.image_processor.post_process_instance_segmentation(
                outputs, threshold=threshold, target_sizes=target_sizes, mask_threshold=threshold, overlap_mask_area_threshold=0.8
            )
        else:
            raise ValueError("Unsupported task type.")

        processed_results = []
        for result in results:
            if self.task == "object_detection":
                processed_results.append({
                    "scores": result["scores"].cpu().numpy(),
                    "labels": result["labels"].cpu().numpy(),
                    "boxes": result["boxes"].cpu().numpy(),
                })
            elif self.task == "segmentation":
                processed_results.append({
                    "segmentation": result["segmentation"].cpu(),
                    "segments_info": result["segments_info"],
                })
        return processed_results

