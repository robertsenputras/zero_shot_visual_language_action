# model.py
# Implements zero-shot object detection for construction site scenes using OWL-VLM.
# Provides functionality for pile detection, visualization, and coordinate normalization.

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from dataclasses import dataclass
from typing import List, Tuple, Union
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from config import ModelConfig, PILE_QUERIES

@dataclass
class DetectionResult:
    """
    Container for object detection results.
    
    Attributes:
        boxes: Array of bounding boxes in format [x1, y1, x2, y2]
        scores: Confidence scores for each detection
        labels: Class labels for each detection
    """
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray

class ObjectDetector:
    """
    Zero-shot object detector specialized for construction site scenes.
    Uses OWL-VLM model for flexible object detection without pre-defined classes.
    """
    
    def __init__(self, config: ModelConfig = ModelConfig()):
        """
        Initializes the detector with specified configuration.
        
        Args:
            config: ModelConfig object containing model parameters and thresholds
        """
        self.config = config
        self.device = config.device if torch.cuda.is_available() else "cpu"
        
        # Initialize OWL-VLM model and processor
        self.processor = AutoProcessor.from_pretrained(config.model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            config.model_name).to(self.device)

    @staticmethod
    def load_image(path_or_url: str) -> np.ndarray:
        """
        Loads an image from local path or URL.
        
        Args:
            path_or_url: Local file path or URL to image
            
        Returns:
            np.ndarray: Loaded image in RGB format
        """
        if path_or_url.startswith("http"):
            img = Image.open(requests.get(path_or_url, stream=True).raw)
            return np.array(img)
        else:
            img = cv2.imread(path_or_url)[:, :, ::-1]  # BGR to RGB
            return img

    @staticmethod
    def compute_iou_batch(box1: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Computes Intersection over Union (IoU) between a box and array of boxes.
        
        Args:
            box1: Single box coordinates [x1, y1, x2, y2]
            boxes: Array of box coordinates Nx[x1, y1, x2, y2]
            
        Returns:
            np.ndarray: IoU scores for each box pair
        """
        # Calculate intersection coordinates
        x1 = np.maximum(box1[0], boxes[:, 0])
        y1 = np.maximum(box1[1], boxes[:, 1])
        x2 = np.minimum(box1[2], boxes[:, 2])
        y2 = np.minimum(box1[3], boxes[:, 3])
        
        # Calculate intersection areas
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate box areas
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Calculate IoU
        union = area1 + area2 - intersection
        iou = intersection / (union + 1e-6)  # Add epsilon to prevent division by zero
        
        return iou

    @staticmethod
    def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Computes IoU between two individual boxes.
        
        Args:
            box1: First box coordinates [x1, y1, x2, y2]
            box2: Second box coordinates [x1, y1, x2, y2]
            
        Returns:
            float: IoU score between the boxes
        """
        # Calculate intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate box areas
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate IoU
        union = area1 + area2 - intersection
        iou = intersection / (union + 1e-6)
        
        return iou

    def apply_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies Fast NMS (Soft-NMS) to filter overlapping detections.
        
        Args:
            boxes: Array of bounding boxes
            scores: Detection confidence scores
            labels: Class labels for each detection
            
        Returns:
            tuple: (filtered_boxes, filtered_scores, filtered_labels)
        """
        if len(boxes) == 0:
            return boxes, scores, labels

        # Track boxes and scores
        indices = np.arange(len(scores))
        updated_scores = scores.copy()
        
        # Sort by confidence score
        order = np.argsort(scores)[::-1]
        boxes = boxes[order]
        updated_scores = updated_scores[order]
        indices = indices[order]
        
        # Apply Soft-NMS with Gaussian penalty
        for i in range(len(boxes)):
            ious = self.compute_iou_batch(boxes[i], boxes[i+1:])
            overlapping = ious > self.config.nms_threshold
            updated_scores[i+1:][overlapping] *= np.exp(-(ious[overlapping]**2)/self.config.nms_sigma)
        
        # Filter boxes by updated scores
        keep = updated_scores > 0.001
        filtered_boxes = boxes[keep]
        filtered_scores = updated_scores[keep]
        filtered_indices = indices[keep]
        
        # Sort by final scores
        order = np.argsort(filtered_scores)[::-1]
        
        return (
            filtered_boxes[order],
            filtered_scores[order],
            labels[filtered_indices[order]]
        )

    def detect(
        self,
        img: Union[str, np.ndarray],
        queries: List[str]
    ) -> DetectionResult:
        """
        Performs zero-shot object detection on construction site image.
        
        Args:
            img: Input image (path/URL or numpy array)
            queries: List of text queries describing objects to detect
            
        Returns:
            DetectionResult: Object containing detection boxes, scores, and labels
        """
        # Load image if path/URL provided
        if isinstance(img, str):
            img = self.load_image(img)

        # Prepare inputs for the model
        inputs = self.processor(
            text=queries,
            images=Image.fromarray(img),
            return_tensors="pt"
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process detections
        target_sizes = torch.tensor([img.shape[:2]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=self.config.detection_threshold,
            target_sizes=target_sizes
        )[0]

        # Extract detection components
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()

        # Apply NMS to filter overlapping detections
        if len(boxes) > 0:
            boxes, scores, labels = self.apply_nms(boxes, scores, labels)
        
        return DetectionResult(boxes=boxes, scores=scores, labels=labels)

    def visualize(
        self,
        img: Union[str, np.ndarray],
        result: DetectionResult,
    ) -> None:
        """
        Visualizes detection results on the image.
        
        Args:
            img: Input image (path/URL or numpy array)
            result: DetectionResult object containing boxes and scores
        """
        # Ensure image is in numpy array format
        if isinstance(img, str):
            img = self.load_image(img)
        elif not isinstance(img, np.ndarray):
            raise TypeError("img must be a path string or numpy array")

        # Setup visualization
        fig, ax = plt.subplots(1, figsize=(12,8))
        ax.imshow(img)
        
        # Separate detections by type
        bucket_boxes = []
        bucket_indices = []
        pile_indices = []
        
        # Filter detections by score and size
        for i, (box, score, label) in enumerate(zip(result.boxes, result.scores, result.labels)):
            if score < self.config.visualization_score_threshold:
                continue
            
            # Skip unreasonably large detections
            if (box[2]-box[0]) > 0.75*img.shape[1] or (box[3]-box[1]) > 0.75*img.shape[0]:
                continue
                
            label_text = PILE_QUERIES[label]
            if "bucket" in label_text:
                bucket_boxes.append(box)
                bucket_indices.append(i)
            else:
                pile_indices.append(i)
        
        # Filter out piles that overlap with buckets
        filtered_pile_indices = []
        for pile_idx in pile_indices:
            keep_pile = True
            pile_box = result.boxes[pile_idx]
            
            for bucket_box in bucket_boxes:
                if self.compute_iou(pile_box, bucket_box) > 0.6:
                    keep_pile = False
                    break
            
            if keep_pile:
                filtered_pile_indices.append(pile_idx)
        
        # Visualize filtered pile detections
        for i in filtered_pile_indices:
            x0, y0, x1, y1 = result.boxes[i]
            score = result.scores[i]
            label = result.labels[i]
            
            if score < self.config.visualization_score_threshold:
                continue
            
            # Draw bounding box
            rect = plt.Rectangle(
                (x0,y0),
                x1-x0,
                y1-y0,
                fill=False,
                linewidth=2,
                edgecolor='r'
            )
            ax.add_patch(rect)
            
            # Add confidence score
            ax.text(
                x0,
                y0-5,
                f"{score:.2f}",
                color='yellow',
                fontsize=12,
                backgroundcolor='black'
            )
            
            # Add class label
            ax.text(
                x0,
                y0-20,
                PILE_QUERIES[label],
                color='white',
                fontsize=12,
                backgroundcolor='black')
        
        ax.axis('off')
        plt.show() 