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
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray

class ObjectDetector:
    def __init__(self, config: ModelConfig = ModelConfig()):
        self.config = config
        self.device = config.device if torch.cuda.is_available() else "cpu"
        
        # Initialize model and processor
        self.processor = AutoProcessor.from_pretrained(config.model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            config.model_name).to(self.device)

    @staticmethod
    def load_image(path_or_url: str) -> np.ndarray:
        """Load image from path or URL"""
        if path_or_url.startswith("http"):
            img = Image.open(requests.get(path_or_url, stream=True).raw)
            return np.array(img)
        else:
            img = cv2.imread(path_or_url)[:, :, ::-1]
            return img

    @staticmethod
    def compute_iou_batch(box1: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Compute IoU between a box and an array of boxes"""
        # Calculate intersection coordinates
        x1 = np.maximum(box1[0], boxes[:, 0])
        y1 = np.maximum(box1[1], boxes[:, 1])
        x2 = np.minimum(box1[2], boxes[:, 2])
        y2 = np.minimum(box1[3], boxes[:, 3])
        
        # Calculate area of intersection
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate area of boxes
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Calculate IoU
        union = area1 + area2 - intersection
        iou = intersection / (union + 1e-6)
        
        return iou

    @staticmethod
    def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two individual boxes"""
        # Calculate intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate area of intersection
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate area of boxes
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
        """Apply Fast NMS (Soft-NMS) to detection results"""
        if len(boxes) == 0:
            return boxes, scores, labels

        # Initialize arrays for keeping track of boxes
        indices = np.arange(len(scores))
        updated_scores = scores.copy()
        
        # Sort boxes by score
        order = np.argsort(scores)[::-1]
        boxes = boxes[order]
        updated_scores = updated_scores[order]
        indices = indices[order]
        
        # Apply Soft-NMS
        for i in range(len(boxes)):
            ious = self.compute_iou_batch(boxes[i], boxes[i+1:])
            overlapping = ious > self.config.nms_threshold
            updated_scores[i+1:][overlapping] *= np.exp(-(ious[overlapping]**2)/self.config.nms_sigma)
        
        # Keep boxes above score threshold
        keep = updated_scores > 0.001
        filtered_boxes = boxes[keep]
        filtered_scores = updated_scores[keep]
        filtered_indices = indices[keep]
        
        # Sort by updated scores
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
        Perform zero-shot object detection on an image
        Args:
            img: Image path/URL or numpy array
            queries: List of text queries to detect
        Returns:
            DetectionResult object containing boxes, scores, and labels
        """
        # Load image if path/URL provided
        if isinstance(img, str):
            img = self.load_image(img)

        # Prepare inputs
        inputs = self.processor(
            text=queries,
            images=Image.fromarray(img),
            return_tensors="pt"
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
        target_sizes = torch.tensor([img.shape[:2]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=self.config.detection_threshold,
            target_sizes=target_sizes
        )[0]

        # Get detection results
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()

        # Apply NMS
        if len(boxes) > 0:
            boxes, scores, labels = self.apply_nms(boxes, scores, labels)
        
        return DetectionResult(boxes=boxes, scores=scores, labels=labels)

    def visualize(
        self,
        img: Union[str, np.ndarray],
        result: DetectionResult,
    ) -> None:
        """Visualize detection results"""
        # Ensure img is numpy array
        if isinstance(img, str):
            img = self.load_image(img)
        elif not isinstance(img, np.ndarray):
            raise TypeError("img must be a path string or numpy array")

        fig, ax = plt.subplots(1, figsize=(12,8))
        ax.imshow(img)
        
        # Separate buckets and piles
        bucket_boxes = []
        bucket_indices = []
        pile_indices = []
        
        for i, (box, score, label) in enumerate(zip(result.boxes, result.scores, result.labels)):
            if score < self.config.visualization_score_threshold:
                continue
            
            if (box[2]-box[0]) > 0.75*img.shape[1] or (box[3]-box[1]) > 0.75*img.shape[0]:
                continue
                
            label_text = PILE_QUERIES[label]
            if "bucket" in label_text:
                bucket_boxes.append(box)
                bucket_indices.append(i)
            else:
                pile_indices.append(i)
        
        # Filter piles that overlap with buckets
        filtered_pile_indices = []
        for pile_idx in pile_indices:
            keep_pile = True
            pile_box = result.boxes[pile_idx]
            
            # Check overlap with buckets
            for bucket_box in bucket_boxes:
                if self.compute_iou(pile_box, bucket_box) > 0.6:
                    keep_pile = False
                    break
            
            if keep_pile:
                filtered_pile_indices.append(pile_idx)
        
        # Draw filtered piles
        for i in filtered_pile_indices:
            x0, y0, x1, y1 = result.boxes[i]
            score = result.scores[i]
            label = result.labels[i]
            
            if score < self.config.visualization_score_threshold:
                continue
            
            rect = plt.Rectangle(
                (x0,y0),
                x1-x0,
                y1-y0,
                fill=False,
                linewidth=2,
                edgecolor='r'
            )
            ax.add_patch(rect)
            ax.text(
                x0,
                y0-5,
                f"{score:.2f}",
                color='yellow',
                fontsize=12,
                backgroundcolor='black'
            )
            ax.text(
                x0,
                y0-20,
                PILE_QUERIES[label],
                color='white',
                fontsize=12,
                backgroundcolor='black')
        
        # # Draw buckets with different color
        # for i in bucket_indices:
        #     x0, y0, x1, y1 = result.boxes[i]
        #     score = result.scores[i]
        #     label = result.labels[i]
            
        #     rect = plt.Rectangle(
        #         (x0,y0),
        #         x1-x0,
        #         y1-y0,
        #         fill=False,
        #         linewidth=2,
        #         edgecolor='b'  # Blue color for buckets
        #     )
        #     ax.add_patch(rect)
        #     ax.text(
        #         x0,
        #         y0-5,
        #         f"{score:.2f}",
        #         color='yellow',
        #         fontsize=12,
        #         backgroundcolor='black'
        #     )
        #     ax.text(
        #         x0,
        #         y0-20,
        #         f"equipment_{PILE_QUERIES[label]}",  # Add equipment_ prefix
        #         color='cyan',
        #         fontsize=12,
        #         backgroundcolor='black')
        
        ax.axis('off')
        plt.show() 