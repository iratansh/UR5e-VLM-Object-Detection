"""
OWL-ViT Visual Language Model Integration.

This module integrates the OWL-ViT (Vision-Language Transformer) model
for zero-shot object detection. It provides:
- Natural language object queries
- Zero-shot object detection
- Bounding box prediction
- Confidence scoring

The model enables flexible object detection without pre-defined classes,
using natural language descriptions like "a red cup" or "a blue box".
"""

import cv2
import torch
import numpy as np
import time
import logging
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import speech_recognition as sr
import threading
import queue
from ObjectDetection import ObjectDetection
from CameraCalibration import CameraCalibration
from WorkSpaceValidator import WorkspaceValidator
from transformers import OwlViTProcessor, OwlViTForObjectDetection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OWLViTDetector:
    """
    OWL-ViT based object detector with natural language queries.
    
    This class provides zero-shot object detection using the OWL-ViT
    model, allowing flexible object detection through text descriptions.
    
    Parameters
    ----------
    model_name : str, optional
        Name of OWL-ViT model variant, by default "google/owlvit-base-patch32"
    device : str, optional
        Device to run model on ('cuda' or 'cpu'), by default None
    confidence_threshold : float, optional
        Detection confidence threshold, by default 0.3
        
    Attributes
    ----------
    processor : OwlViTProcessor
        OWL-ViT image and text processor
    model : OwlViTForObjectDetection
        OWL-ViT model for object detection
    device : torch.device
        Device model is running on
        
    Notes
    -----
    Requirements:
    - PyTorch
    - Transformers library
    - CUDA-capable GPU (optional)
    """
    def __init__(self, model_name: str = "google/owlvit-base-patch32",
                 device: Optional[str] = None,
                 confidence_threshold: float = 0.3):
        self.logger = logging.getLogger(__name__)
        self.confidence_threshold = confidence_threshold
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        try:
            # Load model and processor
            self.processor = OwlViTProcessor.from_pretrained(model_name)
            self.model = OwlViTForObjectDetection.from_pretrained(model_name)
            self.model.to(self.device)
            
            self.logger.info(f"✅ OWL-ViT model loaded on {device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load OWL-ViT model: {e}")
            raise

    def detect_objects(self, image: np.ndarray, queries: List[str]) -> List[Dict]:
        """
        Detect objects in image matching text queries.
        
        Parameters
        ----------
        image : np.ndarray
            RGB image as numpy array
        queries : List[str]
            List of text queries (e.g., ["red cup", "blue box"])
            
        Returns
        -------
        List[Dict]
            List of detections, each containing:
            - label: Matched query text
            - confidence: Detection confidence (0-1)
            - bbox: Bounding box [x1, y1, x2, y2]
            
        Notes
        -----
        Detection process:
        1. Preprocess image and text
        2. Run OWL-ViT inference
        3. Filter by confidence
        4. Convert to output format
        """
        try:
            # Prepare inputs
            inputs = self.processor(
                text=queries,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Run inference
            outputs = self.model(**inputs)
            
            # Post-process outputs
            target_sizes = torch.Tensor([image.shape[:2]])
            results = self.processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=self.confidence_threshold
            )[0]
            
            # Format detections
            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                score_float = score.item()
                if score_float >= self.confidence_threshold:
                    box_int = [int(i) for i in box.tolist()]
                    text_query = queries[label]
                    detections.append({
                        "label": text_query,
                        "confidence": score_float,
                        "bbox": box_int
                    })
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return []

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Parameters
        ----------
        image : np.ndarray
            RGB image as numpy array
            
        Returns
        -------
        torch.Tensor
            Preprocessed image tensor on the correct device
            
        Notes
        -----
        Preprocessing steps:
        1. Convert to PIL Image
        2. Resize to model input size (384x384)
        3. Convert to tensor
        4. Normalize with ImageNet stats
        5. Move to correct device
        
        The preprocessing matches OWL-ViT's training data format
        and ensures optimal model performance.
        """
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(image)
            
            # Apply processor transformations
            inputs = self.processor(
                images=pil_image,
                return_tensors="pt"
            )
            
            # Move to correct device
            image_tensor = inputs.pixel_values.to(self.device)
            
            return image_tensor
            
        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {e}")
            raise

    def _filter_detections(self, boxes: torch.Tensor, scores: torch.Tensor,
                         labels: torch.Tensor) -> List[Dict]:
        """
        Filter and format detection results.
        
        Parameters
        ----------
        boxes : torch.Tensor
            Bounding box coordinates [N, 4]
        scores : torch.Tensor
            Detection confidence scores [N]
        labels : torch.Tensor
            Label indices [N]
            
        Returns
        -------
        List[Dict]
            List of filtered detections with format:
            - bbox: List[int] - [x1, y1, x2, y2]
            - score: float - Confidence score
            - label_idx: int - Label index
            
        Notes
        -----
        Filtering steps:
        1. Remove detections below confidence threshold
        2. Apply non-maximum suppression (IoU threshold 0.5)
        3. Convert coordinates to integer pixels
        4. Format as dictionary for easy access
        
        The filtering ensures only high-quality, non-overlapping
        detections are returned.
        """
        filtered_detections = []
        
        # Convert to numpy for processing
        boxes_np = boxes.cpu().numpy()
        scores_np = scores.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Filter by confidence
        mask = scores_np >= self.confidence_threshold
        boxes_np = boxes_np[mask]
        scores_np = scores_np[mask]
        labels_np = labels_np[mask]
        
        # Apply NMS if multiple detections
        if len(boxes_np) > 0:
            indices = cv2.dnn.NMSBoxes(
                boxes_np.tolist(),
                scores_np.tolist(),
                self.confidence_threshold,
                0.5
            )
            
            for idx in indices:
                filtered_detections.append({
                    'bbox': boxes_np[idx].astype(int).tolist(),
                    'score': float(scores_np[idx]),
                    'label_idx': int(labels_np[idx])
                })
        
        return filtered_detections

    def visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Visualize detections on image.
        
        Parameters
        ----------
        image : np.ndarray
            Original RGB image
        detections : List[Dict]
            List of detections to visualize, each containing:
            - label: str - Object label
            - confidence: float - Detection confidence
            - bbox: List[int] - [x1, y1, x2, y2]
            
        Returns
        -------
        np.ndarray
            Image with visualized detections
            
        Notes
        -----
        Visualization features:
        1. Colored bounding boxes based on confidence:
           - Green: High confidence (>0.7)
           - Yellow: Medium confidence (0.5-0.7)
           - Red: Low confidence (<0.5)
        2. Labels with confidence scores
        3. Box thickness varies with confidence
        4. Semi-transparent overlays
        
        The visualization helps in debugging and understanding
        the model's performance.
        """
        vis_image = image.copy()
        
        for det in detections:
            # Get detection info
            bbox = det['bbox']
            label = det['label']
            conf = det['confidence']
            
            # Determine color based on confidence
            if conf > 0.7:
                color = (0, 255, 0)  # Green
            elif conf > 0.5:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            # Draw box
            thickness = max(1, int(conf * 3))
            cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                         color, thickness)
            
            # Draw label
            label_text = f"{label} ({conf:.2f})"
            cv2.putText(vis_image, label_text, (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_image

    def get_bbox_center(self, bbox: List[int]) -> Tuple[int, int]:
        """
        Calculate the center point of a bounding box.
        
        Parameters
        ----------
        bbox : List[int]
            Bounding box coordinates [x1, y1, x2, y2]
            
        Returns
        -------
        Tuple[int, int]
            Center point coordinates (x, y)
            
        Notes
        -----
        The center is calculated as the midpoint of the box's
        width and height. This is useful for:
        - Grasp point estimation
        - Object tracking
        - Spatial relationship analysis
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return (center_x, center_y)
