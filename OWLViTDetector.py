# vlm_enhanced_vision.py - Vision system with VLM support for natural language commands
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
    """OWL-ViT based object detection with natural language queries"""
    def __init__(self, model_name: str = "google/owlvit-base-patch32", VLM_AVAILABLE: bool = True):
        self.logger = logging.getLogger(__name__)

        if not VLM_AVAILABLE:
            raise ImportError("VLM is not available.")

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        try:
            self.processor = OwlViTProcessor.from_pretrained(model_name)
            self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self.logger.info(f"✅ OWL-ViT loaded on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load OWL-ViT: {e}")
            raise

    
    def detect_with_text_queries(self, image: np.ndarray, text_queries: List[str], 
                               confidence_threshold: float = 0.1) -> List[Tuple[str, float, List[int]]]:
        """Detect objects using natural language descriptions"""
        try:
            # Convert BGR to RGB and create PIL image
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Process inputs
            inputs = self.processor(text=text_queries, images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process results
            target_sizes = torch.Tensor([pil_image.size[::-1]]).to(self.device)
            results = self.processor.post_process_object_detection(
                outputs=outputs, 
                target_sizes=target_sizes, 
                threshold=confidence_threshold
            )
            
            detections = []
            for i, (query, result) in enumerate(zip(text_queries, results)):
                boxes = result["boxes"].cpu().numpy()
                scores = result["scores"].cpu().numpy()
                
                for box, score in zip(boxes, scores):
                    if score >= confidence_threshold:
                        bbox = [int(coord) for coord in box]
                        detections.append((query, float(score), bbox))
            
            return detections
            
        except Exception as e:
            self.logger.error(f"VLM detection failed: {e}")
            return []
    
    def get_bbox_center(self, bbox: List[int]) -> Tuple[int, int]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return (x1 + x2) // 2, (y1 + y2) // 2
