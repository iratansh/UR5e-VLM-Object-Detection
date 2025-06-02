from deepsparse import Pipeline
import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

class DeepSparseObjectDetection:
    def __init__(self, model_path: str = "zoo:yolov5s", confidence_threshold: float = 0.5):
        """
        Initialize DeepSparse object detection pipeline
        Args:
            model_path: Path to DeepSparse model or SparseZoo stub
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        try:
            # Initialize DeepSparse pipeline
            self.pipeline = Pipeline.create(
                task="yolo",
                model_path=model_path,
                batch_size=1,
                num_cores=4,  
                scheduler="multi_stream",  
                num_streams=2
            )
            
            self.logger.info(f"DeepSparse pipeline initialized with model: {model_path}")
            
            # Warm up the model
            dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
            self.pipeline([dummy_input])
            self.logger.info("Model warmed up")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize DeepSparse: {e}")
            raise
        
        # Common graspable objects for filtering
        self.graspable_objects = {
            'bottle', 'cup', 'bowl', 'apple', 'orange', 'banana', 'book',
            'cell phone', 'mouse', 'keyboard', 'remote', 'scissors',
            'teddy bear', 'toothbrush', 'knife', 'spoon', 'fork', 'laptop'
        }

    def detect_objects(self, image: np.ndarray) -> List[Tuple[str, float, List[int]]]:
        """
        Detect objects using DeepSparse
        Args:
            image: Input image as numpy array (BGR format)
        Returns:
            List of (label, confidence, bbox) tuples
        """
        try:
            # Convert BGR to RGB for the model
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.pipeline(rgb_image)
            
            return self.parse_results(results)
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return []

    def parse_results(self, results) -> List[Tuple[str, float, List[int]]]:
        """
        Parse DeepSparse results and filter for graspable objects
        Args:
            results: DeepSparse pipeline results
        Returns:
            List of (label, confidence, bbox) for graspable objects above threshold
        """
        detections = []
        
        if not results or not results.boxes:
            return detections
        
        # Extract detections
        for box in results.boxes:
            confidence = float(box.confidence)
            if confidence >= self.confidence_threshold:
                label = box.class_name
                
                # Filter for graspable objects
                if label in self.graspable_objects:
                    # Convert to integer coordinates
                    bbox = [
                        int(box.x1),
                        int(box.y1),
                        int(box.x2),
                        int(box.y2)
                    ]
                    detections.append((label, confidence, bbox))
        
        return detections

    def get_bbox_center(self, bbox: List[int]) -> Tuple[int, int]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    def benchmark(self, image: np.ndarray, num_runs: int = 100) -> dict:
        """
        Benchmark the model performance
        Args:
            image: Test image
            num_runs: Number of inference runs
        Returns:
            Performance metrics dictionary
        """
        import time
        
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            self.detect_objects(image)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        times = np.array(times)
        return {
            'mean_inference_time': np.mean(times),
            'std_inference_time': np.std(times),
            'min_inference_time': np.min(times),
            'max_inference_time': np.max(times),
            'fps': 1.0 / np.mean(times)
        }