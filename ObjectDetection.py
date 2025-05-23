import torch
import numpy as np
import cv2
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ObjectDetection:
    def __init__(self, model_name: str = 'yolov5s', confidence_threshold: float = 0.5):
        """Enhanced ObjectDetection with improved bounding box accuracy"""
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        try:
            # Load YOLOv5 model with optimized settings for better bounding boxes
            self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
            
            # Critical settings for better bounding box accuracy
            self.model.conf = confidence_threshold  # Confidence threshold
            self.model.iou = 0.35  # Lower IoU for tighter NMS (was 0.45)
            self.model.agnostic = False  # Class-agnostic NMS
            self.model.multi_label = False  # Multiple labels per box
            self.model.max_det = 100  # Maximum detections per image
            
            # Enable model optimizations
            self.model.eval()
            
            # Device optimization for M2 Pro
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.model.to(self.device)
                self.logger.info("🚀 Using Apple Silicon GPU (MPS)")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.model.to(self.device)
                self.logger.info("🚀 Using CUDA GPU")
            else:
                self.device = torch.device("cpu")
                self.logger.info("🖥️ Using CPU")
            
            self.logger.info(f"✅ YOLOv5 {model_name} loaded with optimized bbox settings")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
        
        # Enhanced graspable objects list (exact COCO class names)
        self.graspable_objects = {
            'bottle', 'cup', 'bowl', 'apple', 'orange', 'banana', 'sandwich',
            'book', 'cell phone', 'mouse', 'keyboard', 'remote', 'scissors',
            'teddy bear', 'toothbrush', 'knife', 'spoon', 'fork', 'laptop',
            'vase', 'clock', 'hair drier', 'wine glass', 'backpack', 'handbag'
        }
        
        # Priority objects for robot tasks
        self.priority_objects = {
            'bottle', 'cup', 'apple', 'book', 'cell phone', 'mouse', 'remote'
        }
        
        # Image preprocessing settings for better detection
        self.input_size = 640  # YOLOv5 standard input size
        self.padding_color = (114, 114, 114)  # Gray padding

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for optimal YOLO detection"""
        # Get original dimensions
        h, w = frame.shape[:2]
        
        # Calculate scale to fit into model input size while maintaining aspect ratio
        scale = min(self.input_size / w, self.input_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize frame
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.full((self.input_size, self.input_size, 3), self.padding_color, dtype=np.uint8)
        
        # Calculate padding offsets to center the image
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        
        # Place resized image in center
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        # Store transform info for coordinate conversion
        self.transform_info = {
            'scale': scale,
            'pad_x': pad_x,
            'pad_y': pad_y,
            'orig_w': w,
            'orig_h': h
        }
        
        return padded

    def postprocess_coordinates(self, bbox: List[float]) -> List[int]:
        """Convert model coordinates back to original frame coordinates"""
        x1, y1, x2, y2 = bbox
        
        # Remove padding
        x1 -= self.transform_info['pad_x']
        y1 -= self.transform_info['pad_y']
        x2 -= self.transform_info['pad_x']
        y2 -= self.transform_info['pad_y']
        
        # Scale back to original size
        scale = self.transform_info['scale']
        x1 /= scale
        y1 /= scale
        x2 /= scale
        y2 /= scale
        
        # Clamp to frame boundaries
        x1 = max(0, min(self.transform_info['orig_w'] - 1, x1))
        y1 = max(0, min(self.transform_info['orig_h'] - 1, y1))
        x2 = max(0, min(self.transform_info['orig_w'] - 1, x2))
        y2 = max(0, min(self.transform_info['orig_h'] - 1, y2))
        
        return [int(x1), int(y1), int(x2), int(y2)]

    def detect_objects(self, frame: np.ndarray) -> List[Tuple[str, float, List[int]]]:
        """Detect objects with improved bounding box accuracy"""
        try:
            # Ensure frame is valid
            if frame is None or frame.size == 0:
                return []
            
            # Preprocess frame for better detection
            processed_frame = self.preprocess_frame(frame)
            
            # Run inference with optimized settings
            with torch.no_grad():
                results = self.model(processed_frame, size=self.input_size)
            
            return self.parse_results(results)
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return []

    def parse_results(self, results) -> List[Tuple[str, float, List[int]]]:
        """Enhanced result parsing with improved coordinate handling"""
        detections = []
        
        if len(results.xyxy[0]) == 0:
            return detections
        
        for detection in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2, conf, cls = detection
            
            if conf >= self.confidence_threshold:
                label = self.model.names[int(cls)]
                
                # Filter for graspable objects
                if label.lower() in {obj.lower() for obj in self.graspable_objects}:
                    # Convert coordinates back to original frame
                    bbox = self.postprocess_coordinates([x1, y1, x2, y2])
                    
                    # Validate bounding box
                    if self.is_valid_bbox(bbox):
                        detections.append((label, float(conf), bbox))
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x[1], reverse=True)
        
        return detections

    def is_valid_bbox(self, bbox: List[int]) -> bool:
        """Validate bounding box dimensions"""
        x1, y1, x2, y2 = bbox
        
        # Check if coordinates are valid
        if x1 >= x2 or y1 >= y2:
            return False
        
        # Check minimum size (avoid tiny boxes)
        width = x2 - x1
        height = y2 - y1
        min_size = 10  # Minimum 10 pixels
        
        if width < min_size or height < min_size:
            return False
        
        # Check aspect ratio (avoid extremely distorted boxes)
        aspect_ratio = width / height
        if aspect_ratio > 10 or aspect_ratio < 0.1:
            return False
        
        return True

    def refine_bounding_box(self, frame: np.ndarray, bbox: List[int], 
                           margin: int = 5) -> List[int]:
        """Refine bounding box using edge detection (optional enhancement)"""
        x1, y1, x2, y2 = bbox
        
        # Extract ROI with margin
        roi_x1 = max(0, x1 - margin)
        roi_y1 = max(0, y1 - margin)
        roi_x2 = min(frame.shape[1], x2 + margin)
        roi_y2 = min(frame.shape[0], y2 + margin)
        
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if roi.size == 0:
            return bbox
        
        # Convert to grayscale and apply edge detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            cx, cy, cw, ch = cv2.boundingRect(largest_contour)
            
            # Convert back to original coordinates
            refined_x1 = roi_x1 + cx
            refined_y1 = roi_y1 + cy
            refined_x2 = refined_x1 + cw
            refined_y2 = refined_y1 + ch
            
            # Only use refined box if it's reasonable
            orig_area = (x2 - x1) * (y2 - y1)
            refined_area = cw * ch
            
            if 0.5 * orig_area <= refined_area <= 2.0 * orig_area:
                return [refined_x1, refined_y1, refined_x2, refined_y2]
        
        return bbox

    def get_bbox_center(self, bbox: List[int]) -> Tuple[int, int]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return (x1 + x2) // 2, (y1 + y2) // 2
    
    def get_bbox_area(self, bbox: List[int]) -> int:
        """Get area of bounding box"""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
    
    def get_bbox_dimensions(self, bbox: List[int]) -> Tuple[int, int]:
        """Get width and height of bounding box"""
        x1, y1, x2, y2 = bbox
        return x2 - x1, y2 - y1

    def draw_enhanced_bbox(self, frame: np.ndarray, label: str, confidence: float,
                          bbox: List[int], color: Tuple[int, int, int] = (0, 255, 0)) -> None:
        """Draw enhanced bounding box with better visualization"""
        x1, y1, x2, y2 = bbox
        
        # Main bounding box
        thickness = 3 if label in self.priority_objects else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Corner markers for better visibility
        corner_length = 15
        corner_thickness = 3
        
        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)
        
        # Top-right corner
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)
        
        # Bottom-left corner
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)
        
        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)
        
        # Label with better formatting
        label_text = f"{label} {confidence:.2f}"
        font_scale = 0.6
        font_thickness = 2
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        
        # Background for text
        cv2.rectangle(frame, 
                     (x1, y1 - text_height - baseline - 5), 
                     (x1 + text_width + 5, y1), 
                     color, -1)
        
        # Text
        cv2.putText(frame, label_text, (x1 + 2, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        # Center point
        center_x, center_y = self.get_bbox_center(bbox)
        cv2.circle(frame, (center_x, center_y), 4, color, -1)
        cv2.circle(frame, (center_x, center_y), 8, color, 2)