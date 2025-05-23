import cv2
import numpy as np
from typing import Tuple
from typing import Optional

class CameraCalibration:
    def __init__(self, calibration_file: Optional[str] = None):
        if calibration_file:
            self.load_calibration(calibration_file)
        else:
            self.camera_matrix = np.array([
                [800.0, 0.0, 320.0],    # fx, 0, cx
                [0.0, 800.0, 240.0],    # 0, fy, cy
                [0.0, 0.0, 1.0]         # 0, 0, 1
            ])
            self.dist_coeffs = np.zeros((4, 1))
    
    def load_calibration(self, file_path: str):
        """Load calibration from file"""
        try:
            calib_data = np.load(file_path)
            self.camera_matrix = calib_data['camera_matrix']
            self.dist_coeffs = calib_data['dist_coeffs']
        except Exception as e:
            print(f"Failed to load calibration: {e}")
            self.__init__()
    
    def pixel_to_world(self, pixel_x: int, pixel_y: int, depth: float) -> Tuple[float, float, float]:
        """Convert pixel coordinates to world coordinates"""
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        
        x = (pixel_x - cx) * depth / fx
        y = (pixel_y - cy) * depth / fy
        z = depth
        
        return x, y, z