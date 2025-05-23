# vision/WorkspaceValidator.py
import numpy as np

class WorkspaceValidator:
    def __init__(self):
        """UR5e workspace constraints"""
        # UR5e workspace boundaries (meters, relative to base_link)
        self.x_range = (-0.85, 0.85)
        self.y_range = (-0.85, 0.85)
        self.z_range = (0.0, 1.2)
        
        # Safe picking zone (conservative)
        self.safe_x_range = (-0.4, 0.4)
        self.safe_y_range = (0.1, 0.6)
        self.safe_z_range = (0.05, 0.4)
    
    def is_reachable(self, x: float, y: float, z: float, safe_mode: bool = True) -> bool:
        """Check if position is within robot workspace"""
        if safe_mode:
            x_range, y_range, z_range = self.safe_x_range, self.safe_y_range, self.safe_z_range
        else:
            x_range, y_range, z_range = self.x_range, self.y_range, self.z_range
        
        return (x_range[0] <= x <= x_range[1] and
                y_range[0] <= y <= y_range[1] and
                z_range[0] <= z <= z_range[1])
    
    def get_safety_score(self, x: float, y: float, z: float) -> float:
        """Get safety score (0-1) for a position"""
        safe_center_x = (self.safe_x_range[0] + self.safe_x_range[1]) / 2
        safe_center_y = (self.safe_y_range[0] + self.safe_y_range[1]) / 2
        safe_center_z = (self.safe_z_range[0] + self.safe_z_range[1]) / 2
        
        # Normalized distances
        dx = abs(x - safe_center_x) / (self.safe_x_range[1] - self.safe_x_range[0])
        dy = abs(y - safe_center_y) / (self.safe_y_range[1] - self.safe_y_range[0])
        dz = abs(z - safe_center_z) / (self.safe_z_range[1] - self.safe_z_range[0])
        
        distance_score = 1.0 - min(1.0, np.sqrt(dx**2 + dy**2 + dz**2))
        return max(0.0, distance_score)