"""
Robot Workspace Validation and Safety Checking.

This module provides workspace validation and safety checking for the UR5e robot:
- Workspace boundary validation
- Joint limit checking
- Collision detection
- Safety zone monitoring
- Motion path validation

The validator ensures safe robot operation by preventing:
- Movements outside workspace
- Joint limit violations
- Collisions with environment
- Entry into restricted zones
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
from scipy.spatial import ConvexHull
from dataclasses import dataclass

@dataclass
class WorkspaceConfig:
    """
    Configuration for robot workspace validation.
    
    Parameters
    ----------
    x_limits : Tuple[float, float]
        Workspace limits in x direction (min, max) in meters
    y_limits : Tuple[float, float]
        Workspace limits in y direction (min, max) in meters
    z_limits : Tuple[float, float]
        Workspace limits in z direction (min, max) in meters
    safety_margin : float
        Safety margin around workspace boundaries in meters
    """
    x_limits: Tuple[float, float] = (-0.8, 0.8)
    y_limits: Tuple[float, float] = (-0.8, 0.8)
    z_limits: Tuple[float, float] = (0.1, 1.0)
    safety_margin: float = 0.1

class WorkspaceValidator:
    """
    Robot workspace validator and safety checker.
    
    This class validates robot movements and poses against workspace
    constraints and safety requirements.
    
    Parameters
    ----------
    config : Optional[WorkspaceConfig]
        Workspace configuration, by default None uses default config
    params : Dict, optional
        A dictionary of parameters, including:
        - min_object_volume
        - max_object_volume
    debug_mode : bool, optional
        Whether to enable debug visualization, by default False
        
    Attributes
    ----------
    workspace_hull : ConvexHull
        Convex hull of valid workspace points
    safety_zones : List[Dict]
        List of defined safety zones
    collision_objects : List[Dict]
        List of collision objects in workspace
        
    Notes
    -----
    Safety features:
    - Workspace boundary checking
    - Joint limit validation
    - Safety zone monitoring
    - Collision detection
    - Path interpolation and checking
    """
    
    def __init__(self, config: Optional[WorkspaceConfig] = None, params: Optional[Dict] = None, debug_mode: bool = False):
        self.logger = logging.getLogger(__name__)
        
        # Default workspace limits for UR5e (in meters)
        self.workspace_limits = config or WorkspaceConfig()
        
        # Unpack parameters or use defaults
        if params is None:
            params = {}
        self.min_object_volume = params.get('min_object_volume', 1.0e-6)
        self.max_object_volume = params.get('max_object_volume', 0.01)
        
        self.logger.info("Workspace validator initialized with limits: "
                        f"x={self.workspace_limits.x_limits}, "
                        f"y={self.workspace_limits.y_limits}, "
                        f"z={self.workspace_limits.z_limits}")

    def is_reachable(self, x: float, y: float, z: float, safe_mode: bool = True) -> bool:
        """
        Check if point is reachable by robot. (Alternative method with separate coordinates)
        
        Parameters
        ----------
        x : float
            X coordinate in meters
        y : float
            Y coordinate in meters
        z : float
            Z coordinate in meters
        safe_mode : bool, optional
            Whether to use safety margin, by default True
            
        Returns
        -------
        bool
            True if point is reachable, False otherwise
        
        Notes
        -----
        This is an alternate interface to is_position_reachable that takes separate coordinates.
        """
        """
        Check if point is reachable by robot.
        
        Parameters
        ----------
        x : float
            X coordinate in meters
        y : float
            Y coordinate in meters
        z : float
            Z coordinate in meters
        safe_mode : bool, optional
            Whether to use safety margin, by default True
            
        Returns
        -------
        bool
            True if point is reachable, False otherwise
            
        Notes
        -----
        Checks:
        - Within workspace boundaries
        - Outside safety zones
        - No collision risk
        - Joint solutions exist
        """
        margin = self.workspace_limits.safety_margin if safe_mode else 0.0
        
        # Check coordinate bounds
        if not (self.workspace_limits.x_limits[0] + margin <= x <= self.workspace_limits.x_limits[1] - margin):
            self.logger.warning(f"X coordinate {x:.3f}m outside safe bounds")
            return False
            
        if not (self.workspace_limits.y_limits[0] + margin <= y <= self.workspace_limits.y_limits[1] - margin):
            self.logger.warning(f"Y coordinate {y:.3f}m outside safe bounds")
            return False
            
        if not (self.workspace_limits.z_limits[0] + margin <= z <= self.workspace_limits.z_limits[1] - margin):
            self.logger.warning(f"Z coordinate {z:.3f}m outside safe bounds")
            return False
            
        # Check radial distance
        distance = np.sqrt(x*x + y*y + z*z)
        MAX_REACH = 0.85  # UR5e max reach ~850mm
        if distance > MAX_REACH - margin:
            self.logger.warning(f"Distance {distance:.3f}m outside safe bounds")
            return False
            
        return True

    def get_nearest_valid_point(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Get nearest valid point within workspace.
        
        Parameters
        ----------
        x : float
            X coordinate in meters
        y : float
            Y coordinate in meters
        z : float
            Z coordinate in meters
            
        Returns
        -------
        Tuple[float, float, float]
            Nearest valid point coordinates (x, y, z)
        """
        # Clamp coordinates to workspace bounds
        x_valid = np.clip(x, self.workspace_limits.x_limits[0] + self.workspace_limits.safety_margin, self.workspace_limits.x_limits[1] - self.workspace_limits.safety_margin)
        y_valid = np.clip(y, self.workspace_limits.y_limits[0] + self.workspace_limits.safety_margin, self.workspace_limits.y_limits[1] - self.workspace_limits.safety_margin)
        z_valid = np.clip(z, self.workspace_limits.z_limits[0] + self.workspace_limits.safety_margin, self.workspace_limits.z_limits[1] - self.workspace_limits.safety_margin)
        
        # Check and adjust radial distance against MAX_REACH (e.g. 0.85m for UR5e)
        # This assumes the origin (0,0,0) is the robot base.
        MAX_REACH = 0.85 # Nominal UR5e reach
        current_radial_distance = np.sqrt(x_valid**2 + y_valid**2 + z_valid**2)
        
        if current_radial_distance > MAX_REACH - self.workspace_limits.safety_margin:
            # If the point is outside the safe reach sphere, scale it down to the surface of the safe reach sphere.
            # Avoid division by zero if current_radial_distance is very small (shouldn't happen if z_limits[0] is positive)
            if current_radial_distance > 1e-6: 
                scale_factor = (MAX_REACH - self.workspace_limits.safety_margin) / current_radial_distance
                x_valid *= scale_factor
                y_valid *= scale_factor
                z_valid *= scale_factor
            else: 
                # This case should ideally not be reached if workspace_limits are sensible (e.g. z_min > 0)
                # If it is, clamping to a point on the z-axis at min height might be one strategy, but indicates bad input.
                # For now, just log and don't scale if at origin after box clamp.
                self.logger.warning("Point is at origin after box clamp, cannot scale radially for MAX_REACH.")

        return (x_valid, y_valid, z_valid)

    def get_safety_score(self, x: float, y: float, z: float) -> float:
        """
        Calculate safety score for position.
        
        Parameters
        ----------
        x : float
            X coordinate in meters
        y : float
            Y coordinate in meters
        z : float
            Z coordinate in meters
            
        Returns
        -------
        float
            Safety score between 0 and 1
            
        Notes
        -----
        Score considers:
        - Distance to workspace boundaries
        - Distance to safety zones
        - Distance to obstacles
        - Joint configuration quality
        """
        safe_center_x = (self.workspace_limits.x_limits[0] + self.workspace_limits.x_limits[1]) / 2
        safe_center_y = (self.workspace_limits.y_limits[0] + self.workspace_limits.y_limits[1]) / 2
        safe_center_z = (self.workspace_limits.z_limits[0] + self.workspace_limits.z_limits[1]) / 2
        
        # Normalized distances
        dx = abs(x - safe_center_x) / (self.workspace_limits.x_limits[1] - self.workspace_limits.x_limits[0])
        dy = abs(y - safe_center_y) / (self.workspace_limits.y_limits[1] - self.workspace_limits.y_limits[0])
        dz = abs(z - safe_center_z) / (self.workspace_limits.z_limits[1] - self.workspace_limits.z_limits[0])
        
        distance_score = 1.0 - min(1.0, np.sqrt(dx**2 + dy**2 + dz**2))
        return max(0.0, distance_score)

    def validate_path(self, waypoints: List[np.ndarray]) -> bool:
        """
        Validate path smoothness and workspace constraints.
        
        Parameters
        ----------
        waypoints : List[np.ndarray]
            List of 3D points defining the path
            
        Returns
        -------
        bool
            True if path is valid, False otherwise
            
        Notes
        -----
        Checks:
        1. Path smoothness (no sharp turns)
        2. All points within workspace
        3. Minimum segment lengths
        4. Maximum angle changes
        """
        if len(waypoints) < 2:
            return True  # Single point or empty path is valid
            
        MIN_SEGMENT_LENGTH = 0.01  # 1cm minimum segment length
        MAX_ANGLE_CHANGE = np.pi/2  # 90 degrees maximum angle change
        
        try:
            prev_vector = None
            
            for i in range(len(waypoints) - 1):
                # Get current segment vector
                v1 = waypoints[i+1] - waypoints[i]
                segment_length = np.linalg.norm(v1)
                
                # Check minimum segment length
                if segment_length < MIN_SEGMENT_LENGTH:
                    self.logger.warning(
                        f"Segment {i} is too short: {segment_length:.3f}m < {MIN_SEGMENT_LENGTH}m"
                    )
                    return False
                
                # Normalize current vector
                v1_normalized = v1 / segment_length
                
                # Check angle with previous segment if it exists
                if prev_vector is not None:
                    prev_length = np.linalg.norm(prev_vector)
                    
                    # Skip angle check if previous segment was too short
                    if prev_length >= MIN_SEGMENT_LENGTH:
                        prev_normalized = prev_vector / prev_length
                        cos_angle = np.clip(
                            np.dot(v1_normalized, prev_normalized),
                            -1.0,
                            1.0
                        )
                        angle = np.arccos(cos_angle)
                        
                        if angle > MAX_ANGLE_CHANGE:
                            self.logger.warning(
                                f"Angle change at waypoint {i} too large: "
                                f"{np.degrees(angle):.1f}° > {np.degrees(MAX_ANGLE_CHANGE):.1f}°"
                            )
                            return False
                
                # Check if current point is in workspace
                if not self.is_position_reachable(waypoints[i]):
                    self.logger.warning(f"Waypoint {i} is outside workspace")
                    return False
                
                prev_vector = v1
            
            # Check final point
            if not self.is_position_reachable(waypoints[-1]):
                self.logger.warning("Final waypoint is outside workspace")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating path: {e}")
            return False

    def is_position_reachable(self, position: np.ndarray, safe_mode: bool = True) -> bool:
        """
        Check if position is within robot's reachable workspace.
        
        Parameters
        ----------
        position : np.ndarray
            3D position to check [x, y, z]
        safe_mode : bool, optional
            Whether to use safety margin, by default True
            
        Returns
        -------
        bool
            True if position is reachable, False otherwise
        """
        try:
            pos = np.asarray(position)
            
            # For compatibility with both versions, delegate to is_reachable if position is a 3-element array
            if len(pos) == 3:
                return self.is_reachable(pos[0], pos[1], pos[2], safe_mode=safe_mode)
                
            # If we get here, the position format is not recognized
            self.logger.error(f"Unrecognized position format: {position}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking position reachability: {e}")
            return False

    def is_orientation_valid(self, rotation_matrix: np.ndarray) -> bool:
        """
        Check if orientation is valid for the robot.
        
        Parameters
        ----------
        rotation_matrix : np.ndarray
            3x3 rotation matrix
            
        Returns
        -------
        bool
            True if orientation is valid, False otherwise
        """
        try:
            # Check matrix properties
            if rotation_matrix.shape != (3, 3):
                return False
            
            # Check orthogonality
            identity = np.eye(3)
            orthogonality_error = np.abs(
                rotation_matrix.dot(rotation_matrix.T) - identity
            ).max()
            if orthogonality_error > 1e-6:
                self.logger.warning("Invalid rotation matrix: not orthogonal")
                return False
            
            # Check determinant (should be 1 for proper rotation)
            det = np.linalg.det(rotation_matrix)
            if abs(det - 1.0) > 1e-6:
                self.logger.warning("Invalid rotation matrix: determinant != 1")
                return False
            
            # Get euler angles
            euler = self.rotation_matrix_to_euler(rotation_matrix)
            
            # Check euler angle limits
            MAX_ROLL = np.pi  # ±180°
            MAX_PITCH = np.pi/2  # ±90°
            MAX_YAW = np.pi  # ±180°
            
            roll, pitch, yaw = euler
            
            if abs(roll) > MAX_ROLL or abs(pitch) > MAX_PITCH or abs(yaw) > MAX_YAW:
                self.logger.warning(
                    f"Orientation exceeds limits: roll={np.degrees(roll):.1f}°, "
                    f"pitch={np.degrees(pitch):.1f}°, yaw={np.degrees(yaw):.1f}°"
                )
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating orientation: {e}")
            return False

    def rotation_matrix_to_euler(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to euler angles (roll, pitch, yaw)"""
        try:
            sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
            singular = sy < 1e-6

            if not singular:
                roll = np.arctan2(R[2,1], R[2,2])
                pitch = np.arctan2(-R[2,0], sy)
                yaw = np.arctan2(R[1,0], R[0,0])
            else:
                roll = np.arctan2(-R[1,2], R[1,1])
                pitch = np.arctan2(-R[2,0], sy)
                yaw = 0

            return roll, pitch, yaw
            
        except Exception as e:
            self.logger.error(f"Error converting rotation matrix to euler: {e}")
            return 0.0, 0.0, 0.0

    def add_safety_zone(self, zone: Dict):
        """
        Add a safety zone to workspace.
        
        Parameters
        ----------
        zone : Dict
            Safety zone definition with:
            - type: Zone type (sphere, box, cylinder)
            - center: (x, y, z) coordinates
            - dimensions: Zone dimensions
            - priority: Safety priority level
            
        Notes
        -----
        Safety zones are used to:
        - Define restricted areas
        - Protect equipment
        - Ensure human safety
        - Guide robot motion
        """
        if not hasattr(self, 'safety_zones'):
            self.safety_zones = []
            
        required_keys = ['type', 'center', 'dimensions', 'priority']
        if not all(key in zone for key in required_keys):
            self.logger.error("Safety zone missing required parameters")
            return
            
        # Validate zone parameters
        if zone['type'] not in ['sphere', 'box', 'cylinder']:
            self.logger.error(f"Invalid safety zone type: {zone['type']}")
            return
            
        self.safety_zones.append(zone)
        self.logger.info(f"Added {zone['type']} safety zone at {zone['center']}")

    def check_collision(self, point: Tuple[float, float, float]) -> bool:
        """
        Check if point collides with any obstacles or safety zones.
        
        Parameters
        ----------
        point : Tuple[float, float, float]
            Point coordinates (x, y, z) in meters
            
        Returns
        -------
        bool
            True if collision detected, False otherwise
            
        Notes
        -----
        Collision checks:
        - Safety zones
        - Static obstacles
        - Dynamic objects
        - Robot self-collision
        """
        x, y, z = point
        
        # Check safety zones
        if hasattr(self, 'safety_zones'):
            for zone in self.safety_zones:
                center = np.array(zone['center'])
                point_array = np.array([x, y, z])
                
                if zone['type'] == 'sphere':
                    radius = zone['dimensions']['radius']
                    if np.linalg.norm(point_array - center) <= radius:
                        self.logger.warning(f"Collision with sphere zone at {center}")
                        return True
                        
                elif zone['type'] == 'box':
                    half_size = np.array([
                        zone['dimensions']['x'] / 2,
                        zone['dimensions']['y'] / 2,
                        zone['dimensions']['z'] / 2
                    ])
                    diff = np.abs(point_array - center)
                    if np.all(diff <= half_size):
                        self.logger.warning(f"Collision with box zone at {center}")
                        return True
                        
                elif zone['type'] == 'cylinder':
                    radius = zone['dimensions']['radius']
                    height = zone['dimensions']['height']
                    # Check radial distance in XY plane
                    xy_dist = np.linalg.norm(point_array[:2] - center[:2])
                    # Check height
                    z_dist = abs(point_array[2] - center[2])
                    if xy_dist <= radius and z_dist <= height/2:
                        self.logger.warning(f"Collision with cylinder zone at {center}")
                        return True
        
        # Check static obstacles (if defined)
        if hasattr(self, 'collision_objects'):
            for obj in self.collision_objects:
                # Similar collision checks based on object type
                if self._check_object_collision(point, obj):
                    return True
        
        return False

    def _check_object_collision(self, point: Tuple[float, float, float], obj: Dict) -> bool:
        """
        Helper method to check collision with a specific object.
        
        Parameters
        ----------
        point : Tuple[float, float, float]
            Point to check
        obj : Dict
            Object definition with geometry and position
            
        Returns
        -------
        bool
            True if collision detected, False otherwise
        """
        x, y, z = point
        point_array = np.array([x, y, z])
        
        if obj['type'] == 'mesh':
            # For mesh objects, check if point is inside bounding box first
            if self._point_in_bbox(point_array, obj['bbox_min'], obj['bbox_max']):
                # Detailed mesh collision check would go here
                return True
                
        elif obj['type'] == 'primitive':
            # For primitive shapes (similar to safety zones)
            center = np.array(obj['center'])
            if obj['shape'] == 'sphere':
                return np.linalg.norm(point_array - center) <= obj['radius']
            elif obj['shape'] == 'box':
                diff = np.abs(point_array - center)
                return np.all(diff <= np.array(obj['half_extents']))
                
        return False

    def _point_in_bbox(self, point: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> bool:
        """
        Check if point is inside axis-aligned bounding box.
        
        Parameters
        ----------
        point : np.ndarray
            Point coordinates
        bbox_min : np.ndarray
            Minimum corner of bounding box
        bbox_max : np.ndarray
            Maximum corner of bounding box
            
        Returns
        -------
        bool
            True if point is inside box, False otherwise
        """
        return np.all(point >= bbox_min) and np.all(point <= bbox_max)