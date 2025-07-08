"""
Depth-aware object detection with RealSense and stereo camera support.

This module provides depth-aware object detection capabilities using either:
- Intel RealSense D435/D455 cameras
- Stereo camera pairs

The module handles:
- Camera setup and configuration
- Depth frame acquisition and processing
- 3D object detection and localization
- Point cloud generation and processing
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import threading

# Import CameraCalibration if not already (it should be in the same directory)
from CameraCalibration import CameraCalibration
from WorkSpaceValidator import WorkspaceValidator # For type hint in validate_grasp_3d
from GraspPointDetector import GraspPointDetector # For type hint in augment_vlm_detections_3d

@dataclass
class Detection3D:
    """
    3D detection result with accurate depth information.
    
    Attributes
    ----------
    label : str
        Object class label
    confidence : float
        Detection confidence score
    bbox_2d : List[int]
        2D bounding box [x1, y1, x2, y2]
    center_3d : Tuple[float, float, float]
        3D center point (x, y, z) in meters
    bbox_3d : Dict[str, float]
        3D bounding box dimensions
    grasp_point_3d : Tuple[float, float, float]
        3D grasp point coordinates
    depth_confidence : float
        Confidence in depth measurement
    volume : float
        Estimated object volume
    approach_vector : Tuple[float, float, float]
        Gripper approach direction
    """

    label: str
    confidence: float
    bbox_2d: List[int]  # [x1, y1, x2, y2]
    center_3d: Tuple[float, float, float]  # (x, y, z) in meters
    bbox_3d: Dict[str, float]  # 3D bounding box dimensions
    grasp_point_3d: Tuple[float, float, float]
    depth_confidence: float
    volume: float  # Object volume estimate
    approach_vector: Tuple[float, float, float]  # Gripper approach direction


class DepthAwareDetector:
    """
    Depth-aware object detection system.
    
    This class provides functionality for detecting objects in 3D space using
    depth information from either RealSense or stereo cameras.
    
    Parameters
    ----------
    calibration : CameraCalibration
        Camera calibration object containing necessary parameters
    params : Dict, optional
        A dictionary of parameters, including:
        - use_filters: bool
        - min_valid_pixels: int
        - max_depth_variance: float
        
    Attributes
    ----------
    depth_scale : float
        Scale factor to convert depth values to meters
    color_intrinsics : rs.intrinsics
        Color camera intrinsic parameters
    depth_intrinsics : rs.intrinsics
        Depth camera intrinsic parameters
    depth_to_color_extrin : rs.extrinsics
        Extrinsic transformation between depth and color cameras
    """

    def __init__(self, 
                 calibration: CameraCalibration, 
                 params: Optional[Dict] = None):
        """Initialize depth-aware detection with RealSense D435i optimized settings."""
        self.logger = logging.getLogger(__name__)
        
        # Unpack parameters or use defaults
        if params is None:
            params = {}
        self.use_filters = params.get('use_filters', True)
        self.min_valid_pixels = params.get('min_valid_pixels', 100)
        self.depth_variance_threshold = params.get('max_depth_variance', 0.015)

        self.debug_mode = False # Or make this a parameter
        self.calibration = calibration

        # Get parameters from the CameraCalibration object
        if not self.calibration.camera_matrix_color:
            self.logger.error("Color intrinsics not available. Depth processing will fail.")
            raise ValueError("Color camera intrinsics required for depth processing")

        self.color_intrinsics = self.calibration.get_color_intrinsics_rs()
        self.depth_intrinsics = self.calibration.get_depth_intrinsics_rs()
        self.depth_to_color_extrin = self.calibration.get_depth_to_color_extrinsics_rs()
        self.depth_scale = self.calibration.get_depth_scale_rs()

        if None in [self.color_intrinsics, self.depth_intrinsics, self.depth_to_color_extrin, self.depth_scale]:
            self.logger.error("Critical calibration parameters missing. Check RealSense initialization.")
            raise ValueError("Missing critical calibration parameters")

        # D435i-optimized depth parameters
        self.max_depth = 2.0  # Maximum reliable depth for D435i in meters
        self.min_depth = 0.2  # Minimum reliable depth for D435i in meters
        
        if self.use_filters and self.calibration.source_type == "realsense":
            # Spatial filter - reduces noise while preserving edges
            self.spatial_filter = rs.spatial_filter()
            self.spatial_filter.set_option(rs.option.filter_magnitude, 2)
            self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
            self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
            self.spatial_filter.set_option(rs.option.holes_fill, 1)

            # Temporal filter - reduces temporal noise
            self.temporal_filter = rs.temporal_filter()
            self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
            self.temporal_filter.set_option(rs.option.filter_smooth_delta, 20)
            self.temporal_filter.set_option(rs.option.holes_fill, 3)

            # Hole filling - fills small holes in depth data
            self.hole_filling_filter = rs.hole_filling_filter()
            self.hole_filling_filter.set_option(rs.option.holes_fill, 1)

            # Decimation filter - reduces resolution for better quality
            self.decimation_filter = rs.decimation_filter()
            self.decimation_filter.set_option(rs.option.filter_magnitude, 2)

            self.logger.info("RealSense D435i depth filters initialized with optimized settings")
        else:
            self.spatial_filter = None
            self.temporal_filter = None
            self.hole_filling_filter = None
            self.decimation_filter = None

        # Depth reliability thresholds for D435i (now from params)
        self.edge_threshold = 0.01  # 1cm threshold for edge detection

        self.logger.info(
            f"✅ Depth-aware detection initialized for {self.calibration.source_type}"
            f" (filters: {self.use_filters})"
        )

    def _setup_realsense(self):
        """DEPRECATED: RealSense pipeline is managed externally by UnifiedVisionSystem.
        This class now receives calibration parameters and frames directly.
        """
        self.logger.warning("_setup_realsense is deprecated and should not be called.")
        # Original content commented out or removed as pipeline is external
        pass

    def _setup_stereo_camera(self):
        """DEPRECATED or needs rework if stereo is used without internal pipeline management.
        Set up stereo camera system.
        
        This function:
        - Initializes left and right cameras
        - Sets up stereo matching parameters
        - Loads stereo calibration data
        
        Notes
        -----
        Currently a placeholder implementation for future stereo support
        """
        # This would implement stereo vision depth calculation
        # For now, placeholder implementation
        self.left_camera = cv2.VideoCapture(0)
        self.right_camera = cv2.VideoCapture(1)

        # Stereo calibration parameters (would be loaded from calibration)
        self.stereo_map_left = None
        self.stereo_map_right = None
        self.stereo_matcher = cv2.StereoBM_create(numDisparities=16 * 5, blockSize=21)

        self.logger.info("📷 Stereo camera system initialized")

    def get_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """DEPRECATED: Frames are now passed directly to processing methods like augment_vlm_detections_3d.
        Get synchronized color and depth frames.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Color and depth frames as numpy arrays
            
        Notes
        -----
        For RealSense, uses hardware synchronization
        For stereo, performs software synchronization
        """
        self.logger.warning("get_frames is deprecated. Frames should be passed to processing methods.")
        raise NotImplementedError("get_frames is deprecated and should not be used.")

    def _get_realsense_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """DEPRECATED.
        Get frames from RealSense camera.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Color and filtered depth frames
            
        Notes
        -----
        Applies temporal, spatial, and hole-filling filters to depth
        """
        self.logger.warning("_get_realsense_frames is deprecated.")
        raise NotImplementedError("_get_realsense_frames is deprecated and should not be used.")

    def _get_stereo_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """DEPRECATED.
        Get frames from stereo camera and compute depth"""
        ret_left, left_frame = self.left_camera.read()
        ret_right, right_frame = self.right_camera.read()

        if not ret_left or not ret_right:
            return None, None

        # Compute disparity map
        gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        disparity = self.stereo_matcher.compute(gray_left, gray_right)

        depth_image = disparity.astype(np.float32)

        return left_frame, depth_image

    def detect_objects_3d(self, object_detector, grasp_detector) -> List[Detection3D]:
        """
        Perform 3D object detection with accurate depth information.
        
        Parameters
        ----------
        object_detector : Any
            2D object detection system (YOLO/VLM)
        grasp_detector : Any
            Grasp point detection system
            
        Returns
        -------
        List[Detection3D]
            List of 3D detections with accurate coordinates
        """
        color_frame, depth_frame = self.get_frames()

        if color_frame is None or depth_frame is None:
            return []

        # Get 2D detections
        detections_2d = object_detector.detect_objects(color_frame)

        detections_3d = []

        for label, confidence, bbox_2d in detections_2d:
            # Get grasp point using your existing grasp detector
            grasp_info = grasp_detector.find_grasp_points(color_frame, bbox_2d)

            detection_3d = self._calculate_3d_properties(
                label, confidence, bbox_2d, depth_frame, grasp_info
            )

            if detection_3d:
                detections_3d.append(detection_3d)

        return detections_3d

    def augment_vlm_detections_3d(
        self,
        color_image_np: np.ndarray,
        depth_frame_rs: Optional[rs.frame],
        raw_depth_image_np: np.ndarray,
        vlm_detections: List[Tuple[str, float, List[int]]],
        grasp_detector: GraspPointDetector,
        current_gripper_pose: Optional[np.ndarray] = None
    ) -> List[Detection3D]:
        """
        Augment 2D detections with depth information and 3D properties.
        Now handles eye-in-hand configuration with dynamic camera pose.
        
        Parameters
        ----------
        color_image_np : np.ndarray
            Color image in BGR format
        depth_frame_rs : Optional[rs.frame]
            Raw RealSense depth frame for filtering
        raw_depth_image_np : np.ndarray
            Raw depth image as numpy array
        vlm_detections : List[Tuple[str, float, List[int]]]
            List of 2D detections (label, confidence, bbox)
        grasp_detector : GraspPointDetector
            Grasp point detection instance
        current_gripper_pose : Optional[np.ndarray]
            Current gripper pose for eye-in-hand configuration
            
        Returns
        -------
        List[Detection3D]
            List of 3D detections with depth information
        """
        if len(vlm_detections) == 0:
            return []

        # Apply RealSense depth filters if available
        if depth_frame_rs and self.use_filters:
            filtered_depth = self._apply_depth_filters(depth_frame_rs)
        else:
            filtered_depth = raw_depth_image_np

        detections_3d = []
        
        for label, confidence, bbox_2d in vlm_detections:
            # Get grasp information
            x1, y1, x2, y2 = bbox_2d
            roi = color_image_np[y1:y2, x1:x2]
            grasp_info = grasp_detector.detect_grasp_point(roi)
            
            # Calculate 3D properties with filtered depth
            detection_3d = self._calculate_3d_properties(
                label, confidence, bbox_2d, filtered_depth, grasp_info, current_gripper_pose
            )
            
            if detection_3d:
                detections_3d.append(detection_3d)
                self.logger.info(f"Created 3D detection for {label} at {detection_3d.grasp_point_3d}")
        
        return detections_3d
    
    def _apply_depth_filters(self, depth_frame: rs.frame) -> np.ndarray:
            """Apply optimized D435i depth filters in sequence."""
            if not self.use_filters or not all([
                self.decimation_filter,
                self.spatial_filter,
                self.temporal_filter,
                self.hole_filling_filter
            ]):
                return np.asanyarray(depth_frame.get_data())

            filtered = self.decimation_filter.process(depth_frame)
            filtered = self.spatial_filter.process(filtered)
            filtered = self.temporal_filter.process(filtered)
            filtered = self.hole_filling_filter.process(filtered)
            
            return np.asanyarray(filtered.get_data())

    def _calculate_3d_properties(
        self,
        label: str,
        confidence: float,
        bbox_2d: List[int],
        depth_frame: np.ndarray,
        grasp_info: Dict,
        current_gripper_pose: Optional[np.ndarray] = None  # NEW PARAMETER
    ) -> Optional[Detection3D]:
        """Calculate 3D properties from depth data with eye-in-hand support."""
        try:
            x1, y1, x2, y2 = bbox_2d
            
            # Get depth ROI
            depth_roi = depth_frame[y1:y2, x1:x2] * self.depth_scale
            
            valid_mask = (depth_roi > self.min_depth) & (depth_roi < self.max_depth)
            
            if np.sum(valid_mask) < self.min_valid_pixels:
                self.logger.warning(f"Insufficient valid depth pixels for {label}")
                return None

            valid_depths = depth_roi[valid_mask]
            center_depth = np.median(valid_depths)
            
            # Calculate center in image coordinates
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Convert to 3D coordinates in camera frame
            center_3d = self._pixel_to_3d(center_x, center_y, center_depth)
            if not center_3d:
                return None

            bbox_3d = self._estimate_3d_bbox(bbox_2d, depth_roi, valid_mask)
            
            grasp_pixel_x = x1 + int(grasp_info['x'] * (x2 - x1))
            grasp_pixel_y = y1 + int(grasp_info['y'] * (y2 - y1))
            grasp_depth = self._get_safe_depth_at_point(
                depth_frame, grasp_pixel_x, grasp_pixel_y
            )
            
            if grasp_depth is None:
                self.logger.warning(f"Could not determine grasp point depth for {label}")
                return None
                
            grasp_point_3d = self._pixel_to_3d(
                grasp_pixel_x, grasp_pixel_y, grasp_depth
            )
            if not grasp_point_3d:
                return None

            # Calculate approach vector - adjusted for eye-in-hand
            approach_vector = self._calculate_approach_vector_eye_in_hand(
                grasp_info, grasp_point_3d, current_gripper_pose
            )

            depth_confidence = self._calculate_depth_confidence(
                depth_roi, valid_mask
            )

            return Detection3D(
                label=label,
                confidence=confidence,
                bbox_2d=bbox_2d,
                center_3d=center_3d,
                bbox_3d=bbox_3d,
                grasp_point_3d=grasp_point_3d,
                depth_confidence=depth_confidence,
                volume=bbox_3d['width'] * bbox_3d['height'] * bbox_3d['depth'],
                approach_vector=approach_vector
            )

        except Exception as e:
            self.logger.error(f"Error calculating 3D properties for {label}: {e}")
            return None

    def _pixel_to_3d(
        self, pixel_x: int, pixel_y: int, depth: float
    ) -> Tuple[float, float, float]:
        """
        Convert pixel coordinates to 3D world coordinates.
        
        Parameters
        ----------
        pixel_x : int
            X coordinate in image space
        pixel_y : int
            Y coordinate in image space
        depth : float
            Depth value in meters
            
        Returns
        -------
        Tuple[float, float, float]
            3D point in camera coordinate frame (x, y, z) in meters
            
        Notes
        -----
        Uses camera intrinsics for accurate 3D projection.
        For RealSense, uses rs. intrinsics and rs2_deproject_pixel_to_point.
        For other sources, uses self.calibration.pixel_to_camera.
        """
        if self.calibration.source_type == "realsense" and self.color_intrinsics is not None:
            try:
                # Ensure pixel_x, pixel_y are int for this function's list conversion
                # rs2_deproject_pixel_to_point expects depth in meters.
                point_3d_tuple = rs.rs2_deproject_pixel_to_point(self.color_intrinsics, [int(pixel_x), int(pixel_y)], depth)
                return point_3d_tuple # (x, y, z) tuple
            except Exception as e:
                self.logger.error(f"Error using rs2_deproject_pixel_to_point: {e}")
                # Fall through to generic method if RealSense call fails
        
        # Fallback to generic method using CameraCalibration class logic
        # This uses the numpy camera_matrix and handles undistortion if necessary.
        point_3d_fallback = self.calibration.pixel_to_camera(pixel_x, pixel_y, depth)
        if point_3d_fallback is None or (point_3d_fallback[0]==0.0 and point_3d_fallback[1]==0.0 and point_3d_fallback[2]==0.0 and depth > 0):
             self.logger.warning(f"Fallback self.calibration.pixel_to_camera failed or returned zero for pixel ({pixel_x},{pixel_y}) at depth {depth}.")
             return None # Explicitly return None if fallback fails to produce a non-zero point for valid depth
        return point_3d_fallback

    def _estimate_3d_bbox(
        self, bbox_2d: List[int], depth_roi: np.ndarray, valid_mask: np.ndarray
    ) -> Dict[str, float]:
        """Estimate 3D bounding box dimensions"""
        x1, y1, x2, y2 = bbox_2d

        # Get depth statistics for different regions
        if np.any(valid_mask):
            min_depth = np.min(depth_roi[valid_mask])
            max_depth = np.max(depth_roi[valid_mask])
            avg_depth = np.mean(depth_roi[valid_mask])
        else:
            min_depth = max_depth = avg_depth = 0.5  # Default fallback

        # Estimate 3D dimensions using average depth
        width_pixels = x2 - x1
        height_pixels = y2 - y1

        if self.calibration.source_type == "realsense" and self.color_intrinsics is not None:
            fx = self.color_intrinsics.fx
            fy = self.color_intrinsics.fy

            avg_depth_meters = avg_depth * self.depth_scale
            width_meters = width_pixels * avg_depth_meters / fx
            height_meters = height_pixels * avg_depth_meters / fy
        else:
            # Use generic intrinsics if available
            if self.calibration.camera_matrix_color is not None:
                fx = self.calibration.camera_matrix_color[0,0]
                fy = self.calibration.camera_matrix_color[1,1]
                avg_depth_meters = avg_depth * (self.depth_scale if self.depth_scale is not None else 0.001) # Best guess for scale
                width_meters = width_pixels * avg_depth_meters / fx
                height_meters = height_pixels * avg_depth_meters / fy
            else:
                # Rough approximation for other cameras if no intrinsics at all
                self.logger.warning("Estimating 3D bbox without camera intrinsics. Results will be very approximate.")
                avg_depth_meters = avg_depth * (self.depth_scale if self.depth_scale is not None else 0.001)
                width_meters = width_pixels * avg_depth_meters * 0.001 # Very rough guess
                height_meters = height_pixels * avg_depth_meters * 0.001

        # Assuming min_depth and max_depth from depth_roi are also raw depth values
        min_depth_meters = min_depth * self.depth_scale if self.depth_scale else min_depth * 0.001
        max_depth_meters = max_depth * self.depth_scale if self.depth_scale else max_depth * 0.001
        depth_dim_meters = abs(max_depth_meters - min_depth_meters)

        return {
            "width": abs(width_meters),
            "height": abs(height_meters),
            "depth": abs(depth_dim_meters),
            "min_depth_meters": min_depth_meters,
            "max_depth_meters": max_depth_meters,
            "avg_depth_meters": avg_depth_meters,
        }

    def _get_safe_depth_at_point(
        self, depth_frame: np.ndarray, x: int, y: int, window_size: int = 5
    ) -> float:
        """Get reliable depth at a specific point using local averaging"""
        h, w = depth_frame.shape

        # Ensure coordinates are valid
        x = max(window_size // 2, min(w - window_size // 2 - 1, x))
        y = max(window_size // 2, min(h - window_size // 2 - 1, y))

        x1 = x - window_size // 2
        x2 = x + window_size // 2 + 1
        y1 = y - window_size // 2
        y2 = y + window_size // 2 + 1

        # depth_frame here is expected to be raw (e.g. uint16 for Z16)
        # so we scale it with self.depth_scale (which should be ~0.001 for mm to m)
        if self.depth_scale is None:
            self.logger.error("Depth scale (depth_scale) is not available in _get_safe_depth_at_point. Cannot reliably process depth.")
            # Attempt to access depth_frame directly, hoping it's already scaled or _get_safe_depth_at_point is called with scaled data
            # This is a fallback, proper depth_scale is crucial.
            local_depths_meters = depth_frame[y1:y2, x1:x2] 
        else:
            local_depths_meters = depth_frame[y1:y2, x1:x2] * self.depth_scale

        # Filter valid depths (now in meters)
        valid_depths_meters = local_depths_meters[
            (local_depths_meters > self.min_depth) & (local_depths_meters < self.max_depth)
        ]

        if len(valid_depths_meters) > 0:
            return np.median(valid_depths_meters)  # Use median for robustness (returns meters)
        else:
            # Fallback to single pixel, scaled to meters
            if self.depth_scale is None:
                self.logger.warning("Depth scale missing, returning raw center pixel value from depth_frame in _get_safe_depth_at_point fallback.")
                return depth_frame[y,x] # This would be raw, potentially problematic
            return depth_frame[y, x] * self.depth_scale # (returns meters)

    def _calculate_approach_vector_eye_in_hand(
        self, 
        grasp_info: Dict, 
        grasp_point_3d: Tuple[float, float, float],
        current_gripper_pose: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float]:
        """
        Calculate gripper approach vector for eye-in-hand configuration.
        
        Parameters
        ----------
        grasp_info : Dict
            Grasp information including approach_angle
        grasp_point_3d : Tuple[float, float, float]
            3D grasp point coordinates in camera frame
        current_gripper_pose : Optional[np.ndarray]
            Current gripper pose (not used here but available for future enhancements)
            
        Returns
        -------
        Tuple[float, float, float]
            Normalized approach vector in camera frame
            
        Notes
        -----
        For eye-in-hand configuration where camera looks down at workspace:
        - Default approach is straight down (negative Z in camera frame)
        - Approach angle controls the horizontal tilt
        - Vector is in camera frame and will be transformed to robot frame by caller
        """
        if not grasp_info or "approach_angle" not in grasp_info:
            # Default straight down approach for eye-in-hand
            return (0.0, 0.0, -1.0)

        angle = grasp_info["approach_angle"]

        # For eye-in-hand camera looking down at workspace:
        # - Camera Z-axis points away from camera (down towards workspace)
        # - Camera X and Y define the horizontal plane
        # - We want mostly vertical approach with optional slight tilt
        
        tilt_angle = 0.2  # Maximum tilt from vertical (radians, ~11 degrees)
        
        # Horizontal components based on grasp angle
        approach_x = np.sin(tilt_angle) * np.cos(angle)
        approach_y = np.sin(tilt_angle) * np.sin(angle)
        approach_z = -np.cos(tilt_angle)  # Negative Z is towards workspace

        # Normalize vector
        magnitude = np.sqrt(approach_x**2 + approach_y**2 + approach_z**2)
        if magnitude > 0:
            return (approach_x / magnitude, approach_y / magnitude, approach_z / magnitude)
        else:
            return (0.0, 0.0, -1.0)

    def get_point_cloud(self, color_frame: np.ndarray = None, depth_frame: np.ndarray = None, max_distance: float = 1.0) -> Optional[np.ndarray]:
        """Generate point cloud from provided depth frame or fetch if not provided.
        
        Parameters
        ----------
        color_frame : np.ndarray, optional
            Color frame for point cloud mapping
        depth_frame : np.ndarray, optional
            Depth frame for point cloud generation
        max_distance : float, optional
            Maximum distance for point cloud filtering, by default 1.0
            
        Returns
        -------
        Optional[np.ndarray]
            Point cloud as numpy array, or None if generation fails
        """
        if color_frame is None or depth_frame is None:
            self.logger.warning("get_point_cloud requires frames to be provided externally.")
            return None

        if self.calibration.source_type == "realsense":
            # Use RealSense built-in point cloud generation
            pc = rs.pointcloud()
            # Since frames are provided as numpy arrays, we need to handle them differently
            # This part might need adjustment based on how frames are passed
            self.logger.warning("Point cloud generation from numpy arrays is not fully implemented.")
            return None

        return None

    def validate_grasp_3d(
        self, 
        detection: Detection3D, 
        workspace_validator: WorkspaceValidator,
        current_gripper_pose: Optional[np.ndarray] = None
    ) -> bool:
        """
        Validate if 3D detection is graspable within workspace.
        Now handles eye-in-hand configuration.
        
        Parameters
        ----------
        detection : Detection3D
            3D detection to validate
        workspace_validator : WorkspaceValidator
            Workspace validation system
        current_gripper_pose : Optional[np.ndarray]
            Current gripper pose for eye-in-hand transformation
            
        Returns
        -------
        bool
            True if detection is valid and graspable
        """
        # For eye-in-hand, we need to transform the grasp point to robot frame
        if self.calibration.is_eye_in_hand and current_gripper_pose is not None:
            # Transform grasp point from camera to robot frame
            grasp_point_robot = self.calibration.camera_to_robot(
                detection.grasp_point_3d, current_gripper_pose
            )
            x, y, z = grasp_point_robot
        else:
            x, y, z = detection.grasp_point_3d

        # Check workspace constraints
        if not workspace_validator.is_reachable(x, y, z, safe_mode=True):
            self.logger.warning(f"Position {x, y, z} is outside workspace bounds")
            return False

        # Check depth confidence
        if detection.depth_confidence < 0.5:
            self.logger.warning(
                f"Low depth confidence for {detection.label}: {detection.depth_confidence:.2f}"
            )
            return False

        if not (workspace_validator.min_object_volume <= detection.volume <= workspace_validator.max_object_volume):
            self.logger.warning(
                f"Object {detection.label} size may be problematic: {detection.volume:.6f}m³"
            )
            return False

        # Additional validation for eye-in-hand
        if self.calibration.is_eye_in_hand:
            # Check if the approach vector makes sense for eye-in-hand
            # Should be mostly downward
            approach_z = detection.approach_vector[2]
            if approach_z > -0.7:  # cos(45°) ≈ 0.7
                self.logger.warning(
                    f"Approach vector not suitable for eye-in-hand: z={approach_z:.2f}"
                )
                return False

        return True

    def visualize_3d_detection(
        self, frame: np.ndarray, detection: Detection3D
    ) -> np.ndarray:
        """Visualize 3D detection information on frame with eye-in-hand awareness"""
        x1, y1, x2, y2 = detection.bbox_2d

        # Draw 2D bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw grasp point
        grasp_2d = self._project_3d_to_2d(detection.grasp_point_3d)
        if grasp_2d:
            gx, gy = grasp_2d
            cv2.circle(frame, (int(gx), int(gy)), 8, (0, 0, 255), -1)

            # Draw approach vector (adjusted for eye-in-hand view)
            # For eye-in-hand, the approach is typically more vertical
            approach_scale = 30 if self.calibration.is_eye_in_hand else 50
            approach_end = (
                gx + detection.approach_vector[0] * approach_scale,
                gy + detection.approach_vector[1] * approach_scale,
            )
            cv2.arrowedLine(
                frame,
                (int(gx), int(gy)),
                (int(approach_end[0]), int(approach_end[1])),
                (255, 0, 0),
                2,
            )

        # Add 3D information text with eye-in-hand indicator
        config_text = "Eye-in-hand" if self.calibration.is_eye_in_hand else "Eye-to-hand"
        info_text = [
            f"{detection.label} ({detection.confidence:.2f}) - {config_text}",
            f"3D: ({detection.grasp_point_3d[0]:.3f}, {detection.grasp_point_3d[1]:.3f}, {detection.grasp_point_3d[2]:.3f})",
            f"Depth Conf: {detection.depth_confidence:.2f}",
            f"Volume: {detection.volume*1000000:.1f}cm³",
        ]

        for i, text in enumerate(info_text):
            cv2.putText(
                frame,
                text,
                (x1, y1 - 10 - i * 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        return frame

    def _project_3d_to_2d(
        self, point_3d: Tuple[float, float, float]
    ) -> Optional[Tuple[int, int]]:
        """
        Project 3D point back to 2D pixel coordinates.
        
        Parameters
        ----------
        point_3d : Tuple[float, float, float]
            3D point in camera coordinate frame (x, y, z) in meters
            
        Returns
        -------
        Optional[Tuple[int, int]]
            Pixel coordinates (x, y), or None if projection fails
            
        Notes
        -----
        Checks for points behind camera (z <= 0)
        Uses camera intrinsics for accurate projection
        """
        if self.calibration.source_type == "realsense":
            x, y, z = point_3d

            if z <= 0:
                return None

            fx = self.color_intrinsics.fx
            fy = self.color_intrinsics.fy
            cx = self.color_intrinsics.ppx
            cy = self.color_intrinsics.ppy

            pixel_x = int(x * fx / z + cx)
            pixel_y = int(y * fy / z + cy)

            return (pixel_x, pixel_y)

        return None

    def cleanup(self):
        """
        Clean up resources and stop camera streams.
        
        This function:
        - Stops RealSense pipeline if active
        - Releases stereo camera resources if active
        - Logs cleanup status
        
        Notes
        -----
        Should be called when object detection is complete
        """
        if self.calibration.source_type == "realsense" and hasattr(self, "pipeline"):
            self.pipeline.stop()
        elif self.calibration.source_type == "stereo":
            if hasattr(self, "left_camera"):
                self.left_camera.release()
            if hasattr(self, "right_camera"):
                self.right_camera.release()

        self.logger.info("📷 Camera resources cleaned up")

    def _calculate_depth_confidence(self, depth_frame: np.ndarray, bbox_2d: List[int]) -> float:
        """
        Calculate confidence score for depth measurements in a region.
        
        Parameters
        ----------
        depth_frame : np.ndarray
            Depth frame aligned with color image
        bbox_2d : List[int]
            2D bounding box coordinates [x1, y1, x2, y2]
            
        Returns
        -------
        float
            Confidence score between 0 and 1
            
        Notes
        -----
        Factors considered:
        1. Percentage of valid depth measurements
        2. Depth consistency (low variance)
        3. Edge depth discontinuity
        """
        x1, y1, x2, y2 = bbox_2d
        
        depth_roi = depth_frame[y1:y2, x1:x2]
        
        # No valid measurements
        if depth_roi.size == 0:
            return 0.0
            
        valid_mask = (depth_roi > 0) & (depth_roi * self.depth_scale < self.max_depth)
        valid_percentage = np.sum(valid_mask) / depth_roi.size
        
        if valid_percentage < 0.3:  # Require at least 30% valid measurements
            return 0.0
            
        valid_depths = depth_roi[valid_mask] * self.depth_scale
        depth_std = np.std(valid_depths)
        consistency_score = np.exp(-depth_std / 0.1)  # Penalize high variance
        
        # Check edge discontinuity
        edge_kernel = np.array([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]])
        edges = cv2.filter2D((depth_roi * self.depth_scale).astype(np.float32), -1, edge_kernel)
        edge_score = 1.0 - min(1.0, np.mean(np.abs(edges)) / 0.1)
        
        # Combine scores with weights
        confidence = (0.4 * valid_percentage + 
                     0.4 * consistency_score +
                     0.2 * edge_score)
        
        return float(np.clip(confidence, 0.0, 1.0))
