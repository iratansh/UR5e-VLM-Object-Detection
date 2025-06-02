"""
Camera calibration and coordinate transformation utilities.

This module provides functionality for camera calibration, hand-eye calibration,
and coordinate transformations between camera, robot, and pixel spaces.

The module handles:
- Camera intrinsic parameters (focal length, principal point)
- Distortion coefficients
- Hand-eye calibration between robot and camera
- 3D-2D projections and transformations
- TF2 broadcasting of camera-robot transforms
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
import pyrealsense2 as rs
import logging
from pathlib import Path
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
import tf2_ros
import math

class CameraCalibration:
    """
    Camera calibration and coordinate transformation handler.
    
    This class manages camera calibration parameters and provides
    utilities for transforming between different coordinate frames.
    
    Parameters
    ----------
    node : Optional[Node]
        ROS2 node for transform broadcasting. If None, transforms won't be broadcast.
    hand_eye_file : str, optional
        Path to hand-eye calibration file, by default "hand_eye_calib.npz"
    camera_info_file : str, optional
        Path to camera intrinsics file, by default "camera_info.npz"
        
    Attributes
    ----------
    camera_matrix : np.ndarray
        3x3 camera intrinsic matrix
    dist_coeffs : np.ndarray
        Distortion coefficients
    T_base_to_camera : np.ndarray
        4x4 transformation matrix from robot base to camera frame
    """
    
    def __init__(self, node: Optional[Node] = None):
        """
        Initialize camera calibration with optional ROS2 node for transform broadcasting.
        
        Parameters
        ----------
        node : Optional[Node]
            ROS2 node for transform broadcasting. If None, transforms won't be broadcast.
        """
        self.logger = logging.getLogger(__name__)
        self.node = node
        
        # Initialize transform broadcasters if ROS2 node is provided
        if self.node:
            self.tf_broadcaster = TransformBroadcaster(self.node)
            self.static_tf_broadcaster = StaticTransformBroadcaster(self.node)
        
        # Generic calibration parameters (from file or RealSense)
        self.camera_matrix_color: Optional[np.ndarray] = None # For color camera
        self.dist_coeffs_color: Optional[np.ndarray] = None # For color camera
        
        # Hand-eye calibration
        self.T_base_to_camera: Optional[np.ndarray] = np.eye(4) # Default to identity
        self.T_camera_to_base: Optional[np.ndarray] = np.eye(4) # Default to identity

        # RealSense specific parameters (if applicable)
        self.color_intrinsics_rs: Optional[rs.intrinsics] = None
        self.depth_intrinsics_rs: Optional[rs.intrinsics] = None
        self.depth_to_color_extrinsics_rs: Optional[rs.extrinsics] = None
        self.depth_scale_rs: Optional[float] = None
        
        self.source_type: str = "uninitialized" # e.g., "realsense", "file"

        # Frame IDs for TF2
        self.camera_frame_id = "camera_color_optical_frame"
        self.depth_frame_id = "camera_depth_optical_frame"
        self.camera_link_frame_id = "camera_link"
        self.robot_base_frame_id = "base_link"

        self.logger.info("CameraCalibration object created. Load parameters as needed.")

    # --- Methods to populate from RealSense pipeline ---
    def set_realsense_calibration_params(self, profile: rs.pipeline_profile):
        """
        Set calibration parameters from RealSense pipeline profile.
        
        Parameters
        ----------
        profile : rs.pipeline_profile
            Active RealSense pipeline profile
        """
        if not isinstance(profile, rs.pipeline_profile):
            raise ValueError("Invalid RealSense profile provided")

        try:
            # Get color stream profile and intrinsics
            color_stream = profile.get_stream(rs.stream.color)
            if not color_stream:
                raise ValueError("Color stream not found in RealSense profile")
                
            self.color_intrinsics_rs = color_stream.as_video_stream_profile().get_intrinsics()
            
            # Convert to numpy format
            self.camera_matrix_color = np.array([
                [self.color_intrinsics_rs.fx, 0, self.color_intrinsics_rs.ppx],
                [0, self.color_intrinsics_rs.fy, self.color_intrinsics_rs.ppy],
                [0, 0, 1]
            ])
            self.dist_coeffs_color = np.array(self.color_intrinsics_rs.coeffs)

            # Get depth stream profile and intrinsics
            depth_stream = profile.get_stream(rs.stream.depth)
            if not depth_stream:
                raise ValueError("Depth stream not found in RealSense profile")
                
            self.depth_intrinsics_rs = depth_stream.as_video_stream_profile().get_intrinsics()
            
            # Get depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale_rs = depth_sensor.get_depth_scale()

            # Get extrinsics (depth to color)
            self.depth_to_color_extrinsics_rs = depth_stream.get_extrinsics_to(color_stream)
            
            # Broadcast static transforms between camera frames
            if self.node:
                self._broadcast_camera_transforms()
            
            self.source_type = "realsense"
            self.logger.info(
                f"RealSense calibration parameters set successfully:\n"
                f"  Color: {self.color_intrinsics_rs.width}x{self.color_intrinsics_rs.height}\n"
                f"  Depth: {self.depth_intrinsics_rs.width}x{self.depth_intrinsics_rs.height}\n"
                f"  Scale: {self.depth_scale_rs:.6f} meters/unit"
            )

        except Exception as e:
            self.logger.error(f"Failed to set RealSense calibration parameters: {e}")
            raise

    def _broadcast_camera_transforms(self):
        """Broadcast static transforms between camera frames."""
        if not self.node:
            return
            
        # Create transform from camera_link to color optical frame
        color_tf = TransformStamped()
        color_tf.header.stamp = self.node.get_clock().now().to_msg()
        color_tf.header.frame_id = self.camera_link_frame_id
        color_tf.child_frame_id = self.camera_frame_id
        
        # D435i specific transform (from camera_link to color optical frame)
        # This is the standard transform for the D435i
        color_tf.transform.rotation.x = -0.5
        color_tf.transform.rotation.y = 0.5
        color_tf.transform.rotation.z = -0.5
        color_tf.transform.rotation.w = 0.5
        
        # Depth to color transform from extrinsics
        if self.depth_to_color_extrinsics_rs:
            depth_tf = TransformStamped()
            depth_tf.header.stamp = self.node.get_clock().now().to_msg()
            depth_tf.header.frame_id = self.camera_frame_id
            depth_tf.child_frame_id = self.depth_frame_id
            
            # Convert rotation matrix to quaternion
            R = np.array(self.depth_to_color_extrinsics_rs.rotation).reshape(3, 3)
            quat = self._rotation_matrix_to_quaternion(R)
            
            depth_tf.transform.rotation.x = quat[0]
            depth_tf.transform.rotation.y = quat[1]
            depth_tf.transform.rotation.z = quat[2]
            depth_tf.transform.rotation.w = quat[3]
            
            depth_tf.transform.translation.x = self.depth_to_color_extrinsics_rs.translation[0]
            depth_tf.transform.translation.y = self.depth_to_color_extrinsics_rs.translation[1]
            depth_tf.transform.translation.z = self.depth_to_color_extrinsics_rs.translation[2]
            
            # Broadcast static transforms
            self.static_tf_broadcaster.sendTransform([color_tf, depth_tf])
        else:
            self.static_tf_broadcaster.sendTransform([color_tf])

    def broadcast_hand_eye_transform(self):
        """Broadcast the current hand-eye calibration transform."""
        if not self.node or self.T_base_to_camera is None:
            return
            
        tf = TransformStamped()
        tf.header.stamp = self.node.get_clock().now().to_msg()
        tf.header.frame_id = self.robot_base_frame_id
        tf.child_frame_id = self.camera_link_frame_id
        
        # Extract rotation and translation from transformation matrix
        R = self.T_base_to_camera[:3, :3]
        t = self.T_base_to_camera[:3, 3]
        
        # Convert rotation matrix to quaternion
        quat = self._rotation_matrix_to_quaternion(R)
        
        tf.transform.rotation.x = quat[0]
        tf.transform.rotation.y = quat[1]
        tf.transform.rotation.z = quat[2]
        tf.transform.rotation.w = quat[3]
        
        tf.transform.translation.x = t[0]
        tf.transform.translation.y = t[1]
        tf.transform.translation.z = t[2]
        
        self.tf_broadcaster.sendTransform(tf)

    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion [x,y,z,w]."""
        trace = np.trace(R)
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
        return np.array([qx, qy, qz, qw])

    # --- Getter methods for DepthAwareDetector ---
    def get_color_intrinsics_rs(self) -> Optional[rs.intrinsics]:
        """Get color camera intrinsics in RealSense format."""
        return self.color_intrinsics_rs

    def get_depth_intrinsics_rs(self) -> Optional[rs.intrinsics]:
        """Get depth camera intrinsics in RealSense format."""
        return self.depth_intrinsics_rs

    def get_depth_to_color_extrinsics_rs(self) -> Optional[rs.extrinsics]:
        """Get depth to color extrinsics in RealSense format."""
        return self.depth_to_color_extrinsics_rs

    def get_depth_scale_rs(self) -> Optional[float]:
        """Get depth scale factor in meters/unit."""
        return self.depth_scale_rs

    def _validate_transform(self, T: np.ndarray):
        """
        Validate a transformation matrix.
        
        Parameters
        ----------
        T : np.ndarray
            4x4 transformation matrix to validate
            
        Raises
        ------
        ValueError
            If matrix is not a valid transformation matrix
        """
        if T.shape != (4, 4):
            raise ValueError(f"Transform must be 4x4, got {T.shape}")
        
        # Check rotation matrix properties
        R = T[:3, :3]
        I = np.eye(3)
        
        # Check orthogonality
        if not np.allclose(R @ R.T, I, atol=1e-6):
            raise ValueError("Invalid rotation matrix: not orthogonal")
        
        # Check right-handedness (det should be 1)
        if not np.isclose(np.linalg.det(R), 1.0, atol=1e-6):
            raise ValueError("Invalid rotation matrix: not proper (det != 1)")
        
        # Ensure last row is [0, 0, 0, 1]
        if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError("Invalid homogeneous transform: last row must be [0,0,0,1]")
    
    def load_camera_intrinsics(self, file_path: str):
        """
        Load camera intrinsics from a .npz file.
        Expected keys: 'camera_matrix' (for color), 'dist_coeffs' (for color).
        """
        try:
            calib_data = np.load(file_path)
            if 'camera_matrix' in calib_data and 'dist_coeffs' in calib_data:
                self.camera_matrix_color = calib_data['camera_matrix']
                self.dist_coeffs_color = calib_data['dist_coeffs']
                self.source_type = "file" # Or append if already realsense and loading additional file info
                self.logger.info(f"Loaded camera intrinsics from {file_path}")
                
                # If loaded from file, RealSense specific intrinsics might be None or need conversion
                # For now, we assume these files are primarily for non-RealSense or generic camera_matrix
                if self.color_intrinsics_rs is None and self.camera_matrix_color is not None:
                    self.logger.info("File-loaded intrinsics are set for camera_matrix_color. RealSense specific intrinsics (rs.intrinsics) are not populated from this file type unless explicitly converted.")
            else:
                self.logger.error(f"File {file_path} is missing 'camera_matrix' or 'dist_coeffs'.")
                raise FileNotFoundError(f"Required keys missing in {file_path}")
        except FileNotFoundError:
             self.logger.error(f"Camera intrinsics file not found at {file_path}. Using defaults or uninitialized.")
             # Keep existing defaults or None values for camera_matrix_color and dist_coeffs_color
             # Or set to some safe defaults if preferred and source_type is not 'realsense' already
             if self.source_type != "realsense": # Don't overwrite if realsense already set them
                self.camera_matrix_color = np.array([[500, 0, 320],[0, 500, 240],[0, 0, 1]], dtype=float)
                self.dist_coeffs_color = np.zeros(5, dtype=float)
                self.logger.warning("Using default generic intrinsics due to file load failure.")
             raise # Re-raise FileNotFoundError so UnifiedVisionSystem can log it.
        except Exception as e:
            self.logger.error(f"Failed to load camera intrinsics from {file_path}: {e}")
            # Potentially set defaults here too if not already set by RealSense
            if self.source_type != "realsense":
                self.camera_matrix_color = np.array([[500, 0, 320],[0, 500, 240],[0, 0, 1]], dtype=float)
                self.dist_coeffs_color = np.zeros(5, dtype=float)
                self.logger.warning("Using default generic intrinsics due to error.")
            raise # Re-raise the exception.

    def load_hand_eye_transform(self, file_path: str):
        """
        Load hand-eye calibration matrix from file.
        
        Parameters
        ----------
        file_path : str
            Path to calibration file containing T_base_to_camera
            
        Returns
        -------
        np.ndarray
            4x4 transformation matrix from robot base to camera
            
        Raises
        ------
        FileNotFoundError
            If calibration file is missing or invalid
        """
        try:
            data = np.load(file_path)
            T = data['T_base_to_camera']
            self._validate_transform(T)
            self.T_base_to_camera = T
            self.T_camera_to_base = np.linalg.inv(T) # Pre-calculate inverse
            self.logger.info(f"Loaded hand-eye calibration from {file_path}")
            self.logger.info(f"Translation vector (base to camera, meters): {T[:3, 3]}")
        except FileNotFoundError:
            self.logger.warning(f"Hand-eye calibration file '{file_path}' not found. Using identity transform.")
            self.T_base_to_camera = np.eye(4)
            self.T_camera_to_base = np.eye(4)
        except Exception as e:
            self.logger.error(f"Error loading hand-eye calibration file '{file_path}': {e}")
            self.T_base_to_camera = np.eye(4)
            self.T_camera_to_base = np.eye(4)
            self.logger.warning("Using identity transform for hand-eye due to error.")

    def camera_to_robot(self, point_camera: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Transform point from camera frame to robot base frame.
        
        Parameters
        ----------
        point_camera : Tuple[float, float, float]
            3D point in camera frame (x, y, z) in meters
            
        Returns
        -------
        Tuple[float, float, float]
            3D point in robot base frame (x, y, z) in meters
            
        Notes
        -----
        Uses the stored hand-eye calibration matrix for transformation
        """
        # Convert to homogeneous coordinates
        point_h = np.array([*point_camera, 1.0])
        
        # Transform using hand-eye calibration
        point_robot_h = self.T_base_to_camera @ point_h
        
        return tuple(point_robot_h[:3])

    def robot_to_camera(self, point_robot: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Transform point from robot base frame to camera frame.
        
        Parameters
        ----------
        point_robot : Tuple[float, float, float]
            3D point in robot base frame (x, y, z) in meters
            
        Returns
        -------
        Tuple[float, float, float]
            3D point in camera frame (x, y, z) in meters
            
        Notes
        -----
        Uses the inverse of the stored hand-eye calibration matrix
        """
        # Convert to homogeneous coordinates
        point_h = np.array([*point_robot, 1.0])
        
        # Transform using inverse of hand-eye calibration
        point_camera_h = self.T_camera_to_base @ point_h
        
        return tuple(point_camera_h[:3])

    def pixel_to_camera(self, u: int, v: int, depth: float) -> Tuple[float, float, float]:
        """
        Convert pixel coordinates to camera frame coordinates.
        
        Parameters
        ----------
        u : int
            Pixel x-coordinate
        v : int
            Pixel y-coordinate
        depth : float
            Depth value in meters
            
        Returns
        -------
        Tuple[float, float, float]
            3D point in camera frame (x, y, z) in meters
            
        Notes
        -----
        Uses camera intrinsics and distortion coefficients for back-projection
        """
        if self.camera_matrix_color is None or self.dist_coeffs_color is None:
            self.logger.warning("Camera intrinsics (matrix/dist_coeffs) not loaded. Cannot undistort or deproject pixel.")
            return (0.0,0.0,0.0) # Or raise error

        if np.any(self.dist_coeffs_color):
            # Ensure u, v are float for undistortPoints
            pixel_coords = np.array([[[float(u), float(v)]]], dtype=np.float32)
            undistorted_pixel = cv2.undistortPoints(pixel_coords, 
                                        self.camera_matrix_color,
                                        self.dist_coeffs_color,
                                        None, # R - no rectification rotation
                                        self.camera_matrix_color # P - new camera matrix (use old one for same coord system)
                                        )
            u_undistorted, v_undistorted = undistorted_pixel[0][0]
        else:
            u_undistorted, v_undistorted = float(u), float(v)
            
        # Get camera intrinsics
        fx = self.camera_matrix_color[0, 0]
        fy = self.camera_matrix_color[1, 1]
        cx = self.camera_matrix_color[0, 2]
        cy = self.camera_matrix_color[1, 2]
        
        # Back-project to 3D
        x_cam = (u_undistorted - cx) * depth / fx
        y_cam = (v_undistorted - cy) * depth / fy
        z_cam = depth
        return (x_cam, y_cam, z_cam)

    def pixel_to_robot(self, u: int, v: int, depth: float) -> Optional[Tuple[float, float, float]]:
        """
        Convert pixel coordinates directly to robot base frame.
        
        Parameters
        ----------
        u : int
            Pixel x-coordinate
        v : int
            Pixel y-coordinate
        depth : float
            Depth value in meters
            
        Returns
        -------
        Optional[Tuple[float, float, float]]
            3D point in robot base frame, or None if conversion fails
            
        Notes
        -----
        Combines pixel_to_camera and camera_to_robot transformations
        """
        try:
            # First convert to camera frame
            point_camera = self.pixel_to_camera(u, v, depth)
            
            # Then transform to robot frame
            point_robot = self.camera_to_robot(point_camera)
            
            return point_robot
        except Exception as e:
            self.logger.error(f"Failed to convert pixel to robot coordinates: {e}")
            return None

    def project_to_image(self, point_camera: Tuple[float, float, float]) -> Optional[Tuple[int, int]]:
        """
        Project 3D point from camera frame to 2D pixel coordinates.
        
        Parameters
        ----------
        point_camera : Tuple[float, float, float]
            3D point (x, y, z) in camera frame (meters)
            
        Returns
        -------
        Optional[Tuple[int, int]]
            Pixel coordinates (u, v), or None if projection fails (e.g., behind camera)
            
        Notes
        -----
        Uses camera intrinsics and distortion coefficients for projection
        """
        if self.camera_matrix_color is None or self.dist_coeffs_color is None:
            self.logger.warning("Camera intrinsics (matrix/dist_coeffs) not loaded. Cannot project point.")
            return None

        x, y, z = point_camera
        if z <= 0:  # Point is behind or on the camera plane
            self.logger.warning(f"Point {point_camera} is behind or on the camera plane, cannot project.")
            return None

        # Project 3D points to image plane
        # (rvec, tvec) are set to zero as point_camera is already in camera coordinates
        point_3d_np = np.array([[x, y, z]], dtype=np.float32)
        rvec = np.zeros(3, dtype=np.float32)
        tvec = np.zeros(3, dtype=np.float32)
        
        image_points, _ = cv2.projectPoints(point_3d_np, 
                                            rvec, tvec, 
                                            self.camera_matrix_color, 
                                            self.dist_coeffs_color)
        
        u, v = int(round(image_points[0][0][0])), int(round(image_points[0][0][1]))
        
        # Basic check if projected point is within typical image bounds (e.g. positive)
        # More robust check would use actual image width/height if known
        # if u < 0 or v < 0: 
        #     self.logger.debug(f"Projected point ({u},{v}) is outside typical positive image bounds.")
        #     # Depending on use case, may return None or the coords.
            
        return u, v

    def save_camera_intrinsics(self, file_path: str):
        """Save current camera intrinsics (matrix_color, dist_coeffs_color) to a .npz file."""
        if self.camera_matrix_color is not None and self.dist_coeffs_color is not None:
            try:
                np.savez(file_path, 
                         camera_matrix=self.camera_matrix_color, 
                         dist_coeffs=self.dist_coeffs_color)
                self.logger.info(f"Camera intrinsics saved to {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to save camera intrinsics to {file_path}: {e}")
        else:
            self.logger.warning("Cannot save camera intrinsics, not all parameters are set.")

    def save_hand_eye_calibration(self, T: np.ndarray, file_path: str):
        """
        Save hand-eye calibration matrix to file.
        
        Parameters
        ----------
        T : np.ndarray
            4x4 transformation matrix to save
        file_path : str
            Path to save calibration data
            
        Notes
        -----
        Validates transform matrix before saving
        """
        self._validate_transform(T)
        np.savez(file_path, T_base_to_camera=T)
        self.logger.info(f"Saved hand-eye calibration to {file_path}")
        self.T_base_to_camera = T  # Update current transform

    def transform_vector(self, vector: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Transform a vector from camera frame to robot base frame.
        
        Parameters
        ----------
        vector : Tuple[float, float, float]
            3D vector in camera frame
            
        Returns
        -------
        Tuple[float, float, float]
            3D vector in robot base frame
            
        Notes
        -----
        Only applies rotation, not translation, since vectors
        represent directions not positions
        """
        # Convert to numpy array
        v = np.array(vector)
        
        # Extract rotation matrix from transformation
        R = self.T_base_to_camera[:3, :3]
        
        # Transform vector using only rotation
        v_transformed = R @ v
        
        # Normalize the transformed vector
        v_norm = np.linalg.norm(v_transformed)
        if v_norm > 0:
            v_transformed = v_transformed / v_norm
        
        return tuple(v_transformed)

    def pixel_to_world(self, u: int, v: int, depth: float) -> Tuple[float, float, float]:
        """
        Convert pixel coordinates to world (robot base) coordinates.
        
        Parameters
        ----------
        u : int
            Pixel x-coordinate
        v : int
            Pixel y-coordinate
        depth : float
            Depth value in meters
            
        Returns
        -------
        Tuple[float, float, float]
            3D point in world (robot base) coordinates (x, y, z) in meters
            
        Notes
        -----
        Combines pixel_to_camera and camera_to_robot transformations
        """
        # First convert to camera frame
        point_camera = self.pixel_to_camera(u, v, depth)
        
        # Then transform to robot base frame
        point_world = self.camera_to_robot(point_camera)
        
        return point_world