"""
Hand-Eye Calibration for UR5e Robot.

This module provides the HandEyeCalibrator class for performing
hand-eye calibration between a RealSense camera and a UR5e robot.
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
import time
from pathlib import Path
import logging
from typing import List, Tuple, Optional

class HandEyeCalibrator(Node):
    """
    Node for performing hand-eye calibration between a RealSense camera and a robot.
    
    This class collects synchronized robot and camera poses using ArUco markers
    and computes the transformation between the robot base and camera frames.
    
    Parameters
    ----------
    pipeline : rs.pipeline
        RealSense camera pipeline
    intrinsics : rs.intrinsics
        RealSense camera intrinsics
    eye_in_hand : bool, optional
        True for camera mounted on end-effector, False for static camera, by default True
    
    Attributes
    ----------
    pipeline : rs.pipeline
        RealSense camera pipeline
    robot_poses : List[np.ndarray]
        List of robot end-effector poses
    marker_poses : List[np.ndarray]
        List of detected marker poses
    current_robot_transform : Optional[np.ndarray]
        Current robot transformation matrix
    MARKER_SIZE_METERS : float
        Physical size of the ArUco marker in meters
    eye_in_hand : bool
        Configuration mode: True for eye-in-hand, False for eye-to-hand
        
    Notes
    -----
    Requirements:
    - ROS2 Humble or newer
    - UR5e robot with ROS driver
    - RealSense camera
    - ArUco markers (6x6, 250 dictionary)
    """
    def __init__(self, pipeline: rs.pipeline, intrinsics: rs.intrinsics, eye_in_hand: bool = True):
        super().__init__('hand_eye_calibrator')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Physical size of the ArUco marker (e.g., 0.05 for 5cm).
        # IMPORTANT: This MUST match the actual printed marker size.
        self.MARKER_SIZE_METERS = 0.05  # Default to 5cm
        
        # Configuration mode
        self.eye_in_hand = eye_in_hand
        self.logger.info(f"Hand-eye calibrator initialized in {'EYE-IN-HAND' if eye_in_hand else 'EYE-TO-HAND'} mode")
        
        # Use the existing RealSense pipeline and intrinsics
        self.pipeline = pipeline
        self.intrinsics = intrinsics
        if not self.pipeline or not self.intrinsics:
            raise ValueError("An active RealSense pipeline and camera intrinsics must be provided.")
        
        self.robot_poses: List[np.ndarray] = []
        self.marker_poses: List[np.ndarray] = []
        self.current_robot_transform: Optional[np.ndarray] = None
        
        # ArUco marker setup
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        # Subscribe to robot transform
        self.tf_sub = self.create_subscription(
            TFMessage,
            '/tf',
            self._tf_callback,
            10
        )
        
        self.logger.info("Hand-eye calibrator initialized")
    
    def _tf_callback(self, msg: TFMessage):
        """
        Process robot transform messages.
        
        Parameters
        ----------
        msg : TFMessage
            ROS2 transform message
            
        Notes
        -----
        Updates current_robot_transform when receiving
        transforms between base_link and tool0 frames
        """
        for transform in msg.transforms:
            if transform.header.frame_id == "base_link" and transform.child_frame_id == "tool0":
                self.current_robot_transform = self._transform_to_matrix(transform)
    
    def _transform_to_matrix(self, transform: TransformStamped) -> np.ndarray:
        """
        Convert ROS transform to 4x4 matrix.
        
        Parameters
        ----------
        transform : TransformStamped
            ROS2 transform message
            
        Returns
        -------
        np.ndarray
            4x4 homogeneous transformation matrix
            
        Notes
        -----
        Combines translation and rotation (quaternion)
        into a single transformation matrix
        """
        T = np.eye(4)
        
        # Translation
        T[0:3, 3] = [
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
        ]
        
        # Rotation (quaternion to matrix)
        q = [
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        ]
        
        R = self._quaternion_to_matrix(q)
        T[0:3, 0:3] = R
        
        return T
    
    def _quaternion_to_matrix(self, q: List[float]) -> np.ndarray:
        """
        Convert quaternion to rotation matrix.
        
        Parameters
        ----------
        q : List[float]
            Quaternion [x, y, z, w]
            
        Returns
        -------
        np.ndarray
            3x3 rotation matrix
            
        Notes
        -----
        Uses standard quaternion to rotation matrix formula
        """
        x, y, z, w = q
        
        R = np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
        
        return R
    
    def detect_marker(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect ArUco marker and return its pose.
        
        Parameters
        ----------
        frame : np.ndarray
            RGB camera frame
            
        Returns
        -------
        Optional[np.ndarray]
            4x4 marker pose transformation matrix,
            or None if no marker detected
            
        Notes
        -----
        Uses ArUco 6x6 dictionary
        Estimates pose using camera intrinsics
        Assumes 5cm marker size
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is not None and len(ids) > 0:
            # Get camera matrix from the provided RealSense intrinsics
            intrinsics = self.intrinsics
            
            camera_matrix = np.array([
                [intrinsics.fx, 0, intrinsics.ppx],
                [0, intrinsics.fy, intrinsics.ppy],
                [0, 0, 1]
            ])
            
            dist_coeffs = np.array(intrinsics.coeffs).reshape(-1, 1)
            
            # Estimate marker pose
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                self.MARKER_SIZE_METERS,  # Use the class attribute for marker size
                camera_matrix,
                dist_coeffs
            )
            
            R, _ = cv2.Rodrigues(rvecs[0])
            T = np.eye(4)
            T[0:3, 0:3] = R
            T[0:3, 3] = tvecs[0].ravel()
            
            return T
        
        return None
    
    def collect_calibration_data(self, num_poses: int = 10):
        """
        Collect calibration data from multiple robot poses.
        
        Parameters
        ----------
        num_poses : int, optional
            Number of poses to collect, by default 10
            
        Notes
        -----
        Collection process:
        1. Start RealSense streaming
        2. For each pose:
           - Detect marker
           - Record robot pose
           - Wait for robot movement
        3. Show visualization
        4. Clean up resources
        
        For eye-in-hand: Move robot to different poses while marker is stationary
        For eye-to-hand: Move robot with marker to different poses while camera is stationary
        """
        mode_str = "eye-in-hand" if self.eye_in_hand else "eye-to-hand"
        self.logger.info(f"Starting {mode_str} calibration data collection. Need {num_poses} poses.")
        
        if self.eye_in_hand:
            self.logger.info("Eye-in-hand mode: Keep ArUco marker stationary, move robot to observe from different angles")
        else:
            self.logger.info("Eye-to-hand mode: Move robot with marker attached, camera observes from fixed position")
            
        self.robot_poses = []
        self.marker_poses = []
        
        try:
            # The pipeline is already started by the main system
            
            while len(self.robot_poses) < num_poses:
                # Get frames from the existing pipeline
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                color_image = np.asanyarray(color_frame.get_data())
                
                # Detect marker
                marker_pose = self.detect_marker(color_image)
                
                if marker_pose is not None and self.current_robot_transform is not None:
                    self.robot_poses.append(self.current_robot_transform.copy())
                    self.marker_poses.append(marker_pose)
                    
                    self.logger.info(f"Collected pose {len(self.robot_poses)}/{num_poses}")
                    
                    # Wait for robot to move to next pose
                    time.sleep(2.0)
                
                # Show frame with detected markers
                cv2.imshow('Calibration', color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            # Do not stop the pipeline here, it's managed externally
            cv2.destroyAllWindows()
    
    def compute_calibration(self) -> Optional[np.ndarray]:
        """
        Compute hand-eye calibration matrix.
        
        Returns
        -------
        Optional[np.ndarray]
            4x4 transformation matrix:
            - For eye-in-hand: T_gripper_camera (camera relative to gripper)
            - For eye-to-hand: T_camera_base (camera relative to robot base)
            or None if calibration fails.
        """
        if len(self.robot_poses) < 4 or len(self.marker_poses) < 4:
            self.logger.error("Not enough data points for calibration. Need at least 4.")
            return None
        
        if len(self.robot_poses) != len(self.marker_poses):
            self.logger.error("Mismatch in number of robot and marker poses.")
            return None

        self.logger.info(f"Computing hand-eye calibration with {len(self.robot_poses)} poses.")
        
        R_gripper2base = []
        t_gripper2base = []
        for pose_matrix in self.robot_poses:
            R_gripper2base.append(pose_matrix[:3, :3])
            t_gripper2base.append(pose_matrix[:3, 3].reshape(3, 1))
            
        R_target2cam = []
        t_target2cam = []
        for pose_matrix in self.marker_poses:
            R_target2cam.append(pose_matrix[:3, :3])
            t_target2cam.append(pose_matrix[:3, 3].reshape(3, 1))
            
        try:
            if self.eye_in_hand:
                # Eye-in-hand: Camera mounted on gripper, marker is stationary
                # We have T_base_gripper and T_camera_marker
                # We want to find T_gripper_camera
                # 
                # The equation is: T_base_gripper * T_gripper_camera = T_base_camera * T_camera_marker
                # Rearranging: T_base_gripper1^-1 * T_base_gripper2 * T_gripper_camera = T_gripper_camera * T_camera_marker1^-1 * T_camera_marker2
                # This is the AX = XB form where X = T_gripper_camera
                
                self.logger.info("Computing eye-in-hand calibration (T_gripper_camera)")
                
                # Need to invert marker poses for eye-in-hand
                R_cam2target = []
                t_cam2target = []
                for R, t in zip(R_target2cam, t_target2cam):
                    R_inv = R.T
                    t_inv = -R_inv @ t
                    R_cam2target.append(R_inv)
                    t_cam2target.append(t_inv)
                
                # Compute T_gripper_camera
                R_gripper2cam, t_gripper2cam = cv2.calibrateHandEye(
                    R_gripper2base=R_gripper2base,
                    t_gripper2base=t_gripper2base,
                    R_target2cam=R_cam2target,  # Using inverted poses
                    t_target2cam=t_cam2target,
                    method=cv2.CALIB_HAND_EYE_TSAI  # TSAI method often works well for eye-in-hand
                )
                
                # Build transformation matrix T_gripper_camera
                T_gripper_camera = np.eye(4)
                T_gripper_camera[:3, :3] = R_gripper2cam
                T_gripper_camera[:3, 3] = t_gripper2cam.ravel()
                
                self.logger.info("Eye-in-hand calibration successful.")
                self.logger.info(f"T_gripper_camera:\n{T_gripper_camera}")
                
                # Also compute and log the camera position relative to gripper
                cam_pos_in_gripper = T_gripper_camera[:3, 3]
                self.logger.info(f"Camera position in gripper frame: [{cam_pos_in_gripper[0]:.3f}, {cam_pos_in_gripper[1]:.3f}, {cam_pos_in_gripper[2]:.3f}] meters")
                
                return T_gripper_camera
                
            else:
                # Eye-to-hand: Camera is stationary, marker on gripper
                self.logger.info("Computing eye-to-hand calibration (T_camera_base)")
                
                calibration_method = cv2.CALIB_HAND_EYE_DANIILIDIS 
                
                R_cam2base, t_cam2base = cv2.calibrateHandEye(
                        R_gripper2base=R_gripper2base,
                        t_gripper2base=t_gripper2base,
                        R_target2cam=R_target2cam,
                        t_target2cam=t_target2cam,
                    method=calibration_method
                )
                
                T_camera_base = np.eye(4)
                T_camera_base[:3, :3] = R_cam2base
                T_camera_base[:3, 3] = t_cam2base.ravel()
                
                self.logger.info("Eye-to-hand calibration successful.")
                self.logger.info(f"T_camera_base:\n{T_camera_base}")
                return T_camera_base
            
        except cv2.error as e:
            self.logger.error(f"OpenCV calibration failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during calibration: {e}")
            return None

    def save_calibration(self, T: np.ndarray, file_path: str = 'hand_eye_calib.npz'):
        """
        Save calibration matrix to file.
        
        Parameters
        ----------
        T : np.ndarray
            4x4 transformation matrix to save
        file_path : str, optional
            Output file path, by default 'hand_eye_calib.npz'
            
        Notes
        -----
        Saves matrix in .npz format with metadata about calibration type
        """
        save_data = {
            'T_base_to_camera': T,  # Keep key name for compatibility
            'calibration_type': 'eye_in_hand' if self.eye_in_hand else 'eye_to_hand',
            'is_eye_in_hand': self.eye_in_hand
        }
        np.savez(file_path, **save_data)
        
        calib_type = "T_gripper_camera" if self.eye_in_hand else "T_camera_base"
        self.logger.info(f"Saved {calib_type} calibration to {file_path}") 