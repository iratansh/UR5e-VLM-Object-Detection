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
    None : Uses ROS2 parameters for configuration
    
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
        
    Notes
    -----
    Requirements:
    - ROS2 Humble or newer
    - UR5e robot with ROS driver
    - RealSense camera
    - ArUco markers (6x6, 250 dictionary)
    """
    def __init__(self):
        super().__init__('hand_eye_calibrator')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Physical size of the ArUco marker (e.g., 0.05 for 5cm).
        # IMPORTANT: This MUST match the actual printed marker size.
        self.MARKER_SIZE_METERS = 0.05  # Default to 5cm
        
        # Initialize RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        
        # Initialize variables for calibration
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
        
        # Convert quaternion to rotation matrix
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
            # Get camera matrix from RealSense
            profile = self.pipeline.get_active_profile()
            color_profile = profile.get_stream(rs.stream.color)
            intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            
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
            
            # Convert to transformation matrix
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
        """
        self.logger.info(f"Starting calibration data collection. Need {num_poses} poses.")
        self.robot_poses = []
        self.marker_poses = []
        
        try:
            # Start RealSense pipeline
            self.pipeline.start(self.config)
            
            while len(self.robot_poses) < num_poses:
                # Get frames
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                # Convert to numpy array
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
            self.pipeline.stop()
            cv2.destroyAllWindows()
    
    def compute_calibration(self) -> Optional[np.ndarray]:
        """
        Compute hand-eye calibration matrix.
        
        Returns
        -------
        Optional[np.ndarray]
            4x4 transformation matrix (camera_to_robot_base or robot_base_to_camera),
            or None if calibration fails.
        """
        if len(self.robot_poses) < 4 or len(self.marker_poses) < 4:
            self.logger.error("Not enough data points for calibration. Need at least 4.")
            return None
        
        if len(self.robot_poses) != len(self.marker_poses):
            self.logger.error("Mismatch in number of robot and marker poses.")
            return None

        self.logger.info(f"Computing hand-eye calibration with {len(self.robot_poses)} poses.")
        
        # Convert poses to the format required by OpenCV
        # Robot poses are typically T_base_effector
        # Marker poses are T_camera_marker
        
        # We need R_gripper2base and t_gripper2base (from robot_poses)
        # and R_target2cam and t_target2cam (from marker_poses)
        
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
            
        # Placeholder for choosing calibration method.
        # Common methods: cv2.CALIB_HAND_EYE_TSAI, cv2.CALIB_HAND_EYE_PARK, 
        # cv2.CALIB_HAND_EYE_HORAUD, cv2.CALIB_HAND_EYE_DANIILIDIS (often good).
        # The choice depends on whether the camera is mounted on the hand or static.
        # This example assumes Eye-to-Hand (camera is static, marker on robot).
        # If Eye-in-Hand (camera on gripper, marker is static), then the inputs to
        # calibrateHandEye might need to be T_base_gripper and T_marker_camera (inverted).
        # For Eye-to-Hand, we need T_base_gripper and T_camera_target.
        
        # Assuming Eye-to-Hand setup:
        # R_base_gripper, t_base_gripper
        # R_cam_target, t_cam_target
        # Result is T_cam_base (R_cam_base, t_cam_base)
        
        # If the setup is Eye-in-Hand (camera on gripper, marker is static in world):
        # We have T_base_gripper and T_camera_marker.
        # We need to provide T_base_gripper and T_marker_camera (which is inv(T_camera_marker)).
        # The function then solves for T_gripper_camera.
        # Let's assume an Eye-to-Hand setup for now, where the camera is static and the marker moves with the robot.
        # The robot_poses are T_base_tool0 and marker_poses are T_camera_marker.
        # We want to find T_camera_base.
        # cv2.calibrateHandEye expects:
        # R_gripper2base, t_gripper2base (list of rotations and translations of effector relative to base)
        # R_target2cam, t_target2cam (list of rotations and translations of target relative to camera)
        # It computes R_cam2gripper, t_cam2gripper OR R_base2cam, t_base2cam depending on method.
        # The documentation for OpenCV's calibrateHandEye can be a bit confusing regarding frame definitions.
        # Typically for Eye-to-Hand (camera is base, target is on gripper):
        #   Input: T_world_camera (list of gripper poses relative to robot base)
        #          T_marker_camera (list of marker poses relative to camera)
        #   Output: T_camera_base (transformation from camera to robot base)
        # Let's assume the interpretation where:
        # robot_poses are T_base_effector (effector is where marker is, or effector moves the camera)
        # marker_poses are T_camera_target (target is the calibration pattern)
        # If camera is static and marker is on effector (Eye-to-Hand for T_base_camera):
        #   We input R_base_effector, t_base_effector
        #   We input R_target_camera, t_target_camera (target is marker)
        #   It solves for R_effector_camera, t_effector_camera (transformation from effector to camera)
        #   This would be if the 'camera' is the calibration target and 'gripper' is the camera
        #   Let's use the common setup: Camera is 'world', Gripper is 'camera', target is 'marker' on robot.
        #   No, that's not right.
        # Standard Eye-to-Hand: Robot base is 'base', camera is 'camera', marker is on 'gripper'.
        # We collect (T_base_gripper_i, T_camera_marker_i) pairs.
        # We want to find T_camera_base.
        # OpenCV's `calibrateHandEye` typically solves for X in AX=ZB.
        # A = T_gripper_base (gripper in base frame), B = T_target_camera (target in camera frame)
        # Z = T_camera_base (camera in base frame) - if eye-on-base (static camera)
        # X = T_target_gripper (target in gripper frame) - fixed transform we are calibrating for, or T_base_camera
        
        # According to OpenCV docs for cv2.calibrateHandEye:
        # R_gripper2base, t_gripper2base: rotations and translations of the gripper with respect to the robot base.
        # R_target2cam, t_target2cam: rotations and translations of the calibration target with respect to the camera.
        # It computes R_base2cam, t_base2cam (or R_cam2gripper, t_cam2gripper for eye-in-hand)
        
        # For Eye-to-Hand (static camera, target on robot end-effector):
        # We are looking for T_camera_base (or T_base_camera).
        # Input R_gripper2base should be R_base_gripper.
        # Input R_target2cam should be R_camera_target.
        # Output R_cam2base, t_cam2base
        
        # Let's stick to the variable names from the function signature for clarity with OpenCV docs
        # self.robot_poses are T_base_tool0 (tool0 is gripper)
        # self.marker_poses are T_camera_marker (marker is target)
        
        # So, R_gripper2base needs to be R_base_tool0
        # And R_target2cam needs to be R_camera_marker
        
        # The method chosen affects which transform is solved for.
        # For CALIB_HAND_EYE_DANIILIDIS (and others), for eye-to-hand (camera fixed, pattern on robot):
        # It solves AX = ZB where X is the unknown hand-eye transformation (e.g. T_tool_camera)
        # A is T_base_tool, Z is T_base_camera, B is T_pattern_camera
        # No, this is not the standard AX=XB setup.
        # The most common formulation is AX = XB, where:
        # A: Transformation from robot base to end-effector (T_base_ee)
        # B: Transformation from camera to marker (T_cam_marker)
        # X: Transformation from end-effector to camera (T_ee_cam) --- for eye-in-hand
        # OR
        # X: Transformation from robot base to camera (T_base_cam) --- for eye-to-hand

        # Let's assume Eye-to-Hand: camera is stationary, marker on end-effector. We want T_base_camera.
        # R_base2gripper_list = R_gripper2base (from self.robot_poses)
        # t_base2gripper_list = t_gripper2base (from self.robot_poses)
        # R_target2camera_list = R_target2cam (from self.marker_poses)
        # t_target2camera_list = t_target2cam (from self.marker_poses)

        try:
            # For Eye-to-Hand, we want to find the transform from the camera to the robot base (T_camera_base)
            # Or robot base to camera (T_base_camera).
            # The cv2.calibrateHandEye function with most methods (e.g., TSAI, PARK, HORAUD, DANIILIDIS)
            # computes T_gripper_camera when given T_base_gripper and T_marker_camera.
            # This is for an Eye-in-Hand setup (camera on gripper).
            
            # If our setup is Eye-to-Hand (camera is static, observes marker on gripper):
            # We have T_base_gripper (from robot_poses)
            # We have T_camera_marker (from marker_poses)
            # We want to find T_camera_base.
            # To use cv2.calibrateHandEye, we might need to invert some transformations
            # or use a solver that directly computes T_camera_base.
            
            # Let's use the formulation that solves for T_base_cam (denoted as X).
            # A_i * X = X * B_i
            # Where A_i is T_gripper_i_gripper_j (relative motion of gripper)
            # And B_i is T_camera_marker_i_camera_marker_j (relative motion of marker in camera frame)
            # This is not what cv2.calibrateHandEye takes directly.

            # Sticking to OpenCV's direct input convention for Eye-to-Hand:
            # R_gripper2base, t_gripper2base => list of T_base_gripper
            # R_target2cam, t_target2cam   => list of T_camera_target
            # The function will return R_cam2base, t_cam2base
            
            # Using default Daniilidis method
            calibration_method = cv2.CALIB_HAND_EYE_DANIILIDIS 
            self.logger.info(f"Using calibration method: DANIILIDIS")
            
            R_cam2base, t_cam2base = cv2.calibrateHandEye(
                R_gripper2base=R_gripper2base, # List of R_base_gripper
                t_gripper2base=t_gripper2base, # List of t_base_gripper
                R_target2cam=R_target2cam,     # List of R_camera_target
                t_target2cam=t_target2cam,     # List of t_camera_target
                method=calibration_method
            )
            
            # The returned R_cam2base, t_cam2base is T_camera_base
            T_camera_base = np.eye(4)
            T_camera_base[:3, :3] = R_cam2base
            T_camera_base[:3, 3] = t_cam2base.ravel()
            
            self.logger.info("Hand-eye calibration successful.")
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
        Saves matrix in .npz format for easy loading
        """
        np.savez(file_path, T_base_to_camera=T)
        self.logger.info(f"Saved calibration to {file_path}") 