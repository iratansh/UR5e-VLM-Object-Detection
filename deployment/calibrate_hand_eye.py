"""
Hand-Eye Calibration for UR5e Robot.

This module performs hand-eye calibration between a RealSense camera
and a UR5e robot using ArUco markers. It:
- Captures synchronized robot and camera poses
- Detects ArUco markers for pose estimation
- Computes the transformation between robot and camera frames
- Saves calibration results for use in the vision system

The calibration is essential for accurate 3D object localization
and robot manipulation tasks.
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
import argparse

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
    eye_in_hand : bool
        Whether the configuration is eye-in-hand
    marker_size_meters : float
        Size of the ArUco markers in meters
        
    Notes
    -----
    Requirements:
    - ROS2 Humble or newer
    - UR5e robot with ROS driver
    - RealSense camera
    - ArUco markers (6x6, 250 dictionary)
    """
    def __init__(self, pipeline, intrinsics, eye_in_hand=True):
        super().__init__('hand_eye_calibrator')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.pipeline = pipeline
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        
        self.robot_poses: List[np.ndarray] = []
        self.marker_poses: List[np.ndarray] = []
        self.current_robot_transform: Optional[np.ndarray] = None
        self.eye_in_hand = eye_in_hand
        self.marker_size_meters = 0.05  # Default marker size
        
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
            # Get camera matrix from RealSense
            color_profile = self.pipeline.get_active_profile()
            color_profile = color_profile.get_stream(rs.stream.color)
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
                self.marker_size_meters,  # Marker size in meters
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
        """
        self.logger.info(f"Starting calibration data collection. Need {num_poses} poses.")
        self.robot_poses = []
        self.marker_poses = []
        
        try:
            self.pipeline.start(self.config)
            
            while len(self.robot_poses) < num_poses:
                # Get frames
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                color_image = np.asanyarray(color_frame.get_data())
                
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
    
    def compute_calibration(self) -> np.ndarray:
        """
        Compute hand-eye calibration matrix.
        
        Returns
        -------
        np.ndarray
            4x4 transformation matrix from robot base to camera
            
        Raises
        ------
        ValueError
            If insufficient poses collected
            
        Notes
        -----
        Uses Park's method for hand-eye calibration
        Requires at least 3 different poses
        """
        if len(self.robot_poses) < 3:
            raise ValueError("Need at least 3 poses for calibration")
        
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []
        
        for robot_pose, marker_pose in zip(self.robot_poses, self.marker_poses):
            R_gripper2base.append(robot_pose[0:3, 0:3])
            t_gripper2base.append(robot_pose[0:3, 3])
            R_target2cam.append(marker_pose[0:3, 0:3])
            t_target2cam.append(marker_pose[0:3, 3])
        
        # Compute calibration
        R_base2cam, t_base2cam = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=cv2.CALIB_HAND_EYE_PARK
        )
        
        T_base2cam = np.eye(4)
        T_base2cam[0:3, 0:3] = R_base2cam
        T_base2cam[0:3, 3] = t_base2cam.ravel()
        
        return T_base2cam
    
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

def main():
    parser = argparse.ArgumentParser(description='Perform hand-eye calibration')
    parser.add_argument('--num-poses', type=int, default=10, help='Number of calibration poses')
    parser.add_argument('--save-file', type=str, default='hand_eye_calib.npz', help='Output calibration file')
    parser.add_argument('--marker-size', type=float, default=0.05, help='ArUco marker size in meters')
    parser.add_argument('--eye-in-hand', action='store_true', default=True, help='Use eye-in-hand configuration (default: True)')
    parser.add_argument('--eye-to-hand', action='store_true', help='Use eye-to-hand configuration')
    args = parser.parse_args()
    
    # Determine configuration mode
    eye_in_hand = True  # Default
    if args.eye_to_hand:
        eye_in_hand = False
    
    # Initialize ROS
    rclpy.init()
    
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Configure streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    profile = pipeline.start(config)
    
    # Get camera intrinsics
    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    
    try:
        # Create calibrator with eye-in-hand configuration
        calibrator = HandEyeCalibrator(pipeline, intrinsics, eye_in_hand=eye_in_hand)
        
        calibrator.MARKER_SIZE_METERS = args.marker_size
        
        # Run calibration
        logger.info(f"Starting {'eye-in-hand' if eye_in_hand else 'eye-to-hand'} calibration with {args.num_poses} poses")
        logger.info(f"Marker size: {args.marker_size} meters")
        
        calibrator.collect_calibration_data(args.num_poses)
        
        # Compute calibration
        T_result = calibrator.compute_calibration()
        
        if T_result is not None:
            calibrator.save_calibration(T_result, args.save_file)
            logger.info(f"✅ Calibration saved to {args.save_file}")
            
            # Display results
            if eye_in_hand:
                logger.info("Transformation from gripper to camera (T_gripper_camera):")
            else:
                logger.info("Transformation from camera to base (T_camera_base):")
            logger.info(f"\n{T_result}")
        
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main() 