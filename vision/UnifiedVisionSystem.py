"""
Unified Vision System for Robotic Control.

This module integrates multiple vision and control components:
- Visual Language Models (OWL-ViT) for object detection
- Speech command processing
- Depth-aware detection
- Robot kinematics and control
- Workspace validation
- Grasp point detection

The system enables natural language control of a UR5e robot
for pick-and-place tasks using visual and voice commands.
"""

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, Pose, TransformStamped
from std_msgs.msg import String, Bool, Float32MultiArray
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
import time
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
# Component Imports
from SpeechCommandProcessor import SpeechCommandProcessor
from OWLViTDetector import OWLViTDetector
from DepthAwareDetector import DepthAwareDetection, Detection3D
from UR5eKinematics import UR5eKinematics
from WorkSpaceValidator import WorkspaceValidator
from GraspPointDetector import GraspPointDetector
from CameraCalibration import CameraCalibration 
from HandEyeCalibrator import HandEyeCalibrator
import pyrealsense2 as rs

# Default constants, can be overridden by ROS parameters
DEFAULT_ROBOT_NAMESPACE = 'ur5e'
DEFAULT_HAND_EYE_CALIBRATION_FILE = 'hand_eye_calib.npz'
DEFAULT_CAMERA_CALIBRATION_FILE = 'camera_calibration.npz'

class UnifiedVisionSystem(Node):
    """
    Integrated Vision System for Robotic Control.
    
    This class integrates voice commands, visual language models, depth perception,
    and robotic control into a unified system for object manipulation.
    
    Parameters
    ----------
    None : The class uses ROS2 parameters for configuration
        - robot_namespace : Name of the robot (default: 'ur5e')
        - hand_eye_calibration_file : Path to calibration file
        
    Attributes
    ----------
    calibration : CameraCalibration
        Camera and hand-eye calibration handler
    speech_processor : SpeechCommandProcessor
        Natural language command processor
    vlm_detector : OWLViTDetector
        Visual language model for object detection
    depth_detector : DepthAwareDetection
        Depth-aware object detection system
    kinematics : UR5eKinematics
        Robot kinematics calculator
    workspace_validator : WorkspaceValidator
        Robot workspace safety checker
    grasp_detector : GraspPointDetector
        Grasp point detection system
        
    Notes
    -----
    The system requires:
    - ROS2 Humble or newer
    - UR5e robot with ROS driver
    - RealSense camera
    - PyTorch for visual models
    """
    
    def __init__(self):
        """Initialize UnifiedVisionSystem"""
        super().__init__('unified_vision_system')
        
        # Use ROS2 logger
        self.logger = self.get_logger()
        self.pipeline = None
        self.pipeline_started = False

        # Declare all ROS2 parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('robot_namespace', DEFAULT_ROBOT_NAMESPACE),
                ('hand_eye_calibration_file', DEFAULT_HAND_EYE_CALIBRATION_FILE),
                ('camera_calibration_file', DEFAULT_CAMERA_CALIBRATION_FILE),
                ('realsense_serial', ''),  # Optional: Specific D435i serial number
                ('max_depth', 2.0),
                ('min_depth', 0.1),
                ('depth_fps', 30),
                ('color_fps', 30),
                ('depth_width', 848),
                ('depth_height', 480),
                ('color_width', 848),
                ('color_height', 480),
                ('enable_depth_filters', True)
            ]
        )

        # Get parameters
        self.robot_ns = self.get_parameter('robot_namespace').value
        hand_eye_calib_file_path = self.get_parameter('hand_eye_calibration_file').value
        camera_calib_file_path = self.get_parameter('camera_calibration_file').value
        
        # Set up QoS profiles
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # Initialize TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        try:
            # Initialize RealSense pipeline with parameters
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # Get RealSense parameters
            depth_fps = self.get_parameter('depth_fps').value
            color_fps = self.get_parameter('color_fps').value
            depth_width = self.get_parameter('depth_width').value
            depth_height = self.get_parameter('depth_height').value
            color_width = self.get_parameter('color_width').value
            color_height = self.get_parameter('color_height').value
            
            # Optional: Use specific device
            realsense_serial = self.get_parameter('realsense_serial').value
            if realsense_serial:
                config.enable_device(realsense_serial)
            
            # Configure streams
            config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, depth_fps)
            config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, color_fps)
            
            # Start pipeline
            profile = self.pipeline.start(config)
            self.pipeline_started = True
            self.logger.info("RealSense pipeline started successfully")
            
            # Initialize CameraCalibration object first
            self.calibration = CameraCalibration()
            # Populate it with parameters from the live RealSense pipeline
            self.calibration.set_realsense_calibration_params(profile)
            
            try:
                self.calibration.load_camera_intrinsics(camera_calib_file_path)
                self.logger.info(f"Successfully processed camera intrinsics file path: {camera_calib_file_path}")
            except FileNotFoundError:
                if self.calibration.source_type != "realsense":
                    self.logger.error(f"Camera intrinsics file not found at {camera_calib_file_path} and RealSense not used.")
                else:
                    self.logger.info(f"Using RealSense live intrinsics.")
            except Exception as e:
                self.logger.error(f"Error loading camera intrinsics: {e}")

            # Load hand-eye calibration
            try:
                self.calibration.load_hand_eye_transform(hand_eye_calib_file_path)
                self.logger.info(f"Successfully loaded hand-eye calibration")
            except Exception as e:
                self.logger.error(f"Error loading hand-eye calibration, using identity: {e}")

            # Initialize components with updated calibration
            self.depth_detector = DepthAwareDetection(
                self.calibration,
                use_filters=self.get_parameter('enable_depth_filters').value
            )
            self.speech_processor = SpeechCommandProcessor()
            self.workspace_validator = WorkspaceValidator()
            self.vlm_detector = OWLViTDetector()
            self.kinematics = UR5eKinematics()
            self.grasp_detector = GraspPointDetector(self.calibration)
            
            # Initialize ROS2 publishers/subscribers
            self._init_ros2_communication()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vision system: {e}")
            self.cleanup()
            raise

    def _init_ros2_communication(self):
        """
        Initialize ROS2 communication setup with proper QoS profiles.
        
        This function sets up ROS2 publishers and subscribers for the system.
        """
        self.logger.info("Setting up ROS2 publishers and subscribers...")
        
        # Publishers
        self.joint_command_pub = self.create_publisher(
            Float32MultiArray, 
            f'/{self.robot_ns}/scaled_joint_trajectory_controller/joint_trajectory',
            self.reliable_qos
        )
        
        self.target_pose_pub = self.create_publisher(
            PoseStamped, 
            f'/{self.robot_ns}/target_pose', 
            self.reliable_qos
        )
        
        self.system_status_pub = self.create_publisher(
            String, 
            f'/{self.robot_ns}/system_status', 
            self.reliable_qos
        )
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            f'/{self.robot_ns}/joint_states',
            self._joint_state_callback,
            self.sensor_qos
        )
        
        self.emergency_stop_sub = self.create_subscription(
            Bool,
            f'/{self.robot_ns}/emergency_stop',
            self._emergency_stop_callback,
            self.reliable_qos
        )
        
        # State management
        self.current_robot_joints = np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0]) # Will be updated by callback
        self.robot_state_received = False  # Flag to track if we've received robot state
        self.active_object_queries = []
        self.last_successful_detection_3d: Optional[Detection3D] = None
        self.last_target_pose_matrix: Optional[np.ndarray] = None
        self.last_joint_solution: Optional[List[float]] = None

        # Command queue management
        self.command_queue = []
        self.executing_command = False
        self.command_timeout = 5.0  # seconds
        self.last_command_time = None

        # Create command execution timer
        self.command_timer = self.create_timer(0.1, self._process_command_queue)

        # Start subsystems
        self.logger.info("Starting voice control...")
        self._setup_voice_control()
        self.logger.info("✅ UnifiedVisionSystem initialized successfully.")
        self.publish_status("System initialized. Listening for commands.")

        # Add timeout to motion execution
        self.motion_timeout = 5.0  # seconds

    def _setup_voice_control(self):
        """
        Initialize and start the voice recognition system.
        
        Raises
        ------
        Exception
            If voice control initialization fails
        """
        try:
            # Assuming start_listening might be blocking or needs a thread
            # For now, direct call. If it blocks, will need adjustment.
            # self.speech_processor.start_listening() 
            # If start_listening() is blocking, it should be run in a separate thread:
            # import threading
            # self.speech_thread = threading.Thread(target=self.speech_processor.start_listening, daemon=True)
            # self.speech_thread.start()
            # For now, we assume get_command() below will trigger listening or it's handled internally by SpeechCommandProcessor
            self.logger.info("🎤 Voice control setup. Will process commands in the loop.")
        except Exception as e:
            self.logger.error(f"Failed to initialize or start voice control: {e}")
            self.publish_status(f"Error: Voice control failed: {e}")

    def _process_sound_input(self) -> Optional[List[str]]:
        """
        Handle sound input and process to extract object queries.
        
        Returns
        -------
        Optional[List[str]]
            List of parsed object queries if successful, None otherwise
        """
        # Ensure get_command() is non-blocking or has a timeout
        # If it blocks indefinitely, the main loop will stall.
        command = self.speech_processor.get_command() # This should be non-blocking
        if command:
            self.logger.info(f"Received voice command: '{command}'")
            self.publish_status(f"Processing command: {command}")
            queries = self.speech_processor.parse_object_query(command)
            if queries:
                self.logger.info(f"Parsed object queries: {queries}")
                return queries
            else:
                self.logger.warning(f"Could not parse object queries from command: '{command}'")
                self.publish_status(f"No objects identified in command: {command}")
        return None

    def _perform_vlm_detection(self, color_frame: np.ndarray, queries: List[str]) -> List[Tuple[str, float, List[int]]]:
        """
        Perform VLM-based detection on the color frame for given queries.
        
        Parameters
        ----------
        color_frame : np.ndarray
            RGB image frame to process
        queries : List[str]
            List of text queries to detect objects
            
        Returns
        -------
        List[Tuple[str, float, List[int]]]
            List of detections, each containing (label, confidence, bounding_box)
        """
        if not queries:
            return []
        self.logger.info(f"Performing VLM detection for queries: {queries}...")
        vlm_detections = self.vlm_detector.detect_with_text_queries(
            color_frame, queries, confidence_threshold=0.1
        )
        self.logger.info(f"VLM found {len(vlm_detections)} potential objects.")
        return vlm_detections

    def _perform_depth_aware_detection(self, 
                                     color_image_np: np.ndarray, 
                                     depth_frame_rs_obj: Optional[rs.frame], # pyrealsense2.depth_frame for filtering
                                     raw_depth_image_np: np.ndarray, # Raw depth data as numpy
                                     vlm_detections: List[Tuple[str, float, List[int]]]) -> Optional[Detection3D]:
        """
        Process VLM detections with depth information for 3D graspable object detection.
        
        Parameters
        ----------
        color_image_np : np.ndarray
            RGB image frame as NumPy array
        depth_frame_rs_obj : Optional[rs.frame]
            Raw pyrealsense2.depth_frame object for potential filtering by DepthAwareDetector.
        raw_depth_image_np : np.ndarray
            Depth image data as NumPy array (e.g., Z16 values).
        vlm_detections : List[Tuple[str, float, List[int]]]
            List of 2D VLM detections
            
        Returns
        -------
        Optional[Detection3D]
            First valid and graspable 3D detection, or None if none found
        """
        if not vlm_detections:
            return None
        
        self.logger.info("Augmenting VLM detections with 3D depth information...")
        # Pass all three: color np.ndarray, raw depth rs.frame, and raw depth np.ndarray
        detections_3d: List[Detection3D] = self.depth_detector.augment_vlm_detections_3d(
            color_image_np, depth_frame_rs_obj, raw_depth_image_np, vlm_detections, self.grasp_detector
        )

        if not detections_3d:
            self.logger.warning("No 3D detections were generated from VLM results.")
            return None

        self.logger.info(f"Generated {len(detections_3d)} 3D detections. Validating...")
        
        # Select the first valid and graspable detection according to workspace and grasp validation
        for det_3d in detections_3d:
            if self.depth_detector.validate_grasp_3d(det_3d, self.workspace_validator):
                self.logger.info(f"Valid graspable 3D detection found: {det_3d.label} at {det_3d.grasp_point_3d}")
                return det_3d
        
        self.logger.warning("No valid graspable 3D detection found after validation.")
        return None

    def _calculate_inverse_kinematics(self, target_pose_matrix: np.ndarray) -> Optional[List[float]]:
        """
        Calculate inverse kinematics for target pose.
        
        Parameters
        ----------
        target_pose_matrix : np.ndarray
            4x4 homogeneous transformation matrix for target pose
            
        Returns
        -------
        Optional[List[float]]
            Joint angles solution if found, None otherwise
            
        Notes
        -----
        Includes validation of solutions and selection of best configuration
        """
        self.logger.info("Calculating inverse kinematics...")
        try:
            ik_solutions = self.kinematics.inverse_kinematics(target_pose_matrix)
            if not ik_solutions:
                self.logger.warning("IK: No solutions found for the target pose.")
                return None

            self.logger.info(f"IK: Found {len(ik_solutions)} solutions. Selecting the best one.")
            best_solution = self.kinematics.select_best_solution(ik_solutions, list(self.current_robot_joints))
            
            if best_solution:
                self.logger.info(f"IK: Selected solution (radians): {best_solution}")
                # Optional: Validate solution again before returning
                if not self.kinematics.is_valid_solution(best_solution, target_pose_matrix):
                    self.logger.error("IK: Selected solution failed final validation!")
                    return None
                return best_solution
            else:
                self.logger.warning("IK: No suitable solution found after selection process.")
                return None
        except Exception as e:
            self.logger.error(f"Error during inverse kinematics calculation: {e}", exc_info=True)
            return None

    def _check_joint_limits(self, joint_angles: List[float]) -> bool:
        """
        Check if joint angles are within safe limits.
        
        Parameters
        ----------
        joint_angles : List[float]
            List of 6 joint angles in radians
            
        Returns
        -------
        bool
            True if all joints are within limits, False otherwise
        """
        # UR5e joint limits in radians
        JOINT_LIMITS = [
            (-2 * np.pi, 2 * np.pi),    # Base
            (-2 * np.pi, 2 * np.pi),    # Shoulder (Corrected to +-360 deg)
            (-2 * np.pi, 2 * np.pi),    # Elbow (Corrected to +-360 deg)
            (-2 * np.pi, 2 * np.pi),    # Wrist 1
            (-2 * np.pi, 2 * np.pi),    # Wrist 2
            (-2 * np.pi, 2 * np.pi),    # Wrist 3
        ]
        
        for i, (angle, (min_limit, max_limit)) in enumerate(zip(joint_angles, JOINT_LIMITS)):
            if not min_limit <= angle <= max_limit:
                self.logger.error(f"Joint {i} angle {np.rad2deg(angle):.2f}° exceeds limits [{np.rad2deg(min_limit):.2f}°, {np.rad2deg(max_limit):.2f}°]")
                return False
        return True

    def _check_joint_velocity(self, target_joints: List[float]) -> bool:
        """
        Check if the joint velocity would be safe.
        
        Parameters
        ----------
        target_joints : List[float]
            Target joint angles in radians
            
        Returns
        -------
        bool
            True if velocities are within limits, False otherwise
            
        Notes
        -----
        Uses MAX_JOINT_VELOCITY (π/2 rad/s) as the velocity limit
        """
        MAX_JOINT_VELOCITY = np.pi / 2  # 90 degrees per second max
        
        # Calculate joint velocities (assuming 1 second movement)
        # NOTE: This is a simplification. Actual velocity depends on controller execution time.
        # The robot controller (e.g., JointGroupPositionController) usually handles velocity limits.
        # This check can be a pre-emptive safety measure but may not be fully accurate.
        velocities = np.abs(np.array(target_joints) - self.current_robot_joints)
        
        if np.any(velocities > MAX_JOINT_VELOCITY):
            self.logger.error(f"Joint velocities {np.rad2deg(velocities)}°/s exceed maximum {np.rad2deg(MAX_JOINT_VELOCITY)}°/s")
            return False
        return True

    def _process_command_queue(self):
        """
        Process pending robot commands.
        
        This function checks the command queue and executes the next command
        if conditions are met (robot state received, timeout elapsed).
        """
        if not self.command_queue or self.executing_command:
            return

        if not self.robot_state_received:
            self.logger.warning("Waiting for robot state before executing commands")
            return

        current_time = self.get_clock().now().seconds_nanoseconds()[0]
        if self.last_command_time and (current_time - self.last_command_time) < self.command_timeout:
            return  # Wait for timeout between commands

        self.executing_command = True
        try:
            joint_angles = self.command_queue.pop(0)
            self._format_and_publish_ros2_command(joint_angles)
            self.last_command_time = current_time
        finally:
            self.executing_command = False

    def _format_and_publish_ros2_command(self, joint_angles: List[float]):
        """
        Format and publish joint angles as ROS2 command with safety checks.
        
        Parameters
        ----------
        joint_angles : List[float]
            List of 6 joint angles in radians
            
        Notes
        -----
        Performs joint limit and velocity validation before publishing
        """
        if not self._check_joint_limits(joint_angles):
            self.logger.error("Joint angles exceed safety limits")
            self.publish_status("Error: Command exceeds joint limits")
            return

        if not self._check_joint_velocity(joint_angles):
            self.logger.error("Joint velocities would exceed safety limits")
            self.publish_status("Error: Command would cause excessive joint velocity")
            return

        try:
            self.logger.info(f"Publishing joint command: {np.rad2deg(joint_angles)} degrees")
            ros2_command_msg = self.kinematics.format_ros2_command(joint_angles)
            self.joint_command_pub.publish(ros2_command_msg)
            self.publish_status(f"Command sent: Move to {np.rad2deg(joint_angles).round(2)} deg")
        except Exception as e:
            self.logger.error(f"Failed to publish command: {e}")
            self.publish_status("Error: Command publication failed")

    def _create_target_pose_matrix_from_detection(self, detection_3d: Detection3D) -> np.ndarray:
        """
        Create target pose matrix from 3D detection, handling coordinate frames correctly.
        
        Parameters
        ----------
        detection_3d : Detection3D
            3D detection with grasp point and approach vector in camera frame
            
        Returns
        -------
        np.ndarray
            4x4 transformation matrix in robot base frame
        """
        # First transform the grasp point and approach vector to robot base frame
        grasp_point_robot = self.calibration.camera_to_robot(detection_3d.grasp_point_3d)
        approach_vector_robot = self.calibration.transform_vector(detection_3d.approach_vector)
        
        # Normalize approach vector
        approach_vector_robot = approach_vector_robot / np.linalg.norm(approach_vector_robot)
        
        # Create rotation matrix in robot frame
        z_axis = approach_vector_robot
        x_axis = np.array([1.0, 0.0, 0.0])
        y_axis = np.cross(z_axis, x_axis)
        
        # Handle case where approach vector is aligned with x-axis
        if np.linalg.norm(y_axis) < 1e-6:
            x_axis = np.array([0.0, 1.0, 0.0])
            y_axis = np.cross(z_axis, x_axis)
        
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        
        # Create rotation matrix
        R = np.column_stack((x_axis, y_axis, z_axis))
        
        # Create transformation matrix in robot frame
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = grasp_point_robot
        
        return T

    def _publish_debug_target_pose(self, pose_matrix: np.ndarray):
        """
        Publish target pose as PoseStamped message for visualization in RViz.
        
        Parameters
        ----------
        pose_matrix : np.ndarray
            4x4 homogeneous transformation matrix
        """
        if pose_matrix is None: return

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "base_link" # Or your robot's base frame

        # Position
        pose_msg.pose.position.x = pose_matrix[0, 3]
        pose_msg.pose.position.y = pose_matrix[1, 3]
        pose_msg.pose.position.z = pose_matrix[2, 3]

        # Orientation (from rotation matrix to quaternion)
        # 간단한 변환, 더 강력한 라이브러리(scipy.spatial.transform.Rotation) 사용 가능
        q = self.rotation_matrix_to_quaternion(pose_matrix[:3, :3])
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]
        
        self.target_pose_pub.publish(pose_msg)
        self.logger.info("Published debug target pose to /target_pose_debug")

    def rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """
        Convert 3x3 rotation matrix to quaternion.
        
        Parameters
        ----------
        R : np.ndarray
            3x3 rotation matrix
            
        Returns
        -------
        np.ndarray
            Quaternion as [x, y, z, w]
            
        Notes
        -----
        Consider using a robust library like scipy.spatial.transform.Rotation
        or tf_transformations.quaternions for better numerical stability,
        especially to handle cases where qw might be close to zero.
        """
        # (Copied from a reliable source, e.g., Scipy or transformations.py)
        # Ensure R is a valid rotation matrix
        if not np.allclose(np.dot(R, R.T), np.eye(3)) or not np.isclose(np.linalg.det(R), 1.0):
            self.logger.warning("Provided matrix for quaternion conversion is not a valid rotation matrix.")
            # Fallback or error handling for invalid rotation matrix
            # For now, proceed but log warning. Consider raising an error.

        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2 # S = 4 * qw
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2 # S = 4 * qx
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
            qw = (R[2, 1] - R[1, 2]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2 # S = 4 * qy
            qy = 0.25 * S
            qx = (R[0, 1] + R[1, 0]) / S
            qz = (R[1, 2] + R[2, 1]) / S
            qw = (R[0, 2] - R[2, 0]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2 # S = 4 * qz
            qz = 0.25 * S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qw = (R[1, 0] - R[0, 1]) / S

        return np.array([qx, qy, qz, qw])

    def _visualize_current_detection(self, frame, detection_3d: Optional[Detection3D]):
        """
        Visualize current 3D detection on the frame.
        
        Parameters
        ----------
        frame : Union[np.ndarray, rs.frame]
            RGB image frame, can be either numpy array or RealSense frame
        detection_3d : Optional[Detection3D]
            3D detection to visualize, or None
        """
        if frame is None or detection_3d is None:
            return

        # Convert RealSense frame to numpy array if needed
        if isinstance(frame, rs.frame):
            processed_frame = np.asanyarray(frame.get_data())
        elif isinstance(frame, np.ndarray):
            processed_frame = frame # Already a numpy array
        else:
            self.logger.error(f"Unsupported frame type for visualization: {type(frame)}")
            return

        display_frame = self.depth_detector.visualize_3d_detection(processed_frame.copy(), detection_3d)
        cv2.imshow("Detection Preview", display_frame)
        cv2.waitKey(1)  # Essential for imshow to refresh

    def publish_status(self, message: str):
        """
        Publish status message to ROS2 topic.
        
        Parameters
        ----------
        message : str
            Status message to publish
        """
        status_msg = String()
        status_msg.data = message
        self.system_status_pub.publish(status_msg)
        self.logger.info(f"STATUS: {message}")

    def _joint_state_callback(self, msg: JointState):
        """
        Callback for joint state updates from robot.
        
        Parameters
        ----------
        msg : JointState
            ROS2 JointState message containing current joint positions
            
        Notes
        -----
        Updates current_robot_joints and robot_state_received flag
        """
        if not msg.position or len(msg.position) != 6:
            self.logger.warning("Received invalid joint state message")
            return
            
        self.current_robot_joints = np.array(msg.position)
        if not self.robot_state_received:
            self.robot_state_received = True
            self.logger.info("Received first robot joint state")
            self.publish_status("Robot state monitoring active")

    def _emergency_stop_callback(self, msg: Bool):
        """
        Callback for emergency stop signal.
        
        Parameters
        ----------
        msg : Bool
            ROS2 Bool message indicating emergency stop
            
        Notes
        -----
        Triggers system shutdown on emergency stop
        """
        if msg.data:
            self.logger.warning("Emergency stop signal received. Stopping system.")
            self.publish_status("Emergency stop signal received. Stopping system.")
            self.shutdown()

    def run_pipeline_once(self):
        """Execute one full detection and command generation cycle"""
        if not self.pipeline_started:
            self.logger.error("Pipeline not started")
            return
        
        self.publish_status("Awaiting voice command...")
        self.active_object_queries = self._process_sound_input()

        if not self.active_object_queries:
            return 

        self.publish_status(f"Queries received: {self.active_object_queries}. Capturing frame...")
        
        # 1. Get Frames (Color and Depth) - Depth is always on
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)  # Add timeout
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                self.logger.error("Failed to capture frames from depth camera.")
                self.publish_status("Error: Failed to capture camera frames.")
                return

            # Convert frames to numpy arrays for processing
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            # Keep the rs.depth_frame object for filtering if needed
            depth_frame_rs_obj = depth_frame 

        except rs.error as e:
            self.logger.error(f"RealSense error: {e}")
            return
        except Exception as e:
            self.logger.error(f"Frame capture error: {e}")
            return

        # 2. VLM Detection
        vlm_detections_2d = self._perform_vlm_detection(color_image, self.active_object_queries)
        if not vlm_detections_2d:
            self.logger.warning("VLM detection yielded no results for the current queries.")
            self._visualize_current_detection(color_image, None)
            return

        # 3. Depth-Aware 3D Detection
        # Pass color_image (np.ndarray), depth_frame_rs_obj (rs.frame), and depth_image (np.ndarray from raw depth_frame_rs_obj)
        valid_detection_3d = self._perform_depth_aware_detection(color_image, depth_frame_rs_obj, depth_image, vlm_detections_2d)
        self.last_successful_detection_3d = valid_detection_3d
        self._visualize_current_detection(color_image, self.last_successful_detection_3d)

        if not valid_detection_3d:
            self.logger.warning("No valid, graspable 3D object found after depth processing and validation.")
            self.publish_status("No graspable object found matching criteria.")
            return
        
        self.publish_status(f"Object '{valid_detection_3d.label}' localized in 3D. Calculating kinematics.")

        # Create target pose matrix directly in robot frame
        target_pose_matrix_robot = self._create_target_pose_matrix_from_detection(valid_detection_3d)
        
        # Validate the transformation
        if not self._validate_robot_pose(target_pose_matrix_robot):
            self.logger.error("Generated robot pose is invalid or unsafe")
            self.publish_status("Error: Invalid robot pose generated")
            return
            
        self._publish_debug_target_pose(target_pose_matrix_robot)

        # Calculate inverse kinematics
        joint_solution = self._calculate_inverse_kinematics(target_pose_matrix_robot)
        self.last_joint_solution = joint_solution

        if not joint_solution:
            self.logger.error("Failed to find an IK solution for the target pose.")
            self.publish_status(f"IK failed for {valid_detection_3d.label}. Cannot move.")
            return
        
        self.publish_status(f"IK solution found for {valid_detection_3d.label}. Sending command.")
        self._format_and_publish_ros2_command(joint_solution)
        
        # Clear active queries after execution
        self.active_object_queries = []

    def _validate_robot_pose(self, pose_matrix: np.ndarray) -> bool:
        """
        Validate if the robot pose is safe and reachable.
        
        Parameters
        ----------
        pose_matrix : np.ndarray
            4x4 transformation matrix in robot base frame
            
        Returns
        -------
        bool
            True if pose is valid and safe, False otherwise
        """
        try:
            # Extract position
            position = pose_matrix[:3, 3]
            
            # Check if position is within workspace bounds
            if not self.workspace_validator.is_position_reachable(position):
                self.logger.warning(f"Position {position} is outside workspace bounds")
                return False
            
            # Check orientation constraints
            rotation = pose_matrix[:3, :3]
            if not self.workspace_validator.is_orientation_valid(rotation):
                self.logger.warning("Generated orientation violates constraints")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating robot pose: {e}")
            return False

    def run(self):
        """
        Main execution loop for the Unified Vision System.
        
        This function:
        - Starts the main processing loop
        - Handles ROS2 callbacks
        - Manages system shutdown
        
        Notes
        -----
        Loop continues until ROS2 shutdown or keyboard interrupt
        """
        self.logger.info("🚀 Starting Unified Vision System main loop...")
        try:
            while rclpy.ok():
                self.run_pipeline_once() # Process one cycle
                rclpy.spin_once(self, timeout_sec=0.05) # Handle ROS callbacks, keep responsive
                
                # Visualization update if not handled in run_pipeline_once
                # If using a persistent color_frame, update visualization here
                # For now, visualization is part of the pipeline execution
                
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received. Shutting down...")
        except Exception as e:
            self.logger.critical(f"Critical error in main loop: {e}", exc_info=True)
            self.publish_status(f"Critical system error: {e}")
        finally:
            self.shutdown()

    def shutdown(self):
        """
        Clean up resources on system shutdown.
        
        This function:
        - Stops depth camera
        - Stops voice recognition
        - Closes visualization windows
        - Logs shutdown completion
        """
        self.logger.info("Initiating system shutdown...")
        self.publish_status("System shutting down.")
        if hasattr(self.depth_detector, 'cleanup'):
            self.depth_detector.cleanup()
        if hasattr(self.speech_processor, 'stop_listening'): # Assuming a stop method
             self.speech_processor.stop_listening()
        # if self.speech_thread and self.speech_thread.is_alive():
        #     # Add a way to signal the thread to stop if necessary
        #     self.speech_thread.join(timeout=1.0) # Wait for speech thread to close
        cv2.destroyAllWindows()
        self.logger.info("🔴 Unified Vision System shutdown complete.")

    def cleanup(self):
        """Clean up system resources"""
        self.logger.info("Cleaning up vision system resources...")
        
        try:
            # Stop RealSense pipeline
            if self.pipeline and self.pipeline_started:
                self.pipeline.stop()
                self.logger.info("RealSense pipeline stopped")
            
            # Clean up OpenCV windows
            cv2.destroyAllWindows()
            
            # Clean up other components
            if hasattr(self, 'speech_processor'):
                self.speech_processor.cleanup()
            
            if hasattr(self, 'depth_detector'):
                self.depth_detector.cleanup()
            
            self.logger.info("Vision system cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

    def perform_hand_eye_calibration(self, num_poses: int = 10, save_file: str = 'hand_eye_calib.npz'):
        """
        Perform hand-eye calibration between camera and robot.
        
        Parameters
        ----------
        num_poses : int, optional
            Number of calibration poses to collect, by default 10
        save_file : str, optional
            Path to save calibration results, by default 'hand_eye_calib.npz'
            
        Notes
        -----
        This method:
        1. Creates a HandEyeCalibrator instance. 
           WARNING: HandEyeCalibrator as defined in its own file also tries to initialize a RealSense pipeline.
           This will conflict with the pipeline already active in UnifiedVisionSystem.
           For integrated calibration, HandEyeCalibrator should be adapted to use the existing pipeline
           and a method to get robot poses from this system (e.g., via TF or current joint states).
           Alternatively, run HandEyeCalibrator as a separate ROS2 node/script for calibration.
        2. Collects calibration data from multiple poses
        3. Computes the calibration matrix
        4. Saves the result and updates the current calibration
        """
        self.logger.info("Starting hand-eye calibration procedure...")
        
        try:
            # Create calibrator
            calibrator = HandEyeCalibrator()
            
            # Collect calibration data
            calibrator.collect_calibration_data(num_poses)
            
            # Compute calibration
            T_base2cam = calibrator.compute_calibration()
            
            # Save calibration and update current transform
            calibrator.save_calibration(T_base2cam, save_file)
            self.calibration.load_hand_eye_transform(save_file)
            
            self.logger.info("✅ Hand-eye calibration completed successfully")
            self.publish_status("Hand-eye calibration completed")
            
        except Exception as e:
            self.logger.error(f"Hand-eye calibration failed: {e}")
            self.publish_status("Hand-eye calibration failed")
            raise

def main(args=None):
    rclpy.init(args=args)
    try:
        vision_system_node = UnifiedVisionSystem()
        vision_system_node.run()
    except Exception as e:
        # Log any exceptions that occur during node initialization or run
        if rclpy.ok():
             node_for_logging = rclpy.create_node("unified_vision_system_crash_logger")
             node_for_logging.get_logger().fatal(f"Unhandled exception in UnifiedVisionSystem: {e}", exc_info=True)
             node_for_logging.destroy_node()
        else:
             print(f"Unhandled exception BEFORE/DURING rclpy.init or AFTER rclpy.shutdown: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()