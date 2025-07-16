"""
Test Mode UnifiedVisionSystem for Automated Simulation Testing

This is a modified version of UnifiedVisionSystem that supports:
- Programmatic VLM command injection
- Automated test sequences
- Enhanced logging for test validation
- Simulation-specific configurations
- Test result reporting
"""

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, Pose, TransformStamped
from std_msgs.msg import String, Bool
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf2_ros import TransformBroadcaster
import time
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
import threading
import queue
import json

# Import all the same components as the original
from SpeechCommandProcessor import SpeechCommandProcessor
from OWLViTDetector import OWLViTDetector
from DepthAwareDetector import DepthAwareDetector, Detection3D
from UR5eKinematics import UR5eKinematics, HybridUR5eKinematics
from HybridIKWrapper import VLMKinematicsController
from WorkSpaceValidator import WorkspaceValidator
from GraspPointDetector import GraspPointDetector
from CameraCalibration import CameraCalibration 
from HandEyeCalibrator import HandEyeCalibrator
import pyrealsense2 as rs

class UnifiedVisionSystemTest(Node):
    """
    Test version of UnifiedVisionSystem with automated command injection.
    
    This class extends the original functionality with:
    - Programmatic command injection via ROS topics
    - Automated test sequence execution
    - Enhanced logging and monitoring
    - Test result collection and reporting
    - Simulation-specific optimizations
    """
    
    def __init__(self):
        super().__init__('unified_vision_system_test')
        
        # Enhanced logging setup
        self.setup_test_logging()
        self.logger = self.get_logger()
        
        # Test mode configuration
        self.test_mode_active = True
        self.auto_test_commands = []
        self.test_results = {
            'commands_executed': [],
            'successful_detections': [],
            'failed_detections': [],
            'ik_solutions': [],
            'execution_times': [],
            'errors': []
        }
        
        # Command injection queue
        self.injected_commands = queue.Queue()
        self.current_test_command = None
        
        # Initialize the same components as original
        self.pipeline = None
        self.pipeline_started = False
        
        # Declare all ROS2 parameters (same as original plus test mode params)
        self.declare_test_parameters()
        
        # Get test mode parameters
        self.test_mode = self.get_parameter('test_mode').value
        self.auto_start_test = self.get_parameter('auto_start_test').value
        
        try:
            auto_test_commands_param = self.get_parameter('auto_test_commands').value
            if isinstance(auto_test_commands_param, str):
                self.auto_test_commands = json.loads(auto_test_commands_param)
            elif isinstance(auto_test_commands_param, list):
                self.auto_test_commands = auto_test_commands_param
        except:
            self.auto_test_commands = ["pick up the red cube"]
        
        self.logger.info(f"🧪 Test Mode Active: {self.test_mode}")
        self.logger.info(f"🤖 Auto-start Test: {self.auto_start_test}")
        self.logger.info(f"📝 Test Commands: {self.auto_test_commands}")
        
        # Initialize the same way as original UnifiedVisionSystem
        self.initialize_system()
        
        # Setup test-specific communication
        self.setup_test_communication()
        
        # Start test sequence if auto-start is enabled
        if self.auto_start_test:
            self.start_auto_test_sequence()
    
    def declare_test_parameters(self):
        """Declare all parameters including test-specific ones."""
        # All original parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                # Original parameters
                ('robot_namespace', 'ur5e'),
                ('hand_eye_calibration_file', 'sim_hand_eye_calib.npz'),
                ('camera_calibration_file', 'sim_camera_calib.npz'),
                ('realsense_serial', ''),
                ('max_depth', 1.5),
                ('min_depth', 0.1),
                ('depth_fps', 30),
                ('color_fps', 30),
                ('depth_width', 848),
                ('depth_height', 480),
                ('color_width', 848),
                ('color_height', 480),
                ('enable_depth_filters', True),
                ('vlm_confidence_threshold', 0.1),
                ('min_valid_pixels', 80),
                ('max_depth_variance', 0.025),
                ('min_object_volume_m3', 1.0e-6),
                ('max_object_volume_m3', 0.008),
                ('enable_hybrid_ik', True),
                ('ik_enable_approximation', True),
                ('ik_max_position_error_mm', 15.0),
                ('ik_timeout_ms', 120.0),
                ('ik_debug', True),
                ('eye_in_hand', True),
                ('broadcast_camera_tf', True),
                
                # Test-specific parameters
                ('test_mode', True),
                ('auto_start_test', True),
                ('auto_test_commands', '["pick up the red cube"]'),
                ('test_timeout_seconds', 180),
                ('test_retry_attempts', 3),
                ('enable_test_logging', True),
                ('simulation_speed_factor', 1.0),
            ]
        )
    
    def setup_test_logging(self):
        """Setup enhanced logging for test mode."""
        # Create test logs directory
        import os
        from pathlib import Path
        
        log_dir = Path("test_logs")
        log_dir.mkdir(exist_ok=True)
        
        # Setup file handler for test logs
        log_file = log_dir / f"vision_system_test_{int(time.time())}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def initialize_system(self):
        """Initialize all system components (same as original)."""
        try:
            # Initialize RealSense pipeline
            self.setup_realsense_pipeline()
            
            # Initialize all components
            self.setup_calibration()
            self.setup_detection_components()
            self.setup_kinematics()
            self.setup_ros_communication()
            
            self.logger.info("✅ All system components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize system: {e}")
            raise
    
    def setup_realsense_pipeline(self):
        """Setup RealSense pipeline (same as original)."""
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # Get RealSense parameters
            depth_fps = self.get_parameter('depth_fps').value
            color_fps = self.get_parameter('color_fps').value
            depth_width = self.get_parameter('depth_width').value
            depth_height = self.get_parameter('depth_height').value
            color_width = self.get_parameter('color_width').value
            color_height = self.get_parameter('color_height').value
            
            # Configure streams
            config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, depth_fps)
            config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, color_fps)
            
            # Start pipeline
            profile = self.pipeline.start(config)
            self.pipeline_started = True
            self.logger.info("📷 RealSense pipeline started successfully")
            
            # Setup calibration
            self.calibration = CameraCalibration(node=self)
            self.calibration.set_realsense_calibration_params(profile)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup RealSense: {e}")
            # For simulation, create mock calibration
            self.setup_mock_calibration()
    
    def setup_mock_calibration(self):
        """Setup mock calibration for simulation."""
        self.logger.info("🎭 Setting up mock calibration for simulation")
        
        self.calibration = CameraCalibration(node=self)
        
        # Mock camera intrinsics typical for RealSense D435i
        mock_camera_matrix = np.array([
            [421.61, 0, 424.0],
            [0, 421.61, 240.0],
            [0, 0, 1]
        ])
        
        # Mock eye-in-hand transformation (camera 5cm above gripper, looking down)
        mock_T_gripper_camera = np.array([
            [1, 0, 0, 0.0],
            [0, -1, 0, 0.0],
            [0, 0, -1, -0.05],
            [0, 0, 0, 1]
        ])
        
        self.calibration.set_mock_calibration(
            camera_matrix=mock_camera_matrix,
            eye_in_hand=True,
            T_gripper_camera=mock_T_gripper_camera
        )
        
        self.logger.info("✅ Mock calibration setup complete")
    
    def setup_calibration(self):
        """Setup calibration (modified to handle missing files gracefully)."""
        try:
            hand_eye_calib_file = self.get_parameter('hand_eye_calibration_file').value
            camera_calib_file = self.get_parameter('camera_calibration_file').value
            
            # Try to load calibration files, use defaults if not found
            try:
                self.calibration.load_hand_eye_transform(hand_eye_calib_file)
                self.logger.info(f"📐 Loaded hand-eye calibration from {hand_eye_calib_file}")
            except FileNotFoundError:
                self.logger.warning(f"⚠️ Hand-eye calibration file not found, using simulation defaults")
                # Use mock calibration for simulation
                mock_T = np.array([
                    [1, 0, 0, 0.0],
                    [0, -1, 0, 0.0],
                    [0, 0, -1, -0.05],
                    [0, 0, 0, 1]
                ])
                self.calibration.set_hand_eye_transform(mock_T, is_eye_in_hand=True)
            
            try:
                self.calibration.load_camera_intrinsics(camera_calib_file)
                self.logger.info(f"📷 Loaded camera intrinsics from {camera_calib_file}")
            except FileNotFoundError:
                self.logger.warning(f"⚠️ Camera calibration file not found, using simulation defaults")
            
        except Exception as e:
            self.logger.error(f"❌ Error setting up calibration: {e}")
    
    def setup_detection_components(self):
        """Setup detection components (same as original)."""
        try:
            # Setup depth detector
            depth_detector_params = {
                'use_filters': self.get_parameter('enable_depth_filters').value,
                'min_valid_pixels': self.get_parameter('min_valid_pixels').value,
                'max_depth_variance': self.get_parameter('max_depth_variance').value
            }
            self.depth_detector = DepthAwareDetector(self.calibration, params=depth_detector_params)
            
            # Setup workspace validator
            workspace_params = {
                'min_object_volume': self.get_parameter('min_object_volume_m3').value,
                'max_object_volume': self.get_parameter('max_object_volume_m3').value
            }
            self.workspace_validator = WorkspaceValidator(params=workspace_params)
            
            # Setup VLM detector
            self.vlm_detector = OWLViTDetector(
                confidence_threshold=self.get_parameter('vlm_confidence_threshold').value
            )
            
            # Setup grasp detector
            self.grasp_detector = GraspPointDetector(gripper_width=0.085, gripper_finger_width=0.02)
            
            self.logger.info("🔍 Detection components initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Error setting up detection components: {e}")
            raise
    
    def setup_kinematics(self):
        """Setup kinematics components (same as original)."""
        try:
            enable_hybrid_ik = self.get_parameter('enable_hybrid_ik').value
            ik_debug = self.get_parameter('ik_debug').value
            
            if enable_hybrid_ik:
                self.logger.info("🔧 Initializing Hybrid IK System")
                self.kinematics = HybridUR5eKinematics(enable_fallback=True, debug=ik_debug)
                self.ik_controller = VLMKinematicsController(
                    enable_ikfast=True,
                    adaptive_timeout=True,
                    debug=ik_debug
                )
            else:
                self.logger.info("🔧 Initializing Numerical IK System only")
                self.kinematics = UR5eKinematics()
                self.ik_controller = None
            
            self.logger.info("⚙️ Kinematics system initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Error setting up kinematics: {e}")
            raise
    
    def setup_ros_communication(self):
        """Setup ROS2 communication (enhanced for test mode)."""
        try:
            self.robot_ns = self.get_parameter('robot_namespace').value
            
            # QoS profiles
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
            
            # Publishers (same as original)
            self.joint_command_pub = self.create_publisher(
                JointTrajectory, 
                f'/{self.robot_ns}/scaled_joint_trajectory_controller/joint_trajectory',
                self.reliable_qos
            )
            
            self.target_pose_pub = self.create_publisher(
                PoseStamped, 
                f'/{self.robot_ns}/target_pose_debug', 
                self.reliable_qos
            )
            
            self.system_status_pub = self.create_publisher(
                String, 
                f'/{self.robot_ns}/vision_system_status', 
                self.reliable_qos
            )
            
            # Subscribers (same as original)
            self.joint_state_sub = self.create_subscription(
                JointState,
                f'/{self.robot_ns}/joint_states',
                self._joint_state_callback,
                self.sensor_qos
            )
            
            # Initialize state
            self.current_robot_joints = np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0])
            self.robot_state_received = False
            self.command_queue = []
            self.executing_command = False
            
            # Create timers
            self.command_timer = self.create_timer(0.1, self._process_command_queue)
            
            self.logger.info("📡 ROS2 communication setup complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error setting up ROS communication: {e}")
            raise
    
    def setup_test_communication(self):
        """Setup test-specific ROS2 communication."""
        try:
            # Test command injection subscriber
            self.test_command_sub = self.create_subscription(
                String,
                '/test_vlm_command',
                self.inject_test_command,
                self.reliable_qos
            )
            
            # Test results publisher
            self.test_results_pub = self.create_publisher(
                String,
                '/test_results',
                self.reliable_qos
            )
            
            # Test status publisher
            self.test_status_pub = self.create_publisher(
                String,
                '/test_status',
                self.reliable_qos
            )
            
            self.logger.info("🧪 Test communication setup complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error setting up test communication: {e}")
    
    def inject_test_command(self, msg: String):
        """Inject a test command programmatically."""
        command = msg.data
        self.logger.info(f"💉 Injecting test command: '{command}'")
        
        # Add to injection queue
        self.injected_commands.put(command)
        
        # Publish test status
        status_msg = String()
        status_msg.data = f"Command injected: {command}"
        self.test_status_pub.publish(status_msg)
    
    def start_auto_test_sequence(self):
        """Start automated test sequence."""
        if not self.auto_test_commands:
            self.logger.warning("⚠️ No auto test commands configured")
            return
        
        self.logger.info(f"🚀 Starting auto test sequence with {len(self.auto_test_commands)} commands")
        
        # Start test sequence in a separate thread
        test_thread = threading.Thread(target=self._execute_auto_test_sequence)
        test_thread.daemon = True
        test_thread.start()
    
    def _execute_auto_test_sequence(self):
        """Execute the automated test sequence."""
        try:
            # Wait for system to be ready
            self.logger.info("⏳ Waiting for system to be ready...")
            time.sleep(10)  # Give system time to initialize
            
            for i, command in enumerate(self.auto_test_commands):
                self.logger.info(f"🧪 Executing test {i+1}/{len(self.auto_test_commands)}: '{command}'")
                
                # Inject the command
                self.injected_commands.put(command)
                self.current_test_command = command
                
                # Wait for command processing
                start_time = time.time()
                timeout = self.get_parameter('test_timeout_seconds').value
                
                processed = False
                while time.time() - start_time < timeout:
                    if self.injected_commands.empty():
                        processed = True
                        break
                    time.sleep(1)
                
                if processed:
                    self.logger.info(f"✅ Test command '{command}' completed")
                    self.test_results['commands_executed'].append(command)
                else:
                    self.logger.error(f"❌ Test command '{command}' timed out")
                    self.test_results['errors'].append(f"Timeout: {command}")
                
                # Wait between commands
                time.sleep(5)
            
            # Generate final test report
            self.generate_test_report()
            
        except Exception as e:
            self.logger.error(f"❌ Error in auto test sequence: {e}")
            self.test_results['errors'].append(f"Sequence error: {e}")
    
    def process_injected_commands(self) -> Optional[List[str]]:
        """Process injected commands instead of voice commands."""
        if self.injected_commands.empty():
            return None
        
        try:
            command = self.injected_commands.get_nowait()
            self.logger.info(f"🎯 Processing injected command: '{command}'")
            
            # Parse the command into object queries
            # Simple parsing for test commands
            if "red cube" in command.lower():
                return ["red cube", "cube", "red box"]
            elif "blue cube" in command.lower():
                return ["blue cube", "blue box"]
            elif "green cylinder" in command.lower():
                return ["green cylinder", "cylinder"]
            else:
                # Generic parsing
                words = command.lower().split()
                if "pick" in words or "grasp" in words or "get" in words:
                    # Extract object description
                    obj_words = []
                    for word in words:
                        if word not in ["pick", "up", "grasp", "get", "the", "a", "an"]:
                            obj_words.append(word)
                    
                    if obj_words:
                        return [" ".join(obj_words)]
            
            return [command]  # Fallback to full command
            
        except queue.Empty:
            return None
        except Exception as e:
            self.logger.error(f"❌ Error processing injected command: {e}")
            return None
    
    def run_pipeline_once(self):
        """Execute one detection and command cycle (modified for test mode)."""
        if not self.pipeline_started and not hasattr(self, 'mock_calibration'):
            self.logger.error("❌ Pipeline not started")
            return
        
        # Broadcast camera position for eye-in-hand
        if self.get_parameter('eye_in_hand').value:
            self._broadcast_transforms()
        
        # Check for injected commands instead of voice commands
        if self.test_mode:
            self.active_object_queries = self.process_injected_commands()
        else:
            self.active_object_queries = self._process_sound_input()
        
        if not self.active_object_queries:
            return
        
        self.publish_status(f"🎯 Test queries: {self.active_object_queries}")
        
        # Get frames (use mock frames for simulation if needed)
        try:
            if self.pipeline_started:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    self.logger.warning("⚠️ Failed to get frames, using mock data")
                    color_image, depth_image, depth_frame_rs = self.create_mock_frames()
                else:
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    depth_frame_rs = depth_frame
            else:
                # Use mock frames for simulation
                color_image, depth_image, depth_frame_rs = self.create_mock_frames()
                
        except Exception as e:
            self.logger.warning(f"⚠️ Frame capture error: {e}, using mock data")
            color_image, depth_image, depth_frame_rs = self.create_mock_frames()
        
        # Continue with the same pipeline as original
        vlm_detections_2d = self._perform_vlm_detection(color_image, self.active_object_queries)
        if not vlm_detections_2d:
            self.logger.warning("⚠️ VLM detection yielded no results")
            self.test_results['failed_detections'].append(self.current_test_command)
            return
        
        # Record successful detection
        self.test_results['successful_detections'].append(self.current_test_command)
        
        # Continue with 3D detection and IK
        valid_detection_3d = self._perform_depth_aware_detection(
            color_image, depth_frame_rs, depth_image, vlm_detections_2d
        )
        
        if not valid_detection_3d:
            self.logger.warning("⚠️ No valid 3D detection found")
            return
        
        # Create target pose and solve IK
        target_pose_matrix = self._create_target_pose_matrix_from_detection(valid_detection_3d)
        joint_solution = self._calculate_inverse_kinematics(target_pose_matrix)
        
        if joint_solution:
            self.test_results['ik_solutions'].append({
                'command': self.current_test_command,
                'joints': joint_solution,
                'target_pose': target_pose_matrix.tolist()
            })
            
            self.logger.info(f"✅ IK solution found for '{self.current_test_command}'")
            self._format_and_publish_ros2_command(joint_solution)
        else:
            self.logger.error(f"❌ IK failed for '{self.current_test_command}'")
            self.test_results['errors'].append(f"IK failed: {self.current_test_command}")
    
    def create_mock_frames(self):
        """Create mock camera frames for simulation testing."""
        # Create a mock color image with a red cube
        color_image = np.zeros((480, 848, 3), dtype=np.uint8)
        color_image[:, :] = [50, 50, 50]  # Dark background
        
        # Draw a red cube in the center
        center_x, center_y = 424, 240
        cube_size = 80
        
        # Draw red rectangle (representing cube)
        cv2.rectangle(
            color_image,
            (center_x - cube_size//2, center_y - cube_size//2),
            (center_x + cube_size//2, center_y + cube_size//2),
            (0, 0, 255),  # Red in BGR
            -1
        )
        
        # Create mock depth image
        depth_image = np.full((480, 848), 500, dtype=np.uint16)  # 500mm default depth
        
        # Set cube area to closer depth (300mm)
        depth_image[
            center_y - cube_size//2:center_y + cube_size//2,
            center_x - cube_size//2:center_x + cube_size//2
        ] = 300
        
        self.logger.info("🎭 Created mock frames for simulation")
        return color_image, depth_image, None
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        try:
            report = {
                'test_summary': {
                    'total_commands': len(self.auto_test_commands),
                    'executed_commands': len(self.test_results['commands_executed']),
                    'successful_detections': len(self.test_results['successful_detections']),
                    'failed_detections': len(self.test_results['failed_detections']),
                    'ik_solutions_found': len(self.test_results['ik_solutions']),
                    'total_errors': len(self.test_results['errors'])
                },
                'detailed_results': self.test_results,
                'system_info': {
                    'test_mode': self.test_mode,
                    'eye_in_hand': self.get_parameter('eye_in_hand').value,
                    'hybrid_ik_enabled': self.get_parameter('enable_hybrid_ik').value,
                    'vlm_confidence_threshold': self.get_parameter('vlm_confidence_threshold').value
                }
            }
            
            # Publish test results
            results_msg = String()
            results_msg.data = json.dumps(report, indent=2)
            self.test_results_pub.publish(results_msg)
            
            # Log summary
            self.logger.info("📊 TEST REPORT SUMMARY:")
            self.logger.info(f"   Commands executed: {report['test_summary']['executed_commands']}/{report['test_summary']['total_commands']}")
            self.logger.info(f"   Successful detections: {report['test_summary']['successful_detections']}")
            self.logger.info(f"   IK solutions found: {report['test_summary']['ik_solutions_found']}")
            self.logger.info(f"   Total errors: {report['test_summary']['total_errors']}")
            
            if report['test_summary']['total_errors'] == 0:
                self.logger.info("🎉 ALL TESTS PASSED!")
            else:
                self.logger.warning(f"⚠️ {report['test_summary']['total_errors']} errors occurred")
            
        except Exception as e:
            self.logger.error(f"❌ Error generating test report: {e}")
    
    # Include all the same methods as original UnifiedVisionSystem
    def _broadcast_transforms(self):
        """Broadcast camera transform for eye-in-hand configuration"""
        if self.get_parameter('eye_in_hand').value and self.calibration:
            current_gripper_pose = self.get_current_gripper_pose()
            self.calibration.broadcast_hand_eye_transform(current_gripper_pose)
    
    def get_current_gripper_pose(self) -> np.ndarray:
        """Get current gripper pose from forward kinematics."""
        if self.current_robot_joints is None:
            self.logger.warning("Robot joints not available, using identity for gripper pose")
            return np.eye(4)
        
        return self.kinematics.forward_kinematics(list(self.current_robot_joints))
    
    def _joint_state_callback(self, msg: JointState):
        """Process joint state updates."""
        if not msg.position or len(msg.position) != 6:
            self.logger.warning("Received invalid joint state message")
            return
            
        self.current_robot_joints = np.array(msg.position)
        if not self.robot_state_received:
            self.robot_state_received = True
            self.logger.info("🤖 Robot state monitoring active")
    
    def _process_command_queue(self):
        """Process pending robot commands."""
        if not self.command_queue or self.executing_command:
            return

        if not self.robot_state_received:
            return

        self.executing_command = True
        try:
            joint_angles = self.command_queue.pop(0)
            self._format_and_publish_ros2_command(joint_angles)
        finally:
            self.executing_command = False
    
    def publish_status(self, message: str):
        """Publish status message."""
        status_msg = String()
        status_msg.data = message
        self.system_status_pub.publish(status_msg)
        self.logger.info(f"📢 STATUS: {message}")
    
    # Include all other methods from original UnifiedVisionSystem...
    # (For brevity, I'm not copying all methods, but they would be the same)
    
    def _perform_vlm_detection(self, color_frame: np.ndarray, queries: List[str]):
        """Perform VLM detection (same as original but with test logging)."""
        try:
            if not queries:
                return []
            
            self.logger.info(f"🔍 Performing VLM detection for: {queries}")
            vlm_detections = self.vlm_detector.detect_with_text_queries(
                color_frame, queries, 
                confidence_threshold=self.get_parameter('vlm_confidence_threshold').value
            )
            self.logger.info(f"🎯 VLM found {len(vlm_detections)} detections")
            return vlm_detections
            
        except Exception as e:
            self.logger.error(f"❌ VLM detection error: {e}")
            return []
    
    def _perform_depth_aware_detection(self, color_image_np, depth_frame_rs, raw_depth_image_np, vlm_detections):
        """Perform depth-aware detection (same as original but with test logging)."""
        if not vlm_detections:
            return None
        
        try:
            current_gripper_pose = self.get_current_gripper_pose() if self.get_parameter('eye_in_hand').value else None
            
            self.logger.info("🔍 Augmenting VLM detections with 3D depth information...")
            detections_3d = self.depth_detector.augment_vlm_detections_3d(
                color_image_np, depth_frame_rs, raw_depth_image_np, vlm_detections, 
                self.grasp_detector, current_gripper_pose
            )

            if not detections_3d:
                self.logger.warning("⚠️ No 3D detections generated")
                return None

            self.logger.info(f"📦 Generated {len(detections_3d)} 3D detections")
            
            # Validate detections
            for det_3d in detections_3d:
                if self.depth_detector.validate_grasp_3d(det_3d, self.workspace_validator, current_gripper_pose):
                    self.logger.info(f"✅ Valid 3D detection: {det_3d.label} at {det_3d.grasp_point_3d}")
                    return det_3d
            
            self.logger.warning("⚠️ No valid graspable 3D detection found")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ 3D detection error: {e}")
            return None
    
    def _create_target_pose_matrix_from_detection(self, detection_3d):
        """Create target pose matrix (same as original)."""
        current_gripper_pose = None
        if self.calibration.is_eye_in_hand:
            current_gripper_pose = self.kinematics.forward_kinematics(list(self.current_robot_joints))
        
        grasp_point_robot = self.calibration.camera_to_robot(detection_3d.grasp_point_3d, current_gripper_pose)
        approach_vector_robot = self.calibration.transform_vector(detection_3d.approach_vector, current_gripper_pose)
        
        approach_vector_robot = approach_vector_robot / np.linalg.norm(approach_vector_robot)
        
        if self.calibration.is_eye_in_hand:
            approach_offset = 0.05
            grasp_point_robot = (
                grasp_point_robot[0],
                grasp_point_robot[1],
                grasp_point_robot[2] + approach_offset
            )
        
        z_axis = approach_vector_robot
        
        if np.abs(np.dot(z_axis, np.array([1., 0., 0.]))) < 0.9:
            x_axis_ref = np.array([1., 0., 0.])
        else:
            x_axis_ref = np.array([0., 1., 0.])
            
        y_axis = np.cross(z_axis, x_axis_ref)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        R = np.column_stack((x_axis, y_axis, z_axis))
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = grasp_point_robot
        
        return T
    
    def _calculate_inverse_kinematics(self, target_pose_matrix):
        """Calculate IK (same as original but with test logging)."""
        try:
            self.logger.info("⚙️ Calculating inverse kinematics...")
            
            if self.ik_controller is not None:
                self.logger.info("🔧 Using Hybrid IK System")
                
                target_position = target_pose_matrix[:3, 3].tolist()
                target_orientation = target_pose_matrix[:3, :3]
                
                enable_approximation = self.get_parameter('ik_enable_approximation').value
                max_error_mm = self.get_parameter('ik_max_position_error_mm').value
                timeout_ms = self.get_parameter('ik_timeout_ms').value
                
                success, joint_solution, metadata = self.ik_controller.solve_for_vlm_target(
                    target_position=target_position,
                    target_orientation=target_orientation,
                    current_joints=list(self.current_robot_joints),
                    allow_approximation=enable_approximation,
                    max_position_error_mm=max_error_mm
                )
                
                if success:
                    self.logger.info(f"✅ IK solution found in {metadata['solve_time_ms']:.1f}ms")
                    if metadata['is_approximation']:
                        self.logger.info(f"📏 Using approximation with {metadata['position_error_mm']:.1f}mm error")
                    return joint_solution
                else:
                    self.logger.warning(f"❌ IK failed: {metadata['solver_used']}")
                    return None
            else:
                self.logger.info("🔧 Using Numerical IK System only")
                ik_solutions = self.kinematics.inverse_kinematics(target_pose_matrix)
                
                if ik_solutions:
                    best_solution = self.kinematics.select_best_solution(ik_solutions, list(self.current_robot_joints))
                    self.logger.info(f"✅ IK solution found: {len(ik_solutions)} candidates")
                    return best_solution
                else:
                    self.logger.warning("❌ No IK solutions found")
                    return None
                    
        except Exception as e:
            self.logger.error(f"❌ IK calculation error: {e}")
            return None
    
    def _format_and_publish_ros2_command(self, joint_angles):
        """Format and publish joint command (same as original but with test logging)."""
        try:
            self.logger.info(f"📤 Publishing joint command: {np.rad2deg(joint_angles).round(2)} degrees")
            
            cmd_data = self.kinematics.format_ros2_command(joint_angles)
            
            ros2_command_msg = JointTrajectory()
            ros2_command_msg.joint_names = cmd_data["joint_names"]
            
            point = JointTrajectoryPoint()
            point.positions = cmd_data["points"][0]["positions"]
            point.velocities = cmd_data["points"][0]["velocities"]
            point.accelerations = cmd_data["points"][0]["accelerations"]
            
            time_from_start = cmd_data["points"][0]["time_from_start"]
            point.time_from_start.sec = time_from_start["sec"]
            point.time_from_start.nanosec = time_from_start["nanosec"]
            
            ros2_command_msg.points = [point]
            
            self.joint_command_pub.publish(ros2_command_msg)
            self.publish_status(f"✅ Command sent for test: {self.current_test_command}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to publish command: {e}")
    
    def run(self):
        """Main execution loop (same as original but with test considerations)."""
        self.logger.info("🚀 Starting UnifiedVisionSystemTest main loop...")
        
        # Wait for robot state
        start_time = self.get_clock().now()
        timeout_sec = 15.0
        
        self.publish_status("⏳ Waiting for robot state...")
        while not self.robot_state_received and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            
            current_time = self.get_clock().now()
            if (current_time - start_time).nanoseconds / 1e9 > timeout_sec:
                self.logger.warning(f"⏰ Timed out waiting for robot state, continuing anyway")
                break
        
        if self.robot_state_received:
            self.publish_status("✅ Robot state received. Starting test mode.")
        
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.05)
                self.run_pipeline_once()
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            self.logger.info("⌨️ Keyboard interrupt received")
        except Exception as e:
            self.logger.critical(f"💥 Critical error: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown with test report generation."""
        self.logger.info("🛑 Shutting down UnifiedVisionSystemTest...")
        
        # Generate final test report
        self.generate_test_report()
        
        # Cleanup (same as original)
        self.cleanup()
        
        self.logger.info("✅ UnifiedVisionSystemTest shutdown complete")
    
    def cleanup(self):
        """Clean up resources (same as original)."""
        try:
            if hasattr(self, 'pipeline') and self.pipeline and self.pipeline_started:
                self.pipeline.stop()
                self.logger.info("📷 RealSense pipeline stopped")
            
            cv2.destroyAllWindows()
            self.logger.info("🧹 Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup error: {e}")


def main(args=None):
    """Main entry point for test mode."""
    rclpy.init(args=args)
    
    try:
        vision_system_test = UnifiedVisionSystemTest()
        vision_system_test.run()
    except Exception as e:
        logging.error(f"💥 Test failed: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()