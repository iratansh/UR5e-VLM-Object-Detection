"""
Integrated Vision-Controlled UR5e System
Combines voice commands, object detection, and UR5e control
"""

import cv2
import numpy as np
import time
import logging
import threading
import queue
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# Import your existing modules
try:
    from OWLViTDetector import OWLViTDetector
    from SpeechCommandProcessor import SpeechCommandProcessor
    from CameraCalibration import CameraCalibration
    from UR5eKinematics import UR5eKinematics  # Our kinematics system
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")

# ROS2 imports (optional)
try:
    import rclpy
    from rclpy.node import Node
    from trajectory_msgs.msg import JointTrajectory
    from control_msgs.action import FollowJointTrajectory
    from rclpy.action import ActionClient
    ROS2_AVAILABLE = True
except ImportError:
    print("ROS2 not available - running in simulation mode")
    ROS2_AVAILABLE = False

@dataclass
class SystemState:
    """Current system state"""
    listening: bool = False
    processing: bool = False
    robot_busy: bool = False
    last_command: str = ""
    last_detection: List[Tuple[str, float, List[int]]] = None
    error_message: str = ""

class UR5eControlNode(Node):
    """ROS2 node for UR5e control"""
    
    def __init__(self):
        super().__init__('ur5e_vision_controller')
        
        # Action client for joint trajectory control
        self.trajectory_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            '/joint_trajectory_controller/follow_joint_trajectory'
        )
        
        self.get_logger().info('UR5e Vision Controller Node started')

    def execute_trajectory(self, trajectory: JointTrajectory) -> bool:
        """Execute a joint trajectory on the robot"""
        if not self.trajectory_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Trajectory action server not available')
            return False
        
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = trajectory
        
        future = self.trajectory_client.send_goal_async(goal_msg)
        
        try:
            # Wait for result (simplified - should use proper async handling)
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
            goal_handle = future.result()
            
            if not goal_handle.accepted:
                self.get_logger().error('Trajectory goal rejected')
                return False
            
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=15.0)
            
            return result_future.result().result.error_code == 0
            
        except Exception as e:
            self.get_logger().error(f'Trajectory execution failed: {e}')
            return False

class IntegratedVisionSystem:
    """Main system integrating all components"""
    
    def __init__(self, use_ros2: bool = True, use_camera: bool = True):
        self.logger = logging.getLogger(__name__)
        self.state = SystemState()
        self.use_ros2 = use_ros2 and ROS2_AVAILABLE
        
        self.setup_components(use_camera)
        
        # Command processing queue
        self.command_queue = queue.Queue()
        self.processing_thread = None
        
        self.logger.info("Integrated Vision System initialized")

    def setup_components(self, use_camera: bool):
        """Initialize all system components"""
        
        # Vision components
        try:
            self.detector = OWLViTDetector()
            self.logger.info("✅ Object detector loaded")
        except Exception as e:
            self.logger.error(f"Failed to load detector: {e}")
            self.detector = None
        
        try:
            self.speech = SpeechCommandProcessor()
            self.logger.info("✅ Speech processor loaded")
        except Exception as e:
            self.logger.error(f"Failed to load speech processor: {e}")
            self.speech = None
        
        # Camera and calibration
        self.calibration = CameraCalibration()
        if use_camera:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                self.logger.warning("Camera not available")
                self.camera = None
        else:
            self.camera = None
        
        # ROS2 components
        self.kinematics = UR5eKinematics()
        
        if self.use_ros2:
            try:
                rclpy.init()
                self.ros_node = UR5eControlNode()
                self.logger.info("✅ ROS2 node initialized")
            except Exception as e:
                self.logger.error(f"ROS2 initialization failed: {e}")
                self.use_ros2 = False
                self.ros_node = None
        else:
            self.ros_node = None

    def start_system(self):
        """Start the complete system"""
        self.logger.info("🚀 Starting Integrated Vision System")
        
        if self.speech:
            self.speech.start_listening()
            self.state.listening = True
        
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        self._main_loop()

    def _main_loop(self):
        """Main system loop"""
        print("\n🤖 UR5e Vision Control System")
        print("=" * 50)
        print("Say commands like:")
        print("  - 'pick up the bottle'")
        print("  - 'grab the red cup'") 
        print("  - 'go home'")
        print("  - 'stop' to exit")
        print("-" * 50)
        
        try:
            while True:
                # Check for voice commands
                if self.speech and not self.state.robot_busy:
                    command = self.speech.get_command()
                    if command:
                        if 'stop' in command or 'exit' in command:
                            self.logger.info("Stop command received")
                            break
                        
                        self.command_queue.put(command)
                        self.state.last_command = command
                        print(f"🎤 Command: '{command}'")
                
                # Display current frame with detections
                self._display_camera_feed()
                
                # Small delay
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.logger.info("System interrupted by user")
        finally:
            self.shutdown()

    def _processing_loop(self):
        """Background thread for processing commands"""
        while True:
            try:
                # Wait for command
                command = self.command_queue.get(timeout=1.0)
                self.state.processing = True
                
                success = self._process_voice_command(command)
                
                if success:
                    print(f"✅ Command completed: '{command}'")
                else:
                    print(f"❌ Command failed: '{command}'")
                    if self.state.error_message:
                        print(f"   Error: {self.state.error_message}")
                
                self.state.processing = False
                self.command_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Processing error: {e}")
                self.state.processing = False

    def _process_voice_command(self, command: str) -> bool:
        """Process a voice command through the complete pipeline"""
        
        # 1. Capture current camera frame
        if not self.camera:
            self.state.error_message = "No camera available"
            return False
        
        ret, frame = self.camera.read()
        if not ret:
            self.state.error_message = "Failed to capture camera frame"
            return False
        
        # 2. Detect objects using VLM
        if not self.detector:
            self.state.error_message = "Object detector not available"
            return False
        
        queries = self.speech.parse_object_query(command) if self.speech else ['graspable object']
        if not queries:
            queries = ['graspable object']
        
        print(f"🔍 Searching for: {queries}")
        
        # Detect objects
        detections = self.detector.detect_with_text_queries(frame, queries, confidence_threshold=0.1)
        self.state.last_detection = detections
        
        if not detections:
            self.state.error_message = f"No objects found matching: {queries}"
            return False
        
        print(f"📦 Found {len(detections)} objects")
        for query, confidence, bbox in detections:
            print(f"   - {query}: {confidence:.2f} confidence")
        
        # 3. Format ROS2 commands
        ros2_commands = self.command_formatter.format_command(
            command, detections, self.calibration
        )
        
        if not ros2_commands['success']:
            self.state.error_message = ros2_commands.get('error_message', 'Command formatting failed')
            return False
        
        # 4. Execute robot commands
        return self._execute_robot_commands(ros2_commands)

    def _execute_robot_commands(self, ros2_commands: Dict[str, Any]) -> bool:
        """Execute the formatted ROS2 commands"""
        
        trajectories = ros2_commands.get('trajectories', [])
        gripper_commands = ros2_commands.get('gripper_commands', [])
        
        if not trajectories:
            self.state.error_message = "No trajectories to execute"
            return False
        
        self.state.robot_busy = True
        success = True
        
        try:
            print(f"🤖 Executing {len(trajectories)} trajectories...")
            
            if self.use_ros2 and self.ros_node:
                for i, trajectory in enumerate(trajectories):
                    print(f"   Executing trajectory {i+1}/{len(trajectories)}")
                    
                    for gripper_cmd in gripper_commands:
                        if gripper_cmd['timing'] == i:
                            print(f"   Gripper: {gripper_cmd['action']}")
                            # TODO: Add gripper control
                    
                    if not self.ros_node.execute_trajectory(trajectory):
                        self.state.error_message = f"Trajectory {i+1} execution failed"
                        success = False
                        break
                    
                    time.sleep(0.5)  # Small delay between trajectories
            else:
                # Simulation mode
                print("   Running in simulation mode...")
                for i, trajectory in enumerate(trajectories):
                    print(f"   Simulating trajectory {i+1}: {len(trajectory.points)} points")
                    if trajectory.points:
                        positions = trajectory.points[0].positions
                        print(f"   Target joints: {[f'{p:.2f}' for p in positions]}")
                    time.sleep(1.0)  # Simulate execution time
                    
                # Show ROS2 command for manual execution
                if trajectories:
                    first_trajectory = trajectories[0]
                    if first_trajectory.points:
                        joint_positions = first_trajectory.points[0].positions
                        cmd_string = self.command_formatter.get_joint_command_string(joint_positions)
                        print(f"\n📋 Manual ROS2 command:")
                        print(cmd_string)
                
        except Exception as e:
            self.logger.error(f"Robot execution error: {e}")
            self.state.error_message = str(e)
            success = False
        finally:
            self.state.robot_busy = False
        
        return success

    def _display_camera_feed(self):
        """Display camera feed with detection overlays"""
        if not self.camera:
            return
        
        ret, frame = self.camera.read()
        if not ret:
            return
        
        # Draw detection boxes if available
        if self.state.last_detection:
            for query, confidence, bbox in self.state.last_detection:
                x1, y1, x2, y2 = bbox
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{query}: {confidence:.2f}"
                label