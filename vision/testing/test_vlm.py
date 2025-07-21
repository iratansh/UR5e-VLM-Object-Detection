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
    import pyrealsense2 as rs
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

class RealSenseCamera:
    """RealSense D435i camera interface"""
    
    def __init__(self):
        self.pipeline = None
        self.config = None
        self.align = None
        self.logger = logging.getLogger(__name__)
        
    def start(self, width: int = 848, height: int = 480, fps: int = 30) -> bool:
        """Start RealSense camera pipeline"""
        try:
            # Create pipeline and config
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Configure streams
            self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            
            # Start streaming
            profile = self.pipeline.start(self.config)
            
            # Create align object to align depth to color
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            
            # Get device info
            device = profile.get_device()
            device_name = device.get_info(rs.camera_info.name)
            serial_number = device.get_info(rs.camera_info.serial_number)
            
            self.logger.info(f"✅ RealSense camera started: {device_name} (S/N: {serial_number})")
            self.logger.info(f"   Resolution: {width}x{height} @ {fps}fps")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start RealSense camera: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Read color and depth frames from RealSense"""
        if not self.pipeline:
            return False, None, None
            
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            
            # Align depth to color
            aligned_frames = self.align.process(frames)
            
            # Get aligned frames
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return False, None, None
            
            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            return True, color_image, depth_image
            
        except Exception as e:
            self.logger.warning(f"Failed to read RealSense frame: {e}")
            return False, None, None
    
    def stop(self):
        """Stop RealSense camera pipeline"""
        if self.pipeline:
            try:
                self.pipeline.stop()
                self.logger.info("RealSense camera stopped")
            except Exception as e:
                self.logger.error(f"Error stopping RealSense camera: {e}")
            finally:
                self.pipeline = None
    
    def is_opened(self) -> bool:
        """Check if camera is opened"""
        return self.pipeline is not None

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
            # Use RealSense camera instead of OpenCV
            self.camera = RealSenseCamera()
            if not self.camera.start():
                self.logger.warning("RealSense camera not available, falling back to OpenCV")
                # Fallback to OpenCV camera
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    self.logger.warning("No camera available")
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
        
        # Handle different camera types
        if isinstance(self.camera, RealSenseCamera):
            ret, frame, depth_frame = self.camera.read()
        else:
            # OpenCV camera fallback
            ret, frame = self.camera.read()
            depth_frame = None
            
        if not ret or frame is None:
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
        if not hasattr(self, 'command_formatter'):
            self.command_formatter = CommandFormatter(self.kinematics)
            
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
        
        # Handle different camera types
        if isinstance(self.camera, RealSenseCamera):
            ret, frame, depth_frame = self.camera.read()
        else:
            # OpenCV camera fallback
            ret, frame = self.camera.read()
            depth_frame = None
            
        if not ret or frame is None:
            return
        
        # Draw detection boxes if available
        if self.state.last_detection:
            for query, confidence, bbox in self.state.last_detection:
                x1, y1, x2, y2 = bbox
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{query}: {confidence:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add status overlay
        status_text = []
        if self.state.listening:
            status_text.append("🎤 Listening...")
        if self.state.processing:
            status_text.append("⚙️ Processing...")
        if self.state.robot_busy:
            status_text.append("🤖 Robot busy...")
        if self.state.last_command:
            status_text.append(f"Last: {self.state.last_command}")
        
        # Draw status
        y_offset = 30
        for text in status_text:
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        cv2.imshow('UR5e Vision System', frame)

    def shutdown(self):
        """Clean shutdown of all components"""
        self.logger.info("Shutting down Integrated Vision System...")
        
        self.state.listening = False
        
        if self.speech:
            self.speech.stop_listening()
        
        if self.camera:
            if isinstance(self.camera, RealSenseCamera):
                self.camera.stop()
            else:
                self.camera.release()
        
        if self.use_ros2 and self.ros_node:
            try:
                self.ros_node.destroy_node()
                rclpy.shutdown()
            except Exception as e:
                self.logger.error(f"ROS2 shutdown error: {e}")
        
        cv2.destroyAllWindows()
        self.logger.info("System shutdown complete")


class CommandFormatter:
    """Formats voice commands into ROS2 trajectories"""
    
    def __init__(self, kinematics: UR5eKinematics):
        self.kinematics = kinematics
        self.logger = logging.getLogger(__name__)
    
    def format_command(self, command: str, detections: List[Tuple[str, float, List[int]]], 
                      calibration: CameraCalibration) -> Dict[str, Any]:
        """Convert voice command and detections to ROS2 trajectories"""
        
        try:
            # Parse command type
            if any(word in command.lower() for word in ['pick', 'grab', 'get']):
                return self._create_pick_sequence(detections, calibration)
            elif 'home' in command.lower():
                return self._create_home_sequence()
            elif any(word in command.lower() for word in ['place', 'put', 'drop']):
                return self._create_place_sequence(detections, calibration)
            else:
                return {'success': False, 'error_message': f'Unknown command type: {command}'}
                
        except Exception as e:
            self.logger.error(f"Command formatting error: {e}")
            return {'success': False, 'error_message': str(e)}
    
    def _create_pick_sequence(self, detections: List[Tuple[str, float, List[int]]], 
                             calibration: CameraCalibration) -> Dict[str, Any]:
        """Create pick-up trajectory sequence"""
        
        if not detections:
            return {'success': False, 'error_message': 'No objects to pick'}
        
        # Use the highest confidence detection
        best_detection = max(detections, key=lambda x: x[1])
        query, confidence, bbox = best_detection
        
        # Convert 2D bbox to 3D position (simplified)
        x1, y1, x2, y2 = bbox
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Use camera calibration to get 3D position
        # This is a simplified conversion - you'll need proper depth estimation
        target_position = calibration.pixel_to_world(center_x, center_y, depth=0.5)
        
        if target_position is None:
            return {'success': False, 'error_message': 'Failed to convert to 3D coordinates'}
        
        trajectories = []
        gripper_commands = []
        
        # 1. Approach position (above target)
        approach_pos = [target_position[0], target_position[1], target_position[2] + 0.1]
        approach_joints = self.kinematics.inverse_kinematics(approach_pos, [0, 0, 0, 1])
        
        if approach_joints:
            trajectories.append(self._create_joint_trajectory(approach_joints, duration=3.0))
            gripper_commands.append({'timing': 0, 'action': 'open'})
        
        # 2. Grasp position
        grasp_joints = self.kinematics.inverse_kinematics(target_position, [0, 0, 0, 1])
        if grasp_joints:
            trajectories.append(self._create_joint_trajectory(grasp_joints, duration=2.0))
            gripper_commands.append({'timing': 1, 'action': 'close'})
        
        # 3. Lift position
        lift_pos = [target_position[0], target_position[1], target_position[2] + 0.2]
        lift_joints = self.kinematics.inverse_kinematics(lift_pos, [0, 0, 0, 1])
        if lift_joints:
            trajectories.append(self._create_joint_trajectory(lift_joints, duration=2.0))
        
        if not trajectories:
            return {'success': False, 'error_message': 'Failed to compute trajectories'}
        
        return {
            'success': True,
            'trajectories': trajectories,
            'gripper_commands': gripper_commands,
            'target_object': query
        }
    
    def _create_home_sequence(self) -> Dict[str, Any]:
        """Create home position trajectory"""
        
        # UR5e home position (all joints at 0)
        home_joints = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
        
        trajectory = self._create_joint_trajectory(home_joints, duration=5.0)
        
        return {
            'success': True,
            'trajectories': [trajectory],
            'gripper_commands': [{'timing': 0, 'action': 'open'}]
        }
    
    def _create_place_sequence(self, detections: List[Tuple[str, float, List[int]]], 
                              calibration: CameraCalibration) -> Dict[str, Any]:
        """Create place trajectory sequence"""
        
        # Simple place at a predefined location
        place_position = [0.3, 0.3, 0.2]  # Safe placing position
        
        trajectories = []
        gripper_commands = []
        
        # 1. Move to place position
        place_joints = self.kinematics.inverse_kinematics(place_position, [0, 0, 0, 1])
        if place_joints:
            trajectories.append(self._create_joint_trajectory(place_joints, duration=3.0))
            gripper_commands.append({'timing': 0, 'action': 'open'})
        
        # 2. Return to safe position
        safe_pos = [place_position[0], place_position[1], place_position[2] + 0.2]
        safe_joints = self.kinematics.inverse_kinematics(safe_pos, [0, 0, 0, 1])
        if safe_joints:
            trajectories.append(self._create_joint_trajectory(safe_joints, duration=2.0))
        
        if not trajectories:
            return {'success': False, 'error_message': 'Failed to compute place trajectories'}
        
        return {
            'success': True,
            'trajectories': trajectories,
            'gripper_commands': gripper_commands
        }
    
    def _create_joint_trajectory(self, joint_positions: List[float], duration: float) -> 'JointTrajectory':
        """Create a ROS2 JointTrajectory message"""
        
        if not ROS2_AVAILABLE:
            # Return a mock trajectory for simulation
            class MockTrajectory:
                def __init__(self, positions, duration):
                    self.points = [MockTrajectoryPoint(positions, duration)]
                    self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 
                                      'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
            
            class MockTrajectoryPoint:
                def __init__(self, positions, duration):
                    self.positions = positions
                    self.time_from_start = duration
            
            return MockTrajectory(joint_positions, duration)
        
        # Real ROS2 trajectory
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from builtin_interfaces.msg import Duration
        
        trajectory = JointTrajectory()
        trajectory.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.velocities = [0.0] * len(joint_positions)
        point.accelerations = [0.0] * len(joint_positions)
        
        # Convert duration to ROS2 Duration
        seconds = int(duration)
        nanoseconds = int((duration - seconds) * 1e9)
        point.time_from_start = Duration(sec=seconds, nanosec=nanoseconds)
        
        trajectory.points = [point]
        return trajectory
    
    def get_joint_command_string(self, joint_positions: List[float]) -> str:
        """Generate a manual ROS2 command string"""
        
        positions_str = ' '.join([f'{pos:.3f}' for pos in joint_positions])
        
        return f"""ros2 action send_goal /joint_trajectory_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory "{{
  trajectory: {{
    joint_names: ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'],
    points: [{{
      positions: [{positions_str}],
      time_from_start: {{sec: 3, nanosec: 0}}
    }}]
  }}
}}\""""


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('vision_system.log')
        ]
    )


def main():
    """Main function to run the integrated vision system test"""
    
    print("🚀 UR5e Integrated Vision System Test")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='UR5e Vision Control System')
    parser.add_argument('--no-ros2', action='store_true', help='Run without ROS2 (simulation only)')
    parser.add_argument('--no-camera', action='store_true', help='Run without camera')
    parser.add_argument('--no-speech', action='store_true', help='Run without speech recognition')
    args = parser.parse_args()
    
    try:
        # Initialize system
        system = IntegratedVisionSystem(
            use_ros2=not args.no_ros2,
            use_camera=not args.no_camera
        )
        
        # Add command formatter
        system.command_formatter = CommandFormatter(system.kinematics)
        
        # Disable speech if requested
        if args.no_speech:
            system.speech = None
            print("Speech recognition disabled - use keyboard input")
        
        print("\n✅ System initialized successfully!")
        print("Press 'q' in the camera window or Ctrl+C to exit")
        
        # Start the system
        system.start_system()
        
    except KeyboardInterrupt:
        print("\n\n🛑 System interrupted by user")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        logging.exception("System startup error")
    finally:
        print("👋 Goodbye!")


if __name__ == "__main__":
    main()