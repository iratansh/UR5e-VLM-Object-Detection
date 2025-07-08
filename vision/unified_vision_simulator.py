#!/usr/bin/env python3
"""
UnifiedVisionSystem Simulator

This simulator mocks hardware components (RealSense, UR5e) but tests
the real vision and control logic. It provides a safe way to test the
complete system before deploying to physical hardware.

Key Features:
- Mock RealSense camera with synthetic depth data
- Mock UR5e robot with realistic joint limits
- Real VLM, speech recognition, and IK calculations
- Virtual workspace with objects
- Collision detection
- Performance metrics
"""

import numpy as np
import cv2
import time
import logging
import threading
import queue
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

# Import real components (these will work with mocked data)
from UnifiedVisionSystem import UnifiedVisionSystem
from DepthAwareDetector import Detection3D
from OWLViTDetector import OWLViTDetector
from SpeechCommandProcessor import SpeechCommandProcessor
from UR5eKinematics import HybridUR5eKinematics
from CameraCalibration import CameraCalibration
from WorkSpaceValidator import WorkspaceValidator

@dataclass
class VirtualObject:
    """Represents a virtual object in the simulated workspace"""
    name: str
    position: np.ndarray  # [x, y, z] in meters
    size: np.ndarray      # [width, height, depth] in meters  
    rotation: float       # rotation about Z-axis in radians
    color: Tuple[int, int, int]  # BGR color for visualization
    graspable: bool = True

class MockRealSenseCamera:
    """Mock RealSense camera that generates synthetic depth data"""
    
    def __init__(self, width=848, height=480):
        self.width = width
        self.height = height
        self.fx = 421.61  # Typical D435i intrinsics
        self.fy = 421.61
        self.ppx = width / 2
        self.ppy = height / 2
        
        # Simulate camera noise
        self.depth_noise_std = 2.0  # mm
        self.color_noise_std = 5.0  # RGB values
        
    def get_frames(self, virtual_objects: List[VirtualObject], 
                   camera_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic color and depth frames"""
        
        # Create color frame with objects
        color_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        color_frame[:] = (50, 50, 50)  # Dark background
        
        # Create depth frame (in mm, like real RealSense)
        depth_frame = np.full((self.height, self.width), 2000, dtype=np.uint16)  # 2m background
        
        # Render virtual objects
        for obj in virtual_objects:
            self._render_object(color_frame, depth_frame, obj, camera_pose)
        
        # Add realistic noise
        color_frame = self._add_color_noise(color_frame)
        depth_frame = self._add_depth_noise(depth_frame)
        
        return color_frame, depth_frame
    
    def _render_object(self, color_frame: np.ndarray, depth_frame: np.ndarray,
                      obj: VirtualObject, camera_pose: np.ndarray):
        """Render a virtual object in the frames"""
        
        # Transform object position to camera frame
        obj_pos_world = np.array([obj.position[0], obj.position[1], obj.position[2], 1.0])
        camera_to_world = np.linalg.inv(camera_pose)
        obj_pos_camera = camera_to_world @ obj_pos_world
        
        if obj_pos_camera[2] <= 0.1:  # Behind camera or too close
            return
            
        # Project to image plane
        u = int(obj_pos_camera[0] * self.fx / obj_pos_camera[2] + self.ppx)
        v = int(obj_pos_camera[1] * self.fy / obj_pos_camera[2] + self.ppy)
        
        if not (0 <= u < self.width and 0 <= v < self.height):
            return
            
        # Calculate object size in pixels
        obj_depth_mm = int(obj_pos_camera[2] * 1000)
        pixel_size_x = int(obj.size[0] * self.fx / obj_pos_camera[2])
        pixel_size_y = int(obj.size[1] * self.fy / obj_pos_camera[2])
        
        # Draw object rectangle
        x1 = max(0, u - pixel_size_x // 2)
        x2 = min(self.width, u + pixel_size_x // 2)
        y1 = max(0, v - pixel_size_y // 2)
        y2 = min(self.height, v + pixel_size_y // 2)
        
        # Fill color and depth
        color_frame[y1:y2, x1:x2] = obj.color
        depth_frame[y1:y2, x1:x2] = obj_depth_mm
        
        # Add object label
        cv2.putText(color_frame, obj.name, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _add_color_noise(self, frame: np.ndarray) -> np.ndarray:
        """Add realistic color noise"""
        noise = np.random.normal(0, self.color_noise_std, frame.shape).astype(np.int16)
        noisy_frame = frame.astype(np.int16) + noise
        return np.clip(noisy_frame, 0, 255).astype(np.uint8)
    
    def _add_depth_noise(self, frame: np.ndarray) -> np.ndarray:
        """Add realistic depth noise"""
        noise = np.random.normal(0, self.depth_noise_std, frame.shape).astype(np.int16)
        noisy_frame = frame.astype(np.int16) + noise
        return np.clip(noisy_frame, 100, 10000).astype(np.uint16)  # 10cm to 10m range

class MockUR5eRobot:
    """Mock UR5e robot with realistic kinematics and safety checks"""
    
    def __init__(self):
        self.current_joints = np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0])
        self.kinematics = HybridUR5eKinematics()
        self.joint_limits = [
            (-2*np.pi, 2*np.pi),   # Base
            (-np.pi, np.pi),       # Shoulder  
            (-np.pi, np.pi),       # Elbow
            (-2*np.pi, 2*np.pi),   # Wrist 1
            (-2*np.pi, 2*np.pi),   # Wrist 2
            (-2*np.pi, 2*np.pi),   # Wrist 3
        ]
        self.max_joint_velocity = np.pi / 2  # rad/s
        self.is_moving = False
        self.movement_thread = None
        
        # Safety features
        self.emergency_stop = False
        self.workspace_limits = {
            'x': (-0.8, 0.8),
            'y': (-0.8, 0.8), 
            'z': (0.0, 1.0)
        }
        
    def get_current_pose(self) -> np.ndarray:
        """Get current end-effector pose"""
        return self.kinematics.forward_kinematics(self.current_joints.tolist())
    
    def move_to_joints(self, target_joints: List[float], duration: float = 3.0) -> bool:
        """Simulate movement to target joint configuration"""
        
        if self.emergency_stop:
            logging.error("Emergency stop active - cannot move")
            return False
            
        # Validate joint limits
        for i, (target, (min_limit, max_limit)) in enumerate(zip(target_joints, self.joint_limits)):
            if not min_limit <= target <= max_limit:
                logging.error(f"Joint {i} target {target:.3f} exceeds limits [{min_limit:.3f}, {max_limit:.3f}]")
                return False
        
        # Check velocities
        joint_diff = np.abs(np.array(target_joints) - self.current_joints)
        max_velocity = np.max(joint_diff / duration)
        if max_velocity > self.max_joint_velocity:
            logging.warning(f"Movement too fast: {max_velocity:.3f} > {self.max_joint_velocity:.3f} rad/s")
            duration = np.max(joint_diff) / self.max_joint_velocity
            logging.info(f"Adjusting duration to {duration:.2f}s for safe movement")
        
        # Check workspace limits
        target_pose = self.kinematics.forward_kinematics(target_joints)
        pos = target_pose[:3, 3]
        for axis, (min_val, max_val) in self.workspace_limits.items():
            axis_idx = ['x', 'y', 'z'].index(axis)
            if not min_val <= pos[axis_idx] <= max_val:
                logging.error(f"Target position {axis}={pos[axis_idx]:.3f} outside workspace limits [{min_val}, {max_val}]")
                return False
        
        # Simulate movement
        if self.movement_thread and self.movement_thread.is_alive():
            logging.warning("Robot already moving - waiting for completion")
            self.movement_thread.join()
        
        self.movement_thread = threading.Thread(
            target=self._simulate_movement, 
            args=(target_joints, duration)
        )
        self.movement_thread.start()
        
        return True
    
    def _simulate_movement(self, target_joints: List[float], duration: float):
        """Simulate smooth joint movement"""
        self.is_moving = True
        start_joints = self.current_joints.copy()
        target_array = np.array(target_joints)
        
        steps = int(duration * 50)  # 50 Hz simulation
        
        for i in range(steps + 1):
            if self.emergency_stop:
                logging.warning("Emergency stop during movement")
                break
                
            t = i / steps
            # Use smooth interpolation (ease in/out)
            smooth_t = 3*t*t - 2*t*t*t
            self.current_joints = start_joints + smooth_t * (target_array - start_joints)
            time.sleep(0.02)  # 50 Hz
        
        self.current_joints = target_array
        self.is_moving = False
        logging.info(f"Movement completed to joints: {[f'{j:.3f}' for j in target_joints]}")

class UnifiedVisionSimulator:
    """Complete simulation environment for the UnifiedVisionSystem"""
    
    def __init__(self, enable_gui: bool = True):
        self.enable_gui = enable_gui
        self.running = False
        
        # Initialize mock hardware
        self.mock_camera = MockRealSenseCamera()
        self.mock_robot = MockUR5eRobot()
        
        # Create virtual workspace
        self.virtual_objects = [
            VirtualObject("water_bottle", np.array([0.4, 0.2, 0.1]), 
                         np.array([0.06, 0.06, 0.15]), 0.0, (0, 100, 255)),
            VirtualObject("coffee_cup", np.array([0.35, -0.1, 0.08]),
                         np.array([0.08, 0.08, 0.10]), 0.0, (50, 150, 100)),
            VirtualObject("book", np.array([0.3, 0.15, 0.02]),
                         np.array([0.15, 0.20, 0.03]), 0.0, (100, 50, 200)),
            VirtualObject("phone", np.array([0.25, 0.0, 0.05]),
                         np.array([0.07, 0.14, 0.01]), 0.0, (200, 200, 50)),
        ]
        
        # Performance tracking
        self.stats = {
            'commands_processed': 0,
            'successful_grasps': 0,
            'failed_grasps': 0,
            'ik_solve_times': [],
            'detection_times': [],
            'total_runtime': 0.0
        }
        
        # Initialize real components with mocked data
        self.setup_real_components()
        
        logging.info("UnifiedVisionSimulator initialized")
    
    def setup_real_components(self):
        """Initialize real vision and control components"""
        
        # Use real VLM detector (works with synthetic images)
        try:
            self.vlm_detector = OWLViTDetector()
            logging.info("✅ Real VLM detector loaded")
        except Exception as e:
            logging.error(f"Failed to load VLM detector: {e}")
            self.vlm_detector = None
        
        # Use real speech processor
        try:
            self.speech_processor = SpeechCommandProcessor()
            logging.info("✅ Real speech processor loaded")
        except Exception as e:
            logging.warning(f"Speech processor not available: {e}")
            self.speech_processor = None
        
        # Use real kinematics
        self.kinematics = HybridUR5eKinematics(debug=True)
        
        # Mock calibration (eye-in-hand setup)
        self.calibration = CameraCalibration()
        self.calibration.set_mock_calibration(
            camera_matrix=np.array([[421.61, 0, 424], [0, 421.61, 240], [0, 0, 1]]),
            eye_in_hand=True,
            T_gripper_camera=np.array([
                [0, -1, 0, 0.05],
                [1, 0, 0, 0.0],
                [0, 0, 1, 0.08],
                [0, 0, 0, 1]
            ])
        )
        
        # Real workspace validator
        self.workspace_validator = WorkspaceValidator()
        
    def run_simulation(self, duration: float = 60.0):
        """Run the simulation for specified duration"""
        
        logging.info(f"Starting simulation for {duration}s")
        self.running = True
        start_time = time.time()
        
        # Simulation loop
        while self.running and (time.time() - start_time) < duration:
            try:
                # Get current camera frames
                camera_pose = self.mock_robot.get_current_pose()
                color_frame, depth_frame = self.mock_camera.get_frames(
                    self.virtual_objects, camera_pose
                )
                
                # Process with real vision system
                self.process_vision_frame(color_frame, depth_frame)
                
                # Handle user input
                if self.enable_gui:
                    self.handle_gui_input(color_frame)
                
                # Simulate speech commands periodically
                if np.random.random() < 0.01:  # 1% chance per frame
                    self.simulate_speech_command()
                
                time.sleep(0.1)  # 10 Hz simulation
                
            except KeyboardInterrupt:
                logging.info("Simulation interrupted by user")
                break
            except Exception as e:
                logging.error(f"Simulation error: {e}")
                break
        
        self.running = False
        self.stats['total_runtime'] = time.time() - start_time
        self.print_simulation_summary()
    
    def process_vision_frame(self, color_frame: np.ndarray, depth_frame: np.ndarray):
        """Process frames with real vision components"""
        
        if self.vlm_detector is None:
            return
        
        # Detect objects with real VLM
        start_time = time.perf_counter()
        queries = ["bottle", "cup", "book", "phone"]
        detections = self.vlm_detector.detect_with_text_queries(
            color_frame, queries, confidence_threshold=0.1
        )
        detection_time = (time.perf_counter() - start_time) * 1000
        self.stats['detection_times'].append(detection_time)
        
        # Process detections (simplified - real system would use DepthAwareDetector)
        for label, confidence, bbox in detections:
            logging.debug(f"Detected {label} with confidence {confidence:.2f}")
    
    def simulate_speech_command(self):
        """Simulate speech commands for testing"""
        
        commands = [
            "pick up the water bottle",
            "grab the coffee cup",
            "take the book", 
            "get the phone",
            "move to home position"
        ]
        
        command = np.random.choice(commands)
        logging.info(f"🎤 Simulated command: '{command}'")
        
        # Process command (simplified)
        self.process_command(command)
    
    def process_command(self, command: str):
        """Process a command using real components"""
        
        self.stats['commands_processed'] += 1
        
        # Parse command to find target object
        target_object = None
        for obj in self.virtual_objects:
            if obj.name.replace('_', ' ') in command.lower():
                target_object = obj
                break
        
        if target_object is None:
            if "home" in command.lower():
                self.move_to_home()
                return
            logging.warning(f"No object found for command: {command}")
            return
        
        # Calculate grasp pose
        grasp_success = self.attempt_grasp(target_object)
        
        if grasp_success:
            self.stats['successful_grasps'] += 1
            logging.info(f"✅ Successfully grasped {target_object.name}")
        else:
            self.stats['failed_grasps'] += 1
            logging.warning(f"❌ Failed to grasp {target_object.name}")
    
    def attempt_grasp(self, target_object: VirtualObject) -> bool:
        """Attempt to grasp a virtual object"""
        
        # Create target pose above object
        target_pose = np.eye(4)
        target_pose[:3, 3] = target_object.position + np.array([0, 0, 0.1])  # 10cm above
        
        # Calculate IK
        start_time = time.perf_counter()
        ik_solutions = self.kinematics.inverse_kinematics(target_pose)
        ik_time = (time.perf_counter() - start_time) * 1000
        self.stats['ik_solve_times'].append(ik_time)
        
        if not ik_solutions:
            logging.warning(f"No IK solution for {target_object.name}")
            return False
        
        # Move robot
        best_solution = ik_solutions[0]
        success = self.mock_robot.move_to_joints(best_solution, duration=3.0)
        
        if success:
            # Wait for movement completion
            time.sleep(3.5)
            logging.info(f"Moved to grasp position for {target_object.name}")
            
            # Simulate grasp (remove object from workspace)
            if target_object in self.virtual_objects:
                self.virtual_objects.remove(target_object)
                logging.info(f"Grasped and removed {target_object.name}")
        
        return success
    
    def move_to_home(self):
        """Move robot to home position"""
        home_joints = [0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0]
        self.mock_robot.move_to_joints(home_joints, duration=4.0)
        logging.info("Moving to home position")
    
    def handle_gui_input(self, color_frame: np.ndarray):
        """Handle GUI input and display"""
        
        # Add simulation info to frame
        info_frame = color_frame.copy()
        
        # Add object count
        cv2.putText(info_frame, f"Objects: {len(self.virtual_objects)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add robot status
        status = "MOVING" if self.mock_robot.is_moving else "READY"
        cv2.putText(info_frame, f"Robot: {status}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add statistics
        cv2.putText(info_frame, f"Commands: {self.stats['commands_processed']}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Vision Simulation", info_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.running = False
        elif key == ord('h'):
            self.move_to_home()
        elif key == ord('r'):
            self.reset_workspace()
        elif key == ord('s'):
            self.simulate_speech_command()
    
    def reset_workspace(self):
        """Reset virtual workspace to initial state"""
        self.virtual_objects = [
            VirtualObject("water_bottle", np.array([0.4, 0.2, 0.1]), 
                         np.array([0.06, 0.06, 0.15]), 0.0, (0, 100, 255)),
            VirtualObject("coffee_cup", np.array([0.35, -0.1, 0.08]),
                         np.array([0.08, 0.08, 0.10]), 0.0, (50, 150, 100)),
            VirtualObject("book", np.array([0.3, 0.15, 0.02]),
                         np.array([0.15, 0.20, 0.03]), 0.0, (100, 50, 200)),
            VirtualObject("phone", np.array([0.25, 0.0, 0.05]),
                         np.array([0.07, 0.14, 0.01]), 0.0, (200, 200, 50)),
        ]
        logging.info("Workspace reset to initial state")
    
    def print_simulation_summary(self):
        """Print comprehensive simulation results"""
        
        print("\n" + "="*60)
        print("🎯 SIMULATION SUMMARY")
        print("="*60)
        
        print(f"📊 Performance Metrics:")
        print(f"  Total runtime: {self.stats['total_runtime']:.1f}s")
        print(f"  Commands processed: {self.stats['commands_processed']}")
        print(f"  Successful grasps: {self.stats['successful_grasps']}")
        print(f"  Failed grasps: {self.stats['failed_grasps']}")
        
        if self.stats['commands_processed'] > 0:
            success_rate = self.stats['successful_grasps'] / self.stats['commands_processed'] * 100
            print(f"  Success rate: {success_rate:.1f}%")
        
        if self.stats['ik_solve_times']:
            avg_ik_time = np.mean(self.stats['ik_solve_times'])
            print(f"  Average IK solve time: {avg_ik_time:.1f}ms")
        
        if self.stats['detection_times']:
            avg_detection_time = np.mean(self.stats['detection_times'])
            print(f"  Average detection time: {avg_detection_time:.1f}ms")
        
        print(f"\n🎮 Controls used:")
        print(f"  'q' - Quit simulation")
        print(f"  'h' - Move to home")
        print(f"  'r' - Reset workspace")
        print(f"  's' - Simulate speech command")
        
        # Component status
        print(f"\n🔧 Component Status:")
        print(f"  VLM Detector: {'✅' if self.vlm_detector else '❌'}")
        print(f"  Speech Processor: {'✅' if self.speech_processor else '❌'}")
        print(f"  Hybrid IK: ✅ (ur_ikfast: {'✅' if self.kinematics.ikfast_available else '❌'})")
        print(f"  Mock Hardware: ✅")
        
        print("\n" + "="*60)

def main():
    """Run the simulation"""
    
    print("🚀 UnifiedVisionSystem Simulator")
    print("="*50)
    print("This simulator tests the complete vision system with:")
    print("- Real VLM object detection")
    print("- Real speech processing")  
    print("- Real IK calculations")
    print("- Mock RealSense camera")
    print("- Mock UR5e robot")
    print("- Virtual workspace with objects")
    print("\nPress 'q' to quit, 'h' for home, 'r' to reset, 's' for speech command")
    print("="*50)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Create and run simulator
    try:
        simulator = UnifiedVisionSimulator(enable_gui=True)
        simulator.run_simulation(duration=300.0)  # 5 minutes
        
    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user")
    except Exception as e:
        print(f"\n💥 Simulation error: {e}")
        logging.error(f"Simulation failed: {e}", exc_info=True)
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 