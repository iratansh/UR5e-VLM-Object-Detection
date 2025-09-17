# UnifiedVisionSystemSim.py
"""
Simulation-aware Unified Vision System for Robotic Control.
Supports both real hardware and Gazebo simulation environments.

Updated for reorganized codebase structure:
- Vision/AI components from ../vision/
- Physical hardware components from ../deployment/
"""

import cv2
import rclpy
from rclpy.node import Node
import numpy as np
import logging
from typing import Optional, Dict, Any
import time

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge

# Import the original UnifiedVisionSystem from deployment directory
# and add vision directory for AI/ML components
import sys
import os

# Add paths for the reorganized codebase
current_dir = os.path.dirname(__file__)
deployment_path = os.path.join(current_dir, '..', 'deployment')
vision_path = os.path.join(current_dir, '..', 'vision')

# Add paths if not already present
for path in [deployment_path, vision_path]:
    abs_path = os.path.abspath(path)
    if abs_path not in sys.path:
        sys.path.insert(0, abs_path)

try:
    from UnifiedVisionSystem import UnifiedVisionSystem
    print("✅ Successfully imported UnifiedVisionSystem from deployment directory")
except ImportError as e:
    print(f"❌ Failed to import UnifiedVisionSystem: {e}")
    print(f"Deployment path: {os.path.abspath(deployment_path)}")
    print(f"Vision path: {os.path.abspath(vision_path)}")
    raise

class UnifiedVisionSystemSim(UnifiedVisionSystem):
    """
    Extended UnifiedVisionSystem that supports both real and simulated environments.
    """
    
    def __init__(self):
        try:
            # Initialize without starting hardware first
            super().__init__()

            # Check if we're in simulation mode
            self.declare_parameter('use_sim_time', False)
            self.simulation_mode = self.get_parameter('use_sim_time').value

            if self.simulation_mode:
                self.get_logger().info("🎮 Running in SIMULATION mode")
                self._setup_simulation_interfaces()
            else:
                self.get_logger().info("🤖 Running in REAL HARDWARE mode")
                # Real hardware setup is already done in parent class

            # CV Bridge for ROS image conversion
            self.cv_bridge = CvBridge()

            self.get_logger().info("✅ UnifiedVisionSystemSim initialized successfully")

        except Exception as e:
            print(f"❌ Error initializing UnifiedVisionSystemSim: {e}")
            import traceback
            traceback.print_exc()
            raise
        
    def _setup_simulation_interfaces(self):
        """Setup interfaces for simulation environment."""
        # Override pipeline initialization for simulation
        if hasattr(self, 'pipeline') and self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
            self.pipeline_started = False
        
        # Subscribe to simulated camera topics
        self.color_image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self._sim_color_callback,
            10
        )
        
        self.depth_image_sub = self.create_subscription(
            Image,
            '/camera/depth/image_rect_raw',
            self._sim_depth_callback,
            10
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self._sim_camera_info_callback,
            10
        )
        
        # Store latest frames
        self.latest_color_frame = None
        self.latest_depth_frame = None
        self.camera_info_received = False
        
        self.get_logger().info("✅ Simulation interfaces initialized")
    
    def _sim_color_callback(self, msg: Image):
        """Handle color image from simulation."""
        try:
            self.latest_color_frame = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting color image: {e}")
    
    def _sim_depth_callback(self, msg: Image):
        """Handle depth image from simulation."""
        try:
            # Convert depth image - Gazebo publishes in float32 meters
            if msg.encoding == "32FC1":
                depth_meters = self.cv_bridge.imgmsg_to_cv2(msg, "32FC1")
                # Convert to uint16 millimeters for compatibility
                self.latest_depth_frame = (depth_meters * 1000).astype(np.uint16)
            else:
                self.latest_depth_frame = self.cv_bridge.imgmsg_to_cv2(msg, "16UC1")
        except Exception as e:
            self.get_logger().error(f"Error converting depth image: {e}")
    
    def _sim_camera_info_callback(self, msg: CameraInfo):
        """Handle camera info from simulation."""
        if not self.camera_info_received:
            # Set camera calibration from simulation
            self.calibration.camera_matrix_color = np.array(msg.k).reshape(3, 3)
            self.calibration.dist_coeffs_color = np.array(msg.d)
            
            # Set mock RealSense intrinsics for compatibility
            class MockIntrinsics:
                def __init__(self, camera_info):
                    self.fx = camera_info.k[0]
                    self.fy = camera_info.k[4]
                    self.ppx = camera_info.k[2]
                    self.ppy = camera_info.k[5]
                    self.width = camera_info.width
                    self.height = camera_info.height
                    self.coeffs = list(camera_info.d) if camera_info.d else [0.0] * 5
            
            mock_intrinsics = MockIntrinsics(msg)
            self.calibration.color_intrinsics_rs = mock_intrinsics
            self.calibration.depth_intrinsics_rs = mock_intrinsics  # Same for simulation
            self.calibration.depth_scale_rs = 0.001  # 1mm = 0.001m
            
            self.camera_info_received = True
            self.get_logger().info("✅ Camera calibration set from simulation")
    
    def run_pipeline_once(self):
        """Override to handle simulation vs real hardware."""
        if self.simulation_mode:
            self._run_simulation_pipeline()
        else:
            super().run_pipeline_once()
    
    def _run_simulation_pipeline(self):
        """Run pipeline using simulated camera data."""
        # Wait for camera data
        if self.latest_color_frame is None or self.latest_depth_frame is None:
            return
        
        # Process voice commands
        self.publish_status("Awaiting voice command...")
        self.active_object_queries = self._process_sound_input()
        
        if not self.active_object_queries:
            return
        
        self.publish_status(f"Queries received: {self.active_object_queries}. Processing frames...")
        
        # Use latest frames from simulation
        color_image = self.latest_color_frame.copy()
        depth_image = self.latest_depth_frame.copy()
        
        # Broadcast transforms for eye-in-hand
        if self.eye_in_hand:
            self._broadcast_transforms()
        
        # VLM Detection
        vlm_detections_2d = self._perform_vlm_detection(color_image, self.active_object_queries)
        if not vlm_detections_2d:
            self.get_logger().warning("VLM detection yielded no results")
            self._visualize_current_detection(color_image, None)
            return
        
        # Depth-aware 3D detection (simulation doesn't have rs.frame, so pass None)
        valid_detection_3d = self._perform_depth_aware_detection(
            color_image, None, depth_image, vlm_detections_2d
        )
        
        if not valid_detection_3d:
            self.get_logger().warning("No valid 3D object found")
            self.publish_status("No graspable object found")
            return
        
        self.last_successful_detection_3d = valid_detection_3d
        self._visualize_current_detection(color_image, valid_detection_3d)
        
        # Continue with motion planning...
        self.publish_status(f"Object '{valid_detection_3d.label}' localized. Planning motion...")
        
        # Create target pose
        target_pose_matrix = self._create_target_pose_matrix_from_detection(valid_detection_3d)
        
        if not self._validate_robot_pose(target_pose_matrix):
            self.get_logger().error("Generated pose is invalid")
            return
        
        self._publish_debug_target_pose(target_pose_matrix)
        
        # Calculate IK
        joint_solution = self._calculate_inverse_kinematics(target_pose_matrix)
        
        if not joint_solution:
            self.get_logger().error("IK failed")
            self.publish_status("Cannot reach object")
            return
        
        self.publish_status("Sending motion command...")
        self._format_and_publish_ros2_command(joint_solution)
        
        # Clear queries
        self.active_object_queries = []

def test_import_paths():
    """Test that all necessary modules can be imported with the new structure."""
    try:
        print("🧪 Testing import paths for reorganized codebase...")

        # Test vision directory imports
        import sys
        import os
        vision_path = os.path.join(os.path.dirname(__file__), '..', 'vision')
        sys.path.insert(0, os.path.abspath(vision_path))

        from SpeechCommandProcessor import SpeechCommandProcessor
        print("✅ SpeechCommandProcessor imported successfully")

        from OWLViTDetector import OWLViTDetector
        print("✅ OWLViTDetector imported successfully")

        # Test deployment directory imports
        deployment_path = os.path.join(os.path.dirname(__file__), '..', 'deployment')
        sys.path.insert(0, os.path.abspath(deployment_path))

        from UR5eKinematics import UR5eKinematics
        print("✅ UR5eKinematics imported successfully")

        print("🎉 All import paths working correctly!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main(args=None):
    """Main function to run the UnifiedVisionSystemSim."""
    # Test import paths first
    if not test_import_paths():
        print("❌ Import path test failed. Please check the codebase organization.")
        return

    rclpy.init(args=args)

    try:
        print("🚀 Initializing UnifiedVisionSystemSim with reorganized codebase...")
        vision_system = UnifiedVisionSystemSim()
        print("🎮 Starting simulation vision system...")
        vision_system.run()
    except Exception as e:
        print(f"❌ Error running vision system: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
            print("🔌 ROS2 shutdown complete")

if __name__ == '__main__':
    main()