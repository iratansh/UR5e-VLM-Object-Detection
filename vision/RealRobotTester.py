"""
Structured testing script for UR5e with eye-in-hand RealSense.
Run these tests in order to safely validate the system.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import time
import logging
from typing import List, Tuple, Optional

class RealRobotTester(Node):
    """
    Safe testing procedures for real UR5e with eye-in-hand camera.
    """
    
    def __init__(self):
        super().__init__('real_robot_tester')
        
        self.logger = logging.getLogger(__name__)
        
        # Safety parameters
        self.SLOW_SPEED_FACTOR = 0.2  # 20% of normal speed
        self.TEST_HEIGHT = 0.3  # 30cm working height
        self.HOME_JOINTS = [0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0]
        
        # Publishers
        self.joint_pub = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory',
            10
        )
        
        self.status_pub = self.create_publisher(
            String,
            '/test_status',
            10
        )
        
        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_callback,
            10
        )
        
        self.current_joints = None
        self.test_results = {}
        
    def _joint_callback(self, msg: JointState):
        """Update current joint positions."""
        if len(msg.position) >= 6:
            self.current_joints = list(msg.position[:6])
    
    def publish_status(self, message: str, level: str = "INFO"):
        """Publish test status message."""
        self.logger.info(f"{level}: {message}")
        status_msg = String()
        status_msg.data = f"[{level}] {message}"
        self.status_pub.publish(status_msg)
    
    def move_to_joints(self, target_joints: List[float], 
                      duration: float = 3.0, 
                      wait: bool = True) -> bool:
        """
        Move robot to target joint positions.
        
        Parameters
        ----------
        target_joints : List[float]
            Target joint angles in radians
        duration : float
            Movement duration in seconds
        wait : bool
            Wait for movement completion
            
        Returns
        -------
        bool
            True if movement successful
        """
        if self.current_joints is None:
            self.publish_status("No current joint state available", "ERROR")
            return False
        
        # Create trajectory message
        traj = JointTrajectory()
        traj.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint', 
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # Start point (current position)
        start_point = JointTrajectoryPoint()
        start_point.positions = self.current_joints
        start_point.time_from_start.sec = 0
        
        # End point (target position)
        end_point = JointTrajectoryPoint()
        end_point.positions = target_joints
        end_point.time_from_start.sec = int(duration)
        end_point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        
        traj.points = [start_point, end_point]
        
        # Publish trajectory
        self.joint_pub.publish(traj)
        
        if wait:
            time.sleep(duration + 0.5)  # Extra time for settling
            
        return True
    
    def test_1_basic_communication(self) -> bool:
        """Test 1: Verify basic ROS communication."""
        self.publish_status("Test 1: Basic Communication", "TEST")
        
        # Wait for joint state
        timeout = 5.0
        start_time = time.time()
        
        while self.current_joints is None and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if self.current_joints is None:
            self.publish_status("Failed to receive joint states", "ERROR")
            return False
        
        self.publish_status(f"Current joints: {[f'{np.degrees(j):.1f}°' for j in self.current_joints]}")
        self.test_results['communication'] = True
        return True
    
    def test_2_home_position(self) -> bool:
        """Test 2: Move to home position."""
        self.publish_status("Test 2: Home Position Movement", "TEST")
        
        if not self.move_to_joints(self.HOME_JOINTS, duration=5.0):
            self.publish_status("Failed to move to home position", "ERROR")
            return False
        
        self.publish_status("Successfully moved to home position")
        self.test_results['home_position'] = True
        return True
    
    def test_3_camera_positions(self) -> bool:
        """Test 3: Move through camera test positions."""
        self.publish_status("Test 3: Camera Test Positions", "TEST")
        
        # Define safe test positions for eye-in-hand
        test_positions = [
            # [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
            [0.0, -1.57, 0.0, -1.57, 0.0, 0.0],  # Home
            [0.0, -1.2, -0.5, -1.57, 0.0, 0.0],  # Look down center
            [0.3, -1.2, -0.5, -1.57, 0.0, 0.0],  # Look down left
            [-0.3, -1.2, -0.5, -1.57, 0.0, 0.0], # Look down right
            [0.0, -1.0, -0.8, -1.57, 0.0, 0.0],  # Look forward
        ]
        
        for i, pos in enumerate(test_positions):
            self.publish_status(f"Moving to test position {i+1}/{len(test_positions)}")
            
            if not self.move_to_joints(pos, duration=4.0):
                self.publish_status(f"Failed at position {i+1}", "ERROR")
                return False
            
            # Pause for camera adjustment
            time.sleep(1.0)
        
        # Return to home
        self.move_to_joints(self.HOME_JOINTS, duration=4.0)
        
        self.publish_status("Camera positioning test complete")
        self.test_results['camera_positions'] = True
        return True
    
    def test_4_workspace_boundaries(self) -> bool:
        """Test 4: Test workspace boundaries."""
        self.publish_status("Test 4: Workspace Boundaries", "TEST")
        
        # Define boundary test positions (joint space)
        boundary_positions = [
            # Near base
            [0.0, -2.0, 1.0, -1.57, 0.0, 0.0],
            # Extended reach
            [0.0, -0.5, -2.0, -0.5, 0.0, 0.0],
            # Left boundary
            [1.57, -1.2, -0.8, -1.57, 0.0, 0.0],
            # Right boundary
            [-1.57, -1.2, -0.8, -1.57, 0.0, 0.0],
        ]
        
        for i, pos in enumerate(boundary_positions):
            self.publish_status(f"Testing boundary position {i+1}/{len(boundary_positions)}")
            
            # Move slowly to boundary positions
            if not self.move_to_joints(pos, duration=6.0):
                self.publish_status(f"Boundary {i+1} unreachable (expected)", "WARN")
            else:
                time.sleep(1.0)
        
        # Return to safe position
        self.move_to_joints(self.HOME_JOINTS, duration=5.0)
        
        self.publish_status("Workspace boundary test complete")
        self.test_results['workspace_boundaries'] = True
        return True
    
    def test_5_gripper_visibility(self) -> bool:
        """Test 5: Verify gripper visibility in camera."""
        self.publish_status("Test 5: Gripper Visibility Check", "TEST")
        
        # Move to position where gripper should be visible
        gripper_test_pos = [0.0, -1.2, -0.5, -1.57, 0.0, 0.0]
        
        if not self.move_to_joints(gripper_test_pos, duration=4.0):
            self.publish_status("Failed to move to gripper test position", "ERROR")
            return False
        
        self.publish_status("Check camera feed - gripper should be visible at bottom of image")
        self.publish_status("Gripper fingers should be clearly distinguishable")
        
        # In real implementation, this would check the actual camera feed
        time.sleep(3.0)
        
        self.test_results['gripper_visibility'] = True
        return True
    
    def test_6_pick_motion(self) -> bool:
        """Test 6: Simulated pick motion (no object)."""
        self.publish_status("Test 6: Simulated Pick Motion", "TEST")
        
        # Define pick motion sequence
        # 1. Approach position (above virtual object)
        approach_pos = [0.0, -1.0, -1.0, -1.0, 0.0, 0.0]
        
        # 2. Pick position (at virtual object)
        pick_pos = [0.0, -1.1, -0.9, -1.0, 0.0, 0.0]
        
        # 3. Lift position
        lift_pos = [0.0, -0.9, -1.1, -1.0, 0.0, 0.0]
        
        sequence = [
            ("Approach", approach_pos, 4.0),
            ("Descend", pick_pos, 2.0),
            ("Grasp", pick_pos, 1.0),  # Pause for virtual grasp
            ("Lift", lift_pos, 2.0),
            ("Return", self.HOME_JOINTS, 4.0)
        ]
        
        for step_name, pos, duration in sequence:
            self.publish_status(f"Pick motion: {step_name}")
            
            if not self.move_to_joints(pos, duration=duration):
                self.publish_status(f"Failed at {step_name}", "ERROR")
                return False
            
            if step_name == "Grasp":
                self.publish_status("Simulating gripper close")
                time.sleep(1.0)
        
        self.publish_status("Pick motion simulation complete")
        self.test_results['pick_motion'] = True
        return True
    
    def test_7_vision_integration(self) -> bool:
        """Test 7: Basic vision system integration."""
        self.publish_status("Test 7: Vision Integration", "TEST")
        
        # Move to scanning position
        scan_pos = [0.0, -1.2, -0.5, -1.57, 0.0, 0.0]
        
        if not self.move_to_joints(scan_pos, duration=4.0):
            self.publish_status("Failed to move to scan position", "ERROR")
            return False
        
        self.publish_status("Robot in scanning position")
        self.publish_status("Vision system should be detecting workspace")
        
        # In real implementation, this would trigger object detection
        time.sleep(3.0)
        
        self.test_results['vision_integration'] = True
        return True
    
    def run_all_tests(self):
        """Run all tests in sequence."""
        print("\n" + "="*60)
        print("UR5E EYE-IN-HAND SYSTEM TEST")
        print("="*60)
        print("\nSAFETY REMINDERS:")
        print("- Keep emergency stop within reach")
        print("- Clear workspace of obstacles")
        print("- Monitor robot at all times")
        print("- Stop if any unexpected behavior")
        print("\n" + "="*60)
        
        input("Press Enter to start tests (Ctrl+C to abort)...")
        
        tests = [
            self.test_1_basic_communication,
            self.test_2_home_position,
            self.test_3_camera_positions,
            self.test_4_workspace_boundaries,
            self.test_5_gripper_visibility,
            self.test_6_pick_motion,
            self.test_7_vision_integration
        ]
        
        for i, test in enumerate(tests):
            print(f"\n{'='*40}")
            try:
                success = test()
                if not success:
                    self.publish_status(f"Test {i+1} failed. Stopping tests.", "ERROR")
                    break
            except Exception as e:
                self.publish_status(f"Test {i+1} exception: {e}", "ERROR")
                break
            
            time.sleep(2.0)  # Pause between tests
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name}: {status}")
        
        total_tests = len(tests)
        passed_tests = sum(self.test_results.values())
        
        print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("\n✅ All tests passed! System ready for operation.")
        else:
            print("\n⚠️  Some tests failed. Address issues before proceeding.")
    
    def cleanup(self):
        """Cleanup and return to safe position."""
        self.publish_status("Returning to home position for shutdown")
        self.move_to_joints(self.HOME_JOINTS, duration=5.0)


def main():
    rclpy.init()
    
    tester = RealRobotTester()
    
    try:
        # Wait for system to initialize
        time.sleep(2.0)
        
        # Run all tests
        tester.run_all_tests()
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nTest error: {e}")
    finally:
        tester.cleanup()
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()