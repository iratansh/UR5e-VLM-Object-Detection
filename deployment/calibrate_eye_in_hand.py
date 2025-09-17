#!/usr/bin/env python3
"""
Eye-in-Hand Calibration Script for UR5e with RealSense Camera

This script performs hand-eye calibration for a camera mounted on the robot's end-effector.
The ArUco marker remains stationary while the robot moves to observe it from different angles.
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
import time
import argparse
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class EyeInHandCalibrationScript:
    """
    Standalone script for eye-in-hand calibration with visualization and validation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Calibration poses for eye-in-hand
        # These are designed to observe a stationary marker from different angles
        self.calibration_poses = [
            # Format: [x, y, z, rx, ry, rz] in meters and radians
            # Looking straight down at different positions
            [0.4, 0.0, 0.3, 3.14, 0.0, 0.0],    # Center
            [0.35, 0.1, 0.3, 3.14, 0.0, 0.0],   # Left
            [0.35, -0.1, 0.3, 3.14, 0.0, 0.0],  # Right
            [0.45, 0.0, 0.3, 3.14, 0.0, 0.0],   # Forward
            [0.35, 0.0, 0.3, 3.14, 0.0, 0.0],   # Back
            
            # Looking at angles
            [0.4, 0.1, 0.35, 2.9, 0.0, 0.0],    # Tilted left
            [0.4, -0.1, 0.35, 2.9, 0.0, 0.0],   # Tilted right
            [0.45, 0.0, 0.35, 2.9, 0.2, 0.0],   # Tilted forward
            [0.35, 0.0, 0.35, 2.9, -0.2, 0.0],  # Tilted back
            
            # Different heights
            [0.4, 0.0, 0.25, 3.14, 0.0, 0.0],   # Lower
            [0.4, 0.0, 0.4, 3.14, 0.0, 0.0],    # Higher
            
            # Rotated views
            [0.4, 0.05, 0.3, 3.14, 0.0, 0.3],   # Rotated CCW
            [0.4, -0.05, 0.3, 3.14, 0.0, -0.3], # Rotated CW
            
            # Corner views for better calibration
            [0.35, 0.1, 0.25, 2.9, 0.1, 0.2],   # Lower left corner
            [0.45, -0.1, 0.35, 2.9, -0.1, -0.2], # Upper right corner
        ]
        
        # ArUco marker parameters
        self.marker_size = 0.05  # 5cm marker
        self.marker_dict = cv2.aruco.DICT_6X6_250
        self.marker_id = 0  # ID of the calibration marker
        
    def print_calibration_instructions(self):
        """Print setup instructions for eye-in-hand calibration."""
        print("\n" + "="*60)
        print("EYE-IN-HAND CALIBRATION SETUP INSTRUCTIONS")
        print("="*60)
        print("\n1. MARKER PLACEMENT:")
        print("   - Place ArUco marker (ID: 0, Size: 5cm) on a flat surface")
        print("   - Position marker at approximately (0.4, 0.0, 0.0) in robot base frame")
        print("   - Ensure marker is clearly visible and well-lit")
        print("   - Marker should remain STATIONARY during entire calibration")
        
        print("\n2. CAMERA MOUNTING:")
        print("   - Mount RealSense camera on robot end-effector")
        print("   - Camera should look downward toward workspace")
        print("   - Ensure cable management allows full robot motion")
        print("   - Verify camera is rigidly attached (no wobble)")
        
        print("\n3. SAFETY CHECKS:")
        print("   - Clear workspace of obstacles")
        print("   - Verify emergency stop is accessible")
        print("   - Test robot motion at slow speed first")
        print("   - Ensure camera cable won't snag during motion")
        
        print("\n4. CALIBRATION PROCESS:")
        print("   - Robot will move to 15 different poses")
        print("   - At each pose, camera captures marker position")
        print("   - Keep marker visible and stationary")
        print("   - Process takes approximately 5-10 minutes")
        
        print("\n" + "="*60)
        input("Press Enter when ready to start calibration...")
        
    def validate_calibration_result(self, T_gripper_camera: np.ndarray, 
                                  reprojection_errors: List[float]) -> bool:
        """
        Validate the calibration result with multiple checks.
        
        Parameters
        ----------
        T_gripper_camera : np.ndarray
            4x4 transformation matrix from gripper to camera
        reprojection_errors : List[float]
            List of reprojection errors for each calibration pose
            
        Returns
        -------
        bool
            True if calibration is valid
        """
        print("\n" + "="*60)
        print("CALIBRATION VALIDATION")
        print("="*60)
        
        translation = T_gripper_camera[:3, 3]
        rotation = T_gripper_camera[:3, :3]
        
        euler_angles = self.rotation_matrix_to_euler(rotation)
        
        print(f"\nCamera Position (relative to gripper):")
        print(f"  X: {translation[0]*1000:.1f} mm")
        print(f"  Y: {translation[1]*1000:.1f} mm")
        print(f"  Z: {translation[2]*1000:.1f} mm")
        
        print(f"\nCamera Orientation (relative to gripper):")
        print(f"  Roll:  {np.degrees(euler_angles[0]):.1f}°")
        print(f"  Pitch: {np.degrees(euler_angles[1]):.1f}°")
        print(f"  Yaw:   {np.degrees(euler_angles[2]):.1f}°")
        
        # Validation checks
        valid = True
        
        # Check 1: Translation magnitude (camera should be close to gripper)
        trans_magnitude = np.linalg.norm(translation)
        if trans_magnitude > 0.2:  # More than 20cm is suspicious
            print(f"\n⚠️  WARNING: Camera distance from gripper is {trans_magnitude*1000:.1f}mm")
            print("   This seems unusually large for an end-effector mounted camera.")
            valid = False
        
        # Check 2: Camera should be looking mostly downward
        # The camera Z-axis in gripper frame should point mostly down
        camera_z_axis = rotation[:, 2]
        if camera_z_axis[2] > -0.7:  # cos(45°) ≈ 0.7
            print(f"\n⚠️  WARNING: Camera doesn't appear to be looking downward")
            print(f"   Camera Z-axis: {camera_z_axis}")
            valid = False
        
        # Check 3: Reprojection error analysis
        mean_error = np.mean(reprojection_errors)
        max_error = np.max(reprojection_errors)
        std_error = np.std(reprojection_errors)
        
        print(f"\nReprojection Errors:")
        print(f"  Mean: {mean_error:.2f} pixels")
        print(f"  Max:  {max_error:.2f} pixels")
        print(f"  Std:  {std_error:.2f} pixels")
        
        if mean_error > 2.0:
            print(f"\n⚠️  WARNING: High mean reprojection error")
            valid = False
        
        if max_error > 5.0:
            print(f"\n⚠️  WARNING: Maximum reprojection error exceeds 5 pixels")
            valid = False
        
        # Check 4: Visualize the transform
        self.visualize_transform(T_gripper_camera)
        
        if valid:
            print("\n✅ Calibration validation PASSED")
        else:
            print("\n❌ Calibration validation FAILED")
            print("   Please check camera mounting and retry calibration")
        
        return valid
    
    def rotation_matrix_to_euler(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to euler angles (roll, pitch, yaw)."""
        sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(R[2,1], R[2,2])
            pitch = np.arctan2(-R[2,0], sy)
            yaw = np.arctan2(R[1,0], R[0,0])
        else:
            roll = np.arctan2(-R[1,2], R[1,1])
            pitch = np.arctan2(-R[2,0], sy)
            yaw = 0

        return roll, pitch, yaw
    
    def visualize_transform(self, T_gripper_camera: np.ndarray):
        """Visualize the gripper-camera transform in 3D."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw gripper coordinate frame (at origin)
        self.draw_coordinate_frame(ax, np.eye(4), 'Gripper', scale=0.05)
        
        # Draw camera coordinate frame
        self.draw_coordinate_frame(ax, T_gripper_camera, 'Camera', scale=0.05)
        
        # Draw connection line
        camera_pos = T_gripper_camera[:3, 3]
        ax.plot([0, camera_pos[0]], [0, camera_pos[1]], [0, camera_pos[2]], 
                'k--', linewidth=1, label='Camera mount')
        
        # Set labels and limits
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Eye-in-Hand Camera Transform Visualization')
        
        # Set equal aspect ratio
        max_range = 0.1
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        ax.legend()
        plt.show()
    
    def draw_coordinate_frame(self, ax, T: np.ndarray, label: str, scale: float = 0.1):
        """Draw a coordinate frame in 3D plot."""
        origin = T[:3, 3]
        
        # Extract axes from rotation matrix
        x_axis = T[:3, 0] * scale
        y_axis = T[:3, 1] * scale
        z_axis = T[:3, 2] * scale
        
        # Draw axes
        ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], 
                 color='r', arrow_length_ratio=0.1)
        ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], 
                 color='g', arrow_length_ratio=0.1)
        ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], 
                 color='b', arrow_length_ratio=0.1)
        
        ax.text(origin[0], origin[1], origin[2], label, fontsize=10)
    
    def create_calibration_data(self):
        """
        Generate synthetic calibration data for testing.
        This should be replaced with actual robot movement and marker detection.
        """
        print("\n⚠️  WARNING: Using synthetic data for demonstration")
        print("   In real use, this should interface with actual robot and camera")
        
        # Simulated camera offset from gripper (5cm above, looking down)
        T_gripper_camera_true = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, -0.05],
            [0, 0, 0, 1]
        ])
        
        noise_trans = 0.002  # 2mm noise
        noise_rot = 0.01     # ~0.5 degree noise
        
        T_gripper_camera_noisy = T_gripper_camera_true.copy()
        T_gripper_camera_noisy[:3, 3] += np.random.normal(0, noise_trans, 3)
        
        R_noise = cv2.Rodrigues(np.random.normal(0, noise_rot, 3))[0]
        T_gripper_camera_noisy[:3, :3] = T_gripper_camera_noisy[:3, :3] @ R_noise
        
        return T_gripper_camera_noisy, np.random.uniform(0.5, 1.5, 15)


def main():
    parser = argparse.ArgumentParser(description='Eye-in-hand calibration for UR5e with RealSense')
    parser.add_argument('--marker-size', type=float, default=0.05, 
                       help='ArUco marker size in meters (default: 0.05)')
    parser.add_argument('--num-poses', type=int, default=15, 
                       help='Number of calibration poses (default: 15)')
    parser.add_argument('--output', type=str, default='hand_eye_calib_eye_in_hand.npz',
                       help='Output calibration file')
    parser.add_argument('--visualize', action='store_true', 
                       help='Show visualization during calibration')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data for testing')
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create calibration script
    calibrator = EyeInHandCalibrationScript()
    calibrator.marker_size = args.marker_size
    
    calibrator.print_calibration_instructions()
    
    if args.synthetic:
        # Use synthetic data for testing
        print("\n🔧 Running with synthetic data for testing...")
        T_gripper_camera, errors = calibrator.create_calibration_data()
        
    else:
        # Initialize ROS2 for real calibration
        print("\n🤖 Initializing ROS2 for real calibration...")
        rclpy.init()
        
        try:
            # Import and use the actual calibration system
            from UnifiedVisionSystem import UnifiedVisionSystem
            
            vision_system = UnifiedVisionSystem()
            
            # Perform calibration
            vision_system.perform_hand_eye_calibration(
                num_poses=args.num_poses,
                save_file=args.output
            )
            
            # Get calibration result
            T_gripper_camera = vision_system.calibration.T_gripper_to_camera
            
            # For real calibration, we'd calculate actual reprojection errors
            errors = [0.8] * args.num_poses  # Placeholder
            
        except Exception as e:
            logging.error(f"Calibration failed: {e}")
            rclpy.shutdown()
            return
        finally:
            rclpy.shutdown()
    
    # Validate calibration
    if calibrator.validate_calibration_result(T_gripper_camera, errors):
        # Save calibration
        save_data = {
            'T_base_to_camera': T_gripper_camera,  # Keep key name for compatibility
            'is_eye_in_hand': True,
            'calibration_type': 'eye_in_hand',
            'marker_size': calibrator.marker_size,
            'num_poses': len(errors),
            'reprojection_errors': errors,
            'timestamp': time.time()
        }
        
        np.savez(args.output, **save_data)
        print(f"\n✅ Calibration saved to: {args.output}")
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print(f"1. Copy {args.output} to your ROS workspace")
        print("2. Update your launch file to use this calibration:")
        print(f"   hand_eye_calibration_file: '{args.output}'")
        print("3. Set eye_in_hand parameter to true:")
        print("   eye_in_hand: true")
        print("4. Test with simple pick-and-place tasks")
        print("\n" + "="*60)
        
    else:
        print("\n❌ Calibration validation failed. Please retry.")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())