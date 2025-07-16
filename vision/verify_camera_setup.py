#!/usr/bin/env python3
"""
Camera FOV and Red Cube Position Verification Script

This script helps verify that:
1. The red cube is positioned correctly for the eye-in-hand camera
2. The camera FOV covers the red cube when robot is in scanning position
3. The transforms are set up correctly

Usage:
python3 verify_camera_setup.py
"""

import numpy as np
import math

def calculate_camera_fov():
    """Calculate camera field of view coverage."""
    # RealSense D435i specs
    horizontal_fov = 69.4  # degrees
    vertical_fov = 42.5    # degrees
    
    print("=== RealSense D435i Camera Specifications ===")
    print(f"Horizontal FOV: {horizontal_fov}°")
    print(f"Vertical FOV: {vertical_fov}°")
    print(f"Image Resolution: 848x480")
    print()
    
    return horizontal_fov, vertical_fov

def analyze_robot_camera_position():
    """Analyze the robot and camera positioning."""
    print("=== Robot and Camera Configuration ===")
    
    # Robot base is at origin (0, 0, 0)
    robot_base = np.array([0, 0, 0])
    print(f"Robot base position: {robot_base}")
    
    # Table position from world file
    table_center = np.array([0.5, 0, 0.4])  # Table center
    table_size = np.array([1.2, 0.8, 0.02])  # Table dimensions
    table_height = table_center[2] + table_size[2]/2  # Top surface
    print(f"Table center: {table_center}")
    print(f"Table surface height: {table_height:.3f}m")
    print()
    
    # Red cube positions from world file
    red_cube_pos = np.array([0.4, 0.05, 0.46])
    red_box_pos = np.array([0.5, 0.2, 0.45])
    
    print("=== Target Objects ===")
    print(f"Red cube position: {red_cube_pos}")
    print(f"Red cube size: 0.06m x 0.06m x 0.06m")
    print(f"Red box position: {red_box_pos}")
    print(f"Red box size: 0.08m x 0.05m x 0.07m")
    print()
    
    return red_cube_pos, red_box_pos, table_height

def analyze_camera_positioning():
    """Analyze camera positioning relative to targets."""
    print("=== Eye-in-Hand Camera Analysis ===")
    
    # Camera mounting from URDF
    # Camera mount joint: origin xyz="0.05 0 0.08" rpy="0 π/2 0"
    # Camera joint: origin xyz="0 0 0.02" rpy="0 0 0"
    camera_offset_from_tool0 = np.array([0.05, 0, 0.08 + 0.02])  # 5cm forward, 10cm up
    print(f"Camera offset from tool0: {camera_offset_from_tool0}")
    
    # Typical scanning position (from your test files)
    # [0.0, -1.2, -0.5, -1.57, 0.0, 0.0] - "Look down center"
    print("\nTypical scanning position: [0.0, -1.2, -0.5, -1.57, 0.0, 0.0]")
    
    # UR5e forward kinematics approximation for this pose
    # This is a rough calculation - actual would need full FK
    approx_tool0_pos = np.array([0.4, 0.0, 0.6])  # Approximate end-effector position
    approx_camera_pos = approx_tool0_pos + camera_offset_from_tool0
    
    print(f"Approximate tool0 position: {approx_tool0_pos}")
    print(f"Approximate camera position: {approx_camera_pos}")
    print()
    
    return approx_camera_pos

def check_cube_visibility(camera_pos, cube_pos, horizontal_fov, vertical_fov):
    """Check if red cube is visible from camera position."""
    print("=== Visibility Analysis ===")
    
    # Calculate vector from camera to cube
    camera_to_cube = cube_pos - camera_pos
    distance = np.linalg.norm(camera_to_cube)
    
    print(f"Distance from camera to red cube: {distance:.3f}m")
    
    # Calculate angles
    # Assuming camera is looking down (negative Z direction in world frame)
    horizontal_angle = math.degrees(math.atan2(camera_to_cube[1], camera_to_cube[0]))
    vertical_angle = math.degrees(math.atan2(-camera_to_cube[2], 
                                           math.sqrt(camera_to_cube[0]**2 + camera_to_cube[1]**2)))
    
    print(f"Horizontal angle to cube: {horizontal_angle:.1f}°")
    print(f"Vertical angle to cube: {vertical_angle:.1f}°")
    
    # Check if within FOV
    h_fov_half = horizontal_fov / 2
    v_fov_half = vertical_fov / 2
    
    h_visible = abs(horizontal_angle) <= h_fov_half
    v_visible = abs(vertical_angle) <= v_fov_half
    
    print(f"Horizontal FOV range: ±{h_fov_half:.1f}°")
    print(f"Vertical FOV range: ±{v_fov_half:.1f}°")
    print(f"Horizontally visible: {h_visible}")
    print(f"Vertically visible: {v_visible}")
    print(f"Overall visible: {h_visible and v_visible}")
    print()
    
    return h_visible and v_visible

def optimal_scanning_positions():
    """Suggest optimal robot positions for scanning."""
    print("=== Optimal Scanning Positions ===")
    print("Based on the red cube position [0.4, 0.05, 0.46], here are recommended")
    print("robot joint configurations for optimal viewing:")
    print()
    print("1. Direct overhead view:")
    print("   [0.0, -1.2, -0.5, -1.57, 0.0, 0.0]")
    print("   - Camera looks straight down at workspace center")
    print()
    print("2. Angled view from front:")
    print("   [0.0, -1.0, -0.8, -1.57, 0.0, 0.0]")
    print("   - Camera has slight forward angle")
    print()
    print("3. Side view (left):")
    print("   [0.3, -1.2, -0.5, -1.57, 0.0, 0.0]")
    print("   - Rotated 17° to the left")
    print()

def verification_commands():
    """Print commands to verify the setup."""
    print("=== Verification Commands ===")
    print("After launching the simulation, use these commands to verify:")
    print()
    print("1. Launch the simulation:")
    print("   ros2 launch vision launch_gazebo_with_red_cube.py")
    print()
    print("2. Check camera topics:")
    print("   ros2 topic list | grep camera")
    print("   ros2 topic echo /camera/color/camera_info --once")
    print()
    print("3. View camera feed:")
    print("   ros2 run rqt_image_view rqt_image_view /camera/color/image_raw")
    print()
    print("4. Check robot transforms:")
    print("   ros2 run tf2_tools view_frames")
    print("   ros2 run tf2_ros tf2_echo tool0 camera_link")
    print()
    print("5. Move robot to scanning position:")
    print("   ros2 topic pub /scaled_joint_trajectory_controller/joint_trajectory \\")
    print("   trajectory_msgs/msg/JointTrajectory \\")
    print("   '{joint_names: [shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint], points: [{positions: [0.0, -1.2, -0.5, -1.57, 0.0, 0.0], time_from_start: {sec: 3}}]}'")
    print()

def main():
    """Main verification function."""
    print("=" * 60)
    print("    UR5e Eye-in-Hand Camera and Red Cube Setup Verification")
    print("=" * 60)
    print()
    
    # Calculate camera FOV
    h_fov, v_fov = calculate_camera_fov()
    
    # Analyze positions
    cube_pos, box_pos, table_height = analyze_robot_camera_position()
    camera_pos = analyze_camera_positioning()
    
    # Check visibility
    cube_visible = check_cube_visibility(camera_pos, cube_pos, h_fov, v_fov)
    
    # Recommendations
    optimal_scanning_positions()
    verification_commands()
    
    print("=" * 60)
    if cube_visible:
        print("✅ SETUP LOOKS GOOD: Red cube should be visible to camera!")
    else:
        print("⚠️  SETUP NEEDS ADJUSTMENT: Red cube might not be fully visible")
    print("=" * 60)

if __name__ == "__main__":
    main()
