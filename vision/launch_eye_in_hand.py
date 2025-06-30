"""
ROS2 Launch file for UnifiedVisionSystem with Eye-in-Hand Configuration

This launch file configures the vision system for a camera mounted on the robot's end-effector.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    # Declare launch arguments
    robot_namespace = DeclareLaunchArgument(
        'robot_namespace',
        default_value='ur5e',
        description='Namespace for the robot'
    )
    
    hand_eye_calibration_file = DeclareLaunchArgument(
        'hand_eye_calibration_file',
        default_value='hand_eye_calib_eye_in_hand.npz',
        description='Path to eye-in-hand calibration file'
    )
    
    camera_calibration_file = DeclareLaunchArgument(
        'camera_calibration_file',
        default_value='camera_calibration.npz',
        description='Path to camera intrinsics calibration file'
    )
    
    enable_hybrid_ik = DeclareLaunchArgument(
        'enable_hybrid_ik',
        default_value='true',
        description='Enable hybrid IK system (ur_ikfast + numerical)'
    )
    
    # Vision system node with eye-in-hand configuration
    unified_vision_node = Node(
        package='unified_vision_system',
        executable='unified_vision_system',
        name='unified_vision_system',
        output='screen',
        parameters=[{
            # Robot configuration
            'robot_namespace': LaunchConfiguration('robot_namespace'),
            
            # Eye-in-hand configuration
            'eye_in_hand': True,
            'broadcast_camera_tf': True,
            
            # Calibration files
            'hand_eye_calibration_file': LaunchConfiguration('hand_eye_calibration_file'),
            'camera_calibration_file': LaunchConfiguration('camera_calibration_file'),
            
            # RealSense configuration
            'realsense_serial': '',  # Leave empty to use any connected camera
            'max_depth': 1.5,  # Reduced for eye-in-hand (closer to objects)
            'min_depth': 0.1,  # Minimum depth for eye-in-hand
            'depth_fps': 30,
            'color_fps': 30,
            'depth_width': 848,
            'depth_height': 480,
            'color_width': 848,
            'color_height': 480,
            'enable_depth_filters': True,
            
            # VLM detection parameters
            'vlm_confidence_threshold': 0.15,  # Slightly higher for eye-in-hand
            
            # Depth processing parameters
            'min_valid_pixels': 150,  # Increased for eye-in-hand (closer view)
            'max_depth_variance': 0.010,  # Tighter tolerance for eye-in-hand
            'min_object_volume_m3': 1.0e-6,
            'max_object_volume_m3': 0.005,  # Smaller max volume for eye-in-hand
            
            # Hybrid IK System
            'enable_hybrid_ik': LaunchConfiguration('enable_hybrid_ik'),
            'ik_enable_approximation': True,
            'ik_max_position_error_mm': 8.0,  # Slightly more tolerance for eye-in-hand
            'ik_timeout_ms': 75.0,
            'ik_debug': False,
            
            # Eye-in-hand specific parameters
            'approach_offset_m': 0.08,  # Distance above object for approach
            'grasp_offset_m': 0.05,  # Final grasp offset
            'camera_to_gripper_offset_z': -0.05,  # Camera mounted 5cm above gripper
            
            # Safety parameters for eye-in-hand
            'min_grasp_height': 0.05,  # Minimum height for grasping
            'max_approach_tilt_deg': 15.0,  # Maximum tilt from vertical
            
            # Workspace validation
            'validate_eye_in_hand_workspace': True,
            'eye_in_hand_safety_margin': 0.15,  # Larger safety margin
        }],
        remappings=[
            # Input topics
            ('/ur5e/joint_states', '/joint_states'),
            ('/ur5e/emergency_stop', '/emergency_stop'),
            
            # Output topics
            ('/ur5e/scaled_joint_trajectory_controller/joint_trajectory', 
             '/scaled_joint_trajectory_controller/joint_trajectory'),
            ('/ur5e/target_pose', '/target_pose_debug'),
            ('/ur5e/system_status', '/vision_system_status'),
            
            # TF topics
            ('/tf', '/tf'),
            ('/tf_static', '/tf_static'),
        ]
    )
    
    # Optional: Static transform publisher for camera mount offset
    # This publishes the physical offset of the camera mount relative to the gripper flange
    camera_mount_transform = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_mount_publisher',
        arguments=['0', '0', '-0.05', '0', '0', '0', '1',  # x, y, z, qx, qy, qz, qw
                  'tool0', 'camera_mount']
    )
    
    # Optional: Visualization marker publisher for grasp points
    grasp_visualization = Node(
        package='unified_vision_system',
        executable='grasp_visualizer',
        name='grasp_visualizer',
        output='screen',
        parameters=[{
            'visualization_rate': 10.0,
            'show_approach_vectors': True,
            'show_grasp_frames': True,
        }],
        condition=LaunchConfiguration('enable_visualization', default='false')
    )
    
    return LaunchDescription([
        # Launch arguments
        robot_namespace,
        hand_eye_calibration_file,
        camera_calibration_file,
        enable_hybrid_ik,
        
        # Nodes
        unified_vision_node,
        camera_mount_transform,
        # grasp_visualization,  # Uncomment if visualization node exists
    ])