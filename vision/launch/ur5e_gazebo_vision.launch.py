#!/usr/bin/env python3
"""
Complete Gazebo Launch for UR5e Vision System

This launch file sets up:
- Gazebo simulation with UR5e + RealSense camera
- Robot controllers
- UnifiedVisionSystem node
- RViz visualization
- All necessary transforms and parameters

Usage:
    ros2 launch ur5e_vision ur5e_gazebo_vision.launch.py
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    # Launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    world_file = DeclareLaunchArgument(
        'world_file',
        default_value='ur5e_vision_world.world',
        description='Gazebo world file'
    )
    
    enable_vision_system = DeclareLaunchArgument(
        'enable_vision_system',
        default_value='true',
        description='Enable UnifiedVisionSystem node'
    )
    
    enable_rviz = DeclareLaunchArgument(
        'enable_rviz',
        default_value='true',
        description='Enable RViz visualization'
    )
    
    # Paths
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_ur_description = get_package_share_directory('ur_description')
    
    # World file path
    world_path = os.path.join(
        os.path.dirname(__file__), '..', 'worlds', 
        LaunchConfiguration('world_file')
    )
    
    # Robot description
    robot_description_content = Command([
        'xacro ', 
        os.path.join(os.path.dirname(__file__), '..', 'urdf', 'ur5e_with_realsense.urdf.xacro')
    ])
    
    robot_description = {'robot_description': robot_description_content}
    
    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ]),
        launch_arguments={
            'world': world_path,
            'verbose': 'false',
            'pause': 'false',
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }.items()
    )
    
    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            robot_description,
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ]
    )
    
    # Joint state broadcaster
    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen'
    )
    
    # UR5e controllers
    ur5e_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['scaled_joint_trajectory_controller'],
        output='screen'
    )
    
    # Spawn robot in Gazebo
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'ur5e',
            '-x', '0.0',
            '-y', '0.0', 
            '-z', '0.0'
        ],
        output='screen'
    )
    
    # UnifiedVisionSystem node
    vision_system = Node(
        package='ur5e_vision',
        executable='unified_vision_system',
        name='unified_vision_system',
        output='screen',
        parameters=[{
            # Robot configuration
            'robot_namespace': 'ur5e',
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            
            # Eye-in-hand configuration
            'eye_in_hand': True,
            'broadcast_camera_tf': True,
            
            # Simulation-specific calibration
            'hand_eye_calibration_file': 'sim_hand_eye_calib.npz',
            'camera_calibration_file': 'sim_camera_calib.npz',
            
            # RealSense simulation parameters
            'realsense_serial': '',  # Use simulated camera
            'max_depth': 2.0,
            'min_depth': 0.1,
            'depth_fps': 30,
            'color_fps': 30,
            'depth_width': 848,
            'depth_height': 480,
            'color_width': 848,
            'color_height': 480,
            'enable_depth_filters': True,
            
            # VLM detection parameters (tuned for simulation)
            'vlm_confidence_threshold': 0.15,
            
            # Depth processing parameters
            'min_valid_pixels': 100,
            'max_depth_variance': 0.020,  # More tolerance for simulation
            'min_object_volume_m3': 1.0e-6,
            'max_object_volume_m3': 0.01,
            
            # Hybrid IK System
            'enable_hybrid_ik': True,
            'ik_enable_approximation': True,
            'ik_max_position_error_mm': 10.0,
            'ik_timeout_ms': 100.0,  # More time for simulation
            'ik_debug': True,  # Enable debugging in simulation
            
            # Simulation-specific safety parameters
            'approach_offset_m': 0.10,  # Larger offset for safety
            'grasp_offset_m': 0.05,
            'min_grasp_height': 0.05,
            'max_approach_tilt_deg': 20.0,
            
            # Workspace validation
            'validate_eye_in_hand_workspace': True,
            'eye_in_hand_safety_margin': 0.20,  # Larger margin for simulation
        }],
        remappings=[
            # Camera topics (from Gazebo simulation)
            ('/camera/color/image_raw', '/camera/color/image_raw'),
            ('/camera/depth/image_rect_raw', '/camera/depth/image_rect_raw'),
            ('/camera/color/camera_info', '/camera/color/camera_info'),
            ('/camera/depth/camera_info', '/camera/depth/camera_info'),
            
            # Robot control topics
            ('/scaled_joint_trajectory_controller/joint_trajectory', 
             '/scaled_joint_trajectory_controller/joint_trajectory'),
            ('/joint_states', '/joint_states'),
            
            # TF topics
            ('/tf', '/tf'),
            ('/tf_static', '/tf_static'),
        ],
        condition=LaunchConfiguration('enable_vision_system')
    )
    
    # RViz configuration
    rviz_config_file = os.path.join(
        os.path.dirname(__file__), '..', 'config', 'ur5e_vision_sim.rviz'
    )
    
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen',
        condition=LaunchConfiguration('enable_rviz')
    )
    
    # Camera info publisher (if needed for simulation)
    camera_info_publisher = Node(
        package='ur5e_vision',
        executable='camera_info_publisher',
        name='camera_info_publisher',
        parameters=[{
            'camera_name': 'camera',
            'camera_info_file': os.path.join(
                os.path.dirname(__file__), '..', 'config', 'sim_camera_info.yaml'
            ),
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }],
        remappings=[
            ('/camera_info', '/camera/color/camera_info')
        ]
    )
    
    # Static transform for camera calibration (simulation)
    camera_calibration_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='sim_camera_calibration',
        arguments=[
            '0.05', '0', '0.08',  # x, y, z translation
            '0', '1.5708', '0',   # roll, pitch, yaw (90 degrees pitch to look down)
            'tool0', 'camera_link'
        ],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )
    
    # Test object spawner
    spawn_test_objects = Node(
        package='ur5e_vision',
        executable='spawn_test_objects',
        name='spawn_test_objects',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'randomize_positions': False,
            'object_types': ['bottle', 'cup', 'book', 'phone', 'box']
        }]
    )
    
    # Simulation monitor
    sim_monitor = Node(
        package='ur5e_vision',
        executable='simulation_monitor',
        name='simulation_monitor',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'monitor_rate': 10.0,
            'log_performance': True,
            'auto_test_commands': [
                'pick up the bottle',
                'grab the red cup', 
                'get the book',
                'move to home position'
            ]
        }]
    )
    
    return LaunchDescription([
        # Launch arguments
        use_sim_time,
        world_file,
        enable_vision_system,
        enable_rviz,
        
        # Core simulation
        gazebo,
        robot_state_publisher,
        spawn_robot,
        
        # Robot control
        joint_state_broadcaster,
        ur5e_controller,
        
        # Vision system
        vision_system,
        camera_calibration_tf,
        
        # Visualization and monitoring
        rviz,
        sim_monitor,
        
        # Optional nodes
        # camera_info_publisher,  # Uncomment if needed
        # spawn_test_objects,     # Uncomment for dynamic object spawning
    ]) 