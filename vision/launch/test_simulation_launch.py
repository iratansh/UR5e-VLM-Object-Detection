#!/usr/bin/env python3
"""
ROS2 Launch file for UnifiedVisionSystem Simulation Test

This launch file coordinates all components needed for comprehensive testing:
- Gazebo simulation with UR5e + RealSense camera
- MoveIt2 motion planning
- UnifiedVisionSystem in test mode
- RViz visualization
- Automated test execution
"""

from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, 
    DeclareLaunchArgument, 
    ExecuteProcess, 
    TimerAction,
    LogInfo
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition, UnlessCondition
import os

def generate_launch_description():
    
    # Launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    test_world_file = DeclareLaunchArgument(
        'test_world_file',
        default_value='ur5e_vision_world.world',
        description='Gazebo world file for testing'
    )
    
    enable_rviz = DeclareLaunchArgument(
        'enable_rviz',
        default_value='true',
        description='Enable RViz visualization'
    )
    
    enable_moveit = DeclareLaunchArgument(
        'enable_moveit',
        default_value='true',
        description='Enable MoveIt2 motion planning'
    )
    
    auto_start_test = DeclareLaunchArgument(
        'auto_start_test',
        default_value='true',
        description='Automatically start test sequence'
    )
    
    test_commands = DeclareLaunchArgument(
        'test_commands',
        default_value='["pick up the red cube", "grasp the red box"]',
        description='List of test commands to execute'
    )
    
    # Get package paths
    pkg_gazebo_ros = FindPackageShare('gazebo_ros')
    pkg_ur_description = FindPackageShare('ur_description')
    pkg_ur_moveit_config = FindPackageShare('ur_moveit_config')
    
    # World file path
    world_file_path = PathJoinSubstitution([
        os.path.dirname(__file__), '..', 'worlds', 'ur5e_vision_world.world'
    ])
    
    # Robot description - Use custom UR5e with RealSense camera
    robot_description_content = PathJoinSubstitution([
        FindPackageShare('ur5e_project'), 'urdf', 'ur5e_with_realsense.urdf.xacro'
    ])
    
    robot_description = {'robot_description': robot_description_content}
    
    # 1. Launch Gazebo with UR5e and test world
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([pkg_gazebo_ros, 'launch', 'gazebo.launch.py'])
        ]),
        launch_arguments={
            'world': world_file_path,
            'verbose': 'false',
            'pause': 'false',
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }.items()
    )
    
    # 2. Robot state publisher
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
    
    # 3. Spawn UR5e robot in Gazebo
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
    
    # 4. Joint state broadcaster
    joint_state_broadcaster = TimerAction(
        period=3.0,  # Wait for Gazebo to fully load
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_state_broadcaster'],
                output='screen'
            )
        ]
    )
    
    # 5. UR5e trajectory controller
    ur5e_controller = TimerAction(
        period=5.0,  # Wait for joint_state_broadcaster
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['scaled_joint_trajectory_controller'],
                output='screen'
            )
        ]
    )
    
    # 6. MoveIt2 launch (conditional)
    moveit_launch = TimerAction(
        period=8.0,  # Wait for robot controllers
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    PathJoinSubstitution([pkg_ur_moveit_config, 'launch', 'ur_moveit.launch.py'])
                ]),
                launch_arguments={
                    'ur_type': 'ur5e',
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'launch_rviz': 'false'  # We'll launch RViz separately
                }.items(),
                condition=IfCondition(LaunchConfiguration('enable_moveit'))
            )
        ]
    )
    
    # 7. UnifiedVisionSystem in test mode
    vision_system = TimerAction(
        period=12.0,  # Wait for MoveIt2 to initialize
        actions=[
            Node(
                package='ur5e_vision',
                executable='unified_vision_system_test',
                name='unified_vision_system_test',
                output='screen',
                parameters=[{
                    # Robot configuration
                    'robot_namespace': 'ur5e',
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    
                    # Eye-in-hand configuration
                    'eye_in_hand': True,
                    'broadcast_camera_tf': True,
                    
                    # Test mode configuration
                    'test_mode': True,
                    'auto_test_commands': LaunchConfiguration('test_commands'),
                    'auto_start_test': LaunchConfiguration('auto_start_test'),
                    
                    # Simulation-specific calibration
                    'hand_eye_calibration_file': 'sim_hand_eye_calib.npz',
                    'camera_calibration_file': 'sim_camera_calib.npz',
                    
                    # RealSense simulation parameters
                    'realsense_serial': '',  # Use simulated camera
                    'max_depth': 1.5,
                    'min_depth': 0.1,
                    'depth_fps': 30,
                    'color_fps': 30,
                    'depth_width': 848,
                    'depth_height': 480,
                    'color_width': 848,
                    'color_height': 480,
                    'enable_depth_filters': True,
                    
                    # VLM detection parameters (tuned for simulation)
                    'vlm_confidence_threshold': 0.1,  # Lower for simulation
                    
                    # Depth processing parameters
                    'min_valid_pixels': 80,
                    'max_depth_variance': 0.025,  # More tolerance for simulation
                    'min_object_volume_m3': 1.0e-6,
                    'max_object_volume_m3': 0.008,
                    
                    # Hybrid IK System
                    'enable_hybrid_ik': True,
                    'ik_enable_approximation': True,
                    'ik_max_position_error_mm': 15.0,  # More tolerance for simulation
                    'ik_timeout_ms': 120.0,
                    'ik_debug': True,
                    
                    # Simulation-specific safety parameters
                    'approach_offset_m': 0.12,  # Larger offset for safety
                    'grasp_offset_m': 0.06,
                    'min_grasp_height': 0.04,
                    'max_approach_tilt_deg': 25.0,
                    
                    # Workspace validation
                    'validate_eye_in_hand_workspace': True,
                    'eye_in_hand_safety_margin': 0.25,  # Large margin for simulation
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
                    
                    # Test-specific topics
                    ('/test_vlm_command', '/test_vlm_command'),
                    ('/vision_system_status', '/vision_system_status'),
                    ('/target_pose_debug', '/target_pose_debug'),
                    
                    # TF topics
                    ('/tf', '/tf'),
                    ('/tf_static', '/tf_static'),
                ]
            )
        ]
    )
    
    # 8. RViz for visualization (conditional)
    rviz_launch = TimerAction(
        period=15.0,  # Wait for vision system
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', 'ur5e_vision_simulation_test.rviz'],
                parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
                output='screen',
                condition=IfCondition(LaunchConfiguration('enable_rviz'))
            )
        ]
    )
    
    # 9. Test execution node
    test_executor = TimerAction(
        period=18.0,  # Wait for everything to be ready
        actions=[
            Node(
                package='ur5e_vision',
                executable='simulation_test_executor',
                name='simulation_test_executor',
                output='screen',
                parameters=[{
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'test_commands': LaunchConfiguration('test_commands'),
                    'timeout_seconds': 180,
                    'log_level': 'INFO'
                }],
                condition=IfCondition(LaunchConfiguration('auto_start_test'))
            )
        ]
    )
    
    # 10. Camera info publisher for simulation (if needed)
    camera_info_publisher = TimerAction(
        period=10.0,
        actions=[
            Node(
                package='ur5e_vision',
                executable='camera_info_publisher',
                name='sim_camera_info_publisher',
                parameters=[{
                    'camera_name': 'camera',
                    'width': 848,
                    'height': 480,
                    'fx': 421.61,
                    'fy': 421.61,
                    'cx': 424.0,
                    'cy': 240.0,
                    'use_sim_time': LaunchConfiguration('use_sim_time')
                }],
                remappings=[
                    ('/camera_info', '/camera/color/camera_info')
                ]
            )
        ]
    )
    
    # 11. Static transform for eye-in-hand camera calibration
    camera_calibration_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='sim_eye_in_hand_calibration',
        arguments=[
            '0.0', '0.0', '-0.05',  # 5cm above gripper
            '0', '1.5708', '0',     # 90 degrees pitch to look down
            'tool0', 'camera_link'
        ],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )
    
    # Log messages for user guidance
    startup_log = LogInfo(
        msg="🚀 Starting UnifiedVisionSystem Simulation Test..."
    )
    
    ready_log = TimerAction(
        period=20.0,
        actions=[
            LogInfo(msg="✅ All systems should be ready! Monitor the logs for test progress.")
        ]
    )
    
    return LaunchDescription([
        # Launch arguments
        use_sim_time,
        test_world_file,
        enable_rviz,
        enable_moveit,
        auto_start_test,
        test_commands,
        
        # Startup message
        startup_log,
        
        # Core simulation components
        gazebo_launch,
        robot_state_publisher,
        spawn_robot,
        camera_calibration_tf,
        
        # Robot control
        joint_state_broadcaster,
        ur5e_controller,
        
        # Motion planning
        moveit_launch,
        
        # Vision system
        camera_info_publisher,
        vision_system,
        
        # Visualization
        rviz_launch,
        
        # Test execution
        test_executor,
        
        # Ready message
        ready_log,
    ])