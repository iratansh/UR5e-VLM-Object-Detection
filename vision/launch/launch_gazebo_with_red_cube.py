# launch/launch_simulation_complete.py
"""
Complete launch file for UnifiedVisionSystem simulation testing.
Launches Gazebo, robot controllers, MoveIt2, and vision system.
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
    ExecuteProcess
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    FindExecutable,
    Command
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Launch arguments
    use_rviz = LaunchConfiguration('use_rviz', default='true')
    use_moveit = LaunchConfiguration('use_moveit', default='true')
    
    # Package paths
    ur_gazebo_pkg = FindPackageShare('ur_gazebo')
    ur_moveit_pkg = FindPackageShare('ur_moveit_config')
    vision_pkg = FindPackageShare('unified_vision_system')
    
    # URDF/Xacro file
    xacro_file = os.path.join(
        get_package_share_directory('unified_vision_system'),
        'urdf',
        'ur5e_robotiq_realsense.urdf.xacro'
    )
    
    robot_description_content = Command([
        FindExecutable(name='xacro'), ' ',
        xacro_file
    ])
    
    robot_description = {'robot_description': robot_description_content}
    
    # World file
    world_file = os.path.join(
        get_package_share_directory('unified_vision_system'),
        'worlds',
        'ur5e_vision_world.world'
    )
    
    # 1. Gazebo server and client
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': world_file,
            'pause': 'false',
            'use_sim_time': 'true',
            'gui': 'true',
            'recording': 'false',
            'debug': 'false',
            'verbose': 'true'
        }.items()
    )
    
    # 2. Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': True}]
    )
    
    # 3. Spawn robot in Gazebo
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'ur5e_robotiq_realsense',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.0',
            '-R', '0.0',
            '-P', '0.0',
            '-Y', '0.0'
        ],
        output='screen'
    )
    
    # 4. Load controllers
    load_joint_state_broadcaster = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'joint_state_broadcaster'],
        output='screen'
    )
    
    load_joint_trajectory_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'scaled_joint_trajectory_controller'],
        output='screen'
    )
    
    # 5. Static transform for world to base_link (if needed)
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link']
    )
    
    # 6. MoveIt2 (optional)
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                ur_moveit_pkg,
                'launch',
                'ur_moveit.launch.py'
            ])
        ]),
        launch_arguments={
            'ur_type': 'ur5e',
            'use_sim_time': 'true',
            'launch_rviz': 'false',
            'launch_servo': 'true'
        }.items(),
        condition=IfCondition(use_moveit)
    )
    
    # 7. RViz with MoveIt config
    rviz_config_file = os.path.join(
        get_package_share_directory('unified_vision_system'),
        'config',
        'ur5e_vision_moveit.rviz'
    )
    
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[robot_description, {'use_sim_time': True}],
        condition=IfCondition(use_rviz),
        output='screen'
    )
    
    # 8. Vision System (delayed start)
    vision_system = TimerAction(
        period=10.0,  # Wait for everything to initialize
        actions=[
            Node(
                package='unified_vision_system',
                executable='unified_vision_system_sim',
                name='unified_vision_system',
                output='screen',
                parameters=[{
                    'use_sim_time': True,
                    'robot_namespace': 'ur5e',
                    'eye_in_hand': True,
                    'hand_eye_calibration_file': PathJoinSubstitution([
                        vision_pkg, 'config', 'hand_eye_calib_sim.npz'
                    ]),
                    'enable_hybrid_ik': True,
                    'ik_timeout_ms': 100.0,
                    'vlm_confidence_threshold': 0.15,
                    # Simulation-specific parameters
                    'simulation_mode': True,
                    'enable_depth_filters': False,  # Gazebo depth is clean
                    'min_depth': 0.1,
                    'max_depth': 2.0
                }]
            )
        ]
    )
    
    # Event handlers
    load_joint_state_broadcaster_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_robot,
            on_exit=[load_joint_state_broadcaster]
        )
    )
    
    load_joint_trajectory_controller_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=load_joint_state_broadcaster,
            on_exit=[load_joint_trajectory_controller]
        )
    )
    
    return LaunchDescription([
        # Declare arguments
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Start RViz'
        ),
        DeclareLaunchArgument(
            'use_moveit',
            default_value='true',
            description='Start MoveIt2'
        ),
        
        # Launch nodes
        gazebo,
        robot_state_publisher,
        spawn_robot,
        static_tf,
        
        # Controllers
        load_joint_state_broadcaster_handler,
        load_joint_trajectory_controller_handler,
        
        # Optional components
        moveit,
        rviz,
        
        # Vision system
        vision_system
    ])