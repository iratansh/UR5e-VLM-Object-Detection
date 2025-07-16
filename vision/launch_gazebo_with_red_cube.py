#!/usr/bin/env python3
"""
Simple launch script to verify Gazebo setup with red cube and RealSense camera.

This script launches:
1. Gazebo with the UR5e + RealSense camera + red cube world
2. Robot state publisher
3. Basic controllers

Usage:
ros2 launch vision launch_gazebo_with_red_cube.py
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    
    # Launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    # Get package paths
    pkg_gazebo_ros = FindPackageShare('gazebo_ros')
    
    # World file path - absolute path to your world file
    world_file_path = os.path.join(
        os.path.dirname(__file__), '..', 'worlds', 'ur5e_vision_world.world'
    )
    
    # Robot description - absolute path to your URDF
    robot_description_file = os.path.join(
        os.path.dirname(__file__), '..', 'urdf', 'ur5e_with_realsense.urdf.xacro'
    )
    
    # Process robot description
    from launch.substitutions import Command
    robot_description_content = Command(['xacro ', robot_description_file])
    robot_description = {'robot_description': robot_description_content}
    
    # 1. Launch Gazebo
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([pkg_gazebo_ros, 'launch', 'gazebo.launch.py'])
        ]),
        launch_arguments={
            'world': world_file_path,
            'verbose': 'true',
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
    
    # 3. Spawn robot in Gazebo
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
    
    # 4. Joint state broadcaster (delayed)
    joint_state_broadcaster = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_state_broadcaster'],
                output='screen'
            )
        ]
    )
    
    # 5. Robot controller (delayed)
    robot_controller = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['scaled_joint_trajectory_controller'],
                output='screen'
            )
        ]
    )
    
    return LaunchDescription([
        # Launch arguments
        use_sim_time,
        
        # Nodes
        gazebo_launch,
        robot_state_publisher,
        spawn_robot,
        joint_state_broadcaster,
        robot_controller,
    ])
