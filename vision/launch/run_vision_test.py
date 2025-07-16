#!/usr/bin/env python3
"""
Unified Vision System Simulation Test Runner

This script provides a simple interface to run the complete simulation test
with proper setup, monitoring, and cleanup.

Usage:
    python3 run_vision_test.py [--quick] [--no-rviz] [--commands "cmd1,cmd2,cmd3"]
"""

import os
import sys
import subprocess
import time
import argparse
import signal
import threading
from pathlib import Path
import yaml

class TestRunner:
    """
    Test runner for UnifiedVisionSystem simulation tests.
    """
    
    def __init__(self):
        self.processes = []
        self.test_running = False
        self.test_results = {}
        
    def setup_environment(self):
        """Setup ROS2 environment and check dependencies."""
        print("🔧 Setting up test environment...")
        
        # Check ROS2 installation
        result = subprocess.run(['which', 'ros2'], capture_output=True)
        if result.returncode != 0:
            print("❌ ROS2 not found. Please source your ROS2 installation.")
            return False
        
        # Check if we're in a ROS2 workspace
        if not os.path.exists('src'):
            print("⚠️ Warning: Not in a ROS2 workspace. Some features may not work.")
        
        # Set ROS_DOMAIN_ID if not set
        if 'ROS_DOMAIN_ID' not in os.environ:
            os.environ['ROS_DOMAIN_ID'] = '42'
            print(f"🌐 Set ROS_DOMAIN_ID to {os.environ['ROS_DOMAIN_ID']}")
        
        # Create necessary directories
        Path("test_logs").mkdir(exist_ok=True)
        Path("simulation_test_logs").mkdir(exist_ok=True)
        
        print("✅ Environment setup complete")
        return True
    
    def create_test_configuration(self, test_commands=None, quick_test=False):
        """Create test configuration files."""
        print("📝 Creating test configuration...")
        
        # Default test commands
        if test_commands is None:
            if quick_test:
                test_commands = ["pick up the red cube"]
            else:
                test_commands = [
                    "pick up the red cube",
                    "grasp the red box", 
                    "get the blue cube",
                    "pick up the green cylinder"
                ]
        
        # Create world file with test objects
        self.create_test_world()
        
        # Create RViz configuration
        self.create_rviz_config()
        
        # Create launch configuration
        config = {
            'test_commands': test_commands,
            'timeout_seconds': 60 if quick_test else 180,
            'enable_rviz': True,
            'enable_logging': True,
            'simulation_speed': 1.0
        }
        
        with open('test_config.yaml', 'w') as f:
            yaml.dump(config, f)
        
        print(f"📋 Test configuration created with {len(test_commands)} commands")
        return config
    
    def create_test_world(self):
        """Create Gazebo world file for testing."""
        world_content = '''<?xml version="1.0"?>
<sdf version="1.6">
  <world name="ur5e_vision_test_world">
    
    <!-- Physics -->
    <physics name="default_physics" default="0" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    
    <!-- Lighting optimized for vision -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.3 0.3 0.3 1</specular>
      <direction>-0.3 0.1 -0.9</direction>
    </light>
    
    <light name="ambient_light" type="ambient">
      <color>0.6 0.6 0.6 1</color>
    </light>
    
    <!-- Overhead lighting -->
    <light name="overhead_light" type="point">
      <pose>0.5 0 1.5 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
    </light>
    
    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <!-- Work table -->
    <model name="work_table">
      <pose>0.5 0 0.4 0 0 0</pose>
      <static>true</static>
      <link name="table_link">
        <visual name="table_visual">
          <geometry>
            <box>
              <size>0.8 0.6 0.02</size>
            </box>
          </geometry>
          <material>
            <script>
              <name>Gazebo/Wood</name>
            </script>
          </material>
        </visual>
        <collision name="table_collision">
          <geometry>
            <box>
              <size>0.8 0.6 0.02</size>
            </box>
          </geometry>
        </collision>
      </link>
    </model>
    
    <!-- Test objects positioned for eye-in-hand camera -->
    
    <!-- RED CUBE - Primary test target -->
    <model name="red_cube">
      <pose>0.45 0.0 0.46 0 0 0</pose>
      <static>false</static>
      <link name="cube_link">
        <visual name="cube_visual">
          <geometry>
            <box>
              <size>0.05 0.05 0.05</size>
            </box>
          </geometry>
          <material>
            <script>
              <name>Gazebo/Red</name>
            </script>
          </material>
        </visual>
        <collision name="cube_collision">
          <geometry>
            <box>
              <size>0.05 0.05 0.05</size>
            </box>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>0.8</mu>
                <mu2>0.8</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <inertial>
          <mass>0.1</mass>
          <inertia>
            <ixx>0.0001</ixx>
            <iyy>0.0001</iyy>
            <izz>0.0001</izz>
          </inertia>
        </inertial>
      </link>
    </model>
    
    <!-- BLUE CUBE -->
    <model name="blue_cube">
      <pose>0.35 0.15 0.435 0 0 0</pose>
      <static>false</static>
      <link name="cube_link">
        <visual name="cube_visual">
          <geometry>
            <box>
              <size>0.03 0.03 0.03</size>
            </box>
          </geometry>
          <material>
            <script>
              <name>Gazebo/Blue</name>
            </script>
          </material>
        </visual>
        <collision name="cube_collision">
          <geometry>
            <box>
              <size>0.03 0.03 0.03</size>
            </box>
          </geometry>
        </collision>
        <inertial>
          <mass>0.05</mass>
          <inertia>
            <ixx>0.00005</ixx>
            <iyy>0.00005</iyy>
            <izz>0.00005</izz>
          </inertia>
        </inertial>
      </link>
    </model>
    
    <!-- GREEN CYLINDER -->
    <model name="green_cylinder">
      <pose>0.55 -0.1 0.45 0 0 0</pose>
      <static>false</static>
      <link name="cylinder_link">
        <visual name="cylinder_visual">
          <geometry>
            <cylinder>
              <radius>0.025</radius>
              <length>0.08</length>
            </cylinder>
          </geometry>
          <material>
            <script>
              <name>Gazebo/Green</name>
            </script>
          </material>
        </visual>
        <collision name="cylinder_collision">
          <geometry>
            <cylinder>
              <radius>0.025</radius>
              <length>0.08</length>
            </cylinder>
          </geometry>
        </collision>
        <inertial>
          <mass>0.08</mass>
          <inertia>
            <ixx>0.0001</ixx>
            <iyy>0.0001</iyy>
            <izz>0.00005</izz>
          </inertia>
        </inertial>
      </link>
    </model>
    
  </world>
</sdf>'''
        
        with open('ur5e_vision_test_world.world', 'w') as f:
            f.write(world_content)
        
        print("🌍 Test world file created")
    
    def create_rviz_config(self):
        """Create RViz configuration for visualization."""
        rviz_config = '''Panels:
  - Class: rviz_common/Displays
    Help Height: 78
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /RobotModel1
        - /PlanningScene1
        - /Camera1
      Splitter Ratio: 0.5
    Tree Height: 557
  - Class: rviz_common/Selection
    Name: Selection
  - Class: rviz_common/Tool Properties
    Expanded:
      - /2D Pose Estimate1
      - /2D Nav Goal1
      - /Publish Point1
    Name: Tool Properties
    Splitter Ratio: 0.5886790156364441
  - Class: rviz_common/Views
    Expanded:
      - /Current View1
    Name: Views
    Splitter Ratio: 0.5
Preferences:
  PromptSaveOnExit: true
Toolbars:
  toolButtonStyle: 2
Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 0.5
      Cell Size: 1
      Class: rviz_default_plugins/Grid
      Color: 160; 160; 164
      Enabled: true
      Line Style:
        Line Width: 0.029999999329447746
        Value: Lines
      Name: Grid
      Normal Cell Count: 0
      Offset:
        X: 0
        Y: 0
        Z: 0
      Plane: XY
      Plane Cell Count: 10
      Reference Frame: <Fixed Frame>
      Value: true
    - Alpha: 1
      Class: rviz_default_plugins/RobotModel
      Collision Enabled: false
      Enabled: true
      Links:
        All Links Enabled: true
        Expand Joint Details: false
        Expand Link Details: false
        Expand Tree: false
        Link Tree Style: Links in Alphabetic Order
      Name: RobotModel
      Robot Description: robot_description
      TF Prefix: ""
      Update Interval: 0
      Value: true
      Visual Enabled: true
    - Class: moveit_rviz_plugin/PlanningScene
      Enabled: true
      Move Group Namespace: ""
      Name: PlanningScene
      Planning Scene Topic: /monitored_planning_scene
      Robot Description: robot_description
      Scene Geometry:
        Scene Alpha: 0.8999999761581421
        Scene Color: 50; 230; 50
        Scene Display Time: 0.009999999776482582
        Show Scene Geometry: true
        Voxel Coloring: Z-Axis
        Voxel Rendering: Occupied Voxels
      Scene Robot:
        Attached Body Color: 150; 50; 150
        Links:
          All Links Enabled: true
          Expand Joint Details: false
          Expand Link Details: false
          Expand Tree: false
          Link Tree Style: Links in Alphabetic Order
        Robot Alpha: 1
        Show Robot Collision: false
        Show Robot Visual: true
      Value: true
    - Class: rviz_default_plugins/Camera
      Enabled: true
      Image Rendering: background and overlay
      Image Topic: /camera/color/image_raw
      Name: Camera
      Overlay Alpha: 0.5
      Queue Size: 2
      Transport Hint: raw
      Unreliable: false
      Value: true
      Visibility:
        Grid: true
        PlanningScene: true
        RobotModel: true
        Value: true
      Zoom Factor: 1
    - Alpha: 1
      Autocompute Intensity Bounds: true
      Autocompute Value Bounds:
        Max Value: 10
        Min Value: -10
        Value: true
      Axis: Z
      Channel Name: intensity
      Class: rviz_default_plugins/PointCloud2
      Color: 255; 255; 255
      Color Transformer: RGB8
      Decay Time: 0
      Enabled: true
      Invert Rainbow: false
      Max Color: 255; 255; 255
      Min Color: 0; 0; 0
      Name: PointCloud2
      Position Transformer: XYZ
      Queue Size: 10
      Selectable: true
      Size (Pixels): 3
      Size (m): 0.01
      Style: Flat Squares
      Topic: /camera/depth/color/points
      Unreliable: false
      Use Fixed Frame: true
      Use rainbow: true
      Value: true
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Default Light: true
    Fixed Frame: base_link
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
      Hide Inactive Objects: true
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
    - Class: rviz_default_plugins/Measure
    - Class: rviz_default_plugins/SetInitialPose
      Theta std deviation: 0.2617993950843811
      Topic: /initialpose
      X std deviation: 0.5
      Y std deviation: 0.5
    - Class: rviz_default_plugins/SetGoal
      Topic: /goal_pose
    - Class: rviz_default_plugins/PublishPoint
      Single click: true
      Topic: /clicked_point
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 2.5
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.05999999865889549
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Field of View: 0.7853981633974483
      Focal Point:
        X: 0.5
        Y: 0
        Z: 0.5
      Focal Shape Fixed Size: true
      Focal Shape Size: 0.05000000074505806
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.009999999776482582
      Pitch: 0.4
      Target Frame: <Fixed Frame>
      Yaw: -2.4
    Saved: ~
Window Geometry:
  Camera:
    collapsed: false
  Displays:
    collapsed: false
  Height: 846
  Hide Left Dock: false
  Hide Right Dock: false
  QMainWindow State: 000000ff00000000fd000000040000000000000156000002b0fc0200000009fb0000001200530065006c0065006300740069006f006e00000001e10000009b0000005c00fffffffb0000001e0054006f006f006c002000500072006f007000650072007400690065007302000001ed000001df00000185000000a3fb000000120056006900650077007300200054006f006f02000001df000002110000018500000122fb000000200054006f006f006c002000500072006f0070006500720074006900650073003203000002880000011d000002210000017afb000000100044006900730070006c006100790073010000003d000001d1000000c900fffffffb0000002000730065006c0065006300740069006f006e00200062007500660066006500720200000138000000aa0000023a00000294fb00000014005700690064006500670065007400730100000114000000d90000000000000000fb0000000c004b0069006e0065006300740200000186000001060000030c00000261fb0000000c00430061006d00650072006100000002140000019b0000002800ffffff000000010000010f000002b0fc0200000003fb0000001e0054006f006f006c002000500072006f007000650072007400690065007301000000410000006f0000005c00fffffffb0000000a00560069006500770073010000000000000153000000a400fffffffb0000001200530065006c0065006300740069006f006e010000025a000000b200000000000000000000000200000490000000a9fc0100000001fb0000000a00560069006500770073030000004e00000080000002e10000019700000003000004420000003efc0100000002fb0000000800540069006d00650100000000000004420000000000000000fb0000000800540069006d006501000000000000045000000000000000000000023f000002b000000004000000040000000800000008fc0000000100000002000000010000000a0054006f006f006c00730100000000ffffffff0000000000000000
  Selection:
    collapsed: false
  Tool Properties:
    collapsed: false
  Views:
    collapsed: false
  Width: 1200
  X: 100
  Y: 100'''
        
        with open('ur5e_vision_simulation_test.rviz', 'w') as f:
            f.write(rviz_config)
        
        print("📺 RViz configuration created")
    
    def start_test_sequence(self, config, enable_rviz=True):
        """Start the complete test sequence."""
        print("🚀 Starting unified vision system test...")
        print("="*60)
        
        # Step 1: Launch Gazebo
        print("1. 🌍 Launching Gazebo simulation...")
        gazebo_cmd = [
            'ros2', 'launch', 'gazebo_ros', 'gazebo.launch.py',
            'world:=ur5e_vision_test_world.world',
            'verbose:=false'
        ]
        gazebo_process = self.start_process("Gazebo", gazebo_cmd)
        if not gazebo_process:
            return False
        
        time.sleep(8)  # Wait for Gazebo
        
        # Step 2: Spawn robot
        print("2. 🤖 Spawning UR5e robot...")
        spawn_cmd = [
            'ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
            '-topic', 'robot_description',
            '-entity', 'ur5e'
        ]
        spawn_process = self.start_process("Robot Spawn", spawn_cmd)
        time.sleep(5)
        
        # Step 3: Start robot controllers
        print("3. ⚙️ Starting robot controllers...")
        joint_broadcaster_cmd = [
            'ros2', 'run', 'controller_manager', 'spawner',
            'joint_state_broadcaster'
        ]
        self.start_process("Joint Broadcaster", joint_broadcaster_cmd)
        time.sleep(3)
        
        trajectory_controller_cmd = [
            'ros2', 'run', 'controller_manager', 'spawner',
            'scaled_joint_trajectory_controller'
        ]
        self.start_process("Trajectory Controller", trajectory_controller_cmd)
        time.sleep(3)
        
        # Step 4: Launch MoveIt2 (optional)
        print("4. 🧠 Launching MoveIt2...")
        moveit_cmd = [
            'ros2', 'launch', 'ur_moveit_config', 'ur_moveit.launch.py',
            'ur_type:=ur5e',
            'use_sim_time:=true',
            'launch_rviz:=false'
        ]
        self.start_process("MoveIt2", moveit_cmd)
        time.sleep(5)
        
        # Step 5: Launch RViz (optional)
        if enable_rviz:
            print("5. 📺 Launching RViz...")
            rviz_cmd = [
                'ros2', 'run', 'rviz2', 'rviz2',
                '-d', 'ur5e_vision_simulation_test.rviz'
            ]
            self.start_process("RViz", rviz_cmd)
            time.sleep(3)
        
        # Step 6: Launch vision system test
        print("6. 👁️🤖 Launching UnifiedVisionSystem test...")
        vision_cmd = [
            'python3', 'test_unified_vision_simulation.py'
        ]
        vision_process = self.start_process("Vision System Test", vision_cmd)
        
        print("✅ All components launched!")
        print("📊 Monitor the test logs for progress...")
        print("🛑 Press Ctrl+C to stop the test")
        
        return True
    
    def start_process(self, name, cmd):
        """Start a subprocess and track it."""
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes.append((name, process))
            print(f"   ✅ {name} started (PID: {process.pid})")
            return process
        except Exception as e:
            print(f"   ❌ Failed to start {name}: {e}")
            return None
    
    def monitor_test(self):
        """Monitor test execution."""
        self.test_running = True
        
        def monitor_thread():
            while self.test_running:
                # Check if any critical processes died
                for name, process in self.processes:
                    if process.poll() is not None:
                        print(f"⚠️ Process {name} exited with code {process.returncode}")
                
                time.sleep(5)
        
        monitor = threading.Thread(target=monitor_thread, daemon=True)
        monitor.start()
    
    def cleanup(self):
        """Clean up all processes."""
        print("\n🧹 Cleaning up processes...")
        self.test_running = False
        
        for name, process in reversed(self.processes):
            try:
                if process.poll() is None:
                    print(f"   Terminating {name}...")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print(f"   Force killing {name}...")
                        process.kill()
            except Exception as e:
                print(f"   Error cleaning up {name}: {e}")
        
        print("✅ Cleanup complete")
    
    def print_final_report(self):
        """Print final test report."""
        print("\n" + "="*60)
        print("UNIFIED VISION SYSTEM TEST COMPLETED")
        print("="*60)
        
        log_dir = Path("test_logs")
        if log_dir.exists():
            log_files = list(log_dir.glob("*.log"))
            if log_files:
                print(f"📁 Test logs available in: {log_dir}")
                for log_file in log_files[-3:]:  # Show last 3 log files
                    print(f"   - {log_file.name}")
        
        results_dir = Path("simulation_test_logs")
        if results_dir.exists():
            result_files = list(results_dir.glob("*.yaml"))
            if result_files:
                print(f"📊 Test results available in: {results_dir}")
                for result_file in result_files[-3:]:
                    print(f"   - {result_file.name}")
        
        print("\n🎉 Thank you for testing the UnifiedVisionSystem!")
        print("📧 Please report any issues to the development team.")


def signal_handler(signum, frame, runner):
    """Handle shutdown signals."""
    print(f"\n🛑 Received signal {signum}, shutting down...")
    runner.cleanup()
    sys.exit(0)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run UnifiedVisionSystem simulation tests'
    )
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Run quick test with minimal commands'
    )
    parser.add_argument(
        '--no-rviz', 
        action='store_true',
        help='Skip RViz visualization'
    )
    parser.add_argument(
        '--commands',
        type=str,
        help='Comma-separated list of test commands'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Test timeout in seconds'
    )
    
    args = parser.parse_args()
    
    # Parse commands
    test_commands = None
    if args.commands:
        test_commands = [cmd.strip() for cmd in args.commands.split(',')]
    
    # Create test runner
    runner = TestRunner()
    
    # Setup signal handling
    signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, runner))
    signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, runner))
    
    try:
        print("🚀 UnifiedVisionSystem Simulation Test Runner")
        print("="*60)
        
        # Setup environment
        if not runner.setup_environment():
            print("❌ Environment setup failed")
            return 1
        
        # Create configuration
        config = runner.create_test_configuration(
            test_commands=test_commands,
            quick_test=args.quick
        )
        
        # Start test sequence
        if not runner.start_test_sequence(config, enable_rviz=not args.no_rviz):
            print("❌ Failed to start test sequence")
            return 1
        
        # Start monitoring
        runner.monitor_test()
        
        # Wait for test completion or user interrupt
        print(f"\n⏰ Test will run for up to {args.timeout} seconds...")
        print("   Monitor the logs for detailed progress")
        print("   Press Ctrl+C to stop early")
        
        try:
            time.sleep(args.timeout)
            print(f"\n⏰ Test timeout reached ({args.timeout}s)")
        except KeyboardInterrupt:
            print(f"\n⌨️ Test interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    
    finally:
        runner.cleanup()
        runner.print_final_report()
    
    return 0


if __name__ == '__main__':
    exit(main())