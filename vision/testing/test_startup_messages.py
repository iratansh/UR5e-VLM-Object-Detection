#!/usr/bin/env python3
# scripts/test_startup_message.py

import rclpy
from rclpy.node import Node

class StartupMessage(Node):
    def __init__(self):
        super().__init__('startup_message')
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info('SIMULATION READY!')
        self.get_logger().info('='*60)
        self.get_logger().info('The UR5e robot with Robotiq gripper and RealSense camera')
        self.get_logger().info('should now be visible in Gazebo.')
        self.get_logger().info('')
        self.get_logger().info('To test:')
        self.get_logger().info('1. Check Gazebo - you should see:')
        self.get_logger().info('   - UR5e robot arm')
        self.get_logger().info('   - Robotiq 2F-85 gripper attached to end effector')
        self.get_logger().info('   - RealSense camera mounted on gripper')
        self.get_logger().info('   - Table with red cube')
        self.get_logger().info('')
        self.get_logger().info('2. Check RViz for camera feed:')
        self.get_logger().info('   - Add Camera display')
        self.get_logger().info('   - Set topic to /camera/color/image_raw')
        self.get_logger().info('')
        self.get_logger().info('3. Say "pick up the red cube" to test')
        self.get_logger().info('='*60)

def main():
    rclpy.init()
    node = StartupMessage()
    rclpy.shutdown()

if __name__ == '__main__':
    main()