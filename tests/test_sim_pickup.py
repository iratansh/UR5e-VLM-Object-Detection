# scripts/test_sim_pickup.py
#!/usr/bin/env python3
"""
Simple test script to verify simulation setup.
Sends a "pick up the red cube" command to the vision system.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
import sys

class SimulationTester(Node):
    def __init__(self):
        super().__init__('simulation_tester')
        
        # Create publisher for test commands
        self.test_command_pub = self.create_publisher(
            String, '/test_vlm_command', 10
        )
        
        # Subscribe to system status
        self.status_sub = self.create_subscription(
            String, '/vision_system_status',
            self.status_callback, 10
        )
        
        self.system_ready = False
        self.last_status = ""
        
    def status_callback(self, msg):
        self.last_status = msg.data
        if "initialized" in msg.data.lower():
            self.system_ready = True
        self.get_logger().info(f"System status: {msg.data}")
    
    def send_pickup_command(self):
        """Send pickup command to vision system."""
        # For UnifiedVisionSystem, we need to simulate voice input
        # This would require modifying the speech processor to accept text commands
        # For now, let's log what would happen
        
        self.get_logger().info("TEST: Would send 'pick up the red cube' command")
        self.get_logger().info("To test, say 'pick up the red cube' into microphone")
        
def main():
    rclpy.init()
    
    tester = SimulationTester()
    
    # Wait for system to be ready
    print("Waiting for vision system to initialize...")
    start_time = time.time()
    
    while not tester.system_ready and (time.time() - start_time) < 30:
        rclpy.spin_once(tester, timeout_sec=0.5)
    
    if tester.system_ready:
        print("\n✅ System ready!")
        print("Please say into your microphone: 'pick up the red cube'")
        print("Or press Ctrl+C to exit")
        
        try:
            rclpy.spin(tester)
        except KeyboardInterrupt:
            pass
    else:
        print("\n❌ System failed to initialize within 30 seconds")
        print(f"Last status: {tester.last_status}")
    
    tester.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()