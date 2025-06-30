"""
Example VLM Integration with Hybrid IK System
==============================================

This example demonstrates how to integrate the hybrid ur_ikfast + numerical IK system
with your existing VLM (Vision-Language Model) robotic control pipeline.

The integration provides:
1. Drop-in replacement for existing IK calls
2. Automatic fallback handling for vision noise
3. Performance optimization for real-time interaction
4. Comprehensive error handling and logging

Usage:
    python example_vlm_integration.py
"""

import numpy as np
import math
import time
from typing import List, Optional, Tuple, Dict, Any

# Import the hybrid IK system
from HybridIKWrapper import VLMKinematicsController
from UR5eKinematics import HybridUR5eKinematics

class VLMRobotController:
    """
    Example robot controller that integrates hybrid IK with VLM commands.
    
    This shows how to replace existing IK calls with the hybrid system
    while maintaining compatibility with your current VLM pipeline.
    """
    
    def __init__(self, debug: bool = False):
        """Initialize the VLM robot controller."""
        self.debug = debug
        
        self.kinematics = VLMKinematicsController(
            enable_ikfast=True,
            adaptive_timeout=True,
            debug=debug
        )
        
        # Simulated robot state
        self.current_joints = [0.0, -math.pi/2, 0.0, 0.0, 0.0, 0.0]
        self.is_moving = False
        
        # VLM command tracking
        self.command_history = []
        
        print(f"VLM Robot Controller initialized")
        print(f"Hybrid IK available: {self.kinematics.ik_solver.ikfast_available}")
    
    def process_vlm_command(self, command: str, detected_objects: List[Dict]) -> bool:
        """
        Process a VLM command with detected objects.
        
        Args:
            command: Natural language command (e.g., "pick up the water bottle")
            detected_objects: List of detected objects with positions and types
            
        Returns:
            True if command was successfully executed
        """
        command_start_time = time.perf_counter()
        
        print(f"\n=== Processing VLM Command ===")
        print(f"Command: '{command}'")
        print(f"Detected objects: {len(detected_objects)}")
        
        # Parse command to extract action and target
        action, target_object = self._parse_vlm_command(command)
        
        if action not in ["pick", "grasp", "grab", "take"]:
            print(f"Unsupported action: {action}")
            return False
        
        # Find target object in detected objects
        target_info = self._find_target_object(target_object, detected_objects)
        
        if not target_info:
            print(f"Target object '{target_object}' not found in detected objects")
            return False
        
        print(f"Target found: {target_info['type']} at {target_info['position']}")
        
        success = self._execute_pickup(target_info)
        
        # Track performance
        command_time = (time.perf_counter() - command_start_time) * 1000
        self.command_history.append({
            "command": command,
            "success": success,
            "time_ms": command_time,
            "target": target_info
        })
        
        print(f"Command {'✓ COMPLETED' if success else '✗ FAILED'} in {command_time:.1f}ms")
        
        return success
    
    def _parse_vlm_command(self, command: str) -> Tuple[str, str]:
        """Parse VLM command to extract action and target object."""
        command_lower = command.lower()
        
        # Extract action
        action = "unknown"
        for verb in ["pick up", "pick", "grasp", "grab", "take", "get"]:
            if verb in command_lower:
                action = verb.split()[0]  # Get first word
                break
        
        target = "unknown"
        objects = ["bottle", "can", "box", "book", "cup", "glass", "phone", "remote"]
        for obj in objects:
            if obj in command_lower:
                target = obj
                break
        
        return action, target
    
    def _find_target_object(self, target_name: str, detected_objects: List[Dict]) -> Optional[Dict]:
        """Find target object in detected objects list."""
        for obj in detected_objects:
            if target_name.lower() in obj.get('type', '').lower():
                return obj
            
            # Fuzzy matching for similar objects
            similarities = {
                'bottle': ['container', 'cylinder'],
                'can': ['cylinder', 'container'],
                'box': ['cube', 'rectangular', 'package'],
                'book': ['rectangular', 'flat'],
                'cup': ['container', 'cylinder'],
                'glass': ['container', 'cylinder']
            }
            
            if target_name in similarities:
                for similar in similarities[target_name]:
                    if similar in obj.get('type', '').lower():
                        return obj
        
        return None
    
    def _execute_pickup(self, target_info: Dict) -> bool:
        """
        Execute pickup command using hybrid IK system.
        
        Args:
            target_info: Dictionary with object type and position
            
        Returns:
            True if pickup was successful
        """
        print(f"Executing pickup for {target_info['type']} at {target_info['position']}")
        
        # Use the hybrid IK system for object pickup
        success, joint_solution, metadata = self.kinematics.solve_for_object_pickup(
            object_position=target_info['position'],
            object_type=target_info['type'],
            current_joints=self.current_joints
        )
        
        if success:
            print(f"IK solution found in {metadata['solve_time_ms']:.1f}ms")
            
            if metadata['is_approximation']:
                print(f"Using approximation with {metadata['position_error_mm']:.1f}mm error")
            
            self._move_to_joints(joint_solution)
            
            # Simulate gripper action
            self._close_gripper()
            
            return True
        else:
            print(f"IK failed: no solution found")
            print(f"Solver attempts: {metadata['solver_used']}")
            return False
    
    def _move_to_joints(self, joint_angles: List[float]):
        """Simulate robot motion to target joint angles."""
        print(f"Moving to joints: {[round(math.degrees(j), 1) for j in joint_angles]}°")
        
        # Simulate motion time
        motion_time = 2.0  # seconds
        self.is_moving = True
        
        self.current_joints = joint_angles.copy()
        
        # Simulate motion completion
        time.sleep(0.1)  # Small delay for demonstration
        self.is_moving = False
        
        print("Motion completed")
    
    def _close_gripper(self):
        """Simulate gripper closing."""
        print("Closing gripper...")
        time.sleep(0.05)  # Small delay for demonstration
        print("Object grasped!")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        # VLM-level statistics
        total_commands = len(self.command_history)
        successful_commands = sum(1 for cmd in self.command_history if cmd['success'])
        
        vlm_stats = {
            "total_vlm_commands": total_commands,
            "successful_vlm_commands": successful_commands,
            "vlm_command_success_rate": successful_commands / max(1, total_commands),
        }
        
        if self.command_history:
            command_times = [cmd['time_ms'] for cmd in self.command_history]
            vlm_stats.update({
                "avg_command_time_ms": np.mean(command_times),
                "median_command_time_ms": np.median(command_times),
                "max_command_time_ms": np.max(command_times)
            })
        
        # Get IK-level statistics
        ik_stats = self.kinematics.get_vlm_performance_stats()
        
        return {**vlm_stats, **ik_stats}
    
    def print_comprehensive_summary(self):
        """Print comprehensive performance summary."""
        stats = self.get_performance_summary()
        
        print("\n" + "="*50)
        print("COMPREHENSIVE VLM SYSTEM PERFORMANCE")
        print("="*50)
        
        # Command-level performance
        print(f"Total VLM commands processed: {stats['total_vlm_commands']}")
        print(f"Command success rate: {stats['vlm_command_success_rate']:.1%}")
        
        if 'avg_command_time_ms' in stats:
            print(f"Average command time: {stats['avg_command_time_ms']:.1f}ms")
            print(f"Median command time: {stats['median_command_time_ms']:.1f}ms")
        
        print()
        
        # IK-level performance
        self.kinematics.print_vlm_performance_summary()
        
        # Recent command history
        if self.command_history:
            print(f"\nRecent Commands:")
            for i, cmd in enumerate(self.command_history[-5:], 1):
                status = "✓" if cmd['success'] else "✗"
                print(f"  {i}. {status} '{cmd['command']}' ({cmd['time_ms']:.1f}ms)")


def simulate_vlm_session():
    """Simulate a VLM interaction session with various commands."""
    print("=== VLM Hybrid IK Integration Demo ===")
    
    controller = VLMRobotController(debug=True)
    
    # Simulate detected objects from vision system
    detected_objects = [
        {
            "type": "water_bottle", 
            "position": [0.4, 0.2, 0.1],
            "confidence": 0.95,
            "bbox": [100, 150, 50, 120]
        },
        {
            "type": "coffee_cup",
            "position": [0.35, -0.1, 0.08], 
            "confidence": 0.87,
            "bbox": [200, 180, 60, 80]
        },
        {
            "type": "book",
            "position": [0.3, 0.15, 0.02],
            "confidence": 0.92,
            "bbox": [150, 120, 80, 100]
        },
        {
            "type": "phone",
            "position": [0.25, 0.0, 0.05],
            "confidence": 0.89,
            "bbox": [180, 200, 40, 70]
        }
    ]
    
    # Test various VLM commands
    test_commands = [
        "pick up the water bottle",
        "grab the coffee cup", 
        "take the book",
        "get the phone",
        "pick up the remote control",  # Object not present
        "wave hello"  # Unsupported action
    ]
    
    print(f"\nDetected {len(detected_objects)} objects in workspace")
    for obj in detected_objects:
        print(f"  - {obj['type']} at {obj['position']} (conf: {obj['confidence']:.2f})")
    
    for i, command in enumerate(test_commands, 1):
        print(f"\n{'='*20} Command {i}/{len(test_commands)} {'='*20}")
        success = controller.process_vlm_command(command, detected_objects)
        
        # Small delay between commands
        time.sleep(0.1)
    
    # Show comprehensive performance summary
    controller.print_comprehensive_summary()


def compare_ik_performance():
    """Compare performance between hybrid and numerical-only IK."""
    print("\n=== IK Performance Comparison ===")
    
    hybrid_solver = VLMKinematicsController(debug=False)
    numerical_solver = VLMKinematicsController(debug=False)
    # Force numerical solver to not use ikfast
    numerical_solver.kinematics.ikfast_available = False
    
    # Test poses
    test_poses = [
        [0.4, 0.2, 0.3],    # Reachable
        [0.5, 0.1, 0.2],    # Reachable
        [0.3, 0.3, 0.4],    # Reachable
        [0.6, 0.0, 0.1],    # Near boundary
        [0.8, 0.2, 0.1],    # Potentially unreachable
    ]
    
    print(f"Testing {len(test_poses)} poses...")
    
    hybrid_times = []
    numerical_times = []
    
    for i, pos in enumerate(test_poses):
        print(f"\nPose {i+1}: {pos}")
        
        # Test hybrid solver
        start_time = time.perf_counter()
        success_h, joints_h, meta_h = hybrid_solver.solve_for_vlm_target(
            target_position=pos,
            target_orientation="top_down"
        )
        hybrid_time = (time.perf_counter() - start_time) * 1000
        hybrid_times.append(hybrid_time)
        
        # Test numerical solver
        start_time = time.perf_counter()
        success_n, joints_n, meta_n = numerical_solver.solve_for_vlm_target(
            target_position=pos,
            target_orientation="top_down"
        )
        numerical_time = (time.perf_counter() - start_time) * 1000
        numerical_times.append(numerical_time)
        
        print(f"  Hybrid: {'✓' if success_h else '✗'} {hybrid_time:.1f}ms")
        print(f"  Numerical: {'✓' if success_n else '✗'} {numerical_time:.1f}ms")
        print(f"  Speedup: {numerical_time/hybrid_time:.1f}x")
    
    # Summary statistics
    print(f"\nPerformance Summary:")
    print(f"Hybrid average: {np.mean(hybrid_times):.1f}ms")
    print(f"Numerical average: {np.mean(numerical_times):.1f}ms")
    print(f"Overall speedup: {np.mean(numerical_times)/np.mean(hybrid_times):.1f}x")


if __name__ == "__main__":
    simulate_vlm_session()
    
    # Run performance comparison
    compare_ik_performance() 