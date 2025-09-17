"""
Hybrid IK Wrapper for VLM Integration
=====================================

This module provides a convenient interface for integrating the hybrid ur_ikfast + numerical IK
system with the existing VLM (Vision-Language Model) robotic control pipeline.

Key Features:
- Drop-in replacement for existing IK calls
- Automatic fallback from fast analytical to robust numerical solving
- VLM-specific optimizations for handling vision noise and uncertainty
- Performance monitoring and adaptive timeout management
- Graceful degradation for unreachable poses

Usage Example:
    from HybridIKWrapper import VLMKinematicsController
    
    controller = VLMKinematicsController()
    
    # For VLM pick commands
    success, joints = controller.solve_for_vlm_target(
        target_position=[0.4, 0.2, 0.3],
        target_orientation="top_down_grasp",  # or custom rotation matrix
        current_joints=robot.get_current_joints()
    )
"""

import numpy as np
import math
from typing import List, Optional, Tuple, Union, Dict, Any
import logging
import time

# Try to import scipy, but don't fail if there are library issues
try:
    from scipy.spatial.transform import Rotation as R
    SCIPY_AVAILABLE = True
except ImportError as e:
    print(f"Scipy import failed in HybridIKWrapper: {e}")
    print("This is expected on macOS - will work fine on Ubuntu")
    SCIPY_AVAILABLE = False
    R = None

from UR5eKinematics import HybridUR5eKinematics, UR5eKinematics

class VLMKinematicsController:
    """
    VLM-optimized kinematics controller that intelligently handles vision-based requests.
    
    This controller is specifically designed for Vision-Language Model applications where:
    - Object detection may have small coordinate errors
    - Requested poses might be near workspace boundaries
    - Fast response times are critical for natural interaction
    - Robustness is more important than mathematical precision
    """
    
    def __init__(self, 
                 enable_ikfast: bool = True,
                 adaptive_timeout: bool = True,
                 debug: bool = False):
        """
        Initialize the VLM kinematics controller.
        
        Args:
            enable_ikfast: Use ur_ikfast when available for speed
            adaptive_timeout: Adjust timeout based on recent performance
            debug: Enable debug output
        """
        self.debug = debug
        self.adaptive_timeout = adaptive_timeout
        
        self.ik_solver = HybridUR5eKinematics(
            enable_fallback=True,
            debug=debug
        )
        
        # Adaptive timeout management
        self.base_timeout_ms = 50.0  # Base timeout
        self.timeout_history = []    # Recent solve times
        self.success_history = []    # Recent success rates
        
        # VLM-specific pose presets
        self.grasp_orientations = {
            "top_down": self._create_top_down_orientation(),
            "side_grasp": self._create_side_grasp_orientation(),
            "angled_grasp": self._create_angled_grasp_orientation(),
        }
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.approximate_solutions = 0
        
        if debug:
            print("VLM Kinematics Controller initialized")
            print(f"ur_ikfast available: {self.ik_solver.ikfast_available}")
    
    def solve_for_vlm_target(self,
                           target_position: List[float],
                           target_orientation: Union[str, np.ndarray] = "top_down",
                           current_joints: Optional[List[float]] = None,
                           allow_approximation: bool = True,
                           max_position_error_mm: float = 10.0) -> Tuple[bool, Optional[List[float]], Dict[str, Any]]:
        """
        Solve IK for a VLM-detected target with intelligent fallback handling.
        
        Args:
            target_position: [x, y, z] position in meters
            target_orientation: Preset name or 3x3 rotation matrix
            current_joints: Current robot joint positions
            allow_approximation: Allow approximate solutions for unreachable poses
            max_position_error_mm: Maximum acceptable position error for approximations
            
        Returns:
            Tuple of (success, joint_angles, metadata)
            - success: True if a valid solution was found
            - joint_angles: List of 6 joint angles in radians (None if failed)
            - metadata: Dictionary with solving information
        """
        self.total_requests += 1
        start_time = time.perf_counter()
        
        target_pose = self._create_target_pose(target_position, target_orientation)
        
        # Determine timeout based on adaptive strategy
        timeout_ms = self._get_adaptive_timeout()
        
        # Attempt to solve with hybrid IK
        solutions = self.ik_solver.inverse_kinematics(
            target_pose=target_pose,
            current_joints=current_joints,
            prefer_closest=True,
            timeout_ms=timeout_ms
        )
        
        solve_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Prepare metadata
        metadata = {
            "solve_time_ms": solve_time_ms,
            "timeout_ms": timeout_ms,
            "solutions_found": len(solutions),
            "solver_used": [],
            "is_approximation": False,
            "position_error_mm": 0.0,
            "orientation_error_deg": 0.0
        }
        
        # Track which solvers were used
        if self.ik_solver.ikfast_available and solutions:
            metadata["solver_used"].append("ur_ikfast")
        if len(solutions) > 0:
            metadata["solver_used"].append("numerical")
        
        if solutions:
            # Exact solution found
            best_joints = solutions[0]  # Already sorted by preference
            self.successful_requests += 1
            
            # Validate solution accuracy
            achieved_pose = self.ik_solver.forward_kinematics(best_joints)
            pos_error_mm = np.linalg.norm(achieved_pose[:3, 3] - target_pose[:3, 3]) * 1000
            
            rot_error_matrix = achieved_pose[:3, :3].T @ target_pose[:3, :3]
            rot_angle = math.acos(np.clip((np.trace(rot_error_matrix) - 1) / 2, -1.0, 1.0))
            ori_error_deg = math.degrees(rot_angle)
            
            metadata["position_error_mm"] = pos_error_mm
            metadata["orientation_error_deg"] = ori_error_deg
            
            self._update_performance_tracking(solve_time_ms, True)
            
            if self.debug:
                print(f"Exact solution found in {solve_time_ms:.1f}ms")
                print(f"Position error: {pos_error_mm:.1f}mm, Orientation error: {ori_error_deg:.1f}°")
            
            return True, best_joints, metadata
        
        elif allow_approximation:
            # No exact solution - try to find closest reachable pose
            if self.debug:
                print("No exact solution found, trying approximation...")
            
            best_joints, reachable_pose = self.ik_solver.best_reachable_pose(
                target_pose, current_joints
            )
            
            # Check if approximation is acceptable
            pos_error_mm = np.linalg.norm(reachable_pose[:3, 3] - target_pose[:3, 3]) * 1000
            
            if pos_error_mm <= max_position_error_mm:
                # Acceptable approximation
                self.successful_requests += 1
                self.approximate_solutions += 1
                
                metadata["is_approximation"] = True
                metadata["position_error_mm"] = pos_error_mm
                metadata["solver_used"].append("approximation")
                
                self._update_performance_tracking(solve_time_ms, True)
                
                if self.debug:
                    print(f"Acceptable approximation found with {pos_error_mm:.1f}mm error")
                
                return True, best_joints, metadata
            else:
                # Approximation error too large
                metadata["position_error_mm"] = pos_error_mm
                metadata["solver_used"].append("approximation_failed")
                
                self._update_performance_tracking(solve_time_ms, False)
                
                if self.debug:
                    print(f"Approximation error {pos_error_mm:.1f}mm exceeds limit {max_position_error_mm}mm")
                
                return False, None, metadata
        
        else:
            # No approximation allowed
            self._update_performance_tracking(solve_time_ms, False)
            
            if self.debug:
                print("No solution found and approximation disabled")
            
            return False, None, metadata
    
    def solve_for_object_pickup(self,
                              object_position: List[float],
                              object_type: str = "unknown",
                              current_joints: Optional[List[float]] = None) -> Tuple[bool, Optional[List[float]], Dict[str, Any]]:
        """
        Specialized solver for object pickup tasks from VLM commands.
        
        Args:
            object_position: [x, y, z] position of detected object
            object_type: Type of object (used for grasp strategy)
            current_joints: Current robot joint positions
            
        Returns:
            Tuple of (success, joint_angles, metadata)
        """
        # Select grasp orientation based on object type
        if object_type.lower() in ["bottle", "can", "cylinder"]:
            orientation = "side_grasp"
        elif object_type.lower() in ["box", "book", "flat"]:
            orientation = "top_down"
        else:
            orientation = "angled_grasp"  # Default for unknown objects
        
        approach_position = object_position.copy()
        approach_position[2] += 0.05  # 5cm above
        
        return self.solve_for_vlm_target(
            target_position=approach_position,
            target_orientation=orientation,
            current_joints=current_joints,
            allow_approximation=True,
            max_position_error_mm=15.0  # More tolerance for object pickup
        )
    
    def _create_target_pose(self, position: List[float], orientation: Union[str, np.ndarray]) -> np.ndarray:
        """Create 4x4 pose matrix from position and orientation."""
        pose = np.eye(4)
        pose[:3, 3] = position
        
        if isinstance(orientation, str):
            if orientation in self.grasp_orientations:
                pose[:3, :3] = self.grasp_orientations[orientation]
            else:
                raise ValueError(f"Unknown orientation preset: {orientation}")
        else:
            pose[:3, :3] = orientation
        
        return pose
    
    def _create_top_down_orientation(self) -> np.ndarray:
        """Create top-down grasp orientation matrix."""
        # Z-axis pointing down, X-axis forward
        return np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
    
    def _create_side_grasp_orientation(self) -> np.ndarray:
        """Create side grasp orientation matrix."""
        # Z-axis pointing forward, Y-axis down
        return np.array([
            [0, 0, 1],
            [0, -1, 0],
            [1, 0, 0]
        ])
    
    def _create_angled_grasp_orientation(self) -> np.ndarray:
        """Create angled grasp orientation matrix (45° from vertical)."""
        angle = math.pi / 4  # 45 degrees
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        return np.array([
            [cos_a, 0, sin_a],
            [0, -1, 0],
            [-sin_a, 0, cos_a]
        ])
    
    def _get_adaptive_timeout(self) -> float:
        """Get adaptive timeout based on recent performance."""
        if not self.adaptive_timeout or len(self.timeout_history) < 5:
            return self.base_timeout_ms
        
        # Use 90th percentile of recent solve times plus buffer
        recent_times = self.timeout_history[-10:]  # Last 10 solve times
        percentile_90 = np.percentile(recent_times, 90)
        
        # Adaptive timeout with limits
        adaptive_timeout = min(max(percentile_90 * 1.5, 20.0), 200.0)
        
        return adaptive_timeout
    
    def _update_performance_tracking(self, solve_time_ms: float, success: bool):
        """Update performance tracking for adaptive behavior."""
        self.timeout_history.append(solve_time_ms)
        self.success_history.append(success)
        
        # Keep only recent history
        if len(self.timeout_history) > 50:
            self.timeout_history = self.timeout_history[-25:]
            self.success_history = self.success_history[-25:]
    
    def get_vlm_performance_stats(self) -> Dict[str, Any]:
        """Get VLM-specific performance statistics."""
        base_stats = self.ik_solver.get_performance_stats()
        
        vlm_stats = {
            "total_vlm_requests": self.total_requests,
            "successful_vlm_requests": self.successful_requests,
            "vlm_success_rate": self.successful_requests / max(1, self.total_requests),
            "approximate_solutions": self.approximate_solutions,
            "approximation_rate": self.approximate_solutions / max(1, self.successful_requests),
        }
        
        if self.timeout_history:
            vlm_stats["avg_solve_time_ms"] = np.mean(self.timeout_history)
            vlm_stats["median_solve_time_ms"] = np.median(self.timeout_history)
            vlm_stats["max_solve_time_ms"] = np.max(self.timeout_history)
        
        if self.success_history:
            recent_success = np.mean(self.success_history[-10:]) if len(self.success_history) >= 10 else np.mean(self.success_history)
            vlm_stats["recent_success_rate"] = recent_success
        
        # Combine with base stats
        return {**base_stats, **vlm_stats}
    
    def print_vlm_performance_summary(self):
        """Print comprehensive performance summary for VLM usage."""
        stats = self.get_vlm_performance_stats()
        
        print("\n=== VLM Kinematics Performance Summary ===")
        print(f"Total VLM requests: {stats['total_vlm_requests']}")
        print(f"Success rate: {stats['vlm_success_rate']:.1%}")
        print(f"Approximate solutions: {stats['approximation_rate']:.1%} of successes")
        
        if 'avg_solve_time_ms' in stats:
            print(f"Average solve time: {stats['avg_solve_time_ms']:.1f}ms")
            print(f"Median solve time: {stats['median_solve_time_ms']:.1f}ms")
        
        if 'recent_success_rate' in stats:
            print(f"Recent success rate: {stats['recent_success_rate']:.1%}")
        
        # Show underlying solver performance
        if stats['ikfast_available']:
            print(f"\nur_ikfast: {stats['ikfast_success_rate']:.1%} success, "
                  f"{stats.get('avg_ikfast_time_ms', 0):.2f}ms avg")
        
        print(f"Numerical: {stats['numerical_success_rate']:.1%} success, "
              f"{stats.get('avg_numerical_time_ms', 0):.1f}ms avg")


def test_vlm_integration():
    """Test the VLM integration features."""
    print("=== VLM Integration Test ===")
    
    controller = VLMKinematicsController(debug=True)
    
    # Test 1: Basic object pickup
    print("\n1. Testing basic object pickup:")
    success, joints, metadata = controller.solve_for_object_pickup(
        object_position=[0.4, 0.2, 0.1],
        object_type="bottle",
        current_joints=[0, -math.pi/2, 0, 0, 0, 0]
    )
    
    print(f"Success: {success}")
    if success:
        print(f"Joint solution: {[round(math.degrees(j), 1) for j in joints]}")
    print(f"Metadata: {metadata}")
    
    # Test 2: Different object types
    print("\n2. Testing different object types:")
    object_tests = [
        ([0.3, 0.1, 0.05], "box"),
        ([0.35, -0.1, 0.08], "can"),
        ([0.25, 0.15, 0.03], "unknown")
    ]
    
    for pos, obj_type in object_tests:
        success, joints, metadata = controller.solve_for_object_pickup(
            object_position=pos,
            object_type=obj_type
        )
        print(f"{obj_type} at {pos}: {'✓' if success else '✗'} "
              f"({metadata['solve_time_ms']:.1f}ms)")
    
    # Test 3: VLM target with custom orientation
    print("\n3. Testing custom orientation:")
    custom_rotation = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1]
    ])
    
    success, joints, metadata = controller.solve_for_vlm_target(
        target_position=[0.4, 0.0, 0.2],
        target_orientation=custom_rotation,
        allow_approximation=True
    )
    
    print(f"Custom orientation: {'✓' if success else '✗'}")
    print(f"Is approximation: {metadata['is_approximation']}")
    print(f"Position error: {metadata['position_error_mm']:.1f}mm")
    
    # Performance summary
    controller.print_vlm_performance_summary()


if __name__ == "__main__":
    test_vlm_integration() 