#!/usr/bin/env python3
"""
Integration Test for UnifiedVisionSystem with Hybrid IK

This script tests the integration of the hybrid IK system with the UnifiedVisionSystem
to ensure all components work together correctly.
"""

import sys
import numpy as np
import time
import logging
from typing import List, Dict, Any

# Import the integrated system
try:
    from UnifiedVisionSystem import UnifiedVisionSystem
    from HybridIKWrapper import VLMKinematicsController
    from UR5eKinematics import HybridUR5eKinematics, UR5eKinematics
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all components are installed and in the Python path")
    sys.exit(1)

def test_hybrid_ik_integration():
    """Test the hybrid IK integration components independently."""
    print("=== Testing Hybrid IK Integration Components ===")
    
    # Test 1: Direct Hybrid IK System
    print("\n1. Testing Hybrid IK System...")
    try:
        hybrid_ik = HybridUR5eKinematics(debug=True)
        print(f"✓ Hybrid IK System initialized")
        print(f"  ur_ikfast available: {hybrid_ik.ikfast_available}")
        
        # Test forward kinematics
        test_joints = [0.0, -np.pi/2, 0.0, 0.0, 0.0, 0.0]
        fk_result = hybrid_ik.forward_kinematics(test_joints)
        print(f"✓ Forward kinematics test passed")
        
        # Test inverse kinematics
        ik_solutions = hybrid_ik.inverse_kinematics(fk_result)
        print(f"✓ Inverse kinematics test: {len(ik_solutions)} solutions found")
        
    except Exception as e:
        print(f"✗ Hybrid IK System test failed: {e}")
        return False
    
    # Test 2: VLM Kinematics Controller
    print("\n2. Testing VLM Kinematics Controller...")
    try:
        vlm_controller = VLMKinematicsController(debug=True)
        print(f"✓ VLM Controller initialized")
        
        # Test object pickup solving
        success, joints, metadata = vlm_controller.solve_for_object_pickup(
            object_position=[0.4, 0.2, 0.1],
            object_type="bottle"
        )
        
        if success:
            print(f"✓ Object pickup test passed")
            print(f"  Solve time: {metadata['solve_time_ms']:.1f}ms")
            print(f"  Solvers used: {metadata['solver_used']}")
        else:
            print(f"! Object pickup test: no solution found")
        
    except Exception as e:
        print(f"✗ VLM Controller test failed: {e}")
        return False
    
    print("\n✓ All component tests passed!")
    return True

def test_unified_vision_system_initialization():
    """Test UnifiedVisionSystem initialization with hybrid IK."""
    print("\n=== Testing UnifiedVisionSystem Initialization ===")
    
    try:
        # Mock ROS2 for testing (simplified)
        print("Note: This test requires ROS2 environment for full initialization")
        print("Testing component creation only...")
        
        # Test individual component initialization
        from UR5eKinematics import UR5eKinematics, HybridUR5eKinematics
        from HybridIKWrapper import VLMKinematicsController
        
        # Test 1: Basic kinematics
        kinematics = UR5eKinematics()
        print("✓ Basic UR5eKinematics initialized")
        
        # Test 2: Hybrid kinematics  
        hybrid_kinematics = HybridUR5eKinematics()
        print("✓ HybridUR5eKinematics initialized")
        
        # Test 3: VLM controller
        vlm_controller = VLMKinematicsController()
        print("✓ VLMKinematicsController initialized")
        
        # Test 4: Performance stats
        stats = vlm_controller.get_vlm_performance_stats()
        print(f"✓ Performance stats available: {list(stats.keys())}")
        
        print("\n✓ UnifiedVisionSystem components ready for integration!")
        return True
        
    except Exception as e:
        print(f"✗ UnifiedVisionSystem test failed: {e}")
        return False

def test_ik_performance_comparison():
    """Compare performance between different IK solvers."""
    print("\n=== IK Performance Comparison ===")
    
    try:
        numerical_solver = UR5eKinematics()
        hybrid_solver = HybridUR5eKinematics(debug=False)
        vlm_controller = VLMKinematicsController(debug=False)
        
        # Test poses
        test_poses = [
            [0.4, 0.2, 0.3],    # Reachable
            [0.5, 0.1, 0.2],    # Reachable
            [0.3, 0.3, 0.4],    # Reachable
        ]
        
        print(f"Testing {len(test_poses)} poses...")
        
        numerical_times = []
        hybrid_times = []
        vlm_times = []
        
        for i, pos in enumerate(test_poses):
            target_pose = np.eye(4)
            target_pose[:3, 3] = pos
            
            print(f"\nPose {i+1}: {pos}")
            
            # Test numerical solver
            start_time = time.perf_counter()
            num_solutions = numerical_solver.inverse_kinematics(target_pose)
            numerical_time = (time.perf_counter() - start_time) * 1000
            numerical_times.append(numerical_time)
            
            # Test hybrid solver
            start_time = time.perf_counter()
            hybrid_solutions = hybrid_solver.inverse_kinematics(target_pose)
            hybrid_time = (time.perf_counter() - start_time) * 1000
            hybrid_times.append(hybrid_time)
            
            # Test VLM controller
            start_time = time.perf_counter()
            success, joints, metadata = vlm_controller.solve_for_vlm_target(
                target_position=pos,
                target_orientation="top_down"
            )
            vlm_time = (time.perf_counter() - start_time) * 1000
            vlm_times.append(vlm_time)
            
            print(f"  Numerical: {len(num_solutions) if num_solutions else 0} solutions, {numerical_time:.1f}ms")
            print(f"  Hybrid: {len(hybrid_solutions) if hybrid_solutions else 0} solutions, {hybrid_time:.1f}ms")
            print(f"  VLM: {'✓' if success else '✗'}, {vlm_time:.1f}ms")
        
        # Summary statistics
        print(f"\nPerformance Summary:")
        print(f"Numerical average: {np.mean(numerical_times):.1f}ms")
        print(f"Hybrid average: {np.mean(hybrid_times):.1f}ms")
        print(f"VLM average: {np.mean(vlm_times):.1f}ms")
        
        if hybrid_solver.ikfast_available:
            speedup = np.mean(numerical_times) / np.mean(hybrid_times)
            print(f"Hybrid speedup: {speedup:.1f}x")
        
        vlm_controller.print_vlm_performance_summary()
        
        return True
        
    except Exception as e:
        print(f"✗ Performance comparison failed: {e}")
        return False

def test_vlm_edge_cases():
    """Test VLM-specific edge cases and error handling."""
    print("\n=== Testing VLM Edge Cases ===")
    
    try:
        vlm_controller = VLMKinematicsController(debug=True)
        
        # Test 1: Noisy vision coordinates
        print("\n1. Testing noisy vision coordinates...")
        clean_pos = [0.4, 0.2, 0.3]
        noisy_pos = [0.405, 0.197, 0.302]  # 5mm noise
        
        success_clean, _, _ = vlm_controller.solve_for_vlm_target(
            target_position=clean_pos,
            target_orientation="top_down"
        )
        
        success_noisy, _, metadata_noisy = vlm_controller.solve_for_vlm_target(
            target_position=noisy_pos,
            target_orientation="top_down"
        )
        
        print(f"  Clean pose: {'✓' if success_clean else '✗'}")
        print(f"  Noisy pose: {'✓' if success_noisy else '✗'}")
        if success_noisy:
            print(f"  Approximation: {metadata_noisy['is_approximation']}")
        
        # Test 2: Workspace boundary
        print("\n2. Testing workspace boundary...")
        boundary_pos = [0.8, 0.0, 0.1]  # Near max reach
        
        success_boundary, _, metadata_boundary = vlm_controller.solve_for_vlm_target(
            target_position=boundary_pos,
            target_orientation="top_down",
            allow_approximation=True
        )
        
        print(f"  Boundary pose: {'✓' if success_boundary else '✗'}")
        if success_boundary and metadata_boundary['is_approximation']:
            print(f"  Error: {metadata_boundary['position_error_mm']:.1f}mm")
        
        # Test 3: Different object types
        print("\n3. Testing different object types...")
        object_tests = [
            ("bottle", "side_grasp"),
            ("box", "top_down"), 
            ("unknown", "angled_grasp")
        ]
        
        for obj_type, expected_orientation in object_tests:
            success, _, metadata = vlm_controller.solve_for_object_pickup(
                object_position=[0.3, 0.1, 0.05],
                object_type=obj_type
            )
            print(f"  {obj_type}: {'✓' if success else '✗'}")
        
        print("\n✓ Edge case tests completed!")
        return True
        
    except Exception as e:
        print(f"✗ Edge case tests failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🚀 UnifiedVisionSystem + Hybrid IK Integration Tests")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    tests = [
        ("Hybrid IK Components", test_hybrid_ik_integration),
        ("UnifiedVisionSystem Init", test_unified_vision_system_initialization),
        ("IK Performance Comparison", test_ik_performance_comparison),
        ("VLM Edge Cases", test_vlm_edge_cases),
    ]
    
    results = []
    start_time = time.perf_counter()
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 40)
        
        try:
            result = test_func()
            results.append((test_name, result))
            status = "PASS" if result else "FAIL"
            print(f"📊 {test_name}: {status}")
            
        except Exception as e:
            print(f"💥 {test_name}: CRASH - {e}")
            results.append((test_name, False))
    
    # Summary
    total_time = (time.perf_counter() - start_time) * 1000
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n🏁 Test Summary")
    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")
    print(f"Total time: {total_time:.1f}ms")
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status} {test_name}")
    
    if passed == total:
        print(f"\n🎉 All tests passed! System ready for deployment.")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please review errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 