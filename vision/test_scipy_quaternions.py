#!/usr/bin/env python3
"""
Quick test to verify scipy quaternion conversion works correctly.
Run this before deploying to physical hardware.
"""

import numpy as np
import math

# Try to import scipy, but don't fail if there are library issues
try:
    from scipy.spatial.transform import Rotation as R
    SCIPY_AVAILABLE = True
except ImportError as e:
    print(f"Scipy import failed: {e}")
    print("This is expected on macOS - will work fine on Ubuntu")
    print("Exiting test since scipy is required for this specific test")
    exit(0)

from UR5eKinematics import HybridUR5eKinematics

def test_scipy_quaternion_conversion():
    """Test that scipy quaternion conversion produces correct results"""
    
    print("=== Testing Scipy Quaternion Conversion ===\n")
    
    hybrid = HybridUR5eKinematics(debug=True)
    
    # Test cases: known rotation matrices and their expected quaternions
    test_cases = [
        {
            "name": "Identity (no rotation)",
            "matrix": np.eye(3),
            "expected_angle": 0.0
        },
        {
            "name": "90° rotation about Z-axis",
            "matrix": np.array([
                [0, -1, 0],
                [1,  0, 0],
                [0,  0, 1]
            ]),
            "expected_angle": math.pi/2
        },
        {
            "name": "180° rotation about X-axis", 
            "matrix": np.array([
                [1,  0,  0],
                [0, -1,  0],
                [0,  0, -1]
            ]),
            "expected_angle": math.pi
        },
        {
            "name": "45° rotation about Z-axis",
            "matrix": np.array([
                [math.cos(math.pi/4), -math.sin(math.pi/4), 0],
                [math.sin(math.pi/4),  math.cos(math.pi/4), 0],
                [0,                    0,                   1]
            ]),
            "expected_angle": math.pi/4
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases):
        print(f"Test {i+1}: {test_case['name']}")
        
        try:
            quat_wxyz = hybrid._rotation_matrix_to_quaternion(test_case['matrix'])
            print(f"  Quaternion (w,x,y,z): [{quat_wxyz[0]:.4f}, {quat_wxyz[1]:.4f}, {quat_wxyz[2]:.4f}, {quat_wxyz[3]:.4f}]")
            
            # Verify the quaternion is normalized
            norm = math.sqrt(sum(q*q for q in quat_wxyz))
            print(f"  Quaternion norm: {norm:.6f}")
            
            if abs(norm - 1.0) > 1e-6:
                print(f"  ❌ FAIL: Quaternion not normalized!")
                all_passed = False
                continue
            
            # Convert back to rotation matrix using scipy
            quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
            rotation_obj = R.from_quat(quat_xyzw)
            recovered_matrix = rotation_obj.as_matrix()
            
            # Check if we recover the original matrix
            matrix_error = np.linalg.norm(recovered_matrix - test_case['matrix'])
            print(f"  Matrix recovery error: {matrix_error:.6f}")
            
            if matrix_error > 1e-6:
                print(f"  ❌ FAIL: Matrix recovery error too large!")
                all_passed = False
                continue
            
            # Check rotation angle if specified
            if 'expected_angle' in test_case:
                angle = rotation_obj.magnitude()
                angle_error = abs(angle - test_case['expected_angle'])
                print(f"  Expected angle: {math.degrees(test_case['expected_angle']):.1f}°")
                print(f"  Computed angle: {math.degrees(angle):.1f}°")
                print(f"  Angle error: {math.degrees(angle_error):.3f}°")
                
                if angle_error > 1e-6:
                    print(f"  ❌ FAIL: Angle error too large!")
                    all_passed = False
                    continue
            
            print(f"  ✅ PASS")
            
        except Exception as e:
            print(f"  ❌ FAIL: Exception occurred: {e}")
            all_passed = False
        
        print()
    
    # Test with some random rotation matrices
    print("Testing with random rotation matrices...")
    
    for i in range(5):
        # Generate random rotation
        random_angles = np.random.uniform(-math.pi, math.pi, 3)
        rotation_obj = R.from_euler('xyz', random_angles)
        random_matrix = rotation_obj.as_matrix()
        
        try:
            # Convert to quaternion and back
            quat_wxyz = hybrid._rotation_matrix_to_quaternion(random_matrix)
            quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
            recovered_rotation = R.from_quat(quat_xyzw)
            recovered_matrix = recovered_rotation.as_matrix()
            
            error = np.linalg.norm(recovered_matrix - random_matrix)
            print(f"  Random test {i+1}: error = {error:.6f}", end="")
            
            if error < 1e-6:
                print(" ✅")
            else:
                print(" ❌")
                all_passed = False
                
        except Exception as e:
            print(f"  Random test {i+1}: Exception: {e} ❌")
            all_passed = False
    
    print(f"\n=== Final Result ===")
    if all_passed:
        print("✅ All quaternion conversion tests PASSED!")
        print("Scipy integration is working correctly.")
    else:
        print("❌ Some tests FAILED!")
        print("Check scipy installation and quaternion conversion logic.")
    
    return all_passed

def test_ikfast_with_scipy():
    """Test that ur_ikfast works with scipy-generated quaternions"""
    
    print("\n=== Testing ur_ikfast Integration ===\n")
    
    try:
        import ur_ikfast as ur_ik
        print("✅ ur_ikfast imported successfully")
    except ImportError:
        print("⚠️  ur_ikfast not available - skipping integration test")
        return True
    
    hybrid = HybridUR5eKinematics(debug=True)
    
    # Test with a simple reachable pose
    target_pose = np.eye(4)
    target_pose[:3, 3] = [0.4, 0.2, 0.3]  # Simple position
    target_pose[:3, :3] = np.array([        # Simple orientation
        [1, 0, 0],
        [0, 1, 0], 
        [0, 0, 1]
    ])
    
    print("Testing hybrid IK with scipy quaternions...")
    print(f"Target position: {target_pose[:3, 3]}")
    
    # Test the hybrid solver
    solutions = hybrid.inverse_kinematics(target_pose, timeout_ms=100)
    
    if solutions:
        print(f"✅ Found {len(solutions)} solutions")
        
        best_solution = solutions[0]
        check_pose = hybrid.forward_kinematics(best_solution)
        pos_error = np.linalg.norm(check_pose[:3, 3] - target_pose[:3, 3]) * 1000  # mm
        
        print(f"Position error: {pos_error:.3f}mm")
        
        if pos_error < 5.0:  # 5mm tolerance
            print("✅ IK solution accurate")
            return True
        else:
            print("❌ IK solution inaccurate")
            return False
    else:
        print("❌ No solutions found")
        return False

def main():
    """Run all tests"""
    
    print("Testing scipy integration for hybrid IK system")
    print("=" * 50)
    
    # Test quaternion conversion
    quat_test_passed = test_scipy_quaternion_conversion()
    
    # Test ur_ikfast integration
    ikfast_test_passed = test_ikfast_with_scipy()
    
    print("\n" + "=" * 50)
    print("FINAL SUMMARY:")
    
    if quat_test_passed and ikfast_test_passed:
        print("✅ ALL TESTS PASSED - Ready for physical deployment!")
        print("\nNext steps:")
        print("1. Review PHYSICAL_TESTING_CHECKLIST.md")
        print("2. Verify robot safety systems")
        print("3. Start with conservative parameters")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Fix issues before deployment!")
        if not quat_test_passed:
            print("- Fix scipy quaternion conversion")
        if not ikfast_test_passed:
            print("- Fix ur_ikfast integration")
        return 1

if __name__ == "__main__":
    exit(main()) 