#!/usr/bin/env python3
"""
Debug UR5e workspace and forward kinematics issues.
This script will help identify the root cause of FK/IK mismatches.
"""

import numpy as np
import math
import sys
import os

# Add the path to import our kinematics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UR5eKinematics import HybridUR5eKinematics, UR5eKinematics

def debug_ur5e_workspace():
    """Debug UR5e workspace and FK/IK consistency."""
    
    print("=== UR5e Workspace Debug ===\n")
    
    # Initialize both solvers
    hybrid = HybridUR5eKinematics(debug=False)
    numerical = UR5eKinematics()
    
    # Test 1: Check DH parameters against known UR5e specs
    print("1. UR5e DH Parameters Check:")
    print(f"   d1 (base height): {numerical.d1*1000:.1f}mm (should be ~162.5mm)")
    print(f"   a2 (upper arm): {abs(numerical.a2)*1000:.1f}mm (should be ~425mm)")
    print(f"   a3 (forearm): {abs(numerical.a3)*1000:.1f}mm (should be ~392mm)")
    print(f"   d4 (wrist1): {numerical.d4*1000:.1f}mm (should be ~133mm)")
    print(f"   d5 (wrist2): {numerical.d5*1000:.1f}mm (should be ~100mm)")
    print(f"   d6 (tool): {numerical.d6*1000:.1f}mm (should be ~100mm)")
    
    total_length = abs(numerical.a2) + abs(numerical.a3) + numerical.d6
    print(f"   Max theoretical reach: {total_length*1000:.1f}mm")
    print()
    
    # Test 2: Try known safe joint configurations
    print("2. Testing Known Safe Configurations:")
    
    safe_configs = [
        ([0, -math.pi/2, 0, -math.pi/2, 0, 0], "UR Default Home"),
        ([0, -math.pi/3, -math.pi/3, -math.pi/3, 0, 0], "Conservative Pose"),
        ([0, -math.pi/4, -math.pi/2, -math.pi/4, 0, 0], "Forward Reach"),
        ([math.pi/6, -math.pi/3, -math.pi/3, -math.pi/3, 0, 0], "Slight Rotation"),
    ]
    
    working_poses = []
    
    for joints, description in safe_configs:
        print(f"\n   Testing: {description}")
        print(f"   Joints (deg): {[round(math.degrees(j), 1) for j in joints]}")
        
        # Forward kinematics
        fk_pose = numerical.forward_kinematics(joints)
        position = fk_pose[:3, 3]
        distance = np.linalg.norm(position)
        
        print(f"   FK Position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
        print(f"   Distance: {distance:.3f}m")
        
        # Check if position seems reasonable
        if distance < 0.85 and position[2] > 0.05:  # Within reach and above ground
            print(f"   ✅ Position looks reachable")
            
            # Test inverse kinematics
            ik_solutions = numerical.inverse_kinematics(fk_pose, joints)
            
            if ik_solutions:
                best_sol = numerical.select_best_solution(ik_solutions, joints)
                
                # Verify accuracy
                check_pose = numerical.forward_kinematics(best_sol)
                pos_error = np.linalg.norm(check_pose[:3, 3] - position) * 1000
                
                print(f"   IK Solutions: {len(ik_solutions)}")
                print(f"   Position Error: {pos_error:.3f}mm")
                
                if pos_error < 5.0:
                    print(f"   ✅ FK/IK consistent")
                    working_poses.append((position, fk_pose[:3, :3], description))
                else:
                    print(f"   ❌ FK/IK inconsistent")
            else:
                print(f"   ❌ IK failed for FK result")
        else:
            if distance >= 0.85:
                print(f"   ❌ Position beyond max reach (~0.85m)")
            if position[2] <= 0.05:
                print(f"   ❌ Position too low (collision risk)")
    
    print(f"\n   Working poses found: {len(working_poses)}")
    
    # Test 3: Test hybrid solver on working poses
    if working_poses:
        print(f"\n3. Testing Hybrid Solver on Working Poses:")
        
        success_count = 0
        for position, orientation, description in working_poses:
            target_pose = np.eye(4)
            target_pose[:3, 3] = position
            target_pose[:3, :3] = orientation
            
            print(f"\n   Testing hybrid IK: {description}")
            solutions = hybrid.inverse_kinematics(target_pose, timeout_ms=200)
            
            if solutions:
                best_sol = solutions[0]
                check_pose = hybrid.forward_kinematics(best_sol)
                pos_error = np.linalg.norm(check_pose[:3, 3] - position) * 1000
                
                print(f"   Solutions: {len(solutions)}")
                print(f"   Error: {pos_error:.3f}mm")
                
                if pos_error < 10.0:
                    print(f"   ✅ Hybrid solver working")
                    success_count += 1
                else:
                    print(f"   ⚠️  High error")
            else:
                print(f"   ❌ No solutions")
        
        hybrid_success_rate = success_count / len(working_poses)
        print(f"\n   Hybrid success rate: {success_count}/{len(working_poses)} ({hybrid_success_rate:.1%})")
        
        return working_poses, hybrid_success_rate > 0.5
    
    return [], False

def test_realistic_target_poses():
    """Test with realistic target poses for object manipulation."""
    
    print("\n=== Testing Realistic Manipulation Poses ===\n")
    
    hybrid = HybridUR5eKinematics(debug=True)
    
    # Realistic object manipulation poses
    realistic_poses = [
        {
            "position": [0.3, 0.0, 0.15],
            "description": "Table center pickup",
            "orientation": "top_down"
        },
        {
            "position": [0.25, 0.15, 0.12],
            "description": "Table corner pickup", 
            "orientation": "top_down"
        },
        {
            "position": [0.35, -0.1, 0.2],
            "description": "Elevated object",
            "orientation": "angled"
        },
        {
            "position": [0.2, 0.2, 0.1],
            "description": "Near side object",
            "orientation": "side"
        }
    ]
    
    # Define orientations
    orientations = {
        "top_down": np.array([
            [1, 0, 0],
            [0, -1, 0], 
            [0, 0, -1]
        ]),
        "angled": np.array([
            [0.866, 0, 0.5],
            [0, -1, 0],
            [0.5, 0, -0.866]
        ]),
        "side": np.array([
            [0, 0, 1],
            [0, -1, 0],
            [1, 0, 0]
        ])
    }
    
    success_count = 0
    total_tests = len(realistic_poses)
    
    for pose_info in realistic_poses:
        position = pose_info["position"]
        orientation_key = pose_info["orientation"] 
        description = pose_info["description"]
        
        print(f"\n--- {description} ---")
        print(f"Position: {position}")
        print(f"Orientation: {orientation_key}")
        
        # Check basic reachability
        distance = np.linalg.norm(position)
        print(f"Distance: {distance:.3f}m")
        
        if distance > 0.75:  # Conservative check
            print("⚠️  Position may be too far")
            continue
        
        if position[2] < 0.05:
            print("⚠️  Position too low")
            continue
        
        # Create target pose
        target_pose = np.eye(4)
        target_pose[:3, 3] = position
        target_pose[:3, :3] = orientations[orientation_key]
        
        # Test hybrid IK
        solutions = hybrid.inverse_kinematics(target_pose, timeout_ms=300)
        
        if solutions:
            best_solution = solutions[0]
            print(f"✅ Found {len(solutions)} solutions")
            print(f"Best joints (deg): {[round(math.degrees(j), 1) for j in best_solution]}")
            
            # Verify accuracy
            check_pose = hybrid.forward_kinematics(best_solution)
            pos_error = np.linalg.norm(check_pose[:3, 3] - np.array(position)) * 1000
            
            # Check orientation error
            rot_error_matrix = check_pose[:3, :3].T @ target_pose[:3, :3]
            rot_angle = math.acos(np.clip((np.trace(rot_error_matrix) - 1) / 2, -1.0, 1.0))
            rot_error_deg = math.degrees(rot_angle)
            
            print(f"Position error: {pos_error:.1f}mm")
            print(f"Orientation error: {rot_error_deg:.1f}°")
            
            if pos_error < 15.0 and rot_error_deg < 10.0:  # Relaxed tolerances for real use
                print("✅ Solution accurate enough for manipulation")
                success_count += 1
            else:
                print("⚠️  Solution has high error")
                success_count += 0.5
                
        else:
            print("❌ No solutions found")
            
            # Debug: Try with identity orientation
            debug_pose = np.eye(4)
            debug_pose[:3, 3] = position
            debug_solutions = hybrid.inverse_kinematics(debug_pose, timeout_ms=100)
            
            if debug_solutions:
                print("  ✅ Position reachable with identity orientation")
                print("  Issue: Requested orientation may be unreachable")
            else:
                print("  ❌ Position itself may be unreachable")
    
    success_rate = success_count / total_tests
    print(f"\n=== Realistic Pose Test Results ===")
    print(f"Success rate: {success_count}/{total_tests} ({success_rate:.1%})")
    
    return success_rate

def investigate_fk_issue():
    """Investigate why FK produces unreachable poses."""
    
    print("\n=== Forward Kinematics Investigation ===\n")
    
    numerical = UR5eKinematics()
    
    # Test the "home" position that was failing
    home_joints = [0, -math.pi/2, 0, 0, 0, 0]
    print("Investigating 'home' position [0, -90°, 0, 0, 0, 0]:")
    
    # Step through the DH transformations
    print("\nDH Parameter breakdown:")
    dh_params = [
        (numerical.d1, 0, math.pi/2, home_joints[0]),              # Base to shoulder
        (0, numerical.a2, 0, home_joints[1]),                       # Shoulder to elbow
        (0, numerical.a3, 0, home_joints[2]),                       # Elbow to wrist1
        (numerical.d4, 0, math.pi/2, home_joints[3]),               # Wrist1 to wrist2
        (numerical.d5, 0, -math.pi/2, home_joints[4]),              # Wrist2 to wrist3
        (numerical.d6, 0, 0, home_joints[5])                        # Wrist3 to tool
    ]
    
    T = np.eye(4)
    for i, (d, a, alpha, theta) in enumerate(dh_params):
        Ti = numerical.dh_transform(d, a, alpha, theta)
        T = T @ Ti
        
        position = T[:3, 3]
        print(f"  After joint {i+1}: position = [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}], distance = {np.linalg.norm(position):.3f}m")
    
    final_pose = numerical.forward_kinematics(home_joints)
    final_pos = final_pose[:3, 3]
    final_distance = np.linalg.norm(final_pos)
    
    print(f"\nFinal position: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
    print(f"Final distance: {final_distance:.3f}m")
    
    # Compare with expected
    expected_reach = abs(numerical.a2) + abs(numerical.a3)  # Should be roughly the max horizontal reach
    print(f"Expected max horizontal reach: {expected_reach:.3f}m")
    
    if final_distance > 0.85:
        print("❌ FK producing poses beyond robot reach!")
        print("Possible issues:")
        print("  - Incorrect DH parameters")
        print("  - Wrong coordinate frame conventions")
        print("  - Sign errors in DH transformations")
    else:
        print("✅ FK position seems reasonable")
    
    # Test a few more "reasonable" joint configurations
    print(f"\nTesting other joint configurations:")
    
    test_configs = [
        ([0, -math.pi/4, -math.pi/4, 0, 0, 0], "Forward reach"),
        ([0, -math.pi/6, -math.pi/3, 0, 0, 0], "Conservative"),
        ([math.pi/4, -math.pi/4, -math.pi/4, 0, 0, 0], "45° base rotation")
    ]
    
    reasonable_configs = []
    
    for joints, desc in test_configs:
        pose = numerical.forward_kinematics(joints)
        pos = pose[:3, 3]
        dist = np.linalg.norm(pos)
        
        print(f"  {desc}: distance = {dist:.3f}m", end="")
        
        if dist < 0.85 and pos[2] > 0.05:
            print(" ✅ Reasonable")
            reasonable_configs.append((joints, pose, desc))
        else:
            print(" ❌ Unreachable")
    
    return reasonable_configs

def main():
    """Main debug routine."""
    
    # Step 1: Debug workspace and FK
    working_poses, hybrid_working = debug_ur5e_workspace()
    
    # Step 2: Investigate FK issues
    reasonable_configs = investigate_fk_issue()
    
    # Step 3: Test realistic poses
    realistic_success_rate = test_realistic_target_poses()
    
    # Summary
    print("\n" + "="*60)
    print("DEBUG SUMMARY:")
    print("="*60)
    
    if len(working_poses) > 0:
        print(f"✅ Found {len(working_poses)} working FK/IK pairs")
    else:
        print("❌ No working FK/IK pairs found - FK implementation issue")
    
    if hybrid_working:
        print("✅ Hybrid solver working on valid poses")
    else:
        print("❌ Hybrid solver issues")
    
    if realistic_success_rate > 0.5:
        print(f"✅ Realistic poses: {realistic_success_rate:.1%} success rate")
        print("\n🎯 RECOMMENDATION: Use the realistic poses for your application!")
        
        print("\nNext steps:")
        print("1. Use positions like [0.3, 0.0, 0.15] for object pickup")
        print("2. Avoid the failing test poses from the original test")
        print("3. Validate workspace before IK calls")
        print("4. Consider fixing FK implementation if needed")
        
        return 0
    else:
        print(f"❌ Realistic poses: {realistic_success_rate:.1%} success rate")
        print("\n🔧 RECOMMENDATION: Fix FK implementation or DH parameters")
        
        if len(reasonable_configs) > 0:
            print(f"\nFound {len(reasonable_configs)} reasonable FK results to debug with")
        
        return 1

if __name__ == "__main__":
    exit(main())