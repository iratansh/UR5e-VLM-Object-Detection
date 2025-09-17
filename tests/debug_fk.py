#!/usr/bin/env python3
"""
Quick debug script to check forward kinematics output
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
from UR5eKinematics import UR5eKinematics

def test_fk():
    print("=== Forward Kinematics Debug ===")
    
    kin = UR5eKinematics()
    
    # Print DH parameters
    print(f"DH Parameters:")
    print(f"  d1 = {kin.d1:.4f}m")
    print(f"  a2 = {kin.a2:.4f}m")
    print(f"  a3 = {kin.a3:.4f}m") 
    print(f"  d4 = {kin.d4:.4f}m")
    print(f"  d5 = {kin.d5:.4f}m")
    print(f"  d6 = {kin.d6:.4f}m")
    
    # Test basic configurations
    test_configs = [
        ("Home", [0, -math.pi/2, 0, 0, 0, 0]),
        ("All zeros", [0, 0, 0, 0, 0, 0]),
        ("Conservative", [0, -math.pi/3, -math.pi/3, 0, 0, 0]),
    ]
    
    print("\nTesting configurations:")
    for name, joints in test_configs:
        try:
            pose = kin.forward_kinematics(joints)
            pos = pose[:3, 3]
            distance = np.linalg.norm(pos)
            
            print(f"\n{name}: {[math.degrees(j) for j in joints]}")
            print(f"  Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            print(f"  Distance: {distance:.3f}m")
            
            if distance > 0.85:
                print(f"  ❌ Beyond max reach!")
            else:
                print(f"  ✅ Within reach")
                
        except Exception as e:
            print(f"  ❌ FK Error: {e}")
    
    # Test if a known good position can be reached
    print("\n=== Testing IK on simple position ===")
    
    # Try a very simple, close position
    target_pos = np.array([0.3, 0.0, 0.3])  # 30cm forward, 30cm up
    target_pose = np.eye(4)
    target_pose[:3, 3] = target_pos
    
    print(f"Target: {target_pos} (distance: {np.linalg.norm(target_pos):.3f}m)")
    
    try:
        solutions = kin.inverse_kinematics(target_pose)
        print(f"IK solutions found: {len(solutions)}")
        
        if solutions:
            for i, sol in enumerate(solutions):
                # Verify with FK
                achieved_pose = kin.forward_kinematics(sol)
                achieved_pos = achieved_pose[:3, 3]
                error = np.linalg.norm(achieved_pos - target_pos)
                print(f"  Solution {i+1}: {[math.degrees(j) for j in sol]}")
                print(f"    Achieved: {achieved_pos}")
                print(f"    Error: {error*1000:.1f}mm")
        else:
            print("  ❌ No solutions found!")
            
    except Exception as e:
        print(f"  ❌ IK Error: {e}")

if __name__ == "__main__":
    test_fk()
