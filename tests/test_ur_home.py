#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
from UR5eKinematics import UR5eKinematics

def test_ur_home_position():
    """
    Test what the UR5e "home" position should produce.
    
    According to UR documentation, the robot in home position should have
    the end-effector roughly at the front of the workspace at a reasonable distance.
    """
    
    print("=== UR5e Home Position Test ===")
    
    kinematics = UR5eKinematics()
    
    # UR5e home position is typically [0, -90°, 0, -90°, 0, 0]
    # This should put the end-effector pointing forward at about 0.8m reach
    home_joints_deg = [0, -90, 0, -90, 0, 0]
    home_joints_rad = [math.radians(angle) for angle in home_joints_deg]
    
    print(f"Home joints (deg): {home_joints_deg}")
    print(f"Home joints (rad): {[round(j, 4) for j in home_joints_rad]}")
    
    # Calculate FK
    pose = kinematics.forward_kinematics(home_joints_rad)
    position = pose[:3, 3]
    
    print(f"FK result position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
    print(f"Distance from base: {np.linalg.norm(position):.3f}m")
    
    # Expected: Should be around [0.8, 0, 0.2] with distance ~0.82m
    expected_distance_range = (0.7, 0.9)  # Reasonable range for UR5e
    
    actual_distance = np.linalg.norm(position)
    if expected_distance_range[0] <= actual_distance <= expected_distance_range[1]:
        print("✅ Distance is in expected range!")
    else:
        print(f"❌ Distance {actual_distance:.3f}m is outside expected range {expected_distance_range}")
        print("This suggests an issue with the DH parameters or joint conventions")
    
    # Also test the "all joints at -90°" configuration which should be compact
    print(f"\n=== Testing joints all at -90° ===")
    test_joints = [math.radians(-90)] * 6
    pose2 = kinematics.forward_kinematics(test_joints)
    position2 = pose2[:3, 3]
    distance2 = np.linalg.norm(position2)
    
    print(f"All -90° position: [{position2[0]:.3f}, {position2[1]:.3f}, {position2[2]:.3f}]")
    print(f"Distance: {distance2:.3f}m")
    
    # Test some other common configurations
    print(f"\n=== Testing other configurations ===")
    test_configs = [
        ([0, 0, 0, 0, 0, 0], "All zeros"),
        ([0, -45, -45, 0, 0, 0], "Arm extended forward"),
        ([90, -90, 0, -90, 0, 0], "90° rotated base"),
    ]
    
    for joints_deg, name in test_configs:
        joints_rad = [math.radians(j) for j in joints_deg]
        pose = kinematics.forward_kinematics(joints_rad)
        pos = pose[:3, 3]
        dist = np.linalg.norm(pos)
        print(f"{name}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], dist={dist:.3f}m")

if __name__ == "__main__":
    test_ur_home_position()
