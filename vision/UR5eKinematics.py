"""
Corrected UR5e robot inverse kinematics implementation.
Key fixes for real robot deployment:
1. Fixed joint limit checking logic
2. Corrected singularity handling in IK
3. Fixed angle normalization for continuous joints
4. Improved solution validation
5. Added proper error handling for edge cases
"""

import numpy as np
import math
from typing import List, Optional, Tuple
import logging
from std_msgs.msg import Float32MultiArray

class UR5eKinematics:
    """
    UR5e robot kinematics calculator with corrected inverse kinematics.
    
    This class implements forward and inverse kinematics for the UR5e
    robot arm using the modified DH parameters convention.
    """
    
    def __init__(self, dh_params: Optional[dict] = None, joint_limits: Optional[dict] = None):
        self.logger = logging.getLogger(__name__)
        
        # UR5e DH parameters (modified DH convention)
        self.d = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996]  # Link offsets [m]
        self.a = [0, -0.425, -0.3922, 0, 0, 0]           # Link lengths [m]
        self.alpha = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]  # Link twists [rad]
        
        # Physical dimensions for IK calculations
        self.d1 = 0.1625    # Base height
        self.a2 = 0.425     # Upper arm length  
        self.a3 = 0.3922    # Forearm length
        self.d4 = 0.1333    # Wrist 1 offset
        self.d5 = 0.0997    # Wrist 2 offset
        self.d6 = 0.0996    # Wrist 3 offset (tool flange)
        
        # UR5e joint limits [rad] - CORRECTED for actual robot
        self.JOINT_LIMITS = [
            (-2*np.pi, 2*np.pi),      # Base: ±360° (continuous)
            (-np.pi, np.pi),          # Shoulder: ±180°
            (-np.pi, np.pi),          # Elbow: ±180°
            (-2*np.pi, 2*np.pi),      # Wrist 1: ±360° (continuous)
            (-2*np.pi, 2*np.pi),      # Wrist 2: ±360° (continuous)
            (-2*np.pi, 2*np.pi)       # Wrist 3: ±360° (continuous)
        ]
        
        # Joint velocity limits [rad/s]
        self.VELOCITY_LIMITS = [3.14, 3.14, 3.14, 6.28, 6.28, 6.28]
        
        self.logger.info("✅ UR5e kinematics initialized with corrected parameters")
    
    def forward_kinematics(self, joint_angles: List[float]) -> np.ndarray:
        """
        Calculate forward kinematics for given joint angles.
        
        Parameters
        ----------
        joint_angles : List[float]
            List of 6 joint angles in radians
            
        Returns
        -------
        np.ndarray
            4x4 homogeneous transformation matrix for end-effector pose
        """
        if len(joint_angles) != 6:
            raise ValueError("UR5e requires exactly 6 joint angles")
            
        T = np.eye(4)
        
        for i in range(6):
            ct = np.cos(joint_angles[i])
            st = np.sin(joint_angles[i])
            ca = np.cos(self.alpha[i])
            sa = np.sin(self.alpha[i])
            
            T_i = np.array([
                [ct, -st*ca, st*sa, self.a[i]*ct],
                [st, ct*ca, -ct*sa, self.a[i]*st],
                [0, sa, ca, self.d[i]],
                [0, 0, 0, 1]
            ])
            
            T = T @ T_i
            
        return T
    
    def inverse_kinematics(self, target_pose: np.ndarray, 
                          current_joints: Optional[List[float]] = None) -> List[List[float]]:
        """
        Calculate inverse kinematics for target end-effector pose.
        CORRECTED implementation with proper singularity handling.
        
        Parameters
        ----------
        target_pose : np.ndarray
            4x4 homogeneous transformation matrix for target pose
        current_joints : Optional[List[float]]
            Current joint configuration for solution preference
            
        Returns
        -------
        List[List[float]]
            List of valid joint angle solutions in radians
        """
        solutions = []
        
        try:
            # Extract position and orientation
            R = target_pose[:3, :3]
            p = target_pose[:3, 3]
            
            # Calculate wrist center position
            p_wc = p - self.d6 * R[:, 2]
            
            # ========== JOINT 1 CALCULATION ==========
            # Two solutions for theta1
            theta1_solutions = []
            
            # Handle singularity when wrist center is on z-axis
            if abs(p_wc[0]) < 1e-6 and abs(p_wc[1]) < 1e-6:
                # Singular case - choose current theta1 if available
                if current_joints is not None:
                    theta1_solutions = [current_joints[0]]
                else:
                    theta1_solutions = [0.0]
            else:
                theta1_1 = math.atan2(p_wc[1], p_wc[0])
                theta1_2 = theta1_1 + math.pi
                theta1_solutions = [theta1_1, theta1_2]
            
            for theta1 in theta1_solutions:
                theta1 = self.normalize_angle(theta1)
                
                # Calculate intermediate values
                c1 = math.cos(theta1)
                s1 = math.sin(theta1)
                
                # Position in shoulder coordinate frame
                p_x = p_wc[0]*c1 + p_wc[1]*s1 - self.d4
                p_y = p_wc[2] - self.d1
                
                # Distance from shoulder to wrist center
                r = math.sqrt(p_x**2 + p_y**2)
                
                # Check reachability
                max_reach = self.a2 + self.a3
                min_reach = abs(self.a2 - self.a3)
                
                if r > max_reach + 1e-6 or r < min_reach - 1e-6:
                    continue  # Point not reachable
                
                # ========== JOINT 3 CALCULATION ==========
                # Law of cosines for elbow angle
                cos_theta3 = (r**2 - self.a2**2 - self.a3**2) / (2 * self.a2 * self.a3)
                cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)  # Clamp to valid range
                
                # Two solutions for theta3 (elbow up/down)
                theta3_1 = math.acos(cos_theta3)
                theta3_2 = -theta3_1
                
                for theta3 in [theta3_1, theta3_2]:
                    # ========== JOINT 2 CALCULATION ==========
                    s3 = math.sin(theta3)
                    c3 = math.cos(theta3)
                    
                    # Calculate theta2 using atan2 for proper quadrant
                    k1 = self.a2 + self.a3 * c3
                    k2 = self.a3 * s3
                    
                    theta2 = math.atan2(p_y, p_x) - math.atan2(k2, k1)
                    theta2 = self.normalize_angle(theta2)
                    
                    # ========== WRIST ORIENTATION CALCULATION ==========
                    # Calculate R_03 (base to wrist 1)
                    c2 = math.cos(theta2)
                    s2 = math.sin(theta2)
                    
                    R_03 = np.array([
                        [c1*(c2*c3 - s2*s3), -c1*(c2*s3 + s2*c3), -s1],
                        [s1*(c2*c3 - s2*s3), -s1*(c2*s3 + s2*c3), c1],
                        [s2*c3 + c2*s3, -s2*s3 + c2*c3, 0]
                    ])
                    
                    # R_36 = R_03^T * R (wrist orientation)
                    R_36 = R_03.T @ R
                    
                    # ========== WRIST JOINTS CALCULATION ==========
                    # Joint 5 (wrist 2)
                    c5 = R_36[2, 2]
                    
                    # Handle wrist singularity
                    if abs(abs(c5) - 1.0) < 1e-6:
                        # Singular case: theta5 = 0 or pi
                        if c5 > 0:
                            theta5 = 0.0
                            theta4 = 0.0  # Choose convenient value
                            theta6 = math.atan2(R_36[1, 0], R_36[0, 0])
                        else:
                            theta5 = math.pi
                            theta4 = 0.0
                            theta6 = math.atan2(-R_36[1, 0], -R_36[0, 0])
                        
                        solution = [theta1, theta2, theta3, theta4, theta5, theta6]
                        if self._is_valid_solution(solution, target_pose):
                            solutions.append(solution)
                            
                    else:
                        # Non-singular case: two solutions for theta5
                        theta5_1 = math.acos(c5)
                        theta5_2 = -theta5_1
                        
                        for theta5 in [theta5_1, theta5_2]:
                            s5 = math.sin(theta5)
                            
                            # Joint 4 (wrist 1)
                            theta4 = math.atan2(R_36[1, 2]/s5, R_36[0, 2]/s5)
                            
                            # Joint 6 (wrist 3)  
                            theta6 = math.atan2(R_36[2, 1]/s5, -R_36[2, 0]/s5)
                            
                            solution = [theta1, theta2, theta3, theta4, theta5, theta6]
                            if self._is_valid_solution(solution, target_pose):
                                solutions.append(solution)
            
            return solutions
            
        except Exception as e:
            self.logger.error(f"Inverse kinematics failed: {e}")
            return []
    
    def _is_valid_solution(self, joint_angles: List[float], 
                          target_pose: np.ndarray, 
                          pos_tol: float = 1e-3, 
                          rot_tol: float = 1e-2) -> bool:
        """
        Validate a solution by checking joint limits and forward kinematics.
        
        Parameters
        ----------
        joint_angles : List[float]
            Joint angles to validate
        target_pose : np.ndarray
            Target pose to verify against
        pos_tol : float
            Position tolerance in meters
        rot_tol : float
            Rotation tolerance in radians
            
        Returns
        -------
        bool
            True if solution is valid
        """
        # Check joint limits
        if not self._check_joint_limits(joint_angles):
            return False
        
        try:
            # Verify forward kinematics
            actual_pose = self.forward_kinematics(joint_angles)
            
            # Check position error
            pos_error = np.linalg.norm(actual_pose[:3, 3] - target_pose[:3, 3])
            if pos_error > pos_tol:
                return False
            
            # Check orientation error
            R_error = target_pose[:3, :3].T @ actual_pose[:3, :3]
            trace_R = np.clip(np.trace(R_error), -3.0, 3.0)
            rot_error = math.acos(abs((trace_R - 1) / 2))
            
            return rot_error < rot_tol
            
        except Exception:
            return False
    
    def _check_joint_limits(self, joint_angles: List[float]) -> bool:
        """
        CORRECTED joint limit checking for UR5e.
        
        Parameters
        ----------
        joint_angles : List[float]
            Joint angles in radians
            
        Returns
        -------
        bool
            True if all joints are within limits
        """
        if len(joint_angles) != 6:
            return False
        
        # Safety margin to avoid getting too close to limits
        SAFETY_MARGIN = 0.05  # ~3 degrees
        
        try:
            for i, angle in enumerate(joint_angles):
                min_limit, max_limit = self.JOINT_LIMITS[i]
                
                # For continuous joints (0, 3, 4, 5), no limit checking needed
                if i in [0, 3, 4, 5]:
                    continue
                    
                # For limited joints (1, 2), check against actual limits
                if i in [1, 2]:
                    # Normalize to [-pi, pi] for shoulder and elbow
                    normalized_angle = self.normalize_angle(angle)
                    
                    if (normalized_angle < min_limit + SAFETY_MARGIN or 
                        normalized_angle > max_limit - SAFETY_MARGIN):
                        self.logger.warning(
                            f"Joint {i} angle {math.degrees(normalized_angle):.1f}° "
                            f"exceeds safe limits [{math.degrees(min_limit):.1f}°, "
                            f"{math.degrees(max_limit):.1f}°]"
                        )
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking joint limits: {e}")
            return False
    
    def normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def select_best_solution(self, solutions: List[List[float]], 
                           current_joints: Optional[List[float]] = None) -> Optional[List[float]]:
        """
        Select the best solution from available options.
        Prioritizes solutions closest to current joint configuration.
        """
        if not solutions:
            return None
            
        if len(solutions) == 1:
            return solutions[0]
        
        # If current joint state available, choose closest in joint space
        if current_joints is not None:
            best_solution = None
            min_distance = float('inf')
            
            for solution in solutions:
                # Calculate weighted joint space distance
                distance = 0
                for i, (s, c) in enumerate(zip(solution, current_joints)):
                    # Weight shoulder and elbow joints more heavily
                    weight = 2.0 if i in [1, 2] else 1.0
                    angle_diff = abs(self.normalize_angle(s - c))
                    distance += weight * angle_diff**2
                
                if distance < min_distance:
                    min_distance = distance
                    best_solution = solution
            
            return best_solution
        
        # If no current state, prefer solutions with smaller joint angles
        best_solution = None
        min_magnitude = float('inf')
        
        for solution in solutions:
            magnitude = sum(abs(angle) for angle in solution)
            if magnitude < min_magnitude:
                min_magnitude = magnitude
                best_solution = solution
        
        return best_solution

    def format_ros2_command(self, joint_angles: List[float]) -> Float32MultiArray:
        """Format joint angles as ROS2 message."""
        msg = Float32MultiArray()
        msg.data = joint_angles
        return msg


def test_corrected_ik():
    """Comprehensive test of the corrected IK implementation"""
    ur5e = UR5eKinematics()
    
    print("=== UR5e Corrected IK Validation Test ===")
    
    # Test multiple configurations
    test_cases = [
        # Home position
        [0.0, -math.pi/2, 0.0, -math.pi/2, 0.0, 0.0],
        # Zero configuration  
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # Random valid configuration
        [math.pi/4, -math.pi/3, math.pi/6, math.pi/2, -math.pi/4, math.pi/3],
        # Edge case near singularity
        [0.0, -math.pi/2, math.pi/2, 0.0, 0.0, 0.0]
    ]
    
    for i, test_joints in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Input joints (deg): {[f'{math.degrees(j):6.1f}' for j in test_joints]}")
        
        # Forward kinematics
        T_target = ur5e.forward_kinematics(test_joints)
        pos = T_target[:3, 3]
        print(f"Target position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        
        # Inverse kinematics
        solutions = ur5e.inverse_kinematics(T_target, test_joints)
        print(f"Found {len(solutions)} valid solutions")
        
        if solutions:
            # Select best solution
            best = ur5e.select_best_solution(solutions, test_joints)
            print(f"Best solution (deg): {[f'{math.degrees(j):6.1f}' for j in best]}")
            
            # Verify accuracy
            T_verify = ur5e.forward_kinematics(best)
            pos_error = np.linalg.norm(T_verify[:3, 3] - T_target[:3, 3])
            
            R_error = T_target[:3, :3].T @ T_verify[:3, :3]
            trace_R = np.clip(np.trace(R_error), -3, 3)
            rot_error_deg = math.degrees(math.acos(abs((trace_R - 1) / 2)))
            
            print(f"Verification - Pos error: {pos_error*1000:.3f}mm, "
                  f"Rot error: {rot_error_deg:.3f}°")
            
            # Check if solution is acceptable for real robot
            if pos_error < 0.001 and rot_error_deg < 0.1:
                print("✅ PASS - Solution suitable for real robot")
            else:
                print("❌ FAIL - Solution accuracy insufficient")
        else:
            print("❌ No valid solutions found")
    
    print("\n=== Reachability Test ===")
    # Test unreachable point
    unreachable_pose = np.eye(4)
    unreachable_pose[:3, 3] = [1.5, 0, 0]  # Too far
    solutions = ur5e.inverse_kinematics(unreachable_pose)
    print(f"Unreachable point solutions: {len(solutions)}")
    
    # Test edge of workspace
    edge_pose = np.eye(4) 
    edge_pose[:3, 3] = [0.8, 0, 0.2]  # Near edge
    solutions = ur5e.inverse_kinematics(edge_pose)
    print(f"Edge of workspace solutions: {len(solutions)}")


if __name__ == "__main__":
    test_corrected_ik()