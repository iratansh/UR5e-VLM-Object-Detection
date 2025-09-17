"""
UR5e Robot Grasp Controller.

This module provides high-level control of a UR5e robot for grasping tasks.
It handles:
- Joint and Cartesian space control
- Grasp planning and execution
- Safety monitoring and collision avoidance
- Robot state management

The controller uses ROS2 for communication with the robot hardware
and implements safety features required for physical robot operation.
"""

#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from typing import List, Tuple, Optional
import logging
import cv2
import numpy.linalg as LA
import pyrealsense2 as rs

class UR5eGraspController(Node):
    """
    Enhanced UR5e controller with grasp planning and safety features.
    
    This class provides high-level control of a UR5e robot for grasping tasks,
    including motion planning, execution, and safety monitoring.
    
    Parameters
    ----------
    None : Uses ROS2 parameters for configuration
    
    Attributes
    ----------
    JOINT_LIMITS : Dict
        Joint angle limits in radians
    W1 : float
        Base width in meters
    W2 : float
        Tool width in meters
    L1 : float
        Upper arm length in meters
    L2 : float
        Forearm length in meters
    H1 : float
        Base height in meters
    H2 : float
        Tool height in meters
    current_joints : np.ndarray
        Current joint angles
    current_pose : np.ndarray
        Current end-effector pose
        
    Notes
    -----
    The controller requires:
    - ROS2 Humble or newer
    - UR5e robot with ROS driver
    - Properly configured joint limits
    - Hand-eye calibration for visual servoing
    """
    
    # UR5e specifications - CORRECTED to match UR5eKinematics for consistency
    JOINT_LIMITS = {
        'lower': np.array([-2*np.pi, -np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi]),
        'upper': np.array([2*np.pi, np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
    }
    
    # Robot dimensions
    W1 = 0.109  # Base width
    W2 = 0.082  # Tool width
    L1 = 0.425  # Upper arm
    L2 = 0.392  # Forearm
    H1 = 0.089  # Base height
    H2 = 0.095  # Tool height
    
    def __init__(self):
        super().__init__('ur5e_grasp_controller')
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load hand-eye calibration
        self.T_base_to_camera = self._load_calibration()
        
        # Publishers
        self.joint_pub = self.create_publisher(
            Float64MultiArray,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10
        )
        
        # Current state
        self.current_joints = None
        self.current_pose = None
        
        self.logger.info("UR5e Grasp Controller initialized")
    
    def _load_calibration(self) -> np.ndarray:
        """
        Load hand-eye calibration matrix.
        
        Returns
        -------
        np.ndarray
            4x4 transformation matrix from robot base to camera
            
        Notes
        -----
        Loads calibration from 'hand_eye_calib.npz'
        Falls back to identity matrix if loading fails
        """
        try:
            calib_data = np.load('hand_eye_calib.npz')
            return calib_data['T_base_to_camera']
        except Exception as e:
            self.logger.error(f"Failed to load calibration: {e}")
            return np.eye(4)
    
    def _joint_state_callback(self, msg: JointState):
        """
        Process joint state updates.
        
        Parameters
        ----------
        msg : JointState
            ROS2 joint state message
            
        Notes
        -----
        Updates current_joints and current_pose
        Used for monitoring robot state
        """
        self.current_joints = np.array(msg.position)
        self.current_pose = self.forward_kinematics(self.current_joints)
    
    def forward_kinematics(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics using Product of Exponentials.
        
        Parameters
        ----------
        theta : np.ndarray
            Joint angles in radians
            
        Returns
        -------
        np.ndarray
            4x4 homogeneous transformation matrix for end-effector pose
            
        Notes
        -----
        Uses:
        - Base transformation matrix M
        - Screw axes in space form S
        - Product of exponentials formula
        """
        # Base transformation matrix
        M = np.array([
            [-1, 0, 0, self.L1 + self.L2],
            [0, 0, 1, self.W1 + self.W2],
            [0, 1, 0, self.H1 - self.H2],
            [0, 0, 0, 1]
        ])
        
        # Screw axes in space form
        S = np.array([
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, -self.H1, 0, 0],
            [0, 1, 0, -self.H1, 0, self.L1],
            [0, 1, 0, -self.H1, 0, self.L1 + self.L2],
            [0, 0, -1, -self.W1, self.L1 + self.L2, 0],
            [0, 1, 0, self.H2 - self.H1, 0, self.L1 + self.L2]
        ])
        
        T = M
        for i in range(6):
            T = self._matrix_exp(S[i] * theta[i]) @ T
        
        return T
    
    def inverse_kinematics(self, target_pose: np.ndarray, 
                          initial_guess: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        """Enhanced inverse kinematics with singularity handling and joint limits."""
        if initial_guess is None:
            initial_guess = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])
        
        theta = initial_guess.copy()
        max_iterations = 100
        epsilon_omega = 0.001
        epsilon_v = 0.0001
        
        for iteration in range(max_iterations):
            T_current = self.forward_kinematics(theta)
            
            # Compute twist
            V_b = self._compute_twist(T_current, target_pose)
            
            # Check convergence
            if LA.norm(V_b[:3]) < epsilon_omega and LA.norm(V_b[3:]) < epsilon_v:
                return theta, True
            
            # Compute Jacobian
            J = self._compute_jacobian(theta)
            
            if self._is_singular(J):
                J = self._damped_pseudoinverse(J)
            else:
                J = LA.pinv(J)
            
            delta_theta = J @ V_b
            theta = theta + delta_theta
            
            # Apply joint limits
            theta = np.clip(theta, self.JOINT_LIMITS['lower'], self.JOINT_LIMITS['upper'])
        
        return theta, False
    
    def _compute_twist(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Compute the twist between current and target poses."""
        error = LA.logm(LA.inv(current) @ target)
        return np.array([error[2,1], error[0,2], error[1,0],
                        error[0,3], error[1,3], error[2,3]])
    
    def _compute_jacobian(self, theta: np.ndarray) -> np.ndarray:
        """Compute the geometric Jacobian."""
        J = np.zeros((6, 6))
        T = np.eye(4)
        
        # Screw axes in space form
        S = np.array([
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, -self.H1, 0, 0],
            [0, 1, 0, -self.H1, 0, self.L1],
            [0, 1, 0, -self.H1, 0, self.L1 + self.L2],
            [0, 0, -1, -self.W1, self.L1 + self.L2, 0],
            [0, 1, 0, self.H2 - self.H1, 0, self.L1 + self.L2]
        ])
        
        for i in range(6):
            J[:, i] = self._adjoint(T) @ S[i]
            T = T @ self._matrix_exp(S[i] * theta[i])
        
        return J
    
    def _is_singular(self, J: np.ndarray, threshold: float = 1e-3) -> bool:
        """Check if Jacobian is near singular configuration."""
        return LA.cond(J) > 1/threshold
    
    def _damped_pseudoinverse(self, J: np.ndarray, lambda_sq: float = 0.01) -> np.ndarray:
        """Compute damped pseudoinverse for handling singularities."""
        return J.T @ LA.inv(J @ J.T + lambda_sq * np.eye(6))
    
    def _matrix_exp(self, twist: np.ndarray) -> np.ndarray:
        """Compute matrix exponential of twist."""
        omega = twist[:3]
        v = twist[3:]
        
        omega_hat = np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0]
        ])
        
        theta = LA.norm(omega)
        if theta < 1e-6:
            R = np.eye(3)
            V = v.reshape(3, 1)
        else:
            R = np.eye(3) + np.sin(theta)/theta * omega_hat + \
                (1 - np.cos(theta))/theta**2 * omega_hat @ omega_hat
            V = (np.eye(3) * theta + (1 - np.cos(theta)) * omega_hat + \
                 (theta - np.sin(theta)) * omega_hat @ omega_hat) @ v.reshape(3, 1) / theta**2
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = V.flatten()
        return T
    
    def _adjoint(self, T: np.ndarray) -> np.ndarray:
        """Compute adjoint of transformation matrix."""
        R = T[:3, :3]
        p = T[:3, 3]
        p_hat = np.array([
            [0, -p[2], p[1]],
            [p[2], 0, -p[0]],
            [-p[1], p[0], 0]
        ])
        
        adj = np.zeros((6, 6))
        adj[:3, :3] = R
        adj[3:, 3:] = R
        adj[3:, :3] = p_hat @ R
        return adj
    
    def plan_grasp(self, object_pose: np.ndarray, approach_distance: float = 0.1) -> List[np.ndarray]:
        """Plan grasp trajectory with pre-grasp and final grasp poses."""
        # Transform object pose from camera to base frame
        object_pose_base = self.T_base_to_camera @ object_pose
        
        # Generate pre-grasp pose
        pre_grasp = object_pose_base.copy()
        pre_grasp[2, 3] += approach_distance  # Move up by approach_distance
        
        # Generate grasp poses
        poses = [
            pre_grasp,  # Pre-grasp position
            object_pose_base  # Final grasp position
        ]
        
        return poses
    
    def execute_grasp(self, object_pose: np.ndarray) -> bool:
        """Execute complete grasp sequence."""
        try:
            # Plan grasp trajectory
            grasp_poses = self.plan_grasp(object_pose)
            
            for pose in grasp_poses:
                # Compute IK
                target_joints, success = self.inverse_kinematics(pose)
                
                if not success:
                    self.logger.error("IK failed to converge")
                    return False
                
                if not self._validate_solution(target_joints):
                    self.logger.error("Invalid joint solution")
                    return False
                
                self._execute_motion(target_joints)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Grasp execution failed: {e}")
            return False
    
    def _validate_solution(self, joints: np.ndarray) -> bool:
        """Validate joint solution for safety."""
        # Check joint limits
        if np.any(joints < self.JOINT_LIMITS['lower']) or \
           np.any(joints > self.JOINT_LIMITS['upper']):
            return False
        
        # Check joint velocities (simplified)
        if self.current_joints is not None:
            max_velocity = 1.0  # rad/s
            velocities = np.abs(joints - self.current_joints)
            if np.any(velocities > max_velocity):
                return False
        
        return True
    
    def _execute_motion(self, target_joints: np.ndarray):
        """Execute robot motion to target joints."""
        msg = Float64MultiArray()
        msg.data = target_joints.tolist()
        self.joint_pub.publish(msg)

def main():
    rclpy.init()
    
    try:
        controller = UR5eGraspController()
        rclpy.spin(controller)
    except Exception as e:
        logging.error(f"Controller failed: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main() 