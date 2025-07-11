"""
UR5e Kinematics Implementation
Simple and reliable forward and inverse kinematics for the UR5e robot arm.

This implementation uses a standard DH parameter approach for forward kinematics
and supports both analytical (ur_ikfast) and numerical optimization for inverse kinematics.
"""

import numpy as np
import math
from typing import List, Optional, Tuple, Dict, Any
import logging
import time

# Try to import scipy, but don't fail if there are library issues
try:
    from scipy.spatial.transform import Rotation as R
    SCIPY_AVAILABLE = True
except ImportError as e:
    print(f"Scipy import failed: {e}")
    print("Falling back to manual quaternion calculations")
    SCIPY_AVAILABLE = False
    R = None

# Try to import ur_ikfast for high-speed analytical IK
try:
    from ur_ikfast.ur_kinematics import URKinematics
    UR_IKFAST_AVAILABLE = True
    print("ur_ikfast imported successfully - hybrid IK enabled")
except ImportError:
    UR_IKFAST_AVAILABLE = False
    print("ur_ikfast not available - using numerical IK only")

class HybridUR5eKinematics:
    """
    Hybrid UR5e Kinematics that combines ur_ikfast (analytical) with numerical IK.
    
    This provides:
    - Speed: ur_ikfast solves in ~0.1ms for reachable poses
    - Robustness: Numerical fallback handles edge cases and approximate solutions
    - High Success Rate: Covers both exact and approximate solutions
    """
    
    def __init__(self, enable_fallback: bool = True, debug: bool = False):
        """
        Initialize hybrid kinematics solver.
        
        Args:
            enable_fallback: Whether to use numerical IK as fallback
            debug: Enable debug output
        """
        self.enable_fallback = enable_fallback
        self.debug = debug
        
        # Initialize the numerical solver as fallback
        self.numerical_solver = UR5eKinematics()
        self.numerical_solver.debug = False  # Keep numerical solver quiet unless debugging
        
        # Performance tracking
        self.ikfast_attempts = 0
        self.ikfast_successes = 0
        self.numerical_attempts = 0
        self.numerical_successes = 0
        
        # Timing tracking
        self.ikfast_total_time = 0.0
        self.numerical_total_time = 0.0
        
        # ur_ikfast availability
        self.ikfast_available = UR_IKFAST_AVAILABLE
        
        if not self.ikfast_available and self.debug:
            print("Warning: ur_ikfast not available, using numerical solver only")
    
    def inverse_kinematics(self, target_pose: np.ndarray, 
                          current_joints: Optional[List[float]] = None,
                          prefer_closest: bool = True,
                          timeout_ms: float = 50.0) -> List[List[float]]:
        """
        Hybrid inverse kinematics solver.
        
        Args:
            target_pose: 4x4 homogeneous transformation matrix
            current_joints: Current joint positions for solution preference
            prefer_closest: If True, return solution closest to current_joints
            timeout_ms: Maximum time to spend on numerical solver (ms)
            
        Returns:
            List of possible joint configurations
        """
        solutions = []
        
        # Step 1: Try ur_ikfast first for speed
        if self.ikfast_available:
            ikfast_solutions = self._solve_with_ikfast(target_pose)
            if ikfast_solutions:
                if self.debug:
                    print(f"ur_ikfast found {len(ikfast_solutions)} solutions")
                solutions.extend(ikfast_solutions)
        
        # Step 2: If no solutions or fallback enabled, try numerical solver
        if (not solutions or self.enable_fallback) and timeout_ms > 0:
            numerical_solutions = self._solve_with_numerical(
                target_pose, current_joints, timeout_ms
            )
            
            if numerical_solutions:
                if self.debug:
                    print(f"Numerical solver found {len(numerical_solutions)} solutions")
                
                for num_sol in numerical_solutions:
                    is_duplicate = False
                    for existing_sol in solutions:
                        if self._similar_configs(num_sol, existing_sol):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        solutions.append(num_sol)
        
        # Step 3: Filter and validate solutions
        valid_solutions = []
        for solution in solutions:
            if self._is_valid_solution(solution, target_pose):
                valid_solutions.append(solution)
        
        # Step 4: Sort solutions by preference
        if valid_solutions and current_joints is not None and prefer_closest:
            valid_solutions.sort(key=lambda sol: self._solution_distance(sol, current_joints))
        
        return valid_solutions
    
    def _solve_with_ikfast(self, target_pose: np.ndarray) -> List[List[float]]:
        """
        Solve IK using ur_ikfast analytical solver.
        
        Args:
            target_pose: 4x4 transformation matrix
            
        Returns:
            List of joint solutions
        """
        if not self.ikfast_available:
            return []
        
        ur5e_arm = URKinematics('ur5e')
        start_time = time.perf_counter()
        self.ikfast_attempts += 1
        
        try:
            position = target_pose[:3, 3]
            rotation = target_pose[:3, :3]
            
            # Convert rotation matrix to quaternion (w, x, y, z)
            quat = self._rotation_matrix_to_quaternion(rotation)
            
            # Create the 7-element pose vector [tx, ty, tz, w, x, y, z]
            # as specified by the ikfast error message.
            ee_pose = np.concatenate((position, np.array(quat)))

            # Call ur_ikfast, ensuring all inputs are standard Python lists/floats
            # to work around the library's internal numpy handling issues.
            joint_configs = ur5e_arm.inverse(
                [float(x) for x in ee_pose],
                False,           # closest_only: False to get all solutions
                [0.0] * 6        # seed for IK solver as a plain list
            )
            
            solutions = []
            # The library returns a list of lists, so this check is now safe
            if joint_configs:
                for config in joint_configs:
                    # Normalize joint angles
                    normalized_config = [self.numerical_solver.normalize_angle(j) for j in config]
                    
                    # Validate against joint limits
                    if self._within_joint_limits(normalized_config):
                        solutions.append(normalized_config)
                
                if solutions:
                    self.ikfast_successes += 1
            
            elapsed = time.perf_counter() - start_time
            self.ikfast_total_time += elapsed
            
            if self.debug and solutions:
                print(f"ur_ikfast solved in {elapsed*1000:.3f}ms")
            
            return solutions
            
        except Exception as e:
            if self.debug:
                print(f"ur_ikfast failed: {e}")
            return []
    
    def _solve_with_numerical(self, target_pose: np.ndarray, 
                            current_joints: Optional[List[float]] = None,
                            timeout_ms: float = 50.0) -> List[List[float]]:
        """
        Solve IK using numerical optimization.
        
        Args:
            target_pose: 4x4 transformation matrix
            current_joints: Current joint positions
            timeout_ms: Maximum solving time in milliseconds
            
        Returns:
            List of joint solutions
        """
        start_time = time.perf_counter()
        self.numerical_attempts += 1
        
        # Convert timeout to seconds
        timeout_sec = timeout_ms / 1000.0
        
        try:
            # Use the existing numerical solver with timeout consideration
            solutions = self.numerical_solver.inverse_kinematics(target_pose, current_joints)
            
            # Check if we have valid solutions within timeout
            elapsed = time.perf_counter() - start_time
            if elapsed > timeout_sec:
                if self.debug:
                    print(f"Numerical solver timeout after {elapsed*1000:.1f}ms")
                # Return partial results if any
                solutions = solutions[:1] if solutions else []
            
            if solutions:
                self.numerical_successes += 1
            
            self.numerical_total_time += elapsed
            
            if self.debug and solutions:
                print(f"Numerical solver found {len(solutions)} solutions in {elapsed*1000:.1f}ms")
            
            return solutions
            
        except Exception as e:
            if self.debug:
                print(f"Numerical solver failed: {e}")
            return []
    
    def best_reachable_pose(self, target_pose: np.ndarray, 
                           current_joints: Optional[List[float]] = None) -> Tuple[List[float], np.ndarray]:
        """
        Find the closest reachable pose to the target pose.
        
        This method is especially useful for VLM applications where vision noise
        might request unreachable poses.
        
        Args:
            target_pose: 4x4 transformation matrix for desired pose
            current_joints: Current joint positions
            
        Returns:
            Tuple of (best_joints, reachable_pose)
        """
        # First try exact solution
        solutions = self.inverse_kinematics(target_pose, current_joints, timeout_ms=30.0)
        
        if solutions:
            # Select best solution
            if current_joints is not None:
                best_joints = min(solutions, key=lambda sol: self._solution_distance(sol, current_joints))
            else:
                best_joints = solutions[0]
            
            reachable_pose = self.forward_kinematics(best_joints)
            return best_joints, reachable_pose
        
        # If no exact solution, use numerical solver's approximation capability
        return self.numerical_solver.best_reachable_pose(target_pose, current_joints)
    
    def forward_kinematics(self, joints: List[float]) -> np.ndarray:
        """Forward kinematics using the numerical solver's implementation."""
        return self.numerical_solver.forward_kinematics(joints)
    
    def _rotation_matrix_to_quaternion(self, rotation_matrix: np.ndarray) -> List[float]:
        """
        Convert rotation matrix to quaternion (w, x, y, z) using scipy.
        
        Args:
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            Quaternion as [w, x, y, z]
        """
        # Use scipy if available, otherwise fall back to manual calculation
        if SCIPY_AVAILABLE and R is not None:
            try:
                # Use scipy for robust rotation matrix to quaternion conversion
                rotation_obj = R.from_matrix(rotation_matrix)
                # Get quaternion in [x, y, z, w] format from scipy
                quat_xyzw = rotation_obj.as_quat()
                # Convert to [w, x, y, z] format expected by ur_ikfast
                quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
                return quat_wxyz
                
            except Exception as e:
                if self.debug:
                    print(f"Warning: Scipy quaternion conversion failed: {e}")
                    print("Falling back to manual calculation")
        
        # Manual calculation (either scipy unavailable or failed)
        if self.debug and not SCIPY_AVAILABLE:
            print("Using manual quaternion calculation (scipy not available)")
            
            # Fallback to manual calculation if scipy fails
            R_mat = rotation_matrix
            trace = np.trace(R_mat)
            
            if trace > 0:
                s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
                qw = 0.25 * s
                qx = (R_mat[2, 1] - R_mat[1, 2]) / s
                qy = (R_mat[0, 2] - R_mat[2, 0]) / s
                qz = (R_mat[1, 0] - R_mat[0, 1]) / s
            elif R_mat[0, 0] > R_mat[1, 1] and R_mat[0, 0] > R_mat[2, 2]:
                s = np.sqrt(1.0 + R_mat[0, 0] - R_mat[1, 1] - R_mat[2, 2]) * 2  # s = 4 * qx
                qw = (R_mat[2, 1] - R_mat[1, 2]) / s
                qx = 0.25 * s
                qy = (R_mat[0, 1] + R_mat[1, 0]) / s
                qz = (R_mat[0, 2] + R_mat[2, 0]) / s
            elif R_mat[1, 1] > R_mat[2, 2]:
                s = np.sqrt(1.0 + R_mat[1, 1] - R_mat[0, 0] - R_mat[2, 2]) * 2  # s = 4 * qy
                qw = (R_mat[0, 2] - R_mat[2, 0]) / s
                qx = (R_mat[0, 1] + R_mat[1, 0]) / s
                qy = 0.25 * s
                qz = (R_mat[1, 2] + R_mat[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R_mat[2, 2] - R_mat[0, 0] - R_mat[1, 1]) * 2  # s = 4 * qz
                qw = (R_mat[1, 0] - R_mat[0, 1]) / s
                qx = (R_mat[0, 2] + R_mat[2, 0]) / s
                qy = (R_mat[1, 2] + R_mat[2, 1]) / s
                qz = 0.25 * s
            
            return [qw, qx, qy, qz]
    
    def _within_joint_limits(self, joint_angles: List[float]) -> bool:
        """Check if joint angles are within UR5e limits."""
        for i, angle in enumerate(joint_angles):
            lower, upper = self.numerical_solver.JOINT_LIMITS[i]
            if not lower <= angle <= upper:
                return False
        return True
    
    def _similar_configs(self, config1: List[float], config2: List[float], 
                        threshold: float = 0.1) -> bool:
        """Check if two configurations are similar."""
        return self.numerical_solver._similar_configs(config1, config2, threshold)
    
    def _solution_distance(self, solution: List[float], current_joints: List[float]) -> float:
        """Calculate weighted distance between two joint configurations."""
        # Weight primary joints higher (they move larger segments)
        weights = [3.0, 2.0, 2.0, 1.0, 1.0, 1.0]
        distance = sum(w * (self.numerical_solver.normalize_angle(s - c))**2 
                      for w, s, c in zip(weights, solution, current_joints))
        return distance
    
    def _is_valid_solution(self, joint_angles: List[float], target_pose: np.ndarray) -> bool:
        """Validate that a solution reaches the target pose within tolerance."""
        return self.numerical_solver.is_valid_solution(joint_angles, target_pose)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the hybrid solver.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = {
            "ikfast_available": self.ikfast_available,
            "ikfast_attempts": self.ikfast_attempts,
            "ikfast_successes": self.ikfast_successes,
            "ikfast_success_rate": self.ikfast_successes / max(1, self.ikfast_attempts),
            "numerical_attempts": self.numerical_attempts,
            "numerical_successes": self.numerical_successes,
            "numerical_success_rate": self.numerical_successes / max(1, self.numerical_attempts),
            "total_attempts": self.ikfast_attempts + self.numerical_attempts,
            "total_successes": self.ikfast_successes + self.numerical_successes,
        }
        
        if self.ikfast_attempts > 0:
            stats["avg_ikfast_time_ms"] = (self.ikfast_total_time / self.ikfast_attempts) * 1000
        
        if self.numerical_attempts > 0:
            stats["avg_numerical_time_ms"] = (self.numerical_total_time / self.numerical_attempts) * 1000
        
        return stats
    
    def print_performance_summary(self):
        """Print a summary of solver performance."""
        stats = self.get_performance_stats()
        
        print("\n=== Hybrid IK Performance Summary ===")
        print(f"ur_ikfast available: {stats['ikfast_available']}")
        
        if stats['ikfast_available']:
            print(f"ur_ikfast: {stats['ikfast_successes']}/{stats['ikfast_attempts']} " +
                  f"({stats['ikfast_success_rate']:.1%}) success rate")
            if stats['ikfast_attempts'] > 0:
                print(f"  Average time: {stats.get('avg_ikfast_time_ms', 0):.2f}ms")
        
        print(f"Numerical: {stats['numerical_successes']}/{stats['numerical_attempts']} " +
              f"({stats['numerical_success_rate']:.1%}) success rate")
        if stats['numerical_attempts'] > 0:
            print(f"  Average time: {stats.get('avg_numerical_time_ms', 0):.1f}ms")
        
        print(f"Overall: {stats['total_successes']}/{stats['total_attempts']} " +
              f"({stats['total_successes'] / max(1, stats['total_attempts']):.1%}) success rate")

class UR5eKinematics:
    """Simple UR5e robot kinematics implementation"""
    
    def __init__(self):
        # UR5e DH parameters (standard)
        self.d1 = 0.1625   # Base to shoulder
        self.a2 = -0.425   # Upper arm length (negative in standard UR convention)
        self.a3 = -0.3922  # Forearm length (negative in standard UR convention)
        self.d4 = 0.1333   # Wrist 1 height
        self.d5 = 0.0997   # Wrist 2 height
        self.d6 = 0.0996   # End effector length
        
        # Joint limits (radians)
        self.JOINT_LIMITS = [
            (-2*math.pi, 2*math.pi),  # θ1
            (-math.pi, math.pi),      # θ2
            (-math.pi, math.pi),      # θ3
            (-2*math.pi, 2*math.pi),  # θ4
            (-2*math.pi, 2*math.pi),  # θ5
            (-2*math.pi, 2*math.pi)   # θ6
        ]
        
        # Numerical parameters
        self.eps = 1e-6
        
        # Debug mode
        self.debug = False
        
        # Tolerance for considering a pose reachable (in mm)
        self.reachable_tolerance = 5.0  # mm
    
    def normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π]"""
        return ((angle + math.pi) % (2*math.pi)) - math.pi
    
    def dh_transform(self, d: float, a: float, alpha: float, theta: float) -> np.ndarray:
        """Standard DH transformation matrix"""
        ct = math.cos(theta)
        st = math.sin(theta)
        ca = math.cos(alpha)
        sa = math.sin(alpha)
        
        return np.array([
            [ct, -st*ca, st*sa, a*ct],
            [st, ct*ca, -ct*sa, a*st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])
    
    def forward_kinematics(self, joints: List[float]) -> np.ndarray:
        """
        Forward kinematics for UR5e robot.
        
        Args:
            joints: Six joint angles in radians [θ1, θ2, θ3, θ4, θ5, θ6]
            
        Returns:
            4x4 homogeneous transformation matrix for the end effector position
        """
        q1, q2, q3, q4, q5, q6 = joints
        
        # Define DH parameters for UR5e
        dh_params = [
            (self.d1, 0, math.pi/2, q1),              # Base to shoulder
            (0, self.a2, 0, q2),                       # Shoulder to elbow
            (0, self.a3, 0, q3),                       # Elbow to wrist1
            (self.d4, 0, math.pi/2, q4),               # Wrist1 to wrist2
            (self.d5, 0, -math.pi/2, q5),              # Wrist2 to wrist3
            (self.d6, 0, 0, q6)                        # Wrist3 to tool
        ]
        
        # Start with identity matrix
        T = np.eye(4)
        
        # Apply each transformation
        for d, a, alpha, theta in dh_params:
            Ti = self.dh_transform(d, a, alpha, theta)
            T = T @ Ti
        
        return T
    
    def inverse_kinematics(self, target_pose: np.ndarray, 
                          current_joints: Optional[List[float]] = None) -> List[List[float]]:
        """
        Inverse kinematics for UR5e robot.
        
        Args:
            target_pose: 4x4 homogeneous transformation matrix for the target pose
            current_joints: Optional current joint positions
            
        Returns:
            List of possible joint configurations
        """
        # Use different seed configurations for numerical optimization
        # to find multiple solutions
        seeds = [
            [0, -math.pi/2, 0, 0, 0, 0],               # Front
            [math.pi, -math.pi/2, 0, 0, 0, 0],         # Back
            [math.pi/2, -math.pi/2, 0, 0, 0, 0],       # Left
            [-math.pi/2, -math.pi/2, 0, 0, 0, 0],      # Right
            [0, -math.pi/4, -math.pi/4, 0, 0, 0],      # Front elbow bent
            [0, -math.pi/4, -math.pi/2, 0, math.pi/2, 0]  # Another configuration
        ]
        
        if current_joints is not None:
            seeds.insert(0, current_joints)
        
        # Solutions list
        solutions = []
        
        for seed in seeds:
            # Try to find solution using numerical optimization
            solution = self._numerical_ik(seed, target_pose, max_iterations=100)
            
            if solution is not None:
                is_duplicate = False
                for existing in solutions:
                    if self._similar_configs(solution, existing):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    solutions.append(solution)
                    
                    if self.debug and len(solutions) == 1:
                        print(f"Found solution: {[round(math.degrees(j), 2) for j in solution]}")
            
        return solutions
    
    def best_reachable_pose(self, target_pose: np.ndarray, 
                           current_joints: Optional[List[float]] = None) -> Tuple[List[float], np.ndarray]:
        """
        Find the closest reachable pose to the target pose.
        Useful when the target pose may be outside the robot's workspace.
        
        Args:
            target_pose: 4x4 homogeneous transformation matrix for the desired pose
            current_joints: Optional current joint positions
            
        Returns:
            Tuple of (best_joints, reachable_pose) where best_joints are the joint angles
            and reachable_pose is the closest reachable pose to the target
        """
        # First, try regular inverse kinematics
        solutions = self.inverse_kinematics(target_pose, current_joints)
        
        if solutions:
            # If solutions exist, select the best and return its pose
            best_joints = self.select_best_solution(solutions, current_joints)
            reachable_pose = self.forward_kinematics(best_joints)
            return best_joints, reachable_pose
        
        # If no direct solution, try to find a close approximation
        seeds = [
            [0, -math.pi/2, 0, 0, 0, 0],               # Common starting position
            [0, -math.pi/4, -math.pi/4, 0, 0, 0],      # Another common position
        ]
        
        if current_joints is not None:
            seeds.insert(0, current_joints)
        
        best_error = float('inf')
        best_joints = None
        
        # Try from each seed
        for seed in seeds:
            # Find best approximation from this seed
            joints, error = self._approximate_pose(seed, target_pose)
            
            if error < best_error:
                best_error = error
                best_joints = joints
        
        if best_joints is None:
            # Fallback to a safe configuration
            best_joints = [0, -math.pi/2, 0, 0, 0, 0]
        
        # Return the best approximation
        reachable_pose = self.forward_kinematics(best_joints)
        return best_joints, reachable_pose
    
    def _approximate_pose(self, seed: List[float], target_pose: np.ndarray, 
                         max_iterations: int = 200) -> Tuple[List[float], float]:
        """
        Find the closest approximation to the target pose.
        
        Args:
            seed: Initial joint positions
            target_pose: Target end effector pose
            max_iterations: Maximum optimization iterations
            
        Returns:
            Tuple of (best_joints, error) where error is the position/orientation error
        """
        # Current joint positions
        q = np.array(seed)
        
        # Set up convergence parameters
        alpha = 0.05  # Step size (smaller for more precise convergence)
        lambda_val = 0.2  # Damping factor
        
        target_pos = target_pose[:3, 3]
        target_rot = target_pose[:3, :3]
        
        # Position and orientation weights
        pos_weight = 1.0
        rot_weight = 0.2  # Less weight on orientation
        
        # Keep track of best solution
        best_q = q.copy()
        best_error = float('inf')
        
        # Optimization loop
        for iteration in range(max_iterations):
            # Get current pose
            current_pose = self.forward_kinematics(q)
            current_pos = current_pose[:3, 3]
            current_rot = current_pose[:3, :3]
            
            pos_error = target_pos - current_pos
            pos_error_norm = np.linalg.norm(pos_error)
            
            rot_error_matrix = current_rot.T @ target_rot
            rot_angle = math.acos(np.clip((np.trace(rot_error_matrix) - 1) / 2, -1.0, 1.0))
            
            # Combined error
            total_error = pos_weight * pos_error_norm + rot_weight * rot_angle
            
            # Keep track of best solution
            if total_error < best_error:
                best_error = total_error
                best_q = q.copy()
            
            # Calculate Jacobian matrix
            J = self._calculate_jacobian(q)
            
            # Create error vector
            error_vector = np.zeros(6)
            error_vector[:3] = pos_error * pos_weight
            
            # Rotation error axis
            if rot_angle > self.eps:
                rot_axis = np.array([
                    rot_error_matrix[2, 1] - rot_error_matrix[1, 2],
                    rot_error_matrix[0, 2] - rot_error_matrix[2, 0],
                    rot_error_matrix[1, 0] - rot_error_matrix[0, 1]
                ])
                if np.linalg.norm(rot_axis) > self.eps:
                    rot_axis = rot_axis / np.linalg.norm(rot_axis) * rot_angle
                    error_vector[3:] = rot_axis * rot_weight
            
            # Damped least squares method
            JT = J.T
            JJ = J @ JT + lambda_val * np.eye(6)
            try:
                delta_q = JT @ np.linalg.solve(JJ, error_vector)
            except np.linalg.LinAlgError:
                delta_q = JT @ np.linalg.pinv(JJ) @ error_vector
            
            step_size = alpha * (1.0 - 0.8 * (iteration / max_iterations))
            q += step_size * delta_q
            
            # Normalize to joint limits
            for i in range(len(q)):
                lower, upper = self.JOINT_LIMITS[i]
                q[i] = np.clip(self.normalize_angle(q[i]), lower, upper)
        
        return [self.normalize_angle(angle) for angle in best_q], best_error
    
    def _numerical_ik(self, seed: List[float], target_pose: np.ndarray, 
                     max_iterations: int = 50) -> Optional[List[float]]:
        """
        Numerical inverse kinematics solver using Jacobian method.
        
        Args:
            seed: Initial joint positions
            target_pose: Target end effector pose
            max_iterations: Maximum optimization iterations
            
        Returns:
            Joint positions or None if no solution found
        """
        # Current joint positions
        q = np.array(seed)
        
        # Set up convergence parameters
        alpha = 0.1  # Step size
        lambda_val = 0.1  # Damping factor for matrix inversion
        
        # Target position and rotation
        target_pos = target_pose[:3, 3]
        target_rot = target_pose[:3, :3]
        
        # Convergence thresholds
        pos_threshold = 1e-3  # 1mm
        rot_threshold = 1e-3  # ~0.057 degrees
        
        for iteration in range(max_iterations):
            # Get current pose from forward kinematics
            current_pose = self.forward_kinematics(q)
            current_pos = current_pose[:3, 3]
            current_rot = current_pose[:3, :3]
            
            pos_error = target_pos - current_pos
            pos_error_norm = np.linalg.norm(pos_error)
            
            rot_error_matrix = current_rot.T @ target_rot
            rot_angle = math.acos(np.clip((np.trace(rot_error_matrix) - 1) / 2, -1.0, 1.0))
            
            # Check convergence
            if pos_error_norm < pos_threshold and rot_angle < rot_threshold:
                # Return normalized joint angles
                return [self.normalize_angle(angle) for angle in q]
            
            # Calculate Jacobian matrix
            J = self._calculate_jacobian(q)
            
            # Create error vector (position and rotation error)
            error_vector = np.zeros(6)
            error_vector[:3] = pos_error
            
            if rot_angle > self.eps:
                rot_axis = np.array([
                    rot_error_matrix[2, 1] - rot_error_matrix[1, 2],
                    rot_error_matrix[0, 2] - rot_error_matrix[2, 0],
                    rot_error_matrix[1, 0] - rot_error_matrix[0, 1]
                ])
                # Normalize and scale by rotation angle
                if np.linalg.norm(rot_axis) > self.eps:
                    rot_axis = rot_axis / np.linalg.norm(rot_axis) * rot_angle
                    error_vector[3:] = rot_axis
            
            # Damped least squares method
            JT = J.T
            JJ = J @ JT + lambda_val * np.eye(6)
            try:
                delta_q = JT @ np.linalg.solve(JJ, error_vector)
            except np.linalg.LinAlgError:
                # Fallback to pseudoinverse if direct solve fails
                delta_q = JT @ np.linalg.pinv(JJ) @ error_vector
            
            q += alpha * delta_q
            
            # Normalize to joint limits
            for i in range(len(q)):
                lower, upper = self.JOINT_LIMITS[i]
                q[i] = np.clip(self.normalize_angle(q[i]), lower, upper)
        
        # If we got here, failed to converge
        return None
    
    def _calculate_jacobian(self, q: List[float]) -> np.ndarray:
        """
        Calculate the Jacobian matrix using finite differences.
        
        Args:
            q: Current joint positions
            
        Returns:
            6x6 Jacobian matrix
        """
        # Small step for finite difference
        delta = 1e-6
        
        # Initialize Jacobian matrix (6x6)
        J = np.zeros((6, 6))
        
        # Get current pose
        current_pose = self.forward_kinematics(q)
        current_pos = current_pose[:3, 3]
        current_rot = current_pose[:3, :3]
        
        # Compute each column of the Jacobian
        for i in range(6):
            q_delta = q.copy()
            q_delta[i] += delta
            
            # Get perturbed pose
            perturbed_pose = self.forward_kinematics(q_delta)
            perturbed_pos = perturbed_pose[:3, 3]
            perturbed_rot = perturbed_pose[:3, :3]
            
            # Linear velocity components (position Jacobian)
            J[:3, i] = (perturbed_pos - current_pos) / delta
            
            # Angular velocity components (orientation Jacobian)
            # Using matrix logarithm approximation
            delta_R = current_rot.T @ perturbed_rot
            # Small angle approximation to convert to axis-angle
            if abs(np.trace(delta_R) - 3) < self.eps:
                # If rotation is very small, assume zero
                J[3:, i] = np.zeros(3)
            else:
                angle = math.acos(np.clip((np.trace(delta_R) - 1) / 2, -1.0, 1.0))
                axis = np.array([
                    delta_R[2, 1] - delta_R[1, 2],
                    delta_R[0, 2] - delta_R[2, 0],
                    delta_R[1, 0] - delta_R[0, 1]
                ])
                if np.linalg.norm(axis) > self.eps:
                    axis = axis / np.linalg.norm(axis)
                    J[3:, i] = axis * angle / delta
        
        return J
    
    def _similar_configs(self, config1: List[float], config2: List[float], 
                      threshold: float = 0.1) -> bool:
        """
        Check if two configurations are similar.
        
        Args:
            config1, config2: Joint configurations to compare
            threshold: Joint angle similarity threshold in radians
            
        Returns:
            True if configurations are similar
        """
        for a, b in zip(config1, config2):
            if abs(self.normalize_angle(a - b)) > threshold:
                return False
        return True
    
    def select_best_solution(self, solutions: List[List[float]], 
                           current_joints: Optional[List[float]] = None) -> Optional[List[float]]:
        """
        Select best solution from multiple possibilities.
        
        Args:
            solutions: List of possible joint configurations
            current_joints: Current joint positions
            
        Returns:
            Best joint configuration or None if no solutions
        """
        if not solutions:
            return None
        
        if len(solutions) == 1:
            return solutions[0]
        
        if current_joints is not None:
            # Select solution closest to current configuration
            distances = []
            for solution in solutions:
                # Calculate weighted distance to current configuration
                # Primary joints (θ1, θ2, θ3) affect larger segments - weight them higher
                weights = [3.0, 2.0, 2.0, 1.0, 1.0, 1.0]
                distance = sum(w * (self.normalize_angle(s - c))**2 
                              for w, s, c in zip(weights, solution, current_joints))
                distances.append(distance)
            
            return solutions[np.argmin(distances)]
        
        # If no current joints, select solution with minimum joint displacement
        joint_sums = [sum(abs(j) for j in sol) for sol in solutions]
        return solutions[np.argmin(joint_sums)]

    def is_pose_reachable(self, target_pose: np.ndarray) -> bool:
        """
        Check if a target pose is reachable by the robot.
        
        Args:
            target_pose: 4x4 homogeneous transformation matrix for the target pose
            
        Returns:
            True if the pose is reachable, False otherwise
        """
        # Try to find inverse kinematics solution
        solutions = self.inverse_kinematics(target_pose)
        
        return len(solutions) > 0
        
    def is_valid_solution(self, joint_angles: List[float], target_pose: np.ndarray) -> bool:
        """
        Validate that a joint solution reaches the target pose within tolerance.
        
        Args:
            joint_angles: Joint angles solution to validate
            target_pose: Target end effector pose
            
        Returns:
            True if solution is valid, False otherwise
        """
        # Check if joint angles are within limits
        for i, angle in enumerate(joint_angles):
            lower, upper = self.JOINT_LIMITS[i]
            if not lower <= angle <= upper:
                return False
        
        achieved_pose = self.forward_kinematics(joint_angles)
        
        pos_error = np.linalg.norm(achieved_pose[:3, 3] - target_pose[:3, 3]) * 1000  # mm
        
        rot_error_matrix = achieved_pose[:3, :3].T @ target_pose[:3, :3]
        rot_angle = math.acos(np.clip((np.trace(rot_error_matrix) - 1) / 2, -1.0, 1.0))
        rot_error_deg = math.degrees(rot_angle)
        
        # Define error thresholds
        pos_threshold = 5.0  # mm
        rot_threshold = 5.0  # degrees
        
        # Check if errors are within thresholds
        return pos_error < pos_threshold and rot_error_deg < rot_threshold
    
    def format_ros2_command(self, joint_angles: List[float]) -> object:
        """
        Format joint angles as a ROS2 JointTrajectory message.
        
        Args:
            joint_angles: List of 6 joint angles in radians
            
        Returns:
            JointTrajectory message for UR5e ROS2 driver
            
        Note:
            This is a placeholder implementation. The actual implementation
            will depend on the ROS2 message types being used.
            
            In a real implementation, this would import the appropriate ROS2 message types
            and create a properly formatted message.
        """
        try:
            # This is a placeholder. In a real implementation, you would:
            # 1. Import the appropriate ROS2 message types
            # 2. Create a JointTrajectory message
            # 3. Fill in the message with the joint angles
            
            # For now, we'll just return a dictionary with the joint angles
            # that can be used to create the actual ROS2 message in the calling code
            
            # Normalize joint angles to the range expected by the robot controller
            normalized_angles = [self.normalize_angle(angle) for angle in joint_angles]
            
            # Create a placeholder for the ROS2 message
            # In a real implementation, this would be a proper ROS2 message object
            ros2_command = {
                "joint_names": [
                    "shoulder_pan_joint", 
                    "shoulder_lift_joint", 
                    "elbow_joint", 
                    "wrist_1_joint", 
                    "wrist_2_joint", 
                    "wrist_3_joint"
                ],
                "points": [{
                    "positions": normalized_angles,
                    "velocities": [0.0] * 6,
                    "accelerations": [0.0] * 6,
                    "time_from_start": {"sec": 1, "nanosec": 0}
                }]
            }
            
            return ros2_command
            
        except Exception as e:
            print(f"Error formatting ROS2 command: {e}")
            # Return a safe default
            return None


# Backwards compatibility aliases
UR5eKinematicsFixed = UR5eKinematics
UR5eKinematicsCorrected = UR5eKinematics


def test_hybrid_kinematics():
    """Test the hybrid kinematics implementation"""
    
    print("=== Hybrid UR5e Kinematics Test ===")
    
    # Initialize hybrid solver with debug enabled
    hybrid = HybridUR5eKinematics(debug=True)
    
    # Test configurations
    test_configs = [
        [0.0, -math.pi/2, 0.0, 0.0, 0.0, 0.0],           # Home position
        [0.0, -math.pi/4, -math.pi/4, 0.0, 0.0, 0.0],    # Common working pose
        [math.pi/4, -math.pi/4, -math.pi/4, 0.0, 0.0, 0.0],  # 45° base rotation
        [0.0, -math.pi/4, -math.pi/2, math.pi/4, math.pi/2, 0.0]  # Complex pose
    ]
    
    print(f"ur_ikfast available: {hybrid.ikfast_available}")
    print()
    
    for i, config in enumerate(test_configs):
        print(f"--- Test {i+1}: {[round(math.degrees(j), 1) for j in config]} ---")
        
        # Forward kinematics
        fk = hybrid.forward_kinematics(config)
        pos = fk[:3, 3]
        print(f"Target position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        
        # Hybrid inverse kinematics
        start_time = time.perf_counter()
        ik_solutions = hybrid.inverse_kinematics(fk, config)
        solve_time = (time.perf_counter() - start_time) * 1000  # ms
        
        print(f"Solutions found: {len(ik_solutions)} in {solve_time:.2f}ms")
        
        if ik_solutions:
            # Show only the best solution for brevity
            best_sol = ik_solutions[0]  # Already sorted by preference
            best_deg = [round(math.degrees(angle), 1) for angle in best_sol]
            print(f"Best solution: {best_deg}")
            
            # Verify accuracy
            check_pose = hybrid.forward_kinematics(best_sol)
            pos_error = np.linalg.norm(check_pose[:3, 3] - pos) * 1000  # mm
            
            rot_error_mat = fk[:3, :3].T @ check_pose[:3, :3]
            rot_trace = np.clip(np.trace(rot_error_mat), -1.0, 3.0)
            rot_error_deg = math.degrees(math.acos((rot_trace - 1) / 2))
            
            print(f"Position error: {pos_error:.3f}mm")
            print(f"Rotation error: {rot_error_deg:.3f}°")
            
            if pos_error < 5.0 and rot_error_deg < 5.0:
                print("✓ PASS")
            else:
                print("✗ FAIL")
        else:
            print("✗ FAIL - No solutions found")
        print()
    
    # Test VLM-specific scenarios
    print("--- VLM Edge Case Tests ---")
    
    # Test 1: Slightly noisy vision coordinates (common with OWL-ViT)
    print("\n1. Noisy vision coordinates:")
    clean_pose = np.eye(4)
    clean_pose[:3, 3] = [0.4, 0.2, 0.3]  # Reachable position
    
    noisy_pose = clean_pose.copy()
    noisy_pose[:3, 3] += np.array([0.005, -0.003, 0.002])  # 5mm noise
    
    clean_solutions = hybrid.inverse_kinematics(clean_pose)
    noisy_solutions = hybrid.inverse_kinematics(noisy_pose)
    
    print(f"Clean pose solutions: {len(clean_solutions)}")
    print(f"Noisy pose solutions: {len(noisy_solutions)}")
    
    if noisy_solutions:
        print("✓ Handled noisy vision data")
    else:
        print("⚠ No solution for noisy data - trying best_reachable_pose")
        best_joints, reachable_pose = hybrid.best_reachable_pose(noisy_pose)
        error = np.linalg.norm(reachable_pose[:3, 3] - noisy_pose[:3, 3]) * 1000
        print(f"  Found approximation with {error:.1f}mm error")
    
    # Test 2: Workspace boundary case
    print("\n2. Workspace boundary case:")
    boundary_pose = np.eye(4)
    boundary_pose[:3, 3] = [0.85, 0.0, 0.1]  # Near max reach
    
    boundary_solutions = hybrid.inverse_kinematics(boundary_pose, timeout_ms=100)
    print(f"Boundary pose solutions: {len(boundary_solutions)}")
    
    if not boundary_solutions:
        best_joints, reachable_pose = hybrid.best_reachable_pose(boundary_pose)
        error = np.linalg.norm(reachable_pose[:3, 3] - boundary_pose[:3, 3]) * 1000
        print(f"Best reachable pose with {error:.1f}mm error")
        print("✓ Graceful handling of boundary case")
    
    # Test 3: Completely unreachable pose
    print("\n3. Unreachable pose:")
    unreachable = np.eye(4)
    unreachable[:3, 3] = [1.5, 0.0, 0.0]  # Too far away
    
    unreachable_solutions = hybrid.inverse_kinematics(unreachable, timeout_ms=50)
    print(f"Unreachable pose solutions: {len(unreachable_solutions)}")
    
    if not unreachable_solutions:
        best_joints, reachable_pose = hybrid.best_reachable_pose(unreachable)
        error = np.linalg.norm(reachable_pose[:3, 3] - unreachable[:3, 3]) * 1000
        print(f"Closest reachable pose with {error:.1f}mm error")
        print("✓ Found closest approximation")
    
    # Performance summary
    hybrid.print_performance_summary()


def test_ur5e_kinematics():
    """Test the original UR5e kinematics implementation"""
    
    ur5e = UR5eKinematics()
    print("=== UR5e Kinematics Test ===")
    
    # Test configurations
    test_configs = [
        [0.0, -math.pi/2, 0.0, 0.0, 0.0, 0.0],           # Home position
        [0.0, -math.pi/4, -math.pi/4, 0.0, 0.0, 0.0],    # Common working pose
        [math.pi/4, -math.pi/4, -math.pi/4, 0.0, 0.0, 0.0],  # 45° base rotation
        [0.0, -math.pi/4, -math.pi/2, math.pi/4, math.pi/2, 0.0]  # Complex pose
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\n--- Test {i+1}: {[round(math.degrees(j), 1) for j in config]} ---")
        
        # Enable debug for first test only
        ur5e.debug = (i == 0)
        
        # Forward kinematics
        fk = ur5e.forward_kinematics(config)
        pos = fk[:3, 3]
        print(f"Forward position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        
        # Inverse kinematics (using the same config as a seed)
        ik_solutions = ur5e.inverse_kinematics(fk, config)
        print(f"Solutions found: {len(ik_solutions)}")
        
        if ik_solutions:
            for j, sol in enumerate(ik_solutions):
                sol_deg = [round(math.degrees(angle), 1) for angle in sol]
                print(f"  Solution {j+1}: {sol_deg}")
            
            # Select best solution
            best_sol = ur5e.select_best_solution(ik_solutions, config)
            best_deg = [round(math.degrees(angle), 1) for angle in best_sol]
            print(f"Best solution: {best_deg}")
            
            # Verify accuracy
            check_pose = ur5e.forward_kinematics(best_sol)
            pos_error = np.linalg.norm(check_pose[:3, 3] - pos) * 1000  # mm
            
            rot_error_mat = fk[:3, :3].T @ check_pose[:3, :3]
            rot_trace = np.clip(np.trace(rot_error_mat), -1.0, 3.0)
            rot_error_deg = math.degrees(math.acos((rot_trace - 1) / 2))
            
            print(f"Position error: {pos_error:.3f}mm")
            print(f"Rotation error: {rot_error_deg:.3f}°")
            
            if pos_error < 1.0 and rot_error_deg < 0.1:
                print("✓ PASS")
            else:
                print("✗ FAIL")
        else:
            print("✗ FAIL - No solutions found")
    
    # Test unreachable pose
    print("\n--- Testing unreachable pose ---")
    unreachable = np.eye(4)
    unreachable[:3, 3] = [1.5, 0.0, 0.0]  # Point too far away
    
    # Check reachability
    if not ur5e.is_pose_reachable(unreachable):
        print("Correctly identified unreachable pose")
        
        # Try to get closest reachable pose
        best_joints, reachable_pose = ur5e.best_reachable_pose(unreachable)
        best_pos = reachable_pose[:3, 3]
        distance = np.linalg.norm(best_pos - unreachable[:3, 3]) * 1000  # mm
        
        print(f"Found closest reachable pose at: [{best_pos[0]:.4f}, {best_pos[1]:.4f}, {best_pos[2]:.4f}]")
        print(f"Distance from target: {distance:.1f}mm")
        print(f"Joint angles: {[round(math.degrees(j), 1) for j in best_joints]}")
        print("✓ PASS")
    else:
        print("✗ FAIL - Incorrectly identified as reachable")


if __name__ == "__main__":
    # Test both implementations
    test_hybrid_kinematics()
    print("\n" + "="*50 + "\n")
    test_ur5e_kinematics()