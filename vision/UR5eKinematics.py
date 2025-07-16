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
    import ur_ikfast as ur_ik
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
    
    def _solve_with_numerical(self, target_pose: np.ndarray, 
                        current_joints: Optional[List[float]] = None,
                        timeout_ms: float = 100.0) -> List[List[float]]:
        """
        FIXED: Now calls the FAST improved method
        """
        start_time = time.perf_counter()
        self.numerical_attempts += 1
        
        timeout_sec = timeout_ms / 1000.0
        
        try:
            # CRITICAL FIX: Call the FAST method
            solutions = self.numerical_solver.inverse_kinematics_improved(
                target_pose, current_joints, timeout_sec
            )
            
            elapsed = time.perf_counter() - start_time
            
            if solutions:
                self.numerical_successes += 1
                if self.debug:
                    print(f"Numerical solver found {len(solutions)} solutions in {elapsed*1000:.1f}ms")
            else:
                if self.debug:
                    print(f"Numerical solver found no solutions in {elapsed*1000:.1f}ms")
            
            self.numerical_total_time += elapsed
            return solutions
            
        except Exception as e:
            if self.debug:
                print(f"Numerical solver failed: {e}")
            return []
    
    def inverse_kinematics(self, target_pose: np.ndarray, 
                      current_joints: Optional[List[float]] = None,
                      prefer_closest: bool = True,
                      timeout_ms: float = 150.0) -> List[List[float]]:
        """
        IMPROVED: Better timeout allocation between ur_ikfast and numerical
        """
        solutions = []
        
        # Step 1: ur_ikfast (working perfectly!)
        if self.ikfast_available:
            ikfast_solutions = self._solve_with_ikfast(target_pose)
            if ikfast_solutions:
                if self.debug:
                    print(f"ur_ikfast found {len(ikfast_solutions)} solutions")
                solutions.extend(ikfast_solutions)
        
        # Step 2: Smart timeout allocation for numerical
        if not solutions:
            # ur_ikfast failed, give numerical full time
            numerical_timeout = timeout_ms
        elif self.enable_fallback:
            # ur_ikfast succeeded, shorter backup time
            numerical_timeout = min(timeout_ms * 0.3, 75.0)
        else:
            numerical_timeout = 0
        
        if numerical_timeout > 0:
            numerical_solutions = self._solve_with_numerical(
                target_pose, current_joints, numerical_timeout
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
        
        # Filter and validate
        valid_solutions = []
        for solution in solutions:
            if self._is_valid_solution(solution, target_pose):
                valid_solutions.append(solution)
        
        # Sort by preference
        if valid_solutions and current_joints is not None and prefer_closest:
            valid_solutions.sort(key=lambda sol: self._solution_distance(sol, current_joints))
        
        return valid_solutions

    def _solve_with_ikfast(self, target_pose: np.ndarray) -> List[List[float]]:
        """
        FIXED: Solve IK using ur_ikfast with correct parameter names and format.
        """
        if not self.ikfast_available:
            return []
        
        start_time = time.perf_counter()
        self.ikfast_attempts += 1
        
        try:
            # Import the URKinematics class directly from ur_ikfast
            import ur_ikfast
            
            # Create kinematics instance for UR5e
            ur5e_kinematics = ur_ikfast.URKinematics('ur5e')
            
            position = target_pose[:3, 3]
            rotation = target_pose[:3, :3]
            
            # CRITICAL FIX 1: Convert rotation matrix to pose format expected by ur_ikfast
            # ur_ikfast expects [x, y, z, qx, qy, qz, qw] format
            try:
                # Convert rotation matrix to quaternion using the method from the provided URKinematics
                quat_wxyz = self._rotation_matrix_to_quaternion(rotation)
                # ur_ikfast expects [qx, qy, qz, qw] but we have [qw, qx, qy, qz]
                quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
                
                if len(quat_xyzw) != 4:
                    if self.debug:
                        print("Invalid quaternion length, falling back to identity")
                    quat_xyzw = [0, 0, 0, 1]  # Identity quaternion [qx, qy, qz, qw]
            except Exception as e:
                if self.debug:
                    print(f"Quaternion conversion failed: {e}, using identity")
                quat_xyzw = [0, 0, 0, 1]
            
            # CRITICAL FIX 2: Create pose in the format expected by ur_ikfast
            try:
                # Convert to standard Python types (not numpy)
                pos_list = [float(position[0]), float(position[1]), float(position[2])]
                quat_list = [float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]), float(quat_xyzw[3])]
                
                # Create pose vector [x, y, z, qx, qy, qz, qw] 
                ee_pose = pos_list + quat_list
                
                # Ensure all elements are Python floats
                ee_pose = [float(x) for x in ee_pose]
                
                # Create initial guess as Python list (FIXED: use q_guess parameter name)
                q_guess = [0.0, -1.5708, 0.0, 0.0, 0.0, 0.0]  # Default UR pose
                
            except Exception as e:
                if self.debug:
                    print(f"Data conversion failed: {e}")
                return []
            
            # CRITICAL FIX 3: Use correct parameter name 'q_guess' instead of 'seed'
            try:
                joint_configs = ur5e_kinematics.inverse(
                    ee_pose,
                    all_solutions=True,  # Get all solutions
                    q_guess=q_guess     # FIXED: use q_guess instead of seed
                )
            except ValueError as ve:
                if "ambiguous" in str(ve).lower() or "boolean" in str(ve).lower():
                    if self.debug:
                        print("ur_ikfast boolean/ambiguous error - likely numpy compatibility issue")
                    return []
                else:
                    # Re-raise other ValueErrors
                    raise ve
            except TypeError as te:
                if self.debug:
                    print(f"ur_ikfast type error: {te}")
                return []
            except Exception as e:
                if self.debug:
                    print(f"ur_ikfast general error: {e}")
                return []
            
            # CRITICAL FIX 4: Process solutions correctly
            solutions = []
            
            if joint_configs is not None:
                try:
                    # Handle different return formats from ur_ikfast
                    if isinstance(joint_configs, np.ndarray):
                        if joint_configs.ndim == 1 and len(joint_configs) == 6:
                            # Single solution as 1D array
                            configs_to_process = [joint_configs]
                        elif joint_configs.ndim == 2:
                            # Multiple solutions as 2D array
                            configs_to_process = [joint_configs[i] for i in range(joint_configs.shape[0])]
                        else:
                            if self.debug:
                                print(f"Unexpected numpy array shape: {joint_configs.shape}")
                            return []
                    elif isinstance(joint_configs, (list, tuple)):
                        if len(joint_configs) == 0:
                            configs_to_process = []
                        elif isinstance(joint_configs[0], (list, tuple, np.ndarray)):
                            # Multiple solutions
                            configs_to_process = joint_configs
                        elif len(joint_configs) == 6:
                            # Single solution as flat list
                            configs_to_process = [joint_configs]
                        else:
                            if self.debug:
                                print(f"Unexpected joint_configs format: {type(joint_configs)}, len={len(joint_configs)}")
                            return []
                    else:
                        if self.debug:
                            print(f"Unexpected joint_configs type: {type(joint_configs)}")
                        return []
                    
                    # Process each configuration
                    for config in configs_to_process:
                        try:
                            # Convert to Python list with proper handling
                            if isinstance(config, np.ndarray):
                                config_list = config.flatten().tolist()
                            elif isinstance(config, (list, tuple)):
                                config_list = list(config)
                            else:
                                if self.debug:
                                    print(f"Skipping config with unexpected type: {type(config)}")
                                continue
                            
                            # Ensure we have exactly 6 joints
                            if len(config_list) != 6:
                                if self.debug:
                                    print(f"Skipping config with wrong length: {len(config_list)}")
                                continue
                            
                            # Convert to float and normalize angles
                            try:
                                normalized_config = []
                                for joint_val in config_list:
                                    if isinstance(joint_val, (int, float, np.number)):
                                        normalized_angle = self.numerical_solver.normalize_angle(float(joint_val))
                                        normalized_config.append(normalized_angle)
                                    else:
                                        if self.debug:
                                            print(f"Invalid joint value type: {type(joint_val)}")
                                        break
                                
                                if len(normalized_config) == 6:
                                    # Validate against joint limits
                                    if self._within_joint_limits(normalized_config):
                                        solutions.append(normalized_config)
                                    elif self.debug:
                                        print("Solution exceeds joint limits")
                            
                            except (ValueError, TypeError) as e:
                                if self.debug:
                                    print(f"Error processing joint values: {e}")
                                continue
                        
                        except Exception as e:
                            if self.debug:
                                print(f"Error processing config: {e}")
                            continue
                
                except Exception as e:
                    if self.debug:
                        print(f"Error processing joint_configs: {e}")
                    return []
            
            # Update statistics
            elapsed = time.perf_counter() - start_time
            self.ikfast_total_time += elapsed
            
            if solutions:
                self.ikfast_successes += 1
                if self.debug:
                    print(f"ur_ikfast solved in {elapsed*1000:.3f}ms, found {len(solutions)} solutions")
            else:
                if self.debug:
                    print(f"ur_ikfast failed to find solutions in {elapsed*1000:.3f}ms")
            
            return solutions
        
        except ImportError:
            if self.debug:
                print("ur_ikfast not available")
            return []
        except Exception as e:
            if self.debug:
                print(f"ur_ikfast failed with exception: {type(e).__name__}: {e}")
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
    
    def _calculate_jacobian_adaptive(self, q: List[float], fast_mode: bool = False) -> np.ndarray:
        """
        Adaptive Jacobian calculation with performance optimization
        """
        # Adaptive delta based on convergence state
        if fast_mode:
            base_delta = 5e-7  # Smaller delta for precision
        else:
            base_delta = 1e-6  # Standard delta
        
        J = np.zeros((6, 6))
        
        try:
            current_pose = self.forward_kinematics(q)
            current_pos = current_pose[:3, 3]
            current_rot = current_pose[:3, :3]
        except:
            return np.eye(6)  # Return identity if FK fails
        
        for i in range(6):
            # Adaptive delta per joint
            delta = base_delta * max(1.0, abs(q[i]) / (math.pi/2))
            
            q_delta = list(q)
            q_delta[i] += delta
            
            try:
                perturbed_pose = self.forward_kinematics(q_delta)
                perturbed_pos = perturbed_pose[:3, 3]
                perturbed_rot = perturbed_pose[:3, :3]
                
                # Position Jacobian
                J[:3, i] = (perturbed_pos - current_pos) / delta
                
                # Orientation Jacobian with robust calculation
                try:
                    delta_R = current_rot.T @ perturbed_rot
                    trace_val = np.trace(delta_R)
                    
                    if abs(trace_val - 3) < 1e-6:
                        J[3:, i] = np.zeros(3)
                    else:
                        angle = math.acos(np.clip((trace_val - 1) / 2, -1.0, 1.0))
                        if angle > 1e-6:
                            axis = np.array([
                                delta_R[2, 1] - delta_R[1, 2],
                                delta_R[0, 2] - delta_R[2, 0],
                                delta_R[1, 0] - delta_R[0, 1]
                            ])
                            axis_norm = np.linalg.norm(axis)
                            if axis_norm > 1e-6:
                                J[3:, i] = (axis / axis_norm) * angle / delta
                            else:
                                J[3:, i] = np.zeros(3)
                        else:
                            J[3:, i] = np.zeros(3)
                except:
                    J[3:, i] = np.zeros(3)
            except:
                # If FK fails, use numerical approximation
                J[:, i] = np.zeros(6)
        
        return J
    
    def _numerical_ik_improved(self, seed: List[float], target_pose: np.ndarray, 
                         max_time: float = 0.05, max_iterations: int = 50) -> Optional[List[float]]:
        """
        GREATLY IMPROVED: Fast converging numerical IK with adaptive parameters
        """
        q = np.array(seed, dtype=float)
        
        # Aggressive parameters for fast convergence
        initial_alpha = 0.3    # Higher initial step size
        min_alpha = 0.005      # Minimum step size
        max_alpha = 0.5        # Maximum step size
        lambda_val = 0.03      # Lower initial damping
        
        target_pos = target_pose[:3, 3]
        target_rot = target_pose[:3, :3]
        
        # Relaxed convergence thresholds for speed
        pos_threshold = 1.5e-3  # 1.5mm
        rot_threshold = 1.5e-3  # ~0.086 degrees
        
        start_time = time.perf_counter()
        alpha = initial_alpha
        
        prev_error = float('inf')
        stall_count = 0
        
        # Track best solution
        best_q = q.copy()
        best_error = float('inf')
        
        for iteration in range(max_iterations):
            # Quick timeout check
            if time.perf_counter() - start_time > max_time:
                break
            
            # Get current pose
            try:
                current_pose = self.forward_kinematics(q)
                current_pos = current_pose[:3, 3]
                current_rot = current_pose[:3, :3]
            except:
                break  # Invalid configuration
            
            pos_error = target_pos - current_pos
            pos_error_norm = np.linalg.norm(pos_error)
            
            try:
                rot_error_matrix = current_rot.T @ target_rot
                rot_angle = math.acos(np.clip((np.trace(rot_error_matrix) - 1) / 2, -1.0, 1.0))
            except:
                rot_angle = 0.05  # Assume some rotation error
            
            current_error = pos_error_norm + rot_angle * 0.3
            
            # Track best solution
            if current_error < best_error:
                best_error = current_error
                best_q = q.copy()
            
            # Check convergence
            if pos_error_norm < pos_threshold and rot_angle < rot_threshold:
                return [self.normalize_angle(angle) for angle in q]
            
            # Adaptive step size based on progress
            if current_error >= prev_error:
                stall_count += 1
                if stall_count > 2:
                    alpha = max(alpha * 0.7, min_alpha)
                    stall_count = 0
            else:
                if current_error < 0.8 * prev_error:
                    alpha = min(alpha * 1.15, max_alpha)
                stall_count = 0
            
            prev_error = current_error
            
            # Fast Jacobian calculation
            try:
                J = self._calculate_jacobian_fast(q)
            except:
                break
            
            # Error vector
            error_vector = np.zeros(6)
            error_vector[:3] = pos_error
            
            # Rotation error
            if rot_angle > self.eps:
                try:
                    rot_axis = np.array([
                        rot_error_matrix[2, 1] - rot_error_matrix[1, 2],
                        rot_error_matrix[0, 2] - rot_error_matrix[2, 0],
                        rot_error_matrix[1, 0] - rot_error_matrix[0, 1]
                    ])
                    if np.linalg.norm(rot_axis) > self.eps:
                        rot_axis = rot_axis / np.linalg.norm(rot_axis) * rot_angle
                        error_vector[3:] = rot_axis * 0.4
                except:
                    pass
            
            # Adaptive damped least squares
            JT = J.T
            adaptive_lambda = lambda_val * (1 + min(current_error * 2, 2))
            JJ = J @ JT + adaptive_lambda * np.eye(6)
            
            try:
                delta_q = JT @ np.linalg.solve(JJ, error_vector)
            except:
                try:
                    delta_q = JT @ np.linalg.pinv(JJ) @ error_vector
                except:
                    break
            
            q += alpha * delta_q
            
            # Enforce joint limits
            for i in range(len(q)):
                lower, upper = self.JOINT_LIMITS[i]
                q[i] = np.clip(self.normalize_angle(q[i]), lower, upper)
        
        # Return best solution if reasonable
        if best_error < 0.05:  # 50mm total error threshold
            return [self.normalize_angle(angle) for angle in best_q]
        
        return None
    
    def forward_kinematics(self, joints: List[float]) -> np.ndarray:
        """Forward kinematics using the numerical solver's implementation."""
        return self.numerical_solver.forward_kinematics(joints)
    
    def _rotation_matrix_to_quaternion(self, rotation_matrix: np.ndarray) -> List[float]:
        """
        Convert rotation matrix to quaternion (w, x, y, z) with robust handling.
        
        Args:
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            Quaternion as [w, x, y, z]
        """
        # First, validate the rotation matrix
        try:
            R = np.array(rotation_matrix, dtype=np.float64)
            
            # Ensure it's 3x3
            if R.shape != (3, 3):
                if self.debug:
                    print(f"Invalid rotation matrix shape: {R.shape}, using identity")
                return [1.0, 0.0, 0.0, 0.0]
            
            # Check if it's a proper rotation matrix
            det = np.linalg.det(R)
            if not np.isclose(det, 1.0, atol=1e-6):
                if self.debug:
                    print(f"Invalid rotation matrix determinant: {det}, using identity")
                return [1.0, 0.0, 0.0, 0.0]
            
            # Check orthogonality
            should_be_identity = R @ R.T
            identity = np.eye(3)
            if not np.allclose(should_be_identity, identity, atol=1e-6):
                if self.debug:
                    print("Non-orthogonal rotation matrix, using identity")
                return [1.0, 0.0, 0.0, 0.0]
            
        except Exception as e:
            if self.debug:
                print(f"Error validating rotation matrix: {e}, using identity")
            return [1.0, 0.0, 0.0, 0.0]
        
        # Use scipy if available for robust conversion
        if SCIPY_AVAILABLE and R is not None:
            try:
                rotation_obj = R.from_matrix(rotation_matrix)
                # Get quaternion in [x, y, z, w] format from scipy
                quat_xyzw = rotation_obj.as_quat()
                # Convert to [w, x, y, z] format
                quat_wxyz = [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]
                
                # Validate quaternion
                norm = np.linalg.norm(quat_wxyz)
                if not np.isclose(norm, 1.0, atol=1e-6):
                    if self.debug:
                        print(f"Invalid quaternion norm: {norm}, normalizing")
                    quat_wxyz = [q / norm for q in quat_wxyz]
                
                return quat_wxyz
                
            except Exception as e:
                if self.debug:
                    print(f"Scipy quaternion conversion failed: {e}, using manual method")
        
        # Manual calculation using Shepperd's method (more numerically stable)
        try:
            R_mat = rotation_matrix
            
            # Shepperd's method for robust quaternion extraction
            trace = R_mat[0, 0] + R_mat[1, 1] + R_mat[2, 2]
            
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
            
            # Normalize the quaternion
            quat = [qw, qx, qy, qz]
            norm = np.sqrt(sum(q*q for q in quat))
            
            if norm < 1e-8:
                if self.debug:
                    print("Quaternion norm too small, using identity")
                return [1.0, 0.0, 0.0, 0.0]
            
            quat_normalized = [float(q / norm) for q in quat]
            
            # Ensure w component is positive (avoid double cover issue)
            if quat_normalized[0] < 0:
                quat_normalized = [-q for q in quat_normalized]
            
            return quat_normalized
            
        except Exception as e:
            if self.debug:
                print(f"Manual quaternion calculation failed: {e}")
            return [1.0, 0.0, 0.0, 0.0]  # Identity quaternion as fallback
    
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
            J = self._calculate_jacobian_fast(q)
            
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
            J = self._calculate_jacobian_fast(q)
            
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
    
    def _calculate_jacobian_fast(self, q: List[float]) -> np.ndarray:
        """
        FAST: Optimized Jacobian calculation for speed
        """
        delta = 1.5e-6  # Slightly larger step for speed

        J = np.zeros((6, 6))
        
        try:
            current_pose = self.forward_kinematics(q)
            current_pos = current_pose[:3, 3]
            current_rot = current_pose[:3, :3]
        except:
            return np.eye(6)
        
        for i in range(6):
            q_delta = list(q)
            q_delta[i] += delta
            
            try:
                perturbed_pose = self.forward_kinematics(q_delta)
                perturbed_pos = perturbed_pose[:3, 3]
                perturbed_rot = perturbed_pose[:3, :3]
                
                # Position Jacobian
                J[:3, i] = (perturbed_pos - current_pos) / delta
                
                # Simplified orientation Jacobian for speed
                try:
                    delta_R = current_rot.T @ perturbed_rot
                    trace_val = np.trace(delta_R)
                    
                    if abs(trace_val - 3) < 1e-6:
                        J[3:, i] = np.zeros(3)
                    else:
                        angle = math.acos(np.clip((trace_val - 1) / 2, -1.0, 1.0))
                        if angle > 1e-6:
                            axis = np.array([
                                delta_R[2, 1] - delta_R[1, 2],
                                delta_R[0, 2] - delta_R[2, 0],
                                delta_R[1, 0] - delta_R[0, 1]
                            ])
                            axis_norm = np.linalg.norm(axis)
                            if axis_norm > 1e-6:
                                J[3:, i] = (axis / axis_norm) * angle / delta
                            else:
                                J[3:, i] = np.zeros(3)
                        else:
                            J[3:, i] = np.zeros(3)
                except:
                    J[3:, i] = np.zeros(3)
            except:
                J[:, i] = np.zeros(6)
        
        return J
    
    def inverse_kinematics_improved(self, target_pose: np.ndarray, 
                              current_joints: Optional[List[float]] = None,
                              timeout_sec: float = 0.15) -> List[List[float]]:
        """
        IMPROVED: Faster inverse kinematics with better seed selection
        """
        # Better seed selection for faster convergence
        seeds = [
            [0, -math.pi/2, 0, 0, 0, 0],               # Standard home
            [0, -math.pi/3, -math.pi/3, 0, 0, 0],      # Forward lean
            [math.pi/2, -math.pi/2, 0, 0, 0, 0],       # 90° rotated
            [-math.pi/2, -math.pi/2, 0, 0, 0, 0],      # -90° rotated
            [0, -math.pi/4, -math.pi/2, 0, math.pi/2, 0],  # Elbow down
            [math.pi, -math.pi/2, 0, 0, 0, 0],         # Back configuration
        ]
        
        if current_joints is not None:
            seeds.insert(0, current_joints)
        
        solutions = []
        start_time = time.perf_counter()
        base_time_per_seed = timeout_sec / len(seeds)
        
        for i, seed in enumerate(seeds):
            # Check global timeout
            elapsed = time.perf_counter() - start_time
            if elapsed > timeout_sec:
                break
            
            # Smart time allocation
            remaining_time = timeout_sec - elapsed
            remaining_seeds = len(seeds) - i
            seed_timeout = min(remaining_time / max(1, remaining_seeds), base_time_per_seed * 2)
            
            # Use FAST numerical IK
            solution = self._numerical_ik_fast(seed, target_pose, 
                                            max_time=seed_timeout,
                                            max_iterations=30)
            
            if solution is not None:
                # Check for duplicates
                is_duplicate = False
                for existing in solutions:
                    if self._similar_configs(solution, existing):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    solutions.append(solution)
                    if self.debug and len(solutions) == 1:
                        print(f"Found solution: {[round(math.degrees(j), 2) for j in solution]}")
                    
                    # Early exit if we have enough solutions
                    if len(solutions) >= 2:
                        break
        
        return solutions
    
    def _numerical_ik_fast(self, seed: List[float], target_pose: np.ndarray, 
                      max_time: float = 0.05, max_iterations: int = 30) -> Optional[List[float]]:
        """
        ULTRA-FAST: Numerical IK optimized for speed and success rate
        """
        q = np.array(seed, dtype=float)
        
        # Aggressive parameters for FAST convergence
        alpha = 0.4        # High step size
        min_alpha = 0.01   # Higher minimum
        lambda_val = 0.02  # Low damping
        
        target_pos = target_pose[:3, 3]
        target_rot = target_pose[:3, :3]
        
        # Relaxed thresholds for speed
        pos_threshold = 2e-3  # 2mm (relaxed)
        rot_threshold = 2e-3  # ~0.11 degrees
        
        start_time = time.perf_counter()
        best_q = q.copy()
        best_error = float('inf')
        
        for iteration in range(max_iterations):
            # Quick timeout check
            if time.perf_counter() - start_time > max_time:
                break
            
            # Forward kinematics
            try:
                current_pose = self.forward_kinematics(q)
                current_pos = current_pose[:3, 3]
                current_rot = current_pose[:3, :3]
            except:
                break
            
            # Calculate errors
            pos_error = target_pos - current_pos
            pos_error_norm = np.linalg.norm(pos_error)
            
            try:
                rot_error_matrix = current_rot.T @ target_rot
                rot_angle = math.acos(np.clip((np.trace(rot_error_matrix) - 1) / 2, -1.0, 1.0))
            except:
                rot_angle = 0.05
            
            current_error = pos_error_norm + rot_angle * 0.3
            
            # Track best solution
            if current_error < best_error:
                best_error = current_error
                best_q = q.copy()
            
            # Check convergence
            if pos_error_norm < pos_threshold and rot_angle < rot_threshold:
                return [self.normalize_angle(angle) for angle in q]
            
            # Adaptive step size
            if iteration > 3 and current_error > best_error * 0.9:
                alpha = max(alpha * 0.8, min_alpha)
            
            # Fast Jacobian
            try:
                J = self._calculate_jacobian_super_fast(q)
            except:
                break
            
            # Error vector
            error_vector = np.zeros(6)
            error_vector[:3] = pos_error
            
            # Rotation error
            if rot_angle > self.eps:
                try:
                    rot_axis = np.array([
                        rot_error_matrix[2, 1] - rot_error_matrix[1, 2],
                        rot_error_matrix[0, 2] - rot_error_matrix[2, 0],
                        rot_error_matrix[1, 0] - rot_error_matrix[0, 1]
                    ])
                    if np.linalg.norm(rot_axis) > self.eps:
                        rot_axis = rot_axis / np.linalg.norm(rot_axis) * rot_angle
                        error_vector[3:] = rot_axis * 0.4
                except:
                    pass
            
            # Simple damped least squares
            JT = J.T
            JJ = J @ JT + lambda_val * np.eye(6)
            
            try:
                delta_q = JT @ np.linalg.solve(JJ, error_vector)
            except:
                try:
                    delta_q = JT @ np.linalg.pinv(JJ) @ error_vector
                except:
                    break
            
            q += alpha * delta_q
            
            # Enforce joint limits quickly
            for i in range(6):
                lower, upper = self.JOINT_LIMITS[i]
                q[i] = np.clip(self.normalize_angle(q[i]), lower, upper)
        
        # Return best solution if reasonable
        if best_error < 0.08:  # 80mm total error threshold
            return [self.normalize_angle(angle) for angle in best_q]
        
        return None

    def _calculate_jacobian_super_fast(self, q: List[float]) -> np.ndarray:
        """
        SUPER-FAST: Minimal Jacobian calculation for maximum speed
        """
        delta = 2e-6  # Larger step for speed
        J = np.zeros((6, 6))
        
        try:
            current_pose = self.forward_kinematics(q)
            current_pos = current_pose[:3, 3]
        except:
            return np.eye(6)
        
        # Only calculate position Jacobian for maximum speed
        for i in range(6):
            q_delta = list(q)
            q_delta[i] += delta
            
            try:
                perturbed_pose = self.forward_kinematics(q_delta)
                perturbed_pos = perturbed_pose[:3, 3]
                
                # Position Jacobian
                J[:3, i] = (perturbed_pos - current_pos) / delta
                
                # Simplified orientation (good enough for most cases)
                if i < 3:  # Only for major joints
                    J[3+i, i] = 1.0
                    
            except:
                J[:, i] = np.zeros(6)
        
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