# scripts/create_sim_calibration.py
#!/usr/bin/env python3
"""
Create hand-eye calibration file for simulation.
The transformation is from gripper to camera for eye-in-hand configuration.
"""

import numpy as np
import os

def create_simulation_hand_eye_calibration():
    """Create a hand-eye calibration file for simulation."""
    
    # Eye-in-hand transformation: camera relative to gripper
    # Camera is mounted 5cm above gripper, looking forward/down
    T_gripper_camera = np.array([
        [1.0,  0.0,  0.0,  0.00],  # X: aligned with gripper
        [0.0,  0.866, 0.5, -0.02],  # Y: 30° tilt down
        [0.0, -0.5,  0.866, -0.05], # Z: 5cm above gripper
        [0.0,  0.0,  0.0,  1.0]
    ])
    
    # Save calibration
    save_path = os.path.join(
        os.path.dirname(__file__), '..', 'config', 'hand_eye_calib_sim.npz'
    )
    
    np.savez(
        save_path,
        T_base_to_camera=T_gripper_camera,  # Keep key name for compatibility
        is_eye_in_hand=True,
        calibration_type='eye_in_hand',
        simulation=True
    )
    
    print(f"Created simulation calibration at: {save_path}")
    print(f"Transformation matrix (T_gripper_camera):\n{T_gripper_camera}")

if __name__ == "__main__":
    create_simulation_hand_eye_calibration()