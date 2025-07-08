"""
Safety and calibration verification script for eye-in-hand UR5e setup.
Run before testing with the real robot to ensure safe operation.
"""

import numpy as np
import pyrealsense2 as rs
import cv2
import logging
from typing import Tuple, Optional
import time

class EyeInHandSafetyChecker:
    """
    Safety checker for eye-in-hand configuration with RealSense on UR5e.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Expected camera mounting parameters 
        self.expected_camera_offset_z = -0.10  # Camera 10cm above gripper (negative Z)
        self.expected_camera_tilt = 15.0  # Degrees tilted forward to see gripper
        
        # Safety limits
        self.min_workspace_height = 0.05  # 5cm minimum height
        self.max_approach_speed = 0.1  # 10cm/s max approach speed
        self.gripper_safety_margin = 0.02  # 2cm margin around gripper
        
    def verify_camera_mounting(self, T_gripper_camera: np.ndarray) -> Tuple[bool, str]:
        """
        Verify camera is properly mounted for eye-in-hand operation.
        
        Parameters
        ----------
        T_gripper_camera : np.ndarray
            4x4 transformation matrix from gripper to camera
            
        Returns
        -------
        Tuple[bool, str]
            (is_valid, message)
        """
        # Extract camera position
        camera_pos = T_gripper_camera[:3, 3]
        
        # Check camera is above gripper
        if camera_pos[2] > -0.03:  # Camera should be at least 3cm above
            return False, f"Camera Z offset {camera_pos[2]:.3f}m is too small. Camera may collide with objects."
        
        # Check camera is reasonably centered
        lateral_offset = np.sqrt(camera_pos[0]**2 + camera_pos[1]**2)
        if lateral_offset > 0.05:  # More than 5cm lateral offset
            return False, f"Camera lateral offset {lateral_offset:.3f}m is too large. May cause visibility issues."
        
        # Check camera orientation
        camera_z_axis = T_gripper_camera[:3, 2]
        
        # For eye-in-hand, camera Z should point forward/down to see gripper
        # In gripper frame, this means Z axis should have negative Z component
        if camera_z_axis[2] > -0.5:  # cos(60°) = 0.5
            return False, "Camera not oriented to see gripper and workspace. Check mounting angle."
        
        return True, "Camera mounting verified for eye-in-hand operation."
    
    def check_gripper_visibility(self, pipeline: rs.pipeline) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Check if gripper is visible in camera view.
        
        Parameters
        ----------
        pipeline : rs.pipeline
            Active RealSense pipeline
            
        Returns
        -------
        Tuple[bool, Optional[np.ndarray]]
            (is_visible, annotated_image)
        """
        try:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                return False, None
            
            color_image = np.asanyarray(color_frame.get_data())
            
            # Convert to HSV for gripper detection
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            
            # Define gripper color range (adjust based on your gripper)
            # Example for silver/metallic gripper
            lower_silver = np.array([0, 0, 100])
            upper_silver = np.array([180, 30, 200])
            
            # Create mask
            mask = cv2.inRange(hsv, lower_silver, upper_silver)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for gripper-like shapes in bottom portion of image
            h, w = color_image.shape[:2]
            gripper_detected = False
            annotated = color_image.copy()
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check if in bottom portion of image (where gripper should be)
                    if y > color_image.shape[0] * 0.6:
                        gripper_detected = True
                        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(annotated, "Gripper", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return gripper_detected, annotated
            
        except Exception as e:
            self.logger.error(f"Error checking gripper visibility: {e}")
            return False, None
    
    def validate_workspace_coverage(self, pipeline: rs.pipeline, 
                                  test_height: float = 0.3) -> Tuple[bool, dict]:
        """
        Validate that camera can see the workspace at operating height.
        
        Parameters
        ----------
        pipeline : rs.pipeline
            Active RealSense pipeline
        test_height : float
            Height to test workspace visibility (meters)
            
        Returns
        -------
        Tuple[bool, dict]
            (is_valid, coverage_info)
        """
        try:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return False, {"error": "No frames available"}
            
            # Get camera intrinsics
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            
            # Calculate field of view at test height
            fov_h = 2 * np.arctan(intrinsics.width / (2 * intrinsics.fx))
            fov_v = 2 * np.arctan(intrinsics.height / (2 * intrinsics.fy))
            
            # Calculate workspace coverage at test height
            coverage_width = 2 * test_height * np.tan(fov_h / 2)
            coverage_height = 2 * test_height * np.tan(fov_v / 2)
            
            # Check if coverage is sufficient
            min_coverage = 0.2  # 20cm minimum coverage
            
            coverage_info = {
                "fov_horizontal_deg": np.degrees(fov_h),
                "fov_vertical_deg": np.degrees(fov_v),
                "coverage_width_m": coverage_width,
                "coverage_height_m": coverage_height,
                "test_height_m": test_height
            }
            
            is_valid = coverage_width >= min_coverage and coverage_height >= min_coverage
            
            return is_valid, coverage_info
            
        except Exception as e:
            self.logger.error(f"Error validating workspace coverage: {e}")
            return False, {"error": str(e)}
    
    def generate_safety_report(self, T_gripper_camera: np.ndarray, 
                             pipeline: rs.pipeline) -> dict:
        """
        Generate comprehensive safety report for eye-in-hand setup.
        
        Parameters
        ----------
        T_gripper_camera : np.ndarray
            Gripper to camera transformation
        pipeline : rs.pipeline
            Active RealSense pipeline
            
        Returns
        -------
        dict
            Safety report with all checks and recommendations
        """
        report = {
            "timestamp": time.time(),
            "configuration": "eye-in-hand",
            "checks": {},
            "recommendations": [],
            "ready_for_testing": True
        }
        
        # Check 1: Camera mounting
        mounting_ok, mounting_msg = self.verify_camera_mounting(T_gripper_camera)
        report["checks"]["camera_mounting"] = {
            "passed": mounting_ok,
            "message": mounting_msg
        }
        if not mounting_ok:
            report["ready_for_testing"] = False
            report["recommendations"].append("Adjust camera mounting position")
        
        # Check 2: Gripper visibility
        gripper_visible, gripper_img = self.check_gripper_visibility(pipeline)
        report["checks"]["gripper_visibility"] = {
            "passed": gripper_visible,
            "message": "Gripper visible in camera view" if gripper_visible else "Gripper not detected"
        }
        if not gripper_visible:
            report["recommendations"].append("Adjust camera angle to see gripper")
        
        # Check 3: Workspace coverage
        coverage_ok, coverage_info = self.validate_workspace_coverage(pipeline)
        report["checks"]["workspace_coverage"] = {
            "passed": coverage_ok,
            "info": coverage_info
        }
        if not coverage_ok:
            report["recommendations"].append("Camera field of view may be insufficient")
        
        # Check 4: Depth quality
        depth_quality = self.check_depth_quality(pipeline)
        report["checks"]["depth_quality"] = depth_quality
        if depth_quality["mean_confidence"] < 0.8:
            report["recommendations"].append("Improve lighting for better depth quality")
        
        # Add safety parameters
        report["safety_parameters"] = {
            "min_grasp_height_m": self.min_workspace_height,
            "max_approach_speed_m_s": self.max_approach_speed,
            "gripper_safety_margin_m": self.gripper_safety_margin,
            "recommended_approach_angle_deg": 10.0  # Slight angle to avoid straight-down
        }
        
        return report
    
    def check_depth_quality(self, pipeline: rs.pipeline) -> dict:
        """Check depth sensor quality metrics."""
        try:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            
            if not depth_frame:
                return {"error": "No depth frame"}
            
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Calculate depth statistics
            valid_depths = depth_image[depth_image > 0]
            
            if len(valid_depths) == 0:
                return {"error": "No valid depth data"}
            
            coverage = len(valid_depths) / depth_image.size
            mean_depth = np.mean(valid_depths) * 0.001  # Convert to meters
            std_depth = np.std(valid_depths) * 0.001
            
            # Confidence based on coverage and noise
            confidence = coverage * (1.0 - min(std_depth / mean_depth, 1.0))
            
            return {
                "coverage_ratio": coverage,
                "mean_depth_m": mean_depth,
                "std_depth_m": std_depth,
                "mean_confidence": confidence
            }
            
        except Exception as e:
            return {"error": str(e)}


def main():
    """Run safety checks for eye-in-hand configuration."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*60)
    print("EYE-IN-HAND SAFETY CHECK FOR UR5e")
    print("="*60)
    
    # Initialize RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    
    try:
        pipeline.start(config)
        logger.info("RealSense pipeline started")
        
        # Load calibration
        try:
            calib_data = np.load('hand_eye_calib_eye_in_hand.npz')
            T_gripper_camera = calib_data['T_base_to_camera']  # Contains T_gripper_camera for eye-in-hand
            logger.info("Loaded eye-in-hand calibration")
        except:
            logger.warning("No calibration found, using default")
            # Default mounting: 10cm above gripper, tilted 15° forward
            T_gripper_camera = np.array([
                [1, 0, 0, 0],
                [0, 0.966, 0.259, 0],  # 15° tilt
                [0, -0.259, 0.966, -0.10],
                [0, 0, 0, 1]
            ])
        
        # Run safety checks
        checker = EyeInHandSafetyChecker()
        report = checker.generate_safety_report(T_gripper_camera, pipeline)
        
        # Display report
        print("\n" + "-"*60)
        print("SAFETY CHECK REPORT")
        print("-"*60)
        
        for check_name, check_result in report["checks"].items():
            status = "✅ PASS" if check_result.get("passed", False) else "❌ FAIL"
            print(f"\n{check_name}: {status}")
            if "message" in check_result:
                print(f"  {check_result['message']}")
            if "info" in check_result:
                for key, value in check_result["info"].items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.3f}")
                    else:
                        print(f"  {key}: {value}")
        
        print("\n" + "-"*60)
        print("SAFETY PARAMETERS")
        print("-"*60)
        for param, value in report["safety_parameters"].items():
            print(f"{param}: {value}")
        
        print("\n" + "-"*60)
        print("RECOMMENDATIONS")
        print("-"*60)
        if report["recommendations"]:
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"{i}. {rec}")
        else:
            print("No issues found!")
        
        print("\n" + "-"*60)
        print(f"READY FOR TESTING: {'✅ YES' if report['ready_for_testing'] else '❌ NO'}")
        print("-"*60)
        
        # Show live view with gripper detection
        print("\nPress 'q' to quit live view, 's' to save a snapshot")
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            # Check gripper visibility
            gripper_visible, annotated = checker.check_gripper_visibility(pipeline)
            
            if annotated is not None:
                # Add depth colormap
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                
                # Stack images
                images = np.hstack((annotated, depth_colormap))
                
                # Add status text
                status = "Gripper: VISIBLE" if gripper_visible else "Gripper: NOT VISIBLE"
                cv2.putText(images, status, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, 
                           (0, 255, 0) if gripper_visible else (0, 0, 255), 2)
                
                cv2.imshow('Eye-in-Hand Safety Check', images)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('eye_in_hand_snapshot.png', images)
                logger.info("Saved snapshot")
        
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    
    print("\n✅ Safety check complete!")
    print("\nNEXT STEPS:")
    print("1. Address any failed checks or recommendations")
    print("2. Test with slow movements first")
    print("3. Always have emergency stop ready")
    print("4. Start with objects at safe heights (>10cm)")


if __name__ == '__main__':
    main()