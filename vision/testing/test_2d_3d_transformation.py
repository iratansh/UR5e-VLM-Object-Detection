#!/usr/bin/env python3
"""
2D-3D Coordinate Transformation Test

This test validates the complete pipeline from speech recognition to 3D coordinates:
1. Speech Recognition → Object queries
2. MacBook Camera → RGB frames
3. OWL-ViT → 2D bounding boxes
4. Mock depth + Camera calibration → 3D coordinates
5. Visualization and validation

This allows you to verify the coordinate transformation logic without needing
the physical UR5e or RealSense camera.
"""

import cv2
import numpy as np
import time
import logging
import threading
import queue
from typing import List, Dict, Any, Optional, Tuple
import json
import math

# Import your vision components
from OWLViTDetector import OWLViTDetector
from SpeechCommandProcessor import SpeechCommandProcessor
from CameraCalibration import CameraCalibration
from GraspPointDetector import GraspPointDetector
from DepthAwareDetector import Detection3D
from WorkSpaceValidator import WorkspaceValidator

class MockDepthGenerator:
    """Generate realistic depth maps from RGB images for testing"""
    
    def __init__(self, camera_height: float = 0.5):
        self.camera_height = camera_height  # Height of camera above table in meters
        self.table_height = 0.0  # Assume table is at z=0
        
        # Typical object depths (in meters from camera)
        self.object_depths = {
            'bottle': 0.3,
            'cup': 0.35,
            'book': 0.4,
            'phone': 0.25,
            'box': 0.45,
            'can': 0.3,
            'default': 0.35
        }
        
    def generate_depth_for_detection(self, bbox: List[int], label: str, 
                                   image_shape: Tuple[int, int]) -> Tuple[float, np.ndarray]:
        """
        Generate realistic depth value and depth map for a detected object
        
        Returns:
            depth_value: Depth in meters
            depth_map: Full depth map for the image
        """
        height, width = image_shape[:2]
        
        # Create base depth map (background at camera_height)
        depth_map = np.full((height, width), self.camera_height, dtype=np.float32)
        
        # Get object-specific depth
        object_depth = self.object_depths.get(label.lower(), self.object_depths['default'])
        
        # Add some realistic variation
        depth_variation = np.random.normal(0, 0.02)  # 2cm standard deviation
        actual_depth = max(0.1, object_depth + depth_variation)
        
        # Fill object region with object depth
        x1, y1, x2, y2 = bbox
        depth_map[y1:y2, x1:x2] = actual_depth
        
        # Add some noise to make it realistic
        noise = np.random.normal(0, 0.005, depth_map.shape)  # 5mm noise
        depth_map += noise
        depth_map = np.clip(depth_map, 0.1, 2.0)  # Clip to reasonable range
        
        return actual_depth, depth_map

class CoordinateTransformationTester:
    """Test 2D-3D coordinate transformations with visual validation"""
    
    def __init__(self, use_speech: bool = True):
        self.use_speech = use_speech
        self.running = False
        
        # Initialize components
        self.setup_camera()
        self.setup_vision_components()
        self.setup_coordinate_system()
        
        # Test data storage
        self.test_results = []
        self.current_detections = []
        
        # UI state
        self.show_depth_visualization = True
        self.show_coordinate_overlay = True
        self.current_queries = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def setup_camera(self):
        """Initialize MacBook camera"""
        self.camera = cv2.VideoCapture(0)  # MacBook built-in camera
        
        if not self.camera.isOpened():
            raise RuntimeError("Cannot open MacBook camera")
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual camera properties
        self.camera_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.camera_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.logger.info(f"Camera initialized: {self.camera_width}x{self.camera_height}")
        
    def setup_vision_components(self):
        """Initialize vision processing components"""
        
        # OWL-ViT detector
        try:
            self.vlm_detector = OWLViTDetector()
            self.logger.info("✅ OWL-ViT detector loaded")
        except Exception as e:
            self.logger.error(f"Failed to load OWL-ViT: {e}")
            self.vlm_detector = None
        
        # Speech processor
        if self.use_speech:
            try:
                self.speech_processor = SpeechCommandProcessor()
                self.logger.info("✅ Speech processor loaded")
            except Exception as e:
                self.logger.warning(f"Speech processor not available: {e}")
                self.speech_processor = None
        else:
            self.speech_processor = None
        
        # Grasp point detector
        self.grasp_detector = GraspPointDetector()
        
        # Mock depth generator
        self.depth_generator = MockDepthGenerator(camera_height=0.5)
        
        # Workspace validator
        self.workspace_validator = WorkspaceValidator()
        
    def setup_coordinate_system(self):
        """Setup camera calibration and coordinate system"""
        
        # Mock camera intrinsics (typical MacBook camera values)
        self.camera_matrix = np.array([
            [800.0, 0.0, self.camera_width/2],
            [0.0, 800.0, self.camera_height/2],
            [0.0, 0.0, 1.0]
        ])
        
        self.dist_coeffs = np.zeros(5)  # Assume no distortion for simplicity
        
        # Setup calibration object
        self.calibration = CameraCalibration()
        self.calibration.camera_matrix_color = self.camera_matrix
        self.calibration.dist_coeffs_color = self.dist_coeffs
        self.calibration.source_type = "webcam"
        
        # Mock eye-in-hand transformation (camera mounted on robot end-effector)
        # This represents the transformation from gripper to camera
        self.T_gripper_camera = np.array([
            [0, -1, 0, 0.05],   # Camera X = -Gripper Y, offset 5cm
            [1, 0, 0, 0.0],     # Camera Y = Gripper X
            [0, 0, 1, 0.08],    # Camera Z = Gripper Z, offset 8cm up
            [0, 0, 0, 1]
        ])
        
        self.calibration.set_hand_eye_transform(self.T_gripper_camera, is_eye_in_hand=True)
        
        # Mock current gripper pose (where the robot end-effector would be)
        self.mock_gripper_pose = np.array([
            [1, 0, 0, 0.4],     # 40cm in front of robot base
            [0, 1, 0, 0.0],     # Centered left-right
            [0, 0, 1, 0.3],     # 30cm above table
            [0, 0, 0, 1]
        ])
        
        self.logger.info("✅ Coordinate system setup complete")
        
    def pixel_to_3d_camera_frame(self, pixel_x: int, pixel_y: int, depth: float) -> np.ndarray:
        """Convert pixel coordinates to 3D point in camera frame"""
        
        # Use camera intrinsics to unproject
        fx, fy = self.camera_matrix[0,0], self.camera_matrix[1,1]
        cx, cy = self.camera_matrix[0,2], self.camera_matrix[1,2]
        
        # Convert to normalized camera coordinates
        x_cam = (pixel_x - cx) * depth / fx
        y_cam = (pixel_y - cy) * depth / fy
        z_cam = depth
        
        return np.array([x_cam, y_cam, z_cam])
    
    def camera_to_robot_frame(self, point_camera: np.ndarray) -> np.ndarray:
        """Transform point from camera frame to robot base frame"""
        
        # Transform: camera -> gripper -> robot_base
        point_camera_h = np.array([point_camera[0], point_camera[1], point_camera[2], 1.0])
        
        # Camera to gripper
        T_camera_gripper = np.linalg.inv(self.T_gripper_camera)
        point_gripper_h = T_camera_gripper @ point_camera_h
        
        # Gripper to robot base
        point_robot_h = self.mock_gripper_pose @ point_gripper_h
        
        return point_robot_h[:3]
    
    def process_speech_input(self) -> List[str]:
        """Get object queries from speech input"""
        
        if not self.speech_processor:
            # Fallback: manual input
            print("\nSpeech not available. Enter object to find (or 'quit'):")
            user_input = input("> ").strip()
            if user_input.lower() == 'quit':
                return []
            return [user_input] if user_input else []
        
        try:
            command = self.speech_processor.get_command()
            if command:
                self.logger.info(f"🎤 Speech command: '{command}'")
                queries = self.speech_processor.parse_object_query(command)
                return queries if queries else []
        except Exception as e:
            self.logger.error(f"Speech processing error: {e}")
        
        return []
    
    def detect_objects_2d(self, frame: np.ndarray, queries: List[str]) -> List[Tuple[str, float, List[int]]]:
        """Detect objects in 2D using OWL-ViT"""
        
        if not self.vlm_detector or not queries:
            return []
        
        try:
            start_time = time.perf_counter()
            detections = self.vlm_detector.detect_with_text_queries(
                frame, queries, confidence_threshold=0.1
            )
            detection_time = (time.perf_counter() - start_time) * 1000
            
            self.logger.info(f"🔍 VLM detection: {len(detections)} objects in {detection_time:.1f}ms")
            return detections
            
        except Exception as e:
            self.logger.error(f"VLM detection error: {e}")
            return []
    
    def convert_2d_to_3d(self, detections_2d: List[Tuple[str, float, List[int]]], 
                        frame: np.ndarray) -> List[Dict[str, Any]]:
        """Convert 2D detections to 3D coordinates with full validation"""
        
        detections_3d = []
        
        for label, confidence, bbox in detections_2d:
            x1, y1, x2, y2 = bbox
            
            # Get center point of bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Generate mock depth for this object
            depth_value, depth_map = self.depth_generator.generate_depth_for_detection(
                bbox, label, frame.shape
            )
            
            # Convert center point to 3D camera coordinates
            point_3d_camera = self.pixel_to_3d_camera_frame(center_x, center_y, depth_value)
            
            # Transform to robot base frame
            point_3d_robot = self.camera_to_robot_frame(point_3d_camera)
            
            # Get grasp point (simplified - use center for now)
            grasp_info = {
                'x': 0.5,  # Center of bounding box
                'y': 0.5,
                'approach_angle': 0.0,
                'quality': 0.8
            }
            
            # Calculate grasp point 3D coordinates
            grasp_pixel_x = x1 + int(grasp_info['x'] * (x2 - x1))
            grasp_pixel_y = y1 + int(grasp_info['y'] * (y2 - y1))
            grasp_point_3d_camera = self.pixel_to_3d_camera_frame(grasp_pixel_x, grasp_pixel_y, depth_value)
            grasp_point_3d_robot = self.camera_to_robot_frame(grasp_point_3d_camera)
            
            # Validate workspace constraints
            is_reachable = self.workspace_validator.is_reachable(
                grasp_point_3d_robot[0], grasp_point_3d_robot[1], grasp_point_3d_robot[2]
            )
            
            detection_3d = {
                'label': label,
                'confidence': confidence,
                'bbox_2d': bbox,
                'center_2d': (center_x, center_y),
                'depth_value': depth_value,
                'point_3d_camera': point_3d_camera,
                'point_3d_robot': point_3d_robot,
                'grasp_point_3d_camera': grasp_point_3d_camera,
                'grasp_point_3d_robot': grasp_point_3d_robot,
                'is_reachable': is_reachable,
                'grasp_info': grasp_info,
                'depth_map': depth_map
            }
            
            detections_3d.append(detection_3d)
            
            # Log the transformation
            self.logger.info(f"📍 {label}: 2D({center_x},{center_y}) → 3D_cam{point_3d_camera} → 3D_robot{point_3d_robot}")
            
        return detections_3d
    
    def visualize_results(self, frame: np.ndarray, detections_3d: List[Dict[str, Any]]) -> np.ndarray:
        """Create comprehensive visualization of detection and transformation results"""
        
        vis_frame = frame.copy()
        
        for detection in detections_3d:
            label = detection['label']
            bbox = detection['bbox_2d']
            center_2d = detection['center_2d']
            point_3d_robot = detection['point_3d_robot']
            is_reachable = detection['is_reachable']
            
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            color = (0, 255, 0) if is_reachable else (0, 0, 255)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cv2.circle(vis_frame, center_2d, 5, color, -1)
            
            # Draw grasp point
            grasp_info = detection['grasp_info']
            grasp_x = x1 + int(grasp_info['x'] * (x2 - x1))
            grasp_y = y1 + int(grasp_info['y'] * (y2 - y1))
            cv2.circle(vis_frame, (grasp_x, grasp_y), 3, (255, 0, 255), -1)
            
            # Add text information
            status = "REACHABLE" if is_reachable else "UNREACHABLE"
            text_lines = [
                f"{label} ({detection['confidence']:.2f})",
                f"Depth: {detection['depth_value']:.3f}m",
                f"3D: ({point_3d_robot[0]:.3f}, {point_3d_robot[1]:.3f}, {point_3d_robot[2]:.3f})",
                f"Status: {status}"
            ]
            
            for i, text in enumerate(text_lines):
                y_offset = y1 - 10 - (len(text_lines) - 1 - i) * 20
                cv2.putText(vis_frame, text, (x1, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add coordinate system info
        info_text = [
            f"Camera: {self.camera_width}x{self.camera_height}",
            f"Gripper pose: ({self.mock_gripper_pose[0,3]:.2f}, {self.mock_gripper_pose[1,3]:.2f}, {self.mock_gripper_pose[2,3]:.2f})",
            f"Detections: {len(detections_3d)}",
            f"Queries: {', '.join(self.current_queries) if self.current_queries else 'None'}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(vis_frame, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_frame
    
    def create_depth_visualization(self, detections_3d: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Create depth map visualization"""
        
        if not detections_3d:
            return None
        
        # Use depth map from first detection (they should be similar)
        depth_map = detections_3d[0]['depth_map']
        
        # Normalize depth for visualization
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        # Add depth scale
        cv2.putText(depth_colored, f"Depth: {np.min(depth_map):.2f}m - {np.max(depth_map):.2f}m", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return depth_colored
    
    def save_test_result(self, detections_3d: List[Dict[str, Any]]):
        """Save test results for analysis"""
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        result = {
            'timestamp': timestamp,
            'queries': self.current_queries.copy(),
            'gripper_pose': self.mock_gripper_pose.tolist(),
            'camera_matrix': self.camera_matrix.tolist(),
            'detections': []
        }
        
        for detection in detections_3d:
            result['detections'].append({
                'label': detection['label'],
                'confidence': detection['confidence'],
                'bbox_2d': detection['bbox_2d'],
                'center_2d': detection['center_2d'],
                'depth_value': detection['depth_value'],
                'point_3d_camera': detection['point_3d_camera'].tolist(),
                'point_3d_robot': detection['point_3d_robot'].tolist(),
                'grasp_point_3d_robot': detection['grasp_point_3d_robot'].tolist(),
                'is_reachable': detection['is_reachable']
            })
        
        self.test_results.append(result)
        
        # Save to file
        with open(f'test_results_{timestamp}.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        self.logger.info(f"💾 Test results saved to test_results_{timestamp}.json")
    
    def run_interactive_test(self):
        """Run interactive testing session"""
        
        self.logger.info("🚀 Starting 2D-3D Coordinate Transformation Test")
        self.logger.info("="*60)
        self.logger.info("Controls:")
        self.logger.info("  SPACE - Capture and process current frame")
        self.logger.info("  's' - Get speech input (if available)")
        self.logger.info("  'd' - Toggle depth visualization")
        self.logger.info("  'r' - Reset queries")
        self.logger.info("  'q' - Quit")
        self.logger.info("="*60)
        
        self.running = True
        
        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    self.logger.error("Failed to read camera frame")
                    break
                
                # Process current frame if we have queries
                if self.current_queries:
                    detections_2d = self.detect_objects_2d(frame, self.current_queries)
                    
                    if detections_2d:
                        detections_3d = self.convert_2d_to_3d(detections_2d, frame)
                        self.current_detections = detections_3d
                        
                        # Create visualizations
                        vis_frame = self.visualize_results(frame, detections_3d)
                        
                        if self.show_depth_visualization:
                            depth_vis = self.create_depth_visualization(detections_3d)
                            if depth_vis is not None:
                                cv2.imshow("Depth Visualization", depth_vis)
                    else:
                        vis_frame = frame.copy()
                        cv2.putText(vis_frame, f"Searching for: {', '.join(self.current_queries)}", 
                                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    vis_frame = frame.copy()
                    cv2.putText(vis_frame, "Press 's' for speech input or enter manual query", 
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow("2D-3D Transformation Test", vis_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    if self.current_detections:
                        self.save_test_result(self.current_detections)
                elif key == ord('s'):
                    queries = self.process_speech_input()
                    if queries:
                        self.current_queries = queries
                        self.logger.info(f"🎯 New queries: {queries}")
                elif key == ord('d'):
                    self.show_depth_visualization = not self.show_depth_visualization
                    self.logger.info(f"Depth visualization: {'ON' if self.show_depth_visualization else 'OFF'}")
                elif key == ord('r'):
                    self.current_queries = []
                    self.current_detections = []
                    self.logger.info("🔄 Queries reset")
                
        except KeyboardInterrupt:
            self.logger.info("Test interrupted by user")
        
        finally:
            self.cleanup()
    
    def run_batch_test(self, test_objects: List[str], num_samples: int = 5):
        """Run automated batch test with predefined objects"""
        
        self.logger.info(f"🔬 Running batch test with {len(test_objects)} objects, {num_samples} samples each")
        
        all_results = []
        
        for obj in test_objects:
            self.logger.info(f"\n📋 Testing object: {obj}")
            self.current_queries = [obj]
            
            object_results = []
            
            for sample in range(num_samples):
                self.logger.info(f"  Sample {sample + 1}/{num_samples}")
                
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                # Process
                detections_2d = self.detect_objects_2d(frame, [obj])
                
                if detections_2d:
                    detections_3d = self.convert_2d_to_3d(detections_2d, frame)
                    
                    if detections_3d:
                        detection = detections_3d[0]  # Take first detection
                        
                        result = {
                            'object': obj,
                            'sample': sample,
                            'confidence': detection['confidence'],
                            'depth': detection['depth_value'],
                            'point_3d_robot': detection['point_3d_robot'].tolist(),
                            'is_reachable': detection['is_reachable']
                        }
                        
                        object_results.append(result)
                        self.logger.info(f"    ✅ Detected at {detection['point_3d_robot']}")
                    else:
                        self.logger.info(f"    ❌ No 3D conversion")
                else:
                    self.logger.info(f"    ❌ Not detected")
                
                time.sleep(0.5)  # Brief pause between samples
            
            all_results.extend(object_results)
        
        # Save batch results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        with open(f'batch_test_results_{timestamp}.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Print summary
        self.print_batch_summary(all_results, test_objects)
        
        return all_results
    
    def print_batch_summary(self, results: List[Dict], test_objects: List[str]):
        """Print summary of batch test results"""
        
        print("\n" + "="*60)
        print("📊 BATCH TEST SUMMARY")
        print("="*60)
        
        for obj in test_objects:
            obj_results = [r for r in results if r['object'] == obj]
            
            if obj_results:
                confidences = [r['confidence'] for r in obj_results]
                depths = [r['depth'] for r in obj_results]
                reachable_count = sum(1 for r in obj_results if r['is_reachable'])
                
                print(f"\n🎯 {obj.upper()}:")
                print(f"  Detections: {len(obj_results)}")
                print(f"  Avg confidence: {np.mean(confidences):.3f} ± {np.std(confidences):.3f}")
                print(f"  Avg depth: {np.mean(depths):.3f}m ± {np.std(depths):.3f}m")
                print(f"  Reachable: {reachable_count}/{len(obj_results)} ({reachable_count/len(obj_results)*100:.1f}%)")
                
                # Show coordinate consistency
                positions = [r['point_3d_robot'] for r in obj_results]
                if len(positions) > 1:
                    pos_array = np.array(positions)
                    pos_std = np.std(pos_array, axis=0)
                    print(f"  Position std: ({pos_std[0]:.3f}, {pos_std[1]:.3f}, {pos_std[2]:.3f})m")
            else:
                print(f"\n❌ {obj.upper()}: No detections")
        
        print("\n" + "="*60)
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        
        if hasattr(self, 'camera'):
            self.camera.release()
        
        cv2.destroyAllWindows()
        
        if self.speech_processor:
            try:
                self.speech_processor.cleanup()
            except:
                pass
        
        self.logger.info("🧹 Cleanup complete")

def main():
    """Main test function"""
    
    print("🔬 2D-3D Coordinate Transformation Tester")
    print("="*50)
    print("This test validates:")
    print("✅ Speech Recognition → Object Queries")
    print("✅ MacBook Camera → RGB Frames") 
    print("✅ OWL-ViT → 2D Bounding Boxes")
    print("✅ Mock Depth + Calibration → 3D Coordinates")
    print("✅ Coordinate Frame Transformations")
    print("✅ Workspace Validation")
    print("="*50)
    
    # Ask user for test mode
    print("\nSelect test mode:")
    print("1. Interactive test (manual control)")
    print("2. Batch test (automated)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    try:
        if choice == "2":
            # Batch test mode
            test_objects = ['bottle', 'cup', 'book', 'phone', 'box']
            tester = CoordinateTransformationTester(use_speech=False)
            tester.run_batch_test(test_objects, num_samples=3)
        else:
            # Interactive test mode
            use_speech = input("Enable speech recognition? (y/n): ").lower().startswith('y')
            tester = CoordinateTransformationTester(use_speech=use_speech)
            tester.run_interactive_test()
            
    except KeyboardInterrupt:
        print("\n🛑 Test stopped by user")
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        logging.error(f"Test error: {e}", exc_info=True)
    finally:
        try:
            tester.cleanup()
        except:
            pass

if __name__ == "__main__":
    main() 