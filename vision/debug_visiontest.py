"""
Vision System Debug and Testing Module.

This module provides tools for debugging and testing the vision system:
- Camera setup and configuration testing
- Visual model inference testing
- Performance benchmarking
- Visualization utilities

The module helps identify and diagnose issues in:
- Camera connectivity and settings
- Object detection accuracy
- System latency and performance
- Integration points between components
"""

import cv2
import numpy as np
import time
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ObjectDetection import ObjectDetection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImprovedDebugTest:
    """
    Enhanced vision system debugging and testing.
    
    This class provides tools for testing and debugging various components
    of the vision system, with detailed logging and visualization.
    
    Parameters
    ----------
    log_level : str, optional
        Logging level, by default "INFO"
    save_results : bool, optional
        Whether to save test results, by default True
    
    Attributes
    ----------
    camera : cv2.VideoCapture
        OpenCV camera capture object
    logger : logging.Logger
        Logger for debug information
    test_results : Dict
        Dictionary storing test results
        
    Notes
    -----
    Supports testing of:
    - Camera setup and streaming
    - Object detection models
    - Depth estimation
    - System latency
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Starting Debug Test with BBox Accuracy Testing")
        
        try:
            self.detector = ObjectDetection('yolov5s', confidence_threshold=0.4)
            self.logger.info("Detector initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize detector: {e}")
            raise
        
        # Camera setup
        self.camera = None
        self.setup_camera()
        
        # Debug settings
        self.show_preprocessing = False
        self.show_edge_detection = False
        self.enable_bbox_refinement = False
        self.frame_count = 0
        
        # Performance tracking
        self.bbox_accuracy_scores = []
        self.detection_times = []
        
    def setup_camera(self):
        """
        Setup camera with debug info.
        
        This function:
        - Tests multiple camera indices
        - Configures optimal camera settings
        - Verifies frame capture
        - Warms up camera
        
        Raises
        ------
        RuntimeError
            If no working camera is found
            
        Notes
        -----
        Tests camera indices 0-2
        Sets resolution to 1280x720
        Sets frame rate to 30 FPS
        Configures exposure settings
        """
        for idx in [0, 1, 2]:
            try:
                self.logger.info(f"Trying camera {idx}...")
                cap = cv2.VideoCapture(idx)
                
                if not cap.isOpened():
                    self.logger.warning(f"Camera {idx} not opened")
                    cap.release()
                    continue
                
                # Set optimal camera settings
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  
                cap.set(cv2.CAP_PROP_EXPOSURE, -6) 
                
                # Test frame capture
                ret, frame = cap.read()
                if not ret or frame is None:
                    self.logger.warning(f"Camera {idx} can't capture frames")
                    cap.release()
                    continue
                
                self.logger.info(f"Camera {idx} working - Frame shape: {frame.shape}")
                
                # Warm up camera
                for _ in range(10):
                    cap.read()
                
                self.camera = cap
                return
                
            except Exception as e:
                self.logger.error(f"Camera {idx} error: {e}")
                continue
        
        raise RuntimeError("No camera found!")
    
    def run(self):
        """Enhanced debug main loop"""
        self.logger.info("Starting debug loop...")
        self.logger.info("Commands:")
        self.logger.info("  Q - Quit")
        self.logger.info("  P - Toggle preprocessing view")
        self.logger.info("  E - Toggle edge detection view")
        self.logger.info("  R - Toggle bbox refinement")
        self.logger.info("  S - Save screenshot")
        self.logger.info("  T - Test bbox accuracy")
        self.logger.info("  B - Run benchmark")
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    self.logger.warning("No frame captured")
                    continue
                
                self.frame_count += 1
                
                # Process frame with timing
                start_time = time.perf_counter()
                processed_frame = self.debug_process_frame(frame.copy())
                processing_time = time.perf_counter() - start_time
                self.detection_times.append(processing_time)
                
                # Display different views based on mode
                if self.show_preprocessing:
                    preprocessed = self.detector.preprocess_frame(frame)
                    cv2.imshow('Preprocessed Input', preprocessed)
                
                if self.show_edge_detection:
                    self.show_edge_detection_view(frame)
                
                # Main display
                cv2.imshow('Debug Test - Enhanced BBox Accuracy', processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    self.show_preprocessing = not self.show_preprocessing
                    self.logger.info(f"Preprocessing view: {'ON' if self.show_preprocessing else 'OFF'}")
                    if not self.show_preprocessing:
                        cv2.destroyWindow('Preprocessed Input')
                elif key == ord('e'):
                    self.show_edge_detection = not self.show_edge_detection
                    self.logger.info(f"Edge detection view: {'ON' if self.show_edge_detection else 'OFF'}")
                    if not self.show_edge_detection:
                        cv2.destroyWindow('Edge Detection')
                elif key == ord('r'):
                    self.enable_bbox_refinement = not self.enable_bbox_refinement
                    self.logger.info(f"BBox refinement: {'ON' if self.enable_bbox_refinement else 'OFF'}")
                elif key == ord('s'):
                    self.save_screenshot(processed_frame)
                elif key == ord('t'):
                    self.test_bbox_accuracy(frame)
                elif key == ord('b'):
                    self.run_benchmark(frame)
                
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            self.cleanup()
    
    def debug_process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhanced frame processing with bbox accuracy improvements"""
        start_time = time.perf_counter()
        
        try:
            # Get detections
            detections = self.detector.detect_objects(frame)
            detection_time = time.perf_counter() - start_time
            
            bbox_quality_scores = []
            
            for i, (label, confidence, bbox) in enumerate(detections):
                original_bbox = bbox.copy()
                
                # Optional bbox refinement
                if self.enable_bbox_refinement:
                    refined_bbox = self.detector.refine_bounding_box(frame, bbox)
                    bbox = refined_bbox
                
                quality_score = self.calculate_bbox_quality(frame, bbox)
                bbox_quality_scores.append(quality_score)
                
                # Determine color based on quality
                if quality_score > 0.8:
                    color = (0, 255, 0)  # Excellent 
                elif quality_score > 0.6:
                    color = (0, 255, 255)  # good
                else:
                    color = (0, 0, 255)  # poor
                
                # Draw enhanced bounding box
                self.detector.draw_enhanced_bbox(frame, label, confidence, bbox, color)
                
                # Show refinement comparison if enabled
                if self.enable_bbox_refinement and original_bbox != bbox:
                    self.draw_bbox_comparison(frame, original_bbox, bbox)
                
                x1, y1, x2, y2 = bbox
                quality_text = f"Q: {quality_score:.2f}"
                cv2.putText(frame, quality_text, (x1, y2 + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            if bbox_quality_scores:
                avg_quality = np.mean(bbox_quality_scores)
                self.bbox_accuracy_scores.append(avg_quality)
            
            # Draw comprehensive debug overlay
            self.draw_debug_overlay(frame, len(detections), detection_time, 
                                  np.mean(bbox_quality_scores) if bbox_quality_scores else 0.0)
            
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            cv2.putText(frame, f"ERROR: {str(e)}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def calculate_bbox_quality(self, frame: np.ndarray, bbox: list) -> float:
        """Calculate bounding box quality score based on multiple factors"""
        x1, y1, x2, y2 = bbox
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        # Factor 1: Edge density (higher is better for object boundaries)
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        edge_score = min(1.0, edge_density * 10)  # Normalize
        
        # Factor 2: Contrast (higher is better for distinct objects)
        contrast = np.std(gray_roi) / 255.0
        contrast_score = min(1.0, contrast * 2)
        
        # Factor 3: Aspect ratio reasonableness
        width, height = x2 - x1, y2 - y1
        aspect_ratio = width / height
        aspect_score = 1.0 if 0.2 <= aspect_ratio <= 5.0 else 0.5
        
        # Factor 4: Size reasonableness (not too small or too large)
        area = width * height
        frame_area = frame.shape[0] * frame.shape[1]
        size_ratio = area / frame_area
        size_score = 1.0 if 0.01 <= size_ratio <= 0.5 else 0.5
        
        # Weighted combination
        quality_score = (
            edge_score * 0.4 +
            contrast_score * 0.3 +
            aspect_score * 0.2 +
            size_score * 0.1
        )
        
        return min(1.0, quality_score)
    
    def draw_bbox_comparison(self, frame: np.ndarray, original_bbox: list, 
                           refined_bbox: list):
        """Draw comparison between original and refined bounding boxes"""
        # Draw original in dashed red
        x1o, y1o, x2o, y2o = original_bbox
        cv2.rectangle(frame, (x1o, y1o), (x2o, y2o), (0, 0, 255), 1, cv2.LINE_4)
        
        # Draw refined in solid green (already drawn by main function)
        # Just add a small indicator
        cv2.putText(frame, "R", (refined_bbox[0] - 15, refined_bbox[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    def show_edge_detection_view(self, frame: np.ndarray):
        """Show edge detection visualization"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        edge_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        cv2.imshow('Edge Detection', edge_colored)
    
    def draw_debug_overlay(self, frame: np.ndarray, num_detections: int, 
                          detection_time: float, avg_quality: float):
        """Draw comprehensive debug information"""
        h, w = frame.shape[:2]
        
        # Background for debug info
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        y_offset = 30
        line_height = 22
        
        # Debug information
        debug_info = [
            f"Frame: {self.frame_count}",
            f"Detections: {num_detections}",
            f"Detection time: {detection_time*1000:.1f}ms",
            f"Avg quality score: {avg_quality:.3f}",
            f"Preprocessing: {'ON' if self.show_preprocessing else 'OFF'}",
            f"Edge detection: {'ON' if self.show_edge_detection else 'OFF'}",
            f"BBox refinement: {'ON' if self.enable_bbox_refinement else 'OFF'}",
        ]
        
        # Overall quality assessment
        if len(self.bbox_accuracy_scores) > 10:
            recent_avg = np.mean(self.bbox_accuracy_scores[-10:])
            if recent_avg > 0.8:
                quality_status = "EXCELLENT"
                quality_color = (0, 255, 0)
            elif recent_avg > 0.6:
                quality_status = "GOOD"
                quality_color = (0, 255, 255)
            else:
                quality_status = "NEEDS IMPROVEMENT"
                quality_color = (0, 0, 255)
            
            debug_info.append(f"Overall quality: {quality_status}")
        
        for i, info in enumerate(debug_info):
            color = (255, 255, 255)
            if "quality:" in info.lower() and len(self.bbox_accuracy_scores) > 10:
                color = quality_color
            cv2.putText(frame, info, (15, y_offset + i*line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Controls reminder
        cv2.putText(frame, "P=Preprocess | E=Edges | R=Refine | T=Test | B=Benchmark", 
                   (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    def test_bbox_accuracy(self, frame: np.ndarray):
        """Run comprehensive bbox accuracy test"""
        self.logger.info("Running BBox Accuracy Test...")
        
        # Test with different confidence thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        results = {}
        
        # Store original threshold
        original_threshold = self.detector.confidence_threshold
        original_model_conf = self.detector.model.conf
        
        for thresh in thresholds:
            self.detector.confidence_threshold = thresh
            self.detector.model.conf = thresh
            
            detections = self.detector.detect_objects(frame)
            
            quality_scores = []
            for _, _, bbox in detections:
                quality = self.calculate_bbox_quality(frame, bbox)
                quality_scores.append(quality)
            
            avg_quality = np.mean(quality_scores) if quality_scores else 0.0
            results[thresh] = {
                'count': len(detections),
                'avg_quality': avg_quality
            }
        
        # Restore original settings
        self.detector.confidence_threshold = original_threshold
        self.detector.model.conf = original_model_conf
        
        print("\n" + "="*60)
        print("BBOX ACCURACY TEST RESULTS")
        print("="*60)
        print(f"{'Threshold':<12} {'Detections':<12} {'Avg Quality':<12} {'Status'}")
        print("-" * 60)
        
        for thresh, result in results.items():
            count = result['count']
            quality = result['avg_quality']
            
            if quality > 0.8:
                status = "EXCELLENT"
            elif quality > 0.6:
                status = "GOOD"
            elif quality > 0.4:
                status = "FAIR"
            else:
                status = "POOR"
            
            print(f"{thresh:<12.1f} {count:<12} {quality:<12.3f} {status}")
        
        # Find optimal threshold
        best_thresh = max(results.keys(), key=lambda t: results[t]['avg_quality'])
        print(f"\n🏆 Best threshold: {best_thresh} (Quality: {results[best_thresh]['avg_quality']:.3f})")
        print("="*60 + "\n")
    
    def run_benchmark(self, frame: np.ndarray):
        """Run performance benchmark"""
        self.logger.info("Performance benchmark...")
        
        times = []
        num_runs = 50
        
        print(f"Running {num_runs} detection cycles...")
        
        for i in range(num_runs):
            start_time = time.perf_counter()
            detections = self.detector.detect_objects(frame)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{num_runs}")
        
        times = np.array(times)
        
        print("\n" + "="*60)
        print("⚡ PERFORMANCE BENCHMARK RESULTS")
        print("="*60)
        print(f"Device: {self.detector.device}")
        print(f"Model: YOLOv5s")
        print(f"Runs: {num_runs}")
        print(f"Mean inference time: {np.mean(times)*1000:.2f} ms")
        print(f"Std inference time: {np.std(times)*1000:.2f} ms")
        print(f"Min inference time: {np.min(times)*1000:.2f} ms")
        print(f"Max inference time: {np.max(times)*1000:.2f} ms")
        print(f"Average FPS: {1.0/np.mean(times):.1f}")
        print(f"95th percentile: {np.percentile(times, 95)*1000:.2f} ms")
        print("="*60 + "\n")
    
    def save_screenshot(self, frame: np.ndarray):
        """Save current frame as screenshot"""
        timestamp = int(time.time())
        filename = f"debug_test_screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        self.logger.info(f"📸 Screenshot saved: {filename}")
    
    def print_final_stats(self):
        """Print final statistics"""
        if len(self.bbox_accuracy_scores) > 0:
            print("\n" + "="*60)
            print("📊 FINAL DEBUG SESSION STATISTICS")
            print("="*60)
            print(f"Total frames processed: {self.frame_count}")
            print(f"Average quality score: {np.mean(self.bbox_accuracy_scores):.3f}")
            print(f"Best quality score: {np.max(self.bbox_accuracy_scores):.3f}")
            print(f"Worst quality score: {np.min(self.bbox_accuracy_scores):.3f}")
            
            if len(self.detection_times) > 0:
                print(f"Average detection time: {np.mean(self.detection_times)*1000:.2f} ms")
                print(f"Average FPS: {1.0/np.mean(self.detection_times):.1f}")
            
            # Quality distribution
            excellent = sum(1 for score in self.bbox_accuracy_scores if score > 0.8)
            good = sum(1 for score in self.bbox_accuracy_scores if 0.6 < score <= 0.8)
            fair = sum(1 for score in self.bbox_accuracy_scores if 0.4 < score <= 0.6)
            poor = sum(1 for score in self.bbox_accuracy_scores if score <= 0.4)
            
            total = len(self.bbox_accuracy_scores)
            print(f"\nQuality Distribution:")
            print(f"Excellent (>0.8): {excellent}/{total} ({excellent/total*100:.1f}%)")
            print(f"Good (0.6-0.8): {good}/{total} ({good/total*100:.1f}%)")
            print(f"Fair (0.4-0.6): {fair}/{total} ({fair/total*100:.1f}%)")
            print(f"Poor (<0.4): {poor}/{total} ({poor/total*100:.1f}%)")
            print("="*60)
    
    def cleanup(self):
        """Clean up resources"""
        self.logger.info("🧹 Cleaning up...")
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        self.print_final_stats()
        self.logger.info("✅ Debug test cleanup complete")

def main():
    """Main function"""
    print("Enhanced Debug Vision Test")
    print("This will test bounding box accuracy and performance")
    print("Make sure your camera is connected!")
    print("\nTesting features:")
    print("• BBox quality scoring")
    print("• Edge detection visualization")
    print("• Preprocessing inspection")
    print("• Performance benchmarking")
    print("• Multi-threshold testing")
    
    try:
        debug_test = ImprovedDebugTest()
        debug_test.run()
    except Exception as e:
        logging.error(f"Debug test failed: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())