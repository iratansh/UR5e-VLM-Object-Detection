"""
Accuracy Tester for Vision System
Integrates with existing OWL-ViT detection system to provide comprehensive accuracy evaluation.
"""

import cv2
import numpy as np
import time
import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    from OWLViTDetector import OWLViTDetector
    from SpeechCommandProcessor import SpeechCommandProcessor
    from CameraCalibration import CameraCalibration
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    print("Make sure your modules are in the Python path")

@dataclass
class DetectionResult:
    query: str
    expected_count: int
    detected_count: int
    confidence_scores: List[float]
    processing_time: float
    accuracy_score: float

@dataclass
class CoordinateResult:
    actual_coords: Tuple[float, float, float]
    detected_coords: Tuple[float, float, float]
    euclidean_error: float
    axis_errors: Tuple[float, float, float]
    accuracy_score: float

@dataclass
class SpeechResult:
    command: str
    recognized: bool
    processing_time: float
    confidence: float

class VisionSystemTester:
    def __init__(self, use_camera: bool = True):
        """
        Initialize the VisionSystemTester

        Args:
            use_camera (bool, optional): Whether to use the camera for testing. Defaults to True.
        """
        self.logger = logging.getLogger(__name__)
        self.setup_components(use_camera)
        self.test_results = {
            'detection': [],
            'coordinates': [],
            'speech': [],
            'system_info': self.get_system_info()
        }
        
    def setup_components(self, use_camera: bool=True):
        """
        Initialize the vision system components

        Args:
            use_camera (bool): Whether to use the camera for testing
        """
        try:
            self.detector = OWLViTDetector()
            self.logger.info("OWL-ViT detector loaded")
        except Exception as e:
            self.logger.error(f"Failed to load detector: {e}")
            self.detector = None
            
        try:
            self.speech = SpeechCommandProcessor()
            self.logger.info("Speech processor loaded")
        except Exception as e:
            self.logger.error(f"Failed to load speech processor: {e}")
            self.speech = None
            
        self.calibration = CameraCalibration()
        
        if use_camera:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.logger.warning("Camera not available, using test images")
                self.cap = None
        else:
            self.cap = None
    
    def get_system_info(self) -> Dict:
        """
        Get the system information for testing

        Returns:
            Dict: A dictionary with the following information:
                - timestamp: The current timestamp in ISO format
                - detector_available: Whether the detector is available
                - speech_available: Whether the speech processor is available
                - camera_available: Whether the camera is available
                - camera_matrix: The camera calibration matrix as a list
                - distortion_coeffs: The distortion coefficients as a list
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'detector_available': self.detector is not None,
            'speech_available': self.speech is not None,
            'camera_available': self.cap is not None,
            'camera_matrix': self.calibration.camera_matrix.tolist(),
            'distortion_coeffs': self.calibration.dist_coeffs.tolist()
        }
    
    def test_detection_accuracy(self, test_scenarios: List[Dict]) -> List[DetectionResult]:
        """
        Test the accuracy of the detection system for a given list of scenarios
        
        Args:
            test_scenarios (List[Dict]): A list of dictionaries containing the following information:
                - query (str): The natural language query to test
                - expected_count (int, optional): The expected number of detections. Defaults to 1
                - threshold (float, optional): The confidence threshold for detection. Defaults to 0.1
        
        Returns:
            List[DetectionResult]: A list of DetectionResult objects containing the results of the test
        """
        results = []
        
        if not self.detector:
            self.logger.error("Detector not available")
            return results
            
        for scenario in test_scenarios:
            self.logger.info(f"Testing detection: {scenario['query']}")
            
            # Get test image
            if self.cap:
                ret, frame = self.cap.read()
                if not ret:
                    continue
            else:
                # Create synthetic test image
                frame = self.create_test_image(scenario['query'])
            
            start_time = time.time()
            detections = self.detector.detect_with_text_queries(
                frame, 
                [scenario['query']], 
                confidence_threshold=scenario.get('threshold', 0.1)
            )
            processing_time = time.time() - start_time
            
            # Analyze results
            detected_count = len(detections)
            expected_count = scenario.get('expected_count', 1)
            confidence_scores = [det[1] for det in detections]
            
            # Calculate accuracy (how close detected count is to expected)
            count_accuracy = 1.0 - abs(detected_count - expected_count) / max(expected_count, 1)
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            accuracy_score = (count_accuracy + avg_confidence) / 2.0
            
            result = DetectionResult(
                query=scenario['query'],
                expected_count=expected_count,
                detected_count=detected_count,
                confidence_scores=confidence_scores,
                processing_time=processing_time,
                accuracy_score=accuracy_score
            )
            
            results.append(result)
            self.test_results['detection'].append(asdict(result))
            
            self.logger.info(f"Result: {detected_count}/{expected_count} objects, "
                           f"avg confidence: {avg_confidence:.3f}, "
                           f"accuracy: {accuracy_score:.3f}")
        
        return results
    
    def test_coordinate_accuracy(self, test_points: List[Dict]) -> List[CoordinateResult]:   
        """
        Test 3D coordinate accuracy using camera calibration
        
        Args:
            test_points: List of dictionaries containing test points, where each dictionary
                contains the following keys:
                    - actual (Tuple[float, float, float]): The actual 3D coordinates of the point
                    - pixel (Tuple[int, int]): The pixel coordinates of the point in the image
                    - depth (float, optional): The depth of the point in meters. Defaults to the actual z coordinate
        
        Returns:
            List[CoordinateResult]: A list of CoordinateResult objects containing the results of the test
        """
        results = []
        
        for point in test_points:
            actual = point['actual']  # (x, y, z)
            pixel_coords = point['pixel']  # (px, py)
            depth = point.get('depth', actual[2])
            
            # Calculate detected coordinates using camera calibration
            detected = self.calibration.pixel_to_world(
                pixel_coords[0], pixel_coords[1], depth
            )
            
            # Calculate errors
            axis_errors = (
                abs(actual[0] - detected[0]),
                abs(actual[1] - detected[1]),
                abs(actual[2] - detected[2])
            )
            euclidean_error = np.sqrt(sum(e**2 for e in axis_errors))
            
            # Calculate accuracy (inverse of normalized error)
            max_error = 0.5  # 50cm maximum expected error
            accuracy_score = max(0.0, 1.0 - euclidean_error / max_error)
            
            result = CoordinateResult(
                actual_coords=actual,
                detected_coords=detected,
                euclidean_error=euclidean_error,
                axis_errors=axis_errors,
                accuracy_score=accuracy_score
            )
            
            results.append(result)
            self.test_results['coordinates'].append(asdict(result))
            
            self.logger.info(f"3D accuracy: {euclidean_error*1000:.1f}mm error, "
                           f"score: {accuracy_score:.3f}")
        
        return results
    
    def test_speech_recognition(self, test_commands: List[str], duration: float = 30.0) -> List[SpeechResult]:
        """
        Test speech recognition accuracy.

        Parameters
        ----------
        test_commands : List[str]
            List of commands to test recognition accuracy
        duration : float, optional
            Test duration in seconds, by default 30.0

        Returns
        -------
        List[SpeechResult]
            List of SpeechResult objects containing the results of the test
        """
        results = []
        
        if not self.speech:
            self.logger.error("Speech processor not available")
            return results
        
        self.speech.start_listening()
        self.logger.info(f"Testing speech recognition for {duration} seconds")
        self.logger.info("Say the following commands:")
        for i, cmd in enumerate(test_commands, 1):
            print(f"{i}. {cmd}")
        
        start_time = time.time()
        recognized_commands = []
        
        while time.time() - start_time < duration:
            command = self.speech.get_command()
            if command:
                recognized_commands.append({
                    'command': command,
                    'timestamp': time.time() - start_time
                })
                self.logger.info(f"Recognized: '{command}'")
            time.sleep(0.1)
        
        self.speech.stop_listening()
        
        # Analyze recognition accuracy
        for test_cmd in test_commands:
            recognized = any(test_cmd.lower() in rec['command'].lower() 
                           for rec in recognized_commands)
            
            result = SpeechResult(
                command=test_cmd,
                recognized=recognized,
                processing_time=0.0,  # Not implemented in this example
                confidence=1.0 if recognized else 0.0
            )
            
            results.append(result)
            self.test_results['speech'].append(asdict(result))
        
        return results
    
    def create_test_image(self, object_type: str) -> np.ndarray:
        """
        Create a test image for the given object type. This is a simple example image that includes some basic shapes that might be detected by the object detector.

        Parameters
        ----------
        object_type : str
            Type of object to include in the image

        Returns
        -------
        np.ndarray
            Test image as a numpy array
        """
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Add some basic shapes that might be detected
        if 'bottle' in object_type.lower():
            cv2.rectangle(img, (200, 150), (250, 350), (100, 150, 200), -1)
        elif 'cup' in object_type.lower():
            cv2.circle(img, (320, 240), 50, (150, 100, 50), -1)
        elif 'phone' in object_type.lower():
            cv2.rectangle(img, (280, 200), (360, 320), (50, 50, 50), -1)
        
        # Add some noise
        noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        return img
    
    def run_comprehensive_test(self, config_file: Optional[str] = None) -> Dict:
        """
        Run a comprehensive test of the vision system, including object detection, coordinate accuracy, and speech recognition.
        
        Parameters
        ----------
        config_file : Optional[str], optional
            Path to a JSON configuration file. If not provided, a default configuration will be used.
        
        Returns
        -------
        Dict
            A dictionary containing the test results.
        """
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            config = self.get_default_test_config()
        
        self.logger.info("Starting vision system test")
        
        # Test object detection
        if config.get('test_detection', True):
            detection_results = self.test_detection_accuracy(config['detection_scenarios'])
            self.logger.info(f"Detection test completed: {len(detection_results)} scenarios")
        
        # Test coordinate accuracy
        if config.get('test_coordinates', True):
            coordinate_results = self.test_coordinate_accuracy(config['coordinate_tests'])
            self.logger.info(f"Coordinate test completed: {len(coordinate_results)} points")
        
        # Test speech recognition
        if config.get('test_speech', True) and self.speech:
            speech_results = self.test_speech_recognition(
                config['speech_commands'], 
                config.get('speech_duration', 30.0)
            )
            self.logger.info(f"Speech test completed: {len(speech_results)} commands")
        
        return self.generate_report()
    
    def get_default_test_config(self) -> Dict:
        """Get default test configuration"""
        return {
            'test_detection': True,
            'test_coordinates': True,
            'test_speech': True,
            'detection_scenarios': [
                {'query': 'bottle', 'expected_count': 1, 'threshold': 0.1},
                {'query': 'cup', 'expected_count': 1, 'threshold': 0.1},
                {'query': 'phone', 'expected_count': 1, 'threshold': 0.1},
                {'query': 'water bottle', 'expected_count': 1, 'threshold': 0.2},
                {'query': 'graspable object', 'expected_count': 2, 'threshold': 0.1}
            ],
            'coordinate_tests': [
                {'actual': (0.3, 0.1, 0.3), 'pixel': (400, 300), 'depth': 0.3},
                {'actual': (0.0, 0.0, 0.5), 'pixel': (320, 240), 'depth': 0.5},
                {'actual': (-0.2, 0.2, 0.4), 'pixel': (200, 180), 'depth': 0.4},
                {'actual': (0.4, -0.1, 0.25), 'pixel': (500, 320), 'depth': 0.25}
            ],
            'speech_commands': [
                'pick up the bottle',
                'find the cup',
                'grab the phone',
                'locate the keys',
                'get the remote'
            ],
            'speech_duration': 25.0
        }
    
    def generate_report(self) -> Dict:
        """
        Generate a comprehensive report of the vision system accuracy tests.

        This function compiles the results from various accuracy tests, including
        object detection, 3D coordinate accuracy, and speech recognition, into a
        structured dictionary. The report includes detailed results, summary
        statistics for each test type, and system improvement recommendations.

        Returns:
            Dict: A dictionary containing the following keys:
                - system_info: Information about the system setup and configuration.
                - summary: Summary statistics for detection, coordinate, and speech
                        accuracy tests, such as mean, standard deviation, and count.
                - detailed_results: The raw test results, including individual scores
                                    for each test scenario.
                - recommendations: Suggested actions to improve system accuracy based
                                on the test summary statistics.
                - overall_accuracy: The average accuracy score across all test types.
        """

        report = {
            'system_info': self.test_results['system_info'],
            'summary': {},
            'detailed_results': self.test_results,
            'recommendations': []
        }
        
        # Calculate summary statistics
        if self.test_results['detection']:
            detection_scores = [r['accuracy_score'] for r in self.test_results['detection']]
            report['summary']['detection_accuracy'] = {
                'mean': float(np.mean(detection_scores)),
                'std': float(np.std(detection_scores)),
                'min': float(np.min(detection_scores)),
                'max': float(np.max(detection_scores)),
                'count': len(detection_scores)
            }
        
        if self.test_results['coordinates']:
            coord_scores = [r['accuracy_score'] for r in self.test_results['coordinates']]
            coord_errors = [r['euclidean_error'] for r in self.test_results['coordinates']]
            report['summary']['coordinate_accuracy'] = {
                'mean_accuracy': float(np.mean(coord_scores)),
                'mean_error_mm': float(np.mean(coord_errors) * 1000),
                'std_error_mm': float(np.std(coord_errors) * 1000),
                'max_error_mm': float(np.max(coord_errors) * 1000),
                'count': len(coord_scores)
            }
        
        if self.test_results['speech']:
            speech_success = [r['recognized'] for r in self.test_results['speech']]
            report['summary']['speech_accuracy'] = {
                'recognition_rate': float(np.mean(speech_success)),
                'successful_commands': sum(speech_success),
                'total_commands': len(speech_success)
            }
        
        # Generate recommendations
        report['recommendations'] = self.generate_recommendations(report['summary'])
        
        # Calculate overall system score
        scores = []
        if 'detection_accuracy' in report['summary']:
            scores.append(report['summary']['detection_accuracy']['mean'])
        if 'coordinate_accuracy' in report['summary']:
            scores.append(report['summary']['coordinate_accuracy']['mean_accuracy'])
        if 'speech_accuracy' in report['summary']:
            scores.append(report['summary']['speech_accuracy']['recognition_rate'])
        
        report['overall_accuracy'] = float(np.mean(scores)) if scores else 0.0
        
        return report
    
    def generate_recommendations(self, summary: Dict) -> List[str]:
        """Generate system improvement recommendations"""
        recommendations = []
        
        if 'detection_accuracy' in summary:
            acc = summary['detection_accuracy']['mean']
            if acc < 0.7:
                recommendations.append(
                    f"Detection accuracy ({acc:.1%}) is below recommended threshold. "
                    "Consider: lowering confidence thresholds, using more training data, "
                    "or improving lighting conditions."
                )
        
        if 'coordinate_accuracy' in summary:
            error = summary['coordinate_accuracy']['mean_error_mm']
            if error > 100:  # 10cm
                recommendations.append(
                    f"3D coordinate error ({error:.1f}mm) is high. "
                    "Consider: recalibrating camera, improving depth estimation, "
                    "or validating camera matrix parameters."
                )
        
        if 'speech_accuracy' in summary:
            rate = summary['speech_accuracy']['recognition_rate']
            if rate < 0.8:
                recommendations.append(
                    f"Speech recognition rate ({rate:.1%}) could be improved. "
                    "Consider: reducing ambient noise, using a better microphone, "
                    "or adjusting recognition sensitivity."
                )
        
        return recommendations
    
    def save_report(self, report: Dict, filename: Optional[str] = None):
        """Save the accuracy report to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vision_accuracy_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Report saved to {filename}")
        return filename
    
    def create_visualizations(self, report: Dict, output_dir: str = "accuracy_plots"):
        """Create visualization plots for the test results"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Detection accuracy over time
        if self.test_results['detection']:
            detection_scores = [r['accuracy_score'] for r in self.test_results['detection']]
            queries = [r['query'] for r in self.test_results['detection']]
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(detection_scores)), detection_scores)
            plt.xlabel('Test Scenario')
            plt.ylabel('Accuracy Score')
            plt.title('Object Detection Accuracy by Query')
            plt.xticks(range(len(queries)), queries, rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/detection_accuracy.png")
            plt.close()
        
        # Coordinate error distribution
        if self.test_results['coordinates']:
            errors = [r['euclidean_error'] * 1000 for r in self.test_results['coordinates']]
            
            plt.figure(figsize=(8, 6))
            plt.hist(errors, bins=10, alpha=0.7, edgecolor='black')
            plt.xlabel('3D Position Error (mm)')
            plt.ylabel('Frequency')
            plt.title('Distribution of 3D Coordinate Errors')
            plt.axvline(np.mean(errors), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(errors):.1f}mm')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/coordinate_errors.png")
            plt.close()
        
        self.logger.info(f"Visualizations saved to {output_dir}/")
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        if self.speech:
            self.speech.stop_listening()
        cv2.destroyAllWindows()

def main():
    """Main function to run accuracy testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize tester
    tester = VisionSystemTester(use_camera=True)
    
    try:
        # Run comprehensive test
        print("Vision System Accuracy Testing")
        print("=" * 50)
        report = tester.run_comprehensive_test()
        
        # Display results
        print(f"\nOVERALL ACCURACY: {report['overall_accuracy']:.1%}")
        print("-" * 30)
        
        if 'detection_accuracy' in report['summary']:
            det_acc = report['summary']['detection_accuracy']['mean']
            print(f"Object Detection: {det_acc:.1%}")
        
        if 'coordinate_accuracy' in report['summary']:
            coord_acc = report['summary']['coordinate_accuracy']['mean_accuracy']
            coord_err = report['summary']['coordinate_accuracy']['mean_error_mm']
            print(f"3D Coordinates: {coord_acc:.1%} (avg error: {coord_err:.1f}mm)")
        
        if 'speech_accuracy' in report['summary']:
            speech_acc = report['summary']['speech_accuracy']['recognition_rate']
            print(f"Speech Recognition: {speech_acc:.1%}")
        
        # Show recommendations
        if report['recommendations']:
            print("\nRECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"{i}. {rec}")
        
        # Save report and create visualizations
        filename = tester.save_report(report)
        tester.create_visualizations(report)
        
        print(f"\nFull report saved to: {filename}")
        print("Visualization plots created in: accuracy_plots/")
        
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
    except Exception as e:
        print(f"Error during testing: {e}")
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main()