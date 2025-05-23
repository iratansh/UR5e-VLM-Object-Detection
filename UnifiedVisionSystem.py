import cv2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import time
import numpy as np
import logging
from typing import List, Tuple, Optional
from ObjectDetection import ObjectDetection
from OWLViTDetector import OWLViTDetector
from CameraCalibration import CameraCalibration
from WorkSpaceValidator import WorkspaceValidator


class UnifiedVisionSystem(Node):
    def __init__(self):
        super().__init__('unified_vision_system')

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.logger.info("🔧 Initializing Unified Vision System with VLM + YOLO")

        # Core modules
        self.detector = ObjectDetection('yolov5s', confidence_threshold=0.4)  # Maintain consistency with debug_visiontest.py
        self.vlm_detector = OWLViTDetector()
        self.calibration = CameraCalibration()
        self.workspace = WorkspaceValidator()

        # ROS2
        self.pose_pub = self.create_publisher(PoseStamped, '/object_target_pose', 10)
        self.status_pub = self.create_publisher(String, '/vision_status', 10)

        # Camera
        self.camera = None
        self.setup_camera()

        # Detection settings
        self.text_queries = [
            "red apple", "plastic water bottle", "smartphone",
            "coffee cup", "book", "remote control"
        ]
        self.last_detection_time = 0
        self.detection_cooldown = 2.0

        # Performance tracking
        self.detection_times = []
        self.bbox_accuracy_scores = []

    def setup_camera(self):
        for idx in [0, 1, 2]:
            try:
                self.logger.info(f"🎥 Testing camera {idx}...")
                cap = cv2.VideoCapture(idx)
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                cap.set(cv2.CAP_PROP_EXPOSURE, -6)

                for _ in range(10):
                    cap.read()

                self.camera = cap
                self.logger.info(f"✅ Camera {idx} initialized")
                return
            except Exception as e:
                self.logger.warning(f"Camera {idx} failed: {e}")
        raise RuntimeError("❌ No available camera for Unified Vision System")

    def run(self):
        self.logger.info("▶️ Running unified vision system...")

        try:
            while rclpy.ok():
                ret, frame = self.camera.read()
                if not ret:
                    continue

                start_time = time.perf_counter()
                detections = self.vlm_detector.detect_with_text_queries(frame, self.text_queries, confidence_threshold=0.15)
                processed_frame, best = self.process_detections_debug_style(frame, detections)
                processing_time = time.perf_counter() - start_time
                self.detection_times.append(processing_time)

                if best:
                    self.publish_pose(best)

                cv2.imshow("Unified Vision - VLM + YOLO", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                rclpy.spin_once(self, timeout_sec=0.01)

        except KeyboardInterrupt:
            self.logger.info("🛑 Interrupted by user")
        finally:
            self.camera.release()
            cv2.destroyAllWindows()

    def process_detections_debug_style(self, frame: np.ndarray, detections: List[Tuple[str, float, List[int]]]) -> Tuple[np.ndarray, Optional[dict]]:
        best_score = 0
        best_target = None
        bbox_scores = []

        for label, conf, bbox in detections:
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            depth = 0.3  # placeholder; later integrate depth sensor or model
            x, y, z = self.calibration.pixel_to_world(cx, cy, depth)

            if not self.workspace.is_reachable(x, y, z):
                continue

            score = conf * self.workspace.get_safety_score(x, y, z)
            bbox_scores.append(score)

            color = (0, 255, 0) if score > 0.6 else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if score > best_score:
                best_target = {
                    'label': label,
                    'confidence': conf,
                    'position': (x, y, z),
                    'bbox': bbox
                }
                best_score = score

        if bbox_scores:
            self.bbox_accuracy_scores.append(np.mean(bbox_scores))

        return frame, best_target

    def publish_pose(self, target: dict):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position.x = float(target['position'][0])
        pose_msg.pose.position.y = float(target['position'][1])
        pose_msg.pose.position.z = float(target['position'][2])
        pose_msg.pose.orientation.w = 1.0
        self.pose_pub.publish(pose_msg)

        self.status_pub.publish(String(data=f"Published: {target['label']}"))
        self.logger.info(f"📤 Sent target: {target['label']} @ {target['position']}")


def main(args=None):
    rclpy.init(args=args)
    system = UnifiedVisionSystem()
    system.run()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
