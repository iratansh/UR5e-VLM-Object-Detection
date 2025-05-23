import cv2
import time
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import logging
from ObjectDetection import ObjectDetection
from OWLViTDetector import OWLViTDetector
from CameraCalibration import CameraCalibration
from WorkSpaceValidator import WorkspaceValidator
from SpeechCommandProcessor import SpeechCommandProcessor

class CompleteSystem(Node):
    def __init__(self):
        super().__init__('complete_system')
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.logger.info("Initializing Complete System: Camera, Voice, VLM, and ROS2")

        # Components
        self.detector = ObjectDetection()
        self.vlm = OWLViTDetector()
        self.calibration = CameraCalibration()
        self.workspace = WorkspaceValidator()
        self.speech = SpeechCommandProcessor()

        # ROS2
        self.pose_pub = self.create_publisher(PoseStamped, '/object_target_pose', 10)
        self.status_pub = self.create_publisher(String, '/vision_status', 10)

        # Camera setup
        self.camera = None
        self.setup_camera()

        # Internal state
        self.use_vlm = True
        self.last_command = None
        self.queries = []
        self.last_detection_time = 0
        self.cooldown = 2.0
        self.enable_bbox_refinement = True

        self.speech.start_listening()
        self.logger.info("Voice control listening started")

    def setup_camera(self):
        for idx in [0, 1, 2]:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.camera = cap
                self.logger.info(f"Camera {idx} initialized")
                return
        raise RuntimeError("No available camera")

    def run(self):
        self.logger.info("System running. Speak commands")
        try:
            while rclpy.ok():
                ret, frame = self.camera.read()
                if not ret:
                    continue

                command = self.speech.get_command()
                if command:
                    self.last_command = command
                    self.queries = self.speech.parse_object_query(command)
                    self.logger.info(f"Parsed queries: {self.queries}")

                detections = self.vlm.detect_with_text_queries(frame, self.queries, confidence_threshold=0.15) if self.use_vlm and self.queries else self.detector.detect_objects(frame)
                frame, best = self.process_detections(frame, detections)

                if best and (time.time() - self.last_detection_time > self.cooldown):
                    self.publish_pose(best)
                    self.last_detection_time = time.time()

                cv2.imshow("Complete System With Debug", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                rclpy.spin_once(self, timeout_sec=0.01)
        finally:
            self.cleanup()

    def process_detections(self, frame, detections):
        best_target = None
        best_score = 0

        for label, conf, bbox in detections:
            if self.enable_bbox_refinement:
                bbox = self.detector.refine_bounding_box(frame, bbox)

            cx, cy = self.detector.get_bbox_center(bbox)
            depth = 0.3
            x, y, z = self.calibration.pixel_to_world(cx, cy, depth)

            if not self.workspace.is_reachable(x, y, z):
                continue

            score = conf * self.workspace.get_safety_score(x, y, z)
            color = (0, 255, 0) if score > 0.6 else (0, 255, 255)
            self.detector.draw_enhanced_bbox(frame, label, conf, bbox, color)

            if score > best_score:
                best_score = score
                best_target = {
                    'label': label,
                    'confidence': conf,
                    'position': (x, y, z),
                    'bbox': bbox
                }

        return frame, best_target

    def publish_pose(self, target):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position.x = float(target['position'][0])
        pose_msg.pose.position.y = float(target['position'][1])
        pose_msg.pose.position.z = float(target['position'][2])
        pose_msg.pose.orientation.w = 1.0

        self.pose_pub.publish(pose_msg)
        self.status_pub.publish(String(data=f"Target: {target['label']}"))
        self.logger.info(f"Published target {target['label']} at {target['position']}")

    def cleanup(self):
        self.logger.info("Cleaning up resources")
        if self.camera:
            self.camera.release()
        self.speech.stop_listening()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = CompleteSystem()
    node.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()