import cv2
from OWLViTDetector import OWLViTDetector
from SpeechCommandProcessor import SpeechCommandProcessor
from CameraCalibration import CameraCalibration
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simulate_command_pipeline():
    detector = OWLViTDetector()
    speech = SpeechCommandProcessor()
    speech.start_listening()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No cam available.")
        return

    print("Speak")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            command = speech.get_command()
            if command:
                logger.info(f"Received command: {command}")
                queries = speech.parse_object_query(command)
                if not queries:
                    logger.error("Couldn't understand object type.")
                    continue

                logger.info(f"Looking for: {queries}")
                results = detector.detect_with_text_queries(frame, queries)

                if results:
                    label, score, bbox = results[0]
                    cx, cy = detector.get_bbox_center(bbox)
                    calibration = CameraCalibration()
                    x, y, z = calibration.pixel_to_world(cx, cy, depth=0.3)  


                    logger.info(f"Detected '{label}' at center ({cx},{cy}) → 3D: ({x:.2f}, {y:.2f}, {z:.2f})")
                    logger.info(f"\nSuggested ROS2 Joint Command:")
                    logger.info(f"""
ros2 topic pub /object_target_pose geometry_msgs/msg/PoseStamped "{{ 
  header: {{ frame_id: 'base_link' }},
  pose: {{
    position: {{ x: {x:.2f}, y: {y:.2f}, z: {z:.2f} }},
    orientation: {{ w: 1.0 }}
  }}
}}"
""")
                else:
                    logger.error("No object found.")

            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        speech.stop_listening()

if __name__ == "__main__":
    simulate_command_pipeline()
