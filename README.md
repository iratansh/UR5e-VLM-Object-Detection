### A Multimodal Framework for Natural Language Command Execution in Robotic Systems ###

# TODO (Object Detection and VLM Study) - Waiting for Camera / Microcontroller:
1. Run the Object Detection Model on the Camera to ensure that it works as expected
2. Integrate the Object Detection Model into the UR5e and test with a sample object and see if it is picked up
3. Evaluate the Object Detection Model on its own and perform any operations as necessary
4. Integrate the VLM model into the system
5. Test the VLM with the Object Detection model and evaluate 
6. Research Paper


### Architecture - Camera Based Object Grasping ###
Camera -> ObjectDetection -> CameraCalibration -> WorkspaceValidator -> ROS2 Publisher

### Architecture - VLM Based Object Grasping ###  
User speaks → Microphone → SpeechCommandProcessor
           → parsed object queries → CompleteVLMSystem
           → runs VLMDetector (OWL-ViT)
           → gets bbox + 3D pose (CameraCalibration)
           → filters reachability (WorkspaceValidator)
           → publishes target (ROS2 PoseStamped to UR5e)