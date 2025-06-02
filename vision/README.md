# UR5e Vision-Based Manipulation System

This repo is the official repo for the study titled "A Multimodal Framework for Natural Language Command Execution in Robotic Systems". This system implements vision-based robotic manipulation using a UR5e robot arm and RealSense camera, integrating speech commands, visual language models (VLM), and depth-aware object detection. 

## System Requirements

- Ubuntu 22.04 or later
- Python 3.8+
- ROS2 Humble
- UR5e Robot with ROS2 driver
- Intel RealSense D435i or similar
- CUDA-capable GPU (recommended for VLM)

## Dependencies Installation

1. **ROS2 Humble**
```bash
# Add ROS2 apt repository
sudo apt update && sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install ros-humble-desktop
```

2. **Python Dependencies**
```bash
# Create and activate virtual environment (recommended)
python3 -m venv ~/ros2_ws/src/vision_env
source ~/ros2_ws/src/vision_env/bin/activate

# Install required packages
pip install numpy opencv-python pyrealsense2 transformations
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install scipy matplotlib
```

3. **RealSense SDK**
```bash
# Install RealSense SDK
sudo apt-get install -y software-properties-common
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
sudo apt-get install -y librealsense2-dkms librealsense2-utils librealsense2-dev
```

4. **ROS2 Packages**
```bash
# Install required ROS2 packages
sudo apt install ros-humble-ur-robot-driver
sudo apt install ros-humble-ur-description
sudo apt install ros-humble-ur-msgs
sudo apt install ros-humble-realsense-ros
```

## System Setup

1. **Clone and Build**
```bash
# Create ROS2 workspace if not exists
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Clone the repository
git clone <your-repo-url> UR5e_Unity/ur5e_project

# Build the workspace
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

2. **Environment Setup**
```bash
# Add to ~/.bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Running the System

1. **Start ROS2 Core Components**
```bash
# Terminal 1: Start ROS2 core
ros2 launch ur_robot_driver ur5e_bringup.launch.py robot_ip:=<ROBOT_IP>

# Terminal 2: Start RealSense camera
ros2 launch realsense2_camera rs_launch.py
```

2. **Start Vision System**
```bash
# Terminal 3: Start UnifiedVisionSystem
cd ~/ros2_ws/src/UR5e_Unity/ur5e_project/vision
python3 UnifiedVisionSystem.py
```

## System Components

- `UnifiedVisionSystem.py`: Main system node integrating all components
- `DepthAwareDetector.py`: Handles 3D object detection with RealSense
- `GraspPointDetector.py`: Implements grasp point detection strategies
- `UR5eKinematics.py`: Handles robot kinematics calculations
- `OWLViTDetector.py`: Visual Language Model for object detection
- `SpeechCommandProcessor.py`: Processes voice commands

## Usage

1. The system will start listening for voice commands
2. Example commands:
   - "Pick up the bottle"
   - "Grab the red cup"
   - "Move to the box"

3. The system will:
   - Process the voice command
   - Detect objects using VLM
   - Calculate 3D positions using depth data
   - Plan and execute grasping motions

## Troubleshooting

1. **RealSense Camera Issues**
```bash
# Check if camera is recognized
rs-enumerate-devices

# Test camera stream
realsense-viewer
```

2. **ROS2 Communication Issues**
```bash
# Check ROS2 topics
ros2 topic list

# Monitor robot state
ros2 topic echo /joint_states

# Check node graph
ros2 node list
```

3. **Common Error Solutions**
- If camera not found: Check USB connection and permissions
- If robot not connecting: Verify IP address and network connection
- If vision system crashes: Check CUDA installation and GPU memory

## Safety Notes

- Always keep emergency stop within reach
- Monitor robot operation at all times
- Ensure workspace is clear before operation
- Test in simulation first when possible

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
Authors Email: iratansh@ualberta.ca

