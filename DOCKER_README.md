# Docker Environment for UR5e VLM Object Detection System

This Docker setup provides a complete Ubuntu 22.04 environment with ROS2 Humble, CUDA support, and hardware access for the UR5e VLM Object Detection System.

## Quick Start

### 1. Initial Setup
```bash
# Install Docker and configure hardware access
./docker-setup.sh

# Log out and log back in if you were added to the docker group
```

### 2. Build the Environment
```bash
# Build the Docker image (this will take 15-30 minutes)
./start-docker.sh build
```

### 3. Run the System
```bash
# Start the main container
./start-docker.sh run

# Open a shell in the container
./start-docker.sh shell

# Test the environment
./start-docker.sh test
```

### 4. For Simulation
```bash
# Start with Gazebo simulation
./start-docker.sh simulation

# View simulation logs
./start-docker.sh logs sim
```

## What's Included

### Operating System & Core
- **Ubuntu 22.04** (full compatibility with ROS2 Humble)
- **ROS2 Humble Desktop Full**
- **CUDA 12.1** with NVIDIA runtime support

### Robotics & Simulation
- **MoveIt2** - Motion planning framework
- **RViz2** - 3D visualization
- **Gazebo** - Physics simulation with ros2_control
- **UR5e packages** - Robot descriptions and configurations
- **Robotiq gripper** support

### AI/ML & Vision
- **PyTorch** with CUDA support
- **HuggingFace Transformers** (OWL-ViT, Whisper)
- **OpenCV** for computer vision
- **Intel RealSense SDK** and Python bindings
- **spaCy** for natural language processing

### Hardware Access
- **GPU/CUDA** - Full NVIDIA GPU acceleration
- **USB devices** - RealSense cameras, serial devices
- **Audio system** - Microphone and speaker access
- **X11 forwarding** - GUI applications (RViz, Gazebo)

## Available Commands

```bash
./start-docker.sh build       # Build Docker image
./start-docker.sh run         # Start main container
./start-docker.sh simulation  # Start with Gazebo
./start-docker.sh shell       # Open container shell
./start-docker.sh logs        # View container logs
./start-docker.sh test        # Test environment
./start-docker.sh stop        # Stop all containers
```

## Hardware Testing

### Test RealSense Camera
```bash
./start-docker.sh shell
realsense-viewer
```

### Test Audio System
```bash
./start-docker.sh shell
python -c "import sounddevice; print(sounddevice.query_devices())"
```

### Test GPU/CUDA
```bash
./start-docker.sh shell
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Test ROS2
```bash
./start-docker.sh shell
ros2 topic list
ros2 run demo_nodes_cpp talker
```

## Troubleshooting

### GPU Not Accessible
1. Install NVIDIA drivers on host
2. Install NVIDIA Container Toolkit: `./docker-setup.sh`
3. Restart Docker: `sudo systemctl restart docker`

### USB/Camera Access Issues
1. Check device permissions: `ls -la /dev/video*`
2. Reconnect camera and restart container
3. Run with privileged mode (already enabled)

### Audio Not Working
1. Check PulseAudio is running on host
2. Verify user in audio group: `groups $USER`
3. Test host audio first

### X11/GUI Issues
1. Run `xhost +local:docker` on host
2. Check DISPLAY variable: `echo $DISPLAY`
3. For remote systems, enable X11 forwarding

## Development Workflow

### 1. Code Development
- Mount your workspace: Already configured in docker-compose.yml
- Edit code on host, run in container
- All changes persist automatically

### 2. Testing
```bash
# Run unit tests
./start-docker.sh shell
cd /home/ur5e_user/workspace
python -m pytest vision/testing/

# Test specific components
python vision/test_whisper_integration.py
```

### 3. Simulation Testing
```bash
# Start simulation environment
./start-docker.sh simulation

# In another terminal, run your code
./start-docker.sh shell
cd /home/ur5e_user/workspace
python vision/UnifiedVisionSystemSim.py
```

### 4. Hardware Deployment
```bash
# Connect hardware and start container
./start-docker.sh run

# Test hardware connections
./start-docker.sh test

# Run your system
./start-docker.sh shell
cd /home/ur5e_user/workspace
python vision/UnifiedVisionSystem.py
```

## File Structure

```
├── Dockerfile                 # Main container definition
├── docker-compose.yml         # Multi-service configuration
├── docker-setup.sh           # Initial Docker setup
├── start-docker.sh           # Container management
├── ur5e_vlm_environment.yml  # Conda environment
├── .dockerignore             # Build optimization
└── vision/                   # Your code (mounted)
```

## Environment Variables

Key environment variables set in the container:
- `ROS_DOMAIN_ID=0`
- `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`
- `NVIDIA_VISIBLE_DEVICES=all`
- `DISPLAY=$DISPLAY` (for GUI applications)

## Performance Tips

1. **Build Optimization**: Use `.dockerignore` to exclude unnecessary files
2. **GPU Memory**: Monitor with `nvidia-smi` inside container
3. **Container Resources**: Adjust docker-compose.yml if needed
4. **Volume Mounts**: Use bind mounts for development, volumes for data

## Support

This environment provides complete compatibility with Ubuntu 22.04 and ROS2 Humble, resolving any issues with Ubuntu 24.04. All hardware interfaces are properly configured for seamless development and deployment.