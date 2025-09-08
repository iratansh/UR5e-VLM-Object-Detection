#!/bin/bash

# Complete Environment Setup Script for UR5e VLM Object Detection System
# This script installs ALL necessary dependencies including ROS2, system libraries, and hardware drivers

set -e  # Exit on any error

echo "🚀 Starting comprehensive environment setup for UR5e VLM Object Detection System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Ubuntu/Debian
if ! command -v apt &> /dev/null; then
    print_error "This script requires Ubuntu/Debian with apt package manager"
    exit 1
fi

# Update system packages
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies for hardware access
print_status "Installing system dependencies for hardware interfaces..."
sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    wget \
    curl \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Audio system dependencies
print_status "Installing audio system dependencies..."
sudo apt install -y \
    alsa-base \
    alsa-utils \
    pulseaudio \
    pulseaudio-utils \
    pavucontrol \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libpulse-dev \
    ffmpeg

# USB and hardware access
print_status "Installing USB and hardware interface libraries..."
sudo apt install -y \
    libusb-1.0-0-dev \
    libusb-dev \
    libudev-dev \
    libv4l-dev \
    v4l-utils

# Graphics and display (for visualization)
print_status "Installing graphics dependencies..."
sudo apt install -y \
    mesa-utils \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6

# ROS2 Humble Installation
print_status "Installing ROS2 Humble..."

# Add ROS2 repository
if ! grep -q "packages.ros.org" /etc/apt/sources.list.d/ros2.list 2>/dev/null; then
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    sudo apt update
fi

# Install ROS2 Humble Desktop Full
sudo apt install -y ros-humble-desktop-full

# Install additional ROS2 packages
print_status "Installing ROS2 robotics packages..."
sudo apt install -y \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-gazebo-ros2-control \
    ros-humble-controller-manager \
    ros-humble-joint-trajectory-controller \
    ros-humble-joint-state-broadcaster \
    ros-humble-ur \
    ros-humble-ur-description \
    ros-humble-ur-moveit-config \
    ros-humble-moveit \
    ros-humble-moveit-ros-planning \
    ros-humble-moveit-ros-visualization \
    ros-humble-moveit-ros-planning-interface \
    ros-humble-moveit-ros-move-group \
    ros-humble-moveit-kinematics \
    ros-humble-moveit-planners \
    ros-humble-rviz2 \
    ros-humble-rviz-common \
    ros-humble-rviz-default-plugins \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-image-transport-plugins \
    ros-humble-vision-msgs

# Install Intel RealSense SDK
print_status "Installing Intel RealSense SDK..."
if ! dpkg -l | grep -q librealsense2; then
    # Register the server's public key
    mkdir -p /etc/apt/keyrings
    curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
    
    # Add the server to the list of repositories
    echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" | \
    sudo tee /etc/apt/sources.list.d/librealsense.list
    
    sudo apt update
    
    # Install RealSense SDK
    sudo apt install -y \
        librealsense2-dkms \
        librealsense2-utils \
        librealsense2-dev \
        librealsense2-dbg
fi

# Setup udev rules for hardware access
print_status "Setting up udev rules for hardware access..."

# RealSense udev rules
sudo tee /etc/udev/rules.d/99-realsense-libusb.rules > /dev/null <<EOF
# Intel RealSense cameras udev rules
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b07", GROUP="plugdev", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad1", GROUP="plugdev", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad2", GROUP="plugdev", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad3", GROUP="plugdev", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad4", GROUP="plugdev", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b64", GROUP="plugdev", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b68", GROUP="plugdev", MODE="0666"
EOF

# Audio device rules
sudo tee /etc/udev/rules.d/99-audio-permissions.rules > /dev/null <<EOF
# Audio devices permissions
SUBSYSTEM=="sound", GROUP="audio", MODE="0664"
KERNEL=="controlC[0-9]*", GROUP="audio", MODE="0664"
EOF

# Add user to necessary groups
print_status "Adding user to hardware access groups..."
sudo usermod -a -G audio,video,plugdev,dialout,tty $USER

# Reload udev rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# Check for conda installation
if ! command -v conda &> /dev/null; then
    print_warning "Conda not found. Installing Miniconda..."
    
    # Download and install Miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    
    # Initialize conda
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    
    # Reload bash profile
    source ~/.bashrc
    
    print_success "Miniconda installed successfully"
else
    print_success "Conda already installed"
fi

# Create conda environment
print_status "Creating conda environment from YAML file..."
if [ -f "ur5e_vlm_environment.yml" ]; then
    conda env create -f ur5e_vlm_environment.yml
    print_success "Conda environment created successfully"
else
    print_error "ur5e_vlm_environment.yml not found. Please ensure the file exists."
    exit 1
fi

# Activate environment and install spaCy model
print_status "Installing spaCy English model..."
conda activate ur5e_vlm_environment
python -m spacy download en_core_web_sm

# Setup ROS2 environment
print_status "Setting up ROS2 environment variables..."
echo "# ROS2 Humble setup" >> ~/.bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Create workspace setup script
print_status "Creating workspace setup script..."
cat > setup_workspace.sh << 'EOF'
#!/bin/bash
# Workspace setup script

# Source ROS2
source /opt/ros/humble/setup.bash

# Activate conda environment
conda activate ur5e_vlm_environment

# Set ROS2 environment variables
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Hardware permissions (run if needed)
# sudo chmod 666 /dev/video* 2>/dev/null || true

echo "🎉 Environment activated! ROS2 Humble + UR5e VLM Environment ready"
echo "Hardware interfaces configured for:"
echo "  ✓ Intel RealSense cameras"
echo "  ✓ Audio input/output (microphone/speakers)"
echo "  ✓ USB devices"
echo "  ✓ Video devices"
echo "  ✓ ROS2 Humble"
echo ""
echo "To test your setup:"
echo "  ros2 topic list"
echo "  realsense-viewer"  
echo "  python -c 'import sounddevice; print(sounddevice.query_devices())'"
EOF

chmod +x setup_workspace.sh

# Final system configuration
print_status "Performing final system configuration..."

# Set audio permissions
sudo chmod 666 /dev/snd/* 2>/dev/null || true

# Test installations
print_status "Testing installations..."

# Test RealSense
if command -v realsense-viewer &> /dev/null; then
    print_success "Intel RealSense SDK installed correctly"
else
    print_warning "Intel RealSense SDK may not be properly installed"
fi

# Test audio
if python3 -c "import sounddevice" &> /dev/null; then
    print_success "Audio libraries accessible"
else
    print_warning "Audio libraries may not be properly configured"
fi

print_success "🎉 Complete environment setup finished!"
print_status ""
print_status "IMPORTANT NEXT STEPS:"
print_status "1. Restart your terminal or run: source ~/.bashrc"
print_status "2. Activate the environment: ./setup_workspace.sh"
print_status "3. Test hardware connections:"
print_status "   - Connect Intel RealSense camera and run: realsense-viewer"
print_status "   - Test microphone: python -c 'import sounddevice; print(sounddevice.query_devices())'"
print_status "   - Test ROS2: ros2 topic list"
print_status ""
print_status "Environment includes:"
print_status "✓ ROS2 Humble with UR5e, MoveIt, Gazebo"
print_status "✓ Intel RealSense SDK and Python bindings"
print_status "✓ Audio system (ALSA/PulseAudio) with microphone/speaker support"
print_status "✓ Computer vision (OpenCV, OWL-ViT, transformers)"
print_status "✓ Speech processing (Whisper, TTS, spaCy)"
print_status "✓ Deep learning (PyTorch, HuggingFace)"
print_status "✓ USB device access and hardware permissions"
print_status "✓ All Python dependencies for the UR5e VLM system"
print_status ""
print_warning "Please reboot your system to ensure all hardware permissions take effect!"