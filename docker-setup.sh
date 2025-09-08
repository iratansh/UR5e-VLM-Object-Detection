#!/bin/bash

# Docker Setup Script for UR5e VLM Object Detection System
# This script ensures Docker has proper GPU and hardware access

set -e

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

print_status "🚀 Setting up Docker environment for UR5e VLM System..."

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please do not run this script as root"
    exit 1
fi

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    print_status "Installing Docker..."
    
    # Add Docker's official GPG key
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    
    # Add the repository to Apt sources
    echo \
      "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    
    print_success "Docker installed successfully"
else
    print_success "Docker already installed"
fi

# Add user to docker group
if ! groups $USER | grep -q docker; then
    print_status "Adding user to docker group..."
    sudo usermod -aG docker $USER
    print_warning "Please log out and log back in for group changes to take effect"
else
    print_success "User already in docker group"
fi

# Install NVIDIA Container Toolkit for GPU access
if command -v nvidia-smi &> /dev/null; then
    print_status "Installing NVIDIA Container Toolkit..."
    
    # Add NVIDIA Container Toolkit repository
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    
    # Configure Docker daemon
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    print_success "NVIDIA Container Toolkit installed and configured"
else
    print_warning "NVIDIA GPU not detected. GPU acceleration will not be available."
fi

# Install Docker Compose (if not already installed)
if ! command -v docker-compose &> /dev/null; then
    print_status "Installing Docker Compose..."
    sudo apt-get update
    sudo apt-get install -y docker-compose
    print_success "Docker Compose installed"
else
    print_success "Docker Compose already available"
fi

# Create udev rules for USB device access in containers
print_status "Setting up USB device access rules..."
sudo tee /etc/udev/rules.d/99-docker-usb.rules > /dev/null <<EOF
# Allow Docker containers to access USB devices
SUBSYSTEM=="usb", MODE="0666"
SUBSYSTEM=="tty", MODE="0666"

# Intel RealSense cameras
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b07", GROUP="docker", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad1", GROUP="docker", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad2", GROUP="docker", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad3", GROUP="docker", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad4", GROUP="docker", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b64", GROUP="docker", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b68", GROUP="docker", MODE="0666"

# Audio devices
SUBSYSTEM=="sound", GROUP="audio", MODE="0664"
KERNEL=="controlC[0-9]*", GROUP="audio", MODE="0664"
EOF

sudo udevadm control --reload-rules && sudo udevadm trigger

# Set up X11 forwarding for GUI applications
print_status "Configuring X11 forwarding for GUI applications..."
xhost +local:docker

# Create .dockerignore to optimize build context
print_status "Creating .dockerignore file..."
cat > .dockerignore << EOF
# Git
.git
.gitignore

# Documentation
*.md
LICENSE

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Docker
Dockerfile
docker-compose.yml
.dockerignore

# Temp files
*.tmp
*.temp
EOF

print_success "✅ Docker setup completed!"
print_status ""
print_status "Next steps:"
print_status "1. If you were added to the docker group, please log out and log back in"
print_status "2. Build the Docker image: ./start-docker.sh build"
print_status "3. Run the container: ./start-docker.sh run"
print_status "4. For simulation: ./start-docker.sh simulation"
print_status ""
print_status "Hardware access configured for:"
print_status "✓ NVIDIA GPU/CUDA (if available)"
print_status "✓ Intel RealSense cameras"
print_status "✓ Audio devices (microphone/speakers)"
print_status "✓ USB and serial devices"
print_status "✓ X11 forwarding for GUI applications"