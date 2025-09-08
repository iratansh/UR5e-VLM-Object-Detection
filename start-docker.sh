#!/bin/bash

# UR5e VLM Docker Management Script
# Usage: ./start-docker.sh [build|run|simulation|stop|logs|shell|test]

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

# Function to check Docker and NVIDIA runtime
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please run ./docker-setup.sh first"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running or you don't have permission"
        print_error "Try: sudo systemctl start docker"
        print_error "Or add yourself to docker group and re-login"
        exit 1
    fi
}

# Function to get correct docker compose command
get_compose_cmd() {
    if command -v docker-compose &> /dev/null; then
        echo "docker-compose"
    elif docker compose version &> /dev/null; then
        echo "docker compose"
    else
        print_error "Neither docker-compose nor docker compose found"
        exit 1
    fi
}

# Function to enable X11 forwarding
setup_x11() {
    print_status "Setting up X11 forwarding..."
    xhost +local:docker > /dev/null 2>&1 || print_warning "Could not set xhost permissions"
    export DISPLAY=${DISPLAY:-:0}
}

# Function to build the Docker image
build_image() {
    print_status "🔨 Building UR5e VLM Docker image..."
    COMPOSE_CMD=$(get_compose_cmd)
    $COMPOSE_CMD build --no-cache ur5e-vlm
    print_success "✅ Docker image built successfully!"
}

# Function to run the main container
run_container() {
    setup_x11
    print_status "🚀 Starting UR5e VLM container..."
    
    COMPOSE_CMD=$(get_compose_cmd)
    # Stop any existing container
    $COMPOSE_CMD down > /dev/null 2>&1 || true
    
    # Start the container
    $COMPOSE_CMD up -d ur5e-vlm
    
    print_success "✅ Container started successfully!"
    print_status "Access the container with: ./start-docker.sh shell"
    print_status "View logs with: ./start-docker.sh logs"
}

# Function to run simulation mode
run_simulation() {
    setup_x11
    print_status "🎮 Starting UR5e VLM with Gazebo simulation..."
    
    COMPOSE_CMD=$(get_compose_cmd)
    # Stop any existing containers
    $COMPOSE_CMD down > /dev/null 2>&1 || true
    
    # Start main container and simulation
    $COMPOSE_CMD --profile simulation up -d
    
    print_success "✅ Simulation environment started!"
    print_status "Gazebo should be starting up..."
    print_status "Access main container: ./start-docker.sh shell"
    print_status "View logs: ./start-docker.sh logs"
}

# Function to stop containers
stop_containers() {
    print_status "🛑 Stopping UR5e VLM containers..."
    COMPOSE_CMD=$(get_compose_cmd)
    $COMPOSE_CMD down
    print_success "✅ All containers stopped"
}

# Function to view logs
view_logs() {
    COMPOSE_CMD=$(get_compose_cmd)
    if [ "$2" = "sim" ]; then
        $COMPOSE_CMD logs -f gazebo-sim
    else
        $COMPOSE_CMD logs -f ur5e-vlm
    fi
}

# Function to open shell in container
open_shell() {
    container_name="ur5e-vlm-container"
    
    if ! docker ps | grep -q $container_name; then
        print_error "Container is not running. Start it first with: ./start-docker.sh run"
        exit 1
    fi
    
    print_status "🐚 Opening shell in UR5e VLM container..."
    docker exec -it $container_name /bin/bash
}

# Function to test the environment
test_environment() {
    container_name="ur5e-vlm-container"
    
    if ! docker ps | grep -q $container_name; then
        print_error "Container is not running. Start it first with: ./start-docker.sh run"
        exit 1
    fi
    
    print_status "🧪 Testing UR5e VLM environment..."
    
    docker exec $container_name bash -c "
    source /opt/ros/humble/setup.bash &&
    eval '\$(conda shell.bash hook)' &&
    conda activate ur5e_vlm_environment &&
    echo '✅ ROS2 Topics:' &&
    timeout 5 ros2 topic list || echo '⚠️  ROS2 topics timeout (normal if no nodes running)' &&
    echo '' &&
    echo '✅ GPU Status:' &&
    nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu --format=csv,noheader 2>/dev/null || echo '⚠️  No NVIDIA GPU detected' &&
    echo '' &&
    echo '✅ Audio Devices:' &&
    python -c 'import sounddevice; print(\"Microphones:\", [d[\"name\"] for d in sounddevice.query_devices() if d[\"max_input_channels\"] > 0])' 2>/dev/null || echo '⚠️  Audio not accessible' &&
    echo '' &&
    echo '✅ RealSense:' &&
    python -c 'import pyrealsense2 as rs; ctx = rs.context(); devices = ctx.query_devices(); print(f\"RealSense devices: {len(devices)}\")' &&
    echo '' &&
    echo '✅ Key Libraries:' &&
    python -c 'import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}\")' &&
    python -c 'import cv2; print(f\"OpenCV: {cv2.__version__}\")' &&
    python -c 'import transformers; print(f\"Transformers: {transformers.__version__}\")' &&
    echo '' &&
    echo '🎉 Environment test completed!'
    "
}

# Function to show usage
show_usage() {
    echo "UR5e VLM Docker Management Script"
    echo ""
    echo "Usage: ./start-docker.sh [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build       Build the Docker image"
    echo "  run         Start the main container"
    echo "  simulation  Start container with Gazebo simulation"
    echo "  stop        Stop all containers"
    echo "  logs        View container logs (add 'sim' for simulation logs)"
    echo "  shell       Open shell in running container"
    echo "  test        Test the environment setup"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./start-docker.sh build"
    echo "  ./start-docker.sh run"
    echo "  ./start-docker.sh shell"
    echo "  ./start-docker.sh logs sim"
    echo ""
    echo "Hardware Support:"
    echo "  ✓ NVIDIA GPU/CUDA"
    echo "  ✓ Intel RealSense cameras"
    echo "  ✓ USB devices and serial ports"
    echo "  ✓ Audio input/output"
    echo "  ✓ X11 forwarding for GUI apps"
}

# Main script logic
check_docker

case "${1:-help}" in
    "build")
        build_image
        ;;
    "run")
        run_container
        ;;
    "simulation")
        run_simulation
        ;;
    "stop")
        stop_containers
        ;;
    "logs")
        view_logs "$@"
        ;;
    "shell")
        open_shell
        ;;
    "test")
        test_environment
        ;;
    "help"|*)
        show_usage
        ;;
esac