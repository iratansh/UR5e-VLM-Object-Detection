# Ubuntu 22.04 based Docker environment for UR5e VLM Object Detection System
# Includes ROS2 Humble, CUDA support, hardware access, and all dependencies

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Toronto

# Set locale
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gnupg2 \
    lsb-release \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    build-essential \
    cmake \
    git \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Add ROS2 Humble repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 Humble and robotics packages
RUN apt-get update && apt-get install -y \
    ros-humble-desktop-full \
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
    ros-humble-vision-msgs \
    python3-rosdep \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init && rosdep update

# Install Intel RealSense SDK (without DKMS for Docker compatibility)
RUN mkdir -p /etc/apt/keyrings && \
    curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" | \
    tee /etc/apt/sources.list.d/librealsense.list && \
    apt-get update && apt-get install -y \
    librealsense2-utils \
    librealsense2-dev \
    librealsense2-dbg \
    && rm -rf /var/lib/apt/lists/*

# Install audio system dependencies
RUN apt-get update && apt-get install -y \
    alsa-base \
    alsa-utils \
    pulseaudio \
    pulseaudio-utils \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libpulse-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install USB and hardware interface libraries
RUN apt-get update && apt-get install -y \
    libusb-1.0-0-dev \
    libusb-dev \
    libudev-dev \
    libv4l-dev \
    v4l-utils \
    udev \
    && rm -rf /var/lib/apt/lists/*

# Install graphics dependencies for visualization
RUN apt-get update && apt-get install -y \
    mesa-utils \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV MINICONDA_VERSION=py311_23.11.0-2
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda3 && \
    rm /tmp/miniconda.sh

# Add conda to PATH
ENV PATH=/opt/miniconda3/bin:$PATH

# Copy conda environment file and create environment
COPY ur5e_vlm_environment.yml /tmp/ur5e_vlm_environment.yml
RUN conda env create -f /tmp/ur5e_vlm_environment.yml && \
    conda clean -a -y

# Install spaCy English model
RUN /opt/miniconda3/envs/ur5e_vlm_environment/bin/python -m spacy download en_core_web_sm

# Setup udev rules for hardware access
RUN echo '# Intel RealSense cameras udev rules' > /etc/udev/rules.d/99-realsense-libusb.rules && \
    echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b07", GROUP="plugdev", MODE="0666"' >> /etc/udev/rules.d/99-realsense-libusb.rules && \
    echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad1", GROUP="plugdev", MODE="0666"' >> /etc/udev/rules.d/99-realsense-libusb.rules && \
    echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad2", GROUP="plugdev", MODE="0666"' >> /etc/udev/rules.d/99-realsense-libusb.rules && \
    echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad3", GROUP="plugdev", MODE="0666"' >> /etc/udev/rules.d/99-realsense-libusb.rules && \
    echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad4", GROUP="plugdev", MODE="0666"' >> /etc/udev/rules.d/99-realsense-libusb.rules && \
    echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b64", GROUP="plugdev", MODE="0666"' >> /etc/udev/rules.d/99-realsense-libusb.rules && \
    echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b68", GROUP="plugdev", MODE="0666"' >> /etc/udev/rules.d/99-realsense-libusb.rules

# Audio device rules
RUN echo '# Audio devices permissions' > /etc/udev/rules.d/99-audio-permissions.rules && \
    echo 'SUBSYSTEM=="sound", GROUP="audio", MODE="0664"' >> /etc/udev/rules.d/99-audio-permissions.rules && \
    echo 'KERNEL=="controlC[0-9]*", GROUP="audio", MODE="0664"' >> /etc/udev/rules.d/99-audio-permissions.rules

# Create a non-root user
ARG USERNAME=ur5e_user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Add user to necessary groups
RUN usermod -a -G audio,video,plugdev,dialout,tty $USERNAME

# Set environment variables for ROS2 and conda
ENV ROS_DISTRO=humble
ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Create entrypoint script
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'set -e' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Source ROS2' >> /entrypoint.sh && \
    echo 'source /opt/ros/humble/setup.bash' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Initialize conda' >> /entrypoint.sh && \
    echo 'eval "$(/opt/miniconda3/bin/conda shell.bash hook)"' >> /entrypoint.sh && \
    echo 'conda activate ur5e_vlm_environment' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Set up environment' >> /entrypoint.sh && \
    echo 'export ROS_DOMAIN_ID=0' >> /entrypoint.sh && \
    echo 'export RMW_IMPLEMENTATION=rmw_fastrtps_cpp' >> /entrypoint.sh && \
    echo 'export DISPLAY=$DISPLAY' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Set permissions for devices (if running as root)' >> /entrypoint.sh && \
    echo 'if [ "$EUID" -eq 0 ]; then' >> /entrypoint.sh && \
    echo '  chmod 666 /dev/video* 2>/dev/null || true' >> /entrypoint.sh && \
    echo '  chmod 666 /dev/snd/* 2>/dev/null || true' >> /entrypoint.sh && \
    echo 'fi' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo 'echo "🚀 UR5e VLM Environment Ready!"' >> /entrypoint.sh && \
    echo 'echo "✓ ROS2 Humble activated"' >> /entrypoint.sh && \
    echo 'echo "✓ Conda environment activated"' >> /entrypoint.sh && \
    echo 'echo "✓ Hardware access configured"' >> /entrypoint.sh && \
    echo 'echo "✓ GPU/CUDA available: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo \"Not detected\")"' >> /entrypoint.sh && \
    echo 'echo ""' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo 'exec "$@"' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Switch to non-root user
USER $USERNAME
WORKDIR /home/$USERNAME

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]