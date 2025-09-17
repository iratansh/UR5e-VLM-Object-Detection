# Simulation System for UR5e-VLM-Object-Detection

## 🎮 Updated for Reorganized Codebase

This directory contains simulation-specific files for Gazebo + RViz + MoveIt2 deployment.

### 🔧 Key Components

#### UnifiedVisionSystemSim.py
- **Updated**: Compatible with reorganized codebase structure
- **Dependencies**:
  - Physical hardware components from `../deployment/`
  - AI/ML components from `../vision/`
- **Features**:
  - Automatic path resolution for imports
  - Simulation mode detection via ROS parameters
  - Enhanced error handling and logging
  - Import path testing function

### 🚀 Usage

#### Basic Simulation Launch
```bash
cd launch/
python3 UnifiedVisionSystemSim.py
```

#### Gazebo with Red Cube Environment
```bash
cd launch/
python3 launch_gazebo_with_red_cube.py
```

#### Test Import Paths
```bash
cd launch/
python3 -c "from UnifiedVisionSystemSim import test_import_paths; test_import_paths()"
```

### 📁 Directory Structure Dependencies

```
UR5e-VLM-Object-Detection/
├── launch/                    # 🎮 This directory - Simulation files
│   ├── UnifiedVisionSystemSim.py     # Main simulation system
│   ├── launch_gazebo_with_red_cube.py
│   ├── config/, urdf/, worlds/
│   └── ...
├── deployment/                # 🤖 Physical robot components
│   ├── UnifiedVisionSystem.py        # Base system
│   ├── UR5eKinematics.py
│   ├── CameraCalibration.py
│   └── ...
└── vision/                    # 🧠 AI/ML components
    ├── SpeechCommandProcessor.py
    ├── OWLViTDetector.py
    ├── ConstructionClarificationManager.py
    └── ...
```

### ✅ Recent Updates

1. **Path Resolution**: Smart import path handling for reorganized structure
2. **Error Handling**: Comprehensive error reporting with tracebacks
3. **Logger Updates**: Updated to use `self.get_logger()` instead of `self.logger`
4. **Import Testing**: Built-in function to verify all dependencies load correctly
5. **Documentation**: Clear separation between simulation and deployment code

### 🔍 Troubleshooting

If you encounter import errors:
1. Verify all directories exist: `ls -la ../deployment ../vision`
2. Check file locations: Key files should be in the right directories
3. Run import test: `python3 -c "from UnifiedVisionSystemSim import test_import_paths; test_import_paths()"`
4. Check ROS2 environment: Ensure all ROS2 dependencies are installed

### 🎯 Integration with Research System

The simulation system now properly integrates with the construction HRI research framework:
- **RAG Enhanced**: Uses RAG system from vision directory
- **TTS Integration**: Construction-specific TTS responses
- **Trust Measurement**: Compatible with experimental framework
- **Construction Vocabulary**: Works with trade-specific terminology

Ready for research deployment! 🚀