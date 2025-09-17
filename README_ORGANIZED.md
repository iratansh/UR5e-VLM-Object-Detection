# UR5e-VLM-Object-Detection - Organized Codebase Structure

## 📁 Directory Organization

The codebase has been reorganized into three main directories for better organization and deployment:

### 🧪 `/tests` - All Test Files
Contains all testing files for comprehensive system validation:

- **HRI Testing**: `comprehensive_macos_hri_test.py`, `test_integrated_construction_hri.py`
- **Component Testing**: `test_construction_detection.py`, `test_whisper_integration.py`, `test_tts_integration.py`
- **Integration Testing**: `test_complete_construction_hri_with_rag.py`, `test_rag_integration.py`
- **Performance Testing**: `test_production_readiness.py`, `test_singleton_optimization.py`
- **Vision System Testing**: `test_unified_vision_integration.py`, `test_vlm.py`
- **Robot Testing**: `RealRobotTester.py`, `test_ur_home.py`, `accuracy_tester.py`

### 🚀 `/launch` - Simulation & Gazebo Files
Contains all files necessary for Gazebo + RViz + MoveIt simulation:

- **Simulation System**: `UnifiedVisionSystemSim.py`
- **Launch Scripts**: `launch_gazebo_with_red_cube.py`, `launch_eye_in_hand.py`
- **Test Environment**: `test_unified_vision_system.py`
- **Calibration**: `create_sim_calibration.py`
- **ROS Node**: `unified_vision_system_node.py`
- **Configuration**: `config/`, `urdf/`, `worlds/` directories

### 🤖 `/deployment` - Physical Robot Deployment
Contains all files necessary for physical UR5e robot deployment:

- **Robot Control**: `ur5e_grasp_controller.py`, `UR5eKinematics.py`
- **Vision System**: `UnifiedVisionSystem.py`
- **Calibration**: `CameraCalibration.py`, `calibrate_eye_in_hand.py`, `calibrate_hand_eye.py`
- **Safety Systems**: `EyeInHandSafetyChecker.py`
- **Kinematics**: `HybridIKWrapper.py`, `HandEyeCalibrator.py`
- **Hardware Assets**: `meshes/` directory

### 🧠 `/vision` - Core AI/ML Components
Contains the main AI/ML pipeline components (unchanged location):

- **RAG System**: `EnhancedConstructionRAG.py`, `SharedRAGManager.py`
- **TTS System**: `ConstructionTTSManager.py`, `SharedTTSManager.py`
- **NLP Processing**: `ConstructionHaystackNLP.py`, `SpeechCommandProcessor.py`
- **HRI Components**: `ConstructionClarificationManager.py`
- **Research Framework**: `ExperimentalController.py`, `TrustQuestionnaire.py`, `NASATLXAssessment.py`
- **Detection**: `OWLViTDetector.py`, `DepthAwareDetector.py`, `GraspPointDetector.py`

## 🎯 Usage by Deployment Type

### For Simulation Development (Gazebo + RViz + MoveIt):
```bash
cd launch/
python launch_gazebo_with_red_cube.py
```

### For Physical Robot Deployment:
```bash
cd deployment/
python ur5e_grasp_controller.py
```

### For Running Tests:
```bash
cd tests/
python comprehensive_macos_hri_test.py
```

## 🔧 Key Benefits of This Organization

1. **Clear Separation**: Simulation vs Physical deployment files are clearly separated
2. **Easy Testing**: All tests consolidated in one location
3. **Deployment Ready**: Each directory contains everything needed for its specific use case
4. **Maintainable**: Related files are grouped together logically
5. **Scalable**: Easy to add new tests, launch configurations, or deployment scripts

## 🚀 Ready for Research Deployment

The system is now organized and ready for:
- ✅ Gazebo/RViz simulation testing
- ✅ Physical UR5e robot deployment
- ✅ Comprehensive test suite execution
- ✅ Construction HRI research studies