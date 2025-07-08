# Physical Testing Checklist for Hybrid IK System

This checklist covers all critical factors to consider before deploying the UnifiedVisionSystem with hybrid IK on physical UR5e hardware.

## 🛡️ Safety Considerations (CRITICAL)

### Pre-Testing Safety
- [ ] **Emergency stop button** easily accessible and tested
- [ ] **Workspace cleared** of personnel and obstacles
- [ ] **Safety fence/barriers** in place around robot workspace
- [ ] **Robot speed limits** configured for testing (start with 10-20% max speed)
- [ ] **Force/torque limits** properly configured on robot controller
- [ ] **Collision detection** enabled on robot controller
- [ ] **Manual mode** available to override autonomous operation

### Joint Limits Verification
- [ ] **Software joint limits** match physical robot limits exactly
- [ ] **Verify UR5e DH parameters** against robot documentation
- [ ] **Test joint limit enforcement** with small movements first
- [ ] **Check singularity handling** near workspace boundaries

## 🔧 Hardware Prerequisites

### Robot Controller Setup
- [ ] **UR5e controller** firmware version documented and compatible
- [ ] **Robot calibration** up to date (teach pendant → Installation → Calibration)
- [ ] **Tool center point (TCP)** properly defined for end effector
- [ ] **Payload configuration** matches actual end effector weight
- [ ] **Network connectivity** stable between control computer and robot

### Camera System
- [ ] **Camera calibration** completed and saved (`CameraCalibration.py`)
- [ ] **Hand-eye calibration** completed and verified (`HandEyeCalibrator.py`)
- [ ] **Camera mount** secure and vibration-free
- [ ] **Lighting conditions** consistent and adequate
- [ ] **Camera field of view** covers entire workspace

### Workspace Setup
- [ ] **Robot base coordinate frame** properly established
- [ ] **Camera coordinate frame** aligned with robot frame via hand-eye calibration
- [ ] **Object placement area** within both camera view and robot reach
- [ ] **Table/surface height** measured and configured in software

## 💻 Software Dependencies

### Core Dependencies
- [ ] **ur_ikfast** installed and tested (`pip install ur-ikfast`)
- [ ] **scipy** version >= 1.7.0 for robust quaternion calculations
- [ ] **OpenCV** for vision processing
- [ ] **ROS2** environment properly sourced
- [ ] **UR5e ROS2 driver** installed and functional

### Configuration Files
- [ ] **hybrid_ik_config.yaml** parameters reviewed and adjusted for your setup
- [ ] **Camera intrinsics** loaded correctly
- [ ] **Hand-eye transformation** matrix verified
- [ ] **Workspace boundaries** defined in configuration

## 🧪 Pre-Deployment Testing

### Offline Testing (No Physical Robot)
```bash
# Run comprehensive tests
cd vision/
python test_unified_vision_integration.py

# Test hybrid IK specifically
python -c "from UR5eKinematics import test_hybrid_kinematics; test_hybrid_kinematics()"
```

### Simulation Testing
- [ ] **Test in robot simulator** first (URSim, Gazebo, or MoveIt)
- [ ] **Verify trajectory planning** doesn't cause collisions
- [ ] **Test edge cases** (unreachable poses, singularities)
- [ ] **Performance benchmarking** on target hardware

## 🔗 ROS2 Integration

### ROS2 Environment
- [ ] **ROS2 workspace** built and sourced
- [ ] **UR5e driver nodes** running and responsive
- [ ] **Topic communication** verified (`ros2 topic list`, `ros2 topic echo`)
- [ ] **Service calls** to robot controller working
- [ ] **Parameter server** accessible

### Message Types
- [ ] **JointTrajectory messages** properly formatted
- [ ] **Pose/Transform messages** coordinate frames correct
- [ ] **Action servers** for robot control available

## 📊 Performance Validation

### Benchmarking
- [ ] **IK solving speed** measured on target hardware
- [ ] **Success rate** tested with various poses
- [ ] **Memory usage** monitored during operation
- [ ] **Network latency** measured for ROS2 communication

### Accuracy Testing
- [ ] **End effector positioning accuracy** verified with physical measurements
- [ ] **Repeatability testing** (same command → same result)
- [ ] **Vision-to-robot coordinate accuracy** validated

## 🔍 Vision System Validation

### Object Detection
- [ ] **OWL-ViT model** loading correctly
- [ ] **Detection confidence thresholds** tuned for your objects
- [ ] **Grasp point detection** providing reasonable results
- [ ] **Depth estimation** accurate within workspace

### Coordinate Transformations
- [ ] **Camera to robot** transformations verified
- [ ] **Pixel to 3D world** coordinates accurate
- [ ] **Object pose estimation** tested with known objects

## ⚙️ Configuration Parameters

### Critical Parameters to Verify
```yaml
# hybrid_ik_config.yaml
kinematics:
  enable_hybrid_ik: true          # Enable hybrid solver
  ik_timeout_ms: 100              # Conservative timeout for physical testing
  ik_max_position_error_mm: 2.0   # Tight tolerance for accuracy
  ik_enable_approximation: false  # Disable for initial testing

robot:
  max_velocity: 0.1               # Start with slow movements (10% of max)
  max_acceleration: 0.1           # Conservative acceleration
  joint_velocity_limits: [...]    # Verify these match your robot
```

### UnifiedVisionSystem Parameters
```yaml
# ROS2 parameters
enable_hybrid_ik: true
ik_timeout_ms: 100
ik_debug: true                    # Enable for initial testing
safety_check_enabled: true       # Always keep enabled
workspace_bounds: [...]           # Verify these are correct
```

## 🚀 Deployment Workflow

### Phase 1: Basic Movement Testing
1. **Start with simple poses** in center of workspace
2. **Verify forward kinematics** matches actual robot position
3. **Test single joint movements** first
4. **Gradually increase complexity**

### Phase 2: Vision Integration
1. **Test object detection** without robot movement
2. **Verify coordinate transformations** with static objects
3. **Test grasp point calculation** accuracy
4. **Validate depth perception**

### Phase 3: Integrated Testing
1. **Simple pick operations** with known objects
2. **Test error handling** (unreachable poses, vision failures)
3. **Performance optimization** based on real-world results
4. **Stress testing** with continuous operation

## 🔧 Troubleshooting Checklist

### Common Issues and Solutions

#### IK Solver Issues
- **No solutions found**: Check workspace bounds, joint limits, singularities
- **Slow performance**: Verify ur_ikfast installation, reduce timeout
- **Inaccurate solutions**: Verify DH parameters, check robot calibration

#### Vision Issues  
- **Poor detection**: Check lighting, camera focus, model parameters
- **Coordinate errors**: Re-run hand-eye calibration, verify transformations
- **Depth inaccuracy**: Camera calibration, stereo setup if using

#### ROS2 Communication
- **Node discovery fails**: Check ROS_DOMAIN_ID, network configuration
- **Message delays**: Monitor network traffic, CPU usage
- **Service timeouts**: Increase timeout values, check robot controller

## 📝 Testing Log Template

Create a testing log to track your progress:

```
Date: ___________
Hardware Setup: UR5e + [Camera Model] + [End Effector]
Software Version: [Git commit hash]

Pre-flight Checks:
□ Safety systems verified
□ Hardware connections secure  
□ Software dependencies installed
□ Configuration parameters verified

Test Results:
- IK Success Rate: ___% (analytical), ___% (numerical)
- Average IK Time: ___ms (analytical), ___ms (numerical)
- Vision Detection Rate: ___%
- Coordinate Accuracy: ___mm average error
- Grasp Success Rate: ___%

Issues Encountered:
1. [Issue description]
   - Solution: [What was done]
   - Status: [Resolved/Ongoing]

Notes:
[Additional observations, performance notes, areas for improvement]
```

## 🎯 Success Criteria

Your system is ready for production when:
- [ ] **Safety systems** tested and functional
- [ ] **IK success rate** > 95% for workspace poses
- [ ] **IK solve time** < 50ms average (with ur_ikfast)
- [ ] **Vision accuracy** < 5mm position error
- [ ] **Grasp success rate** > 80% for target objects
- [ ] **System stability** demonstrated over 1+ hour continuous operation
- [ ] **Error handling** gracefully manages all failure modes

## 🆘 Emergency Procedures

### If Something Goes Wrong
1. **Press emergency stop** immediately
2. **Document** what happened (robot position, command sent, error messages)
3. **Check** for any physical damage
4. **Review logs** before restarting
5. **Start with slower/safer** parameters

### Recovery Procedures
- **Robot in safe position**: Use teach pendant to manually move to home
- **Software crash**: Restart nodes in correct order (driver first, then vision)
- **Communication loss**: Check network, restart ROS2 daemon
- **Calibration drift**: Re-run camera and hand-eye calibration

Remember: **Safety first, speed second**. It's better to start conservatively and gradually increase performance than to risk damage to equipment or injury to personnel. 