#!/usr/bin/env python3
"""
Test Environment Validation Script for UnifiedVisionSystem

This script validates that all dependencies and system requirements
are properly installed and configured for running the test suite.

Usage:
    python3 validate_test_environment.py [--fix] [--verbose]

Options:
    --fix      Attempt to fix common issues automatically
    --verbose  Show detailed information about each check
    --quick    Run only essential checks (faster)
"""

import subprocess
import sys
import os
import platform
import importlib
import argparse
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional

class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class ValidationResult:
    """Result of a validation check."""
    def __init__(self, name: str, passed: bool, message: str = "", 
                 suggestion: str = "", critical: bool = True):
        self.name = name
        self.passed = passed
        self.message = message
        self.suggestion = suggestion
        self.critical = critical

class TestEnvironmentValidator:
    """
    Comprehensive validation of the test environment.
    """
    
    def __init__(self, verbose: bool = False, fix_issues: bool = False):
        self.verbose = verbose
        self.fix_issues = fix_issues
        self.results: List[ValidationResult] = []
        self.system_info = self.get_system_info()
        
    def get_system_info(self) -> Dict:
        """Get system information."""
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "architecture": platform.architecture(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
        }
    
    def print_header(self):
        """Print validation header."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}UnifiedVisionSystem Test Environment Validation{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")
        
        if self.verbose:
            print(f"{Colors.BOLD}System Information:{Colors.END}")
            for key, value in self.system_info.items():
                print(f"  {key}: {value}")
            print()
    
    def run_command(self, cmd: List[str], capture_output: bool = True) -> Tuple[bool, str]:
        """Run a shell command and return success status and output."""
        try:
            result = subprocess.run(
                cmd, 
                capture_output=capture_output, 
                text=True, 
                timeout=30
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except FileNotFoundError:
            return False, f"Command not found: {cmd[0]}"
        except Exception as e:
            return False, str(e)
    
    def check_python_version(self) -> ValidationResult:
        """Check Python version compatibility."""
        version = sys.version_info
        required_major, required_minor = 3, 8
        
        if version.major >= required_major and version.minor >= required_minor:
            return ValidationResult(
                "Python Version",
                True,
                f"Python {version.major}.{version.minor}.{version.micro}",
                critical=True
            )
        else:
            return ValidationResult(
                "Python Version",
                False,
                f"Python {version.major}.{version.minor}.{version.micro} (requires >= {required_major}.{required_minor})",
                f"Please upgrade to Python {required_major}.{required_minor}+",
                critical=True
            )
    
    def check_ros2_installation(self) -> ValidationResult:
        """Check if ROS2 is properly installed."""
        success, output = self.run_command(['which', 'ros2'])
        
        if not success:
            return ValidationResult(
                "ROS2 Installation",
                False,
                "ROS2 command not found",
                "Install ROS2 Humble: https://docs.ros.org/en/humble/Installation.html",
                critical=True
            )
        
        # Check ROS2 version
        success, version_output = self.run_command(['ros2', '--version'])
        if success:
            return ValidationResult(
                "ROS2 Installation",
                True,
                f"ROS2 found at {output.strip()}, {version_output.strip()}",
                critical=True
            )
        else:
            return ValidationResult(
                "ROS2 Installation",
                False,
                "ROS2 found but version check failed",
                "Check ROS2 installation integrity",
                critical=True
            )
    
    def check_python_packages(self) -> List[ValidationResult]:
        """Check required Python packages."""
        required_packages = {
            'numpy': 'numpy>=1.21.0',
            'opencv-python': 'cv2',
            'torch': 'torch>=1.12.0',
            'transformers': 'transformers>=4.20.0',
            'pyrealsense2': 'pyrealsense2>=2.50.0',
            'scipy': 'scipy>=1.7.0',
            'matplotlib': 'matplotlib>=3.5.0',
            'pillow': 'PIL',
            'spacy': 'spacy>=3.4.0',
            'rclpy': 'rclpy (ROS2 Python client)',
        }
        
        optional_packages = {
            'ur_ikfast': 'ur_ikfast (Fast analytical IK)',
            'speech_recognition': 'SpeechRecognition>=3.8.1',
            'sklearn': 'scikit-learn (Machine learning utilities)',
        }
        
        results = []
        
        # Check required packages
        for package_name, description in required_packages.items():
            try:
                if package_name == 'opencv-python':
                    import cv2
                    version = cv2.__version__
                    results.append(ValidationResult(
                        f"Python Package: {package_name}",
                        True,
                        f"OpenCV {version}",
                        critical=True
                    ))
                elif package_name == 'pillow':
                    import PIL
                    version = PIL.__version__
                    results.append(ValidationResult(
                        f"Python Package: {package_name}",
                        True,
                        f"Pillow {version}",
                        critical=True
                    ))
                else:
                    module = importlib.import_module(package_name)
                    version = getattr(module, '__version__', 'unknown')
                    results.append(ValidationResult(
                        f"Python Package: {package_name}",
                        True,
                        f"{package_name} {version}",
                        critical=True
                    ))
            except ImportError as e:
                results.append(ValidationResult(
                    f"Python Package: {package_name}",
                    False,
                    f"Import failed: {e}",
                    f"Install with: pip install {description}",
                    critical=True
                ))
        
        # Check optional packages
        for package_name, description in optional_packages.items():
            try:
                module = importlib.import_module(package_name)
                version = getattr(module, '__version__', 'unknown')
                results.append(ValidationResult(
                    f"Optional Package: {package_name}",
                    True,
                    f"{package_name} {version}",
                    critical=False
                ))
            except ImportError:
                results.append(ValidationResult(
                    f"Optional Package: {package_name}",
                    False,
                    "Not installed (optional)",
                    f"Install with: pip install {description}",
                    critical=False
                ))
        
        return results
    
    def check_ros2_packages(self) -> List[ValidationResult]:
        """Check required ROS2 packages."""
        required_packages = [
            'ur_description',
            'ur_moveit_config', 
            'gazebo_ros',
            'controller_manager',
            'joint_state_publisher',
            'robot_state_publisher',
            'rviz2',
            'moveit_core',
        ]
        
        optional_packages = [
            'ur_robot_driver',
            'realsense2_camera',
            'moveit_rviz_plugin',
        ]
        
        results = []
        
        # Check if packages are available
        success, output = self.run_command(['ros2', 'pkg', 'list'])
        
        if not success:
            results.append(ValidationResult(
                "ROS2 Package List",
                False,
                "Cannot list ROS2 packages",
                "Check ROS2 workspace setup",
                critical=True
            ))
            return results
        
        available_packages = output.split('\n')
        
        # Check required packages
        for package in required_packages:
            if package in available_packages:
                results.append(ValidationResult(
                    f"ROS2 Package: {package}",
                    True,
                    "Available",
                    critical=True
                ))
            else:
                results.append(ValidationResult(
                    f"ROS2 Package: {package}",
                    False,
                    "Not found",
                    f"Install with: sudo apt install ros-humble-{package.replace('_', '-')}",
                    critical=True
                ))
        
        # Check optional packages
        for package in optional_packages:
            if package in available_packages:
                results.append(ValidationResult(
                    f"Optional ROS2 Package: {package}",
                    True,
                    "Available",
                    critical=False
                ))
            else:
                results.append(ValidationResult(
                    f"Optional ROS2 Package: {package}",
                    False,
                    "Not found (optional)",
                    f"Install with: sudo apt install ros-humble-{package.replace('_', '-')}",
                    critical=False
                ))
        
        return results
    
    def check_gazebo_installation(self) -> ValidationResult:
        """Check Gazebo simulation environment."""
        success, output = self.run_command(['which', 'gazebo'])
        
        if not success:
            return ValidationResult(
                "Gazebo Simulation",
                False,
                "Gazebo not found",
                "Install with: sudo apt install gazebo",
                critical=True
            )
        
        # Check version
        success, version_output = self.run_command(['gazebo', '--version'])
        if success:
            return ValidationResult(
                "Gazebo Simulation",
                True,
                f"Gazebo found: {version_output.strip()}",
                critical=True
            )
        else:
            return ValidationResult(
                "Gazebo Simulation",
                True,
                f"Gazebo found at {output.strip()}",
                critical=True
            )
    
    def check_environment_variables(self) -> List[ValidationResult]:
        """Check important environment variables."""
        results = []
        
        # ROS2 environment
        ros_distro = os.environ.get('ROS_DISTRO')
        if ros_distro:
            results.append(ValidationResult(
                "ROS_DISTRO",
                True,
                f"ROS_DISTRO={ros_distro}",
                critical=True
            ))
        else:
            results.append(ValidationResult(
                "ROS_DISTRO",
                False,
                "ROS_DISTRO not set",
                "Source ROS2 setup: source /opt/ros/humble/setup.bash",
                critical=True
            ))
        
        # ROS Domain ID
        ros_domain_id = os.environ.get('ROS_DOMAIN_ID')
        if ros_domain_id:
            results.append(ValidationResult(
                "ROS_DOMAIN_ID",
                True,
                f"ROS_DOMAIN_ID={ros_domain_id}",
                critical=False
            ))
        else:
            results.append(ValidationResult(
                "ROS_DOMAIN_ID",
                False,
                "ROS_DOMAIN_ID not set (will use default)",
                "Set with: export ROS_DOMAIN_ID=42",
                critical=False
            ))
        
        # Gazebo model path
        gazebo_model_path = os.environ.get('GAZEBO_MODEL_PATH')
        if gazebo_model_path:
            results.append(ValidationResult(
                "GAZEBO_MODEL_PATH",
                True,
                "Gazebo model path configured",
                critical=False
            ))
        else:
            results.append(ValidationResult(
                "GAZEBO_MODEL_PATH",
                False,
                "GAZEBO_MODEL_PATH not set (may affect model loading)",
                "Set Gazebo model paths if needed",
                critical=False
            ))
        
        return results
    
    def check_hardware_capabilities(self) -> List[ValidationResult]:
        """Check hardware capabilities for simulation."""
        results = []
        
        # Check available memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            if memory_gb >= 8:
                results.append(ValidationResult(
                    "System Memory",
                    True,
                    f"{memory_gb:.1f}GB available",
                    critical=False
                ))
            else:
                results.append(ValidationResult(
                    "System Memory",
                    False,
                    f"{memory_gb:.1f}GB (recommended: 8GB+)",
                    "Consider adding more RAM for better performance",
                    critical=False
                ))
        except ImportError:
            results.append(ValidationResult(
                "System Memory",
                False,
                "Cannot check (psutil not available)",
                "Install psutil: pip install psutil",
                critical=False
            ))
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                results.append(ValidationResult(
                    "GPU Acceleration",
                    True,
                    f"CUDA available: {gpu_count} device(s), {gpu_name}",
                    critical=False
                ))
            else:
                results.append(ValidationResult(
                    "GPU Acceleration",
                    False,
                    "CUDA not available (will use CPU)",
                    "GPU acceleration recommended for VLM models",
                    critical=False
                ))
        except ImportError:
            results.append(ValidationResult(
                "GPU Acceleration",
                False,
                "Cannot check (PyTorch not available)",
                "Install PyTorch to check GPU support",
                critical=False
            ))
        
        return results
    
    def check_file_permissions(self) -> List[ValidationResult]:
        """Check file system permissions."""
        results = []
        
        # Check if we can create test directories
        test_dirs = ['test_logs', 'simulation_test_logs', 'temp_test']
        
        for dir_name in test_dirs:
            try:
                test_path = Path(dir_name)
                test_path.mkdir(exist_ok=True)
                
                # Test write permissions
                test_file = test_path / 'permission_test.txt'
                test_file.write_text('test')
                test_file.unlink()
                
                if dir_name != 'temp_test':  # Keep the log directories
                    results.append(ValidationResult(
                        f"Directory Permissions: {dir_name}",
                        True,
                        "Read/write access OK",
                        critical=False
                    ))
                else:
                    test_path.rmdir()  # Clean up temp test directory
                    
            except PermissionError:
                results.append(ValidationResult(
                    f"Directory Permissions: {dir_name}",
                    False,
                    "Permission denied",
                    f"Check write permissions for {dir_name}",
                    critical=False
                ))
            except Exception as e:
                results.append(ValidationResult(
                    f"Directory Permissions: {dir_name}",
                    False,
                    f"Error: {e}",
                    "Check file system permissions",
                    critical=False
                ))
        
        return results
    
    def attempt_fixes(self):
        """Attempt to fix common issues automatically."""
        if not self.fix_issues:
            return
        
        print(f"\n{Colors.YELLOW}🔧 Attempting to fix common issues...{Colors.END}\n")
        
        # Create necessary directories
        for dir_name in ['test_logs', 'simulation_test_logs']:
            try:
                Path(dir_name).mkdir(exist_ok=True)
                print(f"  ✅ Created directory: {dir_name}")
            except Exception as e:
                print(f"  ❌ Failed to create {dir_name}: {e}")
        
        # Set ROS_DOMAIN_ID if not set
        if not os.environ.get('ROS_DOMAIN_ID'):
            os.environ['ROS_DOMAIN_ID'] = '42'
            print(f"  ✅ Set ROS_DOMAIN_ID=42")
        
        # Try to install missing Python packages
        missing_packages = []
        for result in self.results:
            if (not result.passed and 
                result.critical and 
                "Python Package:" in result.name and
                "pip install" in result.suggestion):
                package_name = result.name.split(": ")[-1]
                missing_packages.append(package_name)
        
        if missing_packages:
            print(f"  🔄 Attempting to install missing Python packages...")
            for package in missing_packages:
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                    print(f"    ✅ Installed {package}")
                except subprocess.CalledProcessError:
                    print(f"    ❌ Failed to install {package}")
    
    def run_validation(self, quick: bool = False) -> bool:
        """Run complete validation."""
        self.print_header()
        
        print(f"{Colors.BOLD}Running validation checks...{Colors.END}\n")
        
        # Core system checks
        self.results.append(self.check_python_version())
        self.results.append(self.check_ros2_installation())
        self.results.append(self.check_gazebo_installation())
        
        # Package checks
        self.results.extend(self.check_python_packages())
        if not quick:
            self.results.extend(self.check_ros2_packages())
        
        # Environment checks
        self.results.extend(self.check_environment_variables())
        
        if not quick:
            # Hardware and system checks
            self.results.extend(self.check_hardware_capabilities())
            self.results.extend(self.check_file_permissions())
        
        # Attempt fixes if requested
        self.attempt_fixes()
        
        # Print results
        self.print_results()
        
        # Return overall success
        critical_failures = [r for r in self.results if not r.passed and r.critical]
        return len(critical_failures) == 0
    
    def print_results(self):
        """Print validation results."""
        print(f"\n{Colors.BOLD}Validation Results:{Colors.END}\n")
        
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)
        critical_failures = [r for r in self.results if not r.passed and r.critical]
        
        # Group results by category
        categories = {}
        for result in self.results:
            category = result.name.split(':')[0] if ':' in result.name else "General"
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        # Print results by category
        for category, results in categories.items():
            print(f"{Colors.BOLD}{Colors.UNDERLINE}{category}:{Colors.END}")
            
            for result in results:
                status_color = Colors.GREEN if result.passed else (Colors.RED if result.critical else Colors.YELLOW)
                status_symbol = "✅" if result.passed else ("❌" if result.critical else "⚠️")
                
                print(f"  {status_symbol} {status_color}{result.name}{Colors.END}")
                
                if self.verbose or not result.passed:
                    if result.message:
                        print(f"    📋 {result.message}")
                    if result.suggestion and not result.passed:
                        print(f"    💡 {result.suggestion}")
                
            print()
        
        # Summary
        print(f"{Colors.BOLD}Summary:{Colors.END}")
        print(f"  Total checks: {total_count}")
        print(f"  Passed: {Colors.GREEN}{passed_count}{Colors.END}")
        print(f"  Failed: {Colors.RED}{total_count - passed_count}{Colors.END}")
        print(f"  Critical failures: {Colors.RED}{len(critical_failures)}{Colors.END}")
        
        if len(critical_failures) == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 Environment validation PASSED!{Colors.END}")
            print(f"{Colors.GREEN}Your system is ready to run the UnifiedVisionSystem tests.{Colors.END}")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}❌ Environment validation FAILED!{Colors.END}")
            print(f"{Colors.RED}Please fix the critical issues before running tests.{Colors.END}")
            
            print(f"\n{Colors.BOLD}Critical Issues to Fix:{Colors.END}")
            for failure in critical_failures:
                print(f"  • {failure.name}: {failure.suggestion}")
    
    def save_report(self, filename: str = "validation_report.json"):
        """Save validation report to file."""
        report = {
            "timestamp": time.time(),
            "system_info": self.system_info,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "suggestion": r.suggestion,
                    "critical": r.critical
                }
                for r in self.results
            ],
            "summary": {
                "total_checks": len(self.results),
                "passed_checks": sum(1 for r in self.results if r.passed),
                "critical_failures": sum(1 for r in self.results if not r.passed and r.critical),
                "overall_success": sum(1 for r in self.results if not r.passed and r.critical) == 0
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📄 Validation report saved to: {filename}")
        except Exception as e:
            print(f"\n❌ Failed to save report: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Validate UnifiedVisionSystem test environment"
    )
    parser.add_argument(
        '--fix', 
        action='store_true',
        help='Attempt to fix common issues automatically'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed information about each check'
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Run only essential checks (faster)'
    )
    parser.add_argument(
        '--report', '-r',
        type=str,
        default=None,
        help='Save validation report to specified file'
    )
    
    args = parser.parse_args()
    
    # Create validator
    validator = TestEnvironmentValidator(
        verbose=args.verbose,
        fix_issues=args.fix
    )
    
    # Run validation
    success = validator.run_validation(quick=args.quick)
    
    # Save report if requested
    if args.report:
        validator.save_report(args.report)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()