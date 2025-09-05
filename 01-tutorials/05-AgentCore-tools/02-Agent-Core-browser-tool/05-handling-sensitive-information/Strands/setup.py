#!/usr/bin/env python3
"""
Comprehensive setup script for Strands AgentCore Browser Tool Sensitive Information Handling Tutorial.

This script installs all dependencies and configures the tutorial environment for production-ready
Strands agents with AgentCore Browser Tool integration.

Requirements: 8.1, 8.2, 8.3, 8.4
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import platform
import shutil
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('setup.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class StrandsAgentCoreSetup:
    """Comprehensive setup manager for Strands AgentCore integration tutorial."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.platform = platform.system().lower()
        self.setup_results = {
            "python_version": self.python_version,
            "platform": self.platform,
            "setup_steps": [],
            "errors": [],
            "warnings": [],
            "success": False
        }
        
    def check_python_version(self) -> bool:
        """Verify Python 3.12+ is being used."""
        logger.info(f"Checking Python version: {self.python_version}")
        
        if sys.version_info < (3, 12):
            error_msg = f"Python 3.12+ required, found {self.python_version}"
            logger.error(error_msg)
            self.setup_results["errors"].append(error_msg)
            return False
            
        logger.info("✓ Python version check passed")
        self.setup_results["setup_steps"].append("python_version_check")
        return True
        
    def check_system_dependencies(self) -> bool:
        """Check for required system dependencies."""
        logger.info("Checking system dependencies...")
        
        required_commands = {
            "git": "Git version control",
            "curl": "HTTP client for downloads",
            "unzip": "Archive extraction"
        }
        
        missing_deps = []
        for cmd, description in required_commands.items():
            if not shutil.which(cmd):
                missing_deps.append(f"{cmd} ({description})")
                
        if missing_deps:
            error_msg = f"Missing system dependencies: {', '.join(missing_deps)}"
            logger.error(error_msg)
            self.setup_results["errors"].append(error_msg)
            return False
            
        logger.info("✓ System dependencies check passed")
        self.setup_results["setup_steps"].append("system_dependencies_check")
        return True
        
    def create_virtual_environment(self) -> bool:
        """Create and activate virtual environment."""
        logger.info("Setting up virtual environment...")
        
        venv_path = self.base_dir / "venv"
        
        try:
            if venv_path.exists():
                logger.info("Virtual environment already exists, removing...")
                shutil.rmtree(venv_path)
                
            # Create virtual environment
            subprocess.run([
                sys.executable, "-m", "venv", str(venv_path)
            ], check=True, capture_output=True, text=True)
            
            logger.info("✓ Virtual environment created successfully")
            self.setup_results["setup_steps"].append("virtual_environment_creation")
            return True
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to create virtual environment: {e.stderr}"
            logger.error(error_msg)
            self.setup_results["errors"].append(error_msg)
            return False
            
    def install_dependencies(self) -> bool:
        """Install Python dependencies from requirements.txt."""
        logger.info("Installing Python dependencies...")
        
        venv_path = self.base_dir / "venv"
        if self.platform == "windows":
            pip_path = venv_path / "Scripts" / "pip"
            python_path = venv_path / "Scripts" / "python"
        else:
            pip_path = venv_path / "bin" / "pip"
            python_path = venv_path / "bin" / "python"
            
        requirements_file = self.base_dir / "requirements.txt"
        
        try:
            # Upgrade pip first
            subprocess.run([
                str(python_path), "-m", "pip", "install", "--upgrade", "pip"
            ], check=True, capture_output=True, text=True)
            
            # Install requirements
            subprocess.run([
                str(pip_path), "install", "-r", str(requirements_file)
            ], check=True, capture_output=True, text=True)
            
            logger.info("✓ Python dependencies installed successfully")
            self.setup_results["setup_steps"].append("python_dependencies_installation")
            return True
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to install dependencies: {e.stderr}"
            logger.error(error_msg)
            self.setup_results["errors"].append(error_msg)
            return False
            
    def setup_environment_files(self) -> bool:
        """Set up environment configuration files."""
        logger.info("Setting up environment configuration...")
        
        env_example = self.base_dir / ".env.example"
        env_file = self.base_dir / ".env"
        
        try:
            if env_example.exists() and not env_file.exists():
                shutil.copy2(env_example, env_file)
                logger.info("✓ Environment file created from template")
                logger.warning("Please configure .env file with your AWS credentials and settings")
                self.setup_results["warnings"].append("Environment file needs manual configuration")
            elif env_file.exists():
                logger.info("✓ Environment file already exists")
            else:
                logger.warning("No .env.example template found")
                self.setup_results["warnings"].append("No environment template available")
                
            self.setup_results["setup_steps"].append("environment_file_setup")
            return True
            
        except Exception as e:
            error_msg = f"Failed to setup environment files: {str(e)}"
            logger.error(error_msg)
            self.setup_results["errors"].append(error_msg)
            return False
            
    def setup_jupyter_kernel(self) -> bool:
        """Set up Jupyter kernel for the virtual environment."""
        logger.info("Setting up Jupyter kernel...")
        
        venv_path = self.base_dir / "venv"
        if self.platform == "windows":
            python_path = venv_path / "Scripts" / "python"
        else:
            python_path = venv_path / "bin" / "python"
            
        try:
            # Install ipykernel and register kernel
            subprocess.run([
                str(python_path), "-m", "ipykernel", "install", 
                "--user", "--name", "strands-agentcore", 
                "--display-name", "Strands AgentCore Tutorial"
            ], check=True, capture_output=True, text=True)
            
            logger.info("✓ Jupyter kernel registered successfully")
            self.setup_results["setup_steps"].append("jupyter_kernel_setup")
            return True
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to setup Jupyter kernel: {e.stderr}"
            logger.error(error_msg)
            self.setup_results["errors"].append(error_msg)
            return False
            
    def create_directories(self) -> bool:
        """Create necessary directories for the tutorial."""
        logger.info("Creating tutorial directories...")
        
        directories = [
            "logs",
            "tutorial_data",
            "tutorial_data/credentials",
            "tutorial_data/outputs",
            "tutorial_data/temp",
            "tutorial_data/cache"
        ]
        
        try:
            for directory in directories:
                dir_path = self.base_dir / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                
            logger.info("✓ Tutorial directories created successfully")
            self.setup_results["setup_steps"].append("directory_creation")
            return True
            
        except Exception as e:
            error_msg = f"Failed to create directories: {str(e)}"
            logger.error(error_msg)
            self.setup_results["errors"].append(error_msg)
            return False
            
    def validate_installation(self) -> bool:
        """Validate that all components are properly installed."""
        logger.info("Validating installation...")
        
        venv_path = self.base_dir / "venv"
        if self.platform == "windows":
            python_path = venv_path / "Scripts" / "python"
        else:
            python_path = venv_path / "bin" / "python"
            
        validation_script = '''
import sys
import importlib

required_packages = [
    "strands_agents",
    "boto3",
    "anthropic",
    "selenium",
    "pandas",
    "cryptography",
    "jupyter",
    "pytest"
]

missing_packages = []
for package in required_packages:
    try:
        importlib.import_module(package.replace("-", "_"))
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f"MISSING: {','.join(missing_packages)}")
    sys.exit(1)
else:
    print("SUCCESS: All packages available")
    sys.exit(0)
'''
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(validation_script)
                temp_script = f.name
                
            result = subprocess.run([
                str(python_path), temp_script
            ], capture_output=True, text=True)
            
            os.unlink(temp_script)
            
            if result.returncode == 0:
                logger.info("✓ Installation validation passed")
                self.setup_results["setup_steps"].append("installation_validation")
                return True
            else:
                error_msg = f"Installation validation failed: {result.stdout}"
                logger.error(error_msg)
                self.setup_results["errors"].append(error_msg)
                return False
                
        except Exception as e:
            error_msg = f"Failed to validate installation: {str(e)}"
            logger.error(error_msg)
            self.setup_results["errors"].append(error_msg)
            return False
            
    def generate_setup_report(self) -> None:
        """Generate comprehensive setup report."""
        logger.info("Generating setup report...")
        
        self.setup_results["success"] = len(self.setup_results["errors"]) == 0
        
        report_file = self.base_dir / "setup_report.json"
        with open(report_file, 'w') as f:
            json.dump(self.setup_results, f, indent=2)
            
        # Generate human-readable report
        readme_content = f"""# Strands AgentCore Setup Report

## Setup Summary
- **Status**: {'✓ SUCCESS' if self.setup_results['success'] else '✗ FAILED'}
- **Python Version**: {self.setup_results['python_version']}
- **Platform**: {self.setup_results['platform']}
- **Completed Steps**: {len(self.setup_results['setup_steps'])}

## Completed Setup Steps
{chr(10).join(f"- {step}" for step in self.setup_results['setup_steps'])}

## Errors
{chr(10).join(f"- {error}" for error in self.setup_results['errors']) if self.setup_results['errors'] else "None"}

## Warnings
{chr(10).join(f"- {warning}" for warning in self.setup_results['warnings']) if self.setup_results['warnings'] else "None"}

## Next Steps
{'1. Review and fix any errors above' if self.setup_results['errors'] else '1. Configure your .env file with AWS credentials'}
2. Run validation tests: `python validate_integration.py`
3. Start with tutorial notebook: `jupyter lab 01_strands_agentcore_secure_login.ipynb`

## Support
If you encounter issues, check:
- Python version is 3.12+
- AWS credentials are properly configured
- All system dependencies are installed
"""
        
        readme_file = self.base_dir / "SETUP_REPORT.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
            
        logger.info(f"Setup report generated: {report_file}")
        
    def run_setup(self) -> bool:
        """Run complete setup process."""
        logger.info("Starting Strands AgentCore tutorial setup...")
        
        setup_steps = [
            ("Python Version Check", self.check_python_version),
            ("System Dependencies Check", self.check_system_dependencies),
            ("Virtual Environment Creation", self.create_virtual_environment),
            ("Python Dependencies Installation", self.install_dependencies),
            ("Environment Files Setup", self.setup_environment_files),
            ("Jupyter Kernel Setup", self.setup_jupyter_kernel),
            ("Directory Creation", self.create_directories),
            ("Installation Validation", self.validate_installation)
        ]
        
        for step_name, step_func in setup_steps:
            logger.info(f"Running: {step_name}")
            if not step_func():
                logger.error(f"Setup failed at: {step_name}")
                self.generate_setup_report()
                return False
                
        self.generate_setup_report()
        logger.info("✓ Setup completed successfully!")
        
        if self.setup_results["warnings"]:
            logger.warning("Setup completed with warnings. Please review SETUP_REPORT.md")
            
        return True

def main():
    """Main setup function."""
    print("Strands AgentCore Browser Tool - Sensitive Information Handling Tutorial Setup")
    print("=" * 80)
    
    setup_manager = StrandsAgentCoreSetup()
    
    try:
        success = setup_manager.run_setup()
        
        if success:
            print("\n✓ Setup completed successfully!")
            print("\nNext steps:")
            print("1. Configure your .env file with AWS credentials")
            print("2. Run: python validate_integration.py")
            print("3. Start tutorial: jupyter lab 01_strands_agentcore_secure_login.ipynb")
            return 0
        else:
            print("\n✗ Setup failed. Check setup.log and SETUP_REPORT.md for details.")
            return 1
            
    except KeyboardInterrupt:
        print("\nSetup interrupted by user.")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error during setup: {str(e)}")
        print(f"\n✗ Setup failed with unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())