#!/usr/bin/env python3
"""
LlamaIndex with AgentCore Browser Tool - Setup Script

This script helps set up the tutorial environment by:
1. Creating necessary directories
2. Installing dependencies
3. Setting up environment configuration
4. Running initial validation

Usage:
    python setup.py [--install-deps] [--create-env] [--validate]
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional

def run_command(command: List[str], cwd: Optional[Path] = None) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)

def check_python_version() -> bool:
    """Check if Python version is 3.9 or higher."""
    version = sys.version_info
    if version >= (3, 9):
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.9+)")
        return False

def check_pip_available() -> bool:
    """Check if pip is available."""
    exit_code, _, _ = run_command([sys.executable, "-m", "pip", "--version"])
    if exit_code == 0:
        print("✅ pip is available")
        return True
    else:
        print("❌ pip is not available")
        return False

def install_dependencies() -> bool:
    """Install dependencies from requirements.txt."""
    print("📦 Installing dependencies...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    # Upgrade pip first
    print("   Upgrading pip...")
    exit_code, stdout, stderr = run_command([
        sys.executable, "-m", "pip", "install", "--upgrade", "pip"
    ])
    
    if exit_code != 0:
        print(f"⚠️  Warning: Failed to upgrade pip: {stderr}")
    
    # Install dependencies
    print("   Installing packages from requirements.txt...")
    exit_code, stdout, stderr = run_command([
        sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
    ])
    
    if exit_code == 0:
        print("✅ Dependencies installed successfully")
        return True
    else:
        print(f"❌ Failed to install dependencies: {stderr}")
        return False

def create_env_file() -> bool:
    """Create .env file from .env.example if it doesn't exist."""
    env_example = Path(__file__).parent / ".env.example"
    env_file = Path(__file__).parent / ".env"
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if not env_example.exists():
        print("❌ .env.example not found")
        return False
    
    try:
        shutil.copy2(env_example, env_file)
        print("✅ Created .env file from .env.example")
        print("📝 Please edit .env file with your actual configuration values")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def create_directories() -> bool:
    """Create necessary directories."""
    base_path = Path(__file__).parent
    directories = [
        "examples",
        "assets", 
        "tutorial_data",
        "logs",
        "vector_store"
    ]
    
    created = []
    for directory in directories:
        dir_path = base_path / directory
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                created.append(directory)
            except Exception as e:
                print(f"❌ Failed to create directory {directory}: {e}")
                return False
    
    if created:
        print(f"✅ Created directories: {', '.join(created)}")
    else:
        print("✅ All directories already exist")
    
    return True

def run_validation() -> bool:
    """Run the validation script."""
    print("🔍 Running validation...")
    
    validation_script = Path(__file__).parent / "validate_integration.py"
    if not validation_script.exists():
        print("❌ validate_integration.py not found")
        return False
    
    exit_code, stdout, stderr = run_command([
        sys.executable, str(validation_script), "--skip-aws", "--skip-browser"
    ])
    
    print(stdout)
    if stderr:
        print(stderr)
    
    if exit_code == 0:
        print("✅ Validation completed successfully")
        return True
    else:
        print("⚠️  Validation completed with issues (see output above)")
        return False

def main():
    """Main setup function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Set up LlamaIndex-AgentCore Browser Tool tutorial environment"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install Python dependencies"
    )
    parser.add_argument(
        "--create-env",
        action="store_true",
        help="Create .env file from template"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after setup"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all setup steps"
    )
    
    args = parser.parse_args()
    
    # If no specific flags, run all steps
    if not any([args.install_deps, args.create_env, args.validate]):
        args.all = True
    
    print("🚀 LlamaIndex-AgentCore Browser Tool Tutorial Setup")
    print("=" * 60)
    
    # Check prerequisites
    if not check_python_version():
        print("\n❌ Setup failed: Python 3.9+ required")
        sys.exit(1)
    
    if not check_pip_available():
        print("\n❌ Setup failed: pip not available")
        sys.exit(1)
    
    # Create directories
    print("\n📁 Setting up directories...")
    if not create_directories():
        print("\n❌ Setup failed: Could not create directories")
        sys.exit(1)
    
    # Install dependencies
    if args.install_deps or args.all:
        print("\n📦 Installing dependencies...")
        if not install_dependencies():
            print("\n❌ Setup failed: Could not install dependencies")
            sys.exit(1)
    
    # Create environment file
    if args.create_env or args.all:
        print("\n⚙️  Setting up environment...")
        if not create_env_file():
            print("\n❌ Setup failed: Could not create .env file")
            sys.exit(1)
    
    # Run validation
    if args.validate or args.all:
        print("\n🔍 Running validation...")
        run_validation()  # Don't fail setup if validation has warnings
    
    print("\n" + "=" * 60)
    print("✅ Setup completed!")
    print("\n📋 Next steps:")
    print("   1. Edit the .env file with your AWS credentials and configuration")
    print("   2. Run: python validate_integration.py")
    print("   3. Start with the first tutorial notebook")
    print("\n📚 Tutorial notebooks:")
    print("   • 01_llamaindex_agentcore_secure_integration.ipynb")
    print("   • 02_llamaindex_sensitive_rag_pipeline.ipynb") 
    print("   • 03_llamaindex_authenticated_web_services.ipynb")
    print("   • 04_production_llamaindex_agentcore_patterns.ipynb")

if __name__ == "__main__":
    main()