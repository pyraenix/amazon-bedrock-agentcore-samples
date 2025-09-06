#!/usr/bin/env python3
"""
Setup Guide for LlamaIndex-AgentCore Browser Integration

This script helps users understand and run the available setup utilities.
"""

import sys
import subprocess
from pathlib import Path

def print_header():
    """Print setup guide header"""
    print("🚀 LlamaIndex-AgentCore Browser Integration Setup Guide")
    print("=" * 60)
    print()

def print_step(step_num, title, description, command=None, optional=False):
    """Print a setup step"""
    status = "Optional" if optional else "Required"
    print(f"📋 Step {step_num}: {title} ({status})")
    print(f"   {description}")
    if command:
        print(f"   Command: {command}")
    print()

def check_python_version():
    """Check if Python 3.12+ is available"""
    if sys.version_info >= (3, 12):
        print("✅ Python 3.12+ detected")
        return True
    else:
        print(f"❌ Python 3.12+ required, found {sys.version}")
        return False

def main():
    """Main setup guide"""
    print_header()
    
    print("🔍 Pre-flight Check:")
    python_ok = check_python_version()
    print()
    
    if not python_ok:
        print("❌ Please install Python 3.12+ before proceeding")
        return False
    
    print("📝 Setup Steps:")
    print()
    
    print_step(
        1, 
        "Environment Setup",
        "Create Python 3.12 virtual environment and install dependencies",
        "python setup_env.py"
    )
    
    print_step(
        2,
        "AWS Permissions Check", 
        "Verify your AWS credentials have the required permissions for AgentCore",
        "python check_iam_permissions.py",
        optional=True
    )
    
    print_step(
        3,
        "AgentCore Configuration",
        "Configure the integration to use real AWS AgentCore services",
        "python configure_real_agentcore.py",
        optional=True
    )
    
    print_step(
        4,
        "Test Installation",
        "Run tests to verify everything is working correctly",
        "python -m pytest tests/ -v"
    )
    
    print("💡 Tips:")
    print("  - Run setup_env.py first to create your Python environment")
    print("  - Use check_iam_permissions.py to troubleshoot AWS access issues")
    print("  - configure_real_agentcore.py helps set up production configurations")
    print("  - All utilities include --help for detailed options")
    print()
    
    print("📚 Documentation:")
    print("  - README.md - Complete usage guide")
    print("  - docs/ - Detailed documentation")
    print("  - examples/ - Working code examples")
    print()
    
    print("🎯 Quick Start:")
    print("  1. python setup_env.py")
    print("  2. source venv312/bin/activate")
    print("  3. python -c 'from integration import LlamaIndexAgentCoreIntegration; print(\"✅ Import successful\")'")
    print()
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    
    print("🎉 Setup guide complete! Choose the steps that apply to your use case.")