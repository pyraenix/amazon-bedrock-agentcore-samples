#!/usr/bin/env python3
"""
Setup script for AWS AgentCore SDK
Attempts to install and configure the real AgentCore SDK
"""

import subprocess
import sys
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command: str, description: str) -> bool:
    """Run a shell command and return success status"""
    try:
        logger.info(f"🔧 {description}...")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ {description} successful")
            if result.stdout.strip():
                logger.info(f"   Output: {result.stdout.strip()}")
            return True
        else:
            logger.error(f"❌ {description} failed")
            if result.stderr.strip():
                logger.error(f"   Error: {result.stderr.strip()}")
            return False
            
    except Exception as e:
        logger.error(f"❌ {description} failed with exception: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    logger.info(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("❌ Python 3.8+ is required for AgentCore SDK")
        return False
    
    logger.info("✅ Python version is compatible")
    return True

def install_agentcore_sdk():
    """Attempt to install the AgentCore SDK"""
    logger.info("📦 Installing AWS AgentCore SDK...")
    
    # Try different installation methods
    installation_methods = [
        ("pip install bedrock-agentcore", "Installing from PyPI"),
        ("pip install bedrock-agentcore --upgrade", "Upgrading from PyPI"),
        ("pip install git+https://github.com/awslabs/amazon-bedrock-agentcore-samples.git", "Installing from GitHub"),
    ]
    
    for command, description in installation_methods:
        if run_command(command, description):
            return True
        logger.warning(f"⚠️  {description} failed, trying next method...")
    
    logger.error("❌ All installation methods failed")
    return False

def clone_agentcore_samples():
    """Clone the AgentCore samples repository"""
    repo_url = "https://github.com/awslabs/amazon-bedrock-agentcore-samples.git"
    clone_dir = "amazon-bedrock-agentcore-samples"
    
    if os.path.exists(clone_dir):
        logger.info(f"📁 Repository {clone_dir} already exists")
        return True
    
    return run_command(
        f"git clone {repo_url}",
        f"Cloning AgentCore samples repository"
    )

def install_dependencies():
    """Install required dependencies"""
    dependencies = [
        ("pip install playwright", "Installing Playwright"),
        ("pip install rich", "Installing Rich console library"),
        ("playwright install chromium", "Installing Playwright browsers"),
    ]
    
    success_count = 0
    for command, description in dependencies:
        if run_command(command, description):
            success_count += 1
        else:
            logger.warning(f"⚠️  {description} failed but continuing...")
    
    logger.info(f"📊 Installed {success_count}/{len(dependencies)} dependencies")
    return success_count > 0

def test_agentcore_import():
    """Test if AgentCore SDK can be imported"""
    try:
        logger.info("🧪 Testing AgentCore SDK import...")
        
        # Try to import the main components
        from bedrock_agentcore.tools.browser_client import browser_session
        logger.info("✅ browser_session import successful")
        
        try:
            from browser_viewer import BrowserViewerServer
            logger.info("✅ BrowserViewerServer import successful")
        except ImportError:
            logger.warning("⚠️  BrowserViewerServer not available (may need samples repo)")
        
        logger.info("✅ AgentCore SDK is ready to use!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ AgentCore SDK import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ AgentCore SDK test failed: {e}")
        return False

def check_aws_credentials():
    """Check if AWS credentials are configured"""
    try:
        import boto3
        
        # Try to get AWS credentials
        session = boto3.Session()
        credentials = session.get_credentials()
        
        if credentials:
            logger.info("✅ AWS credentials are configured")
            
            # Get account info
            sts = boto3.client('sts')
            identity = sts.get_caller_identity()
            logger.info(f"   Account ID: {identity.get('Account')}")
            logger.info(f"   User/Role: {identity.get('Arn', 'Unknown')}")
            return True
        else:
            logger.error("❌ AWS credentials not found")
            return False
            
    except Exception as e:
        logger.error(f"❌ AWS credentials check failed: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("🚀 AWS AgentCore SDK Setup")
    logger.info("=" * 50)
    
    setup_steps = [
        ("Python Version Check", check_python_version),
        ("AWS Credentials Check", check_aws_credentials),
        ("Install Dependencies", install_dependencies),
        ("Clone AgentCore Samples", clone_agentcore_samples),
        ("Install AgentCore SDK", install_agentcore_sdk),
        ("Test AgentCore Import", test_agentcore_import),
    ]
    
    results = {}
    
    for step_name, step_function in setup_steps:
        logger.info(f"\n📋 Step: {step_name}")
        try:
            results[step_name] = step_function()
        except Exception as e:
            logger.error(f"❌ {step_name} failed with exception: {e}")
            results[step_name] = False
    
    # Summary
    logger.info("\n📊 Setup Summary")
    logger.info("=" * 50)
    
    success_count = 0
    for step_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {step_name}: {status}")
        if success:
            success_count += 1
    
    logger.info(f"\n🎯 Overall: {success_count}/{len(results)} steps successful")
    
    if results.get("Test AgentCore Import", False):
        logger.info("\n🎉 AgentCore SDK is ready!")
        logger.info("💡 You can now run: python real_agentcore_demo.py")
    else:
        logger.info("\n⚠️  AgentCore SDK setup incomplete")
        logger.info("💡 The demo will fall back to local browser automation")
        
    # Next steps
    logger.info("\n📋 Next Steps:")
    if not results.get("AWS Credentials Check", False):
        logger.info("  1. Configure AWS credentials: aws configure")
    if not results.get("Test AgentCore Import", False):
        logger.info("  2. Check AgentCore service availability in your AWS region")
        logger.info("  3. Contact AWS Support for AgentCore Browser Tool access")
    logger.info("  4. Run the demo: python real_agentcore_demo.py")

if __name__ == "__main__":
    main()