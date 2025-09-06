#!/usr/bin/env python3
"""
Configure the integration to use real AgentCore Browser Tool service
"""

import json
import yaml
from pathlib import Path
from config import ConfigurationManager, IntegrationConfig

def create_real_agentcore_config(browser_id: str = None, runtime_id: str = None):
    """Create configuration for real AgentCore service"""
    
    print("🔧 Creating Real AgentCore Configuration...")
    
    # Load current configuration
    config_manager = ConfigurationManager()
    current_config = config_manager.load_config()
    
    # Update for real AgentCore service
    config = config_manager.get_integration_config()
    
    # Disable test mode
    config.agentcore_endpoints.test_mode = False
    
    # Set real AgentCore endpoints
    config.agentcore_endpoints.base_url = None  # Use AWS service endpoints
    config.agentcore_endpoints.browser_tool_endpoint = None  # Will use AWS API
    config.agentcore_endpoints.runtime_endpoint = None  # Will use AWS API
    
    # Add browser identifier if provided
    if browser_id:
        # Store browser ID for API calls
        config.browser_config.browser_args.append(f"--browser-id={browser_id}")
    
    # Save the updated configuration
    config_dict = config.to_dict()
    
    # Save as YAML file
    config_file = Path("agentcore_real_config.yaml")
    with open(config_file, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    print(f"  ✅ Configuration saved to: {config_file}")
    
    # Also save as JSON for reference
    json_file = Path("agentcore_real_config.json")
    with open(json_file, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"  ✅ Configuration also saved as: {json_file}")
    
    return config_dict

def create_environment_setup(browser_id: str = None):
    """Create environment variable setup script"""
    
    print("📝 Creating Environment Setup Script...")
    
    env_script = """#!/bin/bash
# Environment variables for real AgentCore Browser Tool

# Disable test mode
export AGENTCORE_TEST_MODE=false

# AWS Configuration (if not using AWS CLI profiles)
# export AWS_ACCESS_KEY_ID=your_access_key
# export AWS_SECRET_ACCESS_KEY=your_secret_key
# export AWS_DEFAULT_REGION=us-east-1

# AgentCore Configuration
export AGENTCORE_BASE_URL=""  # Empty to use AWS service endpoints
export AGENTCORE_BROWSER_TOOL_ENDPOINT=""  # Empty to use AWS API

# Browser Configuration
export BROWSER_HEADLESS=true
export BROWSER_VIEWPORT_WIDTH=1920
export BROWSER_VIEWPORT_HEIGHT=1080
export BROWSER_TIMEOUT=30
"""
    
    if browser_id:
        env_script += f"""
# Browser ID from AWS Console
export AGENTCORE_BROWSER_ID={browser_id}
"""
    
    env_script += """
# LlamaIndex Models
export LLAMAINDEX_LLM_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
export LLAMAINDEX_VISION_MODEL=anthropic.claude-3-sonnet-20240229-v1:0

echo "✅ Environment configured for real AgentCore Browser Tool"
echo "💡 Run: source setup_real_agentcore.sh"
"""
    
    script_file = Path("setup_real_agentcore.sh")
    with open(script_file, 'w') as f:
        f.write(env_script)
    
    # Make executable
    script_file.chmod(0o755)
    
    print(f"  ✅ Environment setup script: {script_file}")
    print(f"  💡 Run: source {script_file}")
    
    return script_file

def create_test_script(browser_id: str = None):
    """Create test script for real AgentCore service"""
    
    print("🧪 Creating Real AgentCore Test Script...")
    
    test_script = f'''#!/usr/bin/env python3
"""
Test real AgentCore Browser Tool integration
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from integration import LlamaIndexAgentCoreBrowserIntegration
from config import ConfigurationManager

def test_real_integration():
    """Test the real AgentCore integration"""
    print("🧪 Testing Real AgentCore Browser Tool Integration")
    print("=" * 60)
    
    try:
        # Load real configuration
        config_manager = ConfigurationManager("agentcore_real_config.yaml")
        config = config_manager.load_config()
        
        print("✅ Configuration loaded successfully")
        print(f"  Test Mode: {{config.get('agentcore_endpoints', {{}}).get('test_mode', 'Unknown')}}")
        
        # Initialize integration
        integration = LlamaIndexAgentCoreBrowserIntegration(config_manager)
        
        print("✅ Integration initialized")
        
        # Test browser session creation
        print("🌐 Testing browser session...")
        session_id = integration.create_browser_session()
        print(f"  ✅ Session created: {{session_id}}")
        
        # Test navigation
        print("🔗 Testing navigation...")
        result = integration.navigate_to_url(session_id, "https://httpbin.org/html")
        print(f"  ✅ Navigation result: {{result.get('status', 'Unknown')}}")
        
        # Test content extraction
        print("📄 Testing content extraction...")
        content = integration.extract_page_content(session_id)
        print(f"  ✅ Content extracted: {{len(content.get('text', ''))}} characters")
        
        # Cleanup
        print("🧹 Cleaning up...")
        integration.close_browser_session(session_id)
        print("  ✅ Session closed")
        
        print("\\n🎉 Real AgentCore integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\\n❌ Test failed: {{e}}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_real_integration()
    sys.exit(0 if success else 1)
'''
    
    test_file = Path("test_real_agentcore_integration.py")
    with open(test_file, 'w') as f:
        f.write(test_script)
    
    # Make executable
    test_file.chmod(0o755)
    
    print(f"  ✅ Test script created: {test_file}")
    
    return test_file

def main():
    """Main configuration setup"""
    print("🚀 AgentCore Real Service Configuration Setup")
    print("=" * 60)
    
    print("📋 Instructions:")
    print("1. Go to AWS Console → AgentCore → Built-in Tools")
    print("2. Create a Browser Tool instance")
    print("3. Note the Browser ID that gets created")
    print("4. Create an Agent Runtime that uses the Browser Tool")
    print("5. Run this script with the Browser ID")
    print()
    
    # Get browser ID from user
    browser_id = input("Enter Browser ID from AWS Console (or press Enter to skip): ").strip()
    if not browser_id:
        browser_id = None
        print("⚠️  No Browser ID provided - you'll need to configure this later")
    else:
        print(f"✅ Using Browser ID: {browser_id}")
    
    print()
    
    # Create configuration files
    config_dict = create_real_agentcore_config(browser_id)
    env_script = create_environment_setup(browser_id)
    test_script = create_test_script(browser_id)
    
    print()
    print("📋 Next Steps:")
    print("=" * 30)
    print("1. Complete AWS Console setup (create Browser Tool and Agent Runtime)")
    print(f"2. Update Browser ID in configuration files if needed")
    print(f"3. Run: source {env_script}")
    print(f"4. Test: python {test_script}")
    print("5. Update your main integration to use the real configuration")
    
    print()
    print("🎉 Configuration setup complete!")
    print("💡 Your integration is ready to use real AgentCore Browser Tool")

if __name__ == "__main__":
    main()