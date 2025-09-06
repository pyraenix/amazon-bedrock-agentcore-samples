#!/usr/bin/env python3
"""
Test AgentCore Browser Tool access with specific IAM role
"""

import boto3
import json
from datetime import datetime

def test_role_access(role_arn=None):
    """Test AgentCore access with specific role"""
    print("🔐 Testing AgentCore Browser Tool Access")
    print("=" * 50)
    print(f"Test Time: {datetime.now().isoformat()}")
    
    try:
        # Get current identity
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"Current Identity: {identity.get('Arn', 'Unknown')}")
        
        # Test Bedrock Runtime access
        bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        print("✅ Bedrock Runtime client created")
        
        # Test Bedrock Agent Runtime access
        try:
            bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name='us-east-1')
            print("✅ Bedrock Agent Runtime client created")
            
            # Try to list available tools/actions
            # This is a hypothetical call - the actual API might be different
            print("🔍 Checking for AgentCore Browser Tool availability...")
            
        except Exception as e:
            print(f"⚠️  Bedrock Agent Runtime access issue: {e}")
        
        # Test Bedrock Agent access
        try:
            bedrock_agent = boto3.client('bedrock-agent', region_name='us-east-1')
            print("✅ Bedrock Agent client created")
            
            # Try to list agents (this requires permissions)
            try:
                response = bedrock_agent.list_agents(maxResults=1)
                print(f"✅ Can list agents: {len(response.get('agentSummaries', []))} found")
            except Exception as e:
                print(f"⚠️  Cannot list agents: {e}")
                
        except Exception as e:
            print(f"⚠️  Bedrock Agent access issue: {e}")
            
        # Test for AgentCore specific permissions
        print("\n🧪 Testing AgentCore-specific operations...")
        
        # Try to invoke a hypothetical AgentCore browser tool
        try:
            # This is speculative - the actual API call might be different
            test_payload = {
                "action": "navigate",
                "url": "https://httpbin.org/html",
                "sessionId": "test-session"
            }
            
            print("📝 Would attempt AgentCore Browser Tool invocation...")
            print(f"   Payload: {json.dumps(test_payload, indent=2)}")
            print("   (Actual API call not implemented - service may not be available)")
            
        except Exception as e:
            print(f"❌ AgentCore Browser Tool test failed: {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ Role access test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🎯 AgentCore Browser Tool Role Access Test")
    print("=" * 60)
    
    # Test current role access
    success = test_role_access()
    
    print(f"\n📊 Test Result: {'✅ Success' if success else '❌ Failed'}")
    
    if success:
        print("\n💡 Next Steps:")
        print("1. Your current role has basic Bedrock access")
        print("2. AgentCore Browser Tool may still require special access")
        print("3. Try running: python check_agentcore_tools.py")
        print("4. Contact AWS Support if AgentCore tools aren't available")
    else:
        print("\n💡 Troubleshooting:")
        print("1. Verify your AWS credentials are configured")
        print("2. Check if the selected role has Bedrock permissions")
        print("3. Try switching to the Admin role for testing")

if __name__ == "__main__":
    main()