#!/usr/bin/env python3
"""
Integration Test Suite: Bedrock AgentCore Browser Tool with Strands

This test suite validates the integration between AWS Bedrock AgentCore
browser capabilities and Strands agents.

Author: AWS Labs Community
License: MIT-0
"""

import os
import time
import boto3
from bedrock_strands_browser_tool import browse_web, get_tool_info
from strands import Agent


def test_aws_connectivity():
    """Test AWS Bedrock connectivity"""
    print("🔐 Testing AWS Connectivity")
    print("-" * 30)
    
    try:
        # Test AWS credentials
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✅ AWS Identity: {identity.get('Arn', 'Unknown')}")
        
        # Test Bedrock access
        bedrock = boto3.client('bedrock', region_name=os.getenv('AWS_REGION', 'us-east-1'))
        models = bedrock.list_foundation_models()
        claude_models = [m for m in models['modelSummaries'] if 'claude' in m['modelId'].lower()]
        print(f"✅ Bedrock Access: {len(claude_models)} Claude models available")
        
        return True
        
    except Exception as e:
        print(f"❌ AWS Connectivity Error: {e}")
        return False


def test_browser_tool_functionality():
    """Test browser tool basic functionality"""
    print("\n🌐 Testing Browser Tool Functionality")
    print("-" * 40)
    
    test_results = []
    
    # Test 1: Basic text extraction
    print("📄 Test 1: Text extraction")
    try:
        result = browse_web("https://example.com", "text")
        success = not result.startswith("Error") and len(result) > 0
        test_results.append(("Text extraction", success))
        
        if success:
            print(f"✅ Success: {len(result)} characters extracted")
        else:
            print(f"❌ Failed: {result}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        test_results.append(("Text extraction", False))
    
    # Test 2: HTML extraction
    print("\n🌐 Test 2: HTML extraction")
    try:
        result = browse_web("https://example.com", "html")
        success = not result.startswith("Error") and "<html" in result.lower()
        test_results.append(("HTML extraction", success))
        
        if success:
            print(f"✅ Success: HTML content extracted ({len(result)} chars)")
        else:
            print(f"❌ Failed: {result[:100]}...")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        test_results.append(("HTML extraction", False))
    
    # Test 3: Metadata extraction
    print("\n📊 Test 3: Metadata extraction")
    try:
        result = browse_web("https://example.com", "metadata")
        success = not result.startswith("Error") and "title" in result.lower()
        test_results.append(("Metadata extraction", success))
        
        if success:
            print(f"✅ Success: {result}")
        else:
            print(f"❌ Failed: {result}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        test_results.append(("Metadata extraction", False))
    
    # Test 4: Error handling
    print("\n🚨 Test 4: Error handling")
    try:
        result = browse_web("invalid-url", "text")
        success = result.startswith("Error")
        test_results.append(("Error handling", success))
        
        if success:
            print(f"✅ Success: Error properly handled")
        else:
            print(f"❌ Failed: Expected error but got: {result[:100]}...")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        test_results.append(("Error handling", False))
    
    return test_results


def test_strands_integration():
    """Test Strands agent integration"""
    print("\n🤖 Testing Strands Agent Integration")
    print("-" * 40)
    
    test_results = []
    
    try:
        # Create agent with AWS Bedrock model using inference profile you have access to
        agent = Agent(
            tools=[browse_web],
            model="us.anthropic.claude-3-sonnet-20240229-v1:0"
        )
        print("✅ Agent created successfully")
        
        # Test agent browsing capability
        print("\n📡 Testing agent browsing...")
        start_time = time.time()
        
        result = agent("Browse https://example.com and tell me what type of website this is in one sentence.")
        
        elapsed = time.time() - start_time
        
        if result and str(result).strip() and len(str(result)) > 10:
            print(f"✅ Success: Agent analysis completed in {elapsed:.1f}s")
            print(f"Response: {result}")
            test_results.append(("Agent integration", True))
        else:
            print(f"❌ Failed: Empty or invalid response: {result}")
            test_results.append(("Agent integration", False))
            
    except Exception as e:
        print(f"❌ Agent Integration Error: {e}")
        test_results.append(("Agent integration", False))
    
    return test_results


def test_performance():
    """Test performance characteristics"""
    print("\n⚡ Testing Performance")
    print("-" * 25)
    
    test_urls = [
        "https://example.com",
        "https://httpbin.org/html",
    ]
    
    performance_results = []
    
    for url in test_urls:
        print(f"\n🎯 Testing: {url}")
        try:
            start_time = time.time()
            result = browse_web(url, "text")
            elapsed = time.time() - start_time
            
            success = not result.startswith("Error")
            performance_results.append({
                "url": url,
                "success": success,
                "time": elapsed,
                "size": len(result) if success else 0
            })
            
            if success:
                print(f"✅ Success: {elapsed:.2f}s, {len(result)} chars")
            else:
                print(f"❌ Failed: {result[:50]}...")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            performance_results.append({
                "url": url,
                "success": False,
                "time": 0,
                "size": 0
            })
    
    return performance_results


def test_tool_info():
    """Test tool information function"""
    print("\n📋 Testing Tool Information")
    print("-" * 30)
    
    try:
        info = get_tool_info()
        required_keys = ["name", "description", "supported_modes"]
        
        for key in required_keys:
            if key in info:
                print(f"✅ {key}: {info[key]}")
            else:
                print(f"❌ Missing key: {key}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Tool Info Error: {e}")
        return False


def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("🧪 Bedrock AgentCore Browser Tool - Integration Test Suite")
    print("=" * 65)
    print("Testing integration between AWS Bedrock AgentCore and Strands agents")
    print("=" * 65)
    
    # Track all test results
    all_results = []
    
    # Test 1: AWS Connectivity
    aws_ok = test_aws_connectivity()
    all_results.append(("AWS Connectivity", aws_ok))
    
    # Test 2: Browser Tool Functionality
    browser_results = test_browser_tool_functionality()
    all_results.extend(browser_results)
    
    # Test 3: Strands Integration (only if AWS is working)
    if aws_ok:
        strands_results = test_strands_integration()
        all_results.extend(strands_results)
    else:
        print("\n⚠️  Skipping Strands integration test due to AWS connectivity issues")
        all_results.append(("Strands Integration", False))
    
    # Test 4: Performance
    performance_results = test_performance()
    perf_success = all(r["success"] for r in performance_results)
    all_results.append(("Performance Test", perf_success))
    
    # Test 5: Tool Info
    tool_info_ok = test_tool_info()
    all_results.append(("Tool Information", tool_info_ok))
    
    # Summary
    print("\n\n📊 Test Results Summary")
    print("=" * 40)
    
    passed = 0
    total = len(all_results)
    
    for test_name, success in all_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if success:
            passed += 1
    
    print("-" * 40)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Integration is fully functional")
        print("✅ Ready for production use")
        return True
    else:
        print(f"\n⚠️  {total-passed} tests failed")
        print("❌ Please check configuration and try again")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)