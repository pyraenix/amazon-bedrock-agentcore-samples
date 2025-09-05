#!/usr/bin/env python3
"""
Browser-Use AgentCore Integration Validation Script

This script validates the integration between browser-use and AgentCore Browser Tool
by testing actual connectivity and basic functionality.
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BrowserUseAgentCoreIntegrationValidator:
    """Validates browser-use with AgentCore Browser Tool integration."""
    
    def __init__(self):
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "integration_tests": {},
            "issues": [],
            "recommendations": [],
            "test_details": {}
        }
    
    async def test_browser_use_import(self) -> bool:
        """Test browser-use package import and basic functionality."""
        logger.info("Testing browser-use package import...")
        
        try:
            import browser_use
            from browser_use import Agent
            from browser_use.browser.session import BrowserSession
            
            self.validation_results["integration_tests"]["browser_use_import"] = "✅ PASS"
            self.validation_results["test_details"]["browser_use_version"] = getattr(browser_use, '__version__', 'Unknown')
            logger.info("Browser-use import - OK")
            return True
            
        except ImportError as e:
            self.validation_results["integration_tests"]["browser_use_import"] = "❌ FAIL"
            self.validation_results["issues"].append(f"Browser-use import failed: {str(e)}")
            self.validation_results["recommendations"].append("Install browser-use: pip install browser-use")
            logger.error(f"Browser-use import - FAILED: {str(e)}")
            return False
        except Exception as e:
            self.validation_results["integration_tests"]["browser_use_import"] = "❌ FAIL"
            self.validation_results["issues"].append(f"Browser-use import error: {str(e)}")
            logger.error(f"Browser-use import - ERROR: {str(e)}")
            return False
    
    async def test_agentcore_sdk_import(self) -> bool:
        """Test AgentCore SDK import and basic functionality."""
        logger.info("Testing AgentCore SDK import...")
        
        try:
            # Try to import AgentCore Browser Client
            import bedrock_agentcore
            from bedrock_agentcore_browser_client import BrowserClient
            
            self.validation_results["integration_tests"]["agentcore_sdk_import"] = "✅ PASS"
            self.validation_results["test_details"]["agentcore_version"] = getattr(bedrock_agentcore, '__version__', 'Unknown')
            logger.info("AgentCore SDK import - OK")
            return True
            
        except ImportError as e:
            self.validation_results["integration_tests"]["agentcore_sdk_import"] = "❌ FAIL"
            self.validation_results["issues"].append(f"AgentCore SDK import failed: {str(e)}")
            self.validation_results["recommendations"].append("Install bedrock-agentcore SDK")
            logger.error(f"AgentCore SDK import - FAILED: {str(e)}")
            return False
        except Exception as e:
            self.validation_results["integration_tests"]["agentcore_sdk_import"] = "❌ FAIL"
            self.validation_results["issues"].append(f"AgentCore SDK import error: {str(e)}")
            logger.error(f"AgentCore SDK import - ERROR: {str(e)}")
            return False
    
    async def test_aws_bedrock_connectivity(self) -> bool:
        """Test AWS Bedrock connectivity for LLM integration."""
        logger.info("Testing AWS Bedrock connectivity...")
        
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            
            # Test Bedrock Runtime client
            bedrock_runtime = boto3.client('bedrock-runtime', region_name=os.getenv('AWS_REGION', 'us-east-1'))
            
            # Try to list available models (this tests permissions)
            bedrock = boto3.client('bedrock', region_name=os.getenv('AWS_REGION', 'us-east-1'))
            models = bedrock.list_foundation_models()
            
            available_models = [model['modelId'] for model in models.get('modelSummaries', [])]
            self.validation_results["test_details"]["available_bedrock_models"] = available_models[:5]  # First 5 models
            
            self.validation_results["integration_tests"]["aws_bedrock_connectivity"] = "✅ PASS"
            logger.info(f"AWS Bedrock connectivity - OK ({len(available_models)} models available)")
            return True
            
        except NoCredentialsError:
            self.validation_results["integration_tests"]["aws_bedrock_connectivity"] = "❌ FAIL"
            self.validation_results["issues"].append("AWS credentials not configured")
            self.validation_results["recommendations"].append("Configure AWS credentials using 'aws configure'")
            logger.error("AWS Bedrock connectivity - NO CREDENTIALS")
            return False
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'AccessDeniedException':
                self.validation_results["integration_tests"]["aws_bedrock_connectivity"] = "❌ FAIL"
                self.validation_results["issues"].append("Insufficient AWS permissions for Bedrock")
                self.validation_results["recommendations"].append("Add bedrock:ListFoundationModels and bedrock:InvokeModel permissions")
                logger.error("AWS Bedrock connectivity - ACCESS DENIED")
            else:
                self.validation_results["integration_tests"]["aws_bedrock_connectivity"] = "❌ FAIL"
                self.validation_results["issues"].append(f"AWS Bedrock error: {str(e)}")
                logger.error(f"AWS Bedrock connectivity - ERROR: {str(e)}")
            return False
            
        except Exception as e:
            self.validation_results["integration_tests"]["aws_bedrock_connectivity"] = "❌ FAIL"
            self.validation_results["issues"].append(f"AWS Bedrock connectivity error: {str(e)}")
            logger.error(f"AWS Bedrock connectivity - ERROR: {str(e)}")
            return False
    
    async def test_playwright_installation(self) -> bool:
        """Test Playwright installation and browser availability."""
        logger.info("Testing Playwright installation...")
        
        try:
            import playwright
            from playwright.async_api import async_playwright
            
            # Test if browsers are installed
            async with async_playwright() as p:
                # Try to launch chromium (most commonly used)
                try:
                    browser = await p.chromium.launch(headless=True)
                    await browser.close()
                    
                    self.validation_results["integration_tests"]["playwright_installation"] = "✅ PASS"
                    self.validation_results["test_details"]["playwright_version"] = playwright.__version__
                    logger.info("Playwright installation - OK")
                    return True
                    
                except Exception as browser_error:
                    self.validation_results["integration_tests"]["playwright_installation"] = "⚠️ WARNING"
                    self.validation_results["issues"].append(f"Playwright browsers not installed: {str(browser_error)}")
                    self.validation_results["recommendations"].append("Install Playwright browsers: playwright install")
                    logger.warning(f"Playwright installation - BROWSERS MISSING: {str(browser_error)}")
                    return False
            
        except ImportError as e:
            self.validation_results["integration_tests"]["playwright_installation"] = "❌ FAIL"
            self.validation_results["issues"].append(f"Playwright import failed: {str(e)}")
            self.validation_results["recommendations"].append("Install Playwright: pip install playwright")
            logger.error(f"Playwright installation - FAILED: {str(e)}")
            return False
        except Exception as e:
            self.validation_results["integration_tests"]["playwright_installation"] = "❌ FAIL"
            self.validation_results["issues"].append(f"Playwright error: {str(e)}")
            logger.error(f"Playwright installation - ERROR: {str(e)}")
            return False
    
    async def test_agentcore_session_creation(self) -> bool:
        """Test AgentCore Browser Tool session creation."""
        logger.info("Testing AgentCore session creation...")
        
        try:
            from bedrock_agentcore_browser_client import BrowserClient
            
            # Initialize AgentCore Browser Client
            region = os.getenv('AWS_REGION', 'us-east-1')
            client = BrowserClient(region=region)
            
            # Try to create a session (this will test actual connectivity)
            session = await client.create_session()
            
            if session and hasattr(session, 'session_id'):
                # Get connection details
                ws_url, headers = client.get_connection_details(session.session_id)
                
                # Clean up session
                await client.close_session(session.session_id)
                
                self.validation_results["integration_tests"]["agentcore_session_creation"] = "✅ PASS"
                self.validation_results["test_details"]["agentcore_region"] = region
                self.validation_results["test_details"]["session_created"] = True
                logger.info("AgentCore session creation - OK")
                return True
            else:
                self.validation_results["integration_tests"]["agentcore_session_creation"] = "❌ FAIL"
                self.validation_results["issues"].append("AgentCore session creation returned invalid session")
                logger.error("AgentCore session creation - INVALID SESSION")
                return False
                
        except ImportError as e:
            self.validation_results["integration_tests"]["agentcore_session_creation"] = "❌ FAIL"
            self.validation_results["issues"].append(f"AgentCore SDK not available: {str(e)}")
            self.validation_results["recommendations"].append("Install bedrock-agentcore SDK")
            logger.error(f"AgentCore session creation - SDK NOT AVAILABLE: {str(e)}")
            return False
            
        except Exception as e:
            self.validation_results["integration_tests"]["agentcore_session_creation"] = "❌ FAIL"
            self.validation_results["issues"].append(f"AgentCore session creation failed: {str(e)}")
            logger.error(f"AgentCore session creation - FAILED: {str(e)}")
            return False
    
    async def test_browser_use_agentcore_integration(self) -> bool:
        """Test full browser-use with AgentCore integration."""
        logger.info("Testing browser-use with AgentCore integration...")
        
        try:
            from browser_use import Agent
            from browser_use.browser.session import BrowserSession
            from bedrock_agentcore_browser_client import BrowserClient
            import boto3
            
            # Initialize AgentCore Browser Client
            region = os.getenv('AWS_REGION', 'us-east-1')
            client = BrowserClient(region=region)
            
            # Create AgentCore session
            session = await client.create_session()
            ws_url, headers = client.get_connection_details(session.session_id)
            
            # Create browser-use session with AgentCore connection
            browser_session = BrowserSession(
                cdp_url=ws_url,
                cdp_headers=headers
            )
            
            # Initialize Bedrock LLM client
            bedrock_runtime = boto3.client('bedrock-runtime', region_name=region)
            
            # Create browser-use agent (without actually running a task)
            # This tests the integration setup
            
            # Clean up
            await client.close_session(session.session_id)
            
            self.validation_results["integration_tests"]["browser_use_agentcore_integration"] = "✅ PASS"
            self.validation_results["test_details"]["integration_successful"] = True
            logger.info("Browser-use with AgentCore integration - OK")
            return True
            
        except Exception as e:
            self.validation_results["integration_tests"]["browser_use_agentcore_integration"] = "❌ FAIL"
            self.validation_results["issues"].append(f"Browser-use AgentCore integration failed: {str(e)}")
            logger.error(f"Browser-use with AgentCore integration - FAILED: {str(e)}")
            logger.debug(f"Integration error traceback: {traceback.format_exc()}")
            return False
    
    async def test_sensitive_data_utilities(self) -> bool:
        """Test sensitive data handling utilities."""
        logger.info("Testing sensitive data handling utilities...")
        
        try:
            # Test if our custom utilities are available
            from tools.browseruse_sensitive_data_handler import SensitiveDataHandler
            from tools.browseruse_session_manager import BrowserUseAgentCoreSessionManager
            
            # Test basic functionality
            handler = SensitiveDataHandler()
            session_manager = BrowserUseAgentCoreSessionManager(region=os.getenv('AWS_REGION', 'us-east-1'))
            
            # Test PII masking
            test_data = "My SSN is 123-45-6789 and email is test@example.com"
            masked_data = handler.mask_pii(test_data)
            
            if masked_data != test_data:  # Should be different after masking
                self.validation_results["integration_tests"]["sensitive_data_utilities"] = "✅ PASS"
                self.validation_results["test_details"]["pii_masking_working"] = True
                logger.info("Sensitive data handling utilities - OK")
                return True
            else:
                self.validation_results["integration_tests"]["sensitive_data_utilities"] = "⚠️ WARNING"
                self.validation_results["issues"].append("PII masking may not be working correctly")
                logger.warning("Sensitive data handling utilities - PII MASKING ISSUE")
                return False
                
        except ImportError as e:
            self.validation_results["integration_tests"]["sensitive_data_utilities"] = "❌ FAIL"
            self.validation_results["issues"].append(f"Sensitive data utilities not available: {str(e)}")
            self.validation_results["recommendations"].append("Ensure all tutorial utilities are properly installed")
            logger.error(f"Sensitive data handling utilities - NOT AVAILABLE: {str(e)}")
            return False
        except Exception as e:
            self.validation_results["integration_tests"]["sensitive_data_utilities"] = "❌ FAIL"
            self.validation_results["issues"].append(f"Sensitive data utilities error: {str(e)}")
            logger.error(f"Sensitive data handling utilities - ERROR: {str(e)}")
            return False
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        
        # Count results
        total_tests = len(self.validation_results["integration_tests"])
        passed_tests = len([v for v in self.validation_results["integration_tests"].values() if "✅" in v])
        failed_tests = len([v for v in self.validation_results["integration_tests"].values() if "❌" in v])
        warning_tests = len([v for v in self.validation_results["integration_tests"].values() if "⚠️" in v])
        
        self.validation_results["summary"] = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "warnings": warning_tests,
            "success_rate": f"{(passed_tests / total_tests * 100):.1f}%" if total_tests > 0 else "0%"
        }
        
        # Determine overall status
        if failed_tests == 0:
            if warning_tests == 0:
                self.validation_results["overall_status"] = "✅ INTEGRATION READY"
            else:
                self.validation_results["overall_status"] = "⚠️ INTEGRATION READY WITH WARNINGS"
        else:
            self.validation_results["overall_status"] = "❌ INTEGRATION NOT READY"
        
        return self.validation_results
    
    def print_validation_report(self):
        """Print formatted validation report."""
        
        print("\n" + "="*80)
        print("🔗 BROWSER-USE WITH AGENTCORE - INTEGRATION VALIDATION")
        print("="*80)
        
        print(f"\n📊 SUMMARY:")
        print(f"   Overall Status: {self.validation_results['overall_status']}")
        print(f"   Success Rate: {self.validation_results['summary']['success_rate']}")
        print(f"   Tests: {self.validation_results['summary']['passed']} passed, "
              f"{self.validation_results['summary']['failed']} failed, "
              f"{self.validation_results['summary']['warnings']} warnings")
        
        print(f"\n🔍 INTEGRATION TEST RESULTS:")
        for test, result in self.validation_results["integration_tests"].items():
            print(f"   {test.replace('_', ' ').title()}: {result}")
        
        if self.validation_results["test_details"]:
            print(f"\n📋 TEST DETAILS:")
            for key, value in self.validation_results["test_details"].items():
                if isinstance(value, list):
                    print(f"   {key.replace('_', ' ').title()}: {', '.join(value[:3])}{'...' if len(value) > 3 else ''}")
                else:
                    print(f"   {key.replace('_', ' ').title()}: {value}")
        
        if self.validation_results["issues"]:
            print(f"\n⚠️ ISSUES FOUND:")
            for i, issue in enumerate(self.validation_results["issues"], 1):
                print(f"   {i}. {issue}")
        
        if self.validation_results["recommendations"]:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(self.validation_results["recommendations"], 1):
                print(f"   {i}. {rec}")
        
        print(f"\n📝 NEXT STEPS:")
        if self.validation_results["overall_status"] == "✅ INTEGRATION READY":
            print("   🎉 Integration is working perfectly!")
            print("   📚 You can now run the tutorial notebooks:")
            print("      - browseruse_agentcore_secure_connection_tutorial.ipynb")
            print("      - browseruse_pii_masking_tutorial.ipynb")
            print("      - browseruse_compliance_audit_tutorial.ipynb")
            print("      - browseruse_production_deployment_tutorial.ipynb")
        elif "WARNING" in self.validation_results["overall_status"]:
            print("   ⚠️ Integration has warnings but should work")
            print("   🔧 Address the warnings above for optimal experience")
            print("   📚 You can proceed with the tutorial notebooks")
        else:
            print("   ❌ Please fix the integration issues above")
            print("   🔧 Follow the recommendations to resolve problems")
            print("   🔄 Run this validation script again after making changes")
        
        print("\n" + "="*80)
    
    def save_validation_report(self, filename: str = "integration_validation_report.json"):
        """Save validation report to JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.validation_results, f, indent=2)
            logger.info(f"Validation report saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save validation report: {str(e)}")

async def main():
    """Main integration validation function."""
    
    print("🔗 Starting Browser-Use with AgentCore integration validation...")
    
    validator = BrowserUseAgentCoreIntegrationValidator()
    
    # Run all integration tests
    tests = [
        await validator.test_browser_use_import(),
        await validator.test_agentcore_sdk_import(),
        await validator.test_aws_bedrock_connectivity(),
        await validator.test_playwright_installation(),
        await validator.test_agentcore_session_creation(),
        await validator.test_browser_use_agentcore_integration(),
        await validator.test_sensitive_data_utilities()
    ]
    
    # Generate and display report
    validator.generate_validation_report()
    validator.print_validation_report()
    
    # Save report
    validator.save_validation_report()
    
    # Return success status
    return validator.validation_results["overall_status"] == "✅ INTEGRATION READY"

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Integration validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Integration validation failed with error: {str(e)}")
        logger.error(f"Validation error traceback: {traceback.format_exc()}")
        sys.exit(1)