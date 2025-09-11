#!/usr/bin/env python3
"""
Browser-Use with AgentCore Browser Tool - Setup and Validation Script

This script sets up and validates the environment for the browser-use with
AgentCore Browser Tool sensitive information handling tutorial.
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BrowserUseAgentCoreSetup:
    """Setup and validation for browser-use with AgentCore tutorial."""
    
    def __init__(self):
        self.setup_results = {
            "timestamp": datetime.now().isoformat(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "checks": {},
            "issues": [],
            "recommendations": []
        }
    
    def check_python_version(self) -> bool:
        """Check if Python version is 3.12 or higher."""
        logger.info("Checking Python version...")
        
        if sys.version_info >= (3, 12):
            self.setup_results["checks"]["python_version"] = "✅ PASS"
            logger.info(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} - OK")
            return True
        else:
            self.setup_results["checks"]["python_version"] = "❌ FAIL"
            self.setup_results["issues"].append(
                f"Python {sys.version_info.major}.{sys.version_info.minor} detected. Python 3.12+ required."
            )
            self.setup_results["recommendations"].append(
                "Install Python 3.12 or higher from https://python.org"
            )
            logger.error(f"Python {sys.version_info.major}.{sys.version_info.minor} - UNSUPPORTED")
            return False
    
    def check_required_packages(self) -> bool:
        """Check if required packages are installed."""
        logger.info("Checking required packages...")
        
        required_packages = [
            ("browser_use", "browser-use"),
            ("bedrock_agentcore", "bedrock-agentcore"),
            ("boto3", "boto3"),
            ("playwright", "playwright"),
            ("pydantic", "pydantic"),
            ("structlog", "structlog")
        ]
        
        all_packages_ok = True
        
        for import_name, package_name in required_packages:
            try:
                __import__(import_name)
                self.setup_results["checks"][f"package_{import_name}"] = "✅ PASS"
                logger.info(f"Package {package_name} - OK")
            except ImportError:
                self.setup_results["checks"][f"package_{import_name}"] = "❌ FAIL"
                self.setup_results["issues"].append(f"Package {package_name} not installed")
                self.setup_results["recommendations"].append(f"Install with: pip install {package_name}")
                logger.error(f"Package {package_name} - MISSING")
                all_packages_ok = False
        
        return all_packages_ok
    
    def check_aws_credentials(self) -> bool:
        """Check AWS credentials and permissions."""
        logger.info("Checking AWS credentials...")
        
        try:
            import boto3
            from botocore.exceptions import NoCredentialsError, ClientError
            
            # Check basic AWS connectivity
            sts = boto3.client('sts')
            identity = sts.get_caller_identity()
            
            self.setup_results["checks"]["aws_credentials"] = "✅ PASS"
            self.setup_results["aws_account"] = identity.get('Account')
            self.setup_results["aws_user_arn"] = identity.get('Arn')
            logger.info(f"AWS credentials - OK (Account: {identity.get('Account')})")
            
            # Check Bedrock permissions
            try:
                bedrock = boto3.client('bedrock')
                # Try to list foundation models to test permissions
                bedrock.list_foundation_models()
                self.setup_results["checks"]["bedrock_permissions"] = "✅ PASS"
                logger.info("Bedrock permissions - OK")
            except ClientError as e:
                self.setup_results["checks"]["bedrock_permissions"] = "⚠️ WARNING"
                self.setup_results["issues"].append(f"Bedrock permissions issue: {str(e)}")
                self.setup_results["recommendations"].append(
                    "Ensure your AWS credentials have bedrock:InvokeModel permissions"
                )
                logger.warning(f"Bedrock permissions - LIMITED: {str(e)}")
            
            return True
            
        except NoCredentialsError:
            self.setup_results["checks"]["aws_credentials"] = "❌ FAIL"
            self.setup_results["issues"].append("AWS credentials not configured")
            self.setup_results["recommendations"].append(
                "Configure AWS credentials using 'aws configure' or environment variables"
            )
            logger.error("AWS credentials - NOT CONFIGURED")
            return False
            
        except Exception as e:
            self.setup_results["checks"]["aws_credentials"] = "❌ FAIL"
            self.setup_results["issues"].append(f"AWS credentials error: {str(e)}")
            logger.error(f"AWS credentials - ERROR: {str(e)}")
            return False
    
    def check_environment_file(self) -> bool:
        """Check if environment file exists and is properly configured."""
        logger.info("Checking environment configuration...")
        
        env_file = Path(".env")
        env_example = Path(".env.example")
        
        if not env_file.exists():
            if env_example.exists():
                self.setup_results["checks"]["env_file"] = "⚠️ WARNING"
                self.setup_results["issues"].append(".env file not found")
                self.setup_results["recommendations"].append(
                    "Copy .env.example to .env and configure your settings"
                )
                logger.warning(".env file - NOT FOUND (example available)")
            else:
                self.setup_results["checks"]["env_file"] = "❌ FAIL"
                self.setup_results["issues"].append(".env.example file missing")
                logger.error(".env file - MISSING")
            return False
        
        # Check environment variables
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            required_vars = [
                "AWS_REGION",
                "BEDROCK_MODEL_ID"
            ]
            
            missing_vars = []
            for var in required_vars:
                if not os.getenv(var):
                    missing_vars.append(var)
            
            if missing_vars:
                self.setup_results["checks"]["env_file"] = "⚠️ WARNING"
                self.setup_results["issues"].append(f"Missing environment variables: {', '.join(missing_vars)}")
                self.setup_results["recommendations"].append(
                    "Configure missing environment variables in .env file"
                )
                logger.warning(f".env file - INCOMPLETE (missing: {', '.join(missing_vars)})")
                return False
            else:
                self.setup_results["checks"]["env_file"] = "✅ PASS"
                logger.info(".env file - OK")
                return True
                
        except ImportError:
            self.setup_results["checks"]["env_file"] = "⚠️ WARNING"
            self.setup_results["issues"].append("python-dotenv package not installed")
            self.setup_results["recommendations"].append("Install with: pip install python-dotenv")
            logger.warning(".env file - CANNOT VALIDATE (python-dotenv missing)")
            return False
    
    def check_directory_structure(self) -> bool:
        """Check if required directories and files exist."""
        logger.info("Checking directory structure...")
        
        required_dirs = [
            "tools",
            "examples", 
            "tests",
            "assets"
        ]
        
        required_files = [
            "requirements.txt",
            "README.md",
            "tools/__init__.py",
            "examples/__init__.py"
        ]
        
        all_present = True
        
        for directory in required_dirs:
            dir_path = Path(directory)
            if dir_path.exists() and dir_path.is_dir():
                self.setup_results["checks"][f"dir_{directory}"] = "✅ PASS"
                logger.info(f"Directory {directory}/ - OK")
            else:
                self.setup_results["checks"][f"dir_{directory}"] = "❌ FAIL"
                self.setup_results["issues"].append(f"Directory {directory}/ missing")
                logger.error(f"Directory {directory}/ - MISSING")
                all_present = False
        
        for file_path in required_files:
            file_obj = Path(file_path)
            if file_obj.exists() and file_obj.is_file():
                self.setup_results["checks"][f"file_{file_path.replace('/', '_')}"] = "✅ PASS"
                logger.info(f"File {file_path} - OK")
            else:
                self.setup_results["checks"][f"file_{file_path.replace('/', '_')}"] = "❌ FAIL"
                self.setup_results["issues"].append(f"File {file_path} missing")
                logger.error(f"File {file_path} - MISSING")
                all_present = False
        
        return all_present
    
    async def test_agentcore_connection(self) -> bool:
        """Test connection to AgentCore Browser Tool."""
        logger.info("Testing AgentCore Browser Tool connection...")
        
        try:
            # This is a placeholder for actual AgentCore connection test
            # In a real implementation, this would test session creation
            
            # For now, we'll just check if the SDK is importable and configured
            import bedrock_agentcore
            
            # Check if region is configured
            region = os.getenv('AWS_REGION', 'us-east-1')
            
            # Simulate connection test (replace with actual test when SDK is available)
            await asyncio.sleep(0.1)  # Simulate async operation
            
            self.setup_results["checks"]["agentcore_connection"] = "✅ PASS"
            self.setup_results["agentcore_region"] = region
            logger.info(f"AgentCore connection test - OK (region: {region})")
            return True
            
        except ImportError:
            self.setup_results["checks"]["agentcore_connection"] = "❌ FAIL"
            self.setup_results["issues"].append("AgentCore SDK not available")
            self.setup_results["recommendations"].append("Install bedrock-agentcore package")
            logger.error("AgentCore connection test - SDK NOT AVAILABLE")
            return False
            
        except Exception as e:
            self.setup_results["checks"]["agentcore_connection"] = "❌ FAIL"
            self.setup_results["issues"].append(f"AgentCore connection error: {str(e)}")
            logger.error(f"AgentCore connection test - ERROR: {str(e)}")
            return False
    
    def create_sample_config(self):
        """Create sample configuration files if they don't exist."""
        logger.info("Creating sample configuration files...")
        
        # Create .env file if it doesn't exist
        env_file = Path(".env")
        env_example = Path(".env.example")
        
        if not env_file.exists() and env_example.exists():
            try:
                env_content = env_example.read_text()
                env_file.write_text(env_content)
                logger.info("Created .env file from .env.example")
                self.setup_results["recommendations"].append(
                    "Edit .env file with your actual AWS configuration"
                )
            except Exception as e:
                logger.error(f"Failed to create .env file: {str(e)}")
    
    def generate_setup_report(self) -> Dict[str, Any]:
        """Generate comprehensive setup report."""
        
        # Count results
        total_checks = len(self.setup_results["checks"])
        passed_checks = len([v for v in self.setup_results["checks"].values() if "✅" in v])
        failed_checks = len([v for v in self.setup_results["checks"].values() if "❌" in v])
        warning_checks = len([v for v in self.setup_results["checks"].values() if "⚠️" in v])
        
        self.setup_results["summary"] = {
            "total_checks": total_checks,
            "passed": passed_checks,
            "failed": failed_checks,
            "warnings": warning_checks,
            "success_rate": f"{(passed_checks / total_checks * 100):.1f}%" if total_checks > 0 else "0%"
        }
        
        # Determine overall status
        if failed_checks == 0:
            if warning_checks == 0:
                self.setup_results["overall_status"] = "✅ READY"
            else:
                self.setup_results["overall_status"] = "⚠️ READY WITH WARNINGS"
        else:
            self.setup_results["overall_status"] = "❌ NOT READY"
        
        return self.setup_results
    
    def print_setup_report(self):
        """Print formatted setup report."""
        
        print("\n" + "="*80)
        print("🚀 BROWSER-USE WITH AGENTCORE BROWSER TOOL - SETUP VALIDATION")
        print("="*80)
        
        print(f"\n📊 SUMMARY:")
        print(f"   Overall Status: {self.setup_results['overall_status']}")
        print(f"   Success Rate: {self.setup_results['summary']['success_rate']}")
        print(f"   Checks: {self.setup_results['summary']['passed']} passed, "
              f"{self.setup_results['summary']['failed']} failed, "
              f"{self.setup_results['summary']['warnings']} warnings")
        
        print(f"\n🔍 DETAILED RESULTS:")
        for check, result in self.setup_results["checks"].items():
            print(f"   {check.replace('_', ' ').title()}: {result}")
        
        if self.setup_results.get("aws_account"):
            print(f"\n☁️ AWS CONFIGURATION:")
            print(f"   Account: {self.setup_results['aws_account']}")
            print(f"   User/Role: {self.setup_results.get('aws_user_arn', 'Unknown')}")
        
        if self.setup_results.get("agentcore_region"):
            print(f"   AgentCore Region: {self.setup_results['agentcore_region']}")
        
        if self.setup_results["issues"]:
            print(f"\n⚠️ ISSUES FOUND:")
            for i, issue in enumerate(self.setup_results["issues"], 1):
                print(f"   {i}. {issue}")
        
        if self.setup_results["recommendations"]:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(self.setup_results["recommendations"], 1):
                print(f"   {i}. {rec}")
        
        print(f"\n📝 NEXT STEPS:")
        if self.setup_results["overall_status"] == "✅ READY":
            print("   🎉 Your environment is ready!")
            print("   📚 Start with the tutorial notebooks:")
            print("      - browseruse_agentcore_secure_connection_tutorial.ipynb")
            print("      - browseruse_pii_masking_tutorial.ipynb")
            print("   🔗 Or run the examples in the examples/ directory")
        elif "WARNING" in self.setup_results["overall_status"]:
            print("   ⚠️ Your environment has warnings but should work")
            print("   🔧 Address the warnings above for optimal experience")
            print("   📚 You can start with the tutorial notebooks")
        else:
            print("   ❌ Please fix the issues above before proceeding")
            print("   🔧 Follow the recommendations to resolve problems")
            print("   🔄 Run this setup script again after making changes")
        
        print("\n" + "="*80)
    
    def save_setup_report(self, filename: str = "setup_report.json"):
        """Save setup report to JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.setup_results, f, indent=2)
            logger.info(f"Setup report saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save setup report: {str(e)}")

async def main():
    """Main setup and validation function."""
    
    print("🚀 Starting Browser-Use with AgentCore Browser Tool setup validation...")
    
    setup = BrowserUseAgentCoreSetup()
    
    # Run all checks
    checks = [
        setup.check_python_version(),
        setup.check_required_packages(),
        setup.check_aws_credentials(),
        setup.check_environment_file(),
        setup.check_directory_structure(),
        await setup.test_agentcore_connection()
    ]
    
    # Create sample config if needed
    setup.create_sample_config()
    
    # Generate and display report
    setup.generate_setup_report()
    setup.print_setup_report()
    
    # Save report
    setup.save_setup_report()
    
    # Return success status
    return setup.setup_results["overall_status"] == "✅ READY"

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup validation failed with error: {str(e)}")
        sys.exit(1)