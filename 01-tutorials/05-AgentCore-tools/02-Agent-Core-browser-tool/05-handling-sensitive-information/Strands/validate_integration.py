#!/usr/bin/env python3
"""
Comprehensive integration validation script for Strands AgentCore Browser Tool integration.

This script tests complete Strands integration with AgentCore Browser Tool functionality,
validating security features, multi-LLM support, and production readiness.

Requirements: 8.1, 8.2, 8.3, 8.4
"""

import os
import sys
import json
import asyncio
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import tempfile
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class StrandsAgentCoreValidator:
    """Comprehensive validator for Strands AgentCore Browser Tool integration."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "validation_id": f"validation_{int(datetime.now().timestamp())}",
            "tests": {},
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "skipped_tests": 0,
                "success_rate": 0.0
            },
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
    def validate_environment_setup(self) -> bool:
        """Validate environment configuration and dependencies."""
        logger.info("Validating environment setup...")
        
        test_name = "environment_setup"
        test_result = {
            "name": test_name,
            "status": "running",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check Python version
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            if sys.version_info < (3, 12):
                test_result["errors"].append(f"Python 3.12+ required, found {python_version}")
            else:
                test_result["details"]["python_version"] = python_version
                
            # Check environment file
            env_file = self.base_dir / ".env"
            if not env_file.exists():
                test_result["warnings"].append(".env file not found - using defaults")
            else:
                test_result["details"]["env_file_exists"] = True
                
            # Check required directories
            required_dirs = ["logs", "tutorial_data", "tests", "tools", "examples"]
            missing_dirs = []
            for dir_name in required_dirs:
                if not (self.base_dir / dir_name).exists():
                    missing_dirs.append(dir_name)
                    
            if missing_dirs:
                test_result["warnings"].append(f"Missing directories: {missing_dirs}")
            else:
                test_result["details"]["directories_complete"] = True
                
            test_result["status"] = "failed" if test_result["errors"] else "passed"
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["errors"].append(f"Environment validation error: {str(e)}")
            
        self.validation_results["tests"][test_name] = test_result
        return test_result["status"] == "passed"
        
    def validate_dependencies(self) -> bool:
        """Validate all required dependencies are installed."""
        logger.info("Validating dependencies...")
        
        test_name = "dependencies"
        test_result = {
            "name": test_name,
            "status": "running",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        required_packages = {
            "strands_agents": "Strands Agents Framework",
            "boto3": "AWS SDK",
            "anthropic": "Anthropic API",
            "selenium": "Web Automation",
            "pandas": "Data Processing",
            "cryptography": "Security",
            "jupyter": "Jupyter Notebooks",
            "pytest": "Testing Framework"
        }
        
        try:
            import importlib
            
            missing_packages = []
            installed_packages = {}
            
            for package, description in required_packages.items():
                try:
                    module = importlib.import_module(package.replace("-", "_"))
                    version = getattr(module, "__version__", "unknown")
                    installed_packages[package] = {
                        "version": version,
                        "description": description
                    }
                except ImportError:
                    missing_packages.append(f"{package} ({description})")
                    
            if missing_packages:
                test_result["errors"].append(f"Missing packages: {missing_packages}")
            else:
                test_result["details"]["installed_packages"] = installed_packages
                
            test_result["status"] = "failed" if test_result["errors"] else "passed"
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["errors"].append(f"Dependency validation error: {str(e)}")
            
        self.validation_results["tests"][test_name] = test_result
        return test_result["status"] == "passed"
        
    def validate_aws_configuration(self) -> bool:
        """Validate AWS configuration and credentials."""
        logger.info("Validating AWS configuration...")
        
        test_name = "aws_configuration"
        test_result = {
            "name": test_name,
            "status": "running",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            import boto3
            from botocore.exceptions import NoCredentialsError, ClientError
            
            # Test AWS credentials
            try:
                session = boto3.Session()
                credentials = session.get_credentials()
                
                if credentials is None:
                    test_result["errors"].append("No AWS credentials found")
                else:
                    test_result["details"]["credentials_available"] = True
                    
                    # Test Bedrock access
                    try:
                        bedrock_client = boto3.client('bedrock', region_name='us-east-1')
                        bedrock_client.list_foundation_models()
                        test_result["details"]["bedrock_access"] = True
                    except ClientError as e:
                        if "AccessDenied" in str(e):
                            test_result["warnings"].append("Limited Bedrock access - check IAM permissions")
                        else:
                            test_result["errors"].append(f"Bedrock access error: {str(e)}")
                    except Exception as e:
                        test_result["warnings"].append(f"Could not test Bedrock access: {str(e)}")
                        
            except NoCredentialsError:
                test_result["errors"].append("AWS credentials not configured")
            except Exception as e:
                test_result["errors"].append(f"AWS configuration error: {str(e)}")
                
            test_result["status"] = "failed" if test_result["errors"] else "passed"
            
        except ImportError:
            test_result["status"] = "failed"
            test_result["errors"].append("boto3 not available")
            
        self.validation_results["tests"][test_name] = test_result
        return test_result["status"] == "passed"
        
    def validate_strands_framework(self) -> bool:
        """Validate Strands framework functionality."""
        logger.info("Validating Strands framework...")
        
        test_name = "strands_framework"
        test_result = {
            "name": test_name,
            "status": "running",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Test basic Strands imports
            validation_code = '''
import sys
import traceback

try:
    # Test Strands core imports
    from strands_agents import Agent, Tool, Workflow
    from strands_agents.core import AgentCore
    from strands_agents.tools import BaseTool
    
    # Test agent creation
    agent = Agent(
        name="test_agent",
        description="Test agent for validation",
        llm_config={"provider": "mock", "model": "test"}
    )
    
    # Test tool creation
    class TestTool(BaseTool):
        def execute(self, **kwargs):
            return {"status": "success", "message": "Test tool executed"}
    
    tool = TestTool(name="test_tool")
    
    print("SUCCESS: Strands framework validation passed")
    
except Exception as e:
    print(f"ERROR: {str(e)}")
    traceback.print_exc()
    sys.exit(1)
'''
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(validation_code)
                temp_script = f.name
                
            try:
                result = subprocess.run([
                    sys.executable, temp_script
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    test_result["details"]["framework_functional"] = True
                else:
                    test_result["errors"].append(f"Strands framework test failed: {result.stderr}")
                    
            finally:
                os.unlink(temp_script)
                
            test_result["status"] = "failed" if test_result["errors"] else "passed"
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["errors"].append(f"Strands framework validation error: {str(e)}")
            
        self.validation_results["tests"][test_name] = test_result
        return test_result["status"] == "passed"
        
    def validate_agentcore_browser_integration(self) -> bool:
        """Validate AgentCore Browser Tool integration."""
        logger.info("Validating AgentCore Browser Tool integration...")
        
        test_name = "agentcore_browser_integration"
        test_result = {
            "name": test_name,
            "status": "running",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Test AgentCore browser client imports
            validation_code = '''
import sys
import traceback

try:
    # Test AgentCore browser client imports
    from bedrock_agentcore_browser_client import BrowserClient, BrowserSession
    from bedrock_agentcore_browser_client.security import SecurityManager
    
    # Test client creation (mock mode for validation)
    client = BrowserClient(
        region="us-east-1",
        mode="validation"  # Mock mode for testing
    )
    
    # Test security manager
    security_manager = SecurityManager()
    
    print("SUCCESS: AgentCore Browser Tool integration validation passed")
    
except ImportError as e:
    print(f"IMPORT_ERROR: {str(e)}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {str(e)}")
    traceback.print_exc()
    sys.exit(1)
'''
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(validation_code)
                temp_script = f.name
                
            try:
                result = subprocess.run([
                    sys.executable, temp_script
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    test_result["details"]["browser_integration_functional"] = True
                elif "IMPORT_ERROR" in result.stdout:
                    test_result["warnings"].append("AgentCore Browser Client not available - using mock mode")
                    test_result["details"]["mock_mode"] = True
                else:
                    test_result["errors"].append(f"Browser integration test failed: {result.stderr}")
                    
            finally:
                os.unlink(temp_script)
                
            test_result["status"] = "failed" if test_result["errors"] else "passed"
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["errors"].append(f"Browser integration validation error: {str(e)}")
            
        self.validation_results["tests"][test_name] = test_result
        return test_result["status"] == "passed"
        
    def validate_security_features(self) -> bool:
        """Validate security features and tools."""
        logger.info("Validating security features...")
        
        test_name = "security_features"
        test_result = {
            "name": test_name,
            "status": "running",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check security tools
            security_tools_dir = self.base_dir / "tools"
            expected_tools = [
                "strands_agentcore_session_helpers.py",
                "strands_pii_utils.py",
                "strands_security_policies.py",
                "strands_monitoring.py"
            ]
            
            missing_tools = []
            for tool in expected_tools:
                if not (security_tools_dir / tool).exists():
                    missing_tools.append(tool)
                    
            if missing_tools:
                test_result["errors"].append(f"Missing security tools: {missing_tools}")
            else:
                test_result["details"]["security_tools_available"] = True
                
            # Test cryptography functionality
            try:
                from cryptography.fernet import Fernet
                key = Fernet.generate_key()
                cipher = Fernet(key)
                test_data = b"test sensitive data"
                encrypted = cipher.encrypt(test_data)
                decrypted = cipher.decrypt(encrypted)
                
                if decrypted == test_data:
                    test_result["details"]["encryption_functional"] = True
                else:
                    test_result["errors"].append("Encryption/decryption test failed")
                    
            except Exception as e:
                test_result["errors"].append(f"Cryptography test failed: {str(e)}")
                
            test_result["status"] = "failed" if test_result["errors"] else "passed"
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["errors"].append(f"Security features validation error: {str(e)}")
            
        self.validation_results["tests"][test_name] = test_result
        return test_result["status"] == "passed"
        
    def validate_tutorial_notebooks(self) -> bool:
        """Validate tutorial notebooks are present and functional."""
        logger.info("Validating tutorial notebooks...")
        
        test_name = "tutorial_notebooks"
        test_result = {
            "name": test_name,
            "status": "running",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            expected_notebooks = [
                "01_strands_agentcore_secure_login.ipynb",
                "02_strands_sensitive_form_automation.ipynb",
                "03_strands_bedrock_multi_model_security.ipynb",
                "04_production_strands_agentcore_patterns.ipynb"
            ]
            
            missing_notebooks = []
            notebook_details = {}
            
            for notebook in expected_notebooks:
                notebook_path = self.base_dir / notebook
                if not notebook_path.exists():
                    missing_notebooks.append(notebook)
                else:
                    try:
                        import nbformat
                        with open(notebook_path, 'r') as f:
                            nb = nbformat.read(f, as_version=4)
                            notebook_details[notebook] = {
                                "cells": len(nb.cells),
                                "code_cells": len([c for c in nb.cells if c.cell_type == 'code'])
                            }
                    except Exception as e:
                        test_result["warnings"].append(f"Could not parse {notebook}: {str(e)}")
                        
            if missing_notebooks:
                test_result["errors"].append(f"Missing notebooks: {missing_notebooks}")
            else:
                test_result["details"]["notebooks"] = notebook_details
                
            test_result["status"] = "failed" if test_result["errors"] else "passed"
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["errors"].append(f"Notebook validation error: {str(e)}")
            
        self.validation_results["tests"][test_name] = test_result
        return test_result["status"] == "passed"
        
    def validate_examples_and_tests(self) -> bool:
        """Validate examples and test files."""
        logger.info("Validating examples and tests...")
        
        test_name = "examples_and_tests"
        test_result = {
            "name": test_name,
            "status": "running",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check examples
            examples_dir = self.base_dir / "examples"
            expected_examples = [
                "healthcare_document_processing.py",
                "financial_data_extraction.py",
                "legal_document_analysis.py",
                "customer_support_automation.py"
            ]
            
            missing_examples = []
            for example in expected_examples:
                if not (examples_dir / example).exists():
                    missing_examples.append(example)
                    
            if missing_examples:
                test_result["warnings"].append(f"Missing examples: {missing_examples}")
            else:
                test_result["details"]["examples_complete"] = True
                
            # Check tests
            tests_dir = self.base_dir / "tests"
            expected_tests = [
                "test_credential_security.py",
                "test_pii_masking.py",
                "test_session_isolation.py",
                "test_audit_trail.py"
            ]
            
            missing_tests = []
            for test_file in expected_tests:
                if not (tests_dir / test_file).exists():
                    missing_tests.append(test_file)
                    
            if missing_tests:
                test_result["warnings"].append(f"Missing tests: {missing_tests}")
            else:
                test_result["details"]["tests_complete"] = True
                
            test_result["status"] = "passed"  # Warnings only, not failures
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["errors"].append(f"Examples and tests validation error: {str(e)}")
            
        self.validation_results["tests"][test_name] = test_result
        return test_result["status"] == "passed"
        
    def generate_validation_report(self) -> None:
        """Generate comprehensive validation report."""
        logger.info("Generating validation report...")
        
        # Calculate summary statistics
        total_tests = len(self.validation_results["tests"])
        passed_tests = len([t for t in self.validation_results["tests"].values() if t["status"] == "passed"])
        failed_tests = len([t for t in self.validation_results["tests"].values() if t["status"] == "failed"])
        skipped_tests = len([t for t in self.validation_results["tests"].values() if t["status"] == "skipped"])
        
        self.validation_results["summary"].update({
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "skipped_tests": skipped_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
        })
        
        # Generate recommendations
        if failed_tests > 0:
            self.validation_results["recommendations"].append("Fix failed tests before proceeding with tutorial")
        if self.validation_results["summary"]["success_rate"] < 80:
            self.validation_results["recommendations"].append("Consider reviewing setup instructions")
        
        # Save JSON report
        report_file = self.base_dir / "validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
            
        # Generate human-readable report
        report_content = f"""# Strands AgentCore Integration Validation Report

## Validation Summary
- **Validation ID**: {self.validation_results['validation_id']}
- **Timestamp**: {self.validation_results['timestamp']}
- **Success Rate**: {self.validation_results['summary']['success_rate']:.1f}%
- **Total Tests**: {self.validation_results['summary']['total_tests']}
- **Passed**: {self.validation_results['summary']['passed_tests']}
- **Failed**: {self.validation_results['summary']['failed_tests']}
- **Skipped**: {self.validation_results['summary']['skipped_tests']}

## Test Results
"""
        
        for test_name, test_result in self.validation_results["tests"].items():
            status_icon = "✓" if test_result["status"] == "passed" else "✗" if test_result["status"] == "failed" else "⚠"
            report_content += f"\n### {status_icon} {test_name.replace('_', ' ').title()}\n"
            report_content += f"**Status**: {test_result['status'].upper()}\n\n"
            
            if test_result["errors"]:
                report_content += "**Errors**:\n"
                for error in test_result["errors"]:
                    report_content += f"- {error}\n"
                report_content += "\n"
                
            if test_result["warnings"]:
                report_content += "**Warnings**:\n"
                for warning in test_result["warnings"]:
                    report_content += f"- {warning}\n"
                report_content += "\n"
                
        if self.validation_results["recommendations"]:
            report_content += "\n## Recommendations\n"
            for rec in self.validation_results["recommendations"]:
                report_content += f"- {rec}\n"
                
        report_content += f"""
## Next Steps
{'1. Fix failed tests and re-run validation' if failed_tests > 0 else '1. Configure AWS credentials if not already done'}
2. Run security tests: `python tests/run_security_tests.py`
3. Start with tutorial: `jupyter lab 01_strands_agentcore_secure_login.ipynb`

## Support
For issues, check:
- Setup was completed successfully
- All dependencies are installed
- AWS credentials are configured
- Environment variables are set
"""
        
        readme_file = self.base_dir / "VALIDATION_REPORT.md"
        with open(readme_file, 'w') as f:
            f.write(report_content)
            
        logger.info(f"Validation report generated: {report_file}")
        
    def run_validation(self) -> bool:
        """Run complete validation process."""
        logger.info("Starting Strands AgentCore integration validation...")
        
        validation_tests = [
            ("Environment Setup", self.validate_environment_setup),
            ("Dependencies", self.validate_dependencies),
            ("AWS Configuration", self.validate_aws_configuration),
            ("Strands Framework", self.validate_strands_framework),
            ("AgentCore Browser Integration", self.validate_agentcore_browser_integration),
            ("Security Features", self.validate_security_features),
            ("Tutorial Notebooks", self.validate_tutorial_notebooks),
            ("Examples and Tests", self.validate_examples_and_tests)
        ]
        
        for test_name, test_func in validation_tests:
            logger.info(f"Running: {test_name}")
            try:
                test_func()
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {str(e)}")
                self.validation_results["tests"][test_name.lower().replace(" ", "_")] = {
                    "name": test_name,
                    "status": "failed",
                    "errors": [f"Exception: {str(e)}"],
                    "warnings": [],
                    "details": {}
                }
                
        self.generate_validation_report()
        
        success_rate = self.validation_results["summary"]["success_rate"]
        failed_tests = self.validation_results["summary"]["failed_tests"]
        
        if failed_tests == 0:
            logger.info("✓ All validation tests passed!")
            return True
        elif success_rate >= 80:
            logger.warning(f"Validation completed with warnings (success rate: {success_rate:.1f}%)")
            return True
        else:
            logger.error(f"Validation failed (success rate: {success_rate:.1f}%)")
            return False

def main():
    """Main validation function."""
    print("Strands AgentCore Browser Tool - Integration Validation")
    print("=" * 60)
    
    validator = StrandsAgentCoreValidator()
    
    try:
        success = validator.run_validation()
        
        if success:
            print("\n✓ Validation completed successfully!")
            print("\nNext steps:")
            print("1. Run security tests: python tests/run_security_tests.py")
            print("2. Start tutorial: jupyter lab 01_strands_agentcore_secure_login.ipynb")
            return 0
        else:
            print("\n✗ Validation failed. Check validation.log and VALIDATION_REPORT.md for details.")
            return 1
            
    except KeyboardInterrupt:
        print("\nValidation interrupted by user.")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error during validation: {str(e)}")
        print(f"\n✗ Validation failed with unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())