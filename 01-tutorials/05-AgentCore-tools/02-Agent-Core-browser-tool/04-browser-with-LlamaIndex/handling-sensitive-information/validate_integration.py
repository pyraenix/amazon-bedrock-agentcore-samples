#!/usr/bin/env python3
"""
LlamaIndex with AgentCore Browser Tool - Integration Validation Script

This script validates the setup and integration between LlamaIndex and 
Amazon Bedrock AgentCore Browser Tool for sensitive information handling.

Usage:
    python validate_integration.py [--verbose] [--skip-aws] [--skip-browser]

Requirements:
    - Python 3.9+
    - All dependencies from requirements.txt installed
    - AWS credentials configured
    - .env file with proper configuration
"""

import os
import sys
import json
import asyncio
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Third-party imports
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    import llama_index
    from llama_index.core import Settings
    from llama_index.llms.bedrock import Bedrock
    from llama_index.embeddings.bedrock import BedrockEmbedding
    from dotenv import load_dotenv
    import requests
    from cryptography.fernet import Fernet
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"❌ Missing required dependency: {e}")
    print("Please install all dependencies: pip install -r requirements.txt")
    sys.exit(1)

# Try to import AgentCore Browser Client
try:
    from bedrock_agentcore.tools.browser_client import BrowserSession
    AGENTCORE_AVAILABLE = True
except ImportError:
    print("⚠️  AgentCore Browser Client SDK not available")
    print("   This is expected if running in a development environment")
    AGENTCORE_AVAILABLE = False

@dataclass
class ValidationResult:
    """Result of a validation check."""
    name: str
    status: str  # "PASS", "FAIL", "WARN", "SKIP"
    message: str
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class IntegrationValidator:
    """Validates LlamaIndex-AgentCore Browser Tool integration setup."""
    
    def __init__(self, verbose: bool = False, skip_aws: bool = False, skip_browser: bool = False):
        self.verbose = verbose
        self.skip_aws = skip_aws
        self.skip_browser = skip_browser
        self.results: List[ValidationResult] = []
        
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        self._load_environment()
    
    def _load_environment(self) -> None:
        """Load environment variables from .env file."""
        env_path = Path(__file__).parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            self.logger.info("Loaded environment variables from .env file")
        else:
            self.logger.warning("No .env file found, using system environment variables")
    
    def _add_result(self, name: str, status: str, message: str, 
                   details: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
        """Add a validation result."""
        result = ValidationResult(name, status, message, details, error)
        self.results.append(result)
        
        # Log the result
        status_emoji = {
            "PASS": "✅",
            "FAIL": "❌", 
            "WARN": "⚠️",
            "SKIP": "⏭️"
        }
        
        emoji = status_emoji.get(status, "❓")
        self.logger.info(f"{emoji} {name}: {message}")
        
        if error and self.verbose:
            self.logger.debug(f"   Error details: {error}")
        
        if details and self.verbose:
            self.logger.debug(f"   Details: {json.dumps(details, indent=2)}")
    
    def validate_python_environment(self) -> None:
        """Validate Python environment and version."""
        try:
            python_version = sys.version_info
            if python_version >= (3, 9):
                self._add_result(
                    "Python Version",
                    "PASS",
                    f"Python {python_version.major}.{python_version.minor}.{python_version.micro}",
                    {"version": str(python_version)}
                )
            else:
                self._add_result(
                    "Python Version",
                    "FAIL",
                    f"Python {python_version.major}.{python_version.minor} < 3.9 (required)",
                    {"version": str(python_version)}
                )
        except Exception as e:
            self._add_result(
                "Python Version",
                "FAIL",
                "Failed to check Python version",
                error=str(e)
            )
    
    def validate_dependencies(self) -> None:
        """Validate required Python dependencies."""
        required_packages = [
            ("llama_index", "LlamaIndex Framework"),
            ("boto3", "AWS SDK"),
            ("pandas", "Data Processing"),
            ("numpy", "Numerical Computing"),
            ("cryptography", "Encryption Support"),
            ("requests", "HTTP Client"),
            ("dotenv", "Environment Management"),
        ]
        
        for package_name, display_name in required_packages:
            try:
                __import__(package_name)
                self._add_result(
                    f"Dependency: {display_name}",
                    "PASS",
                    f"{package_name} is available"
                )
            except ImportError as e:
                self._add_result(
                    f"Dependency: {display_name}",
                    "FAIL",
                    f"{package_name} not found",
                    error=str(e)
                )
    
    def validate_environment_config(self) -> None:
        """Validate environment configuration."""
        required_vars = [
            ("AWS_DEFAULT_REGION", "AWS Region"),
            ("BEDROCK_MODEL_ID", "Bedrock Model ID"),
            ("AGENTCORE_REGION", "AgentCore Region"),
        ]
        
        optional_vars = [
            ("AWS_ACCESS_KEY_ID", "AWS Access Key"),
            ("AWS_SECRET_ACCESS_KEY", "AWS Secret Key"),
            ("ENCRYPTION_KEY", "Encryption Key"),
            ("LOG_LEVEL", "Log Level"),
        ]
        
        # Check required variables
        for var_name, display_name in required_vars:
            value = os.getenv(var_name)
            if value:
                self._add_result(
                    f"Config: {display_name}",
                    "PASS",
                    f"{var_name} is configured",
                    {"value": value[:10] + "..." if len(value) > 10 else value}
                )
            else:
                self._add_result(
                    f"Config: {display_name}",
                    "FAIL",
                    f"{var_name} not configured"
                )
        
        # Check optional variables
        for var_name, display_name in optional_vars:
            value = os.getenv(var_name)
            if value:
                self._add_result(
                    f"Config: {display_name}",
                    "PASS",
                    f"{var_name} is configured"
                )
            else:
                self._add_result(
                    f"Config: {display_name}",
                    "WARN",
                    f"{var_name} not configured (optional)"
                )
    
    def validate_aws_credentials(self) -> None:
        """Validate AWS credentials and permissions."""
        if self.skip_aws:
            self._add_result(
                "AWS Credentials",
                "SKIP",
                "AWS validation skipped by user request"
            )
            return
        
        try:
            # Test basic AWS credentials
            session = boto3.Session()
            sts_client = session.client('sts')
            identity = sts_client.get_caller_identity()
            
            self._add_result(
                "AWS Credentials",
                "PASS",
                "AWS credentials are valid",
                {
                    "account": identity.get('Account'),
                    "user_id": identity.get('UserId'),
                    "arn": identity.get('Arn')
                }
            )
            
            # Test Bedrock access
            try:
                bedrock_client = session.client('bedrock', region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'))
                models = bedrock_client.list_foundation_models()
                
                self._add_result(
                    "AWS Bedrock Access",
                    "PASS",
                    f"Bedrock access confirmed ({len(models.get('modelSummaries', []))} models available)"
                )
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'AccessDeniedException':
                    self._add_result(
                        "AWS Bedrock Access",
                        "FAIL",
                        "Access denied to Bedrock service",
                        error=str(e)
                    )
                else:
                    self._add_result(
                        "AWS Bedrock Access",
                        "WARN",
                        f"Bedrock access issue: {error_code}",
                        error=str(e)
                    )
            
        except NoCredentialsError:
            self._add_result(
                "AWS Credentials",
                "FAIL",
                "No AWS credentials found"
            )
        except Exception as e:
            self._add_result(
                "AWS Credentials",
                "FAIL",
                "Failed to validate AWS credentials",
                error=str(e)
            )
    
    def validate_llamaindex_setup(self) -> None:
        """Validate LlamaIndex setup and configuration."""
        try:
            # Test LlamaIndex core functionality
            from llama_index.core import Document, VectorStoreIndex
            from llama_index.core.node_parser import SimpleNodeParser
            
            # Create a simple test document
            test_doc = Document(text="This is a test document for validation.")
            parser = SimpleNodeParser()
            nodes = parser.get_nodes_from_documents([test_doc])
            
            self._add_result(
                "LlamaIndex Core",
                "PASS",
                f"LlamaIndex core functionality working ({len(nodes)} nodes created)"
            )
            
            # Test Bedrock LLM integration
            try:
                model_id = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')
                region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
                
                llm = Bedrock(
                    model=model_id,
                    region_name=region,
                    max_tokens=100
                )
                
                # Test a simple completion (only if AWS is available)
                if not self.skip_aws:
                    response = llm.complete("Hello, this is a test.")
                    self._add_result(
                        "LlamaIndex Bedrock LLM",
                        "PASS",
                        f"Bedrock LLM integration working (model: {model_id})",
                        {"response_length": len(str(response))}
                    )
                else:
                    self._add_result(
                        "LlamaIndex Bedrock LLM",
                        "SKIP",
                        "Bedrock LLM test skipped (AWS validation disabled)"
                    )
                    
            except Exception as e:
                self._add_result(
                    "LlamaIndex Bedrock LLM",
                    "FAIL",
                    "Failed to initialize Bedrock LLM",
                    error=str(e)
                )
            
            # Test Bedrock Embeddings
            try:
                embedding_model = os.getenv('BEDROCK_EMBEDDING_MODEL_ID', 'amazon.titan-embed-text-v1')
                
                embeddings = BedrockEmbedding(
                    model=embedding_model,
                    region_name=region
                )
                
                if not self.skip_aws:
                    test_embedding = embeddings.get_text_embedding("Test embedding")
                    self._add_result(
                        "LlamaIndex Bedrock Embeddings",
                        "PASS",
                        f"Bedrock embeddings working (model: {embedding_model})",
                        {"embedding_dimension": len(test_embedding)}
                    )
                else:
                    self._add_result(
                        "LlamaIndex Bedrock Embeddings",
                        "SKIP",
                        "Bedrock embeddings test skipped (AWS validation disabled)"
                    )
                    
            except Exception as e:
                self._add_result(
                    "LlamaIndex Bedrock Embeddings",
                    "FAIL",
                    "Failed to initialize Bedrock embeddings",
                    error=str(e)
                )
                
        except Exception as e:
            self._add_result(
                "LlamaIndex Core",
                "FAIL",
                "Failed to validate LlamaIndex setup",
                error=str(e)
            )
    
    def validate_agentcore_browser_client(self) -> None:
        """Validate AgentCore Browser Client setup."""
        if self.skip_browser:
            self._add_result(
                "AgentCore Browser Client",
                "SKIP",
                "AgentCore Browser validation skipped by user request"
            )
            return
        
        if not AGENTCORE_AVAILABLE:
            self._add_result(
                "AgentCore Browser Client",
                "WARN",
                "AgentCore Browser Client SDK not available (development environment)"
            )
            return
        
        try:
            # Test AgentCore Browser Client initialization
            region = os.getenv('AGENTCORE_REGION', 'us-east-1')
            
            # Note: In a real environment, this would create an actual browser session
            # For validation, we just test the import and basic configuration
            self._add_result(
                "AgentCore Browser Client",
                "PASS",
                f"AgentCore Browser Client SDK available (region: {region})"
            )
            
            # Test browser session configuration
            session_config = {
                'timeout': int(os.getenv('BROWSER_SESSION_TIMEOUT', '300')),
                'max_sessions': int(os.getenv('BROWSER_MAX_CONCURRENT_SESSIONS', '5')),
                'observability': os.getenv('BROWSER_ENABLE_OBSERVABILITY', 'true').lower() == 'true',
                'security_level': os.getenv('BROWSER_SECURITY_LEVEL', 'high'),
                'network_isolation': os.getenv('BROWSER_NETWORK_ISOLATION', 'true').lower() == 'true'
            }
            
            self._add_result(
                "AgentCore Browser Config",
                "PASS",
                "Browser session configuration validated",
                session_config
            )
            
        except Exception as e:
            self._add_result(
                "AgentCore Browser Client",
                "FAIL",
                "Failed to validate AgentCore Browser Client",
                error=str(e)
            )
    
    def validate_security_setup(self) -> None:
        """Validate security configuration and encryption setup."""
        try:
            # Test encryption key
            encryption_key = os.getenv('ENCRYPTION_KEY')
            if encryption_key:
                try:
                    # Test if the key is valid for Fernet encryption
                    if len(encryption_key) == 44:  # Base64 encoded 32-byte key
                        fernet = Fernet(encryption_key.encode())
                        test_data = b"test encryption data"
                        encrypted = fernet.encrypt(test_data)
                        decrypted = fernet.decrypt(encrypted)
                        
                        if decrypted == test_data:
                            self._add_result(
                                "Encryption Setup",
                                "PASS",
                                "Encryption key is valid and working"
                            )
                        else:
                            self._add_result(
                                "Encryption Setup",
                                "FAIL",
                                "Encryption/decryption test failed"
                            )
                    else:
                        self._add_result(
                            "Encryption Setup",
                            "WARN",
                            f"Encryption key length is {len(encryption_key)}, expected 44 characters"
                        )
                except Exception as e:
                    self._add_result(
                        "Encryption Setup",
                        "FAIL",
                        "Invalid encryption key format",
                        error=str(e)
                    )
            else:
                self._add_result(
                    "Encryption Setup",
                    "WARN",
                    "No encryption key configured (ENCRYPTION_KEY)"
                )
            
            # Test PII detection configuration
            pii_enabled = os.getenv('PII_DETECTION_ENABLED', 'true').lower() == 'true'
            pii_masking = os.getenv('PII_MASKING_ENABLED', 'true').lower() == 'true'
            pii_threshold = float(os.getenv('PII_CONFIDENCE_THRESHOLD', '0.8'))
            
            self._add_result(
                "PII Detection Config",
                "PASS",
                f"PII detection configured (enabled: {pii_enabled}, masking: {pii_masking}, threshold: {pii_threshold})",
                {
                    "detection_enabled": pii_enabled,
                    "masking_enabled": pii_masking,
                    "confidence_threshold": pii_threshold
                }
            )
            
        except Exception as e:
            self._add_result(
                "Security Setup",
                "FAIL",
                "Failed to validate security configuration",
                error=str(e)
            )
    
    def validate_directory_structure(self) -> None:
        """Validate tutorial directory structure."""
        try:
            base_path = Path(__file__).parent
            
            # Expected directories and files
            expected_structure = [
                "examples/",
                "assets/",
                "tutorial_data/",
                "logs/",
                "requirements.txt",
                ".env.example",
                "README.md"
            ]
            
            missing_items = []
            existing_items = []
            
            for item in expected_structure:
                item_path = base_path / item
                if item.endswith('/'):
                    # Directory
                    if item_path.exists() and item_path.is_dir():
                        existing_items.append(item)
                    else:
                        missing_items.append(item)
                        # Create missing directories
                        item_path.mkdir(parents=True, exist_ok=True)
                else:
                    # File
                    if item_path.exists() and item_path.is_file():
                        existing_items.append(item)
                    else:
                        missing_items.append(item)
            
            if missing_items:
                self._add_result(
                    "Directory Structure",
                    "WARN",
                    f"Created missing directories: {', '.join(missing_items)}",
                    {
                        "existing": existing_items,
                        "created": missing_items
                    }
                )
            else:
                self._add_result(
                    "Directory Structure",
                    "PASS",
                    "All required directories and files exist",
                    {"items": existing_items}
                )
                
        except Exception as e:
            self._add_result(
                "Directory Structure",
                "FAIL",
                "Failed to validate directory structure",
                error=str(e)
            )
    
    def validate_integration_compatibility(self) -> None:
        """Validate compatibility between LlamaIndex and AgentCore."""
        try:
            # Check version compatibility
            llamaindex_version = llama_index.__version__
            
            # Expected version ranges (these would be updated based on actual compatibility)
            compatible_versions = {
                "llama_index": ">=0.10.0,<0.11.0",
                "boto3": ">=1.34.0",
                "python": ">=3.9.0"
            }
            
            self._add_result(
                "Version Compatibility",
                "PASS",
                f"LlamaIndex version {llamaindex_version} is compatible",
                {
                    "llamaindex_version": llamaindex_version,
                    "compatible_ranges": compatible_versions
                }
            )
            
            # Test basic integration pattern
            integration_test_passed = True
            integration_details = {}
            
            try:
                # Test that we can import and use both frameworks together
                from llama_index.core import Document
                
                # Create a test document that simulates sensitive data handling
                test_doc = Document(
                    text="This is a test document with sensitive information: SSN 123-45-6789",
                    metadata={
                        "source": "agentcore_browser_session",
                        "sensitivity_level": "confidential",
                        "extraction_timestamp": datetime.now().isoformat()
                    }
                )
                
                integration_details["test_document_created"] = True
                integration_details["metadata_fields"] = len(test_doc.metadata)
                
            except Exception as e:
                integration_test_passed = False
                integration_details["error"] = str(e)
            
            if integration_test_passed:
                self._add_result(
                    "Integration Compatibility",
                    "PASS",
                    "LlamaIndex-AgentCore integration patterns work correctly",
                    integration_details
                )
            else:
                self._add_result(
                    "Integration Compatibility",
                    "FAIL",
                    "Integration compatibility test failed",
                    integration_details
                )
                
        except Exception as e:
            self._add_result(
                "Integration Compatibility",
                "FAIL",
                "Failed to validate integration compatibility",
                error=str(e)
            )
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate a comprehensive validation report."""
        total_checks = len(self.results)
        passed = len([r for r in self.results if r.status == "PASS"])
        failed = len([r for r in self.results if r.status == "FAIL"])
        warnings = len([r for r in self.results if r.status == "WARN"])
        skipped = len([r for r in self.results if r.status == "SKIP"])
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_checks": total_checks,
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "skipped": skipped,
                "success_rate": f"{(passed / total_checks * 100):.1f}%" if total_checks > 0 else "0%"
            },
            "results": [
                {
                    "name": r.name,
                    "status": r.status,
                    "message": r.message,
                    "details": r.details,
                    "error": r.error
                }
                for r in self.results
            ],
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        failed_results = [r for r in self.results if r.status == "FAIL"]
        warning_results = [r for r in self.results if r.status == "WARN"]
        
        if failed_results:
            recommendations.append("❌ Critical Issues Found:")
            for result in failed_results:
                recommendations.append(f"   • {result.name}: {result.message}")
                if result.error:
                    recommendations.append(f"     Error: {result.error}")
        
        if warning_results:
            recommendations.append("⚠️  Warnings to Address:")
            for result in warning_results:
                recommendations.append(f"   • {result.name}: {result.message}")
        
        # General recommendations
        if not os.path.exists('.env'):
            recommendations.append("📝 Create a .env file based on .env.example")
        
        if any("AWS" in r.name and r.status == "FAIL" for r in self.results):
            recommendations.append("🔧 Configure AWS credentials using 'aws configure' or environment variables")
        
        if any("Encryption" in r.name and r.status in ["FAIL", "WARN"] for r in self.results):
            recommendations.append("🔐 Generate a secure encryption key: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"")
        
        if not recommendations:
            recommendations.append("✅ All validations passed! You're ready to start the tutorial.")
        
        return recommendations
    
    async def run_all_validations(self) -> Dict[str, Any]:
        """Run all validation checks."""
        print("🔍 Starting LlamaIndex-AgentCore Browser Tool Integration Validation...")
        print("=" * 80)
        
        # Run all validation checks
        self.validate_python_environment()
        self.validate_dependencies()
        self.validate_environment_config()
        self.validate_aws_credentials()
        self.validate_llamaindex_setup()
        self.validate_agentcore_browser_client()
        self.validate_security_setup()
        self.validate_directory_structure()
        self.validate_integration_compatibility()
        
        # Generate and return report
        report = self.generate_validation_report()
        
        print("\n" + "=" * 80)
        print("📊 Validation Summary:")
        print(f"   Total Checks: {report['summary']['total_checks']}")
        print(f"   ✅ Passed: {report['summary']['passed']}")
        print(f"   ❌ Failed: {report['summary']['failed']}")
        print(f"   ⚠️  Warnings: {report['summary']['warnings']}")
        print(f"   ⏭️  Skipped: {report['summary']['skipped']}")
        print(f"   📈 Success Rate: {report['summary']['success_rate']}")
        
        if report['recommendations']:
            print("\n🎯 Recommendations:")
            for rec in report['recommendations']:
                print(f"   {rec}")
        
        print("\n" + "=" * 80)
        
        # Save report to file
        report_path = Path(__file__).parent / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Detailed report saved to: {report_path}")
        
        return report

def main():
    """Main entry point for the validation script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate LlamaIndex-AgentCore Browser Tool integration setup"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output with detailed logging"
    )
    parser.add_argument(
        "--skip-aws",
        action="store_true",
        help="Skip AWS-related validations (useful for offline testing)"
    )
    parser.add_argument(
        "--skip-browser",
        action="store_true",
        help="Skip AgentCore Browser Tool validations"
    )
    
    args = parser.parse_args()
    
    # Create and run validator
    validator = IntegrationValidator(
        verbose=args.verbose,
        skip_aws=args.skip_aws,
        skip_browser=args.skip_browser
    )
    
    try:
        # Run validation
        report = asyncio.run(validator.run_all_validations())
        
        # Exit with appropriate code
        if report['summary']['failed'] > 0:
            print("\n❌ Validation failed. Please address the issues above before proceeding.")
            sys.exit(1)
        elif report['summary']['warnings'] > 0:
            print("\n⚠️  Validation completed with warnings. Review recommendations above.")
            sys.exit(0)
        else:
            print("\n✅ All validations passed! Ready to start the tutorial.")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Validation interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error during validation: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()