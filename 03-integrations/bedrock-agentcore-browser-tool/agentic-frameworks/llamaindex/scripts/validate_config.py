#!/usr/bin/env python3
"""
Configuration validation script for llamaindex-agentcore-browser-integration.

This script validates configuration files, checks AWS credentials,
and verifies service connectivity.
"""

import sys
import os
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import aiohttp
import asyncio
from urllib.parse import urlparse

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

class ConfigValidator:
    """Validates configuration and connectivity."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.root_dir = Path(__file__).parent.parent
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def log(self, message: str, level: str = "INFO") -> None:
        """Log validation messages."""
        if self.verbose or level in ["ERROR", "WARNING"]:
            print(f"[{level}] {message}")
    
    def add_error(self, message: str) -> None:
        """Add validation error."""
        self.errors.append(message)
        self.log(message, "ERROR")
    
    def add_warning(self, message: str) -> None:
        """Add validation warning."""
        self.warnings.append(message)
        self.log(message, "WARNING")
    
    def load_config(self, config_path: Path) -> Optional[Dict[str, Any]]:
        """Load configuration file."""
        if not config_path.exists():
            self.add_error(f"Configuration file not found: {config_path}")
            return None
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    config = yaml.safe_load(f)
            
            self.log(f"Loaded configuration from {config_path}")
            return config
            
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            self.add_error(f"Failed to parse configuration file: {e}")
            return None
        except Exception as e:
            self.add_error(f"Failed to load configuration file: {e}")
            return None
    
    def validate_schema(self, config: Dict[str, Any]) -> bool:
        """Validate configuration schema."""
        self.log("Validating configuration schema...")
        
        # Required top-level sections
        required_sections = {
            'aws': ['region'],
            'agentcore': ['browser_tool_endpoint', 'api_version'],
            'browser': ['headless', 'viewport_width', 'viewport_height'],
            'llamaindex': ['llm_model', 'vision_model']
        }
        
        schema_valid = True
        
        for section, required_fields in required_sections.items():
            if section not in config:
                self.add_error(f"Missing required configuration section: {section}")
                schema_valid = False
                continue
            
            section_config = config[section]
            if not isinstance(section_config, dict):
                self.add_error(f"Configuration section '{section}' must be a dictionary")
                schema_valid = False
                continue
            
            for field in required_fields:
                if field not in section_config:
                    self.add_error(f"Missing required field '{field}' in section '{section}'")
                    schema_valid = False
        
        # Validate specific field types and values
        if schema_valid:
            self._validate_field_types(config)
        
        if schema_valid:
            self.log("Configuration schema validation passed")
        
        return schema_valid
    
    def _validate_field_types(self, config: Dict[str, Any]) -> None:
        """Validate specific field types and values."""
        
        # AWS region validation
        aws_config = config.get('aws', {})
        region = aws_config.get('region')
        if region and not isinstance(region, str):
            self.add_error("AWS region must be a string")
        elif region and not self._is_valid_aws_region(region):
            self.add_warning(f"AWS region '{region}' may not be valid")
        
        # Browser configuration validation
        browser_config = config.get('browser', {})
        
        viewport_width = browser_config.get('viewport_width')
        if viewport_width is not None and (not isinstance(viewport_width, int) or viewport_width <= 0):
            self.add_error("Browser viewport_width must be a positive integer")
        
        viewport_height = browser_config.get('viewport_height')
        if viewport_height is not None and (not isinstance(viewport_height, int) or viewport_height <= 0):
            self.add_error("Browser viewport_height must be a positive integer")
        
        headless = browser_config.get('headless')
        if headless is not None and not isinstance(headless, bool):
            self.add_error("Browser headless setting must be a boolean")
        
        # Timeout validations
        page_load_timeout = browser_config.get('page_load_timeout')
        if page_load_timeout is not None and (not isinstance(page_load_timeout, (int, float)) or page_load_timeout <= 0):
            self.add_error("Browser page_load_timeout must be a positive number")
        
        # AgentCore configuration validation
        agentcore_config = config.get('agentcore', {})
        
        endpoint = agentcore_config.get('browser_tool_endpoint')
        if endpoint and not self._is_valid_url(endpoint):
            self.add_error(f"Invalid AgentCore browser tool endpoint URL: {endpoint}")
        
        timeout = agentcore_config.get('timeout')
        if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
            self.add_error("AgentCore timeout must be a positive number")
        
        max_retries = agentcore_config.get('max_retries')
        if max_retries is not None and (not isinstance(max_retries, int) or max_retries < 0):
            self.add_error("AgentCore max_retries must be a non-negative integer")
        
        # LlamaIndex configuration validation
        llamaindex_config = config.get('llamaindex', {})
        
        temperature = llamaindex_config.get('temperature')
        if temperature is not None and (not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2):
            self.add_error("LlamaIndex temperature must be a number between 0 and 2")
        
        max_tokens = llamaindex_config.get('max_tokens')
        if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens <= 0):
            self.add_error("LlamaIndex max_tokens must be a positive integer")
    
    def _is_valid_aws_region(self, region: str) -> bool:
        """Check if AWS region format is valid."""
        import re
        # Basic AWS region format validation
        pattern = r'^[a-z]{2}-[a-z]+-\d+$'
        return bool(re.match(pattern, region))
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL format is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def validate_aws_credentials(self, config: Dict[str, Any]) -> bool:
        """Validate AWS credentials and permissions."""
        self.log("Validating AWS credentials...")
        
        try:
            aws_config = config.get('aws', {})
            region = aws_config.get('region', 'us-east-1')
            
            # Create session based on configuration
            if 'credentials' in aws_config:
                creds = aws_config['credentials']
                session = boto3.Session(
                    aws_access_key_id=creds.get('access_key_id'),
                    aws_secret_access_key=creds.get('secret_access_key'),
                    aws_session_token=creds.get('session_token'),
                    region_name=region
                )
            else:
                profile = aws_config.get('profile', 'default')
                session = boto3.Session(
                    profile_name=profile,
                    region_name=region
                )
            
            # Test credentials with STS
            sts = session.client('sts')
            identity = sts.get_caller_identity()
            
            self.log(f"AWS credentials valid for account: {identity['Account']}")
            self.log(f"User/Role ARN: {identity['Arn']}")
            
            # Test required service access
            self._test_bedrock_access(session, region)
            self._test_cloudwatch_access(session, region)
            
            return True
            
        except NoCredentialsError:
            self.add_error("AWS credentials not found. Please configure credentials.")
            return False
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'InvalidUserID.NotFound':
                self.add_error("AWS credentials are invalid")
            elif error_code == 'AccessDenied':
                self.add_error("AWS credentials do not have sufficient permissions")
            else:
                self.add_error(f"AWS credentials validation failed: {e}")
            return False
        except Exception as e:
            self.add_error(f"Unexpected error validating AWS credentials: {e}")
            return False
    
    def _test_bedrock_access(self, session: boto3.Session, region: str) -> None:
        """Test Bedrock service access."""
        try:
            bedrock = session.client('bedrock', region_name=region)
            bedrock.list_foundation_models()
            self.log("Bedrock access confirmed")
        except ClientError as e:
            if e.response['Error']['Code'] == 'AccessDenied':
                self.add_warning("Bedrock access denied - may need additional permissions")
            else:
                self.add_warning(f"Bedrock access test failed: {e}")
        except Exception as e:
            self.add_warning(f"Could not test Bedrock access: {e}")
    
    def _test_cloudwatch_access(self, session: boto3.Session, region: str) -> None:
        """Test CloudWatch service access."""
        try:
            cloudwatch = session.client('cloudwatch', region_name=region)
            cloudwatch.list_metrics(MaxRecords=1)
            self.log("CloudWatch access confirmed")
        except ClientError as e:
            if e.response['Error']['Code'] == 'AccessDenied':
                self.add_warning("CloudWatch access denied - monitoring may not work")
            else:
                self.add_warning(f"CloudWatch access test failed: {e}")
        except Exception as e:
            self.add_warning(f"Could not test CloudWatch access: {e}")
    
    async def validate_agentcore_connectivity(self, config: Dict[str, Any]) -> bool:
        """Validate AgentCore service connectivity."""
        self.log("Validating AgentCore connectivity...")
        
        agentcore_config = config.get('agentcore', {})
        endpoint = agentcore_config.get('browser_tool_endpoint')
        
        if not endpoint:
            self.add_error("AgentCore browser tool endpoint not configured")
            return False
        
        timeout = agentcore_config.get('timeout', 30)
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                # Test basic connectivity
                try:
                    async with session.get(f"{endpoint}/health") as response:
                        if response.status < 500:
                            self.log("AgentCore endpoint is accessible")
                            return True
                        else:
                            self.add_error(f"AgentCore endpoint returned status {response.status}")
                            return False
                except aiohttp.ClientError as e:
                    self.add_error(f"AgentCore endpoint connectivity failed: {e}")
                    return False
                
        except Exception as e:
            self.add_error(f"Failed to test AgentCore connectivity: {e}")
            return False
    
    def validate_llm_models(self, config: Dict[str, Any]) -> bool:
        """Validate LLM model availability."""
        self.log("Validating LLM model availability...")
        
        llamaindex_config = config.get('llamaindex', {})
        llm_model = llamaindex_config.get('llm_model')
        vision_model = llamaindex_config.get('vision_model')
        
        if not llm_model:
            self.add_error("LLM model not configured")
            return False
        
        if not vision_model:
            self.add_error("Vision model not configured")
            return False
        
        try:
            aws_config = config.get('aws', {})
            region = aws_config.get('region', 'us-east-1')
            
            session = boto3.Session(region_name=region)
            bedrock = session.client('bedrock', region_name=region)
            
            # Get available models
            response = bedrock.list_foundation_models()
            available_models = [model['modelId'] for model in response['modelSummaries']]
            
            # Check if configured models are available
            if llm_model not in available_models:
                self.add_error(f"LLM model '{llm_model}' is not available in region {region}")
                return False
            else:
                self.log(f"LLM model '{llm_model}' is available")
            
            if vision_model not in available_models:
                self.add_error(f"Vision model '{vision_model}' is not available in region {region}")
                return False
            else:
                self.log(f"Vision model '{vision_model}' is available")
            
            return True
            
        except Exception as e:
            self.add_warning(f"Could not validate model availability: {e}")
            return True  # Don't fail validation if we can't check
    
    def validate_security_settings(self, config: Dict[str, Any]) -> bool:
        """Validate security configuration."""
        self.log("Validating security settings...")
        
        security_config = config.get('security', {})
        
        # Check if security is properly configured
        if not security_config:
            self.add_warning("No security configuration found")
            return True
        
        # Validate encryption settings
        encryption_config = security_config.get('encryption', {})
        if encryption_config.get('enabled', False):
            algorithm = encryption_config.get('algorithm')
            if not algorithm:
                self.add_error("Encryption enabled but no algorithm specified")
                return False
            elif algorithm not in ['AES-256-GCM', 'AES-256-CBC']:
                self.add_warning(f"Encryption algorithm '{algorithm}' may not be secure")
        
        # Validate session settings
        session_config = security_config.get('session', {})
        max_duration = session_config.get('max_duration')
        if max_duration and (not isinstance(max_duration, int) or max_duration <= 0):
            self.add_error("Session max_duration must be a positive integer")
            return False
        
        idle_timeout = session_config.get('idle_timeout')
        if idle_timeout and (not isinstance(idle_timeout, int) or idle_timeout <= 0):
            self.add_error("Session idle_timeout must be a positive integer")
            return False
        
        self.log("Security settings validation passed")
        return True
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate validation report."""
        return {
            'validation_status': 'PASSED' if not self.errors else 'FAILED',
            'errors': self.errors,
            'warnings': self.warnings,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings)
        }
    
    async def validate_config_file(self, config_path: Path) -> bool:
        """Validate complete configuration file."""
        self.log(f"Validating configuration file: {config_path}")
        
        # Load configuration
        config = self.load_config(config_path)
        if not config:
            return False
        
        # Run all validations
        validations = [
            self.validate_schema(config),
            self.validate_aws_credentials(config),
            await self.validate_agentcore_connectivity(config),
            self.validate_llm_models(config),
            self.validate_security_settings(config)
        ]
        
        # Check if all validations passed
        all_passed = all(validations)
        
        # Generate report
        report = self.generate_validation_report()
        
        # Print summary
        self._print_validation_summary(config_path, report)
        
        return all_passed
    
    def _print_validation_summary(self, config_path: Path, report: Dict[str, Any]) -> None:
        """Print validation summary."""
        print("\n" + "="*60)
        print("CONFIGURATION VALIDATION REPORT")
        print("="*60)
        print(f"\nConfiguration file: {config_path}")
        print(f"Status: {report['validation_status']}")
        print(f"Errors: {report['error_count']}")
        print(f"Warnings: {report['warning_count']}")
        
        if report['errors']:
            print("\nERRORS:")
            for error in report['errors']:
                print(f"  ❌ {error}")
        
        if report['warnings']:
            print("\nWARNINGS:")
            for warning in report['warnings']:
                print(f"  ⚠️  {warning}")
        
        if report['validation_status'] == 'PASSED':
            print("\n✅ Configuration validation passed!")
        else:
            print("\n❌ Configuration validation failed!")
            print("Please fix the errors above before deploying.")
        
        print("="*60)

async def main():
    """Main validation entry point."""
    parser = argparse.ArgumentParser(
        description="Validate llamaindex-agentcore-browser-integration configuration"
    )
    parser.add_argument(
        "config_file",
        nargs="?",
        default="config.yaml",
        help="Configuration file to validate (default: config.yaml)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config_file)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    
    validator = ConfigValidator(verbose=args.verbose)
    success = await validator.validate_config_file(config_path)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())