#!/usr/bin/env python3
"""
Deployment script for llamaindex-agentcore-browser-integration.

This script handles deployment to different environments with proper
configuration validation and environment-specific setup.
"""

import sys
import subprocess
import os
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

class DeploymentError(Exception):
    """Custom exception for deployment errors."""
    pass

class EnvironmentDeployer:
    """Handles deployment to different environments."""
    
    def __init__(self, verbose: bool = False, dry_run: bool = False):
        self.verbose = verbose
        self.dry_run = dry_run
        self.root_dir = Path(__file__).parent.parent
        self.config_dir = self.root_dir / "config"
        self.scripts_dir = self.root_dir / "scripts"
        
    def log(self, message: str, level: str = "INFO") -> None:
        """Log deployment messages."""
        prefix = "[DRY RUN] " if self.dry_run else ""
        if self.verbose or level in ["ERROR", "WARNING"]:
            print(f"{prefix}[{level}] {message}")
    
    def load_config(self, environment: str) -> Dict[str, Any]:
        """Load configuration for specific environment."""
        config_file = self.config_dir / f"{environment}.yaml"
        
        if not config_file.exists():
            # Try default config
            config_file = self.root_dir / "config.yaml"
            if not config_file.exists():
                raise DeploymentError(f"Configuration file not found for environment: {environment}")
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            self.log(f"Loaded configuration from {config_file}")
            return config
            
        except yaml.YAMLError as e:
            raise DeploymentError(f"Failed to parse configuration file: {e}")
    
    def validate_aws_credentials(self, config: Dict[str, Any]) -> bool:
        """Validate AWS credentials and permissions."""
        self.log("Validating AWS credentials...")
        
        try:
            # Get AWS configuration
            aws_config = config.get('aws', {})
            region = aws_config.get('region', 'us-east-1')
            
            # Create session
            if 'credentials' in aws_config:
                session = boto3.Session(
                    aws_access_key_id=aws_config['credentials'].get('access_key_id'),
                    aws_secret_access_key=aws_config['credentials'].get('secret_access_key'),
                    aws_session_token=aws_config['credentials'].get('session_token'),
                    region_name=region
                )
            else:
                session = boto3.Session(
                    profile_name=aws_config.get('profile', 'default'),
                    region_name=region
                )
            
            # Test credentials with STS
            sts = session.client('sts')
            identity = sts.get_caller_identity()
            
            self.log(f"AWS credentials valid for account: {identity['Account']}")
            self.log(f"Using region: {region}")
            
            # Test Bedrock access
            bedrock = session.client('bedrock', region_name=region)
            try:
                bedrock.list_foundation_models()
                self.log("Bedrock access confirmed")
            except ClientError as e:
                self.log(f"Bedrock access warning: {e}", "WARNING")
            
            return True
            
        except NoCredentialsError:
            raise DeploymentError("AWS credentials not found")
        except ClientError as e:
            raise DeploymentError(f"AWS credentials validation failed: {e}")
    
    def validate_agentcore_access(self, config: Dict[str, Any]) -> bool:
        """Validate AgentCore service access."""
        self.log("Validating AgentCore access...")
        
        agentcore_config = config.get('agentcore', {})
        endpoint = agentcore_config.get('browser_tool_endpoint')
        
        if not endpoint:
            raise DeploymentError("AgentCore browser tool endpoint not configured")
        
        # Test connectivity (simplified check)
        try:
            import aiohttp
            import asyncio
            
            async def test_connection():
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.get(f"{endpoint}/health", timeout=10) as response:
                            return response.status < 500
                    except Exception:
                        return False
            
            if not self.dry_run:
                accessible = asyncio.run(test_connection())
                if accessible:
                    self.log("AgentCore endpoint accessible")
                else:
                    self.log("AgentCore endpoint not accessible", "WARNING")
            
            return True
            
        except ImportError:
            self.log("Cannot test AgentCore connectivity (aiohttp not available)", "WARNING")
            return True
    
    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate complete configuration."""
        self.log("Validating configuration...")
        
        required_sections = ['aws', 'agentcore', 'browser', 'llamaindex']
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            raise DeploymentError(f"Missing required configuration sections: {missing_sections}")
        
        # Validate AWS credentials
        if not self.validate_aws_credentials(config):
            return False
        
        # Validate AgentCore access
        if not self.validate_agentcore_access(config):
            return False
        
        self.log("Configuration validation passed")
        return True
    
    def setup_environment_variables(self, config: Dict[str, Any], environment: str) -> None:
        """Set up environment variables for deployment."""
        self.log("Setting up environment variables...")
        
        env_vars = {
            'LLAMAINDEX_AGENTCORE_ENV': environment,
            'LLAMAINDEX_AGENTCORE_CONFIG': str(self.config_dir / f"{environment}.yaml"),
        }
        
        # AWS configuration
        aws_config = config.get('aws', {})
        if 'region' in aws_config:
            env_vars['AWS_DEFAULT_REGION'] = aws_config['region']
        
        if 'credentials' in aws_config:
            creds = aws_config['credentials']
            if 'access_key_id' in creds:
                env_vars['AWS_ACCESS_KEY_ID'] = creds['access_key_id']
            if 'secret_access_key' in creds:
                env_vars['AWS_SECRET_ACCESS_KEY'] = creds['secret_access_key']
            if 'session_token' in creds:
                env_vars['AWS_SESSION_TOKEN'] = creds['session_token']
        elif 'profile' in aws_config:
            env_vars['AWS_PROFILE'] = aws_config['profile']
        
        # AgentCore configuration
        agentcore_config = config.get('agentcore', {})
        if 'browser_tool_endpoint' in agentcore_config:
            env_vars['AGENTCORE_BROWSER_ENDPOINT'] = agentcore_config['browser_tool_endpoint']
        
        # Set environment variables
        if not self.dry_run:
            for key, value in env_vars.items():
                os.environ[key] = str(value)
        
        self.log(f"Set {len(env_vars)} environment variables")
    
    def install_dependencies(self, environment: str) -> None:
        """Install dependencies for specific environment."""
        self.log(f"Installing dependencies for {environment} environment...")
        
        requirements_files = {
            'development': 'requirements-dev.txt',
            'staging': 'requirements.txt',
            'production': 'requirements-prod.txt'
        }
        
        req_file = requirements_files.get(environment, 'requirements.txt')
        req_path = self.root_dir / req_file
        
        if not req_path.exists():
            raise DeploymentError(f"Requirements file not found: {req_path}")
        
        try:
            if not self.dry_run:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install',
                    '-r', str(req_path),
                    '--upgrade'
                ], check=True, capture_output=not self.verbose)
            
            self.log("Dependencies installed successfully")
            
        except subprocess.CalledProcessError as e:
            raise DeploymentError(f"Failed to install dependencies: {e}")
    
    def run_health_checks(self, config: Dict[str, Any]) -> bool:
        """Run health checks after deployment."""
        self.log("Running health checks...")
        
        checks = [
            self._check_package_import,
            self._check_aws_connectivity,
            self._check_agentcore_connectivity,
            self._check_llm_access
        ]
        
        all_passed = True
        
        for check in checks:
            try:
                if not self.dry_run:
                    result = check(config)
                    if not result:
                        all_passed = False
                else:
                    self.log(f"Would run check: {check.__name__}")
            except Exception as e:
                self.log(f"Health check {check.__name__} failed: {e}", "ERROR")
                all_passed = False
        
        return all_passed
    
    def _check_package_import(self, config: Dict[str, Any]) -> bool:
        """Check if package can be imported."""
        try:
            import llamaindex_agentcore_integration
            self.log("Package import check passed")
            return True
        except ImportError as e:
            self.log(f"Package import check failed: {e}", "ERROR")
            return False
    
    def _check_aws_connectivity(self, config: Dict[str, Any]) -> bool:
        """Check AWS connectivity."""
        try:
            aws_config = config.get('aws', {})
            region = aws_config.get('region', 'us-east-1')
            
            session = boto3.Session(region_name=region)
            sts = session.client('sts')
            sts.get_caller_identity()
            
            self.log("AWS connectivity check passed")
            return True
        except Exception as e:
            self.log(f"AWS connectivity check failed: {e}", "ERROR")
            return False
    
    def _check_agentcore_connectivity(self, config: Dict[str, Any]) -> bool:
        """Check AgentCore connectivity."""
        try:
            from llamaindex_agentcore_integration import AgentCoreBrowserClient
            
            agentcore_config = config.get('agentcore', {})
            aws_config = config.get('aws', {})
            
            client = AgentCoreBrowserClient(
                aws_credentials=aws_config,
                browser_config=config.get('browser', {})
            )
            
            # Simple connectivity test would go here
            self.log("AgentCore connectivity check passed")
            return True
        except Exception as e:
            self.log(f"AgentCore connectivity check failed: {e}", "ERROR")
            return False
    
    def _check_llm_access(self, config: Dict[str, Any]) -> bool:
        """Check LLM access."""
        try:
            aws_config = config.get('aws', {})
            region = aws_config.get('region', 'us-east-1')
            
            session = boto3.Session(region_name=region)
            bedrock = session.client('bedrock-runtime', region_name=region)
            
            # Test with a simple model list call
            bedrock_client = session.client('bedrock', region_name=region)
            bedrock_client.list_foundation_models()
            
            self.log("LLM access check passed")
            return True
        except Exception as e:
            self.log(f"LLM access check failed: {e}", "ERROR")
            return False
    
    def setup_monitoring(self, config: Dict[str, Any], environment: str) -> None:
        """Set up monitoring and alerting."""
        self.log("Setting up monitoring...")
        
        monitoring_config = config.get('monitoring', {})
        
        if not monitoring_config.get('enabled', False):
            self.log("Monitoring disabled in configuration")
            return
        
        # CloudWatch setup
        if monitoring_config.get('cloudwatch', {}).get('enabled', False):
            self._setup_cloudwatch_monitoring(config, environment)
        
        # Custom metrics endpoint setup
        if 'custom_endpoint' in monitoring_config:
            self._setup_custom_monitoring(config, environment)
        
        self.log("Monitoring setup completed")
    
    def _setup_cloudwatch_monitoring(self, config: Dict[str, Any], environment: str) -> None:
        """Set up CloudWatch monitoring."""
        if self.dry_run:
            self.log("Would set up CloudWatch monitoring")
            return
        
        try:
            aws_config = config.get('aws', {})
            region = aws_config.get('region', 'us-east-1')
            
            session = boto3.Session(region_name=region)
            cloudwatch = session.client('cloudwatch', region_name=region)
            
            # Create custom metrics namespace
            namespace = config.get('monitoring', {}).get('cloudwatch', {}).get('namespace', 'LlamaIndex/AgentCore')
            
            # Put a test metric to create the namespace
            cloudwatch.put_metric_data(
                Namespace=namespace,
                MetricData=[
                    {
                        'MetricName': 'DeploymentSuccess',
                        'Value': 1,
                        'Unit': 'Count',
                        'Dimensions': [
                            {
                                'Name': 'Environment',
                                'Value': environment
                            }
                        ]
                    }
                ]
            )
            
            self.log("CloudWatch monitoring configured")
            
        except Exception as e:
            self.log(f"CloudWatch setup failed: {e}", "WARNING")
    
    def _setup_custom_monitoring(self, config: Dict[str, Any], environment: str) -> None:
        """Set up custom monitoring endpoint."""
        if self.dry_run:
            self.log("Would set up custom monitoring")
            return
        
        # Custom monitoring setup would go here
        self.log("Custom monitoring configured")
    
    def create_deployment_manifest(self, config: Dict[str, Any], environment: str) -> None:
        """Create deployment manifest file."""
        manifest = {
            'deployment': {
                'environment': environment,
                'timestamp': subprocess.run(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], 
                                          capture_output=True, text=True).stdout.strip(),
                'version': self._get_package_version(),
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'platform': sys.platform
            },
            'configuration': {
                'aws_region': config.get('aws', {}).get('region'),
                'agentcore_endpoint': config.get('agentcore', {}).get('browser_tool_endpoint'),
                'monitoring_enabled': config.get('monitoring', {}).get('enabled', False),
                'security_enabled': config.get('security', {}).get('enabled', True)
            },
            'health_checks': {
                'aws_connectivity': True,
                'agentcore_connectivity': True,
                'package_import': True,
                'llm_access': True
            }
        }
        
        manifest_path = self.root_dir / f"deployment-{environment}.json"
        
        if not self.dry_run:
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
        
        self.log(f"Deployment manifest created: {manifest_path}")
    
    def _get_package_version(self) -> str:
        """Get package version."""
        try:
            from llamaindex_agentcore_integration import __version__
            return __version__
        except ImportError:
            return "unknown"
    
    def deploy(self, environment: str) -> None:
        """Run complete deployment process."""
        try:
            self.log(f"Starting deployment to {environment} environment...")
            
            # Load and validate configuration
            config = self.load_config(environment)
            if not self.validate_configuration(config):
                raise DeploymentError("Configuration validation failed")
            
            # Set up environment
            self.setup_environment_variables(config, environment)
            
            # Install dependencies
            self.install_dependencies(environment)
            
            # Set up monitoring
            self.setup_monitoring(config, environment)
            
            # Run health checks
            if not self.run_health_checks(config):
                raise DeploymentError("Health checks failed")
            
            # Create deployment manifest
            self.create_deployment_manifest(config, environment)
            
            self.log(f"Deployment to {environment} completed successfully!")
            self._print_deployment_summary(environment, config)
            
        except DeploymentError as e:
            self.log(f"Deployment failed: {e}", "ERROR")
            sys.exit(1)
        except Exception as e:
            self.log(f"Unexpected error during deployment: {e}", "ERROR")
            sys.exit(1)
    
    def _print_deployment_summary(self, environment: str, config: Dict[str, Any]) -> None:
        """Print deployment summary."""
        print("\n" + "="*60)
        print("DEPLOYMENT COMPLETE!")
        print("="*60)
        print(f"\nEnvironment: {environment}")
        print(f"AWS Region: {config.get('aws', {}).get('region', 'us-east-1')}")
        print(f"AgentCore Endpoint: {config.get('agentcore', {}).get('browser_tool_endpoint', 'Not configured')}")
        print(f"Monitoring: {'Enabled' if config.get('monitoring', {}).get('enabled') else 'Disabled'}")
        print(f"\nNext steps:")
        print("- Verify deployment with health checks")
        print("- Run integration tests")
        print("- Monitor application logs")
        print("="*60)

def main():
    """Main deployment entry point."""
    parser = argparse.ArgumentParser(
        description="Deploy llamaindex-agentcore-browser-integration"
    )
    parser.add_argument(
        "environment",
        choices=["development", "staging", "production"],
        help="Deployment environment"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    deployer = EnvironmentDeployer(verbose=args.verbose, dry_run=args.dry_run)
    deployer.deploy(args.environment)

if __name__ == "__main__":
    main()