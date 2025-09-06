#!/usr/bin/env python3
"""
Monitoring setup script for llamaindex-agentcore-browser-integration.

This script sets up CloudWatch dashboards, alarms, and custom metrics
for monitoring the integration in production.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import boto3
from botocore.exceptions import ClientError

class MonitoringSetupError(Exception):
    """Custom exception for monitoring setup errors."""
    pass

class MonitoringSetup:
    """Sets up monitoring and alerting for the integration."""
    
    def __init__(self, verbose: bool = False, dry_run: bool = False):
        self.verbose = verbose
        self.dry_run = dry_run
        self.root_dir = Path(__file__).parent.parent
        
    def log(self, message: str, level: str = "INFO") -> None:
        """Log setup messages."""
        prefix = "[DRY RUN] " if self.dry_run else ""
        if self.verbose or level in ["ERROR", "WARNING"]:
            print(f"{prefix}[{level}] {message}")
    
    def setup_cloudwatch_dashboard(self, 
                                  region: str, 
                                  namespace: str,
                                  environment: str) -> str:
        """Create CloudWatch dashboard."""
        self.log("Setting up CloudWatch dashboard...")
        
        dashboard_name = f"LlamaIndex-AgentCore-{environment}"
        
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "x": 0,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [namespace, "BrowserOperations", "Environment", environment],
                            [namespace, "CaptchaSolved", "Environment", environment],
                            [namespace, "NavigationSuccess", "Environment", environment]
                        ],
                        "period": 300,
                        "stat": "Sum",
                        "region": region,
                        "title": "Browser Operations"
                    }
                },
                {
                    "type": "metric",
                    "x": 12,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [namespace, "ErrorRate", "Environment", environment],
                            [namespace, "TimeoutErrors", "Environment", environment],
                            [namespace, "AuthenticationErrors", "Environment", environment]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": region,
                        "title": "Error Metrics"
                    }
                },
                {
                    "type": "metric",
                    "x": 0,
                    "y": 6,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [namespace, "ResponseTime", "Environment", environment],
                            [namespace, "BrowserSessionDuration", "Environment", environment]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": region,
                        "title": "Performance Metrics"
                    }
                },
                {
                    "type": "metric",
                    "x": 12,
                    "y": 6,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [namespace, "ActiveSessions", "Environment", environment],
                            [namespace, "ConcurrentOperations", "Environment", environment]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": region,
                        "title": "Resource Usage"
                    }
                },
                {
                    "type": "log",
                    "x": 0,
                    "y": 12,
                    "width": 24,
                    "height": 6,
                    "properties": {
                        "query": f"SOURCE '/aws/lambda/llamaindex-agentcore-{environment}'\n| fields @timestamp, @message\n| filter @message like /ERROR/\n| sort @timestamp desc\n| limit 100",
                        "region": region,
                        "title": "Recent Errors",
                        "view": "table"
                    }
                }
            ]
        }
        
        if not self.dry_run:
            try:
                cloudwatch = boto3.client('cloudwatch', region_name=region)
                cloudwatch.put_dashboard(
                    DashboardName=dashboard_name,
                    DashboardBody=json.dumps(dashboard_body)
                )
                self.log(f"Created CloudWatch dashboard: {dashboard_name}")
            except ClientError as e:
                raise MonitoringSetupError(f"Failed to create dashboard: {e}")
        else:
            self.log(f"Would create CloudWatch dashboard: {dashboard_name}")
        
        return dashboard_name
    
    def setup_cloudwatch_alarms(self, 
                               region: str, 
                               namespace: str,
                               environment: str,
                               sns_topic_arn: Optional[str] = None) -> List[str]:
        """Create CloudWatch alarms."""
        self.log("Setting up CloudWatch alarms...")
        
        alarms = [
            {
                "AlarmName": f"LlamaIndex-AgentCore-HighErrorRate-{environment}",
                "ComparisonOperator": "GreaterThanThreshold",
                "EvaluationPeriods": 2,
                "MetricName": "ErrorRate",
                "Namespace": namespace,
                "Period": 300,
                "Statistic": "Average",
                "Threshold": 0.05,  # 5% error rate
                "ActionsEnabled": True,
                "AlarmDescription": "High error rate detected",
                "Dimensions": [
                    {
                        "Name": "Environment",
                        "Value": environment
                    }
                ],
                "Unit": "Percent"
            },
            {
                "AlarmName": f"LlamaIndex-AgentCore-HighResponseTime-{environment}",
                "ComparisonOperator": "GreaterThanThreshold",
                "EvaluationPeriods": 3,
                "MetricName": "ResponseTime",
                "Namespace": namespace,
                "Period": 300,
                "Statistic": "Average",
                "Threshold": 5000,  # 5 seconds
                "ActionsEnabled": True,
                "AlarmDescription": "High response time detected",
                "Dimensions": [
                    {
                        "Name": "Environment",
                        "Value": environment
                    }
                ],
                "Unit": "Milliseconds"
            },
            {
                "AlarmName": f"LlamaIndex-AgentCore-NoOperations-{environment}",
                "ComparisonOperator": "LessThanThreshold",
                "EvaluationPeriods": 3,
                "MetricName": "BrowserOperations",
                "Namespace": namespace,
                "Period": 900,  # 15 minutes
                "Statistic": "Sum",
                "Threshold": 1,
                "ActionsEnabled": True,
                "AlarmDescription": "No browser operations detected",
                "Dimensions": [
                    {
                        "Name": "Environment",
                        "Value": environment
                    }
                ],
                "Unit": "Count",
                "TreatMissingData": "breaching"
            }
        ]
        
        created_alarms = []
        
        if not self.dry_run:
            try:
                cloudwatch = boto3.client('cloudwatch', region_name=region)
                
                for alarm_config in alarms:
                    if sns_topic_arn:
                        alarm_config["AlarmActions"] = [sns_topic_arn]
                        alarm_config["OKActions"] = [sns_topic_arn]
                    
                    cloudwatch.put_metric_alarm(**alarm_config)
                    created_alarms.append(alarm_config["AlarmName"])
                    self.log(f"Created alarm: {alarm_config['AlarmName']}")
                
            except ClientError as e:
                raise MonitoringSetupError(f"Failed to create alarms: {e}")
        else:
            created_alarms = [alarm["AlarmName"] for alarm in alarms]
            for alarm_name in created_alarms:
                self.log(f"Would create alarm: {alarm_name}")
        
        return created_alarms
    
    def setup_sns_topic(self, region: str, environment: str, email: Optional[str] = None) -> str:
        """Create SNS topic for alerts."""
        self.log("Setting up SNS topic for alerts...")
        
        topic_name = f"LlamaIndex-AgentCore-Alerts-{environment}"
        
        if not self.dry_run:
            try:
                sns = boto3.client('sns', region_name=region)
                
                # Create topic
                response = sns.create_topic(Name=topic_name)
                topic_arn = response['TopicArn']
                
                # Subscribe email if provided
                if email:
                    sns.subscribe(
                        TopicArn=topic_arn,
                        Protocol='email',
                        Endpoint=email
                    )
                    self.log(f"Subscribed {email} to alerts")
                
                self.log(f"Created SNS topic: {topic_name}")
                return topic_arn
                
            except ClientError as e:
                raise MonitoringSetupError(f"Failed to create SNS topic: {e}")
        else:
            self.log(f"Would create SNS topic: {topic_name}")
            return f"arn:aws:sns:{region}:123456789012:{topic_name}"
    
    def setup_log_groups(self, region: str, environment: str) -> List[str]:
        """Create CloudWatch log groups."""
        self.log("Setting up CloudWatch log groups...")
        
        log_groups = [
            f"/aws/lambda/llamaindex-agentcore-{environment}",
            f"/llamaindex-agentcore/{environment}/application",
            f"/llamaindex-agentcore/{environment}/security",
            f"/llamaindex-agentcore/{environment}/performance"
        ]
        
        created_groups = []
        
        if not self.dry_run:
            try:
                logs = boto3.client('logs', region_name=region)
                
                for log_group in log_groups:
                    try:
                        logs.create_log_group(
                            logGroupName=log_group,
                            retentionInDays=30 if environment == "development" else 90
                        )
                        created_groups.append(log_group)
                        self.log(f"Created log group: {log_group}")
                    except ClientError as e:
                        if e.response['Error']['Code'] == 'ResourceAlreadyExistsException':
                            self.log(f"Log group already exists: {log_group}")
                            created_groups.append(log_group)
                        else:
                            raise
                
            except ClientError as e:
                raise MonitoringSetupError(f"Failed to create log groups: {e}")
        else:
            created_groups = log_groups
            for log_group in log_groups:
                self.log(f"Would create log group: {log_group}")
        
        return created_groups
    
    def setup_custom_metrics(self, region: str, namespace: str) -> None:
        """Set up custom metric definitions."""
        self.log("Setting up custom metrics...")
        
        # Custom metrics are created automatically when first published
        # This method documents the metrics we'll be using
        
        metrics = [
            {
                "name": "BrowserOperations",
                "description": "Number of browser operations performed",
                "unit": "Count"
            },
            {
                "name": "CaptchaSolved",
                "description": "Number of CAPTCHAs successfully solved",
                "unit": "Count"
            },
            {
                "name": "NavigationSuccess",
                "description": "Successful page navigations",
                "unit": "Count"
            },
            {
                "name": "ErrorRate",
                "description": "Error rate percentage",
                "unit": "Percent"
            },
            {
                "name": "ResponseTime",
                "description": "Average response time",
                "unit": "Milliseconds"
            },
            {
                "name": "BrowserSessionDuration",
                "description": "Average browser session duration",
                "unit": "Seconds"
            },
            {
                "name": "ActiveSessions",
                "description": "Number of active browser sessions",
                "unit": "Count"
            },
            {
                "name": "ConcurrentOperations",
                "description": "Number of concurrent operations",
                "unit": "Count"
            }
        ]
        
        # Save metric definitions for reference
        metrics_file = self.root_dir / "monitoring" / "custom_metrics.json"
        metrics_file.parent.mkdir(exist_ok=True)
        
        if not self.dry_run:
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        self.log(f"Custom metrics documented: {len(metrics)} metrics")
    
    def generate_monitoring_config(self, 
                                 region: str,
                                 environment: str,
                                 dashboard_name: str,
                                 alarm_names: List[str],
                                 sns_topic_arn: str,
                                 log_groups: List[str]) -> None:
        """Generate monitoring configuration file."""
        config = {
            "monitoring": {
                "environment": environment,
                "region": region,
                "dashboard": {
                    "name": dashboard_name,
                    "url": f"https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}#dashboards:name={dashboard_name}"
                },
                "alarms": alarm_names,
                "sns_topic": sns_topic_arn,
                "log_groups": log_groups,
                "namespace": f"LlamaIndex/AgentCore/{environment.title()}"
            }
        }
        
        config_file = self.root_dir / "monitoring" / f"{environment}_monitoring.json"
        config_file.parent.mkdir(exist_ok=True)
        
        if not self.dry_run:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
        
        self.log(f"Monitoring configuration saved: {config_file}")
    
    def setup_monitoring(self, 
                        region: str,
                        environment: str,
                        email: Optional[str] = None) -> None:
        """Set up complete monitoring stack."""
        try:
            self.log(f"Setting up monitoring for {environment} environment...")
            
            namespace = f"LlamaIndex/AgentCore/{environment.title()}"
            
            # Set up SNS topic
            sns_topic_arn = self.setup_sns_topic(region, environment, email)
            
            # Set up log groups
            log_groups = self.setup_log_groups(region, environment)
            
            # Set up dashboard
            dashboard_name = self.setup_cloudwatch_dashboard(region, namespace, environment)
            
            # Set up alarms
            alarm_names = self.setup_cloudwatch_alarms(region, namespace, environment, sns_topic_arn)
            
            # Set up custom metrics
            self.setup_custom_metrics(region, namespace)
            
            # Generate configuration
            self.generate_monitoring_config(
                region, environment, dashboard_name, 
                alarm_names, sns_topic_arn, log_groups
            )
            
            self.log(f"Monitoring setup completed for {environment}!")
            self._print_monitoring_summary(environment, dashboard_name, len(alarm_names))
            
        except MonitoringSetupError as e:
            self.log(f"Monitoring setup failed: {e}", "ERROR")
            sys.exit(1)
        except Exception as e:
            self.log(f"Unexpected error during monitoring setup: {e}", "ERROR")
            sys.exit(1)
    
    def _print_monitoring_summary(self, environment: str, dashboard_name: str, alarm_count: int) -> None:
        """Print monitoring setup summary."""
        print("\n" + "="*60)
        print("MONITORING SETUP COMPLETE!")
        print("="*60)
        print(f"\nEnvironment: {environment}")
        print(f"Dashboard: {dashboard_name}")
        print(f"Alarms created: {alarm_count}")
        print(f"Log groups: 4")
        print(f"\nAccess your monitoring:")
        print("- CloudWatch Console: AWS Console > CloudWatch")
        print("- Dashboard: CloudWatch > Dashboards")
        print("- Alarms: CloudWatch > Alarms")
        print("- Logs: CloudWatch > Log groups")
        print("="*60)

def main():
    """Main monitoring setup entry point."""
    parser = argparse.ArgumentParser(
        description="Set up monitoring for llamaindex-agentcore-browser-integration"
    )
    parser.add_argument(
        "environment",
        choices=["development", "staging", "production"],
        help="Environment to set up monitoring for"
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region (default: us-east-1)"
    )
    parser.add_argument(
        "--email",
        help="Email address for alert notifications"
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
    
    setup = MonitoringSetup(verbose=args.verbose, dry_run=args.dry_run)
    setup.setup_monitoring(args.region, args.environment, args.email)

if __name__ == "__main__":
    main()