# Browser-Use with AgentCore Browser Tool - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying browser-use with Amazon Bedrock AgentCore Browser Tool in production environments for secure handling of sensitive information. The deployment leverages AgentCore's serverless infrastructure to eliminate browser farm management while maintaining enterprise-grade security.

## Prerequisites

### System Requirements

#### Development Environment
- **Python Version**: Python 3.12 or higher
- **Operating System**: Linux, macOS, or Windows with WSL2
- **Memory**: Minimum 8GB RAM, recommended 16GB
- **Storage**: Minimum 10GB free space
- **Network**: Stable internet connection with HTTPS access

#### AWS Requirements
- **AWS Account**: Active AWS account with appropriate permissions
- **AWS CLI**: Version 2.x installed and configured
- **AWS Regions**: AgentCore Browser Tool availability in target region
- **IAM Permissions**: Permissions for AgentCore Browser Tool and Bedrock services

### Required AWS Services

#### Core Services
- **Amazon Bedrock**: For LLM model access
- **AgentCore Browser Tool**: For managed browser runtime
- **AWS IAM**: For access control and authentication
- **Amazon CloudWatch**: For logging and monitoring
- **AWS KMS**: For encryption key management

#### Optional Services
- **Amazon VPC**: For network isolation (recommended for production)
- **AWS Lambda**: For serverless deployment patterns
- **Amazon S3**: For artifact storage and session recordings
- **Amazon DynamoDB**: For session state management

## Installation and Setup

### 1. Environment Setup

#### Create Python Virtual Environment
```bash
# Create virtual environment with Python 3.12
python3.12 -m venv browser-use-agentcore-env

# Activate virtual environment
# On Linux/macOS:
source browser-use-agentcore-env/bin/activate
# On Windows:
browser-use-agentcore-env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

#### Install Dependencies
```bash
# Install from requirements.txt
pip install -r requirements.txt

# Verify installation
python -c "import browser_use; print('Browser-Use installed successfully')"
python -c "import bedrock_agentcore; print('AgentCore SDK installed successfully')"
```

### 2. AWS Configuration

#### Configure AWS Credentials
```bash
# Configure AWS CLI
aws configure

# Verify configuration
aws sts get-caller-identity

# Test AgentCore Browser Tool access
aws bedrock-agentcore list-browser-sessions --region us-east-1
```

#### Set Up IAM Permissions
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "bedrock-agentcore:CreateBrowserSession",
        "bedrock-agentcore:TerminateBrowserSession",
        "bedrock-agentcore:GetBrowserSession",
        "bedrock-agentcore:ListBrowserSessions"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
```

### 3. Environment Configuration

#### Create Environment File
```bash
# Copy example environment file
cp .env.example .env

# Edit environment file with your configuration
nano .env
```

#### Environment Variables
```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_PROFILE=default

# Amazon Bedrock Configuration
BEDROCK_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0

# AgentCore Browser Tool Configuration
AGENTCORE_BROWSER_TOOL_REGION=us-east-1
AGENTCORE_SESSION_TIMEOUT=300
AGENTCORE_ENABLE_LIVE_VIEW=true
AGENTCORE_ENABLE_SESSION_REPLAY=true

# Browser-Use Configuration
BROWSER_USE_HEADLESS=true
BROWSER_USE_TIMEOUT=30000
BROWSER_USE_VIEWPORT_WIDTH=1920
BROWSER_USE_VIEWPORT_HEIGHT=1080

# Security Configuration
ENABLE_PII_MASKING=true
ENABLE_CREDENTIAL_PROTECTION=true
COMPLIANCE_MODE=enterprise
AUDIT_LEVEL=detailed

# Production Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
ENABLE_STRUCTURED_LOGGING=true
```

## Deployment Architectures

### 1. Development Deployment

#### Local Development Setup
```python
# development_deployment.py
import asyncio
from browser_use import Agent
from bedrock_agentcore.tools.browser_client import BrowserClient
from tools.browseruse_agentcore_session_helpers import BrowserUseAgentCoreSessionManager

async def development_setup():
    """Set up development environment."""
    
    # Initialize session manager
    session_manager = BrowserUseAgentCoreSessionManager(region='us-east-1')
    
    # Create development session
    session_details = await session_manager.create_secure_session()
    
    # Create browser-use agent
    agent = Agent(
        task="Development testing with sensitive data handling",
        llm=get_bedrock_model(),
        browser_session=session_details['browser_session']
    )
    
    return agent, session_manager

# Run development setup
if __name__ == "__main__":
    asyncio.run(development_setup())
```

### 2. Production Deployment

#### Serverless Lambda Deployment
```python
# lambda_deployment.py
import json
import asyncio
from typing import Dict, Any

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda handler for browser-use with AgentCore."""
    
    try:
        # Extract task from event
        task = event.get('task', 'Default sensitive data processing task')
        
        # Run browser-use operation
        result = asyncio.run(process_sensitive_data_task(task))
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'result': result
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'success': False,
                'error': str(e)
            })
        }

async def process_sensitive_data_task(task: str) -> Dict[str, Any]:
    """Process sensitive data task using browser-use and AgentCore."""
    
    from tools.browseruse_agentcore_session_helpers import BrowserUseAgentCoreSessionManager
    from tools.browseruse_pii_utils import SensitiveDataHandler
    
    session_manager = None
    
    try:
        # Create secure session
        session_manager = BrowserUseAgentCoreSessionManager()
        session_details = await session_manager.create_secure_session()
        
        # Initialize sensitive data handler
        data_handler = SensitiveDataHandler()
        
        # Create and run browser-use agent
        agent = Agent(
            task=task,
            llm=get_bedrock_model(),
            browser_session=session_details['browser_session']
        )
        
        # Execute task with PII protection
        result = await agent.run()
        
        # Mask any PII in result
        masked_result = data_handler.mask_pii(str(result))
        
        return {
            'task_completed': True,
            'result': masked_result,
            'session_id': session_details['session_id']
        }
        
    finally:
        # Cleanup session
        if session_manager:
            await session_manager.cleanup_session()
```

#### Container Deployment
```dockerfile
# Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Chrome for Playwright
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers
RUN playwright install

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV BROWSER_USE_HEADLESS=true

# Expose port for health checks
EXPOSE 8080

# Run application
CMD ["python", "main.py"]
```

#### Kubernetes Deployment
```yaml
# kubernetes-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: browser-use-agentcore
  labels:
    app: browser-use-agentcore
spec:
  replicas: 3
  selector:
    matchLabels:
      app: browser-use-agentcore
  template:
    metadata:
      labels:
        app: browser-use-agentcore
    spec:
      containers:
      - name: browser-use-agentcore
        image: browser-use-agentcore:latest
        ports:
        - containerPort: 8080
        env:
        - name: AWS_REGION
          value: "us-east-1"
        - name: BEDROCK_MODEL_ID
          value: "anthropic.claude-3-5-sonnet-20241022-v2:0"
        - name: AGENTCORE_BROWSER_TOOL_REGION
          value: "us-east-1"
        - name: ENABLE_PII_MASKING
          value: "true"
        - name: COMPLIANCE_MODE
          value: "enterprise"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: browser-use-agentcore-service
spec:
  selector:
    app: browser-use-agentcore
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: LoadBalancer
```

## Security Configuration

### 1. Network Security

#### VPC Configuration
```yaml
# vpc-configuration.yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'VPC configuration for browser-use with AgentCore'

Resources:
  BrowserUseVPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: browser-use-agentcore-vpc

  PrivateSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref BrowserUseVPC
      CidrBlock: 10.0.1.0/24
      AvailabilityZone: !Select [0, !GetAZs '']
      Tags:
        - Key: Name
          Value: browser-use-private-subnet-1

  PrivateSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref BrowserUseVPC
      CidrBlock: 10.0.2.0/24
      AvailabilityZone: !Select [1, !GetAZs '']
      Tags:
        - Key: Name
          Value: browser-use-private-subnet-2

  SecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for browser-use with AgentCore
      VpcId: !Ref BrowserUseVPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 443
          ToPort: 443
          CidrIp: 0.0.0.0/0
          Description: HTTPS traffic
      SecurityGroupEgress:
        - IpProtocol: tcp
          FromPort: 443
          ToPort: 443
          CidrIp: 0.0.0.0/0
          Description: HTTPS outbound
```

### 2. Encryption Configuration

#### KMS Key Setup
```python
# encryption_setup.py
import boto3
from typing import Dict, Any

def create_kms_key() -> Dict[str, Any]:
    """Create KMS key for browser-use encryption."""
    
    kms_client = boto3.client('kms')
    
    key_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "Enable IAM User Permissions",
                "Effect": "Allow",
                "Principal": {
                    "AWS": f"arn:aws:iam::{boto3.client('sts').get_caller_identity()['Account']}:root"
                },
                "Action": "kms:*",
                "Resource": "*"
            },
            {
                "Sid": "Allow browser-use service",
                "Effect": "Allow",
                "Principal": {
                    "Service": "bedrock-agentcore.amazonaws.com"
                },
                "Action": [
                    "kms:Encrypt",
                    "kms:Decrypt",
                    "kms:ReEncrypt*",
                    "kms:GenerateDataKey*",
                    "kms:DescribeKey"
                ],
                "Resource": "*"
            }
        ]
    }
    
    response = kms_client.create_key(
        Policy=json.dumps(key_policy),
        Description='KMS key for browser-use with AgentCore sensitive data encryption',
        Usage='ENCRYPT_DECRYPT',
        KeySpec='SYMMETRIC_DEFAULT'
    )
    
    return response['KeyMetadata']
```

## Monitoring and Observability

### 1. CloudWatch Configuration

#### Log Groups Setup
```python
# cloudwatch_setup.py
import boto3
from typing import List

def create_log_groups() -> List[str]:
    """Create CloudWatch log groups for browser-use monitoring."""
    
    logs_client = boto3.client('logs')
    
    log_groups = [
        '/aws/browser-use/application',
        '/aws/browser-use/security',
        '/aws/browser-use/compliance',
        '/aws/browser-use/performance'
    ]
    
    created_groups = []
    
    for log_group in log_groups:
        try:
            logs_client.create_log_group(
                logGroupName=log_group,
                retentionInDays=30
            )
            created_groups.append(log_group)
        except logs_client.exceptions.ResourceAlreadyExistsException:
            print(f"Log group {log_group} already exists")
            created_groups.append(log_group)
    
    return created_groups
```

#### Custom Metrics
```python
# custom_metrics.py
import boto3
from datetime import datetime
from typing import Dict, Any

class BrowserUseMetrics:
    """Custom metrics for browser-use operations."""
    
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch')
    
    def put_session_metric(
        self, 
        metric_name: str, 
        value: float, 
        unit: str = 'Count',
        dimensions: Dict[str, str] = None
    ):
        """Put custom metric to CloudWatch."""
        
        if dimensions is None:
            dimensions = {}
        
        self.cloudwatch.put_metric_data(
            Namespace='BrowserUse/AgentCore',
            MetricData=[
                {
                    'MetricName': metric_name,
                    'Value': value,
                    'Unit': unit,
                    'Timestamp': datetime.utcnow(),
                    'Dimensions': [
                        {
                            'Name': key,
                            'Value': value
                        } for key, value in dimensions.items()
                    ]
                }
            ]
        )
    
    def put_pii_detection_metric(self, pii_count: int, session_id: str):
        """Put PII detection metric."""
        self.put_session_metric(
            metric_name='PIIDetections',
            value=pii_count,
            dimensions={'SessionId': session_id}
        )
    
    def put_compliance_violation_metric(self, violation_count: int, framework: str):
        """Put compliance violation metric."""
        self.put_session_metric(
            metric_name='ComplianceViolations',
            value=violation_count,
            dimensions={'Framework': framework}
        )
```

### 2. Dashboard Configuration

#### CloudWatch Dashboard
```json
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["BrowserUse/AgentCore", "SessionsCreated"],
          [".", "SessionsCompleted"],
          [".", "SessionsFailed"]
        ],
        "period": 300,
        "stat": "Sum",
        "region": "us-east-1",
        "title": "Browser-Use Sessions"
      }
    },
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["BrowserUse/AgentCore", "PIIDetections"],
          [".", "ComplianceViolations"]
        ],
        "period": 300,
        "stat": "Sum",
        "region": "us-east-1",
        "title": "Security Metrics"
      }
    },
    {
      "type": "log",
      "properties": {
        "query": "SOURCE '/aws/browser-use/security' | fields @timestamp, level, message\n| filter level = \"ERROR\"\n| sort @timestamp desc\n| limit 100",
        "region": "us-east-1",
        "title": "Security Errors"
      }
    }
  ]
}
```

## Performance Optimization

### 1. Session Management Optimization

#### Connection Pooling
```python
# connection_pool.py
import asyncio
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class PooledSession:
    """Pooled AgentCore session."""
    session_id: str
    ws_url: str
    headers: Dict[str, str]
    created_at: float
    last_used: float
    in_use: bool = False

class AgentCoreSessionPool:
    """Connection pool for AgentCore sessions."""
    
    def __init__(self, max_sessions: int = 10, session_timeout: int = 300):
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
        self.available_sessions: List[PooledSession] = []
        self.in_use_sessions: Dict[str, PooledSession] = {}
        self._lock = asyncio.Lock()
    
    async def get_session(self) -> PooledSession:
        """Get a session from the pool."""
        async with self._lock:
            # Try to reuse an available session
            if self.available_sessions:
                session = self.available_sessions.pop(0)
                session.in_use = True
                session.last_used = time.time()
                self.in_use_sessions[session.session_id] = session
                return session
            
            # Create new session if under limit
            if len(self.in_use_sessions) < self.max_sessions:
                session = await self._create_new_session()
                self.in_use_sessions[session.session_id] = session
                return session
            
            # Wait for available session
            raise Exception("No available sessions in pool")
    
    async def return_session(self, session_id: str):
        """Return a session to the pool."""
        async with self._lock:
            if session_id in self.in_use_sessions:
                session = self.in_use_sessions.pop(session_id)
                session.in_use = False
                session.last_used = time.time()
                self.available_sessions.append(session)
```

### 2. Caching Strategy

#### Result Caching
```python
# result_cache.py
import json
import hashlib
from typing import Any, Optional
import redis

class BrowserUseResultCache:
    """Cache for browser-use operation results."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.default_ttl = 3600  # 1 hour
    
    def _generate_cache_key(self, task: str, url: str) -> str:
        """Generate cache key for task and URL."""
        content = f"{task}:{url}"
        return f"browseruse:{hashlib.md5(content.encode()).hexdigest()}"
    
    def get_cached_result(self, task: str, url: str) -> Optional[Dict[str, Any]]:
        """Get cached result for task and URL."""
        cache_key = self._generate_cache_key(task, url)
        
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            print(f"Cache retrieval error: {e}")
        
        return None
    
    def cache_result(
        self, 
        task: str, 
        url: str, 
        result: Dict[str, Any], 
        ttl: int = None
    ):
        """Cache result for task and URL."""
        cache_key = self._generate_cache_key(task, url)
        ttl = ttl or self.default_ttl
        
        try:
            self.redis_client.setex(
                cache_key, 
                ttl, 
                json.dumps(result)
            )
        except Exception as e:
            print(f"Cache storage error: {e}")
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Session Creation Failures
```python
# troubleshooting.py
import logging
from typing import Dict, Any

class BrowserUseTroubleshooter:
    """Troubleshooting utilities for browser-use with AgentCore."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def diagnose_session_creation_failure(self, error: Exception) -> Dict[str, Any]:
        """Diagnose session creation failures."""
        
        diagnosis = {
            "issue": "Session Creation Failure",
            "error": str(error),
            "possible_causes": [],
            "solutions": []
        }
        
        error_str = str(error).lower()
        
        if "authentication" in error_str or "credentials" in error_str:
            diagnosis["possible_causes"].append("Invalid AWS credentials")
            diagnosis["solutions"].append("Verify AWS credentials and permissions")
            
        elif "region" in error_str:
            diagnosis["possible_causes"].append("Invalid or unsupported region")
            diagnosis["solutions"].append("Check AgentCore Browser Tool availability in region")
            
        elif "quota" in error_str or "limit" in error_str:
            diagnosis["possible_causes"].append("Service quota exceeded")
            diagnosis["solutions"].append("Request quota increase or implement session pooling")
            
        elif "network" in error_str or "timeout" in error_str:
            diagnosis["possible_causes"].append("Network connectivity issues")
            diagnosis["solutions"].append("Check network connectivity and firewall settings")
        
        return diagnosis
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check system health and configuration."""
        
        health_check = {
            "status": "healthy",
            "checks": {},
            "issues": []
        }
        
        # Check AWS credentials
        try:
            import boto3
            sts = boto3.client('sts')
            sts.get_caller_identity()
            health_check["checks"]["aws_credentials"] = "OK"
        except Exception as e:
            health_check["checks"]["aws_credentials"] = "FAILED"
            health_check["issues"].append(f"AWS credentials: {str(e)}")
            health_check["status"] = "unhealthy"
        
        # Check browser-use installation
        try:
            import browser_use
            health_check["checks"]["browser_use"] = "OK"
        except ImportError as e:
            health_check["checks"]["browser_use"] = "FAILED"
            health_check["issues"].append(f"Browser-use installation: {str(e)}")
            health_check["status"] = "unhealthy"
        
        # Check AgentCore SDK
        try:
            import bedrock_agentcore
            health_check["checks"]["agentcore_sdk"] = "OK"
        except ImportError as e:
            health_check["checks"]["agentcore_sdk"] = "FAILED"
            health_check["issues"].append(f"AgentCore SDK installation: {str(e)}")
            health_check["status"] = "unhealthy"
        
        return health_check
```

### Debugging Tools

#### Debug Mode Configuration
```python
# debug_config.py
import logging
import os
from typing import Dict, Any

def setup_debug_logging() -> Dict[str, Any]:
    """Set up debug logging configuration."""
    
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    log_format = os.getenv('LOG_FORMAT', 'json')
    
    if log_format == 'json':
        import json_logging
        json_logging.init_non_web(enable_json=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Enable debug logging for specific modules
    logging.getLogger('browser_use').setLevel(logging.DEBUG)
    logging.getLogger('bedrock_agentcore').setLevel(logging.DEBUG)
    logging.getLogger('boto3').setLevel(logging.INFO)
    logging.getLogger('botocore').setLevel(logging.INFO)
    
    return {
        "log_level": log_level,
        "log_format": log_format,
        "debug_enabled": log_level.upper() == 'DEBUG'
    }
```

## Production Checklist

### Pre-Deployment Checklist

- [ ] **Environment Configuration**
  - [ ] All environment variables configured
  - [ ] AWS credentials properly set up
  - [ ] Region availability verified
  - [ ] Network connectivity tested

- [ ] **Security Configuration**
  - [ ] IAM permissions configured
  - [ ] KMS keys created and configured
  - [ ] VPC and security groups set up
  - [ ] Encryption enabled for data in transit and at rest

- [ ] **Monitoring Setup**
  - [ ] CloudWatch log groups created
  - [ ] Custom metrics configured
  - [ ] Dashboards set up
  - [ ] Alerts configured

- [ ] **Testing Completed**
  - [ ] Unit tests passing
  - [ ] Integration tests passing
  - [ ] Security tests completed
  - [ ] Performance tests completed
  - [ ] Compliance validation completed

- [ ] **Documentation**
  - [ ] Deployment documentation updated
  - [ ] Runbooks created
  - [ ] Troubleshooting guides available
  - [ ] Security procedures documented

### Post-Deployment Checklist

- [ ] **Verification**
  - [ ] Application health checks passing
  - [ ] Session creation working
  - [ ] PII masking functioning
  - [ ] Compliance validation active

- [ ] **Monitoring**
  - [ ] Logs flowing to CloudWatch
  - [ ] Metrics being collected
  - [ ] Alerts functioning
  - [ ] Dashboards displaying data

- [ ] **Performance**
  - [ ] Response times within acceptable limits
  - [ ] Resource utilization optimal
  - [ ] Session pooling working
  - [ ] Caching effective

This deployment guide provides a comprehensive approach to deploying browser-use with AgentCore Browser Tool in production environments while maintaining security, compliance, and performance requirements.