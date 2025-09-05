# Production Deployment Guide for Strands with AgentCore Browser Tool

## Overview

This guide provides comprehensive instructions for deploying Strands agents with AgentCore Browser Tool integration in production environments. It covers security hardening, scalability considerations, monitoring setup, and compliance requirements for enterprise deployments.

## Pre-Deployment Checklist

### 1. Infrastructure Requirements

#### AWS Account Setup
- [ ] AWS account with appropriate billing limits
- [ ] Multi-region setup for disaster recovery
- [ ] VPC configuration with proper network segmentation
- [ ] Security groups and NACLs configured
- [ ] AWS Config enabled for compliance monitoring

#### Service Quotas and Limits
- [ ] Bedrock model access enabled for all required models
- [ ] AgentCore Browser Tool service quotas increased
- [ ] CloudWatch Logs retention policies configured
- [ ] Secrets Manager quotas verified
- [ ] Lambda concurrent execution limits (if using serverless)

#### Security Infrastructure
- [ ] AWS KMS keys created for encryption
- [ ] IAM roles and policies configured with least privilege
- [ ] AWS WAF rules configured for web applications
- [ ] VPC Flow Logs enabled
- [ ] GuardDuty enabled for threat detection

### 2. Application Requirements

#### Strands Framework
- [ ] Strands SDK version compatibility verified
- [ ] Custom security tools tested and validated
- [ ] Multi-LLM configuration tested
- [ ] Performance benchmarks completed
- [ ] Security testing completed

#### Dependencies
- [ ] All Python dependencies pinned to specific versions
- [ ] Security vulnerabilities scanned and resolved
- [ ] Container images scanned (if using containers)
- [ ] Third-party integrations security reviewed

## Deployment Architectures

### 1. Single-Region Production Deployment

```mermaid
graph TB
    subgraph "Production VPC"
        subgraph "Public Subnets"
            ALB[Application Load Balancer]
            NAT[NAT Gateway]
        end
        
        subgraph "Private Subnets"
            APP1[Strands Application 1]
            APP2[Strands Application 2]
            APP3[Strands Application N]
        end
        
        subgraph "Database Subnets"
            RDS[RDS PostgreSQL<br/>Encrypted]
            REDIS[ElastiCache Redis<br/>Encrypted]
        end
    end
    
    subgraph "AWS Services"
        BEDROCK[Amazon Bedrock]
        AGENTCORE[AgentCore Browser Tool]
        SECRETS[Secrets Manager]
        KMS[AWS KMS]
        CLOUDWATCH[CloudWatch]
    end
    
    ALB --> APP1
    ALB --> APP2
    ALB --> APP3
    
    APP1 --> BEDROCK
    APP1 --> AGENTCORE
    APP1 --> SECRETS
    APP1 --> RDS
    APP1 --> REDIS
    
    APP2 --> BEDROCK
    APP2 --> AGENTCORE
    APP2 --> SECRETS
    APP2 --> RDS
    APP2 --> REDIS
    
    APP3 --> BEDROCK
    APP3 --> AGENTCORE
    APP3 --> SECRETS
    APP3 --> RDS
    APP3 --> REDIS
    
    SECRETS --> KMS
    RDS --> KMS
    REDIS --> KMS
    
    APP1 --> CLOUDWATCH
    APP2 --> CLOUDWATCH
    APP3 --> CLOUDWATCH
```

### 2. Multi-Region High Availability Deployment

```mermaid
graph TB
    subgraph "Primary Region (us-east-1)"
        subgraph "Production VPC Primary"
            ALB1[ALB Primary]
            APP1[Strands Apps Primary]
            RDS1[RDS Primary]
        end
        
        subgraph "AWS Services Primary"
            BEDROCK1[Bedrock Primary]
            AGENTCORE1[AgentCore Primary]
        end
    end
    
    subgraph "Secondary Region (us-west-2)"
        subgraph "Production VPC Secondary"
            ALB2[ALB Secondary]
            APP2[Strands Apps Secondary]
            RDS2[RDS Read Replica]
        end
        
        subgraph "AWS Services Secondary"
            BEDROCK2[Bedrock Secondary]
            AGENTCORE2[AgentCore Secondary]
        end
    end
    
    subgraph "Global Services"
        ROUTE53[Route 53<br/>Health Checks]
        CLOUDFRONT[CloudFront<br/>Global CDN]
        WAF[AWS WAF<br/>Global Protection]
    end
    
    ROUTE53 --> ALB1
    ROUTE53 --> ALB2
    CLOUDFRONT --> ALB1
    CLOUDFRONT --> ALB2
    WAF --> CLOUDFRONT
    
    RDS1 --> RDS2
    
    APP1 --> BEDROCK1
    APP1 --> AGENTCORE1
    APP2 --> BEDROCK2
    APP2 --> AGENTCORE2
```

## Deployment Configurations

### 1. Environment Configuration

#### Production Environment Variables
```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=123456789012

# Strands Configuration
STRANDS_ENVIRONMENT=production
STRANDS_LOG_LEVEL=INFO
STRANDS_MAX_WORKERS=20
STRANDS_TIMEOUT=600
STRANDS_RETRY_ATTEMPTS=3

# Multi-LLM Configuration
BEDROCK_PRIMARY_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
BEDROCK_FALLBACK_MODEL=anthropic.claude-3-haiku-20240307-v1:0
BEDROCK_COST_OPTIMIZED_MODEL=amazon.titan-text-express-v1

# AgentCore Configuration
AGENTCORE_BROWSER_REGION=us-east-1
AGENTCORE_SESSION_TIMEOUT=900
AGENTCORE_MAX_CONCURRENT_SESSIONS=50
AGENTCORE_ENABLE_OBSERVABILITY=true

# Security Configuration
ENCRYPTION_KEY_ID=arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012
AUDIT_LOG_LEVEL=INFO
ENABLE_PII_DETECTION=true
PII_CONFIDENCE_THRESHOLD=0.9
COMPLIANCE_MODE=enterprise

# Database Configuration
DATABASE_URL=postgresql://username:password@prod-db.cluster-xyz.us-east-1.rds.amazonaws.com:5432/strandsdb
REDIS_URL=redis://prod-cache.xyz.cache.amazonaws.com:6379

# Monitoring Configuration
CLOUDWATCH_LOG_GROUP=/strands-agentcore/production
CLOUDWATCH_METRICS_NAMESPACE=StrandsAgentCore/Production
ENABLE_XRAY_TRACING=true
```

#### Secrets Management Configuration
```python
# secrets_config.py
SECRETS_CONFIG = {
    'database_credentials': {
        'secret_name': 'strands-agentcore/prod/database',
        'region': 'us-east-1',
        'rotation_enabled': True,
        'rotation_interval_days': 30
    },
    'api_keys': {
        'secret_name': 'strands-agentcore/prod/api-keys',
        'region': 'us-east-1',
        'rotation_enabled': True,
        'rotation_interval_days': 90
    },
    'encryption_keys': {
        'secret_name': 'strands-agentcore/prod/encryption',
        'region': 'us-east-1',
        'rotation_enabled': True,
        'rotation_interval_days': 365
    }
}
```

### 2. Infrastructure as Code

#### Terraform Configuration
```hcl
# main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket         = "strands-agentcore-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "strands-agentcore-prod"
  cidr = "10.0.0.0/16"
  
  azs             = ["us-east-1a", "us-east-1b", "us-east-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  database_subnets = ["10.0.201.0/24", "10.0.202.0/24", "10.0.203.0/24"]
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
  enable_dns_hostnames = true
  enable_dns_support = true
  
  enable_flow_log = true
  flow_log_destination_type = "cloud-watch-logs"
  
  tags = {
    Environment = "production"
    Project     = "strands-agentcore"
  }
}

# KMS Key for Encryption
resource "aws_kms_key" "strands_agentcore" {
  description             = "KMS key for Strands AgentCore encryption"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      }
    ]
  })
  
  tags = {
    Environment = "production"
    Project     = "strands-agentcore"
  }
}

# RDS PostgreSQL Database
resource "aws_db_instance" "strands_agentcore" {
  identifier = "strands-agentcore-prod"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.r6g.xlarge"
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type          = "gp3"
  storage_encrypted     = true
  kms_key_id           = aws_kms_key.strands_agentcore.arn
  
  db_name  = "strandsdb"
  username = "strandsadmin"
  manage_master_user_password = true
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.strands_agentcore.name
  
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  deletion_protection = true
  skip_final_snapshot = false
  final_snapshot_identifier = "strands-agentcore-final-snapshot"
  
  performance_insights_enabled = true
  monitoring_interval         = 60
  monitoring_role_arn        = aws_iam_role.rds_monitoring.arn
  
  tags = {
    Environment = "production"
    Project     = "strands-agentcore"
  }
}

# ElastiCache Redis Cluster
resource "aws_elasticache_replication_group" "strands_agentcore" {
  replication_group_id       = "strands-agentcore-prod"
  description                = "Redis cluster for Strands AgentCore"
  
  node_type                  = "cache.r6g.large"
  port                       = 6379
  parameter_group_name       = "default.redis7"
  
  num_cache_clusters         = 3
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  subnet_group_name = aws_elasticache_subnet_group.strands_agentcore.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  kms_key_id                = aws_kms_key.strands_agentcore.arn
  
  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis_slow_log.name
    destination_type = "cloudwatch-logs"
    log_format       = "text"
    log_type         = "slow-log"
  }
  
  tags = {
    Environment = "production"
    Project     = "strands-agentcore"
  }
}
```

#### CloudFormation Template (Alternative)
```yaml
# cloudformation-template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Strands AgentCore Production Infrastructure'

Parameters:
  Environment:
    Type: String
    Default: production
    Description: Environment name
  
  VpcCidr:
    Type: String
    Default: 10.0.0.0/16
    Description: CIDR block for VPC

Resources:
  # VPC
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: !Ref VpcCidr
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: !Sub '${Environment}-strands-agentcore-vpc'
        - Key: Environment
          Value: !Ref Environment

  # Internet Gateway
  InternetGateway:
    Type: AWS::EC2::InternetGateway
    Properties:
      Tags:
        - Key: Name
          Value: !Sub '${Environment}-strands-agentcore-igw'

  # KMS Key
  KMSKey:
    Type: AWS::KMS::Key
    Properties:
      Description: 'KMS key for Strands AgentCore encryption'
      EnableKeyRotation: true
      KeyPolicy:
        Version: '2012-10-17'
        Statement:
          - Sid: Enable IAM User Permissions
            Effect: Allow
            Principal:
              AWS: !Sub 'arn:aws:iam::${AWS::AccountId}:root'
            Action: 'kms:*'
            Resource: '*'
      Tags:
        - Key: Environment
          Value: !Ref Environment

  # Application Load Balancer
  ApplicationLoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: !Sub '${Environment}-strands-agentcore-alb'
      Type: application
      Scheme: internet-facing
      SecurityGroups:
        - !Ref ALBSecurityGroup
      Subnets:
        - !Ref PublicSubnet1
        - !Ref PublicSubnet2
      Tags:
        - Key: Environment
          Value: !Ref Environment
```

### 3. Container Deployment (ECS/EKS)

#### Docker Configuration
```dockerfile
# Dockerfile
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create non-root user
RUN groupadd -r strands && useradd -r -g strands strands

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R strands:strands /app

# Switch to non-root user
USER strands

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Expose port
EXPOSE 8000

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "300", "app:app"]
```

#### ECS Task Definition
```json
{
  "family": "strands-agentcore-prod",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::123456789012:role/strandsAgentCoreTaskRole",
  "containerDefinitions": [
    {
      "name": "strands-agentcore",
      "image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/strands-agentcore:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "STRANDS_ENVIRONMENT",
          "value": "production"
        },
        {
          "name": "AWS_REGION",
          "value": "us-east-1"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789012:secret:strands-agentcore/prod/database"
        },
        {
          "name": "STRANDS_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789012:secret:strands-agentcore/prod/api-keys"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/strands-agentcore-prod",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "curl -f http://localhost:8000/health || exit 1"
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

#### Kubernetes Deployment (EKS)
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: strands-agentcore
  namespace: production
  labels:
    app: strands-agentcore
    environment: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: strands-agentcore
  template:
    metadata:
      labels:
        app: strands-agentcore
    spec:
      serviceAccountName: strands-agentcore
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: strands-agentcore
        image: 123456789012.dkr.ecr.us-east-1.amazonaws.com/strands-agentcore:latest
        ports:
        - containerPort: 8000
        env:
        - name: STRANDS_ENVIRONMENT
          value: "production"
        - name: AWS_REGION
          value: "us-east-1"
        envFrom:
        - secretRef:
            name: strands-agentcore-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
---
apiVersion: v1
kind: Service
metadata:
  name: strands-agentcore-service
  namespace: production
spec:
  selector:
    app: strands-agentcore
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

## Security Hardening

### 1. Network Security

#### Security Groups Configuration
```python
# security_groups.py
SECURITY_GROUPS = {
    'alb': {
        'ingress': [
            {'protocol': 'tcp', 'port': 443, 'source': '0.0.0.0/0'},
            {'protocol': 'tcp', 'port': 80, 'source': '0.0.0.0/0'}
        ],
        'egress': [
            {'protocol': 'tcp', 'port': 8000, 'source': 'app_security_group'}
        ]
    },
    'app': {
        'ingress': [
            {'protocol': 'tcp', 'port': 8000, 'source': 'alb_security_group'},
            {'protocol': 'tcp', 'port': 22, 'source': 'bastion_security_group'}
        ],
        'egress': [
            {'protocol': 'tcp', 'port': 443, 'source': '0.0.0.0/0'},  # HTTPS outbound
            {'protocol': 'tcp', 'port': 5432, 'source': 'rds_security_group'},
            {'protocol': 'tcp', 'port': 6379, 'source': 'redis_security_group'}
        ]
    },
    'rds': {
        'ingress': [
            {'protocol': 'tcp', 'port': 5432, 'source': 'app_security_group'}
        ],
        'egress': []
    },
    'redis': {
        'ingress': [
            {'protocol': 'tcp', 'port': 6379, 'source': 'app_security_group'}
        ],
        'egress': []
    }
}
```

#### WAF Configuration
```json
{
  "Name": "StrandsAgentCoreWAF",
  "Scope": "CLOUDFRONT",
  "DefaultAction": {
    "Allow": {}
  },
  "Rules": [
    {
      "Name": "AWSManagedRulesCommonRuleSet",
      "Priority": 1,
      "OverrideAction": {
        "None": {}
      },
      "Statement": {
        "ManagedRuleGroupStatement": {
          "VendorName": "AWS",
          "Name": "AWSManagedRulesCommonRuleSet"
        }
      },
      "VisibilityConfig": {
        "SampledRequestsEnabled": true,
        "CloudWatchMetricsEnabled": true,
        "MetricName": "CommonRuleSetMetric"
      }
    },
    {
      "Name": "AWSManagedRulesKnownBadInputsRuleSet",
      "Priority": 2,
      "OverrideAction": {
        "None": {}
      },
      "Statement": {
        "ManagedRuleGroupStatement": {
          "VendorName": "AWS",
          "Name": "AWSManagedRulesKnownBadInputsRuleSet"
        }
      },
      "VisibilityConfig": {
        "SampledRequestsEnabled": true,
        "CloudWatchMetricsEnabled": true,
        "MetricName": "KnownBadInputsMetric"
      }
    },
    {
      "Name": "RateLimitRule",
      "Priority": 3,
      "Action": {
        "Block": {}
      },
      "Statement": {
        "RateBasedStatement": {
          "Limit": 2000,
          "AggregateKeyType": "IP"
        }
      },
      "VisibilityConfig": {
        "SampledRequestsEnabled": true,
        "CloudWatchMetricsEnabled": true,
        "MetricName": "RateLimitMetric"
      }
    }
  ]
}
```

### 2. IAM Security

#### Production IAM Roles
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
        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
        "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-text-express-v1"
      ],
      "Condition": {
        "StringEquals": {
          "aws:RequestedRegion": "us-east-1"
        }
      }
    },
    {
      "Effect": "Allow",
      "Action": [
        "bedrock-agentcore:CreateBrowserSession",
        "bedrock-agentcore:GetBrowserSession",
        "bedrock-agentcore:DeleteBrowserSession",
        "bedrock-agentcore:SendBrowserAction"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "aws:RequestedRegion": "us-east-1"
        },
        "StringLike": {
          "bedrock-agentcore:SessionTag/Environment": "production"
        }
      }
    },
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": [
        "arn:aws:secretsmanager:us-east-1:123456789012:secret:strands-agentcore/prod/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "kms:Decrypt",
        "kms:GenerateDataKey"
      ],
      "Resource": [
        "arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": [
        "arn:aws:logs:us-east-1:123456789012:log-group:/strands-agentcore/production:*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "cloudwatch:PutMetricData"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "cloudwatch:namespace": "StrandsAgentCore/Production"
        }
      }
    }
  ]
}
```

### 3. Data Encryption

#### Encryption Configuration
```python
# encryption_config.py
ENCRYPTION_CONFIG = {
    'data_at_rest': {
        'algorithm': 'AES-256',
        'key_management': 'AWS-KMS',
        'key_rotation': True,
        'key_rotation_interval_days': 365
    },
    'data_in_transit': {
        'protocol': 'TLS-1.3',
        'certificate_validation': True,
        'cipher_suites': [
            'TLS_AES_256_GCM_SHA384',
            'TLS_CHACHA20_POLY1305_SHA256',
            'TLS_AES_128_GCM_SHA256'
        ]
    },
    'application_level': {
        'pii_encryption': True,
        'field_level_encryption': True,
        'encryption_context': {
            'application': 'strands-agentcore',
            'environment': 'production'
        }
    }
}
```

## Monitoring and Observability

### 1. CloudWatch Configuration

#### Custom Metrics
```python
# monitoring.py
import boto3
from datetime import datetime

class ProductionMonitoring:
    def __init__(self, region='us-east-1'):
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.namespace = 'StrandsAgentCore/Production'
        
    def put_custom_metric(self, metric_name, value, unit='Count', dimensions=None):
        """Put custom metric to CloudWatch."""
        self.cloudwatch.put_metric_data(
            Namespace=self.namespace,
            MetricData=[
                {
                    'MetricName': metric_name,
                    'Value': value,
                    'Unit': unit,
                    'Timestamp': datetime.utcnow(),
                    'Dimensions': dimensions or []
                }
            ]
        )
        
    def track_pii_detection(self, pii_count, confidence_score):
        """Track PII detection metrics."""
        self.put_custom_metric('PIIDetectionCount', pii_count)
        self.put_custom_metric('PIIConfidenceScore', confidence_score, 'None')
        
    def track_session_metrics(self, session_duration, session_success):
        """Track browser session metrics."""
        self.put_custom_metric('SessionDuration', session_duration, 'Seconds')
        self.put_custom_metric('SessionSuccess', 1 if session_success else 0)
        
    def track_llm_routing(self, model_name, response_time, cost):
        """Track LLM routing metrics."""
        dimensions = [{'Name': 'ModelName', 'Value': model_name}]
        self.put_custom_metric('LLMResponseTime', response_time, 'Milliseconds', dimensions)
        self.put_custom_metric('LLMCost', cost, 'None', dimensions)
```

#### CloudWatch Alarms
```json
{
  "AlarmName": "StrandsAgentCore-HighErrorRate",
  "AlarmDescription": "High error rate in Strands AgentCore application",
  "MetricName": "ErrorRate",
  "Namespace": "StrandsAgentCore/Production",
  "Statistic": "Average",
  "Period": 300,
  "EvaluationPeriods": 2,
  "Threshold": 5.0,
  "ComparisonOperator": "GreaterThanThreshold",
  "AlarmActions": [
    "arn:aws:sns:us-east-1:123456789012:strands-agentcore-alerts"
  ],
  "TreatMissingData": "breaching"
}
```

### 2. Distributed Tracing

#### X-Ray Configuration
```python
# xray_config.py
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all

# Patch AWS SDK calls
patch_all()

@xray_recorder.capture('strands_workflow_execution')
def execute_secure_workflow(workflow_config):
    """Execute workflow with X-Ray tracing."""
    
    # Add metadata
    xray_recorder.put_metadata('workflow_id', workflow_config['id'])
    xray_recorder.put_metadata('security_level', workflow_config['security_level'])
    
    # Add annotations for filtering
    xray_recorder.put_annotation('environment', 'production')
    xray_recorder.put_annotation('compliance_mode', workflow_config.get('compliance_mode'))
    
    try:
        # Execute workflow
        result = _execute_workflow_internal(workflow_config)
        
        # Add success metadata
        xray_recorder.put_metadata('execution_status', 'success')
        xray_recorder.put_metadata('result_summary', {
            'items_processed': result.get('items_processed', 0),
            'pii_detected': result.get('pii_detected', False)
        })
        
        return result
        
    except Exception as e:
        # Add error metadata
        xray_recorder.put_metadata('execution_status', 'error')
        xray_recorder.put_metadata('error_details', str(e))
        raise
```

### 3. Application Performance Monitoring

#### Custom APM Integration
```python
# apm_integration.py
import time
import psutil
from datadog import initialize, statsd

class APMMonitor:
    def __init__(self):
        initialize(
            api_key=os.getenv('DATADOG_API_KEY'),
            app_key=os.getenv('DATADOG_APP_KEY')
        )
        
    def track_performance_metrics(self):
        """Track system performance metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        statsd.gauge('strands.agentcore.cpu.usage', cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        statsd.gauge('strands.agentcore.memory.usage', memory.percent)
        statsd.gauge('strands.agentcore.memory.available', memory.available)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        statsd.gauge('strands.agentcore.disk.usage', disk.percent)
        
    def track_business_metrics(self, workflow_result):
        """Track business-specific metrics."""
        # Workflow execution time
        statsd.histogram('strands.agentcore.workflow.duration', 
                        workflow_result.get('execution_time', 0))
        
        # PII detection rate
        if workflow_result.get('pii_detected'):
            statsd.increment('strands.agentcore.pii.detected')
            
        # Compliance violations
        violations = workflow_result.get('compliance_violations', 0)
        if violations > 0:
            statsd.increment('strands.agentcore.compliance.violations', violations)
```

## Scaling and Performance

### 1. Auto Scaling Configuration

#### ECS Auto Scaling
```json
{
  "ServiceName": "strands-agentcore-prod",
  "ScalableDimension": "ecs:service:DesiredCount",
  "ServiceNamespace": "ecs",
  "MinCapacity": 3,
  "MaxCapacity": 20,
  "TargetTrackingScalingPolicies": [
    {
      "PolicyName": "CPUUtilization",
      "TargetValue": 70.0,
      "PredefinedMetricSpecification": {
        "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
      },
      "ScaleOutCooldown": 300,
      "ScaleInCooldown": 300
    },
    {
      "PolicyName": "MemoryUtilization",
      "TargetValue": 80.0,
      "PredefinedMetricSpecification": {
        "PredefinedMetricType": "ECSServiceAverageMemoryUtilization"
      },
      "ScaleOutCooldown": 300,
      "ScaleInCooldown": 300
    }
  ]
}
```

#### Application-Level Scaling
```python
# scaling_config.py
SCALING_CONFIG = {
    'session_pool': {
        'min_sessions': 5,
        'max_sessions': 50,
        'scale_up_threshold': 0.8,  # 80% utilization
        'scale_down_threshold': 0.3,  # 30% utilization
        'scale_up_increment': 5,
        'scale_down_increment': 2,
        'cooldown_period': 300  # 5 minutes
    },
    'worker_processes': {
        'min_workers': 4,
        'max_workers': 20,
        'cpu_threshold': 70,
        'memory_threshold': 80,
        'queue_length_threshold': 100
    },
    'llm_routing': {
        'enable_cost_optimization': True,
        'prefer_faster_models_under_load': True,
        'fallback_to_cheaper_models': True,
        'max_concurrent_requests_per_model': 10
    }
}
```

### 2. Performance Optimization

#### Caching Strategy
```python
# caching.py
import redis
import json
from functools import wraps

class ProductionCache:
    def __init__(self, redis_url):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        
    def cache_pii_detection_results(self, ttl=3600):
        """Cache PII detection results."""
        def decorator(func):
            @wraps(func)
            def wrapper(content_hash, *args, **kwargs):
                cache_key = f"pii_detection:{content_hash}"
                
                # Try to get from cache
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    return json.loads(cached_result)
                
                # Execute function
                result = func(content_hash, *args, **kwargs)
                
                # Cache result (without sensitive data)
                cache_data = {
                    'pii_detected': result.get('pii_detected', False),
                    'confidence_score': result.get('confidence_score', 0),
                    'entity_types': result.get('entity_types', [])
                }
                self.redis_client.setex(cache_key, ttl, json.dumps(cache_data))
                
                return result
            return wrapper
        return decorator
        
    def cache_model_responses(self, ttl=1800):
        """Cache LLM model responses for non-sensitive queries."""
        def decorator(func):
            @wraps(func)
            def wrapper(query_hash, model_name, *args, **kwargs):
                # Only cache non-sensitive queries
                if kwargs.get('contains_pii', False):
                    return func(query_hash, model_name, *args, **kwargs)
                
                cache_key = f"llm_response:{model_name}:{query_hash}"
                
                # Try to get from cache
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    return json.loads(cached_result)
                
                # Execute function
                result = func(query_hash, model_name, *args, **kwargs)
                
                # Cache result
                self.redis_client.setex(cache_key, ttl, json.dumps(result))
                
                return result
            return wrapper
        return decorator
```

## Disaster Recovery

### 1. Backup Strategy

#### Database Backups
```python
# backup_config.py
BACKUP_CONFIG = {
    'rds': {
        'automated_backup_retention': 30,  # days
        'backup_window': '03:00-04:00',
        'maintenance_window': 'sun:04:00-sun:05:00',
        'cross_region_backup': True,
        'backup_regions': ['us-west-2'],
        'point_in_time_recovery': True
    },
    'application_data': {
        'backup_frequency': 'daily',
        'backup_retention': 90,  # days
        'backup_encryption': True,
        'backup_verification': True,
        'cross_region_replication': True
    },
    'secrets': {
        'backup_frequency': 'weekly',
        'backup_retention': 365,  # days
        'backup_encryption': True,
        'cross_region_backup': True
    }
}
```

#### Backup Automation
```python
# backup_automation.py
import boto3
from datetime import datetime, timedelta

class BackupAutomation:
    def __init__(self, region='us-east-1'):
        self.backup_client = boto3.client('backup', region_name=region)
        self.rds_client = boto3.client('rds', region_name=region)
        
    def create_backup_plan(self):
        """Create comprehensive backup plan."""
        backup_plan = {
            'BackupPlanName': 'StrandsAgentCoreBackupPlan',
            'Rules': [
                {
                    'RuleName': 'DailyBackups',
                    'TargetBackupVault': 'strands-agentcore-backup-vault',
                    'ScheduleExpression': 'cron(0 2 ? * * *)',  # Daily at 2 AM
                    'StartWindowMinutes': 60,
                    'CompletionWindowMinutes': 120,
                    'Lifecycle': {
                        'DeleteAfterDays': 30,
                        'MoveToColdStorageAfterDays': 7
                    },
                    'RecoveryPointTags': {
                        'Environment': 'production',
                        'Application': 'strands-agentcore'
                    }
                },
                {
                    'RuleName': 'WeeklyBackups',
                    'TargetBackupVault': 'strands-agentcore-backup-vault',
                    'ScheduleExpression': 'cron(0 3 ? * SUN *)',  # Weekly on Sunday at 3 AM
                    'StartWindowMinutes': 60,
                    'CompletionWindowMinutes': 180,
                    'Lifecycle': {
                        'DeleteAfterDays': 365,
                        'MoveToColdStorageAfterDays': 30
                    }
                }
            ]
        }
        
        response = self.backup_client.create_backup_plan(BackupPlan=backup_plan)
        return response['BackupPlanId']
```

### 2. Disaster Recovery Procedures

#### RTO/RPO Targets
```python
# dr_config.py
DISASTER_RECOVERY_CONFIG = {
    'rto_targets': {  # Recovery Time Objective
        'critical_services': 30,  # minutes
        'important_services': 60,  # minutes
        'standard_services': 240  # minutes
    },
    'rpo_targets': {  # Recovery Point Objective
        'critical_data': 15,  # minutes
        'important_data': 60,  # minutes
        'standard_data': 240  # minutes
    },
    'failover_procedures': {
        'automatic_failover': True,
        'manual_approval_required': False,
        'health_check_interval': 30,  # seconds
        'failover_threshold': 3  # consecutive failures
    }
}
```

## Compliance and Governance

### 1. Compliance Monitoring

#### Automated Compliance Checks
```python
# compliance_monitoring.py
import boto3
from datetime import datetime

class ComplianceMonitor:
    def __init__(self, region='us-east-1'):
        self.config_client = boto3.client('config', region_name=region)
        self.compliance_frameworks = ['hipaa', 'pci_dss', 'gdpr', 'sox']
        
    def run_compliance_checks(self):
        """Run automated compliance checks."""
        compliance_results = {}
        
        for framework in self.compliance_frameworks:
            framework_results = self._check_framework_compliance(framework)
            compliance_results[framework] = framework_results
            
        return compliance_results
        
    def _check_framework_compliance(self, framework):
        """Check compliance for specific framework."""
        if framework == 'hipaa':
            return self._check_hipaa_compliance()
        elif framework == 'pci_dss':
            return self._check_pci_dss_compliance()
        elif framework == 'gdpr':
            return self._check_gdpr_compliance()
        elif framework == 'sox':
            return self._check_sox_compliance()
            
    def _check_hipaa_compliance(self):
        """Check HIPAA compliance requirements."""
        checks = {
            'encryption_at_rest': self._verify_encryption_at_rest(),
            'encryption_in_transit': self._verify_encryption_in_transit(),
            'access_controls': self._verify_access_controls(),
            'audit_logging': self._verify_audit_logging(),
            'data_backup': self._verify_data_backup(),
            'incident_response': self._verify_incident_response_plan()
        }
        
        compliance_score = sum(checks.values()) / len(checks) * 100
        
        return {
            'framework': 'hipaa',
            'compliance_score': compliance_score,
            'checks': checks,
            'timestamp': datetime.utcnow().isoformat()
        }
```

### 2. Audit Logging

#### Comprehensive Audit System
```python
# audit_system.py
import json
import boto3
from datetime import datetime
from enum import Enum

class AuditEventType(Enum):
    USER_ACCESS = "user_access"
    DATA_ACCESS = "data_access"
    SYSTEM_CHANGE = "system_change"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_EVENT = "compliance_event"

class ProductionAuditLogger:
    def __init__(self, region='us-east-1'):
        self.cloudtrail_client = boto3.client('cloudtrail', region_name=region)
        self.cloudwatch_logs = boto3.client('logs', region_name=region)
        self.log_group = '/strands-agentcore/audit/production'
        
    def log_audit_event(self, event_type: AuditEventType, event_data: dict):
        """Log audit event with comprehensive details."""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type.value,
            'event_id': self._generate_event_id(),
            'source_ip': event_data.get('source_ip'),
            'user_id': event_data.get('user_id'),
            'session_id': event_data.get('session_id'),
            'resource_accessed': event_data.get('resource_accessed'),
            'action_performed': event_data.get('action_performed'),
            'result': event_data.get('result'),
            'additional_context': event_data.get('additional_context', {})
        }
        
        # Remove sensitive data from audit log
        sanitized_entry = self._sanitize_audit_entry(audit_entry)
        
        # Write to CloudWatch Logs
        self._write_to_cloudwatch_logs(sanitized_entry)
        
        # Write to S3 for long-term retention
        self._write_to_s3_audit_bucket(sanitized_entry)
        
    def _sanitize_audit_entry(self, audit_entry: dict) -> dict:
        """Remove sensitive data from audit entries."""
        # Create a copy to avoid modifying original
        sanitized = audit_entry.copy()
        
        # Remove or mask sensitive fields
        sensitive_fields = ['password', 'api_key', 'token', 'ssn', 'credit_card']
        
        for field in sensitive_fields:
            if field in sanitized:
                sanitized[field] = '[REDACTED]'
                
        return sanitized
```

## Deployment Checklist

### Pre-Deployment
- [ ] Infrastructure provisioned and tested
- [ ] Security groups and IAM roles configured
- [ ] Secrets stored in AWS Secrets Manager
- [ ] Database migrations completed
- [ ] SSL certificates installed and validated
- [ ] Monitoring and alerting configured
- [ ] Backup and disaster recovery tested

### Deployment
- [ ] Blue-green deployment strategy implemented
- [ ] Health checks passing
- [ ] Load balancer configuration verified
- [ ] Auto-scaling policies active
- [ ] Security scanning completed
- [ ] Performance testing completed
- [ ] Compliance validation completed

### Post-Deployment
- [ ] Application functionality verified
- [ ] Security controls validated
- [ ] Monitoring dashboards active
- [ ] Audit logging functional
- [ ] Backup procedures verified
- [ ] Incident response procedures tested
- [ ] Documentation updated
- [ ] Team training completed

## Conclusion

This deployment guide provides a comprehensive framework for deploying Strands agents with AgentCore Browser Tool in production environments. Following these guidelines ensures security, scalability, compliance, and operational excellence for enterprise deployments handling sensitive information.

Regular reviews and updates of these procedures are essential to maintain security posture and adapt to evolving requirements and threats.