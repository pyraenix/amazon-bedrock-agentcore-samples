# Browser-Use + AgentCore Deployment and Troubleshooting Guide

## Overview

This guide provides comprehensive instructions for deploying browser-use with Amazon Bedrock AgentCore Browser Tool in production environments, along with troubleshooting guidance for sensitive information scenarios and performance optimization strategies.

## Production Deployment Guide

### Prerequisites

#### System Requirements
- Python 3.12 or higher
- AWS CLI configured with appropriate permissions
- Amazon Bedrock access in target region
- AgentCore Browser Tool enabled in AWS account

#### Required AWS Permissions
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
                "bedrock-agent:*",
                "bedrock-runtime:*"
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

### Environment Setup

#### 1. Python Environment Configuration

```bash
# Create virtual environment
python3.12 -m venv browseruse-agentcore-env
source browseruse-agentcore-env/bin/activate  # Linux/Mac
# or
browseruse-agentcore-env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

#### 2. Environment Variables

Create a `.env` file with the following configuration:

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_PROFILE=default

# AgentCore Configuration
AGENTCORE_REGION=us-east-1
AGENTCORE_SESSION_TIMEOUT=300
AGENTCORE_ENABLE_LIVE_VIEW=true
AGENTCORE_ENABLE_SESSION_REPLAY=true

# Security Configuration
SECURITY_COMPLIANCE_MODE=enterprise
SECURITY_ISOLATION_LEVEL=micro-vm
SECURITY_AUDIT_LEVEL=detailed

# Browser-Use Configuration
BROWSERUSE_MODEL_NAME=anthropic.claude-3-5-sonnet-20241022-v2:0
BROWSERUSE_MAX_RETRIES=3
BROWSERUSE_TIMEOUT=30

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
AUDIT_LOG_RETENTION_DAYS=90

# Performance Configuration
MAX_CONCURRENT_SESSIONS=10
SESSION_POOL_SIZE=5
CLEANUP_INTERVAL_SECONDS=60
```

#### 3. Configuration Validation

```python
# config_validator.py
import os
import boto3
from typing import Dict, List

class DeploymentConfigValidator:
    """Validate deployment configuration."""
    
    def __init__(self):
        self.required_env_vars = [
            'AWS_REGION',
            'AGENTCORE_REGION',
            'BROWSERUSE_MODEL_NAME'
        ]
        self.validation_errors = []
    
    def validate_environment(self) -> bool:
        """Validate environment configuration."""
        
        # Check required environment variables
        for var in self.required_env_vars:
            if not os.getenv(var):
                self.validation_errors.append(f"Missing required environment variable: {var}")
        
        # Validate AWS credentials
        if not self._validate_aws_credentials():
            self.validation_errors.append("Invalid AWS credentials")
        
        # Validate AgentCore access
        if not self._validate_agentcore_access():
            self.validation_errors.append("Cannot access AgentCore Browser Tool")
        
        # Validate Bedrock model access
        if not self._validate_bedrock_access():
            self.validation_errors.append("Cannot access Bedrock models")
        
        return len(self.validation_errors) == 0
    
    def _validate_aws_credentials(self) -> bool:
        """Validate AWS credentials."""
        try:
            session = boto3.Session()
            sts = session.client('sts')
            sts.get_caller_identity()
            return True
        except Exception:
            return False
    
    def _validate_agentcore_access(self) -> bool:
        """Validate AgentCore Browser Tool access."""
        try:
            # Test AgentCore Browser Tool access
            from bedrock_agentcore.tools.browser_client import BrowserClient
            client = BrowserClient(region=os.getenv('AGENTCORE_REGION'))
            # Perform basic connectivity test
            return True
        except Exception:
            return False
    
    def _validate_bedrock_access(self) -> bool:
        """Validate Bedrock model access."""
        try:
            bedrock = boto3.client('bedrock-runtime', region_name=os.getenv('AWS_REGION'))
            # Test model access (without actually invoking)
            return True
        except Exception:
            return False
    
    def get_validation_report(self) -> Dict:
        """Get validation report."""
        return {
            'is_valid': len(self.validation_errors) == 0,
            'errors': self.validation_errors,
            'timestamp': datetime.utcnow().isoformat()
        }

# Usage
validator = DeploymentConfigValidator()
if validator.validate_environment():
    print("✅ Environment configuration is valid")
else:
    print("❌ Environment configuration errors:")
    for error in validator.validation_errors:
        print(f"  - {error}")
```

### Production Architecture Deployment

#### 1. Containerized Deployment

```dockerfile
# Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 browseruse && chown -R browseruse:browseruse /app
USER browseruse

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Expose port
EXPOSE 8000

# Start application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 2. Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: browseruse-agentcore
  labels:
    app: browseruse-agentcore
spec:
  replicas: 3
  selector:
    matchLabels:
      app: browseruse-agentcore
  template:
    metadata:
      labels:
        app: browseruse-agentcore
    spec:
      containers:
      - name: browseruse-agentcore
        image: browseruse-agentcore:latest
        ports:
        - containerPort: 8000
        env:
        - name: AWS_REGION
          value: "us-east-1"
        - name: AGENTCORE_REGION
          value: "us-east-1"
        - name: SECURITY_COMPLIANCE_MODE
          value: "enterprise"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: browseruse-agentcore-service
spec:
  selector:
    app: browseruse-agentcore
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

#### 3. AWS Lambda Deployment

```python
# lambda_handler.py
import json
import asyncio
from typing import Dict, Any
from browseruse_agentcore_integration import BrowserUseAgentCoreIntegration

def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """AWS Lambda handler for browser-use + AgentCore integration."""
    
    try:
        # Parse request
        task_instruction = event.get('task_instruction')
        security_context = SecurityContext.from_dict(event.get('security_context', {}))
        
        # Initialize integration
        integration = BrowserUseAgentCoreIntegration(
            config=IntegrationConfig.from_environment()
        )
        
        # Execute task
        result = asyncio.run(
            integration.execute_secure_task(task_instruction, security_context)
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'result': result,
                'session_id': security_context.session_id
            })
        }
        
    except ComplianceViolationException as e:
        return {
            'statusCode': 400,
            'body': json.dumps({
                'success': False,
                'error': 'compliance_violation',
                'details': str(e),
                'violations': e.violations
            })
        }
        
    except SecurityException as e:
        return {
            'statusCode': 403,
            'body': json.dumps({
                'success': False,
                'error': 'security_error',
                'details': str(e)
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'success': False,
                'error': 'internal_error',
                'details': str(e)
            })
        }
```

### Performance Optimization

#### 1. Session Pool Management

```python
# session_pool.py
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class PooledSession:
    """Pooled AgentCore session."""
    session_id: str
    created_at: datetime
    last_used: datetime
    is_active: bool
    security_context: SecurityContext

class AgentCoreSessionPool:
    """Session pool for AgentCore Browser Tool sessions."""
    
    def __init__(self, 
                 pool_size: int = 5,
                 max_session_age: int = 300,
                 cleanup_interval: int = 60):
        self.pool_size = pool_size
        self.max_session_age = max_session_age
        self.cleanup_interval = cleanup_interval
        self.sessions: Dict[str, PooledSession] = {}
        self.available_sessions: List[str] = []
        self.lock = asyncio.Lock()
        self._cleanup_task = None
    
    async def start(self):
        """Start the session pool."""
        # Pre-warm the pool
        await self._prewarm_pool()
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop(self):
        """Stop the session pool."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Clean up all sessions
        await self._cleanup_all_sessions()
    
    async def get_session(self, security_context: SecurityContext) -> Optional[str]:
        """Get a session from the pool."""
        async with self.lock:
            # Try to find a compatible session
            for session_id in self.available_sessions:
                session = self.sessions[session_id]
                if self._is_session_compatible(session, security_context):
                    self.available_sessions.remove(session_id)
                    session.last_used = datetime.utcnow()
                    session.is_active = True
                    return session_id
            
            # Create new session if pool not full
            if len(self.sessions) < self.pool_size:
                session_id = await self._create_session(security_context)
                return session_id
            
            return None
    
    async def return_session(self, session_id: str):
        """Return a session to the pool."""
        async with self.lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                session.is_active = False
                session.last_used = datetime.utcnow()
                
                # Add back to available sessions if not expired
                if not self._is_session_expired(session):
                    self.available_sessions.append(session_id)
                else:
                    await self._remove_session(session_id)
    
    async def _prewarm_pool(self):
        """Pre-warm the session pool."""
        default_context = SecurityContext.default()
        
        for _ in range(min(3, self.pool_size)):  # Pre-warm with 3 sessions
            try:
                await self._create_session(default_context)
            except Exception as e:
                print(f"Failed to pre-warm session: {e}")
    
    async def _create_session(self, security_context: SecurityContext) -> str:
        """Create a new session."""
        from bedrock_agentcore.tools.browser_client import BrowserClient
        
        client = BrowserClient(region=security_context.region)
        session = await client.create_session()
        
        pooled_session = PooledSession(
            session_id=session.session_id,
            created_at=datetime.utcnow(),
            last_used=datetime.utcnow(),
            is_active=True,
            security_context=security_context
        )
        
        self.sessions[session.session_id] = pooled_session
        return session.session_id
    
    async def _cleanup_loop(self):
        """Cleanup loop for expired sessions."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Session cleanup error: {e}")
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        async with self.lock:
            expired_sessions = []
            
            for session_id, session in self.sessions.items():
                if self._is_session_expired(session) and not session.is_active:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                await self._remove_session(session_id)
    
    def _is_session_expired(self, session: PooledSession) -> bool:
        """Check if session is expired."""
        age = datetime.utcnow() - session.created_at
        return age.total_seconds() > self.max_session_age
    
    def _is_session_compatible(self, session: PooledSession, context: SecurityContext) -> bool:
        """Check if session is compatible with security context."""
        return (session.security_context.compliance_requirements == 
                context.compliance_requirements and
                session.security_context.isolation_level == 
                context.isolation_level)
```

#### 2. Caching and Optimization

```python
# performance_optimizer.py
import asyncio
from typing import Dict, Any, Optional
from functools import wraps
import hashlib
import json

class PerformanceOptimizer:
    """Performance optimization utilities."""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_response_time': 0,
            'total_requests': 0
        }
    
    def cached_result(self, ttl: int = 300):
        """Decorator for caching results."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Create cache key
                cache_key = self._create_cache_key(func.__name__, args, kwargs)
                
                # Check cache
                cached_result = self._get_cached_result(cache_key)
                if cached_result is not None:
                    self.metrics['cache_hits'] += 1
                    return cached_result
                
                # Execute function
                self.metrics['cache_misses'] += 1
                result = await func(*args, **kwargs)
                
                # Cache result
                self._cache_result(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator
    
    def _create_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Create cache key from function parameters."""
        key_data = {
            'function': func_name,
            'args': str(args),
            'kwargs': sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if not expired."""
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if cached_data['expires_at'] > asyncio.get_event_loop().time():
                return cached_data['result']
            else:
                del self.cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: Any, ttl: int):
        """Cache result with TTL."""
        self.cache[cache_key] = {
            'result': result,
            'expires_at': asyncio.get_event_loop().time() + ttl
        }
    
    async def optimize_concurrent_execution(self, tasks: List[callable], max_concurrent: int = 5):
        """Optimize concurrent task execution."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(task):
            async with semaphore:
                return await task()
        
        results = await asyncio.gather(
            *[execute_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )
        
        return results
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Session Creation Failures

**Problem**: AgentCore session creation fails
```
Error: Failed to create AgentCore session: ConnectionError
```

**Diagnosis Steps**:
```python
# session_diagnostics.py
async def diagnose_session_creation():
    """Diagnose session creation issues."""
    
    print("🔍 Diagnosing session creation...")
    
    # Check AWS credentials
    try:
        import boto3
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✅ AWS credentials valid: {identity['Arn']}")
    except Exception as e:
        print(f"❌ AWS credentials invalid: {e}")
        return
    
    # Check AgentCore service availability
    try:
        from bedrock_agentcore.tools.browser_client import BrowserClient
        client = BrowserClient(region=os.getenv('AWS_REGION'))
        print("✅ AgentCore client initialized")
    except Exception as e:
        print(f"❌ AgentCore client initialization failed: {e}")
        return
    
    # Test session creation
    try:
        session = await client.create_session()
        print(f"✅ Session created successfully: {session.session_id}")
        
        # Clean up test session
        await client.cleanup_session(session.session_id)
        print("✅ Session cleanup successful")
        
    except Exception as e:
        print(f"❌ Session creation failed: {e}")
        
        # Additional diagnostics
        if "AccessDenied" in str(e):
            print("💡 Check IAM permissions for bedrock-agent:* actions")
        elif "ServiceUnavailable" in str(e):
            print("💡 AgentCore service may be unavailable in this region")
        elif "QuotaExceeded" in str(e):
            print("💡 Session quota exceeded, wait or request limit increase")
```

**Solutions**:
1. Verify AWS credentials and permissions
2. Check AgentCore service availability in region
3. Verify session quotas and limits
4. Implement retry logic with exponential backoff

#### 2. PII Detection Issues

**Problem**: PII not being detected or incorrectly masked
```
Warning: Potential PII leakage detected in output
```

**Diagnosis Steps**:
```python
# pii_diagnostics.py
def diagnose_pii_detection():
    """Diagnose PII detection issues."""
    
    test_data = {
        'ssn': '123-45-6789',
        'email': 'user@example.com',
        'phone': '555-123-4567',
        'credit_card': '4111-1111-1111-1111'
    }
    
    pii_detector = PIIDetector()
    
    for pii_type, test_value in test_data.items():
        detected = pii_detector.detect_pii_type(test_value, pii_type)
        if detected:
            print(f"✅ {pii_type.upper()} detection working: {test_value}")
        else:
            print(f"❌ {pii_type.upper()} detection failed: {test_value}")
            
            # Suggest pattern improvements
            print(f"💡 Consider updating regex pattern for {pii_type}")
```

**Solutions**:
1. Update PII detection patterns
2. Implement context-aware detection
3. Add custom PII types for specific domains
4. Validate masking effectiveness

#### 3. Browser-Use Integration Issues

**Problem**: Browser-use agent fails to connect to AgentCore browser
```
Error: WebSocket connection failed to AgentCore browser
```

**Diagnosis Steps**:
```python
# browseruse_diagnostics.py
async def diagnose_browseruse_integration():
    """Diagnose browser-use integration issues."""
    
    # Test AgentCore session
    try:
        from bedrock_agentcore.tools.browser_client import BrowserClient
        client = BrowserClient(region=os.getenv('AWS_REGION'))
        session = await client.create_session()
        
        ws_url, headers = client.get_connection_details(session.session_id)
        print(f"✅ AgentCore session created: {session.session_id}")
        print(f"✅ WebSocket URL: {ws_url}")
        
    except Exception as e:
        print(f"❌ AgentCore session creation failed: {e}")
        return
    
    # Test WebSocket connection
    try:
        import websockets
        async with websockets.connect(ws_url, extra_headers=headers) as websocket:
            print("✅ WebSocket connection successful")
            
            # Test CDP communication
            test_message = json.dumps({
                "id": 1,
                "method": "Runtime.evaluate",
                "params": {"expression": "1+1"}
            })
            
            await websocket.send(test_message)
            response = await websocket.recv()
            print(f"✅ CDP communication successful: {response}")
            
    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")
        
        if "403" in str(e):
            print("💡 Check authentication headers")
        elif "timeout" in str(e):
            print("💡 Check network connectivity and firewall rules")
    
    finally:
        # Cleanup
        try:
            await client.cleanup_session(session.session_id)
        except:
            pass
```

**Solutions**:
1. Verify WebSocket connection parameters
2. Check network connectivity and firewall rules
3. Validate authentication headers
4. Implement connection retry logic

#### 4. Performance Issues

**Problem**: Slow response times or timeouts
```
Error: Task execution timeout after 30 seconds
```

**Performance Monitoring**:
```python
# performance_monitor.py
import time
import psutil
from typing import Dict, Any

class PerformanceMonitor:
    """Monitor performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.metrics['start_cpu'] = psutil.cpu_percent()
        self.metrics['start_memory'] = psutil.virtual_memory().percent
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return metrics."""
        end_time = time.time()
        
        return {
            'execution_time': end_time - self.start_time,
            'cpu_usage': {
                'start': self.metrics['start_cpu'],
                'end': psutil.cpu_percent(),
                'average': (self.metrics['start_cpu'] + psutil.cpu_percent()) / 2
            },
            'memory_usage': {
                'start': self.metrics['start_memory'],
                'end': psutil.virtual_memory().percent,
                'peak': max(self.metrics['start_memory'], psutil.virtual_memory().percent)
            }
        }
    
    async def profile_task_execution(self, task_func, *args, **kwargs):
        """Profile task execution."""
        self.start_monitoring()
        
        try:
            result = await task_func(*args, **kwargs)
            metrics = self.stop_monitoring()
            
            return {
                'result': result,
                'performance_metrics': metrics,
                'success': True
            }
            
        except Exception as e:
            metrics = self.stop_monitoring()
            
            return {
                'error': str(e),
                'performance_metrics': metrics,
                'success': False
            }
```

**Optimization Strategies**:
1. Implement session pooling
2. Use caching for repeated operations
3. Optimize concurrent execution
4. Monitor and tune resource usage

### Error Code Reference

#### Security Errors
- `SEC001`: PII leakage detected
- `SEC002`: Compliance violation
- `SEC003`: Session isolation breach
- `SEC004`: Credential security error
- `SEC005`: Unauthorized access attempt

#### Integration Errors
- `INT001`: AgentCore session creation failed
- `INT002`: WebSocket connection failed
- `INT003`: Browser-use agent initialization failed
- `INT004`: CDP communication error
- `INT005`: Session cleanup failed

#### Performance Errors
- `PERF001`: Task execution timeout
- `PERF002`: Resource limit exceeded
- `PERF003`: Session pool exhausted
- `PERF004`: Memory usage critical

### Monitoring and Alerting

#### Health Check Endpoints

```python
# health_checks.py
from fastapi import FastAPI, HTTPException
from typing import Dict, Any

app = FastAPI()

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Basic health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """Readiness check with dependency validation."""
    
    checks = {
        "aws_credentials": False,
        "agentcore_access": False,
        "bedrock_access": False
    }
    
    # Check AWS credentials
    try:
        boto3.client('sts').get_caller_identity()
        checks["aws_credentials"] = True
    except:
        pass
    
    # Check AgentCore access
    try:
        from bedrock_agentcore.tools.browser_client import BrowserClient
        BrowserClient(region=os.getenv('AWS_REGION'))
        checks["agentcore_access"] = True
    except:
        pass
    
    # Check Bedrock access
    try:
        boto3.client('bedrock-runtime', region_name=os.getenv('AWS_REGION'))
        checks["bedrock_access"] = True
    except:
        pass
    
    all_healthy = all(checks.values())
    
    if not all_healthy:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "status": "ready",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/metrics")
async def metrics_endpoint() -> Dict[str, Any]:
    """Metrics endpoint for monitoring."""
    
    # Collect metrics
    metrics = {
        "active_sessions": len(session_pool.sessions),
        "available_sessions": len(session_pool.available_sessions),
        "total_requests": performance_optimizer.metrics['total_requests'],
        "cache_hit_rate": (
            performance_optimizer.metrics['cache_hits'] / 
            max(1, performance_optimizer.metrics['cache_hits'] + 
                performance_optimizer.metrics['cache_misses'])
        ),
        "avg_response_time": performance_optimizer.metrics['avg_response_time'],
        "system_metrics": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    }
    
    return metrics
```

This comprehensive deployment and troubleshooting guide provides production-ready guidance for deploying browser-use with AgentCore Browser Tool, including performance optimization strategies and detailed troubleshooting procedures for sensitive information scenarios.