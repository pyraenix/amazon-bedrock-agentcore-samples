# NovaAct with Amazon Bedrock AgentCore Browser Tool - Sensitive Information Handling

## Overview

This tutorial demonstrates **how NovaAct's natural language browser automation handles sensitive information when integrated with Amazon Bedrock AgentCore Browser Tool**. You'll learn to build secure, production-ready browser automation solutions that leverage NovaAct's AI-powered natural language processing within AgentCore's managed, containerized browser environment.

## What You'll Learn

- **Real NovaAct-AgentCore Integration**: Working examples using actual NovaAct SDK with AgentCore's browser_session()
- **Secure Sensitive Data Handling**: How NovaAct's AI processes credentials, PII, and payment information within AgentCore's isolated environment
- **Production Security Patterns**: Enterprise-grade patterns for scaling NovaAct automation using AgentCore's managed infrastructure
- **Natural Language Automation**: Leveraging NovaAct's agentic approach for complex web workflows with built-in security

## Architecture Overview

### NovaAct-AgentCore Integration Flow

```
Developer → NovaAct SDK → AgentCore Browser Tool → Secure Browser Session → Web Application
    ↓           ↓              ↓                      ↓                    ↓
Natural     AI Model      Managed Browser        Containerized         Target Site
Language    Processing    Infrastructure         Environment           with Forms
```

### Key Integration Benefits

#### 🔒 **Enhanced Security Through Integration**
1. **Dual Security Layer**: NovaAct's AI processing + AgentCore's containerized isolation
2. **Credential Isolation**: Separate, secure management of NovaAct API keys and AgentCore credentials
3. **Session-Level Protection**: Each NovaAct operation runs within AgentCore's isolated browser session
4. **AI Processing Security**: NovaAct's natural language model processes instructions within secure boundaries
5. **Zero Data Persistence**: AgentCore's ephemeral containers ensure no sensitive data remains after sessions

#### 🚀 **Operational Excellence**
1. **Managed Infrastructure**: AgentCore handles browser scaling, monitoring, and security automatically
2. **Auto-Scaling**: AgentCore's infrastructure scales NovaAct operations based on demand
3. **Built-in Observability**: Real-time monitoring without exposing sensitive data
4. **Automatic Cleanup**: Session termination and resource management handled automatically
5. **Enterprise-Grade Reliability**: Production-ready infrastructure with SLA guarantees

#### 🤖 **AI-Powered Automation**
1. **Natural Language Automation**: NovaAct's AI understands complex instructions for sensitive workflows
2. **Context-Aware Processing**: AI model maintains context across multi-step workflows
3. **Intelligent Error Handling**: AI can adapt to unexpected page changes and errors
4. **Dynamic Form Recognition**: Automatically identifies and fills forms regardless of structure
5. **Smart Wait Strategies**: AI determines optimal wait times for page loads and element availability

#### 🏢 **Enterprise Integration**
1. **AWS Native**: Seamless integration with existing AWS infrastructure and security policies
2. **Compliance Ready**: Built-in features support SOC2, HIPAA, and other compliance requirements
3. **Audit Trail**: Comprehensive logging for security and compliance auditing
4. **Role-Based Access**: Integration with AWS IAM for fine-grained access control
5. **Multi-Region Support**: Deploy across multiple AWS regions for global availability

### Detailed Architecture Benefits

#### **NovaAct AI Model Integration**
```
NovaAct Natural Language Processing within AgentCore Security Boundary:

┌─────────────────────────────────────────────────────────────┐
│ AgentCore Secure Container                                  │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ NovaAct AI Model                                        │ │
│ │ • Processes natural language instructions               │ │
│ │ • Generates browser automation commands                 │ │
│ │ • Maintains workflow context securely                  │ │
│ │ • Handles errors and retries intelligently             │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Managed Browser Instance                                │ │
│ │ • Executes AI-generated commands                        │ │
│ │ • Isolated from other sessions                          │ │
│ │ • Automatic screenshot redaction                        │ │
│ │ • Secure credential handling                            │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

#### **Security Architecture Integration**
- **Network Isolation**: NovaAct AI processing occurs within AgentCore's VPC
- **Process Isolation**: AI model runs in separate process namespace within container
- **Memory Protection**: Secure memory allocation and cleanup for AI processing
- **API Security**: Encrypted communication between NovaAct SDK and AgentCore infrastructure
- **Credential Segregation**: NovaAct API keys and AgentCore credentials managed separately

#### **Scalability Architecture**
- **Horizontal Scaling**: AgentCore automatically scales browser instances for NovaAct operations
- **Load Balancing**: Intelligent distribution of NovaAct sessions across available resources
- **Resource Optimization**: Dynamic resource allocation based on AI processing requirements
- **Geographic Distribution**: Multi-region deployment for global NovaAct automation
- **Cost Optimization**: Pay-per-use model with automatic resource cleanup

## Tutorial Structure

### 📚 Progressive Learning Path

#### 🚀 **Notebook 1: NovaAct-AgentCore Secure Login** (`01_novaact_agentcore_secure_login.ipynb`)
**Focus**: Fundamental integration between NovaAct's AI and AgentCore's secure browser sessions
- How NovaAct connects to AgentCore's CDP endpoints
- AgentCore's containerized browser isolation protecting NovaAct operations
- Secure credential flow between NovaAct SDK and AgentCore infrastructure

#### 🔐 **Notebook 2: NovaAct Sensitive Form Automation** (`02_novaact_sensitive_form_automation.ipynb`)
**Focus**: NovaAct's natural language processing of PII within AgentCore's secure sessions
- PII handling with NovaAct's AI within AgentCore's isolated environment
- AgentCore's built-in data protection features during NovaAct operations
- Secure payment form automation using natural language instructions

#### 🛡️ **Notebook 3: NovaAct-AgentCore Session Security** (`03_novaact_agentcore_session_security.ipynb`)
**Focus**: Session lifecycle management and AgentCore's built-in security features
- AgentCore session lifecycle protecting NovaAct operations
- Error handling that maintains session security
- AgentCore's observability integration for monitoring NovaAct operations

#### 🏭 **Notebook 4: Production NovaAct-AgentCore Patterns** (`04_production_novaact_agentcore_patterns.ipynb`)
**Focus**: Scaling NovaAct automation using AgentCore's managed infrastructure
- Production-ready integration patterns
- AgentCore's auto-scaling capabilities for NovaAct operations
- Monitoring and alerting using AgentCore's built-in dashboards

### 🛠️ **Supporting Examples**

#### Real-World Integration Examples (`examples/`)
- `secure_login_with_novaact.py` - Complete login workflow with NovaAct-AgentCore integration
- `pii_form_automation.py` - PII handling using NovaAct's natural language within AgentCore sessions
- `payment_form_security.py` - Secure payment processing patterns
- `agentcore_session_helpers.py` - AgentCore session utilities and context managers

#### Visual Architecture (`assets/`)
- [`novaact_agentcore_architecture.md`](assets/novaact_agentcore_architecture.md) - Complete integration architecture with Mermaid diagrams
- [`security_flow_diagram.md`](assets/security_flow_diagram.md) - Data protection flow and security layers visualization
- [`containerized_isolation.md`](assets/containerized_isolation.md) - AgentCore's container isolation architecture

#### Architecture Diagrams Overview

**Integration Architecture**: Shows how NovaAct SDK connects to AgentCore Browser Tool, including:
- Developer environment setup and SDK integration
- AgentCore's managed infrastructure components
- NovaAct AI service integration points
- Security layers and data protection mechanisms
- Target web application interaction patterns

**Security Flow**: Illustrates the complete security model, including:
- Session initialization and authentication flow
- NovaAct AI processing within secure boundaries
- Data protection mechanisms during sensitive operations
- Container isolation and network security
- Automatic cleanup and session termination

**Container Isolation**: Details AgentCore's containerization approach:
- Per-session container isolation
- Resource management and scaling
- Network isolation and security groups
- Process and storage isolation mechanisms
- Monitoring and observability integration

## Key Security Features

### NovaAct-AgentCore Security Integration

1. **Containerized AI Processing**: NovaAct's AI model processes instructions within AgentCore's isolated containers
2. **Secure CDP Connection**: NovaAct connects to AgentCore's managed browser via secure CDP endpoints
3. **Credential Isolation**: Both NovaAct API keys and AgentCore credentials are securely managed
4. **Session-Level Protection**: Each NovaAct operation runs within AgentCore's isolated browser session
5. **Built-in Monitoring**: AgentCore's observability tracks NovaAct operations without exposing sensitive data

### Production Security Patterns

- **Dual Authentication**: Secure management of NovaAct API keys and AgentCore credentials
- **Managed Infrastructure**: Leveraging AgentCore's fully managed browser infrastructure
- **Natural Language Security**: NovaAct's AI processes sensitive instructions without exposing data
- **Enterprise Compliance**: Meeting enterprise security requirements through AgentCore's isolation

## Prerequisites

### Required Services
- **NovaAct API Access**: Active NovaAct subscription with API key
- **Amazon Bedrock AgentCore**: Access to AgentCore Browser Tool service
- **AWS Account**: Configured with appropriate permissions for AgentCore

### Technical Requirements
- Python 3.8+
- Jupyter Notebook environment
- AWS CLI configured
- Basic understanding of browser automation concepts

### Recommended Background
- Familiarity with NovaAct's natural language automation
- Understanding of AgentCore Browser Tool concepts
- Experience with secure credential management

## Setup Instructions

### 1. NovaAct SDK Setup

#### Obtain NovaAct API Access
1. Sign up for NovaAct at [https://nova-act.com](https://nova-act.com)
2. Navigate to your account dashboard
3. Generate an API key from the "API Keys" section
4. Copy your API key for environment configuration

#### Install NovaAct SDK
```bash
# Install NovaAct SDK
pip install nova-act

# Verify installation
python -c "import nova_act; print('NovaAct SDK installed successfully')"
```

### 2. AgentCore Browser Client SDK Setup

#### Configure AWS Credentials
```bash
# Configure AWS CLI (if not already done)
aws configure

# Verify AgentCore access
aws bedrock list-foundation-models --region us-east-1
```

#### Install AgentCore Browser Client SDK
```bash
# Install AgentCore Browser Client SDK
pip install bedrock-agentcore-browser-client

# Verify installation
python -c "from bedrock_agentcore.tools.browser_client import browser_session; print('AgentCore Browser Client SDK installed successfully')"
```

### 3. Environment Configuration

#### Set Environment Variables
```bash
# NovaAct configuration
export NOVA_ACT_API_KEY="your-novaact-api-key-here"

# AWS/AgentCore configuration
export AWS_REGION="us-east-1"
export AWS_PROFILE="your-aws-profile"  # Optional, if using named profiles

# Optional: Enable debug logging
export NOVA_ACT_DEBUG="true"
export AGENTCORE_DEBUG="true"
```

#### Create Configuration File (Optional)
```python
# config.py
import os

# NovaAct configuration
NOVA_ACT_CONFIG = {
    'api_key': os.environ.get('NOVA_ACT_API_KEY'),
    'timeout': 30,
    'retry_attempts': 3
}

# AgentCore configuration
AGENTCORE_CONFIG = {
    'region': os.environ.get('AWS_REGION', 'us-east-1'),
    'session_timeout': 300,
    'enable_observability': True
}
```

### 4. Verify Integration Setup

#### Test NovaAct API Connection
```python
import os
from nova_act import NovaAct

# Test NovaAct API connection
try:
    with NovaAct(nova_act_api_key=os.environ['NOVA_ACT_API_KEY']) as nova_act:
        print("✅ NovaAct API connection successful")
except Exception as e:
    print(f"❌ NovaAct API connection failed: {e}")
```

#### Test AgentCore Browser Tool Connection
```python
from bedrock_agentcore.tools.browser_client import browser_session

# Test AgentCore Browser Tool connection
try:
    with browser_session(region='us-east-1') as client:
        ws_url, headers = client.generate_ws_headers()
        print("✅ AgentCore Browser Tool connection successful")
        print(f"CDP Endpoint: {ws_url[:50]}...")
except Exception as e:
    print(f"❌ AgentCore Browser Tool connection failed: {e}")
```

#### Test Complete Integration
```python
import os
from bedrock_agentcore.tools.browser_client import browser_session
from nova_act import NovaAct

# Test complete NovaAct-AgentCore integration
try:
    with browser_session(region='us-east-1') as agentcore_client:
        ws_url, headers = agentcore_client.generate_ws_headers()
        
        with NovaAct(
            cdp_endpoint_url=ws_url,
            cdp_headers=headers,
            nova_act_api_key=os.environ['NOVA_ACT_API_KEY']
        ) as nova_act:
            print("✅ Complete NovaAct-AgentCore integration successful")
except Exception as e:
    print(f"❌ Integration test failed: {e}")
```

## Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export NOVA_ACT_API_KEY="your-novaact-api-key"
export AWS_REGION="us-east-1"
```

### 2. Basic NovaAct-AgentCore Integration

```python
import os
from bedrock_agentcore.tools.browser_client import browser_session
from nova_act import NovaAct

# AgentCore provides secure, managed browser infrastructure
with browser_session(region='us-east-1') as agentcore_client:
    # Get secure CDP connection to AgentCore's managed browser
    ws_url, headers = agentcore_client.generate_ws_headers()
    
    # NovaAct connects to AgentCore's secure browser environment
    with NovaAct(
        cdp_endpoint_url=ws_url,
        cdp_headers=headers,
        nova_act_api_key=os.environ['NOVA_ACT_API_KEY']
    ) as nova_act:
        # NovaAct's AI processes natural language within AgentCore's secure environment
        result = nova_act.act("Navigate to the login page and log in securely")
        print(f"Secure automation completed: {result.success}")
```

### 3. Start with Tutorial Notebooks

Begin with `01_novaact_agentcore_secure_login.ipynb` to understand the fundamental integration patterns.

## Tutorial Learning Objectives

### By the end of this tutorial, you will:

✅ **Understand NovaAct-AgentCore Integration**: How NovaAct's AI model works within AgentCore's managed browser infrastructure

✅ **Implement Secure Automation**: Build browser automation that handles sensitive data using natural language instructions

✅ **Leverage Dual Security**: Combine NovaAct's AI processing security with AgentCore's containerized isolation

✅ **Scale Production Workloads**: Use AgentCore's managed infrastructure to scale NovaAct automation securely

✅ **Monitor Operations Safely**: Utilize AgentCore's built-in observability without exposing sensitive data

## Security Best Practices Covered

### Integration-Specific Security
- Secure connection patterns between NovaAct SDK and AgentCore Browser Tool
- Proper credential management for both services
- Session isolation and cleanup patterns
- Error handling that protects sensitive data

### Production Deployment
- AWS Secrets Manager integration for credential management
- AgentCore's auto-scaling for NovaAct operations
- Monitoring and alerting without data exposure
- Compliance logging patterns

## Troubleshooting Guide

### Common Integration Issues

#### 1. NovaAct API Connection Issues

**Problem**: `NovaActAuthenticationError: Invalid API key`
```
Solution:
1. Verify your NovaAct API key is correct
2. Check if the API key has expired
3. Ensure the API key has proper permissions
4. Verify environment variable is set correctly:
   echo $NOVA_ACT_API_KEY
```

**Problem**: `NovaActTimeoutError: Request timeout`
```
Solution:
1. Check your internet connection
2. Verify NovaAct service status
3. Increase timeout in configuration:
   nova_act = NovaAct(timeout=60)
4. Check for firewall blocking outbound connections
```

#### 2. AgentCore Browser Tool Issues

**Problem**: `AgentCoreAuthenticationError: AWS credentials not found`
```
Solution:
1. Configure AWS credentials:
   aws configure
2. Verify AWS profile:
   aws sts get-caller-identity
3. Check IAM permissions for AgentCore Browser Tool
4. Ensure correct AWS region is set
```

**Problem**: `AgentCoreBrowserSessionError: Failed to create browser session`
```
Solution:
1. Verify AgentCore Browser Tool is available in your region
2. Check AWS service limits and quotas
3. Ensure proper IAM permissions:
   - bedrock:InvokeModel
   - agentcore:CreateBrowserSession
   - agentcore:GetBrowserSession
4. Try a different AWS region if available
```

#### 3. Integration-Specific Issues

**Problem**: `CDPConnectionError: Failed to connect to CDP endpoint`
```
Solution:
1. Verify AgentCore browser session is active
2. Check CDP endpoint URL format
3. Ensure headers are properly set:
   ws_url, headers = agentcore_client.generate_ws_headers()
4. Verify network connectivity to CDP endpoint
```

**Problem**: `NovaActCDPError: CDP command failed`
```
Solution:
1. Check if browser session is still active
2. Verify CDP commands are valid
3. Ensure browser hasn't crashed or been terminated
4. Restart the AgentCore browser session
```

#### 4. Session Management Issues

**Problem**: `SessionTimeoutError: Browser session expired`
```
Solution:
1. Implement proper session lifecycle management
2. Use context managers for automatic cleanup
3. Monitor session duration and renew as needed
4. Implement retry logic for session failures
```

**Problem**: `ResourceLimitError: Too many concurrent sessions`
```
Solution:
1. Implement session pooling and reuse
2. Add proper session cleanup
3. Monitor concurrent session usage
4. Request quota increase if needed
```

### Debugging Tips

#### Enable Debug Logging
```python
import logging

# Enable debug logging for both services
logging.basicConfig(level=logging.DEBUG)

# NovaAct debug logging
os.environ['NOVA_ACT_DEBUG'] = 'true'

# AgentCore debug logging
os.environ['AGENTCORE_DEBUG'] = 'true'
```

#### Monitor Session Health
```python
def monitor_session_health(agentcore_client, nova_act):
    """Monitor the health of NovaAct-AgentCore integration."""
    try:
        # Check AgentCore session
        session_info = agentcore_client.get_session_info()
        print(f"AgentCore session status: {session_info.get('status')}")
        
        # Check NovaAct connection
        health_check = nova_act.health_check()
        print(f"NovaAct connection status: {health_check.get('status')}")
        
        return True
    except Exception as e:
        print(f"Health check failed: {e}")
        return False
```

#### Safe Error Handling Pattern
```python
import logging
from contextlib import contextmanager

@contextmanager
def safe_novaact_agentcore_session():
    """Safe session management with proper error handling."""
    agentcore_client = None
    nova_act = None
    
    try:
        # Create AgentCore session
        agentcore_client = browser_session(region='us-east-1').__enter__()
        ws_url, headers = agentcore_client.generate_ws_headers()
        
        # Create NovaAct connection
        nova_act = NovaAct(
            cdp_endpoint_url=ws_url,
            cdp_headers=headers,
            nova_act_api_key=os.environ['NOVA_ACT_API_KEY']
        ).__enter__()
        
        yield agentcore_client, nova_act
        
    except Exception as e:
        logging.error(f"Session error: {e}")
        raise
    finally:
        # Ensure proper cleanup
        if nova_act:
            try:
                nova_act.__exit__(None, None, None)
            except Exception as e:
                logging.warning(f"NovaAct cleanup error: {e}")
        
        if agentcore_client:
            try:
                agentcore_client.__exit__(None, None, None)
            except Exception as e:
                logging.warning(f"AgentCore cleanup error: {e}")
```

### Performance Optimization

#### Session Reuse Pattern
```python
class NovaActAgentCoreManager:
    """Manage NovaAct-AgentCore sessions efficiently."""
    
    def __init__(self, region='us-east-1'):
        self.region = region
        self.session_pool = {}
    
    def get_session(self, session_id=None):
        """Get or create a NovaAct-AgentCore session."""
        if session_id and session_id in self.session_pool:
            return self.session_pool[session_id]
        
        # Create new session
        agentcore_client = browser_session(region=self.region).__enter__()
        ws_url, headers = agentcore_client.generate_ws_headers()
        
        nova_act = NovaAct(
            cdp_endpoint_url=ws_url,
            cdp_headers=headers,
            nova_act_api_key=os.environ['NOVA_ACT_API_KEY']
        ).__enter__()
        
        session = {
            'agentcore_client': agentcore_client,
            'nova_act': nova_act,
            'created_at': time.time()
        }
        
        if session_id:
            self.session_pool[session_id] = session
        
        return session
    
    def cleanup_expired_sessions(self, max_age=300):
        """Clean up expired sessions."""
        current_time = time.time()
        expired_sessions = [
            sid for sid, session in self.session_pool.items()
            if current_time - session['created_at'] > max_age
        ]
        
        for session_id in expired_sessions:
            self.close_session(session_id)
    
    def close_session(self, session_id):
        """Close a specific session."""
        if session_id in self.session_pool:
            session = self.session_pool[session_id]
            try:
                session['nova_act'].__exit__(None, None, None)
                session['agentcore_client'].__exit__(None, None, None)
            except Exception as e:
                logging.warning(f"Session cleanup error: {e}")
            finally:
                del self.session_pool[session_id]
```

### Security Considerations

#### Credential Security Checklist
- [ ] NovaAct API keys stored securely (not in code)
- [ ] AWS credentials properly configured
- [ ] Environment variables used for sensitive data
- [ ] Proper IAM permissions (least privilege)
- [ ] Session timeouts configured appropriately
- [ ] Debug logging disabled in production
- [ ] Error messages don't expose sensitive data

#### Production Deployment Checklist
- [ ] All credentials managed via AWS Secrets Manager
- [ ] Monitoring and alerting configured
- [ ] Session lifecycle properly managed
- [ ] Resource limits and quotas configured
- [ ] Error handling doesn't expose sensitive data
- [ ] Logging configured for compliance requirements
- [ ] Network security properly configured

## Support and Resources

### Documentation
- [NovaAct SDK Documentation](https://docs.nova-act.com)
- [AgentCore Browser Tool Documentation](../README.md)
- [AWS Bedrock AgentCore Documentation](../../README.md)

### Community
- Report issues or ask questions in the repository discussions
- Contribute improvements via pull requests
- Share your NovaAct-AgentCore integration patterns

### Getting Help
1. **Check this troubleshooting guide first**
2. **Review the tutorial notebooks for working examples**
3. **Check AWS CloudWatch logs for AgentCore issues**
4. **Review NovaAct API documentation for SDK issues**
5. **Open an issue with detailed error information and steps to reproduce**

## Next Steps

1. **Start with Notebook 1**: Begin with basic NovaAct-AgentCore integration patterns
2. **Progress Through Tutorials**: Follow the progressive learning path for comprehensive understanding
3. **Implement Production Patterns**: Apply the patterns to your specific use cases
4. **Contribute Back**: Share your integration experiences and improvements

---

**Ready to get started?** Open `01_novaact_agentcore_secure_login.ipynb` to begin your journey with secure NovaAct-AgentCore browser automation!