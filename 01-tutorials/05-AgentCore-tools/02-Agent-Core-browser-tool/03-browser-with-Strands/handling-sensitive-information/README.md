# Strands with AgentCore Browser Tool - Sensitive Information Handling Tutorial

This comprehensive tutorial demonstrates how Strands agents securely handle sensitive information when integrated with Amazon Bedrock AgentCore Browser Tool. Learn production-ready patterns for secure data handling, including PII protection, credential management, and secure browser automation workflows that meet enterprise security requirements.

## Overview

This tutorial provides real-world integration examples that developers can implement in production environments. Unlike mock demonstrations, these examples showcase actual implementations with:

### Core Security Features
- **Secure Credential Management**: Industry-standard patterns for handling authentication credentials without exposure
- **PII Detection and Masking**: Automatic identification and protection of personally identifiable information
- **Containerized Browser Isolation**: Secure, isolated browser sessions that prevent data leakage
- **Multi-LLM Security Routing**: Intelligent routing between different Bedrock models based on data sensitivity
- **Production Monitoring and Compliance**: Enterprise-grade observability and audit capabilities

### Strands Framework Advantages with AgentCore Browser Tool

#### Code-First Security Architecture
- **Custom Security Tools**: Build domain-specific security tools tailored to your requirements
- **Policy-as-Code**: Define security policies programmatically with full version control
- **Flexible Workflow Orchestration**: Create complex multi-step workflows with granular security controls
- **Extensible Framework**: Easily integrate with existing security infrastructure and tools

#### Multi-LLM Flexibility and Security
- **Provider-Agnostic Design**: Switch between Amazon Bedrock models (Claude, Llama, Titan) based on security requirements
- **Intelligent Model Routing**: Automatically route sensitive data to appropriate models based on compliance needs
- **Cost Optimization**: Use different models for different sensitivity levels to optimize costs
- **Fallback Mechanisms**: Robust fallback systems that maintain security levels during model failures

#### Enterprise Integration Benefits
- **AWS Native**: Seamless integration with AWS security and monitoring services
- **Scalable Architecture**: Support for concurrent sessions and high-throughput operations
- **Compliance Ready**: Built-in support for HIPAA, PCI DSS, and other regulatory requirements
- **Observability**: Comprehensive monitoring and alerting capabilities with custom metrics

#### AgentCore Browser Tool Security Benefits
- **Containerized Isolation**: Each browser session runs in a dedicated, isolated container
- **Network Security**: Controlled network access with security boundaries
- **Resource Management**: Automatic cleanup and resource management
- **Session Lifecycle**: Proper session creation, management, and termination

## Tutorial Structure

### 1. Secure Integration Fundamentals (Notebook 1)
- **File**: `01_strands_agentcore_secure_login.ipynb`
- **Focus**: Basic Strands-AgentCore Browser Tool integration with secure authentication
- **Topics**: Secure session creation, credential injection, multi-LLM configuration, security verification

### 2. Sensitive Form Automation (Notebook 2)
- **File**: `02_strands_sensitive_form_automation.ipynb`
- **Focus**: PII handling and form automation with real-time data protection
- **Topics**: PII detection, secure form filling, data masking, encrypted storage

### 3. Multi-Model Security Framework (Notebook 3)
- **File**: `03_strands_bedrock_multi_model_security.ipynb`
- **Focus**: Bedrock model routing based on data sensitivity and compliance requirements
- **Topics**: Model security policies, intelligent routing, fallback mechanisms, cross-model audit trails

### 4. Production Deployment Patterns (Notebook 4)
- **File**: `04_production_strands_agentcore_patterns.ipynb`
- **Focus**: Enterprise deployment, monitoring, and compliance
- **Topics**: Observability, error handling, compliance reporting, performance optimization

## Prerequisites

### System Requirements
- **Python**: 3.12 or higher (required for latest Strands SDK features)
- **Operating System**: Linux, macOS, or Windows with WSL2
- **Memory**: Minimum 8GB RAM (16GB recommended for production)
- **Network**: Stable internet connection for AWS services

### AWS Account Requirements
- **AWS Account**: Active account with billing configured
- **Bedrock Access**: Enabled in your target region with access to multiple models (Claude, Llama, Titan)
- **AgentCore Browser Tool**: Access to Amazon Bedrock AgentCore services
- **IAM Permissions**: Appropriate roles and policies configured for multi-service access

## Installation

### 1. Clone and Setup Environment

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd amazon-bedrock-agentcore-samples/01-tutorials/05-AgentCore-tools/02-Agent-Core-browser-tool/05-handling-sensitive-information/Strands

# Create virtual environment with Python 3.12
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your AWS credentials and configuration
# Required variables:
# AWS_DEFAULT_REGION=us-east-1
# AWS_ACCESS_KEY_ID=your_access_key
# AWS_SECRET_ACCESS_KEY=your_secret_key
# STRANDS_API_KEY=your_strands_api_key
# BEDROCK_MODEL_IDS=anthropic.claude-3-sonnet-20240229-v1:0,meta.llama2-70b-chat-v1,amazon.titan-text-express-v1
# AGENTCORE_REGION=us-east-1
```

### 3. Validate Installation

```bash
# Run validation script
python validate_integration.py

# Run security tests
python tests/run_security_tests.py --test-type unit

# Test multi-LLM connectivity
python -c "from tools.strands_security_policies import test_multi_llm_connection; test_multi_llm_connection()"
```

### 4. Start Jupyter Environment

```bash
# Launch Jupyter Lab
jupyter lab

# Or use Jupyter Notebook
jupyter notebook
```

## Setup and Configuration

### System Requirements
- **Python**: 3.12 or higher (required for latest Strands SDK features)
- **Operating System**: Linux, macOS, or Windows with WSL2
- **Memory**: Minimum 8GB RAM (16GB recommended for production patterns)
- **Network**: Stable internet connection for AWS services

### Required Dependencies

#### Core Frameworks
```bash
# Strands framework and extensions
pip install strands-agents>=2.0.0
pip install strands-tools-bedrock
pip install strands-security-toolkit

# AgentCore Browser Client SDK
pip install bedrock-agentcore-browser-client>=1.0.0

# AWS SDK and utilities
pip install boto3>=1.34.0
pip install awscli
```

#### Security and Encryption Libraries
```bash
# Cryptography and security
pip install cryptography>=41.0.0
pip install pycryptodome
pip install python-jose[cryptography]

# Data processing and validation
pip install pydantic>=2.0.0
pip install validators
pip install python-dotenv
```

#### Multi-LLM Support Libraries
```bash
# Bedrock model support
pip install anthropic>=0.25.0
pip install langchain-aws
pip install langchain-anthropic

# Additional LLM providers (optional)
pip install openai>=1.0.0
pip install ollama
```

### AWS Setup and Configuration

#### 1. AWS Account Requirements
- **AWS Account**: Active account with billing configured
- **Bedrock Access**: Enabled in your target region with model access for:
  - Anthropic Claude 3 (Sonnet, Haiku, Opus)
  - Meta Llama 2/3 models
  - Amazon Titan models
- **AgentCore Browser Tool**: Service access enabled
- **IAM Permissions**: Comprehensive roles and policies configured

#### 2. Required IAM Permissions
Create an IAM role with the following policies:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
                "bedrock:ListFoundationModels",
                "bedrock:GetFoundationModel"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "bedrock-agentcore:CreateBrowserSession",
                "bedrock-agentcore:GetBrowserSession",
                "bedrock-agentcore:DeleteBrowserSession",
                "bedrock-agentcore:SendBrowserAction",
                "bedrock-agentcore:ListBrowserSessions"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "secretsmanager:GetSecretValue",
                "secretsmanager:CreateSecret",
                "secretsmanager:UpdateSecret"
            ],
            "Resource": "arn:aws:secretsmanager:*:*:secret:strands-agentcore/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "cloudwatch:PutMetricData"
            ],
            "Resource": "*"
        }
    ]
}
```

#### 3. Service Quotas
Ensure adequate service quotas for:
- **Bedrock Model Invocations**: 1000+ per minute per model
- **AgentCore Browser Sessions**: 20+ concurrent sessions
- **CloudWatch Logs**: Sufficient log retention and throughput
- **Secrets Manager**: Adequate secret storage and retrieval limits

### Environment Configuration

#### 1. Environment Variables Setup
```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your configuration
nano .env
```

#### 2. Required Environment Variables
```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# Strands Configuration
STRANDS_API_KEY=your_strands_api_key
STRANDS_WORKSPACE_ID=your_workspace_id
STRANDS_ENABLE_OBSERVABILITY=true

# Multi-LLM Configuration
BEDROCK_PRIMARY_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
BEDROCK_FALLBACK_MODEL=meta.llama2-70b-chat-v1
BEDROCK_COST_OPTIMIZED_MODEL=amazon.titan-text-express-v1
OPENAI_API_KEY=your_openai_key  # Optional: for OpenAI models

# AgentCore Browser Tool Configuration
AGENTCORE_BROWSER_REGION=us-east-1
AGENTCORE_SESSION_TIMEOUT=600
AGENTCORE_MAX_CONCURRENT_SESSIONS=10
AGENTCORE_ENABLE_OBSERVABILITY=true

# Security Configuration
ENCRYPTION_KEY=your_32_byte_encryption_key
AUDIT_LOG_LEVEL=INFO
ENABLE_PII_DETECTION=true
PII_CONFIDENCE_THRESHOLD=0.8
COMPLIANCE_MODE=enterprise  # Options: basic, enterprise, hipaa, pci_dss

# Performance Configuration
STRANDS_MAX_WORKERS=5
STRANDS_TIMEOUT=300
STRANDS_RETRY_ATTEMPTS=3
```

#### 3. Validation and Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Validate AWS credentials
aws sts get-caller-identity

# Test Bedrock model access
aws bedrock list-foundation-models --region us-east-1

# Run integration validation
python validate_integration.py

# Test AgentCore Browser Tool connectivity
python -c "from tools.strands_agentcore_session_helpers import test_connection; test_connection()"

# Validate multi-LLM setup
python -c "from tools.strands_security_policies import validate_multi_llm_setup; validate_multi_llm_setup()"
```

### Development Environment Setup

#### 1. Jupyter Notebook Configuration
```bash
# Install Jupyter with extensions
pip install jupyter jupyterlab ipywidgets

# Install kernel for the virtual environment
python -m ipykernel install --user --name strands-agentcore

# Start Jupyter Lab
jupyter lab
```

#### 2. IDE Configuration (Optional)
For VS Code users:
- Install Python extension
- Configure Python interpreter to use the virtual environment
- Install Jupyter extension for notebook support
- Configure AWS Toolkit extension for AWS integration
- Install Strands extension for enhanced development experience

## Security Features Demonstrated

### Advanced Data Protection
- **Multi-Layer PII Detection**: Custom Strands tools for comprehensive PII identification and masking
- **Dynamic Credential Management**: Secure credential injection with automatic rotation capabilities
- **Session Isolation**: Containerized browser sessions with proper cleanup and resource management
- **End-to-End Encryption**: Secure handling of sensitive data throughout the entire workflow

### Multi-LLM Security Framework
- **Model-Specific Security Policies**: Different security controls for different Bedrock models
- **Intelligent Security Routing**: Automatic routing of sensitive data to appropriate models
- **Cross-Model Audit Trails**: Comprehensive tracking of data across multiple LLM providers
- **Fallback Security Maintenance**: Security level preservation during model failures

### Compliance and Auditing
- **Comprehensive Audit Logging**: Complete operation tracking with custom Strands audit tools
- **Industry-Specific Compliance**: HIPAA, PCI DSS, and GDPR compliance patterns
- **Real-Time Monitoring**: Custom monitoring tools built with Strands framework
- **Automated Compliance Reporting**: Programmatic generation of compliance documentation

### Code-First Security Advantages
- **Custom Security Tool Development**: Build domain-specific security tools with Strands
- **Policy-as-Code Implementation**: Version-controlled security policies and configurations
- **Extensible Security Framework**: Easy integration with existing security infrastructure
- **Granular Control**: Fine-grained control over every aspect of security implementation

## Real-World Examples

The tutorial includes practical examples for:
- **Healthcare Document Processing**: HIPAA-compliant patient data extraction with multi-model routing
- **Financial Data Extraction**: PCI DSS compliant payment processing with intelligent model selection
- **Legal Document Analysis**: Confidentiality controls with attorney-client privilege protection
- **Customer Support Automation**: PII-protected customer service workflows with cost optimization

## Usage Examples

### Quick Start Example

```python
from tools.strands_agentcore_session_helpers import StrandsAgentCoreClient
from tools.strands_security_policies import MultiLLMSecurityManager
from strands_agents import Agent, Workflow

# Initialize secure multi-LLM client
client = StrandsAgentCoreClient(
    region='us-east-1',
    llm_configs={
        'high_security': 'anthropic.claude-3-sonnet-20240229-v1:0',
        'standard': 'meta.llama2-70b-chat-v1',
        'cost_optimized': 'amazon.titan-text-express-v1'
    }
)

# Create security manager
security_manager = MultiLLMSecurityManager(
    compliance_mode='enterprise',
    enable_audit_logging=True
)

# Create secure agent with browser tool
agent = client.create_secure_agent(
    agent_config={
        'name': 'secure_browser_agent',
        'tools': ['secure_browser_tool', 'pii_detection_tool'],
        'security_level': 'high'
    }
)

# Execute secure workflow
workflow = Workflow([
    {'action': 'navigate', 'url': 'https://example.com/sensitive-form'},
    {'action': 'fill_form', 'data': {'username': 'user'}, 'mask_pii': True},
    {'action': 'extract_data', 'sanitize': True}
])

result = agent.execute_workflow(workflow)
print(f"Workflow completed securely: {result}")
```

### Healthcare Data Processing Example

```python
from examples.healthcare_document_processing import HealthcareStrandsAgent

# HIPAA-compliant healthcare data processing
healthcare_agent = HealthcareStrandsAgent(
    compliance_mode="HIPAA",
    encryption_enabled=True,
    audit_logging=True,
    llm_routing_policy="healthcare_sensitive"
)

# Process patient records securely with model routing
patient_workflow = healthcare_agent.create_workflow([
    {'action': 'login_patient_portal', 'credentials_secret': 'healthcare/doctor_creds'},
    {'action': 'extract_patient_data', 'patient_id': '12345', 'mask_phi': True},
    {'action': 'analyze_with_appropriate_model', 'data_sensitivity': 'high'},
    {'action': 'generate_summary', 'protect_phi': True}
])

result = healthcare_agent.execute_workflow(patient_workflow)
```

### Financial Data Extraction Example

```python
from examples.financial_data_extraction import FinancialStrandsAgent

# PCI DSS compliant financial data processing with cost optimization
financial_agent = FinancialStrandsAgent(
    compliance_mode="PCI_DSS",
    tokenize_sensitive_data=True,
    cost_optimization=True
)

# Extract financial information with intelligent model routing
financial_workflow = financial_agent.create_workflow([
    {'action': 'login_banking_portal', 'mfa_enabled': True},
    {'action': 'extract_transaction_data', 'date_range': '30_days'},
    {'action': 'route_to_cost_optimized_model', 'data_type': 'transaction_summary'},
    {'action': 'analyze_spending_patterns', 'protect_account_numbers': True}
])

analysis = financial_agent.execute_workflow(financial_workflow)
```

### Multi-Agent Coordination Example

```python
from tools.strands_agentcore_session_helpers import SessionPool
from strands_agents import Agent, MultiAgentOrchestrator

# Create session pool for concurrent operations
session_pool = SessionPool(max_sessions=5, region='us-east-1')

# Create specialized agents with different security profiles
agents = {
    'data_extractor': Agent.create({
        'name': 'data_extractor',
        'tools': ['secure_browser_tool', 'data_extraction_tool'],
        'llm_model': 'cost_optimized',
        'security_level': 'standard'
    }),
    'pii_analyzer': Agent.create({
        'name': 'pii_analyzer',
        'tools': ['pii_detection_tool', 'data_classification_tool'],
        'llm_model': 'high_security',
        'security_level': 'high'
    }),
    'compliance_checker': Agent.create({
        'name': 'compliance_checker',
        'tools': ['compliance_validation_tool', 'audit_tool'],
        'llm_model': 'high_security',
        'security_level': 'maximum'
    })
}

# Orchestrate multi-agent workflow
orchestrator = MultiAgentOrchestrator(agents, session_pool)

async def process_sensitive_workflow(url, credentials):
    # Coordinate agents with proper security isolation
    workflow_result = await orchestrator.execute_coordinated_workflow({
        'extract_data': {'agent': 'data_extractor', 'url': url, 'credentials': credentials},
        'analyze_pii': {'agent': 'pii_analyzer', 'depends_on': 'extract_data'},
        'validate_compliance': {'agent': 'compliance_checker', 'depends_on': 'analyze_pii'}
    })
    
    return workflow_result
```

### Custom Security Tool Development Example

```python
from strands_agents.tools import BaseTool
from tools.strands_security_policies import SecurityPolicy

class CustomPIIDetectionTool(BaseTool):
    """Custom PII detection tool built with Strands framework."""
    
    def __init__(self, confidence_threshold=0.8, custom_patterns=None):
        super().__init__(name="custom_pii_detector")
        self.confidence_threshold = confidence_threshold
        self.custom_patterns = custom_patterns or []
        
    def execute(self, content: str, context: dict) -> dict:
        """Execute PII detection with custom logic."""
        # Custom PII detection implementation
        pii_results = self.detect_pii(content)
        
        # Apply security policies
        security_policy = SecurityPolicy.get_policy(context.get('data_type'))
        masked_content = self.apply_masking(content, pii_results, security_policy)
        
        return {
            'original_content': content,
            'masked_content': masked_content,
            'pii_detected': pii_results,
            'security_policy_applied': security_policy.name
        }

# Register custom tool with agent
agent.register_tool(CustomPIIDetectionTool(confidence_threshold=0.9))
```

## Getting Started

1. **Setup Environment**: Follow the installation instructions above
2. **Validate Installation**: Run `python validate_integration.py`
3. **Test Multi-LLM Setup**: Run `python -c "from tools.strands_security_policies import test_multi_llm_setup; test_multi_llm_setup()"`
4. **Start with Notebook 1**: Begin with basic integration patterns
5. **Progress Sequentially**: Each notebook builds on the previous one
6. **Explore Examples**: Review the supporting scripts and utilities
7. **Customize Security Tools**: Adapt the custom tools to your specific requirements

## Troubleshooting Guide

### Common Setup Issues

#### 1. AWS Credentials and Permissions
**Problem**: `NoCredentialsError` or `AccessDenied` errors
**Solutions**:
```bash
# Verify AWS credentials
aws sts get-caller-identity

# Check IAM permissions
aws iam get-user
aws iam list-attached-user-policies --user-name YOUR_USERNAME

# Test Bedrock access for multiple models
aws bedrock list-foundation-models --region us-east-1
aws bedrock get-foundation-model --model-identifier anthropic.claude-3-sonnet-20240229-v1:0
```

#### 2. Strands Framework Configuration Issues
**Problem**: Strands agent creation or tool registration fails
**Solutions**:
```python
# Test basic Strands connectivity
from strands_agents import Agent
agent = Agent.create({'name': 'test_agent'})
print(f"Agent created: {agent.id}")

# Verify Strands API key
import os
print(f"Strands API Key configured: {'STRANDS_API_KEY' in os.environ}")

# Test tool registration
from tools.strands_agentcore_session_helpers import SecureBrowserTool
tool = SecureBrowserTool()
agent.register_tool(tool)
```

#### 3. Multi-LLM Configuration Problems
**Problem**: Model routing or fallback mechanisms fail
**Solutions**:
```python
# Test individual model connectivity
from tools.strands_security_policies import MultiLLMSecurityManager

manager = MultiLLMSecurityManager()
models = ['anthropic.claude-3-sonnet-20240229-v1:0', 'meta.llama2-70b-chat-v1']

for model in models:
    try:
        result = manager.test_model_connectivity(model)
        print(f"Model {model}: {'✓' if result else '✗'}")
    except Exception as e:
        print(f"Model {model} error: {e}")
```

#### 4. AgentCore Browser Tool Connection Issues
**Problem**: Browser session creation fails
**Solutions**:
```python
# Test basic connectivity
from tools.strands_agentcore_session_helpers import StrandsAgentCoreClient
client = StrandsAgentCoreClient(region='us-east-1')
session = client.create_secure_session()
print(f"Session created: {session.session_id}")

# Test session pool
from tools.strands_agentcore_session_helpers import SessionPool
pool = SessionPool(max_sessions=2, region='us-east-1')
session = pool.get_session()
print(f"Pool session: {session}")
```

**Common Error Messages**:
- `ServiceQuotaExceededException`: Increase service quotas in AWS console
- `InvalidParameterException`: Check region availability for AgentCore
- `ThrottlingException`: Implement exponential backoff in your code
- `ModelNotAvailableException`: Verify Bedrock model access in your region

#### 5. Environment Configuration Problems
**Problem**: Environment variables not loading correctly
**Solutions**:
```bash
# Verify .env file loading
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('AWS_REGION:', os.getenv('AWS_REGION')); print('STRANDS_API_KEY configured:', bool(os.getenv('STRANDS_API_KEY')))"

# Check file permissions
ls -la .env
chmod 600 .env  # Secure the environment file

# Validate all required variables
python -c "
import os
from dotenv import load_dotenv
load_dotenv()

required_vars = ['AWS_REGION', 'STRANDS_API_KEY', 'BEDROCK_PRIMARY_MODEL', 'AGENTCORE_BROWSER_REGION']
missing = [var for var in required_vars if not os.getenv(var)]
if missing:
    print(f'Missing environment variables: {missing}')
else:
    print('All required environment variables configured')
"
```

### Specific Error Scenarios

#### Session Timeout and Management Issues
```python
# Implement proper session management with Strands
from tools.strands_agentcore_session_helpers import SessionManager

session_manager = SessionManager(
    region='us-east-1',
    timeout=600,  # 10 minutes
    max_retries=3,
    enable_monitoring=True
)

# Use context manager for automatic cleanup
with session_manager.get_session() as session:
    # Your browser automation code here
    agent = session.create_agent()
    result = agent.execute_workflow(workflow)
```

#### PII Detection False Positives/Negatives
```python
# Configure PII detection with custom patterns
from tools.strands_pii_utils import PIIDetector

detector = PIIDetector(
    confidence_threshold=0.8,  # Adjust based on your needs
    custom_patterns=[
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
        r'\b\d{16}\b'  # Credit card pattern
    ],
    whitelist=[
        r'\b\d{4}\b',  # Allow 4-digit numbers (years, etc.)
        r'test@example\.com'  # Allow test emails
    ],
    industry_specific_rules='healthcare'  # or 'financial', 'legal'
)

# Test detection accuracy
test_content = "Patient SSN: 123-45-6789, Email: patient@hospital.com"
result = detector.detect_and_mask(test_content)
print(f"Masked: {result['masked_content']}")
```

#### Multi-LLM Routing Issues
```python
# Debug model routing decisions
from tools.strands_security_policies import MultiLLMSecurityManager

manager = MultiLLMSecurityManager(debug_mode=True)

# Test routing logic
test_data = {
    'content': 'Patient has diabetes',
    'data_type': 'healthcare',
    'sensitivity_level': 'high'
}

selected_model = manager.route_request(test_data)
print(f"Selected model: {selected_model}")
print(f"Routing reason: {manager.get_routing_reason()}")
```

#### Memory and Performance Issues
```python
# Optimize for large-scale operations
from tools.strands_monitoring import ResourceMonitor
from strands_agents import Agent

# Configure resource monitoring
monitor = ResourceMonitor(
    memory_threshold=0.8,  # 80% memory usage alert
    session_limit=10,
    cleanup_interval=300  # 5 minutes
)

# Create memory-efficient agent configuration
agent_config = {
    'name': 'optimized_agent',
    'max_concurrent_workflows': 3,
    'session_reuse': True,
    'cleanup_on_completion': True,
    'memory_limit': '2GB'
}

agent = Agent.create(agent_config)
```

### Performance Optimization

#### 1. Session Pool Management
```python
# Optimize session pooling for Strands workflows
from tools.strands_agentcore_session_helpers import SessionPool

pool = SessionPool(
    max_sessions=10,
    region='us-east-1',
    cleanup_interval=300,
    health_check_interval=60,
    enable_metrics=True
)

# Monitor pool performance
pool_stats = pool.get_statistics()
print(f"Active sessions: {pool_stats['active']}")
print(f"Pool utilization: {pool_stats['utilization']}%")
```

#### 2. Multi-LLM Cost Optimization
```python
# Implement cost-aware model routing
from tools.strands_security_policies import CostOptimizedRouter

router = CostOptimizedRouter(
    cost_budget_per_hour=100.0,  # $100/hour budget
    prefer_cost_efficient=True,
    fallback_to_expensive_on_failure=True
)

# Route based on cost and performance requirements
model = router.select_model(
    data_sensitivity='medium',
    response_time_requirement='fast',
    cost_priority='high'
)
```

#### 3. Concurrent Processing with Security
```python
# Process multiple workflows concurrently with security isolation
import asyncio
from tools.strands_agentcore_session_helpers import SecureWorkflowOrchestrator

orchestrator = SecureWorkflowOrchestrator(
    max_concurrent_workflows=5,
    isolation_level='high',
    enable_audit_logging=True
)

async def process_multiple_workflows(workflows):
    tasks = [
        orchestrator.execute_secure_workflow(workflow) 
        for workflow in workflows
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

### Debugging and Logging

#### Enable Comprehensive Debug Logging
```python
import logging

# Configure logging for all components
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable specific loggers
logging.getLogger('strands_agents').setLevel(logging.DEBUG)
logging.getLogger('bedrock_agentcore').setLevel(logging.DEBUG)
logging.getLogger('strands_security').setLevel(logging.INFO)  # Avoid logging sensitive data
```

#### Monitor Resource Usage and Security Events
```python
# Comprehensive monitoring setup
from tools.strands_monitoring import SecurityMonitor, PerformanceMonitor

security_monitor = SecurityMonitor(
    alert_on_pii_exposure=True,
    alert_on_credential_issues=True,
    alert_on_compliance_violations=True
)

performance_monitor = PerformanceMonitor(
    track_session_usage=True,
    track_model_performance=True,
    track_cost_metrics=True
)

# Start monitoring
security_monitor.start()
performance_monitor.start()

# Your code here

# Generate reports
security_report = security_monitor.generate_report()
performance_report = performance_monitor.generate_report()
```

### Getting Help

#### Documentation Resources
- **Strands Documentation**: https://docs.strands.ai/
- **AWS Bedrock Documentation**: https://docs.aws.amazon.com/bedrock/
- **AgentCore Browser Tool API Reference**: See `assets/api_reference.md`
- **Security Architecture Guide**: See `assets/security_architecture.md`

#### Community Support
- **GitHub Issues**: Report bugs and feature requests
- **AWS Support**: For service-specific issues
- **Strands Community**: Discord and forums for framework discussions

#### Professional Support
For enterprise deployments and custom integrations:
- AWS Professional Services
- Strands Enterprise Support
- Custom consulting services available

## Architecture Overview

The tutorial demonstrates a comprehensive secure integration architecture that combines the flexibility of Strands with the security of AgentCore Browser Tool:

### High-Level Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Developer     │    │   Strands        │    │   AgentCore         │
│   Environment   │───▶│   Framework      │───▶│   Browser Tool      │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                              │                           │
                              ▼                           ▼
                       ┌──────────────┐           ┌──────────────┐
                       │ Multi-LLM    │           │ Secure       │
                       │ Security     │           │ Browser      │
                       │ Router       │           │ Sessions     │
                       └──────────────┘           └──────────────┘
                              │                           │
                              ▼                           ▼
                       ┌──────────────┐           ┌──────────────┐
                       │ Custom       │           │ Web Content  │
                       │ Security     │           │ & Forms      │
                       │ Tools        │           └──────────────┘
                       └──────────────┘
```

### Strands-Specific Security Architecture

The tutorial showcases Strands' unique code-first approach to security:

#### 1. Custom Security Tool Development
- **Domain-Specific Tools**: Build security tools tailored to your industry requirements
- **Policy-as-Code**: Version-controlled security policies and configurations
- **Extensible Framework**: Easy integration with existing security infrastructure
- **Granular Control**: Fine-grained control over every aspect of security implementation

#### 2. Multi-LLM Security Management
- **Provider-Agnostic Security**: Consistent security across different LLM providers
- **Intelligent Model Routing**: Route sensitive data to appropriate models based on compliance
- **Cost-Security Balance**: Optimize costs while maintaining required security levels
- **Fallback Security**: Maintain security levels during model failures

#### 3. Workflow Security Orchestration
- **Secure State Management**: Encrypted state management across workflow steps
- **Multi-Agent Coordination**: Secure coordination between multiple agents
- **Session Pool Management**: Efficient and secure session reuse
- **Error Recovery**: Security-preserving error recovery mechanisms

For detailed architecture diagrams, security specifications, and deployment guides, see the `assets/` directory.

## Why Choose Strands for Sensitive Information Handling?

### Code-First Advantages
- **Custom Security Implementation**: Build exactly the security controls you need
- **Policy Flexibility**: Implement complex, industry-specific security policies
- **Integration Freedom**: Integrate with any existing security infrastructure
- **Full Control**: Complete control over agent behavior and security decisions

### Multi-LLM Benefits
- **Provider Independence**: Not locked into a single LLM provider
- **Cost Optimization**: Use different models for different sensitivity levels
- **Risk Mitigation**: Reduce dependency on single provider availability
- **Compliance Flexibility**: Meet different compliance requirements with appropriate models

### Enterprise Readiness
- **Production Scalability**: Built for enterprise-scale deployments
- **Security First**: Security considerations built into every component
- **Observability**: Comprehensive monitoring and audit capabilities
- **Compliance Support**: Built-in support for major compliance frameworks

This tutorial demonstrates how Strands' unique approach to agent development provides unmatched flexibility and control for handling sensitive information in production environments.