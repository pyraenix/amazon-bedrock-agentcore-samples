# API Reference - Strands AgentCore Browser Tool Integration

## Overview

This document provides comprehensive API reference for all custom tools and utilities developed for the Strands-AgentCore Browser Tool integration. These APIs enable secure handling of sensitive information while maintaining the flexibility and power of the Strands framework.

## Core Integration APIs

### StrandsAgentCoreClient

Main client for integrating Strands agents with AgentCore Browser Tool.

```python
class StrandsAgentCoreClient:
    """Main integration client for Strands-AgentCore Browser Tool."""
    
    def __init__(self, region: str, llm_configs: Dict[str, dict] = None, 
                 security_config: dict = None):
        """
        Initialize the Strands-AgentCore client.
        
        Args:
            region (str): AWS region for AgentCore services
            llm_configs (Dict[str, dict], optional): LLM configuration mapping
            security_config (dict, optional): Security configuration
        """
        
    def create_secure_agent(self, agent_config: dict) -> StrandsAgent:
        """
        Create a secure Strands agent with AgentCore Browser Tool integration.
        
        Args:
            agent_config (dict): Agent configuration including:
                - name (str): Agent name
                - tools (List[str]): List of tool names to register
                - security_level (str): Security level ('basic', 'standard', 'high', 'maximum')
                - llm_model (str, optional): Specific LLM model to use
                - compliance_requirements (List[str], optional): Compliance frameworks
                
        Returns:
            StrandsAgent: Configured agent instance
            
        Raises:
            SecurityConfigurationError: If security configuration is invalid
            AgentCreationError: If agent creation fails
        """
        
    def create_secure_session(self, session_config: dict = None) -> BrowserSession:
        """
        Create a secure browser session.
        
        Args:
            session_config (dict, optional): Session configuration including:
                - timeout (int): Session timeout in seconds
                - isolation_level (str): Isolation level
                - enable_monitoring (bool): Enable session monitoring
                
        Returns:
            BrowserSession: Secure browser session instance
            
        Raises:
            SessionCreationError: If session creation fails
        """
        
    def register_custom_tool(self, tool_class: Type[StrandsTool], 
                           tool_config: dict = None) -> None:
        """
        Register a custom security tool.
        
        Args:
            tool_class (Type[StrandsTool]): Tool class to register
            tool_config (dict, optional): Tool-specific configuration
            
        Raises:
            ToolRegistrationError: If tool registration fails
        """
        
    def execute_secure_workflow(self, workflow: Workflow, 
                               context: dict = None) -> WorkflowResult:
        """
        Execute a secure workflow with comprehensive security controls.
        
        Args:
            workflow (Workflow): Workflow definition
            context (dict, optional): Execution context
            
        Returns:
            WorkflowResult: Workflow execution result
            
        Raises:
            WorkflowExecutionError: If workflow execution fails
            SecurityViolationError: If security violation is detected
        """
```

### SecureBrowserTool

Custom Strands tool for secure browser automation.

```python
class SecureBrowserTool(StrandsTool):
    """Secure browser automation tool for Strands agents."""
    
    def __init__(self, session_config: dict = None, security_policy: SecurityPolicy = None):
        """
        Initialize the secure browser tool.
        
        Args:
            session_config (dict, optional): Browser session configuration
            security_policy (SecurityPolicy, optional): Security policy to apply
        """
        
    def navigate(self, url: str, credentials: dict = None, 
                validate_ssl: bool = True) -> NavigationResult:
        """
        Navigate to a URL with security controls.
        
        Args:
            url (str): Target URL
            credentials (dict, optional): Authentication credentials
            validate_ssl (bool): Validate SSL certificates
            
        Returns:
            NavigationResult: Navigation result with security metadata
            
        Raises:
            NavigationError: If navigation fails
            SecurityViolationError: If security policy is violated
        """
        
    def fill_form(self, form_data: dict, mask_sensitive: bool = True,
                 validate_fields: bool = True) -> FormFillResult:
        """
        Fill form fields with PII protection.
        
        Args:
            form_data (dict): Form field data
            mask_sensitive (bool): Automatically mask sensitive data
            validate_fields (bool): Validate field data before filling
            
        Returns:
            FormFillResult: Form filling result with security metadata
            
        Raises:
            FormFillError: If form filling fails
            PIIViolationError: If PII policy is violated
        """
        
    def extract_data(self, selectors: List[str], sanitize: bool = True,
                    classify_sensitivity: bool = True) -> DataExtractionResult:
        """
        Extract data from web page with security controls.
        
        Args:
            selectors (List[str]): CSS selectors for data extraction
            sanitize (bool): Sanitize extracted data
            classify_sensitivity (bool): Classify data sensitivity
            
        Returns:
            DataExtractionResult: Extracted data with security classification
            
        Raises:
            DataExtractionError: If data extraction fails
            SensitiveDataError: If sensitive data handling fails
        """
        
    def inject_credentials(self, credentials: dict, 
                          credential_source: str = 'aws_secrets_manager') -> CredentialInjectionResult:
        """
        Securely inject credentials into web forms.
        
        Args:
            credentials (dict): Credential data or reference
            credential_source (str): Source of credentials
            
        Returns:
            CredentialInjectionResult: Injection result with audit information
            
        Raises:
            CredentialInjectionError: If credential injection fails
            SecurityViolationError: If security policy is violated
        """
```

## Security and PII Detection APIs

### PIIDetectionTool

Advanced PII detection and masking tool.

```python
class PIIDetectionTool(StrandsTool):
    """Advanced PII detection and masking tool."""
    
    def __init__(self, industry_config: str = 'general', 
                 confidence_threshold: float = 0.8,
                 custom_patterns: List[str] = None):
        """
        Initialize PII detection tool.
        
        Args:
            industry_config (str): Industry-specific configuration
            confidence_threshold (float): Minimum confidence for PII detection
            custom_patterns (List[str], optional): Custom regex patterns
        """
        
    def detect_pii(self, content: str, context: dict = None) -> PIIDetectionResult:
        """
        Detect PII in content using multiple detection methods.
        
        Args:
            content (str): Content to analyze
            context (dict, optional): Additional context for detection
            
        Returns:
            PIIDetectionResult: Detection results with confidence scores
            
        Raises:
            PIIDetectionError: If PII detection fails
        """
        
    def mask_pii(self, content: str, pii_results: PIIDetectionResult,
                masking_strategy: str = 'adaptive') -> PIIMaskingResult:
        """
        Mask detected PII using specified strategy.
        
        Args:
            content (str): Original content
            pii_results (PIIDetectionResult): PII detection results
            masking_strategy (str): Masking strategy to use
            
        Returns:
            PIIMaskingResult: Masked content with masking metadata
            
        Raises:
            PIIMaskingError: If PII masking fails
        """
        
    def classify_sensitivity(self, content: str, 
                           industry_rules: str = None) -> SensitivityClassification:
        """
        Classify content sensitivity level.
        
        Args:
            content (str): Content to classify
            industry_rules (str, optional): Industry-specific classification rules
            
        Returns:
            SensitivityClassification: Sensitivity classification result
            
        Raises:
            ClassificationError: If sensitivity classification fails
        """
```

### CredentialManager

Secure credential management system.

```python
class CredentialManager:
    """Secure credential management system."""
    
    def __init__(self, region: str = 'us-east-1', 
                 kms_key_id: str = None):
        """
        Initialize credential manager.
        
        Args:
            region (str): AWS region
            kms_key_id (str, optional): KMS key ID for encryption
        """
        
    def store_credentials(self, credential_id: str, credentials: dict,
                         metadata: dict = None) -> CredentialStorageResult:
        """
        Store credentials securely in AWS Secrets Manager.
        
        Args:
            credential_id (str): Unique credential identifier
            credentials (dict): Credential data
            metadata (dict, optional): Additional metadata
            
        Returns:
            CredentialStorageResult: Storage result with ARN
            
        Raises:
            CredentialStorageError: If credential storage fails
        """
        
    def retrieve_credentials(self, credential_id: str,
                           audit_context: dict = None) -> dict:
        """
        Retrieve credentials with audit logging.
        
        Args:
            credential_id (str): Credential identifier
            audit_context (dict, optional): Audit context information
            
        Returns:
            dict: Decrypted credential data
            
        Raises:
            CredentialRetrievalError: If credential retrieval fails
            AccessDeniedError: If access is denied
        """
        
    def rotate_credentials(self, credential_id: str,
                          new_credentials: dict = None) -> CredentialRotationResult:
        """
        Rotate credentials with zero-downtime.
        
        Args:
            credential_id (str): Credential identifier
            new_credentials (dict, optional): New credential data
            
        Returns:
            CredentialRotationResult: Rotation result with new version
            
        Raises:
            CredentialRotationError: If credential rotation fails
        """
```

## Multi-LLM Security APIs

### MultiLLMSecurityManager

Intelligent LLM routing and security management.

```python
class MultiLLMSecurityManager:
    """Multi-LLM security management and routing."""
    
    def __init__(self, region: str = 'us-east-1',
                 model_configs: Dict[str, dict] = None,
                 security_policies: Dict[str, SecurityPolicy] = None):
        """
        Initialize multi-LLM security manager.
        
        Args:
            region (str): AWS region
            model_configs (Dict[str, dict], optional): Model configurations
            security_policies (Dict[str, SecurityPolicy], optional): Security policies
        """
        
    def route_request(self, content: str, context: dict,
                     routing_criteria: dict = None) -> LLMRoutingResult:
        """
        Route request to appropriate LLM based on security requirements.
        
        Args:
            content (str): Content to process
            context (dict): Request context
            routing_criteria (dict, optional): Custom routing criteria
            
        Returns:
            LLMRoutingResult: Routing decision with selected model
            
        Raises:
            LLMRoutingError: If routing fails
            SecurityPolicyViolationError: If security policy is violated
        """
        
    def validate_model_security(self, model_id: str, 
                               security_requirements: dict) -> ModelSecurityValidation:
        """
        Validate model meets security requirements.
        
        Args:
            model_id (str): Model identifier
            security_requirements (dict): Security requirements to validate
            
        Returns:
            ModelSecurityValidation: Validation result
            
        Raises:
            ModelValidationError: If model validation fails
        """
        
    def get_fallback_model(self, primary_model: str,
                          security_level: str) -> str:
        """
        Get fallback model maintaining security level.
        
        Args:
            primary_model (str): Primary model that failed
            security_level (str): Required security level
            
        Returns:
            str: Fallback model identifier
            
        Raises:
            NoFallbackModelError: If no suitable fallback model exists
        """
```

### BedrockModelRouter

Bedrock-specific model routing with security controls.

```python
class BedrockModelRouter:
    """Bedrock model routing with security controls."""
    
    def __init__(self, region: str = 'us-east-1',
                 model_security_policies: Dict[str, dict] = None):
        """
        Initialize Bedrock model router.
        
        Args:
            region (str): AWS region
            model_security_policies (Dict[str, dict], optional): Model security policies
        """
        
    def select_model_for_content(self, content: str, 
                                sensitivity_level: str,
                                compliance_requirements: List[str] = None) -> str:
        """
        Select appropriate Bedrock model for content.
        
        Args:
            content (str): Content to process
            sensitivity_level (str): Data sensitivity level
            compliance_requirements (List[str], optional): Compliance requirements
            
        Returns:
            str: Selected Bedrock model identifier
            
        Raises:
            ModelSelectionError: If model selection fails
        """
        
    def validate_model_compliance(self, model_id: str,
                                 compliance_framework: str) -> bool:
        """
        Validate model compliance with framework.
        
        Args:
            model_id (str): Bedrock model identifier
            compliance_framework (str): Compliance framework to validate
            
        Returns:
            bool: True if model is compliant
            
        Raises:
            ComplianceValidationError: If compliance validation fails
        """
```

## Session Management APIs

### SessionPool

Efficient browser session pool management.

```python
class SessionPool:
    """Browser session pool for efficient resource management."""
    
    def __init__(self, max_sessions: int = 10, region: str = 'us-east-1',
                 cleanup_interval: int = 300, health_check_interval: int = 60):
        """
        Initialize session pool.
        
        Args:
            max_sessions (int): Maximum number of concurrent sessions
            region (str): AWS region
            cleanup_interval (int): Cleanup interval in seconds
            health_check_interval (int): Health check interval in seconds
        """
        
    async def get_session(self, session_config: dict = None) -> BrowserSession:
        """
        Get a browser session from the pool.
        
        Args:
            session_config (dict, optional): Session configuration
            
        Returns:
            BrowserSession: Available browser session
            
        Raises:
            SessionPoolExhaustedError: If no sessions are available
            SessionCreationError: If session creation fails
        """
        
    async def return_session(self, session: BrowserSession) -> None:
        """
        Return a session to the pool.
        
        Args:
            session (BrowserSession): Session to return
            
        Raises:
            SessionReturnError: If session return fails
        """
        
    def get_pool_statistics(self) -> PoolStatistics:
        """
        Get session pool statistics.
        
        Returns:
            PoolStatistics: Current pool statistics
        """
        
    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.
        
        Returns:
            int: Number of sessions cleaned up
        """
```

### SessionHealthMonitor

Monitor session health and performance.

```python
class SessionHealthMonitor:
    """Monitor browser session health and performance."""
    
    def __init__(self, session_pool: SessionPool,
                 monitoring_config: dict = None):
        """
        Initialize session health monitor.
        
        Args:
            session_pool (SessionPool): Session pool to monitor
            monitoring_config (dict, optional): Monitoring configuration
        """
        
    def start_monitoring(self) -> None:
        """Start health monitoring."""
        
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        
    def get_session_health(self, session_id: str) -> SessionHealth:
        """
        Get health status of specific session.
        
        Args:
            session_id (str): Session identifier
            
        Returns:
            SessionHealth: Session health information
            
        Raises:
            SessionNotFoundError: If session is not found
        """
        
    def get_pool_health_summary(self) -> PoolHealthSummary:
        """
        Get overall pool health summary.
        
        Returns:
            PoolHealthSummary: Pool health summary
        """
```

## Workflow Orchestration APIs

### SecureWorkflowOrchestrator

Orchestrate secure multi-step workflows.

```python
class SecureWorkflowOrchestrator:
    """Orchestrate secure multi-step workflows."""
    
    def __init__(self, session_pool: SessionPool,
                 security_config: dict = None,
                 audit_config: dict = None):
        """
        Initialize workflow orchestrator.
        
        Args:
            session_pool (SessionPool): Session pool for workflow execution
            security_config (dict, optional): Security configuration
            audit_config (dict, optional): Audit configuration
        """
        
    async def execute_workflow(self, workflow: SecureWorkflow,
                              context: dict = None) -> WorkflowResult:
        """
        Execute secure workflow with comprehensive controls.
        
        Args:
            workflow (SecureWorkflow): Workflow definition
            context (dict, optional): Execution context
            
        Returns:
            WorkflowResult: Workflow execution result
            
        Raises:
            WorkflowExecutionError: If workflow execution fails
            SecurityViolationError: If security violation occurs
        """
        
    async def execute_workflow_plan(self, workflow_plan: dict,
                                   agents: Dict[str, Agent],
                                   isolation_level: str = 'high') -> dict:
        """
        Execute coordinated multi-agent workflow plan.
        
        Args:
            workflow_plan (dict): Workflow plan with dependencies
            agents (Dict[str, Agent]): Available agents
            isolation_level (str): Agent isolation level
            
        Returns:
            dict: Workflow plan execution results
            
        Raises:
            WorkflowPlanExecutionError: If workflow plan execution fails
        """
        
    def create_workflow_checkpoint(self, workflow_id: str,
                                  state: dict) -> CheckpointResult:
        """
        Create workflow checkpoint for recovery.
        
        Args:
            workflow_id (str): Workflow identifier
            state (dict): Current workflow state
            
        Returns:
            CheckpointResult: Checkpoint creation result
            
        Raises:
            CheckpointCreationError: If checkpoint creation fails
        """
```

### WorkflowStateManager

Manage workflow state with encryption and security.

```python
class WorkflowStateManager:
    """Manage workflow state with encryption and security."""
    
    def __init__(self, encryption_key_id: str,
                 storage_backend: str = 'dynamodb'):
        """
        Initialize workflow state manager.
        
        Args:
            encryption_key_id (str): KMS key ID for state encryption
            storage_backend (str): Storage backend for state persistence
        """
        
    def store_workflow_state(self, workflow_id: str, state: dict,
                           metadata: dict = None) -> StateStorageResult:
        """
        Store encrypted workflow state.
        
        Args:
            workflow_id (str): Workflow identifier
            state (dict): Workflow state data
            metadata (dict, optional): Additional metadata
            
        Returns:
            StateStorageResult: State storage result
            
        Raises:
            StateStorageError: If state storage fails
        """
        
    def retrieve_workflow_state(self, workflow_id: str,
                               checkpoint_id: str = None) -> dict:
        """
        Retrieve and decrypt workflow state.
        
        Args:
            workflow_id (str): Workflow identifier
            checkpoint_id (str, optional): Specific checkpoint to retrieve
            
        Returns:
            dict: Decrypted workflow state
            
        Raises:
            StateRetrievalError: If state retrieval fails
        """
        
    def cleanup_workflow_state(self, workflow_id: str,
                              retention_policy: dict = None) -> CleanupResult:
        """
        Clean up workflow state according to retention policy.
        
        Args:
            workflow_id (str): Workflow identifier
            retention_policy (dict, optional): Custom retention policy
            
        Returns:
            CleanupResult: Cleanup operation result
            
        Raises:
            StateCleanupError: If state cleanup fails
        """
```

## Monitoring and Audit APIs

### SecurityAuditLogger

Comprehensive security audit logging system.

```python
class SecurityAuditLogger:
    """Comprehensive security audit logging system."""
    
    def __init__(self, compliance_mode: str,
                 log_retention_days: int = 2555,
                 encryption_enabled: bool = True):
        """
        Initialize security audit logger.
        
        Args:
            compliance_mode (str): Compliance framework mode
            log_retention_days (int): Log retention period in days
            encryption_enabled (bool): Enable log encryption
        """
        
    def log_pii_detection_event(self, event: PIIDetectionEvent) -> None:
        """
        Log PII detection event with compliance requirements.
        
        Args:
            event (PIIDetectionEvent): PII detection event data
            
        Raises:
            AuditLoggingError: If audit logging fails
        """
        
    def log_credential_access_event(self, event: CredentialAccessEvent) -> None:
        """
        Log credential access event.
        
        Args:
            event (CredentialAccessEvent): Credential access event data
            
        Raises:
            AuditLoggingError: If audit logging fails
        """
        
    def log_llm_routing_event(self, event: LLMRoutingEvent) -> None:
        """
        Log LLM routing decision event.
        
        Args:
            event (LLMRoutingEvent): LLM routing event data
            
        Raises:
            AuditLoggingError: If audit logging fails
        """
        
    def generate_compliance_report(self, start_date: datetime,
                                  end_date: datetime,
                                  compliance_framework: str = None) -> ComplianceReport:
        """
        Generate comprehensive compliance report.
        
        Args:
            start_date (datetime): Report start date
            end_date (datetime): Report end date
            compliance_framework (str, optional): Specific compliance framework
            
        Returns:
            ComplianceReport: Generated compliance report
            
        Raises:
            ReportGenerationError: If report generation fails
        """
```

### SecurityMonitor

Real-time security monitoring and alerting.

```python
class SecurityMonitor:
    """Real-time security monitoring and alerting."""
    
    def __init__(self, alert_thresholds: dict,
                 notification_config: dict):
        """
        Initialize security monitor.
        
        Args:
            alert_thresholds (dict): Alert threshold configuration
            notification_config (dict): Notification configuration
        """
        
    def start_monitoring(self) -> None:
        """Start real-time security monitoring."""
        
    def stop_monitoring(self) -> None:
        """Stop security monitoring."""
        
    def monitor_pii_exposure_risk(self, detection_events: List[PIIDetectionEvent]) -> None:
        """
        Monitor for potential PII exposure risks.
        
        Args:
            detection_events (List[PIIDetectionEvent]): PII detection events to analyze
        """
        
    def monitor_credential_security(self, credential_events: List[CredentialEvent]) -> None:
        """
        Monitor credential security events.
        
        Args:
            credential_events (List[CredentialEvent]): Credential events to analyze
        """
        
    def monitor_session_anomalies(self, session_events: List[SessionEvent]) -> None:
        """
        Monitor for session-based security anomalies.
        
        Args:
            session_events (List[SessionEvent]): Session events to analyze
        """
        
    def get_security_metrics(self, time_range: dict) -> SecurityMetrics:
        """
        Get security metrics for specified time range.
        
        Args:
            time_range (dict): Time range for metrics
            
        Returns:
            SecurityMetrics: Security metrics data
        """
```

## Data Models and Types

### Core Data Types

```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any

class SecurityLevel(Enum):
    """Security level enumeration."""
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    MAXIMUM = "maximum"

class SensitivityLevel(Enum):
    """Data sensitivity level enumeration."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class ComplianceFramework(Enum):
    """Compliance framework enumeration."""
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    GDPR = "gdpr"
    SOX = "sox"
    CCPA = "ccpa"

@dataclass
class PIIDetectionResult:
    """PII detection result data model."""
    entities: List[dict]
    confidence_scores: Dict[str, float]
    detection_methods: List[str]
    industry_specific_flags: Dict[str, bool]
    timestamp: datetime
    
@dataclass
class PIIMaskingResult:
    """PII masking result data model."""
    masked_content: str
    masking_log: List[dict]
    original_entity_count: int
    masked_entity_count: int
    masking_strategy: str
    timestamp: datetime

@dataclass
class WorkflowResult:
    """Workflow execution result data model."""
    workflow_id: str
    execution_status: str
    start_time: datetime
    end_time: datetime
    steps_completed: int
    total_steps: int
    security_events: List[dict]
    audit_trail: List[dict]
    result_data: Dict[str, Any]
    
@dataclass
class LLMRoutingResult:
    """LLM routing result data model."""
    selected_model: str
    routing_reason: str
    confidence_score: float
    fallback_models: List[str]
    security_level_maintained: bool
    cost_estimate: float
    timestamp: datetime

@dataclass
class SessionHealth:
    """Browser session health data model."""
    session_id: str
    status: str
    uptime: int
    memory_usage: float
    cpu_usage: float
    network_latency: float
    error_count: int
    last_activity: datetime
    
@dataclass
class ComplianceReport:
    """Compliance report data model."""
    compliance_framework: str
    report_period: Dict[str, datetime]
    total_events: int
    compliance_score: float
    violations: List[dict]
    recommendations: List[str]
    generated_at: datetime
```

### Error Types

```python
class StrandsAgentCoreError(Exception):
    """Base exception for Strands-AgentCore integration."""
    pass

class SecurityViolationError(StrandsAgentCoreError):
    """Raised when security policy is violated."""
    pass

class PIIViolationError(SecurityViolationError):
    """Raised when PII policy is violated."""
    pass

class CredentialSecurityError(SecurityViolationError):
    """Raised when credential security is compromised."""
    pass

class SessionCreationError(StrandsAgentCoreError):
    """Raised when browser session creation fails."""
    pass

class WorkflowExecutionError(StrandsAgentCoreError):
    """Raised when workflow execution fails."""
    pass

class LLMRoutingError(StrandsAgentCoreError):
    """Raised when LLM routing fails."""
    pass

class ComplianceValidationError(StrandsAgentCoreError):
    """Raised when compliance validation fails."""
    pass
```

## Configuration Schemas

### Security Configuration Schema

```python
SECURITY_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "security_level": {
            "type": "string",
            "enum": ["basic", "standard", "high", "maximum"]
        },
        "compliance_frameworks": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["hipaa", "pci_dss", "gdpr", "sox", "ccpa"]
            }
        },
        "pii_detection": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                "industry_specific": {"type": "boolean"},
                "custom_patterns": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        },
        "encryption": {
            "type": "object",
            "properties": {
                "at_rest": {"type": "boolean"},
                "in_transit": {"type": "boolean"},
                "kms_key_id": {"type": "string"}
            }
        },
        "audit_logging": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "retention_days": {"type": "integer", "minimum": 1},
                "log_level": {
                    "type": "string",
                    "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                }
            }
        }
    },
    "required": ["security_level"]
}
```

### LLM Configuration Schema

```python
LLM_CONFIG_SCHEMA = {
    "type": "object",
    "patternProperties": {
        "^[a-zA-Z0-9_-]+$": {
            "type": "object",
            "properties": {
                "model_id": {"type": "string"},
                "security_level": {
                    "type": "string",
                    "enum": ["basic", "standard", "high", "maximum"]
                },
                "data_types": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "compliance_frameworks": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "cost_tier": {
                    "type": "string",
                    "enum": ["budget", "economical", "standard", "premium"]
                },
                "max_concurrent_requests": {"type": "integer", "minimum": 1}
            },
            "required": ["model_id", "security_level"]
        }
    }
}
```

## Usage Examples

### Basic Integration Example

```python
from tools.strands_agentcore_session_helpers import StrandsAgentCoreClient
from tools.strands_security_policies import SecurityPolicy

# Initialize client
client = StrandsAgentCoreClient(
    region='us-east-1',
    llm_configs={
        'high_security': {
            'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0',
            'security_level': 'high'
        }
    }
)

# Create secure agent
agent = client.create_secure_agent({
    'name': 'secure_browser_agent',
    'tools': ['secure_browser_tool', 'pii_detection_tool'],
    'security_level': 'high'
})

# Execute secure workflow
workflow = {
    'steps': [
        {'action': 'navigate', 'url': 'https://example.com'},
        {'action': 'extract_data', 'selector': '.content', 'sanitize': True}
    ]
}

result = client.execute_secure_workflow(workflow)
```

### Advanced Multi-Agent Example

```python
from tools.strands_agentcore_session_helpers import SessionPool, MultiAgentOrchestrator

# Create session pool
session_pool = SessionPool(max_sessions=10, region='us-east-1')

# Create orchestrator
orchestrator = MultiAgentOrchestrator(session_pool)

# Define workflow plan
workflow_plan = {
    'extract_data': {
        'agent': 'data_extractor',
        'config': {'url': 'https://example.com'},
        'dependencies': []
    },
    'analyze_pii': {
        'agent': 'pii_analyzer',
        'config': {'confidence_threshold': 0.8},
        'dependencies': ['extract_data']
    }
}

# Execute coordinated workflow
results = await orchestrator.execute_workflow_plan(workflow_plan, agents)
```

## Error Handling Best Practices

### Exception Handling Pattern

```python
from tools.strands_agentcore_session_helpers import StrandsAgentCoreClient
from tools.strands_security_policies import SecurityViolationError

try:
    client = StrandsAgentCoreClient(region='us-east-1')
    agent = client.create_secure_agent(agent_config)
    result = agent.execute_workflow(workflow)
    
except SecurityViolationError as e:
    # Handle security violations - do not retry
    logger.error(f"Security violation: {e}")
    # Notify security team
    security_alert_handler.send_alert(e)
    raise
    
except SessionCreationError as e:
    # Handle session creation failures - can retry
    logger.warning(f"Session creation failed: {e}")
    # Implement exponential backoff retry
    result = retry_with_backoff(agent.execute_workflow, workflow)
    
except WorkflowExecutionError as e:
    # Handle workflow execution failures
    logger.error(f"Workflow execution failed: {e}")
    # Check if workflow can be resumed from checkpoint
    if e.checkpoint_available:
        result = agent.resume_workflow_from_checkpoint(e.checkpoint_id)
    else:
        raise
        
except Exception as e:
    # Handle unexpected errors
    logger.error(f"Unexpected error: {e}")
    # Perform cleanup
    cleanup_resources()
    raise
```

## Performance Optimization Guidelines

### Session Pool Optimization

```python
# Optimal session pool configuration
session_pool = SessionPool(
    max_sessions=min(20, cpu_count() * 2),  # Scale with CPU cores
    region='us-east-1',
    cleanup_interval=300,  # 5 minutes
    health_check_interval=60,  # 1 minute
    enable_metrics=True
)

# Monitor pool performance
pool_stats = session_pool.get_pool_statistics()
if pool_stats.utilization > 0.8:
    logger.warning("High session pool utilization")
```

### Caching Strategy

```python
from tools.strands_pii_utils import PIIDetectionTool

# Configure PII detection with caching
pii_detector = PIIDetectionTool(
    industry_config='healthcare',
    confidence_threshold=0.8,
    enable_caching=True,
    cache_ttl=3600  # 1 hour
)

# Cache will automatically be used for repeated content
result1 = pii_detector.detect_pii(content)  # Cache miss
result2 = pii_detector.detect_pii(content)  # Cache hit
```

## Conclusion

This API reference provides comprehensive documentation for all custom tools and utilities in the Strands-AgentCore Browser Tool integration. The APIs are designed with security, performance, and compliance as primary considerations, enabling developers to build robust, production-ready applications that handle sensitive information securely.

For additional examples and advanced usage patterns, refer to the tutorial notebooks and supporting documentation in the `assets/` directory.