"""
Strands Tools Package for AgentCore Browser Tool Integration

This package provides custom Strands tools for secure integration with Amazon Bedrock
AgentCore Browser Tool, including browser automation capabilities and sensitive data
handling utilities.

Key Components:
- AgentCoreBrowserTool: Main browser automation tool for Strands agents
- SensitiveDataHandler: Comprehensive sensitive data handling utilities
- PIIDetector: Advanced PII detection engine
- DataMasker: Data masking and sanitization engine
- CredentialManager: Secure credential management with AWS Secrets Manager

Requirements Addressed:
- 1.2: Secure credential management patterns
- 1.3: Proper data isolation and protection mechanisms
- 1.5: Browser automation methods that send commands to AgentCore Browser Tool
- 2.1: PII detection, masking, and classification in Strands workflows
- 2.2: Credential management system that integrates with AWS Secrets Manager
- 2.3: Data sanitization methods that work with Strands' tool output processing
"""

from .agentcore_browser_tool import (
    AgentCoreBrowserTool,
    BrowserSessionConfig,
    CredentialConfig,
    BrowserOperationMetrics,
    create_secure_browser_tool,
    create_authenticated_browser_tool
)

from .sensitive_data_handler import (
    SensitiveDataHandler,
    PIIDetector,
    DataMasker,
    CredentialManager,
    SensitivityLevel,
    PIIType,
    MaskingStrategy,
    PIIPattern,
    PIIDetectionResult,
    SanitizationConfig,
    AuditLogEntry,
    create_secure_data_handler
)

# Note: bedrock_model_router and compliance_validator are implemented in previous tasks
# For now, we'll comment them out to focus on workflow orchestration
# from .bedrock_model_router import (...)
# from .compliance_validator import (...)

from .secure_workflow_orchestrator import (
    SecureWorkflowOrchestrator,
    SecureWorkflow,
    WorkflowStep,
    WorkflowCheckpoint,
    SessionPool,
    SessionPoolConfig,
    EncryptionManager,
    WorkflowStatus,
    SecurityLevel as WorkflowSecurityLevel,
    create_secure_workflow,
    create_workflow_step
)

from .multi_agent_coordinator import (
    MultiAgentCoordinator,
    AgentContext,
    ResourceAllocation,
    SecureDataShare,
    CoordinationConfig,
    ResourceManager,
    SecureDataManager,
    AgentStatus,
    ResourceType,
    IsolationLevel,
    create_agent_task_config,
    create_data_share_permissions
)

__all__ = [
    # Main tool classes
    'AgentCoreBrowserTool',
    'SensitiveDataHandler',
    'PIIDetector',
    'DataMasker',
    'CredentialManager',
    'SecureWorkflowOrchestrator',
    'MultiAgentCoordinator',
    
    # Configuration classes
    'BrowserSessionConfig',
    'CredentialConfig',
    'SanitizationConfig',
    'SessionPoolConfig',
    'CoordinationConfig',
    
    # Data classes
    'BrowserOperationMetrics',
    'PIIDetectionResult',
    'AuditLogEntry',
    'PIIPattern',
    'SecureWorkflow',
    'WorkflowStep',
    'WorkflowCheckpoint',
    'AgentContext',
    'ResourceAllocation',
    'SecureDataShare',
    
    # Manager classes
    'SessionPool',
    'EncryptionManager',
    'ResourceManager',
    'SecureDataManager',
    
    # Enums
    'SensitivityLevel',
    'PIIType',
    'MaskingStrategy',
    'WorkflowStatus',
    'WorkflowSecurityLevel',
    'AgentStatus',
    'ResourceType',
    'IsolationLevel',
    
    # Factory functions
    'create_secure_browser_tool',
    'create_authenticated_browser_tool',
    'create_secure_data_handler',
    'create_secure_workflow',
    'create_workflow_step',
    'create_agent_task_config',
    'create_data_share_permissions'
]

__version__ = "1.0.0"
__author__ = "Strands AgentCore Integration Team"
__description__ = "Strands tools for secure AgentCore Browser Tool integration"