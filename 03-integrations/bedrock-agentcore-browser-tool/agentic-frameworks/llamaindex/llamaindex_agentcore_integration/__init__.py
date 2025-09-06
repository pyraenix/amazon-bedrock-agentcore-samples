"""
LlamaIndex integration with Amazon Bedrock AgentCore Browser Tool.

This package provides LlamaIndex tools and utilities for integrating with
AgentCore's enterprise-grade browser automation capabilities.
"""

from ._version import __version__, get_version, get_version_info, get_build_info

# Import main integration classes
from .integration import LlamaIndexAgentCoreIntegration
from .client import AgentCoreBrowserClient
from .config import BrowserConfiguration, IntegrationConfig

# Import tool classes
from .tools import (
    BrowserNavigationTool,
    CaptchaDetectionTool,
    ScreenshotCaptureTool,
    TextExtractionTool,
    ElementClickTool,
    FormInteractionTool
)

# Import workflow and processing classes
from .workflow_orchestrator import WorkflowOrchestrator
from .document_processor import WebContentDocument, DocumentProcessor
from .incremental_processor import IncrementalProcessor

# Import security and privacy classes
from .security_manager import SecurityManager
from .privacy_manager import PrivacyManager

# Import monitoring and diagnostics
from .monitoring import MonitoringIntegration
from .health_diagnostics import HealthDiagnostics

# Import exception classes
from .exceptions import (
    BrowserToolError,
    BrowserErrorType,
    NavigationError,
    ElementNotFoundError,
    CaptchaError,
    SessionError,
    AuthenticationError
)

__author__ = "AWS AgentCore Team"
__email__ = "agentcore-team@amazon.com"
__license__ = "MIT"

__all__ = [
    # Version info
    "__version__",
    "get_version",
    "get_version_info", 
    "get_build_info",
    
    # Main integration classes
    "LlamaIndexAgentCoreIntegration",
    "AgentCoreBrowserClient",
    "BrowserConfiguration",
    "IntegrationConfig",
    
    # Tool classes
    "BrowserNavigationTool",
    "CaptchaDetectionTool", 
    "ScreenshotCaptureTool",
    "TextExtractionTool",
    "ElementClickTool",
    "FormInteractionTool",
    
    # Workflow and processing
    "WorkflowOrchestrator",
    "WebContentDocument",
    "DocumentProcessor",
    "IncrementalProcessor",
    
    # Security and privacy
    "SecurityManager",
    "PrivacyManager",
    
    # Monitoring and diagnostics
    "MonitoringIntegration",
    "HealthDiagnostics",
    
    # Exceptions
    "BrowserToolError",
    "BrowserErrorType",
    "NavigationError",
    "ElementNotFoundError", 
    "CaptchaError",
    "SessionError",
    "AuthenticationError"
]

# Package metadata
__package_name__ = "llamaindex-agentcore-browser-integration"
__description__ = "LlamaIndex integration with Amazon Bedrock AgentCore Browser Tool"
__url__ = "https://github.com/aws-samples/agentcore-samples"
__documentation__ = "https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html"

def get_package_info() -> dict[str, str]:
    """Get package information."""
    return {
        "name": __package_name__,
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "url": __url__,
        "documentation": __documentation__
    }