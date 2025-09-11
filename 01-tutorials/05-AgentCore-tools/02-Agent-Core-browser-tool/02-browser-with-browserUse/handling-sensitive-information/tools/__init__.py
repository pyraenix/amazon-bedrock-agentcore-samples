"""
Browser-Use with AgentCore Browser Tool - Utility Tools Package

This package contains utility tools and helper functions for integrating
browser-use with Amazon Bedrock AgentCore Browser Tool for secure handling
of sensitive information.
"""

__version__ = "1.0.0"
__author__ = "Amazon Bedrock AgentCore Team"

# Import only the core modules that exist and work
try:
    from .browseruse_sensitive_data_handler import (
        BrowserUseSensitiveDataHandler,
        BrowserUseCredentialManager,
        BrowserUseDataClassifier,
        PIIType,
        ComplianceFramework,
        DataClassification,
        DetectionResult,
        detect_and_mask_pii,
        classify_sensitive_data
    )
except ImportError:
    pass

try:
    from .browseruse_pii_masking import (
        BrowserUsePIIMasking,
        BrowserUsePIIValidator,
        BrowserElementPII,
        FormPIIAnalysis,
        analyze_browser_page_pii,
        mask_browser_form_data,
        validate_browser_pii_handling
    )
except ImportError:
    pass

try:
    from .browseruse_credential_handling import (
        BrowserUseCredentialHandler,
        BrowserUseCredentialIsolation,
        CredentialType,
        CredentialSecurityLevel,
        CredentialScope,
        secure_browser_login,
        secure_api_key_input
    )
except ImportError:
    pass

__all__ = [
    # Core classes that should be available
    "BrowserUseSensitiveDataHandler",
    "BrowserUseCredentialManager", 
    "BrowserUseDataClassifier",
    "BrowserUsePIIMasking",
    "BrowserUsePIIValidator",
    "BrowserUseCredentialHandler",
    "BrowserUseCredentialIsolation",
    
    # Enums and data classes
    "PIIType",
    "ComplianceFramework", 
    "DataClassification",
    "DetectionResult",
    "BrowserElementPII",
    "FormPIIAnalysis",
    "CredentialType",
    "CredentialSecurityLevel",
    "CredentialScope",
    
    # Convenience functions
    "detect_and_mask_pii",
    "classify_sensitive_data",
    "analyze_browser_page_pii",
    "mask_browser_form_data", 
    "validate_browser_pii_handling",
    "secure_browser_login",
    "secure_api_key_input"
]