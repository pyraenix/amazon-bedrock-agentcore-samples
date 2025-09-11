# Browser-Use with AgentCore Browser Tool - API Reference

## Overview

This API reference provides comprehensive documentation for the browser-use integration with Amazon Bedrock AgentCore Browser Tool for secure handling of sensitive information. The API includes classes, methods, and utilities for session management, PII detection, compliance validation, and monitoring.

## Core Classes

### BrowserUseAgentCoreSessionManager

Main class for managing secure browser sessions with AgentCore Browser Tool.

```python
class BrowserUseAgentCoreSessionManager:
    """Manages secure browser sessions for browser-use with AgentCore."""
    
    def __init__(self, region: str = 'us-east-1', **kwargs):
        """
        Initialize session manager.
        
        Args:
            region (str): AWS region for AgentCore Browser Tool
            **kwargs: Additional configuration options
                - session_timeout (int): Session timeout in seconds (default: 300)
                - enable_live_view (bool): Enable live view monitoring (default: True)
                - enable_session_replay (bool): Enable session replay (default: True)
                - compliance_mode (str): Compliance mode ('standard' or 'enterprise')
        """
```

#### Methods

##### create_secure_session()
```python
async def create_secure_session(self) -> Dict[str, Any]:
    """
    Create a secure browser session with AgentCore.
    
    Returns:
        Dict[str, Any]: Session details including:
            - session_id (str): Unique session identifier
            - ws_url (str): WebSocket URL for browser connection
            - headers (Dict[str, str]): Connection headers
            - browser_session (BrowserSession): Browser-use session object
            - live_view_url (str): URL for live view monitoring
    
    Raises:
        SessionCreationError: If session creation fails
        AuthenticationError: If AWS authentication fails
        RegionNotSupportedError: If region doesn't support AgentCore
    
    Example:
        >>> session_manager = BrowserUseAgentCoreSessionManager()
        >>> session_details = await session_manager.create_secure_session()
        >>> print(session_details['session_id'])
        'session-12345-abcde'
    """
```

##### cleanup_session()
```python
async def cleanup_session(self, session_id: str = None) -> bool:
    """
    Clean up browser session resources.
    
    Args:
        session_id (str, optional): Session ID to cleanup. If None, cleans up current session.
    
    Returns:
        bool: True if cleanup successful, False otherwise
    
    Raises:
        SessionNotFoundError: If session ID not found
    
    Example:
        >>> success = await session_manager.cleanup_session()
        >>> print(f"Cleanup successful: {success}")
        True
    """
```

##### get_session_status()
```python
def get_session_status(self, session_id: str) -> Dict[str, Any]:
    """
    Get current status of a browser session.
    
    Args:
        session_id (str): Session ID to check
    
    Returns:
        Dict[str, Any]: Session status including:
            - status (str): 'active', 'inactive', 'terminated'
            - created_at (datetime): Session creation time
            - last_activity (datetime): Last activity timestamp
            - resource_usage (Dict): CPU, memory usage statistics
    
    Example:
        >>> status = session_manager.get_session_status('session-12345')
        >>> print(status['status'])
        'active'
    """
```

### SensitiveDataHandler

Class for detecting and masking sensitive information in browser-use operations.

```python
class SensitiveDataHandler:
    """Handles detection and masking of sensitive information."""
    
    def __init__(self, compliance_frameworks: List[str] = None):
        """
        Initialize sensitive data handler.
        
        Args:
            compliance_frameworks (List[str], optional): List of compliance frameworks
                to enforce. Options: ['hipaa', 'pci_dss', 'gdpr', 'sox']
        """
```

#### Methods

##### detect_pii()
```python
def detect_pii(self, text: str, context: str = None) -> List[PIIDetection]:
    """
    Detect personally identifiable information in text.
    
    Args:
        text (str): Text content to analyze
        context (str, optional): Context for better detection accuracy
    
    Returns:
        List[PIIDetection]: List of detected PII items with:
            - type (str): PII type ('ssn', 'email', 'phone', 'credit_card', etc.)
            - value (str): Detected PII value
            - confidence (float): Detection confidence (0.0-1.0)
            - start_pos (int): Start position in text
            - end_pos (int): End position in text
            - compliance_impact (List[str]): Affected compliance frameworks
    
    Example:
        >>> handler = SensitiveDataHandler()
        >>> detections = handler.detect_pii("My SSN is 123-45-6789")
        >>> print(detections[0].type)
        'ssn'
    """
```

##### mask_pii()
```python
def mask_pii(
    self, 
    text: str, 
    mask_strategy: str = 'partial',
    preserve_format: bool = True
) -> str:
    """
    Mask PII in text content.
    
    Args:
        text (str): Text content to mask
        mask_strategy (str): Masking strategy:
            - 'full': Complete masking with asterisks
            - 'partial': Partial masking (show last 4 digits)
            - 'tokenize': Replace with tokens
        preserve_format (bool): Whether to preserve original format
    
    Returns:
        str: Text with PII masked
    
    Example:
        >>> masked = handler.mask_pii("Credit card: 4111-1111-1111-1111")
        >>> print(masked)
        'Credit card: ****-****-****-1111'
    """
```

##### validate_compliance()
```python
def validate_compliance(
    self, 
    data_types: List[str], 
    operation: str,
    framework: str
) -> ComplianceValidationResult:
    """
    Validate operation against compliance framework.
    
    Args:
        data_types (List[str]): Types of data being processed
        operation (str): Operation being performed
        framework (str): Compliance framework to validate against
    
    Returns:
        ComplianceValidationResult: Validation result with:
            - compliant (bool): Whether operation is compliant
            - violations (List[str]): List of compliance violations
            - recommendations (List[str]): Recommendations for compliance
            - risk_level (str): Risk level ('low', 'medium', 'high', 'critical')
    
    Example:
        >>> result = handler.validate_compliance(['phi'], 'form_fill', 'hipaa')
        >>> print(result.compliant)
        True
    """
```

### BrowserUseMonitor

Class for monitoring and observability of browser-use operations.

```python
class BrowserUseMonitor:
    """Provides monitoring and observability for browser-use operations."""
    
    def __init__(self, enable_metrics: bool = True, enable_logging: bool = True):
        """
        Initialize monitoring system.
        
        Args:
            enable_metrics (bool): Enable CloudWatch metrics
            enable_logging (bool): Enable structured logging
        """
```

#### Methods

##### start_session_monitoring()
```python
def start_session_monitoring(self, session_id: str) -> SessionMonitor:
    """
    Start monitoring for a browser session.
    
    Args:
        session_id (str): Session ID to monitor
    
    Returns:
        SessionMonitor: Monitor object for the session
    
    Example:
        >>> monitor = BrowserUseMonitor()
        >>> session_monitor = monitor.start_session_monitoring('session-12345')
    """
```

##### record_operation()
```python
def record_operation(
    self,
    session_id: str,
    operation_type: str,
    duration: float,
    success: bool = True,
    metadata: Dict[str, Any] = None
) -> None:
    """
    Record an operation for monitoring.
    
    Args:
        session_id (str): Session ID
        operation_type (str): Type of operation performed
        duration (float): Operation duration in seconds
        success (bool): Whether operation was successful
        metadata (Dict[str, Any], optional): Additional metadata
    
    Example:
        >>> monitor.record_operation(
        ...     'session-12345',
        ...     'form_fill',
        ...     2.5,
        ...     success=True,
        ...     metadata={'pii_detected': True}
        ... )
    """
```

##### get_live_view_url()
```python
def get_live_view_url(self, session_id: str) -> Optional[str]:
    """
    Get AgentCore live view URL for session.
    
    Args:
        session_id (str): Session ID
    
    Returns:
        Optional[str]: Live view URL or None if not available
    
    Example:
        >>> url = monitor.get_live_view_url('session-12345')
        >>> print(url)
        'https://agentcore.aws.amazon.com/live-view/session-12345'
    """
```

## Data Classes

### PIIDetection

Data class representing a detected PII item.

```python
@dataclass
class PIIDetection:
    """Represents a detected PII item."""
    
    type: str                    # PII type (ssn, email, phone, etc.)
    value: str                   # Detected value
    confidence: float            # Detection confidence (0.0-1.0)
    start_pos: int              # Start position in text
    end_pos: int                # End position in text
    compliance_impact: List[str] # Affected compliance frameworks
    severity: str               # Severity level (low, medium, high, critical)
    context: Optional[str] = None # Context where PII was found
```

### ComplianceValidationResult

Data class representing compliance validation results.

```python
@dataclass
class ComplianceValidationResult:
    """Represents compliance validation results."""
    
    compliant: bool              # Whether operation is compliant
    framework: str               # Compliance framework validated against
    violations: List[str]        # List of violations found
    recommendations: List[str]   # Recommendations for compliance
    risk_level: str             # Risk level assessment
    audit_trail: List[str]      # Audit trail entries
    timestamp: datetime         # Validation timestamp
```

### SessionMetrics

Data class for session performance metrics.

```python
@dataclass
class SessionMetrics:
    """Represents session performance metrics."""
    
    session_id: str                      # Session identifier
    start_time: datetime                 # Session start time
    end_time: Optional[datetime] = None  # Session end time
    operations_count: int = 0            # Number of operations performed
    pii_detections: int = 0             # Number of PII detections
    compliance_violations: int = 0       # Number of compliance violations
    avg_response_time: float = 0.0      # Average response time
    total_data_processed: int = 0       # Total data processed (bytes)
    error_count: int = 0                # Number of errors encountered
```

## Utility Functions

### Authentication Utilities

```python
def get_aws_credentials(profile: str = None) -> Dict[str, str]:
    """
    Get AWS credentials for AgentCore access.
    
    Args:
        profile (str, optional): AWS profile name
    
    Returns:
        Dict[str, str]: AWS credentials
    
    Raises:
        AuthenticationError: If credentials cannot be obtained
    """

def validate_agentcore_permissions() -> bool:
    """
    Validate that current AWS credentials have required AgentCore permissions.
    
    Returns:
        bool: True if permissions are valid
    
    Raises:
        PermissionError: If required permissions are missing
    """
```

### Configuration Utilities

```python
def load_configuration(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from file or environment variables.
    
    Args:
        config_path (str, optional): Path to configuration file
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    
    Example:
        >>> config = load_configuration('.env')
        >>> print(config['AWS_REGION'])
        'us-east-1'
    """

def validate_configuration(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration for required settings.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
    
    Returns:
        List[str]: List of validation errors (empty if valid)
    """
```

### Error Handling Utilities

```python
def handle_agentcore_error(error: Exception) -> Dict[str, Any]:
    """
    Handle and categorize AgentCore errors.
    
    Args:
        error (Exception): Exception to handle
    
    Returns:
        Dict[str, Any]: Error information including:
            - category (str): Error category
            - message (str): User-friendly error message
            - recovery_suggestions (List[str]): Recovery suggestions
            - retry_recommended (bool): Whether retry is recommended
    """

async def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    *args,
    **kwargs
) -> Any:
    """
    Retry function with exponential backoff.
    
    Args:
        func (Callable): Function to retry
        max_retries (int): Maximum number of retries
        backoff_factor (float): Backoff multiplication factor
        *args: Function arguments
        **kwargs: Function keyword arguments
    
    Returns:
        Any: Function return value
    
    Raises:
        Exception: Last exception if all retries fail
    """
```

## Exception Classes

### Base Exceptions

```python
class BrowserUseAgentCoreError(Exception):
    """Base exception for browser-use AgentCore integration."""
    pass

class SessionCreationError(BrowserUseAgentCoreError):
    """Exception raised when session creation fails."""
    pass

class SessionNotFoundError(BrowserUseAgentCoreError):
    """Exception raised when session is not found."""
    pass

class AuthenticationError(BrowserUseAgentCoreError):
    """Exception raised for authentication failures."""
    pass

class RegionNotSupportedError(BrowserUseAgentCoreError):
    """Exception raised when region is not supported."""
    pass
```

### Security Exceptions

```python
class SensitiveDataError(BrowserUseAgentCoreError):
    """Exception raised for sensitive data handling errors."""
    pass

class PIIMaskingError(SensitiveDataError):
    """Exception raised when PII masking fails."""
    pass

class ComplianceViolationError(SensitiveDataError):
    """Exception raised for compliance violations."""
    pass

class SessionIsolationError(SensitiveDataError):
    """Exception raised when session isolation fails."""
    pass
```

## Constants and Enums

### Compliance Frameworks

```python
class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    GDPR = "gdpr"
    SOX = "sox"
    CCPA = "ccpa"
```

### PII Types

```python
class PIIType(Enum):
    """Types of personally identifiable information."""
    SSN = "ssn"
    EMAIL = "email"
    PHONE = "phone"
    CREDIT_CARD = "credit_card"
    BANK_ACCOUNT = "bank_account"
    DRIVERS_LICENSE = "drivers_license"
    PASSPORT = "passport"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
```

### Session Status

```python
class SessionStatus(Enum):
    """Browser session status values."""
    CREATING = "creating"
    ACTIVE = "active"
    INACTIVE = "inactive"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    ERROR = "error"
```

## Configuration Schema

### Environment Variables

```python
REQUIRED_ENV_VARS = {
    'AWS_REGION': 'AWS region for AgentCore Browser Tool',
    'BEDROCK_MODEL_ID': 'Bedrock model ID for LLM operations'
}

OPTIONAL_ENV_VARS = {
    'AGENTCORE_SESSION_TIMEOUT': 'Session timeout in seconds (default: 300)',
    'ENABLE_PII_MASKING': 'Enable PII masking (default: true)',
    'COMPLIANCE_MODE': 'Compliance mode (standard/enterprise, default: standard)',
    'LOG_LEVEL': 'Logging level (default: INFO)',
    'ENABLE_LIVE_VIEW': 'Enable live view monitoring (default: true)'
}
```

### Configuration File Schema

```json
{
  "aws": {
    "region": "us-east-1",
    "profile": "default"
  },
  "agentcore": {
    "session_timeout": 300,
    "enable_live_view": true,
    "enable_session_replay": true,
    "compliance_mode": "enterprise"
  },
  "browser_use": {
    "headless": true,
    "timeout": 30000,
    "viewport": {
      "width": 1920,
      "height": 1080
    }
  },
  "security": {
    "enable_pii_masking": true,
    "compliance_frameworks": ["hipaa", "pci_dss"],
    "audit_level": "detailed"
  },
  "monitoring": {
    "enable_metrics": true,
    "enable_logging": true,
    "log_level": "INFO"
  }
}
```

## Usage Examples

### Basic Session Management

```python
import asyncio
from tools.browseruse_agentcore_session_helpers import BrowserUseAgentCoreSessionManager

async def basic_session_example():
    """Basic session management example."""
    
    # Initialize session manager
    session_manager = BrowserUseAgentCoreSessionManager(region='us-east-1')
    
    try:
        # Create secure session
        session_details = await session_manager.create_secure_session()
        print(f"Session created: {session_details['session_id']}")
        
        # Use session for browser automation
        # ... browser-use operations here ...
        
        # Check session status
        status = session_manager.get_session_status(session_details['session_id'])
        print(f"Session status: {status['status']}")
        
    finally:
        # Cleanup session
        await session_manager.cleanup_session()
```

### PII Detection and Masking

```python
from tools.browseruse_pii_utils import SensitiveDataHandler

def pii_handling_example():
    """PII detection and masking example."""
    
    # Initialize handler with HIPAA compliance
    handler = SensitiveDataHandler(compliance_frameworks=['hipaa'])
    
    # Sample text with PII
    text = "Patient John Doe, SSN: 123-45-6789, Email: john@example.com"
    
    # Detect PII
    detections = handler.detect_pii(text)
    for detection in detections:
        print(f"Found {detection.type}: {detection.value} (confidence: {detection.confidence})")
    
    # Mask PII
    masked_text = handler.mask_pii(text, mask_strategy='partial')
    print(f"Masked text: {masked_text}")
    
    # Validate compliance
    result = handler.validate_compliance(['phi'], 'data_processing', 'hipaa')
    print(f"HIPAA compliant: {result.compliant}")
```

### Complete Integration Example

```python
import asyncio
from browser_use import Agent
from tools.browseruse_agentcore_session_helpers import BrowserUseAgentCoreSessionManager
from tools.browseruse_pii_utils import SensitiveDataHandler
from tools.browseruse_monitoring import BrowserUseMonitor

async def complete_integration_example():
    """Complete integration example with monitoring."""
    
    # Initialize components
    session_manager = BrowserUseAgentCoreSessionManager()
    data_handler = SensitiveDataHandler(['hipaa', 'pci_dss'])
    monitor = BrowserUseMonitor()
    
    session_id = None
    
    try:
        # Create secure session
        session_details = await session_manager.create_secure_session()
        session_id = session_details['session_id']
        
        # Start monitoring
        session_monitor = monitor.start_session_monitoring(session_id)
        
        # Create browser-use agent
        agent = Agent(
            task="Fill healthcare form with PII protection",
            llm=get_bedrock_model(),
            browser_session=session_details['browser_session']
        )
        
        # Execute task with monitoring
        start_time = time.time()
        result = await agent.run()
        duration = time.time() - start_time
        
        # Record operation
        monitor.record_operation(
            session_id=session_id,
            operation_type='healthcare_form_fill',
            duration=duration,
            success=True,
            metadata={'pii_detected': len(data_handler.detect_pii(str(result))) > 0}
        )
        
        # Mask PII in result
        masked_result = data_handler.mask_pii(str(result))
        print(f"Task completed: {masked_result}")
        
        # Get live view URL
        live_view_url = monitor.get_live_view_url(session_id)
        print(f"Live view: {live_view_url}")
        
    finally:
        # Cleanup
        if session_id:
            await session_manager.cleanup_session(session_id)
```

This API reference provides comprehensive documentation for integrating browser-use with AgentCore Browser Tool for secure handling of sensitive information.