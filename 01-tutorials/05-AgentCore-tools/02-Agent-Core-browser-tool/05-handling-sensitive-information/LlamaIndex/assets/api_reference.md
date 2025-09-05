# API Reference for LlamaIndex-AgentCore Browser Tool Integration

This document provides comprehensive API reference for the LlamaIndex-AgentCore Browser Tool integration components.

## Core Integration Classes

### AgentCoreBrowserLoader

The main loader class for integrating AgentCore Browser Tool with LlamaIndex document loading.

```python
class AgentCoreBrowserLoader(BaseLoader):
    """
    LlamaIndex loader for secure web content extraction via AgentCore Browser Tool.
    
    This loader provides secure web content extraction with built-in PII detection,
    credential management, and containerized isolation.
    """
    
    def __init__(self, 
                 region: str = "us-east-1",
                 security_config: Optional[SecurityConfig] = None,
                 session_config: Optional[SessionConfig] = None):
        """
        Initialize the AgentCore Browser Loader.
        
        Args:
            region: AWS region for AgentCore Browser Tool
            security_config: Security configuration for data protection
            session_config: Browser session configuration
        """
    
    async def load_data(self, 
                       urls: List[str],
                       credentials: Optional[Dict[str, Any]] = None,
                       extraction_config: Optional[ExtractionConfig] = None) -> List[Document]:
        """
        Load documents from web URLs with security controls.
        
        Args:
            urls: List of URLs to extract content from
            credentials: Optional authentication credentials
            extraction_config: Configuration for content extraction
            
        Returns:
            List of LlamaIndex Document objects with security metadata
            
        Raises:
            SecurityError: If security validation fails
            SessionError: If browser session creation fails
            ExtractionError: If content extraction fails
        """
    
    async def load_authenticated_data(self,
                                    auth_config: AuthenticationConfig,
                                    extraction_targets: List[ExtractionTarget]) -> List[Document]:
        """
        Load data from authenticated web services.
        
        Args:
            auth_config: Authentication configuration
            extraction_targets: List of extraction targets with selectors
            
        Returns:
            List of documents with authentication context
        """
    
    def set_security_policy(self, policy: SecurityPolicy) -> None:
        """
        Set security policy for data extraction and processing.
        
        Args:
            policy: Security policy configuration
        """
```

### SecureRAGPipeline

RAG pipeline implementation with security controls for sensitive data.

```python
class SecureRAGPipeline:
    """
    Secure RAG pipeline for processing sensitive web content.
    
    Provides end-to-end security controls for RAG operations including
    secure document ingestion, query sanitization, and response filtering.
    """
    
    def __init__(self,
                 llm: BaseLLM,
                 embedding_model: BaseEmbedding,
                 security_config: SecurityConfig,
                 vector_store: Optional[VectorStore] = None):
        """
        Initialize secure RAG pipeline.
        
        Args:
            llm: Language model for response generation
            embedding_model: Embedding model for document processing
            security_config: Security configuration
            vector_store: Optional vector store (defaults to secure in-memory)
        """
    
    async def ingest_web_documents(self,
                                 web_sources: List[WebSource],
                                 browser_loader: AgentCoreBrowserLoader) -> IngestionResult:
        """
        Ingest documents from web sources with security processing.
        
        Args:
            web_sources: List of web sources to process
            browser_loader: Configured browser loader instance
            
        Returns:
            Ingestion result with security summary
        """
    
    async def query(self,
                   query_str: str,
                   user_context: UserContext,
                   security_level: SecurityLevel = SecurityLevel.HIGH) -> SecureResponse:
        """
        Execute secure query with context filtering and response sanitization.
        
        Args:
            query_str: User query string
            user_context: User context for access control
            security_level: Security level for processing
            
        Returns:
            Secure response with filtered content
        """
    
    def get_security_metrics(self) -> SecurityMetrics:
        """
        Get security metrics for the pipeline.
        
        Returns:
            Security metrics including PII detection stats, access patterns
        """
```

### SessionManager

Manages AgentCore browser sessions with security controls and lifecycle management.

```python
class SessionManager:
    """
    Manages AgentCore browser sessions with security and lifecycle controls.
    
    Provides session pooling, health monitoring, and automatic cleanup
    for secure browser automation.
    """
    
    def __init__(self,
                 region: str,
                 max_sessions: int = 10,
                 session_timeout: int = 300,
                 security_config: Optional[SecurityConfig] = None):
        """
        Initialize session manager.
        
        Args:
            region: AWS region for AgentCore
            max_sessions: Maximum concurrent sessions
            session_timeout: Session timeout in seconds
            security_config: Security configuration
        """
    
    async def create_session(self,
                           session_config: Optional[SessionConfig] = None) -> BrowserSession:
        """
        Create new browser session with security controls.
        
        Args:
            session_config: Optional session configuration
            
        Returns:
            Configured browser session
            
        Raises:
            SessionCreationError: If session creation fails
            SecurityError: If security validation fails
        """
    
    async def get_session(self, session_id: Optional[str] = None) -> BrowserSession:
        """
        Get existing session or create new one from pool.
        
        Args:
            session_id: Optional specific session ID
            
        Returns:
            Browser session instance
        """
    
    async def cleanup_session(self, session_id: str) -> None:
        """
        Cleanup and terminate browser session.
        
        Args:
            session_id: Session ID to cleanup
        """
    
    def get_session_metrics(self) -> SessionMetrics:
        """
        Get session pool metrics and health status.
        
        Returns:
            Session metrics including active sessions, health status
        """
```

## Security Components

### SecurityProcessor

Handles all security-related processing including PII detection, data masking, and audit logging.

```python
class SecurityProcessor:
    """
    Comprehensive security processing for sensitive data handling.
    
    Provides PII detection, data masking, encryption, and audit logging
    for all security-sensitive operations.
    """
    
    def __init__(self, config: SecurityConfig):
        """
        Initialize security processor.
        
        Args:
            config: Security configuration
        """
    
    async def detect_pii(self, content: str) -> PIIDetectionResult:
        """
        Detect personally identifiable information in content.
        
        Args:
            content: Text content to analyze
            
        Returns:
            PII detection result with entity locations and types
        """
    
    async def mask_sensitive_data(self,
                                content: str,
                                pii_entities: List[PIIEntity],
                                masking_policy: MaskingPolicy) -> MaskedContent:
        """
        Apply data masking based on PII entities and policy.
        
        Args:
            content: Original content
            pii_entities: Detected PII entities
            masking_policy: Masking policy to apply
            
        Returns:
            Masked content with metadata
        """
    
    async def encrypt_sensitive_data(self,
                                   data: str,
                                   encryption_config: EncryptionConfig) -> EncryptedData:
        """
        Encrypt sensitive data portions.
        
        Args:
            data: Data to encrypt
            encryption_config: Encryption configuration
            
        Returns:
            Encrypted data with metadata
        """
    
    async def audit_security_event(self,
                                 event: SecurityEvent,
                                 context: SecurityContext) -> None:
        """
        Log security event for audit trail.
        
        Args:
            event: Security event to log
            context: Security context
        """
```

### CredentialManager

Secure credential management for web authentication.

```python
class CredentialManager:
    """
    Secure credential management for web authentication.
    
    Provides secure storage, retrieval, and injection of authentication
    credentials without exposure in logs or memory dumps.
    """
    
    def __init__(self, config: CredentialConfig):
        """
        Initialize credential manager.
        
        Args:
            config: Credential management configuration
        """
    
    async def store_credentials(self,
                              credential_id: str,
                              credentials: Dict[str, Any],
                              encryption_key: Optional[str] = None) -> None:
        """
        Securely store credentials.
        
        Args:
            credential_id: Unique identifier for credentials
            credentials: Credential data to store
            encryption_key: Optional encryption key
        """
    
    async def retrieve_credentials(self,
                                 credential_id: str,
                                 user_context: UserContext) -> Dict[str, Any]:
        """
        Retrieve credentials with access control.
        
        Args:
            credential_id: Credential identifier
            user_context: User context for access control
            
        Returns:
            Decrypted credential data
            
        Raises:
            AccessDeniedError: If user lacks permission
            CredentialNotFoundError: If credentials don't exist
        """
    
    async def inject_credentials(self,
                               session: BrowserSession,
                               credential_id: str,
                               injection_config: InjectionConfig) -> InjectionResult:
        """
        Securely inject credentials into browser session.
        
        Args:
            session: Browser session for injection
            credential_id: Credentials to inject
            injection_config: Injection configuration
            
        Returns:
            Injection result with success status
        """
    
    async def rotate_credentials(self,
                               credential_id: str,
                               rotation_config: RotationConfig) -> None:
        """
        Rotate credentials according to policy.
        
        Args:
            credential_id: Credentials to rotate
            rotation_config: Rotation configuration
        """
```

## Configuration Classes

### SecurityConfig

Configuration for security controls and policies.

```python
@dataclass
class SecurityConfig:
    """
    Comprehensive security configuration.
    """
    
    # PII Detection Configuration
    pii_detection_enabled: bool = True
    pii_confidence_threshold: float = 0.8
    custom_pii_patterns: List[str] = field(default_factory=list)
    
    # Data Masking Configuration
    masking_policy: MaskingPolicy = MaskingPolicy.PARTIAL
    masking_character: str = "*"
    preserve_format: bool = True
    
    # Encryption Configuration
    encryption_enabled: bool = True
    encryption_algorithm: str = "AES-256-GCM"
    key_rotation_interval: int = 86400  # 24 hours
    
    # Audit Configuration
    audit_level: AuditLevel = AuditLevel.COMPREHENSIVE
    audit_retention_days: int = 90
    audit_encryption_enabled: bool = True
    
    # Access Control Configuration
    access_control_enabled: bool = True
    role_based_access: bool = True
    session_based_access: bool = True
    
    # Compliance Configuration
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    data_residency_requirements: Optional[DataResidencyConfig] = None
    
    def validate(self) -> ValidationResult:
        """Validate security configuration."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecurityConfig':
        """Create from dictionary representation."""
```

### SessionConfig

Configuration for browser session management.

```python
@dataclass
class SessionConfig:
    """
    Browser session configuration.
    """
    
    # Session Management
    session_timeout: int = 300  # 5 minutes
    max_idle_time: int = 120    # 2 minutes
    auto_cleanup: bool = True
    
    # Container Configuration
    container_isolation: IsolationLevel = IsolationLevel.HIGH
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    network_policy: NetworkPolicy = NetworkPolicy.RESTRICTED
    
    # Browser Configuration
    browser_type: BrowserType = BrowserType.CHROMIUM
    headless: bool = True
    disable_images: bool = True
    disable_javascript: bool = False
    
    # Security Configuration
    enable_security_monitoring: bool = True
    screenshot_redaction: bool = True
    disable_extensions: bool = True
    
    # Performance Configuration
    page_load_timeout: int = 30
    element_timeout: int = 10
    navigation_timeout: int = 30
    
    def validate(self) -> ValidationResult:
        """Validate session configuration."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
```

### ExtractionConfig

Configuration for web content extraction.

```python
@dataclass
class ExtractionConfig:
    """
    Web content extraction configuration.
    """
    
    # Content Selection
    content_selectors: List[str] = field(default_factory=list)
    exclude_selectors: List[str] = field(default_factory=list)
    extract_links: bool = False
    extract_images: bool = False
    
    # Content Processing
    clean_html: bool = True
    preserve_formatting: bool = False
    extract_metadata: bool = True
    
    # Security Processing
    apply_security_filtering: bool = True
    detect_sensitive_content: bool = True
    mask_on_extraction: bool = True
    
    # Performance Configuration
    max_content_length: int = 1000000  # 1MB
    extraction_timeout: int = 30
    retry_attempts: int = 3
    
    def validate(self) -> ValidationResult:
        """Validate extraction configuration."""
```

## Data Models

### Document

Extended LlamaIndex Document with security metadata.

```python
@dataclass
class SecureDocument(Document):
    """
    LlamaIndex Document extended with security metadata.
    """
    
    # Security Metadata
    security_level: SecurityLevel
    pii_entities: List[PIIEntity] = field(default_factory=list)
    masking_applied: bool = False
    encryption_applied: bool = False
    
    # Source Metadata
    source_url: Optional[str] = None
    extraction_timestamp: Optional[datetime] = None
    session_id: Optional[str] = None
    
    # Audit Metadata
    audit_trail: List[AuditEvent] = field(default_factory=list)
    access_history: List[AccessEvent] = field(default_factory=list)
    
    def add_security_event(self, event: SecurityEvent) -> None:
        """Add security event to audit trail."""
    
    def get_security_summary(self) -> SecuritySummary:
        """Get security summary for the document."""
    
    def is_accessible_by(self, user_context: UserContext) -> bool:
        """Check if document is accessible by user."""
```

### PIIEntity

Represents detected personally identifiable information.

```python
@dataclass
class PIIEntity:
    """
    Represents a detected PII entity.
    """
    
    entity_type: PIIType
    text: str
    start_pos: int
    end_pos: int
    confidence: float
    sensitivity_level: SensitivityLevel
    
    # Masking Information
    masked_text: Optional[str] = None
    masking_method: Optional[MaskingMethod] = None
    
    # Context Information
    context: Optional[str] = None
    surrounding_text: Optional[str] = None
    
    def mask(self, masking_policy: MaskingPolicy) -> str:
        """Apply masking to the entity."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
```

### SecurityEvent

Represents a security-related event for audit logging.

```python
@dataclass
class SecurityEvent:
    """
    Represents a security event for audit logging.
    """
    
    event_type: SecurityEventType
    timestamp: datetime
    session_id: Optional[str]
    user_id: Optional[str]
    
    # Event Details
    description: str
    severity: SecuritySeverity
    category: SecurityCategory
    
    # Context Information
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    
    # Security Metadata
    threat_indicators: List[str] = field(default_factory=list)
    mitigation_actions: List[str] = field(default_factory=list)
    
    def to_audit_log(self) -> Dict[str, Any]:
        """Convert to audit log format."""
    
    def is_critical(self) -> bool:
        """Check if event is critical severity."""
```

## Utility Functions

### Security Utilities

```python
def validate_url_security(url: str, security_policy: SecurityPolicy) -> ValidationResult:
    """
    Validate URL against security policy.
    
    Args:
        url: URL to validate
        security_policy: Security policy to apply
        
    Returns:
        Validation result with security assessment
    """

def sanitize_content(content: str, 
                    sanitization_config: SanitizationConfig) -> SanitizedContent:
    """
    Sanitize content for security.
    
    Args:
        content: Content to sanitize
        sanitization_config: Sanitization configuration
        
    Returns:
        Sanitized content with metadata
    """

def generate_security_hash(data: str, algorithm: str = "SHA-256") -> str:
    """
    Generate security hash for data integrity.
    
    Args:
        data: Data to hash
        algorithm: Hash algorithm to use
        
    Returns:
        Hex-encoded hash string
    """
```

### Session Utilities

```python
def create_session_pool(config: SessionPoolConfig) -> SessionPool:
    """
    Create session pool with configuration.
    
    Args:
        config: Session pool configuration
        
    Returns:
        Configured session pool
    """

def monitor_session_health(session: BrowserSession) -> HealthStatus:
    """
    Monitor browser session health.
    
    Args:
        session: Browser session to monitor
        
    Returns:
        Health status with metrics
    """

def cleanup_expired_sessions(session_manager: SessionManager) -> CleanupResult:
    """
    Cleanup expired sessions.
    
    Args:
        session_manager: Session manager instance
        
    Returns:
        Cleanup result with statistics
    """
```

## Error Classes

### Security Errors

```python
class SecurityError(Exception):
    """Base class for security-related errors."""
    
    def __init__(self, message: str, error_code: str, context: Optional[Dict] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = datetime.utcnow()

class PIIDetectionError(SecurityError):
    """Error in PII detection process."""

class MaskingError(SecurityError):
    """Error in data masking process."""

class EncryptionError(SecurityError):
    """Error in encryption/decryption process."""

class AccessDeniedError(SecurityError):
    """Access denied due to security policy."""
```

### Session Errors

```python
class SessionError(Exception):
    """Base class for session-related errors."""

class SessionCreationError(SessionError):
    """Error creating browser session."""

class SessionTimeoutError(SessionError):
    """Session timeout error."""

class SessionCleanupError(SessionError):
    """Error during session cleanup."""
```

This API reference provides comprehensive documentation for all components in the LlamaIndex-AgentCore Browser Tool integration, enabling developers to effectively use the security-focused integration patterns.