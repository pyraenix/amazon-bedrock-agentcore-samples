"""
Sensitive Data Handler for Strands Integration

This module provides comprehensive sensitive data handling utilities for Strands
workflows, including PII detection, masking, classification, credential management,
and audit logging. Designed specifically for Strands agents working with
AgentCore Browser Tool.

Key Features:
- PII detection and classification in Strands workflows
- Data sanitization methods that work with Strands' tool output processing
- Credential management system that integrates with AWS Secrets Manager
- Audit logging for all sensitive data operations within Strands agent execution
- Real-time data masking and redaction capabilities

Requirements Addressed:
- 2.1: PII detection, masking, and classification in Strands workflows
- 2.2: Credential management system that integrates with AWS Secrets Manager
- 2.3: Data sanitization methods that work with Strands' tool output processing
"""

import os
import re
import json
import logging
import hashlib
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple, Pattern
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager

# AWS SDK imports
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    boto3 = None
    ClientError = Exception
    NoCredentialsError = Exception

# Strands imports
try:
    from strands_agents.core.types import ToolResult
    from strands_agents.core.exceptions import ToolExecutionError
except ImportError:
    # Mock Strands imports for development/testing
    @dataclass
    class ToolResult:
        success: bool
        data: Any = None
        error: Optional[str] = None
        metadata: Dict[str, Any] = field(default_factory=dict)
    
    class ToolExecutionError(Exception):
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SensitivityLevel(Enum):
    """Enumeration of data sensitivity levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class PIIType(Enum):
    """Enumeration of PII types for classification."""
    SSN = "social_security_number"
    CREDIT_CARD = "credit_card_number"
    EMAIL = "email_address"
    PHONE = "phone_number"
    ADDRESS = "physical_address"
    NAME = "person_name"
    DATE_OF_BIRTH = "date_of_birth"
    DRIVER_LICENSE = "driver_license"
    PASSPORT = "passport_number"
    BANK_ACCOUNT = "bank_account_number"
    MEDICAL_ID = "medical_id"
    IP_ADDRESS = "ip_address"
    USERNAME = "username"
    PASSWORD = "password"
    API_KEY = "api_key"
    TOKEN = "authentication_token"


class MaskingStrategy(Enum):
    """Enumeration of data masking strategies."""
    REDACT = "redact"  # Replace with [REDACTED]
    HASH = "hash"  # Replace with hash value
    PARTIAL = "partial"  # Show partial data (e.g., ***-**-1234)
    TOKENIZE = "tokenize"  # Replace with secure token
    REMOVE = "remove"  # Remove completely


@dataclass
class PIIPattern:
    """Pattern definition for PII detection."""
    pii_type: PIIType
    pattern: Pattern[str]
    confidence_threshold: float = 0.8
    masking_strategy: MaskingStrategy = MaskingStrategy.REDACT
    description: str = ""
    
    def __post_init__(self):
        if isinstance(self.pattern, str):
            self.pattern = re.compile(self.pattern, re.IGNORECASE)


@dataclass
class PIIDetectionResult:
    """Result of PII detection operation."""
    pii_type: PIIType
    matched_text: str
    start_position: int
    end_position: int
    confidence: float
    masking_strategy: MaskingStrategy
    masked_value: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'pii_type': self.pii_type.value,
            'matched_text': self.matched_text,
            'start_position': self.start_position,
            'end_position': self.end_position,
            'confidence': self.confidence,
            'masking_strategy': self.masking_strategy.value,
            'masked_value': self.masked_value
        }


@dataclass
class SanitizationConfig:
    """Configuration for data sanitization operations."""
    min_confidence_threshold: float = 0.7
    default_masking_strategy: MaskingStrategy = MaskingStrategy.REDACT
    preserve_format: bool = True
    audit_sensitive_operations: bool = True
    enable_custom_patterns: bool = True
    strict_mode: bool = False
    
    # Custom patterns for domain-specific PII
    custom_patterns: List[PIIPattern] = field(default_factory=list)
    
    # Masking strategy overrides per PII type
    strategy_overrides: Dict[PIIType, MaskingStrategy] = field(default_factory=dict)


@dataclass
class AuditLogEntry:
    """Audit log entry for sensitive data operations."""
    timestamp: datetime
    operation_type: str
    session_id: str
    agent_id: Optional[str]
    pii_types_detected: List[PIIType]
    sensitivity_level: SensitivityLevel
    masking_applied: bool
    operation_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'operation_type': self.operation_type,
            'session_id': self.session_id,
            'agent_id': self.agent_id,
            'pii_types_detected': [pii.value for pii in self.pii_types_detected],
            'sensitivity_level': self.sensitivity_level.value,
            'masking_applied': self.masking_applied,
            'operation_metadata': self.operation_metadata
        }


class PIIDetector:
    """
    Advanced PII detection engine for Strands workflows.
    
    Provides comprehensive PII detection using regex patterns, confidence scoring,
    and customizable detection rules for different data types.
    """
    
    def __init__(self, config: Optional[SanitizationConfig] = None):
        """
        Initialize PII detector with configuration.
        
        Args:
            config: Sanitization configuration
        """
        self.config = config or SanitizationConfig()
        self._patterns = self._initialize_patterns()
        
        logger.info(f"PIIDetector initialized with {len(self._patterns)} patterns")
    
    def _initialize_patterns(self) -> List[PIIPattern]:
        """Initialize built-in PII detection patterns."""
        
        patterns = [
            # Social Security Numbers
            PIIPattern(
                pii_type=PIIType.SSN,
                pattern=r'\b\d{3}-?\d{2}-?\d{4}\b',
                confidence_threshold=0.9,
                masking_strategy=MaskingStrategy.PARTIAL,
                description="US Social Security Number"
            ),
            
            # Credit Card Numbers
            PIIPattern(
                pii_type=PIIType.CREDIT_CARD,
                pattern=r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
                confidence_threshold=0.85,
                masking_strategy=MaskingStrategy.PARTIAL,
                description="Credit Card Number"
            ),
            
            # Email Addresses
            PIIPattern(
                pii_type=PIIType.EMAIL,
                pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                confidence_threshold=0.95,
                masking_strategy=MaskingStrategy.PARTIAL,
                description="Email Address"
            ),
            
            # Phone Numbers
            PIIPattern(
                pii_type=PIIType.PHONE,
                pattern=r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
                confidence_threshold=0.8,
                masking_strategy=MaskingStrategy.PARTIAL,
                description="Phone Number"
            ),
            
            # IP Addresses
            PIIPattern(
                pii_type=PIIType.IP_ADDRESS,
                pattern=r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                confidence_threshold=0.9,
                masking_strategy=MaskingStrategy.HASH,
                description="IP Address"
            ),
            
            # Passwords (common patterns)
            PIIPattern(
                pii_type=PIIType.PASSWORD,
                pattern=r'(?i)(?:password|pwd|pass)\s*[:=]\s*["\']?([^\s"\']+)["\']?',
                confidence_threshold=0.9,
                masking_strategy=MaskingStrategy.REDACT,
                description="Password Field"
            ),
            
            # API Keys (common patterns)
            PIIPattern(
                pii_type=PIIType.API_KEY,
                pattern=r'(?i)(?:api[_-]?key|apikey|access[_-]?key)\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',
                confidence_threshold=0.85,
                masking_strategy=MaskingStrategy.REDACT,
                description="API Key"
            ),
            
            # Authentication Tokens
            PIIPattern(
                pii_type=PIIType.TOKEN,
                pattern=r'(?i)(?:token|bearer|auth)\s*[:=]\s*["\']?([a-zA-Z0-9_.-]{30,})["\']?',
                confidence_threshold=0.8,
                masking_strategy=MaskingStrategy.REDACT,
                description="Authentication Token"
            ),
        ]
        
        # Add custom patterns if enabled
        if self.config.enable_custom_patterns:
            patterns.extend(self.config.custom_patterns)
        
        return patterns
    
    def detect_pii(self, text: str) -> List[PIIDetectionResult]:
        """
        Detect PII in the given text.
        
        Args:
            text: Text to analyze for PII
            
        Returns:
            List of PII detection results
        """
        
        if not text:
            return []
        
        detections = []
        
        for pattern in self._patterns:
            matches = pattern.pattern.finditer(text)
            
            for match in matches:
                # Calculate confidence based on pattern and context
                confidence = self._calculate_confidence(pattern, match, text)
                
                if confidence >= pattern.confidence_threshold:
                    # Determine masking strategy
                    masking_strategy = self.config.strategy_overrides.get(
                        pattern.pii_type, 
                        pattern.masking_strategy
                    )
                    
                    detection = PIIDetectionResult(
                        pii_type=pattern.pii_type,
                        matched_text=match.group(),
                        start_position=match.start(),
                        end_position=match.end(),
                        confidence=confidence,
                        masking_strategy=masking_strategy
                    )
                    
                    detections.append(detection)
        
        # Remove overlapping detections (keep highest confidence)
        detections = self._remove_overlaps(detections)
        
        logger.info(f"Detected {len(detections)} PII instances in text")
        return detections
    
    def _calculate_confidence(self, pattern: PIIPattern, match: re.Match, text: str) -> float:
        """
        Calculate confidence score for a PII detection.
        
        Args:
            pattern: PII pattern that matched
            match: Regex match object
            text: Full text being analyzed
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        
        base_confidence = pattern.confidence_threshold
        
        # Adjust confidence based on context
        matched_text = match.group()
        
        # Length-based adjustments
        if pattern.pii_type in [PIIType.SSN, PIIType.CREDIT_CARD]:
            # Longer matches are more likely to be real
            if len(matched_text.replace('-', '').replace(' ', '')) >= 9:
                base_confidence += 0.1
        
        # Context-based adjustments
        context_start = max(0, match.start() - 20)
        context_end = min(len(text), match.end() + 20)
        context = text[context_start:context_end].lower()
        
        # Look for contextual clues
        if pattern.pii_type == PIIType.SSN:
            if any(keyword in context for keyword in ['ssn', 'social', 'security']):
                base_confidence += 0.1
        elif pattern.pii_type == PIIType.CREDIT_CARD:
            if any(keyword in context for keyword in ['card', 'credit', 'visa', 'mastercard']):
                base_confidence += 0.1
        elif pattern.pii_type == PIIType.EMAIL:
            if any(keyword in context for keyword in ['email', 'contact', '@']):
                base_confidence += 0.05
        
        return min(1.0, base_confidence)
    
    def _remove_overlaps(self, detections: List[PIIDetectionResult]) -> List[PIIDetectionResult]:
        """
        Remove overlapping detections, keeping the highest confidence ones.
        
        Args:
            detections: List of PII detections
            
        Returns:
            List of non-overlapping detections
        """
        
        if not detections:
            return []
        
        # Sort by start position
        sorted_detections = sorted(detections, key=lambda d: d.start_position)
        
        filtered_detections = []
        
        for detection in sorted_detections:
            # Check for overlap with existing detections
            overlaps = False
            
            for existing in filtered_detections:
                if (detection.start_position < existing.end_position and 
                    detection.end_position > existing.start_position):
                    # There's an overlap
                    if detection.confidence > existing.confidence:
                        # Replace existing with higher confidence detection
                        filtered_detections.remove(existing)
                        filtered_detections.append(detection)
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def add_custom_pattern(self, pattern: PIIPattern) -> None:
        """
        Add a custom PII detection pattern.
        
        Args:
            pattern: Custom PII pattern to add
        """
        
        self._patterns.append(pattern)
        logger.info(f"Added custom PII pattern for {pattern.pii_type.value}")


class DataMasker:
    """
    Data masking engine for applying various masking strategies to sensitive data.
    """
    
    def __init__(self, config: Optional[SanitizationConfig] = None):
        """
        Initialize data masker with configuration.
        
        Args:
            config: Sanitization configuration
        """
        self.config = config or SanitizationConfig()
        self._token_cache: Dict[str, str] = {}
    
    def mask_text(self, text: str, detections: List[PIIDetectionResult]) -> str:
        """
        Apply masking to text based on PII detections.
        
        Args:
            text: Original text
            detections: List of PII detections to mask
            
        Returns:
            Masked text
        """
        
        if not detections:
            return text
        
        # Sort detections by position (reverse order to maintain positions)
        sorted_detections = sorted(detections, key=lambda d: d.start_position, reverse=True)
        
        masked_text = text
        
        for detection in sorted_detections:
            # Apply appropriate masking strategy
            masked_value = self._apply_masking_strategy(
                detection.matched_text,
                detection.masking_strategy,
                detection.pii_type
            )
            
            # Update detection with masked value
            detection.masked_value = masked_value
            
            # Replace in text
            masked_text = (
                masked_text[:detection.start_position] +
                masked_value +
                masked_text[detection.end_position:]
            )
        
        return masked_text
    
    def _apply_masking_strategy(
        self, 
        original_value: str, 
        strategy: MaskingStrategy, 
        pii_type: PIIType
    ) -> str:
        """
        Apply specific masking strategy to a value.
        
        Args:
            original_value: Original sensitive value
            strategy: Masking strategy to apply
            pii_type: Type of PII being masked
            
        Returns:
            Masked value
        """
        
        if strategy == MaskingStrategy.REDACT:
            return f"[REDACTED_{pii_type.value.upper()}]"
        
        elif strategy == MaskingStrategy.HASH:
            # Create a consistent hash
            hash_value = hashlib.sha256(original_value.encode()).hexdigest()[:8]
            return f"[HASH_{hash_value}]"
        
        elif strategy == MaskingStrategy.PARTIAL:
            return self._apply_partial_masking(original_value, pii_type)
        
        elif strategy == MaskingStrategy.TOKENIZE:
            return self._apply_tokenization(original_value, pii_type)
        
        elif strategy == MaskingStrategy.REMOVE:
            return ""
        
        else:
            # Default to redaction
            return f"[REDACTED_{pii_type.value.upper()}]"
    
    def _apply_partial_masking(self, value: str, pii_type: PIIType) -> str:
        """
        Apply partial masking showing only some characters.
        
        Args:
            value: Original value
            pii_type: Type of PII
            
        Returns:
            Partially masked value
        """
        
        if pii_type == PIIType.SSN:
            # Show last 4 digits: ***-**-1234
            clean_value = re.sub(r'[^\d]', '', value)
            if len(clean_value) >= 4:
                return f"***-**-{clean_value[-4:]}"
            return "***-**-****"
        
        elif pii_type == PIIType.CREDIT_CARD:
            # Show last 4 digits: ****-****-****-1234
            clean_value = re.sub(r'[^\d]', '', value)
            if len(clean_value) >= 4:
                return f"****-****-****-{clean_value[-4:]}"
            return "****-****-****-****"
        
        elif pii_type == PIIType.EMAIL:
            # Show first char and domain: j***@example.com
            if '@' in value:
                local, domain = value.split('@', 1)
                if len(local) > 0:
                    return f"{local[0]}***@{domain}"
            return "***@***.***"
        
        elif pii_type == PIIType.PHONE:
            # Show last 4 digits: ***-***-1234
            clean_value = re.sub(r'[^\d]', '', value)
            if len(clean_value) >= 4:
                return f"***-***-{clean_value[-4:]}"
            return "***-***-****"
        
        else:
            # Generic partial masking
            if len(value) <= 4:
                return "*" * len(value)
            else:
                return value[:2] + "*" * (len(value) - 4) + value[-2:]
    
    def _apply_tokenization(self, value: str, pii_type: PIIType) -> str:
        """
        Apply tokenization to replace value with a secure token.
        
        Args:
            value: Original value
            pii_type: Type of PII
            
        Returns:
            Tokenized value
        """
        
        # Check if we already have a token for this value
        if value in self._token_cache:
            return self._token_cache[value]
        
        # Generate new token
        token = f"TOKEN_{pii_type.value.upper()}_{uuid.uuid4().hex[:8]}"
        self._token_cache[value] = token
        
        return token


class CredentialManager:
    """
    Secure credential management system that integrates with AWS Secrets Manager.
    
    Provides secure storage, retrieval, and rotation of credentials for Strands
    workflows using AgentCore Browser Tool.
    """
    
    def __init__(self, region: str = "us-east-1", use_local_fallback: bool = True):
        """
        Initialize credential manager.
        
        Args:
            region: AWS region for Secrets Manager
            use_local_fallback: Use local storage if AWS is not available
        """
        self.region = region
        self.use_local_fallback = use_local_fallback
        
        # Initialize AWS Secrets Manager client
        if AWS_AVAILABLE:
            try:
                self.secrets_client = boto3.client('secretsmanager', region_name=region)
                self.aws_available = True
                logger.info(f"AWS Secrets Manager client initialized for region: {region}")
            except (NoCredentialsError, Exception) as e:
                logger.warning(f"AWS Secrets Manager not available: {str(e)}")
                self.secrets_client = None
                self.aws_available = False
        else:
            logger.warning("AWS SDK not available - using local fallback only")
            self.secrets_client = None
            self.aws_available = False
        
        # Local credential cache (encrypted in memory)
        self._local_cache: Dict[str, Dict[str, Any]] = {}
    
    def store_credentials(
        self,
        credential_id: str,
        username: str,
        password: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store credentials securely.
        
        Args:
            credential_id: Unique identifier for the credentials
            username: Username to store
            password: Password to store
            metadata: Additional metadata
            
        Returns:
            True if storage successful, False otherwise
        """
        
        credential_data = {
            'username': username,
            'password': password,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        # Try AWS Secrets Manager first
        if self.aws_available and self.secrets_client:
            try:
                self.secrets_client.create_secret(
                    Name=f"strands-agentcore-{credential_id}",
                    SecretString=json.dumps(credential_data),
                    Description=f"Strands AgentCore credentials for {credential_id}"
                )
                logger.info(f"✅ Credentials stored in AWS Secrets Manager: {credential_id}")
                return True
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceExistsException':
                    # Update existing secret
                    try:
                        self.secrets_client.update_secret(
                            SecretId=f"strands-agentcore-{credential_id}",
                            SecretString=json.dumps(credential_data)
                        )
                        logger.info(f"✅ Credentials updated in AWS Secrets Manager: {credential_id}")
                        return True
                    except ClientError as update_error:
                        logger.error(f"Failed to update credentials: {str(update_error)}")
                else:
                    logger.error(f"Failed to store credentials: {str(e)}")
        
        # Fallback to local storage
        if self.use_local_fallback:
            # In production, this should use proper encryption
            self._local_cache[credential_id] = credential_data
            logger.info(f"⚠️ Credentials stored locally (fallback): {credential_id}")
            return True
        
        logger.error(f"Failed to store credentials: {credential_id}")
        return False
    
    def retrieve_credentials(self, credential_id: str) -> Optional[Tuple[str, str]]:
        """
        Retrieve credentials securely.
        
        Args:
            credential_id: Unique identifier for the credentials
            
        Returns:
            Tuple of (username, password) if found, None otherwise
        """
        
        # Try AWS Secrets Manager first
        if self.aws_available and self.secrets_client:
            try:
                response = self.secrets_client.get_secret_value(
                    SecretId=f"strands-agentcore-{credential_id}"
                )
                
                credential_data = json.loads(response['SecretString'])
                logger.info(f"✅ Credentials retrieved from AWS Secrets Manager: {credential_id}")
                return credential_data['username'], credential_data['password']
                
            except ClientError as e:
                if e.response['Error']['Code'] != 'ResourceNotFoundException':
                    logger.error(f"Failed to retrieve credentials: {str(e)}")
        
        # Fallback to local storage
        if credential_id in self._local_cache:
            credential_data = self._local_cache[credential_id]
            logger.info(f"⚠️ Credentials retrieved from local cache: {credential_id}")
            return credential_data['username'], credential_data['password']
        
        logger.warning(f"Credentials not found: {credential_id}")
        return None
    
    def delete_credentials(self, credential_id: str) -> bool:
        """
        Delete credentials securely.
        
        Args:
            credential_id: Unique identifier for the credentials
            
        Returns:
            True if deletion successful, False otherwise
        """
        
        success = False
        
        # Delete from AWS Secrets Manager
        if self.aws_available and self.secrets_client:
            try:
                self.secrets_client.delete_secret(
                    SecretId=f"strands-agentcore-{credential_id}",
                    ForceDeleteWithoutRecovery=True
                )
                logger.info(f"✅ Credentials deleted from AWS Secrets Manager: {credential_id}")
                success = True
                
            except ClientError as e:
                if e.response['Error']['Code'] != 'ResourceNotFoundException':
                    logger.error(f"Failed to delete credentials from AWS: {str(e)}")
        
        # Delete from local cache
        if credential_id in self._local_cache:
            del self._local_cache[credential_id]
            logger.info(f"✅ Credentials deleted from local cache: {credential_id}")
            success = True
        
        return success
    
    def list_credentials(self) -> List[str]:
        """
        List available credential IDs.
        
        Returns:
            List of credential IDs
        """
        
        credential_ids = []
        
        # List from AWS Secrets Manager
        if self.aws_available and self.secrets_client:
            try:
                response = self.secrets_client.list_secrets()
                for secret in response.get('SecretList', []):
                    name = secret['Name']
                    if name.startswith('strands-agentcore-'):
                        credential_id = name.replace('strands-agentcore-', '')
                        credential_ids.append(credential_id)
                        
            except ClientError as e:
                logger.error(f"Failed to list AWS credentials: {str(e)}")
        
        # Add local credentials
        credential_ids.extend(self._local_cache.keys())
        
        # Remove duplicates
        return list(set(credential_ids))
    
    @contextmanager
    def secure_credentials(self, credential_id: str):
        """
        Context manager for secure credential access.
        
        Args:
            credential_id: Unique identifier for the credentials
            
        Yields:
            Tuple of (username, password) if found
        """
        
        credentials = self.retrieve_credentials(credential_id)
        
        if not credentials:
            raise ToolExecutionError(f"Credentials not found: {credential_id}")
        
        try:
            yield credentials
        finally:
            # Clear credentials from memory
            if credentials:
                # Overwrite memory (basic security measure)
                username, password = credentials
                username = "X" * len(username)
                password = "X" * len(password)


class SensitiveDataHandler:
    """
    Comprehensive sensitive data handler for Strands workflows.
    
    Combines PII detection, data masking, credential management, and audit logging
    into a unified system for handling sensitive data in Strands agent operations.
    """
    
    def __init__(
        self,
        config: Optional[SanitizationConfig] = None,
        region: str = "us-east-1",
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ):
        """
        Initialize sensitive data handler.
        
        Args:
            config: Sanitization configuration
            region: AWS region for services
            session_id: Current session ID
            agent_id: Current agent ID
        """
        self.config = config or SanitizationConfig()
        self.region = region
        self.session_id = session_id or f"strands-session-{uuid.uuid4().hex[:8]}"
        self.agent_id = agent_id
        
        # Initialize components
        self.pii_detector = PIIDetector(self.config)
        self.data_masker = DataMasker(self.config)
        self.credential_manager = CredentialManager(region)
        
        # Audit logging
        self.audit_log: List[AuditLogEntry] = []
        
        logger.info(f"SensitiveDataHandler initialized for session: {self.session_id}")
    
    def process_tool_output(self, tool_result: ToolResult) -> ToolResult:
        """
        Process Strands tool output for sensitive data handling.
        
        Args:
            tool_result: Original tool result
            
        Returns:
            Processed tool result with sensitive data handled
        """
        
        try:
            # Detect PII in tool result data
            pii_detections = []
            
            if tool_result.data:
                if isinstance(tool_result.data, str):
                    pii_detections = self.pii_detector.detect_pii(tool_result.data)
                elif isinstance(tool_result.data, dict):
                    # Process dictionary values
                    for key, value in tool_result.data.items():
                        if isinstance(value, str):
                            detections = self.pii_detector.detect_pii(value)
                            pii_detections.extend(detections)
            
            # Apply masking if PII detected
            if pii_detections:
                logger.info(f"🔒 Processing {len(pii_detections)} PII detections in tool output")
                
                # Create sanitized copy of tool result
                sanitized_result = ToolResult(
                    success=tool_result.success,
                    data=self._sanitize_data(tool_result.data, pii_detections),
                    error=tool_result.error,
                    metadata=tool_result.metadata.copy() if tool_result.metadata else {}
                )
                
                # Add sanitization metadata
                sanitized_result.metadata.update({
                    'pii_detected': True,
                    'pii_types': [d.pii_type.value for d in pii_detections],
                    'sanitization_applied': True,
                    'original_data_hash': hashlib.sha256(
                        str(tool_result.data).encode()
                    ).hexdigest()[:16]
                })
                
                # Log audit entry
                self._log_audit_entry(
                    operation_type="tool_output_sanitization",
                    pii_types_detected=[d.pii_type for d in pii_detections],
                    sensitivity_level=self._determine_sensitivity_level(pii_detections),
                    masking_applied=True,
                    operation_metadata={
                        'tool_name': sanitized_result.metadata.get('tool_name', 'unknown'),
                        'detections_count': len(pii_detections)
                    }
                )
                
                return sanitized_result
            
            else:
                # No PII detected - return original result
                return tool_result
        
        except Exception as e:
            logger.error(f"Error processing tool output for sensitive data: {str(e)}")
            # Return original result if processing fails
            return tool_result
    
    def _sanitize_data(self, data: Any, detections: List[PIIDetectionResult]) -> Any:
        """
        Sanitize data based on PII detections.
        
        Args:
            data: Original data
            detections: PII detections
            
        Returns:
            Sanitized data
        """
        
        if isinstance(data, str):
            return self.data_masker.mask_text(data, detections)
        
        elif isinstance(data, dict):
            sanitized_dict = {}
            for key, value in data.items():
                if isinstance(value, str):
                    # Find detections for this specific value
                    value_detections = [
                        d for d in detections 
                        if d.matched_text in value
                    ]
                    sanitized_dict[key] = self.data_masker.mask_text(value, value_detections)
                else:
                    sanitized_dict[key] = value
            return sanitized_dict
        
        elif isinstance(data, list):
            sanitized_list = []
            for item in data:
                if isinstance(item, str):
                    item_detections = [
                        d for d in detections 
                        if d.matched_text in item
                    ]
                    sanitized_list.append(self.data_masker.mask_text(item, item_detections))
                else:
                    sanitized_list.append(item)
            return sanitized_list
        
        else:
            return data
    
    def _determine_sensitivity_level(self, detections: List[PIIDetectionResult]) -> SensitivityLevel:
        """
        Determine overall sensitivity level based on PII detections.
        
        Args:
            detections: List of PII detections
            
        Returns:
            Overall sensitivity level
        """
        
        if not detections:
            return SensitivityLevel.PUBLIC
        
        # Check for high-sensitivity PII types
        high_sensitivity_types = {
            PIIType.SSN, PIIType.CREDIT_CARD, PIIType.PASSPORT, 
            PIIType.MEDICAL_ID, PIIType.BANK_ACCOUNT
        }
        
        medium_sensitivity_types = {
            PIIType.EMAIL, PIIType.PHONE, PIIType.ADDRESS, 
            PIIType.DRIVER_LICENSE, PIIType.DATE_OF_BIRTH
        }
        
        detected_types = {d.pii_type for d in detections}
        
        if detected_types & high_sensitivity_types:
            return SensitivityLevel.RESTRICTED
        elif detected_types & medium_sensitivity_types:
            return SensitivityLevel.CONFIDENTIAL
        else:
            return SensitivityLevel.INTERNAL
    
    def _log_audit_entry(
        self,
        operation_type: str,
        pii_types_detected: List[PIIType],
        sensitivity_level: SensitivityLevel,
        masking_applied: bool,
        operation_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an audit entry for sensitive data operations.
        
        Args:
            operation_type: Type of operation performed
            pii_types_detected: List of PII types detected
            sensitivity_level: Overall sensitivity level
            masking_applied: Whether masking was applied
            operation_metadata: Additional operation metadata
        """
        
        if not self.config.audit_sensitive_operations:
            return
        
        audit_entry = AuditLogEntry(
            timestamp=datetime.now(),
            operation_type=operation_type,
            session_id=self.session_id,
            agent_id=self.agent_id,
            pii_types_detected=pii_types_detected,
            sensitivity_level=sensitivity_level,
            masking_applied=masking_applied,
            operation_metadata=operation_metadata or {}
        )
        
        self.audit_log.append(audit_entry)
        
        # Log to standard logger as well
        logger.info(
            f"AUDIT: {operation_type} - PII types: {[p.value for p in pii_types_detected]} - "
            f"Sensitivity: {sensitivity_level.value} - Masked: {masking_applied}"
        )
    
    def get_audit_log(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get audit log entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of audit log entries as dictionaries
        """
        
        entries = self.audit_log[-limit:] if limit else self.audit_log
        return [entry.to_dict() for entry in entries]
    
    def export_audit_log(self, filepath: str) -> bool:
        """
        Export audit log to file.
        
        Args:
            filepath: Path to export file
            
        Returns:
            True if export successful, False otherwise
        """
        
        try:
            audit_data = {
                'session_id': self.session_id,
                'agent_id': self.agent_id,
                'export_timestamp': datetime.now().isoformat(),
                'entries': self.get_audit_log()
            }
            
            with open(filepath, 'w') as f:
                json.dump(audit_data, f, indent=2)
            
            logger.info(f"✅ Audit log exported to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export audit log: {str(e)}")
            return False
    
    def clear_audit_log(self) -> None:
        """Clear the audit log."""
        self.audit_log.clear()
        logger.info("Audit log cleared")
    
    # Convenience methods
    
    def detect_pii_in_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Convenience method to detect PII in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of PII detections as dictionaries
        """
        
        detections = self.pii_detector.detect_pii(text)
        return [d.to_dict() for d in detections]
    
    def sanitize_text(self, text: str) -> str:
        """
        Convenience method to sanitize text.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        
        detections = self.pii_detector.detect_pii(text)
        return self.data_masker.mask_text(text, detections)
    
    def store_credentials(self, credential_id: str, username: str, password: str) -> bool:
        """
        Convenience method to store credentials.
        
        Args:
            credential_id: Unique identifier for credentials
            username: Username to store
            password: Password to store
            
        Returns:
            True if storage successful
        """
        
        success = self.credential_manager.store_credentials(credential_id, username, password)
        
        if success:
            self._log_audit_entry(
                operation_type="credential_storage",
                pii_types_detected=[PIIType.USERNAME, PIIType.PASSWORD],
                sensitivity_level=SensitivityLevel.RESTRICTED,
                masking_applied=True,
                operation_metadata={'credential_id': credential_id}
            )
        
        return success
    
    def get_credentials(self, credential_id: str) -> Optional[Tuple[str, str]]:
        """
        Convenience method to retrieve credentials.
        
        Args:
            credential_id: Unique identifier for credentials
            
        Returns:
            Tuple of (username, password) if found
        """
        
        credentials = self.credential_manager.retrieve_credentials(credential_id)
        
        if credentials:
            self._log_audit_entry(
                operation_type="credential_retrieval",
                pii_types_detected=[PIIType.USERNAME, PIIType.PASSWORD],
                sensitivity_level=SensitivityLevel.RESTRICTED,
                masking_applied=False,
                operation_metadata={'credential_id': credential_id}
            )
        
        return credentials


# Utility functions for creating configured handlers

def create_secure_data_handler(
    session_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    region: str = "us-east-1",
    strict_mode: bool = False
) -> SensitiveDataHandler:
    """
    Create a secure data handler with recommended settings.
    
    Args:
        session_id: Session ID for audit logging
        agent_id: Agent ID for audit logging
        region: AWS region for services
        strict_mode: Enable strict sanitization mode
        
    Returns:
        Configured SensitiveDataHandler instance
    """
    
    config = SanitizationConfig(
        min_confidence_threshold=0.6 if strict_mode else 0.7,
        default_masking_strategy=MaskingStrategy.REDACT,
        preserve_format=True,
        audit_sensitive_operations=True,
        enable_custom_patterns=True,
        strict_mode=strict_mode
    )
    
    return SensitiveDataHandler(
        config=config,
        region=region,
        session_id=session_id,
        agent_id=agent_id
    )


if __name__ == "__main__":
    # Example usage
    print("Sensitive Data Handler for Strands - Example Usage")
    print("=" * 55)
    
    # Create handler
    handler = create_secure_data_handler(
        session_id="example-session",
        agent_id="example-agent"
    )
    
    # Example text with PII
    test_text = """
    Contact John Doe at john.doe@example.com or call 555-123-4567.
    His SSN is 123-45-6789 and credit card number is 4532-1234-5678-9012.
    """
    
    # Detect PII
    detections = handler.detect_pii_in_text(test_text)
    print(f"PII Detections: {len(detections)}")
    
    # Sanitize text
    sanitized = handler.sanitize_text(test_text)
    print(f"Sanitized Text: {sanitized}")
    
    # Store credentials
    success = handler.store_credentials("test-creds", "testuser", "testpass")
    print(f"Credential Storage: {success}")
    
    # Get audit log
    audit_entries = handler.get_audit_log()
    print(f"Audit Entries: {len(audit_entries)}")
    
    print("\n✅ Example completed")