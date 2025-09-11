"""
Browser-Use Sensitive Data Handler

This module provides comprehensive utilities for detecting, masking, and securely handling
sensitive information during browser-use operations with AgentCore Browser Tool.
Includes PII detection, credential security, and data classification specifically
designed for browser automation workflows.
"""

import re
import hashlib
import logging
import json
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import base64
import secrets


class DataClassification(Enum):
    """Data classification levels for sensitive information."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class PIIType(Enum):
    """Types of personally identifiable information."""
    SSN = "ssn"
    EMAIL = "email"
    PHONE = "phone"
    CREDIT_CARD = "credit_card"
    DATE_OF_BIRTH = "date_of_birth"
    ADDRESS = "address"
    NAME = "name"
    MEDICAL_RECORD = "medical_record"
    DRIVER_LICENSE = "driver_license"
    PASSPORT = "passport"
    BANK_ACCOUNT = "bank_account"
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"


class ComplianceFramework(Enum):
    """Compliance frameworks for data handling."""
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    GDPR = "gdpr"
    SOX = "sox"
    FERPA = "ferpa"
    CCPA = "ccpa"


@dataclass
class PIIPattern:
    """Pattern definition for PII detection."""
    pii_type: PIIType
    pattern: str
    confidence_threshold: float = 0.8
    context_keywords: List[str] = field(default_factory=list)
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)


@dataclass
class SensitiveDataContext:
    """Context information for sensitive data handling."""
    data_type: str
    classification: DataClassification
    compliance_requirements: List[ComplianceFramework]
    retention_policy: str
    masking_rules: Dict[str, str] = field(default_factory=dict)
    audit_level: str = "standard"
    encryption_required: bool = True


@dataclass
class DetectionResult:
    """Result of PII detection operation."""
    pii_type: PIIType
    value: str
    masked_value: str
    confidence: float
    start_position: int
    end_position: int
    context: str
    compliance_impact: List[ComplianceFramework]


class BrowserUseSensitiveDataHandler:
    """
    Comprehensive sensitive data handler for browser-use operations.
    
    Provides PII detection, masking, credential security, and compliance
    validation specifically designed for browser automation workflows
    with AgentCore Browser Tool integration.
    """
    
    # PII Detection Patterns
    PII_PATTERNS = [
        PIIPattern(
            pii_type=PIIType.SSN,
            pattern=r'\b(?:\d{3}-?\d{2}-?\d{4}|\d{9})\b',
            confidence_threshold=0.9,
            context_keywords=['ssn', 'social', 'security', 'tax', 'id'],
            compliance_frameworks=[ComplianceFramework.HIPAA, ComplianceFramework.SOX]
        ),
        PIIPattern(
            pii_type=PIIType.EMAIL,
            pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            confidence_threshold=0.95,
            context_keywords=['email', 'mail', 'contact'],
            compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.CCPA]
        ),
        PIIPattern(
            pii_type=PIIType.PHONE,
            pattern=r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            confidence_threshold=0.85,
            context_keywords=['phone', 'tel', 'mobile', 'cell'],
            compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.CCPA]
        ),
        PIIPattern(
            pii_type=PIIType.CREDIT_CARD,
            pattern=r'\b(?:4[0-9]{3}[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}|5[1-5][0-9]{2}[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}|3[47][0-9]{1}[-\s]?[0-9]{6}[-\s]?[0-9]{5}|3[0-9]{3}[-\s]?[0-9]{6}[-\s]?[0-9]{5}|6(?:011|5[0-9]{2})[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4})\b',
            confidence_threshold=0.95,
            context_keywords=['card', 'credit', 'payment', 'visa', 'mastercard'],
            compliance_frameworks=[ComplianceFramework.PCI_DSS]
        ),
        PIIPattern(
            pii_type=PIIType.DATE_OF_BIRTH,
            pattern=r'\b(?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12][0-9]|3[01])[-/](?:19|20)\d{2}\b',
            confidence_threshold=0.8,
            context_keywords=['birth', 'dob', 'born', 'birthday'],
            compliance_frameworks=[ComplianceFramework.HIPAA, ComplianceFramework.GDPR]
        ),
        PIIPattern(
            pii_type=PIIType.MEDICAL_RECORD,
            pattern=r'\b(?:MRN|MR|MEDICAL)\s*[#:]?\s*([A-Z0-9]{6,12})\b',
            confidence_threshold=0.9,
            context_keywords=['medical', 'patient', 'record', 'mrn', 'health'],
            compliance_frameworks=[ComplianceFramework.HIPAA]
        ),
        PIIPattern(
            pii_type=PIIType.IP_ADDRESS,
            pattern=r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            confidence_threshold=0.9,
            context_keywords=['ip', 'address', 'network'],
            compliance_frameworks=[ComplianceFramework.GDPR]
        )
    ]
    
    def __init__(self, 
                 compliance_frameworks: Optional[List[ComplianceFramework]] = None,
                 custom_patterns: Optional[List[PIIPattern]] = None):
        """
        Initialize the sensitive data handler.
        
        Args:
            compliance_frameworks: List of compliance frameworks to enforce
            custom_patterns: Additional custom PII patterns
        """
        self.logger = logging.getLogger(__name__)
        self.compliance_frameworks = compliance_frameworks or []
        self.patterns = self.PII_PATTERNS.copy()
        
        if custom_patterns:
            self.patterns.extend(custom_patterns)
        
        # Note: We include all patterns by default for comprehensive PII detection
        # Compliance filtering is applied during validation, not detection
        # This ensures we detect all PII types regardless of specific compliance framework
        
        self.logger.info(f"Initialized with {len(self.patterns)} PII patterns")
    
    def detect_pii(self, text: str, context: Optional[str] = None) -> List[DetectionResult]:
        """
        Detect personally identifiable information in text.
        
        Args:
            text: Text to analyze for PII
            context: Additional context for improved detection
            
        Returns:
            List of detected PII items with details
        """
        if not text:
            return []
        
        detections = []
        text_lower = text.lower()
        context_lower = (context or "").lower()
        
        for pattern in self.patterns:
            matches = re.finditer(pattern.pattern, text, re.IGNORECASE)
            
            for match in matches:
                # Calculate confidence based on context
                confidence = pattern.confidence_threshold
                
                # Boost confidence if context keywords are present
                if pattern.context_keywords:
                    context_text = text_lower + " " + context_lower
                    keyword_matches = sum(1 for keyword in pattern.context_keywords 
                                        if keyword in context_text)
                    if keyword_matches > 0:
                        confidence = min(1.0, confidence + (keyword_matches * 0.1))
                
                # Create masked value
                masked_value = self._mask_value(match.group(), pattern.pii_type)
                
                detection = DetectionResult(
                    pii_type=pattern.pii_type,
                    value=match.group(),
                    masked_value=masked_value,
                    confidence=confidence,
                    start_position=match.start(),
                    end_position=match.end(),
                    context=context or "",
                    compliance_impact=pattern.compliance_frameworks
                )
                
                detections.append(detection)
        
        # Sort by position for consistent processing
        detections.sort(key=lambda x: x.start_position)
        
        self.logger.info(f"Detected {len(detections)} PII items in text")
        return detections
    
    def _mask_value(self, value: str, pii_type: PIIType) -> str:
        """
        Mask a PII value based on its type.
        
        Args:
            value: Original value to mask
            pii_type: Type of PII
            
        Returns:
            Masked value
        """
        if pii_type == PIIType.SSN:
            return "XXX-XX-" + value[-4:] if len(value) >= 4 else "XXX-XX-XXXX"
        elif pii_type == PIIType.EMAIL:
            parts = value.split('@')
            if len(parts) == 2:
                username = parts[0]
                domain = parts[1]
                masked_username = username[0] + '*' * (len(username) - 2) + username[-1] if len(username) > 2 else '*' * len(username)
                return f"{masked_username}@{domain}"
            return "***@***.***"
        elif pii_type == PIIType.PHONE:
            # Keep last 4 digits
            digits_only = re.sub(r'\D', '', value)
            if len(digits_only) >= 4:
                return "XXX-XXX-" + digits_only[-4:]
            return "XXX-XXX-XXXX"
        elif pii_type == PIIType.CREDIT_CARD:
            # Keep last 4 digits
            digits_only = re.sub(r'\D', '', value)
            if len(digits_only) >= 4:
                return "**** **** **** " + digits_only[-4:]
            return "**** **** **** ****"
        elif pii_type == PIIType.DATE_OF_BIRTH:
            return "XX/XX/XXXX"
        elif pii_type == PIIType.MEDICAL_RECORD:
            return "MRN-XXXXXXXX"
        elif pii_type == PIIType.IP_ADDRESS:
            parts = value.split('.')
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.XXX.XXX"
            return "XXX.XXX.XXX.XXX"
        else:
            # Generic masking
            if len(value) <= 3:
                return '*' * len(value)
            return value[0] + '*' * (len(value) - 2) + value[-1]
    
    def mask_text(self, text: str, context: Optional[str] = None) -> Tuple[str, List[DetectionResult]]:
        """
        Mask all PII in text and return both masked text and detection details.
        
        Args:
            text: Text to mask
            context: Additional context for detection
            
        Returns:
            Tuple of (masked_text, detection_results)
        """
        if not text:
            return text, []
        
        detections = self.detect_pii(text, context)
        
        if not detections:
            return text, []
        
        # Apply masking from end to start to preserve positions
        masked_text = text
        for detection in reversed(detections):
            masked_text = (
                masked_text[:detection.start_position] + 
                detection.masked_value + 
                masked_text[detection.end_position:]
            )
        
        self.logger.info(f"Masked {len(detections)} PII items in text")
        return masked_text, detections
    
    def classify_data(self, text: str, context: Optional[Dict[str, Any]] = None) -> DataClassification:
        """
        Classify data based on sensitivity level.
        
        Args:
            text: Text to classify
            context: Additional context for classification
            
        Returns:
            Data classification level
        """
        if not text:
            return DataClassification.PUBLIC
        
        detections = self.detect_pii(text)
        
        if not detections:
            return DataClassification.PUBLIC
        
        # Determine classification based on PII types found
        high_sensitivity_types = {PIIType.SSN, PIIType.CREDIT_CARD, PIIType.MEDICAL_RECORD}
        medium_sensitivity_types = {PIIType.EMAIL, PIIType.PHONE, PIIType.DATE_OF_BIRTH}
        
        detected_types = {d.pii_type for d in detections}
        
        if detected_types & high_sensitivity_types:
            return DataClassification.RESTRICTED
        elif detected_types & medium_sensitivity_types:
            return DataClassification.CONFIDENTIAL
        else:
            return DataClassification.INTERNAL
    
    def validate_compliance(self, 
                          text: str, 
                          required_frameworks: List[ComplianceFramework],
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate text against compliance requirements.
        
        Args:
            text: Text to validate
            required_frameworks: Required compliance frameworks
            context: Additional context for validation
            
        Returns:
            Compliance validation results
        """
        detections = self.detect_pii(text)
        
        violations = []
        warnings = []
        
        for detection in detections:
            for framework in required_frameworks:
                if framework in detection.compliance_impact:
                    if detection.confidence > 0.8:
                        violations.append({
                            'framework': framework.value,
                            'pii_type': detection.pii_type.value,
                            'position': f"{detection.start_position}-{detection.end_position}",
                            'confidence': detection.confidence
                        })
                    else:
                        warnings.append({
                            'framework': framework.value,
                            'pii_type': detection.pii_type.value,
                            'position': f"{detection.start_position}-{detection.end_position}",
                            'confidence': detection.confidence
                        })
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'warnings': warnings,
            'total_pii_detected': len(detections),
            'frameworks_checked': [f.value for f in required_frameworks]
        }


class BrowserUseCredentialManager:
    """
    Secure credential management for browser-use operations.
    
    Provides secure storage, retrieval, and handling of credentials
    within AgentCore's isolated environment.
    """
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        """
        Initialize the credential manager.
        
        Args:
            encryption_key: Optional encryption key. Generated if not provided.
        """
        self.logger = logging.getLogger(__name__)
        self.encryption_key = encryption_key or self._generate_encryption_key()
        self.credentials_store: Dict[str, Dict[str, Any]] = {}
        self.access_log: List[Dict[str, Any]] = []
    
    def _generate_encryption_key(self) -> bytes:
        """Generate a secure encryption key."""
        return secrets.token_bytes(32)
    
    def _encrypt_value(self, value: str) -> str:
        """Encrypt a credential value."""
        # Simple base64 encoding for demo - in production use proper encryption
        encoded = base64.b64encode(value.encode()).decode()
        return encoded
    
    def _decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a credential value."""
        # Simple base64 decoding for demo - in production use proper decryption
        try:
            decoded = base64.b64decode(encrypted_value.encode()).decode()
            return decoded
        except Exception as e:
            self.logger.error(f"Failed to decrypt value: {e}")
            raise
    
    def store_credential(self, 
                        credential_id: str, 
                        credential_type: str,
                        value: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Securely store a credential.
        
        Args:
            credential_id: Unique identifier for the credential
            credential_type: Type of credential (password, api_key, token, etc.)
            value: Credential value to encrypt and store
            metadata: Additional metadata about the credential
        """
        encrypted_value = self._encrypt_value(value)
        
        self.credentials_store[credential_id] = {
            'type': credential_type,
            'encrypted_value': encrypted_value,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat(),
            'access_count': 0,
            'last_accessed': None
        }
        
        self.logger.info(f"Stored credential: {credential_id} (type: {credential_type})")
    
    def retrieve_credential(self, credential_id: str) -> Optional[str]:
        """
        Retrieve and decrypt a credential.
        
        Args:
            credential_id: Identifier of the credential to retrieve
            
        Returns:
            Decrypted credential value or None if not found
        """
        if credential_id not in self.credentials_store:
            self.logger.warning(f"Credential not found: {credential_id}")
            return None
        
        credential_data = self.credentials_store[credential_id]
        
        try:
            decrypted_value = self._decrypt_value(credential_data['encrypted_value'])
            
            # Update access tracking
            credential_data['access_count'] += 1
            credential_data['last_accessed'] = datetime.now().isoformat()
            
            # Log access
            self.access_log.append({
                'credential_id': credential_id,
                'accessed_at': datetime.now().isoformat(),
                'type': credential_data['type']
            })
            
            self.logger.info(f"Retrieved credential: {credential_id}")
            return decrypted_value
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve credential {credential_id}: {e}")
            return None
    
    def delete_credential(self, credential_id: str) -> bool:
        """
        Delete a stored credential.
        
        Args:
            credential_id: Identifier of the credential to delete
            
        Returns:
            True if deleted, False if not found
        """
        if credential_id in self.credentials_store:
            del self.credentials_store[credential_id]
            self.logger.info(f"Deleted credential: {credential_id}")
            return True
        else:
            self.logger.warning(f"Credential not found for deletion: {credential_id}")
            return False
    
    def list_credentials(self) -> List[Dict[str, Any]]:
        """
        List all stored credentials (without values).
        
        Returns:
            List of credential metadata
        """
        credentials = []
        for cred_id, cred_data in self.credentials_store.items():
            credentials.append({
                'credential_id': cred_id,
                'type': cred_data['type'],
                'created_at': cred_data['created_at'],
                'access_count': cred_data['access_count'],
                'last_accessed': cred_data['last_accessed'],
                'metadata': cred_data['metadata']
            })
        return credentials
    
    def get_access_log(self) -> List[Dict[str, Any]]:
        """
        Get the credential access log.
        
        Returns:
            List of access log entries
        """
        return self.access_log.copy()
    
    def clear_all_credentials(self) -> None:
        """Clear all stored credentials (emergency cleanup)."""
        self.credentials_store.clear()
        self.access_log.clear()
        self.logger.warning("All credentials cleared")


class BrowserUseDataClassifier:
    """
    Data classification utility for browser-use operations.
    
    Provides automated classification of data based on content,
    context, and compliance requirements.
    """
    
    def __init__(self, 
                 compliance_frameworks: Optional[List[ComplianceFramework]] = None):
        """
        Initialize the data classifier.
        
        Args:
            compliance_frameworks: Compliance frameworks to consider
        """
        self.logger = logging.getLogger(__name__)
        self.compliance_frameworks = compliance_frameworks or []
        self.pii_handler = BrowserUseSensitiveDataHandler(compliance_frameworks)
    
    def classify_form_data(self, form_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Classify form data based on field names and values.
        
        Args:
            form_data: Dictionary of form field names and values
            
        Returns:
            Classification results with recommendations
        """
        classifications = {}
        overall_classification = DataClassification.PUBLIC
        pii_detected = []
        compliance_issues = []
        
        for field_name, field_value in form_data.items():
            # Analyze field name for hints
            field_context = self._analyze_field_name(field_name)
            
            # Detect PII in field value
            detections = self.pii_handler.detect_pii(str(field_value), field_name)
            
            # Determine field classification
            if detections:
                field_classification = DataClassification.CONFIDENTIAL
                for detection in detections:
                    if detection.pii_type in {PIIType.SSN, PIIType.CREDIT_CARD, PIIType.MEDICAL_RECORD}:
                        field_classification = DataClassification.RESTRICTED
                        break
                pii_detected.extend(detections)
            else:
                field_classification = field_context.get('classification', DataClassification.PUBLIC)
            
            classifications[field_name] = {
                'classification': field_classification,
                'pii_detected': detections,
                'context': field_context,
                'recommendations': self._get_field_recommendations(field_classification, detections)
            }
            
            # Update overall classification
            if field_classification.value == DataClassification.RESTRICTED.value:
                overall_classification = DataClassification.RESTRICTED
            elif (field_classification.value == DataClassification.CONFIDENTIAL.value and 
                  overall_classification.value != DataClassification.RESTRICTED.value):
                overall_classification = DataClassification.CONFIDENTIAL
        
        # Check compliance requirements
        if self.compliance_frameworks and pii_detected:
            for detection in pii_detected:
                for framework in self.compliance_frameworks:
                    if framework in detection.compliance_impact:
                        compliance_issues.append({
                            'framework': framework.value,
                            'pii_type': detection.pii_type.value,
                            'field': field_name,
                            'severity': 'high' if detection.confidence > 0.9 else 'medium'
                        })
        
        return {
            'overall_classification': overall_classification,
            'field_classifications': classifications,
            'total_pii_detected': len(pii_detected),
            'compliance_issues': compliance_issues,
            'recommendations': self._get_overall_recommendations(overall_classification, compliance_issues)
        }
    
    def _analyze_field_name(self, field_name: str) -> Dict[str, Any]:
        """Analyze field name to determine likely content type."""
        field_lower = field_name.lower()
        
        # Common field patterns
        if any(keyword in field_lower for keyword in ['ssn', 'social', 'security']):
            return {'classification': DataClassification.RESTRICTED, 'likely_pii': PIIType.SSN}
        elif any(keyword in field_lower for keyword in ['email', 'mail']):
            return {'classification': DataClassification.CONFIDENTIAL, 'likely_pii': PIIType.EMAIL}
        elif any(keyword in field_lower for keyword in ['phone', 'tel', 'mobile']):
            return {'classification': DataClassification.CONFIDENTIAL, 'likely_pii': PIIType.PHONE}
        elif any(keyword in field_lower for keyword in ['card', 'credit', 'payment']):
            return {'classification': DataClassification.RESTRICTED, 'likely_pii': PIIType.CREDIT_CARD}
        elif any(keyword in field_lower for keyword in ['birth', 'dob', 'birthday']):
            return {'classification': DataClassification.CONFIDENTIAL, 'likely_pii': PIIType.DATE_OF_BIRTH}
        elif any(keyword in field_lower for keyword in ['address', 'street', 'city', 'zip']):
            return {'classification': DataClassification.CONFIDENTIAL, 'likely_pii': PIIType.ADDRESS}
        elif any(keyword in field_lower for keyword in ['name', 'first', 'last', 'full']):
            return {'classification': DataClassification.CONFIDENTIAL, 'likely_pii': PIIType.NAME}
        else:
            return {'classification': DataClassification.PUBLIC, 'likely_pii': None}
    
    def _get_field_recommendations(self, 
                                 classification: DataClassification, 
                                 detections: List[DetectionResult]) -> List[str]:
        """Get recommendations for handling a specific field."""
        recommendations = []
        
        if classification == DataClassification.RESTRICTED:
            recommendations.extend([
                "Use secure input methods",
                "Enable field-level encryption",
                "Implement audit logging",
                "Require additional authentication"
            ])
        elif classification == DataClassification.CONFIDENTIAL:
            recommendations.extend([
                "Mask field in logs",
                "Use secure transmission",
                "Implement access controls"
            ])
        
        if detections:
            pii_types = {d.pii_type for d in detections}
            if PIIType.CREDIT_CARD in pii_types:
                recommendations.append("Ensure PCI-DSS compliance")
            if PIIType.MEDICAL_RECORD in pii_types:
                recommendations.append("Ensure HIPAA compliance")
            if PIIType.SSN in pii_types:
                recommendations.append("Implement strong access controls")
        
        return recommendations
    
    def _get_overall_recommendations(self, 
                                   classification: DataClassification,
                                   compliance_issues: List[Dict[str, Any]]) -> List[str]:
        """Get overall recommendations for the form."""
        recommendations = []
        
        if classification == DataClassification.RESTRICTED:
            recommendations.extend([
                "Use AgentCore's micro-VM isolation",
                "Enable comprehensive audit logging",
                "Implement session recording",
                "Use encrypted data transmission",
                "Require multi-factor authentication"
            ])
        
        if compliance_issues:
            frameworks = {issue['framework'] for issue in compliance_issues}
            for framework in frameworks:
                recommendations.append(f"Ensure {framework.upper()} compliance measures")
        
        return recommendations


# Convenience functions for common operations
def detect_and_mask_pii(text: str, 
                       compliance_frameworks: Optional[List[ComplianceFramework]] = None) -> Tuple[str, List[DetectionResult]]:
    """
    Convenience function to detect and mask PII in text.
    
    Args:
        text: Text to process
        compliance_frameworks: Compliance frameworks to consider
        
    Returns:
        Tuple of (masked_text, detection_results)
    """
    handler = BrowserUseSensitiveDataHandler(compliance_frameworks)
    return handler.mask_text(text)


def classify_sensitive_data(text: str, 
                          context: Optional[Dict[str, Any]] = None) -> DataClassification:
    """
    Convenience function to classify data sensitivity.
    
    Args:
        text: Text to classify
        context: Additional context
        
    Returns:
        Data classification level
    """
    handler = BrowserUseSensitiveDataHandler()
    return handler.classify_data(text, context)


# Example usage and testing
if __name__ == "__main__":
    def example_usage():
        """Example usage of the sensitive data handler."""
        
        # Example 1: PII Detection and Masking
        sample_text = """
        Patient John Doe, SSN: 123-45-6789, was born on 03/15/1985.
        Contact email: john.doe@email.com, phone: (555) 123-4567.
        Credit card: 4532-1234-5678-9012 for payment.
        Medical Record Number: MRN-ABC123456
        """
        
        handler = BrowserUseSensitiveDataHandler([ComplianceFramework.HIPAA, ComplianceFramework.PCI_DSS])
        
        # Detect PII
        detections = handler.detect_pii(sample_text)
        print(f"Detected {len(detections)} PII items:")
        for detection in detections:
            print(f"  - {detection.pii_type.value}: {detection.value} -> {detection.masked_value}")
        
        # Mask text
        masked_text, _ = handler.mask_text(sample_text)
        print(f"\nMasked text:\n{masked_text}")
        
        # Classify data
        classification = handler.classify_data(sample_text)
        print(f"\nData classification: {classification.value}")
        
        # Validate compliance
        compliance_result = handler.validate_compliance(
            sample_text, 
            [ComplianceFramework.HIPAA, ComplianceFramework.PCI_DSS]
        )
        print(f"\nCompliance validation: {compliance_result}")
        
        # Example 2: Credential Management
        cred_manager = BrowserUseCredentialManager()
        
        # Store credentials
        cred_manager.store_credential("user_password", "password", "secret123", {"user": "john.doe"})
        cred_manager.store_credential("api_key", "api_key", "sk-1234567890abcdef", {"service": "openai"})
        
        # Retrieve credentials
        password = cred_manager.retrieve_credential("user_password")
        print(f"\nRetrieved password: {password}")
        
        # List credentials
        credentials = cred_manager.list_credentials()
        print(f"\nStored credentials: {len(credentials)}")
        for cred in credentials:
            print(f"  - {cred['credential_id']} ({cred['type']})")
        
        # Example 3: Form Data Classification
        classifier = BrowserUseDataClassifier([ComplianceFramework.HIPAA])
        
        form_data = {
            "first_name": "John",
            "last_name": "Doe",
            "email": "john.doe@email.com",
            "ssn": "123-45-6789",
            "credit_card": "4532-1234-5678-9012",
            "phone": "(555) 123-4567"
        }
        
        classification_result = classifier.classify_form_data(form_data)
        print(f"\nForm classification: {classification_result['overall_classification'].value}")
        print(f"PII detected: {classification_result['total_pii_detected']}")
        print(f"Compliance issues: {len(classification_result['compliance_issues'])}")
    
    # Run example
    example_usage()