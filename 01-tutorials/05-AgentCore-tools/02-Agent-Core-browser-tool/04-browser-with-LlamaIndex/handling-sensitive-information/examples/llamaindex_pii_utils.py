#!/usr/bin/env python3
"""
LlamaIndex PII Utilities for Sensitive Data Identification

This module provides comprehensive utilities for detecting, classifying, and handling
Personally Identifiable Information (PII) in LlamaIndex documents. Includes GDPR compliance,
data anonymization, and secure data handling patterns.

Key Features:
- Comprehensive PII detection patterns
- GDPR compliance utilities
- Data anonymization and pseudonymization
- Sensitive data classification
- Audit logging for data processing
- Integration with LlamaIndex document processing

Requirements: 1.2, 2.5, 4.2, 5.4
"""

import os
import logging
import re
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from uuid import uuid4

from llama_index.core import Document
from llama_index.core.schema import BaseNode

# Configure logging
logger = logging.getLogger(__name__)


class PIIType(Enum):
    """Types of Personally Identifiable Information"""
    # Personal identifiers
    FULL_NAME = "full_name"
    FIRST_NAME = "first_name"
    LAST_NAME = "last_name"
    EMAIL = "email_address"
    PHONE = "phone_number"
    SSN = "social_security_number"
    PASSPORT = "passport_number"
    DRIVERS_LICENSE = "drivers_license"
    
    # Financial information
    CREDIT_CARD = "credit_card_number"
    BANK_ACCOUNT = "bank_account_number"
    ROUTING_NUMBER = "routing_number"
    TAX_ID = "tax_identification_number"
    
    # Location information
    ADDRESS = "physical_address"
    ZIP_CODE = "zip_code"
    IP_ADDRESS = "ip_address"
    GPS_COORDINATES = "gps_coordinates"
    
    # Dates and identifiers
    DATE_OF_BIRTH = "date_of_birth"
    MEDICAL_RECORD = "medical_record_number"
    EMPLOYEE_ID = "employee_id"
    CUSTOMER_ID = "customer_id"
    
    # Biometric and health
    BIOMETRIC = "biometric_data"
    HEALTH_INFO = "health_information"
    
    # Digital identifiers
    USERNAME = "username"
    PASSWORD = "password"
    API_KEY = "api_key"
    TOKEN = "authentication_token"


class DataSensitivityLevel(Enum):
    """Data sensitivity classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class GDPRCategory(Enum):
    """GDPR data categories"""
    PERSONAL_DATA = "personal_data"
    SENSITIVE_PERSONAL_DATA = "sensitive_personal_data"
    CRIMINAL_DATA = "criminal_data"
    BIOMETRIC_DATA = "biometric_data"
    HEALTH_DATA = "health_data"
    GENETIC_DATA = "genetic_data"


@dataclass
class PIIMatch:
    """Represents a detected PII match"""
    pii_type: PIIType
    value: str
    start_pos: int
    end_pos: int
    confidence: float
    context: str
    gdpr_category: Optional[GDPRCategory] = None
    sensitivity_level: DataSensitivityLevel = DataSensitivityLevel.CONFIDENTIAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "pii_type": self.pii_type.value,
            "value": self.value,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": self.confidence,
            "context": self.context,
            "gdpr_category": self.gdpr_category.value if self.gdpr_category else None,
            "sensitivity_level": self.sensitivity_level.value
        }


@dataclass
class PIIDetectionResult:
    """Results of PII detection analysis"""
    document_id: str
    total_pii_found: int
    pii_matches: List[PIIMatch]
    pii_types_detected: Set[PIIType]
    gdpr_categories: Set[GDPRCategory]
    overall_sensitivity: DataSensitivityLevel
    detection_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "document_id": self.document_id,
            "total_pii_found": self.total_pii_found,
            "pii_matches": [match.to_dict() for match in self.pii_matches],
            "pii_types_detected": [pii_type.value for pii_type in self.pii_types_detected],
            "gdpr_categories": [category.value for category in self.gdpr_categories],
            "overall_sensitivity": self.overall_sensitivity.value,
            "detection_timestamp": self.detection_timestamp.isoformat()
        }


class PIIDetector:
    """Comprehensive PII detector for LlamaIndex documents"""
    
    def __init__(self, custom_patterns: Optional[Dict[PIIType, str]] = None):
        self.patterns = self._initialize_patterns()
        if custom_patterns:
            self.patterns.update(custom_patterns)
        
        # GDPR category mappings
        self.gdpr_mappings = {
            PIIType.FULL_NAME: GDPRCategory.PERSONAL_DATA,
            PIIType.EMAIL: GDPRCategory.PERSONAL_DATA,
            PIIType.PHONE: GDPRCategory.PERSONAL_DATA,
            PIIType.SSN: GDPRCategory.SENSITIVE_PERSONAL_DATA,
            PIIType.HEALTH_INFO: GDPRCategory.HEALTH_DATA,
            PIIType.BIOMETRIC: GDPRCategory.BIOMETRIC_DATA,
            PIIType.CREDIT_CARD: GDPRCategory.PERSONAL_DATA,
            PIIType.ADDRESS: GDPRCategory.PERSONAL_DATA,
            PIIType.DATE_OF_BIRTH: GDPRCategory.PERSONAL_DATA,
        }
        
        logger.info("PII detector initialized")
    
    def _initialize_patterns(self) -> Dict[PIIType, str]:
        """Initialize PII detection patterns"""
        return {
            # Names
            PIIType.FULL_NAME: r'\b[A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b',
            PIIType.FIRST_NAME: r'\b[A-Z][a-z]{2,}\b',
            PIIType.LAST_NAME: r'\b[A-Z][a-z]{2,}\b',
            
            # Contact information
            PIIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            PIIType.PHONE: r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            
            # Government IDs
            PIIType.SSN: r'\b\d{3}-\d{2}-\d{4}\b',
            PIIType.PASSPORT: r'\b[A-Z]{1,2}\d{6,9}\b',
            PIIType.DRIVERS_LICENSE: r'\b[A-Z]{1,2}\d{6,8}\b',
            
            # Financial
            PIIType.CREDIT_CARD: r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            PIIType.BANK_ACCOUNT: r'\b\d{8,17}\b',
            PIIType.ROUTING_NUMBER: r'\b\d{9}\b',
            PIIType.TAX_ID: r'\b\d{2}-\d{7}\b',
            
            # Location
            PIIType.ZIP_CODE: r'\b\d{5}(?:-\d{4})?\b',
            PIIType.IP_ADDRESS: r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            PIIType.GPS_COORDINATES: r'\b-?\d{1,3}\.\d+,\s*-?\d{1,3}\.\d+\b',
            
            # Dates
            PIIType.DATE_OF_BIRTH: r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            
            # Medical and employee IDs
            PIIType.MEDICAL_RECORD: r'\bMRN[:\s]*\d{6,10}\b',
            PIIType.EMPLOYEE_ID: r'\bEMP[:\s]*\d{4,8}\b',
            PIIType.CUSTOMER_ID: r'\bCUST[:\s]*\d{4,10}\b',
            
            # Digital identifiers
            PIIType.USERNAME: r'\b[a-zA-Z0-9._-]{3,20}\b',
            PIIType.API_KEY: r'\b[A-Za-z0-9]{20,}\b',
            PIIType.TOKEN: r'\b[A-Za-z0-9+/]{20,}={0,2}\b',
        }
    
    def detect_pii_in_text(self, text: str, document_id: str = None) -> PIIDetectionResult:
        """Detect PII in text content"""
        if document_id is None:
            document_id = f"doc_{uuid4().hex[:8]}"
        
        pii_matches = []
        pii_types_detected = set()
        gdpr_categories = set()
        
        for pii_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                # Extract context around the match
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end]
                
                # Calculate confidence based on pattern specificity
                confidence = self._calculate_confidence(pii_type, match.group(), context)
                
                # Skip low confidence matches for generic patterns
                if confidence < 0.5 and pii_type in [PIIType.FIRST_NAME, PIIType.LAST_NAME, PIIType.USERNAME]:
                    continue
                
                pii_match = PIIMatch(
                    pii_type=pii_type,
                    value=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=confidence,
                    context=context,
                    gdpr_category=self.gdpr_mappings.get(pii_type),
                    sensitivity_level=self._determine_sensitivity_level(pii_type)
                )
                
                pii_matches.append(pii_match)
                pii_types_detected.add(pii_type)
                
                if pii_match.gdpr_category:
                    gdpr_categories.add(pii_match.gdpr_category)
        
        # Determine overall sensitivity
        overall_sensitivity = self._determine_overall_sensitivity(pii_types_detected)
        
        result = PIIDetectionResult(
            document_id=document_id,
            total_pii_found=len(pii_matches),
            pii_matches=pii_matches,
            pii_types_detected=pii_types_detected,
            gdpr_categories=gdpr_categories,
            overall_sensitivity=overall_sensitivity
        )
        
        logger.info(f"Detected {len(pii_matches)} PII instances in document {document_id}")
        return result
    
    def detect_pii_in_document(self, document: Document) -> PIIDetectionResult:
        """Detect PII in LlamaIndex document"""
        doc_id = document.doc_id or f"doc_{uuid4().hex[:8]}"
        return self.detect_pii_in_text(document.text, doc_id)
    
    def detect_pii_in_node(self, node: BaseNode) -> PIIDetectionResult:
        """Detect PII in LlamaIndex node"""
        node_id = node.node_id or f"node_{uuid4().hex[:8]}"
        return self.detect_pii_in_text(node.text, node_id)
    
    def _calculate_confidence(self, pii_type: PIIType, value: str, context: str) -> float:
        """Calculate confidence score for PII match"""
        base_confidence = 0.7
        
        # Adjust confidence based on PII type
        high_confidence_types = {
            PIIType.EMAIL, PIIType.SSN, PIIType.CREDIT_CARD, 
            PIIType.PHONE, PIIType.IP_ADDRESS
        }
        
        if pii_type in high_confidence_types:
            base_confidence = 0.9
        
        # Adjust based on context
        context_lower = context.lower()
        
        # Increase confidence for contextual clues
        context_clues = {
            PIIType.EMAIL: ['email', 'e-mail', 'contact', '@'],
            PIIType.PHONE: ['phone', 'tel', 'call', 'number'],
            PIIType.SSN: ['ssn', 'social security', 'social'],
            PIIType.ADDRESS: ['address', 'street', 'avenue', 'road'],
            PIIType.CREDIT_CARD: ['card', 'credit', 'payment', 'visa', 'mastercard'],
        }
        
        if pii_type in context_clues:
            for clue in context_clues[pii_type]:
                if clue in context_lower:
                    base_confidence = min(1.0, base_confidence + 0.1)
                    break
        
        return base_confidence
    
    def _determine_sensitivity_level(self, pii_type: PIIType) -> DataSensitivityLevel:
        """Determine sensitivity level for PII type"""
        sensitivity_mapping = {
            PIIType.SSN: DataSensitivityLevel.RESTRICTED,
            PIIType.CREDIT_CARD: DataSensitivityLevel.RESTRICTED,
            PIIType.PASSPORT: DataSensitivityLevel.RESTRICTED,
            PIIType.HEALTH_INFO: DataSensitivityLevel.RESTRICTED,
            PIIType.BIOMETRIC: DataSensitivityLevel.RESTRICTED,
            PIIType.MEDICAL_RECORD: DataSensitivityLevel.RESTRICTED,
            
            PIIType.FULL_NAME: DataSensitivityLevel.CONFIDENTIAL,
            PIIType.EMAIL: DataSensitivityLevel.CONFIDENTIAL,
            PIIType.PHONE: DataSensitivityLevel.CONFIDENTIAL,
            PIIType.ADDRESS: DataSensitivityLevel.CONFIDENTIAL,
            PIIType.DATE_OF_BIRTH: DataSensitivityLevel.CONFIDENTIAL,
            
            PIIType.ZIP_CODE: DataSensitivityLevel.INTERNAL,
            PIIType.FIRST_NAME: DataSensitivityLevel.INTERNAL,
            PIIType.LAST_NAME: DataSensitivityLevel.INTERNAL,
        }
        
        return sensitivity_mapping.get(pii_type, DataSensitivityLevel.CONFIDENTIAL)
    
    def _determine_overall_sensitivity(self, pii_types: Set[PIIType]) -> DataSensitivityLevel:
        """Determine overall sensitivity level for document"""
        if not pii_types:
            return DataSensitivityLevel.PUBLIC
        
        max_sensitivity = DataSensitivityLevel.PUBLIC
        
        for pii_type in pii_types:
            sensitivity = self._determine_sensitivity_level(pii_type)
            if sensitivity.value > max_sensitivity.value:
                max_sensitivity = sensitivity
        
        return max_sensitivity


class PIIAnonymizer:
    """Anonymize and pseudonymize PII in documents"""
    
    def __init__(self, anonymization_key: Optional[str] = None):
        self.anonymization_key = anonymization_key or os.getenv("PII_ANONYMIZATION_KEY", "default_key")
        self.anonymization_cache: Dict[str, str] = {}
        
        logger.info("PII anonymizer initialized")
    
    def anonymize_text(self, text: str, pii_result: PIIDetectionResult, 
                      anonymization_method: str = "mask") -> str:
        """Anonymize PII in text"""
        anonymized_text = text
        
        # Sort matches by position (reverse order to maintain positions)
        sorted_matches = sorted(pii_result.pii_matches, key=lambda x: x.start_pos, reverse=True)
        
        for match in sorted_matches:
            if anonymization_method == "mask":
                replacement = self._mask_value(match)
            elif anonymization_method == "pseudonymize":
                replacement = self._pseudonymize_value(match)
            elif anonymization_method == "remove":
                replacement = ""
            else:
                replacement = f"[REDACTED_{match.pii_type.value.upper()}]"
            
            anonymized_text = (
                anonymized_text[:match.start_pos] + 
                replacement + 
                anonymized_text[match.end_pos:]
            )
        
        return anonymized_text
    
    def anonymize_document(self, document: Document, pii_result: PIIDetectionResult,
                          anonymization_method: str = "mask") -> Document:
        """Anonymize PII in LlamaIndex document"""
        anonymized_text = self.anonymize_text(document.text, pii_result, anonymization_method)
        
        # Create new document with anonymized content
        anonymized_doc = Document(
            text=anonymized_text,
            doc_id=document.doc_id,
            metadata={
                **document.metadata,
                "pii_anonymized": True,
                "anonymization_method": anonymization_method,
                "pii_types_found": [pii_type.value for pii_type in pii_result.pii_types_detected],
                "anonymization_timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return anonymized_doc
    
    def _mask_value(self, match: PIIMatch) -> str:
        """Mask PII value"""
        value = match.value
        
        if match.pii_type == PIIType.EMAIL:
            # Mask email: j***@example.com
            parts = value.split('@')
            if len(parts) == 2:
                username = parts[0]
                domain = parts[1]
                masked_username = username[0] + '*' * (len(username) - 1) if len(username) > 1 else '*'
                return f"{masked_username}@{domain}"
        
        elif match.pii_type == PIIType.PHONE:
            # Mask phone: ***-***-1234
            return "***-***-" + value[-4:] if len(value) >= 4 else "***-***-****"
        
        elif match.pii_type == PIIType.CREDIT_CARD:
            # Mask credit card: ****-****-****-1234
            clean_value = re.sub(r'[\s-]', '', value)
            return "****-****-****-" + clean_value[-4:] if len(clean_value) >= 4 else "****-****-****-****"
        
        elif match.pii_type == PIIType.SSN:
            # Mask SSN: ***-**-1234
            return "***-**-" + value[-4:] if len(value) >= 4 else "***-**-****"
        
        elif match.pii_type in [PIIType.FULL_NAME, PIIType.FIRST_NAME, PIIType.LAST_NAME]:
            # Mask names: J*** D***
            words = value.split()
            masked_words = []
            for word in words:
                if len(word) > 1:
                    masked_words.append(word[0] + '*' * (len(word) - 1))
                else:
                    masked_words.append('*')
            return ' '.join(masked_words)
        
        else:
            # Generic masking
            if len(value) <= 4:
                return '*' * len(value)
            else:
                return value[:2] + '*' * (len(value) - 4) + value[-2:]
    
    def _pseudonymize_value(self, match: PIIMatch) -> str:
        """Pseudonymize PII value using consistent hashing"""
        cache_key = f"{match.pii_type.value}:{match.value}"
        
        if cache_key in self.anonymization_cache:
            return self.anonymization_cache[cache_key]
        
        # Create deterministic pseudonym
        hash_input = f"{self.anonymization_key}:{match.value}:{match.pii_type.value}"
        hash_value = hashlib.sha256(hash_input.encode()).hexdigest()
        
        if match.pii_type == PIIType.EMAIL:
            pseudonym = f"user{hash_value[:8]}@example.com"
        elif match.pii_type == PIIType.PHONE:
            pseudonym = f"555-{hash_value[:3]}-{hash_value[3:7]}"
        elif match.pii_type in [PIIType.FULL_NAME, PIIType.FIRST_NAME, PIIType.LAST_NAME]:
            pseudonym = f"Person{hash_value[:6]}"
        elif match.pii_type == PIIType.SSN:
            pseudonym = f"***-**-{hash_value[:4]}"
        elif match.pii_type == PIIType.CREDIT_CARD:
            pseudonym = f"****-****-****-{hash_value[:4]}"
        else:
            pseudonym = f"ID{hash_value[:8]}"
        
        self.anonymization_cache[cache_key] = pseudonym
        return pseudonym


class PIIAuditLogger:
    """Audit logger for PII processing activities"""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file or "pii_audit.log"
        self.audit_logger = logging.getLogger("pii_audit")
        
        # Configure audit logger
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.audit_logger.addHandler(handler)
        self.audit_logger.setLevel(logging.INFO)
    
    def log_pii_detection(self, result: PIIDetectionResult, source: str = "unknown"):
        """Log PII detection event"""
        audit_entry = {
            "event": "pii_detection",
            "timestamp": datetime.utcnow().isoformat(),
            "document_id": result.document_id,
            "source": source,
            "pii_count": result.total_pii_found,
            "pii_types": [pii_type.value for pii_type in result.pii_types_detected],
            "gdpr_categories": [category.value for category in result.gdpr_categories],
            "sensitivity_level": result.overall_sensitivity.value
        }
        
        self.audit_logger.info(f"PII_DETECTION: {json.dumps(audit_entry)}")
    
    def log_pii_anonymization(self, document_id: str, method: str, pii_types: List[PIIType]):
        """Log PII anonymization event"""
        audit_entry = {
            "event": "pii_anonymization",
            "timestamp": datetime.utcnow().isoformat(),
            "document_id": document_id,
            "anonymization_method": method,
            "pii_types_anonymized": [pii_type.value for pii_type in pii_types]
        }
        
        self.audit_logger.info(f"PII_ANONYMIZATION: {json.dumps(audit_entry)}")
    
    def log_gdpr_compliance_check(self, document_id: str, compliant: bool, issues: List[str]):
        """Log GDPR compliance check"""
        audit_entry = {
            "event": "gdpr_compliance_check",
            "timestamp": datetime.utcnow().isoformat(),
            "document_id": document_id,
            "compliant": compliant,
            "issues": issues
        }
        
        self.audit_logger.info(f"GDPR_COMPLIANCE: {json.dumps(audit_entry)}")


class GDPRComplianceChecker:
    """GDPR compliance checker for PII processing"""
    
    def __init__(self):
        self.required_protections = {
            GDPRCategory.SENSITIVE_PERSONAL_DATA: ["encryption", "anonymization", "access_control"],
            GDPRCategory.HEALTH_DATA: ["encryption", "anonymization", "access_control", "audit_logging"],
            GDPRCategory.BIOMETRIC_DATA: ["encryption", "anonymization", "access_control", "audit_logging"],
            GDPRCategory.PERSONAL_DATA: ["anonymization", "access_control"]
        }
    
    def check_compliance(self, pii_result: PIIDetectionResult, 
                        applied_protections: List[str]) -> Tuple[bool, List[str]]:
        """Check GDPR compliance for detected PII"""
        issues = []
        
        for gdpr_category in pii_result.gdpr_categories:
            required = self.required_protections.get(gdpr_category, [])
            
            for protection in required:
                if protection not in applied_protections:
                    issues.append(f"Missing {protection} for {gdpr_category.value}")
        
        compliant = len(issues) == 0
        return compliant, issues


# Utility functions for common PII operations

def detect_and_anonymize_document(document: Document, anonymization_method: str = "mask") -> Tuple[Document, PIIDetectionResult]:
    """Detect PII and anonymize document in one step"""
    detector = PIIDetector()
    anonymizer = PIIAnonymizer()
    
    # Detect PII
    pii_result = detector.detect_pii_in_document(document)
    
    # Anonymize if PII found
    if pii_result.total_pii_found > 0:
        anonymized_doc = anonymizer.anonymize_document(document, pii_result, anonymization_method)
        return anonymized_doc, pii_result
    
    return document, pii_result


def batch_process_documents(documents: List[Document], anonymization_method: str = "mask") -> List[Tuple[Document, PIIDetectionResult]]:
    """Process multiple documents for PII detection and anonymization"""
    detector = PIIDetector()
    anonymizer = PIIAnonymizer()
    audit_logger = PIIAuditLogger()
    
    results = []
    
    for document in documents:
        # Detect PII
        pii_result = detector.detect_pii_in_document(document)
        
        # Log detection
        audit_logger.log_pii_detection(pii_result, "batch_processing")
        
        # Anonymize if needed
        if pii_result.total_pii_found > 0:
            anonymized_doc = anonymizer.anonymize_document(document, pii_result, anonymization_method)
            audit_logger.log_pii_anonymization(document.doc_id, anonymization_method, list(pii_result.pii_types_detected))
            results.append((anonymized_doc, pii_result))
        else:
            results.append((document, pii_result))
    
    return results


def create_pii_summary_report(pii_results: List[PIIDetectionResult]) -> Dict[str, Any]:
    """Create summary report of PII detection results"""
    total_documents = len(pii_results)
    documents_with_pii = sum(1 for result in pii_results if result.total_pii_found > 0)
    total_pii_instances = sum(result.total_pii_found for result in pii_results)
    
    # Count PII types
    pii_type_counts = {}
    for result in pii_results:
        for pii_type in result.pii_types_detected:
            pii_type_counts[pii_type.value] = pii_type_counts.get(pii_type.value, 0) + 1
    
    # Count GDPR categories
    gdpr_category_counts = {}
    for result in pii_results:
        for category in result.gdpr_categories:
            gdpr_category_counts[category.value] = gdpr_category_counts.get(category.value, 0) + 1
    
    # Sensitivity distribution
    sensitivity_counts = {}
    for result in pii_results:
        sensitivity = result.overall_sensitivity.value
        sensitivity_counts[sensitivity] = sensitivity_counts.get(sensitivity, 0) + 1
    
    return {
        "summary": {
            "total_documents": total_documents,
            "documents_with_pii": documents_with_pii,
            "pii_detection_rate": documents_with_pii / total_documents if total_documents > 0 else 0,
            "total_pii_instances": total_pii_instances,
            "average_pii_per_document": total_pii_instances / total_documents if total_documents > 0 else 0
        },
        "pii_types": pii_type_counts,
        "gdpr_categories": gdpr_category_counts,
        "sensitivity_distribution": sensitivity_counts,
        "report_timestamp": datetime.utcnow().isoformat()
    }


# Example usage
if __name__ == "__main__":
    # Example document with PII
    sample_text = """
    John Doe's email is john.doe@example.com and his phone number is 555-123-4567.
    His SSN is 123-45-6789 and he lives at 123 Main Street, Anytown, NY 12345.
    Credit card: 4532-1234-5678-9012
    """
    
    document = Document(text=sample_text, doc_id="sample_doc")
    
    # Detect and anonymize
    anonymized_doc, pii_result = detect_and_anonymize_document(document, "mask")
    
    print("Original text:")
    print(document.text)
    print("\nAnonymized text:")
    print(anonymized_doc.text)
    print(f"\nPII detected: {pii_result.total_pii_found} instances")
    print(f"PII types: {[pii_type.value for pii_type in pii_result.pii_types_detected]}")