"""
Sensitive Data Handler for LlamaIndex AgentCore Integration

This module provides comprehensive sensitive data detection, classification, and
sanitization capabilities for LlamaIndex documents extracted via AgentCore Browser Tool.
It implements PII detection, data masking, and document sanitization to ensure
sensitive information is properly protected during indexing and retrieval.

Key Features:
- PII detection using pattern matching and NLP techniques
- Data classification and sensitivity tagging
- Document sanitization with configurable masking strategies
- Metadata enrichment for sensitive content tracking
- Compliance-ready audit logging

Requirements Addressed:
- 1.4: PII detection and masking during web content extraction
- 2.2: Document sanitization methods for sensitive content
- 2.4: Data classification and sensitivity tagging
"""

import re
import logging
import hashlib
import json
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# LlamaIndex imports
from llama_index.core.schema import Document

# NLP and text processing
try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    displacy = None
    SPACY_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SensitivityLevel(Enum):
    """Classification levels for sensitive data."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    
    @property
    def priority(self):
        """Get priority level for comparison."""
        priorities = {
            "public": 1,
            "internal": 2,
            "confidential": 3,
            "restricted": 4
        }
        return priorities[self.value]
    
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.priority < other.priority
        return NotImplemented
    
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.priority <= other.priority
        return NotImplemented
    
    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.priority > other.priority
        return NotImplemented
    
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.priority >= other.priority
        return NotImplemented


class DataType(Enum):
    """Types of sensitive data that can be detected."""
    PII = "personally_identifiable_information"
    FINANCIAL = "financial_information"
    HEALTH = "health_information"
    CREDENTIALS = "authentication_credentials"
    BUSINESS = "business_confidential"
    CONTACT = "contact_information"
    GOVERNMENT_ID = "government_identification"
    BIOMETRIC = "biometric_data"


class MaskingStrategy(Enum):
    """Strategies for masking sensitive data."""
    FULL_MASK = "full_mask"  # Replace with ***
    PARTIAL_MASK = "partial_mask"  # Show first/last chars
    HASH_MASK = "hash_mask"  # Replace with hash
    REDACT = "redact"  # Remove completely
    PLACEHOLDER = "placeholder"  # Replace with [TYPE] placeholder


@dataclass
class SensitiveDataMatch:
    """Represents a detected sensitive data match."""
    data_type: DataType
    sensitivity_level: SensitivityLevel
    start_pos: int
    end_pos: int
    original_text: str
    confidence: float
    pattern_name: str
    context: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'data_type': self.data_type.value,
            'sensitivity_level': self.sensitivity_level.value,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'original_text_hash': hashlib.sha256(self.original_text.encode()).hexdigest()[:16],
            'confidence': self.confidence,
            'pattern_name': self.pattern_name,
            'context': self.context[:50] + "..." if len(self.context) > 50 else self.context
        }


@dataclass
class SanitizationConfig:
    """Configuration for data sanitization."""
    default_masking_strategy: MaskingStrategy = MaskingStrategy.PARTIAL_MASK
    masking_strategies: Dict[DataType, MaskingStrategy] = field(default_factory=dict)
    min_confidence_threshold: float = 0.7
    preserve_document_structure: bool = True
    add_sanitization_metadata: bool = True
    audit_sensitive_operations: bool = True
    
    def __post_init__(self):
        """Set default masking strategies for different data types."""
        if not self.masking_strategies:
            self.masking_strategies = {
                DataType.PII: MaskingStrategy.PARTIAL_MASK,
                DataType.FINANCIAL: MaskingStrategy.FULL_MASK,
                DataType.HEALTH: MaskingStrategy.REDACT,
                DataType.CREDENTIALS: MaskingStrategy.FULL_MASK,
                DataType.BUSINESS: MaskingStrategy.PARTIAL_MASK,
                DataType.CONTACT: MaskingStrategy.PARTIAL_MASK,
                DataType.GOVERNMENT_ID: MaskingStrategy.HASH_MASK,
                DataType.BIOMETRIC: MaskingStrategy.REDACT
            }


class SensitiveDataDetector:
    """
    Advanced sensitive data detector using pattern matching and NLP.
    
    Detects various types of sensitive information including PII, financial data,
    health information, and credentials using a combination of regex patterns
    and named entity recognition.
    """
    
    def __init__(self, load_nlp_model: bool = True):
        """
        Initialize the sensitive data detector.
        
        Args:
            load_nlp_model: Whether to load spaCy NLP model for enhanced detection
        """
        self.nlp_model = None
        
        if load_nlp_model and SPACY_AVAILABLE:
            try:
                # Try to load spaCy model
                self.nlp_model = spacy.load("en_core_web_sm")
                logger.info("✅ spaCy NLP model loaded for enhanced PII detection")
            except OSError:
                logger.warning("⚠️ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                logger.info("Falling back to pattern-based detection only")
        elif load_nlp_model and not SPACY_AVAILABLE:
            logger.warning("⚠️ spaCy not available. Install with: pip install spacy")
            logger.info("Falling back to pattern-based detection only")
        
        # Initialize detection patterns
        self._init_detection_patterns()
    
    def _init_detection_patterns(self):
        """Initialize regex patterns for sensitive data detection."""
        
        self.patterns = {
            # Social Security Numbers
            'ssn': {
                'pattern': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
                'data_type': DataType.GOVERNMENT_ID,
                'sensitivity': SensitivityLevel.RESTRICTED,
                'confidence': 0.9
            },
            
            # Credit Card Numbers
            'credit_card': {
                'pattern': re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
                'data_type': DataType.FINANCIAL,
                'sensitivity': SensitivityLevel.RESTRICTED,
                'confidence': 0.85
            },
            
            # Email Addresses
            'email': {
                'pattern': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
                'data_type': DataType.CONTACT,
                'sensitivity': SensitivityLevel.CONFIDENTIAL,
                'confidence': 0.95
            },
            
            # Phone Numbers
            'phone': {
                'pattern': re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
                'data_type': DataType.CONTACT,
                'sensitivity': SensitivityLevel.CONFIDENTIAL,
                'confidence': 0.8
            },
            
            # Driver's License (US format)
            'drivers_license': {
                'pattern': re.compile(r'\b[A-Z]{1,2}\d{6,8}\b'),
                'data_type': DataType.GOVERNMENT_ID,
                'sensitivity': SensitivityLevel.RESTRICTED,
                'confidence': 0.7
            },
            
            # Bank Account Numbers
            'bank_account': {
                'pattern': re.compile(r'\b\d{8,17}\b'),
                'data_type': DataType.FINANCIAL,
                'sensitivity': SensitivityLevel.RESTRICTED,
                'confidence': 0.6  # Lower confidence due to potential false positives
            },
            
            # IP Addresses
            'ip_address': {
                'pattern': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
                'data_type': DataType.BUSINESS,
                'sensitivity': SensitivityLevel.INTERNAL,
                'confidence': 0.9
            },
            
            # Medical Record Numbers
            'medical_record': {
                'pattern': re.compile(r'\bMRN:?\s*\d{6,10}\b', re.IGNORECASE),
                'data_type': DataType.HEALTH,
                'sensitivity': SensitivityLevel.RESTRICTED,
                'confidence': 0.9
            },
            
            # Passport Numbers
            'passport': {
                'pattern': re.compile(r'\b[A-Z]{1,2}\d{6,9}\b'),
                'data_type': DataType.GOVERNMENT_ID,
                'sensitivity': SensitivityLevel.RESTRICTED,
                'confidence': 0.75
            },
            
            # Date of Birth patterns
            'date_of_birth': {
                'pattern': re.compile(r'\b(?:DOB|Date of Birth):?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', re.IGNORECASE),
                'data_type': DataType.PII,
                'sensitivity': SensitivityLevel.CONFIDENTIAL,
                'confidence': 0.85
            },
            
            # API Keys and Tokens
            'api_key': {
                'pattern': re.compile(r'(?:api[_-]?key|token|secret)["\s:=]+([A-Za-z0-9+/_-]{16,})', re.IGNORECASE),
                'data_type': DataType.CREDENTIALS,
                'sensitivity': SensitivityLevel.RESTRICTED,
                'confidence': 0.8
            },
            
            # AWS Access Keys
            'aws_access_key': {
                'pattern': re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
                'data_type': DataType.CREDENTIALS,
                'sensitivity': SensitivityLevel.RESTRICTED,
                'confidence': 0.95
            }
        }
        
        logger.info(f"Initialized {len(self.patterns)} sensitive data detection patterns")
    
    def detect_sensitive_data(self, text: str, context: str = "") -> List[SensitiveDataMatch]:
        """
        Detect sensitive data in text using pattern matching and NLP.
        
        Args:
            text: Text to analyze
            context: Additional context for better detection
            
        Returns:
            List of detected sensitive data matches
        """
        
        matches = []
        
        # Pattern-based detection
        pattern_matches = self._detect_with_patterns(text, context)
        matches.extend(pattern_matches)
        
        # NLP-based detection (if model is available)
        if self.nlp_model:
            nlp_matches = self._detect_with_nlp(text, context)
            matches.extend(nlp_matches)
        
        # Remove duplicates and overlapping matches
        matches = self._deduplicate_matches(matches)
        
        logger.info(f"Detected {len(matches)} sensitive data matches in text")
        return matches
    
    def _detect_with_patterns(self, text: str, context: str = "") -> List[SensitiveDataMatch]:
        """Detect sensitive data using regex patterns."""
        
        matches = []
        
        for pattern_name, pattern_info in self.patterns.items():
            pattern = pattern_info['pattern']
            
            for match in pattern.finditer(text):
                sensitive_match = SensitiveDataMatch(
                    data_type=pattern_info['data_type'],
                    sensitivity_level=pattern_info['sensitivity'],
                    start_pos=match.start(),
                    end_pos=match.end(),
                    original_text=match.group(),
                    confidence=pattern_info['confidence'],
                    pattern_name=pattern_name,
                    context=context
                )
                matches.append(sensitive_match)
        
        return matches
    
    def _detect_with_nlp(self, text: str, context: str = "") -> List[SensitiveDataMatch]:
        """Detect sensitive data using NLP named entity recognition."""
        
        matches = []
        
        try:
            doc = self.nlp_model(text)
            
            for ent in doc.ents:
                # Map spaCy entity types to our data types
                data_type, sensitivity = self._map_entity_to_data_type(ent.label_)
                
                if data_type:
                    sensitive_match = SensitiveDataMatch(
                        data_type=data_type,
                        sensitivity_level=sensitivity,
                        start_pos=ent.start_char,
                        end_pos=ent.end_char,
                        original_text=ent.text,
                        confidence=0.8,  # NLP confidence
                        pattern_name=f"nlp_{ent.label_.lower()}",
                        context=context
                    )
                    matches.append(sensitive_match)
        
        except Exception as e:
            logger.warning(f"NLP detection failed: {str(e)}")
        
        return matches
    
    def _map_entity_to_data_type(self, entity_label: str) -> Tuple[Optional[DataType], SensitivityLevel]:
        """Map spaCy entity labels to our data types."""
        
        entity_mapping = {
            'PERSON': (DataType.PII, SensitivityLevel.CONFIDENTIAL),
            'ORG': (DataType.BUSINESS, SensitivityLevel.INTERNAL),
            'GPE': (DataType.PII, SensitivityLevel.INTERNAL),  # Geopolitical entity
            'MONEY': (DataType.FINANCIAL, SensitivityLevel.CONFIDENTIAL),
            'DATE': (DataType.PII, SensitivityLevel.INTERNAL),
            'TIME': (DataType.PII, SensitivityLevel.INTERNAL),
            'CARDINAL': (None, SensitivityLevel.PUBLIC),  # Numbers - usually not sensitive
            'ORDINAL': (None, SensitivityLevel.PUBLIC)
        }
        
        return entity_mapping.get(entity_label, (None, SensitivityLevel.PUBLIC))
    
    def _deduplicate_matches(self, matches: List[SensitiveDataMatch]) -> List[SensitiveDataMatch]:
        """Remove duplicate and overlapping matches."""
        
        if not matches:
            return matches
        
        # Sort by start position
        matches.sort(key=lambda x: x.start_pos)
        
        deduplicated = []
        
        for match in matches:
            # Check if this match overlaps with any existing match
            overlaps = False
            
            for existing in deduplicated:
                if (match.start_pos < existing.end_pos and 
                    match.end_pos > existing.start_pos):
                    # Overlapping matches - keep the one with higher confidence
                    if match.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        deduplicated.append(match)
                    overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(match)
        
        return deduplicated


class DocumentSanitizer:
    """
    Document sanitizer that applies masking strategies to sensitive data.
    
    Provides configurable sanitization of LlamaIndex documents while preserving
    document structure and adding comprehensive metadata about sanitization operations.
    """
    
    def __init__(self, config: Optional[SanitizationConfig] = None):
        """
        Initialize the document sanitizer.
        
        Args:
            config: Sanitization configuration
        """
        self.config = config or SanitizationConfig()
        self.detector = SensitiveDataDetector()
        
        logger.info("DocumentSanitizer initialized with configuration:")
        logger.info(f"  Default masking strategy: {self.config.default_masking_strategy.value}")
        logger.info(f"  Confidence threshold: {self.config.min_confidence_threshold}")
    
    def sanitize_document(self, document: Document) -> Document:
        """
        Sanitize a LlamaIndex document by detecting and masking sensitive data.
        
        Args:
            document: LlamaIndex document to sanitize
            
        Returns:
            Sanitized document with masked sensitive data and enriched metadata
        """
        
        logger.info(f"Sanitizing document from source: {document.metadata.get('source', 'unknown')}")
        
        # Detect sensitive data
        sensitive_matches = self.detector.detect_sensitive_data(
            document.text,
            context=document.metadata.get('source', '')
        )
        
        # Filter matches by confidence threshold
        high_confidence_matches = [
            match for match in sensitive_matches 
            if match.confidence >= self.config.min_confidence_threshold
        ]
        
        logger.info(f"Found {len(high_confidence_matches)} high-confidence sensitive data matches")
        
        # Apply sanitization
        sanitized_text = self._apply_sanitization(document.text, high_confidence_matches)
        
        # Create sanitized document
        sanitized_document = Document(
            text=sanitized_text,
            metadata=document.metadata.copy()
        )
        
        # Add sanitization metadata
        if self.config.add_sanitization_metadata:
            self._add_sanitization_metadata(sanitized_document, high_confidence_matches)
        
        # Audit logging
        if self.config.audit_sensitive_operations:
            self._audit_sanitization(document, high_confidence_matches)
        
        return sanitized_document
    
    def _apply_sanitization(self, text: str, matches: List[SensitiveDataMatch]) -> str:
        """Apply masking strategies to sensitive data matches."""
        
        if not matches:
            return text
        
        # Sort matches by position (reverse order to maintain positions)
        matches.sort(key=lambda x: x.start_pos, reverse=True)
        
        sanitized_text = text
        
        for match in matches:
            # Get masking strategy for this data type
            strategy = self.config.masking_strategies.get(
                match.data_type, 
                self.config.default_masking_strategy
            )
            
            # Apply masking
            masked_value = self._apply_masking_strategy(match.original_text, strategy, match.data_type)
            
            # Replace in text
            sanitized_text = (
                sanitized_text[:match.start_pos] + 
                masked_value + 
                sanitized_text[match.end_pos:]
            )
        
        return sanitized_text
    
    def _apply_masking_strategy(self, original_text: str, strategy: MaskingStrategy, data_type: DataType) -> str:
        """Apply a specific masking strategy to text."""
        
        if strategy == MaskingStrategy.FULL_MASK:
            return "*" * len(original_text)
        
        elif strategy == MaskingStrategy.PARTIAL_MASK:
            if len(original_text) <= 4:
                return "*" * len(original_text)
            else:
                # Show first and last character
                return original_text[0] + "*" * (len(original_text) - 2) + original_text[-1]
        
        elif strategy == MaskingStrategy.HASH_MASK:
            # Create a short hash
            hash_value = hashlib.sha256(original_text.encode()).hexdigest()[:8]
            return f"[HASH:{hash_value}]"
        
        elif strategy == MaskingStrategy.REDACT:
            return ""  # Remove completely
        
        elif strategy == MaskingStrategy.PLACEHOLDER:
            return f"[{data_type.value.upper()}]"
        
        else:
            # Default to partial mask
            return self._apply_masking_strategy(original_text, MaskingStrategy.PARTIAL_MASK, data_type)
    
    def _add_sanitization_metadata(self, document: Document, matches: List[SensitiveDataMatch]) -> None:
        """Add sanitization metadata to the document."""
        
        # Sanitization summary
        document.metadata['sanitization'] = {
            'sanitized': True,
            'timestamp': datetime.now().isoformat(),
            'sensitive_data_detected': len(matches),
            'data_types_found': list(set(match.data_type.value for match in matches)),
            'sensitivity_levels': list(set(match.sensitivity_level.value for match in matches)),
            'sanitization_config': {
                'default_strategy': self.config.default_masking_strategy.value,
                'confidence_threshold': self.config.min_confidence_threshold
            }
        }
        
        # Detailed match information (without original text)
        document.metadata['sensitive_matches'] = [
            match.to_dict() for match in matches
        ]
        
        # Classification based on highest sensitivity level
        if matches:
            highest_sensitivity = max(match.sensitivity_level for match in matches)
            document.metadata['classification'] = highest_sensitivity.value
        else:
            document.metadata['classification'] = SensitivityLevel.PUBLIC.value
    
    def _audit_sanitization(self, original_document: Document, matches: List[SensitiveDataMatch]) -> None:
        """Audit the sanitization operation."""
        
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': 'document_sanitization',
            'source': original_document.metadata.get('source', 'unknown'),
            'session_id': original_document.metadata.get('session_id', 'unknown'),
            'sensitive_data_count': len(matches),
            'data_types': list(set(match.data_type.value for match in matches)),
            'sensitivity_levels': list(set(match.sensitivity_level.value for match in matches))
        }
        
        # Log audit entry (in production, this would go to a secure audit log)
        logger.info(f"AUDIT: Document sanitization completed - {json.dumps(audit_entry)}")


class SensitiveDataClassifier:
    """
    Classifier for determining document sensitivity levels and data types.
    
    Provides document-level classification based on detected sensitive data
    and configurable classification rules.
    """
    
    def __init__(self):
        """Initialize the classifier."""
        self.detector = SensitiveDataDetector()
    
    def classify_document(self, document: Document) -> Dict[str, Any]:
        """
        Classify a document's sensitivity level and data types.
        
        Args:
            document: LlamaIndex document to classify
            
        Returns:
            Classification results including sensitivity level and data types
        """
        
        # Detect sensitive data
        matches = self.detector.detect_sensitive_data(document.text)
        
        if not matches:
            return {
                'sensitivity_level': SensitivityLevel.PUBLIC.value,
                'data_types': [],
                'sensitive_data_count': 0,
                'classification_confidence': 1.0,
                'requires_special_handling': False
            }
        
        # Determine overall sensitivity level
        sensitivity_levels = [match.sensitivity_level for match in matches]
        overall_sensitivity = max(sensitivity_levels)
        
        # Get unique data types
        data_types = list(set(match.data_type.value for match in matches))
        
        # Calculate classification confidence
        avg_confidence = sum(match.confidence for match in matches) / len(matches)
        
        # Determine if special handling is required
        requires_special_handling = any(
            match.data_type in [DataType.HEALTH, DataType.FINANCIAL, DataType.CREDENTIALS]
            for match in matches
        )
        
        return {
            'sensitivity_level': overall_sensitivity.value,
            'data_types': data_types,
            'sensitive_data_count': len(matches),
            'classification_confidence': avg_confidence,
            'requires_special_handling': requires_special_handling,
            'detailed_matches': [match.to_dict() for match in matches]
        }


# Utility functions for common use cases

def sanitize_documents(
    documents: List[Document], 
    config: Optional[SanitizationConfig] = None
) -> List[Document]:
    """
    Sanitize a list of LlamaIndex documents.
    
    Args:
        documents: List of documents to sanitize
        config: Sanitization configuration
        
    Returns:
        List of sanitized documents
    """
    
    sanitizer = DocumentSanitizer(config)
    sanitized_docs = []
    
    for doc in documents:
        try:
            sanitized_doc = sanitizer.sanitize_document(doc)
            sanitized_docs.append(sanitized_doc)
        except Exception as e:
            logger.error(f"Failed to sanitize document: {str(e)}")
            # Add error metadata to original document
            doc.metadata['sanitization_error'] = str(e)
            sanitized_docs.append(doc)
    
    logger.info(f"Sanitized {len(sanitized_docs)} documents")
    return sanitized_docs


def classify_documents(documents: List[Document]) -> List[Dict[str, Any]]:
    """
    Classify a list of documents for sensitivity.
    
    Args:
        documents: List of documents to classify
        
    Returns:
        List of classification results
    """
    
    classifier = SensitiveDataClassifier()
    classifications = []
    
    for doc in documents:
        try:
            classification = classifier.classify_document(doc)
            classifications.append(classification)
        except Exception as e:
            logger.error(f"Failed to classify document: {str(e)}")
            classifications.append({
                'sensitivity_level': SensitivityLevel.PUBLIC.value,
                'error': str(e)
            })
    
    return classifications


def create_secure_sanitization_config(
    strict_mode: bool = False,
    preserve_structure: bool = True
) -> SanitizationConfig:
    """
    Create a sanitization configuration for secure environments.
    
    Args:
        strict_mode: Use strict sanitization (higher security)
        preserve_structure: Preserve document structure
        
    Returns:
        Configured SanitizationConfig
    """
    
    if strict_mode:
        # Strict mode - more aggressive sanitization
        masking_strategies = {
            DataType.PII: MaskingStrategy.REDACT,
            DataType.FINANCIAL: MaskingStrategy.REDACT,
            DataType.HEALTH: MaskingStrategy.REDACT,
            DataType.CREDENTIALS: MaskingStrategy.REDACT,
            DataType.BUSINESS: MaskingStrategy.HASH_MASK,
            DataType.CONTACT: MaskingStrategy.FULL_MASK,
            DataType.GOVERNMENT_ID: MaskingStrategy.REDACT,
            DataType.BIOMETRIC: MaskingStrategy.REDACT
        }
        confidence_threshold = 0.5  # Lower threshold for strict mode
    else:
        # Standard mode - balanced sanitization
        masking_strategies = {
            DataType.PII: MaskingStrategy.PARTIAL_MASK,
            DataType.FINANCIAL: MaskingStrategy.FULL_MASK,
            DataType.HEALTH: MaskingStrategy.HASH_MASK,
            DataType.CREDENTIALS: MaskingStrategy.FULL_MASK,
            DataType.BUSINESS: MaskingStrategy.PARTIAL_MASK,
            DataType.CONTACT: MaskingStrategy.PARTIAL_MASK,
            DataType.GOVERNMENT_ID: MaskingStrategy.HASH_MASK,
            DataType.BIOMETRIC: MaskingStrategy.REDACT
        }
        confidence_threshold = 0.7
    
    return SanitizationConfig(
        default_masking_strategy=MaskingStrategy.PARTIAL_MASK,
        masking_strategies=masking_strategies,
        min_confidence_threshold=confidence_threshold,
        preserve_document_structure=preserve_structure,
        add_sanitization_metadata=True,
        audit_sensitive_operations=True
    )


# Example usage
if __name__ == "__main__":
    print("Sensitive Data Handler for LlamaIndex")
    print("=" * 50)
    
    # Example document with sensitive data
    sample_text = """
    John Doe's personal information:
    Email: john.doe@example.com
    Phone: (555) 123-4567
    SSN: 123-45-6789
    Credit Card: 4532 1234 5678 9012
    Medical Record: MRN: 1234567
    """
    
    sample_document = Document(
        text=sample_text,
        metadata={
            'source': 'https://example.com/profile',
            'session_id': 'test-session-123'
        }
    )
    
    print("\n1. Sensitive Data Detection Example")
    detector = SensitiveDataDetector()
    matches = detector.detect_sensitive_data(sample_text)
    
    print(f"Detected {len(matches)} sensitive data matches:")
    for match in matches:
        print(f"  - {match.data_type.value}: {match.pattern_name} "
              f"(confidence: {match.confidence:.2f})")
    
    print("\n2. Document Classification Example")
    classifier = SensitiveDataClassifier()
    classification = classifier.classify_document(sample_document)
    
    print(f"Document classification:")
    print(f"  - Sensitivity Level: {classification['sensitivity_level']}")
    print(f"  - Data Types: {classification['data_types']}")
    print(f"  - Requires Special Handling: {classification['requires_special_handling']}")
    
    print("\n3. Document Sanitization Example")
    
    # Standard sanitization
    config = create_secure_sanitization_config(strict_mode=False)
    sanitizer = DocumentSanitizer(config)
    sanitized_doc = sanitizer.sanitize_document(sample_document)
    
    print("Original text (first 100 chars):")
    print(f"  {sample_text[:100]}...")
    
    print("\nSanitized text (first 100 chars):")
    print(f"  {sanitized_doc.text[:100]}...")
    
    print(f"\nSanitization metadata:")
    print(f"  - Sensitive data detected: {sanitized_doc.metadata['sanitization']['sensitive_data_detected']}")
    print(f"  - Data types found: {sanitized_doc.metadata['sanitization']['data_types_found']}")
    print(f"  - Classification: {sanitized_doc.metadata['classification']}")
    
    print("\n4. Strict Mode Sanitization Example")
    strict_config = create_secure_sanitization_config(strict_mode=True)
    strict_sanitizer = DocumentSanitizer(strict_config)
    strict_sanitized_doc = strict_sanitizer.sanitize_document(sample_document)
    
    print("Strict mode sanitized text (first 100 chars):")
    print(f"  {strict_sanitized_doc.text[:100]}...")
    
    print("\n✅ Sensitive data handling examples completed")


class SensitiveDataHandler:
    """
    Main handler class that combines detection, classification, and sanitization.
    
    This class provides a unified interface for handling sensitive data in LlamaIndex
    documents extracted via AgentCore Browser Tool.
    """
    
    def __init__(self, config: Optional[SanitizationConfig] = None):
        """
        Initialize the sensitive data handler.
        
        Args:
            config: Sanitization configuration (uses default if None)
        """
        self.config = config or create_secure_sanitization_config()
        self.detector = SensitiveDataDetector()
        self.sanitizer = DocumentSanitizer(self.config)
        self.classifier = SensitiveDataClassifier()
        
    def process_document(self, document: Document) -> Document:
        """
        Process a document through the complete sensitive data handling pipeline.
        
        Args:
            document: LlamaIndex document to process
            
        Returns:
            Processed document with sensitive data handled
        """
        # Classify the document
        classification = self.classifier.classify_document(document)
        
        # Sanitize if needed
        if classification['requires_special_handling']:
            sanitized_doc = self.sanitizer.sanitize_document(document)
            # Add classification to metadata
            sanitized_doc.metadata['classification'] = classification
            return sanitized_doc
        else:
            # Add classification metadata even if no sanitization needed
            document.metadata['classification'] = classification
            return document
            
    def detect_pii_in_document(self, document: Document) -> List[SensitiveDataMatch]:
        """
        Detect PII in a document.
        
        Args:
            document: Document to analyze
            
        Returns:
            List of detected PII matches
        """
        return self.detector.detect_sensitive_data(document.text)
        
    def sanitize_document(self, document: Document, compliance_mode: str = "standard") -> Document:
        """
        Sanitize a document based on compliance requirements.
        
        Args:
            document: Document to sanitize
            compliance_mode: Compliance mode ("standard", "HIPAA", "PCI_DSS", etc.)
            
        Returns:
            Sanitized document
        """
        if compliance_mode == "HIPAA":
            config = create_secure_sanitization_config(strict_mode=True)
            sanitizer = DocumentSanitizer(config)
        elif compliance_mode == "PCI_DSS":
            config = create_secure_sanitization_config(strict_mode=True)
            sanitizer = DocumentSanitizer(config)
        else:
            sanitizer = self.sanitizer
            
        return sanitizer.sanitize_document(document)