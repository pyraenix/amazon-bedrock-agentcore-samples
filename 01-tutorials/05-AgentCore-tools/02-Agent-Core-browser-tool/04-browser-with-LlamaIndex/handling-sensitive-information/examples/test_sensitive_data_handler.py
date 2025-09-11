"""
Unit Tests for Sensitive Data Handler

This module provides comprehensive unit tests for the sensitive data detection,
classification, and sanitization components used in LlamaIndex AgentCore integration.

Test Categories:
- Sensitive data detection and pattern matching
- Data classification and sensitivity tagging
- Document sanitization and masking strategies
- Configuration and security features
- Integration with LlamaIndex documents

Requirements Addressed:
- 1.4: PII detection and masking during web content extraction
- 2.2: Document sanitization methods for sensitive content
- 2.4: Data classification and sensitivity tagging
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import re
from datetime import datetime

# Import the modules under test
from sensitive_data_handler import (
    SensitiveDataDetector,
    DocumentSanitizer,
    SensitiveDataClassifier,
    SensitiveDataMatch,
    SanitizationConfig,
    SensitivityLevel,
    DataType,
    MaskingStrategy,
    sanitize_documents,
    classify_documents,
    create_secure_sanitization_config
)

# LlamaIndex imports for testing
from llama_index.core.schema import Document


class TestSensitivityLevel(unittest.TestCase):
    """Test cases for SensitivityLevel enum."""
    
    def test_sensitivity_levels(self):
        """Test all sensitivity levels are defined."""
        levels = [level.value for level in SensitivityLevel]
        expected_levels = ["public", "internal", "confidential", "restricted"]
        
        self.assertEqual(set(levels), set(expected_levels))


class TestDataType(unittest.TestCase):
    """Test cases for DataType enum."""
    
    def test_data_types(self):
        """Test all data types are defined."""
        types = [dtype.value for dtype in DataType]
        expected_types = [
            "personally_identifiable_information",
            "financial_information",
            "health_information",
            "authentication_credentials",
            "business_confidential",
            "contact_information",
            "government_identification",
            "biometric_data"
        ]
        
        self.assertEqual(set(types), set(expected_types))


class TestMaskingStrategy(unittest.TestCase):
    """Test cases for MaskingStrategy enum."""
    
    def test_masking_strategies(self):
        """Test all masking strategies are defined."""
        strategies = [strategy.value for strategy in MaskingStrategy]
        expected_strategies = ["full_mask", "partial_mask", "hash_mask", "redact", "placeholder"]
        
        self.assertEqual(set(strategies), set(expected_strategies))


class TestSensitiveDataMatch(unittest.TestCase):
    """Test cases for SensitiveDataMatch."""
    
    def test_match_creation(self):
        """Test creating a sensitive data match."""
        match = SensitiveDataMatch(
            data_type=DataType.PII,
            sensitivity_level=SensitivityLevel.CONFIDENTIAL,
            start_pos=10,
            end_pos=20,
            original_text="test@example.com",
            confidence=0.95,
            pattern_name="email",
            context="Contact information"
        )
        
        self.assertEqual(match.data_type, DataType.PII)
        self.assertEqual(match.sensitivity_level, SensitivityLevel.CONFIDENTIAL)
        self.assertEqual(match.start_pos, 10)
        self.assertEqual(match.end_pos, 20)
        self.assertEqual(match.original_text, "test@example.com")
        self.assertEqual(match.confidence, 0.95)
        self.assertEqual(match.pattern_name, "email")
        self.assertEqual(match.context, "Contact information")
    
    def test_match_to_dict(self):
        """Test converting match to dictionary."""
        match = SensitiveDataMatch(
            data_type=DataType.FINANCIAL,
            sensitivity_level=SensitivityLevel.RESTRICTED,
            start_pos=0,
            end_pos=10,
            original_text="1234567890",
            confidence=0.8,
            pattern_name="credit_card"
        )
        
        match_dict = match.to_dict()
        
        self.assertEqual(match_dict['data_type'], "financial_information")
        self.assertEqual(match_dict['sensitivity_level'], "restricted")
        self.assertEqual(match_dict['start_pos'], 0)
        self.assertEqual(match_dict['end_pos'], 10)
        self.assertEqual(match_dict['confidence'], 0.8)
        self.assertEqual(match_dict['pattern_name'], "credit_card")
        self.assertIn('original_text_hash', match_dict)
        self.assertEqual(len(match_dict['original_text_hash']), 16)  # SHA256 hash truncated to 16 chars


class TestSanitizationConfig(unittest.TestCase):
    """Test cases for SanitizationConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = SanitizationConfig()
        
        self.assertEqual(config.default_masking_strategy, MaskingStrategy.PARTIAL_MASK)
        self.assertEqual(config.min_confidence_threshold, 0.7)
        self.assertTrue(config.preserve_document_structure)
        self.assertTrue(config.add_sanitization_metadata)
        self.assertTrue(config.audit_sensitive_operations)
        
        # Check that default masking strategies are set
        self.assertIn(DataType.PII, config.masking_strategies)
        self.assertIn(DataType.FINANCIAL, config.masking_strategies)
        self.assertIn(DataType.CREDENTIALS, config.masking_strategies)
    
    def test_custom_config(self):
        """Test custom configuration."""
        custom_strategies = {
            DataType.PII: MaskingStrategy.FULL_MASK,
            DataType.FINANCIAL: MaskingStrategy.REDACT
        }
        
        config = SanitizationConfig(
            default_masking_strategy=MaskingStrategy.HASH_MASK,
            masking_strategies=custom_strategies,
            min_confidence_threshold=0.9,
            preserve_document_structure=False,
            add_sanitization_metadata=False,
            audit_sensitive_operations=False
        )
        
        self.assertEqual(config.default_masking_strategy, MaskingStrategy.HASH_MASK)
        self.assertEqual(config.masking_strategies, custom_strategies)
        self.assertEqual(config.min_confidence_threshold, 0.9)
        self.assertFalse(config.preserve_document_structure)
        self.assertFalse(config.add_sanitization_metadata)
        self.assertFalse(config.audit_sensitive_operations)


class TestSensitiveDataDetector(unittest.TestCase):
    """Test cases for SensitiveDataDetector."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Initialize detector without NLP model for faster testing
        self.detector = SensitiveDataDetector(load_nlp_model=False)
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        self.assertIsNotNone(self.detector.patterns)
        self.assertGreater(len(self.detector.patterns), 0)
        self.assertIsNone(self.detector.nlp_model)  # No NLP model in test
    
    def test_email_detection(self):
        """Test email address detection."""
        text = "Contact us at support@example.com or admin@test.org"
        matches = self.detector.detect_sensitive_data(text)
        
        email_matches = [m for m in matches if m.pattern_name == 'email']
        self.assertEqual(len(email_matches), 2)
        
        # Check first email
        match1 = email_matches[0]
        self.assertEqual(match1.data_type, DataType.CONTACT)
        self.assertEqual(match1.sensitivity_level, SensitivityLevel.CONFIDENTIAL)
        self.assertEqual(match1.original_text, "support@example.com")
        self.assertEqual(match1.confidence, 0.95)
    
    def test_phone_detection(self):
        """Test phone number detection."""
        text = "Call me at (555) 123-4567 or 555.987.6543"
        matches = self.detector.detect_sensitive_data(text)
        
        phone_matches = [m for m in matches if m.pattern_name == 'phone']
        self.assertEqual(len(phone_matches), 2)
        
        # Check phone number properties
        for match in phone_matches:
            self.assertEqual(match.data_type, DataType.CONTACT)
            self.assertEqual(match.sensitivity_level, SensitivityLevel.CONFIDENTIAL)
            self.assertEqual(match.confidence, 0.8)
    
    def test_ssn_detection(self):
        """Test Social Security Number detection."""
        text = "My SSN is 123-45-6789 and another is 987654321"
        matches = self.detector.detect_sensitive_data(text)
        
        ssn_matches = [m for m in matches if m.pattern_name == 'ssn']
        self.assertEqual(len(ssn_matches), 2)
        
        # Check SSN properties
        for match in ssn_matches:
            self.assertEqual(match.data_type, DataType.GOVERNMENT_ID)
            self.assertEqual(match.sensitivity_level, SensitivityLevel.RESTRICTED)
            self.assertEqual(match.confidence, 0.9)
    
    def test_credit_card_detection(self):
        """Test credit card number detection."""
        text = "Credit card: 4532 1234 5678 9012 and 4532123456789012"
        matches = self.detector.detect_sensitive_data(text)
        
        cc_matches = [m for m in matches if m.pattern_name == 'credit_card']
        self.assertEqual(len(cc_matches), 2)
        
        # Check credit card properties
        for match in cc_matches:
            self.assertEqual(match.data_type, DataType.FINANCIAL)
            self.assertEqual(match.sensitivity_level, SensitivityLevel.RESTRICTED)
            self.assertEqual(match.confidence, 0.85)
    
    def test_api_key_detection(self):
        """Test API key detection."""
        text = 'API_KEY="sk-1234567890abcdef" and token: ghp_abcdefghijklmnopqrstuvwxyz123456'
        matches = self.detector.detect_sensitive_data(text)
        
        api_matches = [m for m in matches if m.pattern_name == 'api_key']
        self.assertGreater(len(api_matches), 0)
        
        # Check API key properties
        for match in api_matches:
            self.assertEqual(match.data_type, DataType.CREDENTIALS)
            self.assertEqual(match.sensitivity_level, SensitivityLevel.RESTRICTED)
    
    def test_aws_access_key_detection(self):
        """Test AWS access key detection."""
        text = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
        matches = self.detector.detect_sensitive_data(text)
        
        aws_matches = [m for m in matches if m.pattern_name == 'aws_access_key']
        self.assertEqual(len(aws_matches), 1)
        
        match = aws_matches[0]
        self.assertEqual(match.data_type, DataType.CREDENTIALS)
        self.assertEqual(match.sensitivity_level, SensitivityLevel.RESTRICTED)
        self.assertEqual(match.confidence, 0.95)
    
    def test_no_sensitive_data(self):
        """Test text with no sensitive data."""
        text = "This is just regular text with no sensitive information."
        matches = self.detector.detect_sensitive_data(text)
        
        self.assertEqual(len(matches), 0)
    
    def test_overlapping_matches_deduplication(self):
        """Test deduplication of overlapping matches."""
        # This would require creating overlapping patterns, which is complex
        # For now, test that deduplication method exists and works with empty list
        matches = self.detector._deduplicate_matches([])
        self.assertEqual(len(matches), 0)
    
    def test_context_parameter(self):
        """Test that context parameter is properly handled."""
        text = "Email: test@example.com"
        context = "User profile page"
        
        matches = self.detector.detect_sensitive_data(text, context)
        
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].context, context)


class TestDocumentSanitizer(unittest.TestCase):
    """Test cases for DocumentSanitizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = SanitizationConfig()
        self.sanitizer = DocumentSanitizer(self.config)
        
        # Sample document with sensitive data
        self.sample_document = Document(
            text="Contact John Doe at john.doe@example.com or (555) 123-4567. SSN: 123-45-6789",
            metadata={
                'source': 'https://example.com/profile',
                'session_id': 'test-session'
            }
        )
    
    def test_sanitizer_initialization(self):
        """Test sanitizer initialization."""
        self.assertIsInstance(self.sanitizer.config, SanitizationConfig)
        self.assertIsNotNone(self.sanitizer.detector)
    
    def test_document_sanitization(self):
        """Test basic document sanitization."""
        sanitized_doc = self.sanitizer.sanitize_document(self.sample_document)
        
        # Check that document is returned
        self.assertIsInstance(sanitized_doc, Document)
        
        # Check that sensitive data is masked
        self.assertNotIn("john.doe@example.com", sanitized_doc.text)
        self.assertNotIn("123-45-6789", sanitized_doc.text)
        
        # Check that some content remains
        self.assertIn("Contact", sanitized_doc.text)
        self.assertIn("John Doe", sanitized_doc.text)
    
    def test_sanitization_metadata(self):
        """Test that sanitization metadata is added."""
        sanitized_doc = self.sanitizer.sanitize_document(self.sample_document)
        
        # Check sanitization metadata
        self.assertIn('sanitization', sanitized_doc.metadata)
        sanitization_meta = sanitized_doc.metadata['sanitization']
        
        self.assertTrue(sanitization_meta['sanitized'])
        self.assertIn('timestamp', sanitization_meta)
        self.assertGreater(sanitization_meta['sensitive_data_detected'], 0)
        self.assertIn('data_types_found', sanitization_meta)
        
        # Check classification
        self.assertIn('classification', sanitized_doc.metadata)
        
        # Check sensitive matches
        self.assertIn('sensitive_matches', sanitized_doc.metadata)
        self.assertIsInstance(sanitized_doc.metadata['sensitive_matches'], list)
    
    def test_masking_strategies(self):
        """Test different masking strategies."""
        # Test full mask
        config = SanitizationConfig(default_masking_strategy=MaskingStrategy.FULL_MASK)
        sanitizer = DocumentSanitizer(config)
        
        test_text = "Email: test@example.com"
        result = sanitizer._apply_masking_strategy("test@example.com", MaskingStrategy.FULL_MASK, DataType.CONTACT)
        self.assertEqual(result, "*" * len("test@example.com"))
        
        # Test partial mask
        result = sanitizer._apply_masking_strategy("test@example.com", MaskingStrategy.PARTIAL_MASK, DataType.CONTACT)
        self.assertTrue(result.startswith("t"))
        self.assertTrue(result.endswith("m"))
        self.assertIn("*", result)
        
        # Test hash mask
        result = sanitizer._apply_masking_strategy("test@example.com", MaskingStrategy.HASH_MASK, DataType.CONTACT)
        self.assertTrue(result.startswith("[HASH:"))
        self.assertTrue(result.endswith("]"))
        
        # Test redact
        result = sanitizer._apply_masking_strategy("test@example.com", MaskingStrategy.REDACT, DataType.CONTACT)
        self.assertEqual(result, "")
        
        # Test placeholder
        result = sanitizer._apply_masking_strategy("test@example.com", MaskingStrategy.PLACEHOLDER, DataType.CONTACT)
        self.assertEqual(result, "[CONTACT_INFORMATION]")
    
    def test_confidence_threshold_filtering(self):
        """Test that low-confidence matches are filtered out."""
        # Create config with high confidence threshold
        config = SanitizationConfig(min_confidence_threshold=0.95)
        sanitizer = DocumentSanitizer(config)
        
        # Document with mixed confidence matches
        doc = Document(
            text="Email: test@example.com and some numbers: 12345678",  # Email high confidence, numbers low
            metadata={'source': 'test'}
        )
        
        sanitized_doc = sanitizer.sanitize_document(doc)
        
        # Email should be masked (high confidence)
        self.assertNotIn("test@example.com", sanitized_doc.text)
        
        # Numbers might remain (depending on pattern confidence)
        # This test verifies the filtering mechanism exists
        self.assertIn('sanitization', sanitized_doc.metadata)
    
    def test_no_sensitive_data_document(self):
        """Test sanitizing document with no sensitive data."""
        clean_doc = Document(
            text="This is a clean document with no sensitive information.",
            metadata={'source': 'test'}
        )
        
        sanitized_doc = self.sanitizer.sanitize_document(clean_doc)
        
        # Text should remain unchanged
        self.assertEqual(sanitized_doc.text, clean_doc.text)
        
        # Should still have sanitization metadata
        self.assertIn('sanitization', sanitized_doc.metadata)
        self.assertEqual(sanitized_doc.metadata['sanitization']['sensitive_data_detected'], 0)
        self.assertEqual(sanitized_doc.metadata['classification'], SensitivityLevel.PUBLIC.value)


class TestSensitiveDataClassifier(unittest.TestCase):
    """Test cases for SensitiveDataClassifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = SensitiveDataClassifier()
    
    def test_classifier_initialization(self):
        """Test classifier initialization."""
        self.assertIsNotNone(self.classifier.detector)
    
    def test_classify_sensitive_document(self):
        """Test classifying document with sensitive data."""
        doc = Document(
            text="SSN: 123-45-6789, Email: test@example.com, Credit Card: 4532 1234 5678 9012",
            metadata={'source': 'test'}
        )
        
        classification = self.classifier.classify_document(doc)
        
        # Should be classified as restricted due to SSN and credit card
        self.assertEqual(classification['sensitivity_level'], SensitivityLevel.RESTRICTED.value)
        self.assertGreater(classification['sensitive_data_count'], 0)
        self.assertTrue(classification['requires_special_handling'])
        self.assertIn('government_identification', classification['data_types'])
        self.assertIn('financial_information', classification['data_types'])
        self.assertIn('contact_information', classification['data_types'])
    
    def test_classify_public_document(self):
        """Test classifying document with no sensitive data."""
        doc = Document(
            text="This is public information about our company services.",
            metadata={'source': 'test'}
        )
        
        classification = self.classifier.classify_document(doc)
        
        self.assertEqual(classification['sensitivity_level'], SensitivityLevel.PUBLIC.value)
        self.assertEqual(classification['sensitive_data_count'], 0)
        self.assertFalse(classification['requires_special_handling'])
        self.assertEqual(classification['data_types'], [])
        self.assertEqual(classification['classification_confidence'], 1.0)
    
    def test_special_handling_requirements(self):
        """Test detection of documents requiring special handling."""
        # Document with health information
        health_doc = Document(
            text="Patient MRN: 1234567 has diabetes.",
            metadata={'source': 'test'}
        )
        
        classification = self.classifier.classify_document(health_doc)
        self.assertTrue(classification['requires_special_handling'])
        
        # Document with only contact info (no special handling)
        contact_doc = Document(
            text="Contact us at info@company.com",
            metadata={'source': 'test'}
        )
        
        classification = self.classifier.classify_document(contact_doc)
        self.assertFalse(classification['requires_special_handling'])


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_sanitize_documents(self):
        """Test sanitizing multiple documents."""
        docs = [
            Document(text="Email: test1@example.com", metadata={'source': 'doc1'}),
            Document(text="Phone: (555) 123-4567", metadata={'source': 'doc2'}),
            Document(text="Clean document", metadata={'source': 'doc3'})
        ]
        
        sanitized_docs = sanitize_documents(docs)
        
        self.assertEqual(len(sanitized_docs), 3)
        
        # Check that all documents have sanitization metadata
        for doc in sanitized_docs:
            self.assertIn('sanitization', doc.metadata)
    
    def test_classify_documents(self):
        """Test classifying multiple documents."""
        docs = [
            Document(text="SSN: 123-45-6789", metadata={'source': 'doc1'}),
            Document(text="Email: test@example.com", metadata={'source': 'doc2'}),
            Document(text="Public information", metadata={'source': 'doc3'})
        ]
        
        classifications = classify_documents(docs)
        
        self.assertEqual(len(classifications), 3)
        
        # Check classification structure
        for classification in classifications:
            self.assertIn('sensitivity_level', classification)
            self.assertIn('data_types', classification)
            self.assertIn('sensitive_data_count', classification)
    
    def test_create_secure_sanitization_config(self):
        """Test creating secure sanitization configurations."""
        # Standard mode
        standard_config = create_secure_sanitization_config(strict_mode=False)
        self.assertEqual(standard_config.min_confidence_threshold, 0.7)
        self.assertEqual(standard_config.masking_strategies[DataType.PII], MaskingStrategy.PARTIAL_MASK)
        
        # Strict mode
        strict_config = create_secure_sanitization_config(strict_mode=True)
        self.assertEqual(strict_config.min_confidence_threshold, 0.5)
        self.assertEqual(strict_config.masking_strategies[DataType.PII], MaskingStrategy.REDACT)
        self.assertEqual(strict_config.masking_strategies[DataType.FINANCIAL], MaskingStrategy.REDACT)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration test scenarios."""
    
    def test_complete_sanitization_workflow(self):
        """Test complete sanitization workflow."""
        # Create document with multiple types of sensitive data
        doc = Document(
            text="""
            Patient Information:
            Name: John Doe
            Email: john.doe@hospital.com
            Phone: (555) 123-4567
            SSN: 123-45-6789
            Medical Record: MRN: 7654321
            Credit Card: 4532 1234 5678 9012
            """,
            metadata={
                'source': 'https://hospital.com/patient/123',
                'session_id': 'medical-session-456'
            }
        )
        
        # Classify document
        classifier = SensitiveDataClassifier()
        classification = classifier.classify_document(doc)
        
        # Should be restricted due to health and financial data
        self.assertEqual(classification['sensitivity_level'], SensitivityLevel.RESTRICTED.value)
        self.assertTrue(classification['requires_special_handling'])
        
        # Sanitize document
        config = create_secure_sanitization_config(strict_mode=True)
        sanitizer = DocumentSanitizer(config)
        sanitized_doc = sanitizer.sanitize_document(doc)
        
        # Verify sensitive data is removed/masked
        sensitive_patterns = [
            "john.doe@hospital.com",
            "123-45-6789",
            "4532 1234 5678 9012",
            "MRN: 7654321"
        ]
        
        for pattern in sensitive_patterns:
            self.assertNotIn(pattern, sanitized_doc.text)
        
        # Verify metadata is comprehensive
        self.assertIn('sanitization', sanitized_doc.metadata)
        self.assertIn('classification', sanitized_doc.metadata)
        self.assertIn('sensitive_matches', sanitized_doc.metadata)
        
        # Verify classification matches expectation
        self.assertEqual(sanitized_doc.metadata['classification'], SensitivityLevel.RESTRICTED.value)
    
    def test_batch_processing_with_mixed_sensitivity(self):
        """Test batch processing of documents with mixed sensitivity levels."""
        docs = [
            Document(text="Public company information", metadata={'source': 'public'}),
            Document(text="Internal email: staff@company.com", metadata={'source': 'internal'}),
            Document(text="Customer SSN: 123-45-6789", metadata={'source': 'confidential'}),
            Document(text="Credit card: 4532 1234 5678 9012", metadata={'source': 'restricted'})
        ]
        
        # Classify all documents
        classifications = classify_documents(docs)
        
        # Verify different sensitivity levels
        sensitivity_levels = [c['sensitivity_level'] for c in classifications]
        self.assertIn(SensitivityLevel.PUBLIC.value, sensitivity_levels)
        self.assertIn(SensitivityLevel.CONFIDENTIAL.value, sensitivity_levels)
        self.assertIn(SensitivityLevel.RESTRICTED.value, sensitivity_levels)
        
        # Sanitize all documents
        sanitized_docs = sanitize_documents(docs)
        
        # Verify all documents are processed
        self.assertEqual(len(sanitized_docs), 4)
        
        # Verify each document has appropriate metadata
        for doc in sanitized_docs:
            self.assertIn('sanitization', doc.metadata)
            self.assertIn('classification', doc.metadata)


if __name__ == '__main__':
    # Configure test logging
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests
    
    # Run the tests
    unittest.main(verbosity=2)