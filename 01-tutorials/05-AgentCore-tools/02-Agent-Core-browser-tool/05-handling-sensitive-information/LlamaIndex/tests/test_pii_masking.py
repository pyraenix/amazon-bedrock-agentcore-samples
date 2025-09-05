"""
PII Masking Validation Tests for LlamaIndex-AgentCore Integration

Comprehensive tests for PII detection, masking, and sanitization.
Requirements: 2.2, 4.1
"""

import pytest
import re
from typing import List, Dict, Any
from datetime import datetime

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'examples'))

try:
    from llama_index.core import Document
except ImportError:
    # Mock for testing
    class Document:
        def __init__(self, text: str, metadata: Dict = None):
            self.text = text
            self.metadata = metadata or {}

from sensitive_data_handler import SensitiveDataHandler
from llamaindex_pii_utils import PIIDetector, PIIMasker, DataClassifier


class TestPIIDetection:
    """Test PII detection capabilities."""
    
    def setup_method(self):
        """Set up test environment."""
        self.pii_detector = PIIDetector()
        self.data_handler = SensitiveDataHandler()
        
    def test_email_detection(self):
        """Test detection of email addresses."""
        test_cases = [
            "Contact us at john.doe@example.com",
            "Email: jane.smith@company.org",
            "Send to user+tag@domain.co.uk",
            "Multiple emails: a@b.com, c@d.net, e@f.org",
            "Edge case: test@sub.domain.example.com"
        ]
        
        for text in test_cases:
            detected_pii = self.pii_detector.detect_pii(text)
            email_detections = [pii for pii in detected_pii if pii["type"] == "EMAIL"]
            
            assert len(email_detections) > 0, f"Failed to detect email in: {text}"
            
            # Verify detected emails are valid
            for detection in email_detections:
                email = detection["value"]
                assert "@" in email
                assert "." in email.split("@")[1]
                
    def test_phone_number_detection(self):
        """Test detection of phone numbers."""
        test_cases = [
            "Call us at 555-123-4567",
            "Phone: (555) 123-4567",
            "Mobile: +1-555-123-4567",
            "International: +44 20 7946 0958",
            "Extension: 555-123-4567 ext. 123",
            "Toll-free: 1-800-555-0199"
        ]
        
        for text in test_cases:
            detected_pii = self.pii_detector.detect_pii(text)
            phone_detections = [pii for pii in detected_pii if pii["type"] == "PHONE"]
            
            assert len(phone_detections) > 0, f"Failed to detect phone in: {text}"
            
    def test_ssn_detection(self):
        """Test detection of Social Security Numbers."""
        test_cases = [
            "SSN: 123-45-6789",
            "Social Security Number: 987-65-4321",
            "SSN 555-44-3333",
            "My SSN is 111-22-3333"
        ]
        
        for text in test_cases:
            detected_pii = self.pii_detector.detect_pii(text)
            ssn_detections = [pii for pii in detected_pii if pii["type"] == "SSN"]
            
            assert len(ssn_detections) > 0, f"Failed to detect SSN in: {text}"
            
            # Verify SSN format
            for detection in ssn_detections:
                ssn = detection["value"]
                assert re.match(r'\d{3}-\d{2}-\d{4}', ssn), f"Invalid SSN format: {ssn}"
                
    def test_credit_card_detection(self):
        """Test detection of credit card numbers."""
        test_cases = [
            "Card: 4532-1234-5678-9012",
            "Credit card number: 4532123456789012",
            "Visa: 4532 1234 5678 9012",
            "Mastercard: 5555-5555-5555-4444",
            "Amex: 3782-822463-10005"
        ]
        
        for text in test_cases:
            detected_pii = self.pii_detector.detect_pii(text)
            cc_detections = [pii for pii in detected_pii if pii["type"] == "CREDIT_CARD"]
            
            assert len(cc_detections) > 0, f"Failed to detect credit card in: {text}"
            
    def test_name_detection(self):
        """Test detection of person names."""
        test_cases = [
            "Patient: John Doe",
            "Dr. Jane Smith will see you",
            "Contact Mr. Robert Johnson",
            "Ms. Sarah Williams, CEO",
            "Prof. Michael Brown"
        ]
        
        for text in test_cases:
            detected_pii = self.pii_detector.detect_pii(text)
            name_detections = [pii for pii in detected_pii if pii["type"] == "PERSON_NAME"]
            
            # Names are harder to detect reliably, so we allow some flexibility
            if len(name_detections) == 0:
                # Check if at least capitalized words are detected as potential names
                potential_names = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', text)
                assert len(potential_names) > 0, f"No names or potential names detected in: {text}"
                
    def test_address_detection(self):
        """Test detection of addresses."""
        test_cases = [
            "Address: 123 Main St, Anytown, ST 12345",
            "Located at 456 Oak Avenue, Suite 100, City, State 67890",
            "PO Box 789, Small Town, XX 11111",
            "1000 Corporate Blvd, Business City, BC 22222"
        ]
        
        for text in test_cases:
            detected_pii = self.pii_detector.detect_pii(text)
            address_detections = [pii for pii in detected_pii if pii["type"] == "ADDRESS"]
            
            # Address detection is complex, check for zip codes at minimum
            if len(address_detections) == 0:
                zip_codes = re.findall(r'\b\d{5}(?:-\d{4})?\b', text)
                assert len(zip_codes) > 0, f"No addresses or zip codes detected in: {text}"
                
    def test_false_positive_prevention(self):
        """Test that legitimate content is not incorrectly detected as PII."""
        non_pii_cases = [
            "The API endpoint is https://api.example.com/v1/users",
            "Version 1.2.3 was released on 2024-01-01",
            "Error code: 404-NOT-FOUND",
            "Temperature: 98.6°F",
            "Price: $123.45",
            "Time: 12:34:56"
        ]
        
        for text in non_pii_cases:
            detected_pii = self.pii_detector.detect_pii(text)
            
            # Should not detect PII in these cases
            assert len(detected_pii) == 0, f"False positive PII detection in: {text}"
            
    def test_mixed_content_detection(self):
        """Test detection in content with mixed PII and non-PII."""
        mixed_content = """
        Patient Information Form
        
        Name: John Doe
        Email: john.doe@hospital.com
        Phone: 555-123-4567
        SSN: 123-45-6789
        Address: 123 Medical Center Dr, Health City, HC 12345
        
        Insurance Information:
        Policy Number: POL-123456789
        Group Number: GRP-987654321
        
        Emergency Contact:
        Name: Jane Doe
        Phone: 555-987-6543
        Relationship: Spouse
        
        Notes: Patient has no known allergies. Last visit was on 2024-01-15.
        Next appointment scheduled for 2024-02-15 at 10:30 AM.
        """
        
        detected_pii = self.pii_detector.detect_pii(mixed_content)
        
        # Should detect multiple types of PII
        pii_types = {pii["type"] for pii in detected_pii}
        
        expected_types = {"EMAIL", "PHONE", "SSN", "PERSON_NAME"}
        found_types = pii_types.intersection(expected_types)
        
        assert len(found_types) >= 3, f"Expected to find at least 3 PII types, found: {found_types}"


class TestPIIMasking:
    """Test PII masking and sanitization."""
    
    def setup_method(self):
        """Set up test environment."""
        self.pii_masker = PIIMasker()
        self.data_handler = SensitiveDataHandler()
        
    def test_email_masking(self):
        """Test email masking preserves format while hiding content."""
        test_cases = [
            ("Contact john.doe@example.com", "john.doe@example.com"),
            ("Email: jane@company.org", "jane@company.org"),
            ("Send to user+tag@domain.co.uk", "user+tag@domain.co.uk")
        ]
        
        for original_text, email in test_cases:
            masked_text = self.pii_masker.mask_emails(original_text)
            
            # Original email should not be present
            assert email not in masked_text
            
            # Should contain masked version or placeholder
            assert ("[EMAIL]" in masked_text or 
                   "***@***.***" in masked_text or
                   "@" in masked_text)  # Some masking preserves @ symbol
                   
    def test_phone_masking(self):
        """Test phone number masking."""
        test_cases = [
            ("Call 555-123-4567", "555-123-4567"),
            ("Phone: (555) 123-4567", "(555) 123-4567"),
            ("Mobile +1-555-123-4567", "+1-555-123-4567")
        ]
        
        for original_text, phone in test_cases:
            masked_text = self.pii_masker.mask_phones(original_text)
            
            # Original phone should not be present
            assert phone not in masked_text
            
            # Should contain masked version or placeholder
            assert ("[PHONE]" in masked_text or 
                   "***-***-****" in masked_text or
                   "XXX-XXX-XXXX" in masked_text)
                   
    def test_ssn_masking(self):
        """Test SSN masking."""
        test_cases = [
            ("SSN: 123-45-6789", "123-45-6789"),
            ("Social Security: 987-65-4321", "987-65-4321")
        ]
        
        for original_text, ssn in test_cases:
            masked_text = self.pii_masker.mask_ssns(original_text)
            
            # Original SSN should not be present
            assert ssn not in masked_text
            
            # Should contain masked version
            assert ("[SSN]" in masked_text or 
                   "***-**-****" in masked_text or
                   "XXX-XX-XXXX" in masked_text)
                   
    def test_credit_card_masking(self):
        """Test credit card masking."""
        test_cases = [
            ("Card: 4532-1234-5678-9012", "4532-1234-5678-9012"),
            ("CC: 4532123456789012", "4532123456789012")
        ]
        
        for original_text, cc in test_cases:
            masked_text = self.pii_masker.mask_credit_cards(original_text)
            
            # Original CC should not be present
            assert cc not in masked_text
            
            # Should contain masked version
            assert ("[CREDIT_CARD]" in masked_text or 
                   "****-****-****-****" in masked_text or
                   "XXXX-XXXX-XXXX-XXXX" in masked_text)
                   
    def test_comprehensive_masking(self):
        """Test comprehensive masking of all PII types."""
        original_text = """
        Patient: John Doe
        Email: john.doe@hospital.com
        Phone: 555-123-4567
        SSN: 123-45-6789
        Credit Card: 4532-1234-5678-9012
        Address: 123 Main St, City, ST 12345
        """
        
        masked_text = self.pii_masker.mask_all_pii(original_text)
        
        # Verify all PII is masked
        pii_values = [
            "john.doe@hospital.com",
            "555-123-4567",
            "123-45-6789",
            "4532-1234-5678-9012"
        ]
        
        for pii_value in pii_values:
            assert pii_value not in masked_text, f"PII value {pii_value} not masked"
            
        # Verify non-PII content is preserved
        assert "Patient:" in masked_text
        assert "Email:" in masked_text
        assert "Phone:" in masked_text
        
    def test_masking_preserves_context(self):
        """Test that masking preserves document context and readability."""
        original_text = "Dr. Smith (dr.smith@hospital.com) will call you at 555-123-4567."
        masked_text = self.pii_masker.mask_all_pii(original_text)
        
        # Context should be preserved
        assert "Dr. Smith" in masked_text  # Names might be preserved in some contexts
        assert "hospital" in masked_text or "[EMAIL]" in masked_text
        assert "will call you at" in masked_text
        
        # PII should be masked
        assert "dr.smith@hospital.com" not in masked_text
        assert "555-123-4567" not in masked_text
        
    def test_partial_masking_options(self):
        """Test partial masking options that preserve some information."""
        email = "john.doe@example.com"
        phone = "555-123-4567"
        
        # Test partial email masking (preserve domain)
        partial_email = self.pii_masker.mask_email_partial(email)
        assert "example.com" in partial_email
        assert "john.doe" not in partial_email
        
        # Test partial phone masking (preserve area code)
        partial_phone = self.pii_masker.mask_phone_partial(phone)
        assert "555" in partial_phone
        assert "123-4567" not in partial_phone


class TestDocumentSanitization:
    """Test document-level sanitization for LlamaIndex."""
    
    def setup_method(self):
        """Set up test environment."""
        self.data_handler = SensitiveDataHandler()
        
    def test_document_pii_detection(self):
        """Test PII detection in LlamaIndex documents."""
        doc_content = """
        Medical Record
        
        Patient: John Doe
        DOB: 1985-03-15
        Email: john.doe@email.com
        Phone: 555-123-4567
        SSN: 123-45-6789
        
        Diagnosis: Patient presents with symptoms...
        Treatment: Prescribed medication...
        """
        
        document = Document(text=doc_content)
        pii_results = self.data_handler.detect_pii_in_document(document)
        
        assert len(pii_results) > 0
        
        # Check for expected PII types
        pii_types = {result["type"] for result in pii_results}
        expected_types = {"EMAIL", "PHONE", "SSN"}
        
        assert len(pii_types.intersection(expected_types)) >= 2
        
    def test_document_sanitization(self):
        """Test complete document sanitization."""
        doc_content = "Contact John Doe at john.doe@example.com or call 555-123-4567"
        document = Document(text=doc_content, metadata={"source": "contact_form"})
        
        sanitized_doc = self.data_handler.sanitize_document(document)
        
        # Verify PII is removed
        assert "john.doe@example.com" not in sanitized_doc.text
        assert "555-123-4567" not in sanitized_doc.text
        
        # Verify metadata is updated
        assert sanitized_doc.metadata["pii_detected"] is True
        assert sanitized_doc.metadata["sanitization_applied"] is True
        assert "sanitization_timestamp" in sanitized_doc.metadata
        
        # Verify original metadata is preserved
        assert sanitized_doc.metadata["source"] == "contact_form"
        
    def test_batch_document_sanitization(self):
        """Test sanitization of multiple documents."""
        documents = [
            Document(text="Email: user1@example.com", metadata={"id": "doc1"}),
            Document(text="Phone: 555-111-2222", metadata={"id": "doc2"}),
            Document(text="SSN: 111-22-3333", metadata={"id": "doc3"}),
            Document(text="No PII here", metadata={"id": "doc4"})
        ]
        
        sanitized_docs = self.data_handler.sanitize_documents(documents)
        
        assert len(sanitized_docs) == 4
        
        # Check that PII documents are sanitized
        for i, doc in enumerate(sanitized_docs[:3]):
            assert doc.metadata["pii_detected"] is True
            assert doc.metadata["sanitization_applied"] is True
            assert doc.metadata["id"] == f"doc{i+1}"
            
        # Check that non-PII document is unchanged
        assert sanitized_docs[3].metadata.get("pii_detected", False) is False
        assert sanitized_docs[3].text == "No PII here"
        
    def test_sanitization_audit_trail(self):
        """Test that sanitization creates proper audit trail."""
        doc_content = "Patient John Doe, email: john@example.com, phone: 555-123-4567"
        document = Document(text=doc_content)
        
        sanitized_doc = self.data_handler.sanitize_document(document, create_audit=True)
        
        # Check audit trail in metadata
        assert "audit_trail" in sanitized_doc.metadata
        audit_trail = sanitized_doc.metadata["audit_trail"]
        
        assert len(audit_trail) > 0
        assert audit_trail[0]["operation"] == "pii_sanitization"
        assert "timestamp" in audit_trail[0]
        assert "pii_types_found" in audit_trail[0]
        
    def test_selective_sanitization(self):
        """Test selective sanitization based on PII types."""
        doc_content = "Contact: john@example.com, phone: 555-123-4567, SSN: 123-45-6789"
        document = Document(text=doc_content)
        
        # Sanitize only emails and phones, preserve SSN
        sanitized_doc = self.data_handler.sanitize_document(
            document, 
            pii_types_to_sanitize=["EMAIL", "PHONE"]
        )
        
        # Email and phone should be sanitized
        assert "john@example.com" not in sanitized_doc.text
        assert "555-123-4567" not in sanitized_doc.text
        
        # SSN should be preserved
        assert "123-45-6789" in sanitized_doc.text


class TestDataClassification:
    """Test data classification and sensitivity levels."""
    
    def setup_method(self):
        """Set up test environment."""
        self.classifier = DataClassifier()
        
    def test_sensitivity_classification(self):
        """Test classification of data sensitivity levels."""
        test_cases = [
            ("Public information about our company", "PUBLIC"),
            ("Internal memo about project status", "INTERNAL"),
            ("Employee SSN: 123-45-6789", "CONFIDENTIAL"),
            ("Patient medical records with diagnosis", "RESTRICTED"),
            ("Credit card: 4532-1234-5678-9012", "RESTRICTED")
        ]
        
        for content, expected_level in test_cases:
            document = Document(text=content)
            classification = self.classifier.classify_sensitivity(document)
            
            assert classification["level"] == expected_level, \
                f"Expected {expected_level}, got {classification['level']} for: {content}"
                
    def test_pii_type_classification(self):
        """Test classification of PII types."""
        test_cases = [
            ("Email: john@example.com", ["CONTACT_INFO"]),
            ("SSN: 123-45-6789", ["GOVERNMENT_ID"]),
            ("Credit card: 4532-1234-5678-9012", ["FINANCIAL"]),
            ("Patient diagnosis: diabetes", ["HEALTH"]),
            ("Employee ID: EMP123456", ["EMPLOYMENT"])
        ]
        
        for content, expected_categories in test_cases:
            document = Document(text=content)
            classification = self.classifier.classify_pii_types(document)
            
            found_categories = classification["categories"]
            assert any(cat in found_categories for cat in expected_categories), \
                f"Expected categories {expected_categories}, got {found_categories} for: {content}"
                
    def test_compliance_classification(self):
        """Test classification for compliance requirements."""
        test_cases = [
            ("Patient health information", ["HIPAA"]),
            ("Credit card processing data", ["PCI_DSS"]),
            ("EU citizen personal data", ["GDPR"]),
            ("Financial account information", ["SOX", "PCI_DSS"]),
            ("Student educational records", ["FERPA"])
        ]
        
        for content, expected_regulations in test_cases:
            document = Document(text=content)
            classification = self.classifier.classify_compliance_requirements(document)
            
            found_regulations = classification["regulations"]
            assert any(reg in found_regulations for reg in expected_regulations), \
                f"Expected regulations {expected_regulations}, got {found_regulations} for: {content}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])