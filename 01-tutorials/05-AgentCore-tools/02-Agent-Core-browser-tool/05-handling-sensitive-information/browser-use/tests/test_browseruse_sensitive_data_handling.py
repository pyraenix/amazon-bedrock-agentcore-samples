"""
Browser-Use Sensitive Data Handling Tests

This module provides comprehensive tests for PII detection and masking in browser-use operations,
credential security and isolation, and compliance validation during browser-use sensitive operations.

Requirements covered:
- 6.1: Unit tests for AgentCore Browser Client integration and browser-use sensitive data handling
- 6.3: Security testing for sensitive data exposure prevention
- 6.4: Compliance audit tests for regulatory compliance verification
"""

import pytest
import json
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock

# Import the modules we're testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))

from browseruse_sensitive_data_handler import (
    BrowserUseSensitiveDataHandler,
    BrowserUseCredentialManager,
    BrowserUseDataClassifier,
    PIIType,
    DataClassification,
    ComplianceFramework,
    DetectionResult,
    SensitiveDataContext,
    detect_and_mask_pii,
    classify_sensitive_data
)


class TestBrowserUseSensitiveDataHandler:
    """Test suite for browser-use sensitive data handler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = BrowserUseSensitiveDataHandler([
            ComplianceFramework.HIPAA,
            ComplianceFramework.PCI_DSS,
            ComplianceFramework.GDPR
        ])
    
    def test_detect_pii_ssn(self):
        """Test SSN detection in various formats."""
        test_cases = [
            ("SSN: 123-45-6789", "123-45-6789"),
            ("Social Security Number 123456789", "123456789"),
            ("Tax ID: 987-65-4321", "987-65-4321"),
            ("Patient SSN is 555-44-3333", "555-44-3333")
        ]
        
        for text, expected_ssn in test_cases:
            detections = self.handler.detect_pii(text)
            
            ssn_detections = [d for d in detections if d.pii_type == PIIType.SSN]
            assert len(ssn_detections) == 1, f"Failed to detect SSN in: {text}"
            
            detection = ssn_detections[0]
            assert expected_ssn in detection.value
            assert detection.confidence >= 0.8
            assert ComplianceFramework.HIPAA in detection.compliance_impact
    
    def test_detect_pii_email(self):
        """Test email detection."""
        test_cases = [
            "Contact: john.doe@example.com",
            "Email address: user+tag@domain.co.uk",
            "Send to: test.email@subdomain.example.org"
        ]
        
        for text in test_cases:
            detections = self.handler.detect_pii(text)
            
            email_detections = [d for d in detections if d.pii_type == PIIType.EMAIL]
            assert len(email_detections) == 1, f"Failed to detect email in: {text}"
            
            detection = email_detections[0]
            assert "@" in detection.value
            assert detection.confidence >= 0.9
            assert ComplianceFramework.GDPR in detection.compliance_impact
    
    def test_detect_pii_phone(self):
        """Test phone number detection."""
        test_cases = [
            ("Phone: (555) 123-4567", "(555) 123-4567"),
            ("Call 555-123-4567", "555-123-4567"),
            ("Mobile: +1-555-123-4567", "+1-555-123-4567"),
            ("Tel: 555.123.4567", "555.123.4567")
        ]
        
        for text, expected_phone in test_cases:
            detections = self.handler.detect_pii(text)
            
            phone_detections = [d for d in detections if d.pii_type == PIIType.PHONE]
            assert len(phone_detections) >= 1, f"Failed to detect phone in: {text}"
            
            # Check if any detection matches expected phone
            found_match = any(expected_phone in d.value for d in phone_detections)
            assert found_match, f"Expected phone {expected_phone} not found in detections"
    
    def test_detect_pii_credit_card(self):
        """Test credit card detection."""
        test_cases = [
            ("Visa: 4532-1234-5678-9012", "4532-1234-5678-9012"),
            ("MasterCard 5555555555554444", "5555555555554444"),
            ("Amex: 3782-822463-10005", "3782-822463-10005"),
            ("Discover 6011111111111117", "6011111111111117")
        ]
        
        for text, expected_card in test_cases:
            detections = self.handler.detect_pii(text)
            
            card_detections = [d for d in detections if d.pii_type == PIIType.CREDIT_CARD]
            assert len(card_detections) >= 1, f"Failed to detect credit card in: {text}"
            
            detection = card_detections[0]
            assert detection.confidence >= 0.9
            assert ComplianceFramework.PCI_DSS in detection.compliance_impact
    
    def test_detect_pii_date_of_birth(self):
        """Test date of birth detection."""
        test_cases = [
            "DOB: 03/15/1985",
            "Born on 12/25/1990",
            "Birthday: 07-04-1976"
        ]
        
        for text in test_cases:
            detections = self.handler.detect_pii(text, context="birth date")
            
            dob_detections = [d for d in detections if d.pii_type == PIIType.DATE_OF_BIRTH]
            assert len(dob_detections) >= 1, f"Failed to detect DOB in: {text}"
            
            detection = dob_detections[0]
            assert ComplianceFramework.HIPAA in detection.compliance_impact
            assert ComplianceFramework.GDPR in detection.compliance_impact
    
    def test_detect_pii_medical_record(self):
        """Test medical record number detection."""
        test_cases = [
            "MRN: ABC123456",
            "Medical Record Number: MR-789012",
            "Patient ID: MEDICAL-XYZ789"
        ]
        
        for text in test_cases:
            detections = self.handler.detect_pii(text, context="medical patient")
            
            mrn_detections = [d for d in detections if d.pii_type == PIIType.MEDICAL_RECORD]
            assert len(mrn_detections) >= 1, f"Failed to detect MRN in: {text}"
            
            detection = mrn_detections[0]
            assert ComplianceFramework.HIPAA in detection.compliance_impact
    
    def test_detect_pii_ip_address(self):
        """Test IP address detection."""
        test_cases = [
            "Server IP: 192.168.1.100",
            "Connect to 10.0.0.1",
            "Public IP 203.0.113.42"
        ]
        
        for text in test_cases:
            detections = self.handler.detect_pii(text)
            
            ip_detections = [d for d in detections if d.pii_type == PIIType.IP_ADDRESS]
            assert len(ip_detections) >= 1, f"Failed to detect IP in: {text}"
            
            detection = ip_detections[0]
            assert ComplianceFramework.GDPR in detection.compliance_impact
    
    def test_mask_pii_values(self):
        """Test PII masking functionality."""
        test_cases = [
            (PIIType.SSN, "123-45-6789", "XXX-XX-6789"),
            (PIIType.EMAIL, "john.doe@example.com", "j*****e@example.com"),
            (PIIType.PHONE, "(555) 123-4567", "XXX-XXX-4567"),
            (PIIType.CREDIT_CARD, "4532-1234-5678-9012", "**** **** **** 9012"),
            (PIIType.DATE_OF_BIRTH, "03/15/1985", "XX/XX/XXXX"),
            (PIIType.MEDICAL_RECORD, "MRN-ABC123456", "MRN-XXXXXXXX"),
            (PIIType.IP_ADDRESS, "192.168.1.100", "192.168.XXX.XXX")
        ]
        
        for pii_type, original, expected_mask in test_cases:
            masked = self.handler._mask_value(original, pii_type)
            assert masked == expected_mask, f"Incorrect masking for {pii_type}: {masked} != {expected_mask}"
    
    def test_mask_text_comprehensive(self):
        """Test comprehensive text masking with multiple PII types."""
        text = """
        Patient: John Doe
        SSN: 123-45-6789
        Email: john.doe@hospital.com
        Phone: (555) 123-4567
        DOB: 03/15/1985
        Credit Card: 4532-1234-5678-9012
        MRN: ABC123456
        IP: 192.168.1.100
        """
        
        masked_text, detections = self.handler.mask_text(text)
        
        # Verify multiple PII types detected
        pii_types_found = {d.pii_type for d in detections}
        expected_types = {
            PIIType.SSN, PIIType.EMAIL, PIIType.PHONE, 
            PIIType.DATE_OF_BIRTH, PIIType.CREDIT_CARD, PIIType.IP_ADDRESS
        }
        
        assert len(pii_types_found & expected_types) >= 5, f"Expected multiple PII types, found: {pii_types_found}"
        
        # Verify original values are not in masked text
        assert "123-45-6789" not in masked_text
        assert "john.doe@hospital.com" not in masked_text
        assert "(555) 123-4567" not in masked_text
        assert "4532-1234-5678-9012" not in masked_text
        
        # Verify masked values are present
        assert "XXX-XX-6789" in masked_text
        assert "**** **** **** 9012" in masked_text
    
    def test_classify_data_levels(self):
        """Test data classification based on PII content."""
        test_cases = [
            ("Hello world", DataClassification.PUBLIC),
            ("Email: user@example.com", DataClassification.CONFIDENTIAL),
            ("Phone: 555-123-4567", DataClassification.CONFIDENTIAL),
            ("SSN: 123-45-6789", DataClassification.RESTRICTED),
            ("Credit Card: 4532-1234-5678-9012", DataClassification.RESTRICTED),
            ("MRN: ABC123456", DataClassification.RESTRICTED)
        ]
        
        for text, expected_classification in test_cases:
            classification = self.handler.classify_data(text)
            assert classification == expected_classification, f"Incorrect classification for: {text}"
    
    def test_validate_compliance_hipaa(self):
        """Test HIPAA compliance validation."""
        hipaa_text = """
        Patient: John Doe
        SSN: 123-45-6789
        DOB: 03/15/1985
        MRN: ABC123456
        """
        
        result = self.handler.validate_compliance(
            hipaa_text, 
            [ComplianceFramework.HIPAA]
        )
        
        assert result['compliant'] is False
        assert len(result['violations']) > 0
        assert result['total_pii_detected'] > 0
        
        # Check specific HIPAA violations
        hipaa_violations = [v for v in result['violations'] if v['framework'] == 'hipaa']
        assert len(hipaa_violations) > 0
    
    def test_validate_compliance_pci_dss(self):
        """Test PCI-DSS compliance validation."""
        pci_text = """
        Payment Information:
        Card Number: 4532-1234-5678-9012
        Expiry: 12/25
        CVV: 123
        """
        
        result = self.handler.validate_compliance(
            pci_text,
            [ComplianceFramework.PCI_DSS]
        )
        
        assert result['compliant'] is False
        assert len(result['violations']) > 0
        
        # Check PCI-DSS violations
        pci_violations = [v for v in result['violations'] if v['framework'] == 'pci_dss']
        assert len(pci_violations) > 0
    
    def test_validate_compliance_gdpr(self):
        """Test GDPR compliance validation."""
        gdpr_text = """
        User Profile:
        Email: user@example.com
        Phone: +44 20 7946 0958
        IP Address: 203.0.113.42
        """
        
        result = self.handler.validate_compliance(
            gdpr_text,
            [ComplianceFramework.GDPR]
        )
        
        assert result['compliant'] is False
        assert len(result['violations']) > 0
        
        # Check GDPR violations
        gdpr_violations = [v for v in result['violations'] if v['framework'] == 'gdpr']
        assert len(gdpr_violations) > 0
    
    def test_context_enhanced_detection(self):
        """Test that context improves PII detection confidence."""
        # Test with context
        text = "ID: 123456789"
        
        # Without context - might be detected as generic number
        detections_no_context = self.handler.detect_pii(text)
        
        # With SSN context - should boost confidence
        detections_with_context = self.handler.detect_pii(text, context="social security number")
        
        # Find SSN detections
        ssn_no_context = [d for d in detections_no_context if d.pii_type == PIIType.SSN]
        ssn_with_context = [d for d in detections_with_context if d.pii_type == PIIType.SSN]
        
        if ssn_no_context and ssn_with_context:
            assert ssn_with_context[0].confidence >= ssn_no_context[0].confidence
    
    def test_multiple_pii_same_type(self):
        """Test detection of multiple PII items of the same type."""
        text = """
        Primary email: john@example.com
        Secondary email: john.doe@work.com
        Backup email: j.doe@personal.org
        """
        
        detections = self.handler.detect_pii(text)
        email_detections = [d for d in detections if d.pii_type == PIIType.EMAIL]
        
        assert len(email_detections) == 3
        
        # Verify all emails are different
        email_values = {d.value for d in email_detections}
        assert len(email_values) == 3
    
    def test_edge_cases_and_false_positives(self):
        """Test edge cases and potential false positives."""
        # These should NOT be detected as PII
        non_pii_text = """
        Version: 1.2.3
        Port: 8080
        Count: 123456789 items
        Date: 2023-12-25 (Christmas)
        Code: ABC123 (not medical)
        """
        
        detections = self.handler.detect_pii(non_pii_text)
        
        # Should have minimal or no high-confidence detections
        high_confidence_detections = [d for d in detections if d.confidence > 0.9]
        
        # Allow some low-confidence detections but not high-confidence false positives
        assert len(high_confidence_detections) <= 1, f"Too many false positives: {high_confidence_detections}"


class TestBrowserUseCredentialManager:
    """Test suite for browser-use credential manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.credential_manager = BrowserUseCredentialManager()
    
    def test_store_and_retrieve_credential(self):
        """Test basic credential storage and retrieval."""
        credential_id = "test_password"
        credential_type = "password"
        credential_value = "super_secret_password_123"
        metadata = {"service": "test_service", "user": "test_user"}
        
        # Store credential
        self.credential_manager.store_credential(
            credential_id, credential_type, credential_value, metadata
        )
        
        # Verify storage
        assert credential_id in self.credential_manager.credentials_store
        
        # Retrieve credential
        retrieved_value = self.credential_manager.retrieve_credential(credential_id)
        
        assert retrieved_value == credential_value
        
        # Verify access tracking
        credential_data = self.credential_manager.credentials_store[credential_id]
        assert credential_data['access_count'] == 1
        assert credential_data['last_accessed'] is not None
    
    def test_store_multiple_credential_types(self):
        """Test storing different types of credentials."""
        credentials = [
            ("api_key", "api_key", "sk-1234567890abcdef", {"service": "openai"}),
            ("db_password", "password", "db_pass_456", {"database": "production"}),
            ("oauth_token", "token", "oauth_xyz789", {"provider": "google"}),
            ("ssh_key", "private_key", "-----BEGIN RSA PRIVATE KEY-----", {"server": "prod"})
        ]
        
        for cred_id, cred_type, cred_value, metadata in credentials:
            self.credential_manager.store_credential(cred_id, cred_type, cred_value, metadata)
        
        # Verify all stored
        assert len(self.credential_manager.credentials_store) == 4
        
        # Verify retrieval
        for cred_id, _, cred_value, _ in credentials:
            retrieved = self.credential_manager.retrieve_credential(cred_id)
            assert retrieved == cred_value
    
    def test_credential_encryption(self):
        """Test that credentials are encrypted in storage."""
        credential_id = "test_encryption"
        credential_value = "plaintext_password"
        
        self.credential_manager.store_credential(
            credential_id, "password", credential_value
        )
        
        # Verify raw stored value is encrypted (not plaintext)
        stored_data = self.credential_manager.credentials_store[credential_id]
        encrypted_value = stored_data['encrypted_value']
        
        assert encrypted_value != credential_value
        assert len(encrypted_value) > 0
        
        # Verify decryption works
        decrypted = self.credential_manager.retrieve_credential(credential_id)
        assert decrypted == credential_value
    
    def test_credential_not_found(self):
        """Test retrieval of non-existent credential."""
        result = self.credential_manager.retrieve_credential("non_existent")
        assert result is None
    
    def test_delete_credential(self):
        """Test credential deletion."""
        credential_id = "to_delete"
        self.credential_manager.store_credential(
            credential_id, "password", "delete_me"
        )
        
        # Verify stored
        assert credential_id in self.credential_manager.credentials_store
        
        # Delete
        result = self.credential_manager.delete_credential(credential_id)
        assert result is True
        
        # Verify deleted
        assert credential_id not in self.credential_manager.credentials_store
        
        # Try to delete again
        result = self.credential_manager.delete_credential(credential_id)
        assert result is False
    
    def test_list_credentials(self):
        """Test listing stored credentials without exposing values."""
        credentials = [
            ("cred1", "password", "value1", {"service": "service1"}),
            ("cred2", "api_key", "value2", {"service": "service2"}),
            ("cred3", "token", "value3", {"service": "service3"})
        ]
        
        for cred_id, cred_type, cred_value, metadata in credentials:
            self.credential_manager.store_credential(cred_id, cred_type, cred_value, metadata)
        
        # List credentials
        credential_list = self.credential_manager.list_credentials()
        
        assert len(credential_list) == 3
        
        # Verify no values are exposed
        for cred_info in credential_list:
            assert 'credential_id' in cred_info
            assert 'type' in cred_info
            assert 'created_at' in cred_info
            assert 'access_count' in cred_info
            assert 'metadata' in cred_info
            assert 'encrypted_value' not in cred_info
            assert 'value' not in cred_info
    
    def test_access_logging(self):
        """Test credential access logging."""
        credential_id = "logged_access"
        self.credential_manager.store_credential(
            credential_id, "password", "test_value"
        )
        
        # Initial access log should be empty
        access_log = self.credential_manager.get_access_log()
        initial_count = len(access_log)
        
        # Access credential multiple times
        for i in range(3):
            self.credential_manager.retrieve_credential(credential_id)
        
        # Verify access log updated
        access_log = self.credential_manager.get_access_log()
        assert len(access_log) == initial_count + 3
        
        # Verify log entries
        recent_entries = access_log[-3:]
        for entry in recent_entries:
            assert entry['credential_id'] == credential_id
            assert entry['type'] == 'password'
            assert 'accessed_at' in entry
    
    def test_clear_all_credentials(self):
        """Test emergency credential cleanup."""
        # Store multiple credentials
        for i in range(5):
            self.credential_manager.store_credential(
                f"cred_{i}", "password", f"value_{i}"
            )
        
        assert len(self.credential_manager.credentials_store) == 5
        
        # Clear all
        self.credential_manager.clear_all_credentials()
        
        assert len(self.credential_manager.credentials_store) == 0
        assert len(self.credential_manager.access_log) == 0
    
    def test_credential_metadata_tracking(self):
        """Test credential metadata and tracking."""
        credential_id = "metadata_test"
        metadata = {
            "service": "test_service",
            "environment": "production",
            "owner": "test_user",
            "expires": "2024-12-31"
        }
        
        self.credential_manager.store_credential(
            credential_id, "api_key", "test_key", metadata
        )
        
        # Verify metadata stored
        credential_list = self.credential_manager.list_credentials()
        cred_info = next(c for c in credential_list if c['credential_id'] == credential_id)
        
        assert cred_info['metadata'] == metadata
        assert cred_info['access_count'] == 0
        assert cred_info['last_accessed'] is None
        
        # Access and verify tracking
        self.credential_manager.retrieve_credential(credential_id)
        
        updated_list = self.credential_manager.list_credentials()
        updated_info = next(c for c in updated_list if c['credential_id'] == credential_id)
        
        assert updated_info['access_count'] == 1
        assert updated_info['last_accessed'] is not None


class TestBrowserUseDataClassifier:
    """Test suite for browser-use data classifier."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = BrowserUseDataClassifier([
            ComplianceFramework.HIPAA,
            ComplianceFramework.PCI_DSS,
            ComplianceFramework.GDPR
        ])
    
    def test_classify_form_data_public(self):
        """Test classification of public form data."""
        form_data = {
            "name": "John Doe",
            "company": "Example Corp",
            "message": "Hello, I'm interested in your services"
        }
        
        result = self.classifier.classify_form_data(form_data)
        
        # Should be mostly public/confidential
        assert result['overall_classification'] in [DataClassification.PUBLIC, DataClassification.CONFIDENTIAL]
        assert result['total_pii_detected'] >= 0  # Name might be detected as PII
        assert len(result['compliance_issues']) == 0  # No high-risk PII
    
    def test_classify_form_data_healthcare(self):
        """Test classification of healthcare form data."""
        form_data = {
            "patient_name": "John Doe",
            "ssn": "123-45-6789",
            "date_of_birth": "03/15/1985",
            "medical_record_number": "MRN-ABC123456",
            "diagnosis": "Hypertension",
            "insurance_id": "INS-789012"
        }
        
        result = self.classifier.classify_form_data(form_data)
        
        # Should be restricted due to SSN and MRN
        assert result['overall_classification'] == DataClassification.RESTRICTED
        assert result['total_pii_detected'] > 0
        assert len(result['compliance_issues']) > 0
        
        # Check for HIPAA compliance issues
        hipaa_issues = [issue for issue in result['compliance_issues'] if issue['framework'] == 'hipaa']
        assert len(hipaa_issues) > 0
    
    def test_classify_form_data_financial(self):
        """Test classification of financial form data."""
        form_data = {
            "cardholder_name": "John Doe",
            "credit_card_number": "4532-1234-5678-9012",
            "expiry_date": "12/25",
            "cvv": "123",
            "billing_address": "123 Main St, Anytown, ST 12345"
        }
        
        result = self.classifier.classify_form_data(form_data)
        
        # Should be restricted due to credit card
        assert result['overall_classification'] == DataClassification.RESTRICTED
        assert result['total_pii_detected'] > 0
        
        # Check for PCI-DSS compliance issues
        pci_issues = [issue for issue in result['compliance_issues'] if issue['framework'] == 'pci_dss']
        assert len(pci_issues) > 0
    
    def test_classify_form_data_mixed_sensitivity(self):
        """Test classification of mixed sensitivity form data."""
        form_data = {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "phone": "(555) 123-4567",
            "company": "Example Corp",
            "message": "Please contact me about your services"
        }
        
        result = self.classifier.classify_form_data(form_data)
        
        # Should be confidential due to email and phone
        assert result['overall_classification'] == DataClassification.CONFIDENTIAL
        assert result['total_pii_detected'] > 0
        
        # Check field-level classifications
        assert result['field_classifications']['email']['classification'] == DataClassification.CONFIDENTIAL
        assert result['field_classifications']['phone']['classification'] == DataClassification.CONFIDENTIAL
    
    def test_field_name_analysis(self):
        """Test field name analysis for PII hints."""
        test_cases = [
            ("ssn", DataClassification.RESTRICTED),
            ("social_security_number", DataClassification.RESTRICTED),
            ("email", DataClassification.CONFIDENTIAL),
            ("phone", DataClassification.CONFIDENTIAL),
            ("credit_card", DataClassification.RESTRICTED),
            ("date_of_birth", DataClassification.CONFIDENTIAL),
            ("name", DataClassification.CONFIDENTIAL),
            ("company", DataClassification.PUBLIC)
        ]
        
        for field_name, expected_classification in test_cases:
            context = self.classifier._analyze_field_name(field_name)
            assert context['classification'] == expected_classification
    
    def test_get_field_recommendations(self):
        """Test field-level security recommendations."""
        # Test restricted field recommendations
        restricted_recommendations = self.classifier._get_field_recommendations(
            DataClassification.RESTRICTED, []
        )
        
        assert "secure input methods" in " ".join(restricted_recommendations).lower()
        assert "encryption" in " ".join(restricted_recommendations).lower()
        assert "audit" in " ".join(restricted_recommendations).lower()
        
        # Test confidential field recommendations
        confidential_recommendations = self.classifier._get_field_recommendations(
            DataClassification.CONFIDENTIAL, []
        )
        
        assert "mask" in " ".join(confidential_recommendations).lower()
        assert "secure" in " ".join(confidential_recommendations).lower()
    
    def test_get_overall_recommendations(self):
        """Test overall form security recommendations."""
        compliance_issues = [
            {'framework': 'hipaa', 'pii_type': 'ssn', 'severity': 'high'},
            {'framework': 'pci_dss', 'pii_type': 'credit_card', 'severity': 'high'}
        ]
        
        recommendations = self.classifier._get_overall_recommendations(
            DataClassification.RESTRICTED, compliance_issues
        )
        
        assert "micro-vm" in " ".join(recommendations).lower()
        assert "audit" in " ".join(recommendations).lower()
        assert "hipaa" in " ".join(recommendations).lower()
        assert "pci-dss" in " ".join(recommendations).lower()


class TestConvenienceFunctions:
    """Test convenience functions for common operations."""
    
    def test_detect_and_mask_pii_function(self):
        """Test convenience function for PII detection and masking."""
        text = "Contact John at john.doe@example.com or call (555) 123-4567"
        
        masked_text, detections = detect_and_mask_pii(text, [ComplianceFramework.GDPR])
        
        # Verify masking occurred
        assert "john.doe@example.com" not in masked_text
        assert "(555) 123-4567" not in masked_text
        
        # Verify detections
        assert len(detections) >= 2  # Email and phone
        pii_types = {d.pii_type for d in detections}
        assert PIIType.EMAIL in pii_types
        assert PIIType.PHONE in pii_types
    
    def test_classify_sensitive_data_function(self):
        """Test convenience function for data classification."""
        test_cases = [
            ("Hello world", DataClassification.PUBLIC),
            ("Email: user@example.com", DataClassification.CONFIDENTIAL),
            ("SSN: 123-45-6789", DataClassification.RESTRICTED)
        ]
        
        for text, expected_classification in test_cases:
            classification = classify_sensitive_data(text)
            assert classification == expected_classification


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = BrowserUseSensitiveDataHandler([ComplianceFramework.HIPAA])
        self.credential_manager = BrowserUseCredentialManager()
        self.classifier = BrowserUseDataClassifier([ComplianceFramework.HIPAA])
    
    def test_healthcare_form_processing_scenario(self):
        """Test complete healthcare form processing scenario."""
        # Simulate healthcare form data
        form_data = {
            "patient_name": "Jane Smith",
            "ssn": "987-65-4321",
            "date_of_birth": "07/22/1978",
            "medical_record": "MRN-XYZ789012",
            "insurance_number": "INS-456789",
            "diagnosis": "Type 2 Diabetes",
            "medications": "Metformin 500mg",
            "emergency_contact": "John Smith (555) 987-6543"
        }
        
        # Step 1: Classify the form data
        classification_result = self.classifier.classify_form_data(form_data)
        
        assert classification_result['overall_classification'] == DataClassification.RESTRICTED
        assert len(classification_result['compliance_issues']) > 0
        
        # Step 2: Process each field for PII
        processed_fields = {}
        total_detections = []
        
        for field_name, field_value in form_data.items():
            masked_value, detections = self.handler.mask_text(str(field_value), field_name)
            processed_fields[field_name] = masked_value
            total_detections.extend(detections)
        
        # Verify PII was masked
        assert "987-65-4321" not in processed_fields["ssn"]
        assert "MRN-XYZ789012" not in processed_fields["medical_record"]
        assert "(555) 987-6543" not in processed_fields["emergency_contact"]
        
        # Step 3: Validate HIPAA compliance
        full_text = " ".join(form_data.values())
        compliance_result = self.handler.validate_compliance(
            full_text, [ComplianceFramework.HIPAA]
        )
        
        assert compliance_result['compliant'] is False
        assert len(compliance_result['violations']) > 0
        
        # Step 4: Store credentials securely (if any)
        if "password" in form_data:
            self.credential_manager.store_credential(
                "patient_portal_password", "password", form_data["password"]
            )
    
    def test_financial_form_processing_scenario(self):
        """Test complete financial form processing scenario."""
        form_data = {
            "cardholder_name": "Robert Johnson",
            "credit_card": "4532-1234-5678-9012",
            "expiry": "12/25",
            "cvv": "123",
            "ssn": "555-44-3333",
            "annual_income": "$75,000",
            "employer": "Tech Corp Inc"
        }
        
        # Classify and process
        classification_result = self.classifier.classify_form_data(form_data)
        assert classification_result['overall_classification'] == DataClassification.RESTRICTED
        
        # Mask PII
        full_text = " ".join(form_data.values())
        masked_text, detections = self.handler.mask_text(full_text)
        
        # Verify credit card and SSN are masked
        assert "4532-1234-5678-9012" not in masked_text
        assert "555-44-3333" not in masked_text
        
        # Verify PCI-DSS compliance issues detected
        pii_types = {d.pii_type for d in detections}
        assert PIIType.CREDIT_CARD in pii_types
        assert PIIType.SSN in pii_types
    
    def test_error_handling_and_recovery(self):
        """Test error handling in sensitive data processing."""
        # Test with malformed data
        malformed_data = {
            "field1": None,
            "field2": "",
            "field3": 12345,  # Non-string data
            "field4": ["list", "data"],  # Complex data
        }
        
        # Should handle gracefully without crashing
        try:
            classification_result = self.classifier.classify_form_data(malformed_data)
            assert classification_result is not None
        except Exception as e:
            pytest.fail(f"Should handle malformed data gracefully: {e}")
        
        # Test with invalid credential operations
        invalid_credential = self.credential_manager.retrieve_credential("non_existent")
        assert invalid_credential is None
        
        # Test with empty text
        empty_detections = self.handler.detect_pii("")
        assert len(empty_detections) == 0
        
        empty_masked, empty_det = self.handler.mask_text("")
        assert empty_masked == ""
        assert len(empty_det) == 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])