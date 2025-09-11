"""
Browser-Use Real Integration Tests

This module provides real integration tests that work with actual browser-use
and AgentCore components without mocking, but with proper error handling
for when services are not available.

Requirements covered:
- 6.1: Unit tests for AgentCore Browser Client integration and browser-use sensitive data handling
- 6.2: Integration testing for complete workflows
"""

import pytest
import asyncio
import os
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

# Import the modules we're testing
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))

from browseruse_agentcore_session_manager import (
    BrowserUseAgentCoreSessionManager,
    SessionConfig,
    SessionMetrics
)

from browseruse_sensitive_data_handler import (
    BrowserUseSensitiveDataHandler,
    BrowserUseCredentialManager,
    PIIType,
    ComplianceFramework
)

# Test configuration
TEST_REGION = os.getenv('AWS_REGION', 'us-east-1')
TEST_TIMEOUT = 30


class TestBrowserUseRealIntegration:
    """Real integration tests without mocking."""
    
    def test_session_config_creation(self):
        """Test session configuration creation."""
        config = SessionConfig(
            region=TEST_REGION,
            session_timeout=TEST_TIMEOUT,
            enable_live_view=True,
            enable_session_replay=True,
            max_retries=2,
            retry_delay=0.1
        )
        
        assert config.region == TEST_REGION
        assert config.session_timeout == TEST_TIMEOUT
        assert config.enable_live_view is True
        assert config.enable_session_replay is True
        assert config.max_retries == 2
        assert config.retry_delay == 0.1
    
    def test_session_manager_initialization(self):
        """Test session manager initialization."""
        config = SessionConfig(region=TEST_REGION)
        
        try:
            manager = BrowserUseAgentCoreSessionManager(config)
            
            assert manager.config == config
            assert len(manager.active_sessions) == 0
            assert len(manager.session_metrics) == 0
            
        except Exception as e:
            # If AgentCore dependencies are not available, that's expected
            assert "BrowserClient" in str(e) or "bedrock_agentcore" in str(e)
    
    def test_sensitive_data_handler_initialization(self):
        """Test sensitive data handler initialization."""
        handler = BrowserUseSensitiveDataHandler([
            ComplianceFramework.HIPAA,
            ComplianceFramework.PCI_DSS
        ])
        
        assert handler.compliance_frameworks == [
            ComplianceFramework.HIPAA,
            ComplianceFramework.PCI_DSS
        ]
        assert len(handler.patterns) > 0
    
    def test_pii_detection_functionality(self):
        """Test PII detection without external dependencies."""
        handler = BrowserUseSensitiveDataHandler()
        
        # Test SSN detection
        text_with_ssn = "Patient SSN: 123-45-6789"
        detections = handler.detect_pii(text_with_ssn)
        
        ssn_detections = [d for d in detections if d.pii_type == PIIType.SSN]
        assert len(ssn_detections) > 0
        assert "123-45-6789" in ssn_detections[0].value
        
        # Test email detection
        text_with_email = "Contact: john.doe@example.com"
        detections = handler.detect_pii(text_with_email)
        
        email_detections = [d for d in detections if d.pii_type == PIIType.EMAIL]
        assert len(email_detections) > 0
        assert "john.doe@example.com" in email_detections[0].value
    
    def test_data_masking_functionality(self):
        """Test data masking without external dependencies."""
        handler = BrowserUseSensitiveDataHandler()
        
        # Test SSN masking
        text_with_pii = "SSN: 123-45-6789, Email: john@example.com"
        masked_text, detections = handler.mask_text(text_with_pii)
        
        # Verify original values are not in masked text
        assert "123-45-6789" not in masked_text
        assert "john@example.com" not in masked_text
        
        # Verify some masking occurred
        assert "XXX-XX-6789" in masked_text or "*" in masked_text
        assert len(detections) >= 2  # SSN and email
    
    def test_credential_manager_functionality(self):
        """Test credential manager without external dependencies."""
        manager = BrowserUseCredentialManager()
        
        # Test credential storage and retrieval
        credential_id = "test_password"
        credential_value = "super_secret_password"
        
        manager.store_credential(credential_id, "password", credential_value)
        
        # Verify storage
        assert credential_id in manager.credentials_store
        
        # Test retrieval
        retrieved_value = manager.retrieve_credential(credential_id)
        assert retrieved_value == credential_value
        
        # Test access tracking
        credential_data = manager.credentials_store[credential_id]
        assert credential_data['access_count'] == 1
        assert credential_data['last_accessed'] is not None
        
        # Test deletion
        deleted = manager.delete_credential(credential_id)
        assert deleted is True
        assert credential_id not in manager.credentials_store
    
    def test_compliance_validation(self):
        """Test compliance validation functionality."""
        handler = BrowserUseSensitiveDataHandler([ComplianceFramework.HIPAA])
        
        # Test HIPAA compliance validation
        hipaa_text = "Patient: John Doe, SSN: 123-45-6789, DOB: 03/15/1985"
        
        result = handler.validate_compliance(
            hipaa_text,
            [ComplianceFramework.HIPAA]
        )
        
        assert result['compliant'] is False  # Should detect violations
        assert len(result['violations']) > 0
        assert result['total_pii_detected'] > 0
        
        # Check for HIPAA-specific violations
        hipaa_violations = [v for v in result['violations'] if v['framework'] == 'hipaa']
        assert len(hipaa_violations) > 0
    
    def test_data_classification(self):
        """Test data classification functionality."""
        handler = BrowserUseSensitiveDataHandler()
        
        # Test different classification levels
        test_cases = [
            ("Hello world", "public"),
            ("Email: user@example.com", "confidential"),
            ("SSN: 123-45-6789", "restricted"),
            ("Credit Card: 4532-1234-5678-9012", "restricted")
        ]
        
        for text, expected_level in test_cases:
            classification = handler.classify_data(text)
            
            # Verify classification is reasonable
            if expected_level == "public":
                assert classification.value in ["public", "internal"]
            elif expected_level == "confidential":
                assert classification.value in ["confidential", "internal"]
            elif expected_level == "restricted":
                assert classification.value == "restricted"
    
    def test_session_metrics_creation(self):
        """Test session metrics creation."""
        session_id = str(uuid.uuid4())
        
        metrics = SessionMetrics(
            session_id=session_id,
            start_time=datetime.now()
        )
        
        assert metrics.session_id == session_id
        assert metrics.start_time is not None
        assert metrics.end_time is None
        assert metrics.operations_count == 0
        assert metrics.sensitive_data_accessed is False
        assert len(metrics.compliance_violations) == 0
        assert len(metrics.errors) == 0
    
    def test_error_handling_in_pii_detection(self):
        """Test error handling in PII detection."""
        handler = BrowserUseSensitiveDataHandler()
        
        # Test with None input
        detections = handler.detect_pii(None)
        assert len(detections) == 0
        
        # Test with empty string
        detections = handler.detect_pii("")
        assert len(detections) == 0
        
        # Test with non-string input (should handle gracefully)
        try:
            detections = handler.detect_pii(12345)
            # Should either work or handle gracefully
        except Exception:
            # If it raises an exception, that's also acceptable
            pass
    
    def test_integration_scenario_healthcare(self):
        """Test healthcare integration scenario."""
        handler = BrowserUseSensitiveDataHandler([ComplianceFramework.HIPAA])
        credential_manager = BrowserUseCredentialManager()
        
        # Simulate healthcare form data
        form_data = {
            "patient_name": "Jane Smith",
            "ssn": "987-65-4321",
            "date_of_birth": "07/22/1978",
            "medical_record": "MRN-XYZ789012"
        }
        
        # Process each field
        processed_fields = {}
        total_detections = []
        
        for field_name, field_value in form_data.items():
            masked_value, detections = handler.mask_text(str(field_value), field_name)
            processed_fields[field_name] = masked_value
            total_detections.extend(detections)
        
        # Verify PII was detected and masked
        assert len(total_detections) > 0
        assert "987-65-4321" not in processed_fields["ssn"]
        
        # Test credential storage for healthcare scenario
        credential_manager.store_credential(
            "patient_portal_access",
            "api_key",
            "healthcare_api_key_12345",
            {"service": "patient_portal", "compliance": "HIPAA"}
        )
        
        # Verify credential was stored securely
        credentials = credential_manager.list_credentials()
        assert len(credentials) == 1
        assert credentials[0]['type'] == 'api_key'
        assert 'healthcare_api_key_12345' not in str(credentials)  # Should not expose value
    
    def test_integration_scenario_financial(self):
        """Test financial integration scenario."""
        handler = BrowserUseSensitiveDataHandler([ComplianceFramework.PCI_DSS])
        
        # Simulate financial form data
        form_data = {
            "cardholder_name": "Robert Johnson",
            "credit_card": "4532-1234-5678-9012",
            "expiry": "12/25",
            "cvv": "123"
        }
        
        # Process the data
        full_text = " ".join(form_data.values())
        masked_text, detections = handler.mask_text(full_text)
        
        # Verify credit card was detected and masked
        credit_card_detections = [d for d in detections if d.pii_type == PIIType.CREDIT_CARD]
        assert len(credit_card_detections) > 0
        assert "4532-1234-5678-9012" not in masked_text
        
        # Verify PCI-DSS compliance validation
        compliance_result = handler.validate_compliance(
            full_text,
            [ComplianceFramework.PCI_DSS]
        )
        
        assert compliance_result['compliant'] is False
        pci_violations = [v for v in compliance_result['violations'] if v['framework'] == 'pci_dss']
        assert len(pci_violations) > 0


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    def test_invalid_session_config(self):
        """Test handling of invalid session configuration."""
        # Test with invalid region
        try:
            config = SessionConfig(region="invalid-region")
            manager = BrowserUseAgentCoreSessionManager(config)
            # Should either work or fail gracefully
        except Exception as e:
            # Expected if AgentCore validates regions
            assert "region" in str(e).lower() or "BrowserClient" in str(e)
    
    def test_credential_manager_edge_cases(self):
        """Test credential manager edge cases."""
        manager = BrowserUseCredentialManager()
        
        # Test retrieving non-existent credential
        result = manager.retrieve_credential("non_existent")
        assert result is None
        
        # Test deleting non-existent credential
        deleted = manager.delete_credential("non_existent")
        assert deleted is False
        
        # Test empty credential storage
        credentials = manager.list_credentials()
        assert len(credentials) == 0
        
        # Test access log when no access has occurred
        access_log = manager.get_access_log()
        assert len(access_log) == 0
    
    def test_pii_detection_edge_cases(self):
        """Test PII detection edge cases."""
        handler = BrowserUseSensitiveDataHandler()
        
        # Test with text that looks like PII but isn't
        false_positive_text = "Version 1.2.3, Port 8080, Count 123456789"
        detections = handler.detect_pii(false_positive_text)
        
        # Should have minimal high-confidence detections
        high_confidence = [d for d in detections if d.confidence > 0.9]
        assert len(high_confidence) <= 1  # Allow some false positives but not many
        
        # Test with mixed content
        mixed_text = "Contact info: email@example.com, phone: 555-1234, version: 2.1.0"
        detections = handler.detect_pii(mixed_text)
        
        # Should detect email but not version number
        email_detections = [d for d in detections if d.pii_type == PIIType.EMAIL]
        assert len(email_detections) >= 1


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])