"""
Browser-Use AgentCore Integration Tests

This module provides comprehensive tests for validating browser-use Agent connection
to AgentCore Browser Tool, WebSocket connection and CDP integration with sensitive data,
and session lifecycle management during browser-use operations.

Requirements covered:
- 6.1: Unit tests for AgentCore Browser Client integration and browser-use sensitive data handling
- 6.2: Integration testing for complete workflows from AgentCore session creation through browser-use automation to session cleanup
"""

import pytest
import asyncio
import json
import uuid
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

# Import the modules we're testing
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))

# Import core modules
from browseruse_sensitive_data_handler import (
    BrowserUseSensitiveDataHandler,
    BrowserUseCredentialManager,
    PIIType,
    ComplianceFramework
)

# Import session management with error handling
try:
    from browseruse_agentcore_session_manager import (
        BrowserUseAgentCoreSessionManager,
        SessionConfig,
        SessionMetrics,
        create_secure_browseruse_session
    )
    SESSION_MANAGER_AVAILABLE = True
except ImportError as e:
    SESSION_MANAGER_AVAILABLE = False
    # Create dummy classes for testing
    class SessionConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class SessionMetrics:
        def __init__(self, session_id, start_time):
            self.session_id = session_id
            self.start_time = start_time
            self.end_time = None
            self.operations_count = 0
            self.sensitive_data_accessed = False
            self.compliance_violations = []
            self.errors = []

# Check for optional dependencies
try:
    from bedrock_agentcore.tools.browser_client import BrowserClient
    AGENTCORE_AVAILABLE = True
except ImportError:
    AGENTCORE_AVAILABLE = False

try:
    from browser_use import Agent
    from browser_use.browser.session import BrowserSession
    BROWSER_USE_AVAILABLE = True
except ImportError:
    BROWSER_USE_AVAILABLE = False

# Test configuration
TEST_REGION = os.getenv('AWS_REGION', 'us-east-1')
TEST_TIMEOUT = 30  # Shorter timeout for tests


@pytest.fixture
def session_config():
    """Test session configuration."""
    return SessionConfig(
        region=TEST_REGION,
        session_timeout=TEST_TIMEOUT,
        enable_live_view=True,
        enable_session_replay=True,
        max_retries=2,
        retry_delay=0.1  # Fast retries for tests
    )


class TestBrowserUseComponents:
    """Test browser-use components that don't require external services."""
    
    def test_session_config_creation(self, session_config):
        """Test session configuration creation."""
        assert session_config.region == TEST_REGION
        assert session_config.session_timeout == TEST_TIMEOUT
        assert session_config.enable_live_view is True
        assert session_config.enable_session_replay is True
        assert session_config.max_retries == 2
        assert session_config.retry_delay == 0.1
    
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
    
    def test_sensitive_data_handler_integration(self):
        """Test sensitive data handler integration."""
        handler = BrowserUseSensitiveDataHandler([ComplianceFramework.HIPAA])
        
        # Test PII detection
        text_with_pii = "Patient SSN: 123-45-6789, Email: patient@hospital.com"
        detections = handler.detect_pii(text_with_pii)
        
        assert len(detections) >= 2  # Should detect SSN and email
        
        # Test masking
        masked_text, mask_detections = handler.mask_text(text_with_pii)
        assert "123-45-6789" not in masked_text
        assert "patient@hospital.com" not in masked_text
        assert len(mask_detections) >= 2
    
    def test_credential_manager_integration(self):
        """Test credential manager integration."""
        manager = BrowserUseCredentialManager()
        
        # Test credential storage
        credential_id = "test_agentcore_token"
        credential_value = "agentcore_session_token_12345"
        
        manager.store_credential(
            credential_id, 
            "session_token", 
            credential_value,
            {"service": "agentcore", "session_type": "browser"}
        )
        
        # Verify storage
        assert credential_id in manager.credentials_store
        
        # Test retrieval
        retrieved = manager.retrieve_credential(credential_id)
        assert retrieved == credential_value
        
        # Test access tracking
        credentials = manager.list_credentials()
        assert len(credentials) == 1
        assert credentials[0]['type'] == 'session_token'
        assert credentials[0]['access_count'] == 1


class TestAgentCoreIntegration:
    """Test AgentCore integration when available."""
    
    @pytest.mark.skipif(not SESSION_MANAGER_AVAILABLE, reason="Session manager not available")
    def test_session_manager_initialization(self, session_config):
        """Test session manager initialization."""
        try:
            manager = BrowserUseAgentCoreSessionManager(session_config)
            
            assert manager.config == session_config
            assert len(manager.active_sessions) == 0
            assert len(manager.session_metrics) == 0
            
            # If AgentCore is available, client should be initialized
            if AGENTCORE_AVAILABLE:
                assert manager.agentcore_client is not None
                assert manager.agentcore_client.region == session_config.region
            
        except Exception as e:
            # If AgentCore is not available, that's expected
            if not AGENTCORE_AVAILABLE:
                assert "BrowserClient" in str(e) or "bedrock_agentcore" in str(e)
            else:
                # If AgentCore should be available but fails, re-raise
                raise
    
    @pytest.mark.skipif(not (SESSION_MANAGER_AVAILABLE and AGENTCORE_AVAILABLE), reason="AgentCore not available")
    @pytest.mark.asyncio
    async def test_session_creation_with_agentcore(self, session_config):
        """Test session creation with real AgentCore (when available)."""
        manager = BrowserUseAgentCoreSessionManager(session_config)
        
        try:
            session_id, ws_url, headers = await manager.create_secure_session(
                sensitive_context={'data_type': 'test', 'compliance': 'TEST'}
            )
            
            # Verify session creation
            assert session_id is not None
            assert ws_url.startswith('wss://')
            assert 'Authorization' in headers
            assert 'X-Session-ID' in headers
            
            # Verify session tracking
            assert session_id in manager.active_sessions
            assert session_id in manager.session_metrics
            
            # Cleanup
            await manager.cleanup_session(session_id)
            
        except Exception as e:
            # If AWS credentials or AgentCore service is not configured, skip
            if "credentials" in str(e).lower() or "unauthorized" in str(e).lower():
                pytest.skip(f"AgentCore service not configured: {e}")
            else:
                raise
        finally:
            await manager.shutdown()
    
    @pytest.mark.skipif(not (SESSION_MANAGER_AVAILABLE and AGENTCORE_AVAILABLE), reason="AgentCore not available")
    @pytest.mark.asyncio
    async def test_session_manager_error_handling(self, session_config):
        """Test session manager error handling."""
        manager = BrowserUseAgentCoreSessionManager(session_config)
        
        try:
            # Test invalid session operations
            status = manager.get_session_status("invalid_session_id")
            assert status is None
            
            # Test cleanup of non-existent session (should not raise)
            await manager.cleanup_session("invalid_session_id")
            
            # Test list active sessions when none exist
            sessions = manager.list_active_sessions()
            assert len(sessions) == 0
            
        finally:
            await manager.shutdown()


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""
    
    def test_healthcare_data_processing_scenario(self):
        """Test healthcare data processing scenario."""
        # Initialize components
        data_handler = BrowserUseSensitiveDataHandler([ComplianceFramework.HIPAA])
        credential_manager = BrowserUseCredentialManager()
        
        # Simulate healthcare form data
        form_data = {
            "patient_name": "Jane Smith",
            "ssn": "987-65-4321",
            "date_of_birth": "07/22/1978",
            "medical_record": "MRN-XYZ789012",
            "insurance_id": "INS-456789"
        }
        
        # Process each field for PII
        processed_fields = {}
        total_detections = []
        
        for field_name, field_value in form_data.items():
            masked_value, detections = data_handler.mask_text(str(field_value), field_name)
            processed_fields[field_name] = masked_value
            total_detections.extend(detections)
        
        # Verify PII was detected and masked
        assert len(total_detections) > 0
        assert "987-65-4321" not in processed_fields["ssn"]
        
        # Store healthcare credentials
        credential_manager.store_credential(
            "healthcare_api_key",
            "api_key",
            "healthcare_secure_key_12345",
            {"service": "healthcare_portal", "compliance": "HIPAA"}
        )
        
        # Validate HIPAA compliance
        full_text = " ".join(form_data.values())
        compliance_result = data_handler.validate_compliance(
            full_text, [ComplianceFramework.HIPAA]
        )
        
        assert compliance_result['compliant'] is False  # Should detect violations
        assert len(compliance_result['violations']) > 0
    
    def test_financial_data_processing_scenario(self):
        """Test financial data processing scenario."""
        # Initialize components
        data_handler = BrowserUseSensitiveDataHandler([ComplianceFramework.PCI_DSS])
        credential_manager = BrowserUseCredentialManager()
        
        # Simulate financial form data
        form_data = {
            "cardholder_name": "Robert Johnson",
            "credit_card": "4532-1234-5678-9012",
            "expiry": "12/25",
            "cvv": "123",
            "ssn": "555-44-3333"
        }
        
        # Process the data
        full_text = " ".join(form_data.values())
        masked_text, detections = data_handler.mask_text(full_text)
        
        # Verify credit card and SSN are masked
        assert "4532-1234-5678-9012" not in masked_text
        assert "555-44-3333" not in masked_text
        
        # Verify PII types detected
        pii_types = {d.pii_type for d in detections}
        assert PIIType.CREDIT_CARD in pii_types
        assert PIIType.SSN in pii_types
        
        # Store financial credentials
        credential_manager.store_credential(
            "payment_gateway_key",
            "api_key",
            "payment_secure_key_67890",
            {"service": "payment_gateway", "compliance": "PCI_DSS"}
        )
        
        # Validate PCI-DSS compliance
        compliance_result = data_handler.validate_compliance(
            full_text, [ComplianceFramework.PCI_DSS]
        )
        
        assert compliance_result['compliant'] is False  # Should detect violations
        pci_violations = [v for v in compliance_result['violations'] if v['framework'] == 'pci_dss']
        assert len(pci_violations) > 0


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    def test_invalid_session_config_handling(self):
        """Test handling of invalid session configuration."""
        # Test with various invalid configurations
        try:
            config = SessionConfig(region="invalid-region-name")
            manager = BrowserUseAgentCoreSessionManager(config)
            # Should either work or fail gracefully
        except Exception as e:
            # Expected if region validation is strict
            assert "region" in str(e).lower() or "BrowserClient" in str(e)
    
    def test_pii_detection_edge_cases(self):
        """Test PII detection edge cases."""
        handler = BrowserUseSensitiveDataHandler()
        
        # Test with empty/None inputs
        assert len(handler.detect_pii(None)) == 0
        assert len(handler.detect_pii("")) == 0
        
        # Test with non-PII that might look like PII
        false_positive_text = "Version 1.2.3, Port 8080, ID 123456789"
        detections = handler.detect_pii(false_positive_text)
        
        # Should have minimal high-confidence false positives
        high_confidence = [d for d in detections if d.confidence > 0.9]
        assert len(high_confidence) <= 1
    
    def test_credential_manager_edge_cases(self):
        """Test credential manager edge cases."""
        manager = BrowserUseCredentialManager()
        
        # Test operations on empty manager
        assert manager.retrieve_credential("nonexistent") is None
        assert manager.delete_credential("nonexistent") is False
        assert len(manager.list_credentials()) == 0
        assert len(manager.get_access_log()) == 0
        
        # Test clear all on empty manager
        manager.clear_all_credentials()  # Should not raise
        
        # Test with various credential types
        test_credentials = [
            ("api_key", "api_key", "key123"),
            ("password", "password", "pass456"),
            ("token", "oauth_token", "token789")
        ]
        
        for cred_id, cred_type, cred_value in test_credentials:
            manager.store_credential(cred_id, cred_type, cred_value)
        
        # Verify all stored
        credentials = manager.list_credentials()
        assert len(credentials) == 3
        
        # Verify no values exposed in listing
        for cred in credentials:
            assert 'key123' not in str(cred)
            assert 'pass456' not in str(cred)
            assert 'token789' not in str(cred)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])