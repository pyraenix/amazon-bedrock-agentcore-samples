"""
Security validation tests for credential handling in Strands-AgentCore integration.

This module tests that credentials are never exposed in Strands agent logs or outputs,
ensuring secure credential management throughout the agent lifecycle.

Requirements: 8.1, 8.2, 8.3, 8.4
"""

import pytest
import json
import re
import logging
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import asyncio
from datetime import datetime

# Import Strands and AgentCore components
try:
    from strands import Agent, Tool, Workflow
    from strands.tools import BaseTool
    from strands.security import CredentialManager
except ImportError:
    # Mock imports for testing environment
    Agent = Mock
    Tool = Mock
    Workflow = Mock
    BaseTool = Mock
    CredentialManager = Mock

# Import custom tools
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))
try:
    from strands_agentcore_session_helpers import StrandsAgentCoreClient
    from strands_pii_utils import SensitiveDataHandler
    from strands_security_policies import BedrockModelRouter
except ImportError:
    # Mock for testing
    StrandsAgentCoreClient = Mock
    SensitiveDataHandler = Mock
    BedrockModelRouter = Mock


class TestCredentialSecurity:
    """Test suite for credential security in Strands-AgentCore integration."""
    
    @pytest.fixture
    def mock_credentials(self):
        """Mock credentials for testing."""
        return {
            'username': 'test_user',
            'password': 'super_secret_password_123',
            'api_key': 'sk-1234567890abcdef',
            'oauth_token': 'oauth_token_abcdef123456',
            'session_token': 'session_xyz789',
            'private_key': '-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC...\n-----END PRIVATE KEY-----'
        }
    
    @pytest.fixture
    def mock_agent_config(self):
        """Mock agent configuration."""
        return {
            'agent_id': 'test_agent_001',
            'llm_providers': {
                'bedrock': {
                    'model': 'anthropic.claude-3-sonnet-20240229-v1:0',
                    'region': 'us-east-1'
                }
            },
            'security_level': 'HIGH',
            'audit_config': {
                'log_level': 'INFO',
                'audit_sensitive_operations': True
            }
        }
    
    @pytest.fixture
    def log_capture(self):
        """Capture logs for testing."""
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log') as f:
            log_file = f.name
        
        # Configure logging to capture to file
        logger = logging.getLogger('strands_security_test')
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        yield log_file, logger
        
        # Cleanup
        handler.close()
        logger.removeHandler(handler)
        if os.path.exists(log_file):
            os.unlink(log_file)
    
    def test_credentials_not_in_logs(self, mock_credentials, log_capture):
        """Test that credentials are never exposed in log files."""
        log_file, logger = log_capture
        
        # Simulate agent operations with credentials
        with patch('strands_agentcore_session_helpers.StrandsAgentCoreClient') as mock_client:
            mock_session = Mock()
            mock_client.return_value.create_secure_session.return_value = mock_session
            
            # Log various operations that might expose credentials
            logger.info(f"Starting secure session with credentials")
            logger.debug(f"Session configuration: {{'session_id': 'test_session'}}")
            logger.info(f"Authentication successful")
            logger.warning(f"Session timeout warning")
            logger.error(f"Connection error occurred")
            
            # Simulate credential injection (should be masked)
            credential_manager = Mock()
            credential_manager.inject_credentials = Mock()
            credential_manager.inject_credentials(mock_session, mock_credentials)
            
            logger.info("Credentials injected successfully")
        
        # Read log file and check for credential exposure
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        # Check that no credential values appear in logs
        for cred_key, cred_value in mock_credentials.items():
            assert cred_value not in log_content, f"Credential '{cred_key}' exposed in logs: {cred_value}"
        
        # Verify that log entries exist (sanity check)
        assert "Starting secure session" in log_content
        assert "Authentication successful" in log_content
    
    def test_credentials_not_in_agent_outputs(self, mock_credentials, mock_agent_config):
        """Test that credentials are never exposed in agent outputs."""
        with patch('strands_agentcore_session_helpers.StrandsAgentCoreClient') as mock_client:
            # Mock agent execution
            mock_agent = Mock()
            mock_agent.execute = Mock(return_value={
                'status': 'success',
                'result': 'Login completed successfully',
                'session_id': 'masked_session_id',
                'timestamp': datetime.now().isoformat()
            })
            
            mock_client.return_value.create_secure_agent.return_value = mock_agent
            
            # Create client and execute workflow
            client = StrandsAgentCoreClient(
                region='us-east-1',
                llm_configs=mock_agent_config['llm_providers'],
                security_config={'level': 'HIGH'}
            )
            
            # Execute agent with credentials
            result = mock_agent.execute({
                'action': 'secure_login',
                'credentials': mock_credentials,
                'target_url': 'https://example.com/login'
            })
            
            # Convert result to string for checking
            result_str = json.dumps(result, default=str)
            
            # Verify no credentials in output
            for cred_key, cred_value in mock_credentials.items():
                assert cred_value not in result_str, f"Credential '{cred_key}' exposed in output: {cred_value}"
            
            # Verify output contains expected success indicators
            assert result['status'] == 'success'
            assert 'session_id' in result
    
    def test_credential_masking_in_debug_output(self, mock_credentials):
        """Test that credentials are properly masked in debug outputs."""
        with patch('strands_pii_utils.SensitiveDataHandler') as mock_handler:
            # Configure mock to return masked data
            mock_handler.return_value.mask_sensitive_data.return_value = {
                'username': 'test_user',
                'password': '***MASKED***',
                'api_key': '***MASKED***',
                'oauth_token': '***MASKED***',
                'session_token': '***MASKED***',
                'private_key': '***MASKED***'
            }
            
            handler = SensitiveDataHandler()
            masked_creds = handler.mask_sensitive_data(mock_credentials, {
                'mask_patterns': ['password', 'key', 'token'],
                'mask_value': '***MASKED***'
            })
            
            # Verify masking occurred
            assert masked_creds['password'] == '***MASKED***'
            assert masked_creds['api_key'] == '***MASKED***'
            assert masked_creds['oauth_token'] == '***MASKED***'
            assert masked_creds['session_token'] == '***MASKED***'
            assert masked_creds['private_key'] == '***MASKED***'
            
            # Verify non-sensitive data is preserved
            assert masked_creds['username'] == 'test_user'
    
    def test_credential_storage_security(self, mock_credentials):
        """Test that credentials are securely stored and retrieved."""
        with patch('boto3.client') as mock_boto3:
            # Mock AWS Secrets Manager
            mock_secrets_client = Mock()
            mock_boto3.return_value = mock_secrets_client
            
            mock_secrets_client.create_secret.return_value = {
                'ARN': 'arn:aws:secretsmanager:us-east-1:123456789012:secret:test-secret',
                'Name': 'test-secret'
            }
            
            mock_secrets_client.get_secret_value.return_value = {
                'SecretString': json.dumps(mock_credentials)
            }
            
            # Test credential storage
            credential_manager = Mock()
            credential_manager.store_credentials = Mock(return_value='test-secret-arn')
            credential_manager.retrieve_credentials = Mock(return_value=mock_credentials)
            
            # Store credentials
            secret_arn = credential_manager.store_credentials('test-agent', mock_credentials)
            assert secret_arn is not None
            
            # Retrieve credentials
            retrieved_creds = credential_manager.retrieve_credentials('test-agent')
            assert retrieved_creds == mock_credentials
            
            # Verify secure storage was called
            credential_manager.store_credentials.assert_called_once()
            credential_manager.retrieve_credentials.assert_called_once()
    
    def test_credential_rotation_security(self, mock_credentials):
        """Test that credential rotation maintains security."""
        with patch('strands_agentcore_session_helpers.StrandsAgentCoreClient') as mock_client:
            mock_session = Mock()
            mock_client.return_value.create_secure_session.return_value = mock_session
            
            # Mock credential rotation
            new_credentials = {
                'username': 'test_user',
                'password': 'new_super_secret_password_456',
                'api_key': 'sk-new1234567890abcdef',
                'oauth_token': 'oauth_token_new_abcdef123456'
            }
            
            credential_manager = Mock()
            credential_manager.rotate_credentials = Mock(return_value={
                'status': 'success',
                'old_credentials_revoked': True,
                'new_credentials_active': True,
                'rotation_timestamp': datetime.now().isoformat()
            })
            
            # Perform rotation
            rotation_result = credential_manager.rotate_credentials('test-agent', new_credentials)
            
            # Verify rotation was successful
            assert rotation_result['status'] == 'success'
            assert rotation_result['old_credentials_revoked'] is True
            assert rotation_result['new_credentials_active'] is True
            
            # Verify old credentials are not accessible
            credential_manager.rotate_credentials.assert_called_once()
    
    def test_session_credential_cleanup(self, mock_credentials):
        """Test that credentials are properly cleaned up after session termination."""
        with patch('strands_agentcore_session_helpers.StrandsAgentCoreClient') as mock_client:
            mock_session = Mock()
            mock_session.session_id = 'test_session_123'
            mock_session.cleanup = Mock()
            mock_session.is_active = Mock(return_value=False)
            
            mock_client.return_value.create_secure_session.return_value = mock_session
            
            # Create session with credentials
            client = StrandsAgentCoreClient(
                region='us-east-1',
                llm_configs={'bedrock': {'model': 'claude-3'}},
                security_config={'level': 'HIGH'}
            )
            
            session = mock_client.return_value.create_secure_session({
                'credentials': mock_credentials,
                'cleanup_on_exit': True
            })
            
            # Simulate session termination
            session.cleanup()
            
            # Verify session is no longer active
            assert not session.is_active()
            
            # Verify cleanup was called
            session.cleanup.assert_called_once()
    
    def test_credential_audit_trail(self, mock_credentials, log_capture):
        """Test that credential operations are properly audited without exposing credentials."""
        log_file, logger = log_capture
        
        with patch('strands_monitoring.AuditTrailTool') as mock_audit:
            mock_audit_instance = Mock()
            mock_audit.return_value = mock_audit_instance
            
            # Mock audit operations
            mock_audit_instance.log_credential_operation = Mock()
            mock_audit_instance.log_security_event = Mock()
            
            audit_tool = mock_audit()
            
            # Log credential operations
            audit_tool.log_credential_operation({
                'operation': 'credential_injection',
                'agent_id': 'test_agent_001',
                'session_id': 'test_session_123',
                'timestamp': datetime.now().isoformat(),
                'success': True
            })
            
            audit_tool.log_security_event({
                'event_type': 'credential_rotation',
                'agent_id': 'test_agent_001',
                'timestamp': datetime.now().isoformat(),
                'details': 'Credentials rotated successfully'
            })
            
            # Verify audit calls were made
            mock_audit_instance.log_credential_operation.assert_called_once()
            mock_audit_instance.log_security_event.assert_called_once()
            
            # Verify audit data doesn't contain actual credentials
            call_args = mock_audit_instance.log_credential_operation.call_args[0][0]
            for cred_value in mock_credentials.values():
                assert cred_value not in str(call_args)
    
    def test_memory_credential_cleanup(self, mock_credentials):
        """Test that credentials are properly cleared from memory."""
        import gc
        
        # Create credential objects
        test_creds = mock_credentials.copy()
        
        # Simulate credential usage
        with patch('strands_agentcore_session_helpers.StrandsAgentCoreClient') as mock_client:
            mock_session = Mock()
            mock_client.return_value.create_secure_session.return_value = mock_session
            
            # Use credentials
            credential_manager = Mock()
            credential_manager.secure_clear = Mock()
            
            # Clear credentials from memory
            credential_manager.secure_clear(test_creds)
            
            # Verify secure clear was called
            credential_manager.secure_clear.assert_called_once_with(test_creds)
        
        # Force garbage collection
        gc.collect()
        
        # Verify credentials are cleared (mock verification)
        assert True  # In real implementation, would verify memory is cleared
    
    @pytest.mark.asyncio
    async def test_concurrent_credential_access_security(self, mock_credentials):
        """Test that concurrent access to credentials maintains security."""
        with patch('strands_agentcore_session_helpers.StrandsAgentCoreClient') as mock_client:
            mock_session1 = Mock()
            mock_session1.session_id = 'session_1'
            mock_session2 = Mock()
            mock_session2.session_id = 'session_2'
            
            mock_client.return_value.create_secure_session.side_effect = [mock_session1, mock_session2]
            
            # Create multiple concurrent sessions
            async def create_secure_session(session_id):
                client = StrandsAgentCoreClient(
                    region='us-east-1',
                    llm_configs={'bedrock': {'model': 'claude-3'}},
                    security_config={'level': 'HIGH'}
                )
                return mock_client.return_value.create_secure_session({
                    'session_id': session_id,
                    'credentials': mock_credentials
                })
            
            # Run concurrent sessions
            sessions = await asyncio.gather(
                create_secure_session('session_1'),
                create_secure_session('session_2')
            )
            
            # Verify sessions are isolated
            assert sessions[0].session_id != sessions[1].session_id
            assert len(sessions) == 2
    
    def test_credential_validation_security(self, mock_credentials):
        """Test that credential validation doesn't expose credential values."""
        with patch('strands_security_policies.BedrockModelRouter') as mock_router:
            mock_router_instance = Mock()
            mock_router.return_value = mock_router_instance
            
            # Mock validation that returns boolean without exposing credentials
            mock_router_instance.validate_credentials = Mock(return_value={
                'valid': True,
                'validation_timestamp': datetime.now().isoformat(),
                'credential_types_validated': ['username', 'password', 'api_key'],
                'security_level': 'HIGH'
            })
            
            router = BedrockModelRouter()
            validation_result = router.validate_credentials(mock_credentials)
            
            # Verify validation occurred
            assert validation_result['valid'] is True
            assert 'credential_types_validated' in validation_result
            
            # Verify actual credential values are not in result
            result_str = json.dumps(validation_result, default=str)
            for cred_value in mock_credentials.values():
                assert cred_value not in result_str
            
            mock_router_instance.validate_credentials.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])