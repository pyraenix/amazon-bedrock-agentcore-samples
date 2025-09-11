"""
Credential Security Tests for LlamaIndex-AgentCore Integration

Focused tests for credential isolation, encryption, and secure handling.
Requirements: 1.3, 2.2
"""

import pytest
import os
import json
import tempfile
import logging
from unittest.mock import Mock, patch, MagicMock
from cryptography.fernet import Fernet
import hashlib
import base64

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'examples'))

from agentcore_session_helpers import SessionManager, CredentialManager
from agentcore_browser_loader import AgentCoreBrowserLoader


class TestCredentialEncryption:
    """Test credential encryption and secure storage."""
    
    def setup_method(self):
        """Set up test environment."""
        self.credential_manager = CredentialManager()
        self.test_credentials = {
            "username": "test_user",
            "password": "super_secret_password_123!",
            "api_key": "sk-1234567890abcdef",
            "oauth_token": "oauth_token_xyz789",
            "session_cookie": "session_abc123def456"
        }
        
    def test_credential_encryption_at_rest(self):
        """Test that credentials are encrypted when stored."""
        encrypted_data = self.credential_manager.encrypt_credentials(self.test_credentials)
        
        # Verify data is encrypted (not plaintext)
        encrypted_str = json.dumps(encrypted_data) if isinstance(encrypted_data, dict) else str(encrypted_data)
        
        for credential_value in self.test_credentials.values():
            assert credential_value not in encrypted_str
            
        # Verify we can decrypt back to original
        decrypted_data = self.credential_manager.decrypt_credentials(encrypted_data)
        assert decrypted_data == self.test_credentials
        
    def test_credential_key_derivation(self):
        """Test that encryption keys are properly derived."""
        # Test with different master keys
        key1 = self.credential_manager._derive_key("master_key_1", "salt_1")
        key2 = self.credential_manager._derive_key("master_key_2", "salt_1")
        key3 = self.credential_manager._derive_key("master_key_1", "salt_2")
        
        # Keys should be different for different inputs
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3
        
        # Same inputs should produce same key
        key1_repeat = self.credential_manager._derive_key("master_key_1", "salt_1")
        assert key1 == key1_repeat
        
    def test_credential_secure_deletion(self):
        """Test that credentials are securely deleted from memory."""
        session_id = "test_session_123"
        
        # Store credentials
        self.credential_manager.store_credentials(session_id, self.test_credentials)
        
        # Verify credentials exist
        retrieved = self.credential_manager.get_credentials(session_id)
        assert retrieved["password"] == "super_secret_password_123!"
        
        # Securely delete credentials
        self.credential_manager.secure_delete_credentials(session_id)
        
        # Verify credentials are gone
        assert self.credential_manager.get_credentials(session_id) is None
        
        # Verify no traces in memory (basic check)
        manager_state = str(self.credential_manager.__dict__)
        assert "super_secret_password_123!" not in manager_state
        
    def test_credential_rotation(self):
        """Test credential rotation functionality."""
        session_id = "rotation_test_session"
        
        # Store initial credentials
        self.credential_manager.store_credentials(session_id, self.test_credentials)
        
        # Rotate credentials
        new_credentials = {
            "username": "test_user",
            "password": "new_rotated_password_456!",
            "api_key": "sk-new-key-fedcba0987654321"
        }
        
        self.credential_manager.rotate_credentials(session_id, new_credentials)
        
        # Verify new credentials are stored
        retrieved = self.credential_manager.get_credentials(session_id)
        assert retrieved["password"] == "new_rotated_password_456!"
        assert retrieved["api_key"] == "sk-new-key-fedcba0987654321"
        
        # Verify old credentials are not accessible
        assert retrieved["password"] != "super_secret_password_123!"


class TestCredentialIsolation:
    """Test credential isolation between sessions and processes."""
    
    def setup_method(self):
        """Set up test environment."""
        self.session_manager = SessionManager()
        self.browser_loader = AgentCoreBrowserLoader()
        
    def test_session_credential_isolation(self):
        """Test that credentials are isolated between different sessions."""
        # Create multiple sessions with different credentials
        sessions = []
        for i in range(3):
            creds = {
                "username": f"user_{i}",
                "password": f"password_{i}_secret",
                "api_key": f"key_{i}_abcdef"
            }
            session_id = self.session_manager.create_session(creds)
            sessions.append((session_id, creds))
            
        # Verify each session has only its own credentials
        for session_id, original_creds in sessions:
            retrieved_creds = self.session_manager.get_session_credentials(session_id)
            
            # Should match original credentials
            assert retrieved_creds["username"] == original_creds["username"]
            assert retrieved_creds["password"] == original_creds["password"]
            
            # Should not contain other sessions' credentials
            for other_session_id, other_creds in sessions:
                if other_session_id != session_id:
                    assert retrieved_creds["username"] != other_creds["username"]
                    assert retrieved_creds["password"] != other_creds["password"]
                    
    def test_process_credential_isolation(self):
        """Test that credentials are isolated between different processes."""
        import multiprocessing
        import queue
        
        def worker_process(session_data, result_queue):
            """Worker process that handles credentials."""
            try:
                session_manager = SessionManager()
                session_id = session_manager.create_session(session_data["credentials"])
                
                # Try to access credentials
                creds = session_manager.get_session_credentials(session_id)
                result_queue.put({
                    "success": True,
                    "username": creds["username"],
                    "session_id": session_id
                })
            except Exception as e:
                result_queue.put({"success": False, "error": str(e)})
                
        # Create separate processes with different credentials
        processes = []
        result_queue = multiprocessing.Queue()
        
        for i in range(2):
            session_data = {
                "credentials": {
                    "username": f"process_user_{i}",
                    "password": f"process_password_{i}"
                }
            }
            
            process = multiprocessing.Process(
                target=worker_process,
                args=(session_data, result_queue)
            )
            processes.append(process)
            process.start()
            
        # Wait for processes and collect results
        results = []
        for process in processes:
            process.join()
            
        # Get results from queue
        while not result_queue.empty():
            results.append(result_queue.get())
            
        # Verify each process handled its own credentials
        assert len(results) == 2
        assert all(result["success"] for result in results)
        
        usernames = [result["username"] for result in results]
        assert "process_user_0" in usernames
        assert "process_user_1" in usernames
        
    def test_memory_isolation_between_loaders(self):
        """Test that different browser loaders don't share credential memory."""
        # Create multiple browser loaders
        loader1 = AgentCoreBrowserLoader(session_id="loader1")
        loader2 = AgentCoreBrowserLoader(session_id="loader2")
        
        # Set different credentials for each
        creds1 = {"username": "loader1_user", "password": "loader1_pass"}
        creds2 = {"username": "loader2_user", "password": "loader2_pass"}
        
        loader1.set_credentials(creds1)
        loader2.set_credentials(creds2)
        
        # Verify credentials are isolated
        retrieved1 = loader1.get_credentials()
        retrieved2 = loader2.get_credentials()
        
        assert retrieved1["username"] == "loader1_user"
        assert retrieved2["username"] == "loader2_user"
        assert retrieved1["password"] != retrieved2["password"]


class TestCredentialLogging:
    """Test that credentials are never exposed in logs."""
    
    def setup_method(self):
        """Set up test environment."""
        self.session_manager = SessionManager()
        self.browser_loader = AgentCoreBrowserLoader()
        self.sensitive_credentials = {
            "password": "very_secret_password_123!",
            "api_key": "sk-super-secret-api-key-xyz",
            "oauth_token": "oauth_very_sensitive_token_abc",
            "private_key": "FAKE_PRIVATE_KEY_FOR_TESTING_PURPOSES_ONLY"
        }
        
    def test_credentials_not_in_debug_logs(self, caplog):
        """Test that credentials don't appear in debug logs."""
        with caplog.at_level(logging.DEBUG):
            # Perform operations that might log credentials
            session_id = self.session_manager.create_session(self.sensitive_credentials)
            self.browser_loader.load_with_credentials(
                url="https://example.com/login",
                credentials=self.sensitive_credentials
            )
            self.session_manager.cleanup_session(session_id)
            
            # Check that no sensitive values appear in logs
            log_text = caplog.text
            
            for credential_value in self.sensitive_credentials.values():
                # Handle multi-line values like private keys
                credential_lines = credential_value.split('\n')
                for line in credential_lines:
                    if line.strip():  # Skip empty lines
                        assert line not in log_text
                        
    def test_credentials_not_in_error_logs(self, caplog):
        """Test that credentials don't appear in error logs."""
        with caplog.at_level(logging.ERROR):
            # Force an error condition
            with patch.object(self.browser_loader, '_authenticate') as mock_auth:
                mock_auth.side_effect = Exception("Authentication failed")
                
                try:
                    self.browser_loader.load_with_credentials(
                        url="https://example.com/login",
                        credentials=self.sensitive_credentials
                    )
                except Exception:
                    pass
                    
            # Verify no credentials in error logs
            log_text = caplog.text
            for credential_value in self.sensitive_credentials.values():
                assert credential_value not in log_text
                
    def test_credential_masking_in_logs(self, caplog):
        """Test that credential keys are masked when they must appear in logs."""
        with caplog.at_level(logging.INFO):
            # Operation that might log credential metadata
            self.session_manager.log_credential_operation("password_update", self.sensitive_credentials)
            
            log_text = caplog.text
            
            # Credential keys might appear but values should be masked
            if "password" in log_text:
                assert "***" in log_text or "[MASKED]" in log_text
                assert "very_secret_password_123!" not in log_text
                
    def test_structured_logging_credential_safety(self):
        """Test that structured logging doesn't expose credentials."""
        import json
        
        # Simulate structured logging
        log_data = {
            "operation": "authentication",
            "user": "test_user",
            "credentials": self.sensitive_credentials,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        # Process through logging system
        safe_log_data = self.session_manager._sanitize_log_data(log_data)
        
        # Verify credentials are sanitized
        log_json = json.dumps(safe_log_data)
        
        for credential_value in self.sensitive_credentials.values():
            assert credential_value not in log_json
            
        # Verify structure is preserved but values are masked
        assert safe_log_data["operation"] == "authentication"
        assert safe_log_data["user"] == "test_user"
        assert "credentials" in safe_log_data
        assert safe_log_data["credentials"]["password"] == "[MASKED]"


class TestCredentialValidation:
    """Test credential validation and security checks."""
    
    def setup_method(self):
        """Set up test environment."""
        self.credential_manager = CredentialManager()
        
    def test_weak_password_detection(self):
        """Test detection of weak passwords."""
        weak_passwords = [
            "123456",
            "password",
            "qwerty",
            "abc123",
            "password123"
        ]
        
        for weak_password in weak_passwords:
            credentials = {"username": "test", "password": weak_password}
            validation_result = self.credential_manager.validate_credentials(credentials)
            
            assert validation_result["valid"] is False
            assert "weak_password" in validation_result["issues"]
            
    def test_strong_password_acceptance(self):
        """Test acceptance of strong passwords."""
        strong_passwords = [
            "MyStr0ng!P@ssw0rd#2024",
            "C0mpl3x&S3cur3*P@ss!",
            "Ungu3ss@bl3#P@ssw0rd$2024"
        ]
        
        for strong_password in strong_passwords:
            credentials = {"username": "test", "password": strong_password}
            validation_result = self.credential_manager.validate_credentials(credentials)
            
            assert validation_result["valid"] is True
            assert "weak_password" not in validation_result.get("issues", [])
            
    def test_credential_format_validation(self):
        """Test validation of credential formats."""
        # Test API key format
        invalid_api_keys = [
            "invalid-key",
            "sk-",
            "sk-tooshort",
            ""
        ]
        
        for invalid_key in invalid_api_keys:
            credentials = {"api_key": invalid_key}
            validation_result = self.credential_manager.validate_credentials(credentials)
            
            assert validation_result["valid"] is False
            assert "invalid_api_key_format" in validation_result["issues"]
            
        # Test valid API key
        valid_credentials = {"api_key": "sk-1234567890abcdef1234567890abcdef"}
        validation_result = self.credential_manager.validate_credentials(valid_credentials)
        assert validation_result["valid"] is True
        
    def test_credential_expiry_validation(self):
        """Test validation of credential expiry."""
        from datetime import datetime, timedelta
        
        # Expired credentials
        expired_credentials = {
            "username": "test",
            "password": "valid_password",
            "expires_at": (datetime.now() - timedelta(days=1)).isoformat()
        }
        
        validation_result = self.credential_manager.validate_credentials(expired_credentials)
        assert validation_result["valid"] is False
        assert "expired_credentials" in validation_result["issues"]
        
        # Valid credentials
        valid_credentials = {
            "username": "test",
            "password": "valid_password",
            "expires_at": (datetime.now() + timedelta(days=30)).isoformat()
        }
        
        validation_result = self.credential_manager.validate_credentials(valid_credentials)
        assert validation_result["valid"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])