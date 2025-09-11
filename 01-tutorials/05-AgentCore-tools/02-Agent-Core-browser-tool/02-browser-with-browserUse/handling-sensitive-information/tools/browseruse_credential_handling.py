"""
Browser-Use Credential Handling

This module provides secure credential input functions for browser-use Agent operations,
implements credential isolation within AgentCore's micro-VM environment, and creates
credential validation and audit trail for browser-use sensitive operations.

Integrates with AgentCore Browser Tool's security features for enterprise-grade credential management.
"""

import os
import json
import logging
import asyncio
import hashlib
import secrets
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import base64
from pathlib import Path

# Import from the same directory
import sys
import os
sys.path.append(os.path.dirname(__file__))

from browseruse_sensitive_data_handler import BrowserUseCredentialManager


class CredentialType(Enum):
    """Types of credentials that can be handled."""
    PASSWORD = "password"
    API_KEY = "api_key"
    TOKEN = "token"
    CERTIFICATE = "certificate"
    SSH_KEY = "ssh_key"
    DATABASE_CONNECTION = "database_connection"
    OAUTH_TOKEN = "oauth_token"
    SESSION_COOKIE = "session_cookie"
    BIOMETRIC = "biometric"
    MFA_CODE = "mfa_code"


class CredentialSecurityLevel(Enum):
    """Security levels for credential handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CredentialScope(Enum):
    """Scope of credential usage."""
    SESSION = "session"  # Valid for current session only
    TEMPORARY = "temporary"  # Valid for limited time
    PERSISTENT = "persistent"  # Stored for reuse
    SHARED = "shared"  # Shared across sessions


@dataclass
class CredentialPolicy:
    """Policy for credential handling."""
    credential_type: CredentialType
    security_level: CredentialSecurityLevel
    scope: CredentialScope
    max_age_minutes: int = 60
    require_encryption: bool = True
    require_audit: bool = True
    allow_caching: bool = False
    require_mfa: bool = False
    isolation_required: bool = True


@dataclass
class CredentialMetadata:
    """Metadata for stored credentials."""
    credential_id: str
    credential_type: CredentialType
    security_level: CredentialSecurityLevel
    scope: CredentialScope
    created_at: datetime
    expires_at: Optional[datetime]
    last_accessed: Optional[datetime]
    access_count: int = 0
    source_session: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class CredentialAccessLog:
    """Log entry for credential access."""
    credential_id: str
    access_type: str  # "create", "read", "update", "delete"
    timestamp: datetime
    session_id: Optional[str]
    user_agent: Optional[str]
    ip_address: Optional[str]
    success: bool
    error_message: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)


class BrowserUseCredentialHandler:
    """
    Secure credential handling for browser-use Agent operations.
    
    Provides secure credential input, storage, and management within
    AgentCore's micro-VM isolated environment with comprehensive
    audit trails and validation.
    """
    
    # Default credential policies
    DEFAULT_POLICIES = {
        CredentialType.PASSWORD: CredentialPolicy(
            credential_type=CredentialType.PASSWORD,
            security_level=CredentialSecurityLevel.HIGH,
            scope=CredentialScope.SESSION,
            max_age_minutes=30,
            require_encryption=True,
            require_audit=True,
            allow_caching=False,
            require_mfa=True,
            isolation_required=True
        ),
        CredentialType.API_KEY: CredentialPolicy(
            credential_type=CredentialType.API_KEY,
            security_level=CredentialSecurityLevel.CRITICAL,
            scope=CredentialScope.TEMPORARY,
            max_age_minutes=60,
            require_encryption=True,
            require_audit=True,
            allow_caching=False,
            require_mfa=False,
            isolation_required=True
        ),
        CredentialType.TOKEN: CredentialPolicy(
            credential_type=CredentialType.TOKEN,
            security_level=CredentialSecurityLevel.HIGH,
            scope=CredentialScope.SESSION,
            max_age_minutes=15,
            require_encryption=True,
            require_audit=True,
            allow_caching=False,
            require_mfa=False,
            isolation_required=True
        ),
        CredentialType.MFA_CODE: CredentialPolicy(
            credential_type=CredentialType.MFA_CODE,
            security_level=CredentialSecurityLevel.CRITICAL,
            scope=CredentialScope.SESSION,
            max_age_minutes=5,
            require_encryption=True,
            require_audit=True,
            allow_caching=False,
            require_mfa=False,
            isolation_required=True
        )
    }
    
    def __init__(self, 
                 session_id: Optional[str] = None,
                 agentcore_session_config: Optional[Dict[str, Any]] = None,
                 custom_policies: Optional[Dict[CredentialType, CredentialPolicy]] = None):
        """
        Initialize the credential handler.
        
        Args:
            session_id: AgentCore session ID for isolation
            agentcore_session_config: AgentCore session configuration
            custom_policies: Custom credential policies
        """
        self.logger = logging.getLogger(__name__)
        self.session_id = session_id or self._generate_session_id()
        self.agentcore_config = agentcore_session_config or {}
        
        # Initialize credential manager
        self.credential_manager = BrowserUseCredentialManager()
        
        # Credential policies
        self.policies = self.DEFAULT_POLICIES.copy()
        if custom_policies:
            self.policies.update(custom_policies)
        
        # Credential metadata and audit logs
        self.credential_metadata: Dict[str, CredentialMetadata] = {}
        self.access_logs: List[CredentialAccessLog] = []
        
        # Security features
        self.isolation_enabled = True
        self.audit_enabled = True
        self.encryption_enabled = True
        
        self.logger.info(f"Initialized credential handler for session: {self.session_id}")
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"browseruse-{secrets.token_hex(16)}"
    
    async def secure_credential_input(self, 
                                    credential_type: CredentialType,
                                    prompt: str = "Enter credential:",
                                    validation_callback: Optional[Callable] = None) -> Optional[str]:
        """
        Securely collect credential input from user.
        
        Args:
            credential_type: Type of credential being collected
            prompt: Prompt message for user
            validation_callback: Optional validation function
            
        Returns:
            Credential ID if successful, None if failed
        """
        policy = self.policies.get(credential_type, self.DEFAULT_POLICIES[CredentialType.PASSWORD])
        
        try:
            # Log credential input attempt
            self._log_access("input_attempt", None, success=True, 
                           additional_context={'credential_type': credential_type.value})
            
            # In a real implementation, this would use secure input methods
            # For demo purposes, we'll simulate secure input
            credential_value = await self._simulate_secure_input(prompt, credential_type)
            
            if not credential_value:
                self._log_access("input_failed", None, success=False, 
                               error_message="No credential provided")
                return None
            
            # Validate credential if callback provided
            if validation_callback:
                is_valid = await validation_callback(credential_value)
                if not is_valid:
                    self._log_access("validation_failed", None, success=False,
                                   error_message="Credential validation failed")
                    return None
            
            # Store credential securely
            credential_id = await self.store_credential(
                credential_type=credential_type,
                credential_value=credential_value,
                metadata={'input_method': 'secure_input', 'validated': validation_callback is not None}
            )
            
            self.logger.info(f"Secure credential input completed: {credential_id}")
            return credential_id
            
        except Exception as e:
            self.logger.error(f"Secure credential input failed: {e}")
            self._log_access("input_error", None, success=False, error_message=str(e))
            return None
    
    async def _simulate_secure_input(self, prompt: str, credential_type: CredentialType) -> Optional[str]:
        """
        Simulate secure credential input.
        
        In a real implementation, this would use:
        - Secure input dialogs
        - Hardware security modules
        - Biometric authentication
        - MFA verification
        """
        # Simulate different credential types
        if credential_type == CredentialType.PASSWORD:
            return "secure_password_123!"
        elif credential_type == CredentialType.API_KEY:
            return f"api_key_{secrets.token_hex(16)}"
        elif credential_type == CredentialType.TOKEN:
            return f"token_{secrets.token_hex(20)}"
        elif credential_type == CredentialType.MFA_CODE:
            return "123456"  # Simulated MFA code
        else:
            return f"credential_{secrets.token_hex(12)}"
    
    async def store_credential(self, 
                             credential_type: CredentialType,
                             credential_value: str,
                             metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a credential securely with proper isolation.
        
        Args:
            credential_type: Type of credential
            credential_value: Credential value to store
            metadata: Additional metadata
            
        Returns:
            Credential ID
        """
        policy = self.policies.get(credential_type, self.DEFAULT_POLICIES[CredentialType.PASSWORD])
        credential_id = f"{credential_type.value}_{secrets.token_hex(8)}"
        
        try:
            # Store in credential manager
            self.credential_manager.store_credential(
                credential_id=credential_id,
                credential_type=credential_type.value,
                value=credential_value,
                metadata=metadata
            )
            
            # Create metadata entry
            expires_at = None
            if policy.max_age_minutes >= 0:  # Include 0 for immediate expiration
                expires_at = datetime.now() + timedelta(minutes=policy.max_age_minutes)
            
            credential_metadata = CredentialMetadata(
                credential_id=credential_id,
                credential_type=credential_type,
                security_level=policy.security_level,
                scope=policy.scope,
                created_at=datetime.now(),
                expires_at=expires_at,
                last_accessed=None,
                access_count=0,
                source_session=self.session_id,
                tags=metadata.get('tags', []) if metadata else []
            )
            
            self.credential_metadata[credential_id] = credential_metadata
            
            # Log credential storage
            self._log_access("store", credential_id, success=True,
                           additional_context={'credential_type': credential_type.value})
            
            self.logger.info(f"Credential stored: {credential_id}")
            return credential_id
            
        except Exception as e:
            self.logger.error(f"Failed to store credential: {e}")
            self._log_access("store_error", credential_id, success=False, error_message=str(e))
            raise
    
    async def retrieve_credential(self, credential_id: str) -> Optional[str]:
        """
        Retrieve a stored credential with security checks.
        
        Args:
            credential_id: ID of credential to retrieve
            
        Returns:
            Credential value if authorized, None otherwise
        """
        try:
            # Check if credential exists
            if credential_id not in self.credential_metadata:
                self._log_access("retrieve_not_found", credential_id, success=False,
                               error_message="Credential not found")
                return None
            
            metadata = self.credential_metadata[credential_id]
            
            # Check expiration - if max_age_minutes is 0, it expires immediately
            current_time = datetime.now()
            if metadata.expires_at and current_time >= metadata.expires_at:
                self._log_access("retrieve_expired", credential_id, success=False,
                               error_message="Credential expired")
                await self.delete_credential(credential_id)
                return None
            
            # Check session isolation
            if (metadata.scope == CredentialScope.SESSION and 
                metadata.source_session != self.session_id):
                self._log_access("retrieve_isolation_violation", credential_id, success=False,
                               error_message="Session isolation violation")
                return None
            
            # Retrieve from credential manager
            credential_value = self.credential_manager.retrieve_credential(credential_id)
            
            if credential_value:
                # Update access tracking
                metadata.last_accessed = datetime.now()
                metadata.access_count += 1
                
                # Log successful access
                self._log_access("retrieve", credential_id, success=True)
                
                self.logger.info(f"Credential retrieved: {credential_id}")
                return credential_value
            else:
                self._log_access("retrieve_failed", credential_id, success=False,
                               error_message="Failed to decrypt credential")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve credential {credential_id}: {e}")
            self._log_access("retrieve_error", credential_id, success=False, error_message=str(e))
            return None
    
    async def delete_credential(self, credential_id: str) -> bool:
        """
        Delete a stored credential.
        
        Args:
            credential_id: ID of credential to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            # Delete from credential manager
            deleted = self.credential_manager.delete_credential(credential_id)
            
            # Remove metadata
            if credential_id in self.credential_metadata:
                del self.credential_metadata[credential_id]
            
            # Log deletion
            self._log_access("delete", credential_id, success=deleted)
            
            if deleted:
                self.logger.info(f"Credential deleted: {credential_id}")
            else:
                self.logger.warning(f"Credential not found for deletion: {credential_id}")
            
            return deleted
            
        except Exception as e:
            self.logger.error(f"Failed to delete credential {credential_id}: {e}")
            self._log_access("delete_error", credential_id, success=False, error_message=str(e))
            return False
    
    async def validate_credential_access(self, 
                                       credential_id: str,
                                       required_security_level: CredentialSecurityLevel = CredentialSecurityLevel.MEDIUM) -> bool:
        """
        Validate if credential access is authorized.
        
        Args:
            credential_id: ID of credential to validate
            required_security_level: Minimum required security level
            
        Returns:
            True if access is authorized
        """
        if credential_id not in self.credential_metadata:
            return False
        
        metadata = self.credential_metadata[credential_id]
        
        # Check security level
        security_levels = {
            CredentialSecurityLevel.LOW: 0,
            CredentialSecurityLevel.MEDIUM: 1,
            CredentialSecurityLevel.HIGH: 2,
            CredentialSecurityLevel.CRITICAL: 3
        }
        
        if security_levels[metadata.security_level] < security_levels[required_security_level]:
            return False
        
        # Check expiration
        if metadata.expires_at and datetime.now() > metadata.expires_at:
            return False
        
        # Check session isolation
        if (metadata.scope == CredentialScope.SESSION and 
            metadata.source_session != self.session_id):
            return False
        
        return True
    
    def _log_access(self, 
                   access_type: str, 
                   credential_id: Optional[str],
                   success: bool,
                   error_message: Optional[str] = None,
                   additional_context: Optional[Dict[str, Any]] = None) -> None:
        """Log credential access for audit trail."""
        if not self.audit_enabled:
            return
        
        log_entry = CredentialAccessLog(
            credential_id=credential_id or "unknown",
            access_type=access_type,
            timestamp=datetime.now(),
            session_id=self.session_id,
            user_agent=None,  # Would be populated in real implementation
            ip_address=None,  # Would be populated in real implementation
            success=success,
            error_message=error_message,
            additional_context=additional_context or {}
        )
        
        self.access_logs.append(log_entry)
    
    def get_credential_metadata(self, credential_id: str) -> Optional[CredentialMetadata]:
        """Get metadata for a credential."""
        return self.credential_metadata.get(credential_id)
    
    def list_credentials(self, 
                        credential_type: Optional[CredentialType] = None,
                        security_level: Optional[CredentialSecurityLevel] = None) -> List[CredentialMetadata]:
        """
        List stored credentials with optional filtering.
        
        Args:
            credential_type: Filter by credential type
            security_level: Filter by security level
            
        Returns:
            List of credential metadata
        """
        credentials = list(self.credential_metadata.values())
        
        if credential_type:
            credentials = [c for c in credentials if c.credential_type == credential_type]
        
        if security_level:
            credentials = [c for c in credentials if c.security_level == security_level]
        
        return credentials
    
    def get_access_logs(self, 
                       credential_id: Optional[str] = None,
                       access_type: Optional[str] = None,
                       since: Optional[datetime] = None) -> List[CredentialAccessLog]:
        """
        Get access logs with optional filtering.
        
        Args:
            credential_id: Filter by credential ID
            access_type: Filter by access type
            since: Filter by timestamp
            
        Returns:
            List of access log entries
        """
        logs = self.access_logs.copy()
        
        if credential_id:
            logs = [log for log in logs if log.credential_id == credential_id]
        
        if access_type:
            logs = [log for log in logs if log.access_type == access_type]
        
        if since:
            logs = [log for log in logs if log.timestamp >= since]
        
        return logs
    
    async def cleanup_expired_credentials(self) -> int:
        """
        Clean up expired credentials.
        
        Returns:
            Number of credentials cleaned up
        """
        cleaned_count = 0
        current_time = datetime.now()
        
        expired_credentials = []
        for credential_id, metadata in self.credential_metadata.items():
            if metadata.expires_at and current_time > metadata.expires_at:
                expired_credentials.append(credential_id)
        
        for credential_id in expired_credentials:
            if await self.delete_credential(credential_id):
                cleaned_count += 1
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} expired credentials")
        
        return cleaned_count
    
    async def emergency_cleanup(self) -> None:
        """Emergency cleanup of all credentials."""
        self.logger.warning("Emergency credential cleanup initiated")
        
        # Clear all credentials
        self.credential_manager.clear_all_credentials()
        self.credential_metadata.clear()
        
        # Log emergency cleanup
        self._log_access("emergency_cleanup", None, success=True,
                       additional_context={'reason': 'emergency_cleanup'})
        
        self.logger.warning("Emergency credential cleanup completed")
    
    def generate_audit_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive audit report.
        
        Returns:
            Audit report with statistics and logs
        """
        current_time = datetime.now()
        
        # Credential statistics
        total_credentials = len(self.credential_metadata)
        active_credentials = sum(1 for m in self.credential_metadata.values() 
                               if not m.expires_at or m.expires_at > current_time)
        expired_credentials = total_credentials - active_credentials
        
        # Access statistics
        total_accesses = len(self.access_logs)
        successful_accesses = sum(1 for log in self.access_logs if log.success)
        failed_accesses = total_accesses - successful_accesses
        
        # Security level distribution
        security_distribution = {}
        for metadata in self.credential_metadata.values():
            level = metadata.security_level.value
            security_distribution[level] = security_distribution.get(level, 0) + 1
        
        return {
            'report_timestamp': current_time.isoformat(),
            'session_id': self.session_id,
            'credential_statistics': {
                'total_credentials': total_credentials,
                'active_credentials': active_credentials,
                'expired_credentials': expired_credentials,
                'security_level_distribution': security_distribution
            },
            'access_statistics': {
                'total_accesses': total_accesses,
                'successful_accesses': successful_accesses,
                'failed_accesses': failed_accesses,
                'success_rate': successful_accesses / total_accesses if total_accesses > 0 else 0
            },
            'recent_access_logs': [
                {
                    'credential_id': log.credential_id,
                    'access_type': log.access_type,
                    'timestamp': log.timestamp.isoformat(),
                    'success': log.success,
                    'error_message': log.error_message
                }
                for log in self.access_logs[-10:]  # Last 10 entries
            ]
        }


class BrowserUseCredentialIsolation:
    """
    Credential isolation utilities for AgentCore micro-VM environment.
    
    Provides session-level isolation, secure credential boundaries,
    and integration with AgentCore's security features.
    """
    
    def __init__(self, agentcore_session_id: str):
        """
        Initialize credential isolation.
        
        Args:
            agentcore_session_id: AgentCore session ID for isolation
        """
        self.logger = logging.getLogger(__name__)
        self.agentcore_session_id = agentcore_session_id
        self.isolation_boundaries: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info(f"Initialized credential isolation for session: {agentcore_session_id}")
    
    def create_isolation_boundary(self, 
                                boundary_id: str,
                                security_level: CredentialSecurityLevel,
                                allowed_operations: List[str]) -> Dict[str, Any]:
        """
        Create an isolation boundary for credentials.
        
        Args:
            boundary_id: Unique identifier for the boundary
            security_level: Security level for the boundary
            allowed_operations: List of allowed operations
            
        Returns:
            Boundary configuration
        """
        boundary_config = {
            'boundary_id': boundary_id,
            'agentcore_session_id': self.agentcore_session_id,
            'security_level': security_level,
            'allowed_operations': allowed_operations,
            'created_at': datetime.now(),
            'active': True,
            'credential_count': 0
        }
        
        self.isolation_boundaries[boundary_id] = boundary_config
        
        self.logger.info(f"Created isolation boundary: {boundary_id}")
        return boundary_config
    
    def validate_boundary_access(self, 
                                boundary_id: str,
                                operation: str,
                                credential_security_level: CredentialSecurityLevel) -> bool:
        """
        Validate access to an isolation boundary.
        
        Args:
            boundary_id: Boundary to validate
            operation: Operation being attempted
            credential_security_level: Security level of credential
            
        Returns:
            True if access is allowed
        """
        if boundary_id not in self.isolation_boundaries:
            return False
        
        boundary = self.isolation_boundaries[boundary_id]
        
        # Check if boundary is active
        if not boundary['active']:
            return False
        
        # Check allowed operations
        if operation not in boundary['allowed_operations']:
            return False
        
        # Check security level compatibility
        security_levels = {
            CredentialSecurityLevel.LOW: 0,
            CredentialSecurityLevel.MEDIUM: 1,
            CredentialSecurityLevel.HIGH: 2,
            CredentialSecurityLevel.CRITICAL: 3
        }
        
        boundary_level = security_levels[boundary['security_level']]
        credential_level = security_levels[credential_security_level]
        
        return credential_level <= boundary_level
    
    def get_isolation_status(self) -> Dict[str, Any]:
        """Get current isolation status."""
        return {
            'agentcore_session_id': self.agentcore_session_id,
            'active_boundaries': len([b for b in self.isolation_boundaries.values() if b['active']]),
            'total_boundaries': len(self.isolation_boundaries),
            'boundaries': list(self.isolation_boundaries.keys())
        }


# Convenience functions for browser-use credential handling
async def secure_browser_login(username: str, 
                             password_prompt: str = "Enter password:",
                             session_id: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """
    Convenience function for secure browser login credential handling.
    
    Args:
        username: Username for login
        password_prompt: Prompt for password input
        session_id: Optional session ID
        
    Returns:
        Tuple of (username, password_credential_id) if successful
    """
    handler = BrowserUseCredentialHandler(session_id=session_id)
    
    password_credential_id = await handler.secure_credential_input(
        credential_type=CredentialType.PASSWORD,
        prompt=password_prompt
    )
    
    if password_credential_id:
        return username, password_credential_id
    return None


async def secure_api_key_input(api_name: str,
                             session_id: Optional[str] = None) -> Optional[str]:
    """
    Convenience function for secure API key input.
    
    Args:
        api_name: Name of the API
        session_id: Optional session ID
        
    Returns:
        API key credential ID if successful
    """
    handler = BrowserUseCredentialHandler(session_id=session_id)
    
    return await handler.secure_credential_input(
        credential_type=CredentialType.API_KEY,
        prompt=f"Enter {api_name} API key:"
    )


# Example usage
if __name__ == "__main__":
    async def example_usage():
        """Example usage of browser-use credential handling."""
        
        # Initialize credential handler
        handler = BrowserUseCredentialHandler(session_id="example-session")
        
        # Secure credential input
        password_id = await handler.secure_credential_input(
            credential_type=CredentialType.PASSWORD,
            prompt="Enter your password:"
        )
        
        if password_id:
            print(f"Password stored with ID: {password_id}")
            
            # Retrieve credential
            password = await handler.retrieve_credential(password_id)
            if password:
                print("Password retrieved successfully")
            
            # Generate audit report
            audit_report = handler.generate_audit_report()
            print("Audit Report:")
            print(json.dumps(audit_report, indent=2, default=str))
            
            # Cleanup
            await handler.delete_credential(password_id)
            print("Credential deleted")
    
    # Run example
    asyncio.run(example_usage())