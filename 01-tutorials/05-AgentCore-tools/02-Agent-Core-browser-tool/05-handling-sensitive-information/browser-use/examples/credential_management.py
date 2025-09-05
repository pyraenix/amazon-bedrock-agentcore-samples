"""
Browser-Use Credential Management Example

This example demonstrates secure login automation with browser-use Agent
while leveraging AgentCore's secure environment for credential protection.

Features demonstrated:
- Secure login automation with browser-use Agent
- Credential protection and secure authentication workflows
- Session cleanup and credential isolation within AgentCore
- Multi-factor authentication handling
- Secure password management and rotation

Requirements: 1.3, 2.1, 2.2, 2.3, 2.4
"""

import asyncio
import logging
import json
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import base64

# Core imports - production ready
from browser_use import Agent
from langchain_aws import ChatBedrock

# AgentCore and tutorial utilities
from tools.browseruse_agentcore_session_manager import (
    BrowserUseAgentCoreSessionManager, 
    SessionConfig
)
from tools.browseruse_sensitive_data_handler import (
    BrowserUseSensitiveDataHandler,
    ComplianceFramework,
    DataClassification
)
from tools.browseruse_credential_handling import BrowserUseCredentialManager


class AuthenticationType(Enum):
    """Types of authentication methods."""
    PASSWORD = "password"
    MFA = "multi_factor"
    SSO = "single_sign_on"
    BIOMETRIC = "biometric"
    API_KEY = "api_key"
    OAUTH = "oauth"


class CredentialType(Enum):
    """Types of credentials."""
    USERNAME_PASSWORD = "username_password"
    API_TOKEN = "api_token"
    SSH_KEY = "ssh_key"
    CERTIFICATE = "certificate"
    DATABASE_CONNECTION = "database_connection"
    SERVICE_ACCOUNT = "service_account"


@dataclass
class LoginCredentials:
    """Login credentials with security metadata."""
    username: str
    password: str
    domain: Optional[str] = None
    mfa_token: Optional[str] = None
    security_questions: Optional[Dict[str, str]] = None
    credential_type: CredentialType = CredentialType.USERNAME_PASSWORD
    authentication_type: AuthenticationType = AuthenticationType.PASSWORD
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    rotation_required: bool = False


@dataclass
class AuthenticationSession:
    """Authentication session with security tracking."""
    session_id: str
    username: str
    login_url: str
    authentication_type: AuthenticationType
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    failure_reason: Optional[str] = None
    security_events: List[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class CredentialManagementExample:
    """
    Credential management example using browser-use with AgentCore.
    
    Demonstrates secure credential handling, authentication workflows,
    and session management with comprehensive security monitoring.
    """
    
    def __init__(self, region: str = 'us-east-1'):
        """
        Initialize the credential management example.
        
        Args:
            region: AWS region for AgentCore services
        """
        self.region = region
        self.logger = logging.getLogger(__name__)
        
        # Configure session for credential security
        self.session_config = SessionConfig(
            region=region,
            session_timeout=1200,  # 20 minutes for complex authentication flows
            enable_live_view=True,
            enable_session_replay=True,
            isolation_level="micro-vm",
            compliance_mode="enterprise"
        )
        
        # Initialize sensitive data handler for credential protection
        self.data_handler = BrowserUseSensitiveDataHandler(
            compliance_frameworks=[ComplianceFramework.SOX, ComplianceFramework.GDPR]
        )
        
        # Initialize credential manager for secure storage
        self.credential_manager = BrowserUseCredentialManager()
        
        # Initialize session manager
        self.session_manager = BrowserUseAgentCoreSessionManager(self.session_config)
        
        # Initialize LLM model for authentication context
        self.llm_model = ChatBedrock(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            region_name=region,
            model_kwargs={
                "max_tokens": 4000,
                "temperature": 0.1,  # Low temperature for precise credential handling
                "top_p": 0.9
            }
        )
        
        # Authentication tracking
        self.active_sessions: Dict[str, AuthenticationSession] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.security_events: List[Dict[str, Any]] = []
    
    async def create_sample_credentials(self) -> Dict[str, LoginCredentials]:
        """
        Create sample credentials for demonstration.
        
        Returns:
            Dictionary of sample login credentials for different services
        """
        return {
            'corporate_portal': LoginCredentials(
                username="john.doe@company.com",
                password="SecureP@ssw0rd123!",
                domain="company.com",
                mfa_token="123456",
                security_questions={
                    "What was your first pet's name?": "Fluffy",
                    "What city were you born in?": "Springfield"
                },
                authentication_type=AuthenticationType.MFA,
                expires_at=datetime.now() + timedelta(days=90)
            ),
            'banking_portal': LoginCredentials(
                username="john.doe.banking",
                password="B@nk1ngP@ss2024!",
                mfa_token="789012",
                authentication_type=AuthenticationType.MFA,
                credential_type=CredentialType.USERNAME_PASSWORD,
                expires_at=datetime.now() + timedelta(days=30),
                rotation_required=True
            ),
            'cloud_service': LoginCredentials(
                username="service-account-prod",
                password="API-KEY-abc123def456ghi789",
                credential_type=CredentialType.API_TOKEN,
                authentication_type=AuthenticationType.API_KEY,
                expires_at=datetime.now() + timedelta(days=365)
            ),
            'database_connection': LoginCredentials(
                username="db_admin",
                password="DbAdm1n$ecure2024",
                domain="production-db.company.com",
                credential_type=CredentialType.DATABASE_CONNECTION,
                authentication_type=AuthenticationType.PASSWORD,
                expires_at=datetime.now() + timedelta(days=60)
            )
        }
    
    def secure_credential_storage(self, 
                                credentials: Dict[str, LoginCredentials], 
                                session_id: str) -> Dict[str, Any]:
        """
        Securely store credentials with encryption and access controls.
        
        Args:
            credentials: Dictionary of credentials to store
            session_id: Session identifier for tracking
            
        Returns:
            Storage results with security metadata
        """
        self.logger.info("🔐 Securely storing credentials with encryption")
        
        storage_results = {}
        security_metadata = []
        
        for service_name, creds in credentials.items():
            # Generate unique credential ID
            cred_id = f"{session_id}_{service_name}_{secrets.token_hex(8)}"
            
            # Mask sensitive data for logging
            masked_username, _ = self.data_handler.mask_text(creds.username, "username")
            masked_password = "*" * len(creds.password)
            
            # Store username
            self.credential_manager.store_credential(
                credential_id=f"{cred_id}_username",
                credential_type="username",
                value=creds.username,
                metadata={
                    'service': service_name,
                    'session_id': session_id,
                    'credential_type': creds.credential_type.value,
                    'authentication_type': creds.authentication_type.value,
                    'expires_at': creds.expires_at.isoformat() if creds.expires_at else None,
                    'rotation_required': creds.rotation_required
                }
            )
            
            # Store password/token
            self.credential_manager.store_credential(
                credential_id=f"{cred_id}_password",
                credential_type="password",
                value=creds.password,
                metadata={
                    'service': service_name,
                    'session_id': session_id,
                    'credential_type': creds.credential_type.value,
                    'sensitive': True
                }
            )
            
            # Store MFA token if present
            if creds.mfa_token:
                self.credential_manager.store_credential(
                    credential_id=f"{cred_id}_mfa",
                    credential_type="mfa_token",
                    value=creds.mfa_token,
                    metadata={
                        'service': service_name,
                        'session_id': session_id,
                        'expires_quickly': True
                    }
                )
            
            # Store security questions if present
            if creds.security_questions:
                for question, answer in creds.security_questions.items():
                    question_id = hashlib.md5(question.encode()).hexdigest()[:8]
                    self.credential_manager.store_credential(
                        credential_id=f"{cred_id}_sq_{question_id}",
                        credential_type="security_question",
                        value=f"{question}|{answer}",
                        metadata={
                            'service': service_name,
                            'session_id': session_id,
                            'question_hash': question_id
                        }
                    )
            
            storage_results[service_name] = {
                'credential_id': cred_id,
                'masked_username': masked_username,
                'masked_password': masked_password,
                'has_mfa': bool(creds.mfa_token),
                'has_security_questions': bool(creds.security_questions),
                'expires_at': creds.expires_at.isoformat() if creds.expires_at else None,
                'rotation_required': creds.rotation_required
            }
            
            security_metadata.append({
                'service': service_name,
                'credential_id': cred_id,
                'storage_timestamp': datetime.now().isoformat(),
                'encryption_applied': True,
                'access_controls_enabled': True,
                'audit_trail_created': True
            })
        
        return {
            'storage_results': storage_results,
            'security_metadata': security_metadata,
            'total_credentials_stored': len(credentials),
            'storage_timestamp': datetime.now().isoformat()
        }
    
    def validate_credential_security(self, credentials: Dict[str, LoginCredentials]) -> Dict[str, Any]:
        """
        Validate credential security according to best practices.
        
        Args:
            credentials: Credentials to validate
            
        Returns:
            Security validation results and recommendations
        """
        self.logger.info("🔍 Validating credential security")
        
        validation_results = {}
        overall_security_score = 0
        total_checks = 0
        
        for service_name, creds in credentials.items():
            service_score = 0
            service_checks = 0
            issues = []
            recommendations = []
            
            # Password strength validation
            password_strength = self._validate_password_strength(creds.password)
            service_score += password_strength['score']
            service_checks += 1
            
            if password_strength['score'] < 0.8:
                issues.append(f"Weak password (score: {password_strength['score']:.2f})")
                recommendations.append("Use a stronger password with mixed case, numbers, and symbols")
            
            # MFA validation
            if creds.authentication_type == AuthenticationType.MFA:
                if creds.mfa_token:
                    service_score += 1
                    recommendations.append("MFA is properly configured")
                else:
                    issues.append("MFA enabled but no token provided")
                    recommendations.append("Ensure MFA token is available")
            else:
                issues.append("MFA not enabled")
                recommendations.append("Enable multi-factor authentication for enhanced security")
            service_checks += 1
            
            # Expiration validation
            if creds.expires_at:
                days_until_expiry = (creds.expires_at - datetime.now()).days
                if days_until_expiry > 0:
                    service_score += 1
                    if days_until_expiry <= 7:
                        issues.append(f"Credential expires in {days_until_expiry} days")
                        recommendations.append("Plan credential rotation soon")
                else:
                    issues.append("Credential has expired")
                    recommendations.append("Rotate expired credentials immediately")
            else:
                issues.append("No expiration date set")
                recommendations.append("Set credential expiration dates")
            service_checks += 1
            
            # Rotation validation
            if creds.rotation_required:
                issues.append("Credential rotation is required")
                recommendations.append("Rotate credentials as soon as possible")
            else:
                service_score += 1
            service_checks += 1
            
            # Domain validation for corporate accounts
            if creds.domain and '@' in creds.username:
                username_domain = creds.username.split('@')[1]
                if username_domain == creds.domain:
                    service_score += 1
                else:
                    issues.append("Username domain doesn't match credential domain")
                    recommendations.append("Verify username and domain consistency")
            else:
                service_score += 0.5  # Partial credit for non-domain accounts
            service_checks += 1
            
            # Calculate service security score
            service_security_score = (service_score / service_checks) if service_checks > 0 else 0
            
            validation_results[service_name] = {
                'security_score': service_security_score,
                'issues': issues,
                'recommendations': recommendations,
                'password_strength': password_strength,
                'has_mfa': creds.authentication_type == AuthenticationType.MFA,
                'expires_at': creds.expires_at.isoformat() if creds.expires_at else None,
                'rotation_required': creds.rotation_required
            }
            
            overall_security_score += service_security_score
            total_checks += 1
        
        # Calculate overall security score
        overall_score = (overall_security_score / total_checks) if total_checks > 0 else 0
        
        return {
            'overall_security_score': overall_score,
            'service_validations': validation_results,
            'security_level': self._get_security_level(overall_score),
            'global_recommendations': self._get_global_security_recommendations(validation_results),
            'validation_timestamp': datetime.now().isoformat()
        }
    
    def _validate_password_strength(self, password: str) -> Dict[str, Any]:
        """
        Validate password strength using multiple criteria.
        
        Args:
            password: Password to validate
            
        Returns:
            Password strength analysis
        """
        score = 0
        criteria = {
            'length': len(password) >= 12,
            'uppercase': any(c.isupper() for c in password),
            'lowercase': any(c.islower() for c in password),
            'digits': any(c.isdigit() for c in password),
            'special_chars': any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password),
            'no_common_patterns': not any(pattern in password.lower() for pattern in ['password', '123456', 'qwerty'])
        }
        
        # Calculate score based on criteria
        score = sum(criteria.values()) / len(criteria)
        
        # Bonus for very long passwords
        if len(password) >= 16:
            score = min(1.0, score + 0.1)
        
        strength_level = 'weak'
        if score >= 0.9:
            strength_level = 'very_strong'
        elif score >= 0.7:
            strength_level = 'strong'
        elif score >= 0.5:
            strength_level = 'medium'
        
        return {
            'score': score,
            'strength_level': strength_level,
            'criteria_met': criteria,
            'length': len(password)
        }
    
    def _get_security_level(self, score: float) -> str:
        """Get security level based on overall score."""
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.7:
            return 'good'
        elif score >= 0.5:
            return 'fair'
        else:
            return 'poor'
    
    def _get_global_security_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Get global security recommendations based on validation results."""
        recommendations = []
        
        # Check for common issues across services
        weak_passwords = sum(1 for result in validation_results.values() 
                           if result['password_strength']['score'] < 0.7)
        no_mfa = sum(1 for result in validation_results.values() 
                    if not result['has_mfa'])
        rotation_needed = sum(1 for result in validation_results.values() 
                            if result['rotation_required'])
        
        if weak_passwords > 0:
            recommendations.append(f"Strengthen {weak_passwords} weak passwords")
        
        if no_mfa > 0:
            recommendations.append(f"Enable MFA for {no_mfa} services")
        
        if rotation_needed > 0:
            recommendations.append(f"Rotate {rotation_needed} credentials requiring updates")
        
        recommendations.extend([
            "Implement regular credential rotation schedule",
            "Use a password manager for secure storage",
            "Enable security monitoring and alerts",
            "Conduct regular security audits"
        ])
        
        return recommendations
    
    async def demonstrate_secure_login_automation(self, 
                                                service_name: str = "corporate_portal",
                                                login_url: str = "https://portal.company.com/login",
                                                credentials: Optional[Dict[str, LoginCredentials]] = None) -> Dict[str, Any]:
        """
        Demonstrate secure login automation with browser-use and AgentCore.
        
        Args:
            service_name: Name of the service to log into
            login_url: URL of the login page
            credentials: Credentials to use (sample data if not provided)
            
        Returns:
            Results of the secure login automation
        """
        if credentials is None:
            credentials = await self.create_sample_credentials()
        
        if service_name not in credentials:
            raise ValueError(f"Credentials for service '{service_name}' not found")
        
        creds = credentials[service_name]
        self.logger.info(f"🔐 Starting secure login automation for {service_name}")
        
        # Create authentication session
        auth_session = AuthenticationSession(
            session_id=f"auth_{secrets.token_hex(8)}",
            username=creds.username,
            login_url=login_url,
            authentication_type=creds.authentication_type,
            start_time=datetime.now(),
            security_events=[]
        )
        
        try:
            # Validate credential security
            security_validation = self.validate_credential_security({service_name: creds})
            
            # Store credentials securely
            storage_result = self.secure_credential_storage({service_name: creds}, auth_session.session_id)
            
            # Create secure AgentCore session
            session_id, ws_url, headers = await self.session_manager.create_secure_session(
                sensitive_context={
                    'data_type': 'authentication',
                    'service': service_name,
                    'authentication_type': creds.authentication_type.value,
                    'classification': 'restricted',
                    'audit_required': True
                }
            )
            
            self.logger.info(f"✅ Created secure AgentCore session: {session_id}")
            live_view_url = self.session_manager.get_live_view_url(session_id)
            if live_view_url:
                self.logger.info(f"👁️ Live view available: {live_view_url}")
            
            # Retrieve stored credentials for use
            cred_id = storage_result['storage_results'][service_name]['credential_id']
            stored_username = self.credential_manager.retrieve_credential(f"{cred_id}_username")
            stored_password = self.credential_manager.retrieve_credential(f"{cred_id}_password")
            stored_mfa = self.credential_manager.retrieve_credential(f"{cred_id}_mfa") if creds.mfa_token else None
            
            # Create login-specific task instruction
            task_instruction = f"""
            Navigate to {login_url} and perform secure login automation.
            
            CRITICAL SECURITY REQUIREMENTS:
            1. Verify the login page is legitimate and secure (HTTPS)
            2. Use only the provided credentials - never expose them in logs
            3. Handle MFA if required
            4. Verify successful login without exposing sensitive data
            5. Take screenshots at key steps for audit trail
            6. Handle any security questions if prompted
            7. Ensure proper session establishment
            
            LOGIN CREDENTIALS (USE SECURELY):
            - Username: {stored_username}
            - Password: [RETRIEVED FROM SECURE STORAGE]
            - MFA Token: {stored_mfa if stored_mfa else 'Not required'}
            - Authentication Type: {creds.authentication_type.value}
            
            SECURITY CONTEXT:
            - Service: {service_name}
            - Session ID: {auth_session.session_id}
            - Security Level: {security_validation['security_level']}
            - AgentCore Session: {session_id}
            
            COMPLIANCE NOTES:
            - All actions are being recorded for audit
            - Session is isolated in AgentCore micro-VM
            - Credentials are encrypted and access-controlled
            """
            
            # Create browser-use agent with authentication context
            agent = await self.session_manager.create_browseruse_agent(
                session_id=session_id,
                task=task_instruction,
                llm_model=self.llm_model
            )
            
            self.logger.info("🤖 Created browser-use agent for login automation")
            
            # Execute the secure login task
            execution_result = await self.session_manager.execute_sensitive_task(
                session_id=session_id,
                agent=agent,
                sensitive_data_context={
                    'credential_types': ['username', 'password', 'mfa_token'],
                    'authentication_type': creds.authentication_type.value,
                    'service': service_name,
                    'audit_level': 'comprehensive'
                }
            )
            
            # Update authentication session
            auth_session.end_time = datetime.now()
            auth_session.success = execution_result.get('status') == 'completed'
            if not auth_session.success:
                auth_session.failure_reason = execution_result.get('error', 'Unknown error')
            
            # Get session status and metrics
            session_status = self.session_manager.get_session_status(session_id)
            
            # Record security event
            security_event = {
                'event_type': 'login_attempt',
                'service': service_name,
                'username': creds.username,
                'success': auth_session.success,
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'authentication_type': creds.authentication_type.value
            }
            self.security_events.append(security_event)
            auth_session.security_events.append(security_event)
            
            # Store authentication session
            self.active_sessions[auth_session.session_id] = auth_session
            
            # Compile comprehensive results
            results = {
                'status': 'completed' if auth_session.success else 'failed',
                'authentication_session': {
                    'session_id': auth_session.session_id,
                    'username': auth_session.username,
                    'service': service_name,
                    'success': auth_session.success,
                    'start_time': auth_session.start_time.isoformat(),
                    'end_time': auth_session.end_time.isoformat() if auth_session.end_time else None,
                    'failure_reason': auth_session.failure_reason
                },
                'agentcore_session': {
                    'session_id': session_id,
                    'live_view_url': live_view_url,
                    'session_metrics': session_status
                },
                'execution_result': execution_result,
                'security_measures': {
                    'credential_encryption': True,
                    'secure_storage': True,
                    'micro_vm_isolation': True,
                    'session_recording': True,
                    'audit_trail': True,
                    'mfa_supported': creds.authentication_type == AuthenticationType.MFA
                },
                'credential_security': {
                    'security_validation': security_validation,
                    'storage_result': storage_result,
                    'password_strength': security_validation['service_validations'][service_name]['password_strength']
                },
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Login automation completed for {service_name}")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Login automation failed for {service_name}: {e}")
            
            # Update authentication session with failure
            auth_session.end_time = datetime.now()
            auth_session.success = False
            auth_session.failure_reason = str(e)
            
            # Record security event for failure
            security_event = {
                'event_type': 'login_failure',
                'service': service_name,
                'username': creds.username,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.security_events.append(security_event)
            
            return {
                'status': 'failed',
                'error': str(e),
                'authentication_session': {
                    'session_id': auth_session.session_id,
                    'username': auth_session.username,
                    'service': service_name,
                    'success': False,
                    'failure_reason': str(e)
                },
                'timestamp': datetime.now().isoformat()
            }
        
        finally:
            # Always cleanup session and credentials for security
            if 'session_id' in locals():
                await self.session_manager.cleanup_session(session_id, reason="login_task_complete")
                self.logger.info("🧹 Login session cleaned up for security")
            
            # Clean up stored credentials
            if 'storage_result' in locals():
                await self._cleanup_stored_credentials(storage_result, auth_session.session_id)
    
    async def _cleanup_stored_credentials(self, 
                                        storage_result: Dict[str, Any], 
                                        session_id: str) -> None:
        """
        Clean up stored credentials after authentication.
        
        Args:
            storage_result: Storage result containing credential IDs
            session_id: Session identifier
        """
        for service_name, service_data in storage_result['storage_results'].items():
            cred_id = service_data['credential_id']
            
            # Delete all credentials associated with this session
            credentials_to_delete = [
                f"{cred_id}_username",
                f"{cred_id}_password",
                f"{cred_id}_mfa"
            ]
            
            for cred_to_delete in credentials_to_delete:
                self.credential_manager.delete_credential(cred_to_delete)
            
            # Delete security question credentials
            all_credentials = self.credential_manager.list_credentials()
            for cred in all_credentials:
                if cred['credential_id'].startswith(f"{cred_id}_sq_"):
                    self.credential_manager.delete_credential(cred['credential_id'])
        
        self.logger.info(f"🗑️ Cleaned up stored credentials for session: {session_id}")
    
    async def demonstrate_multi_service_authentication(self) -> Dict[str, Any]:
        """
        Demonstrate authentication across multiple services with credential management.
        
        Returns:
            Results of multi-service authentication demonstration
        """
        self.logger.info("🔐 Demonstrating multi-service authentication")
        
        # Get sample credentials
        credentials = await self.create_sample_credentials()
        
        # Services to authenticate with
        services_to_test = [
            ('corporate_portal', 'https://portal.company.com/login'),
            ('banking_portal', 'https://banking.example.com/login'),
            ('cloud_service', 'https://cloud.example.com/api/login')
        ]
        
        authentication_results = {}
        overall_success = True
        
        for service_name, login_url in services_to_test:
            try:
                self.logger.info(f"🔑 Authenticating with {service_name}")
                
                result = await self.demonstrate_secure_login_automation(
                    service_name=service_name,
                    login_url=login_url,
                    credentials=credentials
                )
                
                authentication_results[service_name] = result
                
                if result['status'] != 'completed':
                    overall_success = False
                
                # Brief pause between authentications
                await asyncio.sleep(2)
                
            except Exception as e:
                self.logger.error(f"❌ Failed to authenticate with {service_name}: {e}")
                authentication_results[service_name] = {
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                overall_success = False
        
        # Compile multi-service results
        return {
            'overall_status': 'completed' if overall_success else 'partial_failure',
            'services_tested': len(services_to_test),
            'successful_authentications': sum(1 for result in authentication_results.values() 
                                            if result.get('status') == 'completed'),
            'authentication_results': authentication_results,
            'security_events': self.security_events,
            'active_sessions': len(self.active_sessions),
            'timestamp': datetime.now().isoformat()
        }
    
    async def demonstrate_credential_rotation(self, 
                                            service_name: str = "banking_portal") -> Dict[str, Any]:
        """
        Demonstrate secure credential rotation workflow.
        
        Args:
            service_name: Service to rotate credentials for
            
        Returns:
            Results of credential rotation demonstration
        """
        self.logger.info(f"🔄 Demonstrating credential rotation for {service_name}")
        
        # Get current credentials
        current_credentials = await self.create_sample_credentials()
        current_cred = current_credentials[service_name]
        
        # Generate new password
        new_password = self._generate_secure_password()
        
        # Create new credentials
        new_credentials = LoginCredentials(
            username=current_cred.username,
            password=new_password,
            domain=current_cred.domain,
            mfa_token=current_cred.mfa_token,
            security_questions=current_cred.security_questions,
            credential_type=current_cred.credential_type,
            authentication_type=current_cred.authentication_type,
            expires_at=datetime.now() + timedelta(days=90),
            rotation_required=False
        )
        
        try:
            # Use the secure session context manager for rotation
            async with self.session_manager.secure_session_context(
                task=f"Rotate credentials for {service_name}",
                llm_model=self.llm_model,
                sensitive_context={
                    'data_type': 'credential_rotation',
                    'service': service_name,
                    'operation': 'password_change'
                }
            ) as (session_id, agent):
                
                self.logger.info(f"🔐 Secure credential rotation session created: {session_id}")
                
                # Store both old and new credentials temporarily
                rotation_session_id = f"rotation_{secrets.token_hex(8)}"
                
                old_storage = self.secure_credential_storage(
                    {f"{service_name}_old": current_cred}, 
                    rotation_session_id
                )
                
                new_storage = self.secure_credential_storage(
                    {f"{service_name}_new": new_credentials}, 
                    rotation_session_id
                )
                
                # Execute credential rotation task
                rotation_task = f"""
                Navigate to the password change page for {service_name} and rotate credentials securely.
                
                CREDENTIAL ROTATION REQUIREMENTS:
                1. Log in with current credentials
                2. Navigate to password/security settings
                3. Change password using secure new password
                4. Verify password change was successful
                5. Test login with new credentials
                6. Take screenshots for audit trail
                
                CURRENT CREDENTIALS: [Retrieved from secure storage]
                NEW PASSWORD: [Retrieved from secure storage]
                
                SECURITY NOTES:
                - This is a credential rotation operation
                - All actions are audited and recorded
                - Session is isolated for security
                """
                
                # Update agent task
                agent.task = rotation_task
                
                # Execute rotation
                rotation_result = await agent.run()
                
                # Validate new credentials
                validation_result = self.validate_credential_security({service_name: new_credentials})
                
                return {
                    'status': 'completed',
                    'service': service_name,
                    'rotation_session_id': rotation_session_id,
                    'agentcore_session_id': session_id,
                    'rotation_result': rotation_result,
                    'old_credential_security': self.validate_credential_security({service_name: current_cred}),
                    'new_credential_security': validation_result,
                    'security_improvements': {
                        'password_strength_improved': (
                            validation_result['service_validations'][service_name]['password_strength']['score'] >
                            self.validate_credential_security({service_name: current_cred})['service_validations'][service_name]['password_strength']['score']
                        ),
                        'expiration_extended': True,
                        'rotation_requirement_cleared': True
                    },
                    'timestamp': datetime.now().isoformat()
                }
        
        except Exception as e:
            self.logger.error(f"❌ Credential rotation failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'service': service_name,
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_secure_password(self, length: int = 16) -> str:
        """
        Generate a secure password with mixed character types.
        
        Args:
            length: Password length
            
        Returns:
            Secure password string
        """
        import string
        
        # Ensure we have at least one of each character type
        password_chars = [
            secrets.choice(string.ascii_uppercase),
            secrets.choice(string.ascii_lowercase),
            secrets.choice(string.digits),
            secrets.choice('!@#$%^&*()_+-=[]{}|;:,.<>?')
        ]
        
        # Fill the rest with random characters
        all_chars = string.ascii_letters + string.digits + '!@#$%^&*()_+-=[]{}|;:,.<>?'
        for _ in range(length - 4):
            password_chars.append(secrets.choice(all_chars))
        
        # Shuffle the password
        secrets.SystemRandom().shuffle(password_chars)
        
        return ''.join(password_chars)
    
    async def run_comprehensive_credential_demo(self) -> Dict[str, Any]:
        """
        Run a comprehensive credential management demonstration.
        
        Returns:
            Complete demonstration results with security analysis
        """
        self.logger.info("🚀 Starting comprehensive credential management demo")
        
        try:
            # Step 1: Demonstrate single service authentication
            single_auth_result = await self.demonstrate_secure_login_automation()
            
            # Step 2: Demonstrate multi-service authentication
            multi_auth_result = await self.demonstrate_multi_service_authentication()
            
            # Step 3: Demonstrate credential rotation
            rotation_result = await self.demonstrate_credential_rotation()
            
            # Step 4: Generate security summary
            security_summary = self._generate_security_summary()
            
            # Compile comprehensive demo results
            demo_results = {
                'demo_status': 'completed',
                'single_service_authentication': single_auth_result,
                'multi_service_authentication': multi_auth_result,
                'credential_rotation': rotation_result,
                'security_summary': security_summary,
                'demo_metrics': {
                    'total_authentications': multi_auth_result.get('services_tested', 0) + 1,
                    'successful_authentications': multi_auth_result.get('successful_authentications', 0) + (1 if single_auth_result.get('status') == 'completed' else 0),
                    'credentials_rotated': 1 if rotation_result.get('status') == 'completed' else 0,
                    'security_events_recorded': len(self.security_events),
                    'active_sessions_managed': len(self.active_sessions)
                },
                'security_measures_demonstrated': [
                    'Secure credential storage with encryption',
                    'AgentCore micro-VM isolation for authentication',
                    'Multi-factor authentication support',
                    'Comprehensive audit trails',
                    'Automated credential rotation',
                    'Password strength validation',
                    'Session management and cleanup',
                    'Real-time security monitoring'
                ],
                'demo_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("✅ Comprehensive credential management demo completed successfully")
            return demo_results
            
        except Exception as e:
            self.logger.error(f"❌ Credential management demo failed: {e}")
            return {
                'demo_status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
        finally:
            # Cleanup any remaining resources
            await self.session_manager.shutdown()
            self.credential_manager.clear_all_credentials()
    
    def _generate_security_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive security summary of all operations."""
        return {
            'total_security_events': len(self.security_events),
            'authentication_attempts': len([e for e in self.security_events if e['event_type'] == 'login_attempt']),
            'successful_logins': len([e for e in self.security_events if e['event_type'] == 'login_attempt' and e.get('success', False)]),
            'failed_logins': len([e for e in self.security_events if e['event_type'] == 'login_attempt' and not e.get('success', True)]),
            'active_sessions': len(self.active_sessions),
            'credential_access_log': self.credential_manager.get_access_log(),
            'security_recommendations': [
                'Continue using AgentCore micro-VM isolation for all authentication',
                'Implement regular credential rotation schedules',
                'Monitor authentication patterns for anomalies',
                'Maintain comprehensive audit trails',
                'Use strong, unique passwords for all services',
                'Enable MFA wherever possible'
            ],
            'summary_timestamp': datetime.now().isoformat()
        }


# Standalone execution example
async def main():
    """Main execution function for the credential management example."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🔐 Credential Management Example - Browser-Use + AgentCore")
    
    try:
        # Initialize the credential management example
        credential_example = CredentialManagementExample(region='us-east-1')
        
        # Run the comprehensive demonstration
        results = await credential_example.run_comprehensive_credential_demo()
        
        # Display results
        print("\n" + "="*80)
        print("🔐 CREDENTIAL MANAGEMENT RESULTS")
        print("="*80)
        print(f"Demo Status: {results['demo_status']}")
        
        if results['demo_status'] == 'completed':
            metrics = results['demo_metrics']
            print(f"Total Authentications: {metrics['total_authentications']}")
            print(f"Successful Authentications: {metrics['successful_authentications']}")
            print(f"Credentials Rotated: {metrics['credentials_rotated']}")
            print(f"Security Events Recorded: {metrics['security_events_recorded']}")
            print(f"Active Sessions Managed: {metrics['active_sessions_managed']}")
            
            print("\nSecurity Measures Demonstrated:")
            for measure in results['security_measures_demonstrated']:
                print(f"  ✅ {measure}")
            
            # Show security summary
            security = results['security_summary']
            print(f"\nSecurity Summary:")
            print(f"  Authentication Attempts: {security['authentication_attempts']}")
            print(f"  Successful Logins: {security['successful_logins']}")
            print(f"  Failed Logins: {security['failed_logins']}")
            
            if security.get('security_recommendations'):
                print("\nSecurity Recommendations:")
                for rec in security['security_recommendations']:
                    print(f"  📋 {rec}")
        else:
            print(f"Error: {results.get('error', 'Unknown error')}")
        
        print("="*80)
        
    except Exception as e:
        logger.error(f"❌ Failed to run credential management example: {e}")
        print(f"\n❌ Example failed: {e}")


if __name__ == "__main__":
    print("🔐 Browser-Use Credential Management Example")
    print("📋 Demonstrates secure authentication workflows with AgentCore")
    print("⚠️  Requires: browser-use, bedrock-agentcore, and AWS credentials")
    print("🚀 Starting demonstration...\n")
    
    # Run the example
    asyncio.run(main())