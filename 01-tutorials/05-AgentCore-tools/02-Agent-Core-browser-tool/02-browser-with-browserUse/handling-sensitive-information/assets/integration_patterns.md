# Browser-Use with AgentCore Browser Tool - Integration Patterns

## Overview

This document outlines the integration patterns for connecting browser-use with Amazon Bedrock AgentCore Browser Tool to handle sensitive information securely. These patterns provide reusable approaches for different enterprise scenarios while maintaining security and compliance requirements.

## Core Integration Patterns

### 1. Secure Session Pattern

The Secure Session Pattern establishes a secure connection between browser-use and AgentCore Browser Tool with proper authentication and isolation.

```python
import asyncio
from browser_use import Agent
from bedrock_agentcore.tools.browser_client import BrowserClient
from typing import Dict, Optional

class SecureBrowserUseSession:
    """Secure session management for browser-use with AgentCore."""
    
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.agentcore_client = None
        self.browser_session = None
        self.session_id = None
    
    async def create_secure_session(self) -> Dict[str, str]:
        """Create a secure AgentCore browser session."""
        try:
            # Initialize AgentCore Browser Client
            self.agentcore_client = BrowserClient(region=self.region)
            
            # Create isolated browser session
            session = await self.agentcore_client.create_session()
            self.session_id = session.session_id
            
            # Get WebSocket connection details
            ws_url, headers = self.agentcore_client.get_connection_details(
                self.session_id
            )
            
            return {
                'ws_url': ws_url,
                'headers': headers,
                'session_id': self.session_id
            }
            
        except Exception as e:
            await self.cleanup_session()
            raise Exception(f"Failed to create secure session: {str(e)}")
    
    async def cleanup_session(self):
        """Clean up AgentCore session resources."""
        if self.agentcore_client and self.session_id:
            try:
                await self.agentcore_client.terminate_session(self.session_id)
            except Exception as e:
                print(f"Warning: Session cleanup failed: {str(e)}")
            finally:
                self.session_id = None
                self.browser_session = None
```

### 2. Sensitive Data Handler Pattern

The Sensitive Data Handler Pattern provides automatic detection and masking of sensitive information during browser-use operations.

```python
import re
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class PIIPattern:
    """Pattern definition for PII detection."""
    name: str
    pattern: str
    mask_char: str = '*'
    compliance_type: str = 'general'

class SensitiveDataHandler:
    """Handler for sensitive data detection and masking."""
    
    def __init__(self):
        self.pii_patterns = self._initialize_pii_patterns()
    
    def _initialize_pii_patterns(self) -> List[PIIPattern]:
        """Initialize PII detection patterns."""
        return [
            PIIPattern(
                name="ssn",
                pattern=r'\b\d{3}-\d{2}-\d{4}\b',
                compliance_type="hipaa"
            ),
            PIIPattern(
                name="credit_card",
                pattern=r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
                compliance_type="pci_dss"
            ),
            PIIPattern(
                name="email",
                pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                compliance_type="gdpr"
            ),
            PIIPattern(
                name="phone",
                pattern=r'\b\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
                compliance_type="general"
            )
        ]
    
    def detect_pii(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII in text content."""
        detected_pii = []
        
        for pattern in self.pii_patterns:
            matches = re.finditer(pattern.pattern, text, re.IGNORECASE)
            for match in matches:
                detected_pii.append({
                    'type': pattern.name,
                    'value': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'compliance_type': pattern.compliance_type
                })
        
        return detected_pii
    
    def mask_pii(self, text: str, mask_char: str = '*') -> str:
        """Mask PII in text content."""
        masked_text = text
        
        for pattern in self.pii_patterns:
            def mask_match(match):
                original = match.group()
                if pattern.name == 'credit_card':
                    # Show only last 4 digits for credit cards
                    return f"****-****-****-{original[-4:]}"
                elif pattern.name == 'ssn':
                    # Show only last 4 digits for SSN
                    return f"***-**-{original[-4:]}"
                else:
                    # Full masking for other PII types
                    return mask_char * len(original)
            
            masked_text = re.sub(pattern.pattern, mask_match, masked_text, flags=re.IGNORECASE)
        
        return masked_text
```

### 3. Compliance Validation Pattern

The Compliance Validation Pattern ensures that browser-use operations comply with regulatory requirements.

```python
from enum import Enum
from typing import Dict, List, Optional
import json
import logging

class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    GDPR = "gdpr"
    SOX = "sox"

class ComplianceValidator:
    """Validator for compliance requirements."""
    
    def __init__(self):
        self.compliance_rules = self._load_compliance_rules()
        self.logger = logging.getLogger(__name__)
    
    def _load_compliance_rules(self) -> Dict[str, Dict]:
        """Load compliance rules configuration."""
        return {
            ComplianceFramework.HIPAA.value: {
                "required_controls": [
                    "data_encryption",
                    "audit_trail",
                    "access_controls",
                    "session_isolation"
                ],
                "data_types": ["phi", "medical_records", "patient_info"],
                "retention_period": 2555,  # 7 years in days
                "audit_level": "detailed"
            },
            ComplianceFramework.PCI_DSS.value: {
                "required_controls": [
                    "card_data_masking",
                    "secure_transmission",
                    "audit_logging",
                    "access_restrictions"
                ],
                "data_types": ["credit_card", "payment_info", "cardholder_data"],
                "retention_period": 365,  # 1 year in days
                "audit_level": "comprehensive"
            },
            ComplianceFramework.GDPR.value: {
                "required_controls": [
                    "data_minimization",
                    "consent_management",
                    "right_to_erasure",
                    "data_portability"
                ],
                "data_types": ["personal_data", "sensitive_personal_data"],
                "retention_period": 1095,  # 3 years in days
                "audit_level": "standard"
            }
        }
    
    def validate_operation(
        self, 
        operation: str, 
        data_types: List[str], 
        framework: ComplianceFramework
    ) -> Dict[str, Any]:
        """Validate operation against compliance framework."""
        
        framework_rules = self.compliance_rules.get(framework.value, {})
        validation_result = {
            "compliant": True,
            "framework": framework.value,
            "operation": operation,
            "violations": [],
            "recommendations": []
        }
        
        # Check required controls
        required_controls = framework_rules.get("required_controls", [])
        for control in required_controls:
            if not self._validate_control(control, operation):
                validation_result["compliant"] = False
                validation_result["violations"].append(
                    f"Missing required control: {control}"
                )
        
        # Check data type compatibility
        allowed_data_types = framework_rules.get("data_types", [])
        for data_type in data_types:
            if data_type not in allowed_data_types:
                validation_result["compliant"] = False
                validation_result["violations"].append(
                    f"Data type '{data_type}' not allowed under {framework.value}"
                )
        
        # Add recommendations
        if not validation_result["compliant"]:
            validation_result["recommendations"] = self._get_recommendations(
                framework, validation_result["violations"]
            )
        
        return validation_result
    
    def _validate_control(self, control: str, operation: str) -> bool:
        """Validate specific compliance control."""
        # Implementation would check if the control is properly implemented
        # This is a simplified version for demonstration
        control_implementations = {
            "data_encryption": True,  # Assume AgentCore provides encryption
            "audit_trail": True,      # Assume audit trail is enabled
            "access_controls": True,  # Assume proper access controls
            "session_isolation": True, # Assume AgentCore provides isolation
            "card_data_masking": True, # Assume PII masking is implemented
            "secure_transmission": True, # Assume HTTPS/TLS is used
            "audit_logging": True,    # Assume comprehensive logging
            "data_minimization": True, # Assume only necessary data is processed
            "consent_management": False, # Would need to be implemented
            "right_to_erasure": False,  # Would need to be implemented
            "data_portability": False   # Would need to be implemented
        }
        
        return control_implementations.get(control, False)
    
    def _get_recommendations(
        self, 
        framework: ComplianceFramework, 
        violations: List[str]
    ) -> List[str]:
        """Get recommendations for compliance violations."""
        recommendations = []
        
        for violation in violations:
            if "consent_management" in violation:
                recommendations.append(
                    "Implement user consent tracking and management system"
                )
            elif "right_to_erasure" in violation:
                recommendations.append(
                    "Implement secure data deletion capabilities"
                )
            elif "data_portability" in violation:
                recommendations.append(
                    "Implement data export functionality in machine-readable format"
                )
        
        return recommendations
```

### 4. Monitoring and Observability Pattern

The Monitoring and Observability Pattern provides comprehensive monitoring of browser-use operations with AgentCore.

```python
import time
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class SessionMetrics:
    """Metrics for browser-use session monitoring."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    operations_count: int = 0
    pii_detections: int = 0
    compliance_violations: int = 0
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}

class BrowserUseMonitor:
    """Monitor for browser-use operations with AgentCore."""
    
    def __init__(self):
        self.active_sessions: Dict[str, SessionMetrics] = {}
        self.completed_sessions: List[SessionMetrics] = []
    
    def start_session_monitoring(self, session_id: str) -> SessionMetrics:
        """Start monitoring a browser-use session."""
        metrics = SessionMetrics(
            session_id=session_id,
            start_time=datetime.now()
        )
        self.active_sessions[session_id] = metrics
        return metrics
    
    def record_operation(
        self, 
        session_id: str, 
        operation_type: str, 
        duration: float,
        pii_detected: bool = False,
        compliance_violation: bool = False
    ):
        """Record an operation in the session."""
        if session_id in self.active_sessions:
            metrics = self.active_sessions[session_id]
            metrics.operations_count += 1
            
            if pii_detected:
                metrics.pii_detections += 1
            
            if compliance_violation:
                metrics.compliance_violations += 1
            
            # Update performance metrics
            if operation_type not in metrics.performance_metrics:
                metrics.performance_metrics[operation_type] = []
            
            metrics.performance_metrics[operation_type].append(duration)
    
    def end_session_monitoring(self, session_id: str) -> Optional[SessionMetrics]:
        """End monitoring for a session."""
        if session_id in self.active_sessions:
            metrics = self.active_sessions[session_id]
            metrics.end_time = datetime.now()
            
            # Move to completed sessions
            self.completed_sessions.append(metrics)
            del self.active_sessions[session_id]
            
            return metrics
        
        return None
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary for a specific session."""
        # Check active sessions
        if session_id in self.active_sessions:
            return asdict(self.active_sessions[session_id])
        
        # Check completed sessions
        for session in self.completed_sessions:
            if session.session_id == session_id:
                return asdict(session)
        
        return None
    
    def get_live_view_url(self, session_id: str) -> Optional[str]:
        """Get AgentCore live view URL for session."""
        # This would integrate with AgentCore's live view API
        if session_id in self.active_sessions:
            return f"https://agentcore.aws.amazon.com/live-view/{session_id}"
        
        return None
```

### 5. Error Handling and Recovery Pattern

The Error Handling and Recovery Pattern provides robust error handling for sensitive data operations.

```python
import asyncio
import logging
from typing import Optional, Callable, Any
from contextlib import asynccontextmanager

class SensitiveOperationError(Exception):
    """Base exception for sensitive operation errors."""
    pass

class ComplianceViolationError(SensitiveOperationError):
    """Exception for compliance violations."""
    pass

class SessionIsolationError(SensitiveOperationError):
    """Exception for session isolation failures."""
    pass

class PIIMaskingError(SensitiveOperationError):
    """Exception for PII masking failures."""
    pass

class BrowserUseErrorHandler:
    """Error handler for browser-use operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @asynccontextmanager
    async def safe_sensitive_operation(
        self, 
        session_id: str,
        operation_name: str,
        cleanup_callback: Optional[Callable] = None
    ):
        """Context manager for safe sensitive operations."""
        try:
            self.logger.info(f"Starting sensitive operation: {operation_name}")
            yield
            self.logger.info(f"Completed sensitive operation: {operation_name}")
            
        except ComplianceViolationError as e:
            self.logger.error(f"Compliance violation in {operation_name}: {str(e)}")
            # Immediate session termination for compliance violations
            if cleanup_callback:
                await cleanup_callback(session_id, emergency=True)
            raise
            
        except SessionIsolationError as e:
            self.logger.error(f"Session isolation failure in {operation_name}: {str(e)}")
            # Emergency cleanup and session recreation
            if cleanup_callback:
                await cleanup_callback(session_id, emergency=True)
            raise
            
        except PIIMaskingError as e:
            self.logger.error(f"PII masking failure in {operation_name}: {str(e)}")
            # Continue with enhanced masking
            self.logger.info("Applying enhanced PII masking")
            
        except Exception as e:
            self.logger.error(f"Unexpected error in {operation_name}: {str(e)}")
            # Standard cleanup
            if cleanup_callback:
                await cleanup_callback(session_id, emergency=False)
            raise SensitiveOperationError(f"Operation failed: {str(e)}")
    
    async def handle_emergency_cleanup(
        self, 
        session_id: str, 
        agentcore_client: Any,
        emergency: bool = False
    ):
        """Handle emergency cleanup of sensitive operations."""
        try:
            if emergency:
                self.logger.warning(f"Emergency cleanup for session: {session_id}")
                # Immediate session termination
                await agentcore_client.terminate_session(session_id)
                # Clear any cached sensitive data
                await self._clear_sensitive_cache(session_id)
            else:
                self.logger.info(f"Standard cleanup for session: {session_id}")
                # Graceful session cleanup
                await agentcore_client.cleanup_session(session_id)
                
        except Exception as e:
            self.logger.error(f"Cleanup failed for session {session_id}: {str(e)}")
    
    async def _clear_sensitive_cache(self, session_id: str):
        """Clear any cached sensitive data."""
        # Implementation would clear any cached sensitive data
        # This is a placeholder for the actual implementation
        pass
```

## Integration Pattern Examples

### Healthcare Form Automation Pattern

```python
async def healthcare_form_automation_pattern():
    """Pattern for automating healthcare forms with HIPAA compliance."""
    
    # Initialize components
    session_manager = SecureBrowserUseSession()
    data_handler = SensitiveDataHandler()
    compliance_validator = ComplianceValidator()
    monitor = BrowserUseMonitor()
    error_handler = BrowserUseErrorHandler()
    
    session_id = None
    
    try:
        # Create secure session
        session_details = await session_manager.create_secure_session()
        session_id = session_details['session_id']
        
        # Start monitoring
        monitor.start_session_monitoring(session_id)
        
        # Validate HIPAA compliance
        validation_result = compliance_validator.validate_operation(
            operation="healthcare_form_automation",
            data_types=["phi", "medical_records"],
            framework=ComplianceFramework.HIPAA
        )
        
        if not validation_result["compliant"]:
            raise ComplianceViolationError(
                f"HIPAA compliance validation failed: {validation_result['violations']}"
            )
        
        # Create browser-use agent with secure session
        async with error_handler.safe_sensitive_operation(
            session_id=session_id,
            operation_name="healthcare_form_automation",
            cleanup_callback=error_handler.handle_emergency_cleanup
        ):
            # Browser automation logic here
            # This would include PII detection and masking
            pass
            
    finally:
        # Cleanup
        if session_id:
            monitor.end_session_monitoring(session_id)
            await session_manager.cleanup_session()
```

### Financial Data Processing Pattern

```python
async def financial_data_processing_pattern():
    """Pattern for processing financial data with PCI-DSS compliance."""
    
    # Similar structure to healthcare pattern but with PCI-DSS validation
    # and specific financial data handling requirements
    pass
```

## Best Practices

### 1. Session Management
- Always use secure session creation with proper authentication
- Implement automatic session cleanup and timeout
- Monitor session health and performance metrics
- Use session isolation for sensitive operations

### 2. Data Protection
- Implement real-time PII detection and masking
- Use compliance-specific data handling rules
- Encrypt sensitive data in transit and at rest
- Minimize data collection and processing

### 3. Error Handling
- Implement comprehensive error handling for sensitive operations
- Use emergency cleanup procedures for compliance violations
- Log errors without exposing sensitive information
- Provide clear error messages and recovery guidance

### 4. Monitoring and Auditing
- Enable comprehensive audit trails for all operations
- Monitor performance and security metrics
- Use real-time monitoring for sensitive data operations
- Implement compliance reporting and validation

### 5. Testing and Validation
- Test all integration patterns with mock sensitive data
- Validate compliance requirements in test environments
- Perform security testing and penetration testing
- Implement automated compliance validation

These integration patterns provide a solid foundation for building secure, compliant browser automation solutions using browser-use with AgentCore Browser Tool.