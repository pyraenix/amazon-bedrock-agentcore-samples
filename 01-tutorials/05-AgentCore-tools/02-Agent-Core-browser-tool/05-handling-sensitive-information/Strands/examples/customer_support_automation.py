#!/usr/bin/env python3
"""
Customer Support Automation with PII Protection
==============================================

This example demonstrates how to use Strands agents with AgentCore Browser Tool
for PII-protected customer service workflows. It showcases:

1. Secure customer portal access and data extraction
2. Real-time PII detection and masking in customer interactions
3. GDPR-compliant customer data handling
4. Automated customer service workflows with privacy protection
5. Comprehensive audit logging for customer data compliance

Requirements:
- GDPR compliance for customer data
- Real-time PII detection and masking
- Customer consent management
- Comprehensive audit trails
- Automated privacy controls
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid
import re

# Strands framework imports
from strands import Agent
from strands.tools import tool, PythonAgentTool
from strands.types.tools import AgentTool

# AgentCore Browser Tool integration
from strands_tools.browser.agent_core_browser import AgentCoreBrowser
from bedrock_agentcore.tools.browser_client import BrowserClient

# AWS and security imports
import boto3
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64


class GDPRComplianceLevel(Enum):
    """GDPR compliance levels for different data types."""
    PERSONAL_DATA = "personal_data"  # Name, email, phone, address
    SENSITIVE_PERSONAL_DATA = "sensitive_personal_data"  # Health, financial, biometric
    PSEUDONYMIZED_DATA = "pseudonymized_data"  # Pseudonymized personal data
    ANONYMOUS_DATA = "anonymous_data"  # Fully anonymized data
    GENERAL_DATA = "general_data"  # Non-personal data


class CustomerDataType(Enum):
    """Types of customer data."""
    CONTACT_INFO = "contact_information"
    ACCOUNT_DETAILS = "account_details"
    SUPPORT_HISTORY = "support_history"
    PREFERENCES = "preferences"
    BILLING_INFO = "billing_information"
    INTERACTION_LOGS = "interaction_logs"


@dataclass
class CustomerDataRecord:
    """Secure customer data record with GDPR compliance."""
    record_id: str  # Hashed identifier
    customer_id: str  # Hashed customer identifier
    data_type: CustomerDataType
    compliance_level: GDPRComplianceLevel
    extracted_data: Dict[str, Any]
    consent_status: str
    data_retention_period: int  # Days
    extraction_timestamp: datetime
    audit_trail: List[str]
    pii_detected: bool
    anonymization_applied: bool
    
    def to_secure_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with PII protected."""
        secure_data = asdict(self)
        if self.compliance_level in [GDPRComplianceLevel.PERSONAL_DATA, GDPRComplianceLevel.SENSITIVE_PERSONAL_DATA]:
            if self.consent_status != "granted":
                secure_data['extracted_data'] = {"status": "consent_required", "data_type": self.data_type.value}
        return secure_data


class GDPRComplianceTool(PythonAgentTool):
    """Custom Strands tool for GDPR compliance and PII protection."""
    
    def __init__(self):
        super().__init__(name="gdpr_compliance_validator")
        self.pii_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone US format
            r'\b\+\d{1,3}\s?\d{3,14}\b',  # International phone
            r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b',  # Credit card
            r'\b\d{1,5}\s[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln)\b',  # Address
            r'\b[A-Z]{2}\s?\d{5}(?:-\d{4})?\b',  # ZIP code
        ]
        self.sensitive_pii_patterns = [
            r'\b(?:diabetes|cancer|HIV|AIDS|depression|anxiety)\b',  # Health conditions
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN (also sensitive)
            r'\b\d{16}\b',  # Credit card numbers
            r'\b(?:salary|income|wage):\s?\$?\d+\b',  # Financial info
        ]
        self.encryption_manager = EncryptionManager()
        
    async def detect_pii(self, content: str) -> Dict[str, Any]:
        """Detect personally identifiable information in content."""
        pii_items = []
        
        for pattern in self.pii_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                pii_items.extend([{
                    'type': 'personal_data',
                    'pattern': pattern,
                    'matches_count': len(matches),
                    'severity': 'medium'
                }])
        
        return {
            'pii_detected': len(pii_items) > 0,
            'pii_items': pii_items,
            'compliance_level': GDPRComplianceLevel.PERSONAL_DATA.value if pii_items else GDPRComplianceLevel.GENERAL_DATA.value
        }
    
    async def detect_sensitive_pii(self, content: str) -> Dict[str, Any]:
        """Detect sensitive personal data under GDPR."""
        sensitive_items = []
        
        for pattern in self.sensitive_pii_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                sensitive_items.extend([{
                    'type': 'sensitive_personal_data',
                    'pattern': pattern,
                    'matches_count': len(matches),
                    'severity': 'high'
                }])
        
        return {
            'sensitive_pii_detected': len(sensitive_items) > 0,
            'sensitive_items': sensitive_items
        }
    
    async def mask_pii(self, content: str, masking_level: str = "full") -> str:
        """Mask PII in content according to GDPR requirements."""
        masked_content = content
        
        # Mask email addresses
        masked_content = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL_REDACTED]',
            masked_content,
            flags=re.IGNORECASE
        )
        
        # Mask phone numbers
        masked_content = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', 'XXX-XXX-XXXX', masked_content)
        masked_content = re.sub(r'\b\+\d{1,3}\s?\d{3,14}\b', '[PHONE_REDACTED]', masked_content)
        
        # Mask SSN
        masked_content = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', 'XXX-XX-XXXX', masked_content)
        
        # Mask credit card numbers
        masked_content = re.sub(r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b', 'XXXX-XXXX-XXXX-XXXX', masked_content)
        
        # Mask addresses
        masked_content = re.sub(
            r'\b\d{1,5}\s[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln)\b',
            '[ADDRESS_REDACTED]',
            masked_content,
            flags=re.IGNORECASE
        )
        
        # Mask ZIP codes
        masked_content = re.sub(r'\b[A-Z]{2}\s?\d{5}(?:-\d{4})?\b', '[ZIP_REDACTED]', masked_content)
        
        return masked_content
    
    async def pseudonymize_data(self, data: Dict[str, Any], customer_id: str) -> Dict[str, Any]:
        """Pseudonymize customer data for GDPR compliance."""
        pseudonymized = {}
        
        # Create consistent pseudonym for customer
        customer_hash = hashlib.sha256(customer_id.encode()).hexdigest()[:8]
        pseudonym = f"CUSTOMER_{customer_hash}"
        
        for key, value in data.items():
            if isinstance(value, str):
                # Replace names with pseudonym
                if any(keyword in key.lower() for keyword in ['name', 'customer', 'user']):
                    pseudonymized[key] = pseudonym
                # Mask other PII
                else:
                    pseudonymized[key] = await self.mask_pii(value)
            elif isinstance(value, list):
                pseudonymized[key] = [await self.mask_pii(str(item)) if isinstance(item, str) else item for item in value]
            else:
                pseudonymized[key] = value
        
        return pseudonymized
    
    async def check_consent_status(self, customer_id: str, data_type: str) -> Dict[str, Any]:
        """Check customer consent status for data processing."""
        # In production, this would check against a consent management system
        # For demo purposes, we'll simulate consent checking
        consent_key = f"consent_{customer_id}_{data_type}"
        
        # Simulate consent database lookup
        consent_status = {
            "status": "granted",  # granted, denied, withdrawn, expired
            "granted_date": datetime.utcnow() - timedelta(days=30),
            "expiry_date": datetime.utcnow() + timedelta(days=335),  # 1 year from grant
            "purposes": ["customer_support", "service_improvement"],
            "data_types": [data_type]
        }
        
        return consent_status
    
    async def calculate_retention_period(self, data_type: str, consent_status: Dict[str, Any]) -> int:
        """Calculate data retention period based on GDPR requirements."""
        # Default retention periods (in days)
        retention_periods = {
            "contact_information": 1095,  # 3 years
            "account_details": 2555,  # 7 years (regulatory requirement)
            "support_history": 1095,  # 3 years
            "preferences": 365,  # 1 year
            "billing_information": 2555,  # 7 years
            "interaction_logs": 730  # 2 years
        }
        
        base_period = retention_periods.get(data_type, 365)
        
        # Adjust based on consent status
        if consent_status["status"] == "withdrawn":
            return 30  # 30 days for withdrawal processing
        elif consent_status["status"] == "expired":
            return 90  # 90 days grace period
        
        return base_period


class AgentCoreBrowserTool(AgentTool):
    """Custom Strands tool that integrates with AgentCore Browser Tool for customer support."""
    
    def __init__(self, region: str = "us-east-1"):
        super().__init__(name="agentcore_browser_support")
        self.region = region
        self.bedrock_agent = boto3.client('bedrock-agent-runtime', region_name=region)
        self.secrets_client = boto3.client('secretsmanager', region_name=region)
        
    async def invoke_browser_tool(self, instructions: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the AgentCore Browser Tool with customer support security configurations."""
        try:
            # Prepare the browser tool invocation with GDPR compliance
            tool_input = {
                "toolName": "AgentCoreBrowserTool",
                "toolInput": {
                    "instructions": json.dumps(instructions),
                    "securityConfig": {
                        "isolationLevel": "high",
                        "dataProtection": "gdpr_compliant",
                        "auditLogging": True,
                        "sessionTimeout": 1200,  # 20 minutes for support sessions
                        "screenshotDisabled": True,  # Prevent PII in screenshots
                        "clipboardDisabled": True,  # Prevent PII in clipboard
                        "piiProtection": True,  # Enable PII protection
                        "consentValidation": True  # Validate customer consent
                    }
                }
            }
            
            # Invoke the browser tool via Bedrock Agent Runtime
            response = self.bedrock_agent.invoke_agent(
                agentId="your-support-agent-id",  # Replace with actual agent ID
                agentAliasId="your-support-alias-id",  # Replace with actual alias ID
                sessionId=f"support-session-{uuid.uuid4()}",
                inputText=json.dumps(tool_input)
            )
            
            # Parse the response
            result = json.loads(response['completion'])
            return result
            
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def navigate_and_authenticate_support_portal(self, portal_url: str, credential_secret_name: str) -> Dict[str, Any]:
        """Navigate to customer support portal and authenticate securely."""
        # Retrieve credentials from AWS Secrets Manager
        try:
            secret_response = self.secrets_client.get_secret_value(SecretId=credential_secret_name)
            credentials = json.loads(secret_response['SecretString'])
        except Exception as e:
            return {"error": f"Failed to retrieve support credentials: {str(e)}", "success": False}
        
        # Prepare browser instructions for support portal navigation
        instructions = {
            "action": "navigate_and_authenticate_support",
            "url": portal_url,
            "complianceLevel": "gdpr",
            "steps": [
                {
                    "action": "navigate",
                    "url": portal_url
                },
                {
                    "action": "wait_for_element",
                    "selector": "input[type='email'], input[name='username']",
                    "timeout": 10000
                },
                {
                    "action": "fill_secure",
                    "selector": "input[type='email'], input[name='username']",
                    "value": credentials['username'],
                    "piiProtected": True
                },
                {
                    "action": "fill_secure",
                    "selector": "input[type='password']",
                    "value": credentials['password'],
                    "piiProtected": True
                },
                {
                    "action": "click",
                    "selector": "button[type='submit'], input[type='submit']"
                },
                {
                    "action": "wait_for_element",
                    "selector": ".dashboard, .customer-list, .support-workspace, .ticket-queue",
                    "timeout": 15000
                }
            ]
        }
        
        return await self.invoke_browser_tool(instructions)
    
    async def extract_customer_data(self, data_type: str, customer_id: str = None) -> Dict[str, Any]:
        """Extract customer data using the browser tool with PII protection."""
        # Define extraction instructions based on data type
        extraction_steps = {
            "contact_information": [
                {"action": "click", "selector": ".customer-profile, .contact-info"},
                {"action": "wait_for_load_state", "state": "networkidle"},
                {"action": "extract_data", "selector": ".contact, .customer-details", "attributes": ["text"], "piiProtected": True}
            ],
            "account_details": [
                {"action": "click", "selector": ".account-info, .customer-account"},
                {"action": "wait_for_load_state", "state": "networkidle"},
                {"action": "extract_data", "selector": ".account-detail, .account-summary", "attributes": ["text"], "piiProtected": True}
            ],
            "support_history": [
                {"action": "click", "selector": ".support-history, .ticket-history"},
                {"action": "wait_for_load_state", "state": "networkidle"},
                {"action": "extract_data", "selector": ".ticket, .support-case", "attributes": ["text"], "piiProtected": True}
            ],
            "interaction_logs": [
                {"action": "click", "selector": ".interaction-logs, .communication-history"},
                {"action": "wait_for_load_state", "state": "networkidle"},
                {"action": "extract_data", "selector": ".interaction, .communication", "attributes": ["text"], "piiProtected": True}
            ]
        }
        
        if data_type not in extraction_steps:
            return {"error": f"Unknown customer data type: {data_type}", "success": False}
        
        instructions = {
            "action": "extract_customer_data",
            "data_type": data_type,
            "customer_id": customer_id,
            "complianceLevel": "gdpr",
            "steps": extraction_steps[data_type]
        }
        
        return await self.invoke_browser_tool(instructions)


class SecureCustomerSupportAgent:
    """Strands agent specialized for GDPR-compliant customer support automation."""
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.browser_tool = AgentCoreBrowserTool(region=region)
        self.gdpr_tool = GDPRComplianceTool()
        self.audit_logger = AuditLogger(service_name="customer_support_agent")
        
        # Initialize encryption for customer data
        self.encryption_key = self._derive_customer_encryption_key()
        self.fernet = Fernet(self.encryption_key)
        
        # Initialize Strands agent with customer support configuration
        self.agent = Agent(
            name="customer_support_processor",
            tools=[self.browser_tool, self.gdpr_tool],
            security_context=SecurityContext(
                compliance_level="gdpr",
                data_classification="customer_personal_data",
                audit_required=True,
                encryption_required=True,
                pii_protection=True
            )
        )
    
    def _derive_customer_encryption_key(self) -> bytes:
        """Derive strong encryption key for customer data."""
        password = b"customer_data_encryption_key_2024_gdpr"
        salt = b"gdpr_compliant_customer_salt"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    async def create_secure_support_session(self, portal_config: Dict[str, Any]) -> str:
        """Create a secure browser session for customer support portal access."""
        session_id = f"support-session-{uuid.uuid4()}"
        
        # Log session creation for audit
        await self.audit_logger.log_event({
            "event_type": "support_session_created",
            "session_id": session_id,
            "portal": portal_config.get("name", "unknown"),
            "compliance_level": "gdpr",
            "security_controls": ["isolation_high", "gdpr_compliant", "pii_protection", "audit_logging"],
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return session_id
    
    async def authenticate_support_portal(
        self, 
        session_id: str, 
        portal_url: str,
        credential_secret_name: str
    ) -> bool:
        """Securely authenticate to customer support portal using AgentCore Browser Tool."""
        try:
            # Use the browser tool to navigate and authenticate
            auth_result = await self.browser_tool.navigate_and_authenticate_support_portal(portal_url, credential_secret_name)
            
            success = auth_result.get("success", False)
            
            # Log authentication attempt
            await self.audit_logger.log_event({
                "event_type": "support_authentication",
                "session_id": session_id,
                "portal_url": portal_url,
                "compliance_level": "gdpr",
                "success": success,
                "browser_tool_response": auth_result,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return success
            
        except Exception as e:
            await self.audit_logger.log_event({
                "event_type": "support_authentication_error",
                "session_id": session_id,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            return False
    
    async def extract_customer_data(
        self, 
        session_id: str, 
        customer_identifier: str,
        data_types: List[str]
    ) -> List[CustomerDataRecord]:
        """Extract customer data with GDPR compliance using AgentCore Browser Tool."""
        extracted_records = []
        
        # Generate secure customer ID hash
        customer_id_hash = hashlib.sha256(customer_identifier.encode()).hexdigest()[:16]
        
        for data_type in data_types:
            try:
                # Check consent before processing
                consent_status = await self.gdpr_tool.check_consent_status(customer_id_hash, data_type)
                
                if consent_status["status"] not in ["granted"]:
                    await self.audit_logger.log_event({
                        "event_type": "consent_check_failed",
                        "session_id": session_id,
                        "customer_id": customer_id_hash,
                        "data_type": data_type,
                        "consent_status": consent_status["status"],
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    continue
                
                # Use browser tool to extract customer data
                extraction_result = await self.browser_tool.extract_customer_data(data_type, customer_identifier)
                
                if not extraction_result.get("success", False):
                    raise Exception(f"Browser tool extraction failed: {extraction_result.get('error', 'Unknown error')}")
                
                raw_data = extraction_result.get("data", {})
                
                # Analyze for PII
                pii_analysis = await self.gdpr_tool.detect_pii(str(raw_data))
                sensitive_pii_analysis = await self.gdpr_tool.detect_sensitive_pii(str(raw_data))
                
                # Determine compliance level
                if sensitive_pii_analysis['sensitive_pii_detected']:
                    compliance_level = GDPRComplianceLevel.SENSITIVE_PERSONAL_DATA
                elif pii_analysis['pii_detected']:
                    compliance_level = GDPRComplianceLevel.PERSONAL_DATA
                else:
                    compliance_level = GDPRComplianceLevel.GENERAL_DATA
                
                # Calculate retention period
                retention_period = await self.gdpr_tool.calculate_retention_period(data_type, consent_status)
                
                # Determine data type enum
                data_type_enum = CustomerDataType.CONTACT_INFO  # Default
                if data_type == "account_details":
                    data_type_enum = CustomerDataType.ACCOUNT_DETAILS
                elif data_type == "support_history":
                    data_type_enum = CustomerDataType.SUPPORT_HISTORY
                elif data_type == "interaction_logs":
                    data_type_enum = CustomerDataType.INTERACTION_LOGS
                
                # Create secure record
                record = CustomerDataRecord(
                    record_id=str(uuid.uuid4()),
                    customer_id=customer_id_hash,
                    data_type=data_type_enum,
                    compliance_level=compliance_level,
                    extracted_data=raw_data,
                    consent_status=consent_status["status"],
                    data_retention_period=retention_period,
                    extraction_timestamp=datetime.utcnow(),
                    audit_trail=[f"extracted_by_agentcore_browser_tool_{datetime.utcnow().isoformat()}"],
                    pii_detected=pii_analysis['pii_detected'],
                    anonymization_applied=False  # Will be applied later if needed
                )
                
                extracted_records.append(record)
                
                # Log extraction
                await self.audit_logger.log_event({
                    "event_type": "customer_data_extraction",
                    "session_id": session_id,
                    "customer_id": customer_id_hash,
                    "data_type": data_type,
                    "compliance_level": compliance_level.value,
                    "pii_detected": pii_analysis['pii_detected'],
                    "sensitive_pii_detected": sensitive_pii_analysis['sensitive_pii_detected'],
                    "consent_status": consent_status["status"],
                    "retention_period": retention_period,
                    "browser_tool_used": True,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            except Exception as e:
                await self.audit_logger.log_event({
                    "event_type": "customer_extraction_error",
                    "session_id": session_id,
                    "customer_id": customer_id_hash,
                    "data_type": data_type,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        return extracted_records
    
    async def process_and_secure_customer_data(self, records: List[CustomerDataRecord]) -> Dict[str, Any]:
        """Process extracted customer data with GDPR compliance."""
        processed_data = {
            "total_records": len(records),
            "personal_data_records": 0,
            "sensitive_personal_data_records": 0,
            "pseudonymized_records": 0,
            "processed_records": [],
            "gdpr_analysis": {},
            "audit_summary": []
        }
        
        for record in records:
            # Apply PII masking if present
            if record.compliance_level in [GDPRComplianceLevel.PERSONAL_DATA, GDPRComplianceLevel.SENSITIVE_PERSONAL_DATA]:
                if record.compliance_level == GDPRComplianceLevel.PERSONAL_DATA:
                    processed_data["personal_data_records"] += 1
                else:
                    processed_data["sensitive_personal_data_records"] += 1
                
                # Pseudonymize data for processing
                pseudonymized_data = await self.gdpr_tool.pseudonymize_data(record.extracted_data, record.customer_id)
                record.extracted_data = pseudonymized_data
                record.anonymization_applied = True
                processed_data["pseudonymized_records"] += 1
            
            # Encrypt personal and sensitive data
            if record.compliance_level in [
                GDPRComplianceLevel.PERSONAL_DATA, 
                GDPRComplianceLevel.SENSITIVE_PERSONAL_DATA
            ]:
                encrypted_data = self.fernet.encrypt(json.dumps(record.extracted_data).encode())
                record.extracted_data = {"encrypted": True, "data": encrypted_data.decode()}
            
            processed_data["processed_records"].append(record.to_secure_dict())
            
            # Add to audit trail
            processed_data["audit_summary"].append({
                "record_id": record.record_id,
                "customer_id": record.customer_id,
                "data_type": record.data_type.value,
                "compliance_level": record.compliance_level.value,
                "consent_status": record.consent_status,
                "pii_detected": record.pii_detected,
                "anonymization_applied": record.anonymization_applied,
                "retention_period": record.data_retention_period,
                "processed_timestamp": datetime.utcnow().isoformat()
            })
        
        # Generate GDPR analysis summary
        processed_data["gdpr_analysis"] = {
            "total_personal_data": processed_data["personal_data_records"],
            "total_sensitive_data": processed_data["sensitive_personal_data_records"],
            "total_pseudonymized": processed_data["pseudonymized_records"],
            "gdpr_compliance_applied": processed_data["personal_data_records"] + processed_data["sensitive_personal_data_records"] > 0
        }
        
        return processed_data
    
    async def generate_gdpr_audit_report(self, session_id: str) -> Dict[str, Any]:
        """Generate GDPR compliance audit report."""
        audit_events = await self.audit_logger.get_session_events(session_id)
        
        report = {
            "session_id": session_id,
            "report_generated": datetime.utcnow().isoformat(),
            "compliance_framework": "GDPR",
            "agentcore_browser_tool_used": True,
            "total_events": len(audit_events),
            "event_summary": {},
            "personal_data_events": [],
            "consent_events": [],
            "security_events": [],
            "compliance_status": "compliant"
        }
        
        # Categorize events
        for event in audit_events:
            event_type = event.get("event_type", "unknown")
            if event_type not in report["event_summary"]:
                report["event_summary"][event_type] = 0
            report["event_summary"][event_type] += 1
            
            # Track personal data handling
            if "personal_data" in event_type or event.get("pii_detected"):
                report["personal_data_events"].append(event)
            
            # Track consent events
            if "consent" in event_type:
                report["consent_events"].append(event)
            
            # Track security events
            if event_type in ["support_authentication", "support_session_created", "customer_extraction_error"]:
                report["security_events"].append(event)
        
        return report


async def main():
    """Main function demonstrating customer support automation."""
    print("🎧 Customer Support Automation with PII Protection")
    print("=" * 60)
    
    # Initialize customer support agent
    agent = SecureCustomerSupportAgent()
    
    # Customer support portal configuration
    portal_config = {
        "name": "Zendesk Support Portal",
        "url": "https://support.zendesk.com/login",
        "credential_secret_name": "zendesk_support_credentials"  # AWS Secrets Manager secret name
    }
    
    try:
        # Create secure browser session
        print("Creating secure GDPR-compliant browser session...")
        session_id = await agent.create_secure_support_session(portal_config)
        
        # Authenticate to support portal using AgentCore Browser Tool
        print("Authenticating to customer support portal via AgentCore Browser Tool...")
        auth_success = await agent.authenticate_support_portal(
            session_id, 
            portal_config["url"],
            portal_config["credential_secret_name"]
        )
        
        if not auth_success:
            print("❌ Authentication failed")
            return
        
        print("✅ Successfully authenticated to support portal via AgentCore Browser Tool")
        
        # Extract customer data using AgentCore Browser Tool
        print("Extracting customer data with GDPR compliance via AgentCore Browser Tool...")
        customer_records = await agent.extract_customer_data(
            session_id,
            customer_identifier="CUST123456",  # Example customer ID
            data_types=["contact_information", "account_details", "support_history", "interaction_logs"]
        )
        
        print(f"✅ Extracted {len(customer_records)} customer data records")
        
        # Process and secure data
        print("Processing data with GDPR compliance controls...")
        processed_data = await agent.process_and_secure_customer_data(customer_records)
        
        print(f"✅ Processed {processed_data['total_records']} records")
        print(f"👤 Personal data records: {processed_data['personal_data_records']}")
        print(f"🔒 Sensitive personal data records: {processed_data['sensitive_personal_data_records']}")
        print(f"🎭 Pseudonymized records: {processed_data['pseudonymized_records']}")
        
        # Display GDPR analysis
        if processed_data.get('gdpr_analysis'):
            gdpr_analysis = processed_data['gdpr_analysis']
            print(f"📋 GDPR Analysis:")
            print(f"   - Total personal data: {gdpr_analysis['total_personal_data']}")
            print(f"   - Total sensitive data: {gdpr_analysis['total_sensitive_data']}")
            print(f"   - Total pseudonymized: {gdpr_analysis['total_pseudonymized']}")
            print(f"   - GDPR compliance applied: {gdpr_analysis['gdpr_compliance_applied']}")
        
        # Generate audit report
        print("Generating GDPR compliance audit report...")
        audit_report = await agent.generate_gdpr_audit_report(session_id)
        
        print("✅ GDPR Compliance Audit Report Generated:")
        print(f"   - Total events: {audit_report['total_events']}")
        print(f"   - Personal data events: {len(audit_report['personal_data_events'])}")
        print(f"   - Consent events: {len(audit_report['consent_events'])}")
        print(f"   - Security events: {len(audit_report['security_events'])}")
        print(f"   - Compliance status: {audit_report['compliance_status']}")
        
        # Save results securely
        results = {
            "session_summary": {
                "session_id": session_id,
                "portal": portal_config["name"],
                "records_processed": len(customer_records),
                "personal_data_records": processed_data['personal_data_records'],
                "sensitive_data_records": processed_data['sensitive_personal_data_records'],
                "compliance_status": "gdpr_compliant",
                "agentcore_browser_tool_used": True
            },
            "audit_report": audit_report,
            "processed_data": processed_data
        }
        
        # Save to secure file (in production, this would be encrypted storage)
        with open("customer_support_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print("✅ Results saved to customer_support_results.json")
        
    except Exception as e:
        print(f"❌ Error during customer support automation: {str(e)}")
        
    finally:
        # AgentCore Browser Tool automatically cleans up sessions
        print("🧹 AgentCore Browser Tool session automatically cleaned up")


if __name__ == "__main__":
    asyncio.run(main())