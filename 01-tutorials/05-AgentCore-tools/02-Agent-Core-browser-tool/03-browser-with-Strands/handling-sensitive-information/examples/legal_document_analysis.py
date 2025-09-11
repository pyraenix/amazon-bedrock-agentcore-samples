#!/usr/bin/env python3
"""
Legal Document Analysis with Attorney-Client Privilege Protection
===============================================================

This example demonstrates how to use Strands agents with AgentCore Browser Tool
for confidential legal document processing with attorney-client privilege protection. It showcases:

1. Secure legal portal access with multi-factor authentication
2. Attorney-client privilege protection and confidentiality controls
3. Legal document classification and sensitivity analysis
4. Comprehensive audit logging for legal compliance
5. Privileged communication protection

Requirements:
- Attorney-client privilege protection
- Confidentiality controls for legal documents
- Comprehensive audit trails for legal compliance
- Document classification and sensitivity analysis
- Secure communication channel protection
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


class LegalPrivilegeLevel(Enum):
    """Legal privilege levels for different document types."""
    ATTORNEY_CLIENT_PRIVILEGED = "attorney_client_privileged"  # Highest protection
    WORK_PRODUCT = "work_product"  # High protection - attorney work product
    CONFIDENTIAL_LEGAL = "confidential_legal"  # Confidential legal documents
    SENSITIVE_LEGAL = "sensitive_legal"  # Sensitive legal information
    GENERAL_LEGAL = "general_legal"  # Standard legal documents


class DocumentType(Enum):
    """Types of legal documents."""
    CONTRACT = "contract"
    LITIGATION_DOCUMENT = "litigation_document"
    LEGAL_MEMO = "legal_memo"
    CLIENT_COMMUNICATION = "client_communication"
    COURT_FILING = "court_filing"
    DISCOVERY_DOCUMENT = "discovery_document"
    SETTLEMENT_AGREEMENT = "settlement_agreement"
    LEGAL_OPINION = "legal_opinion"


@dataclass
class LegalDocumentRecord:
    """Secure legal document record with privilege protection."""
    document_id: str  # Hashed identifier
    client_id: str  # Hashed client identifier
    document_type: DocumentType
    privilege_level: LegalPrivilegeLevel
    extracted_content: Dict[str, Any]
    privilege_protected: bool
    extraction_timestamp: datetime
    audit_trail: List[str]
    confidentiality_score: float
    attorney_work_product: bool
    
    def to_secure_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with privileged content protected."""
        secure_data = asdict(self)
        if self.privilege_level == LegalPrivilegeLevel.ATTORNEY_CLIENT_PRIVILEGED:
            secure_data['extracted_content'] = {"status": "privileged", "access": "attorney_only"}
        elif self.privilege_level == LegalPrivilegeLevel.WORK_PRODUCT:
            secure_data['extracted_content'] = {"status": "work_product", "access": "legal_team_only"}
        return secure_data


class LegalPrivilegeTool(PythonAgentTool):
    """Custom Strands tool for legal privilege and confidentiality validation."""
    
    def __init__(self):
        super().__init__(name="legal_privilege_validator")
        self.privileged_patterns = [
            r'attorney[- ]client',
            r'privileged\s+communication',
            r'confidential\s+legal',
            r'work\s+product',
            r'legal\s+advice',
            r'counsel\s+recommends',
            r'attorney\s+opinion',
            r'legal\s+strategy',
        ]
        self.sensitive_legal_patterns = [
            r'settlement\s+amount',
            r'damages\s+calculation',
            r'litigation\s+strategy',
            r'witness\s+statement',
            r'expert\s+opinion',
            r'discovery\s+response',
            r'plea\s+agreement',
            r'confidential\s+settlement',
        ]
        self.encryption_manager = EncryptionManager()
        
    async def detect_attorney_client_privilege(self, content: str) -> Dict[str, Any]:
        """Detect attorney-client privileged content."""
        privileged_items = []
        
        for pattern in self.privileged_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                privileged_items.extend([{
                    'type': 'attorney_client_privilege',
                    'pattern': pattern,
                    'matches_count': len(matches),
                    'severity': 'critical'
                }])
        
        return {
            'privileged_content_detected': len(privileged_items) > 0,
            'privileged_items': privileged_items,
            'privilege_level': LegalPrivilegeLevel.ATTORNEY_CLIENT_PRIVILEGED.value if privileged_items else LegalPrivilegeLevel.GENERAL_LEGAL.value
        }
    
    async def detect_work_product(self, content: str) -> Dict[str, Any]:
        """Detect attorney work product content."""
        work_product_indicators = [
            r'legal\s+analysis',
            r'case\s+strategy',
            r'litigation\s+plan',
            r'attorney\s+notes',
            r'legal\s+research',
            r'case\s+preparation',
            r'trial\s+strategy',
            r'legal\s+memorandum',
        ]
        
        work_product_items = []
        for pattern in work_product_indicators:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                work_product_items.extend([{
                    'type': 'work_product',
                    'pattern': pattern,
                    'matches_count': len(matches),
                    'severity': 'high'
                }])
        
        return {
            'work_product_detected': len(work_product_items) > 0,
            'work_product_items': work_product_items
        }
    
    async def detect_sensitive_legal_content(self, content: str) -> Dict[str, Any]:
        """Detect sensitive legal content."""
        sensitive_items = []
        
        for pattern in self.sensitive_legal_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                sensitive_items.extend([{
                    'type': 'sensitive_legal',
                    'pattern': pattern,
                    'matches_count': len(matches),
                    'severity': 'medium'
                }])
        
        return {
            'sensitive_content_detected': len(sensitive_items) > 0,
            'sensitive_items': sensitive_items
        }
    
    async def mask_privileged_content(self, content: str) -> str:
        """Mask privileged content to protect attorney-client privilege."""
        masked_content = content
        
        # Mask attorney-client privileged communications
        for pattern in self.privileged_patterns:
            masked_content = re.sub(pattern, '[ATTORNEY-CLIENT PRIVILEGED]', masked_content, flags=re.IGNORECASE)
        
        # Mask sensitive legal information
        for pattern in self.sensitive_legal_patterns:
            masked_content = re.sub(pattern, '[CONFIDENTIAL LEGAL INFORMATION]', masked_content, flags=re.IGNORECASE)
        
        # Mask monetary amounts in legal context
        masked_content = re.sub(r'\$[\d,]+\.?\d*', '[MONETARY AMOUNT REDACTED]', masked_content)
        
        # Mask dates that might be sensitive
        masked_content = re.sub(r'\b\d{1,2}/\d{1,2}/\d{4}\b', '[DATE REDACTED]', masked_content)
        
        return masked_content
    
    async def calculate_confidentiality_score(self, document_data: Dict[str, Any]) -> float:
        """Calculate confidentiality risk score for legal document."""
        score = 0.0
        content = str(document_data)
        
        # Check for privileged content
        if any(re.search(pattern, content, re.IGNORECASE) for pattern in self.privileged_patterns):
            score += 0.4
        
        # Check for work product
        work_product_patterns = [r'legal\s+analysis', r'case\s+strategy', r'attorney\s+notes']
        if any(re.search(pattern, content, re.IGNORECASE) for pattern in work_product_patterns):
            score += 0.3
        
        # Check for sensitive legal content
        if any(re.search(pattern, content, re.IGNORECASE) for pattern in self.sensitive_legal_patterns):
            score += 0.2
        
        # Check for client-specific information
        client_patterns = [r'client\s+name', r'case\s+number', r'matter\s+number']
        if any(re.search(pattern, content, re.IGNORECASE) for pattern in client_patterns):
            score += 0.1
        
        return min(score, 1.0)


class AgentCoreBrowserTool(AgentTool):
    """Custom Strands tool that integrates with AgentCore Browser Tool for legal document processing."""
    
    def __init__(self, region: str = "us-east-1"):
        super().__init__(name="agentcore_browser_legal")
        self.region = region
        self.bedrock_agent = boto3.client('bedrock-agent-runtime', region_name=region)
        self.secrets_client = boto3.client('secretsmanager', region_name=region)
        
    async def invoke_browser_tool(self, instructions: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the AgentCore Browser Tool with legal-specific security configurations."""
        try:
            # Prepare the browser tool invocation with enhanced security for legal documents
            tool_input = {
                "toolName": "AgentCoreBrowserTool",
                "toolInput": {
                    "instructions": json.dumps(instructions),
                    "securityConfig": {
                        "isolationLevel": "maximum",
                        "dataProtection": "attorney_client_privileged",
                        "auditLogging": True,
                        "sessionTimeout": 1800,  # 30 minutes for legal sessions
                        "screenshotDisabled": True,  # Prevent privileged content in screenshots
                        "clipboardDisabled": True,  # Prevent privileged content in clipboard
                        "networkMonitoring": True,  # Monitor for data exfiltration
                        "memoryProtection": True,  # Protect against memory dumps
                        "privilegeProtection": True  # Special protection for attorney-client privilege
                    }
                }
            }
            
            # Invoke the browser tool via Bedrock Agent Runtime
            response = self.bedrock_agent.invoke_agent(
                agentId="your-legal-agent-id",  # Replace with actual agent ID
                agentAliasId="your-legal-alias-id",  # Replace with actual alias ID
                sessionId=f"legal-session-{uuid.uuid4()}",
                inputText=json.dumps(tool_input)
            )
            
            # Parse the response
            result = json.loads(response['completion'])
            return result
            
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def navigate_and_authenticate_legal_portal(self, portal_url: str, credential_secret_name: str) -> Dict[str, Any]:
        """Navigate to legal portal and authenticate securely."""
        # Retrieve credentials from AWS Secrets Manager
        try:
            secret_response = self.secrets_client.get_secret_value(SecretId=credential_secret_name)
            credentials = json.loads(secret_response['SecretString'])
        except Exception as e:
            return {"error": f"Failed to retrieve legal credentials: {str(e)}", "success": False}
        
        # Prepare browser instructions for legal portal navigation
        instructions = {
            "action": "navigate_and_authenticate_legal",
            "url": portal_url,
            "privilegeLevel": "attorney_client",
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
                    "privilegeProtected": True
                },
                {
                    "action": "fill_secure",
                    "selector": "input[type='password']",
                    "value": credentials['password'],
                    "privilegeProtected": True
                },
                {
                    "action": "click",
                    "selector": "button[type='submit'], input[type='submit']"
                },
                {
                    "action": "wait_for_element",
                    "selector": ".dashboard, .case-list, .document-library, .legal-workspace",
                    "timeout": 15000
                }
            ]
        }
        
        return await self.invoke_browser_tool(instructions)
    
    async def extract_legal_documents(self, document_type: str, case_id: str = None) -> Dict[str, Any]:
        """Extract legal documents using the browser tool with privilege protection."""
        # Define extraction instructions based on document type
        extraction_steps = {
            "contracts": [
                {"action": "click", "selector": ".contracts, .contract-library"},
                {"action": "wait_for_load_state", "state": "networkidle"},
                {"action": "extract_data", "selector": ".contract, .contract-document", "attributes": ["text", "data-*"], "privilegeProtected": True}
            ],
            "litigation": [
                {"action": "click", "selector": ".litigation, .case-documents"},
                {"action": "wait_for_load_state", "state": "networkidle"},
                {"action": "extract_data", "selector": ".litigation-doc, .court-filing", "attributes": ["text"], "privilegeProtected": True}
            ],
            "client_communications": [
                {"action": "click", "selector": ".communications, .client-emails"},
                {"action": "wait_for_load_state", "state": "networkidle"},
                {"action": "extract_data", "selector": ".email, .communication", "attributes": ["text"], "privilegeProtected": True}
            ],
            "legal_memos": [
                {"action": "click", "selector": ".memos, .legal-memoranda"},
                {"action": "wait_for_load_state", "state": "networkidle"},
                {"action": "extract_data", "selector": ".memo, .legal-memo", "attributes": ["text"], "privilegeProtected": True}
            ]
        }
        
        if document_type not in extraction_steps:
            return {"error": f"Unknown legal document type: {document_type}", "success": False}
        
        instructions = {
            "action": "extract_legal_documents",
            "document_type": document_type,
            "case_id": case_id,
            "privilegeLevel": "attorney_client",
            "steps": extraction_steps[document_type]
        }
        
        return await self.invoke_browser_tool(instructions)


class SecureLegalAgent:
    """Strands agent specialized for attorney-client privileged legal document processing."""
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.browser_tool = AgentCoreBrowserTool(region=region)
        self.privilege_tool = LegalPrivilegeTool()
        self.audit_logger = AuditLogger(service_name="legal_agent")
        
        # Initialize strong encryption for legal documents
        self.encryption_key = self._derive_legal_encryption_key()
        self.fernet = Fernet(self.encryption_key)
        
        # Initialize Strands agent with legal-specific configuration
        self.agent = Agent(
            name="legal_document_processor",
            tools=[self.browser_tool, self.privilege_tool],
            security_context=SecurityContext(
                compliance_level="attorney_client_privileged",
                data_classification="legal_confidential",
                audit_required=True,
                encryption_required=True,
                privilege_protection=True
            )
        )
    
    def _derive_legal_encryption_key(self) -> bytes:
        """Derive strong encryption key for legal documents."""
        password = b"legal_document_encryption_key_2024_privileged"
        salt = b"attorney_client_privilege_salt"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    async def create_secure_legal_session(self, portal_config: Dict[str, Any]) -> str:
        """Create a secure browser session for legal portal access."""
        session_id = f"legal-session-{uuid.uuid4()}"
        
        # Log session creation for audit
        await self.audit_logger.log_event({
            "event_type": "legal_session_created",
            "session_id": session_id,
            "portal": portal_config.get("name", "unknown"),
            "privilege_level": "attorney_client",
            "security_controls": ["isolation_maximum", "privilege_protected", "audit_logging"],
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return session_id
    
    async def authenticate_legal_portal(
        self, 
        session_id: str, 
        portal_url: str,
        credential_secret_name: str
    ) -> bool:
        """Securely authenticate to legal portal using AgentCore Browser Tool."""
        try:
            # Use the browser tool to navigate and authenticate
            auth_result = await self.browser_tool.navigate_and_authenticate_legal_portal(portal_url, credential_secret_name)
            
            success = auth_result.get("success", False)
            
            # Log authentication attempt
            await self.audit_logger.log_event({
                "event_type": "legal_authentication",
                "session_id": session_id,
                "portal_url": portal_url,
                "privilege_level": "attorney_client",
                "success": success,
                "browser_tool_response": auth_result,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return success
            
        except Exception as e:
            await self.audit_logger.log_event({
                "event_type": "legal_authentication_error",
                "session_id": session_id,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            return False
    
    async def extract_legal_documents(
        self, 
        session_id: str, 
        case_identifier: str,
        document_types: List[str]
    ) -> List[LegalDocumentRecord]:
        """Extract legal documents with attorney-client privilege protection."""
        extracted_records = []
        
        # Generate secure case ID hash
        case_id_hash = hashlib.sha256(case_identifier.encode()).hexdigest()[:16]
        
        for doc_type in document_types:
            try:
                # Use browser tool to extract documents
                extraction_result = await self.browser_tool.extract_legal_documents(doc_type, case_identifier)
                
                if not extraction_result.get("success", False):
                    raise Exception(f"Browser tool extraction failed: {extraction_result.get('error', 'Unknown error')}")
                
                raw_data = extraction_result.get("data", {})
                
                # Analyze for attorney-client privilege
                privilege_analysis = await self.privilege_tool.detect_attorney_client_privilege(str(raw_data))
                work_product_analysis = await self.privilege_tool.detect_work_product(str(raw_data))
                sensitive_analysis = await self.privilege_tool.detect_sensitive_legal_content(str(raw_data))
                
                # Determine privilege level
                if privilege_analysis['privileged_content_detected']:
                    privilege_level = LegalPrivilegeLevel.ATTORNEY_CLIENT_PRIVILEGED
                elif work_product_analysis['work_product_detected']:
                    privilege_level = LegalPrivilegeLevel.WORK_PRODUCT
                elif sensitive_analysis['sensitive_content_detected']:
                    privilege_level = LegalPrivilegeLevel.CONFIDENTIAL_LEGAL
                else:
                    privilege_level = LegalPrivilegeLevel.GENERAL_LEGAL
                
                # Calculate confidentiality score
                confidentiality_score = await self.privilege_tool.calculate_confidentiality_score(raw_data)
                
                # Determine document type enum
                doc_type_enum = DocumentType.CONTRACT  # Default
                if doc_type == "litigation":
                    doc_type_enum = DocumentType.LITIGATION_DOCUMENT
                elif doc_type == "client_communications":
                    doc_type_enum = DocumentType.CLIENT_COMMUNICATION
                elif doc_type == "legal_memos":
                    doc_type_enum = DocumentType.LEGAL_MEMO
                
                # Create secure record
                record = LegalDocumentRecord(
                    document_id=str(uuid.uuid4()),
                    client_id=case_id_hash,
                    document_type=doc_type_enum,
                    privilege_level=privilege_level,
                    extracted_content=raw_data,
                    privilege_protected=privilege_level in [LegalPrivilegeLevel.ATTORNEY_CLIENT_PRIVILEGED, LegalPrivilegeLevel.WORK_PRODUCT],
                    extraction_timestamp=datetime.utcnow(),
                    audit_trail=[f"extracted_by_agentcore_browser_tool_{datetime.utcnow().isoformat()}"],
                    confidentiality_score=confidentiality_score,
                    attorney_work_product=work_product_analysis['work_product_detected']
                )
                
                extracted_records.append(record)
                
                # Log extraction
                await self.audit_logger.log_event({
                    "event_type": "legal_document_extraction",
                    "session_id": session_id,
                    "case_id": case_id_hash,
                    "document_type": doc_type,
                    "privilege_level": privilege_level.value,
                    "privileged_content_detected": privilege_analysis['privileged_content_detected'],
                    "work_product_detected": work_product_analysis['work_product_detected'],
                    "confidentiality_score": confidentiality_score,
                    "browser_tool_used": True,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            except Exception as e:
                await self.audit_logger.log_event({
                    "event_type": "legal_extraction_error",
                    "session_id": session_id,
                    "case_id": case_id_hash,
                    "document_type": doc_type,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        return extracted_records
    
    async def process_and_secure_legal_data(self, records: List[LegalDocumentRecord]) -> Dict[str, Any]:
        """Process extracted legal data with attorney-client privilege protection."""
        processed_data = {
            "total_records": len(records),
            "privileged_records": 0,
            "work_product_records": 0,
            "confidential_records": 0,
            "processed_records": [],
            "privilege_analysis": {},
            "audit_summary": []
        }
        
        for record in records:
            # Mask privileged content if present
            if record.privilege_level == LegalPrivilegeLevel.ATTORNEY_CLIENT_PRIVILEGED:
                processed_data["privileged_records"] += 1
                
                # Mask privileged content
                masked_data = {}
                for key, value in record.extracted_content.items():
                    if isinstance(value, str):
                        masked_data[key] = await self.privilege_tool.mask_privileged_content(value)
                    elif isinstance(value, list):
                        masked_data[key] = [
                            await self.privilege_tool.mask_privileged_content(str(item)) if isinstance(item, str) else item
                            for item in value
                        ]
                    else:
                        masked_data[key] = value
                
                record.extracted_content = masked_data
            
            # Handle work product
            if record.privilege_level == LegalPrivilegeLevel.WORK_PRODUCT:
                processed_data["work_product_records"] += 1
            
            # Handle confidential legal documents
            if record.privilege_level == LegalPrivilegeLevel.CONFIDENTIAL_LEGAL:
                processed_data["confidential_records"] += 1
            
            # Encrypt privileged and confidential records
            if record.privilege_level in [
                LegalPrivilegeLevel.ATTORNEY_CLIENT_PRIVILEGED, 
                LegalPrivilegeLevel.WORK_PRODUCT,
                LegalPrivilegeLevel.CONFIDENTIAL_LEGAL
            ]:
                encrypted_data = self.fernet.encrypt(json.dumps(record.extracted_content).encode())
                record.extracted_content = {"encrypted": True, "data": encrypted_data.decode()}
            
            processed_data["processed_records"].append(record.to_secure_dict())
            
            # Add to audit trail
            processed_data["audit_summary"].append({
                "document_id": record.document_id,
                "client_id": record.client_id,
                "document_type": record.document_type.value,
                "privilege_level": record.privilege_level.value,
                "confidentiality_score": record.confidentiality_score,
                "privilege_protected": record.privilege_protected,
                "attorney_work_product": record.attorney_work_product,
                "processed_timestamp": datetime.utcnow().isoformat()
            })
        
        # Generate privilege analysis summary
        processed_data["privilege_analysis"] = {
            "total_privileged": processed_data["privileged_records"],
            "total_work_product": processed_data["work_product_records"],
            "total_confidential": processed_data["confidential_records"],
            "privilege_protection_applied": processed_data["privileged_records"] + processed_data["work_product_records"] > 0
        }
        
        return processed_data
    
    async def generate_legal_audit_report(self, session_id: str) -> Dict[str, Any]:
        """Generate legal compliance audit report with privilege protection."""
        audit_events = await self.audit_logger.get_session_events(session_id)
        
        report = {
            "session_id": session_id,
            "report_generated": datetime.utcnow().isoformat(),
            "compliance_framework": "Attorney-Client Privilege",
            "agentcore_browser_tool_used": True,
            "total_events": len(audit_events),
            "event_summary": {},
            "privileged_content_events": [],
            "work_product_events": [],
            "security_events": [],
            "compliance_status": "compliant"
        }
        
        # Categorize events
        for event in audit_events:
            event_type = event.get("event_type", "unknown")
            if event_type not in report["event_summary"]:
                report["event_summary"][event_type] = 0
            report["event_summary"][event_type] += 1
            
            # Track privileged content handling
            if "privileged_content" in event_type or event.get("privileged_content_detected"):
                report["privileged_content_events"].append(event)
            
            # Track work product handling
            if "work_product" in event_type or event.get("work_product_detected"):
                report["work_product_events"].append(event)
            
            # Track security events
            if event_type in ["legal_authentication", "legal_session_created", "legal_extraction_error"]:
                report["security_events"].append(event)
        
        return report


async def main():
    """Main function demonstrating legal document analysis."""
    print("⚖️  Legal Document Analysis with Attorney-Client Privilege Protection")
    print("=" * 70)
    
    # Initialize legal agent
    agent = SecureLegalAgent()
    
    # Legal portal configuration
    portal_config = {
        "name": "LexisNexis Legal Portal",
        "url": "https://legal.lexisnexis.com/login",
        "credential_secret_name": "lexisnexis_legal_credentials"  # AWS Secrets Manager secret name
    }
    
    try:
        # Create secure browser session
        print("Creating secure attorney-client privileged browser session...")
        session_id = await agent.create_secure_legal_session(portal_config)
        
        # Authenticate to legal portal using AgentCore Browser Tool
        print("Authenticating to legal portal via AgentCore Browser Tool...")
        auth_success = await agent.authenticate_legal_portal(
            session_id, 
            portal_config["url"],
            portal_config["credential_secret_name"]
        )
        
        if not auth_success:
            print("❌ Authentication failed")
            return
        
        print("✅ Successfully authenticated to legal portal via AgentCore Browser Tool")
        
        # Extract legal documents using AgentCore Browser Tool
        print("Extracting legal documents with privilege protection via AgentCore Browser Tool...")
        legal_records = await agent.extract_legal_documents(
            session_id,
            case_identifier="CASE2024-001",  # Example case ID
            document_types=["contracts", "litigation", "client_communications", "legal_memos"]
        )
        
        print(f"✅ Extracted {len(legal_records)} legal document records")
        
        # Process and secure data
        print("Processing data with attorney-client privilege protection...")
        processed_data = await agent.process_and_secure_legal_data(legal_records)
        
        print(f"✅ Processed {processed_data['total_records']} records")
        print(f"🔒 Attorney-client privileged records: {processed_data['privileged_records']}")
        print(f"📋 Work product records: {processed_data['work_product_records']}")
        print(f"🔐 Confidential legal records: {processed_data['confidential_records']}")
        
        # Display privilege analysis
        if processed_data.get('privilege_analysis'):
            privilege_analysis = processed_data['privilege_analysis']
            print(f"⚖️  Privilege Analysis:")
            print(f"   - Total privileged: {privilege_analysis['total_privileged']}")
            print(f"   - Total work product: {privilege_analysis['total_work_product']}")
            print(f"   - Total confidential: {privilege_analysis['total_confidential']}")
            print(f"   - Privilege protection applied: {privilege_analysis['privilege_protection_applied']}")
        
        # Generate audit report
        print("Generating legal compliance audit report...")
        audit_report = await agent.generate_legal_audit_report(session_id)
        
        print("✅ Legal Compliance Audit Report Generated:")
        print(f"   - Total events: {audit_report['total_events']}")
        print(f"   - Privileged content events: {len(audit_report['privileged_content_events'])}")
        print(f"   - Work product events: {len(audit_report['work_product_events'])}")
        print(f"   - Security events: {len(audit_report['security_events'])}")
        print(f"   - Compliance status: {audit_report['compliance_status']}")
        
        # Save results securely
        results = {
            "session_summary": {
                "session_id": session_id,
                "portal": portal_config["name"],
                "records_processed": len(legal_records),
                "privileged_records": processed_data['privileged_records'],
                "work_product_records": processed_data['work_product_records'],
                "compliance_status": "attorney_client_privileged",
                "agentcore_browser_tool_used": True
            },
            "audit_report": audit_report,
            "processed_data": processed_data
        }
        
        # Save to secure file (in production, this would be encrypted storage)
        with open("legal_processing_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print("✅ Results saved to legal_processing_results.json")
        
    except Exception as e:
        print(f"❌ Error during legal document analysis: {str(e)}")
        
    finally:
        # AgentCore Browser Tool automatically cleans up sessions
        print("🧹 AgentCore Browser Tool session automatically cleaned up")


if __name__ == "__main__":
    asyncio.run(main())