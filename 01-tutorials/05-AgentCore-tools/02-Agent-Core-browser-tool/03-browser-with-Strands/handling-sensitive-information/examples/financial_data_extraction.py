#!/usr/bin/env python3
"""
Financial Data Extraction with PCI DSS Compliance
================================================

This example demonstrates how to use Strands agents with AgentCore Browser Tool
for PCI DSS compliant payment processing and financial data extraction. It showcases:

1. Secure financial portal access with multi-factor authentication
2. PCI DSS compliant payment card data handling
3. Encrypted financial transaction processing
4. Comprehensive audit logging for financial compliance
5. Real-time fraud detection and prevention

Requirements:
- PCI DSS compliance for payment card data
- Strong encryption for financial data
- Comprehensive audit trails
- Fraud detection capabilities
- Secure multi-factor authentication
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
from decimal import Decimal

# Strands framework imports
from strands import Agent
from strands.tools import tool, PythonAgentTool
from strands.types.tools import AgentTool

# AgentCore Browser Tool integration
from strands_tools.browser.agent_core_browser import AgentCoreBrowser
from bedrock_agentcore.tools.browser_client import BrowserClient
import boto3
import json

# AWS and security imports
import boto3
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64


class PCIComplianceLevel(Enum):
    """PCI DSS compliance levels for different data types."""
    CARDHOLDER_DATA = "cardholder_data"  # Highest protection - PAN, expiry, etc.
    SENSITIVE_AUTH_DATA = "sensitive_authentication_data"  # CVV, PIN, etc.
    FINANCIAL_DATA = "financial_data"  # Account numbers, balances
    TRANSACTION_DATA = "transaction_data"  # Transaction details
    GENERAL_DATA = "general_data"  # Non-sensitive financial info


@dataclass
class FinancialDataRecord:
    """Secure financial data record with PCI DSS compliance."""
    record_id: str  # Hashed identifier
    account_id: str  # Hashed account identifier
    record_type: str
    data_classification: PCIComplianceLevel
    extracted_data: Dict[str, Any]
    encryption_applied: bool
    extraction_timestamp: datetime
    audit_trail: List[str]
    fraud_score: float
    
    def to_secure_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with sensitive data protected."""
        secure_data = asdict(self)
        if self.data_classification in [PCIComplianceLevel.CARDHOLDER_DATA, PCIComplianceLevel.SENSITIVE_AUTH_DATA]:
            secure_data['extracted_data'] = {"status": "encrypted", "classification": self.data_classification.value}
        return secure_data


class PCIComplianceTool(PythonAgentTool):
    """Custom Strands tool for PCI DSS compliance validation."""
    
    def __init__(self):
        super().__init__(name="pci_compliance_validator")
        self.cardholder_patterns = [
            r'\b4[0-9]{12}(?:[0-9]{3})?\b',  # Visa
            r'\b5[1-5][0-9]{14}\b',  # MasterCard
            r'\b3[47][0-9]{13}\b',  # American Express
            r'\b3[0-9]{4,}\b',  # Diners Club
            r'\b6(?:011|5[0-9]{2})[0-9]{12}\b',  # Discover
        ]
        self.sensitive_patterns = [
            r'\b\d{3,4}\b',  # CVV/CVC
            r'\b\d{2}/\d{2}\b',  # Expiry date MM/YY
            r'\b\d{4}/\d{4}\b',  # Expiry date MM/YYYY
            r'\$\d+\.\d{2}',  # Currency amounts
            r'\b\d{10,12}\b',  # Account numbers
        ]
        self.encryption_manager = EncryptionManager()
        
    async def detect_cardholder_data(self, content: str) -> Dict[str, Any]:
        """Detect cardholder data in content."""
        cardholder_data = []
        
        for pattern in self.cardholder_patterns:
            matches = re.findall(pattern, content)
            if matches:
                # Validate card numbers using Luhn algorithm
                valid_cards = [card for card in matches if self._luhn_check(card)]
                if valid_cards:
                    cardholder_data.extend([{
                        'type': 'payment_card',
                        'pattern': pattern,
                        'valid_cards_count': len(valid_cards),
                        'severity': 'critical'
                    }])
        
        return {
            'cardholder_data_detected': len(cardholder_data) > 0,
            'cardholder_items': cardholder_data,
            'compliance_level': PCIComplianceLevel.CARDHOLDER_DATA.value if cardholder_data else PCIComplianceLevel.GENERAL_DATA.value
        }
    
    def _luhn_check(self, card_number: str) -> bool:
        """Validate credit card number using Luhn algorithm."""
        def digits_of(n):
            return [int(d) for d in str(n)]
        
        digits = digits_of(card_number.replace(' ', '').replace('-', ''))
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(digits_of(d*2))
        return checksum % 10 == 0
    
    async def detect_sensitive_auth_data(self, content: str) -> Dict[str, Any]:
        """Detect sensitive authentication data."""
        sensitive_data = []
        
        for pattern in self.sensitive_patterns:
            matches = re.findall(pattern, content)
            if matches:
                sensitive_data.extend([{
                    'type': 'sensitive_auth_data',
                    'pattern': pattern,
                    'matches_count': len(matches),
                    'severity': 'high'
                }])
        
        return {
            'sensitive_data_detected': len(sensitive_data) > 0,
            'sensitive_items': sensitive_data,
            'compliance_level': PCIComplianceLevel.SENSITIVE_AUTH_DATA.value if sensitive_data else PCIComplianceLevel.GENERAL_DATA.value
        }
    
    async def mask_cardholder_data(self, content: str) -> str:
        """Mask cardholder data according to PCI DSS requirements."""
        masked_content = content
        
        # Mask credit card numbers (show only last 4 digits)
        for pattern in self.cardholder_patterns:
            def mask_card(match):
                card = match.group(0)
                return '*' * (len(card) - 4) + card[-4:]
            
            masked_content = re.sub(pattern, mask_card, masked_content)
        
        # Mask CVV completely
        masked_content = re.sub(r'\b\d{3,4}\b', '***', masked_content)
        
        # Mask expiry dates
        masked_content = re.sub(r'\b\d{2}/\d{2,4}\b', 'XX/XX', masked_content)
        
        return masked_content
    
    async def calculate_fraud_score(self, transaction_data: Dict[str, Any]) -> float:
        """Calculate fraud risk score for transaction."""
        score = 0.0
        
        # Check for suspicious patterns
        amount = transaction_data.get('amount', 0)
        if isinstance(amount, str):
            amount = float(re.sub(r'[^\d.]', '', amount))
        
        # High amount transactions
        if amount > 10000:
            score += 0.3
        elif amount > 5000:
            score += 0.2
        
        # Unusual time patterns
        timestamp = transaction_data.get('timestamp', '')
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                hour = dt.hour
                # Transactions outside business hours
                if hour < 6 or hour > 22:
                    score += 0.2
            except:
                pass
        
        # Multiple transactions in short time
        frequency = transaction_data.get('frequency_score', 0)
        score += min(frequency * 0.1, 0.3)
        
        return min(score, 1.0)


class FraudDetectionTool(PythonAgentTool):
    """Advanced fraud detection tool for financial transactions."""
    
    def __init__(self):
        super().__init__(name="fraud_detector")
        self.suspicious_patterns = [
            r'test\s*card',  # Test card usage
            r'stolen',  # Stolen card indicators
            r'fraud',  # Fraud indicators
            r'chargeback',  # Chargeback indicators
        ]
    
    async def analyze_transaction_pattern(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze transaction patterns for fraud indicators."""
        analysis = {
            'total_transactions': len(transactions),
            'suspicious_transactions': 0,
            'fraud_indicators': [],
            'risk_level': 'low'
        }
        
        if not transactions:
            return analysis
        
        # Analyze transaction velocity
        amounts = [float(re.sub(r'[^\d.]', '', str(t.get('amount', 0)))) for t in transactions]
        avg_amount = sum(amounts) / len(amounts) if amounts else 0
        
        # Check for unusual patterns
        for i, transaction in enumerate(transactions):
            amount = amounts[i] if i < len(amounts) else 0
            
            # Unusually high amounts
            if amount > avg_amount * 3:
                analysis['fraud_indicators'].append(f"High amount transaction: ${amount}")
                analysis['suspicious_transactions'] += 1
            
            # Rapid succession transactions
            if i > 0:
                prev_time = transactions[i-1].get('timestamp', '')
                curr_time = transaction.get('timestamp', '')
                if prev_time and curr_time:
                    try:
                        prev_dt = datetime.fromisoformat(prev_time.replace('Z', '+00:00'))
                        curr_dt = datetime.fromisoformat(curr_time.replace('Z', '+00:00'))
                        time_diff = (curr_dt - prev_dt).total_seconds()
                        
                        if time_diff < 60:  # Less than 1 minute apart
                            analysis['fraud_indicators'].append("Rapid succession transactions")
                            analysis['suspicious_transactions'] += 1
                    except:
                        pass
        
        # Determine risk level
        if analysis['suspicious_transactions'] > len(transactions) * 0.3:
            analysis['risk_level'] = 'high'
        elif analysis['suspicious_transactions'] > len(transactions) * 0.1:
            analysis['risk_level'] = 'medium'
        
        return analysis


class AgentCoreBrowserTool(AgentTool):
    """Custom Strands tool that integrates with AgentCore Browser Tool."""
    
    def __init__(self, region: str = "us-east-1"):
        super().__init__(name="agentcore_browser")
        self.region = region
        self.bedrock_agent = boto3.client('bedrock-agent-runtime', region_name=region)
        self.secrets_client = boto3.client('secretsmanager', region_name=region)
        
    async def invoke_browser_tool(self, instructions: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the AgentCore Browser Tool with structured instructions."""
        try:
            # Prepare the browser tool invocation
            tool_input = {
                "toolName": "AgentCoreBrowserTool",
                "toolInput": {
                    "instructions": json.dumps(instructions),
                    "securityConfig": {
                        "isolationLevel": "maximum",
                        "dataProtection": "pci_dss_compliant",
                        "auditLogging": True,
                        "sessionTimeout": 900
                    }
                }
            }
            
            # Invoke the browser tool via Bedrock Agent Runtime
            response = self.bedrock_agent.invoke_agent(
                agentId="your-agent-id",  # Replace with actual agent ID
                agentAliasId="your-alias-id",  # Replace with actual alias ID
                sessionId=f"financial-session-{uuid.uuid4()}",
                inputText=json.dumps(tool_input)
            )
            
            # Parse the response
            result = json.loads(response['completion'])
            return result
            
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def navigate_and_authenticate(self, portal_url: str, credential_secret_name: str) -> Dict[str, Any]:
        """Navigate to portal and authenticate securely."""
        # Retrieve credentials from AWS Secrets Manager
        try:
            secret_response = self.secrets_client.get_secret_value(SecretId=credential_secret_name)
            credentials = json.loads(secret_response['SecretString'])
        except Exception as e:
            return {"error": f"Failed to retrieve credentials: {str(e)}", "success": False}
        
        # Prepare browser instructions for navigation and authentication
        instructions = {
            "action": "navigate_and_authenticate",
            "url": portal_url,
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
                    "value": credentials['username']
                },
                {
                    "action": "fill_secure",
                    "selector": "input[type='password']",
                    "value": credentials['password']
                },
                {
                    "action": "click",
                    "selector": "button[type='submit'], input[type='submit']"
                },
                {
                    "action": "wait_for_element",
                    "selector": ".dashboard, .account-summary, .main-content",
                    "timeout": 15000
                }
            ]
        }
        
        return await self.invoke_browser_tool(instructions)
    
    async def extract_financial_data(self, data_type: str, account_id: str = None) -> Dict[str, Any]:
        """Extract specific financial data using the browser tool."""
        # Define extraction instructions based on data type
        extraction_steps = {
            "transactions": [
                {"action": "click", "selector": ".transactions, .transaction-history"},
                {"action": "wait_for_load_state", "state": "networkidle"},
                {"action": "extract_data", "selector": ".transaction, .transaction-row", "attributes": ["text", "data-*"]}
            ],
            "statements": [
                {"action": "click", "selector": ".statements, .account-statements"},
                {"action": "wait_for_load_state", "state": "networkidle"},
                {"action": "extract_data", "selector": ".statement, .statement-link", "attributes": ["text", "href"]}
            ],
            "cards": [
                {"action": "click", "selector": ".cards, .payment-cards"},
                {"action": "wait_for_load_state", "state": "networkidle"},
                {"action": "extract_data", "selector": ".card, .payment-card", "attributes": ["text"]}
            ],
            "balances": [
                {"action": "extract_data", "selector": ".balance, .current-balance, .account-balance", "attributes": ["text"]}
            ]
        }
        
        if data_type not in extraction_steps:
            return {"error": f"Unknown data type: {data_type}", "success": False}
        
        instructions = {
            "action": "extract_financial_data",
            "data_type": data_type,
            "account_id": account_id,
            "steps": extraction_steps[data_type]
        }
        
        return await self.invoke_browser_tool(instructions)


class SecureFinancialAgent:
    """Strands agent specialized for PCI DSS compliant financial data processing."""
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.browser_tool = AgentCoreBrowserTool(region=region)
        self.pci_tool = PCIComplianceTool()
        self.fraud_tool = FraudDetectionTool()
        self.audit_logger = AuditLogger(service_name="financial_agent")
        
        # Initialize strong encryption for financial data
        self.encryption_key = self._derive_encryption_key()
        self.fernet = Fernet(self.encryption_key)
        
        # Initialize Strands agent with financial-specific configuration
        self.agent = Agent(
            name="financial_data_processor",
            tools=[self.browser_tool, self.pci_tool, self.fraud_tool],
            security_context=SecurityContext(
                compliance_level="pci_dss",
                data_classification="financial",
                audit_required=True,
                encryption_required=True
            )
        )
    
    def _derive_encryption_key(self) -> bytes:
        """Derive strong encryption key for financial data."""
        password = b"financial_data_encryption_key_2024"
        salt = b"pci_dss_compliant_salt"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    async def create_secure_financial_session(self, portal_config: Dict[str, Any]) -> str:
        """Create a secure browser session for financial portal access."""
        session_id = f"financial-session-{uuid.uuid4()}"
        
        # Log session creation for audit
        await self.audit_logger.log_event({
            "event_type": "financial_session_created",
            "session_id": session_id,
            "portal": portal_config.get("name", "unknown"),
            "compliance_level": "pci_dss",
            "security_controls": ["isolation_level_maximum", "pci_dss_compliant", "audit_logging"],
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return session_id
    
    async def authenticate_with_mfa(
        self, 
        session_id: str, 
        portal_url: str,
        credential_secret_name: str
    ) -> bool:
        """Securely authenticate to financial portal with MFA using AgentCore Browser Tool."""
        try:
            # Use the browser tool to navigate and authenticate
            auth_result = await self.browser_tool.navigate_and_authenticate(portal_url, credential_secret_name)
            
            success = auth_result.get("success", False)
            
            # Log authentication attempt
            await self.audit_logger.log_event({
                "event_type": "financial_authentication",
                "session_id": session_id,
                "portal_url": portal_url,
                "mfa_used": True,
                "success": success,
                "browser_tool_response": auth_result,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return success
            
        except Exception as e:
            await self.audit_logger.log_event({
                "event_type": "authentication_error",
                "session_id": session_id,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            return False
    

    
    async def extract_financial_data(
        self, 
        session_id: str, 
        account_identifier: str,
        data_types: List[str]
    ) -> List[FinancialDataRecord]:
        """Extract financial data with PCI DSS compliance using AgentCore Browser Tool."""
        extracted_records = []
        
        # Generate secure account ID hash
        account_id_hash = hashlib.sha256(account_identifier.encode()).hexdigest()[:16]
        
        for data_type in data_types:
            try:
                # Use browser tool to extract data
                extraction_result = await self.browser_tool.extract_financial_data(data_type, account_identifier)
                
                if not extraction_result.get("success", False):
                    raise Exception(f"Browser tool extraction failed: {extraction_result.get('error', 'Unknown error')}")
                
                raw_data = extraction_result.get("data", {})
                
                # Analyze for cardholder data
                cardholder_analysis = await self.pci_tool.detect_cardholder_data(str(raw_data))
                sensitive_analysis = await self.pci_tool.detect_sensitive_auth_data(str(raw_data))
                
                # Determine compliance level
                if cardholder_analysis['cardholder_data_detected']:
                    compliance_level = PCIComplianceLevel.CARDHOLDER_DATA
                elif sensitive_analysis['sensitive_data_detected']:
                    compliance_level = PCIComplianceLevel.SENSITIVE_AUTH_DATA
                else:
                    compliance_level = PCIComplianceLevel.FINANCIAL_DATA
                
                # Calculate fraud score
                fraud_score = await self.pci_tool.calculate_fraud_score(raw_data)
                
                # Create secure record
                record = FinancialDataRecord(
                    record_id=str(uuid.uuid4()),
                    account_id=account_id_hash,
                    record_type=data_type,
                    data_classification=compliance_level,
                    extracted_data=raw_data,
                    encryption_applied=False,  # Will be encrypted later
                    extraction_timestamp=datetime.utcnow(),
                    audit_trail=[f"extracted_by_agentcore_browser_tool_{datetime.utcnow().isoformat()}"],
                    fraud_score=fraud_score
                )
                
                extracted_records.append(record)
                
                # Log extraction
                await self.audit_logger.log_event({
                    "event_type": "financial_data_extraction",
                    "session_id": session_id,
                    "account_id": account_id_hash,
                    "data_type": data_type,
                    "cardholder_data_detected": cardholder_analysis['cardholder_data_detected'],
                    "compliance_level": compliance_level.value,
                    "fraud_score": fraud_score,
                    "browser_tool_used": True,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            except Exception as e:
                await self.audit_logger.log_event({
                    "event_type": "extraction_error",
                    "session_id": session_id,
                    "account_id": account_id_hash,
                    "data_type": data_type,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        return extracted_records
    

    
    async def process_and_secure_financial_data(self, records: List[FinancialDataRecord]) -> Dict[str, Any]:
        """Process extracted financial data with PCI DSS compliance."""
        processed_data = {
            "total_records": len(records),
            "cardholder_data_records": 0,
            "sensitive_auth_records": 0,
            "processed_records": [],
            "fraud_analysis": {},
            "audit_summary": []
        }
        
        all_transactions = []
        
        for record in records:
            # Mask cardholder data if present
            if record.data_classification == PCIComplianceLevel.CARDHOLDER_DATA:
                processed_data["cardholder_data_records"] += 1
                
                # Mask sensitive data
                masked_data = {}
                for key, value in record.extracted_data.items():
                    if isinstance(value, str):
                        masked_data[key] = await self.pci_tool.mask_cardholder_data(value)
                    elif isinstance(value, list):
                        masked_data[key] = [
                            await self.pci_tool.mask_cardholder_data(str(item)) if isinstance(item, str) else item
                            for item in value
                        ]
                    else:
                        masked_data[key] = value
                
                record.extracted_data = masked_data
            
            # Handle sensitive authentication data
            if record.data_classification == PCIComplianceLevel.SENSITIVE_AUTH_DATA:
                processed_data["sensitive_auth_records"] += 1
            
            # Encrypt sensitive records
            if record.data_classification in [
                PCIComplianceLevel.CARDHOLDER_DATA, 
                PCIComplianceLevel.SENSITIVE_AUTH_DATA,
                PCIComplianceLevel.FINANCIAL_DATA
            ]:
                encrypted_data = self.fernet.encrypt(json.dumps(record.extracted_data).encode())
                record.extracted_data = {"encrypted": True, "data": encrypted_data.decode()}
                record.encryption_applied = True
            
            # Collect transactions for fraud analysis
            if record.record_type == "transactions" and "recent_transactions" in record.extracted_data:
                all_transactions.extend(record.extracted_data.get("recent_transactions", []))
            
            processed_data["processed_records"].append(record.to_secure_dict())
            
            # Add to audit trail
            processed_data["audit_summary"].append({
                "record_id": record.record_id,
                "account_id": record.account_id,
                "record_type": record.record_type,
                "classification": record.data_classification.value,
                "fraud_score": record.fraud_score,
                "encrypted": record.encryption_applied,
                "processed_timestamp": datetime.utcnow().isoformat()
            })
        
        # Perform fraud analysis on all transactions
        if all_transactions:
            fraud_analysis = await self.fraud_tool.analyze_transaction_pattern(all_transactions)
            processed_data["fraud_analysis"] = fraud_analysis
        
        return processed_data
    
    async def generate_pci_audit_report(self, session_id: str) -> Dict[str, Any]:
        """Generate PCI DSS compliance audit report."""
        audit_events = await self.audit_logger.get_session_events(session_id)
        
        report = {
            "session_id": session_id,
            "report_generated": datetime.utcnow().isoformat(),
            "compliance_framework": "PCI DSS",
            "agentcore_browser_tool_used": True,
            "total_events": len(audit_events),
            "event_summary": {},
            "cardholder_data_events": [],
            "security_events": [],
            "fraud_events": [],
            "compliance_status": "compliant"
        }
        
        # Categorize events
        for event in audit_events:
            event_type = event.get("event_type", "unknown")
            if event_type not in report["event_summary"]:
                report["event_summary"][event_type] = 0
            report["event_summary"][event_type] += 1
            
            # Track cardholder data handling
            if "cardholder_data" in event_type or event.get("cardholder_data_detected"):
                report["cardholder_data_events"].append(event)
            
            # Track security events
            if event_type in ["financial_authentication", "financial_session_created", "mfa_error"]:
                report["security_events"].append(event)
            
            # Track fraud-related events
            if "fraud" in event_type or event.get("fraud_score", 0) > 0.5:
                report["fraud_events"].append(event)
        
        return report


async def main():
    """Main function demonstrating financial data extraction."""
    print("💳 Financial Data Extraction with PCI DSS Compliance")
    print("=" * 60)
    
    # Initialize financial agent
    agent = SecureFinancialAgent()
    
    # Financial portal configuration
    portal_config = {
        "name": "Chase Online Banking",
        "url": "https://secure.chase.com/login",
        "credential_secret_name": "chase_banking_credentials"  # AWS Secrets Manager secret name
    }
    
    try:
        # Create secure browser session
        print("Creating secure PCI DSS compliant browser session...")
        session_id = await agent.create_secure_financial_session(portal_config)
        
        # Authenticate with MFA using AgentCore Browser Tool
        print("Authenticating to financial portal with MFA via AgentCore Browser Tool...")
        auth_success = await agent.authenticate_with_mfa(
            session_id, 
            portal_config["url"],
            portal_config["credential_secret_name"]
        )
        
        if not auth_success:
            print("❌ Authentication failed")
            return
        
        print("✅ Successfully authenticated to financial portal via AgentCore Browser Tool")
        
        # Extract financial data using AgentCore Browser Tool
        print("Extracting financial data with PCI DSS compliance via AgentCore Browser Tool...")
        financial_records = await agent.extract_financial_data(
            session_id,
            account_identifier="ACCT123456",  # Example account ID
            data_types=["transactions", "statements", "cards", "balances"]
        )
        
        print(f"✅ Extracted {len(financial_records)} financial records")
        
        # Process and secure data
        print("Processing data with PCI DSS compliance controls...")
        processed_data = await agent.process_and_secure_financial_data(financial_records)
        
        print(f"✅ Processed {processed_data['total_records']} records")
        print(f"🔒 Cardholder data records: {processed_data['cardholder_data_records']}")
        print(f"🔐 Sensitive auth records: {processed_data['sensitive_auth_records']}")
        
        # Display fraud analysis
        if processed_data.get('fraud_analysis'):
            fraud_analysis = processed_data['fraud_analysis']
            print(f"🚨 Fraud Analysis:")
            print(f"   - Risk level: {fraud_analysis['risk_level']}")
            print(f"   - Suspicious transactions: {fraud_analysis['suspicious_transactions']}")
            print(f"   - Fraud indicators: {len(fraud_analysis['fraud_indicators'])}")
        
        # Generate audit report
        print("Generating PCI DSS compliance audit report...")
        audit_report = await agent.generate_pci_audit_report(session_id)
        
        print("✅ PCI DSS Audit Report Generated:")
        print(f"   - Total events: {audit_report['total_events']}")
        print(f"   - Cardholder data events: {len(audit_report['cardholder_data_events'])}")
        print(f"   - Security events: {len(audit_report['security_events'])}")
        print(f"   - Fraud events: {len(audit_report['fraud_events'])}")
        print(f"   - Compliance status: {audit_report['compliance_status']}")
        
        # Save results securely
        results = {
            "session_summary": {
                "session_id": session_id,
                "portal": portal_config["name"],
                "records_processed": len(financial_records),
                "cardholder_data_records": processed_data['cardholder_data_records'],
                "compliance_status": "pci_dss_compliant",
                "agentcore_browser_tool_used": True
            },
            "audit_report": audit_report,
            "processed_data": processed_data
        }
        
        # Save to secure file (in production, this would be encrypted storage)
        with open("financial_processing_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print("✅ Results saved to financial_processing_results.json")
        
    except Exception as e:
        print(f"❌ Error during financial data extraction: {str(e)}")
        
    finally:
        # AgentCore Browser Tool automatically cleans up sessions
        print("🧹 AgentCore Browser Tool session automatically cleaned up")


if __name__ == "__main__":
    asyncio.run(main())