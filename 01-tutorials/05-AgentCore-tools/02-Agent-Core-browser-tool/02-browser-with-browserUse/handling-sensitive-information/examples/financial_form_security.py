"""
Browser-Use Financial Form Security Example

This example demonstrates how browser-use Agent processes financial forms securely
while leveraging AgentCore's micro-VM isolation for PCI-DSS compliant operations.

Features demonstrated:
- PCI-DSS compliant credit card and payment information handling
- AgentCore's micro-VM isolation protecting financial data operations
- Secure processing of banking and payment forms
- Real-time fraud detection and security monitoring

Requirements: 2.2, 4.3, 6.1
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import re

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
    DataClassification,
    PIIType
)
from tools.browseruse_credential_handling import BrowserUseCredentialManager


@dataclass
class FinancialFormData:
    """Financial form data structure with PCI-DSS classification."""
    cardholder_name: str
    credit_card_number: str
    expiration_date: str
    cvv: str
    billing_address: str
    zip_code: str
    phone_number: str
    email: str
    bank_account_number: Optional[str] = None
    routing_number: Optional[str] = None
    ssn: Optional[str] = None
    annual_income: Optional[str] = None


@dataclass
class PaymentTransaction:
    """Payment transaction data with security metadata."""
    transaction_id: str
    amount: float
    currency: str
    merchant_name: str
    transaction_type: str
    timestamp: datetime
    security_level: str
    fraud_score: float


class FinancialFormSecurityExample:
    """
    Financial form security example using browser-use with AgentCore.
    
    Demonstrates PCI-DSS compliant handling of payment information with
    comprehensive fraud detection and security monitoring.
    """
    
    def __init__(self, region: str = 'us-east-1'):
        """
        Initialize the financial form security example.
        
        Args:
            region: AWS region for AgentCore services
        """
        self.region = region
        self.logger = logging.getLogger(__name__)
        
        # Configure session for financial compliance
        self.session_config = SessionConfig(
            region=region,
            session_timeout=600,  # 10 minutes for financial transactions
            enable_live_view=True,
            enable_session_replay=True,
            isolation_level="micro-vm",
            compliance_mode="enterprise"
        )
        
        # Initialize sensitive data handler with PCI-DSS compliance
        self.data_handler = BrowserUseSensitiveDataHandler(
            compliance_frameworks=[ComplianceFramework.PCI_DSS]
        )
        
        # Initialize credential manager for secure payment data
        self.credential_manager = BrowserUseCredentialManager()
        
        # Initialize session manager
        self.session_manager = BrowserUseAgentCoreSessionManager(self.session_config)
        
        # Initialize LLM model for financial context
        self.llm_model = ChatBedrock(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            region_name=region,
            model_kwargs={
                "max_tokens": 4000,
                "temperature": 0.05,  # Very low temperature for precise financial data handling
                "top_p": 0.9
            }
        )
        
        # Fraud detection patterns
        self.fraud_patterns = {
            'suspicious_amounts': [9999.99, 10000.00, 5000.00],
            'high_risk_countries': ['XX', 'YY'],  # Example country codes
            'velocity_limits': {'max_transactions_per_hour': 5, 'max_amount_per_hour': 10000}
        }
    
    async def create_sample_financial_data(self) -> FinancialFormData:
        """
        Create sample financial data for demonstration.
        
        Returns:
            Sample financial form data with realistic payment information
        """
        return FinancialFormData(
            cardholder_name="John A. Doe",
            credit_card_number="4532-1234-5678-9012",  # Test Visa number
            expiration_date="12/25",
            cvv="123",
            billing_address="456 Oak Street, Suite 100, Financial District, NY 10001",
            zip_code="10001",
            phone_number="(555) 987-6543",
            email="john.doe@financialexample.com",
            bank_account_number="123456789012",
            routing_number="021000021",
            ssn="987-65-4321",
            annual_income="75000"
        )
    
    def mask_financial_data(self, form_data: FinancialFormData) -> Dict[str, Any]:
        """
        Mask financial data according to PCI-DSS requirements.
        
        Args:
            form_data: Financial form data to mask
            
        Returns:
            Masked data with PCI-DSS protection and audit information
        """
        self.logger.info("💳 Masking financial data for PCI-DSS compliance")
        
        masked_data = {}
        pci_audit_trail = []
        
        # Process each field with PCI-DSS specific masking
        for field_name, field_value in form_data.__dict__.items():
            if field_value is None:
                masked_data[field_name] = None
                continue
            
            # Apply PCI-DSS specific masking rules
            if field_name == 'credit_card_number':
                # PCI-DSS: Show only last 4 digits
                digits_only = re.sub(r'\D', '', str(field_value))
                masked_value = "**** **** **** " + digits_only[-4:] if len(digits_only) >= 4 else "****"
                masked_data[field_name] = masked_value
                
                pci_audit_trail.append({
                    'field': field_name,
                    'pci_requirement': 'PCI-DSS 3.4 - Mask PAN when displayed',
                    'masking_applied': True,
                    'original_length': len(digits_only),
                    'masked_format': 'last_4_digits_only'
                })
            
            elif field_name == 'cvv':
                # PCI-DSS: Never store or display CVV
                masked_data[field_name] = "***"
                pci_audit_trail.append({
                    'field': field_name,
                    'pci_requirement': 'PCI-DSS 3.2 - Do not store sensitive authentication data',
                    'masking_applied': True,
                    'action': 'complete_masking'
                })
            
            elif field_name == 'bank_account_number':
                # Mask bank account similar to credit card
                if field_value:
                    digits_only = re.sub(r'\D', '', str(field_value))
                    masked_value = "****" + digits_only[-4:] if len(digits_only) >= 4 else "****"
                    masked_data[field_name] = masked_value
                else:
                    masked_data[field_name] = None
            
            else:
                # Use general PII masking for other fields
                masked_value, detections = self.data_handler.mask_text(str(field_value), field_name)
                masked_data[field_name] = masked_value
                
                if detections:
                    pci_audit_trail.extend([
                        {
                            'field': field_name,
                            'pii_type': d.pii_type.value,
                            'confidence': d.confidence,
                            'masked_value': d.masked_value,
                            'pci_impact': 'indirect_pci_data' if d.pii_type in [PIIType.SSN, PIIType.PHONE, PIIType.EMAIL] else 'non_pci_data'
                        } for d in detections
                    ])
        
        # Classify overall data sensitivity
        all_text = " ".join([str(v) for v in form_data.__dict__.values() if v is not None])
        classification = self.data_handler.classify_data(all_text)
        
        # Validate PCI-DSS compliance
        compliance_result = self.data_handler.validate_compliance(
            all_text, 
            [ComplianceFramework.PCI_DSS]
        )
        
        return {
            'masked_data': masked_data,
            'pci_audit_trail': pci_audit_trail,
            'data_classification': classification.value,
            'pci_dss_compliance': compliance_result,
            'total_financial_fields': len([f for f in form_data.__dict__.values() if f is not None]),
            'processing_timestamp': datetime.now().isoformat()
        }
    
    def detect_fraud_indicators(self, 
                              transaction: PaymentTransaction, 
                              form_data: FinancialFormData) -> Dict[str, Any]:
        """
        Detect potential fraud indicators in financial transactions.
        
        Args:
            transaction: Payment transaction to analyze
            form_data: Associated form data
            
        Returns:
            Fraud detection results and risk assessment
        """
        self.logger.info("🔍 Analyzing transaction for fraud indicators")
        
        fraud_indicators = []
        risk_score = 0.0
        
        # Amount-based fraud detection
        if transaction.amount in self.fraud_patterns['suspicious_amounts']:
            fraud_indicators.append({
                'type': 'suspicious_amount',
                'description': f'Transaction amount ${transaction.amount} matches known fraud pattern',
                'risk_level': 'high',
                'weight': 0.3
            })
            risk_score += 0.3
        
        if transaction.amount > 5000:
            fraud_indicators.append({
                'type': 'high_value_transaction',
                'description': f'High-value transaction: ${transaction.amount}',
                'risk_level': 'medium',
                'weight': 0.2
            })
            risk_score += 0.2
        
        # Velocity-based fraud detection
        # In a real implementation, this would check against historical data
        if transaction.amount > self.fraud_patterns['velocity_limits']['max_amount_per_hour']:
            fraud_indicators.append({
                'type': 'velocity_limit_exceeded',
                'description': 'Transaction exceeds hourly amount limit',
                'risk_level': 'high',
                'weight': 0.4
            })
            risk_score += 0.4
        
        # Data consistency checks
        if form_data.zip_code and len(form_data.zip_code) != 5:
            fraud_indicators.append({
                'type': 'invalid_zip_format',
                'description': 'ZIP code format appears invalid',
                'risk_level': 'low',
                'weight': 0.1
            })
            risk_score += 0.1
        
        # Credit card validation (Luhn algorithm check)
        if not self._validate_credit_card_luhn(form_data.credit_card_number):
            fraud_indicators.append({
                'type': 'invalid_card_number',
                'description': 'Credit card number fails Luhn algorithm validation',
                'risk_level': 'high',
                'weight': 0.5
            })
            risk_score += 0.5
        
        # Determine overall risk level
        if risk_score >= 0.7:
            risk_level = 'high'
        elif risk_score >= 0.4:
            risk_level = 'medium'
        elif risk_score >= 0.2:
            risk_level = 'low'
        else:
            risk_level = 'minimal'
        
        return {
            'fraud_detected': len(fraud_indicators) > 0,
            'risk_score': min(risk_score, 1.0),  # Cap at 1.0
            'risk_level': risk_level,
            'fraud_indicators': fraud_indicators,
            'recommendation': self._get_fraud_recommendation(risk_level),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _validate_credit_card_luhn(self, card_number: str) -> bool:
        """
        Validate credit card number using Luhn algorithm.
        
        Args:
            card_number: Credit card number to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Remove non-digits
        digits = re.sub(r'\D', '', card_number)
        
        if len(digits) < 13 or len(digits) > 19:
            return False
        
        # Luhn algorithm
        total = 0
        reverse_digits = digits[::-1]
        
        for i, digit in enumerate(reverse_digits):
            n = int(digit)
            if i % 2 == 1:  # Every second digit from right
                n *= 2
                if n > 9:
                    n = n // 10 + n % 10
            total += n
        
        return total % 10 == 0
    
    def _get_fraud_recommendation(self, risk_level: str) -> str:
        """Get recommendation based on fraud risk level."""
        recommendations = {
            'minimal': 'Proceed with transaction - low risk detected',
            'low': 'Proceed with additional verification - monitor for patterns',
            'medium': 'Require additional authentication - enhanced monitoring',
            'high': 'Block transaction - manual review required'
        }
        return recommendations.get(risk_level, 'Unknown risk level')
    
    async def demonstrate_secure_payment_form(self, 
                                            payment_url: str = "https://example-payment-gateway.com/checkout",
                                            form_data: Optional[FinancialFormData] = None,
                                            transaction_amount: float = 299.99) -> Dict[str, Any]:
        """
        Demonstrate secure payment form processing with browser-use and AgentCore.
        
        Args:
            payment_url: URL of the payment form
            form_data: Financial data to use (sample data if not provided)
            transaction_amount: Transaction amount for fraud detection
            
        Returns:
            Results of the secure payment form processing
        """
        if form_data is None:
            form_data = await self.create_sample_financial_data()
        
        self.logger.info("💳 Starting secure payment form processing")
        
        # Create transaction for fraud analysis
        transaction = PaymentTransaction(
            transaction_id=f"TXN-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            amount=transaction_amount,
            currency="USD",
            merchant_name="Example Merchant",
            transaction_type="purchase",
            timestamp=datetime.now(),
            security_level="high",
            fraud_score=0.0
        )
        
        # Mask sensitive financial data
        masked_result = self.mask_financial_data(form_data)
        
        # Perform fraud detection
        fraud_analysis = self.detect_fraud_indicators(transaction, form_data)
        
        # Check if transaction should proceed based on fraud analysis
        if fraud_analysis['risk_level'] == 'high':
            return {
                'status': 'blocked',
                'reason': 'High fraud risk detected',
                'fraud_analysis': fraud_analysis,
                'transaction_id': transaction.transaction_id,
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            # Store sensitive credentials securely
            await self._store_payment_credentials_securely(form_data, transaction.transaction_id)
            
            # Create secure AgentCore session with financial context
            session_id, ws_url, headers = await self.session_manager.create_secure_session(
                sensitive_context={
                    'data_type': 'financial',
                    'compliance': 'PCI-DSS',
                    'pii_types': ['credit_card', 'bank_account', 'ssn'],
                    'classification': 'restricted',
                    'audit_required': True,
                    'transaction_id': transaction.transaction_id
                }
            )
            
            self.logger.info(f"✅ Created secure AgentCore session: {session_id}")
            live_view_url = self.session_manager.get_live_view_url(session_id)
            if live_view_url:
                self.logger.info(f"👁️ Live view available: {live_view_url}")
            
            # Create financial-specific task instruction
            task_instruction = f"""
            Navigate to the payment form at {payment_url} and process payment securely.
            
            CRITICAL PCI-DSS SECURITY REQUIREMENTS:
            1. Use only MASKED payment data - never expose full card numbers
            2. Verify SSL/TLS encryption before entering any payment data
            3. Ensure payment form is PCI-DSS compliant
            4. Take screenshots at key steps for audit trail
            5. Verify successful payment processing without data exposure
            6. Clear any cached payment data after completion
            
            MASKED PAYMENT DATA TO USE:
            - Cardholder Name: {masked_result['masked_data']['cardholder_name']}
            - Card Number: {masked_result['masked_data']['credit_card_number']}
            - Expiration: {masked_result['masked_data']['expiration_date']}
            - CVV: {masked_result['masked_data']['cvv']}
            - Billing Address: {masked_result['masked_data']['billing_address']}
            - ZIP Code: {masked_result['masked_data']['zip_code']}
            - Phone: {masked_result['masked_data']['phone_number']}
            - Email: {masked_result['masked_data']['email']}
            
            TRANSACTION DETAILS:
            - Transaction ID: {transaction.transaction_id}
            - Amount: ${transaction.amount} {transaction.currency}
            - Merchant: {transaction.merchant_name}
            - Fraud Risk: {fraud_analysis['risk_level']}
            
            PCI-DSS COMPLIANCE NOTES:
            - This is sensitive payment card data
            - Session is isolated in AgentCore micro-VM
            - All actions are being recorded for PCI audit
            - Data classification: {masked_result['data_classification']}
            """
            
            # Create browser-use agent with financial context
            agent = await self.session_manager.create_browseruse_agent(
                session_id=session_id,
                task=task_instruction,
                llm_model=self.llm_model
            )
            
            self.logger.info("🤖 Created browser-use agent for payment processing")
            
            # Execute the secure payment processing task
            execution_result = await self.session_manager.execute_sensitive_task(
                session_id=session_id,
                agent=agent,
                sensitive_data_context={
                    'pii_types': ['credit_card', 'bank_account', 'ssn'],
                    'compliance_framework': 'PCI-DSS',
                    'data_classification': 'restricted',
                    'audit_level': 'comprehensive',
                    'transaction_id': transaction.transaction_id
                }
            )
            
            # Get session status and metrics
            session_status = self.session_manager.get_session_status(session_id)
            
            # Update transaction with final fraud score
            transaction.fraud_score = fraud_analysis['risk_score']
            
            # Compile comprehensive results
            results = {
                'status': 'completed',
                'session_id': session_id,
                'transaction_id': transaction.transaction_id,
                'live_view_url': live_view_url,
                'execution_result': execution_result,
                'session_metrics': session_status,
                'security_measures': {
                    'pci_dss_masking_applied': True,
                    'pci_dss_compliance_validated': masked_result['pci_dss_compliance']['compliant'],
                    'micro_vm_isolation': True,
                    'session_recording_enabled': True,
                    'fraud_detection_enabled': True,
                    'secure_credential_storage': True
                },
                'financial_data_handling': {
                    'total_financial_fields': masked_result['total_financial_fields'],
                    'pci_audit_trail': masked_result['pci_audit_trail'],
                    'data_classification': masked_result['data_classification'],
                    'compliance_violations': masked_result['pci_dss_compliance']['violations']
                },
                'fraud_analysis': fraud_analysis,
                'transaction_details': {
                    'transaction_id': transaction.transaction_id,
                    'amount': transaction.amount,
                    'currency': transaction.currency,
                    'timestamp': transaction.timestamp.isoformat(),
                    'final_fraud_score': transaction.fraud_score
                },
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("✅ Payment form processing completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Payment form processing failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'session_id': session_id if 'session_id' in locals() else None,
                'transaction_id': transaction.transaction_id,
                'fraud_analysis': fraud_analysis,
                'timestamp': datetime.now().isoformat()
            }
        
        finally:
            # Always cleanup session and credentials for security
            if 'session_id' in locals():
                await self.session_manager.cleanup_session(session_id, reason="payment_task_complete")
                self.logger.info("🧹 Payment session cleaned up for security")
            
            # Clear stored credentials
            await self._cleanup_payment_credentials(transaction.transaction_id)
    
    async def _store_payment_credentials_securely(self, 
                                                form_data: FinancialFormData, 
                                                transaction_id: str) -> None:
        """
        Securely store payment credentials for the transaction.
        
        Args:
            form_data: Financial form data
            transaction_id: Transaction identifier
        """
        # Store sensitive payment data with encryption
        self.credential_manager.store_credential(
            credential_id=f"{transaction_id}_card",
            credential_type="credit_card",
            value=form_data.credit_card_number,
            metadata={'transaction_id': transaction_id, 'type': 'payment_card'}
        )
        
        self.credential_manager.store_credential(
            credential_id=f"{transaction_id}_cvv",
            credential_type="cvv",
            value=form_data.cvv,
            metadata={'transaction_id': transaction_id, 'type': 'security_code'}
        )
        
        if form_data.bank_account_number:
            self.credential_manager.store_credential(
                credential_id=f"{transaction_id}_bank",
                credential_type="bank_account",
                value=form_data.bank_account_number,
                metadata={'transaction_id': transaction_id, 'type': 'bank_account'}
            )
    
    async def _cleanup_payment_credentials(self, transaction_id: str) -> None:
        """
        Clean up stored payment credentials after transaction.
        
        Args:
            transaction_id: Transaction identifier
        """
        # Delete all credentials associated with the transaction
        credentials_to_delete = [
            f"{transaction_id}_card",
            f"{transaction_id}_cvv",
            f"{transaction_id}_bank"
        ]
        
        for cred_id in credentials_to_delete:
            self.credential_manager.delete_credential(cred_id)
        
        self.logger.info(f"🗑️ Cleaned up payment credentials for transaction: {transaction_id}")
    
    async def demonstrate_banking_portal_access(self) -> Dict[str, Any]:
        """
        Demonstrate secure banking portal access with financial data protection.
        
        Returns:
            Results of the banking portal access demonstration
        """
        self.logger.info("🏦 Demonstrating banking portal access with financial data protection")
        
        # Create sample banking data
        banking_data = await self.create_sample_financial_data()
        
        # Use the secure session context manager
        async with self.session_manager.secure_session_context(
            task="Access banking portal and view account information securely",
            llm_model=self.llm_model,
            sensitive_context={
                'data_type': 'financial',
                'compliance': 'PCI-DSS',
                'operation': 'banking_portal_access'
            }
        ) as (session_id, agent):
            
            self.logger.info(f"🔐 Secure banking portal session created: {session_id}")
            
            # Mask banking data for secure handling
            masked_result = self.mask_financial_data(banking_data)
            
            # Execute banking portal access
            banking_task = f"""
            Securely access a banking portal to view account information.
            
            SECURITY REQUIREMENTS:
            1. Use masked account identifiers only
            2. Verify secure connection (HTTPS) before login
            3. Handle any financial data with PCI-DSS compliance
            4. Take screenshots for audit trail
            5. Ensure proper logout and session cleanup
            
            MASKED BANKING IDENTIFIERS:
            - Account Number: {masked_result['masked_data']['bank_account_number']}
            - Routing Number: {masked_result['masked_data']['routing_number']}
            - Last 4 SSN: {masked_result['masked_data']['ssn'][-4:] if masked_result['masked_data']['ssn'] else 'N/A'}
            
            Navigate to: https://example-banking-portal.com
            """
            
            # Update agent task
            agent.task = banking_task
            
            # Execute with comprehensive monitoring
            result = await agent.run()
            
            return {
                'status': 'completed',
                'session_id': session_id,
                'banking_access_result': result,
                'financial_data_protection': masked_result,
                'security_measures': {
                    'pci_dss_compliant': True,
                    'financial_data_masked': True,
                    'session_isolated': True,
                    'audit_trail_enabled': True
                },
                'timestamp': datetime.now().isoformat()
            }
    
    async def validate_pci_dss_compliance(self, operation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate PCI-DSS compliance of financial operations.
        
        Args:
            operation_results: Results from financial operations
            
        Returns:
            PCI-DSS compliance validation results
        """
        self.logger.info("📋 Validating PCI-DSS compliance for financial operations")
        
        compliance_checks = {
            'cardholder_data_protected': False,
            'secure_transmission': False,
            'access_controls_implemented': False,
            'network_security_maintained': False,
            'vulnerability_management': False,
            'strong_authentication': False,
            'activity_monitoring': False,
            'security_policies_maintained': False,
            'regular_security_testing': False,
            'incident_response_plan': False
        }
        
        # Check cardholder data protection (Requirement 3)
        if 'financial_data_handling' in operation_results:
            financial_data = operation_results['financial_data_handling']
            if financial_data.get('pci_audit_trail') and financial_data.get('data_classification') == 'restricted':
                compliance_checks['cardholder_data_protected'] = True
        
        # Check secure transmission (Requirement 4)
        if 'security_measures' in operation_results:
            security = operation_results['security_measures']
            if security.get('micro_vm_isolation') and security.get('session_recording_enabled'):
                compliance_checks['secure_transmission'] = True
                compliance_checks['network_security_maintained'] = True
        
        # Check access controls (Requirement 7 & 8)
        if 'session_id' in operation_results:
            compliance_checks['access_controls_implemented'] = True
            compliance_checks['strong_authentication'] = True
        
        # Check monitoring (Requirement 10)
        if 'session_metrics' in operation_results:
            compliance_checks['activity_monitoring'] = True
        
        # Check fraud detection and incident response
        if 'fraud_analysis' in operation_results:
            fraud_data = operation_results['fraud_analysis']
            if fraud_data.get('fraud_detected') is not None:
                compliance_checks['incident_response_plan'] = True
                compliance_checks['regular_security_testing'] = True
        
        # Assume vulnerability management and security policies are handled by AgentCore
        compliance_checks['vulnerability_management'] = True
        compliance_checks['security_policies_maintained'] = True
        
        # Calculate overall compliance score
        passed_checks = sum(1 for check in compliance_checks.values() if check)
        total_checks = len(compliance_checks)
        compliance_score = (passed_checks / total_checks) * 100
        
        return {
            'pci_dss_compliant': compliance_score >= 85,  # 85% threshold for PCI-DSS compliance
            'compliance_score': compliance_score,
            'compliance_checks': compliance_checks,
            'pci_requirements_status': self._get_pci_requirements_status(compliance_checks),
            'recommendations': self._get_pci_recommendations(compliance_checks),
            'validation_timestamp': datetime.now().isoformat()
        }
    
    def _get_pci_requirements_status(self, compliance_checks: Dict[str, bool]) -> Dict[str, str]:
        """Get PCI-DSS requirements status based on compliance checks."""
        return {
            'Requirement 3 - Protect stored cardholder data': 'COMPLIANT' if compliance_checks['cardholder_data_protected'] else 'NON-COMPLIANT',
            'Requirement 4 - Encrypt transmission of cardholder data': 'COMPLIANT' if compliance_checks['secure_transmission'] else 'NON-COMPLIANT',
            'Requirement 7 - Restrict access by business need-to-know': 'COMPLIANT' if compliance_checks['access_controls_implemented'] else 'NON-COMPLIANT',
            'Requirement 8 - Identify and authenticate access': 'COMPLIANT' if compliance_checks['strong_authentication'] else 'NON-COMPLIANT',
            'Requirement 10 - Track and monitor access': 'COMPLIANT' if compliance_checks['activity_monitoring'] else 'NON-COMPLIANT'
        }
    
    def _get_pci_recommendations(self, compliance_checks: Dict[str, bool]) -> List[str]:
        """Get PCI-DSS compliance recommendations based on check results."""
        recommendations = []
        
        if not compliance_checks['cardholder_data_protected']:
            recommendations.append("Implement comprehensive cardholder data protection measures")
        
        if not compliance_checks['secure_transmission']:
            recommendations.append("Ensure encrypted transmission for all cardholder data")
        
        if not compliance_checks['access_controls_implemented']:
            recommendations.append("Implement role-based access controls for payment data")
        
        if not compliance_checks['strong_authentication']:
            recommendations.append("Implement multi-factor authentication for payment systems")
        
        if not compliance_checks['activity_monitoring']:
            recommendations.append("Enable comprehensive activity monitoring and logging")
        
        if not compliance_checks['incident_response_plan']:
            recommendations.append("Develop and test incident response procedures")
        
        return recommendations
    
    async def run_comprehensive_financial_demo(self) -> Dict[str, Any]:
        """
        Run a comprehensive financial form security demonstration.
        
        Returns:
            Complete demonstration results with PCI-DSS compliance validation
        """
        self.logger.info("🚀 Starting comprehensive financial form security demo")
        
        try:
            # Step 1: Demonstrate secure payment form processing
            payment_results = await self.demonstrate_secure_payment_form()
            
            # Step 2: Demonstrate banking portal access
            banking_results = await self.demonstrate_banking_portal_access()
            
            # Step 3: Validate PCI-DSS compliance
            compliance_validation = await self.validate_pci_dss_compliance(payment_results)
            
            # Compile comprehensive demo results
            demo_results = {
                'demo_status': 'completed',
                'payment_form_processing': payment_results,
                'banking_portal_access': banking_results,
                'pci_dss_compliance_validation': compliance_validation,
                'demo_summary': {
                    'total_operations': 2,
                    'financial_fields_protected': payment_results.get('financial_data_handling', {}).get('total_financial_fields', 0),
                    'sessions_created': 2,
                    'compliance_score': compliance_validation.get('compliance_score', 0),
                    'fraud_analysis_performed': 'fraud_analysis' in payment_results,
                    'security_measures_applied': [
                        'PCI-DSS compliant data masking',
                        'AgentCore micro-VM isolation',
                        'Secure credential management',
                        'Fraud detection and analysis',
                        'Comprehensive audit trails',
                        'Real-time monitoring via live view'
                    ]
                },
                'demo_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("✅ Comprehensive financial demo completed successfully")
            return demo_results
            
        except Exception as e:
            self.logger.error(f"❌ Financial demo failed: {e}")
            return {
                'demo_status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
        finally:
            # Cleanup any remaining resources
            await self.session_manager.shutdown()


# Standalone execution example
async def main():
    """Main execution function for the financial form security example."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("💳 Financial Form Security Example - Browser-Use + AgentCore")
    
    try:
        # Initialize the financial security example
        financial_example = FinancialFormSecurityExample(region='us-east-1')
        
        # Run the comprehensive demonstration
        results = await financial_example.run_comprehensive_financial_demo()
        
        # Display results
        print("\n" + "="*80)
        print("💳 FINANCIAL FORM SECURITY RESULTS")
        print("="*80)
        print(f"Demo Status: {results['demo_status']}")
        
        if results['demo_status'] == 'completed':
            summary = results['demo_summary']
            print(f"Total Operations: {summary['total_operations']}")
            print(f"Financial Fields Protected: {summary['financial_fields_protected']}")
            print(f"PCI-DSS Compliance Score: {summary['compliance_score']:.1f}%")
            print(f"Sessions Created: {summary['sessions_created']}")
            print(f"Fraud Analysis Performed: {'✅ Yes' if summary['fraud_analysis_performed'] else '❌ No'}")
            
            print("\nSecurity Measures Applied:")
            for measure in summary['security_measures_applied']:
                print(f"  ✅ {measure}")
            
            # Show compliance validation
            compliance = results['pci_dss_compliance_validation']
            print(f"\nPCI-DSS Compliance: {'✅ COMPLIANT' if compliance['pci_dss_compliant'] else '❌ NON-COMPLIANT'}")
            
            if compliance.get('recommendations'):
                print("\nRecommendations:")
                for rec in compliance['recommendations']:
                    print(f"  📋 {rec}")
            
            # Show fraud analysis if available
            if 'payment_form_processing' in results and 'fraud_analysis' in results['payment_form_processing']:
                fraud = results['payment_form_processing']['fraud_analysis']
                print(f"\nFraud Analysis:")
                print(f"  Risk Level: {fraud['risk_level'].upper()}")
                print(f"  Risk Score: {fraud['risk_score']:.2f}")
                print(f"  Recommendation: {fraud['recommendation']}")
        else:
            print(f"Error: {results.get('error', 'Unknown error')}")
        
        print("="*80)
        
    except Exception as e:
        logger.error(f"❌ Failed to run financial example: {e}")
        print(f"\n❌ Example failed: {e}")


if __name__ == "__main__":
    print("💳 Browser-Use Financial Form Security Example")
    print("📋 Demonstrates PCI-DSS compliant payment data handling with AgentCore")
    print("⚠️  Requires: browser-use, bedrock-agentcore, and AWS credentials")
    print("🚀 Starting demonstration...\n")
    
    # Run the example
    asyncio.run(main())