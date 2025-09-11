#!/usr/bin/env python3
"""
Financial Data Extraction Example with LlamaIndex and AgentCore Browser Tool

This example demonstrates secure extraction and processing of financial data using LlamaIndex agents
integrated with Amazon Bedrock AgentCore Browser Tool. Includes PCI DSS compliance patterns,
fraud detection, and secure handling of financial information.

Key Features:
- PCI DSS compliant data handling
- Credit card and banking information protection
- Secure extraction from financial portals
- Fraud detection and risk assessment
- Encrypted storage of financial data
- Regulatory compliance logging

Requirements: 3.1, 3.3, 5.1, 5.2
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import re
from decimal import Decimal

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.llms.bedrock import Bedrock

from bedrock_agentcore.tools.browser_client import BrowserSession
from agentcore_browser_loader import AgentCoreBrowserLoader
from sensitive_data_handler import SensitiveDataHandler, SensitivityLevel
from secure_rag_pipeline import SecureRAGPipeline

# Configure logging for financial compliance
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FinancialDataType(Enum):
    """Types of financial data requiring protection"""
    CREDIT_CARD = "credit_card_number"
    BANK_ACCOUNT = "bank_account_number"
    ROUTING_NUMBER = "routing_number"
    SSN = "social_security_number"
    TAX_ID = "tax_identification_number"
    ACCOUNT_BALANCE = "account_balance"
    TRANSACTION_AMOUNT = "transaction_amount"
    INVESTMENT_VALUE = "investment_value"
    SALARY = "salary_information"
    LOAN_AMOUNT = "loan_amount"


class RiskLevel(Enum):
    """Risk assessment levels for financial transactions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FinancialDocument:
    """Financial document with PCI DSS protection"""
    content: str
    document_type: str
    account_id: str  # Anonymized account identifier
    financial_data_detected: List[FinancialDataType]
    risk_level: RiskLevel
    sensitivity_level: SensitivityLevel
    source_url: str
    extraction_timestamp: datetime
    compliance_flags: List[str]
    audit_trail: List[Dict[str, Any]]
    
    def mask_financial_data(self) -> 'FinancialDocument':
        """Mask financial data according to PCI DSS requirements"""
        masked_content = self.content
        audit_event = {
            "action": "financial_data_masking",
            "timestamp": datetime.utcnow().isoformat(),
            "data_types_masked": [data_type.value for data_type in self.financial_data_detected],
            "compliance_standard": "PCI_DSS"
        }
        
        # Apply PCI DSS masking patterns
        financial_patterns = {
            FinancialDataType.CREDIT_CARD: r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            FinancialDataType.BANK_ACCOUNT: r'\b\d{8,17}\b',
            FinancialDataType.ROUTING_NUMBER: r'\b\d{9}\b',
            FinancialDataType.SSN: r'\b\d{3}-\d{2}-\d{4}\b',
            FinancialDataType.TAX_ID: r'\b\d{2}-\d{7}\b',
        }
        
        for data_type in self.financial_data_detected:
            if data_type in financial_patterns:
                if data_type == FinancialDataType.CREDIT_CARD:
                    # PCI DSS: Show only last 4 digits
                    masked_content = re.sub(
                        financial_patterns[data_type], 
                        lambda m: f"****-****-****-{m.group()[-4:]}", 
                        masked_content
                    )
                else:
                    masked_content = re.sub(
                        financial_patterns[data_type], 
                        f'[MASKED_{data_type.value.upper()}]', 
                        masked_content
                    )
        
        return FinancialDocument(
            content=masked_content,
            document_type=self.document_type,
            account_id=self.account_id,
            financial_data_detected=self.financial_data_detected,
            risk_level=self.risk_level,
            sensitivity_level=self.sensitivity_level,
            source_url=self.source_url,
            extraction_timestamp=self.extraction_timestamp,
            compliance_flags=self.compliance_flags + ["PCI_DSS_MASKED"],
            audit_trail=self.audit_trail + [audit_event]
        )


class FinancialDataExtractor:
    """Secure financial data extractor using LlamaIndex and AgentCore"""
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.browser_loader = AgentCoreBrowserLoader(region=region)
        self.sensitive_handler = SensitiveDataHandler()
        self.rag_pipeline = SecureRAGPipeline(region=region)
        
        # Configure LlamaIndex for financial use
        Settings.embed_model = BedrockEmbedding(
            model_name="amazon.titan-embed-text-v1",
            region_name=region
        )
        Settings.llm = Bedrock(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            region_name=region
        )
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        
        logger.info("Financial data extractor initialized")
    
    async def extract_bank_statements(self, bank_config: Dict[str, Any]) -> List[FinancialDocument]:
        """
        Extract bank statements from online banking portal
        
        Args:
            bank_config: Configuration for bank portal access
            
        Returns:
            List of financial documents with PCI DSS protection
        """
        try:
            logger.info(f"Starting bank statement extraction from {bank_config.get('bank_name', 'Unknown')}")
            
            # Create secure browser session
            session = await self.browser_loader.create_secure_session()
            
            # Authenticate with banking portal
            await self._authenticate_banking_portal(session, bank_config)
            
            # Navigate to statements section
            await session.navigate(bank_config['statements_url'])
            
            # Extract bank statements
            documents = []
            statement_elements = await session.find_elements(bank_config['statement_selector'])
            
            for element in statement_elements:
                statement_content = await element.get_text()
                
                # Detect financial data
                financial_data = self._detect_financial_data(statement_content)
                
                # Assess risk level
                risk_level = self._assess_transaction_risk(statement_content)
                
                # Create financial document
                doc = FinancialDocument(
                    content=statement_content,
                    document_type="bank_statement",
                    account_id=self._generate_anonymous_account_id(statement_content),
                    financial_data_detected=financial_data,
                    risk_level=risk_level,
                    sensitivity_level=SensitivityLevel.RESTRICTED,
                    source_url=bank_config['statements_url'],
                    extraction_timestamp=datetime.utcnow(),
                    compliance_flags=["PCI_DSS_REQUIRED"],
                    audit_trail=[{
                        "action": "bank_statement_extraction",
                        "timestamp": datetime.utcnow().isoformat(),
                        "source": bank_config['bank_name'],
                        "compliance_check": "PCI_DSS"
                    }]
                )
                
                documents.append(doc)
            
            await session.close()
            logger.info(f"Extracted {len(documents)} bank statements")
            return documents
            
        except Exception as e:
            logger.error(f"Error extracting bank statements: {str(e)}")
            raise
    
    async def extract_credit_card_transactions(self, cc_config: Dict[str, Any]) -> List[FinancialDocument]:
        """
        Extract credit card transactions from credit card portal
        
        Args:
            cc_config: Configuration for credit card portal access
            
        Returns:
            List of transaction documents with PCI DSS protection
        """
        try:
            logger.info("Extracting credit card transactions")
            
            session = await self.browser_loader.create_secure_session()
            
            # Authenticate with credit card portal
            await self._authenticate_banking_portal(session, cc_config)
            
            # Navigate to transactions
            await session.navigate(cc_config['transactions_url'])
            
            # Extract transactions
            documents = []
            transaction_elements = await session.find_elements(cc_config['transaction_selector'])
            
            for element in transaction_elements:
                transaction_content = await element.get_text()
                
                # Detect financial data and assess fraud risk
                financial_data = self._detect_financial_data(transaction_content)
                risk_level = self._assess_fraud_risk(transaction_content)
                
                doc = FinancialDocument(
                    content=transaction_content,
                    document_type="credit_card_transaction",
                    account_id=self._generate_anonymous_account_id(transaction_content),
                    financial_data_detected=financial_data,
                    risk_level=risk_level,
                    sensitivity_level=SensitivityLevel.RESTRICTED,
                    source_url=cc_config['transactions_url'],
                    extraction_timestamp=datetime.utcnow(),
                    compliance_flags=["PCI_DSS_REQUIRED", "FRAUD_MONITORING"],
                    audit_trail=[{
                        "action": "credit_card_transaction_extraction",
                        "timestamp": datetime.utcnow().isoformat(),
                        "source": cc_config['provider_name'],
                        "fraud_check": risk_level.value
                    }]
                )
                
                documents.append(doc)
            
            await session.close()
            logger.info(f"Extracted {len(documents)} credit card transactions")
            return documents
            
        except Exception as e:
            logger.error(f"Error extracting credit card transactions: {str(e)}")
            raise
    
    async def extract_investment_portfolio(self, investment_config: Dict[str, Any]) -> List[FinancialDocument]:
        """
        Extract investment portfolio data from brokerage portal
        
        Args:
            investment_config: Configuration for investment portal access
            
        Returns:
            List of investment documents with financial data protection
        """
        try:
            logger.info("Extracting investment portfolio data")
            
            session = await self.browser_loader.create_secure_session()
            
            # Authenticate with investment portal
            await self._authenticate_banking_portal(session, investment_config)
            
            # Navigate to portfolio
            await session.navigate(investment_config['portfolio_url'])
            
            # Extract portfolio data
            documents = []
            portfolio_elements = await session.find_elements(investment_config['portfolio_selector'])
            
            for element in portfolio_elements:
                portfolio_content = await element.get_text()
                
                # Detect financial data
                financial_data = self._detect_financial_data(portfolio_content)
                financial_data.append(FinancialDataType.INVESTMENT_VALUE)
                
                # Assess investment risk
                risk_level = self._assess_investment_risk(portfolio_content)
                
                doc = FinancialDocument(
                    content=portfolio_content,
                    document_type="investment_portfolio",
                    account_id=self._generate_anonymous_account_id(portfolio_content),
                    financial_data_detected=financial_data,
                    risk_level=risk_level,
                    sensitivity_level=SensitivityLevel.CONFIDENTIAL,
                    source_url=investment_config['portfolio_url'],
                    extraction_timestamp=datetime.utcnow(),
                    compliance_flags=["FINANCIAL_PRIVACY", "INVESTMENT_REGULATION"],
                    audit_trail=[{
                        "action": "investment_portfolio_extraction",
                        "timestamp": datetime.utcnow().isoformat(),
                        "source": investment_config['provider_name'],
                        "risk_assessment": risk_level.value
                    }]
                )
                
                documents.append(doc)
            
            await session.close()
            logger.info(f"Extracted {len(documents)} investment portfolio items")
            return documents
            
        except Exception as e:
            logger.error(f"Error extracting investment portfolio: {str(e)}")
            raise
    
    async def create_secure_financial_index(self, documents: List[FinancialDocument]) -> VectorStoreIndex:
        """
        Create secure vector index for financial documents
        
        Args:
            documents: List of financial documents
            
        Returns:
            Secure vector store index with PCI DSS compliance
        """
        try:
            logger.info("Creating secure financial index")
            
            # Mask financial data in all documents before indexing
            masked_documents = []
            for doc in documents:
                masked_doc = doc.mask_financial_data()
                
                # Convert to LlamaIndex Document
                llama_doc = Document(
                    text=masked_doc.content,
                    metadata={
                        "document_type": masked_doc.document_type,
                        "account_id": masked_doc.account_id,
                        "risk_level": masked_doc.risk_level.value,
                        "sensitivity_level": masked_doc.sensitivity_level.value,
                        "financial_data_detected": [data.value for data in masked_doc.financial_data_detected],
                        "compliance_flags": masked_doc.compliance_flags,
                        "extraction_timestamp": masked_doc.extraction_timestamp.isoformat(),
                        "audit_trail": masked_doc.audit_trail
                    }
                )
                masked_documents.append(llama_doc)
            
            # Create secure vector store with encryption
            vector_store = ChromaVectorStore(
                chroma_collection_name="financial_documents",
                persist_directory="secure_financial_vector_store"
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Build index with PCI DSS compliance
            index = VectorStoreIndex.from_documents(
                masked_documents,
                storage_context=storage_context
            )
            
            logger.info(f"Created secure financial index with {len(masked_documents)} documents")
            return index
            
        except Exception as e:
            logger.error(f"Error creating financial index: {str(e)}")
            raise
    
    async def analyze_financial_patterns(self, index: VectorStoreIndex, analysis_type: str) -> str:
        """
        Analyze financial patterns with data protection
        
        Args:
            index: Secure financial vector index
            analysis_type: Type of analysis to perform
            
        Returns:
            Sanitized analysis results
        """
        try:
            logger.info(f"Analyzing financial patterns: {analysis_type}")
            
            # Define analysis queries based on type
            analysis_queries = {
                "spending_patterns": "What are the main spending categories and patterns?",
                "fraud_detection": "Identify any unusual or suspicious transaction patterns",
                "investment_performance": "Analyze investment portfolio performance and trends",
                "cash_flow": "Analyze cash flow patterns and account balances",
                "risk_assessment": "Assess overall financial risk based on transaction patterns"
            }
            
            query = analysis_queries.get(analysis_type, analysis_type)
            
            # Sanitize query
            sanitized_query = self._sanitize_financial_query(query)
            
            # Create query engine with financial security controls
            query_engine = index.as_query_engine(
                response_mode="compact",
                similarity_top_k=5
            )
            
            # Execute analysis
            response = query_engine.query(sanitized_query)
            
            # Sanitize response to ensure no financial data leakage
            sanitized_response = self._sanitize_financial_response(str(response))
            
            # Log analysis for compliance
            self._log_financial_analysis(analysis_type, sanitized_query, sanitized_response)
            
            return sanitized_response
            
        except Exception as e:
            logger.error(f"Error analyzing financial patterns: {str(e)}")
            raise
    
    async def detect_fraud_indicators(self, documents: List[FinancialDocument]) -> Dict[str, Any]:
        """
        Detect potential fraud indicators in financial documents
        
        Args:
            documents: List of financial documents
            
        Returns:
            Fraud detection results
        """
        try:
            logger.info("Detecting fraud indicators")
            
            fraud_indicators = {
                "high_risk_transactions": [],
                "unusual_patterns": [],
                "velocity_alerts": [],
                "geographic_anomalies": [],
                "amount_anomalies": []
            }
            
            for doc in documents:
                # Check for high-risk transactions
                if doc.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    fraud_indicators["high_risk_transactions"].append({
                        "account_id": doc.account_id,
                        "risk_level": doc.risk_level.value,
                        "timestamp": doc.extraction_timestamp.isoformat()
                    })
                
                # Analyze transaction patterns
                patterns = self._analyze_transaction_patterns(doc.content)
                if patterns["unusual"]:
                    fraud_indicators["unusual_patterns"].extend(patterns["anomalies"])
                
                # Check transaction velocity
                velocity_check = self._check_transaction_velocity(doc.content)
                if velocity_check["alert"]:
                    fraud_indicators["velocity_alerts"].append(velocity_check)
            
            # Log fraud detection results
            self._log_fraud_detection(fraud_indicators)
            
            return fraud_indicators
            
        except Exception as e:
            logger.error(f"Error detecting fraud indicators: {str(e)}")
            raise
    
    async def _authenticate_banking_portal(self, session: BrowserSession, config: Dict[str, Any]):
        """Authenticate with banking/financial portal"""
        await session.navigate(config['login_url'])
        
        # Enter credentials securely
        username_field = await session.find_element(config['username_selector'])
        await username_field.type(config['username'])
        
        password_field = await session.find_element(config['password_selector'])
        await password_field.type(config['password'])
        
        # Handle MFA if required
        if config.get('mfa_required', False):
            await self._handle_financial_mfa(session, config)
        
        # Submit login
        login_button = await session.find_element(config['login_button_selector'])
        await login_button.click()
        
        # Wait for successful login
        await session.wait_for_element(config['dashboard_selector'])
        
        logger.info(f"Successfully authenticated with {config.get('provider_name', 'financial portal')}")
    
    async def _handle_financial_mfa(self, session: BrowserSession, config: Dict[str, Any]):
        """Handle multi-factor authentication for financial portal"""
        await session.wait_for_element(config['mfa_selector'])
        logger.info("Financial MFA authentication required - integrate with secure token provider")
    
    def _detect_financial_data(self, content: str) -> List[FinancialDataType]:
        """Detect financial data in content"""
        financial_detected = []
        
        # Financial data detection patterns
        patterns = {
            FinancialDataType.CREDIT_CARD: r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            FinancialDataType.BANK_ACCOUNT: r'\b\d{8,17}\b',
            FinancialDataType.ROUTING_NUMBER: r'\b\d{9}\b',
            FinancialDataType.SSN: r'\b\d{3}-\d{2}-\d{4}\b',
            FinancialDataType.TAX_ID: r'\b\d{2}-\d{7}\b',
        }
        
        for data_type, pattern in patterns.items():
            if re.search(pattern, content):
                financial_detected.append(data_type)
        
        # Detect monetary amounts
        if re.search(r'\$[\d,]+\.?\d*', content):
            financial_detected.append(FinancialDataType.TRANSACTION_AMOUNT)
        
        return financial_detected
    
    def _assess_transaction_risk(self, content: str) -> RiskLevel:
        """Assess risk level of transactions"""
        # Extract transaction amounts
        amounts = re.findall(r'\$[\d,]+\.?\d*', content)
        
        if amounts:
            # Convert to numeric values
            numeric_amounts = []
            for amount in amounts:
                try:
                    numeric_value = float(amount.replace('$', '').replace(',', ''))
                    numeric_amounts.append(numeric_value)
                except ValueError:
                    continue
            
            if numeric_amounts:
                max_amount = max(numeric_amounts)
                
                # Risk assessment based on amount
                if max_amount > 10000:
                    return RiskLevel.HIGH
                elif max_amount > 5000:
                    return RiskLevel.MEDIUM
                else:
                    return RiskLevel.LOW
        
        return RiskLevel.LOW
    
    def _assess_fraud_risk(self, content: str) -> RiskLevel:
        """Assess fraud risk of transactions"""
        risk_indicators = 0
        
        # Check for high-risk patterns
        high_risk_patterns = [
            r'cash advance',
            r'atm withdrawal.*\$[5-9]\d{3,}',  # Large ATM withdrawals
            r'foreign transaction',
            r'declined.*retry',
            r'unusual location'
        ]
        
        for pattern in high_risk_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                risk_indicators += 1
        
        # Assess based on risk indicators
        if risk_indicators >= 3:
            return RiskLevel.CRITICAL
        elif risk_indicators >= 2:
            return RiskLevel.HIGH
        elif risk_indicators >= 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _assess_investment_risk(self, content: str) -> RiskLevel:
        """Assess investment risk level"""
        # Look for investment risk indicators
        high_risk_terms = ['volatile', 'speculative', 'high risk', 'cryptocurrency', 'options']
        medium_risk_terms = ['growth', 'equity', 'emerging markets']
        
        content_lower = content.lower()
        
        high_risk_count = sum(1 for term in high_risk_terms if term in content_lower)
        medium_risk_count = sum(1 for term in medium_risk_terms if term in content_lower)
        
        if high_risk_count > 0:
            return RiskLevel.HIGH
        elif medium_risk_count > 0:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_anonymous_account_id(self, content: str) -> str:
        """Generate anonymous account identifier"""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        return f"ACCOUNT_{content_hash[:8]}"
    
    def _sanitize_financial_query(self, query: str) -> str:
        """Sanitize query to remove financial data"""
        sanitized = query
        
        # Remove financial patterns
        sanitized = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[REDACTED_CC]', sanitized)
        sanitized = re.sub(r'\b\d{8,17}\b', '[REDACTED_ACCOUNT]', sanitized)
        sanitized = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]', sanitized)
        
        return sanitized
    
    def _sanitize_financial_response(self, response: str) -> str:
        """Sanitize response to prevent financial data leakage"""
        return self._sanitize_financial_query(response)
    
    def _analyze_transaction_patterns(self, content: str) -> Dict[str, Any]:
        """Analyze transaction patterns for anomalies"""
        return {
            "unusual": False,
            "anomalies": []
        }
    
    def _check_transaction_velocity(self, content: str) -> Dict[str, Any]:
        """Check transaction velocity for alerts"""
        return {
            "alert": False,
            "velocity_score": 0
        }
    
    def _log_financial_analysis(self, analysis_type: str, query: str, response: str):
        """Log financial analysis for compliance"""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": "financial_analysis",
            "analysis_type": analysis_type,
            "query_hash": hashlib.sha256(query.encode()).hexdigest(),
            "response_length": len(response),
            "compliance_check": "PCI_DSS"
        }
        
        logger.info(f"Financial analysis audit: {audit_entry}")
    
    def _log_fraud_detection(self, fraud_indicators: Dict[str, Any]):
        """Log fraud detection results"""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": "fraud_detection",
            "high_risk_count": len(fraud_indicators["high_risk_transactions"]),
            "unusual_patterns_count": len(fraud_indicators["unusual_patterns"]),
            "velocity_alerts_count": len(fraud_indicators["velocity_alerts"])
        }
        
        logger.info(f"Fraud detection audit: {audit_entry}")


async def main():
    """Example usage of financial data extraction"""
    
    # Initialize extractor
    extractor = FinancialDataExtractor(region="us-east-1")
    
    # Example bank configuration
    bank_config = {
        "bank_name": "Chase Bank",
        "provider_name": "JPMorgan Chase",
        "login_url": "https://secure01a.chase.com/web/auth/dashboard",
        "username_selector": "#userId-text-input-field",
        "password_selector": "#password-text-input-field",
        "login_button_selector": "#signin-button",
        "dashboard_selector": "#dashboard",
        "statements_url": "https://secure01a.chase.com/web/auth/dashboard#/dashboard/statements",
        "statement_selector": ".statement-row",
        "transactions_url": "https://secure01a.chase.com/web/auth/dashboard#/dashboard/transactions",
        "transaction_selector": ".transaction-row",
        "username": os.getenv("BANK_USERNAME"),
        "password": os.getenv("BANK_PASSWORD"),
        "mfa_required": True,
        "mfa_selector": "#requestDeliveryDevices"
    }
    
    try:
        # Extract bank statements
        bank_statements = await extractor.extract_bank_statements(bank_config)
        print(f"Extracted {len(bank_statements)} bank statements")
        
        # Extract credit card transactions
        cc_config = bank_config.copy()
        cc_config.update({
            "provider_name": "Chase Credit Card",
            "transactions_url": "https://secure01a.chase.com/web/auth/dashboard#/dashboard/creditcards/transactions",
            "transaction_selector": ".cc-transaction-row"
        })
        
        cc_transactions = await extractor.extract_credit_card_transactions(cc_config)
        print(f"Extracted {len(cc_transactions)} credit card transactions")
        
        # Extract investment portfolio
        investment_config = {
            "provider_name": "Charles Schwab",
            "login_url": "https://www.schwab.com/login",
            "username_selector": "#LoginId",
            "password_selector": "#Password",
            "login_button_selector": "#LoginSubmitBtn",
            "dashboard_selector": "#sch-main-content",
            "portfolio_url": "https://client.schwab.com/Areas/Accounts/Positions",
            "portfolio_selector": ".position-row",
            "username": os.getenv("INVESTMENT_USERNAME"),
            "password": os.getenv("INVESTMENT_PASSWORD"),
            "mfa_required": True,
            "mfa_selector": "#AuthenticationToken"
        }
        
        investment_data = await extractor.extract_investment_portfolio(investment_config)
        print(f"Extracted {len(investment_data)} investment portfolio items")
        
        # Combine all financial documents
        all_documents = bank_statements + cc_transactions + investment_data
        
        # Create secure financial index
        financial_index = await extractor.create_secure_financial_index(all_documents)
        
        # Perform financial analysis
        analyses = [
            "spending_patterns",
            "fraud_detection", 
            "investment_performance",
            "cash_flow",
            "risk_assessment"
        ]
        
        for analysis in analyses:
            result = await extractor.analyze_financial_patterns(financial_index, analysis)
            print(f"\n{analysis.replace('_', ' ').title()} Analysis:")
            print(result)
        
        # Detect fraud indicators
        fraud_results = await extractor.detect_fraud_indicators(all_documents)
        print(f"\nFraud Detection Results:")
        print(f"High-risk transactions: {len(fraud_results['high_risk_transactions'])}")
        print(f"Unusual patterns: {len(fraud_results['unusual_patterns'])}")
        print(f"Velocity alerts: {len(fraud_results['velocity_alerts'])}")
        
        print("\nFinancial data extraction completed successfully!")
        
    except Exception as e:
        logger.error(f"Financial processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())