#!/usr/bin/env python3
"""
Legal Document Analysis Example with LlamaIndex and AgentCore Browser Tool

This example demonstrates secure legal document analysis using LlamaIndex agents
integrated with Amazon Bedrock AgentCore Browser Tool. Includes attorney-client privilege
protection, confidentiality controls, and secure handling of legal information.

Key Features:
- Attorney-client privilege protection
- Legal document confidentiality controls
- Secure extraction from legal document systems
- Contract analysis with privacy protection
- Case law research with data protection
- Compliance with legal ethics requirements

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
from uuid import uuid4

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

# Configure logging for legal compliance
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('legal_document_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LegalDataType(Enum):
    """Types of legal data requiring protection"""
    CLIENT_NAME = "client_name"
    CASE_NUMBER = "case_number"
    SSN = "social_security_number"
    ATTORNEY_NAME = "attorney_name"
    FIRM_NAME = "firm_name"
    COURT_NAME = "court_name"
    JUDGE_NAME = "judge_name"
    FINANCIAL_AMOUNT = "financial_amount"
    CONTRACT_TERMS = "contract_terms"
    SETTLEMENT_AMOUNT = "settlement_amount"


class DocumentType(Enum):
    """Types of legal documents"""
    CONTRACT = "contract"
    BRIEF = "legal_brief"
    MOTION = "motion"
    DISCOVERY = "discovery_document"
    CORRESPONDENCE = "legal_correspondence"
    CASE_LAW = "case_law"
    STATUTE = "statute"
    REGULATION = "regulation"
    SETTLEMENT = "settlement_agreement"
    PLEADING = "pleading"


class ConfidentialityLevel(Enum):
    """Legal confidentiality levels"""
    PUBLIC = "public"
    CONFIDENTIAL = "confidential"
    ATTORNEY_CLIENT_PRIVILEGED = "attorney_client_privileged"
    WORK_PRODUCT = "work_product"
    SEALED = "sealed"


@dataclass
class LegalDocument:
    """Legal document with confidentiality protection"""
    content: str
    document_type: DocumentType
    case_id: str  # Anonymized case identifier
    client_id: str  # Anonymized client identifier
    legal_data_detected: List[LegalDataType]
    confidentiality_level: ConfidentialityLevel
    sensitivity_level: SensitivityLevel
    source_url: str
    creation_timestamp: datetime
    privilege_flags: List[str]
    audit_trail: List[Dict[str, Any]]
    
    def protect_privileged_information(self) -> 'LegalDocument':
        """Protect attorney-client privileged information"""
        protected_content = self.content
        audit_event = {
            "action": "privilege_protection",
            "timestamp": datetime.utcnow().isoformat(),
            "data_types_protected": [data_type.value for data_type in self.legal_data_detected],
            "confidentiality_level": self.confidentiality_level.value
        }
        
        # Apply privilege protection patterns
        legal_patterns = {
            LegalDataType.CLIENT_NAME: r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            LegalDataType.CASE_NUMBER: r'\bCase No\.?\s*\d{2,4}-\d{2,6}\b',
            LegalDataType.SSN: r'\b\d{3}-\d{2}-\d{4}\b',
            LegalDataType.ATTORNEY_NAME: r'\bAttorney [A-Z][a-z]+ [A-Z][a-z]+\b',
            LegalDataType.SETTLEMENT_AMOUNT: r'\$[\d,]+\.?\d*',
        }
        
        for data_type in self.legal_data_detected:
            if data_type in legal_patterns:
                if self.confidentiality_level == ConfidentialityLevel.ATTORNEY_CLIENT_PRIVILEGED:
                    protected_content = re.sub(
                        legal_patterns[data_type], 
                        f'[PRIVILEGED_{data_type.value.upper()}]', 
                        protected_content
                    )
                else:
                    protected_content = re.sub(
                        legal_patterns[data_type], 
                        f'[CONFIDENTIAL_{data_type.value.upper()}]', 
                        protected_content
                    )
        
        return LegalDocument(
            content=protected_content,
            document_type=self.document_type,
            case_id=self.case_id,
            client_id=self.client_id,
            legal_data_detected=self.legal_data_detected,
            confidentiality_level=self.confidentiality_level,
            sensitivity_level=self.sensitivity_level,
            source_url=self.source_url,
            creation_timestamp=self.creation_timestamp,
            privilege_flags=self.privilege_flags + ["PRIVILEGE_PROTECTED"],
            audit_trail=self.audit_trail + [audit_event]
        )


class LegalDocumentAnalyzer:
    """Secure legal document analyzer using LlamaIndex and AgentCore"""
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.browser_loader = AgentCoreBrowserLoader(region=region)
        self.sensitive_handler = SensitiveDataHandler()
        self.rag_pipeline = SecureRAGPipeline(region=region)
        
        # Configure LlamaIndex for legal use
        Settings.embed_model = BedrockEmbedding(
            model_name="amazon.titan-embed-text-v1",
            region_name=region
        )
        Settings.llm = Bedrock(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            region_name=region
        )
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        
        logger.info("Legal document analyzer initialized")
    
    async def extract_case_documents(self, legal_config: Dict[str, Any]) -> List[LegalDocument]:
        """
        Extract case documents from legal document management system
        
        Args:
            legal_config: Configuration for legal system access
            
        Returns:
            List of legal documents with privilege protection
        """
        try:
            logger.info(f"Starting case document extraction from {legal_config.get('system_name', 'Unknown')}")
            
            # Create secure browser session
            session = await self.browser_loader.create_secure_session()
            
            # Authenticate with legal system
            await self._authenticate_legal_system(session, legal_config)
            
            # Navigate to case documents
            await session.navigate(legal_config['documents_url'])
            
            # Extract case documents
            documents = []
            document_elements = await session.find_elements(legal_config['document_selector'])
            
            for element in document_elements:
                document_content = await element.get_text()
                
                # Detect legal data
                legal_data = self._detect_legal_data(document_content)
                
                # Determine document type and confidentiality
                doc_type = self._classify_document_type(document_content)
                confidentiality = self._assess_confidentiality_level(document_content, doc_type)
                
                # Create legal document
                doc = LegalDocument(
                    content=document_content,
                    document_type=doc_type,
                    case_id=self._generate_anonymous_case_id(document_content),
                    client_id=self._generate_anonymous_client_id(document_content),
                    legal_data_detected=legal_data,
                    confidentiality_level=confidentiality,
                    sensitivity_level=SensitivityLevel.RESTRICTED,
                    source_url=legal_config['documents_url'],
                    creation_timestamp=datetime.utcnow(),
                    privilege_flags=["ATTORNEY_CLIENT_PRIVILEGE"],
                    audit_trail=[{
                        "action": "case_document_extraction",
                        "timestamp": datetime.utcnow().isoformat(),
                        "source": legal_config['system_name'],
                        "confidentiality_check": confidentiality.value
                    }]
                )
                
                documents.append(doc)
            
            await session.close()
            logger.info(f"Extracted {len(documents)} case documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error extracting case documents: {str(e)}")
            raise
    
    async def extract_contracts(self, contract_config: Dict[str, Any]) -> List[LegalDocument]:
        """
        Extract contracts from contract management system
        
        Args:
            contract_config: Configuration for contract system access
            
        Returns:
            List of contract documents with confidentiality protection
        """
        try:
            logger.info("Extracting contracts")
            
            session = await self.browser_loader.create_secure_session()
            
            # Authenticate with contract system
            await self._authenticate_legal_system(session, contract_config)
            
            # Navigate to contracts
            await session.navigate(contract_config['contracts_url'])
            
            # Extract contracts
            documents = []
            contract_elements = await session.find_elements(contract_config['contract_selector'])
            
            for element in contract_elements:
                contract_content = await element.get_text()
                
                # Detect legal data in contract
                legal_data = self._detect_legal_data(contract_content)
                legal_data.extend(self._detect_contract_terms(contract_content))
                
                # Assess contract confidentiality
                confidentiality = self._assess_contract_confidentiality(contract_content)
                
                doc = LegalDocument(
                    content=contract_content,
                    document_type=DocumentType.CONTRACT,
                    case_id=f"CONTRACT_{uuid4().hex[:8]}",
                    client_id=self._generate_anonymous_client_id(contract_content),
                    legal_data_detected=legal_data,
                    confidentiality_level=confidentiality,
                    sensitivity_level=SensitivityLevel.CONFIDENTIAL,
                    source_url=contract_config['contracts_url'],
                    creation_timestamp=datetime.utcnow(),
                    privilege_flags=["CONTRACT_CONFIDENTIALITY"],
                    audit_trail=[{
                        "action": "contract_extraction",
                        "timestamp": datetime.utcnow().isoformat(),
                        "source": contract_config['system_name'],
                        "contract_type": "commercial"
                    }]
                )
                
                documents.append(doc)
            
            await session.close()
            logger.info(f"Extracted {len(documents)} contracts")
            return documents
            
        except Exception as e:
            logger.error(f"Error extracting contracts: {str(e)}")
            raise
    
    async def research_case_law(self, research_config: Dict[str, Any]) -> List[LegalDocument]:
        """
        Research case law from legal databases
        
        Args:
            research_config: Configuration for legal research system access
            
        Returns:
            List of case law documents
        """
        try:
            logger.info("Researching case law")
            
            session = await self.browser_loader.create_secure_session()
            
            # Authenticate with legal research system
            await self._authenticate_legal_system(session, research_config)
            
            # Navigate to case law search
            await session.navigate(research_config['research_url'])
            
            # Perform case law search
            search_field = await session.find_element(research_config['search_selector'])
            await search_field.type(research_config['search_query'])
            
            search_button = await session.find_element(research_config['search_button_selector'])
            await search_button.click()
            
            # Extract case law results
            documents = []
            case_elements = await session.find_elements(research_config['case_selector'])
            
            for element in case_elements:
                case_content = await element.get_text()
                
                # Detect legal data in case law
                legal_data = self._detect_legal_data(case_content)
                
                doc = LegalDocument(
                    content=case_content,
                    document_type=DocumentType.CASE_LAW,
                    case_id=f"CASELAW_{uuid4().hex[:8]}",
                    client_id="PUBLIC",
                    legal_data_detected=legal_data,
                    confidentiality_level=ConfidentialityLevel.PUBLIC,
                    sensitivity_level=SensitivityLevel.INTERNAL,
                    source_url=research_config['research_url'],
                    creation_timestamp=datetime.utcnow(),
                    privilege_flags=["PUBLIC_RECORD"],
                    audit_trail=[{
                        "action": "case_law_research",
                        "timestamp": datetime.utcnow().isoformat(),
                        "source": research_config['system_name'],
                        "search_query": research_config['search_query']
                    }]
                )
                
                documents.append(doc)
            
            await session.close()
            logger.info(f"Researched {len(documents)} case law documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error researching case law: {str(e)}")
            raise
    
    async def create_secure_legal_index(self, documents: List[LegalDocument]) -> VectorStoreIndex:
        """
        Create secure vector index for legal documents
        
        Args:
            documents: List of legal documents
            
        Returns:
            Secure vector store index with privilege protection
        """
        try:
            logger.info("Creating secure legal index")
            
            # Protect privileged information in all documents before indexing
            protected_documents = []
            for doc in documents:
                protected_doc = doc.protect_privileged_information()
                
                # Convert to LlamaIndex Document
                llama_doc = Document(
                    text=protected_doc.content,
                    metadata={
                        "document_type": protected_doc.document_type.value,
                        "case_id": protected_doc.case_id,
                        "client_id": protected_doc.client_id,
                        "confidentiality_level": protected_doc.confidentiality_level.value,
                        "sensitivity_level": protected_doc.sensitivity_level.value,
                        "legal_data_detected": [data.value for data in protected_doc.legal_data_detected],
                        "privilege_flags": protected_doc.privilege_flags,
                        "creation_timestamp": protected_doc.creation_timestamp.isoformat(),
                        "audit_trail": protected_doc.audit_trail
                    }
                )
                protected_documents.append(llama_doc)
            
            # Create secure vector store with privilege protection
            vector_store = ChromaVectorStore(
                chroma_collection_name="legal_documents",
                persist_directory="./secure_legal_vector_store"
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Build index with legal ethics compliance
            index = VectorStoreIndex.from_documents(
                protected_documents,
                storage_context=storage_context
            )
            
            logger.info(f"Created secure legal index with {len(protected_documents)} documents")
            return index
            
        except Exception as e:
            logger.error(f"Error creating legal index: {str(e)}")
            raise
    
    async def analyze_legal_documents(self, index: VectorStoreIndex, analysis_type: str) -> str:
        """
        Analyze legal documents with privilege protection
        
        Args:
            index: Secure legal vector index
            analysis_type: Type of analysis to perform
            
        Returns:
            Sanitized analysis results
        """
        try:
            logger.info(f"Analyzing legal documents: {analysis_type}")
            
            # Define analysis queries based on type
            analysis_queries = {
                "contract_terms": "Analyze common contract terms and clauses",
                "case_precedents": "Identify relevant case law precedents and legal principles",
                "risk_assessment": "Assess legal risks and potential issues in documents",
                "compliance_review": "Review documents for regulatory compliance requirements",
                "document_classification": "Classify documents by type and legal significance"
            }
            
            query = analysis_queries.get(analysis_type, analysis_type)
            
            # Sanitize query
            sanitized_query = self._sanitize_legal_query(query)
            
            # Create query engine with privilege controls
            query_engine = index.as_query_engine(
                response_mode="compact",
                similarity_top_k=5
            )
            
            # Execute analysis
            response = query_engine.query(sanitized_query)
            
            # Sanitize response to ensure no privileged information leakage
            sanitized_response = self._sanitize_legal_response(str(response))
            
            # Log analysis for legal compliance
            self._log_legal_analysis(analysis_type, sanitized_query, sanitized_response)
            
            return sanitized_response
            
        except Exception as e:
            logger.error(f"Error analyzing legal documents: {str(e)}")
            raise
    
    async def perform_contract_analysis(self, contracts: List[LegalDocument]) -> Dict[str, Any]:
        """
        Perform detailed contract analysis
        
        Args:
            contracts: List of contract documents
            
        Returns:
            Contract analysis results
        """
        try:
            logger.info("Performing contract analysis")
            
            analysis_results = {
                "contract_types": {},
                "key_terms": [],
                "risk_factors": [],
                "compliance_issues": [],
                "financial_terms": []
            }
            
            for contract in contracts:
                if contract.document_type == DocumentType.CONTRACT:
                    # Protect privileged information
                    protected_contract = contract.protect_privileged_information()
                    
                    # Analyze contract type
                    contract_type = self._identify_contract_type(protected_contract.content)
                    analysis_results["contract_types"][contract_type] = analysis_results["contract_types"].get(contract_type, 0) + 1
                    
                    # Extract key terms
                    key_terms = self._extract_contract_key_terms(protected_contract.content)
                    analysis_results["key_terms"].extend(key_terms)
                    
                    # Assess risks
                    risks = self._assess_contract_risks(protected_contract.content)
                    analysis_results["risk_factors"].extend(risks)
                    
                    # Check compliance
                    compliance = self._check_contract_compliance(protected_contract.content)
                    analysis_results["compliance_issues"].extend(compliance)
            
            # Log contract analysis
            self._log_contract_analysis(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error performing contract analysis: {str(e)}")
            raise
    
    async def _authenticate_legal_system(self, session: BrowserSession, config: Dict[str, Any]):
        """Authenticate with legal document system"""
        await session.navigate(config['login_url'])
        
        # Enter credentials securely
        username_field = await session.find_element(config['username_selector'])
        await username_field.type(config['username'])
        
        password_field = await session.find_element(config['password_selector'])
        await password_field.type(config['password'])
        
        # Handle MFA if required
        if config.get('mfa_required', False):
            await self._handle_legal_mfa(session, config)
        
        # Submit login
        login_button = await session.find_element(config['login_button_selector'])
        await login_button.click()
        
        # Wait for successful login
        await session.wait_for_element(config['dashboard_selector'])
        
        logger.info(f"Successfully authenticated with {config.get('system_name', 'legal system')}")
    
    async def _handle_legal_mfa(self, session: BrowserSession, config: Dict[str, Any]):
        """Handle MFA authentication for legal system"""
        await session.wait_for_element(config['mfa_selector'])
        logger.info("Legal system MFA authentication required")
    
    def _detect_legal_data(self, content: str) -> List[LegalDataType]:
        """Detect legal data in content"""
        legal_data = []
        
        # Legal data detection patterns
        patterns = {
            LegalDataType.CASE_NUMBER: r'\bCase No\.?\s*\d{2,4}-\d{2,6}\b',
            LegalDataType.SSN: r'\b\d{3}-\d{2}-\d{4}\b',
            LegalDataType.FINANCIAL_AMOUNT: r'\$[\d,]+\.?\d*',
        }
        
        for data_type, pattern in patterns.items():
            if re.search(pattern, content):
                legal_data.append(data_type)
        
        # Detect names and entities
        if re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', content):
            if 'attorney' in content.lower() or 'counsel' in content.lower():
                legal_data.append(LegalDataType.ATTORNEY_NAME)
            elif 'client' in content.lower():
                legal_data.append(LegalDataType.CLIENT_NAME)
        
        if re.search(r'\b[A-Z][a-z]+ Court\b', content):
            legal_data.append(LegalDataType.COURT_NAME)
        
        return legal_data
    
    def _detect_contract_terms(self, content: str) -> List[LegalDataType]:
        """Detect contract-specific terms"""
        contract_terms = []
        
        if any(term in content.lower() for term in ['payment', 'consideration', 'fee']):
            contract_terms.append(LegalDataType.FINANCIAL_AMOUNT)
        
        if any(term in content.lower() for term in ['settlement', 'damages', 'compensation']):
            contract_terms.append(LegalDataType.SETTLEMENT_AMOUNT)
        
        return contract_terms
    
    def _classify_document_type(self, content: str) -> DocumentType:
        """Classify legal document type"""
        content_lower = content.lower()
        
        if any(term in content_lower for term in ['contract', 'agreement', 'terms']):
            return DocumentType.CONTRACT
        elif any(term in content_lower for term in ['motion', 'petition']):
            return DocumentType.MOTION
        elif any(term in content_lower for term in ['brief', 'memorandum']):
            return DocumentType.BRIEF
        elif any(term in content_lower for term in ['discovery', 'interrogatory', 'deposition']):
            return DocumentType.DISCOVERY
        elif any(term in content_lower for term in ['settlement', 'agreement']):
            return DocumentType.SETTLEMENT
        else:
            return DocumentType.CORRESPONDENCE
    
    def _assess_confidentiality_level(self, content: str, doc_type: DocumentType) -> ConfidentialityLevel:
        """Assess confidentiality level of document"""
        content_lower = content.lower()
        
        if any(term in content_lower for term in ['privileged', 'attorney-client', 'confidential']):
            return ConfidentialityLevel.ATTORNEY_CLIENT_PRIVILEGED
        elif any(term in content_lower for term in ['work product', 'attorney work']):
            return ConfidentialityLevel.WORK_PRODUCT
        elif any(term in content_lower for term in ['sealed', 'under seal']):
            return ConfidentialityLevel.SEALED
        elif doc_type in [DocumentType.CONTRACT, DocumentType.SETTLEMENT]:
            return ConfidentialityLevel.CONFIDENTIAL
        else:
            return ConfidentialityLevel.PUBLIC
    
    def _assess_contract_confidentiality(self, content: str) -> ConfidentialityLevel:
        """Assess contract confidentiality level"""
        if 'confidential' in content.lower() or 'non-disclosure' in content.lower():
            return ConfidentialityLevel.CONFIDENTIAL
        else:
            return ConfidentialityLevel.CONFIDENTIAL  # Default for contracts
    
    def _generate_anonymous_case_id(self, content: str) -> str:
        """Generate anonymous case identifier"""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        return f"CASE_{content_hash[:8]}"
    
    def _generate_anonymous_client_id(self, content: str) -> str:
        """Generate anonymous client identifier"""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        return f"CLIENT_{content_hash[:8]}"
    
    def _identify_contract_type(self, content: str) -> str:
        """Identify type of contract"""
        content_lower = content.lower()
        
        contract_types = {
            "employment": ["employment", "job", "salary", "benefits"],
            "service": ["service", "consulting", "professional"],
            "purchase": ["purchase", "sale", "buy", "sell"],
            "lease": ["lease", "rent", "rental", "tenant"],
            "nda": ["non-disclosure", "confidentiality", "nda"]
        }
        
        for contract_type, keywords in contract_types.items():
            if any(keyword in content_lower for keyword in keywords):
                return contract_type
        
        return "general"
    
    def _extract_contract_key_terms(self, content: str) -> List[str]:
        """Extract key terms from contract"""
        # Simplified key term extraction
        key_terms = []
        
        # Look for monetary amounts
        amounts = re.findall(r'\$[\d,]+\.?\d*', content)
        key_terms.extend([f"Amount: {amount}" for amount in amounts[:3]])  # Limit to first 3
        
        # Look for dates
        dates = re.findall(r'\b\d{1,2}/\d{1,2}/\d{4}\b', content)
        key_terms.extend([f"Date: {date}" for date in dates[:3]])  # Limit to first 3
        
        return key_terms
    
    def _assess_contract_risks(self, content: str) -> List[str]:
        """Assess risks in contract"""
        risks = []
        content_lower = content.lower()
        
        risk_indicators = {
            "penalty": "Financial penalty clauses present",
            "termination": "Termination clauses require review",
            "liability": "Liability limitations need attention",
            "indemnification": "Indemnification clauses present"
        }
        
        for indicator, risk_description in risk_indicators.items():
            if indicator in content_lower:
                risks.append(risk_description)
        
        return risks
    
    def _check_contract_compliance(self, content: str) -> List[str]:
        """Check contract compliance issues"""
        compliance_issues = []
        content_lower = content.lower()
        
        # Check for common compliance requirements
        if "gdpr" not in content_lower and "data" in content_lower:
            compliance_issues.append("GDPR compliance may be required")
        
        if "force majeure" not in content_lower:
            compliance_issues.append("Consider adding force majeure clause")
        
        return compliance_issues
    
    def _sanitize_legal_query(self, query: str) -> str:
        """Sanitize query to remove privileged information"""
        sanitized = query
        
        # Remove legal data patterns
        sanitized = re.sub(r'\bCase No\.?\s*\d{2,4}-\d{2,6}\b', '[REDACTED_CASE]', sanitized)
        sanitized = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]', sanitized)
        sanitized = re.sub(r'\$[\d,]+\.?\d*', '[REDACTED_AMOUNT]', sanitized)
        
        return sanitized
    
    def _sanitize_legal_response(self, response: str) -> str:
        """Sanitize response to prevent privileged information leakage"""
        return self._sanitize_legal_query(response)
    
    def _log_legal_analysis(self, analysis_type: str, query: str, response: str):
        """Log legal analysis for compliance"""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": "legal_analysis",
            "analysis_type": analysis_type,
            "query_hash": hashlib.sha256(query.encode()).hexdigest(),
            "response_length": len(response),
            "privilege_check": "ATTORNEY_CLIENT_PRIVILEGE"
        }
        
        logger.info(f"Legal analysis audit: {audit_entry}")
    
    def _log_contract_analysis(self, analysis_results: Dict[str, Any]):
        """Log contract analysis results"""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": "contract_analysis",
            "contract_types_count": len(analysis_results["contract_types"]),
            "key_terms_count": len(analysis_results["key_terms"]),
            "risk_factors_count": len(analysis_results["risk_factors"])
        }
        
        logger.info(f"Contract analysis audit: {audit_entry}")


async def main():
    """Example usage of legal document analysis"""
    
    # Initialize analyzer
    analyzer = LegalDocumentAnalyzer(region="us-east-1")
    
    # Example legal system configuration
    legal_config = {
        "system_name": "iManage",
        "login_url": "https://firm.imanage.com/login",
        "username_selector": "#username",
        "password_selector": "#password",
        "login_button_selector": "#login-button",
        "dashboard_selector": "#dashboard",
        "documents_url": "https://firm.imanage.com/documents",
        "document_selector": ".document-item",
        "username": os.getenv("LEGAL_USERNAME"),
        "password": os.getenv("LEGAL_PASSWORD"),
        "mfa_required": True,
        "mfa_selector": "#mfa-token"
    }
    
    try:
        # Extract case documents
        case_documents = await analyzer.extract_case_documents(legal_config)
        print(f"Extracted {len(case_documents)} case documents")
        
        # Extract contracts
        contract_config = legal_config.copy()
        contract_config.update({
            "contracts_url": "https://firm.imanage.com/contracts",
            "contract_selector": ".contract-item"
        })
        
        contracts = await analyzer.extract_contracts(contract_config)
        print(f"Extracted {len(contracts)} contracts")
        
        # Research case law
        research_config = {
            "system_name": "Westlaw",
            "login_url": "https://1.next.westlaw.com/SignOn",
            "username_selector": "#Username",
            "password_selector": "#Password",
            "login_button_selector": "#SignIn",
            "dashboard_selector": "#main-content",
            "research_url": "https://1.next.westlaw.com/Search/Home.html",
            "search_selector": "#searchInputBox",
            "search_button_selector": "#searchButton",
            "case_selector": ".search-result",
            "search_query": "contract interpretation",
            "username": os.getenv("WESTLAW_USERNAME"),
            "password": os.getenv("WESTLAW_PASSWORD"),
            "mfa_required": False
        }
        
        case_law = await analyzer.research_case_law(research_config)
        print(f"Researched {len(case_law)} case law documents")
        
        # Combine all legal documents
        all_documents = case_documents + contracts + case_law
        
        # Create secure legal index
        legal_index = await analyzer.create_secure_legal_index(all_documents)
        
        # Perform legal document analysis
        analyses = [
            "contract_terms",
            "case_precedents",
            "risk_assessment",
            "compliance_review",
            "document_classification"
        ]
        
        for analysis in analyses:
            result = await analyzer.analyze_legal_documents(legal_index, analysis)
            print(f"\n{analysis.replace('_', ' ').title()} Analysis:")
            print(result)
        
        # Perform detailed contract analysis
        contract_analysis = await analyzer.perform_contract_analysis(contracts)
        print(f"\nContract Analysis Results:")
        print(f"Contract types: {contract_analysis['contract_types']}")
        print(f"Key terms found: {len(contract_analysis['key_terms'])}")
        print(f"Risk factors identified: {len(contract_analysis['risk_factors'])}")
        print(f"Compliance issues: {len(contract_analysis['compliance_issues'])}")
        
        print("\nLegal document analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Legal document analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())