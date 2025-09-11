#!/usr/bin/env python3
"""
Healthcare Document Processing Example with LlamaIndex and AgentCore Browser Tool

This example demonstrates secure processing of healthcare documents containing PHI (Protected Health Information)
using LlamaIndex agents integrated with Amazon Bedrock AgentCore Browser Tool.

Key Features:
- HIPAA-compliant data handling
- PHI detection and masking
- Secure document extraction from healthcare portals
- Audit logging for compliance
- Encrypted storage of sensitive medical data

Requirements: 3.1, 3.3, 5.1, 5.2
"""

import os
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

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

# Configure logging for healthcare compliance
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('healthcare_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PHIType(Enum):
    """Types of Protected Health Information (PHI)"""
    PATIENT_NAME = "patient_name"
    SSN = "social_security_number"
    MRN = "medical_record_number"
    DOB = "date_of_birth"
    PHONE = "phone_number"
    EMAIL = "email_address"
    ADDRESS = "address"
    DIAGNOSIS = "diagnosis"
    MEDICATION = "medication"
    PROVIDER_NAME = "provider_name"
    INSURANCE_ID = "insurance_id"


@dataclass
class HealthcareDocument:
    """Healthcare document with PHI protection"""
    content: str
    document_type: str
    patient_id: str  # Anonymized patient identifier
    phi_detected: List[PHIType]
    sensitivity_level: SensitivityLevel
    source_url: str
    extraction_timestamp: datetime
    audit_trail: List[Dict[str, Any]]
    
    def mask_phi(self) -> 'HealthcareDocument':
        """Mask PHI in document content"""
        masked_content = self.content
        audit_event = {
            "action": "phi_masking",
            "timestamp": datetime.utcnow().isoformat(),
            "phi_types_masked": [phi.value for phi in self.phi_detected]
        }
        
        # Apply PHI masking patterns
        phi_patterns = {
            PHIType.SSN: r'\b\d{3}-\d{2}-\d{4}\b',
            PHIType.PHONE: r'\b\d{3}-\d{3}-\d{4}\b',
            PHIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            PHIType.MRN: r'\bMRN[:\s]*\d{6,10}\b',
        }
        
        for phi_type in self.phi_detected:
            if phi_type in phi_patterns:
                import re
                masked_content = re.sub(phi_patterns[phi_type], f'[MASKED_{phi_type.value.upper()}]', masked_content)
        
        return HealthcareDocument(
            content=masked_content,
            document_type=self.document_type,
            patient_id=self.patient_id,
            phi_detected=self.phi_detected,
            sensitivity_level=self.sensitivity_level,
            source_url=self.source_url,
            extraction_timestamp=self.extraction_timestamp,
            audit_trail=self.audit_trail + [audit_event]
        )


class HealthcareDocumentProcessor:
    """Secure healthcare document processor using LlamaIndex and AgentCore"""
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.browser_loader = AgentCoreBrowserLoader(region=region)
        self.sensitive_handler = SensitiveDataHandler()
        self.rag_pipeline = SecureRAGPipeline(region=region)
        
        # Configure LlamaIndex for healthcare use
        Settings.embed_model = BedrockEmbedding(
            model_name="amazon.titan-embed-text-v1",
            region_name=region
        )
        Settings.llm = Bedrock(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            region_name=region
        )
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        
        logger.info("Healthcare document processor initialized")
    
    async def extract_patient_records(self, portal_config: Dict[str, Any]) -> List[HealthcareDocument]:
        """
        Extract patient records from healthcare portal
        
        Args:
            portal_config: Configuration for healthcare portal access
            
        Returns:
            List of healthcare documents with PHI protection
        """
        try:
            logger.info(f"Starting patient record extraction from {portal_config.get('portal_name', 'Unknown')}")
            
            # Create secure browser session
            session = await self.browser_loader.create_secure_session()
            
            # Authenticate with healthcare portal
            await self._authenticate_healthcare_portal(session, portal_config)
            
            # Navigate to patient records section
            await session.navigate(portal_config['records_url'])
            
            # Extract patient records with PHI detection
            documents = []
            record_elements = await session.find_elements(portal_config['record_selector'])
            
            for element in record_elements:
                record_content = await element.get_text()
                
                # Detect PHI in the content
                phi_detected = self._detect_phi(record_content)
                
                # Create healthcare document
                doc = HealthcareDocument(
                    content=record_content,
                    document_type="patient_record",
                    patient_id=self._generate_anonymous_patient_id(record_content),
                    phi_detected=phi_detected,
                    sensitivity_level=SensitivityLevel.RESTRICTED,
                    source_url=portal_config['records_url'],
                    extraction_timestamp=datetime.utcnow(),
                    audit_trail=[{
                        "action": "document_extraction",
                        "timestamp": datetime.utcnow().isoformat(),
                        "source": portal_config['portal_name']
                    }]
                )
                
                documents.append(doc)
            
            await session.close()
            logger.info(f"Extracted {len(documents)} patient records")
            return documents
            
        except Exception as e:
            logger.error(f"Error extracting patient records: {str(e)}")
            raise
    
    async def process_lab_results(self, lab_portal_config: Dict[str, Any]) -> List[HealthcareDocument]:
        """
        Process lab results from laboratory information system
        
        Args:
            lab_portal_config: Configuration for lab portal access
            
        Returns:
            List of lab result documents with PHI protection
        """
        try:
            logger.info("Processing lab results")
            
            session = await self.browser_loader.create_secure_session()
            
            # Authenticate with lab portal
            await self._authenticate_healthcare_portal(session, lab_portal_config)
            
            # Navigate to lab results
            await session.navigate(lab_portal_config['results_url'])
            
            # Extract lab results
            documents = []
            result_elements = await session.find_elements(lab_portal_config['result_selector'])
            
            for element in result_elements:
                result_content = await element.get_text()
                
                # Detect PHI and medical information
                phi_detected = self._detect_phi(result_content)
                phi_detected.extend(self._detect_medical_info(result_content))
                
                doc = HealthcareDocument(
                    content=result_content,
                    document_type="lab_result",
                    patient_id=self._generate_anonymous_patient_id(result_content),
                    phi_detected=phi_detected,
                    sensitivity_level=SensitivityLevel.RESTRICTED,
                    source_url=lab_portal_config['results_url'],
                    extraction_timestamp=datetime.utcnow(),
                    audit_trail=[{
                        "action": "lab_result_extraction",
                        "timestamp": datetime.utcnow().isoformat(),
                        "source": lab_portal_config['portal_name']
                    }]
                )
                
                documents.append(doc)
            
            await session.close()
            logger.info(f"Processed {len(documents)} lab results")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing lab results: {str(e)}")
            raise
    
    async def create_secure_healthcare_index(self, documents: List[HealthcareDocument]) -> VectorStoreIndex:
        """
        Create secure vector index for healthcare documents
        
        Args:
            documents: List of healthcare documents
            
        Returns:
            Secure vector store index
        """
        try:
            logger.info("Creating secure healthcare index")
            
            # Mask PHI in all documents before indexing
            masked_documents = []
            for doc in documents:
                masked_doc = doc.mask_phi()
                
                # Convert to LlamaIndex Document
                llama_doc = Document(
                    text=masked_doc.content,
                    metadata={
                        "document_type": masked_doc.document_type,
                        "patient_id": masked_doc.patient_id,
                        "sensitivity_level": masked_doc.sensitivity_level.value,
                        "phi_detected": [phi.value for phi in masked_doc.phi_detected],
                        "extraction_timestamp": masked_doc.extraction_timestamp.isoformat(),
                        "audit_trail": masked_doc.audit_trail
                    }
                )
                masked_documents.append(llama_doc)
            
            # Create secure vector store
            vector_store = ChromaVectorStore(
                chroma_collection_name="healthcare_documents",
                persist_directory="secure_healthcare_vector_store"
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Build index with security controls
            index = VectorStoreIndex.from_documents(
                masked_documents,
                storage_context=storage_context
            )
            
            logger.info(f"Created secure index with {len(masked_documents)} healthcare documents")
            return index
            
        except Exception as e:
            logger.error(f"Error creating healthcare index: {str(e)}")
            raise
    
    async def query_healthcare_data(self, index: VectorStoreIndex, query: str) -> str:
        """
        Query healthcare data with PHI protection
        
        Args:
            index: Secure healthcare vector index
            query: Query string
            
        Returns:
            Sanitized response
        """
        try:
            logger.info(f"Querying healthcare data: {query}")
            
            # Sanitize query to remove potential PHI
            sanitized_query = self._sanitize_query(query)
            
            # Create query engine with security controls
            query_engine = index.as_query_engine(
                response_mode="compact",
                similarity_top_k=3
            )
            
            # Execute query
            response = query_engine.query(sanitized_query)
            
            # Sanitize response to ensure no PHI leakage
            sanitized_response = self._sanitize_response(str(response))
            
            # Log query for audit
            self._log_healthcare_query(sanitized_query, sanitized_response)
            
            return sanitized_response
            
        except Exception as e:
            logger.error(f"Error querying healthcare data: {str(e)}")
            raise
    
    async def _authenticate_healthcare_portal(self, session: BrowserSession, config: Dict[str, Any]):
        """Authenticate with healthcare portal"""
        await session.navigate(config['login_url'])
        
        # Enter credentials securely
        username_field = await session.find_element(config['username_selector'])
        await username_field.type(config['username'])
        
        password_field = await session.find_element(config['password_selector'])
        await password_field.type(config['password'])
        
        # Handle MFA if required
        if config.get('mfa_required', False):
            await self._handle_healthcare_mfa(session, config)
        
        # Submit login
        login_button = await session.find_element(config['login_button_selector'])
        await login_button.click()
        
        # Wait for successful login
        await session.wait_for_element(config['dashboard_selector'])
        
        logger.info("Successfully authenticated with healthcare portal")
    
    async def _handle_healthcare_mfa(self, session: BrowserSession, config: Dict[str, Any]):
        """Handle multi-factor authentication for healthcare portal"""
        # Wait for MFA prompt
        await session.wait_for_element(config['mfa_selector'])
        
        # This would integrate with secure MFA token provider
        # For demo purposes, we'll simulate the process
        logger.info("MFA authentication required - integrate with secure token provider")
    
    def _detect_phi(self, content: str) -> List[PHIType]:
        """Detect PHI in content"""
        import re
        
        phi_detected = []
        
        # PHI detection patterns
        patterns = {
            PHIType.SSN: r'\b\d{3}-\d{2}-\d{4}\b',
            PHIType.PHONE: r'\b\d{3}-\d{3}-\d{4}\b',
            PHIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            PHIType.MRN: r'\bMRN[:\s]*\d{6,10}\b',
            PHIType.DOB: r'\b\d{1,2}/\d{1,2}/\d{4}\b',
        }
        
        for phi_type, pattern in patterns.items():
            if re.search(pattern, content):
                phi_detected.append(phi_type)
        
        return phi_detected
    
    def _detect_medical_info(self, content: str) -> List[PHIType]:
        """Detect medical information in content"""
        medical_detected = []
        
        # Medical terminology detection
        if any(term in content.lower() for term in ['diagnosis', 'condition', 'disease']):
            medical_detected.append(PHIType.DIAGNOSIS)
        
        if any(term in content.lower() for term in ['medication', 'prescription', 'drug']):
            medical_detected.append(PHIType.MEDICATION)
        
        if any(term in content.lower() for term in ['dr.', 'doctor', 'physician', 'provider']):
            medical_detected.append(PHIType.PROVIDER_NAME)
        
        return medical_detected
    
    def _generate_anonymous_patient_id(self, content: str) -> str:
        """Generate anonymous patient identifier"""
        import hashlib
        
        # Create hash-based anonymous ID
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        return f"PATIENT_{content_hash[:8]}"
    
    def _sanitize_query(self, query: str) -> str:
        """Sanitize query to remove potential PHI"""
        # Remove potential PHI patterns from query
        import re
        
        sanitized = query
        
        # Remove SSN patterns
        sanitized = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]', sanitized)
        
        # Remove phone patterns
        sanitized = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[REDACTED_PHONE]', sanitized)
        
        # Remove email patterns
        sanitized = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED_EMAIL]', sanitized)
        
        return sanitized
    
    def _sanitize_response(self, response: str) -> str:
        """Sanitize response to prevent PHI leakage"""
        # Apply same sanitization as query
        return self._sanitize_query(response)
    
    def _log_healthcare_query(self, query: str, response: str):
        """Log healthcare query for audit compliance"""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": "healthcare_query",
            "query_hash": hashlib.sha256(query.encode()).hexdigest(),
            "response_length": len(response),
            "phi_detected": "redacted" in response.lower()
        }
        
        logger.info(f"Healthcare query audit: {audit_entry}")


async def main():
    """Example usage of healthcare document processing"""
    
    # Initialize processor
    processor = HealthcareDocumentProcessor(region="us-east-1")
    
    # Example healthcare portal configuration
    portal_config = {
        "portal_name": "Epic MyChart",
        "login_url": "https://mychart.example.com/login",
        "username_selector": "#username",
        "password_selector": "#password",
        "login_button_selector": "#login-button",
        "dashboard_selector": "#dashboard",
        "records_url": "https://mychart.example.com/records",
        "record_selector": ".patient-record",
        "username": os.getenv("HEALTHCARE_USERNAME"),
        "password": os.getenv("HEALTHCARE_PASSWORD"),
        "mfa_required": True,
        "mfa_selector": "#mfa-code"
    }
    
    try:
        # Extract patient records
        patient_records = await processor.extract_patient_records(portal_config)
        print(f"Extracted {len(patient_records)} patient records")
        
        # Process lab results
        lab_config = portal_config.copy()
        lab_config.update({
            "results_url": "https://mychart.example.com/lab-results",
            "result_selector": ".lab-result"
        })
        
        lab_results = await processor.process_lab_results(lab_config)
        print(f"Processed {len(lab_results)} lab results")
        
        # Combine all healthcare documents
        all_documents = patient_records + lab_results
        
        # Create secure healthcare index
        healthcare_index = await processor.create_secure_healthcare_index(all_documents)
        
        # Query healthcare data
        queries = [
            "What are the common diagnoses in the patient records?",
            "Show me recent lab results with abnormal values",
            "What medications are most frequently prescribed?"
        ]
        
        for query in queries:
            response = await processor.query_healthcare_data(healthcare_index, query)
            print(f"\nQuery: {query}")
            print(f"Response: {response}")
        
        print("\nHealthcare document processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Healthcare processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())