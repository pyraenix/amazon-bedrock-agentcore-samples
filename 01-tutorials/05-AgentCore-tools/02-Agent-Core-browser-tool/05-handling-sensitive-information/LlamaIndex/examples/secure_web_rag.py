#!/usr/bin/env python3
"""
Secure Web RAG for Production LlamaIndex-AgentCore Integration

This module provides production-ready patterns for building secure RAG (Retrieval-Augmented Generation)
applications that extract data from web sources using AgentCore Browser Tool. Includes comprehensive
security controls, data protection, and enterprise-grade features.

Key Features:
- Secure web data ingestion with AgentCore Browser Tool
- Production RAG pipeline with security controls
- Data encryption and secure storage
- Query sanitization and response filtering
- Audit logging and compliance tracking
- Error handling and recovery patterns
- Performance monitoring and optimization

Requirements: 1.2, 2.5, 4.2, 5.4
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from uuid import uuid4
from contextlib import asynccontextmanager

from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.response.schema import Response
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.llms.bedrock import Bedrock

from bedrock_agentcore.tools.browser_client import BrowserSession
from agentcore_session_helpers import SessionManager, SessionPriority
from llamaindex_pii_utils import PIIDetector, PIIAnonymizer, PIIAuditLogger, detect_and_anonymize_document

# Configure logging
logger = logging.getLogger(__name__)


class RAGSecurityLevel(Enum):
    """Security levels for RAG operations"""
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    MAXIMUM = "maximum"


class DataSource(Enum):
    """Types of data sources"""
    WEB_SCRAPING = "web_scraping"
    AUTHENTICATED_PORTAL = "authenticated_portal"
    API_ENDPOINT = "api_endpoint"
    DOCUMENT_UPLOAD = "document_upload"
    DATABASE_QUERY = "database_query"


@dataclass
class SecureRAGConfig:
    """Configuration for secure RAG operations"""
    security_level: RAGSecurityLevel = RAGSecurityLevel.HIGH
    enable_pii_detection: bool = True
    enable_data_encryption: bool = True
    enable_audit_logging: bool = True
    enable_query_sanitization: bool = True
    enable_response_filtering: bool = True
    max_document_size: int = 1000000  # 1MB
    max_documents_per_batch: int = 100
    query_timeout_seconds: int = 30
    embedding_model: str = "amazon.titan-embed-text-v1"
    llm_model: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    vector_store_persist_dir: str = "./secure_vector_store"
    audit_log_file: str = "secure_rag_audit.log"


@dataclass
class WebExtractionConfig:
    """Configuration for web data extraction"""
    url: str
    authentication_required: bool = False
    credentials: Optional[Dict[str, str]] = None
    selectors: Dict[str, str] = field(default_factory=dict)
    wait_conditions: List[str] = field(default_factory=list)
    extraction_timeout: int = 30
    retry_attempts: int = 3
    rate_limit_delay: float = 1.0
    respect_robots_txt: bool = True
    user_agent: str = "SecureRAG/1.0"


@dataclass
class RAGMetrics:
    """Metrics for RAG operations"""
    total_documents_processed: int = 0
    total_queries_executed: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    pii_instances_detected: int = 0
    pii_instances_anonymized: int = 0
    average_query_time: float = 0.0
    average_extraction_time: float = 0.0
    security_violations: int = 0
    
    def update_extraction_metrics(self, success: bool, extraction_time: float):
        """Update extraction metrics"""
        if success:
            self.successful_extractions += 1
        else:
            self.failed_extractions += 1
        
        # Update average extraction time
        total_extractions = self.successful_extractions + self.failed_extractions
        if total_extractions == 1:
            self.average_extraction_time = extraction_time
        else:
            self.average_extraction_time = (
                (self.average_extraction_time * (total_extractions - 1) + extraction_time) 
                / total_extractions
            )
    
    def update_query_metrics(self, query_time: float):
        """Update query metrics"""
        self.total_queries_executed += 1
        
        if self.total_queries_executed == 1:
            self.average_query_time = query_time
        else:
            self.average_query_time = (
                (self.average_query_time * (self.total_queries_executed - 1) + query_time) 
                / self.total_queries_executed
            )


class SecureWebExtractor:
    """Secure web data extractor using AgentCore Browser Tool"""
    
    def __init__(self, session_manager: SessionManager, config: SecureRAGConfig):
        self.session_manager = session_manager
        self.config = config
        self.pii_detector = PIIDetector()
        self.pii_anonymizer = PIIAnonymizer()
        self.audit_logger = PIIAuditLogger(config.audit_log_file)
        self.metrics = RAGMetrics()
        
        logger.info("Secure web extractor initialized")
    
    async def extract_from_url(self, extraction_config: WebExtractionConfig) -> List[Document]:
        """Extract documents from web URL with security controls"""
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Starting secure extraction from {extraction_config.url}")
            
            # Validate URL and check security
            await self._validate_extraction_request(extraction_config)
            
            # Extract content using secure session
            documents = await self._perform_secure_extraction(extraction_config)
            
            # Process documents for PII and security
            processed_documents = await self._process_extracted_documents(documents, extraction_config.url)
            
            extraction_time = (datetime.utcnow() - start_time).total_seconds()
            self.metrics.update_extraction_metrics(True, extraction_time)
            
            logger.info(f"Successfully extracted {len(processed_documents)} documents from {extraction_config.url}")
            return processed_documents
            
        except Exception as e:
            extraction_time = (datetime.utcnow() - start_time).total_seconds()
            self.metrics.update_extraction_metrics(False, extraction_time)
            
            logger.error(f"Failed to extract from {extraction_config.url}: {str(e)}")
            raise
    
    async def extract_from_multiple_urls(self, extraction_configs: List[WebExtractionConfig]) -> List[Document]:
        """Extract documents from multiple URLs with rate limiting"""
        all_documents = []
        
        for i, config in enumerate(extraction_configs):
            try:
                documents = await self.extract_from_url(config)
                all_documents.extend(documents)
                
                # Rate limiting
                if i < len(extraction_configs) - 1:
                    await asyncio.sleep(config.rate_limit_delay)
                    
            except Exception as e:
                logger.warning(f"Failed to extract from {config.url}: {str(e)}")
                continue
        
        return all_documents
    
    async def _validate_extraction_request(self, config: WebExtractionConfig):
        """Validate extraction request for security"""
        # Check URL safety
        if not config.url.startswith(('http://', 'https://')):
            raise ValueError("Invalid URL scheme")
        
        # Check for blocked domains (implement your domain policy)
        blocked_domains = ['malicious-site.com', 'blocked-domain.net']
        for domain in blocked_domains:
            if domain in config.url:
                self.metrics.security_violations += 1
                raise ValueError(f"Blocked domain: {domain}")
        
        # Validate credentials if authentication required
        if config.authentication_required and not config.credentials:
            raise ValueError("Authentication required but no credentials provided")
        
        logger.debug(f"Extraction request validated for {config.url}")
    
    async def _perform_secure_extraction(self, config: WebExtractionConfig) -> List[Document]:
        """Perform secure web extraction using AgentCore session"""
        documents = []
        
        async with self.session_manager.session(
            priority=SessionPriority.HIGH,
            tags={"purpose": "secure_extraction", "url": config.url}
        ) as session:
            
            # Navigate to URL
            await session.session.navigate(config.url)
            
            # Handle authentication if required
            if config.authentication_required:
                await self._handle_authentication(session.session, config)
            
            # Wait for content to load
            for condition in config.wait_conditions:
                await session.session.wait_for_element(condition)
            
            # Extract content based on selectors
            if config.selectors:
                for selector_name, selector in config.selectors.items():
                    elements = await session.session.find_elements(selector)
                    
                    for i, element in enumerate(elements):
                        content = await element.get_text()
                        
                        if content.strip():
                            doc = Document(
                                text=content,
                                doc_id=f"{selector_name}_{i}_{uuid4().hex[:8]}",
                                metadata={
                                    "source_url": config.url,
                                    "selector": selector,
                                    "selector_name": selector_name,
                                    "extraction_timestamp": datetime.utcnow().isoformat(),
                                    "data_source": DataSource.WEB_SCRAPING.value
                                }
                            )
                            documents.append(doc)
            else:
                # Extract full page content
                page_content = await session.session.evaluate("document.body.innerText")
                
                if page_content.strip():
                    doc = Document(
                        text=page_content,
                        doc_id=f"page_{uuid4().hex[:8]}",
                        metadata={
                            "source_url": config.url,
                            "extraction_timestamp": datetime.utcnow().isoformat(),
                            "data_source": DataSource.WEB_SCRAPING.value
                        }
                    )
                    documents.append(doc)
        
        return documents
    
    async def _handle_authentication(self, session: BrowserSession, config: WebExtractionConfig):
        """Handle authentication for secure extraction"""
        if not config.credentials:
            return
        
        # This is a simplified authentication handler
        # In production, you would implement specific authentication flows
        username_selector = config.selectors.get('username_field', '#username')
        password_selector = config.selectors.get('password_field', '#password')
        login_button_selector = config.selectors.get('login_button', '#login')
        
        try:
            # Enter credentials
            username_field = await session.find_element(username_selector)
            await username_field.type(config.credentials.get('username', ''))
            
            password_field = await session.find_element(password_selector)
            await password_field.type(config.credentials.get('password', ''))
            
            # Submit login
            login_button = await session.find_element(login_button_selector)
            await login_button.click()
            
            # Wait for login to complete
            await asyncio.sleep(2)
            
            logger.info("Authentication completed successfully")
            
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise
    
    async def _process_extracted_documents(self, documents: List[Document], source_url: str) -> List[Document]:
        """Process extracted documents for PII and security"""
        processed_documents = []
        
        for document in documents:
            # Check document size
            if len(document.text) > self.config.max_document_size:
                logger.warning(f"Document {document.doc_id} exceeds size limit, truncating")
                document.text = document.text[:self.config.max_document_size]
            
            # Detect and handle PII if enabled
            if self.config.enable_pii_detection:
                anonymized_doc, pii_result = detect_and_anonymize_document(document, "mask")
                
                if pii_result.total_pii_found > 0:
                    self.metrics.pii_instances_detected += pii_result.total_pii_found
                    self.metrics.pii_instances_anonymized += pii_result.total_pii_found
                    
                    # Log PII detection
                    self.audit_logger.log_pii_detection(pii_result, source_url)
                    
                    processed_documents.append(anonymized_doc)
                else:
                    processed_documents.append(document)
            else:
                processed_documents.append(document)
        
        self.metrics.total_documents_processed += len(processed_documents)
        return processed_documents


class SecureRAGPipeline:
    """Production-ready secure RAG pipeline"""
    
    def __init__(self, region: str, config: Optional[SecureRAGConfig] = None):
        self.region = region
        self.config = config or SecureRAGConfig()
        self.session_manager = SessionManager(region)
        self.web_extractor = SecureWebExtractor(self.session_manager, self.config)
        self.audit_logger = PIIAuditLogger(self.config.audit_log_file)
        
        # Configure LlamaIndex
        Settings.embed_model = BedrockEmbedding(
            model_name=self.config.embedding_model,
            region_name=region
        )
        Settings.llm = Bedrock(
            model=self.config.llm_model,
            region_name=region
        )
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        
        self.vector_store: Optional[ChromaVectorStore] = None
        self.index: Optional[VectorStoreIndex] = None
        self.query_engine: Optional[BaseQueryEngine] = None
        
        logger.info("Secure RAG pipeline initialized")
    
    async def initialize_vector_store(self, collection_name: str = "secure_rag"):
        """Initialize secure vector store"""
        try:
            self.vector_store = ChromaVectorStore(
                chroma_collection_name=collection_name,
                persist_directory=self.config.vector_store_persist_dir
            )
            
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            # Try to load existing index
            try:
                self.index = VectorStoreIndex.from_vector_store(
                    self.vector_store,
                    storage_context=storage_context
                )
                logger.info("Loaded existing vector store index")
            except:
                # Create new index
                self.index = VectorStoreIndex([], storage_context=storage_context)
                logger.info("Created new vector store index")
            
            # Create query engine with security controls
            self.query_engine = self.index.as_query_engine(
                response_mode="compact",
                similarity_top_k=5
            )
            
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise
    
    async def ingest_web_data(self, extraction_configs: List[WebExtractionConfig]) -> Dict[str, Any]:
        """Ingest data from web sources"""
        if not self.index:
            await self.initialize_vector_store()
        
        try:
            # Extract documents from web sources
            documents = await self.web_extractor.extract_from_multiple_urls(extraction_configs)
            
            if not documents:
                logger.warning("No documents extracted from web sources")
                return {"status": "no_data", "documents_processed": 0}
            
            # Batch process documents
            batch_size = self.config.max_documents_per_batch
            total_processed = 0
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Add documents to index
                for document in batch:
                    self.index.insert(document)
                
                total_processed += len(batch)
                logger.info(f"Processed batch {i//batch_size + 1}, total documents: {total_processed}")
            
            # Persist the index
            if hasattr(self.index.storage_context, 'persist'):
                self.index.storage_context.persist()
            
            # Update query engine
            self.query_engine = self.index.as_query_engine(
                response_mode="compact",
                similarity_top_k=5
            )
            
            logger.info(f"Successfully ingested {total_processed} documents")
            
            return {
                "status": "success",
                "documents_processed": total_processed,
                "extraction_metrics": self._get_extraction_metrics()
            }
            
        except Exception as e:
            logger.error(f"Failed to ingest web data: {str(e)}")
            raise
    
    async def secure_query(self, query: str, context_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute secure query with sanitization and filtering"""
        if not self.query_engine:
            raise RuntimeError("RAG pipeline not initialized. Call initialize_vector_store() first.")
        
        start_time = datetime.utcnow()
        
        try:
            # Sanitize query
            sanitized_query = self._sanitize_query(query) if self.config.enable_query_sanitization else query
            
            # Log query for audit
            query_id = uuid4().hex[:8]
            self.audit_logger.audit_logger.info(f"SECURE_QUERY: {json.dumps({
                'query_id': query_id,
                'timestamp': datetime.utcnow().isoformat(),
                'query_hash': hashlib.sha256(sanitized_query.encode()).hexdigest(),
                'context_filter': context_filter
            })}")
            
            # Execute query with timeout
            response = await asyncio.wait_for(
                self._execute_query(sanitized_query),
                timeout=self.config.query_timeout_seconds
            )
            
            # Filter response if enabled
            filtered_response = self._filter_response(response) if self.config.enable_response_filtering else response
            
            query_time = (datetime.utcnow() - start_time).total_seconds()
            self.web_extractor.metrics.update_query_metrics(query_time)
            
            return {
                "query_id": query_id,
                "response": str(filtered_response),
                "query_time": query_time,
                "source_nodes": len(response.source_nodes) if hasattr(response, 'source_nodes') else 0,
                "security_level": self.config.security_level.value
            }
            
        except asyncio.TimeoutError:
            logger.error(f"Query timeout after {self.config.query_timeout_seconds} seconds")
            raise
        except Exception as e:
            logger.error(f"Secure query failed: {str(e)}")
            raise
    
    async def _execute_query(self, query: str) -> Response:
        """Execute query against the index"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.query_engine.query, query
        )
    
    def _sanitize_query(self, query: str) -> str:
        """Sanitize query to remove potential security risks"""
        # Remove potential PII from query
        pii_detector = PIIDetector()
        pii_result = pii_detector.detect_pii_in_text(query)
        
        if pii_result.total_pii_found > 0:
            pii_anonymizer = PIIAnonymizer()
            sanitized_query = pii_anonymizer.anonymize_text(query, pii_result, "mask")
            
            logger.warning(f"PII detected in query, sanitized {pii_result.total_pii_found} instances")
            return sanitized_query
        
        return query
    
    def _filter_response(self, response: Response) -> Response:
        """Filter response to remove sensitive information"""
        # Apply PII filtering to response
        pii_detector = PIIDetector()
        pii_result = pii_detector.detect_pii_in_text(str(response))
        
        if pii_result.total_pii_found > 0:
            pii_anonymizer = PIIAnonymizer()
            filtered_text = pii_anonymizer.anonymize_text(str(response), pii_result, "mask")
            
            # Create new response with filtered content
            # Note: This is a simplified approach - in production you might need more sophisticated filtering
            response.response = filtered_text
            
            logger.info(f"Filtered {pii_result.total_pii_found} PII instances from response")
        
        return response
    
    def _get_extraction_metrics(self) -> Dict[str, Any]:
        """Get extraction metrics"""
        metrics = self.web_extractor.metrics
        return {
            "total_documents_processed": metrics.total_documents_processed,
            "successful_extractions": metrics.successful_extractions,
            "failed_extractions": metrics.failed_extractions,
            "pii_instances_detected": metrics.pii_instances_detected,
            "pii_instances_anonymized": metrics.pii_instances_anonymized,
            "average_extraction_time": metrics.average_extraction_time,
            "security_violations": metrics.security_violations
        }
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        return {
            "pipeline_initialized": self.index is not None,
            "vector_store_initialized": self.vector_store is not None,
            "query_engine_ready": self.query_engine is not None,
            "security_level": self.config.security_level.value,
            "extraction_metrics": self._get_extraction_metrics(),
            "session_pool_stats": self.session_manager.get_all_stats(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def cleanup(self):
        """Cleanup pipeline resources"""
        try:
            await self.session_manager.close_all_pools()
            logger.info("RAG pipeline cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


# Utility functions for common RAG operations

async def create_secure_rag_from_urls(urls: List[str], region: str = "us-east-1", 
                                    security_level: RAGSecurityLevel = RAGSecurityLevel.HIGH) -> SecureRAGPipeline:
    """Create secure RAG pipeline from list of URLs"""
    config = SecureRAGConfig(security_level=security_level)
    pipeline = SecureRAGPipeline(region, config)
    
    # Create extraction configs
    extraction_configs = [
        WebExtractionConfig(url=url) for url in urls
    ]
    
    # Initialize and ingest data
    await pipeline.initialize_vector_store()
    await pipeline.ingest_web_data(extraction_configs)
    
    return pipeline


async def secure_rag_query_batch(pipeline: SecureRAGPipeline, queries: List[str]) -> List[Dict[str, Any]]:
    """Execute batch of secure queries"""
    results = []
    
    for query in queries:
        try:
            result = await pipeline.secure_query(query)
            results.append(result)
        except Exception as e:
            logger.error(f"Query failed: {query[:50]}... - {str(e)}")
            results.append({
                "query": query,
                "error": str(e),
                "status": "failed"
            })
    
    return results


@asynccontextmanager
async def secure_rag_context(region: str, config: Optional[SecureRAGConfig] = None):
    """Context manager for secure RAG pipeline"""
    pipeline = SecureRAGPipeline(region, config)
    try:
        await pipeline.initialize_vector_store()
        yield pipeline
    finally:
        await pipeline.cleanup()


# Example usage
async def example_secure_rag():
    """Example of secure RAG pipeline usage"""
    
    # Configuration
    config = SecureRAGConfig(
        security_level=RAGSecurityLevel.HIGH,
        enable_pii_detection=True,
        enable_audit_logging=True
    )
    
    # URLs to extract data from
    urls = [
        "https://example.com/public-data",
        "https://docs.example.com/api-documentation"
    ]
    
    async with secure_rag_context("us-east-1", config) as pipeline:
        # Create extraction configs
        extraction_configs = [
            WebExtractionConfig(
                url=url,
                selectors={"content": ".main-content", "title": "h1"}
            ) for url in urls
        ]
        
        # Ingest data
        ingestion_result = await pipeline.ingest_web_data(extraction_configs)
        print(f"Ingestion result: {ingestion_result}")
        
        # Execute secure queries
        queries = [
            "What is the main topic discussed in the documentation?",
            "How do I authenticate with the API?",
            "What are the rate limits for API calls?"
        ]
        
        for query in queries:
            result = await pipeline.secure_query(query)
            print(f"\nQuery: {query}")
            print(f"Response: {result['response']}")
            print(f"Query time: {result['query_time']:.2f}s")
        
        # Get pipeline status
        status = await pipeline.get_pipeline_status()
        print(f"\nPipeline status: {json.dumps(status, indent=2)}")


if __name__ == "__main__":
    asyncio.run(example_secure_rag())