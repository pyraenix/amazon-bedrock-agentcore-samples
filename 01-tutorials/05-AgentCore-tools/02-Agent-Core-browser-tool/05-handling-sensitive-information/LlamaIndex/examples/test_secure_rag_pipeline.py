"""
Unit Tests for Secure RAG Pipeline

Comprehensive test suite for the SecureRAGPipeline class and related components.
Tests cover secure data ingestion, encrypted vector storage, query sanitization,
response filtering, and security controls.

Requirements Addressed:
- 3.1: Unit tests for secure RAG operations with web data
- 3.2: Testing secure vector storage with encryption
- 3.3: Testing query engines that handle sensitive context without data leakage
"""

import unittest
import tempfile
import shutil
import os
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime

# Import components to test
from secure_rag_pipeline import (
    SecureRAGPipeline, SecureRAGConfig, SecureVectorStore, SecureQueryEngine,
    QueryMetrics, create_secure_rag_pipeline, create_high_security_rag_pipeline
)
from sensitive_data_handler import (
    SanitizationConfig, create_secure_sanitization_config,
    SensitivityLevel, DataType
)
from agentcore_browser_loader import BrowserSessionConfig

# LlamaIndex imports for testing
from llama_index.core.schema import Document, TextNode, NodeWithScore
from llama_index.core.response import Response
try:
    from llama_index.core.response.schema import Response
except ImportError:
    # Fallback for different LlamaIndex versions
    try:
        from llama_index.core.base.response.schema import Response
    except ImportError:
        # Create a simple Response class for testing
        class Response:
            def __init__(self, response="", source_nodes=None):
                self.response = response
                self.source_nodes = source_nodes or []


class TestSecureRAGConfig(unittest.TestCase):
    """Test SecureRAGConfig configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SecureRAGConfig()
        
        self.assertEqual(config.storage_dir, "./secure_vector_store")
        self.assertTrue(config.enable_encryption)
        self.assertIsNotNone(config.encryption_key)
        self.assertEqual(config.embedding_model, "amazon.titan-embed-text-v1")
        self.assertEqual(config.similarity_top_k, 5)
        self.assertTrue(config.enable_query_sanitization)
        self.assertTrue(config.enable_response_filtering)
        self.assertTrue(config.audit_all_operations)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = SecureRAGConfig(
            storage_dir="./custom_storage",
            enable_encryption=False,
            embedding_model="custom-model",
            similarity_top_k=10
        )
        
        self.assertEqual(config.storage_dir, "./custom_storage")
        self.assertFalse(config.enable_encryption)
        self.assertEqual(config.embedding_model, "custom-model")
        self.assertEqual(config.similarity_top_k, 10)
    
    def test_encryption_key_generation(self):
        """Test automatic encryption key generation."""
        config1 = SecureRAGConfig(enable_encryption=True, encryption_key=None)
        config2 = SecureRAGConfig(enable_encryption=True, encryption_key=None)
        
        # Each config should have a unique encryption key
        self.assertIsNotNone(config1.encryption_key)
        self.assertIsNotNone(config2.encryption_key)
        self.assertNotEqual(config1.encryption_key, config2.encryption_key)


class TestQueryMetrics(unittest.TestCase):
    """Test QueryMetrics tracking class."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = QueryMetrics(query_id="test-query-123")
        
        self.assertEqual(metrics.query_id, "test-query-123")
        self.assertIsInstance(metrics.timestamp, datetime)
        self.assertEqual(metrics.documents_retrieved, 0)
        self.assertEqual(metrics.response_length, 0)
        self.assertFalse(metrics.response_sanitized)
        self.assertEqual(len(metrics.security_violations), 0)
        self.assertEqual(len(metrics.audit_events), 0)
    
    def test_add_audit_event(self):
        """Test adding audit events."""
        metrics = QueryMetrics(query_id="test-query")
        
        metrics.add_audit_event("test_event", {"key": "value"})
        
        self.assertEqual(len(metrics.audit_events), 1)
        event = metrics.audit_events[0]
        self.assertEqual(event['event_type'], "test_event")
        self.assertEqual(event['details'], {"key": "value"})
        self.assertIn('timestamp', event)
    
    def test_add_security_violation(self):
        """Test adding security violations."""
        metrics = QueryMetrics(query_id="test-query")
        
        with patch('secure_rag_pipeline.logger') as mock_logger:
            metrics.add_security_violation("test violation")
            
            self.assertEqual(len(metrics.security_violations), 1)
            self.assertEqual(metrics.security_violations[0], "test violation")
            mock_logger.warning.assert_called_once()
    
    def test_to_dict(self):
        """Test metrics serialization to dictionary."""
        metrics = QueryMetrics(query_id="test-query")
        metrics.query_length = 100
        metrics.documents_retrieved = 5
        metrics.response_length = 200
        metrics.add_security_violation("test violation")
        
        result = metrics.to_dict()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['query_id'], "test-query")
        self.assertEqual(result['query_characteristics']['query_length'], 100)
        self.assertEqual(result['retrieval_metrics']['documents_retrieved'], 5)
        self.assertEqual(result['response_metrics']['response_length'], 200)
        self.assertEqual(result['security_metrics']['security_violations'], 1)


class TestSecureVectorStore(unittest.TestCase):
    """Test SecureVectorStore with encryption capabilities."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = SecureRAGConfig(
            storage_dir=self.temp_dir,
            enable_encryption=True
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('secure_rag_pipeline.ENCRYPTION_AVAILABLE', True)
    def test_initialization_with_encryption(self):
        """Test vector store initialization with encryption."""
        store = SecureVectorStore(self.config)
        
        self.assertEqual(str(store.storage_dir), self.temp_dir)
        self.assertIsNotNone(store.cipher_suite)
        self.assertIsNotNone(store.storage_context)
    
    @patch('secure_rag_pipeline.ENCRYPTION_AVAILABLE', False)
    def test_initialization_without_encryption_library(self):
        """Test initialization when encryption library is not available."""
        with patch('secure_rag_pipeline.logger') as mock_logger:
            store = SecureVectorStore(self.config)
            
            self.assertIsNone(store.cipher_suite)
            mock_logger.warning.assert_called()
    
    @patch('secure_rag_pipeline.ENCRYPTION_AVAILABLE', True)
    def test_encrypt_decrypt_data(self):
        """Test data encryption and decryption."""
        store = SecureVectorStore(self.config)
        
        original_data = "This is sensitive test data"
        encrypted_data = store.encrypt_data(original_data)
        decrypted_data = store.decrypt_data(encrypted_data)
        
        self.assertNotEqual(original_data, encrypted_data)
        self.assertEqual(original_data, decrypted_data)
    
    def test_encrypt_decrypt_without_cipher(self):
        """Test encryption/decryption when cipher is not available."""
        config = SecureRAGConfig(enable_encryption=False)
        store = SecureVectorStore(config)
        
        original_data = "Test data"
        encrypted_data = store.encrypt_data(original_data)
        decrypted_data = store.decrypt_data(encrypted_data)
        
        # Should return original data unchanged
        self.assertEqual(original_data, encrypted_data)
        self.assertEqual(original_data, decrypted_data)


class TestSecureQueryEngine(unittest.TestCase):
    """Test SecureQueryEngine with security controls."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = SecureRAGConfig()
        
        # Mock LlamaIndex components
        self.mock_index = Mock()
        self.mock_base_query_engine = Mock()
        self.mock_index.as_query_engine.return_value = self.mock_base_query_engine
        
        # Create test documents
        self.test_documents = [
            Document(
                text="This is public information about our company.",
                metadata={'source': 'https://example.com/public'}
            ),
            Document(
                text="John Doe's email is john.doe@example.com and SSN is 123-45-6789.",
                metadata={'source': 'https://example.com/sensitive'}
            )
        ]
    
    def test_initialization(self):
        """Test query engine initialization."""
        engine = SecureQueryEngine(self.mock_index, self.config)
        
        self.assertEqual(engine.index, self.mock_index)
        self.assertEqual(engine.config, self.config)
        self.assertIsNotNone(engine.sanitizer)
        self.assertIsNotNone(engine.classifier)
        self.mock_index.as_query_engine.assert_called_once()
    
    @patch('secure_rag_pipeline.uuid.uuid4')
    def test_query_execution(self, mock_uuid):
        """Test secure query execution."""
        mock_uuid.return_value.hex = "abcd1234" * 4
        
        # Mock response
        mock_response = Mock(spec=Response)
        mock_response.response = "This is a test response"
        mock_response.source_nodes = []
        self.mock_base_query_engine.query.return_value = mock_response
        
        engine = SecureQueryEngine(self.mock_index, self.config)
        
        with patch.object(engine, '_sanitize_query', return_value="sanitized query") as mock_sanitize_query, \
             patch.object(engine, '_execute_secure_retrieval', return_value=mock_response) as mock_retrieval, \
             patch.object(engine, '_sanitize_response', return_value=mock_response) as mock_sanitize_response, \
             patch.object(engine, '_audit_query_operation') as mock_audit:
            
            result = engine.query("test query")
            
            self.assertEqual(result, mock_response)
            mock_sanitize_query.assert_called_once()
            mock_retrieval.assert_called_once()
            mock_sanitize_response.assert_called_once()
            mock_audit.assert_called_once()
    
    def test_query_sanitization_enabled(self):
        """Test query sanitization when enabled."""
        self.config.enable_query_sanitization = True
        engine = SecureQueryEngine(self.mock_index, self.config)
        
        # Mock classifier to detect sensitive data
        with patch.object(engine.classifier, 'classify_document') as mock_classify:
            mock_classify.return_value = {
                'sensitive_data_count': 1,
                'data_types': ['email'],
                'sensitivity_level': 'confidential'
            }
            
            # Mock sanitizer
            with patch.object(engine.sanitizer, 'sanitize_document') as mock_sanitize:
                mock_sanitized_doc = Mock()
                mock_sanitized_doc.text = "sanitized query text"
                mock_sanitize.return_value = mock_sanitized_doc
                
                metrics = QueryMetrics(query_id="test")
                result = engine._sanitize_query("query with john.doe@example.com", metrics)
                
                self.assertEqual(result, "sanitized query text")
                self.assertTrue(metrics.contains_sensitive_data)
                mock_classify.assert_called_once()
                mock_sanitize.assert_called_once()
    
    def test_query_sanitization_disabled(self):
        """Test query sanitization when disabled."""
        self.config.enable_query_sanitization = False
        engine = SecureQueryEngine(self.mock_index, self.config)
        
        metrics = QueryMetrics(query_id="test")
        result = engine._sanitize_query("original query", metrics)
        
        self.assertEqual(result, "original query")
        self.assertFalse(metrics.contains_sensitive_data)
    
    def test_context_filtering(self):
        """Test sensitive context filtering."""
        self.config.enable_context_filtering = True
        engine = SecureQueryEngine(self.mock_index, self.config)
        
        # Create real LlamaIndex nodes
        public_node = TextNode(
            text="Public information",
            id_="node1"
        )
        
        sensitive_node = TextNode(
            text="Sensitive SSN: 123-45-6789",
            id_="node2"
        )
        
        # Create real NodeWithScore objects
        public_node_with_score = NodeWithScore(node=public_node, score=0.8)
        sensitive_node_with_score = NodeWithScore(node=sensitive_node, score=0.9)
        
        # Create real Response object
        response = Response(
            response="Test response",
            source_nodes=[public_node_with_score, sensitive_node_with_score]
        )
        
        # Mock classifier
        with patch.object(engine.classifier, 'classify_document') as mock_classify:
            def classify_side_effect(doc):
                if "SSN" in doc.text:
                    return {
                        'requires_special_handling': True,
                        'sensitivity_level': 'restricted',
                        'data_types': ['government_id']
                    }
                else:
                    return {
                        'requires_special_handling': False,
                        'sensitivity_level': 'public',
                        'data_types': []
                    }
            
            mock_classify.side_effect = classify_side_effect
            
            metrics = QueryMetrics(query_id="test")
            result = engine._filter_sensitive_context(response, metrics)
            
            # Should filter out the sensitive node
            self.assertEqual(len(result.source_nodes), 1)
            self.assertEqual(result.source_nodes[0].node.text, "Public information")
            self.assertEqual(metrics.sensitive_documents_filtered, 1)
    
    def test_response_sanitization(self):
        """Test response sanitization."""
        self.config.enable_response_filtering = True
        engine = SecureQueryEngine(self.mock_index, self.config)
        
        # Create real Response object
        response = Response(
            response="Response with john.doe@example.com",
            source_nodes=[]
        )
        
        # Mock sanitizer
        with patch.object(engine.sanitizer, 'sanitize_document') as mock_sanitize:
            mock_sanitized_doc = Mock()
            mock_sanitized_doc.text = "Response with j***@example.com"
            mock_sanitize.return_value = mock_sanitized_doc
            
            metrics = QueryMetrics(query_id="test")
            result = engine._sanitize_response(response, metrics)
            
            self.assertEqual(result.response, "Response with j***@example.com")
            self.assertTrue(metrics.response_sanitized)
            mock_sanitize.assert_called_once()
    
    def test_response_truncation(self):
        """Test response truncation for long responses."""
        self.config.max_response_length = 50
        engine = SecureQueryEngine(self.mock_index, self.config)
        
        long_response = "This is a very long response that exceeds the maximum length limit"
        
        # Create real Response object
        response = Response(
            response=long_response,
            source_nodes=[]
        )
        
        with patch.object(engine.sanitizer, 'sanitize_document') as mock_sanitize:
            mock_sanitized_doc = Mock()
            mock_sanitized_doc.text = long_response
            mock_sanitize.return_value = mock_sanitized_doc
            
            metrics = QueryMetrics(query_id="test")
            result = engine._sanitize_response(response, metrics)
            
            self.assertTrue(len(result.response) <= 53)  # 50 + "..."
            self.assertTrue(result.response.endswith("..."))


class TestSecureRAGPipeline(unittest.TestCase):
    """Test SecureRAGPipeline integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = SecureRAGConfig(
            storage_dir=self.temp_dir,
            enable_encryption=False  # Disable for testing
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('secure_rag_pipeline.BedrockEmbedding')
    def test_initialization(self, mock_bedrock_embedding):
        """Test pipeline initialization."""
        pipeline = SecureRAGPipeline(config=self.config)
        
        self.assertEqual(pipeline.config, self.config)
        self.assertIsNotNone(pipeline.secure_vector_store)
        self.assertIsNotNone(pipeline.sanitizer)
        self.assertIsNotNone(pipeline.classifier)
        self.assertIsNotNone(pipeline.pipeline_id)
        self.assertIsNone(pipeline.index)  # Not created until documents are added
        self.assertIsNone(pipeline.query_engine)
    
    @patch('secure_rag_pipeline.AgentCoreBrowserLoader')
    @patch('secure_rag_pipeline.VectorStoreIndex')
    def test_ingest_web_data(self, mock_vector_index, mock_loader_class):
        """Test web data ingestion."""
        # Mock loader
        mock_loader = Mock()
        mock_documents = [
            Document(text="Test document 1", metadata={'source': 'https://example.com/1'}),
            Document(text="Test document 2", metadata={'source': 'https://example.com/2'})
        ]
        mock_loader.load_data.return_value = mock_documents
        mock_loader.get_session_metrics.return_value = {'documents_loaded': 2}
        mock_loader_class.return_value = mock_loader
        
        # Mock index
        mock_index = Mock()
        mock_vector_index.from_documents.return_value = mock_index
        
        pipeline = SecureRAGPipeline(config=self.config)
        
        with patch.object(pipeline.classifier, 'classify_document') as mock_classify:
            mock_classify.return_value = {
                'requires_special_handling': False,
                'sensitivity_level': 'public',
                'data_types': []
            }
            
            results = pipeline.ingest_web_data(['https://example.com/1', 'https://example.com/2'])
            
            self.assertEqual(results['documents_loaded'], 2)
            self.assertEqual(results['documents_indexed'], 2)
            self.assertIsNotNone(pipeline.index)
            self.assertIsNotNone(pipeline.query_engine)
            mock_loader.load_data.assert_called_once()
            mock_loader.cleanup_session.assert_called_once()
    
    def test_query_without_ingestion(self):
        """Test querying without ingesting documents first."""
        pipeline = SecureRAGPipeline(config=self.config)
        
        with self.assertRaises(ValueError) as context:
            pipeline.query("test query")
        
        self.assertIn("No documents have been ingested", str(context.exception))
    
    @patch('secure_rag_pipeline.VectorStoreIndex')
    def test_query_with_ingestion(self, mock_vector_index):
        """Test querying after document ingestion."""
        # Setup pipeline with mock index and query engine
        mock_index = Mock()
        mock_vector_index.from_documents.return_value = mock_index
        
        pipeline = SecureRAGPipeline(config=self.config)
        
        # Simulate document ingestion
        test_documents = [Document(text="Test document", metadata={'source': 'test'})]
        with patch.object(pipeline.classifier, 'classify_document') as mock_classify:
            mock_classify.return_value = {
                'requires_special_handling': False,
                'sensitivity_level': 'public',
                'data_types': []
            }
            pipeline._process_and_index_documents(test_documents)
        
        # Mock query engine response
        mock_response = Mock(spec=Response)
        mock_response.response = "Test response"
        pipeline.query_engine.query = Mock(return_value=mock_response)
        
        result = pipeline.query("test query")
        
        self.assertEqual(result, mock_response)
        pipeline.query_engine.query.assert_called_once_with("test query")
    
    def test_get_pipeline_status(self):
        """Test pipeline status reporting."""
        pipeline = SecureRAGPipeline(config=self.config)
        
        status = pipeline.get_pipeline_status()
        
        self.assertIsInstance(status, dict)
        self.assertEqual(status['pipeline_id'], pipeline.pipeline_id)
        self.assertIn('configuration', status)
        self.assertIn('index_status', status)
        self.assertIn('security_features', status)
        self.assertFalse(status['index_status']['initialized'])
        self.assertFalse(status['index_status']['query_engine_ready'])
    
    def test_cleanup(self):
        """Test pipeline cleanup."""
        pipeline = SecureRAGPipeline(config=self.config)
        
        with patch.object(pipeline.secure_vector_store, 'persist') as mock_persist:
            pipeline.cleanup()
            mock_persist.assert_called_once()


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions for creating RAG pipelines."""
    
    @patch('secure_rag_pipeline.SecureRAGPipeline')
    def test_create_secure_rag_pipeline(self, mock_pipeline_class):
        """Test secure RAG pipeline creation utility."""
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        result = create_secure_rag_pipeline(
            storage_dir="./test_storage",
            enable_encryption=True,
            region="us-west-2"
        )
        
        self.assertEqual(result, mock_pipeline)
        mock_pipeline_class.assert_called_once()
        
        # Check that the config was created with correct parameters
        call_args = mock_pipeline_class.call_args
        config = call_args[1]['config']
        self.assertEqual(config.storage_dir, "./test_storage")
        self.assertTrue(config.enable_encryption)
        self.assertEqual(config.embedding_region, "us-west-2")
    
    @patch('secure_rag_pipeline.SecureRAGPipeline')
    def test_create_high_security_rag_pipeline(self, mock_pipeline_class):
        """Test high-security RAG pipeline creation utility."""
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        result = create_high_security_rag_pipeline(
            storage_dir="./high_security_storage",
            region="us-east-1"
        )
        
        self.assertEqual(result, mock_pipeline)
        mock_pipeline_class.assert_called_once()
        
        # Check that high-security settings were applied
        call_args = mock_pipeline_class.call_args
        config = call_args[1]['config']
        self.assertEqual(config.storage_dir, "./high_security_storage")
        self.assertTrue(config.enable_encryption)
        self.assertEqual(config.similarity_top_k, 3)  # Limited context
        self.assertEqual(config.max_response_length, 1000)  # Shorter responses
        self.assertEqual(config.max_sensitive_context, 0.1)  # Strict filtering


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complete RAG scenarios."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up integration test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('secure_rag_pipeline.AgentCoreBrowserLoader')
    @patch('secure_rag_pipeline.VectorStoreIndex')
    @patch('secure_rag_pipeline.BedrockEmbedding')
    def test_end_to_end_secure_rag_workflow(self, mock_bedrock, mock_vector_index, mock_loader_class):
        """Test complete end-to-end secure RAG workflow."""
        # Setup mocks
        mock_loader = Mock()
        mock_documents = [
            Document(
                text="Public company information about our services.",
                metadata={'source': 'https://example.com/public'}
            ),
            Document(
                text="Employee John Doe (john.doe@example.com) handles customer support.",
                metadata={'source': 'https://example.com/internal'}
            )
        ]
        mock_loader.load_data.return_value = mock_documents
        mock_loader.get_session_metrics.return_value = {'documents_loaded': 2}
        mock_loader_class.return_value = mock_loader
        
        mock_index = Mock()
        mock_vector_index.from_documents.return_value = mock_index
        
        # Create pipeline
        config = SecureRAGConfig(
            storage_dir=self.temp_dir,
            enable_encryption=False,  # Disable for testing
            enable_query_sanitization=True,
            enable_response_filtering=True
        )
        pipeline = SecureRAGPipeline(config=config)
        
        # Test ingestion
        results = pipeline.ingest_web_data(['https://example.com/public', 'https://example.com/internal'])
        
        self.assertEqual(results['documents_loaded'], 2)
        self.assertEqual(results['documents_indexed'], 2)
        self.assertIsNotNone(pipeline.query_engine)
        
        # Test querying
        mock_response = Mock(spec=Response)
        mock_response.response = "Company provides excellent services."
        mock_response.source_nodes = []
        pipeline.query_engine.query = Mock(return_value=mock_response)
        
        response = pipeline.query("What services does the company provide?")
        
        self.assertEqual(response, mock_response)
        
        # Test status
        status = pipeline.get_pipeline_status()
        self.assertTrue(status['index_status']['initialized'])
        self.assertTrue(status['index_status']['query_engine_ready'])
        
        # Test cleanup
        pipeline.cleanup()


if __name__ == '__main__':
    # Configure test logging
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
    
    # Run tests
    unittest.main(verbosity=2)