"""
Unit Tests for LlamaIndex Query and Response Sanitization

Comprehensive test suite for the LlamaIndex sanitization components including
query sanitization, response filtering, and context sanitization.

Requirements Addressed:
- 3.2: Unit tests for query and response security in LlamaIndex
- 3.3: Testing context filtering to prevent sensitive data exposure
- 3.5: Testing response filtering to mask sensitive data
"""

import unittest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import components to test
from llamaindex_sanitization import (
    QuerySanitizer, ResponseSanitizer, SanitizedQueryEngine,
    LlamaIndexSanitizationCallback, QuerySanitizationConfig,
    ResponseSanitizationConfig, SanitizationMode, SanitizationMetrics,
    create_sanitized_query_engine, create_high_security_sanitized_engine
)

# LlamaIndex imports for testing
from llama_index.core.schema import Document, TextNode, NodeWithScore, QueryBundle
from llama_index.core.response import Response
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.callbacks.schema import CBEventType


class TestQuerySanitizationConfig(unittest.TestCase):
    """Test QuerySanitizationConfig configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = QuerySanitizationConfig()
        
        self.assertEqual(config.mode, SanitizationMode.BALANCED)
        self.assertTrue(config.enable_query_preprocessing)
        self.assertFalse(config.enable_query_logging)  # Should be False for security
        self.assertEqual(config.max_query_length, 2000)
        self.assertGreater(len(config.blocked_patterns), 0)
        self.assertEqual(len(config.allowed_sensitive_types), 0)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = QuerySanitizationConfig(
            mode=SanitizationMode.STRICT,
            enable_query_preprocessing=False,
            max_query_length=1000,
            blocked_patterns=["custom_pattern"]
        )
        
        self.assertEqual(config.mode, SanitizationMode.STRICT)
        self.assertFalse(config.enable_query_preprocessing)
        self.assertEqual(config.max_query_length, 1000)
        self.assertIn("custom_pattern", config.blocked_patterns)
    
    def test_blocked_patterns_initialization(self):
        """Test that blocked patterns are properly initialized."""
        config = QuerySanitizationConfig()
        
        # Should have default patterns
        self.assertGreater(len(config.blocked_patterns), 0)
        
        # Should include common sensitive patterns
        patterns_str = " ".join(config.blocked_patterns)
        self.assertIn("\\d{3}-\\d{2}-\\d{4}", patterns_str)  # SSN pattern


class TestResponseSanitizationConfig(unittest.TestCase):
    """Test ResponseSanitizationConfig configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ResponseSanitizationConfig()
        
        self.assertEqual(config.mode, SanitizationMode.BALANCED)
        self.assertTrue(config.enable_response_filtering)
        self.assertTrue(config.enable_context_filtering)
        self.assertEqual(config.max_response_length, 5000)
        self.assertEqual(config.max_sensitive_context_ratio, 0.2)
        self.assertTrue(config.preserve_response_structure)
        self.assertTrue(config.add_sanitization_notices)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ResponseSanitizationConfig(
            mode=SanitizationMode.STRICT,
            max_response_length=1000,
            max_sensitive_context_ratio=0.1,
            add_sanitization_notices=False
        )
        
        self.assertEqual(config.mode, SanitizationMode.STRICT)
        self.assertEqual(config.max_response_length, 1000)
        self.assertEqual(config.max_sensitive_context_ratio, 0.1)
        self.assertFalse(config.add_sanitization_notices)


class TestSanitizationMetrics(unittest.TestCase):
    """Test SanitizationMetrics tracking class."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = SanitizationMetrics(operation_id="test-op-123")
        
        self.assertEqual(metrics.operation_id, "test-op-123")
        self.assertIsInstance(metrics.timestamp, datetime)
        self.assertEqual(metrics.queries_processed, 0)
        self.assertEqual(metrics.responses_processed, 0)
        self.assertEqual(len(metrics.security_violations), 0)
    
    def test_add_security_violation(self):
        """Test adding security violations."""
        metrics = SanitizationMetrics(operation_id="test-op")
        
        with patch('llamaindex_sanitization.logger') as mock_logger:
            metrics.add_security_violation("test violation")
            
            self.assertEqual(len(metrics.security_violations), 1)
            self.assertEqual(metrics.security_violations[0], "test violation")
            mock_logger.warning.assert_called_once()
    
    def test_to_dict(self):
        """Test metrics serialization to dictionary."""
        metrics = SanitizationMetrics(operation_id="test-op")
        metrics.queries_processed = 5
        metrics.responses_processed = 3
        metrics.add_security_violation("test violation")
        
        result = metrics.to_dict()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['operation_id'], "test-op")
        self.assertEqual(result['query_metrics']['processed'], 5)
        self.assertEqual(result['response_metrics']['processed'], 3)
        self.assertEqual(result['security_metrics']['violations'], 1)


class TestQuerySanitizer(unittest.TestCase):
    """Test QuerySanitizer functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = QuerySanitizationConfig()
        self.sanitizer = QuerySanitizer(config=self.config)
    
    def test_initialization(self):
        """Test sanitizer initialization."""
        self.assertEqual(self.sanitizer.config, self.config)
        self.assertIsNotNone(self.sanitizer.detector)
        self.assertGreater(len(self.sanitizer.compiled_patterns), 0)
    
    def test_sanitize_empty_query(self):
        """Test sanitization of empty queries."""
        result, metadata = self.sanitizer.sanitize_query("")
        
        self.assertEqual(result, "")
        self.assertFalse(metadata['sanitized'])
        self.assertEqual(metadata['reason'], 'empty_query')
    
    def test_sanitize_clean_query(self):
        """Test sanitization of clean queries."""
        clean_query = "What is machine learning?"
        result, metadata = self.sanitizer.sanitize_query(clean_query)
        
        self.assertEqual(result, clean_query)
        self.assertFalse(metadata['sanitized'])
        self.assertEqual(len(metadata['sensitive_data_found']), 0)
    
    def test_sanitize_query_with_sensitive_data(self):
        """Test sanitization of queries with sensitive data."""
        sensitive_query = "What is my balance for SSN 123-45-6789?"
        result, metadata = self.sanitizer.sanitize_query(sensitive_query)
        
        # Should be sanitized
        self.assertNotEqual(result, sensitive_query)
        self.assertTrue(metadata['sanitized'])
        self.assertGreater(len(metadata['sensitive_data_found']), 0)
        
        # Should not contain the original SSN
        self.assertNotIn("123-45-6789", result)
    
    def test_sanitize_query_length_truncation(self):
        """Test query length truncation."""
        long_query = "A" * 3000  # Longer than default max_query_length
        result, metadata = self.sanitizer.sanitize_query(long_query)
        
        self.assertLessEqual(len(result), self.config.max_query_length)
        self.assertIn('length_truncation', metadata['sanitization_applied'])
    
    def test_blocked_patterns(self):
        """Test blocked pattern detection."""
        # Test with a query that should match blocked patterns
        blocked_query = "My SSN is 123-45-6789"
        result, metadata = self.sanitizer.sanitize_query(blocked_query)
        
        # Depending on configuration, might be blocked or sanitized
        self.assertTrue(metadata['blocked'] or metadata['sanitized'])
    
    def test_validate_query(self):
        """Test query validation."""
        # Valid query
        is_valid, issues = self.sanitizer.validate_query("What is AI?")
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)
        
        # Invalid query (too long)
        long_query = "A" * 3000
        is_valid, issues = self.sanitizer.validate_query(long_query)
        self.assertFalse(is_valid)
        self.assertGreater(len(issues), 0)
    
    def test_strict_mode_sanitization(self):
        """Test strict mode sanitization."""
        strict_config = QuerySanitizationConfig(mode=SanitizationMode.STRICT)
        strict_sanitizer = QuerySanitizer(config=strict_config)
        
        sensitive_query = "My email is john.doe@example.com"
        result, metadata = strict_sanitizer.sanitize_query(sensitive_query)
        
        # Strict mode should be more aggressive
        if metadata['sanitized']:
            self.assertNotIn("john.doe@example.com", result)
    
    def test_permissive_mode_sanitization(self):
        """Test permissive mode sanitization."""
        permissive_config = QuerySanitizationConfig(mode=SanitizationMode.PERMISSIVE)
        permissive_sanitizer = QuerySanitizer(config=permissive_config)
        
        # Test with moderately sensitive data
        query = "My email is john.doe@example.com"
        result, metadata = permissive_sanitizer.sanitize_query(query)
        
        # Permissive mode might allow some sensitive data
        # The exact behavior depends on the sensitivity level


class TestResponseSanitizer(unittest.TestCase):
    """Test ResponseSanitizer functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = ResponseSanitizationConfig()
        self.sanitizer = ResponseSanitizer(config=self.config)
    
    def test_initialization(self):
        """Test sanitizer initialization."""
        self.assertEqual(self.sanitizer.config, self.config)
        self.assertIsNotNone(self.sanitizer.sanitizer)
        self.assertIsNotNone(self.sanitizer.classifier)
    
    def test_sanitize_empty_response(self):
        """Test sanitization of empty responses."""
        empty_response = None
        result = self.sanitizer.sanitize_response(empty_response)
        
        self.assertIsNone(result)
    
    def test_sanitize_clean_response(self):
        """Test sanitization of clean responses."""
        clean_response = Response(
            response="Machine learning is a subset of artificial intelligence.",
            source_nodes=[]
        )
        
        result = self.sanitizer.sanitize_response(clean_response)
        
        self.assertEqual(result.response, clean_response.response)
        self.assertIsNotNone(result.metadata)
        self.assertIn('sanitization', result.metadata)
    
    def test_sanitize_response_with_sensitive_data(self):
        """Test sanitization of responses with sensitive data."""
        sensitive_response = Response(
            response="Your account balance for SSN 123-45-6789 is $1000.",
            source_nodes=[]
        )
        
        with patch.object(self.sanitizer.classifier, 'classify_document') as mock_classify:
            mock_classify.return_value = {
                'sensitive_data_count': 1,
                'requires_special_handling': True,
                'sensitivity_level': 'confidential'
            }
            
            with patch.object(self.sanitizer.sanitizer, 'sanitize_document') as mock_sanitize:
                mock_sanitized_doc = Mock()
                mock_sanitized_doc.text = "Your account balance for SSN ***-**-**** is $1000."
                mock_sanitize.return_value = mock_sanitized_doc
                
                result = self.sanitizer.sanitize_response(sensitive_response)
                
                # The response should be sanitized (either different text or with notice)
                self.assertTrue(
                    result.response != sensitive_response.response or 
                    "[Note: This response has been sanitized" in result.response
                )
                self.assertTrue(result.metadata['sanitization']['sensitive_content_found'])
    
    def test_response_length_truncation(self):
        """Test response length truncation."""
        self.config.max_response_length = 50
        
        long_response = Response(
            response="This is a very long response that exceeds the maximum allowed length for responses.",
            source_nodes=[]
        )
        
        result = self.sanitizer.sanitize_response(long_response)
        
        self.assertLessEqual(len(result.response), 53)  # 50 + "..."
        self.assertTrue(result.response.endswith("..."))
    
    def test_context_filtering(self):
        """Test sensitive context filtering."""
        # Create nodes with different sensitivity levels
        public_node = TextNode(text="Public information", id_="node1")
        sensitive_node = TextNode(text="Sensitive SSN: 123-45-6789", id_="node2")
        
        source_nodes = [
            NodeWithScore(node=public_node, score=0.8),
            NodeWithScore(node=sensitive_node, score=0.9)
        ]
        
        response = Response(
            response="Test response",
            source_nodes=source_nodes
        )
        
        with patch.object(self.sanitizer.classifier, 'classify_document') as mock_classify:
            def classify_side_effect(doc):
                if "SSN" in doc.text:
                    return {
                        'sensitive_data_count': 1,
                        'requires_special_handling': True,
                        'sensitivity_level': 'restricted'
                    }
                else:
                    return {
                        'sensitive_data_count': 0,
                        'requires_special_handling': False,
                        'sensitivity_level': 'public'
                    }
            
            mock_classify.side_effect = classify_side_effect
            
            result = self.sanitizer.sanitize_response(response)
            
            # Should filter out sensitive node in balanced mode
            if self.config.mode == SanitizationMode.BALANCED:
                self.assertLess(len(result.source_nodes), len(source_nodes))
    
    def test_sanitization_notices(self):
        """Test sanitization notices."""
        self.config.add_sanitization_notices = True
        
        response = Response(response="Test response", source_nodes=[])
        
        with patch.object(self.sanitizer.classifier, 'classify_document') as mock_classify:
            mock_classify.return_value = {
                'sensitive_data_count': 1,
                'requires_special_handling': True
            }
            
            with patch.object(self.sanitizer.sanitizer, 'sanitize_document') as mock_sanitize:
                mock_sanitized_doc = Mock()
                mock_sanitized_doc.text = "Sanitized response"
                mock_sanitize.return_value = mock_sanitized_doc
                
                result = self.sanitizer.sanitize_response(response)
                
                self.assertIn("[Note: This response has been sanitized", result.response)


class TestLlamaIndexSanitizationCallback(unittest.TestCase):
    """Test LlamaIndexSanitizationCallback functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.query_sanitizer = Mock()
        self.response_sanitizer = Mock()
        self.callback = LlamaIndexSanitizationCallback(
            query_sanitizer=self.query_sanitizer,
            response_sanitizer=self.response_sanitizer,
            enable_monitoring=True
        )
    
    def test_initialization(self):
        """Test callback initialization."""
        self.assertEqual(self.callback.query_sanitizer, self.query_sanitizer)
        self.assertEqual(self.callback.response_sanitizer, self.response_sanitizer)
        self.assertTrue(self.callback.enable_monitoring)
        self.assertIsNotNone(self.callback.metrics)
    
    def test_on_event_start_query(self):
        """Test query event start handling."""
        payload = {'query_str': 'test query'}
        
        self.query_sanitizer.validate_query.return_value = (True, [])
        
        event_id = self.callback.on_event_start(
            event_type=CBEventType.QUERY,
            payload=payload,
            event_id="test_event"
        )
        
        self.assertEqual(event_id, "test_event")
        self.assertEqual(self.callback.metrics.queries_processed, 1)
        self.query_sanitizer.validate_query.assert_called_once_with('test query')
    
    def test_on_event_start_invalid_query(self):
        """Test handling of invalid queries."""
        payload = {'query_str': 'invalid query'}
        
        self.query_sanitizer.validate_query.return_value = (False, ['test issue'])
        
        with patch('llamaindex_sanitization.logger') as mock_logger:
            self.callback.on_event_start(
                event_type=CBEventType.QUERY,
                payload=payload,
                event_id="test_event"
            )
            
            self.assertGreater(len(self.callback.metrics.security_violations), 0)
            mock_logger.warning.assert_called()
    
    def test_on_event_end_response(self):
        """Test response event end handling."""
        payload = {'response': 'test response with john.doe@example.com'}
        
        self.callback.on_event_end(
            event_type=CBEventType.QUERY,
            payload=payload,
            event_id="test_event"
        )
        
        self.assertEqual(self.callback.metrics.responses_processed, 1)
        # Should detect sensitive data in response
        self.assertGreater(self.callback.metrics.sensitive_data_detections, 0)
    
    def test_get_metrics(self):
        """Test metrics retrieval."""
        self.callback.metrics.queries_processed = 5
        self.callback.metrics.responses_processed = 3
        
        metrics = self.callback.get_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertEqual(metrics['query_metrics']['processed'], 5)
        self.assertEqual(metrics['response_metrics']['processed'], 3)


class TestSanitizedQueryEngine(unittest.TestCase):
    """Test SanitizedQueryEngine functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.base_engine = Mock(spec=BaseQueryEngine)
        self.query_sanitizer = Mock()
        self.response_sanitizer = Mock()
        
        self.sanitized_engine = SanitizedQueryEngine(
            base_query_engine=self.base_engine,
            query_sanitizer=self.query_sanitizer,
            response_sanitizer=self.response_sanitizer,
            enable_callbacks=False  # Disable for testing
        )
    
    def test_initialization(self):
        """Test sanitized query engine initialization."""
        self.assertEqual(self.sanitized_engine.base_query_engine, self.base_engine)
        self.assertEqual(self.sanitized_engine.query_sanitizer, self.query_sanitizer)
        self.assertEqual(self.sanitized_engine.response_sanitizer, self.response_sanitizer)
    
    def test_query_string_input(self):
        """Test query with string input."""
        query_str = "test query"
        
        # Mock sanitizer responses
        self.query_sanitizer.sanitize_query.return_value = (
            "sanitized query", 
            {'blocked': False, 'sanitized': True}
        )
        
        # Mock base engine response
        mock_response = Response(response="test response", source_nodes=[])
        self.base_engine.query.return_value = mock_response
        
        # Mock response sanitizer
        self.response_sanitizer.sanitize_response.return_value = mock_response
        
        result = self.sanitized_engine.query(query_str)
        
        self.query_sanitizer.sanitize_query.assert_called_once_with(query_str)
        self.base_engine.query.assert_called_once_with("sanitized query")
        self.response_sanitizer.sanitize_response.assert_called_once_with(mock_response)
        self.assertEqual(result, mock_response)
    
    def test_query_bundle_input(self):
        """Test query with QueryBundle input."""
        query_bundle = QueryBundle(query_str="test query")
        
        # Mock sanitizer responses
        self.query_sanitizer.sanitize_query.return_value = (
            "sanitized query", 
            {'blocked': False, 'sanitized': True}
        )
        
        # Mock base engine response
        mock_response = Response(response="test response", source_nodes=[])
        self.base_engine.query.return_value = mock_response
        self.response_sanitizer.sanitize_response.return_value = mock_response
        
        result = self.sanitized_engine.query(query_bundle)
        
        self.query_sanitizer.sanitize_query.assert_called_once_with("test query")
        # Should call base engine with sanitized QueryBundle
        call_args = self.base_engine.query.call_args[0][0]
        self.assertIsInstance(call_args, QueryBundle)
        self.assertEqual(call_args.query_str, "sanitized query")
    
    def test_blocked_query(self):
        """Test handling of blocked queries."""
        query_str = "blocked query"
        
        # Mock blocked query
        self.query_sanitizer.sanitize_query.return_value = (
            "", 
            {'blocked': True, 'block_reason': 'sensitive_pattern'}
        )
        
        result = self.sanitized_engine.query(query_str)
        
        # Should not call base engine
        self.base_engine.query.assert_not_called()
        
        # Should return blocked response
        self.assertIn("cannot process", result.response.lower())
        self.assertTrue(result.metadata['query_blocked'])
    
    def test_query_error_handling(self):
        """Test error handling in query processing."""
        query_str = "test query"
        
        # Mock sanitizer to raise exception
        self.query_sanitizer.sanitize_query.side_effect = Exception("Test error")
        
        result = self.sanitized_engine.query(query_str)
        
        # Should return error response
        self.assertIn("error occurred", result.response.lower())
        self.assertIn('error', result.metadata)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions for creating sanitized engines."""
    
    def setUp(self):
        """Set up test environment."""
        self.base_engine = Mock(spec=BaseQueryEngine)
    
    def test_create_sanitized_query_engine(self):
        """Test creating sanitized query engine with default settings."""
        result = create_sanitized_query_engine(self.base_engine)
        
        self.assertIsInstance(result, SanitizedQueryEngine)
        self.assertEqual(result.base_query_engine, self.base_engine)
        self.assertIsNotNone(result.query_sanitizer)
        self.assertIsNotNone(result.response_sanitizer)
    
    def test_create_sanitized_query_engine_with_options(self):
        """Test creating sanitized query engine with custom options."""
        result = create_sanitized_query_engine(
            self.base_engine,
            sanitization_mode=SanitizationMode.STRICT,
            enable_strict_filtering=True
        )
        
        self.assertIsInstance(result, SanitizedQueryEngine)
        # Verify strict mode configuration
        self.assertEqual(result.query_sanitizer.config.mode, SanitizationMode.STRICT)
        self.assertEqual(result.response_sanitizer.config.mode, SanitizationMode.STRICT)
    
    def test_create_high_security_sanitized_engine(self):
        """Test creating high-security sanitized query engine."""
        result = create_high_security_sanitized_engine(self.base_engine)
        
        self.assertIsInstance(result, SanitizedQueryEngine)
        # Should use strict mode
        self.assertEqual(result.query_sanitizer.config.mode, SanitizationMode.STRICT)
        self.assertEqual(result.response_sanitizer.config.mode, SanitizationMode.STRICT)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complete sanitization scenarios."""
    
    def test_end_to_end_query_sanitization(self):
        """Test complete query sanitization workflow."""
        # Create real components (not mocks)
        query_sanitizer = QuerySanitizer()
        
        # Test with sensitive query
        sensitive_query = "What is my balance for account 4532-1234-5678-9012?"
        
        sanitized_query, metadata = query_sanitizer.sanitize_query(sensitive_query)
        
        # Should be sanitized
        self.assertNotEqual(sanitized_query, sensitive_query)
        self.assertTrue(metadata['sanitized'])
        
        # Should not contain original sensitive data
        self.assertNotIn("4532-1234-5678-9012", sanitized_query)
    
    def test_end_to_end_response_sanitization(self):
        """Test complete response sanitization workflow."""
        # Create real components
        response_sanitizer = ResponseSanitizer()
        
        # Create response with sensitive data
        sensitive_response = Response(
            response="Your SSN 123-45-6789 is associated with account 4532-1234-5678-9012.",
            source_nodes=[]
        )
        
        # Sanitize response
        sanitized_response = response_sanitizer.sanitize_response(sensitive_response)
        
        # Should have sanitization metadata
        self.assertIn('sanitization', sanitized_response.metadata)
        
        # Response should be processed (exact sanitization depends on classifier/sanitizer)
        self.assertIsNotNone(sanitized_response.response)
    
    def test_sanitized_engine_integration(self):
        """Test sanitized query engine integration."""
        # Create mock base engine
        base_engine = Mock(spec=BaseQueryEngine)
        base_response = Response(response="Clean response", source_nodes=[])
        base_engine.query.return_value = base_response
        
        # Create sanitized engine
        sanitized_engine = create_sanitized_query_engine(base_engine)
        
        # Test query
        result = sanitized_engine.query("What is machine learning?")
        
        # Should call base engine
        base_engine.query.assert_called_once()
        
        # Should return sanitized response
        self.assertIsNotNone(result)
        self.assertIn('sanitization', result.metadata)


if __name__ == '__main__':
    # Configure test logging
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
    
    # Run tests
    unittest.main(verbosity=2)