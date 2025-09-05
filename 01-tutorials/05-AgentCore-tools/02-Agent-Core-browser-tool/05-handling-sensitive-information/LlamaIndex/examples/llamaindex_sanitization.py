"""
LlamaIndex Query and Response Sanitization

This module provides specialized sanitization capabilities for LlamaIndex queries and responses.
It implements query preprocessing, response filtering, and context sanitization to prevent
sensitive data exposure in RAG operations.

Key Features:
- Query sanitization for LlamaIndex queries containing sensitive information
- Response filtering to mask sensitive data in LlamaIndex agent responses
- Context filtering to prevent sensitive data exposure in RAG responses
- Real-time sensitive data detection and masking
- Comprehensive audit logging for sanitization operations

Requirements Addressed:
- 3.2: Query sanitization for LlamaIndex queries containing sensitive information
- 3.3: Response filtering to mask sensitive data in LlamaIndex agent responses
- 3.5: Context filtering to prevent sensitive data exposure in RAG responses
"""

import logging
import json
import hashlib
import re
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# LlamaIndex imports
from llama_index.core.schema import Document, QueryBundle
from llama_index.core.response import Response
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.base import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload

# Import our sensitive data handling components
from sensitive_data_handler import (
    SensitiveDataDetector, DocumentSanitizer, SensitiveDataClassifier,
    SanitizationConfig, SensitiveDataMatch, DataType, SensitivityLevel,
    MaskingStrategy, create_secure_sanitization_config
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SanitizationMode(Enum):
    """Modes for sanitization operations."""
    STRICT = "strict"  # Aggressive sanitization, may impact functionality
    BALANCED = "balanced"  # Balance between security and functionality
    PERMISSIVE = "permissive"  # Minimal sanitization, preserve functionality


@dataclass
class QuerySanitizationConfig:
    """Configuration for query sanitization."""
    mode: SanitizationMode = SanitizationMode.BALANCED
    enable_query_preprocessing: bool = True
    enable_query_logging: bool = False  # Don't log queries by default
    max_query_length: int = 2000
    blocked_patterns: List[str] = field(default_factory=list)
    allowed_sensitive_types: List[DataType] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize default blocked patterns."""
        if not self.blocked_patterns:
            self.blocked_patterns = [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN patterns
                r'\b4\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card patterns
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email patterns (if strict)
            ]


@dataclass
class ResponseSanitizationConfig:
    """Configuration for response sanitization."""
    mode: SanitizationMode = SanitizationMode.BALANCED
    enable_response_filtering: bool = True
    enable_context_filtering: bool = True
    max_response_length: int = 5000
    max_sensitive_context_ratio: float = 0.2  # Max 20% sensitive content
    preserve_response_structure: bool = True
    add_sanitization_notices: bool = True


@dataclass
class SanitizationMetrics:
    """Metrics for sanitization operations."""
    operation_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Query metrics
    queries_processed: int = 0
    queries_sanitized: int = 0
    queries_blocked: int = 0
    
    # Response metrics
    responses_processed: int = 0
    responses_sanitized: int = 0
    sensitive_content_filtered: int = 0
    
    # Context metrics
    context_nodes_processed: int = 0
    context_nodes_filtered: int = 0
    
    # Security metrics
    sensitive_data_detections: int = 0
    security_violations: List[str] = field(default_factory=list)
    
    def add_security_violation(self, violation: str):
        """Add a security violation."""
        self.security_violations.append(violation)
        logger.warning(f"Security violation: {violation}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'operation_id': self.operation_id,
            'timestamp': self.timestamp.isoformat(),
            'query_metrics': {
                'processed': self.queries_processed,
                'sanitized': self.queries_sanitized,
                'blocked': self.queries_blocked
            },
            'response_metrics': {
                'processed': self.responses_processed,
                'sanitized': self.responses_sanitized,
                'content_filtered': self.sensitive_content_filtered
            },
            'context_metrics': {
                'nodes_processed': self.context_nodes_processed,
                'nodes_filtered': self.context_nodes_filtered
            },
            'security_metrics': {
                'sensitive_detections': self.sensitive_data_detections,
                'violations': len(self.security_violations)
            }
        }


class QuerySanitizer:
    """
    Sanitizer for LlamaIndex queries containing sensitive information.
    
    Provides preprocessing and sanitization of user queries before they are
    processed by LlamaIndex query engines, preventing sensitive data from
    being used in retrieval or generation.
    """
    
    def __init__(
        self,
        config: Optional[QuerySanitizationConfig] = None,
        detector: Optional[SensitiveDataDetector] = None
    ):
        """
        Initialize query sanitizer.
        
        Args:
            config: Query sanitization configuration
            detector: Sensitive data detector instance
        """
        self.config = config or QuerySanitizationConfig()
        self.detector = detector or SensitiveDataDetector()
        
        # Compile blocked patterns for efficiency
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.config.blocked_patterns
        ]
        
        logger.info(f"QuerySanitizer initialized with mode: {self.config.mode.value}")
        logger.info(f"Blocked patterns: {len(self.compiled_patterns)}")
    
    def sanitize_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Sanitize a query string for safe processing.
        
        Args:
            query: Original query string
            context: Optional context information
            
        Returns:
            Tuple of (sanitized_query, sanitization_metadata)
        """
        if not query or not query.strip():
            return query, {'sanitized': False, 'reason': 'empty_query'}
        
        original_query = query
        sanitization_metadata = {
            'original_length': len(query),
            'sanitized': False,
            'blocked': False,
            'sensitive_data_found': [],
            'sanitization_applied': [],
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Step 1: Length validation
            if len(query) > self.config.max_query_length:
                query = query[:self.config.max_query_length]
                sanitization_metadata['sanitization_applied'].append('length_truncation')
                logger.info(f"Query truncated to {self.config.max_query_length} characters")
            
            # Step 2: Blocked pattern detection (only in strict mode)
            if self.config.mode == SanitizationMode.STRICT:
                blocked, block_reason = self._check_blocked_patterns(query)
                if blocked:
                    sanitization_metadata['blocked'] = True
                    sanitization_metadata['block_reason'] = block_reason
                    logger.warning(f"Query blocked due to: {block_reason}")
                    return "", sanitization_metadata
            
            # Step 3: Sensitive data detection
            sensitive_matches = self.detector.detect_sensitive_data(query)
            
            if sensitive_matches:
                sanitization_metadata['sensitive_data_found'] = [
                    {
                        'type': match.data_type.value,
                        'confidence': match.confidence,
                        'position': [match.start_pos, match.end_pos]
                    }
                    for match in sensitive_matches
                ]
                
                # Apply sanitization based on mode
                query = self._apply_query_sanitization(query, sensitive_matches, sanitization_metadata)
            
            # Step 4: Final validation
            if query != original_query:
                sanitization_metadata['sanitized'] = True
                sanitization_metadata['final_length'] = len(query)
                logger.info(f"Query sanitized: {len(sensitive_matches)} sensitive items processed")
            
            return query, sanitization_metadata
            
        except Exception as e:
            logger.error(f"Query sanitization failed: {str(e)}")
            sanitization_metadata['error'] = str(e)
            # Return original query on error, but log the issue
            return original_query, sanitization_metadata
    
    def _check_blocked_patterns(self, query: str) -> Tuple[bool, Optional[str]]:
        """Check if query contains blocked patterns."""
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(query):
                return True, f"blocked_pattern_{i}"
        return False, None
    
    def _apply_query_sanitization(
        self, 
        query: str, 
        matches: List[SensitiveDataMatch], 
        metadata: Dict[str, Any]
    ) -> str:
        """Apply sanitization to query based on sensitive data matches."""
        
        if self.config.mode == SanitizationMode.STRICT:
            # Strict mode: Remove or heavily mask sensitive data
            sanitized_query = self._strict_query_sanitization(query, matches)
            metadata['sanitization_applied'].append('strict_masking')
            
        elif self.config.mode == SanitizationMode.BALANCED:
            # Balanced mode: Mask sensitive data but preserve query intent
            sanitized_query = self._balanced_query_sanitization(query, matches)
            metadata['sanitization_applied'].append('balanced_masking')
            
        else:  # PERMISSIVE
            # Permissive mode: Only mask highly sensitive data
            sanitized_query = self._permissive_query_sanitization(query, matches)
            metadata['sanitization_applied'].append('permissive_masking')
        
        return sanitized_query
    
    def _strict_query_sanitization(self, query: str, matches: List[SensitiveDataMatch]) -> str:
        """Apply strict sanitization - aggressive masking."""
        # Sort matches by position (reverse order to maintain positions)
        matches.sort(key=lambda x: x.start_pos, reverse=True)
        
        sanitized_query = query
        for match in matches:
            # Replace with generic placeholder
            placeholder = f"[{match.data_type.value.upper()}]"
            sanitized_query = (
                sanitized_query[:match.start_pos] + 
                placeholder + 
                sanitized_query[match.end_pos:]
            )
        
        return sanitized_query
    
    def _balanced_query_sanitization(self, query: str, matches: List[SensitiveDataMatch]) -> str:
        """Apply balanced sanitization - preserve query intent."""
        matches.sort(key=lambda x: x.start_pos, reverse=True)
        
        sanitized_query = query
        for match in matches:
            if match.sensitivity_level in [SensitivityLevel.RESTRICTED, SensitivityLevel.CONFIDENTIAL]:
                # Mask highly sensitive data
                if match.data_type in [DataType.CREDENTIALS, DataType.FINANCIAL, DataType.HEALTH]:
                    placeholder = f"[{match.data_type.value.upper()}]"
                else:
                    # Partial masking for other sensitive data
                    original = match.original_text
                    if len(original) > 4:
                        masked = original[:2] + "*" * (len(original) - 4) + original[-2:]
                    else:
                        masked = "*" * len(original)
                    placeholder = masked
                
                sanitized_query = (
                    sanitized_query[:match.start_pos] + 
                    placeholder + 
                    sanitized_query[match.end_pos:]
                )
        
        return sanitized_query
    
    def _permissive_query_sanitization(self, query: str, matches: List[SensitiveDataMatch]) -> str:
        """Apply permissive sanitization - minimal masking."""
        matches.sort(key=lambda x: x.start_pos, reverse=True)
        
        sanitized_query = query
        for match in matches:
            # Only mask the most sensitive data types
            if match.data_type in [DataType.CREDENTIALS, DataType.FINANCIAL] and \
               match.sensitivity_level == SensitivityLevel.RESTRICTED:
                placeholder = f"[{match.data_type.value.upper()}]"
                sanitized_query = (
                    sanitized_query[:match.start_pos] + 
                    placeholder + 
                    sanitized_query[match.end_pos:]
                )
        
        return sanitized_query
    
    def validate_query(self, query: str) -> Tuple[bool, List[str]]:
        """
        Validate a query for security issues.
        
        Args:
            query: Query to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check length
        if len(query) > self.config.max_query_length:
            issues.append(f"Query too long: {len(query)} > {self.config.max_query_length}")
        
        # Check blocked patterns
        blocked, reason = self._check_blocked_patterns(query)
        if blocked:
            issues.append(f"Contains blocked pattern: {reason}")
        
        # Check for sensitive data
        matches = self.detector.detect_sensitive_data(query)
        high_risk_matches = [
            match for match in matches 
            if match.sensitivity_level == SensitivityLevel.RESTRICTED
        ]
        
        if high_risk_matches:
            issues.append(f"Contains {len(high_risk_matches)} high-risk sensitive data items")
        
        return len(issues) == 0, issues


class ResponseSanitizer:
    """
    Sanitizer for LlamaIndex responses containing sensitive information.
    
    Provides post-processing sanitization of LlamaIndex responses, including
    response text filtering, context sanitization, and metadata cleaning.
    """
    
    def __init__(
        self,
        config: Optional[ResponseSanitizationConfig] = None,
        sanitizer: Optional[DocumentSanitizer] = None,
        classifier: Optional[SensitiveDataClassifier] = None
    ):
        """
        Initialize response sanitizer.
        
        Args:
            config: Response sanitization configuration
            sanitizer: Document sanitizer instance
            classifier: Document classifier instance
        """
        self.config = config or ResponseSanitizationConfig()
        self.sanitizer = sanitizer or DocumentSanitizer()
        self.classifier = classifier or SensitiveDataClassifier()
        
        logger.info(f"ResponseSanitizer initialized with mode: {self.config.mode.value}")
        logger.info(f"Context filtering enabled: {self.config.enable_context_filtering}")
    
    def sanitize_response(self, response: Response, context: Optional[Dict[str, Any]] = None) -> Response:
        """
        Sanitize a LlamaIndex response.
        
        Args:
            response: Original LlamaIndex response
            context: Optional context information
            
        Returns:
            Sanitized response
        """
        if not response:
            return response
        
        sanitization_metadata = {
            'timestamp': datetime.now().isoformat(),
            'original_response_length': len(response.response) if response.response else 0,
            'source_nodes_count': len(response.source_nodes) if response.source_nodes else 0,
            'sanitization_applied': [],
            'sensitive_content_found': False
        }
        
        try:
            # Step 1: Sanitize main response text
            if response.response and self.config.enable_response_filtering:
                sanitized_text, text_metadata = self._sanitize_response_text(response.response)
                response.response = sanitized_text
                
                if text_metadata.get('sanitized', False):
                    sanitization_metadata['sanitization_applied'].append('response_text_sanitized')
                    sanitization_metadata['sensitive_content_found'] = True
            
            # Step 2: Sanitize source nodes (context filtering)
            if response.source_nodes and self.config.enable_context_filtering:
                filtered_nodes, context_metadata = self._filter_sensitive_context(response.source_nodes)
                response.source_nodes = filtered_nodes
                
                if context_metadata.get('nodes_filtered', 0) > 0:
                    sanitization_metadata['sanitization_applied'].append('context_filtered')
                    sanitization_metadata['nodes_filtered'] = context_metadata['nodes_filtered']
            
            # Step 3: Length validation and truncation
            if response.response and len(response.response) > self.config.max_response_length:
                response.response = response.response[:self.config.max_response_length] + "..."
                sanitization_metadata['sanitization_applied'].append('length_truncated')
            
            # Step 4: Add sanitization notice if configured
            if self.config.add_sanitization_notices and sanitization_metadata['sensitive_content_found']:
                notice = "\n\n[Note: This response has been sanitized to protect sensitive information]"
                response.response = (response.response or "") + notice
            
            # Add metadata to response
            if not hasattr(response, 'metadata') or response.metadata is None:
                response.metadata = {}
            response.metadata['sanitization'] = sanitization_metadata
            
            logger.info(f"Response sanitized: {len(sanitization_metadata['sanitization_applied'])} operations applied")
            return response
            
        except Exception as e:
            logger.error(f"Response sanitization failed: {str(e)}")
            # Add error metadata but return original response
            if not hasattr(response, 'metadata') or response.metadata is None:
                response.metadata = {}
            response.metadata['sanitization_error'] = str(e)
            return response
    
    def _sanitize_response_text(self, response_text: str) -> Tuple[str, Dict[str, Any]]:
        """Sanitize the main response text."""
        # Create a temporary document for sanitization
        temp_doc = Document(text=response_text)
        
        # Classify the document
        classification = self.classifier.classify_document(temp_doc)
        
        metadata = {
            'sanitized': False,
            'classification': classification
        }
        
        # Apply sanitization if sensitive content is detected
        if classification['sensitive_data_count'] > 0:
            sanitized_doc = self.sanitizer.sanitize_document(temp_doc)
            metadata['sanitized'] = True
            return sanitized_doc.text, metadata
        
        return response_text, metadata
    
    def _filter_sensitive_context(self, source_nodes) -> Tuple[List, Dict[str, Any]]:
        """Filter sensitive content from source nodes."""
        if not source_nodes:
            return source_nodes, {'nodes_filtered': 0}
        
        filtered_nodes = []
        nodes_filtered = 0
        total_sensitive_ratio = 0.0
        
        for node in source_nodes:
            try:
                # Get node text
                node_text = node.node.text if hasattr(node.node, 'text') else str(node.node)
                
                # Classify node content
                temp_doc = Document(text=node_text)
                classification = self.classifier.classify_document(temp_doc)
                
                # Determine if node should be filtered
                should_filter = False
                
                if self.config.mode == SanitizationMode.STRICT:
                    # Strict mode: Filter any sensitive content
                    if classification['sensitive_data_count'] > 0:
                        should_filter = True
                
                elif self.config.mode == SanitizationMode.BALANCED:
                    # Balanced mode: Filter high-sensitivity content
                    if classification['requires_special_handling']:
                        should_filter = True
                
                else:  # PERMISSIVE
                    # Permissive mode: Only filter restricted content
                    if classification['sensitivity_level'] == 'restricted':
                        should_filter = True
                
                if should_filter:
                    nodes_filtered += 1
                    logger.debug(f"Filtered sensitive node: {classification['sensitivity_level']}")
                else:
                    # Check overall sensitive content ratio
                    if classification['sensitive_data_count'] > 0:
                        total_sensitive_ratio += 1.0 / len(source_nodes)
                    
                    filtered_nodes.append(node)
                    
            except Exception as e:
                logger.warning(f"Failed to process node for filtering: {str(e)}")
                # Keep node on error
                filtered_nodes.append(node)
        
        # Additional check: if too much sensitive content overall, apply additional filtering
        if total_sensitive_ratio > self.config.max_sensitive_context_ratio:
            logger.warning(f"High sensitive content ratio: {total_sensitive_ratio:.2f}")
            # Could implement additional filtering logic here
        
        metadata = {
            'nodes_filtered': nodes_filtered,
            'total_nodes': len(source_nodes),
            'sensitive_ratio': total_sensitive_ratio
        }
        
        return filtered_nodes, metadata


class LlamaIndexSanitizationCallback(BaseCallbackHandler):
    """
    LlamaIndex callback handler for real-time sanitization monitoring.
    
    Integrates with LlamaIndex's callback system to provide real-time
    monitoring and sanitization of queries and responses during RAG operations.
    """
    
    def __init__(
        self,
        query_sanitizer: Optional[QuerySanitizer] = None,
        response_sanitizer: Optional[ResponseSanitizer] = None,
        enable_monitoring: bool = True
    ):
        """
        Initialize sanitization callback.
        
        Args:
            query_sanitizer: Query sanitizer instance
            response_sanitizer: Response sanitizer instance
            enable_monitoring: Enable monitoring and logging
        """
        super().__init__(
            event_starts_to_ignore=[],
            event_ends_to_ignore=[]
        )
        
        self.query_sanitizer = query_sanitizer
        self.response_sanitizer = response_sanitizer
        self.enable_monitoring = enable_monitoring
        
        # Metrics tracking
        self.metrics = SanitizationMetrics(operation_id=f"callback-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        
        logger.info("LlamaIndexSanitizationCallback initialized")
    
    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Start a trace (required by BaseCallbackHandler)."""
        pass
    
    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """End a trace (required by BaseCallbackHandler)."""
        pass
    
    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Handle event start."""
        if not self.enable_monitoring:
            return event_id
        
        try:
            if event_type == CBEventType.QUERY:
                # Monitor query events
                if payload and 'query_str' in payload:
                    query_str = payload['query_str']
                    self.metrics.queries_processed += 1
                    
                    # Validate query
                    if self.query_sanitizer:
                        is_valid, issues = self.query_sanitizer.validate_query(query_str)
                        if not is_valid:
                            self.metrics.add_security_violation(f"Invalid query: {issues}")
                            logger.warning(f"Query validation failed: {issues}")
            
            elif event_type == CBEventType.RETRIEVE:
                # Monitor retrieval events
                if payload:
                    self.metrics.context_nodes_processed += payload.get('node_count', 0)
            
        except Exception as e:
            logger.error(f"Sanitization callback error on event start: {str(e)}")
        
        return event_id
    
    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Handle event end."""
        if not self.enable_monitoring:
            return
        
        try:
            if event_type == CBEventType.QUERY:
                # Monitor query completion
                self.metrics.responses_processed += 1
                
                # Check if response contains sensitive data
                if payload and 'response' in payload:
                    response_text = str(payload['response'])
                    
                    # Quick sensitive data check
                    if self._contains_sensitive_patterns(response_text):
                        self.metrics.sensitive_data_detections += 1
                        logger.info("Sensitive data detected in response")
            
        except Exception as e:
            logger.error(f"Sanitization callback error on event end: {str(e)}")
    
    def _contains_sensitive_patterns(self, text: str) -> bool:
        """Quick check for common sensitive data patterns."""
        sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b4\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card
        ]
        
        for pattern in sensitive_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get sanitization metrics."""
        return self.metrics.to_dict()


class SanitizedQueryEngine:
    """
    Wrapper for LlamaIndex query engines with integrated sanitization.
    
    Provides a drop-in replacement for LlamaIndex query engines that
    automatically applies query and response sanitization.
    """
    
    def __init__(
        self,
        base_query_engine: BaseQueryEngine,
        query_sanitizer: Optional[QuerySanitizer] = None,
        response_sanitizer: Optional[ResponseSanitizer] = None,
        enable_callbacks: bool = True
    ):
        """
        Initialize sanitized query engine.
        
        Args:
            base_query_engine: Base LlamaIndex query engine
            query_sanitizer: Query sanitizer instance
            response_sanitizer: Response sanitizer instance
            enable_callbacks: Enable sanitization callbacks
        """
        self.base_query_engine = base_query_engine
        self.query_sanitizer = query_sanitizer or QuerySanitizer()
        self.response_sanitizer = response_sanitizer or ResponseSanitizer()
        
        # Setup callbacks if enabled
        if enable_callbacks:
            callback = LlamaIndexSanitizationCallback(
                query_sanitizer=self.query_sanitizer,
                response_sanitizer=self.response_sanitizer
            )
            
            # Add callback to base engine if it supports callbacks
            if hasattr(self.base_query_engine, 'callback_manager'):
                if self.base_query_engine.callback_manager is None:
                    self.base_query_engine.callback_manager = CallbackManager([callback])
                else:
                    self.base_query_engine.callback_manager.add_handler(callback)
        
        logger.info("SanitizedQueryEngine initialized")
    
    def query(self, query: Union[str, QueryBundle], **kwargs) -> Response:
        """
        Execute a sanitized query.
        
        Args:
            query: Query string or QueryBundle
            **kwargs: Additional query parameters
            
        Returns:
            Sanitized response
        """
        # Extract query string
        if isinstance(query, QueryBundle):
            query_str = query.query_str
        else:
            query_str = str(query)
        
        try:
            # Step 1: Sanitize query
            sanitized_query_str, query_metadata = self.query_sanitizer.sanitize_query(query_str)
            
            # Check if query was blocked
            if query_metadata.get('blocked', False):
                logger.warning(f"Query blocked: {query_metadata.get('block_reason')}")
                return Response(
                    response="I cannot process this query due to security restrictions.",
                    metadata={'query_blocked': True, 'reason': query_metadata.get('block_reason')}
                )
            
            # Step 2: Create sanitized query bundle
            if isinstance(query, QueryBundle):
                sanitized_query = QueryBundle(
                    query_str=sanitized_query_str,
                    custom_embedding_strs=query.custom_embedding_strs,
                    embedding=query.embedding
                )
            else:
                sanitized_query = sanitized_query_str
            
            # Step 3: Execute base query
            response = self.base_query_engine.query(sanitized_query, **kwargs)
            
            # Step 4: Sanitize response
            sanitized_response = self.response_sanitizer.sanitize_response(response)
            
            # Step 5: Add query sanitization metadata
            if not hasattr(sanitized_response, 'metadata') or sanitized_response.metadata is None:
                sanitized_response.metadata = {}
            sanitized_response.metadata['query_sanitization'] = query_metadata
            
            return sanitized_response
            
        except Exception as e:
            logger.error(f"Sanitized query execution failed: {str(e)}")
            return Response(
                response="An error occurred while processing your query.",
                metadata={'error': str(e)}
            )
    
    def aquery(self, query: Union[str, QueryBundle], **kwargs):
        """Async query method (if supported by base engine)."""
        if hasattr(self.base_query_engine, 'aquery'):
            # For async, we'd need to implement async sanitization
            # For now, fall back to sync
            return self.query(query, **kwargs)
        else:
            raise NotImplementedError("Base query engine does not support async queries")


# Utility functions for easy integration

def create_sanitized_query_engine(
    base_query_engine: BaseQueryEngine,
    sanitization_mode: SanitizationMode = SanitizationMode.BALANCED,
    enable_strict_filtering: bool = False
) -> SanitizedQueryEngine:
    """
    Create a sanitized query engine with recommended settings.
    
    Args:
        base_query_engine: Base LlamaIndex query engine
        sanitization_mode: Sanitization mode to use
        enable_strict_filtering: Enable strict filtering for high-security environments
        
    Returns:
        Configured SanitizedQueryEngine
    """
    # Configure query sanitization
    query_config = QuerySanitizationConfig(
        mode=sanitization_mode,
        enable_query_preprocessing=True,
        enable_query_logging=False  # Don't log queries for security
    )
    
    # Configure response sanitization
    response_config = ResponseSanitizationConfig(
        mode=sanitization_mode,
        enable_response_filtering=True,
        enable_context_filtering=True,
        max_sensitive_context_ratio=0.1 if enable_strict_filtering else 0.2
    )
    
    # Create sanitizers
    query_sanitizer = QuerySanitizer(config=query_config)
    response_sanitizer = ResponseSanitizer(config=response_config)
    
    return SanitizedQueryEngine(
        base_query_engine=base_query_engine,
        query_sanitizer=query_sanitizer,
        response_sanitizer=response_sanitizer,
        enable_callbacks=False  # Disable callbacks for now due to compatibility issues
    )


def create_high_security_sanitized_engine(
    base_query_engine: BaseQueryEngine
) -> SanitizedQueryEngine:
    """
    Create a high-security sanitized query engine.
    
    Args:
        base_query_engine: Base LlamaIndex query engine
        
    Returns:
        High-security configured SanitizedQueryEngine
    """
    return create_sanitized_query_engine(
        base_query_engine=base_query_engine,
        sanitization_mode=SanitizationMode.STRICT,
        enable_strict_filtering=True
    )


# Example usage
if __name__ == "__main__":
    print("LlamaIndex Query and Response Sanitization")
    print("=" * 50)
    
    # Example query sanitization
    query_sanitizer = QuerySanitizer()
    
    test_queries = [
        "What is my account balance for card 4532-1234-5678-9012?",
        "Can you help me with my SSN 123-45-6789?",
        "Tell me about machine learning algorithms",
        "What is the weather like today?"
    ]
    
    print("\n1. Query Sanitization Examples:")
    for query in test_queries:
        sanitized, metadata = query_sanitizer.sanitize_query(query)
        print(f"Original: {query}")
        print(f"Sanitized: {sanitized}")
        print(f"Metadata: {metadata.get('sanitization_applied', [])}")
        print("-" * 40)
    
    # Example response sanitization
    response_sanitizer = ResponseSanitizer()
    
    print("\n2. Response Sanitization Ready")
    print("Use with LlamaIndex Response objects for automatic sanitization")
    
    print("\n3. Integration Example:")
    print("# Create sanitized query engine")
    print("# sanitized_engine = create_sanitized_query_engine(your_query_engine)")
    print("# response = sanitized_engine.query('Your query here')")