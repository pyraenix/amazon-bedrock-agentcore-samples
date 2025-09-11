"""
LlamaIndex-AgentCore Browser Tool Integration Security Tests

This module contains comprehensive security tests specifically for validating:
1. LlamaIndex agents securely handling sensitive information via AgentCore Browser Tool
2. RAG pipeline security when processing web data through AgentCore's containerized browser
3. Secure credential management for LlamaIndex-AgentCore authenticated web access
4. Session isolation between LlamaIndex operations using AgentCore Browser sessions
5. Audit trail completeness for LlamaIndex workflows involving sensitive web data

These tests validate the specific integration patterns described in the tutorial requirements,
focusing on real-world scenarios where LlamaIndex agents interact with sensitive web applications
through Amazon Bedrock AgentCore Browser Tool.

Requirements: 1.1-1.5, 2.1-2.5, 3.1-3.5, 4.1-4.5, 5.1-5.5
"""

import pytest
import asyncio
import json
import os
import re
import tempfile
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import uuid

# Import LlamaIndex components
try:
    from llama_index.core import Document, VectorStoreIndex
    from llama_index.core.base.base_query_engine import BaseQueryEngine
    from llama_index.core.base.response.schema import Response
    from llama_index.core.agent import ReActAgent
    from llama_index.core.tools import FunctionTool
except ImportError:
    # Fallback for different LlamaIndex versions
    Document = Mock
    VectorStoreIndex = Mock
    BaseQueryEngine = Mock
    Response = Mock
    ReActAgent = Mock
    FunctionTool = Mock

# Import AgentCore components
try:
    from bedrock_agentcore.tools.browser_client import BrowserSession, BrowserClient
    from bedrock_agentcore.tools.browser_tool import BrowserTool
except ImportError:
    # Mock for testing environment
    class BrowserSession:
        def __init__(self, session_id: str):
            self.session_id = session_id
            self.cookies = {}
            self.local_storage = {}
    
    class BrowserClient:
        def __init__(self, region: str):
            self.region = region
    
    class BrowserTool:
        def __init__(self, client: BrowserClient):
            self.client = client

# Import our LlamaIndex-AgentCore integration components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'examples'))

from agentcore_browser_loader import AgentCoreBrowserLoader
from sensitive_data_handler import SensitiveDataHandler
from secure_rag_pipeline import SecureRAGPipeline
from agentcore_session_helpers import SessionManager
from llamaindex_monitoring import SecurityAuditor


class TestLlamaIndexAgentCoreCredentialSecurity:
    """Test credential security in LlamaIndex-AgentCore Browser Tool integration."""
    
    def setup_method(self):
        """Set up test environment for LlamaIndex-AgentCore integration."""
        self.test_credentials = {
            "username": "healthcare_user",
            "password": "secure_medical_pass_123!",
            "api_key": "sk-llamaindex-agentcore-key-12345",
            "session_token": "agentcore_session_token_abcdef",
            "oauth_token": "oauth_healthcare_portal_xyz789"
        }
        self.session_manager = SessionManager()
        self.browser_loader = AgentCoreBrowserLoader()
        self.rag_pipeline = SecureRAGPipeline()
        
    def test_llamaindex_agent_credential_isolation(self, caplog):
        """Test that LlamaIndex agents don't expose credentials when using AgentCore Browser Tool."""
        with caplog.at_level(logging.DEBUG):
            # Create LlamaIndex agent with AgentCore Browser Tool
            browser_tool = self.browser_loader.create_browser_tool(self.test_credentials)
            
            # Simulate LlamaIndex agent using the browser tool for sensitive web access
            agent_response = self.browser_loader.load_authenticated_content(
                url="https://healthcare-portal.example.com/patient-records",
                credentials=self.test_credentials,
                agent_context="Extract patient information for RAG indexing"
            )
            
            # Verify no credential values appear in any logs
            log_text = caplog.text.lower()
            assert "secure_medical_pass_123!" not in log_text
            assert "sk-llamaindex-agentcore-key-12345" not in log_text
            assert "agentcore_session_token_abcdef" not in log_text
            assert "oauth_healthcare_portal_xyz789" not in log_text
            
            # Verify credential keys are masked when they appear
            if "password" in log_text:
                assert "***" in log_text or "[MASKED]" in log_text
            
    def test_llamaindex_rag_error_credential_protection(self):
        """Test that LlamaIndex RAG pipeline errors don't expose AgentCore credentials."""
        with patch.object(self.browser_loader, '_create_agentcore_session') as mock_session:
            mock_session.side_effect = Exception("AgentCore Browser Tool authentication failed")
            
            try:
                # Simulate LlamaIndex RAG pipeline trying to ingest web data via AgentCore
                documents = self.browser_loader.load_documents_for_rag(
                    urls=["https://financial-portal.example.com/reports"],
                    credentials=self.test_credentials,
                    rag_context="Financial document analysis for investment insights"
                )
                self.rag_pipeline.ingest_documents(documents)
            except Exception as e:
                error_message = str(e).lower()
                # Verify no credential values in error messages
                assert "secure_medical_pass_123!" not in error_message
                assert "sk-llamaindex-agentcore-key-12345" not in error_message
                assert "agentcore_session_token_abcdef" not in error_message
                assert "oauth_healthcare_portal_xyz789" not in error_message
                
    def test_credential_memory_cleanup(self):
        """Test that credentials are properly cleaned from memory."""
        session_id = self.session_manager.create_session(self.test_credentials)
        
        # Verify session exists
        assert self.session_manager.get_session(session_id) is not None
        
        # Clean up session
        self.session_manager.cleanup_session(session_id)
        
        # Verify credentials are removed from memory
        assert self.session_manager.get_session(session_id) is None
        
        # Verify no credential traces in session manager's internal state
        session_data = str(self.session_manager.__dict__)
        assert "test_password_123" not in session_data
        assert "sk-test-api-key-12345" not in session_data
        
    def test_credential_encryption_at_rest(self):
        """Test that credentials are encrypted when stored."""
        encrypted_creds = self.session_manager._encrypt_credentials(self.test_credentials)
        
        # Verify credentials are encrypted (not plaintext)
        encrypted_str = json.dumps(encrypted_creds)
        assert "test_password_123" not in encrypted_str
        assert "sk-test-api-key-12345" not in encrypted_str
        
        # Verify we can decrypt them back
        decrypted_creds = self.session_manager._decrypt_credentials(encrypted_creds)
        assert decrypted_creds["password"] == "test_password_123"
        assert decrypted_creds["api_key"] == "sk-test-api-key-12345"
        
    def test_session_credential_isolation(self):
        """Test that credentials are isolated between sessions."""
        creds1 = {"username": "user1", "password": "pass1"}
        creds2 = {"username": "user2", "password": "pass2"}
        
        session1_id = self.session_manager.create_session(creds1)
        session2_id = self.session_manager.create_session(creds2)
        
        # Verify sessions are different
        assert session1_id != session2_id
        
        # Verify credentials are isolated
        session1_creds = self.session_manager.get_session_credentials(session1_id)
        session2_creds = self.session_manager.get_session_credentials(session2_id)
        
        assert session1_creds["username"] == "user1"
        assert session2_creds["username"] == "user2"
        assert session1_creds["password"] != session2_creds["password"]


class TestLlamaIndexRAGSecurityWithAgentCore:
    """Test LlamaIndex RAG pipeline security when processing web data via AgentCore Browser Tool."""
    
    def setup_method(self):
        """Set up test environment for LlamaIndex RAG with AgentCore integration."""
        self.rag_pipeline = SecureRAGPipeline()
        self.browser_loader = AgentCoreBrowserLoader()
        self.data_handler = SensitiveDataHandler()
        self.session_manager = SessionManager()
        
        # Realistic sensitive web content that LlamaIndex might encounter via AgentCore
        self.healthcare_content = """
        Patient Portal - Confidential Medical Records
        
        Patient: Dr. Sarah Johnson
        DOB: 1985-03-15
        Email: sarah.johnson@healthsystem.com
        Phone: 555-987-6543
        SSN: 123-45-6789
        Insurance ID: INS-987654321
        
        Medical History:
        - Diabetes Type 2 diagnosed 2020
        - Hypertension managed with medication
        - Last visit: 2024-01-15
        
        Current Medications:
        - Metformin 500mg twice daily
        - Lisinopril 10mg once daily
        
        Lab Results (Latest):
        - HbA1c: 7.2%
        - Blood Pressure: 130/85 mmHg
        - Cholesterol: 180 mg/dL
        """
        
        self.financial_content = """
        Investment Portfolio Dashboard - Private Client
        
        Client: Michael Chen
        Account: ACC-789123456
        Email: m.chen@privateclient.com
        Phone: 555-234-5678
        SSN: 987-65-4321
        
        Portfolio Summary:
        - Total Assets: $2,450,000
        - Cash Position: $125,000
        - Equity Holdings: $1,800,000
        - Fixed Income: $525,000
        
        Recent Transactions:
        - 01/15/2024: Purchased 500 shares AAPL at $185.50
        - 01/12/2024: Sold 1000 shares MSFT at $412.25
        - 01/10/2024: Dividend received: $3,250
        
        Credit Card: 4532-1234-5678-9012
        Bank Account: 123456789 (Routing: 021000021)
        """
        
    def test_llamaindex_rag_pii_detection_via_agentcore(self):
        """Test that LlamaIndex RAG pipeline detects PII in web content loaded via AgentCore."""
        # Simulate AgentCore Browser Tool loading sensitive healthcare content
        with patch.object(self.browser_loader, 'load_web_content') as mock_load:
            mock_load.return_value = [Document(text=self.healthcare_content)]
            
            # Load documents through AgentCore for RAG processing
            documents = self.browser_loader.load_documents_for_rag(
                urls=["https://healthcare-portal.example.com/patient/12345"],
                session_config={"security_level": "HIPAA_COMPLIANT"}
            )
            
            # Detect PII in the loaded content
            for doc in documents:
                pii_results = self.data_handler.detect_pii_in_document(doc)
                
                # Verify healthcare-specific PII is detected
                pii_types = {result["type"] for result in pii_results}
                expected_types = {"EMAIL", "PHONE", "SSN", "PERSON_NAME", "MEDICAL_ID"}
                
                assert len(pii_types.intersection(expected_types)) >= 3, \
                    f"Expected healthcare PII types not detected: {pii_types}"
                    
    def test_llamaindex_rag_document_sanitization_for_agentcore_content(self):
        """Test that LlamaIndex sanitizes documents loaded via AgentCore before RAG indexing."""
        # Create document from AgentCore browser content
        financial_doc = Document(
            text=self.financial_content,
            metadata={
                "source": "agentcore_browser_session",
                "url": "https://investment-portal.example.com/portfolio/789123456",
                "session_id": "agentcore_session_abc123",
                "extraction_method": "browser_automation"
            }
        )
        
        # Sanitize document before RAG indexing
        sanitized_doc = self.data_handler.sanitize_document(financial_doc)
        
        # Verify financial PII is removed
        sensitive_values = [
            "m.chen@privateclient.com",
            "555-234-5678", 
            "987-65-4321",
            "4532-1234-5678-9012",
            "123456789"
        ]
        
        for value in sensitive_values:
            assert value not in sanitized_doc.text, \
                f"Sensitive financial data '{value}' not sanitized from AgentCore content"
                
        # Verify sanitization metadata is preserved
        assert sanitized_doc.metadata["pii_detected"] is True
        assert sanitized_doc.metadata["sanitization_applied"] is True
        assert "agentcore_browser_session" in sanitized_doc.metadata["source"]
        
    def test_llamaindex_rag_query_security_with_agentcore_data(self):
        """Test that LlamaIndex RAG queries don't leak sensitive data from AgentCore-loaded content."""
        # Ingest sanitized healthcare documents via AgentCore
        healthcare_doc = Document(text=self.healthcare_content)
        sanitized_doc = self.data_handler.sanitize_document(healthcare_doc)
        
        # Index the sanitized document in RAG pipeline
        self.rag_pipeline.ingest_documents([sanitized_doc])
        
        # Query the RAG system about the healthcare data
        sensitive_queries = [
            "What is the patient's social security number?",
            "What is Dr. Johnson's email address?",
            "What is the patient's phone number?",
            "What are the patient's insurance details?"
        ]
        
        for query in sensitive_queries:
            response = self.rag_pipeline.query(query)
            
            # Verify response doesn't contain original PII
            assert "sarah.johnson@healthsystem.com" not in response.response
            assert "555-987-6543" not in response.response
            assert "123-45-6789" not in response.response
            assert "INS-987654321" not in response.response
            
            # Verify response indicates data protection
            assert any(phrase in response.response.lower() for phrase in [
                "protected", "confidential", "not available", "privacy", "redacted"
            ]), f"Response should indicate data protection for query: {query}"
        
    def test_email_masking(self):
        """Test that email addresses are properly masked."""
        content = "Contact us at john.doe@example.com for more information."
        masked_content = self.data_handler.mask_pii(content)
        
        assert "john.doe@example.com" not in masked_content
        assert "[EMAIL]" in masked_content or "***@***.***" in masked_content
        
    def test_phone_number_masking(self):
        """Test that phone numbers are properly masked."""
        content = "Call us at 555-123-4567 or (555) 123-4567."
        masked_content = self.data_handler.mask_pii(content)
        
        assert "555-123-4567" not in masked_content
        assert "[PHONE]" in masked_content or "***-***-****" in masked_content
        
    def test_ssn_masking(self):
        """Test that SSNs are properly masked."""
        content = "SSN: 123-45-6789"
        masked_content = self.data_handler.mask_pii(content)
        
        assert "123-45-6789" not in masked_content
        assert "[SSN]" in masked_content or "***-**-****" in masked_content
        
    def test_credit_card_masking(self):
        """Test that credit card numbers are properly masked."""
        content = "Card number: 4532-1234-5678-9012"
        masked_content = self.data_handler.mask_pii(content)
        
        assert "4532-1234-5678-9012" not in masked_content
        assert "[CREDIT_CARD]" in masked_content or "****-****-****-****" in masked_content
        
    def test_document_pii_detection(self):
        """Test PII detection in LlamaIndex documents."""
        doc_content = """
        Patient Information:
        Name: John Doe
        Email: john.doe@hospital.com
        Phone: 555-123-4567
        SSN: 123-45-6789
        """
        
        document = Document(text=doc_content)
        pii_detected = self.data_handler.detect_pii_in_document(document)
        
        assert len(pii_detected) > 0
        assert any(pii["type"] == "EMAIL" for pii in pii_detected)
        assert any(pii["type"] == "PHONE" for pii in pii_detected)
        assert any(pii["type"] == "SSN" for pii in pii_detected)
        
    def test_document_sanitization(self):
        """Test document sanitization removes PII."""
        doc_content = "Contact John Doe at john.doe@example.com or 555-123-4567"
        document = Document(text=doc_content)
        
        sanitized_doc = self.data_handler.sanitize_document(document)
        
        assert "john.doe@example.com" not in sanitized_doc.text
        assert "555-123-4567" not in sanitized_doc.text
        assert sanitized_doc.metadata.get("pii_detected") is True
        assert sanitized_doc.metadata.get("sanitization_applied") is True
        
    def test_pii_masking_preserves_context(self):
        """Test that PII masking preserves document context and meaning."""
        content = "Dr. Smith can be reached at dr.smith@hospital.com for patient John Doe."
        masked_content = self.data_handler.mask_pii(content)
        
        # Should preserve non-PII context
        assert "Dr. Smith" in masked_content
        assert "hospital" in masked_content or "[EMAIL]" in masked_content
        assert "patient" in masked_content
        
    def test_false_positive_prevention(self):
        """Test that legitimate content is not incorrectly masked as PII."""
        content = "The API endpoint is https://api.example.com/v1/users"
        masked_content = self.data_handler.mask_pii(content)
        
        # API endpoints should not be masked as emails
        assert "api.example.com" in masked_content
        assert content == masked_content  # No masking should occur


class TestLlamaIndexAgentWithAgentCoreBrowserTool:
    """Test LlamaIndex Agent integration with AgentCore Browser Tool for sensitive operations."""
    
    def setup_method(self):
        """Set up test environment for LlamaIndex Agent + AgentCore Browser Tool."""
        self.session_manager = SessionManager()
        self.browser_loader = AgentCoreBrowserLoader()
        self.rag_pipeline = SecureRAGPipeline()
        
        # Mock LlamaIndex Agent with AgentCore Browser Tool
        self.browser_tool = FunctionTool.from_defaults(
            fn=self.browser_loader.browse_and_extract,
            name="agentcore_browser_tool",
            description="Securely browse web content using AgentCore Browser Tool"
        )
        
    def test_llamaindex_agent_secure_web_browsing(self):
        """Test LlamaIndex Agent using AgentCore Browser Tool for secure web access."""
        # Create LlamaIndex Agent with AgentCore Browser Tool
        agent = ReActAgent.from_tools(
            tools=[self.browser_tool],
            verbose=True
        )
        
        # Test agent browsing sensitive healthcare portal
        with patch.object(self.browser_loader, 'browse_and_extract') as mock_browse:
            mock_browse.return_value = {
                "content": "Patient records accessed successfully",
                "metadata": {
                    "url": "https://healthcare-portal.example.com/patients",
                    "session_id": "agentcore_session_123",
                    "security_level": "HIPAA_COMPLIANT"
                }
            }
            
            # Agent task: Extract patient information for analysis
            response = agent.chat(
                "Browse the healthcare portal and extract patient information for medical analysis. "
                "Ensure all PII is properly protected."
            )
            
            # Verify agent used AgentCore Browser Tool
            mock_browse.assert_called()
            
            # Verify response doesn't contain raw sensitive data
            assert "Patient records accessed successfully" in str(response)
            assert "HIPAA_COMPLIANT" not in str(response)  # Internal metadata shouldn't be exposed
            
    def test_llamaindex_agent_multi_site_session_isolation(self):
        """Test that LlamaIndex Agent maintains session isolation across multiple sites via AgentCore."""
        # Create agent with browser tool
        agent = ReActAgent.from_tools(tools=[self.browser_tool])
        
        # Simulate agent accessing multiple sensitive sites
        healthcare_session = self.session_manager.create_session({
            "site": "healthcare-portal.example.com",
            "credentials": {"username": "doctor1", "password": "medical_pass_123"},
            "compliance": "HIPAA"
        })
        
        financial_session = self.session_manager.create_session({
            "site": "investment-portal.example.com", 
            "credentials": {"username": "advisor1", "password": "finance_pass_456"},
            "compliance": "SOX"
        })
        
        # Verify sessions are isolated
        healthcare_data = self.session_manager.get_session_data(healthcare_session)
        financial_data = self.session_manager.get_session_data(financial_session)
        
        assert healthcare_data["site"] != financial_data["site"]
        assert healthcare_data["credentials"]["password"] != financial_data["credentials"]["password"]
        assert healthcare_data["compliance"] != financial_data["compliance"]
        
        # Verify no cross-contamination
        assert "medical_pass_123" not in str(financial_data)
        assert "finance_pass_456" not in str(healthcare_data)
        
    def test_llamaindex_agent_rag_workflow_with_agentcore(self):
        """Test complete LlamaIndex Agent RAG workflow using AgentCore Browser Tool."""
        # Create agent with both browser tool and RAG capabilities
        rag_tool = FunctionTool.from_defaults(
            fn=self.rag_pipeline.query,
            name="rag_query_tool",
            description="Query the RAG system for information"
        )
        
        agent = ReActAgent.from_tools(
            tools=[self.browser_tool, rag_tool],
            verbose=True
        )
        
        # Mock browser tool loading legal documents
        with patch.object(self.browser_loader, 'browse_and_extract') as mock_browse:
            mock_browse.return_value = {
                "documents": [
                    Document(
                        text="Legal Contract - Confidential Attorney-Client Privileged Information",
                        metadata={"source": "legal-portal.example.com", "classification": "CONFIDENTIAL"}
                    )
                ],
                "session_info": {
                    "session_id": "agentcore_legal_session_456",
                    "security_level": "ATTORNEY_CLIENT_PRIVILEGE"
                }
            }
            
            # Agent task: Browse legal documents and answer questions
            response = agent.chat(
                "Browse the legal document portal, extract contract information, "
                "index it in the RAG system, and then answer questions about contract terms. "
                "Maintain attorney-client privilege throughout."
            )
            
            # Verify agent used both tools appropriately
            mock_browse.assert_called()
            
            # Verify response maintains confidentiality
            assert "CONFIDENTIAL" not in str(response)  # Classification shouldn't be exposed
            assert "agentcore_legal_session_456" not in str(response)  # Session ID shouldn't be exposed
        
    def test_session_data_isolation(self):
        """Test that session data is isolated between different sessions."""
        # Create two sessions with different data
        session1_data = {"user_id": "user1", "role": "admin"}
        session2_data = {"user_id": "user2", "role": "user"}
        
        session1_id = self.session_manager.create_session(session1_data)
        session2_id = self.session_manager.create_session(session2_data)
        
        # Verify sessions are isolated
        session1_retrieved = self.session_manager.get_session_data(session1_id)
        session2_retrieved = self.session_manager.get_session_data(session2_id)
        
        assert session1_retrieved["user_id"] == "user1"
        assert session2_retrieved["user_id"] == "user2"
        assert session1_retrieved["role"] != session2_retrieved["role"]
        
    def test_session_cookie_isolation(self):
        """Test that browser cookies are isolated between sessions."""
        with patch('bedrock_agentcore.tools.browser_client.BrowserSession') as mock_session:
            mock_session1 = Mock()
            mock_session2 = Mock()
            
            # Create two browser sessions
            loader1 = AgentCoreBrowserLoader(session_id="session1")
            loader2 = AgentCoreBrowserLoader(session_id="session2")
            
            # Simulate cookie setting
            loader1.set_cookie("auth_token", "token1")
            loader2.set_cookie("auth_token", "token2")
            
            # Verify cookies are isolated
            assert loader1.get_cookie("auth_token") == "token1"
            assert loader2.get_cookie("auth_token") == "token2"
            
    def test_session_memory_isolation(self):
        """Test that session memory is properly isolated."""
        session1_id = self.session_manager.create_session({"data": "session1_data"})
        session2_id = self.session_manager.create_session({"data": "session2_data"})
        
        # Modify session1 data
        self.session_manager.update_session_data(session1_id, {"data": "modified_data"})
        
        # Verify session2 is unaffected
        session2_data = self.session_manager.get_session_data(session2_id)
        assert session2_data["data"] == "session2_data"
        
    def test_session_cleanup_isolation(self):
        """Test that cleaning up one session doesn't affect others."""
        session1_id = self.session_manager.create_session({"data": "session1"})
        session2_id = self.session_manager.create_session({"data": "session2"})
        
        # Clean up session1
        self.session_manager.cleanup_session(session1_id)
        
        # Verify session1 is gone but session2 remains
        assert self.session_manager.get_session(session1_id) is None
        assert self.session_manager.get_session(session2_id) is not None
        
    def test_concurrent_session_access(self):
        """Test that concurrent access to different sessions is safe."""
        import threading
        import time
        
        results = {}
        
        def access_session(session_id, user_data):
            session_data = self.session_manager.get_session_data(session_id)
            time.sleep(0.1)  # Simulate processing time
            results[session_id] = session_data
            
        # Create sessions
        session1_id = self.session_manager.create_session({"user": "user1"})
        session2_id = self.session_manager.create_session({"user": "user2"})
        
        # Access sessions concurrently
        thread1 = threading.Thread(target=access_session, args=(session1_id, "user1"))
        thread2 = threading.Thread(target=access_session, args=(session2_id, "user2"))
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        # Verify results are correct and isolated
        assert results[session1_id]["user"] == "user1"
        assert results[session2_id]["user"] == "user2"


class TestAuditTrailCompleteness:
    """Test audit trail completeness for sensitive operations in LlamaIndex workflows."""
    
    def setup_method(self):
        """Set up test environment."""
        self.auditor = SecurityAuditor()
        self.rag_pipeline = SecureRAGPipeline()
        self.data_handler = SensitiveDataHandler()
        
    def test_sensitive_operation_logging(self):
        """Test that all sensitive operations are properly logged."""
        # Simulate sensitive operations
        operations = [
            ("credential_access", {"user": "test_user"}),
            ("pii_detection", {"document_id": "doc123"}),
            ("data_masking", {"field": "email"}),
            ("session_creation", {"session_id": "sess456"})
        ]
        
        for operation, metadata in operations:
            self.auditor.log_sensitive_operation(operation, metadata)
            
        # Verify all operations are logged
        audit_logs = self.auditor.get_audit_logs()
        assert len(audit_logs) == 4
        
        logged_operations = [log["operation"] for log in audit_logs]
        assert "credential_access" in logged_operations
        assert "pii_detection" in logged_operations
        assert "data_masking" in logged_operations
        assert "session_creation" in logged_operations
        
    def test_audit_log_integrity(self):
        """Test that audit logs maintain integrity and cannot be tampered with."""
        operation_data = {"operation": "sensitive_query", "user": "test_user"}
        log_entry = self.auditor.log_sensitive_operation("test_operation", operation_data)
        
        # Verify log entry has integrity hash
        assert "integrity_hash" in log_entry
        assert "timestamp" in log_entry
        assert "operation" in log_entry
        
        # Verify integrity check
        assert self.auditor.verify_log_integrity(log_entry) is True
        
        # Tamper with log entry
        tampered_entry = log_entry.copy()
        tampered_entry["operation"] = "modified_operation"
        
        # Verify tampering is detected
        assert self.auditor.verify_log_integrity(tampered_entry) is False
        
    def test_pii_access_audit(self):
        """Test that PII access is properly audited."""
        document = Document(text="Contact John Doe at john.doe@example.com")
        
        # Process document with PII
        self.data_handler.detect_pii_in_document(document)
        
        # Verify PII access is audited
        audit_logs = self.auditor.get_audit_logs()
        pii_logs = [log for log in audit_logs if log["operation"] == "pii_detection"]
        
        assert len(pii_logs) > 0
        assert "document_id" in pii_logs[0]["metadata"]
        assert "pii_types" in pii_logs[0]["metadata"]
        
    def test_query_audit_trail(self):
        """Test that RAG queries are properly audited."""
        query = "What is John Doe's email address?"
        
        # Execute query through secure pipeline
        with patch.object(self.rag_pipeline, '_execute_query') as mock_query:
            mock_query.return_value = Mock(response="Email information is protected")
            self.rag_pipeline.query(query)
            
        # Verify query is audited
        audit_logs = self.auditor.get_audit_logs()
        query_logs = [log for log in audit_logs if log["operation"] == "rag_query"]
        
        assert len(query_logs) > 0
        assert "query_hash" in query_logs[0]["metadata"]  # Query should be hashed, not stored plaintext
        assert "timestamp" in query_logs[0]
        
    def test_session_lifecycle_audit(self):
        """Test that session lifecycle events are audited."""
        session_manager = SessionManager()
        
        # Create session
        session_id = session_manager.create_session({"user": "test_user"})
        
        # Update session
        session_manager.update_session_data(session_id, {"last_activity": datetime.now()})
        
        # Cleanup session
        session_manager.cleanup_session(session_id)
        
        # Verify all lifecycle events are audited
        audit_logs = self.auditor.get_audit_logs()
        session_logs = [log for log in audit_logs if "session" in log["operation"]]
        
        operations = [log["operation"] for log in session_logs]
        assert "session_created" in operations
        assert "session_updated" in operations
        assert "session_cleanup" in operations
        
    def test_audit_log_retention(self):
        """Test that audit logs are properly retained and rotated."""
        # Generate multiple audit entries
        for i in range(100):
            self.auditor.log_sensitive_operation(f"test_operation_{i}", {"index": i})
            
        # Verify logs are retained
        audit_logs = self.auditor.get_audit_logs()
        assert len(audit_logs) == 100
        
        # Test log rotation (if implemented)
        if hasattr(self.auditor, 'rotate_logs'):
            self.auditor.rotate_logs(max_entries=50)
            audit_logs = self.auditor.get_audit_logs()
            assert len(audit_logs) <= 50
            
    def test_audit_log_export(self):
        """Test that audit logs can be exported for compliance."""
        # Generate test audit data
        operations = ["login", "data_access", "pii_detection", "logout"]
        for op in operations:
            self.auditor.log_sensitive_operation(op, {"test": True})
            
        # Export audit logs
        exported_logs = self.auditor.export_audit_logs(
            start_date=datetime.now() - timedelta(hours=1),
            end_date=datetime.now()
        )
        
        assert len(exported_logs) == 4
        assert all("timestamp" in log for log in exported_logs)
        assert all("operation" in log for log in exported_logs)
        assert all("integrity_hash" in log for log in exported_logs)


class TestLlamaIndexAgentCoreEndToEndSecurity:
    """End-to-end security tests for complete LlamaIndex-AgentCore Browser Tool workflows."""
    
    def setup_method(self):
        """Set up test environment for complete integration testing."""
        self.session_manager = SessionManager()
        self.browser_loader = AgentCoreBrowserLoader()
        self.data_handler = SensitiveDataHandler()
        self.rag_pipeline = SecureRAGPipeline()
        self.auditor = SecurityAuditor()
        
        # Real-world scenario configurations
        self.healthcare_scenario = {
            "portal_url": "https://healthcare-portal.example.com",
            "credentials": {
                "username": "dr_smith",
                "password": "medical_secure_pass_2024!",
                "mfa_token": "123456"
            },
            "compliance_requirements": ["HIPAA", "SOC2"],
            "data_classification": "PHI"
        }
        
        self.financial_scenario = {
            "portal_url": "https://investment-platform.example.com", 
            "credentials": {
                "username": "advisor_jones",
                "password": "finance_secure_pass_2024!",
                "api_key": "sk-financial-api-key-xyz789"
            },
            "compliance_requirements": ["SOX", "PCI_DSS"],
            "data_classification": "FINANCIAL_PII"
        }
        
    def test_healthcare_llamaindex_agentcore_workflow(self):
        """Test complete healthcare workflow: LlamaIndex Agent + AgentCore Browser Tool + RAG."""
        # Clear audit logs for clean test
        self.auditor.clear_audit_logs()
        
        # 1. Create secure AgentCore browser session for healthcare portal
        session_id = self.session_manager.create_agentcore_session(
            credentials=self.healthcare_scenario["credentials"],
            compliance_level="HIPAA_COMPLIANT",
            data_classification=self.healthcare_scenario["data_classification"]
        )
        
        # 2. LlamaIndex Agent uses AgentCore Browser Tool to access patient portal
        with patch.object(self.browser_loader, 'load_authenticated_healthcare_data') as mock_load:
            mock_load.return_value = [
                Document(
                    text="""
                    Patient Medical Record - CONFIDENTIAL
                    
                    Patient: Jane Smith
                    DOB: 1990-05-15
                    Email: jane.smith@email.com
                    Phone: 555-123-4567
                    MRN: MR123456789
                    
                    Diagnosis: Type 2 Diabetes
                    Treatment Plan: Metformin 500mg BID
                    Last Visit: 2024-01-15
                    Next Appointment: 2024-02-15
                    
                    Lab Results:
                    - HbA1c: 7.2% (Target: <7.0%)
                    - Glucose: 145 mg/dL
                    - Blood Pressure: 130/85 mmHg
                    """,
                    metadata={
                        "source": "healthcare_portal_agentcore",
                        "session_id": session_id,
                        "compliance": "HIPAA",
                        "classification": "PHI"
                    }
                )
            ]
            
            # Load healthcare documents via AgentCore
            healthcare_docs = self.browser_loader.load_authenticated_healthcare_data(
                portal_url=self.healthcare_scenario["portal_url"],
                session_id=session_id,
                patient_ids=["12345", "67890"]
            )
        
        # 3. Sanitize healthcare documents for RAG processing
        sanitized_docs = []
        for doc in healthcare_docs:
            sanitized_doc = self.data_handler.sanitize_document(
                doc, 
                compliance_mode="HIPAA",
                preserve_medical_context=True
            )
            sanitized_docs.append(sanitized_doc)
        
        # 4. Index sanitized healthcare documents in RAG pipeline
        self.rag_pipeline.ingest_documents(
            sanitized_docs,
            index_config={"security_level": "PHI", "encryption": True}
        )
        
        # 5. LlamaIndex Agent queries healthcare RAG system
        healthcare_queries = [
            "What is the patient's current treatment plan?",
            "What are the latest lab results?", 
            "When is the next appointment scheduled?",
            "What is the patient's contact information?"  # This should be protected
        ]
        
        query_responses = []
        for query in healthcare_queries:
            response = self.rag_pipeline.query(
                query,
                security_context={"compliance": "HIPAA", "user_role": "healthcare_provider"}
            )
            query_responses.append(response)
        
        # 6. Cleanup AgentCore session
        self.session_manager.cleanup_agentcore_session(session_id)
        
        # Verify security throughout healthcare workflow
        audit_logs = self.auditor.get_audit_logs()
        
        # Check that all healthcare operations were audited
        operations = [log["operation"] for log in audit_logs]
        expected_operations = [
            "agentcore_session_created",
            "healthcare_portal_access", 
            "phi_detection",
            "hipaa_sanitization",
            "secure_rag_indexing",
            "healthcare_rag_query",
            "agentcore_session_cleanup"
        ]
        
        for expected_op in expected_operations:
            assert expected_op in operations, f"Missing audit log for: {expected_op}"
        
        # Verify PHI was properly protected in responses
        for response in query_responses:
            # Should not contain original PHI
            assert "jane.smith@email.com" not in response.response
            assert "555-123-4567" not in response.response
            assert "MR123456789" not in response.response
            
            # Medical context should be preserved (non-PII medical info)
            if "treatment plan" in response.response.lower():
                assert "metformin" in response.response.lower() or "diabetes" in response.response.lower()
        
        # Verify compliance metadata is maintained
        for log in audit_logs:
            if log["operation"] == "healthcare_rag_query":
                assert log["metadata"]["compliance"] == "HIPAA"
                assert log["metadata"]["data_classification"] == "PHI"
                
    def test_financial_llamaindex_agentcore_workflow(self):
        """Test complete financial workflow: LlamaIndex Agent + AgentCore Browser Tool + RAG."""
        # Clear audit logs
        self.auditor.clear_audit_logs()
        
        # 1. Create secure AgentCore session for financial platform
        session_id = self.session_manager.create_agentcore_session(
            credentials=self.financial_scenario["credentials"],
            compliance_level="SOX_COMPLIANT", 
            data_classification=self.financial_scenario["data_classification"]
        )
        
        # 2. Load financial data via AgentCore Browser Tool
        with patch.object(self.browser_loader, 'load_authenticated_financial_data') as mock_load:
            mock_load.return_value = [
                Document(
                    text="""
                    Investment Portfolio Report - CONFIDENTIAL
                    
                    Client: Robert Johnson
                    Account: ACC-789456123
                    Email: r.johnson@client.com
                    Phone: 555-987-6543
                    SSN: 987-65-4321
                    
                    Portfolio Value: $1,250,000
                    Cash Position: $75,000
                    
                    Holdings:
                    - AAPL: 1000 shares @ $185.50
                    - MSFT: 500 shares @ $412.25
                    - GOOGL: 200 shares @ $142.80
                    
                    Recent Transactions:
                    - 01/15/2024: Buy 100 AAPL @ $185.50
                    - 01/12/2024: Sell 200 MSFT @ $410.00
                    
                    Credit Card: 4532-9876-5432-1098
                    Bank Account: 987654321
                    """,
                    metadata={
                        "source": "financial_portal_agentcore",
                        "session_id": session_id,
                        "compliance": "SOX",
                        "classification": "FINANCIAL_PII"
                    }
                )
            ]
            
            financial_docs = self.browser_loader.load_authenticated_financial_data(
                portal_url=self.financial_scenario["portal_url"],
                session_id=session_id,
                account_ids=["ACC-789456123"]
            )
        
        # 3. Sanitize financial documents
        sanitized_docs = []
        for doc in financial_docs:
            sanitized_doc = self.data_handler.sanitize_document(
                doc,
                compliance_mode="SOX",
                preserve_financial_context=True
            )
            sanitized_docs.append(sanitized_doc)
        
        # 4. Index in RAG with financial security controls
        self.rag_pipeline.ingest_documents(
            sanitized_docs,
            index_config={"security_level": "FINANCIAL_PII", "encryption": True}
        )
        
        # 5. Query financial RAG system
        financial_queries = [
            "What is the current portfolio value?",
            "What are the recent transactions?",
            "What is the client's contact information?",  # Should be protected
            "What is the account number?"  # Should be protected
        ]
        
        for query in financial_queries:
            response = self.rag_pipeline.query(
                query,
                security_context={"compliance": "SOX", "user_role": "financial_advisor"}
            )
            
            # Verify financial PII is protected
            assert "r.johnson@client.com" not in response.response
            assert "555-987-6543" not in response.response
            assert "987-65-4321" not in response.response
            assert "4532-9876-5432-1098" not in response.response
            assert "987654321" not in response.response
            
            # Financial context should be preserved
            if "portfolio" in query.lower():
                assert "$" in response.response or "value" in response.response.lower()
        
        # 6. Cleanup
        self.session_manager.cleanup_agentcore_session(session_id)
        
        # Verify SOX compliance in audit logs
        audit_logs = self.auditor.get_audit_logs()
        sox_logs = [log for log in audit_logs if log.get("metadata", {}).get("compliance") == "SOX"]
        assert len(sox_logs) > 0, "SOX compliance not properly audited"
        
    def test_security_under_error_conditions(self):
        """Test that security is maintained even when errors occur."""
        credentials = {"username": "test", "password": "secret"}
        
        # Simulate various error conditions
        with patch.object(self.browser_loader, 'load_web_content') as mock_load:
            mock_load.side_effect = Exception("Network error")
            
            try:
                session_id = self.session_manager.create_session(credentials)
                self.browser_loader.load_web_content("https://example.com")
            except Exception:
                pass
            finally:
                # Verify credentials are still protected even after error
                audit_logs = self.auditor.get_audit_logs()
                
                # Check that no credentials appear in error logs
                for log in audit_logs:
                    log_str = json.dumps(log)
                    assert "secret" not in log_str
                    assert credentials["password"] not in log_str
                    
    def test_concurrent_security_operations(self):
        """Test security under concurrent operations."""
        import threading
        import time
        
        results = []
        
        def secure_operation(thread_id):
            try:
                # Create session
                creds = {"user": f"user_{thread_id}", "pass": f"pass_{thread_id}"}
                session_id = self.session_manager.create_session(creds)
                
                # Process sensitive data
                doc = Document(text=f"Email: user{thread_id}@example.com")
                sanitized = self.data_handler.sanitize_document(doc)
                
                # Cleanup
                self.session_manager.cleanup_session(session_id)
                
                results.append({"thread_id": thread_id, "success": True})
            except Exception as e:
                results.append({"thread_id": thread_id, "success": False, "error": str(e)})
                
        # Run concurrent operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=secure_operation, args=(i,))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Verify all operations succeeded
        assert len(results) == 5
        assert all(result["success"] for result in results)
        
        # Verify audit logs are complete and consistent
        audit_logs = self.auditor.get_audit_logs()
        assert len(audit_logs) >= 15  # At least 3 operations per thread


if __name__ == "__main__":
    pytest.main([__file__, "-v"])