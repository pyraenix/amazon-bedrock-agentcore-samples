"""
Audit Trail Completeness Tests for LlamaIndex-AgentCore Integration

Tests to verify audit trail completeness for sensitive operations in LlamaIndex workflows.
Requirements: 4.1, 4.4
"""

import pytest
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import uuid
import tempfile
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'examples'))

from llamaindex_monitoring import SecurityAuditor, AuditLogger, ComplianceReporter
from agentcore_session_helpers import SessionManager
from sensitive_data_handler import SensitiveDataHandler
from secure_rag_pipeline import SecureRAGPipeline

try:
    from llama_index.core import Document
except ImportError:
    # Mock for testing
    class Document:
        def __init__(self, text: str, metadata: Dict = None):
            self.text = text
            self.metadata = metadata or {}


class TestAuditLogGeneration:
    """Test that audit logs are generated for all sensitive operations."""
    
    def setup_method(self):
        """Set up test environment."""
        self.auditor = SecurityAuditor()
        self.session_manager = SessionManager()
        self.data_handler = SensitiveDataHandler()
        
    def test_session_creation_audit(self):
        """Test that session creation is audited."""
        # Create a session
        session_data = {"user_id": "test_user", "role": "admin"}
        session_id = self.session_manager.create_session(session_data)
        
        # Verify audit log was created
        audit_logs = self.auditor.get_audit_logs()
        session_logs = [log for log in audit_logs if log["operation"] == "session_created"]
        
        assert len(session_logs) > 0
        
        session_log = session_logs[-1]  # Get the most recent
        assert session_log["session_id"] == session_id
        assert session_log["metadata"]["user_id"] == "test_user"
        assert "timestamp" in session_log
        assert "integrity_hash" in session_log
        
    def test_credential_access_audit(self):
        """Test that credential access is audited."""
        credentials = {"username": "test_user", "password": "secret"}
        session_id = self.session_manager.create_session(credentials)
        
        # Access credentials
        retrieved_creds = self.session_manager.get_session_credentials(session_id)
        
        # Verify audit log
        audit_logs = self.auditor.get_audit_logs()
        cred_logs = [log for log in audit_logs if log["operation"] == "credential_access"]
        
        assert len(cred_logs) > 0
        
        cred_log = cred_logs[-1]
        assert cred_log["session_id"] == session_id
        assert "username" in cred_log["metadata"]
        # Password should not be in audit log
        assert "password" not in str(cred_log)
        
    def test_pii_detection_audit(self):
        """Test that PII detection is audited."""
        document = Document(text="Contact John Doe at john.doe@example.com or 555-123-4567")
        
        # Detect PII
        pii_results = self.data_handler.detect_pii_in_document(document)
        
        # Verify audit log
        audit_logs = self.auditor.get_audit_logs()
        pii_logs = [log for log in audit_logs if log["operation"] == "pii_detection"]
        
        assert len(pii_logs) > 0
        
        pii_log = pii_logs[-1]
        assert "document_id" in pii_log["metadata"]
        assert "pii_types" in pii_log["metadata"]
        assert "EMAIL" in pii_log["metadata"]["pii_types"]
        assert "PHONE" in pii_log["metadata"]["pii_types"]
        
    def test_data_sanitization_audit(self):
        """Test that data sanitization is audited."""
        document = Document(text="Email: user@example.com, Phone: 555-123-4567")
        
        # Sanitize document
        sanitized_doc = self.data_handler.sanitize_document(document)
        
        # Verify audit log
        audit_logs = self.auditor.get_audit_logs()
        sanitization_logs = [log for log in audit_logs if log["operation"] == "data_sanitization"]
        
        assert len(sanitization_logs) > 0
        
        sanitization_log = sanitization_logs[-1]
        assert "document_id" in sanitization_log["metadata"]
        assert "pii_types_masked" in sanitization_log["metadata"]
        assert "sanitization_method" in sanitization_log["metadata"]
        
    def test_rag_query_audit(self):
        """Test that RAG queries are audited."""
        rag_pipeline = SecureRAGPipeline()
        
        # Execute a query
        query = "What is the contact information?"
        with patch.object(rag_pipeline, '_execute_query') as mock_query:
            mock_query.return_value = Mock(response="Contact information is protected")
            response = rag_pipeline.query(query)
            
        # Verify audit log
        audit_logs = self.auditor.get_audit_logs()
        query_logs = [log for log in audit_logs if log["operation"] == "rag_query"]
        
        assert len(query_logs) > 0
        
        query_log = query_logs[-1]
        assert "query_hash" in query_log["metadata"]  # Query should be hashed
        assert "response_length" in query_log["metadata"]
        assert "timestamp" in query_log
        
        # Original query should not be in log for privacy
        assert query not in str(query_log)
        
    def test_session_cleanup_audit(self):
        """Test that session cleanup is audited."""
        session_id = self.session_manager.create_session({"user": "test"})
        
        # Cleanup session
        self.session_manager.cleanup_session(session_id)
        
        # Verify audit log
        audit_logs = self.auditor.get_audit_logs()
        cleanup_logs = [log for log in audit_logs if log["operation"] == "session_cleanup"]
        
        assert len(cleanup_logs) > 0
        
        cleanup_log = cleanup_logs[-1]
        assert cleanup_log["session_id"] == session_id
        assert "cleanup_reason" in cleanup_log["metadata"]
        assert "resources_cleaned" in cleanup_log["metadata"]


class TestAuditLogIntegrity:
    """Test audit log integrity and tamper detection."""
    
    def setup_method(self):
        """Set up test environment."""
        self.auditor = SecurityAuditor()
        
    def test_audit_log_hash_generation(self):
        """Test that audit logs have integrity hashes."""
        operation_data = {
            "operation": "test_operation",
            "user": "test_user",
            "timestamp": datetime.now().isoformat(),
            "metadata": {"test": "data"}
        }
        
        log_entry = self.auditor.create_audit_log(operation_data)
        
        # Verify hash is present
        assert "integrity_hash" in log_entry
        assert len(log_entry["integrity_hash"]) > 0
        
        # Verify hash is deterministic
        log_entry2 = self.auditor.create_audit_log(operation_data)
        assert log_entry["integrity_hash"] == log_entry2["integrity_hash"]
        
    def test_audit_log_tamper_detection(self):
        """Test that tampering with audit logs is detected."""
        operation_data = {
            "operation": "sensitive_operation",
            "user": "test_user",
            "metadata": {"sensitive": True}
        }
        
        log_entry = self.auditor.create_audit_log(operation_data)
        original_hash = log_entry["integrity_hash"]
        
        # Verify original log is valid
        assert self.auditor.verify_log_integrity(log_entry) is True
        
        # Tamper with the log
        tampered_log = log_entry.copy()
        tampered_log["operation"] = "modified_operation"
        
        # Verify tampering is detected
        assert self.auditor.verify_log_integrity(tampered_log) is False
        
        # Tamper with metadata
        tampered_log2 = log_entry.copy()
        tampered_log2["metadata"]["sensitive"] = False
        
        assert self.auditor.verify_log_integrity(tampered_log2) is False
        
    def test_audit_log_chain_integrity(self):
        """Test integrity of audit log chains."""
        # Create multiple audit logs
        operations = ["op1", "op2", "op3", "op4"]
        log_entries = []
        
        for op in operations:
            log_entry = self.auditor.create_audit_log({
                "operation": op,
                "user": "test_user"
            })
            log_entries.append(log_entry)
            
        # Verify chain integrity
        chain_valid = self.auditor.verify_log_chain_integrity(log_entries)
        assert chain_valid is True
        
        # Break the chain by modifying a middle entry
        log_entries[2]["operation"] = "modified_op3"
        
        chain_valid = self.auditor.verify_log_chain_integrity(log_entries)
        assert chain_valid is False
        
    def test_audit_log_timestamp_validation(self):
        """Test that audit log timestamps are validated."""
        # Create log with future timestamp
        future_time = datetime.now() + timedelta(hours=1)
        log_entry = self.auditor.create_audit_log({
            "operation": "test_op",
            "timestamp": future_time.isoformat()
        })
        
        # Should detect invalid timestamp
        timestamp_valid = self.auditor.validate_log_timestamp(log_entry)
        assert timestamp_valid is False
        
        # Create log with reasonable timestamp
        current_time = datetime.now()
        log_entry2 = self.auditor.create_audit_log({
            "operation": "test_op",
            "timestamp": current_time.isoformat()
        })
        
        timestamp_valid = self.auditor.validate_log_timestamp(log_entry2)
        assert timestamp_valid is True


class TestAuditLogCompleteness:
    """Test that audit logs capture all required information."""
    
    def setup_method(self):
        """Set up test environment."""
        self.auditor = SecurityAuditor()
        self.session_manager = SessionManager()
        
    def test_required_fields_present(self):
        """Test that all required fields are present in audit logs."""
        required_fields = [
            "operation",
            "timestamp",
            "session_id",
            "user_id",
            "integrity_hash",
            "metadata"
        ]
        
        # Create a session and perform an operation
        session_id = self.session_manager.create_session({"user_id": "test_user"})
        
        # Get the audit log
        audit_logs = self.auditor.get_audit_logs()
        session_log = [log for log in audit_logs if log["operation"] == "session_created"][-1]
        
        # Verify all required fields are present
        for field in required_fields:
            assert field in session_log, f"Required field '{field}' missing from audit log"
            
    def test_sensitive_operation_metadata(self):
        """Test that sensitive operations include appropriate metadata."""
        # Perform a sensitive operation
        credentials = {"username": "user", "password": "secret", "api_key": "key123"}
        session_id = self.session_manager.create_session(credentials)
        
        # Get credential access log
        audit_logs = self.auditor.get_audit_logs()
        cred_logs = [log for log in audit_logs if log["operation"] == "credential_access"]
        
        if len(cred_logs) > 0:
            cred_log = cred_logs[-1]
            
            # Should include metadata about credential types
            assert "credential_types" in cred_log["metadata"]
            assert "username" in cred_log["metadata"]["credential_types"]
            assert "api_key" in cred_log["metadata"]["credential_types"]
            
            # Should not include actual credential values
            log_str = json.dumps(cred_log)
            assert "secret" not in log_str
            assert "key123" not in log_str
            
    def test_error_operation_audit(self):
        """Test that error conditions are properly audited."""
        # Force an error condition
        with patch.object(self.session_manager, '_validate_session_data') as mock_validate:
            mock_validate.side_effect = ValueError("Invalid session data")
            
            try:
                self.session_manager.create_session({"invalid": "data"})
            except ValueError:
                pass
                
        # Verify error is audited
        audit_logs = self.auditor.get_audit_logs()
        error_logs = [log for log in audit_logs if log["operation"] == "session_creation_error"]
        
        assert len(error_logs) > 0
        
        error_log = error_logs[-1]
        assert "error_type" in error_log["metadata"]
        assert "error_message" in error_log["metadata"]
        assert error_log["metadata"]["error_type"] == "ValueError"
        
    def test_performance_metrics_in_audit(self):
        """Test that performance metrics are included in audit logs."""
        # Perform an operation that should include timing
        start_time = datetime.now()
        session_id = self.session_manager.create_session({"user": "test"})
        
        # Get the audit log
        audit_logs = self.auditor.get_audit_logs()
        session_log = [log for log in audit_logs if log["operation"] == "session_created"][-1]
        
        # Should include performance metadata
        if "performance" in session_log["metadata"]:
            perf_data = session_log["metadata"]["performance"]
            assert "duration_ms" in perf_data
            assert "memory_usage" in perf_data
            assert perf_data["duration_ms"] >= 0


class TestAuditLogRetention:
    """Test audit log retention and archival."""
    
    def setup_method(self):
        """Set up test environment."""
        self.auditor = SecurityAuditor()
        
    def test_audit_log_retention_policy(self):
        """Test that audit logs follow retention policies."""
        # Generate logs over time
        old_logs = []
        recent_logs = []
        
        # Create old logs (simulate)
        for i in range(10):
            old_time = datetime.now() - timedelta(days=400)  # Very old
            log_entry = self.auditor.create_audit_log({
                "operation": f"old_operation_{i}",
                "timestamp": old_time.isoformat()
            })
            old_logs.append(log_entry)
            
        # Create recent logs
        for i in range(10):
            recent_time = datetime.now() - timedelta(days=10)  # Recent
            log_entry = self.auditor.create_audit_log({
                "operation": f"recent_operation_{i}",
                "timestamp": recent_time.isoformat()
            })
            recent_logs.append(log_entry)
            
        # Apply retention policy (e.g., keep logs for 365 days)
        self.auditor.apply_retention_policy(retention_days=365)
        
        # Verify old logs are archived/removed and recent logs remain
        current_logs = self.auditor.get_audit_logs()
        
        # Recent logs should still be present
        recent_operations = [log["operation"] for log in current_logs if "recent_operation" in log["operation"]]
        assert len(recent_operations) == 10
        
        # Old logs should be archived (not in current logs)
        old_operations = [log["operation"] for log in current_logs if "old_operation" in log["operation"]]
        assert len(old_operations) == 0
        
    def test_audit_log_archival(self):
        """Test that old audit logs are properly archived."""
        # Create logs to archive
        logs_to_archive = []
        for i in range(5):
            old_time = datetime.now() - timedelta(days=400)
            log_entry = self.auditor.create_audit_log({
                "operation": f"archive_test_{i}",
                "timestamp": old_time.isoformat()
            })
            logs_to_archive.append(log_entry)
            
        # Archive the logs
        archive_path = self.auditor.archive_logs(logs_to_archive)
        
        # Verify archive was created
        assert os.path.exists(archive_path)
        
        # Verify archived logs can be retrieved
        archived_logs = self.auditor.retrieve_archived_logs(archive_path)
        assert len(archived_logs) == 5
        
        # Verify integrity of archived logs
        for log in archived_logs:
            assert self.auditor.verify_log_integrity(log) is True


class TestComplianceReporting:
    """Test compliance reporting from audit logs."""
    
    def setup_method(self):
        """Set up test environment."""
        self.auditor = SecurityAuditor()
        self.compliance_reporter = ComplianceReporter()
        
    def test_gdpr_compliance_report(self):
        """Test GDPR compliance reporting."""
        # Generate logs with personal data operations
        operations = [
            {"operation": "personal_data_access", "data_subject": "user123"},
            {"operation": "personal_data_modification", "data_subject": "user123"},
            {"operation": "personal_data_deletion", "data_subject": "user456"},
            {"operation": "consent_granted", "data_subject": "user789"},
            {"operation": "consent_withdrawn", "data_subject": "user789"}
        ]
        
        for op in operations:
            self.auditor.create_audit_log(op)
            
        # Generate GDPR report
        gdpr_report = self.compliance_reporter.generate_gdpr_report(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now()
        )
        
        # Verify report contains required sections
        assert "data_subject_requests" in gdpr_report
        assert "consent_management" in gdpr_report
        assert "data_processing_activities" in gdpr_report
        
        # Verify specific operations are captured
        assert gdpr_report["data_subject_requests"]["access_requests"] >= 1
        assert gdpr_report["data_subject_requests"]["deletion_requests"] >= 1
        assert gdpr_report["consent_management"]["consents_granted"] >= 1
        assert gdpr_report["consent_management"]["consents_withdrawn"] >= 1
        
    def test_hipaa_compliance_report(self):
        """Test HIPAA compliance reporting."""
        # Generate logs with healthcare data operations
        operations = [
            {"operation": "phi_access", "patient_id": "patient123", "user": "doctor1"},
            {"operation": "phi_modification", "patient_id": "patient123", "user": "nurse1"},
            {"operation": "phi_disclosure", "patient_id": "patient456", "recipient": "insurance"},
            {"operation": "audit_log_access", "user": "admin1"}
        ]
        
        for op in operations:
            self.auditor.create_audit_log(op)
            
        # Generate HIPAA report
        hipaa_report = self.compliance_reporter.generate_hipaa_report(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now()
        )
        
        # Verify report contains required sections
        assert "phi_access_log" in hipaa_report
        assert "disclosures" in hipaa_report
        assert "audit_trail_access" in hipaa_report
        
        # Verify operations are properly categorized
        assert len(hipaa_report["phi_access_log"]) >= 2  # access and modification
        assert len(hipaa_report["disclosures"]) >= 1
        
    def test_sox_compliance_report(self):
        """Test SOX compliance reporting."""
        # Generate logs with financial data operations
        operations = [
            {"operation": "financial_data_access", "user": "accountant1", "data_type": "revenue"},
            {"operation": "financial_report_generation", "user": "cfo", "report_type": "quarterly"},
            {"operation": "audit_trail_review", "user": "auditor1", "scope": "q4_2024"},
            {"operation": "access_control_change", "user": "admin", "target_user": "accountant2"}
        ]
        
        for op in operations:
            self.auditor.create_audit_log(op)
            
        # Generate SOX report
        sox_report = self.compliance_reporter.generate_sox_report(
            start_date=datetime.now() - timedelta(days=90),
            end_date=datetime.now()
        )
        
        # Verify report contains required sections
        assert "financial_data_access" in sox_report
        assert "privileged_user_activities" in sox_report
        assert "access_control_changes" in sox_report
        
    def test_audit_log_export_for_compliance(self):
        """Test exporting audit logs for compliance purposes."""
        # Generate various audit logs
        for i in range(20):
            self.auditor.create_audit_log({
                "operation": f"compliance_test_{i}",
                "user": f"user_{i % 5}",
                "metadata": {"test": True}
            })
            
        # Export logs for compliance
        export_data = self.auditor.export_for_compliance(
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now(),
            format="json"
        )
        
        # Verify export format
        assert isinstance(export_data, str)  # JSON string
        exported_logs = json.loads(export_data)
        
        assert len(exported_logs) == 20
        
        # Verify each log has required compliance fields
        for log in exported_logs:
            assert "operation" in log
            assert "timestamp" in log
            assert "user" in log
            assert "integrity_hash" in log
            
        # Test CSV export
        csv_export = self.auditor.export_for_compliance(
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now(),
            format="csv"
        )
        
        assert isinstance(csv_export, str)
        assert "operation,timestamp,user" in csv_export  # CSV headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])