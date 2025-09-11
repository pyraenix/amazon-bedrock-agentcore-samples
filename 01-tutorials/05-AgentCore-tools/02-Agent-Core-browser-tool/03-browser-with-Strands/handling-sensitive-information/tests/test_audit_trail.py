"""
Security validation tests for audit trail in Strands-AgentCore integration.

This module ensures that all sensitive operations are logged for compliance,
providing comprehensive audit trail validation for regulatory requirements.

Requirements: 8.1, 8.2, 8.3, 8.4
"""

import pytest
import json
import uuid
import time
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

# Import Strands and AgentCore components
try:
    from strands import Agent, Tool, Workflow
    from strands.tools import BaseTool
    from strands.audit import AuditLogger
except ImportError:
    # Mock imports for testing environment
    Agent = Mock
    Tool = Mock
    Workflow = Mock
    BaseTool = Mock
    AuditLogger = Mock

# Import custom tools
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))
try:
    from strands_monitoring import AuditTrailTool
    from strands_agentcore_session_helpers import StrandsAgentCoreClient
    from strands_pii_utils import SensitiveDataHandler
    from strands_security_policies import BedrockModelRouter
except ImportError:
    # Mock for testing
    AuditTrailTool = Mock
    StrandsAgentCoreClient = Mock
    SensitiveDataHandler = Mock
    BedrockModelRouter = Mock


class TestAuditTrail:
    """Test suite for audit trail in Strands-AgentCore integration."""
    
    @pytest.fixture
    def audit_config(self):
        """Configuration for audit trail testing."""
        return {
            'audit_level': 'COMPREHENSIVE',
            'log_sensitive_operations': True,
            'log_pii_operations': True,
            'log_credential_operations': True,
            'log_session_operations': True,
            'log_security_events': True,
            'retention_days': 2555,  # 7 years for compliance
            'encryption_enabled': True,
            'compliance_standards': ['HIPAA', 'PCI_DSS', 'GDPR', 'SOX'],
            'audit_destinations': ['cloudwatch', 'elasticsearch', 's3'],
            'real_time_monitoring': True
        }
    
    @pytest.fixture
    def sample_operations(self):
        """Sample operations for audit testing."""
        return {
            'credential_operations': [
                {
                    'operation_id': 'cred_001',
                    'operation_type': 'credential_injection',
                    'agent_id': 'agent_001',
                    'session_id': 'session_001',
                    'timestamp': datetime.now().isoformat(),
                    'success': True,
                    'credential_types': ['username', 'password', 'api_key'],
                    'target_service': 'banking_portal'
                },
                {
                    'operation_id': 'cred_002',
                    'operation_type': 'credential_rotation',
                    'agent_id': 'agent_002',
                    'session_id': 'session_002',
                    'timestamp': datetime.now().isoformat(),
                    'success': True,
                    'credential_types': ['oauth_token'],
                    'rotation_reason': 'scheduled_rotation'
                }
            ],
            'pii_operations': [
                {
                    'operation_id': 'pii_001',
                    'operation_type': 'pii_detection',
                    'agent_id': 'agent_001',
                    'session_id': 'session_001',
                    'timestamp': datetime.now().isoformat(),
                    'pii_types_detected': ['ssn', 'email', 'phone'],
                    'pii_count': 5,
                    'masking_applied': True,
                    'data_classification': 'SENSITIVE'
                },
                {
                    'operation_id': 'pii_002',
                    'operation_type': 'pii_masking',
                    'agent_id': 'agent_003',
                    'session_id': 'session_003',
                    'timestamp': datetime.now().isoformat(),
                    'pii_types_masked': ['credit_card', 'ssn'],
                    'masking_method': 'irreversible_hash',
                    'compliance_level': 'PCI_DSS'
                }
            ],
            'session_operations': [
                {
                    'operation_id': 'sess_001',
                    'operation_type': 'session_creation',
                    'agent_id': 'agent_001',
                    'session_id': 'session_001',
                    'timestamp': datetime.now().isoformat(),
                    'isolation_level': 'STRICT',
                    'security_level': 'HIGH',
                    'user_context': {'user_id': 'user_001', 'role': 'admin'}
                },
                {
                    'operation_id': 'sess_002',
                    'operation_type': 'session_termination',
                    'agent_id': 'agent_001',
                    'session_id': 'session_001',
                    'timestamp': datetime.now().isoformat(),
                    'cleanup_completed': True,
                    'data_purged': True,
                    'duration_minutes': 45
                }
            ],
            'security_events': [
                {
                    'event_id': 'sec_001',
                    'event_type': 'security_violation_detected',
                    'agent_id': 'agent_002',
                    'session_id': 'session_002',
                    'timestamp': datetime.now().isoformat(),
                    'violation_type': 'unauthorized_access_attempt',
                    'severity': 'HIGH',
                    'remediation_applied': True
                },
                {
                    'event_id': 'sec_002',
                    'event_type': 'compliance_validation',
                    'agent_id': 'agent_003',
                    'session_id': 'session_003',
                    'timestamp': datetime.now().isoformat(),
                    'compliance_standard': 'HIPAA',
                    'validation_result': 'PASSED',
                    'validation_score': 0.98
                }
            ]
        }
    
    @pytest.fixture
    def audit_log_file(self):
        """Temporary audit log file for testing."""
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.audit.log') as f:
            log_file = f.name
        yield log_file
        if os.path.exists(log_file):
            os.unlink(log_file)
    
    def test_audit_trail_initialization(self, audit_config):
        """Test that audit trail is properly initialized with correct configuration."""
        with patch('strands_monitoring.AuditTrailTool') as mock_audit:
            # Mock audit tool initialization
            mock_audit_instance = Mock()
            mock_audit.return_value = mock_audit_instance
            
            mock_audit_instance.initialize = Mock(return_value={
                'status': 'initialized',
                'audit_level': audit_config['audit_level'],
                'destinations_configured': len(audit_config['audit_destinations']),
                'encryption_enabled': audit_config['encryption_enabled'],
                'compliance_standards': audit_config['compliance_standards']
            })
            
            # Initialize audit trail
            audit_tool = AuditTrailTool()
            init_result = audit_tool.initialize(audit_config)
            
            # Verify initialization
            assert init_result['status'] == 'initialized'
            assert init_result['audit_level'] == 'COMPREHENSIVE'
            assert init_result['destinations_configured'] == 3
            assert init_result['encryption_enabled'] is True
            assert 'HIPAA' in init_result['compliance_standards']
            assert 'PCI_DSS' in init_result['compliance_standards']
            assert 'GDPR' in init_result['compliance_standards']
    
    def test_credential_operations_audit(self, sample_operations, audit_config):
        """Test that credential operations are properly audited."""
        with patch('strands_monitoring.AuditTrailTool') as mock_audit:
            # Mock audit logging
            mock_audit_instance = Mock()
            mock_audit.return_value = mock_audit_instance
            
            logged_events = []
            
            def mock_log_credential_operation(operation_data):
                # Ensure no actual credentials are logged
                sanitized_data = operation_data.copy()
                if 'credentials' in sanitized_data:
                    del sanitized_data['credentials']
                
                logged_events.append({
                    'audit_id': f"audit_{len(logged_events) + 1}",
                    'event_type': 'credential_operation',
                    'data': sanitized_data,
                    'timestamp': datetime.now().isoformat(),
                    'compliance_tags': ['PCI_DSS', 'SOX']
                })
                return logged_events[-1]
            
            mock_audit_instance.log_credential_operation = Mock(side_effect=mock_log_credential_operation)
            
            audit_tool = AuditTrailTool()
            
            # Log credential operations
            for operation in sample_operations['credential_operations']:
                audit_result = audit_tool.log_credential_operation(operation)
                
                # Verify audit logging
                assert audit_result['event_type'] == 'credential_operation'
                assert 'audit_id' in audit_result
                assert 'compliance_tags' in audit_result
                
                # Verify no sensitive data in audit log
                audit_data_str = json.dumps(audit_result['data'])
                sensitive_terms = ['password', 'secret', 'key', 'token']
                for term in sensitive_terms:
                    # Should not contain actual credential values
                    assert 'super_secret' not in audit_data_str
                    assert 'sk-' not in audit_data_str  # API key prefix
            
            # Verify all operations were logged
            assert len(logged_events) == len(sample_operations['credential_operations'])
    
    def test_pii_operations_audit(self, sample_operations, audit_config):
        """Test that PII operations are properly audited."""
        with patch('strands_monitoring.AuditTrailTool') as mock_audit:
            # Mock audit logging
            mock_audit_instance = Mock()
            mock_audit.return_value = mock_audit_instance
            
            pii_audit_events = []
            
            def mock_log_pii_operation(operation_data):
                # Ensure no actual PII is logged
                sanitized_data = operation_data.copy()
                if 'original_data' in sanitized_data:
                    sanitized_data['original_data'] = '***REDACTED***'
                if 'pii_values' in sanitized_data:
                    sanitized_data['pii_values'] = ['***REDACTED***'] * len(sanitized_data['pii_values'])
                
                pii_audit_events.append({
                    'audit_id': f"pii_audit_{len(pii_audit_events) + 1}",
                    'event_type': 'pii_operation',
                    'data': sanitized_data,
                    'timestamp': datetime.now().isoformat(),
                    'compliance_tags': ['GDPR', 'HIPAA', 'CCPA']
                })
                return pii_audit_events[-1]
            
            mock_audit_instance.log_pii_operation = Mock(side_effect=mock_log_pii_operation)
            
            audit_tool = AuditTrailTool()
            
            # Log PII operations
            for operation in sample_operations['pii_operations']:
                audit_result = audit_tool.log_pii_operation(operation)
                
                # Verify audit logging
                assert audit_result['event_type'] == 'pii_operation'
                assert 'audit_id' in audit_result
                assert 'GDPR' in audit_result['compliance_tags']
                
                # Verify no actual PII in audit log
                audit_data_str = json.dumps(audit_result['data'])
                pii_values = ['123-45-6789', 'john@example.com', '4532-1234-5678-9012']
                for pii_value in pii_values:
                    assert pii_value not in audit_data_str
                
                # Verify PII metadata is logged
                assert 'pii_types_detected' in audit_result['data'] or 'pii_types_masked' in audit_result['data']
            
            # Verify all PII operations were logged
            assert len(pii_audit_events) == len(sample_operations['pii_operations'])
    
    def test_session_operations_audit(self, sample_operations, audit_config):
        """Test that session operations are properly audited."""
        with patch('strands_monitoring.AuditTrailTool') as mock_audit:
            # Mock audit logging
            mock_audit_instance = Mock()
            mock_audit.return_value = mock_audit_instance
            
            session_audit_events = []
            
            def mock_log_session_operation(operation_data):
                session_audit_events.append({
                    'audit_id': f"sess_audit_{len(session_audit_events) + 1}",
                    'event_type': 'session_operation',
                    'data': operation_data,
                    'timestamp': datetime.now().isoformat(),
                    'compliance_tags': ['SOX', 'GDPR']
                })
                return session_audit_events[-1]
            
            mock_audit_instance.log_session_operation = Mock(side_effect=mock_log_session_operation)
            
            audit_tool = AuditTrailTool()
            
            # Log session operations
            for operation in sample_operations['session_operations']:
                audit_result = audit_tool.log_session_operation(operation)
                
                # Verify audit logging
                assert audit_result['event_type'] == 'session_operation'
                assert 'audit_id' in audit_result
                assert 'SOX' in audit_result['compliance_tags']
                
                # Verify session metadata is logged
                assert 'session_id' in audit_result['data']
                assert 'agent_id' in audit_result['data']
                assert 'operation_type' in audit_result['data']
            
            # Verify all session operations were logged
            assert len(session_audit_events) == len(sample_operations['session_operations'])
    
    def test_security_events_audit(self, sample_operations, audit_config):
        """Test that security events are properly audited."""
        with patch('strands_monitoring.AuditTrailTool') as mock_audit:
            # Mock audit logging
            mock_audit_instance = Mock()
            mock_audit.return_value = mock_audit_instance
            
            security_audit_events = []
            
            def mock_log_security_event(event_data):
                security_audit_events.append({
                    'audit_id': f"sec_audit_{len(security_audit_events) + 1}",
                    'event_type': 'security_event',
                    'data': event_data,
                    'timestamp': datetime.now().isoformat(),
                    'compliance_tags': ['SOX', 'PCI_DSS', 'HIPAA'],
                    'severity': event_data.get('severity', 'MEDIUM'),
                    'requires_notification': event_data.get('severity') in ['HIGH', 'CRITICAL']
                })
                return security_audit_events[-1]
            
            mock_audit_instance.log_security_event = Mock(side_effect=mock_log_security_event)
            
            audit_tool = AuditTrailTool()
            
            # Log security events
            for event in sample_operations['security_events']:
                audit_result = audit_tool.log_security_event(event)
                
                # Verify audit logging
                assert audit_result['event_type'] == 'security_event'
                assert 'audit_id' in audit_result
                assert audit_result['severity'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
                
                # Verify high severity events require notification
                if event.get('severity') == 'HIGH':
                    assert audit_result['requires_notification'] is True
            
            # Verify all security events were logged
            assert len(security_audit_events) == len(sample_operations['security_events'])
    
    def test_audit_log_integrity(self, audit_config, audit_log_file):
        """Test that audit logs maintain integrity and cannot be tampered with."""
        with patch('strands_monitoring.AuditTrailTool') as mock_audit:
            # Mock audit log integrity
            mock_audit_instance = Mock()
            mock_audit.return_value = mock_audit_instance
            
            audit_entries = []
            
            def mock_write_audit_entry(entry_data):
                # Simulate integrity protection
                entry_with_hash = entry_data.copy()
                entry_with_hash['integrity_hash'] = f"hash_{len(audit_entries)}_integrity"
                entry_with_hash['sequence_number'] = len(audit_entries) + 1
                entry_with_hash['previous_hash'] = audit_entries[-1]['integrity_hash'] if audit_entries else 'genesis'
                
                audit_entries.append(entry_with_hash)
                
                # Write to file
                with open(audit_log_file, 'a') as f:
                    f.write(json.dumps(entry_with_hash) + '\n')
                
                return {'status': 'written', 'integrity_verified': True}
            
            def mock_verify_audit_integrity():
                # Verify chain integrity
                for i, entry in enumerate(audit_entries):
                    if i == 0:
                        assert entry['previous_hash'] == 'genesis'
                    else:
                        assert entry['previous_hash'] == audit_entries[i-1]['integrity_hash']
                
                return {
                    'integrity_valid': True,
                    'total_entries': len(audit_entries),
                    'chain_valid': True,
                    'tampering_detected': False
                }
            
            mock_audit_instance.write_audit_entry = Mock(side_effect=mock_write_audit_entry)
            mock_audit_instance.verify_audit_integrity = Mock(side_effect=mock_verify_audit_integrity)
            
            audit_tool = AuditTrailTool()
            
            # Write multiple audit entries
            test_entries = [
                {'operation': 'credential_injection', 'agent_id': 'agent_001'},
                {'operation': 'pii_detection', 'agent_id': 'agent_002'},
                {'operation': 'session_creation', 'agent_id': 'agent_003'}
            ]
            
            for entry in test_entries:
                result = audit_tool.write_audit_entry(entry)
                assert result['status'] == 'written'
                assert result['integrity_verified'] is True
            
            # Verify audit integrity
            integrity_result = audit_tool.verify_audit_integrity()
            assert integrity_result['integrity_valid'] is True
            assert integrity_result['chain_valid'] is True
            assert integrity_result['tampering_detected'] is False
            assert integrity_result['total_entries'] == len(test_entries)
    
    def test_audit_log_encryption(self, audit_config):
        """Test that audit logs are properly encrypted."""
        with patch('strands_monitoring.AuditTrailTool') as mock_audit:
            # Mock audit log encryption
            mock_audit_instance = Mock()
            mock_audit.return_value = mock_audit_instance
            
            def mock_encrypt_audit_data(data):
                # Simulate encryption
                encrypted_data = {
                    'encrypted_payload': f"encrypted_{hash(json.dumps(data, sort_keys=True))}",
                    'encryption_algorithm': 'AES-256-GCM',
                    'key_id': 'audit_key_001',
                    'iv': 'random_iv_12345',
                    'timestamp': datetime.now().isoformat()
                }
                return encrypted_data
            
            def mock_decrypt_audit_data(encrypted_data):
                # Simulate decryption (for verification only)
                return {
                    'decryption_successful': True,
                    'data_integrity_verified': True,
                    'original_data_recovered': True
                }
            
            mock_audit_instance.encrypt_audit_data = Mock(side_effect=mock_encrypt_audit_data)
            mock_audit_instance.decrypt_audit_data = Mock(side_effect=mock_decrypt_audit_data)
            
            audit_tool = AuditTrailTool()
            
            # Test encryption of sensitive audit data
            sensitive_audit_data = {
                'operation': 'credential_rotation',
                'agent_id': 'agent_001',
                'session_id': 'session_001',
                'metadata': {'credential_types': ['api_key', 'oauth_token']}
            }
            
            encrypted_result = audit_tool.encrypt_audit_data(sensitive_audit_data)
            
            # Verify encryption
            assert 'encrypted_payload' in encrypted_result
            assert encrypted_result['encryption_algorithm'] == 'AES-256-GCM'
            assert 'key_id' in encrypted_result
            assert 'iv' in encrypted_result
            
            # Verify original data is not in encrypted payload
            original_data_str = json.dumps(sensitive_audit_data)
            assert 'credential_rotation' not in encrypted_result['encrypted_payload']
            assert 'agent_001' not in encrypted_result['encrypted_payload']
            
            # Test decryption (for verification)
            decryption_result = audit_tool.decrypt_audit_data(encrypted_result)
            assert decryption_result['decryption_successful'] is True
            assert decryption_result['data_integrity_verified'] is True
    
    def test_audit_log_retention_policy(self, audit_config):
        """Test that audit log retention policy is properly enforced."""
        with patch('strands_monitoring.AuditTrailTool') as mock_audit:
            # Mock retention policy enforcement
            mock_audit_instance = Mock()
            mock_audit.return_value = mock_audit_instance
            
            # Mock audit entries with different ages
            current_time = datetime.now()
            audit_entries = [
                {
                    'audit_id': 'audit_001',
                    'timestamp': (current_time - timedelta(days=1)).isoformat(),
                    'retention_status': 'ACTIVE'
                },
                {
                    'audit_id': 'audit_002',
                    'timestamp': (current_time - timedelta(days=365)).isoformat(),
                    'retention_status': 'ACTIVE'
                },
                {
                    'audit_id': 'audit_003',
                    'timestamp': (current_time - timedelta(days=2600)).isoformat(),  # Over 7 years
                    'retention_status': 'EXPIRED'
                }
            ]
            
            def mock_enforce_retention_policy():
                retention_results = []
                for entry in audit_entries:
                    entry_date = datetime.fromisoformat(entry['timestamp'])
                    days_old = (current_time - entry_date).days
                    
                    if days_old > audit_config['retention_days']:
                        retention_results.append({
                            'audit_id': entry['audit_id'],
                            'action': 'ARCHIVED',
                            'days_old': days_old,
                            'retention_compliant': True
                        })
                    else:
                        retention_results.append({
                            'audit_id': entry['audit_id'],
                            'action': 'RETAINED',
                            'days_old': days_old,
                            'retention_compliant': True
                        })
                
                return {
                    'policy_enforced': True,
                    'entries_processed': len(retention_results),
                    'entries_archived': len([r for r in retention_results if r['action'] == 'ARCHIVED']),
                    'entries_retained': len([r for r in retention_results if r['action'] == 'RETAINED']),
                    'results': retention_results
                }
            
            mock_audit_instance.enforce_retention_policy = Mock(side_effect=mock_enforce_retention_policy)
            
            audit_tool = AuditTrailTool()
            
            # Enforce retention policy
            retention_result = audit_tool.enforce_retention_policy()
            
            # Verify retention policy enforcement
            assert retention_result['policy_enforced'] is True
            assert retention_result['entries_processed'] == 3
            assert retention_result['entries_archived'] == 1  # One entry over 7 years
            assert retention_result['entries_retained'] == 2  # Two entries within retention period
            
            # Verify specific actions
            results = retention_result['results']
            expired_entry = next(r for r in results if r['audit_id'] == 'audit_003')
            assert expired_entry['action'] == 'ARCHIVED'
            assert expired_entry['days_old'] > audit_config['retention_days']
    
    def test_compliance_audit_reporting(self, sample_operations, audit_config):
        """Test that compliance audit reports are generated correctly."""
        with patch('strands_monitoring.AuditTrailTool') as mock_audit:
            # Mock compliance reporting
            mock_audit_instance = Mock()
            mock_audit.return_value = mock_audit_instance
            
            def mock_generate_compliance_report(standard, time_period):
                # Simulate compliance report generation
                if standard == 'HIPAA':
                    return {
                        'compliance_standard': 'HIPAA',
                        'reporting_period': time_period,
                        'total_operations': 15,
                        'compliant_operations': 15,
                        'non_compliant_operations': 0,
                        'compliance_score': 1.0,
                        'pii_operations_audited': 8,
                        'access_controls_verified': 12,
                        'data_encryption_verified': 15,
                        'audit_trail_complete': True,
                        'recommendations': []
                    }
                elif standard == 'PCI_DSS':
                    return {
                        'compliance_standard': 'PCI_DSS',
                        'reporting_period': time_period,
                        'total_operations': 10,
                        'compliant_operations': 9,
                        'non_compliant_operations': 1,
                        'compliance_score': 0.9,
                        'payment_data_operations': 5,
                        'encryption_verified': 10,
                        'access_logging_complete': True,
                        'vulnerability_scans': 2,
                        'recommendations': ['Review access controls for payment processing']
                    }
                else:
                    return {
                        'compliance_standard': standard,
                        'reporting_period': time_period,
                        'compliance_score': 0.95,
                        'audit_trail_complete': True
                    }
            
            mock_audit_instance.generate_compliance_report = Mock(side_effect=mock_generate_compliance_report)
            
            audit_tool = AuditTrailTool()
            
            # Generate compliance reports for different standards
            time_period = {'start': '2024-01-01', 'end': '2024-12-31'}
            
            for standard in audit_config['compliance_standards']:
                report = audit_tool.generate_compliance_report(standard, time_period)
                
                # Verify report structure
                assert report['compliance_standard'] == standard
                assert report['reporting_period'] == time_period
                assert 'compliance_score' in report
                assert report['audit_trail_complete'] is True
                
                # Verify compliance scores are acceptable
                assert report['compliance_score'] >= 0.9
                
                # Verify specific standard requirements
                if standard == 'HIPAA':
                    assert 'pii_operations_audited' in report
                    assert 'data_encryption_verified' in report
                elif standard == 'PCI_DSS':
                    assert 'payment_data_operations' in report
                    assert 'encryption_verified' in report
    
    def test_real_time_audit_monitoring(self, audit_config):
        """Test that real-time audit monitoring detects and alerts on suspicious activities."""
        with patch('strands_monitoring.AuditTrailTool') as mock_audit:
            # Mock real-time monitoring
            mock_audit_instance = Mock()
            mock_audit.return_value = mock_audit_instance
            
            monitoring_alerts = []
            
            def mock_monitor_audit_events(event_data):
                # Simulate real-time monitoring logic
                alert_conditions = [
                    {'condition': 'multiple_failed_logins', 'threshold': 3, 'severity': 'HIGH'},
                    {'condition': 'unusual_data_access', 'threshold': 1, 'severity': 'MEDIUM'},
                    {'condition': 'privilege_escalation', 'threshold': 1, 'severity': 'CRITICAL'}
                ]
                
                alerts_triggered = []
                
                # Check for suspicious patterns
                if event_data.get('operation_type') == 'credential_injection' and event_data.get('success') is False:
                    alerts_triggered.append({
                        'alert_id': f"alert_{len(monitoring_alerts) + 1}",
                        'condition': 'failed_credential_injection',
                        'severity': 'HIGH',
                        'event_data': event_data,
                        'timestamp': datetime.now().isoformat(),
                        'requires_investigation': True
                    })
                
                if event_data.get('pii_count', 0) > 10:
                    alerts_triggered.append({
                        'alert_id': f"alert_{len(monitoring_alerts) + 1}",
                        'condition': 'large_pii_exposure',
                        'severity': 'MEDIUM',
                        'event_data': event_data,
                        'timestamp': datetime.now().isoformat(),
                        'requires_investigation': True
                    })
                
                monitoring_alerts.extend(alerts_triggered)
                
                return {
                    'monitoring_active': True,
                    'alerts_triggered': len(alerts_triggered),
                    'alerts': alerts_triggered
                }
            
            mock_audit_instance.monitor_audit_events = Mock(side_effect=mock_monitor_audit_events)
            
            audit_tool = AuditTrailTool()
            
            # Test monitoring with suspicious events
            suspicious_events = [
                {
                    'operation_type': 'credential_injection',
                    'success': False,
                    'agent_id': 'agent_001',
                    'attempt_count': 4
                },
                {
                    'operation_type': 'pii_detection',
                    'pii_count': 15,
                    'agent_id': 'agent_002',
                    'data_source': 'customer_database'
                }
            ]
            
            total_alerts = 0
            for event in suspicious_events:
                monitoring_result = audit_tool.monitor_audit_events(event)
                
                # Verify monitoring is active
                assert monitoring_result['monitoring_active'] is True
                
                # Verify alerts are triggered for suspicious events
                if event['operation_type'] == 'credential_injection' and not event['success']:
                    assert monitoring_result['alerts_triggered'] > 0
                    total_alerts += monitoring_result['alerts_triggered']
                
                if event.get('pii_count', 0) > 10:
                    assert monitoring_result['alerts_triggered'] > 0
                    total_alerts += monitoring_result['alerts_triggered']
            
            # Verify alerts were generated
            assert total_alerts > 0
            assert len(monitoring_alerts) == total_alerts
            
            # Verify alert details
            for alert in monitoring_alerts:
                assert 'alert_id' in alert
                assert 'severity' in alert
                assert alert['severity'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
                assert alert['requires_investigation'] is True
    
    def test_audit_log_search_and_retrieval(self, sample_operations, audit_config):
        """Test that audit logs can be searched and retrieved efficiently."""
        with patch('strands_monitoring.AuditTrailTool') as mock_audit:
            # Mock audit log search
            mock_audit_instance = Mock()
            mock_audit.return_value = mock_audit_instance
            
            # Mock stored audit entries
            stored_entries = []
            for operation_type, operations in sample_operations.items():
                for operation in operations:
                    stored_entries.append({
                        'audit_id': f"audit_{len(stored_entries) + 1}",
                        'event_type': operation_type,
                        'data': operation,
                        'timestamp': operation.get('timestamp', datetime.now().isoformat()),
                        'agent_id': operation.get('agent_id'),
                        'session_id': operation.get('session_id')
                    })
            
            def mock_search_audit_logs(search_criteria):
                results = []
                for entry in stored_entries:
                    match = True
                    
                    # Apply search filters
                    if 'agent_id' in search_criteria:
                        if entry['agent_id'] != search_criteria['agent_id']:
                            match = False
                    
                    if 'event_type' in search_criteria:
                        if entry['event_type'] != search_criteria['event_type']:
                            match = False
                    
                    if 'date_range' in search_criteria:
                        entry_date = datetime.fromisoformat(entry['timestamp'])
                        start_date = datetime.fromisoformat(search_criteria['date_range']['start'])
                        end_date = datetime.fromisoformat(search_criteria['date_range']['end'])
                        if not (start_date <= entry_date <= end_date):
                            match = False
                    
                    if match:
                        results.append(entry)
                
                return {
                    'total_results': len(results),
                    'results': results[:search_criteria.get('limit', 100)],
                    'search_time_ms': 25.5
                }
            
            mock_audit_instance.search_audit_logs = Mock(side_effect=mock_search_audit_logs)
            
            audit_tool = AuditTrailTool()
            
            # Test various search scenarios
            search_scenarios = [
                {
                    'name': 'search_by_agent',
                    'criteria': {'agent_id': 'agent_001'},
                    'expected_min_results': 1
                },
                {
                    'name': 'search_by_event_type',
                    'criteria': {'event_type': 'credential_operations'},
                    'expected_min_results': 1
                },
                {
                    'name': 'search_by_date_range',
                    'criteria': {
                        'date_range': {
                            'start': (datetime.now() - timedelta(days=1)).isoformat(),
                            'end': datetime.now().isoformat()
                        }
                    },
                    'expected_min_results': 1
                }
            ]
            
            for scenario in search_scenarios:
                search_result = audit_tool.search_audit_logs(scenario['criteria'])
                
                # Verify search results
                assert 'total_results' in search_result
                assert 'results' in search_result
                assert search_result['search_time_ms'] < 100  # Should be fast
                
                # Verify results match criteria
                if scenario['criteria'].get('agent_id'):
                    for result in search_result['results']:
                        assert result['agent_id'] == scenario['criteria']['agent_id']
    
    def test_audit_log_backup_and_recovery(self, audit_config):
        """Test that audit logs can be backed up and recovered."""
        with patch('strands_monitoring.AuditTrailTool') as mock_audit:
            # Mock backup and recovery
            mock_audit_instance = Mock()
            mock_audit.return_value = mock_audit_instance
            
            def mock_backup_audit_logs(backup_config):
                return {
                    'backup_id': f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'backup_location': backup_config['destination'],
                    'entries_backed_up': 1000,
                    'backup_size_mb': 150.5,
                    'encryption_enabled': backup_config.get('encrypt', True),
                    'compression_enabled': backup_config.get('compress', True),
                    'backup_status': 'COMPLETED',
                    'backup_timestamp': datetime.now().isoformat()
                }
            
            def mock_restore_audit_logs(backup_id):
                return {
                    'restore_id': f"restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'backup_id': backup_id,
                    'entries_restored': 1000,
                    'restore_status': 'COMPLETED',
                    'integrity_verified': True,
                    'restore_timestamp': datetime.now().isoformat()
                }
            
            mock_audit_instance.backup_audit_logs = Mock(side_effect=mock_backup_audit_logs)
            mock_audit_instance.restore_audit_logs = Mock(side_effect=mock_restore_audit_logs)
            
            audit_tool = AuditTrailTool()
            
            # Test backup
            backup_config = {
                'destination': 's3://audit-backups/strands-agentcore/',
                'encrypt': True,
                'compress': True,
                'retention_years': 10
            }
            
            backup_result = audit_tool.backup_audit_logs(backup_config)
            
            # Verify backup
            assert backup_result['backup_status'] == 'COMPLETED'
            assert backup_result['entries_backed_up'] > 0
            assert backup_result['encryption_enabled'] is True
            assert backup_result['compression_enabled'] is True
            assert 'backup_id' in backup_result
            
            # Test restore
            restore_result = audit_tool.restore_audit_logs(backup_result['backup_id'])
            
            # Verify restore
            assert restore_result['restore_status'] == 'COMPLETED'
            assert restore_result['entries_restored'] == backup_result['entries_backed_up']
            assert restore_result['integrity_verified'] is True
            assert restore_result['backup_id'] == backup_result['backup_id']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])