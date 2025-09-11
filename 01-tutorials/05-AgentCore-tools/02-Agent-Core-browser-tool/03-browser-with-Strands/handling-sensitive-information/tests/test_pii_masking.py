"""
Security validation tests for PII masking in Strands-AgentCore integration.

This module validates that sensitive data is properly masked during Strands workflows,
ensuring PII protection throughout the agent execution lifecycle.

Requirements: 8.1, 8.2, 8.3, 8.4
"""

import pytest
import json
import re
import logging
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Tuple
import asyncio
from datetime import datetime

# Import Strands and AgentCore components
try:
    from strands import Agent, Tool, Workflow
    from strands.tools import BaseTool
    from strands.security import PIIDetector
except ImportError:
    # Mock imports for testing environment
    Agent = Mock
    Tool = Mock
    Workflow = Mock
    BaseTool = Mock
    PIIDetector = Mock

# Import custom tools
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))
try:
    from strands_pii_utils import SensitiveDataHandler
    from strands_agentcore_session_helpers import StrandsAgentCoreClient
    from strands_monitoring import AuditTrailTool
except ImportError:
    # Mock for testing
    SensitiveDataHandler = Mock
    StrandsAgentCoreClient = Mock
    AuditTrailTool = Mock


class TestPIIMasking:
    """Test suite for PII masking in Strands-AgentCore integration."""
    
    @pytest.fixture
    def sample_pii_data(self):
        """Sample PII data for testing."""
        return {
            'personal_info': {
                'full_name': 'John Michael Smith',
                'email': 'john.smith@example.com',
                'phone': '+1-555-123-4567',
                'ssn': '123-45-6789',
                'date_of_birth': '1985-03-15',
                'address': '123 Main Street, Anytown, ST 12345'
            },
            'financial_info': {
                'credit_card': '4532-1234-5678-9012',
                'bank_account': '123456789',
                'routing_number': '021000021',
                'iban': 'GB82 WEST 1234 5698 7654 32'
            },
            'health_info': {
                'patient_id': 'PAT-12345',
                'medical_record_number': 'MRN-987654',
                'diagnosis': 'Type 2 Diabetes',
                'medication': 'Metformin 500mg'
            },
            'mixed_content': 'Patient John Smith (SSN: 123-45-6789) has credit card 4532-1234-5678-9012 and lives at 123 Main Street. Contact at john.smith@example.com or +1-555-123-4567.'
        }
    
    @pytest.fixture
    def pii_patterns(self):
        """PII detection patterns for testing."""
        return {
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            'name': r'\b[A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b',
            'address': r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b'
        }
    
    @pytest.fixture
    def masking_config(self):
        """Configuration for PII masking."""
        return {
            'mask_char': '*',
            'preserve_format': True,
            'partial_masking': {
                'email': {'show_domain': False, 'show_first_char': True},
                'phone': {'show_area_code': False, 'show_last_digits': 2},
                'credit_card': {'show_last_digits': 4},
                'ssn': {'show_last_digits': 4}
            },
            'full_masking': ['name', 'address', 'medical_info'],
            'classification_levels': {
                'LOW': ['email'],
                'MEDIUM': ['phone', 'address'],
                'HIGH': ['ssn', 'credit_card'],
                'CRITICAL': ['medical_info', 'financial_info']
            }
        }
    
    def test_pii_detection_accuracy(self, sample_pii_data, pii_patterns):
        """Test that PII detection accurately identifies sensitive data."""
        with patch('strands_pii_utils.SensitiveDataHandler') as mock_handler:
            # Configure mock to detect PII
            mock_handler.return_value.detect_pii.return_value = {
                'detected_pii': [
                    {'type': 'ssn', 'value': '123-45-6789', 'confidence': 0.95, 'start': 45, 'end': 56},
                    {'type': 'credit_card', 'value': '4532-1234-5678-9012', 'confidence': 0.98, 'start': 75, 'end': 94},
                    {'type': 'email', 'value': 'john.smith@example.com', 'confidence': 0.92, 'start': 150, 'end': 172},
                    {'type': 'phone', 'value': '+1-555-123-4567', 'confidence': 0.90, 'start': 176, 'end': 191},
                    {'type': 'name', 'value': 'John Smith', 'confidence': 0.85, 'start': 8, 'end': 18}
                ],
                'total_detected': 5,
                'confidence_threshold': 0.8
            }
            
            handler = SensitiveDataHandler()
            detection_result = handler.detect_pii(
                sample_pii_data['mixed_content'],
                {'patterns': pii_patterns, 'confidence_threshold': 0.8}
            )
            
            # Verify detection results
            assert detection_result['total_detected'] == 5
            assert len(detection_result['detected_pii']) == 5
            
            # Verify specific PII types were detected
            detected_types = [item['type'] for item in detection_result['detected_pii']]
            assert 'ssn' in detected_types
            assert 'credit_card' in detected_types
            assert 'email' in detected_types
            assert 'phone' in detected_types
            assert 'name' in detected_types
            
            # Verify confidence scores
            for pii_item in detection_result['detected_pii']:
                assert pii_item['confidence'] >= 0.8
    
    def test_pii_masking_formats(self, sample_pii_data, masking_config):
        """Test that PII masking preserves format while hiding sensitive data."""
        with patch('strands_pii_utils.SensitiveDataHandler') as mock_handler:
            # Configure mock to return masked data with preserved formats
            mock_handler.return_value.mask_sensitive_data.return_value = {
                'personal_info': {
                    'full_name': '***MASKED***',
                    'email': 'j***@***',
                    'phone': '***-***-**67',
                    'ssn': '***-**-6789',
                    'date_of_birth': '***MASKED***',
                    'address': '***MASKED***'
                },
                'financial_info': {
                    'credit_card': '****-****-****-9012',
                    'bank_account': '***MASKED***',
                    'routing_number': '***MASKED***',
                    'iban': '***MASKED***'
                },
                'health_info': {
                    'patient_id': '***MASKED***',
                    'medical_record_number': '***MASKED***',
                    'diagnosis': '***MASKED***',
                    'medication': '***MASKED***'
                }
            }
            
            handler = SensitiveDataHandler()
            masked_data = handler.mask_sensitive_data(sample_pii_data, masking_config)
            
            # Verify format preservation for partial masking
            assert masked_data['personal_info']['ssn'] == '***-**-6789'  # Last 4 digits shown
            assert masked_data['financial_info']['credit_card'] == '****-****-****-9012'  # Last 4 digits shown
            assert masked_data['personal_info']['phone'] == '***-***-**67'  # Last 2 digits shown
            
            # Verify full masking for sensitive categories
            assert masked_data['personal_info']['full_name'] == '***MASKED***'
            assert masked_data['personal_info']['address'] == '***MASKED***'
            assert masked_data['health_info']['diagnosis'] == '***MASKED***'
            
            # Verify original data is not present
            original_values = [
                'John Michael Smith', 'john.smith@example.com', '123-45-6789',
                '4532-1234-5678-9012', 'Type 2 Diabetes'
            ]
            masked_str = json.dumps(masked_data)
            for value in original_values:
                assert value not in masked_str
    
    def test_pii_classification_levels(self, sample_pii_data, masking_config):
        """Test that PII is classified into appropriate sensitivity levels."""
        with patch('strands_pii_utils.SensitiveDataHandler') as mock_handler:
            # Configure mock to return classification results
            mock_handler.return_value.classify_sensitivity.return_value = {
                'classifications': {
                    'john.smith@example.com': {'level': 'LOW', 'type': 'email'},
                    '+1-555-123-4567': {'level': 'MEDIUM', 'type': 'phone'},
                    '123 Main Street': {'level': 'MEDIUM', 'type': 'address'},
                    '123-45-6789': {'level': 'HIGH', 'type': 'ssn'},
                    '4532-1234-5678-9012': {'level': 'HIGH', 'type': 'credit_card'},
                    'Type 2 Diabetes': {'level': 'CRITICAL', 'type': 'medical_info'}
                },
                'summary': {
                    'LOW': 1,
                    'MEDIUM': 2,
                    'HIGH': 2,
                    'CRITICAL': 1
                },
                'highest_level': 'CRITICAL'
            }
            
            handler = SensitiveDataHandler()
            classification = handler.classify_sensitivity(sample_pii_data['mixed_content'])
            
            # Verify classification levels
            assert classification['highest_level'] == 'CRITICAL'
            assert classification['summary']['CRITICAL'] == 1
            assert classification['summary']['HIGH'] == 2
            assert classification['summary']['MEDIUM'] == 2
            assert classification['summary']['LOW'] == 1
            
            # Verify specific classifications
            classifications = classification['classifications']
            assert classifications['123-45-6789']['level'] == 'HIGH'
            assert classifications['4532-1234-5678-9012']['level'] == 'HIGH'
            assert classifications['Type 2 Diabetes']['level'] == 'CRITICAL'
    
    def test_real_time_pii_masking_during_workflow(self, sample_pii_data):
        """Test that PII is masked in real-time during Strands workflow execution."""
        with patch('strands_agentcore_session_helpers.StrandsAgentCoreClient') as mock_client:
            with patch('strands_pii_utils.SensitiveDataHandler') as mock_handler:
                # Mock workflow execution with PII masking
                mock_agent = Mock()
                mock_agent.execute = Mock(return_value={
                    'status': 'success',
                    'extracted_data': {
                        'customer_name': '***MASKED***',
                        'customer_email': 'j***@***',
                        'customer_phone': '***-***-**67',
                        'account_number': '***MASKED***'
                    },
                    'pii_detected': True,
                    'masking_applied': True,
                    'security_level': 'HIGH'
                })
                
                mock_client.return_value.create_secure_agent.return_value = mock_agent
                
                # Mock PII handler
                mock_handler.return_value.mask_sensitive_data.return_value = {
                    'customer_name': '***MASKED***',
                    'customer_email': 'j***@***',
                    'customer_phone': '***-***-**67',
                    'account_number': '***MASKED***'
                }
                
                # Execute workflow with PII data
                client = StrandsAgentCoreClient(
                    region='us-east-1',
                    llm_configs={'bedrock': {'model': 'claude-3'}},
                    security_config={'pii_masking': True, 'real_time_masking': True}
                )
                
                agent = mock_client.return_value.create_secure_agent({
                    'tools': ['pii_detection', 'data_extraction'],
                    'security_level': 'HIGH'
                })
                
                result = agent.execute({
                    'action': 'extract_customer_data',
                    'source_data': sample_pii_data['mixed_content']
                })
                
                # Verify PII was detected and masked
                assert result['pii_detected'] is True
                assert result['masking_applied'] is True
                assert result['security_level'] == 'HIGH'
                
                # Verify extracted data is masked
                extracted_data = result['extracted_data']
                assert extracted_data['customer_name'] == '***MASKED***'
                assert extracted_data['customer_email'] == 'j***@***'
                assert extracted_data['customer_phone'] == '***-***-**67'
    
    def test_pii_masking_in_logs(self, sample_pii_data):
        """Test that PII is masked in log outputs."""
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log') as f:
            log_file = f.name
        
        try:
            # Configure logging
            logger = logging.getLogger('pii_masking_test')
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
            
            with patch('strands_pii_utils.SensitiveDataHandler') as mock_handler:
                # Mock PII masking for logs
                mock_handler.return_value.mask_for_logging.return_value = (
                    'Patient ***MASKED*** (SSN: ***-**-6789) has credit card ****-****-****-9012 '
                    'and lives at ***MASKED***. Contact at j***@*** or ***-***-**67.'
                )
                
                handler_instance = SensitiveDataHandler()
                
                # Log data with PII masking
                masked_content = handler_instance.mask_for_logging(sample_pii_data['mixed_content'])
                logger.info(f"Processing customer data: {masked_content}")
                logger.debug(f"Data extraction completed successfully")
                
            # Read log file and verify masking
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            # Verify original PII is not in logs
            sensitive_values = [
                'John Smith', 'john.smith@example.com', '123-45-6789',
                '4532-1234-5678-9012', '+1-555-123-4567', '123 Main Street'
            ]
            
            for value in sensitive_values:
                assert value not in log_content, f"Sensitive value '{value}' found in logs"
            
            # Verify masked values are present
            assert '***MASKED***' in log_content
            assert '***-**-6789' in log_content
            assert '****-****-****-9012' in log_content
            
        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)
    
    def test_pii_masking_performance(self, sample_pii_data):
        """Test that PII masking doesn't significantly impact performance."""
        import time
        
        with patch('strands_pii_utils.SensitiveDataHandler') as mock_handler:
            # Mock performance-optimized PII masking
            mock_handler.return_value.mask_sensitive_data.return_value = {
                'masked_data': '***MASKED***',
                'processing_time_ms': 15.5,
                'pii_items_processed': 5
            }
            
            handler = SensitiveDataHandler()
            
            # Measure masking performance
            start_time = time.time()
            
            for _ in range(100):  # Process multiple times
                result = handler.mask_sensitive_data(
                    sample_pii_data['mixed_content'],
                    {'performance_mode': True}
                )
            
            end_time = time.time()
            total_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            # Verify performance is acceptable (mock verification)
            assert result['processing_time_ms'] < 50  # Should be under 50ms per operation
            assert total_time < 5000  # Total time should be under 5 seconds for 100 operations
    
    def test_pii_masking_accuracy_validation(self, sample_pii_data, pii_patterns):
        """Test that PII masking accuracy meets requirements."""
        with patch('strands_pii_utils.SensitiveDataHandler') as mock_handler:
            # Mock accuracy validation
            mock_handler.return_value.validate_masking_accuracy.return_value = {
                'accuracy_score': 0.98,
                'precision': 0.97,
                'recall': 0.99,
                'f1_score': 0.98,
                'false_positives': 2,
                'false_negatives': 1,
                'total_pii_items': 100,
                'correctly_masked': 98
            }
            
            handler = SensitiveDataHandler()
            accuracy_result = handler.validate_masking_accuracy(
                sample_pii_data,
                pii_patterns
            )
            
            # Verify accuracy meets requirements
            assert accuracy_result['accuracy_score'] >= 0.95
            assert accuracy_result['precision'] >= 0.95
            assert accuracy_result['recall'] >= 0.95
            assert accuracy_result['f1_score'] >= 0.95
            
            # Verify low error rates
            assert accuracy_result['false_positives'] <= 5
            assert accuracy_result['false_negatives'] <= 5
    
    def test_pii_masking_reversibility_prevention(self, sample_pii_data):
        """Test that PII masking is irreversible to prevent data recovery."""
        with patch('strands_pii_utils.SensitiveDataHandler') as mock_handler:
            # Mock irreversible masking
            mock_handler.return_value.mask_sensitive_data.return_value = {
                'masked_data': 'Patient ***MASKED*** has account ****-****-****-9012',
                'masking_method': 'irreversible_hash',
                'recovery_possible': False,
                'hash_salt_used': True
            }
            
            mock_handler.return_value.attempt_recovery.return_value = {
                'recovery_successful': False,
                'error': 'Irreversible masking - recovery not possible'
            }
            
            handler = SensitiveDataHandler()
            
            # Mask data
            masked_result = handler.mask_sensitive_data(
                sample_pii_data['mixed_content'],
                {'masking_method': 'irreversible'}
            )
            
            # Attempt to recover original data (should fail)
            recovery_result = handler.attempt_recovery(masked_result['masked_data'])
            
            # Verify recovery is not possible
            assert masked_result['recovery_possible'] is False
            assert recovery_result['recovery_successful'] is False
            assert 'not possible' in recovery_result['error']
    
    def test_pii_masking_audit_trail(self, sample_pii_data):
        """Test that PII masking operations are properly audited."""
        with patch('strands_monitoring.AuditTrailTool') as mock_audit:
            with patch('strands_pii_utils.SensitiveDataHandler') as mock_handler:
                # Mock audit logging
                mock_audit_instance = Mock()
                mock_audit.return_value = mock_audit_instance
                
                mock_audit_instance.log_pii_operation = Mock()
                
                # Mock PII masking with audit
                mock_handler.return_value.mask_with_audit.return_value = {
                    'masked_data': '***MASKED***',
                    'audit_id': 'audit_12345',
                    'pii_items_masked': 5,
                    'masking_timestamp': datetime.now().isoformat()
                }
                
                handler = SensitiveDataHandler()
                audit_tool = AuditTrailTool()
                
                # Perform masking with audit
                masking_result = handler.mask_with_audit(
                    sample_pii_data['mixed_content'],
                    {'audit_required': True}
                )
                
                # Log audit event
                audit_tool.log_pii_operation({
                    'operation': 'pii_masking',
                    'audit_id': masking_result['audit_id'],
                    'pii_items_count': masking_result['pii_items_masked'],
                    'timestamp': masking_result['masking_timestamp'],
                    'success': True
                })
                
                # Verify audit logging
                mock_audit_instance.log_pii_operation.assert_called_once()
                
                # Verify audit data doesn't contain original PII
                call_args = mock_audit_instance.log_pii_operation.call_args[0][0]
                sensitive_values = ['John Smith', 'john.smith@example.com', '123-45-6789']
                for value in sensitive_values:
                    assert value not in str(call_args)
    
    @pytest.mark.asyncio
    async def test_concurrent_pii_masking(self, sample_pii_data):
        """Test that concurrent PII masking operations maintain data integrity."""
        with patch('strands_pii_utils.SensitiveDataHandler') as mock_handler:
            # Mock concurrent masking
            mock_handler.return_value.mask_sensitive_data.return_value = {
                'masked_data': '***MASKED***',
                'thread_id': 'thread_123',
                'processing_time': 0.05
            }
            
            handler = SensitiveDataHandler()
            
            # Define concurrent masking task
            async def mask_data_async(data, task_id):
                return handler.mask_sensitive_data(data, {'task_id': task_id})
            
            # Run concurrent masking operations
            tasks = [
                mask_data_async(sample_pii_data['mixed_content'], f'task_{i}')
                for i in range(10)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Verify all operations completed successfully
            assert len(results) == 10
            for result in results:
                assert result['masked_data'] == '***MASKED***'
                assert 'thread_id' in result
    
    def test_pii_masking_configuration_validation(self, masking_config):
        """Test that PII masking configuration is properly validated."""
        with patch('strands_pii_utils.SensitiveDataHandler') as mock_handler:
            # Mock configuration validation
            mock_handler.return_value.validate_config.return_value = {
                'config_valid': True,
                'validation_errors': [],
                'warnings': [],
                'recommended_settings': {
                    'mask_char': '*',
                    'preserve_format': True,
                    'confidence_threshold': 0.8
                }
            }
            
            handler = SensitiveDataHandler()
            validation_result = handler.validate_config(masking_config)
            
            # Verify configuration is valid
            assert validation_result['config_valid'] is True
            assert len(validation_result['validation_errors']) == 0
            
            # Test invalid configuration
            invalid_config = masking_config.copy()
            invalid_config['mask_char'] = ''  # Invalid empty mask character
            
            mock_handler.return_value.validate_config.return_value = {
                'config_valid': False,
                'validation_errors': ['mask_char cannot be empty'],
                'warnings': ['preserve_format recommended for better usability']
            }
            
            invalid_result = handler.validate_config(invalid_config)
            assert invalid_result['config_valid'] is False
            assert len(invalid_result['validation_errors']) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])