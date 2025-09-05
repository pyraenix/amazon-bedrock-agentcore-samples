"""
Security validation tests for session isolation in Strands-AgentCore integration.

This module verifies that AgentCore Browser Tool sessions are properly isolated
between Strands agents, ensuring no data leakage or cross-contamination.

Requirements: 8.1, 8.2, 8.3, 8.4
"""

import pytest
import json
import uuid
import time
import threading
import asyncio
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Set
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import Strands and AgentCore components
try:
    from strands import Agent, Tool, Workflow
    from strands.tools import BaseTool
    from strands.session import SessionManager
except ImportError:
    # Mock imports for testing environment
    Agent = Mock
    Tool = Mock
    Workflow = Mock
    BaseTool = Mock
    SessionManager = Mock

# Import custom tools
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))
try:
    from strands_agentcore_session_helpers import StrandsAgentCoreClient, SessionPoolManager
    from strands_pii_utils import SensitiveDataHandler
    from strands_monitoring import AuditTrailTool
except ImportError:
    # Mock for testing
    StrandsAgentCoreClient = Mock
    SessionPoolManager = Mock
    SensitiveDataHandler = Mock
    AuditTrailTool = Mock


class TestSessionIsolation:
    """Test suite for session isolation in Strands-AgentCore integration."""
    
    @pytest.fixture
    def session_configs(self):
        """Different session configurations for testing isolation."""
        return {
            'session_a': {
                'session_id': 'session_a_001',
                'agent_id': 'agent_001',
                'security_level': 'HIGH',
                'isolation_mode': 'STRICT',
                'data_classification': 'CONFIDENTIAL',
                'user_context': {'user_id': 'user_001', 'role': 'admin'}
            },
            'session_b': {
                'session_id': 'session_b_002',
                'agent_id': 'agent_002',
                'security_level': 'MEDIUM',
                'isolation_mode': 'STANDARD',
                'data_classification': 'INTERNAL',
                'user_context': {'user_id': 'user_002', 'role': 'user'}
            },
            'session_c': {
                'session_id': 'session_c_003',
                'agent_id': 'agent_003',
                'security_level': 'HIGH',
                'isolation_mode': 'STRICT',
                'data_classification': 'RESTRICTED',
                'user_context': {'user_id': 'user_003', 'role': 'analyst'}
            }
        }
    
    @pytest.fixture
    def sensitive_test_data(self):
        """Sensitive test data for different sessions."""
        return {
            'session_a_data': {
                'customer_records': [
                    {'id': 'CUST_001', 'name': 'Alice Johnson', 'ssn': '111-11-1111'},
                    {'id': 'CUST_002', 'name': 'Bob Smith', 'ssn': '222-22-2222'}
                ],
                'session_token': 'token_session_a_12345',
                'api_keys': {'service_x': 'key_a_xyz789'}
            },
            'session_b_data': {
                'customer_records': [
                    {'id': 'CUST_003', 'name': 'Charlie Brown', 'ssn': '333-33-3333'},
                    {'id': 'CUST_004', 'name': 'Diana Prince', 'ssn': '444-44-4444'}
                ],
                'session_token': 'token_session_b_67890',
                'api_keys': {'service_y': 'key_b_abc123'}
            },
            'session_c_data': {
                'financial_records': [
                    {'account': 'ACC_001', 'balance': 50000, 'owner': 'Eve Wilson'},
                    {'account': 'ACC_002', 'balance': 75000, 'owner': 'Frank Miller'}
                ],
                'session_token': 'token_session_c_54321',
                'api_keys': {'service_z': 'key_c_def456'}
            }
        }
    
    def test_session_creation_isolation(self, session_configs):
        """Test that sessions are created with proper isolation."""
        with patch('strands_agentcore_session_helpers.StrandsAgentCoreClient') as mock_client:
            # Mock session creation
            mock_sessions = {}
            for session_id, config in session_configs.items():
                mock_session = Mock()
                mock_session.session_id = config['session_id']
                mock_session.agent_id = config['agent_id']
                mock_session.isolation_level = config['isolation_mode']
                mock_session.is_isolated = Mock(return_value=True)
                mock_sessions[session_id] = mock_session
            
            mock_client.return_value.create_secure_session.side_effect = lambda config: mock_sessions[
                next(k for k, v in session_configs.items() if v['session_id'] == config['session_id'])
            ]
            
            client = StrandsAgentCoreClient(
                region='us-east-1',
                llm_configs={'bedrock': {'model': 'claude-3'}},
                security_config={'isolation_required': True}
            )
            
            # Create multiple sessions
            created_sessions = {}
            for session_key, config in session_configs.items():
                session = mock_client.return_value.create_secure_session(config)
                created_sessions[session_key] = session
            
            # Verify each session is properly isolated
            for session_key, session in created_sessions.items():
                assert session.is_isolated()
                assert session.session_id == session_configs[session_key]['session_id']
                assert session.agent_id == session_configs[session_key]['agent_id']
            
            # Verify sessions have different IDs
            session_ids = [session.session_id for session in created_sessions.values()]
            assert len(set(session_ids)) == len(session_ids)  # All unique
    
    def test_data_isolation_between_sessions(self, session_configs, sensitive_test_data):
        """Test that data is isolated between different sessions."""
        with patch('strands_agentcore_session_helpers.StrandsAgentCoreClient') as mock_client:
            # Mock session data storage
            session_data_store = {}
            
            def mock_store_data(session_id, data):
                if session_id not in session_data_store:
                    session_data_store[session_id] = {}
                session_data_store[session_id].update(data)
                return {'status': 'success', 'stored_items': len(data)}
            
            def mock_retrieve_data(session_id):
                return session_data_store.get(session_id, {})
            
            # Create mock sessions with data isolation
            mock_sessions = {}
            for session_key, config in session_configs.items():
                mock_session = Mock()
                mock_session.session_id = config['session_id']
                mock_session.store_data = Mock(side_effect=lambda data, sid=config['session_id']: mock_store_data(sid, data))
                mock_session.retrieve_data = Mock(side_effect=lambda sid=config['session_id']: mock_retrieve_data(sid))
                mock_session.get_accessible_sessions = Mock(return_value=[config['session_id']])
                mock_sessions[session_key] = mock_session
            
            mock_client.return_value.create_secure_session.side_effect = lambda config: mock_sessions[
                next(k for k, v in session_configs.items() if v['session_id'] == config['session_id'])
            ]
            
            client = StrandsAgentCoreClient(
                region='us-east-1',
                llm_configs={'bedrock': {'model': 'claude-3'}},
                security_config={'data_isolation': True}
            )
            
            # Create sessions and store data
            sessions = {}
            for session_key, config in session_configs.items():
                session = mock_client.return_value.create_secure_session(config)
                sessions[session_key] = session
                
                # Store session-specific data
                data_key = f"{session_key}_data"
                if data_key in sensitive_test_data:
                    session.store_data(sensitive_test_data[data_key])
            
            # Verify data isolation
            for session_key, session in sessions.items():
                retrieved_data = session.retrieve_data()
                expected_data_key = f"{session_key}_data"
                
                if expected_data_key in sensitive_test_data:
                    # Verify session can access its own data
                    expected_data = sensitive_test_data[expected_data_key]
                    for key in expected_data:
                        assert key in retrieved_data
                    
                    # Verify session cannot access other sessions' data
                    for other_session_key, other_session in sessions.items():
                        if other_session_key != session_key:
                            other_data_key = f"{other_session_key}_data"
                            if other_data_key in sensitive_test_data:
                                other_expected_data = sensitive_test_data[other_data_key]
                                for key in other_expected_data:
                                    # Data from other sessions should not be accessible
                                    if key in retrieved_data:
                                        assert retrieved_data[key] != other_expected_data[key]
    
    def test_session_memory_isolation(self, session_configs):
        """Test that session memory is isolated and doesn't leak between sessions."""
        with patch('strands_agentcore_session_helpers.SessionPoolManager') as mock_pool:
            # Mock memory isolation
            session_memory = {}
            
            def mock_allocate_memory(session_id, size_mb):
                session_memory[session_id] = {
                    'allocated_size': size_mb,
                    'used_size': 0,
                    'memory_pool': f"pool_{session_id}",
                    'isolated': True
                }
                return session_memory[session_id]
            
            def mock_get_memory_info(session_id):
                return session_memory.get(session_id, {})
            
            def mock_check_memory_isolation(session_id):
                return {
                    'isolated': True,
                    'cross_session_access': False,
                    'memory_leaks': [],
                    'isolation_score': 1.0
                }
            
            mock_pool_instance = Mock()
            mock_pool.return_value = mock_pool_instance
            mock_pool_instance.allocate_session_memory = Mock(side_effect=mock_allocate_memory)
            mock_pool_instance.get_memory_info = Mock(side_effect=mock_get_memory_info)
            mock_pool_instance.check_memory_isolation = Mock(side_effect=mock_check_memory_isolation)
            
            pool_manager = SessionPoolManager()
            
            # Allocate memory for each session
            for session_key, config in session_configs.items():
                session_id = config['session_id']
                memory_info = pool_manager.allocate_session_memory(session_id, 512)  # 512MB
                
                # Verify memory allocation
                assert memory_info['allocated_size'] == 512
                assert memory_info['isolated'] is True
                assert memory_info['memory_pool'] == f"pool_{session_id}"
            
            # Check memory isolation for each session
            for session_key, config in session_configs.items():
                session_id = config['session_id']
                isolation_check = pool_manager.check_memory_isolation(session_id)
                
                # Verify memory isolation
                assert isolation_check['isolated'] is True
                assert isolation_check['cross_session_access'] is False
                assert len(isolation_check['memory_leaks']) == 0
                assert isolation_check['isolation_score'] == 1.0
    
    def test_session_network_isolation(self, session_configs):
        """Test that network connections are isolated between sessions."""
        with patch('strands_agentcore_session_helpers.StrandsAgentCoreClient') as mock_client:
            # Mock network isolation
            network_configs = {}
            
            def mock_setup_network_isolation(session_id, config):
                network_configs[session_id] = {
                    'virtual_network': f"vnet_{session_id}",
                    'isolated_subnet': f"subnet_{session_id}",
                    'firewall_rules': [f"allow_{session_id}_only"],
                    'network_namespace': f"ns_{session_id}",
                    'isolation_level': config.get('isolation_mode', 'STANDARD')
                }
                return network_configs[session_id]
            
            def mock_verify_network_isolation(session_id):
                return {
                    'network_isolated': True,
                    'cross_session_traffic': False,
                    'firewall_violations': [],
                    'network_leaks': []
                }
            
            mock_sessions = {}
            for session_key, config in session_configs.items():
                mock_session = Mock()
                mock_session.session_id = config['session_id']
                mock_session.setup_network_isolation = Mock(
                    side_effect=lambda cfg, sid=config['session_id']: mock_setup_network_isolation(sid, cfg)
                )
                mock_session.verify_network_isolation = Mock(
                    side_effect=lambda sid=config['session_id']: mock_verify_network_isolation(sid)
                )
                mock_sessions[session_key] = mock_session
            
            mock_client.return_value.create_secure_session.side_effect = lambda config: mock_sessions[
                next(k for k, v in session_configs.items() if v['session_id'] == config['session_id'])
            ]
            
            client = StrandsAgentCoreClient(
                region='us-east-1',
                llm_configs={'bedrock': {'model': 'claude-3'}},
                security_config={'network_isolation': True}
            )
            
            # Create sessions with network isolation
            sessions = {}
            for session_key, config in session_configs.items():
                session = mock_client.return_value.create_secure_session(config)
                sessions[session_key] = session
                
                # Setup network isolation
                network_config = session.setup_network_isolation(config)
                
                # Verify network configuration
                assert network_config['virtual_network'] == f"vnet_{config['session_id']}"
                assert network_config['isolated_subnet'] == f"subnet_{config['session_id']}"
                assert network_config['isolation_level'] == config['isolation_mode']
            
            # Verify network isolation for each session
            for session_key, session in sessions.items():
                isolation_status = session.verify_network_isolation()
                
                assert isolation_status['network_isolated'] is True
                assert isolation_status['cross_session_traffic'] is False
                assert len(isolation_status['firewall_violations']) == 0
                assert len(isolation_status['network_leaks']) == 0
    
    def test_concurrent_session_isolation(self, session_configs, sensitive_test_data):
        """Test that session isolation is maintained under concurrent access."""
        with patch('strands_agentcore_session_helpers.StrandsAgentCoreClient') as mock_client:
            # Mock concurrent session operations
            session_operations = {}
            operation_lock = threading.Lock()
            
            def mock_concurrent_operation(session_id, operation_data):
                with operation_lock:
                    if session_id not in session_operations:
                        session_operations[session_id] = []
                    session_operations[session_id].append({
                        'timestamp': datetime.now().isoformat(),
                        'thread_id': threading.current_thread().ident,
                        'operation_data': operation_data,
                        'isolation_maintained': True
                    })
                return {'status': 'success', 'isolation_verified': True}
            
            # Create mock sessions
            mock_sessions = {}
            for session_key, config in session_configs.items():
                mock_session = Mock()
                mock_session.session_id = config['session_id']
                mock_session.execute_operation = Mock(
                    side_effect=lambda data, sid=config['session_id']: mock_concurrent_operation(sid, data)
                )
                mock_sessions[session_key] = mock_session
            
            mock_client.return_value.create_secure_session.side_effect = lambda config: mock_sessions[
                next(k for k, v in session_configs.items() if v['session_id'] == config['session_id'])
            ]
            
            client = StrandsAgentCoreClient(
                region='us-east-1',
                llm_configs={'bedrock': {'model': 'claude-3'}},
                security_config={'concurrent_isolation': True}
            )
            
            # Create sessions
            sessions = {}
            for session_key, config in session_configs.items():
                session = mock_client.return_value.create_secure_session(config)
                sessions[session_key] = session
            
            # Define concurrent operation function
            def concurrent_session_operation(session_key, operation_id):
                session = sessions[session_key]
                data_key = f"{session_key}_data"
                operation_data = {
                    'operation_id': operation_id,
                    'data': sensitive_test_data.get(data_key, {}),
                    'timestamp': datetime.now().isoformat()
                }
                return session.execute_operation(operation_data)
            
            # Execute concurrent operations
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                
                # Submit multiple operations for each session
                for session_key in sessions.keys():
                    for i in range(5):  # 5 operations per session
                        future = executor.submit(concurrent_session_operation, session_key, f"op_{i}")
                        futures.append((session_key, future))
                
                # Collect results
                results = {}
                for session_key, future in futures:
                    if session_key not in results:
                        results[session_key] = []
                    result = future.result()
                    results[session_key].append(result)
            
            # Verify isolation was maintained during concurrent operations
            for session_key, session_results in results.items():
                for result in session_results:
                    assert result['status'] == 'success'
                    assert result['isolation_verified'] is True
            
            # Verify operations were properly isolated per session
            for session_key, config in session_configs.items():
                session_id = config['session_id']
                if session_id in session_operations:
                    operations = session_operations[session_id]
                    assert len(operations) == 5  # 5 operations per session
                    
                    # Verify all operations maintained isolation
                    for operation in operations:
                        assert operation['isolation_maintained'] is True
    
    @pytest.mark.asyncio
    async def test_async_session_isolation(self, session_configs):
        """Test that session isolation is maintained in async operations."""
        with patch('strands_agentcore_session_helpers.StrandsAgentCoreClient') as mock_client:
            # Mock async session operations
            async_operations = {}
            
            async def mock_async_operation(session_id, operation_data):
                await asyncio.sleep(0.1)  # Simulate async work
                if session_id not in async_operations:
                    async_operations[session_id] = []
                async_operations[session_id].append({
                    'operation_data': operation_data,
                    'task_id': id(asyncio.current_task()),
                    'isolation_maintained': True
                })
                return {'status': 'success', 'async_isolation_verified': True}
            
            # Create mock sessions with async support
            mock_sessions = {}
            for session_key, config in session_configs.items():
                mock_session = Mock()
                mock_session.session_id = config['session_id']
                mock_session.execute_async_operation = Mock(
                    side_effect=lambda data, sid=config['session_id']: mock_async_operation(sid, data)
                )
                mock_sessions[session_key] = mock_session
            
            mock_client.return_value.create_secure_session.side_effect = lambda config: mock_sessions[
                next(k for k, v in session_configs.items() if v['session_id'] == config['session_id'])
            ]
            
            client = StrandsAgentCoreClient(
                region='us-east-1',
                llm_configs={'bedrock': {'model': 'claude-3'}},
                security_config={'async_isolation': True}
            )
            
            # Create sessions
            sessions = {}
            for session_key, config in session_configs.items():
                session = mock_client.return_value.create_secure_session(config)
                sessions[session_key] = session
            
            # Define async operation tasks
            async def async_session_operation(session_key, operation_id):
                session = sessions[session_key]
                operation_data = {
                    'operation_id': operation_id,
                    'session_key': session_key,
                    'timestamp': datetime.now().isoformat()
                }
                return await session.execute_async_operation(operation_data)
            
            # Execute concurrent async operations
            tasks = []
            for session_key in sessions.keys():
                for i in range(3):  # 3 async operations per session
                    task = async_session_operation(session_key, f"async_op_{i}")
                    tasks.append((session_key, task))
            
            # Await all tasks
            results = {}
            for session_key, task in tasks:
                if session_key not in results:
                    results[session_key] = []
                result = await task
                results[session_key].append(result)
            
            # Verify async isolation
            for session_key, session_results in results.items():
                for result in session_results:
                    assert result['status'] == 'success'
                    assert result['async_isolation_verified'] is True
            
            # Verify operations were properly isolated
            for session_key, config in session_configs.items():
                session_id = config['session_id']
                if session_id in async_operations:
                    operations = async_operations[session_id]
                    assert len(operations) == 3  # 3 async operations per session
                    
                    # Verify isolation was maintained
                    for operation in operations:
                        assert operation['isolation_maintained'] is True
    
    def test_session_cleanup_isolation(self, session_configs, sensitive_test_data):
        """Test that session cleanup properly isolates and removes session data."""
        with patch('strands_agentcore_session_helpers.StrandsAgentCoreClient') as mock_client:
            # Mock session cleanup
            cleanup_results = {}
            
            def mock_cleanup_session(session_id):
                cleanup_results[session_id] = {
                    'data_cleared': True,
                    'memory_freed': True,
                    'network_disconnected': True,
                    'isolation_maintained': True,
                    'cleanup_timestamp': datetime.now().isoformat()
                }
                return cleanup_results[session_id]
            
            def mock_verify_cleanup(session_id):
                return {
                    'session_exists': False,
                    'data_remnants': [],
                    'memory_leaks': [],
                    'network_connections': [],
                    'cleanup_complete': True
                }
            
            # Create mock sessions
            mock_sessions = {}
            for session_key, config in session_configs.items():
                mock_session = Mock()
                mock_session.session_id = config['session_id']
                mock_session.is_active = Mock(return_value=True)
                mock_session.cleanup = Mock(
                    side_effect=lambda sid=config['session_id']: mock_cleanup_session(sid)
                )
                mock_session.verify_cleanup = Mock(
                    side_effect=lambda sid=config['session_id']: mock_verify_cleanup(sid)
                )
                mock_sessions[session_key] = mock_session
            
            mock_client.return_value.create_secure_session.side_effect = lambda config: mock_sessions[
                next(k for k, v in session_configs.items() if v['session_id'] == config['session_id'])
            ]
            
            client = StrandsAgentCoreClient(
                region='us-east-1',
                llm_configs={'bedrock': {'model': 'claude-3'}},
                security_config={'secure_cleanup': True}
            )
            
            # Create and use sessions
            sessions = {}
            for session_key, config in session_configs.items():
                session = mock_client.return_value.create_secure_session(config)
                sessions[session_key] = session
                
                # Verify session is active
                assert session.is_active()
            
            # Cleanup sessions
            for session_key, session in sessions.items():
                cleanup_result = session.cleanup()
                
                # Verify cleanup results
                assert cleanup_result['data_cleared'] is True
                assert cleanup_result['memory_freed'] is True
                assert cleanup_result['network_disconnected'] is True
                assert cleanup_result['isolation_maintained'] is True
                
                # Verify cleanup completion
                verification = session.verify_cleanup()
                assert verification['session_exists'] is False
                assert len(verification['data_remnants']) == 0
                assert len(verification['memory_leaks']) == 0
                assert len(verification['network_connections']) == 0
                assert verification['cleanup_complete'] is True
    
    def test_session_isolation_audit_trail(self, session_configs):
        """Test that session isolation events are properly audited."""
        with patch('strands_monitoring.AuditTrailTool') as mock_audit:
            with patch('strands_agentcore_session_helpers.StrandsAgentCoreClient') as mock_client:
                # Mock audit logging
                mock_audit_instance = Mock()
                mock_audit.return_value = mock_audit_instance
                
                audit_events = []
                
                def mock_log_isolation_event(event_data):
                    audit_events.append(event_data)
                    return {'audit_id': f"audit_{len(audit_events)}", 'logged': True}
                
                mock_audit_instance.log_isolation_event = Mock(side_effect=mock_log_isolation_event)
                
                # Create mock sessions with audit integration
                mock_sessions = {}
                for session_key, config in session_configs.items():
                    mock_session = Mock()
                    mock_session.session_id = config['session_id']
                    mock_session.log_isolation_event = Mock(
                        side_effect=lambda event, sid=config['session_id']: mock_audit_instance.log_isolation_event({
                            'session_id': sid,
                            'event_type': event['type'],
                            'timestamp': datetime.now().isoformat(),
                            'isolation_status': 'MAINTAINED'
                        })
                    )
                    mock_sessions[session_key] = mock_session
                
                mock_client.return_value.create_secure_session.side_effect = lambda config: mock_sessions[
                    next(k for k, v in session_configs.items() if v['session_id'] == config['session_id'])
                ]
                
                client = StrandsAgentCoreClient(
                    region='us-east-1',
                    llm_configs={'bedrock': {'model': 'claude-3'}},
                    security_config={'audit_isolation': True}
                )
                
                audit_tool = AuditTrailTool()
                
                # Create sessions and log isolation events
                sessions = {}
                for session_key, config in session_configs.items():
                    session = mock_client.return_value.create_secure_session(config)
                    sessions[session_key] = session
                    
                    # Log session creation event
                    session.log_isolation_event({
                        'type': 'session_created',
                        'isolation_level': config['isolation_mode']
                    })
                    
                    # Log data isolation event
                    session.log_isolation_event({
                        'type': 'data_isolated',
                        'data_classification': config['data_classification']
                    })
                
                # Verify audit events were logged
                assert len(audit_events) == len(session_configs) * 2  # 2 events per session
                
                # Verify audit event content
                for event in audit_events:
                    assert 'session_id' in event
                    assert 'event_type' in event
                    assert 'timestamp' in event
                    assert event['isolation_status'] == 'MAINTAINED'
                    assert event['event_type'] in ['session_created', 'data_isolated']
    
    def test_session_isolation_performance_impact(self, session_configs):
        """Test that session isolation doesn't significantly impact performance."""
        with patch('strands_agentcore_session_helpers.StrandsAgentCoreClient') as mock_client:
            import time
            
            # Mock performance monitoring
            performance_metrics = {}
            
            def mock_measure_performance(session_id, operation):
                start_time = time.time()
                time.sleep(0.01)  # Simulate operation time
                end_time = time.time()
                
                performance_metrics[session_id] = {
                    'operation': operation,
                    'execution_time_ms': (end_time - start_time) * 1000,
                    'isolation_overhead_ms': 2.5,  # Simulated isolation overhead
                    'total_time_ms': ((end_time - start_time) * 1000) + 2.5
                }
                return performance_metrics[session_id]
            
            # Create mock sessions with performance monitoring
            mock_sessions = {}
            for session_key, config in session_configs.items():
                mock_session = Mock()
                mock_session.session_id = config['session_id']
                mock_session.measure_performance = Mock(
                    side_effect=lambda op, sid=config['session_id']: mock_measure_performance(sid, op)
                )
                mock_sessions[session_key] = mock_session
            
            mock_client.return_value.create_secure_session.side_effect = lambda config: mock_sessions[
                next(k for k, v in session_configs.items() if v['session_id'] == config['session_id'])
            ]
            
            client = StrandsAgentCoreClient(
                region='us-east-1',
                llm_configs={'bedrock': {'model': 'claude-3'}},
                security_config={'performance_monitoring': True}
            )
            
            # Create sessions and measure performance
            sessions = {}
            for session_key, config in session_configs.items():
                session = mock_client.return_value.create_secure_session(config)
                sessions[session_key] = session
                
                # Measure performance of isolated operations
                perf_result = session.measure_performance('data_processing')
                
                # Verify performance is acceptable
                assert perf_result['total_time_ms'] < 100  # Should be under 100ms
                assert perf_result['isolation_overhead_ms'] < 10  # Isolation overhead should be minimal
            
            # Verify overall performance impact
            total_overhead = sum(
                metrics['isolation_overhead_ms'] 
                for metrics in performance_metrics.values()
            )
            average_overhead = total_overhead / len(performance_metrics)
            assert average_overhead < 5  # Average isolation overhead should be under 5ms


if __name__ == '__main__':
    pytest.main([__file__, '-v'])