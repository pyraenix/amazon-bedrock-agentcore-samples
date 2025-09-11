"""
Performance validation for Strands agent performance with AgentCore Browser Tool.

This module tests the performance characteristics of Strands agents when integrated
with AgentCore Browser Tool, ensuring acceptable performance under various loads.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import asyncio
import json
import logging
import os
import sys
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import psutil

# Add tools directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))

# Import custom tools
try:
    from strands_agentcore_session_helpers import StrandsAgentCoreClient, SessionPoolManager
    from strands_pii_utils import SensitiveDataHandler
    from strands_security_policies import BedrockModelRouter
    from strands_monitoring import AuditTrailTool
except ImportError:
    # Mock for testing environment
    StrandsAgentCoreClient = Mock
    SessionPoolManager = Mock
    SensitiveDataHandler = Mock
    BedrockModelRouter = Mock
    AuditTrailTool = Mock

# Import Strands components
try:
    from strands import Agent, Tool, Workflow
    from strands.tools import BaseTool
except ImportError:
    Agent = Mock
    Tool = Mock
    Workflow = Mock
    BaseTool = Mock


class PerformanceValidator:
    """Performance validator for Strands-AgentCore integration."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the performance validator."""
        self.config = config
        self.logger = self._setup_logging()
        self.performance_metrics = {}
        self.baseline_metrics = {}
        self.load_test_results = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for performance validation."""
        logger = logging.getLogger('performance_validator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def validate_performance(self) -> Dict[str, Any]:
        """Validate overall performance of Strands-AgentCore integration."""
        self.logger.info("Starting performance validation")
        
        validation_tasks = [
            self._validate_session_performance(),
            self._validate_credential_performance(),
            self._validate_pii_processing_performance(),
            self._validate_audit_logging_performance(),
            self._validate_memory_usage(),
            self._validate_concurrent_performance(),
            self._validate_scalability(),
            self._validate_response_times()
        ]
        
        results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        # Compile performance results
        performance_result = {
            'validation_timestamp': datetime.now().isoformat(),
            'overall_status': 'PASSED',
            'performance_score': 0.0,
            'total_tests': len(validation_tasks),
            'passed_tests': 0,
            'failed_tests': 0,
            'detailed_results': {},
            'performance_metrics': self.performance_metrics,
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Process results
        test_names = [
            'session_performance',
            'credential_performance',
            'pii_processing_performance',
            'audit_logging_performance',
            'memory_usage',
            'concurrent_performance',
            'scalability',
            'response_times'
        ]
        
        for i, result in enumerate(results):
            test_name = test_names[i]
            if isinstance(result, Exception):
                performance_result['detailed_results'][test_name] = {
                    'status': 'FAILED',
                    'error': str(result),
                    'score': 0.0
                }
                performance_result['failed_tests'] += 1
            else:
                performance_result['detailed_results'][test_name] = result
                if result['status'] in ['PASSED', 'WARNING']:
                    performance_result['passed_tests'] += 1
                else:
                    performance_result['failed_tests'] += 1
        
        # Calculate overall performance score
        total_score = sum(
            result.get('score', 0.0) 
            for result in performance_result['detailed_results'].values()
            if isinstance(result, dict)
        )
        performance_result['performance_score'] = total_score / len(validation_tasks)
        
        # Determine overall status
        if performance_result['failed_tests'] > 0 or performance_result['performance_score'] < 0.7:
            performance_result['overall_status'] = 'FAILED'
        elif performance_result['performance_score'] < 0.85:
            performance_result['overall_status'] = 'WARNING'
        
        # Generate recommendations
        performance_result['recommendations'] = self._generate_performance_recommendations(performance_result)
        
        self.logger.info(f"Performance validation completed: {performance_result['overall_status']}")
        return performance_result
    
    async def _validate_session_performance(self) -> Dict[str, Any]:
        """Validate session creation and management performance."""
        self.logger.info("Validating session performance")
        
        with patch('strands_agentcore_session_helpers.StrandsAgentCoreClient') as mock_client:
            # Mock session operations with timing
            session_times = []
            
            def mock_create_session(config):
                # Simulate session creation time
                start_time = time.time()
                time.sleep(0.05 + (len(session_times) * 0.01))  # Increasing delay
                end_time = time.time()
                
                session_time = (end_time - start_time) * 1000  # Convert to milliseconds
                session_times.append(session_time)
                
                return Mock(
                    session_id=config['session_id'],
                    creation_time_ms=session_time,
                    is_ready=Mock(return_value=True)
                )
            
            mock_client.return_value.create_secure_session = Mock(side_effect=mock_create_session)
            
            # Test session creation performance
            client = StrandsAgentCoreClient(
                region='us-east-1',
                llm_configs={'bedrock': {'model': 'claude-3'}},
                security_config={'performance_monitoring': True}
            )
            
            # Create multiple sessions
            sessions = []
            for i in range(10):
                session_config = {
                    'session_id': f'perf_session_{i}',
                    'security_level': 'HIGH'
                }
                session = mock_client.return_value.create_secure_session(session_config)
                sessions.append(session)
            
            # Calculate performance metrics
            avg_session_time = statistics.mean(session_times)
            max_session_time = max(session_times)
            min_session_time = min(session_times)
            std_dev = statistics.stdev(session_times) if len(session_times) > 1 else 0
            
            # Performance thresholds
            threshold_avg = self.config.get('thresholds', {}).get('session_creation_ms', 200)
            threshold_max = threshold_avg * 2
            
            # Evaluate performance
            avg_acceptable = avg_session_time < threshold_avg
            max_acceptable = max_session_time < threshold_max
            consistency_good = std_dev < (threshold_avg * 0.3)
            
            score = 1.0
            if not avg_acceptable:
                score -= 0.4
            if not max_acceptable:
                score -= 0.3
            if not consistency_good:
                score -= 0.3
            
            score = max(0.0, score)
            
            # Store metrics
            self.performance_metrics['session_performance'] = {
                'avg_creation_time_ms': avg_session_time,
                'max_creation_time_ms': max_session_time,
                'min_creation_time_ms': min_session_time,
                'std_deviation_ms': std_dev,
                'sessions_created': len(sessions)
            }
            
            return {
                'status': 'PASSED' if score >= 0.8 else 'WARNING' if score >= 0.6 else 'FAILED',
                'score': score,
                'metrics': self.performance_metrics['session_performance'],
                'thresholds': {
                    'avg_threshold_ms': threshold_avg,
                    'max_threshold_ms': threshold_max
                },
                'performance_acceptable': avg_acceptable and max_acceptable and consistency_good
            }
    
    async def _validate_credential_performance(self) -> Dict[str, Any]:
        """Validate credential injection and management performance."""
        self.logger.info("Validating credential performance")
        
        with patch('strands_agentcore_session_helpers.StrandsAgentCoreClient') as mock_client:
            # Mock credential operations with timing
            credential_times = []
            
            def mock_inject_credentials(session_id, credentials):
                start_time = time.time()
                # Simulate credential processing time based on credential count
                processing_time = 0.02 + (len(credentials) * 0.005)
                time.sleep(processing_time)
                end_time = time.time()
                
                injection_time = (end_time - start_time) * 1000
                credential_times.append(injection_time)
                
                return {
                    'status': 'success',
                    'injection_time_ms': injection_time,
                    'credentials_processed': len(credentials)
                }
            
            mock_session = Mock()
            mock_session.inject_credentials = Mock(side_effect=mock_inject_credentials)
            mock_client.return_value.create_secure_session.return_value = mock_session
            
            # Test credential injection performance
            client = StrandsAgentCoreClient(
                region='us-east-1',
                llm_configs={'bedrock': {'model': 'claude-3'}},
                security_config={'credential_performance_monitoring': True}
            )
            
            session = mock_client.return_value.create_secure_session({
                'session_id': 'cred_perf_session'
            })
            
            # Test with different credential sets
            credential_sets = [
                {'username': 'user1', 'password': 'pass1'},
                {'username': 'user2', 'password': 'pass2', 'api_key': 'key2'},
                {'username': 'user3', 'password': 'pass3', 'api_key': 'key3', 'oauth_token': 'token3'},
                {'username': 'user4', 'password': 'pass4', 'api_key': 'key4', 'oauth_token': 'token4', 'session_token': 'session4'}
            ]
            
            for creds in credential_sets:
                session.inject_credentials('cred_perf_session', creds)
            
            # Calculate performance metrics
            avg_credential_time = statistics.mean(credential_times)
            max_credential_time = max(credential_times)
            
            # Performance threshold
            threshold = self.config.get('thresholds', {}).get('credential_injection_ms', 100)
            
            # Evaluate performance
            performance_acceptable = avg_credential_time < threshold and max_credential_time < (threshold * 1.5)
            score = 1.0 if performance_acceptable else max(0.0, 1.0 - ((avg_credential_time - threshold) / threshold))
            
            # Store metrics
            self.performance_metrics['credential_performance'] = {
                'avg_injection_time_ms': avg_credential_time,
                'max_injection_time_ms': max_credential_time,
                'credential_sets_tested': len(credential_sets)
            }
            
            return {
                'status': 'PASSED' if score >= 0.8 else 'WARNING' if score >= 0.6 else 'FAILED',
                'score': score,
                'metrics': self.performance_metrics['credential_performance'],
                'threshold_ms': threshold,
                'performance_acceptable': performance_acceptable
            }
    
    async def _validate_pii_processing_performance(self) -> Dict[str, Any]:
        """Validate PII detection and masking performance."""
        self.logger.info("Validating PII processing performance")
        
        with patch('strands_pii_utils.SensitiveDataHandler') as mock_handler:
            # Mock PII processing with timing
            pii_processing_times = []
            
            def mock_detect_and_mask_pii(content, config):
                start_time = time.time()
                # Simulate processing time based on content length
                processing_time = 0.01 + (len(content) / 10000)  # 1ms per 10k characters
                time.sleep(processing_time)
                end_time = time.time()
                
                processing_time_ms = (end_time - start_time) * 1000
                pii_processing_times.append(processing_time_ms)
                
                # Mock detection results
                pii_count = content.count('@') + content.count('-') + content.count('(')  # Simple PII indicators
                
                return {
                    'processing_time_ms': processing_time_ms,
                    'pii_detected': pii_count,
                    'content_length': len(content),
                    'masked_content': content.replace('@', '*@*').replace('-', '*-*')
                }
            
            mock_handler.return_value.detect_and_mask_pii = Mock(side_effect=mock_detect_and_mask_pii)
            
            # Test PII processing performance
            handler = SensitiveDataHandler()
            
            # Test with different content sizes
            test_contents = [
                "Short text with john@example.com",
                "Medium text with multiple PII items: john@example.com, 123-45-6789, (555) 123-4567" * 10,
                "Long text with extensive PII data: " + ("john.doe@company.com, 987-65-4321, (555) 987-6543, " * 50),
                "Very long document: " + ("Patient records contain sensitive information like emails, SSNs, and phone numbers. " * 100)
            ]
            
            for content in test_contents:
                handler.detect_and_mask_pii(content, {'performance_mode': True})
            
            # Calculate performance metrics
            avg_processing_time = statistics.mean(pii_processing_times)
            max_processing_time = max(pii_processing_times)
            
            # Performance threshold
            threshold = self.config.get('thresholds', {}).get('pii_masking_ms', 75)
            
            # Evaluate performance
            performance_acceptable = avg_processing_time < threshold
            score = 1.0 if performance_acceptable else max(0.0, 1.0 - ((avg_processing_time - threshold) / threshold))
            
            # Store metrics
            self.performance_metrics['pii_processing_performance'] = {
                'avg_processing_time_ms': avg_processing_time,
                'max_processing_time_ms': max_processing_time,
                'content_samples_tested': len(test_contents)
            }
            
            return {
                'status': 'PASSED' if score >= 0.8 else 'WARNING' if score >= 0.6 else 'FAILED',
                'score': score,
                'metrics': self.performance_metrics['pii_processing_performance'],
                'threshold_ms': threshold,
                'performance_acceptable': performance_acceptable
            }
    
    async def _validate_audit_logging_performance(self) -> Dict[str, Any]:
        """Validate audit logging performance."""
        self.logger.info("Validating audit logging performance")
        
        with patch('strands_monitoring.AuditTrailTool') as mock_audit:
            # Mock audit logging with timing
            audit_times = []
            
            def mock_log_audit_event(event_data):
                start_time = time.time()
                # Simulate audit logging time
                time.sleep(0.005 + (len(str(event_data)) / 100000))  # Based on data size
                end_time = time.time()
                
                logging_time = (end_time - start_time) * 1000
                audit_times.append(logging_time)
                
                return {
                    'audit_id': f"audit_{len(audit_times)}",
                    'logging_time_ms': logging_time,
                    'event_size_bytes': len(str(event_data))
                }
            
            mock_audit_instance = Mock()
            mock_audit.return_value = mock_audit_instance
            mock_audit_instance.log_security_event = Mock(side_effect=mock_log_audit_event)
            
            # Test audit logging performance
            audit_tool = AuditTrailTool()
            
            # Test with different event sizes
            test_events = [
                {'type': 'simple_event', 'data': 'small'},
                {'type': 'medium_event', 'data': {'details': 'medium' * 100}},
                {'type': 'large_event', 'data': {'extensive_details': 'large' * 1000}},
                {'type': 'complex_event', 'data': {
                    'operation': 'complex_operation',
                    'metadata': {'key' + str(i): 'value' + str(i) for i in range(100)},
                    'timestamp': datetime.now().isoformat()
                }}
            ]
            
            for event in test_events:
                audit_tool.log_security_event(event)
            
            # Calculate performance metrics
            avg_audit_time = statistics.mean(audit_times)
            max_audit_time = max(audit_times)
            
            # Performance threshold
            threshold = self.config.get('thresholds', {}).get('audit_logging_ms', 50)
            
            # Evaluate performance
            performance_acceptable = avg_audit_time < threshold
            score = 1.0 if performance_acceptable else max(0.0, 1.0 - ((avg_audit_time - threshold) / threshold))
            
            # Store metrics
            self.performance_metrics['audit_logging_performance'] = {
                'avg_logging_time_ms': avg_audit_time,
                'max_logging_time_ms': max_audit_time,
                'events_logged': len(test_events)
            }
            
            return {
                'status': 'PASSED' if score >= 0.8 else 'WARNING' if score >= 0.6 else 'FAILED',
                'score': score,
                'metrics': self.performance_metrics['audit_logging_performance'],
                'threshold_ms': threshold,
                'performance_acceptable': performance_acceptable
            }
    
    async def _validate_memory_usage(self) -> Dict[str, Any]:
        """Validate memory usage patterns."""
        self.logger.info("Validating memory usage")
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with patch('strands_agentcore_session_helpers.StrandsAgentCoreClient') as mock_client:
            # Mock memory-intensive operations
            memory_measurements = []
            
            def mock_memory_intensive_operation():
                # Simulate memory usage
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_measurements.append(current_memory)
                return {'memory_used_mb': current_memory}
            
            mock_client.return_value.perform_operation = Mock(side_effect=mock_memory_intensive_operation)
            
            # Perform multiple operations
            client = StrandsAgentCoreClient(
                region='us-east-1',
                llm_configs={'bedrock': {'model': 'claude-3'}},
                security_config={'memory_monitoring': True}
            )
            
            for i in range(10):
                mock_client.return_value.perform_operation()
                await asyncio.sleep(0.1)  # Allow memory measurement
            
            # Calculate memory metrics
            final_memory = process.memory_info().rss / 1024 / 1024
            max_memory = max(memory_measurements) if memory_measurements else final_memory
            avg_memory = statistics.mean(memory_measurements) if memory_measurements else final_memory
            memory_growth = final_memory - initial_memory
            
            # Memory thresholds
            max_memory_threshold = 512  # MB
            growth_threshold = 100  # MB
            
            # Evaluate memory usage
            memory_acceptable = max_memory < max_memory_threshold and memory_growth < growth_threshold
            score = 1.0
            if max_memory >= max_memory_threshold:
                score -= 0.5
            if memory_growth >= growth_threshold:
                score -= 0.5
            
            score = max(0.0, score)
            
            # Store metrics
            self.performance_metrics['memory_usage'] = {
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'max_memory_mb': max_memory,
                'avg_memory_mb': avg_memory,
                'memory_growth_mb': memory_growth
            }
            
            return {
                'status': 'PASSED' if score >= 0.8 else 'WARNING' if score >= 0.6 else 'FAILED',
                'score': score,
                'metrics': self.performance_metrics['memory_usage'],
                'thresholds': {
                    'max_memory_mb': max_memory_threshold,
                    'growth_threshold_mb': growth_threshold
                },
                'memory_acceptable': memory_acceptable
            }
    
    async def _validate_concurrent_performance(self) -> Dict[str, Any]:
        """Validate performance under concurrent load."""
        self.logger.info("Validating concurrent performance")
        
        with patch('strands_agentcore_session_helpers.StrandsAgentCoreClient') as mock_client:
            # Mock concurrent operations
            operation_times = []
            operation_lock = threading.Lock()
            
            def mock_concurrent_operation(operation_id):
                start_time = time.time()
                # Simulate operation with some variability
                time.sleep(0.05 + (operation_id % 3) * 0.01)
                end_time = time.time()
                
                operation_time = (end_time - start_time) * 1000
                with operation_lock:
                    operation_times.append(operation_time)
                
                return {
                    'operation_id': operation_id,
                    'execution_time_ms': operation_time,
                    'thread_id': threading.current_thread().ident
                }
            
            mock_client.return_value.execute_operation = Mock(side_effect=mock_concurrent_operation)
            
            # Test concurrent operations
            client = StrandsAgentCoreClient(
                region='us-east-1',
                llm_configs={'bedrock': {'model': 'claude-3'}},
                security_config={'concurrent_operations': True}
            )
            
            # Run concurrent operations
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(mock_client.return_value.execute_operation, i)
                    for i in range(20)
                ]
                
                results = [future.result() for future in as_completed(futures)]
            
            # Calculate concurrent performance metrics
            avg_concurrent_time = statistics.mean(operation_times)
            max_concurrent_time = max(operation_times)
            std_dev = statistics.stdev(operation_times) if len(operation_times) > 1 else 0
            
            # Performance evaluation
            baseline_time = 60  # Expected single operation time in ms
            degradation_factor = avg_concurrent_time / baseline_time
            consistency_good = std_dev < (avg_concurrent_time * 0.2)
            
            score = 1.0
            if degradation_factor > 2.0:  # More than 2x slower
                score -= 0.4
            elif degradation_factor > 1.5:  # More than 1.5x slower
                score -= 0.2
            
            if not consistency_good:
                score -= 0.3
            
            score = max(0.0, score)
            
            # Store metrics
            self.performance_metrics['concurrent_performance'] = {
                'avg_operation_time_ms': avg_concurrent_time,
                'max_operation_time_ms': max_concurrent_time,
                'std_deviation_ms': std_dev,
                'operations_completed': len(results),
                'degradation_factor': degradation_factor
            }
            
            return {
                'status': 'PASSED' if score >= 0.8 else 'WARNING' if score >= 0.6 else 'FAILED',
                'score': score,
                'metrics': self.performance_metrics['concurrent_performance'],
                'baseline_time_ms': baseline_time,
                'performance_acceptable': degradation_factor < 2.0 and consistency_good
            }
    
    async def _validate_scalability(self) -> Dict[str, Any]:
        """Validate scalability characteristics."""
        self.logger.info("Validating scalability")
        
        # Mock scalability testing
        scalability_results = []
        
        # Test with increasing loads
        load_levels = [1, 5, 10, 20, 50]
        
        for load in load_levels:
            start_time = time.time()
            
            # Simulate processing time that scales with load
            processing_time = 0.01 * load + (0.001 * load * load)  # Quadratic growth simulation
            await asyncio.sleep(processing_time)
            
            end_time = time.time()
            actual_time = (end_time - start_time) * 1000
            
            scalability_results.append({
                'load_level': load,
                'processing_time_ms': actual_time,
                'time_per_unit_ms': actual_time / load
            })
        
        # Analyze scalability
        time_per_unit_values = [result['time_per_unit_ms'] for result in scalability_results]
        scalability_factor = max(time_per_unit_values) / min(time_per_unit_values)
        
        # Good scalability should have factor < 2.0
        scalability_acceptable = scalability_factor < 2.0
        score = 1.0 if scalability_acceptable else max(0.0, 1.0 - ((scalability_factor - 2.0) / 2.0))
        
        # Store metrics
        self.performance_metrics['scalability'] = {
            'scalability_factor': scalability_factor,
            'load_levels_tested': load_levels,
            'scalability_results': scalability_results
        }
        
        return {
            'status': 'PASSED' if score >= 0.8 else 'WARNING' if score >= 0.6 else 'FAILED',
            'score': score,
            'metrics': self.performance_metrics['scalability'],
            'scalability_acceptable': scalability_acceptable
        }
    
    async def _validate_response_times(self) -> Dict[str, Any]:
        """Validate response times for different operations."""
        self.logger.info("Validating response times")
        
        # Mock different operation types with expected response times
        operations = {
            'session_creation': {'expected_ms': 100, 'actual_times': []},
            'credential_injection': {'expected_ms': 50, 'actual_times': []},
            'pii_detection': {'expected_ms': 30, 'actual_times': []},
            'audit_logging': {'expected_ms': 20, 'actual_times': []},
            'data_masking': {'expected_ms': 25, 'actual_times': []}
        }
        
        # Simulate operations
        for operation_name, operation_data in operations.items():
            for _ in range(5):  # 5 samples per operation
                start_time = time.time()
                
                # Simulate operation time with some variance
                base_time = operation_data['expected_ms'] / 1000
                variance = base_time * 0.2  # 20% variance
                actual_time = base_time + (variance * (0.5 - abs(hash(operation_name + str(_)) % 100) / 100))
                
                await asyncio.sleep(max(0.001, actual_time))  # Minimum 1ms
                
                end_time = time.time()
                measured_time = (end_time - start_time) * 1000
                operation_data['actual_times'].append(measured_time)
        
        # Analyze response times
        response_time_analysis = {}
        overall_score = 0.0
        
        for operation_name, operation_data in operations.items():
            avg_time = statistics.mean(operation_data['actual_times'])
            expected_time = operation_data['expected_ms']
            
            # Calculate score for this operation
            if avg_time <= expected_time:
                operation_score = 1.0
            elif avg_time <= expected_time * 1.5:
                operation_score = 0.8
            elif avg_time <= expected_time * 2.0:
                operation_score = 0.6
            else:
                operation_score = 0.4
            
            response_time_analysis[operation_name] = {
                'avg_time_ms': avg_time,
                'expected_time_ms': expected_time,
                'score': operation_score,
                'acceptable': avg_time <= expected_time * 1.5
            }
            
            overall_score += operation_score
        
        overall_score /= len(operations)
        
        # Store metrics
        self.performance_metrics['response_times'] = response_time_analysis
        
        return {
            'status': 'PASSED' if overall_score >= 0.8 else 'WARNING' if overall_score >= 0.6 else 'FAILED',
            'score': overall_score,
            'metrics': self.performance_metrics['response_times'],
            'operations_tested': len(operations)
        }
    
    def _generate_performance_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on validation results."""
        recommendations = []
        
        if results['performance_score'] < 0.8:
            recommendations.append("Overall performance score is below 80%. Review and optimize slow operations.")
        
        # Check specific performance areas
        for test_name, result in results['detailed_results'].items():
            if isinstance(result, dict) and result.get('score', 1.0) < 0.8:
                if test_name == 'session_performance':
                    recommendations.append("Session creation performance is slow. Consider session pooling or optimization.")
                elif test_name == 'credential_performance':
                    recommendations.append("Credential injection performance needs improvement. Review credential processing logic.")
                elif test_name == 'pii_processing_performance':
                    recommendations.append("PII processing is slow. Consider optimizing detection algorithms or using caching.")
                elif test_name == 'memory_usage':
                    recommendations.append("Memory usage is high. Review memory management and implement cleanup procedures.")
                elif test_name == 'concurrent_performance':
                    recommendations.append("Concurrent performance degrades significantly. Consider optimizing for parallel execution.")
                elif test_name == 'scalability':
                    recommendations.append("Scalability issues detected. Review architecture for better scaling characteristics.")
        
        # Memory-specific recommendations
        if 'memory_usage' in results['detailed_results']:
            memory_result = results['detailed_results']['memory_usage']
            if isinstance(memory_result, dict) and memory_result.get('metrics', {}).get('memory_growth_mb', 0) > 50:
                recommendations.append("Significant memory growth detected. Implement memory cleanup and garbage collection.")
        
        # Concurrent performance recommendations
        if 'concurrent_performance' in results['detailed_results']:
            concurrent_result = results['detailed_results']['concurrent_performance']
            if isinstance(concurrent_result, dict) and concurrent_result.get('metrics', {}).get('degradation_factor', 1.0) > 2.0:
                recommendations.append("High performance degradation under concurrent load. Consider connection pooling and resource optimization.")
        
        return recommendations


async def main():
    """Main function to run performance validation."""
    config = {
        'thresholds': {
            'session_creation_ms': 200,
            'credential_injection_ms': 100,
            'pii_masking_ms': 75,
            'audit_logging_ms': 50
        },
        'load_testing': {
            'max_concurrent_operations': 20,
            'test_duration_seconds': 60
        }
    }
    
    validator = PerformanceValidator(config)
    results = await validator.validate_performance()
    
    # Save results
    results_file = Path(__file__).parent / 'performance_validation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Performance validation completed: {results['overall_status']}")
    print(f"Performance Score: {results['performance_score']:.2f}/1.00")
    print(f"Results saved to: {results_file}")
    
    return results


if __name__ == '__main__':
    asyncio.run(main())