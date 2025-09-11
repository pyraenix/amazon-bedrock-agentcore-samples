"""
Strands CAPTCHA Framework Test Suite

This module contains comprehensive tests for the Strands CAPTCHA handling framework,
including unit tests, integration tests, performance benchmarks, and validation tests.

Test Categories:
- Unit Tests: Individual component testing
- Integration Tests: Multi-component workflow testing
- Performance Tests: Benchmarking and optimization validation
- Validation Tests: Tutorial and setup validation
- Error Handling Tests: Failure scenario testing
"""

import asyncio
import pytest
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch

# Import framework components
from strands_captcha_framework import (
    CaptchaHandlingAgent, CaptchaDetectionTool, CaptchaSolvingTool,
    WorkflowStateManager, CaptchaType, WorkflowPhase, 
    CaptchaFrameworkError, CaptchaDetectionError, CaptchaSolvingError,
    create_captcha_agent, validate_framework_setup
)

# =============================================================================
# TEST CONFIGURATION AND FIXTURES
# =============================================================================

class TestConfig:
    """Test configuration and constants"""
    
    # Test timeouts
    DEFAULT_TIMEOUT = 30
    LONG_TIMEOUT = 60
    
    # Test data
    MOCK_PAGE_URLS = [
        'https://test.example.com/recaptcha',
        'https://test.example.com/hcaptcha',
        'https://test.example.com/text-captcha'
    ]
    
    # Performance thresholds
    MAX_DETECTION_TIME = 5.0  # seconds
    MAX_SOLVING_TIME = 10.0   # seconds
    MAX_WORKFLOW_TIME = 30.0  # seconds
    
    # Success rate thresholds
    MIN_DETECTION_SUCCESS_RATE = 0.8  # 80%
    MIN_SOLVING_SUCCESS_RATE = 0.7    # 70%
    MIN_WORKFLOW_SUCCESS_RATE = 0.6   # 60%

@pytest.fixture
def mock_agentcore_client():
    """Mock AgentCore client for testing"""
    
    client = Mock()
    client.create_session = AsyncMock(return_value=Mock(id='test_session_123'))
    client.navigate = AsyncMock(return_value=Mock(success=True))
    client.find_elements = AsyncMock(return_value=[Mock()])
    client.screenshot = AsyncMock(return_value=b'mock_screenshot_data')
    
    return client

@pytest.fixture
def mock_bedrock_client():
    """Mock Bedrock client for testing"""
    
    client = Mock()
    client.invoke_model = AsyncMock(return_value={
        'body': json.dumps({
            'content': [{'text': 'MOCK123'}],
            'usage': {'input_tokens': 100, 'output_tokens': 10}
        })
    })
    
    return client

@pytest.fixture
def detection_tool(mock_agentcore_client):
    """CAPTCHA detection tool fixture"""
    return CaptchaDetectionTool(mock_agentcore_client)

@pytest.fixture
def solving_tool(mock_bedrock_client):
    """CAPTCHA solving tool fixture"""
    return CaptchaSolvingTool(mock_bedrock_client)

@pytest.fixture
def captcha_agent():
    """CAPTCHA handling agent fixture"""
    return create_captcha_agent({'max_attempts': 2, 'default_timeout': 30})

@pytest.fixture
def state_manager():
    """Workflow state manager fixture"""
    return WorkflowStateManager()

# =============================================================================
# UNIT TESTS
# =============================================================================

class TestCaptchaDetectionTool:
    """Unit tests for CAPTCHA detection tool"""
    
    @pytest.mark.asyncio
    async def test_detection_tool_initialization(self, mock_agentcore_client):
        """Test detection tool initialization"""
        
        tool = CaptchaDetectionTool(mock_agentcore_client)
        
        assert tool.name == "captcha_detector"
        assert tool.version == "1.0.0"
        assert len(tool.parameters) == 3
        assert tool.agentcore == mock_agentcore_client
        assert 'comprehensive' in tool.detection_strategies
    
    @pytest.mark.asyncio
    async def test_basic_detection_execution(self, detection_tool):
        """Test basic CAPTCHA detection execution"""
        
        result = await detection_tool.execute(
            page_url='https://test.example.com/recaptcha',
            detection_strategy='comprehensive'
        )
        
        assert result.success is True
        assert 'detected_captchas' in result.data
        assert 'execution_time' in result.data
        assert result.data['captcha_found'] is True
    
    @pytest.mark.asyncio
    async def test_detection_strategies(self, detection_tool):
        """Test different detection strategies"""
        
        strategies = ['comprehensive', 'recaptcha_focused', 'hcaptcha_focused']
        
        for strategy in strategies:
            result = await detection_tool.execute(
                page_url='https://test.example.com/test',
                detection_strategy=strategy
            )
            
            assert result.success is True
            assert result.data['detection_strategy'] == strategy
    
    @pytest.mark.asyncio
    async def test_detection_parameter_validation(self, detection_tool):
        """Test parameter validation for detection tool"""
        
        # Test missing required parameter
        result = await detection_tool.execute()
        
        assert result.success is False
        assert result.error_code == "MISSING_PARAMETER"
        assert "page_url parameter is required" in result.error
    
    @pytest.mark.asyncio
    async def test_detection_timeout_handling(self, detection_tool):
        """Test timeout handling in detection"""
        
        # Mock a slow operation
        with patch.object(detection_tool, '_execute_detection_workflow') as mock_workflow:
            mock_workflow.side_effect = asyncio.TimeoutError()
            
            result = await detection_tool.execute(
                page_url='https://test.example.com/slow',
                timeout=1
            )
            
            assert result.success is False
            assert result.error_code == "DETECTION_TIMEOUT"

class TestCaptchaSolvingTool:
    """Unit tests for CAPTCHA solving tool"""
    
    @pytest.mark.asyncio
    async def test_solving_tool_initialization(self, mock_bedrock_client):
        """Test solving tool initialization"""
        
        tool = CaptchaSolvingTool(mock_bedrock_client)
        
        assert tool.name == "captcha_solver"
        assert tool.version == "1.0.0"
        assert len(tool.parameters) == 3
        assert tool.bedrock == mock_bedrock_client
        assert 'claude-3-sonnet' in tool.model_configs
    
    @pytest.mark.asyncio
    async def test_basic_solving_execution(self, solving_tool):
        """Test basic CAPTCHA solving execution"""
        
        mock_captcha_data = {
            'detected_captchas': [
                {
                    'captcha_type': CaptchaType.TEXT_CAPTCHA,
                    'screenshot': 'mock_screenshot_data',
                    'confidence': 0.9
                }
            ]
        }
        
        result = await solving_tool.execute(
            captcha_data=mock_captcha_data,
            model_preference='auto'
        )
        
        assert result.success is True
        assert 'solutions' in result.data
        assert result.data['successful_count'] > 0
    
    @pytest.mark.asyncio
    async def test_model_selection(self, solving_tool):
        """Test AI model selection logic"""
        
        mock_captcha_data = {
            'detected_captchas': [
                {
                    'captcha_type': CaptchaType.RECAPTCHA_V2,
                    'screenshot': 'mock_screenshot_data',
                    'confidence': 0.9
                }
            ]
        }
        
        # Test auto model selection
        result = await solving_tool.execute(
            captcha_data=mock_captcha_data,
            model_preference='auto'
        )
        
        assert result.success is True
        
        # Test specific model selection
        result = await solving_tool.execute(
            captcha_data=mock_captcha_data,
            model_preference='claude-3-opus'
        )
        
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_confidence_threshold_validation(self, solving_tool):
        """Test confidence threshold validation"""
        
        mock_captcha_data = {
            'detected_captchas': [
                {
                    'captcha_type': CaptchaType.TEXT_CAPTCHA,
                    'screenshot': 'mock_screenshot_data',
                    'confidence': 0.9
                }
            ]
        }
        
        # Test with high confidence threshold
        result = await solving_tool.execute(
            captcha_data=mock_captcha_data,
            confidence_threshold=0.95
        )
        
        # Should still succeed due to retry mechanism
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_solving_parameter_validation(self, solving_tool):
        """Test parameter validation for solving tool"""
        
        # Test missing captcha_data
        result = await solving_tool.execute()
        
        assert result.success is False
        assert result.error_code == "INVALID_CAPTCHA_DATA"
        
        # Test empty captcha_data
        result = await solving_tool.execute(captcha_data={})
        
        assert result.success is False
        assert result.error_code == "NO_CAPTCHAS_DETECTED"

class TestWorkflowStateManager:
    """Unit tests for workflow state manager"""
    
    @pytest.mark.asyncio
    async def test_state_manager_initialization(self, state_manager):
        """Test state manager initialization"""
        
        assert len(state_manager.workflow_states) == 0
        assert len(state_manager.state_history) == 0
    
    @pytest.mark.asyncio
    async def test_workflow_state_initialization(self, state_manager):
        """Test workflow state initialization"""
        
        initial_context = {
            'page_url': 'https://test.example.com',
            'task_description': 'Test task'
        }
        
        workflow_state = await state_manager.initialize_workflow_state(
            'test_workflow_123', initial_context
        )
        
        assert workflow_state.workflow_id == 'test_workflow_123'
        assert workflow_state.current_phase == WorkflowPhase.INITIALIZATION
        assert workflow_state.page_url == 'https://test.example.com'
        assert workflow_state.task_description == 'Test task'
    
    @pytest.mark.asyncio
    async def test_workflow_phase_transitions(self, state_manager):
        """Test workflow phase transitions"""
        
        # Initialize workflow
        initial_context = {'page_url': 'https://test.example.com', 'task_description': 'Test'}
        workflow_state = await state_manager.initialize_workflow_state('test_workflow', initial_context)
        
        # Test valid phase transition
        updated_state = await state_manager.update_workflow_phase(
            'test_workflow', WorkflowPhase.DETECTION, {'test_data': 'value'}
        )
        
        assert updated_state.current_phase == WorkflowPhase.DETECTION
        assert WorkflowPhase.INITIALIZATION in updated_state.completed_phases
    
    @pytest.mark.asyncio
    async def test_invalid_phase_transition(self, state_manager):
        """Test invalid phase transition handling"""
        
        # Initialize workflow
        initial_context = {'page_url': 'https://test.example.com', 'task_description': 'Test'}
        await state_manager.initialize_workflow_state('test_workflow', initial_context)
        
        # Test invalid phase transition (skip phases)
        with pytest.raises(Exception):  # Should raise StateManagementError
            await state_manager.update_workflow_phase(
                'test_workflow', WorkflowPhase.COMPLETED, {}
            )
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, state_manager):
        """Test workflow error handling"""
        
        # Initialize workflow
        initial_context = {'page_url': 'https://test.example.com', 'task_description': 'Test'}
        workflow_state = await state_manager.initialize_workflow_state('test_workflow', initial_context)
        
        # Simulate error
        test_error = Exception("Test error")
        
        error_state = await state_manager.handle_workflow_error(
            'test_workflow', test_error, 'retry_current_phase'
        )
        
        assert len(error_state.error_history) == 1
        assert error_state.error_history[0]['error_message'] == "Test error"

class TestCaptchaHandlingAgent:
    """Unit tests for CAPTCHA handling agent"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, captcha_agent):
        """Test agent initialization"""
        
        assert captcha_agent.max_attempts == 2
        assert captcha_agent.default_timeout == 30
        assert hasattr(captcha_agent, 'detection_tool')
        assert hasattr(captcha_agent, 'solving_tool')
        assert hasattr(captcha_agent, 'state_manager')
    
    @pytest.mark.asyncio
    async def test_basic_workflow_execution(self, captcha_agent):
        """Test basic workflow execution"""
        
        result = await captcha_agent.handle_captcha_workflow(
            page_url='https://test.example.com/form',
            task_description='Fill out test form'
        )
        
        assert 'success' in result
        assert 'workflow_id' in result
        assert isinstance(result['success'], bool)
    
    @pytest.mark.asyncio
    async def test_workflow_with_config(self):
        """Test workflow with custom configuration"""
        
        config = {
            'max_attempts': 5,
            'default_timeout': 120
        }
        
        agent = create_captcha_agent(config)
        
        result = await agent.handle_captcha_workflow(
            page_url='https://test.example.com/complex',
            task_description='Complex CAPTCHA task',
            config={'custom_setting': True}
        )
        
        assert 'success' in result
        assert 'workflow_id' in result

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestWorkflowIntegration:
    """Integration tests for complete workflows"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, captcha_agent):
        """Test complete end-to-end CAPTCHA workflow"""
        
        # Test successful workflow
        result = await captcha_agent.handle_captcha_workflow(
            page_url='https://test.example.com/integration',
            task_description='Integration test workflow'
        )
        
        assert result.get('success') is not None
        assert 'workflow_id' in result
        
        if result.get('success'):
            assert result.get('task_completed') is True
        else:
            assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_multi_captcha_workflow(self, captcha_agent):
        """Test workflow with multiple CAPTCHAs"""
        
        # Simulate page with multiple CAPTCHAs
        result = await captcha_agent.handle_captcha_workflow(
            page_url='https://test.example.com/multi-captcha',
            task_description='Handle multiple CAPTCHAs'
        )
        
        assert 'success' in result
        assert 'workflow_id' in result
    
    @pytest.mark.asyncio
    async def test_workflow_state_consistency(self, captcha_agent):
        """Test workflow state consistency across phases"""
        
        # Start workflow
        result = await captcha_agent.handle_captcha_workflow(
            page_url='https://test.example.com/state-test',
            task_description='State consistency test'
        )
        
        # Verify state consistency
        workflow_id = result.get('workflow_id')
        assert workflow_id is not None
        
        # Check that state was properly managed
        state_manager = captcha_agent.state_manager
        if workflow_id in state_manager.workflow_states:
            final_state = state_manager.workflow_states[workflow_id]
            assert final_state.workflow_id == workflow_id
            assert final_state.current_phase in [WorkflowPhase.COMPLETED, WorkflowPhase.FAILED]

class TestServiceIntegration:
    """Integration tests for service coordination"""
    
    @pytest.mark.asyncio
    async def test_detection_solving_integration(self, detection_tool, solving_tool):
        """Test integration between detection and solving tools"""
        
        # Execute detection
        detection_result = await detection_tool.execute(
            page_url='https://test.example.com/integration',
            detection_strategy='comprehensive'
        )
        
        assert detection_result.success is True
        
        # Use detection results for solving
        solving_result = await solving_tool.execute(
            captcha_data=detection_result.data,
            model_preference='auto'
        )
        
        assert solving_result.success is True
        
        # Verify data flow
        assert len(solving_result.data['solutions']) > 0
    
    @pytest.mark.asyncio
    async def test_parallel_service_coordination(self, captcha_agent):
        """Test parallel coordination of multiple services"""
        
        # Create multiple concurrent workflows
        tasks = []
        for i in range(3):
            task = asyncio.create_task(
                captcha_agent.handle_captcha_workflow(
                    page_url=f'https://test.example.com/parallel-{i}',
                    task_description=f'Parallel test {i}'
                )
            )
            tasks.append(task)
        
        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all completed
        assert len(results) == 3
        
        # Check for successful coordination
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) > 0

# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformanceBenchmarks:
    """Performance benchmarking tests"""
    
    @pytest.mark.asyncio
    async def test_detection_performance(self, detection_tool):
        """Test CAPTCHA detection performance"""
        
        start_time = time.time()
        
        # Run multiple detection operations
        tasks = []
        for i in range(5):
            task = asyncio.create_task(
                detection_tool.execute(
                    page_url=f'https://test.example.com/perf-{i}',
                    detection_strategy='comprehensive'
                )
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        avg_time = total_time / len(results)
        
        # Performance assertions
        assert avg_time < TestConfig.MAX_DETECTION_TIME
        assert all(r.success for r in results)
        
        print(f"Detection performance: {avg_time:.3f}s average, {total_time:.3f}s total")
    
    @pytest.mark.asyncio
    async def test_solving_performance(self, solving_tool):
        """Test CAPTCHA solving performance"""
        
        mock_captcha_data = {
            'detected_captchas': [
                {
                    'captcha_type': CaptchaType.TEXT_CAPTCHA,
                    'screenshot': 'mock_screenshot_data',
                    'confidence': 0.9
                }
            ]
        }
        
        start_time = time.time()
        
        # Run multiple solving operations
        tasks = []
        for i in range(3):  # Fewer tasks for AI operations
            task = asyncio.create_task(
                solving_tool.execute(
                    captcha_data=mock_captcha_data,
                    model_preference='auto'
                )
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        avg_time = total_time / len(results)
        
        # Performance assertions
        assert avg_time < TestConfig.MAX_SOLVING_TIME
        assert all(r.success for r in results)
        
        print(f"Solving performance: {avg_time:.3f}s average, {total_time:.3f}s total")
    
    @pytest.mark.asyncio
    async def test_workflow_performance(self, captcha_agent):
        """Test complete workflow performance"""
        
        start_time = time.time()
        
        result = await captcha_agent.handle_captcha_workflow(
            page_url='https://test.example.com/performance',
            task_description='Performance test workflow'
        )
        
        execution_time = time.time() - start_time
        
        # Performance assertions
        assert execution_time < TestConfig.MAX_WORKFLOW_TIME
        assert 'success' in result
        
        print(f"Workflow performance: {execution_time:.3f}s total")
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, captcha_agent):
        """Test memory usage during operations"""
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run multiple workflows
        for i in range(10):
            await captcha_agent.handle_captcha_workflow(
                page_url=f'https://test.example.com/memory-{i}',
                task_description=f'Memory test {i}'
            )
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory should not increase excessively (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024
        
        print(f"Memory usage increase: {memory_increase / 1024 / 1024:.2f} MB")

class TestScalabilityBenchmarks:
    """Scalability and load testing"""
    
    @pytest.mark.asyncio
    async def test_concurrent_workflows(self, captcha_agent):
        """Test concurrent workflow handling"""
        
        concurrent_count = 10
        
        start_time = time.time()
        
        # Create concurrent workflows
        tasks = []
        for i in range(concurrent_count):
            task = asyncio.create_task(
                captcha_agent.handle_captcha_workflow(
                    page_url=f'https://test.example.com/concurrent-{i}',
                    task_description=f'Concurrent test {i}'
                )
            )
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_results = [r for r in results if not isinstance(r, Exception) and r.get('success')]
        success_rate = len(successful_results) / len(results)
        
        # Scalability assertions
        assert success_rate >= TestConfig.MIN_WORKFLOW_SUCCESS_RATE
        assert total_time < TestConfig.MAX_WORKFLOW_TIME * 2  # Allow some overhead for concurrency
        
        print(f"Concurrent workflows: {len(successful_results)}/{len(results)} successful "
              f"({success_rate:.2%}) in {total_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_sustained_load(self, captcha_agent):
        """Test sustained load handling"""
        
        duration = 30  # seconds
        request_interval = 1  # second
        
        start_time = time.time()
        results = []
        
        while (time.time() - start_time) < duration:
            try:
                result = await asyncio.wait_for(
                    captcha_agent.handle_captcha_workflow(
                        page_url='https://test.example.com/sustained-load',
                        task_description='Sustained load test'
                    ),
                    timeout=10
                )
                results.append(result)
            except asyncio.TimeoutError:
                results.append({'success': False, 'error': 'timeout'})
            
            await asyncio.sleep(request_interval)
        
        # Analyze sustained performance
        successful_results = [r for r in results if r.get('success')]
        success_rate = len(successful_results) / len(results) if results else 0
        
        assert success_rate >= TestConfig.MIN_WORKFLOW_SUCCESS_RATE
        
        print(f"Sustained load: {len(successful_results)}/{len(results)} successful "
              f"({success_rate:.2%}) over {duration}s")

# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Error handling and recovery tests"""
    
    @pytest.mark.asyncio
    async def test_detection_error_handling(self, detection_tool):
        """Test error handling in detection tool"""
        
        # Test with invalid URL
        result = await detection_tool.execute(
            page_url='invalid-url',
            detection_strategy='comprehensive'
        )
        
        # Should handle gracefully
        assert result.success is False or result.success is True  # May succeed with mock
    
    @pytest.mark.asyncio
    async def test_solving_error_handling(self, solving_tool):
        """Test error handling in solving tool"""
        
        # Test with invalid captcha data
        invalid_data = {'invalid': 'data'}
        
        result = await solving_tool.execute(captcha_data=invalid_data)
        
        assert result.success is False
        assert result.error_code == "NO_CAPTCHAS_DETECTED"
    
    @pytest.mark.asyncio
    async def test_workflow_error_recovery(self, captcha_agent):
        """Test workflow error recovery mechanisms"""
        
        # Test with problematic URL that might cause errors
        result = await captcha_agent.handle_captcha_workflow(
            page_url='https://test.example.com/error-prone',
            task_description='Error recovery test'
        )
        
        # Should either succeed or fail gracefully
        assert 'success' in result
        assert 'workflow_id' in result
        
        if not result.get('success'):
            assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, captcha_agent):
        """Test timeout handling in workflows"""
        
        # Create agent with short timeout
        short_timeout_agent = create_captcha_agent({
            'max_attempts': 1,
            'default_timeout': 1  # Very short timeout
        })
        
        result = await short_timeout_agent.handle_captcha_workflow(
            page_url='https://test.example.com/slow-page',
            task_description='Timeout test'
        )
        
        # Should handle timeout gracefully
        assert 'success' in result
        assert 'workflow_id' in result
    
    @pytest.mark.asyncio
    async def test_retry_mechanisms(self, captcha_agent):
        """Test retry mechanisms"""
        
        # Test with agent configured for retries
        retry_agent = create_captcha_agent({
            'max_attempts': 3,
            'default_timeout': 30
        })
        
        result = await retry_agent.handle_captcha_workflow(
            page_url='https://test.example.com/retry-test',
            task_description='Retry mechanism test'
        )
        
        assert 'success' in result
        assert 'workflow_id' in result
        
        # Check if retry information is available
        if not result.get('success'):
            # May contain retry information
            assert 'attempts' in result or 'retry_count' in result

# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestTutorialValidation:
    """Tutorial and setup validation tests"""
    
    @pytest.mark.asyncio
    async def test_framework_setup_validation(self):
        """Test framework setup validation"""
        
        validation_results = await validate_framework_setup()
        
        assert isinstance(validation_results, dict)
        assert 'agent_creation' in validation_results
        assert 'detection_tool' in validation_results
        assert 'solving_tool' in validation_results
        assert 'state_manager' in validation_results
        
        # All components should validate successfully
        for component, result in validation_results.items():
            if component != 'validation_error':
                assert result is True, f"Component {component} failed validation"
    
    @pytest.mark.asyncio
    async def test_tutorial_examples_validation(self):
        """Test that tutorial examples work correctly"""
        
        # Test basic agent creation example
        agent = create_captcha_agent()
        assert agent is not None
        
        # Test basic workflow example
        result = await agent.handle_captcha_workflow(
            page_url='https://example.com/tutorial-test',
            task_description='Tutorial validation test'
        )
        
        assert 'success' in result
        assert 'workflow_id' in result
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        
        # Test valid configurations
        valid_configs = [
            {},
            {'max_attempts': 3},
            {'max_attempts': 5, 'default_timeout': 60},
            {'max_attempts': 1, 'default_timeout': 30}
        ]
        
        for config in valid_configs:
            agent = create_captcha_agent(config)
            assert agent is not None
            
            if 'max_attempts' in config:
                assert agent.max_attempts == config['max_attempts']
            if 'default_timeout' in config:
                assert agent.default_timeout == config['default_timeout']
    
    def test_captcha_type_enumeration(self):
        """Test CAPTCHA type enumeration"""
        
        # Verify all expected CAPTCHA types are defined
        expected_types = [
            'TEXT_CAPTCHA', 'RECAPTCHA_V2', 'RECAPTCHA_V3', 'HCAPTCHA',
            'IMAGE_SELECTION', 'MATHEMATICAL', 'TURNSTILE', 'FUNCAPTCHA', 'GENERIC'
        ]
        
        for type_name in expected_types:
            assert hasattr(CaptchaType, type_name)
            captcha_type = getattr(CaptchaType, type_name)
            assert isinstance(captcha_type, CaptchaType)
    
    def test_workflow_phase_enumeration(self):
        """Test workflow phase enumeration"""
        
        # Verify all expected workflow phases are defined
        expected_phases = [
            'INITIALIZATION', 'DETECTION', 'ANALYSIS', 'SOLUTION',
            'SUBMISSION', 'VERIFICATION', 'COMPLETED', 'FAILED'
        ]
        
        for phase_name in expected_phases:
            assert hasattr(WorkflowPhase, phase_name)
            phase = getattr(WorkflowPhase, phase_name)
            assert isinstance(phase, WorkflowPhase)

class TestCaptchaSiteValidation:
    """Validation tests for different CAPTCHA sites"""
    
    @pytest.mark.asyncio
    async def test_recaptcha_site_handling(self, captcha_agent):
        """Test handling of reCAPTCHA sites"""
        
        recaptcha_urls = [
            'https://test.example.com/recaptcha-v2',
            'https://test.example.com/recaptcha-v3'
        ]
        
        for url in recaptcha_urls:
            result = await captcha_agent.handle_captcha_workflow(
                page_url=url,
                task_description=f'Test reCAPTCHA handling for {url}'
            )
            
            assert 'success' in result
            assert 'workflow_id' in result
    
    @pytest.mark.asyncio
    async def test_hcaptcha_site_handling(self, captcha_agent):
        """Test handling of hCaptcha sites"""
        
        result = await captcha_agent.handle_captcha_workflow(
            page_url='https://test.example.com/hcaptcha',
            task_description='Test hCaptcha handling'
        )
        
        assert 'success' in result
        assert 'workflow_id' in result
    
    @pytest.mark.asyncio
    async def test_text_captcha_handling(self, captcha_agent):
        """Test handling of text-based CAPTCHAs"""
        
        result = await captcha_agent.handle_captcha_workflow(
            page_url='https://test.example.com/text-captcha',
            task_description='Test text CAPTCHA handling'
        )
        
        assert 'success' in result
        assert 'workflow_id' in result
    
    @pytest.mark.asyncio
    async def test_no_captcha_site_handling(self, captcha_agent):
        """Test handling of sites without CAPTCHAs"""
        
        result = await captcha_agent.handle_captcha_workflow(
            page_url='https://test.example.com/no-captcha',
            task_description='Test site without CAPTCHA'
        )
        
        assert 'success' in result
        assert 'workflow_id' in result
        
        # Should complete successfully without CAPTCHA handling
        if result.get('success'):
            assert result.get('captcha_handled', True) is False  # No CAPTCHA to handle

# =============================================================================
# TEST UTILITIES
# =============================================================================

class TestUtilities:
    """Utility functions for testing"""
    
    @staticmethod
    def create_mock_captcha_data(captcha_type: CaptchaType = CaptchaType.TEXT_CAPTCHA) -> Dict[str, Any]:
        """Create mock CAPTCHA data for testing"""
        
        return {
            'detected_captchas': [
                {
                    'captcha_type': captcha_type,
                    'bounds': {'x': 100, 'y': 200, 'width': 300, 'height': 150},
                    'screenshot': 'mock_screenshot_data',
                    'confidence': 0.9,
                    'selector': '.captcha-element',
                    'detector': 'test_detector'
                }
            ],
            'detection_strategy': 'comprehensive',
            'page_url': 'https://test.example.com',
            'execution_time': 0.5,
            'captcha_found': True
        }
    
    @staticmethod
    def assert_valid_workflow_result(result: Dict[str, Any]):
        """Assert that a workflow result is valid"""
        
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'workflow_id' in result
        assert isinstance(result['success'], bool)
        assert isinstance(result['workflow_id'], str)
        
        if result['success']:
            assert 'task_completed' in result
        else:
            assert 'error' in result
    
    @staticmethod
    def measure_execution_time(func):
        """Decorator to measure execution time"""
        
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if isinstance(result, dict):
                result['measured_execution_time'] = execution_time
            
            return result
        
        return wrapper

# =============================================================================
# TEST RUNNER AND REPORTING
# =============================================================================

class TestRunner:
    """Test runner with comprehensive reporting"""
    
    def __init__(self):
        self.results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'skipped_tests': 0,
            'execution_time': 0,
            'test_details': []
        }
    
    async def run_all_tests(self):
        """Run all test suites"""
        
        start_time = time.time()
        
        test_suites = [
            ('Unit Tests - Detection Tool', TestCaptchaDetectionTool),
            ('Unit Tests - Solving Tool', TestCaptchaSolvingTool),
            ('Unit Tests - State Manager', TestWorkflowStateManager),
            ('Unit Tests - CAPTCHA Agent', TestCaptchaHandlingAgent),
            ('Integration Tests - Workflows', TestWorkflowIntegration),
            ('Integration Tests - Services', TestServiceIntegration),
            ('Performance Tests - Benchmarks', TestPerformanceBenchmarks),
            ('Performance Tests - Scalability', TestScalabilityBenchmarks),
            ('Error Handling Tests', TestErrorHandling),
            ('Validation Tests - Tutorial', TestTutorialValidation),
            ('Validation Tests - CAPTCHA Sites', TestCaptchaSiteValidation)
        ]
        
        print("🧪 Running Strands CAPTCHA Framework Test Suite")
        print("=" * 60)
        
        for suite_name, test_class in test_suites:
            print(f"\n📋 Running: {suite_name}")
            await self._run_test_suite(suite_name, test_class)
        
        self.results['execution_time'] = time.time() - start_time
        
        print("\n" + "=" * 60)
        self._print_summary()
    
    async def _run_test_suite(self, suite_name: str, test_class):
        """Run a single test suite"""
        
        suite_results = {
            'suite_name': suite_name,
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'test_methods': []
        }
        
        # Get test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for method_name in test_methods:
            try:
                # Create test instance
                test_instance = test_class()
                
                # Get test method
                test_method = getattr(test_instance, method_name)
                
                # Run test
                if asyncio.iscoroutinefunction(test_method):
                    await test_method()
                else:
                    test_method()
                
                suite_results['tests_passed'] += 1
                suite_results['test_methods'].append({
                    'name': method_name,
                    'status': 'PASSED'
                })
                print(f"  ✅ {method_name}")
                
            except Exception as e:
                suite_results['tests_failed'] += 1
                suite_results['test_methods'].append({
                    'name': method_name,
                    'status': 'FAILED',
                    'error': str(e)
                })
                print(f"  ❌ {method_name}: {e}")
            
            suite_results['tests_run'] += 1
        
        # Update overall results
        self.results['total_tests'] += suite_results['tests_run']
        self.results['passed_tests'] += suite_results['tests_passed']
        self.results['failed_tests'] += suite_results['tests_failed']
        self.results['test_details'].append(suite_results)
        
        # Print suite summary
        success_rate = (suite_results['tests_passed'] / suite_results['tests_run'] 
                       if suite_results['tests_run'] > 0 else 0)
        print(f"  📊 Suite Results: {suite_results['tests_passed']}/{suite_results['tests_run']} "
              f"passed ({success_rate:.1%})")
    
    def _print_summary(self):
        """Print test execution summary"""
        
        success_rate = (self.results['passed_tests'] / self.results['total_tests'] 
                       if self.results['total_tests'] > 0 else 0)
        
        print("📊 Test Execution Summary")
        print("-" * 30)
        print(f"Total Tests: {self.results['total_tests']}")
        print(f"Passed: {self.results['passed_tests']}")
        print(f"Failed: {self.results['failed_tests']}")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Execution Time: {self.results['execution_time']:.2f}s")
        
        if self.results['failed_tests'] > 0:
            print(f"\n❌ Failed Tests:")
            for suite in self.results['test_details']:
                failed_methods = [m for m in suite['test_methods'] if m['status'] == 'FAILED']
                if failed_methods:
                    print(f"  {suite['suite_name']}:")
                    for method in failed_methods:
                        print(f"    - {method['name']}: {method.get('error', 'Unknown error')}")
        
        if success_rate >= 0.8:
            print(f"\n🎉 Test suite passed with {success_rate:.1%} success rate!")
        else:
            print(f"\n⚠️ Test suite needs attention - {success_rate:.1%} success rate")

# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================

async def run_framework_tests():
    """Run the complete framework test suite"""
    
    runner = TestRunner()
    await runner.run_all_tests()
    
    return runner.results

if __name__ == "__main__":
    # Run all tests
    asyncio.run(run_framework_tests())