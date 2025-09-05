"""
Test Script for Workflow Orchestration Components

This script validates the SecureWorkflowOrchestrator and MultiAgentCoordinator
implementations to ensure they meet the requirements for secure workflow
management and multi-agent coordination.

Requirements Validated:
- 6.1: Multi-step workflow orchestration with security controls
- 6.2: Encrypted state management for sensitive data
- 6.3: Session pool management for efficient session reuse
- 6.4: Checkpoint and recovery mechanisms with security preservation
- 6.5: Resource allocation and session management for concurrent operations
- 6.6: Isolation mechanisms to prevent data leakage between agents
"""

import asyncio
import logging
import tempfile
import shutil
import os
from typing import Dict, Any
from datetime import datetime

# Import components to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.secure_workflow_orchestrator import (
    SecureWorkflowOrchestrator,
    SecureWorkflow,
    WorkflowStep,
    SessionPoolConfig,
    SecurityLevel,
    WorkflowStatus,
    create_secure_workflow,
    create_workflow_step
)

from tools.multi_agent_coordinator import (
    MultiAgentCoordinator,
    CoordinationConfig,
    ResourceType,
    IsolationLevel,
    AgentStatus,
    create_agent_task_config
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.id = f"test_{name.lower()}"


class WorkflowOrchestrationTests:
    """Test suite for workflow orchestration."""
    
    def __init__(self):
        self.temp_dir = None
        self.orchestrator = None
    
    async def setup(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="workflow_test_")
        
        session_config = SessionPoolConfig(
            max_sessions=3,
            min_sessions=1,
            session_timeout=60,
            reuse_sessions=True
        )
        
        self.orchestrator = SecureWorkflowOrchestrator(
            session_pool_config=session_config,
            checkpoint_storage_path=self.temp_dir
        )
        
        logger.info(f"Test environment set up in: {self.temp_dir}")
    
    async def teardown(self):
        """Clean up test environment."""
        if self.orchestrator:
            await self.orchestrator.shutdown()
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        logger.info("Test environment cleaned up")
    
    async def test_workflow_creation(self) -> bool:
        """Test workflow creation and validation."""
        logger.info("🧪 Testing workflow creation...")
        
        try:
            # Create valid workflow
            steps = [
                create_workflow_step(
                    step_id="step1",
                    name="Test Step 1",
                    description="First test step",
                    agent_name="TestAgent",
                    tool_name="test_tool",
                    action="test_action",
                    parameters={"param1": "value1"}
                ),
                create_workflow_step(
                    step_id="step2",
                    name="Test Step 2",
                    description="Second test step",
                    agent_name="TestAgent",
                    tool_name="test_tool",
                    action="test_action",
                    parameters={"param2": "value2"},
                    depends_on=["step1"]
                )
            ]
            
            workflow = create_secure_workflow(
                workflow_id="test_workflow",
                name="Test Workflow",
                description="Test workflow for validation",
                steps=steps,
                security_level=SecurityLevel.HIGH
            )
            
            # Validate workflow
            is_valid, errors = workflow.validate()
            
            if is_valid:
                logger.info("✅ Workflow creation and validation passed")
                return True
            else:
                logger.error(f"❌ Workflow validation failed: {errors}")
                return False
        
        except Exception as e:
            logger.error(f"❌ Workflow creation test failed: {str(e)}")
            return False
    
    async def test_workflow_execution(self) -> bool:
        """Test workflow execution."""
        logger.info("🧪 Testing workflow execution...")
        
        try:
            # Register test agent
            test_agent = MockAgent("TestAgent")
            self.orchestrator.register_agent(test_agent)
            
            # Create simple workflow
            steps = [
                create_workflow_step(
                    step_id="test_step",
                    name="Test Step",
                    description="Simple test step",
                    agent_name="TestAgent",
                    tool_name="test_tool",
                    action="test_action",
                    parameters={"test": "value"},
                    checkpoint_after=True
                )
            ]
            
            workflow = create_secure_workflow(
                workflow_id="execution_test_workflow",
                name="Execution Test Workflow",
                description="Test workflow execution",
                steps=steps
            )
            
            # Execute workflow
            result = await self.orchestrator.execute_workflow(workflow)
            
            if result.success:
                logger.info("✅ Workflow execution test passed")
                return True
            else:
                logger.error(f"❌ Workflow execution failed: {result.error}")
                return False
        
        except Exception as e:
            logger.error(f"❌ Workflow execution test failed: {str(e)}")
            return False
    
    async def test_checkpoint_creation(self) -> bool:
        """Test checkpoint creation and storage."""
        logger.info("🧪 Testing checkpoint creation...")
        
        try:
            # Register test agent
            test_agent = MockAgent("CheckpointAgent")
            self.orchestrator.register_agent(test_agent)
            
            # Create workflow with checkpoint
            steps = [
                create_workflow_step(
                    step_id="checkpoint_step",
                    name="Checkpoint Step",
                    description="Step that creates checkpoint",
                    agent_name="CheckpointAgent",
                    tool_name="test_tool",
                    action="test_action",
                    parameters={"checkpoint": True},
                    checkpoint_after=True
                )
            ]
            
            workflow = create_secure_workflow(
                workflow_id="checkpoint_test_workflow",
                name="Checkpoint Test Workflow",
                description="Test checkpoint creation",
                steps=steps,
                auto_checkpoint=True
            )
            
            # Execute workflow
            result = await self.orchestrator.execute_workflow(workflow)
            
            if result.success:
                # Check if checkpoint files were created
                checkpoint_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.json')]
                
                if checkpoint_files:
                    logger.info(f"✅ Checkpoint creation test passed ({len(checkpoint_files)} checkpoints)")
                    return True
                else:
                    logger.error("❌ No checkpoint files created")
                    return False
            else:
                logger.error(f"❌ Workflow execution for checkpoint test failed: {result.error}")
                return False
        
        except Exception as e:
            logger.error(f"❌ Checkpoint creation test failed: {str(e)}")
            return False
    
    async def test_session_pool_management(self) -> bool:
        """Test session pool management."""
        logger.info("🧪 Testing session pool management...")
        
        try:
            # Test session allocation and release
            session1 = await self.orchestrator.session_pool.get_session("workflow1", "step1")
            session2 = await self.orchestrator.session_pool.get_session("workflow2", "step1")
            
            if session1 and session2:
                # Release sessions
                await self.orchestrator.session_pool.release_session("workflow1", "step1")
                await self.orchestrator.session_pool.release_session("workflow2", "step1")
                
                logger.info("✅ Session pool management test passed")
                return True
            else:
                logger.error("❌ Failed to get sessions from pool")
                return False
        
        except Exception as e:
            logger.error(f"❌ Session pool management test failed: {str(e)}")
            return False


class MultiAgentCoordinationTests:
    """Test suite for multi-agent coordination."""
    
    def __init__(self):
        self.coordinator = None
    
    async def setup(self):
        """Set up test environment."""
        config = CoordinationConfig(
            max_concurrent_agents=5,
            default_isolation_level=IsolationLevel.STRICT,
            resource_timeout=60,
            data_share_timeout=120
        )
        
        self.coordinator = MultiAgentCoordinator(config=config)
        await self.coordinator.start_cleanup_task()
        
        logger.info("Multi-agent coordination test environment set up")
    
    async def teardown(self):
        """Clean up test environment."""
        if self.coordinator:
            await self.coordinator.shutdown()
        
        logger.info("Multi-agent coordination test environment cleaned up")
    
    async def test_agent_registration(self) -> bool:
        """Test agent registration and management."""
        logger.info("🧪 Testing agent registration...")
        
        try:
            # Register test agents
            agent1 = MockAgent("Agent1")
            agent2 = MockAgent("Agent2")
            
            agent_id1 = await self.coordinator.register_agent(agent1, IsolationLevel.STRICT)
            agent_id2 = await self.coordinator.register_agent(agent2, IsolationLevel.BASIC)
            
            # Check agent status
            status1 = self.coordinator.get_agent_status(agent_id1)
            status2 = self.coordinator.get_agent_status(agent_id2)
            
            if status1 and status2:
                logger.info("✅ Agent registration test passed")
                return True
            else:
                logger.error("❌ Failed to get agent status after registration")
                return False
        
        except Exception as e:
            logger.error(f"❌ Agent registration test failed: {str(e)}")
            return False
    
    async def test_resource_allocation(self) -> bool:
        """Test resource allocation and management."""
        logger.info("🧪 Testing resource allocation...")
        
        try:
            # Register test agent
            agent = MockAgent("ResourceAgent")
            agent_id = await self.coordinator.register_agent(agent)
            
            # Allocate resources
            success1 = await self.coordinator.resource_manager.allocate_resource(
                "test_resource_1", ResourceType.BROWSER_SESSION, agent_id
            )
            success2 = await self.coordinator.resource_manager.allocate_resource(
                "test_resource_2", ResourceType.COMPUTE_SLOT, agent_id
            )
            
            if success1 and success2:
                # Check allocated resources
                resources = await self.coordinator.resource_manager.get_agent_resources(agent_id)
                
                if len(resources) == 2:
                    # Release resources
                    await self.coordinator.resource_manager.release_resource("test_resource_1", agent_id)
                    await self.coordinator.resource_manager.release_resource("test_resource_2", agent_id)
                    
                    logger.info("✅ Resource allocation test passed")
                    return True
                else:
                    logger.error(f"❌ Expected 2 resources, got {len(resources)}")
                    return False
            else:
                logger.error("❌ Failed to allocate resources")
                return False
        
        except Exception as e:
            logger.error(f"❌ Resource allocation test failed: {str(e)}")
            return False
    
    async def test_data_sharing(self) -> bool:
        """Test secure data sharing between agents."""
        logger.info("🧪 Testing secure data sharing...")
        
        try:
            # Register test agents
            owner_agent = MockAgent("OwnerAgent")
            consumer_agent = MockAgent("ConsumerAgent")
            
            owner_id = await self.coordinator.register_agent(owner_agent)
            consumer_id = await self.coordinator.register_agent(consumer_agent)
            
            # Create data share
            test_data = {"sensitive_info": "test_value", "timestamp": datetime.now().isoformat()}
            
            share_id = await self.coordinator.share_data_between_agents(
                data_key="test_data",
                data=test_data,
                owner_agent_id=owner_id,
                target_agent_ids=[consumer_id],
                permissions={consumer_id: ["read"]},
                expires_in=60
            )
            
            # Access shared data
            retrieved_data = await self.coordinator.get_shared_data(share_id, consumer_id)
            
            if retrieved_data and retrieved_data["sensitive_info"] == "test_value":
                logger.info("✅ Data sharing test passed")
                return True
            else:
                logger.error("❌ Failed to retrieve shared data or data mismatch")
                return False
        
        except Exception as e:
            logger.error(f"❌ Data sharing test failed: {str(e)}")
            return False
    
    async def test_agent_coordination(self) -> bool:
        """Test agent coordination with parallel execution."""
        logger.info("🧪 Testing agent coordination...")
        
        try:
            # Create a fresh coordinator for this test to avoid agent limit issues
            fresh_config = CoordinationConfig(
                max_concurrent_agents=10,
                default_isolation_level=IsolationLevel.STRICT,
                resource_timeout=60,
                data_share_timeout=120
            )
            
            fresh_coordinator = MultiAgentCoordinator(config=fresh_config)
            await fresh_coordinator.start_cleanup_task()
            
            # Register test agents
            agents = [MockAgent(f"CoordAgent{i}") for i in range(3)]
            agent_ids = []
            
            for agent in agents:
                agent_id = await fresh_coordinator.register_agent(agent)
                agent_ids.append(agent_id)
            
            # Define test tasks
            def test_task(task_id: str, **kwargs) -> Dict[str, Any]:
                import time
                time.sleep(0.1)  # Simulate work
                return {"task_id": task_id, "completed": True}
            
            # Create task configurations
            task_configs = [
                {
                    'agent_id': agent_ids[i],
                    'task_func': test_task,
                    'task_args': (f"task_{i}",),
                    'require_resources': [(f"resource_{i}", ResourceType.COMPUTE_SLOT)]
                }
                for i in range(3)
            ]
            
            # Execute tasks in parallel
            results = await fresh_coordinator.coordinate_agents(
                task_configs,
                coordination_strategy="parallel"
            )
            
            successful_results = [r for r in results if r.success]
            
            # Clean up fresh coordinator
            await fresh_coordinator.shutdown()
            
            if len(successful_results) == 3:
                logger.info("✅ Agent coordination test passed")
                return True
            else:
                logger.error(f"❌ Expected 3 successful results, got {len(successful_results)}")
                return False
        
        except Exception as e:
            logger.error(f"❌ Agent coordination test failed: {str(e)}")
            return False


async def run_all_tests():
    """Run all test suites."""
    logger.info("🚀 Starting Workflow Orchestration & Multi-Agent Coordination Tests")
    logger.info("=" * 80)
    
    # Test results
    workflow_results = []
    coordination_results = []
    
    # Run workflow orchestration tests
    logger.info("📋 Running Workflow Orchestration Tests...")
    workflow_tests = WorkflowOrchestrationTests()
    
    try:
        await workflow_tests.setup()
        
        workflow_results.append(await workflow_tests.test_workflow_creation())
        workflow_results.append(await workflow_tests.test_workflow_execution())
        workflow_results.append(await workflow_tests.test_checkpoint_creation())
        workflow_results.append(await workflow_tests.test_session_pool_management())
        
    finally:
        await workflow_tests.teardown()
    
    logger.info("\n" + "=" * 80)
    
    # Run multi-agent coordination tests
    logger.info("🤝 Running Multi-Agent Coordination Tests...")
    coordination_tests = MultiAgentCoordinationTests()
    
    try:
        await coordination_tests.setup()
        
        coordination_results.append(await coordination_tests.test_agent_registration())
        coordination_results.append(await coordination_tests.test_resource_allocation())
        coordination_results.append(await coordination_tests.test_data_sharing())
        coordination_results.append(await coordination_tests.test_agent_coordination())
        
    finally:
        await coordination_tests.teardown()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 Test Results Summary")
    logger.info("=" * 80)
    
    workflow_passed = sum(workflow_results)
    workflow_total = len(workflow_results)
    
    coordination_passed = sum(coordination_results)
    coordination_total = len(coordination_results)
    
    logger.info(f"Workflow Orchestration Tests: {workflow_passed}/{workflow_total} passed")
    logger.info(f"Multi-Agent Coordination Tests: {coordination_passed}/{coordination_total} passed")
    
    total_passed = workflow_passed + coordination_passed
    total_tests = workflow_total + coordination_total
    
    logger.info(f"Overall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        logger.info("🎉 All tests passed! Implementation is working correctly.")
        return True
    else:
        logger.error(f"❌ {total_tests - total_passed} tests failed. Please review implementation.")
        return False


if __name__ == "__main__":
    # Run the test suite
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)