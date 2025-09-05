"""
Workflow Orchestration Example for Strands Integration

This example demonstrates how to use the SecureWorkflowOrchestrator and
MultiAgentCoordinator to manage complex multi-step workflows with multiple
Strands agents using AgentCore Browser Tool.

Key Features Demonstrated:
- Multi-step workflow creation and execution
- Multi-agent coordination with resource sharing
- Secure data sharing between agents
- Session pool management
- Checkpoint and recovery mechanisms
- Comprehensive security controls

Requirements Addressed:
- 6.1: Multi-step workflow orchestration with security controls
- 6.2: Encrypted state management for sensitive data
- 6.3: Session pool management for efficient session reuse
- 6.4: Checkpoint and recovery mechanisms with security preservation
- 6.5: Resource allocation and session management for concurrent operations
- 6.6: Isolation mechanisms to prevent data leakage between agents
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

# Import Strands workflow orchestration components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.secure_workflow_orchestrator import (
    SecureWorkflowOrchestrator,
    SecureWorkflow,
    WorkflowStep,
    SessionPoolConfig,
    SecurityLevel,
    create_secure_workflow,
    create_workflow_step
)

from tools.multi_agent_coordinator import (
    MultiAgentCoordinator,
    CoordinationConfig,
    ResourceType,
    IsolationLevel,
    create_agent_task_config,
    create_data_share_permissions
)

from tools.agentcore_browser_tool import AgentCoreBrowserTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockStrandsAgent:
    """Mock Strands agent for demonstration purposes."""
    
    def __init__(self, name: str):
        self.name = name
        self.id = f"mock_{name.lower()}"
    
    def execute_task(self, task_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a mock task."""
        logger.info(f"Agent {self.name} executing task: {task_name}")
        
        # Simulate task execution
        import time
        time.sleep(1)
        
        return {
            'task_name': task_name,
            'agent_name': self.name,
            'result': f"Task {task_name} completed by {self.name}",
            'timestamp': datetime.now().isoformat(),
            'parameters': kwargs
        }


async def demonstrate_workflow_orchestration():
    """Demonstrate secure workflow orchestration with multiple agents."""
    logger.info("🚀 Starting Workflow Orchestration Demonstration")
    
    # Create session pool configuration
    session_config = SessionPoolConfig(
        max_sessions=5,
        min_sessions=2,
        session_timeout=300,
        reuse_sessions=True,
        isolation_level="strict"
    )
    
    # Initialize workflow orchestrator
    orchestrator = SecureWorkflowOrchestrator(
        session_pool_config=session_config,
        checkpoint_storage_path="./demo_checkpoints"
    )
    
    # Create mock agents
    data_extraction_agent = MockStrandsAgent("DataExtractor")
    form_filler_agent = MockStrandsAgent("FormFiller")
    validator_agent = MockStrandsAgent("Validator")
    
    # Register agents
    orchestrator.register_agent(data_extraction_agent)
    orchestrator.register_agent(form_filler_agent)
    orchestrator.register_agent(validator_agent)
    
    # Create workflow steps
    steps = [
        create_workflow_step(
            step_id="extract_data",
            name="Extract Customer Data",
            description="Extract customer information from secure portal",
            agent_name="DataExtractor",
            tool_name="agentcore_browser",
            action="extract_data",
            parameters={
                "selectors": [".customer-name", ".customer-email", ".customer-phone"],
                "extract_type": "text"
            },
            security_level=SecurityLevel.HIGH,
            sensitive_data=True,
            checkpoint_after=True
        ),
        
        create_workflow_step(
            step_id="fill_application",
            name="Fill Application Form",
            description="Fill application form with extracted data",
            agent_name="FormFiller",
            tool_name="agentcore_browser",
            action="fill_form",
            parameters={
                "form_selector": "#application-form",
                "use_extracted_data": True
            },
            depends_on=["extract_data"],
            security_level=SecurityLevel.HIGH,
            sensitive_data=True,
            checkpoint_after=True
        ),
        
        create_workflow_step(
            step_id="validate_submission",
            name="Validate Form Submission",
            description="Validate that form was submitted correctly",
            agent_name="Validator",
            tool_name="agentcore_browser",
            action="extract_data",
            parameters={
                "selectors": [".success-message", ".confirmation-number"],
                "extract_type": "text"
            },
            depends_on=["fill_application"],
            security_level=SecurityLevel.MEDIUM,
            checkpoint_after=True
        )
    ]
    
    # Create secure workflow
    workflow = create_secure_workflow(
        workflow_id="customer_application_workflow",
        name="Customer Application Processing",
        description="Secure workflow for processing customer applications with sensitive data",
        steps=steps,
        security_level=SecurityLevel.HIGH,
        auto_checkpoint=True,
        checkpoint_interval=60,
        error_handling_strategy="retry_and_checkpoint"
    )
    
    logger.info(f"📋 Created workflow with {len(workflow.steps)} steps")
    
    # Execute workflow
    try:
        logger.info("▶️ Starting workflow execution...")
        result = await orchestrator.execute_workflow(workflow)
        
        if result.success:
            logger.info("✅ Workflow completed successfully!")
            logger.info(f"Completed steps: {result.data['completed_steps']}")
            logger.info(f"Execution time: {result.data['execution_time']:.2f} seconds")
        else:
            logger.error(f"❌ Workflow failed: {result.error}")
    
    except Exception as e:
        logger.error(f"❌ Workflow execution error: {str(e)}")
    
    finally:
        # Cleanup
        await orchestrator.shutdown()


async def demonstrate_multi_agent_coordination():
    """Demonstrate multi-agent coordination with resource sharing."""
    logger.info("🤝 Starting Multi-Agent Coordination Demonstration")
    
    # Create coordination configuration
    coord_config = CoordinationConfig(
        max_concurrent_agents=5,
        default_isolation_level=IsolationLevel.STRICT,
        resource_timeout=300,
        data_share_timeout=600,
        enable_cross_agent_communication=True,
        audit_all_operations=True
    )
    
    # Initialize multi-agent coordinator
    coordinator = MultiAgentCoordinator(
        config=coord_config,
        session_pool_config=SessionPoolConfig(max_sessions=3)
    )
    
    # Start cleanup task
    await coordinator.start_cleanup_task()
    
    # Create and register agents
    agents = [
        MockStrandsAgent("WebScraper"),
        MockStrandsAgent("DataProcessor"),
        MockStrandsAgent("ReportGenerator")
    ]
    
    agent_ids = []
    for agent in agents:
        agent_id = await coordinator.register_agent(
            agent,
            isolation_level=IsolationLevel.STRICT,
            security_clearance="high"
        )
        agent_ids.append(agent_id)
        logger.info(f"Registered agent: {agent.name} -> {agent_id}")
    
    # Define agent tasks
    def scrape_data(**kwargs) -> Dict[str, Any]:
        """Mock web scraping task."""
        return {
            "scraped_data": {
                "customers": ["Alice", "Bob", "Charlie"],
                "orders": [1001, 1002, 1003],
                "revenue": 15000
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def process_data(shared_data=None, **kwargs) -> Dict[str, Any]:
        """Mock data processing task."""
        if shared_data:
            processed_count = len(shared_data.get("customers", []))
        else:
            processed_count = 0
        
        return {
            "processed_records": processed_count,
            "processing_time": 2.5,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_report(processed_data=None, **kwargs) -> Dict[str, Any]:
        """Mock report generation task."""
        return {
            "report_generated": True,
            "report_id": "RPT-2024-001",
            "timestamp": datetime.now().isoformat()
        }
    
    # Create task configurations
    task_configs = [
        create_agent_task_config(
            agent_id=agent_ids[0],
            task_func=scrape_data,
            require_resources=[("browser_session_1", ResourceType.BROWSER_SESSION)],
            stop_on_failure=True
        ),
        create_agent_task_config(
            agent_id=agent_ids[1],
            task_func=process_data,
            require_resources=[("compute_slot_1", ResourceType.COMPUTE_SLOT)],
            stop_on_failure=True
        ),
        create_agent_task_config(
            agent_id=agent_ids[2],
            task_func=generate_report,
            require_resources=[("compute_slot_2", ResourceType.COMPUTE_SLOT)],
            stop_on_failure=False
        )
    ]
    
    # Execute tasks in parallel
    logger.info("▶️ Executing agent tasks in parallel...")
    results = await coordinator.coordinate_agents(
        task_configs,
        coordination_strategy="parallel"
    )
    
    # Process results
    successful_tasks = [r for r in results if r.success]
    failed_tasks = [r for r in results if not r.success]
    
    logger.info(f"✅ Successful tasks: {len(successful_tasks)}")
    logger.info(f"❌ Failed tasks: {len(failed_tasks)}")
    
    # Demonstrate data sharing
    if successful_tasks:
        logger.info("📤 Demonstrating secure data sharing...")
        
        # Share data from first agent to others
        share_id = await coordinator.share_data_between_agents(
            data_key="scraped_customer_data",
            data=successful_tasks[0].data,
            owner_agent_id=agent_ids[0],
            target_agent_ids=agent_ids[1:],
            permissions=create_data_share_permissions({
                agent_ids[1]: ["read", "write"],
                agent_ids[2]: ["read"]
            }),
            expires_in=300  # 5 minutes
        )
        
        logger.info(f"📋 Data share created: {share_id}")
        
        # Access shared data
        for i, agent_id in enumerate(agent_ids[1:], 1):
            shared_data = await coordinator.get_shared_data(share_id, agent_id)
            if shared_data:
                logger.info(f"✅ Agent {i+1} successfully accessed shared data")
            else:
                logger.warning(f"❌ Agent {i+1} failed to access shared data")
    
    # Show agent statuses
    logger.info("📊 Agent Status Summary:")
    for agent_id in agent_ids:
        status = coordinator.get_agent_status(agent_id)
        if status:
            logger.info(f"  {status['agent_name']}: {status['status']} (Resources: {len(status['assigned_resources'])})")
    
    # Cleanup
    await coordinator.shutdown()


async def demonstrate_checkpoint_recovery():
    """Demonstrate checkpoint and recovery mechanisms."""
    logger.info("💾 Starting Checkpoint Recovery Demonstration")
    
    # Initialize orchestrator
    orchestrator = SecureWorkflowOrchestrator(
        checkpoint_storage_path="./demo_checkpoints"
    )
    
    # Register a mock agent
    test_agent = MockStrandsAgent("TestAgent")
    orchestrator.register_agent(test_agent)
    
    # Create a simple workflow
    steps = [
        create_workflow_step(
            step_id="step1",
            name="First Step",
            description="First step that will succeed",
            agent_name="TestAgent",
            tool_name="mock_tool",
            action="test_action",
            parameters={"test": "value1"},
            checkpoint_after=True
        ),
        create_workflow_step(
            step_id="step2",
            name="Second Step",
            description="Second step that will create checkpoint",
            agent_name="TestAgent",
            tool_name="mock_tool",
            action="test_action",
            parameters={"test": "value2"},
            depends_on=["step1"],
            checkpoint_after=True
        )
    ]
    
    workflow = create_secure_workflow(
        workflow_id="checkpoint_test_workflow",
        name="Checkpoint Test Workflow",
        description="Workflow to test checkpoint functionality",
        steps=steps,
        auto_checkpoint=True
    )
    
    # Execute workflow (this will create checkpoints)
    logger.info("▶️ Executing workflow with checkpoints...")
    result = await orchestrator.execute_workflow(workflow)
    
    if result.success:
        logger.info("✅ Workflow completed with checkpoints created")
        
        # Get workflow state to show checkpoints
        workflow_state = orchestrator.get_workflow_status(workflow.workflow_id)
        if workflow_state and 'state' in workflow_state:
            checkpoints = workflow_state['state'].get('checkpoints', [])
            logger.info(f"📋 Checkpoints created: {len(checkpoints)}")
            
            # Demonstrate recovery (in a real scenario, this would be after a failure)
            if checkpoints:
                logger.info("🔄 Demonstrating checkpoint recovery...")
                recovery_result = await orchestrator.recover_from_checkpoint(checkpoints[-1])
                
                if recovery_result.success:
                    logger.info("✅ Successfully recovered from checkpoint")
                else:
                    logger.error(f"❌ Checkpoint recovery failed: {recovery_result.error}")
    
    # Cleanup
    await orchestrator.shutdown()


async def main():
    """Main demonstration function."""
    logger.info("🎯 Strands Workflow Orchestration & Multi-Agent Coordination Demo")
    logger.info("=" * 70)
    
    try:
        # Run demonstrations
        await demonstrate_workflow_orchestration()
        logger.info("\n" + "=" * 70)
        
        await demonstrate_multi_agent_coordination()
        logger.info("\n" + "=" * 70)
        
        await demonstrate_checkpoint_recovery()
        logger.info("\n" + "=" * 70)
        
        logger.info("🎉 All demonstrations completed successfully!")
    
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())