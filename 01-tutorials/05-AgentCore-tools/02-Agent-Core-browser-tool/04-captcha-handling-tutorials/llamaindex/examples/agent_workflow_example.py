"""
LlamaIndex Agent Workflow Example for CAPTCHA Handling

This example demonstrates how to use LlamaIndex ReActAgent with CAPTCHA tools
for automated workflow execution.
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

# LlamaIndex imports
from llama_index.core.agent import ReActAgent
from llama_index.llms.bedrock import Bedrock
from llama_index.core.base.response.schema import Response

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llamaindex_captcha_tools import CaptchaDetectionTool, CaptchaSolvingTool


class CaptchaAgentWorkflow:
    """LlamaIndex agent workflow for CAPTCHA handling"""
    
    def __init__(self):
        """Initialize the CAPTCHA agent workflow"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize mock clients (replace with real clients in production)
        self.agentcore_client = None  # BrowserClient()
        self.bedrock_client = None    # boto3.client('bedrock-runtime')
        
        # Initialize LlamaIndex LLM (mock for tutorial)
        # In production: self.llm = Bedrock(model="anthropic.claude-3-sonnet-20240229-v1:0")
        self.llm = None  # Mock LLM
        
        # Create CAPTCHA tools
        self.detection_tool = CaptchaDetectionTool(self.agentcore_client)
        self.solving_tool = CaptchaSolvingTool(self.bedrock_client, None)  # Mock vision LLM
        
        # Create ReActAgent with CAPTCHA tools
        self.agent = self._create_captcha_agent()
        
        print("✅ CAPTCHA agent workflow initialized")
    
    def _create_captcha_agent(self):
        """Create ReActAgent with CAPTCHA handling capabilities"""
        # Mock agent creation for tutorial
        # In production, would use:
        # return ReActAgent.from_tools(
        #     tools=[self.detection_tool, self.solving_tool],
        #     llm=self.llm,
        #     verbose=True,
        #     system_prompt=self._create_system_prompt()
        # )
        
        return MockReActAgent([self.detection_tool, self.solving_tool])
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for CAPTCHA handling agent"""
        return """
        You are an expert CAPTCHA handling agent with access to specialized tools.
        
        Your capabilities:
        1. Detect various types of CAPTCHAs on web pages using captcha_detector
        2. Solve CAPTCHAs using AI models with captcha_solver
        3. Handle errors gracefully and provide fallback strategies
        
        Workflow for CAPTCHA handling:
        1. Use captcha_detector to scan the page for CAPTCHAs
        2. If CAPTCHAs are found, analyze the type and complexity
        3. Use captcha_solver to solve each detected CAPTCHA
        4. Provide detailed feedback about success or failure
        5. If solving fails, suggest alternative approaches
        
        Always be thorough in your analysis and provide clear explanations
        of your reasoning and actions.
        """
    
    async def execute_captcha_workflow(self, page_url: str, task_description: str = None) -> Dict[str, Any]:
        """
        Execute complete CAPTCHA handling workflow
        
        Args:
            page_url: URL to process
            task_description: Optional description of the task
            
        Returns:
            Workflow execution results
        """
        print(f"\n🤖 Executing CAPTCHA workflow for: {page_url}")
        if task_description:
            print(f"📋 Task: {task_description}")
        
        workflow_start = datetime.now()
        
        try:
            # Create agent prompt
            prompt = self._create_workflow_prompt(page_url, task_description)
            
            # Execute workflow with agent
            print("🔄 Agent processing workflow...")
            response = await self._execute_agent_workflow(prompt)
            
            # Process and analyze results
            results = self._process_workflow_results(response, workflow_start)
            
            # Display results
            self._display_workflow_results(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "page_url": page_url,
                "execution_time": (datetime.now() - workflow_start).total_seconds()
            }
    
    def _create_workflow_prompt(self, page_url: str, task_description: str = None) -> str:
        """Create prompt for agent workflow execution"""
        base_prompt = f"""
        Please execute a complete CAPTCHA handling workflow for the page: {page_url}
        
        Steps to follow:
        1. Use captcha_detector to scan the page for any CAPTCHAs
        2. Analyze the detection results and identify CAPTCHA types
        3. For each detected CAPTCHA, use captcha_solver to solve it
        4. Provide a comprehensive summary of the results
        
        Be thorough and provide detailed feedback about each step.
        """
        
        if task_description:
            base_prompt += f"\n\nAdditional context: {task_description}"
        
        return base_prompt
    
    async def _execute_agent_workflow(self, prompt: str) -> Dict[str, Any]:
        """Execute the agent workflow (mock implementation)"""
        # Mock agent execution for tutorial
        # In production, would use: response = await self.agent.achat(prompt)
        
        # Simulate agent workflow execution
        print("  🔍 Step 1: Detecting CAPTCHAs...")
        await asyncio.sleep(0.5)  # Simulate processing time
        
        detection_result = self.detection_tool.call("https://example.com/login")
        
        print("  🧠 Step 2: Analyzing detection results...")
        await asyncio.sleep(0.3)
        
        solving_results = []
        if detection_result.get("captcha_found"):
            print("  🔧 Step 3: Solving detected CAPTCHAs...")
            await asyncio.sleep(1.0)
            
            solving_result = self.solving_tool.call(detection_result)
            solving_results.append(solving_result)
        
        print("  📊 Step 4: Generating summary...")
        await asyncio.sleep(0.2)
        
        return {
            "detection_result": detection_result,
            "solving_results": solving_results,
            "workflow_completed": True
        }
    
    def _process_workflow_results(self, response: Dict[str, Any], start_time: datetime) -> Dict[str, Any]:
        """Process and structure workflow results"""
        execution_time = (datetime.now() - start_time).total_seconds()
        
        detection_result = response.get("detection_result", {})
        solving_results = response.get("solving_results", [])
        
        # Calculate success metrics
        captchas_detected = len(detection_result.get("captcha_types", []))
        captchas_solved = sum(1 for r in solving_results if r.get("success", False))
        
        return {
            "success": response.get("workflow_completed", False),
            "execution_time": execution_time,
            "captchas_detected": captchas_detected,
            "captchas_solved": captchas_solved,
            "success_rate": (captchas_solved / captchas_detected * 100) if captchas_detected > 0 else 0,
            "detection_result": detection_result,
            "solving_results": solving_results,
            "workflow_steps": [
                "CAPTCHA Detection",
                "Result Analysis", 
                "CAPTCHA Solving",
                "Summary Generation"
            ]
        }
    
    def _display_workflow_results(self, results: Dict[str, Any]) -> None:
        """Display workflow execution results"""
        print("\n📊 Workflow Results")
        print("=" * 40)
        
        print(f"✅ Success: {'Yes' if results['success'] else 'No'}")
        print(f"⏱️  Execution Time: {results['execution_time']:.2f}s")
        print(f"🔍 CAPTCHAs Detected: {results['captchas_detected']}")
        print(f"✅ CAPTCHAs Solved: {results['captchas_solved']}")
        print(f"📈 Success Rate: {results['success_rate']:.1f}%")
        
        # Show detection details
        detection = results.get("detection_result", {})
        if detection.get("captcha_found"):
            print(f"\n🎯 Detection Details:")
            print(f"   Types: {', '.join(detection.get('captcha_types', []))}")
            print(f"   Primary: {detection.get('primary_captcha_type', 'unknown')}")
        
        # Show solving details
        solving_results = results.get("solving_results", [])
        if solving_results:
            print(f"\n🔧 Solving Details:")
            for i, result in enumerate(solving_results, 1):
                status = "✅ Success" if result.get("success") else "❌ Failed"
                captcha_type = result.get("captcha_type", "unknown")
                confidence = result.get("confidence_score", 0)
                print(f"   CAPTCHA {i}: {status} ({captcha_type}, {confidence:.2f} confidence)")
        
        print("=" * 40)
    
    async def run_workflow_examples(self) -> None:
        """Run multiple workflow examples"""
        print("🚀 Running LlamaIndex Agent Workflow Examples")
        print("=" * 55)
        
        # Test scenarios
        scenarios = [
            {
                "url": "https://example.com/login",
                "task": "Login to user account"
            },
            {
                "url": "https://example.com/register", 
                "task": "Create new user account"
            },
            {
                "url": "https://example.com/contact",
                "task": "Submit contact form"
            }
        ]
        
        all_results = []
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n🎯 Scenario {i}: {scenario['task']}")
            print("-" * 30)
            
            result = await self.execute_captcha_workflow(
                scenario["url"], 
                scenario["task"]
            )
            all_results.append(result)
        
        # Overall summary
        self._print_overall_summary(all_results)
    
    def _print_overall_summary(self, results: List[Dict[str, Any]]) -> None:
        """Print overall summary of all workflow executions"""
        print("\n📈 Overall Workflow Summary")
        print("=" * 35)
        
        total_workflows = len(results)
        successful_workflows = sum(1 for r in results if r.get("success"))
        total_captchas = sum(r.get("captchas_detected", 0) for r in results)
        total_solved = sum(r.get("captchas_solved", 0) for r in results)
        avg_execution_time = sum(r.get("execution_time", 0) for r in results) / total_workflows
        
        print(f"Total Workflows: {total_workflows}")
        print(f"Successful Workflows: {successful_workflows}")
        print(f"Workflow Success Rate: {(successful_workflows/total_workflows)*100:.1f}%")
        print(f"Total CAPTCHAs Processed: {total_captchas}")
        print(f"Total CAPTCHAs Solved: {total_solved}")
        print(f"CAPTCHA Solving Rate: {(total_solved/total_captchas)*100:.1f}%" if total_captchas > 0 else "N/A")
        print(f"Average Execution Time: {avg_execution_time:.2f}s")
        
        print("=" * 35)


class MockReActAgent:
    """Mock ReActAgent for tutorial purposes"""
    
    def __init__(self, tools):
        self.tools = tools
    
    async def achat(self, prompt: str):
        """Mock agent chat method"""
        return {"response": "Mock agent response", "tools_used": len(self.tools)}


async def main():
    """Main function to run the agent workflow example"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create and run workflow example
        workflow = CaptchaAgentWorkflow()
        await workflow.run_workflow_examples()
        
        print("\n✅ Agent workflow example completed successfully!")
        
    except Exception as e:
        print(f"❌ Example failed: {str(e)}")
        logging.error(f"Workflow example execution failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())

# Note: Enhanced workflow integration would require additional modules:
# from llamaindex_captcha_workflow import (
#     CaptchaAgentWithWorkflow,
#     CaptchaHandlingWorkflow,
#     CaptchaHandlingMiddleware,
#     CaptchaWorkflowState,
#     WorkflowConfig,
#     RetryStrategy
# )
# from complete_workflow_examples import (
#     EnhancedCaptchaWorkflow,
#     CaptchaWorkflowOrchestrator,
#     ProductionCaptchaWorkflowManager
# )