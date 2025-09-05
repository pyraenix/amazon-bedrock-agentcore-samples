"""
Production CAPTCHA Detection Example with LlamaIndex + AgentCore Browser Tool

This example demonstrates production-ready CAPTCHA detection using LlamaIndex tools
with real Amazon Bedrock AgentCore Browser Tool integration.

Prerequisites:
1. AgentCore Browser Tool configured in AWS Console
2. AWS credentials configured (aws configure)
3. Bedrock model access enabled
"""

import asyncio
import logging
from typing import Dict, Any

# LlamaIndex imports
from llama_index.core.tools import BaseTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.bedrock import Bedrock

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llamaindex_captcha_tools import CaptchaDetectionTool, CaptchaSolvingTool, CaptchaToolSpec


class ProductionCaptchaExample:
    """Production example of CAPTCHA detection with LlamaIndex + AgentCore Browser Tool"""
    
    def __init__(self, region: str = "us-east-1"):
        """Initialize the production CAPTCHA example"""
        self.region = region
        self.logger = logging.getLogger(__name__)
        
        # Create production CAPTCHA tools
        self.detection_tool = CaptchaDetectionTool(region=region)
        self.solving_tool = CaptchaSolvingTool(region=region)
        
        # Create comprehensive tool spec
        self.tool_spec = CaptchaToolSpec(region=region)
        
        print(f"✅ Production CAPTCHA detection example initialized (region: {region})")
    
    def detect_captcha_simple(self, page_url: str) -> Dict[str, Any]:
        """
        Simple CAPTCHA detection on a page using AgentCore Browser Tool
        
        Args:
            page_url: URL to check for CAPTCHAs
            
        Returns:
            Detection results from AgentCore Browser Tool
        """
        print(f"🔍 Detecting CAPTCHAs on: {page_url}")
        
        try:
            # Use production detection tool
            result = self.detection_tool.call(page_url)
            
            if result.get("captcha_found"):
                print(f"✅ Found {len(result.get('captcha_types', []))} CAPTCHA type(s):")
                for captcha_type in result.get("captcha_types", []):
                    print(f"   - {captcha_type}")
                
                # Get live view URL if available
                live_view_url = self.detection_tool.get_live_view_url()
                if live_view_url:
                    print(f"👁️ Live view: {live_view_url}")
            else:
                print("ℹ️ No CAPTCHAs detected")
            
            return result
            
        except Exception as e:
            print(f"❌ Detection failed: {e}")
            return {"error": str(e), "captcha_found": False}
    
    async def detect_and_solve_workflow(self, page_url: str) -> Dict[str, Any]:
        """
        Complete workflow: detect and attempt to solve CAPTCHAs
        
        Args:
            page_url: URL to process
            
        Returns:
            Complete workflow results
        """
        print(f"🚀 Starting complete CAPTCHA workflow for: {page_url}")
        
        try:
            # Step 1: Detect CAPTCHAs
            detection_result = self.detection_tool.call(page_url)
            
            workflow_result = {
                "page_url": page_url,
                "detection_result": detection_result,
                "solving_results": [],
                "workflow_success": False
            }
            
            if not detection_result.get("captcha_found"):
                print("ℹ️ No CAPTCHAs found - workflow complete")
                workflow_result["workflow_success"] = True
                return workflow_result
            
            print(f"🎯 Found CAPTCHAs, attempting to solve...")
            
            # Step 2: Attempt to solve each detected CAPTCHA
            solving_result = self.solving_tool.call(detection_result)
            workflow_result["solving_results"].append(solving_result)
            
            if solving_result.get("success"):
                print(f"✅ CAPTCHA solved: {solving_result.get('solution')}")
                workflow_result["workflow_success"] = True
            else:
                print(f"❌ CAPTCHA solving failed: {solving_result.get('error')}")
            
            return workflow_result
            
        except Exception as e:
            print(f"❌ Workflow failed: {e}")
            return {
                "page_url": page_url,
                "error": str(e),
                "workflow_success": False
            }
        
        finally:
            # Clean up resources
            await self.detection_tool.cleanup()
    
    def create_llamaindex_agent(self) -> ReActAgent:
        """
        Create a LlamaIndex ReActAgent with CAPTCHA handling capabilities
        
        Returns:
            Configured ReActAgent with CAPTCHA tools
        """
        print("🤖 Creating LlamaIndex ReActAgent with CAPTCHA capabilities...")
        
        # Initialize Bedrock LLM
        llm = Bedrock(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            region_name=self.region
        )
        
        # Create tools list
        tools = [self.detection_tool, self.solving_tool]
        
        # Create ReActAgent
        agent = ReActAgent.from_tools(
            tools=tools,
            llm=llm,
            verbose=True,
            max_iterations=10
        )
        
        print("✅ LlamaIndex ReActAgent created with CAPTCHA tools")
        return agent
    
    async def agent_workflow_example(self, page_url: str) -> str:
        """
        Example of using LlamaIndex agent for CAPTCHA handling
        
        Args:
            page_url: URL to process with the agent
            
        Returns:
            Agent response
        """
        print(f"🤖 Running agent workflow for: {page_url}")
        
        try:
            # Create agent
            agent = self.create_llamaindex_agent()
            
            # Agent prompt
            prompt = f"""
            Please help me analyze the webpage at {page_url} for CAPTCHAs.
            
            Tasks:
            1. Navigate to the page and detect any CAPTCHAs present
            2. If CAPTCHAs are found, analyze them and attempt to solve them
            3. Provide a summary of what you found and any solutions
            4. Include details about the CAPTCHA types and your confidence in the solutions
            
            Please be thorough in your analysis and provide clear explanations.
            """
            
            # Run agent
            response = agent.chat(prompt)
            
            print("✅ Agent workflow completed")
            return str(response)
            
        except Exception as e:
            print(f"❌ Agent workflow failed: {e}")
            return f"Agent workflow failed: {e}"
        
        finally:
            # Clean up resources
            await self.detection_tool.cleanup()


async def main():
    """Main function to run the production CAPTCHA detection examples"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Production LlamaIndex CAPTCHA Detection with AgentCore Browser Tool")
    print("=" * 70)
    
    try:
        # Create production example
        example = ProductionCaptchaExample(region="us-east-1")
        
        # Test URLs with real CAPTCHA sites
        test_urls = [
            "https://www.google.com/recaptcha/api2/demo",
            "https://accounts.hcaptcha.com/demo"
        ]
        
        for url in test_urls:
            print(f"\n{'='*50}")
            print(f"Testing: {url}")
            print('='*50)
            
            # Simple detection
            print("\n1️⃣ Simple Detection Test:")
            detection_result = example.detect_captcha_simple(url)
            
            # Complete workflow
            print("\n2️⃣ Complete Workflow Test:")
            workflow_result = await example.detect_and_solve_workflow(url)
            
            # Agent workflow (optional - requires more setup)
            print("\n3️⃣ Agent Workflow Test:")
            try:
                agent_response = await example.agent_workflow_example(url)
                print(f"Agent response: {agent_response[:200]}...")
            except Exception as e:
                print(f"Agent workflow skipped: {e}")
        
        print("\n✅ Production CAPTCHA detection examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Example failed: {str(e)}")
        logging.error(f"Example execution failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())