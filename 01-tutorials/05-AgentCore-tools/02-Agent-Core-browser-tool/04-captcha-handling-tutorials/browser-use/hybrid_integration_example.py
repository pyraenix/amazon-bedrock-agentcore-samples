#!/usr/bin/env python3
"""
Hybrid Integration Example: browser-use + AgentCore Browser Tool

This example demonstrates the TRUE INTEGRATION between browser-use CAPTCHA
capabilities and AgentCore Browser Tool managed infrastructure.

Architecture:
- browser-use: Provides CAPTCHA detection and solving algorithms
- AgentCore Browser Tool: Provides managed browser infrastructure
- AWS Bedrock: Provides AI-powered CAPTCHA analysis

Usage:
    python hybrid_integration_example.py
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

# AgentCore Browser Tool SDK
from bedrock_agentcore.tools import BrowserClient
from bedrock_agentcore import BedrockAgentCoreApp, BedrockAgentCoreContext

# browser-use for CAPTCHA logic
from browser_use import Browser

# AWS Bedrock for AI analysis
import boto3
from botocore.exceptions import ClientError

# Rich for beautiful output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn


class HybridCaptchaHandler:
    """
    Hybrid CAPTCHA handler combining browser-use + AgentCore Browser Tool
    
    This demonstrates the integration pattern where:
    1. AgentCore provides managed browser infrastructure
    2. browser-use provides CAPTCHA detection algorithms
    3. Both work together for enterprise-grade CAPTCHA handling
    """
    
    def __init__(self, region: str = "us-east-1"):
        self.console = Console()
        self.region = region
        
        # AgentCore Browser Tool components
        self.agentcore_browser = BrowserClient(region=region)
        self.agentcore_session_id = None
        
        # browser-use components
        self.browser_use_instance = None
        
        # AWS Bedrock for AI analysis
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=region)
        
        # Test sites for demonstration
        self.test_sites = [
            {
                "name": "Google reCAPTCHA v2 Demo",
                "url": "https://www.google.com/recaptcha/api2/demo",
                "expected_captcha": "recaptcha_v2"
            },
            {
                "name": "hCaptcha Demo", 
                "url": "https://accounts.hcaptcha.com/demo",
                "expected_captcha": "hcaptcha"
            }
        ]
    
    async def initialize_hybrid_session(self) -> Dict[str, Any]:
        """
        Initialize the hybrid session combining AgentCore + browser-use
        
        Returns:
            Session information and connection details
        """
        try:
            self.console.print(Panel.fit(
                "🚀 Initializing Hybrid AgentCore + browser-use Session",
                style="bold blue"
            ))
            
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=self.console
            ) as progress:
                
                # Step 1: Start AgentCore managed browser session
                task1 = progress.add_task("Starting AgentCore browser session...", total=100)
                
                self.agentcore_session_id = self.agentcore_browser.start(
                    name="hybrid-captcha-session",
                    session_timeout_seconds=3600
                )
                
                progress.update(task1, advance=50)
                
                # Get AgentCore browser connection details
                ws_url, headers = self.agentcore_browser.generate_ws_headers()
                
                progress.update(task1, advance=100)
                
                # Step 2: Initialize browser-use with AgentCore browser
                task2 = progress.add_task("Connecting browser-use to AgentCore browser...", total=100)
                
                # Configure browser-use to use AgentCore's managed browser
                self.browser_use_instance = Browser(
                    # Connect to AgentCore's managed browser via WebSocket
                    browser_config={
                        "headless": False,  # AgentCore provides live view
                        "ws_endpoint": ws_url,
                        "ws_headers": headers
                    }
                )
                
                await self.browser_use_instance.start()
                
                progress.update(task2, advance=100)
            
            session_info = {
                "agentcore_session_id": self.agentcore_session_id,
                "browser_use_connected": True,
                "ws_url": ws_url,
                "live_view_url": self.agentcore_browser.generate_live_view_url(expires=3600),
                "timestamp": time.time()
            }
            
            self.console.print("✅ [green]Hybrid session initialized successfully![/green]")
            self.console.print(f"📊 AgentCore Session: {self.agentcore_session_id}")
            self.console.print(f"👁️ Live View: {session_info['live_view_url']}")
            
            return session_info
            
        except Exception as e:
            self.console.print(f"❌ [red]Hybrid session initialization failed: {e}[/red]")
            raise
    
    async def detect_captcha_hybrid(self, url: str) -> Dict[str, Any]:
        """
        Detect CAPTCHA using browser-use algorithms on AgentCore infrastructure
        
        This demonstrates the core integration pattern:
        1. Navigate using browser-use (running on AgentCore browser)
        2. Use browser-use's CAPTCHA detection algorithms
        3. Leverage AgentCore's enterprise features (live view, session management)
        
        Args:
            url: URL to analyze for CAPTCHAs
            
        Returns:
            CAPTCHA detection results with hybrid metadata
        """
        if not self.browser_use_instance:
            raise RuntimeError("Hybrid session not initialized")
        
        try:
            self.console.print(f"🌐 [blue]Navigating to: {url}[/blue]")
            self.console.print("🔍 [yellow]Using browser-use CAPTCHA detection on AgentCore infrastructure[/yellow]")
            
            # Use browser-use to navigate and detect CAPTCHAs
            # This runs on AgentCore's managed browser infrastructure
            page = await self.browser_use_instance.new_page()
            await page.goto(url)
            
            # Use browser-use's built-in CAPTCHA detection capabilities
            captcha_results = await self.browser_use_instance.detect_captchas(page)
            
            # Take screenshot using AgentCore capabilities
            screenshot_path = f"hybrid_captcha_{int(time.time())}.png"
            await page.screenshot(path=screenshot_path)
            
            # Get page metadata
            page_title = await page.title()
            page_url = page.url
            
            result = {
                "url": page_url,
                "page_title": page_title,
                "captcha_results": captcha_results,
                "screenshot_path": screenshot_path,
                "detection_method": "browser-use_on_agentcore",
                "agentcore_session_id": self.agentcore_session_id,
                "timestamp": time.time(),
                "success": True
            }
            
            if captcha_results:
                self.console.print(f"🎯 [green]Detected {len(captcha_results)} CAPTCHA(s) using browser-use[/green]")
                for captcha in captcha_results:
                    self.console.print(f"  • {captcha.get('type', 'unknown')} CAPTCHA")
            else:
                self.console.print("ℹ️ [blue]No CAPTCHAs detected[/blue]")
            
            return result
            
        except Exception as e:
            self.console.print(f"❌ [red]Hybrid CAPTCHA detection failed: {e}[/red]")
            return {
                "url": url,
                "success": False,
                "error": str(e),
                "agentcore_session_id": self.agentcore_session_id,
                "timestamp": time.time()
            }
    
    async def solve_captcha_with_bedrock(self, captcha_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve CAPTCHA using AWS Bedrock AI models via AgentCore integration
        
        This demonstrates how to use AgentCore's AI integration capabilities
        for intelligent CAPTCHA solving.
        """
        if not captcha_result.get("success") or not captcha_result.get("captcha_results"):
            return {"status": "no_captcha", "solution": None}
        
        try:
            self.console.print("🤖 [magenta]Analyzing CAPTCHA with AWS Bedrock via AgentCore...[/magenta]")
            
            # Load screenshot for AI analysis
            screenshot_path = captcha_result.get("screenshot_path")
            if not screenshot_path or not Path(screenshot_path).exists():
                raise Exception("Screenshot not available for AI analysis")
            
            with open(screenshot_path, 'rb') as f:
                import base64
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Use Bedrock Claude 3 Sonnet for CAPTCHA analysis
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this CAPTCHA image and provide guidance on how to solve it. Describe what type of CAPTCHA it is and what actions would be needed to complete it."
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_data
                                }
                            }
                        ]
                    }
                ]
            }
            
            # Call Bedrock
            response = self.bedrock_client.invoke_model(
                modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            analysis_text = response_body['content'][0]['text']
            
            return {
                "status": "analyzed",
                "analysis": analysis_text,
                "captcha_count": len(captcha_result.get("captcha_results", [])),
                "model_used": "claude-3-sonnet",
                "agentcore_session": self.agentcore_session_id
            }
            
        except Exception as e:
            self.console.print(f"❌ [red]Bedrock analysis failed: {e}[/red]")
            return {"status": "error", "analysis": f"Analysis failed: {str(e)}"}
    
    async def run_hybrid_demo(self) -> Dict[str, Any]:
        """
        Run complete hybrid demonstration showing AgentCore + browser-use integration
        
        Returns:
            Complete demo results
        """
        try:
            # Initialize hybrid session
            session_info = await self.initialize_hybrid_session()
            
            results = []
            
            # Test CAPTCHA detection on multiple sites
            for site in self.test_sites:
                self.console.print(f"\n🧪 [yellow]Testing: {site['name']}[/yellow]")
                
                # Detect CAPTCHAs using hybrid approach
                detection_result = await self.detect_captcha_hybrid(site['url'])
                
                # Analyze with Bedrock if CAPTCHAs found
                if detection_result.get("success") and detection_result.get("captcha_results"):
                    analysis_result = await self.solve_captcha_with_bedrock(detection_result)
                    detection_result["ai_analysis"] = analysis_result
                
                results.append(detection_result)
                
                # Brief pause between tests
                await asyncio.sleep(2)
            
            # Generate summary report
            report = {
                "session_info": session_info,
                "test_results": results,
                "summary": {
                    "total_sites_tested": len(self.test_sites),
                    "successful_detections": sum(1 for r in results if r.get("success")),
                    "captchas_found": sum(len(r.get("captcha_results", [])) for r in results),
                    "ai_analyses_performed": sum(1 for r in results if r.get("ai_analysis")),
                    "demo_completed_at": time.time()
                }
            }
            
            # Display summary
            self.display_demo_summary(report)
            
            return report
            
        except Exception as e:
            self.console.print(f"❌ [red]Hybrid demo failed: {e}[/red]")
            return {"error": str(e)}
        
        finally:
            await self.cleanup()
    
    def display_demo_summary(self, report: Dict[str, Any]) -> None:
        """Display a beautiful summary of the hybrid demo results"""
        
        summary = report.get("summary", {})
        
        # Create summary table
        table = Table(title="Hybrid Integration Demo Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Sites Tested", str(summary.get("total_sites_tested", 0)))
        table.add_row("Successful Detections", str(summary.get("successful_detections", 0)))
        table.add_row("CAPTCHAs Found", str(summary.get("captchas_found", 0)))
        table.add_row("AI Analyses", str(summary.get("ai_analyses_performed", 0)))
        table.add_row("AgentCore Session", report.get("session_info", {}).get("agentcore_session_id", "N/A"))
        
        self.console.print("\n")
        self.console.print(table)
        
        self.console.print("\n🎉 [green]Hybrid Integration Demo Completed![/green]")
        self.console.print("✅ [green]Successfully demonstrated browser-use + AgentCore Browser Tool integration[/green]")
        
        # Save detailed report
        report_file = f"hybrid_demo_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.console.print(f"📄 [cyan]Detailed report saved to: {report_file}[/cyan]")
    
    async def cleanup(self) -> None:
        """Clean up hybrid session resources"""
        try:
            self.console.print("🧹 [yellow]Cleaning up hybrid session...[/yellow]")
            
            # Close browser-use instance
            if self.browser_use_instance:
                await self.browser_use_instance.close()
            
            # Stop AgentCore browser session
            if self.agentcore_browser and self.agentcore_session_id:
                self.agentcore_browser.stop()
                self.console.print(f"✅ [green]AgentCore session stopped: {self.agentcore_session_id}[/green]")
            
        except Exception as e:
            self.console.print(f"⚠️ [yellow]Cleanup warning: {e}[/yellow]")


async def main():
    """
    Main function demonstrating hybrid AgentCore + browser-use integration
    """
    console = Console()
    
    console.print(Panel.fit(
        "🎯 Hybrid Integration Demo: AgentCore Browser Tool + browser-use",
        style="bold blue"
    ))
    
    console.print("This demo shows how to combine:")
    console.print("• 🏢 AgentCore Browser Tool (managed infrastructure)")
    console.print("• 🧠 browser-use (CAPTCHA detection algorithms)")
    console.print("• 🤖 AWS Bedrock (AI-powered analysis)")
    console.print()
    
    # Run the hybrid demo
    handler = HybridCaptchaHandler(region="us-east-1")
    
    try:
        report = await handler.run_hybrid_demo()
        
        if "error" in report:
            console.print(f"❌ [red]Demo failed: {report['error']}[/red]")
            return 1
        else:
            console.print("\n💡 [cyan]Key Integration Benefits Demonstrated:[/cyan]")
            console.print("   • Enterprise browser infrastructure (AgentCore)")
            console.print("   • Proven CAPTCHA algorithms (browser-use)")
            console.print("   • AI-powered analysis (Bedrock)")
            console.print("   • Live browser viewing and session management")
            console.print("   • Production-ready error handling and monitoring")
            return 0
    
    except KeyboardInterrupt:
        console.print("\n⚠️ [yellow]Demo interrupted by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n❌ [red]Unexpected error: {e}[/red]")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)