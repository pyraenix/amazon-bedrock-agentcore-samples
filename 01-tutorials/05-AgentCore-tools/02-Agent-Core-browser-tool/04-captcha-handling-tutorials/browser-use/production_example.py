#!/usr/bin/env python3
"""
Production Example: AgentCore Browser Tool CAPTCHA Detection

This example demonstrates production-ready usage of Amazon Bedrock AgentCore
Browser Tool for CAPTCHA detection and handling.

Prerequisites:
1. AgentCore Browser Tool created in AWS Console
2. AWS credentials configured with proper permissions
3. All dependencies installed (pip install -r requirements.txt)
4. Playwright browsers installed (playwright install)

Usage:
    python production_example.py
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Import our production handler
from agentcore_captcha_handler import AgentCoreCaptchaHandler, CaptchaDetectionResult

# AgentCore Browser Tool SDK
from bedrock_agentcore.tools import BrowserClient

# Rich for beautiful output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn


class ProductionCaptchaDetectionSuite:
    """
    Production-ready CAPTCHA detection suite using AgentCore Browser Tool
    """
    
    def __init__(self, region: str = "us-east-1"):
        self.console = Console()
        self.region = region
        self.handler = None
        self.results = []
        
        # Production test sites with known CAPTCHA implementations
        self.test_sites = [
            {
                "name": "Google reCAPTCHA v2 Demo",
                "url": "https://www.google.com/recaptcha/api2/demo",
                "expected_captcha": "recaptcha_v2",
                "description": "Standard reCAPTCHA v2 checkbox implementation"
            },
            {
                "name": "hCaptcha Demo",
                "url": "https://accounts.hcaptcha.com/demo",
                "expected_captcha": "hcaptcha",
                "description": "hCaptcha accessibility-focused implementation"
            },
            {
                "name": "Cloudflare Challenge",
                "url": "https://nopecha.com/demo/cloudflare",
                "expected_captcha": "cloudflare_turnstile",
                "description": "Cloudflare Turnstile challenge"
            }
        ]
    
    async def initialize_handler(self) -> bool:
        """
        Initialize the AgentCore CAPTCHA handler
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.console.print(Panel.fit(
                "🚀 Initializing AgentCore Browser Tool",
                style="bold blue"
            ))
            
            self.handler = AgentCoreCaptchaHandler(region=self.region)
            init_result = await self.handler.initialize()
            
            self.console.print(f"✅ [green]Session ID: {init_result['session_id']}[/green]")
            
            # Get live view URL for monitoring
            live_view_url = await self.handler.get_live_view_url()
            if live_view_url:
                self.console.print(f"👁️ [cyan]Live View: {live_view_url}[/cyan]")
            
            return True
            
        except Exception as e:
            self.console.print(f"❌ [red]Initialization failed: {e}[/red]")
            return False
    
    async def run_captcha_detection_suite(self) -> List[CaptchaDetectionResult]:
        """
        Run CAPTCHA detection on all test sites
        
        Returns:
            List of detection results
        """
        if not self.handler:
            raise RuntimeError("Handler not initialized")
        
        results = []
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task(
                "Running CAPTCHA detection suite...",
                total=len(self.test_sites)
            )
            
            for site in self.test_sites:
                self.console.print(f"\n🧪 [yellow]Testing: {site['name']}[/yellow]")
                self.console.print(f"📝 [dim]{site['description']}[/dim]")
                
                try:
                    result = await self.handler.detect_captcha(
                        url=site['url'],
                        take_screenshot=True
                    )
                    
                    # Add test site metadata to result
                    result.expected_captcha = site['expected_captcha']
                    result.site_name = site['name']
                    
                    results.append(result)
                    
                    if result.success:
                        detected_types = [c['type'] for c in result.detected_captchas]
                        if site['expected_captcha'] in detected_types:
                            self.console.print(f"✅ [green]Expected CAPTCHA detected: {site['expected_captcha']}[/green]")
                        else:
                            self.console.print(f"⚠️ [yellow]Expected {site['expected_captcha']}, found: {detected_types}[/yellow]")
                    else:
                        self.console.print(f"❌ [red]Detection failed: {result.error_message}[/red]")
                    
                except Exception as e:
                    self.console.print(f"❌ [red]Test failed: {e}[/red]")
                    # Create error result
                    error_result = CaptchaDetectionResult(
                        url=site['url'],
                        session_id=self.handler.session_id,
                        detected_captchas=[],
                        page_screenshot=None,
                        page_title="",
                        timestamp=__import__('time').time(),
                        success=False,
                        error_message=str(e)
                    )
                    error_result.expected_captcha = site['expected_captcha']
                    error_result.site_name = site['name']
                    results.append(error_result)
                
                progress.update(task, advance=1)
                
                # Brief pause between tests
                await asyncio.sleep(2)
        
        return results
    
    def generate_report(self, results: List[CaptchaDetectionResult]) -> Dict[str, Any]:
        """
        Generate comprehensive test report
        
        Args:
            results: List of detection results
            
        Returns:
            Report dictionary
        """
        # Create summary table
        table = Table(title="CAPTCHA Detection Results")
        table.add_column("Site", style="cyan")
        table.add_column("Expected", style="yellow")
        table.add_column("Detected", style="green")
        table.add_column("Status", style="white")
        table.add_column("Screenshots", style="blue")
        
        successful_detections = 0
        total_captchas_found = 0
        
        for result in results:
            site_name = getattr(result, 'site_name', 'Unknown')
            expected = getattr(result, 'expected_captcha', 'Unknown')
            
            if result.success:
                detected_types = [c['type'] for c in result.detected_captchas]
                detected_str = ', '.join(detected_types) if detected_types else 'None'
                
                if expected in detected_types:
                    status = "✅ PASS"
                    successful_detections += 1
                else:
                    status = "⚠️ PARTIAL" if detected_types else "❌ MISS"
                
                total_captchas_found += len(result.detected_captchas)
                
                # Count screenshots
                screenshot_count = len([c for c in result.detected_captchas if c.get('screenshot')])
                if result.page_screenshot:
                    screenshot_count += 1
                screenshot_str = f"{screenshot_count} files"
            else:
                detected_str = "ERROR"
                status = "❌ FAIL"
                screenshot_str = "0 files"
            
            table.add_row(site_name, expected, detected_str, status, screenshot_str)
        
        self.console.print("\n")
        self.console.print(table)
        
        # Generate summary statistics
        total_tests = len(results)
        success_rate = (successful_detections / total_tests * 100) if total_tests > 0 else 0
        
        summary = {
            "timestamp": __import__('time').time(),
            "total_tests": total_tests,
            "successful_detections": successful_detections,
            "success_rate_percent": round(success_rate, 2),
            "total_captchas_found": total_captchas_found,
            "session_id": self.handler.session_id if self.handler else None,
            "region": self.region
        }
        
        # Print summary
        self.console.print(f"\n📊 [bold]Test Summary:[/bold]")
        self.console.print(f"   • Total Tests: {total_tests}")
        self.console.print(f"   • Successful Detections: {successful_detections}")
        self.console.print(f"   • Success Rate: {success_rate:.1f}%")
        self.console.print(f"   • Total CAPTCHAs Found: {total_captchas_found}")
        
        # Detailed results for JSON report
        detailed_results = []
        for result in results:
            detailed_results.append({
                "site_name": getattr(result, 'site_name', 'Unknown'),
                "url": result.url,
                "expected_captcha": getattr(result, 'expected_captcha', 'Unknown'),
                "detected_captchas": result.detected_captchas,
                "page_title": result.page_title,
                "success": result.success,
                "error_message": result.error_message,
                "timestamp": result.timestamp,
                "page_screenshot": result.page_screenshot
            })
        
        return {
            "summary": summary,
            "detailed_results": detailed_results
        }
    
    async def cleanup(self):
        """Clean up resources"""
        if self.handler:
            await self.handler.cleanup()
    
    async def run_production_suite(self) -> Dict[str, Any]:
        """
        Run the complete production test suite
        
        Returns:
            Complete test report
        """
        try:
            # Initialize
            if not await self.initialize_handler():
                return {"error": "Failed to initialize AgentCore handler"}
            
            # Run tests
            results = await self.run_captcha_detection_suite()
            
            # Generate report
            report = self.generate_report(results)
            
            # Save report
            report_file = f"captcha_detection_report_{int(__import__('time').time())}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.console.print(f"\n📄 [cyan]Detailed report saved to: {report_file}[/cyan]")
            
            return report
            
        except Exception as e:
            self.console.print(f"❌ [red]Production suite failed: {e}[/red]")
            return {"error": str(e)}
        
        finally:
            await self.cleanup()


async def main():
    """
    Main function to run the production example
    """
    console = Console()
    
    console.print(Panel.fit(
        "🎯 AgentCore Browser Tool - Production CAPTCHA Detection Suite",
        style="bold blue"
    ))
    
    # Check if we're in the right directory
    if not Path("agentcore_captcha_handler.py").exists():
        console.print("❌ [red]Error: agentcore_captcha_handler.py not found[/red]")
        console.print("Please run this script from the tutorial directory.")
        sys.exit(1)
    
    # Run the production suite
    suite = ProductionCaptchaDetectionSuite(region="us-east-1")
    
    try:
        report = await suite.run_production_suite()
        
        if "error" in report:
            console.print(f"❌ [red]Suite failed: {report['error']}[/red]")
            sys.exit(1)
        else:
            summary = report.get("summary", {})
            success_rate = summary.get("success_rate_percent", 0)
            
            if success_rate >= 80:
                console.print("\n🎉 [green]Production suite completed successfully![/green]")
                console.print(f"✅ [green]Success rate: {success_rate}%[/green]")
            else:
                console.print("\n⚠️ [yellow]Production suite completed with issues[/yellow]")
                console.print(f"⚠️ [yellow]Success rate: {success_rate}% (below 80% threshold)[/yellow]")
            
            console.print("\n💡 [cyan]Next steps:[/cyan]")
            console.print("   • Review the detailed JSON report")
            console.print("   • Check screenshot files for visual verification")
            console.print("   • Integrate the handler into your production application")
    
    except KeyboardInterrupt:
        console.print("\n⚠️ [yellow]Suite interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n❌ [red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())