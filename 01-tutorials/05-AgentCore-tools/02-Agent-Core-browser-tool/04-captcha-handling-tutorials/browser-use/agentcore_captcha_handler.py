#!/usr/bin/env python3
"""
Production AgentCore Browser Tool CAPTCHA Handler

This is the production-ready implementation for CAPTCHA detection and handling
using Amazon Bedrock AgentCore Browser Tool. This implementation assumes
AgentCore Browser Tool is properly configured in AWS Console.

Prerequisites:
1. AgentCore Browser Tool created in AWS Console
2. Proper AWS credentials configured
3. Required packages installed (see requirements.txt)

Usage:
    handler = AgentCoreCaptchaHandler(region="us-east-1")
    await handler.initialize()
    result = await handler.detect_captcha("https://example.com")
    await handler.cleanup()
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass

# AgentCore Browser Tool SDK
from bedrock_agentcore.tools import BrowserClient

# Playwright for browser automation
from playwright.async_api import async_playwright, Browser, Page, Playwright

# AWS Bedrock for AI analysis
import boto3
from botocore.exceptions import ClientError

# Utilities
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn


@dataclass
class CaptchaDetectionResult:
    """Result of CAPTCHA detection operation"""
    url: str
    session_id: str
    detected_captchas: List[Dict[str, Any]]
    page_screenshot: Optional[str]
    page_title: str
    timestamp: float
    success: bool
    error_message: Optional[str] = None


class AgentCoreCaptchaHandler:
    """
    Production-ready CAPTCHA handler using Amazon Bedrock AgentCore Browser Tool
    
    This class provides enterprise-grade CAPTCHA detection and handling capabilities
    using AWS managed browser infrastructure.
    """
    
    def __init__(self, region: str = "us-east-1", session_timeout: int = 3600):
        """
        Initialize the AgentCore CAPTCHA handler
        
        Args:
            region: AWS region for AgentCore Browser Tool
            session_timeout: Browser session timeout in seconds
        """
        self.region = region
        self.session_timeout = session_timeout
        self.console = Console()
        
        # AgentCore Browser Tool components
        self.browser_client = BrowserClient(region=region)
        self.session_id = None
        
        # Playwright components
        self.playwright = None
        self.browser = None
        self.page = None
        
        # AWS Bedrock for AI analysis
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=region)
        
        # CAPTCHA detection patterns
        self.captcha_selectors = {
            'recaptcha_v2': [
                'iframe[src*="recaptcha/api2/anchor"]',
                '.g-recaptcha[data-sitekey]',
                'iframe[title="reCAPTCHA"]',
                '#recaptcha-anchor'
            ],
            'recaptcha_v3': [
                '.g-recaptcha[data-sitekey][data-size="invisible"]',
                'script[src*="recaptcha/releases/"]'
            ],
            'hcaptcha': [
                'iframe[src*="hcaptcha.com"]',
                '.h-captcha[data-sitekey]',
                'iframe[title*="hCaptcha"]',
                '.hcaptcha-box'
            ],
            'cloudflare_turnstile': [
                '.cf-turnstile',
                'iframe[src*="challenges.cloudflare.com"]'
            ],
            'image_captcha': [
                'img[src*="captcha"]',
                'img[alt*="CAPTCHA" i]',
                '.captcha-image',
                'img[src*="verification"]'
            ],
            'text_captcha': [
                'input[name*="captcha" i]',
                'input[placeholder*="captcha" i]',
                '.captcha-input',
                'input[aria-label*="captcha" i]'
            ]
        }
    
    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize the AgentCore browser session and Playwright connection
        
        Returns:
            Dict containing initialization status and session details
        """
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                
                # Start AgentCore browser session
                task1 = progress.add_task("Starting AgentCore browser session...", total=None)
                self.session_id = self.browser_client.start(
                    name="captcha-detection-session",
                    session_timeout_seconds=self.session_timeout
                )
                progress.update(task1, completed=True)
                
                # Get WebSocket connection details
                task2 = progress.add_task("Getting browser connection details...", total=None)
                ws_url, headers = self.browser_client.generate_ws_headers()
                progress.update(task2, completed=True)
                
                # Initialize Playwright
                task3 = progress.add_task("Connecting Playwright to AgentCore browser...", total=None)
                self.playwright = await async_playwright().start()
                
                # Connect to the remote AgentCore browser
                self.browser = await self.playwright.chromium.connect_over_cdp(
                    endpoint_url=ws_url,
                    headers=headers
                )
                
                # Get or create page
                contexts = self.browser.contexts
                if contexts and contexts[0].pages:
                    self.page = contexts[0].pages[0]
                else:
                    context = await self.browser.new_context(
                        viewport={'width': 1920, 'height': 1080},
                        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                    )
                    self.page = await context.new_page()
                
                progress.update(task3, completed=True)
            
            self.console.print(f"✅ [green]AgentCore browser session initialized: {self.session_id}[/green]")
            
            return {
                "session_id": self.session_id,
                "ws_url": ws_url,
                "status": "initialized",
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.console.print(f"❌ [red]Initialization failed: {e}[/red]")
            raise
    
    async def detect_captcha(self, url: str, take_screenshot: bool = True) -> CaptchaDetectionResult:
        """
        Detect CAPTCHAs on the specified URL
        
        Args:
            url: URL to analyze for CAPTCHAs
            take_screenshot: Whether to take screenshots of detected CAPTCHAs
            
        Returns:
            CaptchaDetectionResult with detection details
        """
        if not self.page:
            raise RuntimeError("Browser not initialized. Call initialize() first.")
        
        try:
            self.console.print(f"🌐 [blue]Navigating to: {url}[/blue]")
            
            # Navigate to the URL
            await self.page.goto(url, wait_until="networkidle", timeout=30000)
            
            # Wait for page to fully load
            await asyncio.sleep(2)
            
            self.console.print("🔍 [yellow]Scanning for CAPTCHA elements...[/yellow]")
            
            detected_captchas = []
            
            # Check for each CAPTCHA type
            for captcha_type, selectors in self.captcha_selectors.items():
                for selector in selectors:
                    try:
                        elements = await self.page.query_selector_all(selector)
                        if elements:
                            self.console.print(f"✅ [green]Found {captcha_type}: {selector}[/green]")
                            
                            captcha_info = {
                                "type": captcha_type,
                                "selector": selector,
                                "element_count": len(elements),
                                "screenshot": None
                            }
                            
                            # Take screenshot of the CAPTCHA element
                            if take_screenshot:
                                try:
                                    screenshot_path = f"captcha_{captcha_type}_{int(time.time())}.png"
                                    await elements[0].screenshot(path=screenshot_path)
                                    captcha_info["screenshot"] = screenshot_path
                                    self.console.print(f"📸 [cyan]Screenshot saved: {screenshot_path}[/cyan]")
                                except Exception as e:
                                    self.console.print(f"⚠️ [yellow]Screenshot failed: {e}[/yellow]")
                            
                            detected_captchas.append(captcha_info)
                            break  # Found this type, move to next
                    
                    except Exception as e:
                        # Continue checking other selectors
                        continue
            
            # Take full page screenshot
            page_screenshot = None
            if take_screenshot:
                try:
                    page_screenshot = f"page_screenshot_{int(time.time())}.png"
                    await self.page.screenshot(path=page_screenshot, full_page=True)
                    self.console.print(f"📸 [cyan]Page screenshot saved: {page_screenshot}[/cyan]")
                except Exception as e:
                    self.console.print(f"⚠️ [yellow]Page screenshot failed: {e}[/yellow]")
            
            # Get page title
            page_title = await self.page.title()
            
            result = CaptchaDetectionResult(
                url=url,
                session_id=self.session_id,
                detected_captchas=detected_captchas,
                page_screenshot=page_screenshot,
                page_title=page_title,
                timestamp=time.time(),
                success=True
            )
            
            if detected_captchas:
                self.console.print(f"🎯 [green]Detected {len(detected_captchas)} CAPTCHA(s)[/green]")
            else:
                self.console.print("ℹ️ [blue]No CAPTCHAs detected[/blue]")
            
            return result
            
        except Exception as e:
            self.console.print(f"❌ [red]CAPTCHA detection failed: {e}[/red]")
            return CaptchaDetectionResult(
                url=url,
                session_id=self.session_id or "unknown",
                detected_captchas=[],
                page_screenshot=None,
                page_title="",
                timestamp=time.time(),
                success=False,
                error_message=str(e)
            )
    
    async def get_live_view_url(self) -> Optional[str]:
        """
        Get the live view URL for real-time browser visualization
        
        Returns:
            Live view URL or None if not available
        """
        try:
            if self.browser_client and self.session_id:
                live_view_url = self.browser_client.generate_live_view_url(expires=3600)
                self.console.print(f"👁️ [cyan]Live view URL: {live_view_url}[/cyan]")
                return live_view_url
        except Exception as e:
            self.console.print(f"⚠️ [yellow]Live view URL generation failed: {e}[/yellow]")
        return None
    
    async def take_control(self) -> None:
        """Enable manual control of the browser session"""
        try:
            if self.browser_client:
                self.browser_client.take_control()
                self.console.print("🎮 [green]Manual control enabled[/green]")
        except Exception as e:
            self.console.print(f"❌ [red]Take control failed: {e}[/red]")
    
    async def release_control(self) -> None:
        """Release manual control back to automation"""
        try:
            if self.browser_client:
                self.browser_client.release_control()
                self.console.print("🤖 [green]Control released to automation[/green]")
        except Exception as e:
            self.console.print(f"❌ [red]Release control failed: {e}[/red]")
    
    async def cleanup(self) -> None:
        """
        Clean up all resources and stop the browser session
        """
        try:
            self.console.print("🧹 [yellow]Cleaning up resources...[/yellow]")
            
            # Close Playwright resources
            if self.page:
                await self.page.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            
            # Stop AgentCore browser session
            if self.browser_client and self.session_id:
                self.browser_client.stop()
                self.console.print(f"✅ [green]AgentCore browser session stopped: {self.session_id}[/green]")
            
        except Exception as e:
            self.console.print(f"⚠️ [yellow]Cleanup warning: {e}[/yellow]")


async def main():
    """
    Example usage of the AgentCore CAPTCHA handler
    """
    console = Console()
    console.print("🎯 [bold blue]AgentCore Browser Tool CAPTCHA Detection Demo[/bold blue]")
    console.print("=" * 60)
    
    # Initialize the handler
    handler = AgentCoreCaptchaHandler(region="us-east-1")
    
    try:
        # Initialize the browser session
        init_result = await handler.initialize()
        console.print(f"📊 [green]Initialization result: {init_result}[/green]")
        
        # Get live view URL
        live_view_url = await handler.get_live_view_url()
        if live_view_url:
            console.print(f"👁️ [cyan]Live view available at: {live_view_url}[/cyan]")
        
        # Test CAPTCHA detection on various sites
        test_urls = [
            "https://www.google.com/recaptcha/api2/demo",
            "https://accounts.hcaptcha.com/demo"
        ]
        
        for url in test_urls:
            console.print(f"\n🧪 [yellow]Testing: {url}[/yellow]")
            result = await handler.detect_captcha(url)
            
            if result.success:
                console.print(f"✅ [green]Detection completed for {result.page_title}[/green]")
                for captcha in result.detected_captchas:
                    console.print(f"  🎯 Found: {captcha['type']} ({captcha['element_count']} elements)")
            else:
                console.print(f"❌ [red]Detection failed: {result.error_message}[/red]")
        
        console.print("\n✅ [green]Demo completed successfully![/green]")
        
    except Exception as e:
        console.print(f"❌ [red]Demo failed: {e}[/red]")
        
    finally:
        # Always cleanup
        await handler.cleanup()


if __name__ == "__main__":
    asyncio.run(main())