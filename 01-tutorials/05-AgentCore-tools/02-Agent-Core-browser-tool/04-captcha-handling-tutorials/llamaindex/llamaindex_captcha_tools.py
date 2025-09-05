"""
LlamaIndex CAPTCHA Tools - Production Implementation

This module provides production-ready LlamaIndex BaseTool implementations for CAPTCHA 
detection and solving using Amazon Bedrock AgentCore Browser Tool.

Prerequisites:
1. AgentCore Browser Tool configured in AWS Console
2. Proper AWS credentials configured
3. Required packages installed (see requirements.txt)
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, Optional, List
import base64
from io import BytesIO

from llama_index.core.tools import BaseTool
from llama_index.core.tools.tool_spec.base import ToolMetadata
from llama_index.multi_modal_llms.bedrock import BedrockMultiModal
from llama_index.core.schema import ImageDocument

import boto3
from PIL import Image
from pydantic import BaseModel

# Note: AgentCore Browser Tool is accessed via AWS API, not local SDK
# The actual browser sessions are managed by AWS infrastructure


class CaptchaDetectionTool(BaseTool):
    """
    Production LlamaIndex tool for detecting CAPTCHAs using AgentCore Browser Tool
    """
    
    metadata = ToolMetadata(
        name="captcha_detector",
        description=(
            "Detects various types of CAPTCHAs on web pages including reCAPTCHA, "
            "hCaptcha, image-based CAPTCHAs, and text-based CAPTCHAs using "
            "Amazon Bedrock AgentCore Browser Tool. Returns detailed information "
            "about detected CAPTCHAs with screenshots and element details."
        )
    )
    
    def __init__(self, region: str = "us-east-1", session_timeout: int = 3600):
        """
        Initialize CAPTCHA detection tool with AgentCore Browser Tool
        
        Args:
            region: AWS region for AgentCore Browser Tool
            session_timeout: Browser session timeout in seconds
        
        Args:
            region: AWS region for AgentCore Browser Tool
            session_timeout: Browser session timeout in seconds
        """
        super().__init__()
        self.region = region
        self.session_timeout = session_timeout
        self.logger = logging.getLogger(__name__)
        
        # AgentCore Browser Tool client (AWS managed service)
        self.agentcore_client = boto3.client('bedrock-agentcore', region_name=region)
        self.session_id = None
        
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
    
    async def _initialize_browser_session(self) -> bool:
        """Initialize AgentCore browser session if not already initialized"""
        if self.session_id and self.page:
            return True
            
        try:
            # Start AgentCore browser session
            self.session_id = self.browser_client.start(
                name="llamaindex-captcha-detection",
                session_timeout_seconds=self.session_timeout
            )
            
            # Get WebSocket connection details
            ws_url, headers = self.browser_client.generate_ws_headers()
            
            # Initialize Playwright
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
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )
                self.page = await context.new_page()
            
            self.logger.info(f"AgentCore browser session initialized: {self.session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize browser session: {e}")
            return False
    
    def call(self, page_url: str) -> Dict[str, Any]:
        """
        Detect CAPTCHAs on the specified page using AgentCore Browser Tool
        
        Args:
            page_url: URL of the page to scan for CAPTCHAs
            
        Returns:
            Dictionary containing CAPTCHA detection results
        """
        # Run async detection in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._detect_captcha_async(page_url))
        finally:
            loop.close()
    
    def __call__(self, page_url: str) -> Dict[str, Any]:
        """LlamaIndex BaseTool requires __call__ method"""
        return self.call(page_url)
    
    async def _detect_captcha_async(self, page_url: str) -> Dict[str, Any]:
        """
        Async implementation of CAPTCHA detection using AgentCore Browser Tool
        """
        try:
            # Initialize browser session if needed
            if not await self._initialize_browser_session():
                return {
                    "page_url": page_url,
                    "captcha_found": False,
                    "error": "Failed to initialize browser session",
                    "error_type": "BrowserInitializationError"
                }
            
            self.logger.info(f"Navigating to: {page_url}")
            
            # Navigate to the URL
            await self.page.goto(page_url, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(2)  # Wait for page to fully load
            
            detected_captchas = []
            
            # Check for each CAPTCHA type
            for captcha_type, selectors in self.captcha_selectors.items():
                for selector in selectors:
                    try:
                        elements = await self.page.query_selector_all(selector)
                        if elements:
                            self.logger.info(f"Found {captcha_type}: {selector}")
                            
                            captcha_info = {
                                "type": captcha_type,
                                "selector": selector,
                                "element_count": len(elements),
                                "screenshot": None
                            }
                            
                            # Take screenshot of the CAPTCHA element
                            try:
                                screenshot_path = f"captcha_{captcha_type}_{int(time.time())}.png"
                                await elements[0].screenshot(path=screenshot_path)
                                captcha_info["screenshot"] = screenshot_path
                                self.logger.info(f"Screenshot saved: {screenshot_path}")
                            except Exception as e:
                                self.logger.warning(f"Screenshot failed: {e}")
                            
                            detected_captchas.append(captcha_info)
                            break  # Found this type, move to next
                    
                    except Exception:
                        # Continue checking other selectors
                        continue
            
            # Take full page screenshot
            page_screenshot = None
            try:
                page_screenshot = f"page_screenshot_{int(time.time())}.png"
                await self.page.screenshot(path=page_screenshot, full_page=True)
                self.logger.info(f"Page screenshot saved: {page_screenshot}")
            except Exception as e:
                self.logger.warning(f"Page screenshot failed: {e}")
            
            # Get page title
            page_title = await self.page.title()
            
            detection_results = {
                "page_url": page_url,
                "session_id": self.session_id,
                "captcha_found": len(detected_captchas) > 0,
                "captcha_types": [c["type"] for c in detected_captchas],
                "elements_detected": detected_captchas,
                "page_screenshot": page_screenshot,
                "page_title": page_title,
                "screenshot_taken": page_screenshot is not None,
                "analysis_timestamp": time.time(),
                "primary_captcha_type": detected_captchas[0]["type"] if detected_captchas else None
            }
            
            self.logger.info(f"CAPTCHA detection completed for {page_url}: {detection_results['captcha_found']}")
            return detection_results
            
        except Exception as e:
            self.logger.error(f"Error detecting CAPTCHA on {page_url}: {str(e)}")
            return {
                "page_url": page_url,
                "session_id": self.session_id,
                "captcha_found": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def acall(self, page_url: str) -> Dict[str, Any]:
        """Async version of CAPTCHA detection"""
        return await self._detect_captcha_async(page_url)
    
    async def cleanup(self) -> None:
        """Clean up browser session and resources"""
        try:
            if self.page:
                await self.page.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            if self.browser_client and self.session_id:
                self.browser_client.stop()
                self.logger.info(f"AgentCore browser session stopped: {self.session_id}")
        except Exception as e:
            self.logger.warning(f"Cleanup warning: {e}")
    
    def get_live_view_url(self) -> Optional[str]:
        """Get live view URL for real-time browser visualization"""
        try:
            if self.browser_client and self.session_id:
                return self.browser_client.generate_live_view_url(expires=3600)
        except Exception as e:
            self.logger.warning(f"Live view URL generation failed: {e}")
        return None


class CaptchaSolvingTool(BaseTool):
    """
    Production LlamaIndex tool for solving CAPTCHAs using Bedrock AI models
    """
    
    metadata = ToolMetadata(
        name="captcha_solver",
        description=(
            "Solves various types of CAPTCHAs using Amazon Bedrock AI vision and "
            "language models. Supports text extraction, image recognition, and "
            "pattern analysis. Returns solution with confidence score and reasoning."
        )
    )
    
    def __init__(self, region: str = "us-east-1", model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"):
        """
        Initialize CAPTCHA solving tool with Bedrock AI models
        
        Args:
            region: AWS region for Bedrock
            model_id: Bedrock model ID for AI analysis
        """
        super().__init__()
        self.region = region
        self.model_id = model_id
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=region)
        self.logger = logging.getLogger(__name__)
    
    def call(self, captcha_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the provided CAPTCHA using Bedrock AI models
        
        Args:
            captcha_data: Dictionary containing CAPTCHA information from detection
            
        Returns:
            Dictionary containing solution results
        """
        # Run async solving in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._solve_captcha_async(captcha_data))
        finally:
            loop.close()
    
    def __call__(self, captcha_data: Dict[str, Any]) -> Dict[str, Any]:
        """LlamaIndex BaseTool requires __call__ method"""
        return self.call(captcha_data)
    
    async def _solve_captcha_async(self, captcha_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async implementation of CAPTCHA solving using Bedrock AI
        """
        try:
            captcha_type = captcha_data.get("primary_captcha_type", "unknown")
            
            # Route to appropriate solving method based on CAPTCHA type
            if captcha_type == "text_captcha":
                return await self._solve_text_captcha(captcha_data)
            elif captcha_type == "image_captcha":
                return await self._solve_image_captcha(captcha_data)
            elif captcha_type in ["recaptcha_v2", "recaptcha_v3", "hcaptcha"]:
                return await self._solve_interactive_captcha(captcha_data)
            else:
                return self._solve_unknown_captcha(captcha_data)
                
        except Exception as e:
            self.logger.error(f"Error solving CAPTCHA: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "captcha_type": captcha_data.get("primary_captcha_type", "unknown")
            }
    
    async def acall(self, captcha_data: Dict[str, Any]) -> Dict[str, Any]:
        """Async version of CAPTCHA solving"""
        return await self._solve_captcha_async(captcha_data)
    
    async def _analyze_image_with_bedrock(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """Analyze image using Bedrock Claude with vision capabilities"""
        try:
            # Read and encode image
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
            
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare request for Claude 3 Sonnet with vision
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            }
            
            # Call Bedrock
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            
            return {
                "success": True,
                "analysis": response_body['content'][0]['text'],
                "model_id": self.model_id,
                "processing_time": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Bedrock image analysis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _solve_text_captcha(self, captcha_data: Dict[str, Any]) -> Dict[str, Any]:
        """Solve text-based CAPTCHA using Bedrock Claude with vision"""
        try:
            # Find screenshot from detected elements
            screenshot_path = None
            for element in captcha_data.get("elements_detected", []):
                if element.get("screenshot") and element.get("type") == "text_captcha":
                    screenshot_path = element["screenshot"]
                    break
            
            if not screenshot_path:
                return {
                    "success": False,
                    "error": "No screenshot available for text CAPTCHA analysis",
                    "captcha_type": "text_captcha"
                }
            
            # Analyze with Bedrock Claude Vision
            prompt = """
            Analyze this text CAPTCHA image and extract the text characters shown.
            
            Instructions:
            1. Look carefully at the distorted text in the image
            2. Identify each character, ignoring visual distortions
            3. Return only the extracted text, no additional explanation
            4. If the text is unclear, provide your best interpretation
            
            Respond with just the text characters you see.
            """
            
            analysis_result = await self._analyze_image_with_bedrock(screenshot_path, prompt)
            
            if analysis_result.get("success"):
                extracted_text = analysis_result["analysis"].strip()
                
                return {
                    "success": True,
                    "solution": extracted_text,
                    "solution_type": "text",
                    "confidence_score": 0.85,  # Could be enhanced with confidence analysis
                    "captcha_type": "text_captcha",
                    "method": "Bedrock Claude 3 Sonnet Vision Analysis",
                    "processing_time": analysis_result.get("processing_time", 0.0),
                    "reasoning": f"Extracted text from CAPTCHA image: {extracted_text}",
                    "screenshot_analyzed": screenshot_path
                }
            else:
                return {
                    "success": False,
                    "error": f"Bedrock analysis failed: {analysis_result.get('error')}",
                    "captcha_type": "text_captcha"
                }
            
        except Exception as e:
            self.logger.error(f"Error solving text CAPTCHA: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "captcha_type": "text_captcha"
            }
    
    async def _solve_image_captcha(self, captcha_data: Dict[str, Any]) -> Dict[str, Any]:
        """Solve image-based CAPTCHA using Bedrock Claude with vision"""
        try:
            # Find screenshot from detected elements
            screenshot_path = None
            for element in captcha_data.get("elements_detected", []):
                if element.get("screenshot") and element.get("type") == "image_captcha":
                    screenshot_path = element["screenshot"]
                    break
            
            if not screenshot_path:
                return {
                    "success": False,
                    "error": "No screenshot available for image CAPTCHA analysis",
                    "captcha_type": "image_captcha"
                }
            
            # Analyze with Bedrock Claude Vision
            prompt = """
            Analyze this image-based CAPTCHA and identify what objects or patterns are shown.
            
            Instructions:
            1. Examine the image carefully for objects, patterns, or specific items
            2. If this is a "select all images with X" type CAPTCHA, identify what X represents
            3. Describe what you see in the image that matches the CAPTCHA criteria
            4. Provide specific details about objects, their positions, or characteristics
            
            Describe what you observe in the CAPTCHA image and what should be selected.
            """
            
            analysis_result = await self._analyze_image_with_bedrock(screenshot_path, prompt)
            
            if analysis_result.get("success"):
                analysis_text = analysis_result["analysis"]
                
                return {
                    "success": True,
                    "solution": analysis_text,
                    "solution_type": "image_analysis",
                    "confidence_score": 0.78,  # Could be enhanced with confidence analysis
                    "captcha_type": "image_captcha",
                    "method": "Bedrock Claude 3 Sonnet Vision Analysis",
                    "processing_time": analysis_result.get("processing_time", 0.0),
                    "reasoning": f"Image analysis: {analysis_text}",
                    "screenshot_analyzed": screenshot_path
                }
            else:
                return {
                    "success": False,
                    "error": f"Bedrock analysis failed: {analysis_result.get('error')}",
                    "captcha_type": "image_captcha"
                }
            
        except Exception as e:
            self.logger.error(f"Error solving image CAPTCHA: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "captcha_type": "image_captcha"
            }
    
    async def _solve_interactive_captcha(self, captcha_data: Dict[str, Any]) -> Dict[str, Any]:
        """Solve interactive CAPTCHAs (reCAPTCHA, hCaptcha) using Bedrock Claude with vision"""
        try:
            captcha_type = captcha_data.get("primary_captcha_type", "recaptcha_v2")
            
            # Find screenshot from detected elements
            screenshot_path = None
            for element in captcha_data.get("elements_detected", []):
                if element.get("screenshot") and element.get("type") in ["recaptcha_v2", "recaptcha_v3", "hcaptcha"]:
                    screenshot_path = element["screenshot"]
                    break
            
            if not screenshot_path:
                # For interactive CAPTCHAs, we might need to handle them differently
                # as they often require user interaction to reveal the challenge
                return {
                    "success": False,
                    "error": f"Interactive {captcha_type} detected but requires user interaction to reveal challenge",
                    "captcha_type": captcha_type,
                    "recommendation": "Manual intervention required for interactive CAPTCHA completion"
                }
            
            # Analyze with Bedrock Claude Vision
            prompt = f"""
            Analyze this {captcha_type} challenge image.
            
            Instructions:
            1. Identify the type of challenge (image grid selection, checkbox, etc.)
            2. If it's an image grid, describe what objects need to be selected
            3. If it's a checkbox challenge, describe what's visible
            4. Provide specific guidance on how to solve this challenge
            5. Note any text instructions visible in the CAPTCHA
            
            Describe the {captcha_type} challenge and provide solution guidance.
            """
            
            analysis_result = await self._analyze_image_with_bedrock(screenshot_path, prompt)
            
            if analysis_result.get("success"):
                analysis_text = analysis_result["analysis"]
                
                return {
                    "success": True,
                    "solution": analysis_text,
                    "solution_type": "interactive_analysis",
                    "confidence_score": 0.70,  # Lower confidence for interactive CAPTCHAs
                    "captcha_type": captcha_type,
                    "method": f"Bedrock Claude 3 Sonnet {captcha_type} Analysis",
                    "processing_time": analysis_result.get("processing_time", 0.0),
                    "reasoning": f"{captcha_type} analysis: {analysis_text}",
                    "screenshot_analyzed": screenshot_path,
                    "requires_interaction": True
                }
            else:
                return {
                    "success": False,
                    "error": f"Bedrock analysis failed: {analysis_result.get('error')}",
                    "captcha_type": captcha_type
                }
            
        except Exception as e:
            self.logger.error(f"Error solving interactive CAPTCHA: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "captcha_type": captcha_data.get("primary_captcha_type", "interactive")
            }
    
    def _solve_unknown_captcha(self, captcha_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle unknown CAPTCHA types"""
        return {
            "success": False,
            "error": "Unknown CAPTCHA type - unable to solve",
            "captcha_type": captcha_data.get("primary_captcha_type", "unknown"),
            "suggested_action": "Manual intervention required",
            "confidence_score": 0.0,
            "method": "Unknown CAPTCHA type handler"
        }


class CaptchaToolSpec:
    """
    Production tool specification for comprehensive CAPTCHA handling using AgentCore Browser Tool
    """
    
    spec_functions = [
        "detect_captcha",
        "solve_captcha",
        "analyze_captcha_image",
        "get_live_view_url",
        "cleanup_session"
    ]
    
    def __init__(self, region: str = "us-east-1", model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"):
        """Initialize production CAPTCHA tool specification"""
        self.detection_tool = CaptchaDetectionTool(region=region)
        self.solving_tool = CaptchaSolvingTool(region=region, model_id=model_id)
        self.logger = logging.getLogger(__name__)
    
    def detect_captcha(self, page_url: str) -> Dict[str, Any]:
        """Detect CAPTCHA on page using AgentCore Browser Tool"""
        return self.detection_tool.call(page_url)
    
    def solve_captcha(self, captcha_data: Dict[str, Any]) -> Dict[str, Any]:
        """Solve detected CAPTCHA using Bedrock AI models"""
        return self.solving_tool.call(captcha_data)
    
    def analyze_captcha_image(self, image_path: str, captcha_type: str = "unknown") -> Dict[str, Any]:
        """Analyze CAPTCHA image using Bedrock Claude with vision"""
        try:
            # Use the solving tool's Bedrock analysis capability
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                prompt = f"""
                Analyze this {captcha_type} CAPTCHA image and provide detailed analysis.
                
                Instructions:
                1. Identify the type of CAPTCHA challenge
                2. Describe what you see in the image
                3. Provide solution guidance if possible
                4. Note any text or instructions visible
                
                Provide comprehensive analysis of this CAPTCHA.
                """
                
                result = loop.run_until_complete(
                    self.solving_tool._analyze_image_with_bedrock(image_path, prompt)
                )
                
                if result.get("success"):
                    return {
                        "success": True,
                        "captcha_type": captcha_type,
                        "analysis": result["analysis"],
                        "method": "Bedrock Claude 3 Sonnet Vision Analysis",
                        "processing_time": result.get("processing_time", 0.0)
                    }
                else:
                    return {
                        "success": False,
                        "error": result.get("error"),
                        "captcha_type": captcha_type
                    }
            finally:
                loop.close()
            
        except Exception as e:
            self.logger.error(f"Error analyzing CAPTCHA image: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "captcha_type": captcha_type
            }
    
    def get_live_view_url(self) -> Optional[str]:
        """Get live view URL for browser session"""
        return self.detection_tool.get_live_view_url()
    
    async def cleanup_session(self) -> None:
        """Clean up browser session and resources"""
        await self.detection_tool.cleanup()


# Production example usage and testing
async def test_production_captcha_tools():
    """Test production CAPTCHA tools functionality with AgentCore Browser Tool"""
    print("🚀 Testing Production LlamaIndex CAPTCHA Tools with AgentCore Browser Tool")
    print("=" * 70)
    
    # Create production tools
    detection_tool = CaptchaDetectionTool(region="us-east-1")
    solving_tool = CaptchaSolvingTool(region="us-east-1")
    
    try:
        # Test detection on a known CAPTCHA site
        test_url = "https://www.google.com/recaptcha/api2/demo"
        print(f"🔍 Testing CAPTCHA detection on: {test_url}")
        
        detection_result = detection_tool.call(test_url)
        print(f"✅ Detection result: {detection_result}")
        
        # Get live view URL if available
        live_view_url = detection_tool.get_live_view_url()
        if live_view_url:
            print(f"👁️ Live view URL: {live_view_url}")
        
        # Test solving if CAPTCHA detected
        if detection_result.get("captcha_found"):
            print("🧠 Attempting to solve detected CAPTCHA...")
            solving_result = solving_tool.call(detection_result)
            print(f"✅ Solving result: {solving_result}")
        else:
            print("ℹ️ No CAPTCHA detected to solve")
        
        return detection_result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None
        
    finally:
        # Clean up resources
        print("🧹 Cleaning up resources...")
        await detection_tool.cleanup()


def test_captcha_tool_spec():
    """Test the complete CAPTCHA tool specification"""
    print("🛠️ Testing CaptchaToolSpec...")
    
    # Create tool spec
    tool_spec = CaptchaToolSpec(region="us-east-1")
    
    # Test detection
    test_url = "https://accounts.hcaptcha.com/demo"
    result = tool_spec.detect_captcha(test_url)
    print(f"Tool spec detection result: {result}")
    
    return result


if __name__ == "__main__":
    # Run production tests
    print("Running production CAPTCHA tools test...")
    asyncio.run(test_production_captcha_tools())
    
    print("\nRunning tool spec test...")
    test_captcha_tool_spec()