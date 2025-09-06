"""
Test fixtures and utilities for LlamaIndex-AgentCore browser integration tests.

This module provides reusable test fixtures, mock objects, and utilities
for consistent testing across all test modules.
"""

import pytest
import json
import tempfile
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock
import base64

# Import components for fixtures
from interfaces import (
    BrowserResponse, ElementSelector, BrowserAction, SessionStatus
)
from vision_models import CaptchaType, CaptchaAnalysisResult
from config import (
    IntegrationConfig, BrowserConfiguration, AWSCredentials, AgentCoreEndpoints
)
from exceptions import AgentCoreBrowserError, BrowserErrorType


class TestDataGenerator:
    """Generates test data for various scenarios."""
    
    @staticmethod
    def create_browser_response(
        success: bool = True,
        data: Optional[Dict] = None,
        error_message: Optional[str] = None,
        session_id: str = "test-session-123",
        operation_id: Optional[str] = None
    ) -> BrowserResponse:
        """Create a mock browser response."""
        return BrowserResponse(
            success=success,
            data=data or {},
            error_message=error_message,
            session_id=session_id,
            timestamp=datetime.utcnow().isoformat(),
            operation_id=operation_id or f"op-{datetime.utcnow().timestamp()}"
        )
    
    @staticmethod
    def create_navigation_response(
        url: str = "https://example.com",
        title: str = "Example Page",
        status_code: int = 200,
        load_time_ms: int = 1500
    ) -> BrowserResponse:
        """Create a navigation response."""
        return TestDataGenerator.create_browser_response(
            data={
                "current_url": url,
                "page_title": title,
                "status_code": status_code,
                "load_time_ms": load_time_ms,
                "page_ready": True,
                "page_state": "complete"
            }
        )
    
    @staticmethod
    def create_text_extraction_response(
        text: str = "Sample extracted text content",
        element_found: bool = True,
        element_count: int = 1,
        extraction_method: str = "full_page"
    ) -> BrowserResponse:
        """Create a text extraction response."""
        return TestDataGenerator.create_browser_response(
            data={
                "text": text,
                "element_found": element_found,
                "element_count": element_count,
                "extraction_method": extraction_method,
                "element_info": {"tag": "div", "id": "content"}
            }
        )
    
    @staticmethod
    def create_screenshot_response(
        format: str = "png",
        width: int = 1920,
        height: int = 1080,
        element_found: bool = True
    ) -> BrowserResponse:
        """Create a screenshot response."""
        # Create a minimal 1x1 PNG image in base64
        screenshot_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        return TestDataGenerator.create_browser_response(
            data={
                "screenshot_data": screenshot_data,
                "format": format,
                "size": {"width": width, "height": height},
                "element_found": element_found,
                "element_info": {"tag": "body"}
            }
        )
    
    @staticmethod
    def create_click_response(
        element_found: bool = True,
        click_successful: bool = True,
        page_changed: bool = True,
        new_url: Optional[str] = None
    ) -> BrowserResponse:
        """Create a click response."""
        return TestDataGenerator.create_browser_response(
            data={
                "element_found": element_found,
                "click_successful": click_successful,
                "page_changed": page_changed,
                "new_url": new_url or "https://example.com/clicked",
                "response_time_ms": 500,
                "element_info": {"tag": "button", "text": "Click me"}
            }
        )
    
    @staticmethod
    def create_type_response(
        text: str = "test input",
        element_found: bool = True,
        field_cleared: bool = True,
        element_type: str = "input"
    ) -> BrowserResponse:
        """Create a type text response."""
        return TestDataGenerator.create_browser_response(
            data={
                "element_found": element_found,
                "text_entered": text,
                "field_cleared": field_cleared,
                "element_type": element_type,
                "typing_time_ms": len(text) * 50,
                "element_info": {"tag": "input", "type": "text"}
            }
        )
    
    @staticmethod
    def create_captcha_analysis(
        detected: bool = True,
        captcha_type: CaptchaType = CaptchaType.RECAPTCHA_V2,
        confidence_score: float = 0.95,
        solution: Optional[str] = None
    ) -> CaptchaAnalysisResult:
        """Create a CAPTCHA analysis result."""
        return CaptchaAnalysisResult(
            captcha_detected=detected,
            captcha_type=captcha_type,
            confidence_score=confidence_score,
            solution=solution,
            solution_confidence=0.9 if solution else None,
            challenge_text="Please complete the CAPTCHA",
            visual_elements=[{"type": "checkbox", "location": {"x": 100, "y": 200}}],
            processing_time_ms=1500,
            model_used="anthropic.claude-3-sonnet-20240229-v1:0"
        )
    
    @staticmethod
    def create_test_image_data() -> bytes:
        """Create test image data."""
        # 1x1 PNG image in base64
        base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        return base64.b64decode(base64_data)
    
    @staticmethod
    def create_test_config() -> Dict[str, Any]:
        """Create a test configuration dictionary."""
        return {
            "aws_credentials": {
                "region": "us-east-1",
                "access_key_id": "AKIA...",
                "secret_access_key": "test-secret"
            },
            "agentcore_endpoints": {
                "browser_tool_url": "https://test-agentcore.amazonaws.com",
                "api_version": "v1"
            },
            "browser_config": {
                "headless": True,
                "viewport_width": 1920,
                "viewport_height": 1080,
                "timeout_seconds": 30,
                "enable_javascript": True,
                "enable_images": True,
                "enable_cookies": True
            },
            "llm_model": "anthropic.claude-3-sonnet-20240229-v1:0",
            "vision_model": "anthropic.claude-3-sonnet-20240229-v1:0",
            "test_mode": True
        }
    
    @staticmethod
    def create_test_urls() -> List[str]:
        """Create a list of test URLs."""
        return [
            "https://example.com",
            "https://httpbin.org/html",
            "https://httpbin.org/json",
            "https://httpbin.org/forms/post",
            "https://httpbin.org/status/200",
            "https://httpbin.org/delay/1"
        ]
    
    @staticmethod
    def create_captcha_test_urls() -> List[str]:
        """Create a list of CAPTCHA test URLs."""
        return [
            "https://www.google.com/recaptcha/api2/demo",
            "https://hcaptcha.com/1/demo",
            "https://funcaptcha.com/demo"
        ]


class MockBrowserClient:
    """Mock browser client for testing."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or TestDataGenerator.create_test_config()
        self.session_id = None
        self.responses = {}
        self.call_history = []
    
    def set_response(self, method: str, response: BrowserResponse):
        """Set mock response for a method."""
        self.responses[method] = response
    
    def get_call_history(self) -> List[Dict[str, Any]]:
        """Get history of method calls."""
        return self.call_history.copy()
    
    def clear_call_history(self):
        """Clear call history."""
        self.call_history.clear()
    
    async def create_session(self) -> str:
        """Mock create session."""
        self.call_history.append({"method": "create_session", "args": [], "kwargs": {}})
        
        if "create_session" in self.responses:
            response = self.responses["create_session"]
            if response.success:
                self.session_id = response.data.get("session_id", "mock-session-123")
                return self.session_id
            else:
                raise AgentCoreBrowserError(response.error_message or "Session creation failed")
        
        self.session_id = "mock-session-123"
        return self.session_id
    
    async def navigate(self, url: str, wait_for_load: bool = True, timeout: int = None) -> BrowserResponse:
        """Mock navigate method."""
        self.call_history.append({
            "method": "navigate",
            "args": [url],
            "kwargs": {"wait_for_load": wait_for_load, "timeout": timeout}
        })
        
        if "navigate" in self.responses:
            return self.responses["navigate"]
        
        return TestDataGenerator.create_navigation_response(url=url)
    
    async def extract_text(self, element_selector=None) -> BrowserResponse:
        """Mock extract_text method."""
        self.call_history.append({
            "method": "extract_text",
            "args": [],
            "kwargs": {"element_selector": element_selector}
        })
        
        if "extract_text" in self.responses:
            return self.responses["extract_text"]
        
        return TestDataGenerator.create_text_extraction_response()
    
    async def take_screenshot(self, element_selector=None, full_page=False) -> BrowserResponse:
        """Mock take_screenshot method."""
        self.call_history.append({
            "method": "take_screenshot",
            "args": [],
            "kwargs": {"element_selector": element_selector, "full_page": full_page}
        })
        
        if "take_screenshot" in self.responses:
            return self.responses["take_screenshot"]
        
        return TestDataGenerator.create_screenshot_response()
    
    async def click_element(self, element_selector, wait_for_response=True, timeout=None) -> BrowserResponse:
        """Mock click_element method."""
        self.call_history.append({
            "method": "click_element",
            "args": [element_selector],
            "kwargs": {"wait_for_response": wait_for_response, "timeout": timeout}
        })
        
        if "click_element" in self.responses:
            return self.responses["click_element"]
        
        return TestDataGenerator.create_click_response()
    
    async def type_text(self, element_selector, text, clear_first=True, typing_delay=None) -> BrowserResponse:
        """Mock type_text method."""
        self.call_history.append({
            "method": "type_text",
            "args": [element_selector, text],
            "kwargs": {"clear_first": clear_first, "typing_delay": typing_delay}
        })
        
        if "type_text" in self.responses:
            return self.responses["type_text"]
        
        return TestDataGenerator.create_type_response(text=text)
    
    async def close_session(self) -> BrowserResponse:
        """Mock close_session method."""
        self.call_history.append({"method": "close_session", "args": [], "kwargs": {}})
        
        if "close_session" in self.responses:
            response = self.responses["close_session"]
            if response.success:
                self.session_id = None
            return response
        
        self.session_id = None
        return TestDataGenerator.create_browser_response(
            data={"session_closed": True, "cleanup_successful": True}
        )


class TestConfigManager:
    """Manages test configurations and temporary files."""
    
    def __init__(self):
        self.temp_files = []
    
    def create_temp_config_file(self, config: Optional[Dict] = None) -> str:
        """Create a temporary configuration file."""
        config = config or TestDataGenerator.create_test_config()
        
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        
        with temp_file as f:
            json.dump(config, f, indent=2)
        
        self.temp_files.append(temp_file.name)
        return temp_file.name
    
    def cleanup(self):
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except OSError:
                pass  # File may already be deleted
        self.temp_files.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# Pytest fixtures
@pytest.fixture
def test_data():
    """Provide test data generator."""
    return TestDataGenerator()


@pytest.fixture
def mock_browser_client():
    """Provide mock browser client."""
    return MockBrowserClient()


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return TestDataGenerator.create_test_config()


@pytest.fixture
def test_urls():
    """Provide test URLs."""
    return TestDataGenerator.create_test_urls()


@pytest.fixture
def captcha_test_urls():
    """Provide CAPTCHA test URLs."""
    return TestDataGenerator.create_captcha_test_urls()


@pytest.fixture
def temp_config_file():
    """Provide temporary configuration file."""
    with TestConfigManager() as config_manager:
        config_file = config_manager.create_temp_config_file()
        yield config_file


@pytest.fixture
def element_selectors():
    """Provide common element selectors for testing."""
    return {
        "button": ElementSelector(css_selector="button.submit"),
        "input": ElementSelector(css_selector="input[name='username']"),
        "link": ElementSelector(css_selector="a.nav-link"),
        "form": ElementSelector(css_selector="form#login-form"),
        "div": ElementSelector(css_selector="div.content"),
        "xpath_button": ElementSelector(xpath="//button[@type='submit']"),
        "xpath_input": ElementSelector(xpath="//input[@name='password']")
    }


@pytest.fixture
def browser_responses():
    """Provide common browser responses for testing."""
    return {
        "navigation_success": TestDataGenerator.create_navigation_response(),
        "navigation_failure": TestDataGenerator.create_browser_response(
            success=False, error_message="Navigation failed"
        ),
        "text_extraction": TestDataGenerator.create_text_extraction_response(),
        "screenshot": TestDataGenerator.create_screenshot_response(),
        "click_success": TestDataGenerator.create_click_response(),
        "type_success": TestDataGenerator.create_type_response(),
        "session_created": TestDataGenerator.create_browser_response(
            data={"session_id": "new-session-456", "status": "active"}
        ),
        "session_closed": TestDataGenerator.create_browser_response(
            data={"session_closed": True, "cleanup_successful": True}
        )
    }


@pytest.fixture
def captcha_analysis():
    """Provide CAPTCHA analysis results for testing."""
    return {
        "recaptcha_detected": TestDataGenerator.create_captcha_analysis(
            captcha_type=CaptchaType.RECAPTCHA_V2
        ),
        "hcaptcha_detected": TestDataGenerator.create_captcha_analysis(
            captcha_type=CaptchaType.HCAPTCHA
        ),
        "text_captcha": TestDataGenerator.create_captcha_analysis(
            captcha_type=CaptchaType.TEXT,
            solution="HELLO"
        ),
        "no_captcha": TestDataGenerator.create_captcha_analysis(
            detected=False, confidence_score=0.1
        )
    }


@pytest.fixture
def error_scenarios():
    """Provide error scenarios for testing."""
    return {
        "navigation_error": AgentCoreBrowserError(
            "Navigation failed",
            error_type=BrowserErrorType.NAVIGATION_FAILED,
            recoverable=True
        ),
        "element_not_found": AgentCoreBrowserError(
            "Element not found",
            error_type=BrowserErrorType.ELEMENT_NOT_FOUND,
            recoverable=False
        ),
        "timeout_error": AgentCoreBrowserError(
            "Operation timed out",
            error_type=BrowserErrorType.TIMEOUT,
            recoverable=True
        ),
        "session_error": AgentCoreBrowserError(
            "Session expired",
            error_type=BrowserErrorType.SESSION_EXPIRED,
            recoverable=True
        )
    }


class TestUtilities:
    """Utility functions for tests."""
    
    @staticmethod
    def assert_browser_response_valid(response: BrowserResponse):
        """Assert that a browser response is valid."""
        assert isinstance(response, BrowserResponse)
        assert hasattr(response, 'success')
        assert hasattr(response, 'data')
        assert hasattr(response, 'session_id')
        assert hasattr(response, 'timestamp')
        assert isinstance(response.data, dict)
    
    @staticmethod
    def assert_tool_metadata_valid(tool):
        """Assert that a tool has valid metadata."""
        assert hasattr(tool, 'metadata')
        assert hasattr(tool.metadata, 'name')
        assert hasattr(tool.metadata, 'description')
        assert hasattr(tool.metadata, 'fn_schema')
        assert isinstance(tool.metadata.name, str)
        assert isinstance(tool.metadata.description, str)
        assert len(tool.metadata.name) > 0
        assert len(tool.metadata.description) > 0
    
    @staticmethod
    def assert_config_valid(config: Dict[str, Any]):
        """Assert that a configuration is valid."""
        required_keys = [
            "aws_credentials", "agentcore_endpoints", 
            "browser_config", "llm_model"
        ]
        
        for key in required_keys:
            assert key in config, f"Missing required config key: {key}"
        
        # Check AWS credentials
        aws_creds = config["aws_credentials"]
        assert "region" in aws_creds
        assert "access_key_id" in aws_creds
        assert "secret_access_key" in aws_creds
        
        # Check browser config
        browser_config = config["browser_config"]
        assert "headless" in browser_config
        assert "viewport_width" in browser_config
        assert "viewport_height" in browser_config
    
    @staticmethod
    def create_mock_llm_response(content: str) -> Mock:
        """Create a mock LLM response."""
        mock_response = Mock()
        mock_response.response = content
        mock_response.content = content
        return mock_response
    
    @staticmethod
    def create_mock_vision_response(analysis: str) -> Mock:
        """Create a mock vision model response."""
        mock_response = Mock()
        mock_response.content = [{
            'text': analysis
        }]
        return mock_response


# Export utilities for easy import
__all__ = [
    'TestDataGenerator',
    'MockBrowserClient', 
    'TestConfigManager',
    'TestUtilities'
]