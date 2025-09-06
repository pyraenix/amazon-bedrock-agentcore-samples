"""
Comprehensive unit tests for all LlamaIndex-AgentCore browser integration components.

This module provides complete unit test coverage for all components including:
- AgentCore browser client functionality
- LlamaIndex tool implementations
- Configuration management
- Error handling
- Response parsing
- Security and privacy features

Tests use real API calls where possible while maintaining isolation through mocking
for external dependencies.
"""

import asyncio
import pytest
import json
import base64
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path

# Import all components to test
from client import AgentCoreBrowserClient
from tools import (
    BrowserNavigationTool, TextExtractionTool, ScreenshotCaptureTool,
    ElementClickTool, FormInteractionTool, CaptchaDetectionTool
)
from integration import LlamaIndexAgentCoreIntegration
from config import (
    ConfigurationManager, IntegrationConfig, BrowserConfiguration,
    AWSCredentials, AgentCoreEndpoints
)
from interfaces import (
    IBrowserClient, BrowserResponse, ElementSelector, BrowserAction,
    SessionStatus
)
from vision_models import CaptchaType, CaptchaAnalysisResult
from exceptions import (
    AgentCoreBrowserError, NavigationError, ElementNotFoundError,
    TimeoutError, SessionError, BrowserErrorType
)
from error_handler import ErrorHandler, RetryConfig
from response_parser import ResponseParser
from security_manager import SecurityManager
from privacy_manager import PrivacyManager
from document_processor import DocumentProcessor, WebContentDocument
from vision_models import VisionModelClient
from captcha_workflows import CaptchaWorkflowManager


class TestFixtures:
    """Common test fixtures and utilities."""
    
    @staticmethod
    def create_mock_browser_response(
        success: bool = True,
        data: Optional[Dict] = None,
        error_message: Optional[str] = None
    ) -> BrowserResponse:
        """Create a mock browser response."""
        return BrowserResponse(
            success=success,
            data=data or {},
            error_message=error_message,
            session_id="test-session-123",
            timestamp=datetime.utcnow().isoformat(),
            operation_id=f"op-{datetime.utcnow().timestamp()}"
        )
    
    @staticmethod
    def create_test_config() -> Dict[str, Any]:
        """Create test configuration."""
        return {
            "aws_credentials": {
                "region": "us-east-1",
                "access_key_id": "test-key",
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
                "timeout_seconds": 30
            },
            "llm_model": "anthropic.claude-3-sonnet-20240229-v1:0",
            "vision_model": "anthropic.claude-3-sonnet-20240229-v1:0"
        }
    
    @staticmethod
    def create_test_screenshot_data() -> str:
        """Create test screenshot data (base64 encoded 1x1 PNG)."""
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="


class TestConfigurationManager:
    """Unit tests for ConfigurationManager."""
    
    def test_configuration_manager_creation(self):
        """Test ConfigurationManager can be created with defaults."""
        config_manager = ConfigurationManager()
        assert config_manager is not None
        assert isinstance(config_manager.config, IntegrationConfig)
    
    def test_load_from_dict(self):
        """Test loading configuration from dictionary."""
        test_config = TestFixtures.create_test_config()
        config_manager = ConfigurationManager()
        config_manager.load_from_dict(test_config)
        
        assert config_manager.config.aws_credentials.region == "us-east-1"
        assert config_manager.config.browser_config.headless is True
        assert config_manager.config.llm_model == "anthropic.claude-3-sonnet-20240229-v1:0"
    
    def test_load_from_file(self):
        """Test loading configuration from file."""
        test_config = TestFixtures.create_test_config()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            config_file = f.name
        
        try:
            config_manager = ConfigurationManager()
            config_manager.load_from_file(config_file)
            
            assert config_manager.config.aws_credentials.region == "us-east-1"
            assert config_manager.config.browser_config.viewport_width == 1920
        finally:
            os.unlink(config_file)
    
    def test_validate_configuration(self):
        """Test configuration validation."""
        config_manager = ConfigurationManager()
        test_config = TestFixtures.create_test_config()
        config_manager.load_from_dict(test_config)
        
        # Should not raise exception for valid config
        config_manager.validate_configuration()
        
        # Test invalid config
        invalid_config = test_config.copy()
        invalid_config["aws_credentials"]["region"] = ""
        config_manager.load_from_dict(invalid_config)
        
        with pytest.raises(ValueError):
            config_manager.validate_configuration()
    
    def test_get_browser_config(self):
        """Test getting browser configuration."""
        config_manager = ConfigurationManager()
        test_config = TestFixtures.create_test_config()
        config_manager.load_from_dict(test_config)
        
        browser_config = config_manager.get_browser_config()
        assert isinstance(browser_config, BrowserConfiguration)
        assert browser_config.headless is True
        assert browser_config.viewport_width == 1920
    
    def test_get_aws_credentials(self):
        """Test getting AWS credentials."""
        config_manager = ConfigurationManager()
        test_config = TestFixtures.create_test_config()
        config_manager.load_from_dict(test_config)
        
        credentials = config_manager.get_aws_credentials()
        assert isinstance(credentials, AWSCredentials)
        assert credentials.region == "us-east-1"
        assert credentials.access_key_id == "test-key"


class TestAgentCoreBrowserClient:
    """Unit tests for AgentCoreBrowserClient."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return TestFixtures.create_test_config()
    
    @pytest.fixture
    def browser_client(self, mock_config):
        """Create browser client with mock configuration."""
        return AgentCoreBrowserClient(mock_config)
    
    def test_client_initialization(self, browser_client):
        """Test client initialization."""
        assert browser_client is not None
        assert browser_client.config is not None
        assert browser_client.session_id is None  # Not created yet
    
    @pytest.mark.asyncio
    async def test_create_session(self, browser_client):
        """Test session creation."""
        with patch.object(browser_client, '_make_api_call') as mock_call:
            mock_call.return_value = TestFixtures.create_mock_browser_response(
                data={"session_id": "new-session-123", "status": "active"}
            )
            
            session_id = await browser_client.create_session()
            
            assert session_id == "new-session-123"
            assert browser_client.session_id == "new-session-123"
            mock_call.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_navigate(self, browser_client):
        """Test navigation functionality."""
        browser_client.session_id = "test-session-123"
        
        with patch.object(browser_client, '_make_api_call') as mock_call:
            mock_call.return_value = TestFixtures.create_mock_browser_response(
                data={
                    "current_url": "https://example.com",
                    "page_title": "Example Page",
                    "status_code": 200,
                    "load_time_ms": 1500
                }
            )
            
            response = await browser_client.navigate("https://example.com")
            
            assert response.success is True
            assert response.data["current_url"] == "https://example.com"
            assert response.data["page_title"] == "Example Page"
            mock_call.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extract_text(self, browser_client):
        """Test text extraction functionality."""
        browser_client.session_id = "test-session-123"
        
        with patch.object(browser_client, '_make_api_call') as mock_call:
            mock_call.return_value = TestFixtures.create_mock_browser_response(
                data={
                    "text": "Sample page content",
                    "element_found": True,
                    "element_count": 1,
                    "extraction_method": "full_page"
                }
            )
            
            response = await browser_client.extract_text()
            
            assert response.success is True
            assert response.data["text"] == "Sample page content"
            assert response.data["element_found"] is True
            mock_call.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_take_screenshot(self, browser_client):
        """Test screenshot functionality."""
        browser_client.session_id = "test-session-123"
        
        with patch.object(browser_client, '_make_api_call') as mock_call:
            mock_call.return_value = TestFixtures.create_mock_browser_response(
                data={
                    "screenshot_data": TestFixtures.create_test_screenshot_data(),
                    "format": "png",
                    "size": {"width": 1920, "height": 1080}
                }
            )
            
            response = await browser_client.take_screenshot()
            
            assert response.success is True
            assert response.data["screenshot_data"] != ""
            assert response.data["format"] == "png"
            mock_call.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_click_element(self, browser_client):
        """Test element clicking functionality."""
        browser_client.session_id = "test-session-123"
        
        with patch.object(browser_client, '_make_api_call') as mock_call:
            mock_call.return_value = TestFixtures.create_mock_browser_response(
                data={
                    "element_found": True,
                    "click_successful": True,
                    "page_changed": True,
                    "element_info": {"tag": "button", "text": "Click me"}
                }
            )
            
            selector = ElementSelector(css_selector="button.submit")
            response = await browser_client.click_element(selector)
            
            assert response.success is True
            assert response.data["element_found"] is True
            assert response.data["click_successful"] is True
            mock_call.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_type_text(self, browser_client):
        """Test text typing functionality."""
        browser_client.session_id = "test-session-123"
        
        with patch.object(browser_client, '_make_api_call') as mock_call:
            mock_call.return_value = TestFixtures.create_mock_browser_response(
                data={
                    "element_found": True,
                    "text_entered": "test input",
                    "field_cleared": True,
                    "element_type": "input"
                }
            )
            
            selector = ElementSelector(css_selector="input[name='username']")
            response = await browser_client.type_text(selector, "test input")
            
            assert response.success is True
            assert response.data["text_entered"] == "test input"
            assert response.data["field_cleared"] is True
            mock_call.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close_session(self, browser_client):
        """Test session closure."""
        browser_client.session_id = "test-session-123"
        
        with patch.object(browser_client, '_make_api_call') as mock_call:
            mock_call.return_value = TestFixtures.create_mock_browser_response(
                data={"session_closed": True, "cleanup_successful": True}
            )
            
            response = await browser_client.close_session()
            
            assert response.success is True
            assert response.data["session_closed"] is True
            assert browser_client.session_id is None
            mock_call.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, browser_client):
        """Test error handling in client operations."""
        browser_client.session_id = "test-session-123"
        
        with patch.object(browser_client, '_make_api_call') as mock_call:
            mock_call.return_value = TestFixtures.create_mock_browser_response(
                success=False,
                error_message="Navigation failed"
            )
            
            response = await browser_client.navigate("https://invalid-url.com")
            
            assert response.success is False
            assert response.error_message == "Navigation failed"


class TestLlamaIndexTools:
    """Unit tests for all LlamaIndex tool implementations."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock browser client."""
        client = Mock(spec=AgentCoreBrowserClient)
        client.session_id = "test-session-123"
        return client
    
    @pytest.fixture
    def navigation_tool(self, mock_client):
        """Create navigation tool."""
        return BrowserNavigationTool(mock_client)
    
    @pytest.fixture
    def text_extraction_tool(self, mock_client):
        """Create text extraction tool."""
        return TextExtractionTool(mock_client)
    
    @pytest.fixture
    def screenshot_tool(self, mock_client):
        """Create screenshot tool."""
        return ScreenshotCaptureTool(mock_client)
    
    @pytest.fixture
    def click_tool(self, mock_client):
        """Create click tool."""
        return ElementClickTool(mock_client)
    
    @pytest.fixture
    def form_tool(self, mock_client):
        """Create form interaction tool."""
        return FormInteractionTool(mock_client)
    
    @pytest.fixture
    def captcha_tool(self, mock_client):
        """Create CAPTCHA detection tool."""
        return CaptchaDetectionTool(mock_client)
    
    def test_navigation_tool_metadata(self, navigation_tool):
        """Test navigation tool metadata."""
        metadata = navigation_tool.metadata
        assert metadata.name == "navigate_browser"
        assert "navigate" in metadata.description.lower()
        assert metadata.fn_schema is not None
    
    @pytest.mark.asyncio
    async def test_navigation_tool_call(self, navigation_tool, mock_client):
        """Test navigation tool call."""
        mock_client.navigate = AsyncMock(return_value=TestFixtures.create_mock_browser_response(
            data={"current_url": "https://example.com", "page_title": "Test Page"}
        ))
        
        result = await navigation_tool.acall(url="https://example.com")
        
        assert result["success"] is True
        assert result["url"] == "https://example.com"
        assert result["title"] == "Test Page"
        mock_client.navigate.assert_called_once()
    
    def test_text_extraction_tool_metadata(self, text_extraction_tool):
        """Test text extraction tool metadata."""
        metadata = text_extraction_tool.metadata
        assert metadata.name == "extract_text"
        assert "extract" in metadata.description.lower()
        assert metadata.fn_schema is not None
    
    @pytest.mark.asyncio
    async def test_text_extraction_tool_call(self, text_extraction_tool, mock_client):
        """Test text extraction tool call."""
        mock_client.extract_text = AsyncMock(return_value=TestFixtures.create_mock_browser_response(
            data={"text": "Sample content", "element_found": True}
        ))
        
        result = await text_extraction_tool.acall()
        
        assert result["success"] is True
        assert result["text"] == "Sample content"
        assert result["element_found"] is True
        mock_client.extract_text.assert_called_once()
    
    def test_screenshot_tool_metadata(self, screenshot_tool):
        """Test screenshot tool metadata."""
        metadata = screenshot_tool.metadata
        assert metadata.name == "capture_screenshot"
        assert "screenshot" in metadata.description.lower()
        assert metadata.fn_schema is not None
    
    @pytest.mark.asyncio
    async def test_screenshot_tool_call(self, screenshot_tool, mock_client):
        """Test screenshot tool call."""
        mock_client.take_screenshot = AsyncMock(return_value=TestFixtures.create_mock_browser_response(
            data={
                "screenshot_data": TestFixtures.create_test_screenshot_data(),
                "format": "png"
            }
        ))
        
        result = await screenshot_tool.acall()
        
        assert result["success"] is True
        assert result["screenshot_data"] != ""
        assert result["screenshot_format"] == "png"
        mock_client.take_screenshot.assert_called_once()
    
    def test_click_tool_metadata(self, click_tool):
        """Test click tool metadata."""
        metadata = click_tool.metadata
        assert metadata.name == "click_element"
        assert "click" in metadata.description.lower()
        assert metadata.fn_schema is not None
    
    @pytest.mark.asyncio
    async def test_click_tool_call(self, click_tool, mock_client):
        """Test click tool call."""
        mock_client.click_element = AsyncMock(return_value=TestFixtures.create_mock_browser_response(
            data={"element_found": True, "click_successful": True}
        ))
        
        result = await click_tool.acall(css_selector="button.submit")
        
        assert result["success"] is True
        assert result["element_found"] is True
        assert result["click_successful"] is True
        mock_client.click_element.assert_called_once()
    
    def test_form_tool_metadata(self, form_tool):
        """Test form tool metadata."""
        metadata = form_tool.metadata
        assert metadata.name == "interact_with_form"
        assert "form" in metadata.description.lower()
        assert metadata.fn_schema is not None
    
    @pytest.mark.asyncio
    async def test_form_tool_call(self, form_tool, mock_client):
        """Test form tool call."""
        mock_client.type_text = AsyncMock(return_value=TestFixtures.create_mock_browser_response(
            data={"element_found": True, "text_entered": "test input"}
        ))
        
        result = await form_tool.acall(
            css_selector="input[name='username']",
            text="test input"
        )
        
        assert result["success"] is True
        assert result["element_found"] is True
        assert result["text_entered"] == "test input"
        mock_client.type_text.assert_called_once()
    
    def test_captcha_tool_metadata(self, captcha_tool):
        """Test CAPTCHA tool metadata."""
        metadata = captcha_tool.metadata
        assert metadata.name == "detect_captcha"
        assert "captcha" in metadata.description.lower()
        assert metadata.fn_schema is not None
    
    @pytest.mark.asyncio
    async def test_captcha_tool_call(self, captcha_tool, mock_client):
        """Test CAPTCHA tool call."""
        mock_client.extract_text = AsyncMock(return_value=TestFixtures.create_mock_browser_response(
            data={"text": "Please complete the CAPTCHA"}
        ))
        mock_client.take_screenshot = AsyncMock(return_value=TestFixtures.create_mock_browser_response(
            data={"screenshot_data": TestFixtures.create_test_screenshot_data()}
        ))
        
        result = await captcha_tool.acall(detection_strategy="comprehensive")
        
        assert result["success"] is True
        assert "captcha_detected" in result
        assert "detection_strategy" in result
        mock_client.extract_text.assert_called()
        mock_client.take_screenshot.assert_called()


class TestErrorHandling:
    """Unit tests for error handling components."""
    
    def test_browser_error_creation(self):
        """Test browser error creation."""
        error = AgentCoreBrowserError(
            "Test error",
            error_type=BrowserErrorType.NAVIGATION_FAILED,
            recoverable=True,
            details={"url": "https://example.com"}
        )
        
        assert str(error) == "Test error"
        assert error.error_type == BrowserErrorType.NAVIGATION_FAILED
        assert error.recoverable is True
        assert error.details["url"] == "https://example.com"
    
    def test_specific_error_types(self):
        """Test specific error type creation."""
        nav_error = NavigationError("Navigation failed", url="https://example.com")
        assert isinstance(nav_error, AgentCoreBrowserError)
        assert nav_error.error_type == BrowserErrorType.NAVIGATION_FAILED
        
        element_error = ElementNotFoundError("Element not found", selector="#missing")
        assert isinstance(element_error, AgentCoreBrowserError)
        assert element_error.error_type == BrowserErrorType.ELEMENT_NOT_FOUND
        
        timeout_error = TimeoutError("Operation timed out", timeout_seconds=30)
        assert isinstance(timeout_error, AgentCoreBrowserError)
        assert timeout_error.error_type == BrowserErrorType.TIMEOUT
    
    def test_error_handler_creation(self):
        """Test error handler creation."""
        handler = ErrorHandler(max_retries=3, retry_delay=1.0)
        assert handler.max_retries == 3
        assert handler.retry_delay == 1.0
    
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery mechanisms."""
        handler = ErrorHandler(max_retries=2, retry_delay=0.1)
        
        # Mock a recoverable error
        error = AgentCoreBrowserError(
            "Temporary failure",
            error_type=BrowserErrorType.TIMEOUT,
            recoverable=True
        )
        
        # Mock operation that succeeds on retry
        call_count = 0
        async def mock_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise error
            return "success"
        
        # Test recovery
        with patch.object(handler, '_get_recovery_strategy') as mock_strategy:
            mock_strategy.return_value = mock_operation
            result = await handler.handle_error(error, "test_operation", {})
            assert result == "success"
    
    def test_retry_config(self):
        """Test retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=True
        )
        
        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True


class TestResponseParser:
    """Unit tests for response parsing functionality."""
    
    def test_response_parser_creation(self):
        """Test response parser creation."""
        parser = ResponseParser()
        assert parser is not None
    
    def test_parse_navigation_response(self):
        """Test parsing navigation response."""
        parser = ResponseParser()
        raw_response = {
            "current_url": "https://example.com",
            "page_title": "Example Page",
            "status_code": 200,
            "load_time_ms": 1500
        }
        
        parsed = parser.parse_navigation_response(raw_response)
        
        assert parsed["url"] == "https://example.com"
        assert parsed["title"] == "Example Page"
        assert parsed["status_code"] == 200
        assert parsed["load_time"] == 1500
    
    def test_parse_text_extraction_response(self):
        """Test parsing text extraction response."""
        parser = ResponseParser()
        raw_response = {
            "text": "Sample content",
            "element_found": True,
            "element_count": 1,
            "extraction_method": "full_page"
        }
        
        parsed = parser.parse_text_extraction_response(raw_response)
        
        assert parsed["text"] == "Sample content"
        assert parsed["element_found"] is True
        assert parsed["text_length"] == len("Sample content")
        assert parsed["extraction_method"] == "full_page"
    
    def test_parse_screenshot_response(self):
        """Test parsing screenshot response."""
        parser = ResponseParser()
        raw_response = {
            "screenshot_data": TestFixtures.create_test_screenshot_data(),
            "format": "png",
            "size": {"width": 1920, "height": 1080}
        }
        
        parsed = parser.parse_screenshot_response(raw_response)
        
        assert parsed["screenshot_data"] != ""
        assert parsed["screenshot_format"] == "png"
        assert parsed["image_width"] == 1920
        assert parsed["image_height"] == 1080
    
    def test_parse_captcha_detection_response(self):
        """Test parsing CAPTCHA detection response."""
        parser = ResponseParser()
        raw_response = {
            "captcha_elements": [
                {"type": "recaptcha", "selector": ".g-recaptcha"}
            ],
            "visual_indicators": ["captcha_image.png"],
            "confidence_score": 0.95
        }
        
        parsed = parser.parse_captcha_detection_response(raw_response)
        
        assert parsed["captcha_detected"] is True
        assert len(parsed["captcha_types"]) > 0
        assert parsed["confidence_score"] == 0.95
        assert "captcha_elements" in parsed


class TestSecurityAndPrivacy:
    """Unit tests for security and privacy components."""
    
    def test_security_manager_creation(self):
        """Test security manager creation."""
        credentials = {"region": "us-east-1", "access_key_id": "test"}
        manager = SecurityManager(credentials)
        assert manager is not None
        assert manager.credentials == credentials
    
    def test_input_sanitization(self):
        """Test input sanitization."""
        manager = SecurityManager({})
        
        # Test dangerous input
        dangerous_input = "<script>alert('xss')</script>"
        sanitized = manager.sanitize_input(dangerous_input)
        assert "<script>" not in sanitized
        assert "alert" in sanitized  # Content preserved, tags removed
        
        # Test safe input
        safe_input = "Hello world"
        sanitized = manager.sanitize_input(safe_input)
        assert sanitized == safe_input
    
    def test_credential_validation(self):
        """Test credential validation."""
        manager = SecurityManager({
            "region": "us-east-1",
            "access_key_id": "AKIA...",
            "secret_access_key": "secret"
        })
        
        # Mock the validation (would normally check AWS permissions)
        with patch.object(manager, '_check_aws_permissions') as mock_check:
            mock_check.return_value = True
            result = manager.validate_credentials()
            assert result is True
    
    def test_privacy_manager_creation(self):
        """Test privacy manager creation."""
        manager = PrivacyManager()
        assert manager is not None
        assert len(manager.pii_patterns) > 0
    
    def test_pii_scrubbing(self):
        """Test PII scrubbing functionality."""
        manager = PrivacyManager()
        
        # Test text with PII
        text_with_pii = "My SSN is 123-45-6789 and email is test@example.com"
        scrubbed = manager.scrub_pii(text_with_pii)
        
        assert "123-45-6789" not in scrubbed
        assert "test@example.com" not in scrubbed
        assert "[REDACTED]" in scrubbed
    
    def test_data_access_logging(self):
        """Test data access logging."""
        manager = PrivacyManager()
        
        # Mock logging
        with patch.object(manager, '_send_to_audit_system') as mock_log:
            manager.log_data_access("extract_text", "web_content", "user123")
            mock_log.assert_called_once()
            
            # Check log entry structure
            log_entry = mock_log.call_args[0][0]
            assert log_entry["operation"] == "extract_text"
            assert log_entry["data_type"] == "web_content"
            assert log_entry["user_id"] == "user123"


class TestDocumentProcessing:
    """Unit tests for document processing components."""
    
    def test_web_content_document_creation(self):
        """Test WebContentDocument creation."""
        doc = WebContentDocument(
            text="Sample web content",
            source_url="https://example.com",
            page_title="Example Page",
            extraction_timestamp=datetime.utcnow()
        )
        
        assert doc.text == "Sample web content"
        assert doc.metadata["source_url"] == "https://example.com"
        assert doc.metadata["page_title"] == "Example Page"
        assert doc.metadata["extraction_method"] == "agentcore_browser_tool"
    
    def test_document_processor_creation(self):
        """Test DocumentProcessor creation."""
        config_path = "test_config.json"
        
        # Create temporary config file
        test_config = TestFixtures.create_test_config()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            config_path = f.name
        
        try:
            processor = DocumentProcessor(config_path=config_path)
            assert processor is not None
            assert processor.config is not None
        finally:
            os.unlink(config_path)
    
    @pytest.mark.asyncio
    async def test_document_processing_workflow(self):
        """Test document processing workflow."""
        # Mock the processor components
        with patch('document_processor.AgentCoreBrowserClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Mock browser responses
            mock_client.navigate = AsyncMock(return_value=TestFixtures.create_mock_browser_response(
                data={"current_url": "https://example.com", "page_title": "Test Page"}
            ))
            mock_client.extract_text = AsyncMock(return_value=TestFixtures.create_mock_browser_response(
                data={"text": "Sample content", "element_found": True}
            ))
            mock_client.take_screenshot = AsyncMock(return_value=TestFixtures.create_mock_browser_response(
                data={"screenshot_data": TestFixtures.create_test_screenshot_data()}
            ))
            
            # Create processor with mock config
            test_config = TestFixtures.create_test_config()
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(test_config, f)
                config_path = f.name
            
            try:
                processor = DocumentProcessor(config_path=config_path)
                result = await processor.process_url("https://example.com")
                
                assert result is not None
                assert result.document is not None
                assert result.document.text == "Sample content"
                assert result.document.metadata["source_url"] == "https://example.com"
            finally:
                os.unlink(config_path)


class TestVisionModels:
    """Unit tests for vision model components."""
    
    def test_vision_model_client_creation(self):
        """Test vision model client creation."""
        credentials = {"region": "us-east-1"}
        client = VisionModelClient(credentials)
        assert client is not None
        assert client.credentials == credentials
    
    @pytest.mark.asyncio
    async def test_captcha_analysis(self):
        """Test CAPTCHA analysis with vision models."""
        credentials = {"region": "us-east-1"}
        client = VisionModelClient(credentials)
        
        # Mock the Bedrock client
        with patch.object(client, 'bedrock_client') as mock_bedrock:
            mock_bedrock.invoke_model.return_value = {
                'body': Mock(read=Mock(return_value=json.dumps({
                    'content': [{
                        'text': 'This appears to be a reCAPTCHA challenge with text "HELLO"'
                    }]
                }).encode()))
            }
            
            screenshot_data = TestFixtures.create_test_screenshot_data()
            analysis = await client.analyze_captcha_image(screenshot_data)
            
            assert analysis is not None
            assert isinstance(analysis, CaptchaAnalysisResult)
            assert analysis.captcha_detected is True
            assert analysis.captcha_type in [CaptchaType.RECAPTCHA_V2, CaptchaType.TEXT]


class TestIntegrationClass:
    """Unit tests for the main integration class."""
    
    def test_integration_creation(self):
        """Test integration class creation."""
        test_config = TestFixtures.create_test_config()
        
        with patch('integration.Bedrock') as mock_bedrock:
            with patch('integration.BedrockMultiModal') as mock_vision:
                integration = LlamaIndexAgentCoreIntegration(test_config)
                
                assert integration is not None
                assert integration.config == test_config
                assert integration.browser_client is not None
                assert integration.tools is not None
                assert len(integration.tools) > 0
    
    def test_tool_creation(self):
        """Test tool creation in integration."""
        test_config = TestFixtures.create_test_config()
        
        with patch('integration.Bedrock') as mock_bedrock:
            with patch('integration.BedrockMultiModal') as mock_vision:
                integration = LlamaIndexAgentCoreIntegration(test_config)
                tools = integration._create_browser_tools()
                
                assert len(tools) >= 6  # All main tools
                tool_names = [tool.metadata.name for tool in tools]
                
                expected_tools = [
                    "navigate_browser", "extract_text", "capture_screenshot",
                    "click_element", "interact_with_form", "detect_captcha"
                ]
                
                for expected_tool in expected_tools:
                    assert expected_tool in tool_names
    
    @pytest.mark.asyncio
    async def test_web_content_processing(self):
        """Test web content processing workflow."""
        test_config = TestFixtures.create_test_config()
        
        with patch('integration.Bedrock') as mock_bedrock:
            with patch('integration.BedrockMultiModal') as mock_vision:
                with patch('integration.ReActAgent') as mock_agent_class:
                    # Mock agent response
                    mock_agent = Mock()
                    mock_agent.achat = AsyncMock(return_value=Mock(
                        response="Successfully processed the web content"
                    ))
                    mock_agent_class.from_tools.return_value = mock_agent
                    
                    integration = LlamaIndexAgentCoreIntegration(test_config)
                    result = await integration.process_web_content("https://example.com")
                    
                    assert result is not None
                    mock_agent.achat.assert_called_once()


if __name__ == "__main__":
    # Run all unit tests
    pytest.main([__file__, "-v", "--tb=short"])