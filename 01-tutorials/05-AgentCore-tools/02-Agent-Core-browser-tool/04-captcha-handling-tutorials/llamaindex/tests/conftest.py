"""
Pytest configuration and fixtures for LlamaIndex CAPTCHA integration tests.

This module provides shared fixtures and configuration for all test modules.
"""

import pytest
import asyncio
import os
import sys
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_browser_client():
    """Provide a mock browser client for testing."""
    browser = Mock()
    
    # Set up common browser methods
    browser.get = Mock()
    browser.find_elements = Mock(return_value=[])
    browser.find_element = Mock()
    browser.get_screenshot_as_png = Mock(return_value=b'mock_screenshot_data')
    browser.execute_script = Mock(return_value=None)
    browser.quit = Mock()
    
    # Mock element interactions
    mock_element = Mock()
    mock_element.click = Mock()
    mock_element.send_keys = Mock()
    mock_element.clear = Mock()
    mock_element.is_displayed = Mock(return_value=True)
    mock_element.is_enabled = Mock(return_value=True)
    mock_element.get_attribute = Mock(return_value="mock_attribute")
    mock_element.text = "Mock element text"
    mock_element.tag_name = "div"
    
    browser.find_element.return_value = mock_element
    
    return browser


@pytest.fixture
def mock_bedrock_client():
    """Provide a mock Bedrock client for testing."""
    bedrock = Mock()
    
    # Mock successful model invocation
    mock_response_body = Mock()
    mock_response_body.read = Mock(return_value=b'{"completion": "MOCK_SOLUTION", "confidence": 0.95}')
    
    bedrock.invoke_model = Mock(return_value={
        'body': mock_response_body,
        'contentType': 'application/json'
    })
    
    return bedrock


@pytest.fixture
def mock_captcha_data():
    """Provide mock CAPTCHA data for testing."""
    return {
        "captcha_type": "text",
        "image_data": b"mock_image_data",
        "element_selector": "#captcha-input",
        "page_url": "https://test-site.com",
        "screenshot_path": "/tmp/mock_screenshot.png"
    }


@pytest.fixture
def mock_detection_result():
    """Provide mock CAPTCHA detection result."""
    return {
        "found": True,
        "captcha_type": "text",
        "element_info": {
            "selector": "#captcha-input",
            "tag_name": "input",
            "attributes": {"type": "text", "name": "captcha"}
        },
        "screenshot_path": "/tmp/captcha_screenshot.png",
        "confidence": 0.9
    }


@pytest.fixture
def mock_solving_result():
    """Provide mock CAPTCHA solving result."""
    return {
        "solution": "ABC123",
        "solution_type": "text",
        "confidence": 0.85,
        "processing_time": 2.5,
        "model_used": "anthropic.claude-3-sonnet-20240229-v1:0"
    }


@pytest.fixture
def sample_captcha_elements():
    """Provide sample CAPTCHA elements for testing."""
    elements = []
    
    # Text CAPTCHA elements
    text_input = Mock()
    text_input.tag_name = "input"
    text_input.get_attribute = Mock(side_effect=lambda attr: {
        'type': 'text',
        'name': 'captcha',
        'id': 'captcha-input'
    }.get(attr))
    text_input.is_displayed = Mock(return_value=True)
    
    captcha_image = Mock()
    captcha_image.tag_name = "img"
    captcha_image.get_attribute = Mock(side_effect=lambda attr: {
        'src': '/captcha-image.png',
        'alt': 'CAPTCHA Image'
    }.get(attr))
    captcha_image.is_displayed = Mock(return_value=True)
    
    elements.extend([text_input, captcha_image])
    
    # reCAPTCHA elements
    recaptcha_div = Mock()
    recaptcha_div.tag_name = "div"
    recaptcha_div.get_attribute = Mock(side_effect=lambda attr: {
        'class': 'g-recaptcha',
        'data-sitekey': 'mock-site-key'
    }.get(attr))
    recaptcha_div.is_displayed = Mock(return_value=True)
    
    elements.append(recaptcha_div)
    
    return elements


@pytest.fixture
def mock_llm_response():
    """Provide mock LLM response for testing."""
    response = Mock()
    response.text = "Mock LLM response text"
    response.response = "Mock response content"
    response.source_nodes = []
    response.metadata = {"model": "mock-model", "tokens": 100}
    
    return response


@pytest.fixture
def event_loop():
    """Provide event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_directory(tmp_path):
    """Provide temporary directory for test files."""
    return tmp_path


@pytest.fixture
def mock_workflow_events():
    """Provide mock workflow events for testing."""
    from unittest.mock import Mock
    
    start_event = Mock()
    start_event.page_url = "https://test-site.com"
    start_event.task = "Test CAPTCHA handling"
    
    detected_event = Mock()
    detected_event.captcha_data = {
        "captcha_type": "text",
        "element_selector": "#captcha-input"
    }
    detected_event.page_url = "https://test-site.com"
    
    solved_event = Mock()
    solved_event.solution = "ABC123"
    solved_event.captcha_data = detected_event.captcha_data
    
    return {
        "start": start_event,
        "detected": detected_event,
        "solved": solved_event
    }


@pytest.fixture
def mock_error_scenarios():
    """Provide mock error scenarios for testing."""
    return {
        "browser_timeout": TimeoutError("Browser operation timed out"),
        "connection_error": ConnectionError("Failed to connect to browser"),
        "model_error": Exception("Bedrock model unavailable"),
        "rate_limit": Exception("Rate limit exceeded"),
        "invalid_data": ValueError("Invalid CAPTCHA data format"),
        "permission_error": PermissionError("Insufficient permissions")
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test."""
    # Set test environment variables
    os.environ['TESTING'] = 'true'
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    
    yield
    
    # Clean up after test
    if 'TESTING' in os.environ:
        del os.environ['TESTING']


@pytest.fixture
def mock_performance_metrics():
    """Provide mock performance metrics for testing."""
    return {
        "detection_time": 1.2,
        "solving_time": 3.5,
        "total_time": 4.7,
        "memory_usage": 150.5,  # MB
        "cpu_usage": 25.3,      # %
        "success_rate": 0.85,
        "retry_count": 1
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark tests based on their names or paths
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)
            
        # Mark slow tests
        if "performance" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.slow)


# Custom assertions for CAPTCHA testing
def assert_captcha_detection_result(result):
    """Assert that a CAPTCHA detection result has the expected structure."""
    assert isinstance(result, dict)
    assert "found" in result
    assert isinstance(result["found"], bool)
    
    if result["found"]:
        assert "captcha_type" in result
        assert "element_info" in result
        assert "confidence" in result
        assert isinstance(result["confidence"], (int, float))
        assert 0 <= result["confidence"] <= 1


def assert_captcha_solving_result(result):
    """Assert that a CAPTCHA solving result has the expected structure."""
    assert isinstance(result, (str, dict))
    
    if isinstance(result, dict):
        assert "solution" in result
        assert "confidence" in result
        assert isinstance(result["confidence"], (int, float))
        assert 0 <= result["confidence"] <= 1


# Add custom assertions to pytest namespace
pytest.assert_captcha_detection_result = assert_captcha_detection_result
pytest.assert_captcha_solving_result = assert_captcha_solving_result