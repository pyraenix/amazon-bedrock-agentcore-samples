"""
Pytest configuration and shared fixtures for LlamaIndex-AgentCore browser integration tests.

This file configures pytest settings, markers, and provides session-scoped
fixtures that are shared across all test modules.
"""

import pytest
import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test utilities
from tests.test_fixtures import TestDataGenerator, MockBrowserClient, TestConfigManager


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test (requires real AgentCore access)"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "captcha: mark test as CAPTCHA-related (may require special setup)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names and paths."""
    for item in items:
        # Add markers based on test file names
        if "test_unit" in item.fspath.basename:
            item.add_marker(pytest.mark.unit)
        elif "test_integration" in item.fspath.basename:
            item.add_marker(pytest.mark.integration)
        elif "performance" in item.name.lower():
            item.add_marker(pytest.mark.performance)
        
        # Add slow marker for tests that might take longer
        if any(keyword in item.name.lower() for keyword in ["concurrent", "batch", "workflow"]):
            item.add_marker(pytest.mark.slow)
        
        # Add captcha marker for CAPTCHA-related tests
        if "captcha" in item.name.lower():
            item.add_marker(pytest.mark.captcha)


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_environment():
    """Provide test environment information."""
    return {
        "python_version": sys.version,
        "platform": sys.platform,
        "cwd": os.getcwd(),
        "has_aws_credentials": bool(os.getenv("AWS_ACCESS_KEY_ID")),
        "has_agentcore_config": os.path.exists("agentcore_config.json"),
        "test_mode": True
    }


@pytest.fixture(scope="session")
def session_config_manager():
    """Provide a session-scoped configuration manager."""
    config_manager = TestConfigManager()
    yield config_manager
    config_manager.cleanup()


@pytest.fixture(scope="function")
def isolated_config_manager():
    """Provide a function-scoped configuration manager for isolated tests."""
    config_manager = TestConfigManager()
    yield config_manager
    config_manager.cleanup()


@pytest.fixture
def skip_if_no_credentials():
    """Skip test if no AWS credentials are available."""
    if not os.getenv("AWS_ACCESS_KEY_ID"):
        pytest.skip("No AWS credentials available for integration tests")


@pytest.fixture
def skip_if_no_agentcore():
    """Skip test if no AgentCore configuration is available."""
    if not os.path.exists("agentcore_config.json") and not os.getenv("AGENTCORE_BROWSER_URL"):
        pytest.skip("No AgentCore configuration available for integration tests")


@pytest.fixture(autouse=True)
def setup_test_logging(caplog):
    """Set up logging for tests."""
    import logging
    
    # Set logging level for test modules
    logging.getLogger("client").setLevel(logging.DEBUG)
    logging.getLogger("tools").setLevel(logging.DEBUG)
    logging.getLogger("integration").setLevel(logging.DEBUG)
    
    # Capture logs
    caplog.set_level(logging.DEBUG)


@pytest.fixture
def mock_aws_credentials():
    """Provide mock AWS credentials for testing."""
    return {
        "region": "us-east-1",
        "access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    }


@pytest.fixture
def real_aws_credentials():
    """Provide real AWS credentials if available."""
    credentials = {
        "region": os.getenv("AWS_REGION", "us-east-1"),
        "access_key_id": os.getenv("AWS_ACCESS_KEY_ID", ""),
        "secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", "")
    }
    
    if not credentials["access_key_id"]:
        pytest.skip("No real AWS credentials available")
    
    return credentials


@pytest.fixture
def agentcore_config():
    """Provide AgentCore configuration."""
    config_file = "agentcore_config.json"
    
    if os.path.exists(config_file):
        import json
        with open(config_file, 'r') as f:
            return json.load(f)
    
    # Create default config from environment
    return {
        "aws_credentials": {
            "region": os.getenv("AWS_REGION", "us-east-1"),
            "access_key_id": os.getenv("AWS_ACCESS_KEY_ID", ""),
            "secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", "")
        },
        "agentcore_endpoints": {
            "browser_tool_url": os.getenv("AGENTCORE_BROWSER_URL", "https://agentcore.amazonaws.com"),
            "api_version": "v1"
        },
        "browser_config": {
            "headless": True,
            "viewport_width": 1920,
            "viewport_height": 1080,
            "timeout_seconds": 30
        },
        "llm_model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "vision_model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "test_mode": True
    }


# Performance test fixtures
@pytest.fixture
def performance_config():
    """Provide performance test configuration."""
    return {
        "max_response_time_ms": 10000,
        "max_memory_increase_mb": 100,
        "concurrent_sessions": 3,
        "requests_per_session": 5,
        "success_rate_threshold": 0.8
    }


# Error simulation fixtures
@pytest.fixture
def error_simulation():
    """Provide error simulation utilities."""
    class ErrorSimulator:
        def __init__(self):
            self.error_count = 0
            self.max_errors = 0
        
        def set_max_errors(self, count: int):
            """Set maximum number of errors to simulate."""
            self.max_errors = count
            self.error_count = 0
        
        def should_error(self) -> bool:
            """Check if should simulate an error."""
            if self.error_count < self.max_errors:
                self.error_count += 1
                return True
            return False
        
        def reset(self):
            """Reset error simulation."""
            self.error_count = 0
            self.max_errors = 0
    
    return ErrorSimulator()


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Clean up temporary files after each test."""
    temp_files = []
    
    yield temp_files
    
    # Cleanup
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        except OSError:
            pass  # Ignore cleanup errors


@pytest.fixture(scope="session", autouse=True)
def session_cleanup():
    """Perform session-level cleanup."""
    yield
    
    # Clean up any session-level resources
    temp_files = [
        "integration_test_config.json",
        "test_results.json",
        "performance_results.json"
    ]
    
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        except OSError:
            pass


# Test data fixtures
@pytest.fixture
def sample_html_content():
    """Provide sample HTML content for testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page</title>
    </head>
    <body>
        <h1>Welcome to Test Page</h1>
        <p>This is a sample paragraph.</p>
        <form id="test-form">
            <input type="text" name="username" placeholder="Username">
            <input type="password" name="password" placeholder="Password">
            <button type="submit">Submit</button>
        </form>
        <div class="g-recaptcha" data-sitekey="test-key"></div>
    </body>
    </html>
    """


@pytest.fixture
def sample_json_content():
    """Provide sample JSON content for testing."""
    return {
        "title": "Sample JSON",
        "data": {
            "items": [
                {"id": 1, "name": "Item 1"},
                {"id": 2, "name": "Item 2"}
            ]
        },
        "metadata": {
            "timestamp": "2024-01-01T12:00:00Z",
            "version": "1.0"
        }
    }


# Async test utilities
@pytest.fixture
def async_test_timeout():
    """Provide timeout for async tests."""
    return 30  # seconds


def pytest_runtest_setup(item):
    """Set up individual test runs."""
    # Skip integration tests if not explicitly requested
    if "integration" in item.keywords and not item.config.getoption("--run-integration", default=False):
        pytest.skip("Integration tests skipped (use --run-integration to run)")
    
    # Skip performance tests if not explicitly requested
    if "performance" in item.keywords and not item.config.getoption("--run-performance", default=False):
        pytest.skip("Performance tests skipped (use --run-performance to run)")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require real AgentCore access"
    )
    parser.addoption(
        "--run-performance",
        action="store_true", 
        default=False,
        help="Run performance tests"
    )
    parser.addoption(
        "--run-captcha",
        action="store_true",
        default=False,
        help="Run CAPTCHA-related tests"
    )


# Test result collection
@pytest.fixture(scope="session")
def test_results():
    """Collect test results for reporting."""
    results = {
        "unit_tests": {"passed": 0, "failed": 0, "skipped": 0},
        "integration_tests": {"passed": 0, "failed": 0, "skipped": 0},
        "performance_tests": {"passed": 0, "failed": 0, "skipped": 0},
        "total_duration": 0,
        "start_time": None,
        "end_time": None
    }
    
    yield results
    
    # Save results to file
    import json
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)