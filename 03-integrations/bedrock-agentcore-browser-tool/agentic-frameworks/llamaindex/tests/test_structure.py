"""
Basic tests to verify the project structure and imports.
"""

import pytest
import sys
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_package_imports():
    """Test that all main package components can be imported."""
    try:
        # Test core interfaces
        from interfaces import (
            IBrowserClient, IToolWrapper, IConfigurationManager,
            BrowserResponse, ElementSelector, BrowserAction, SessionStatus
        )
        
        # Test configuration
        from config import (
            ConfigurationManager, IntegrationConfig, BrowserConfiguration,
            AWSCredentials, AgentCoreEndpoints
        )
        
        # Test exceptions
        from exceptions import (
            AgentCoreBrowserError, NavigationError, ElementNotFoundError,
            TimeoutError, SessionError, BrowserErrorType
        )
        
        # Test client
        from client import AgentCoreBrowserClient
        
        # Test integration
        from integration import LlamaIndexAgentCoreIntegration
        
        assert True  # If we get here, all imports succeeded
        
    except ImportError as e:
        pytest.fail(f"Failed to import package components: {e}")


def test_configuration_manager_creation():
    """Test that ConfigurationManager can be created."""
    from config import ConfigurationManager
    
    config_manager = ConfigurationManager()
    assert config_manager is not None


def test_integration_config_creation():
    """Test that IntegrationConfig can be created with defaults."""
    from config import IntegrationConfig
    
    config = IntegrationConfig()
    assert config is not None
    assert config.aws_credentials is not None
    assert config.browser_config is not None
    assert config.llm_model is not None


def test_browser_response_creation():
    """Test that BrowserResponse can be created."""
    from interfaces import BrowserResponse
    
    response = BrowserResponse(
        success=True,
        data={"test": "data"}
    )
    assert response.success is True
    assert response.data == {"test": "data"}


def test_element_selector_creation():
    """Test that ElementSelector can be created."""
    from interfaces import ElementSelector
    
    selector = ElementSelector(css_selector="#test")
    assert selector.css_selector == "#test"


def test_browser_error_creation():
    """Test that browser errors can be created."""
    from exceptions import AgentCoreBrowserError, BrowserErrorType
    
    error = AgentCoreBrowserError(
        "Test error",
        error_type=BrowserErrorType.NAVIGATION_FAILED,
        recoverable=True
    )
    assert "Test error" in str(error)
    assert "navigation_failed" in str(error)
    assert error.error_type == BrowserErrorType.NAVIGATION_FAILED
    assert error.recoverable is True


def test_project_structure():
    """Test that the project structure is correct."""
    base_path = Path(__file__).parent.parent
    
    # Check that all required files exist
    required_files = [
        "__init__.py",
        "interfaces.py",
        "config.py",
        "exceptions.py",
        "client.py",
        "integration.py",
        "requirements.txt",
        "setup.py",
        "README.md",
        "config.example.yaml"
    ]
    
    for file_name in required_files:
        file_path = base_path / file_name
        assert file_path.exists(), f"Required file {file_name} does not exist"
    
    # Check that tests directory exists
    tests_path = base_path / "tests"
    assert tests_path.exists(), "Tests directory does not exist"
    assert (tests_path / "__init__.py").exists(), "Tests __init__.py does not exist"


if __name__ == "__main__":
    pytest.main([__file__])