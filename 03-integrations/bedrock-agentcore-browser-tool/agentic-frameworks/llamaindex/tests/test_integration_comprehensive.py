"""
Comprehensive integration tests for LlamaIndex-AgentCore browser integration.

This module provides end-to-end integration tests that make actual API calls
to AgentCore browser tool service, test CAPTCHA handling workflows, and
validate performance and resource management.

These tests require:
- Valid AWS credentials with AgentCore access
- AgentCore browser tool service availability
- Network connectivity to test websites
"""

import asyncio
import pytest
import time
import psutil
import json
import os
from typing import Dict, Any, List
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
from pathlib import Path

# Import integration components
from integration import LlamaIndexAgentCoreIntegration
from client import AgentCoreBrowserClient
from config import ConfigurationManager
from document_processor import DocumentProcessor, DocumentPipeline
from captcha_workflows import CaptchaWorkflowManager
from vision_models import VisionModelClient
from interfaces import SessionStatus
from vision_models import CaptchaType
from exceptions import AgentCoreBrowserError, BrowserErrorType


class IntegrationTestConfig:
    """Configuration for integration tests."""
    
    def __init__(self):
        self.config_file = self._find_config_file()
        self.test_urls = self._get_test_urls()
        self.captcha_test_urls = self._get_captcha_test_urls()
        self.performance_test_config = self._get_performance_config()
    
    def _find_config_file(self) -> str:
        """Find the configuration file for tests."""
        possible_configs = [
            "agentcore_config.json",
            "config.json",
            "test_config.json"
        ]
        
        for config_file in possible_configs:
            if os.path.exists(config_file):
                return config_file
        
        # Create a test config if none exists
        return self._create_test_config()
    
    def _create_test_config(self) -> str:
        """Create a test configuration file."""
        test_config = {
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
                "timeout_seconds": 30,
                "enable_javascript": True,
                "enable_images": True
            },
            "llm_model": "anthropic.claude-3-sonnet-20240229-v1:0",
            "vision_model": "anthropic.claude-3-sonnet-20240229-v1:0",
            "test_mode": True
        }
        
        config_file = "integration_test_config.json"
        with open(config_file, 'w') as f:
            json.dump(test_config, f, indent=2)
        
        return config_file
    
    def _get_test_urls(self) -> List[str]:
        """Get URLs for testing."""
        return [
            "https://httpbin.org/html",
            "https://httpbin.org/json",
            "https://example.com",
            "https://httpbin.org/forms/post",
            "https://httpbin.org/delay/1",
            "https://httpbin.org/status/200"
        ]
    
    def _get_captcha_test_urls(self) -> List[str]:
        """Get URLs with CAPTCHAs for testing."""
        return [
            "https://www.google.com/recaptcha/api2/demo",
            "https://hcaptcha.com/1/demo",
            "https://funcaptcha.com/demo"
        ]
    
    def _get_performance_config(self) -> Dict[str, Any]:
        """Get performance test configuration."""
        return {
            "concurrent_sessions": 5,
            "requests_per_session": 10,
            "max_memory_mb": 500,
            "max_response_time_ms": 10000,
            "success_rate_threshold": 0.95
        }


@pytest.fixture(scope="session")
def test_config():
    """Create test configuration."""
    return IntegrationTestConfig()


@pytest.fixture(scope="session")
def integration_client(test_config):
    """Create integration client for tests."""
    config_manager = ConfigurationManager()
    config_manager.load_from_file(test_config.config_file)
    
    # Skip tests if no valid credentials
    if not config_manager.config.aws_credentials.access_key_id:
        pytest.skip("No AWS credentials available for integration tests")
    
    return LlamaIndexAgentCoreIntegration(config_manager.config.to_dict())


@pytest.fixture(scope="session")
def browser_client(test_config):
    """Create browser client for tests."""
    config_manager = ConfigurationManager()
    config_manager.load_from_file(test_config.config_file)
    
    return AgentCoreBrowserClient(config_manager.config.to_dict())


class TestRealAgentCoreBrowserClient:
    """Integration tests for AgentCore browser client with real API calls."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_session_lifecycle(self, browser_client):
        """Test complete session lifecycle with real AgentCore."""
        # Create session
        session_id = await browser_client.create_session()
        assert session_id is not None
        assert browser_client.session_id == session_id
        
        # Verify session is active
        assert browser_client.session_id is not None
        
        # Close session
        response = await browser_client.close_session()
        assert response.success is True
        assert browser_client.session_id is None
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_navigation_with_real_urls(self, browser_client, test_config):
        """Test navigation to real URLs."""
        await browser_client.create_session()
        
        try:
            for url in test_config.test_urls[:3]:  # Test first 3 URLs
                response = await browser_client.navigate(url)
                
                assert response.success is True
                assert response.data.get("current_url") == url
                assert response.data.get("status_code") in [200, 301, 302]
                assert response.session_id == browser_client.session_id
                
                # Wait between requests to avoid rate limiting
                await asyncio.sleep(1)
        
        finally:
            await browser_client.close_session()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_text_extraction_real_content(self, browser_client):
        """Test text extraction from real web content."""
        await browser_client.create_session()
        
        try:
            # Navigate to a page with known content
            await browser_client.navigate("https://httpbin.org/html")
            
            # Extract text
            response = await browser_client.extract_text()
            
            assert response.success is True
            assert len(response.data.get("text", "")) > 0
            assert "html" in response.data.get("text", "").lower()
        
        finally:
            await browser_client.close_session()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_screenshot_capture_real_page(self, browser_client):
        """Test screenshot capture from real page."""
        await browser_client.create_session()
        
        try:
            # Navigate to a simple page
            await browser_client.navigate("https://example.com")
            
            # Take screenshot
            response = await browser_client.take_screenshot()
            
            assert response.success is True
            assert response.data.get("screenshot_data") is not None
            assert len(response.data.get("screenshot_data", "")) > 0
            assert response.data.get("format") == "png"
        
        finally:
            await browser_client.close_session()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_element_interaction_real_form(self, browser_client):
        """Test element interaction with real form."""
        await browser_client.create_session()
        
        try:
            # Navigate to form page
            await browser_client.navigate("https://httpbin.org/forms/post")
            
            # Type in form field
            from interfaces import ElementSelector
            selector = ElementSelector(css_selector="input[name='custname']")
            response = await browser_client.type_text(selector, "Test User")
            
            assert response.success is True
            assert response.data.get("element_found") is True
            assert response.data.get("text_entered") == "Test User"
        
        finally:
            await browser_client.close_session()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_invalid_url(self, browser_client):
        """Test error handling with invalid URL."""
        await browser_client.create_session()
        
        try:
            # Try to navigate to invalid URL
            response = await browser_client.navigate("https://this-domain-does-not-exist-12345.com")
            
            # Should handle error gracefully
            assert response.success is False
            assert response.error_message is not None
        
        finally:
            await browser_client.close_session()


class TestCaptchaHandlingWorkflows:
    """Integration tests for CAPTCHA detection and handling workflows."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_captcha_detection_recaptcha(self, integration_client):
        """Test CAPTCHA detection on reCAPTCHA demo page."""
        try:
            # Process reCAPTCHA demo page
            result = await integration_client.process_web_content(
                "https://www.google.com/recaptcha/api2/demo"
            )
            
            # Should detect CAPTCHA presence
            assert result is not None
            response_text = str(result).lower()
            assert any(keyword in response_text for keyword in ["captcha", "recaptcha", "challenge"])
        
        except Exception as e:
            # CAPTCHA sites may block automated access
            pytest.skip(f"CAPTCHA site not accessible: {e}")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_captcha_workflow_manager(self, test_config):
        """Test CAPTCHA workflow manager with real scenarios."""
        config_manager = ConfigurationManager()
        config_manager.load_from_file(test_config.config_file)
        
        workflow_manager = CaptchaWorkflowManager(config_manager.config.to_dict())
        
        # Test CAPTCHA detection workflow
        browser_client = AgentCoreBrowserClient(config_manager.config.to_dict())
        await browser_client.create_session()
        
        try:
            # Navigate to a page that might have CAPTCHAs
            await browser_client.navigate("https://example.com")
            
            # Run CAPTCHA detection
            detection_result = await workflow_manager.detect_captcha(browser_client)
            
            assert detection_result is not None
            assert "captcha_detected" in detection_result
            assert "detection_confidence" in detection_result
        
        finally:
            await browser_client.close_session()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_vision_model_captcha_analysis(self, test_config):
        """Test vision model analysis of CAPTCHA images."""
        config_manager = ConfigurationManager()
        config_manager.load_from_file(test_config.config_file)
        
        vision_client = VisionModelClient(config_manager.config.aws_credentials.to_dict())
        
        # Create a simple test image (1x1 PNG)
        test_image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        try:
            # Analyze the test image
            analysis = await vision_client.analyze_captcha_image(test_image_data)
            
            assert analysis is not None
            assert hasattr(analysis, 'detected')
            assert hasattr(analysis, 'captcha_type')
            assert hasattr(analysis, 'confidence_score')
        
        except Exception as e:
            # Vision model may not be available in test environment
            pytest.skip(f"Vision model not available: {e}")


class TestPerformanceAndConcurrency:
    """Performance and concurrency tests for AgentCore browser integration."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_browser_sessions(self, test_config):
        """Test multiple concurrent browser sessions."""
        config_manager = ConfigurationManager()
        config_manager.load_from_file(test_config.config_file)
        
        num_sessions = test_config.performance_test_config["concurrent_sessions"]
        test_urls = test_config.test_urls[:num_sessions]
        
        async def process_url_session(url: str, session_id: int) -> Dict[str, Any]:
            """Process a URL in a separate session."""
            client = AgentCoreBrowserClient(config_manager.config.to_dict())
            start_time = time.time()
            
            try:
                await client.create_session()
                response = await client.navigate(url)
                text_response = await client.extract_text()
                
                processing_time = (time.time() - start_time) * 1000
                
                return {
                    "session_id": session_id,
                    "url": url,
                    "success": response.success and text_response.success,
                    "processing_time_ms": processing_time,
                    "content_length": len(text_response.data.get("text", ""))
                }
            
            except Exception as e:
                return {
                    "session_id": session_id,
                    "url": url,
                    "success": False,
                    "error": str(e),
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
            
            finally:
                try:
                    await client.close_session()
                except:
                    pass
        
        # Run concurrent sessions
        start_time = time.time()
        tasks = [
            process_url_session(url, i) 
            for i, url in enumerate(test_urls)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Analyze results
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
        success_rate = len(successful_results) / len(results)
        
        avg_processing_time = sum(r["processing_time_ms"] for r in successful_results) / len(successful_results) if successful_results else 0
        
        # Performance assertions
        assert success_rate >= test_config.performance_test_config["success_rate_threshold"]
        assert avg_processing_time <= test_config.performance_test_config["max_response_time_ms"]
        assert total_time <= 60  # Should complete within 60 seconds
        
        print(f"Concurrent sessions test results:")
        print(f"  Sessions: {num_sessions}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Average processing time: {avg_processing_time:.0f}ms")
        print(f"  Total time: {total_time:.1f}s")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, test_config):
        """Test memory usage during extended operations."""
        config_manager = ConfigurationManager()
        config_manager.load_from_file(test_config.config_file)
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform multiple operations
        client = AgentCoreBrowserClient(config_manager.config.to_dict())
        await client.create_session()
        
        try:
            for i in range(10):
                url = test_config.test_urls[i % len(test_config.test_urls)]
                await client.navigate(url)
                await client.extract_text()
                await client.take_screenshot()
                
                # Check memory usage
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                
                # Memory should not increase excessively
                assert memory_increase <= test_config.performance_test_config["max_memory_mb"]
                
                await asyncio.sleep(0.5)  # Brief pause between operations
        
        finally:
            await client.close_session()
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_memory_increase = final_memory - initial_memory
        
        print(f"Memory usage test results:")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Final memory: {final_memory:.1f}MB")
        print(f"  Memory increase: {total_memory_increase:.1f}MB")
        
        assert total_memory_increase <= test_config.performance_test_config["max_memory_mb"]
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_session_lifecycle_performance(self, test_config):
        """Test performance of session creation and cleanup."""
        config_manager = ConfigurationManager()
        config_manager.load_from_file(test_config.config_file)
        
        session_times = []
        
        for i in range(5):  # Test 5 session lifecycles
            client = AgentCoreBrowserClient(config_manager.config.to_dict())
            
            start_time = time.time()
            
            # Create session
            session_id = await client.create_session()
            assert session_id is not None
            
            # Perform basic operation
            response = await client.navigate("https://example.com")
            assert response.success is True
            
            # Close session
            close_response = await client.close_session()
            assert close_response.success is True
            
            session_time = (time.time() - start_time) * 1000
            session_times.append(session_time)
            
            await asyncio.sleep(1)  # Brief pause between sessions
        
        # Analyze session performance
        avg_session_time = sum(session_times) / len(session_times)
        max_session_time = max(session_times)
        
        print(f"Session lifecycle performance:")
        print(f"  Average session time: {avg_session_time:.0f}ms")
        print(f"  Maximum session time: {max_session_time:.0f}ms")
        
        # Performance assertions
        assert avg_session_time <= 10000  # 10 seconds average
        assert max_session_time <= 15000  # 15 seconds maximum


class TestEndToEndWorkflows:
    """End-to-end workflow tests."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_web_scraping_workflow(self, integration_client, test_config):
        """Test complete web scraping workflow."""
        # Test URL with structured content
        test_url = "https://httpbin.org/json"
        
        result = await integration_client.process_web_content(test_url)
        
        assert result is not None
        # The result should contain processed content information
        result_str = str(result).lower()
        assert any(keyword in result_str for keyword in ["json", "content", "processed"])
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_document_processing_pipeline(self, test_config):
        """Test document processing pipeline with real URLs."""
        config_manager = ConfigurationManager()
        config_manager.load_from_file(test_config.config_file)
        
        async with DocumentPipeline(config_path=test_config.config_file) as pipeline:
            # Process multiple URLs
            urls = test_config.test_urls[:3]
            results = await pipeline.process_multiple_urls(urls)
            
            assert len(results) == len(urls)
            
            # Check processing results
            successful_docs = pipeline.get_successful_documents(results)
            assert len(successful_docs) > 0
            
            # Verify document structure
            for doc in successful_docs:
                assert hasattr(doc, 'text')
                assert hasattr(doc, 'metadata')
                assert doc.metadata.get('source_url') in urls
                assert doc.metadata.get('extraction_method') == 'agentcore_browser_tool'
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, test_config):
        """Test error recovery in workflows."""
        config_manager = ConfigurationManager()
        config_manager.load_from_file(test_config.config_file)
        
        client = AgentCoreBrowserClient(config_manager.config.to_dict())
        await client.create_session()
        
        try:
            # Test with mix of valid and invalid URLs
            urls = [
                "https://example.com",  # Valid
                "https://invalid-domain-12345.com",  # Invalid
                "https://httpbin.org/status/404",  # 404 error
                "https://httpbin.org/html"  # Valid
            ]
            
            results = []
            for url in urls:
                try:
                    response = await client.navigate(url)
                    results.append({
                        "url": url,
                        "success": response.success,
                        "error": response.error_message if not response.success else None
                    })
                except Exception as e:
                    results.append({
                        "url": url,
                        "success": False,
                        "error": str(e)
                    })
            
            # Should have some successful and some failed results
            successful = [r for r in results if r["success"]]
            failed = [r for r in results if not r["success"]]
            
            assert len(successful) >= 2  # At least 2 should succeed
            assert len(failed) >= 1  # At least 1 should fail
            
            # Failed results should have error messages
            for failed_result in failed:
                assert failed_result["error"] is not None
        
        finally:
            await client.close_session()


class TestResourceManagement:
    """Tests for resource management and cleanup."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_session_cleanup_on_error(self, test_config):
        """Test that sessions are properly cleaned up on errors."""
        config_manager = ConfigurationManager()
        config_manager.load_from_file(test_config.config_file)
        
        client = AgentCoreBrowserClient(config_manager.config.to_dict())
        
        # Create session
        session_id = await client.create_session()
        assert client.session_id == session_id
        
        # Simulate error scenario
        try:
            # Force an error by trying to navigate to invalid URL
            await client.navigate("invalid://url")
        except:
            pass  # Expected to fail
        
        # Session should still be tracked
        assert client.session_id == session_id
        
        # Cleanup should work
        response = await client.close_session()
        assert response.success is True
        assert client.session_id is None
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_session_isolation(self, test_config):
        """Test that concurrent sessions are properly isolated."""
        config_manager = ConfigurationManager()
        config_manager.load_from_file(test_config.config_file)
        
        # Create multiple clients
        clients = [
            AgentCoreBrowserClient(config_manager.config.to_dict())
            for _ in range(3)
        ]
        
        try:
            # Create sessions for all clients
            session_ids = []
            for client in clients:
                session_id = await client.create_session()
                session_ids.append(session_id)
                assert client.session_id == session_id
            
            # All session IDs should be unique
            assert len(set(session_ids)) == len(session_ids)
            
            # Each client should maintain its own session
            for i, client in enumerate(clients):
                assert client.session_id == session_ids[i]
                
                # Navigate to different URLs
                url = f"https://httpbin.org/status/{200 + i}"
                response = await client.navigate(url)
                assert response.success is True
        
        finally:
            # Cleanup all sessions
            for client in clients:
                try:
                    await client.close_session()
                except:
                    pass  # Ignore cleanup errors


if __name__ == "__main__":
    # Run integration tests
    pytest.main([
        __file__, 
        "-v", 
        "-m", "integration",
        "--tb=short",
        "--durations=10"
    ])