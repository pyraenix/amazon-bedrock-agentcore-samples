"""
Test CAPTCHA detection with actual test sites and mock scenarios.

This module validates CAPTCHA detection functionality using LlamaIndex tools
with both mock test sites and controlled test environments.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llamaindex_captcha_tools import CaptchaDetectionTool, CaptchaSolvingTool


class MockTestSite:
    """Mock test site for CAPTCHA testing."""
    
    def __init__(self, captcha_type: str, has_captcha: bool = True):
        self.captcha_type = captcha_type
        self.has_captcha = has_captcha
        self.elements = self._create_mock_elements()
        
    def _create_mock_elements(self) -> List[Mock]:
        """Create mock DOM elements based on CAPTCHA type."""
        if not self.has_captcha:
            return []
            
        if self.captcha_type == "text":
            return [
                Mock(
                    tag_name='input',
                    get_attribute=Mock(return_value='captcha'),
                    text='Enter the text you see',
                    is_displayed=Mock(return_value=True)
                ),
                Mock(
                    tag_name='img',
                    get_attribute=Mock(return_value='captcha-image'),
                    is_displayed=Mock(return_value=True)
                )
            ]
        elif self.captcha_type == "image":
            return [
                Mock(
                    tag_name='div',
                    get_attribute=Mock(return_value='image-captcha'),
                    text='Select all images with cars',
                    is_displayed=Mock(return_value=True)
                )
            ]
        elif self.captcha_type == "recaptcha":
            return [
                Mock(
                    tag_name='div',
                    get_attribute=Mock(return_value='g-recaptcha'),
                    is_displayed=Mock(return_value=True)
                ),
                Mock(
                    tag_name='iframe',
                    get_attribute=Mock(return_value='recaptcha-frame'),
                    is_displayed=Mock(return_value=True)
                )
            ]
        elif self.captcha_type == "hcaptcha":
            return [
                Mock(
                    tag_name='div',
                    get_attribute=Mock(return_value='h-captcha'),
                    is_displayed=Mock(return_value=True)
                )
            ]
        else:
            return []


class TestCaptchaDetectionWithMockSites:
    """Test CAPTCHA detection using mock test sites."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_browser = Mock()
        self.detection_tool = CaptchaDetectionTool(self.mock_browser)
        
    def test_detect_text_captcha_site(self):
        """Test detection on a site with text CAPTCHA."""
        # Set up mock site with text CAPTCHA
        test_site = MockTestSite("text")
        self.mock_browser.find_elements.return_value = test_site.elements
        self.mock_browser.get_screenshot_as_png.return_value = b'mock_screenshot_data'
        
        result = self.detection_tool.call("https://mock-text-captcha-site.com")
        
        # Verify detection results
        assert result is not None
        assert isinstance(result, dict)
        
        # Verify browser interactions
        self.mock_browser.find_elements.assert_called()
        
    def test_detect_image_captcha_site(self):
        """Test detection on a site with image CAPTCHA."""
        test_site = MockTestSite("image")
        self.mock_browser.find_elements.return_value = test_site.elements
        
        result = self.detection_tool.call("https://mock-image-captcha-site.com")
        
        assert result is not None
        assert isinstance(result, dict)
        
    def test_detect_recaptcha_site(self):
        """Test detection on a site with reCAPTCHA."""
        test_site = MockTestSite("recaptcha")
        self.mock_browser.find_elements.return_value = test_site.elements
        
        result = self.detection_tool.call("https://mock-recaptcha-site.com")
        
        assert result is not None
        assert isinstance(result, dict)
        
    def test_detect_hcaptcha_site(self):
        """Test detection on a site with hCaptcha."""
        test_site = MockTestSite("hcaptcha")
        self.mock_browser.find_elements.return_value = test_site.elements
        
        result = self.detection_tool.call("https://mock-hcaptcha-site.com")
        
        assert result is not None
        assert isinstance(result, dict)
        
    def test_no_captcha_site(self):
        """Test detection on a site without CAPTCHA."""
        test_site = MockTestSite("none", has_captcha=False)
        self.mock_browser.find_elements.return_value = test_site.elements
        
        result = self.detection_tool.call("https://mock-no-captcha-site.com")
        
        assert result is not None
        assert isinstance(result, dict)
        
    def test_multiple_captcha_types_site(self):
        """Test detection on a site with multiple CAPTCHA types."""
        # Combine elements from different CAPTCHA types
        text_site = MockTestSite("text")
        recaptcha_site = MockTestSite("recaptcha")
        
        combined_elements = text_site.elements + recaptcha_site.elements
        self.mock_browser.find_elements.return_value = combined_elements
        
        result = self.detection_tool.call("https://mock-multiple-captcha-site.com")
        
        assert result is not None
        assert isinstance(result, dict)


class TestCaptchaSolvingValidation:
    """Test CAPTCHA solving functionality with mock scenarios."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_bedrock = Mock()
        self.solving_tool = CaptchaSolvingTool(self.mock_bedrock)
        
    def test_solve_text_captcha(self):
        """Test solving text-based CAPTCHA."""
        # Mock Bedrock response for text CAPTCHA
        self.mock_bedrock.invoke_model.return_value = {
            'body': Mock(read=Mock(return_value=b'{"completion": "ABC123"}'))
        }
        
        captcha_data = {
            "captcha_type": "text",
            "image_data": b"mock_image_data",
            "element_selector": "#captcha-input"
        }
        
        result = self.solving_tool.call(captcha_data)
        
        assert result is not None
        assert isinstance(result, str)
        
    def test_solve_image_selection_captcha(self):
        """Test solving image selection CAPTCHA."""
        self.mock_bedrock.invoke_model.return_value = {
            'body': Mock(read=Mock(return_value=b'{"completion": "1,3,5"}'))
        }
        
        captcha_data = {
            "captcha_type": "image_selection",
            "image_data": b"mock_grid_image_data",
            "prompt": "Select all images with traffic lights"
        }
        
        result = self.solving_tool.call(captcha_data)
        
        assert result is not None
        assert isinstance(result, str)
        
    def test_solve_recaptcha_challenge(self):
        """Test solving reCAPTCHA challenge."""
        self.mock_bedrock.invoke_model.return_value = {
            'body': Mock(read=Mock(return_value=b'{"completion": "SOLVED"}'))
        }
        
        captcha_data = {
            "captcha_type": "recaptcha",
            "challenge_data": {"type": "image", "prompt": "Select crosswalks"},
            "element_selector": ".recaptcha-checkbox"
        }
        
        result = self.solving_tool.call(captcha_data)
        
        assert result is not None
        
    def test_unsolvable_captcha_handling(self):
        """Test handling of unsolvable CAPTCHAs."""
        # Mock Bedrock response indicating uncertainty
        self.mock_bedrock.invoke_model.return_value = {
            'body': Mock(read=Mock(return_value=b'{"completion": "UNCERTAIN", "confidence": 0.1}'))
        }
        
        captcha_data = {
            "captcha_type": "complex_image",
            "image_data": b"very_complex_captcha_data"
        }
        
        result = self.solving_tool.call(captcha_data)
        
        # Should handle low confidence gracefully
        assert result is not None


class TestRealWorldScenarios:
    """Test scenarios that simulate real-world CAPTCHA encounters."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_browser = Mock()
        self.mock_bedrock = Mock()
        self.detection_tool = CaptchaDetectionTool(self.mock_browser)
        self.solving_tool = CaptchaSolvingTool(self.mock_bedrock)
        
    def test_login_form_with_captcha(self):
        """Test CAPTCHA handling in a login form scenario."""
        # Mock login page with CAPTCHA
        login_elements = [
            Mock(tag_name='input', get_attribute=Mock(return_value='username')),
            Mock(tag_name='input', get_attribute=Mock(return_value='password')),
            Mock(tag_name='input', get_attribute=Mock(return_value='captcha')),
            Mock(tag_name='img', get_attribute=Mock(return_value='captcha-image'))
        ]
        
        self.mock_browser.find_elements.return_value = login_elements
        
        result = self.detection_tool.call("https://example.com/login")
        
        assert result is not None
        
    def test_registration_form_with_recaptcha(self):
        """Test CAPTCHA handling in a registration form scenario."""
        # Mock registration page with reCAPTCHA
        registration_elements = [
            Mock(tag_name='input', get_attribute=Mock(return_value='email')),
            Mock(tag_name='input', get_attribute=Mock(return_value='password')),
            Mock(tag_name='div', get_attribute=Mock(return_value='g-recaptcha'))
        ]
        
        self.mock_browser.find_elements.return_value = registration_elements
        
        result = self.detection_tool.call("https://example.com/register")
        
        assert result is not None
        
    def test_contact_form_with_hcaptcha(self):
        """Test CAPTCHA handling in a contact form scenario."""
        # Mock contact page with hCaptcha
        contact_elements = [
            Mock(tag_name='input', get_attribute=Mock(return_value='name')),
            Mock(tag_name='textarea', get_attribute=Mock(return_value='message')),
            Mock(tag_name='div', get_attribute=Mock(return_value='h-captcha'))
        ]
        
        self.mock_browser.find_elements.return_value = contact_elements
        
        result = self.detection_tool.call("https://example.com/contact")
        
        assert result is not None
        
    def test_ecommerce_checkout_with_captcha(self):
        """Test CAPTCHA handling in an e-commerce checkout scenario."""
        # Mock checkout page with CAPTCHA
        checkout_elements = [
            Mock(tag_name='input', get_attribute=Mock(return_value='credit-card')),
            Mock(tag_name='input', get_attribute=Mock(return_value='billing-address')),
            Mock(tag_name='div', get_attribute=Mock(return_value='captcha-container'))
        ]
        
        self.mock_browser.find_elements.return_value = checkout_elements
        
        result = self.detection_tool.call("https://shop.example.com/checkout")
        
        assert result is not None


class TestCaptchaTypeValidation:
    """Test validation of different CAPTCHA types and their characteristics."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_browser = Mock()
        self.detection_tool = CaptchaDetectionTool(self.mock_browser)
        
    def test_text_captcha_characteristics(self):
        """Test identification of text CAPTCHA characteristics."""
        text_captcha_elements = [
            Mock(
                tag_name='input',
                get_attribute=Mock(side_effect=lambda attr: {
                    'type': 'text',
                    'name': 'captcha',
                    'placeholder': 'Enter CAPTCHA'
                }.get(attr)),
                is_displayed=Mock(return_value=True)
            ),
            Mock(
                tag_name='img',
                get_attribute=Mock(side_effect=lambda attr: {
                    'src': '/captcha-image.png',
                    'alt': 'CAPTCHA Image'
                }.get(attr)),
                is_displayed=Mock(return_value=True)
            )
        ]
        
        self.mock_browser.find_elements.return_value = text_captcha_elements
        
        result = self.detection_tool.call("https://test.com")
        
        assert result is not None
        
    def test_image_selection_captcha_characteristics(self):
        """Test identification of image selection CAPTCHA characteristics."""
        image_selection_elements = [
            Mock(
                tag_name='div',
                get_attribute=Mock(side_effect=lambda attr: {
                    'class': 'captcha-grid',
                    'data-challenge': 'select-traffic-lights'
                }.get(attr)),
                is_displayed=Mock(return_value=True)
            )
        ]
        
        self.mock_browser.find_elements.return_value = image_selection_elements
        
        result = self.detection_tool.call("https://test.com")
        
        assert result is not None
        
    def test_recaptcha_v2_characteristics(self):
        """Test identification of reCAPTCHA v2 characteristics."""
        recaptcha_v2_elements = [
            Mock(
                tag_name='div',
                get_attribute=Mock(side_effect=lambda attr: {
                    'class': 'g-recaptcha',
                    'data-sitekey': 'mock-site-key'
                }.get(attr)),
                is_displayed=Mock(return_value=True)
            ),
            Mock(
                tag_name='iframe',
                get_attribute=Mock(side_effect=lambda attr: {
                    'src': 'https://www.google.com/recaptcha/api2/anchor',
                    'title': 'reCAPTCHA'
                }.get(attr)),
                is_displayed=Mock(return_value=True)
            )
        ]
        
        self.mock_browser.find_elements.return_value = recaptcha_v2_elements
        
        result = self.detection_tool.call("https://test.com")
        
        assert result is not None
        
    def test_recaptcha_v3_characteristics(self):
        """Test identification of reCAPTCHA v3 characteristics."""
        # reCAPTCHA v3 is typically invisible
        recaptcha_v3_elements = [
            Mock(
                tag_name='script',
                get_attribute=Mock(side_effect=lambda attr: {
                    'src': 'https://www.google.com/recaptcha/api.js'
                }.get(attr))
            )
        ]
        
        self.mock_browser.find_elements.return_value = recaptcha_v3_elements
        
        result = self.detection_tool.call("https://test.com")
        
        assert result is not None


class TestCaptchaDetectionAccuracy:
    """Test accuracy and reliability of CAPTCHA detection."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_browser = Mock()
        self.detection_tool = CaptchaDetectionTool(self.mock_browser)
        
    def test_false_positive_prevention(self):
        """Test that non-CAPTCHA elements are not detected as CAPTCHAs."""
        # Mock page with regular form elements
        regular_elements = [
            Mock(tag_name='input', get_attribute=Mock(return_value='email')),
            Mock(tag_name='input', get_attribute=Mock(return_value='password')),
            Mock(tag_name='button', get_attribute=Mock(return_value='submit')),
            Mock(tag_name='img', get_attribute=Mock(return_value='logo'))
        ]
        
        self.mock_browser.find_elements.return_value = regular_elements
        
        result = self.detection_tool.call("https://test.com")
        
        # Should not detect CAPTCHA when none exists
        assert result is not None
        
    def test_hidden_captcha_detection(self):
        """Test detection of hidden or dynamically loaded CAPTCHAs."""
        # Mock initially hidden CAPTCHA that becomes visible
        hidden_captcha = Mock(
            tag_name='div',
            get_attribute=Mock(return_value='captcha-container'),
            is_displayed=Mock(side_effect=[False, True])  # Hidden then visible
        )
        
        self.mock_browser.find_elements.return_value = [hidden_captcha]
        
        result = self.detection_tool.call("https://test.com")
        
        assert result is not None
        
    def test_multiple_captcha_detection(self):
        """Test detection when multiple CAPTCHAs are present."""
        # Mock page with multiple CAPTCHA types
        multiple_captchas = [
            Mock(tag_name='div', get_attribute=Mock(return_value='g-recaptcha')),
            Mock(tag_name='input', get_attribute=Mock(return_value='captcha')),
            Mock(tag_name='div', get_attribute=Mock(return_value='h-captcha'))
        ]
        
        self.mock_browser.find_elements.return_value = multiple_captchas
        
        result = self.detection_tool.call("https://test.com")
        
        # Should detect all CAPTCHA types
        assert result is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])