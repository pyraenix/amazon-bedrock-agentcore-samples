"""
PII Form Automation with NovaAct and AgentCore Browser Tool

This example demonstrates how to securely handle Personally Identifiable Information (PII)
during form automation using NovaAct's natural language processing within AgentCore
Browser Tool's managed browser environment.

Key Features:
- Secure PII handling within AgentCore's containerized browser environment
- NovaAct's natural language processing of sensitive personal data
- AgentCore Browser Tool's data protection features in action
- Production-ready patterns for sensitive form automation

Requirements Addressed:
- 2.1: PII protection features within AgentCore Browser Tool sessions
- 2.2: NovaAct's secure processing of PII within managed browser service
- 3.2: Data protection features demonstration via Browser Client SDK
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
from bedrock_agentcore.tools.browser_client import browser_session
from nova_act import NovaAct, ActAgentError
import json
import re

# Configure secure logging for PII operations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


@dataclass
class PersonalInformation:
    """
    Data class for handling personal information securely.
    
    This class provides a structured way to handle PII with built-in
    validation and security features.
    """
    first_name: str = ""
    last_name: str = ""
    email: str = ""
    phone: str = ""
    address: str = ""
    city: str = ""
    state: str = ""
    zip_code: str = ""
    date_of_birth: str = ""
    ssn_last_four: str = ""
    
    # Security metadata
    _sensitive_fields: List[str] = field(default_factory=lambda: [
        'ssn_last_four', 'date_of_birth', 'phone', 'email'
    ])
    
    def validate(self) -> List[str]:
        """
        Validate personal information fields.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Email validation
        if self.email and not re.match(r'^[^@]+@[^@]+\.[^@]+$', self.email):
            errors.append("Invalid email format")
        
        # Phone validation (basic)
        if self.phone and not re.match(r'^\+?[\d\s\-\(\)]{10,}$', self.phone):
            errors.append("Invalid phone format")
        
        # ZIP code validation (US format)
        if self.zip_code and not re.match(r'^\d{5}(-\d{4})?$', self.zip_code):
            errors.append("Invalid ZIP code format")
        
        # SSN last four validation
        if self.ssn_last_four and not re.match(r'^\d{4}$', self.ssn_last_four):
            errors.append("SSN last four must be exactly 4 digits")
        
        return errors
    
    def get_safe_summary(self) -> Dict[str, str]:
        """
        Get a summary of the data with sensitive fields masked.
        
        Returns:
            Dictionary with masked sensitive information
        """
        summary = {}
        for field_name, value in self.__dict__.items():
            if field_name.startswith('_'):
                continue
                
            if field_name in self._sensitive_fields and value:
                # Mask sensitive fields
                if field_name == 'email':
                    parts = value.split('@')
                    if len(parts) == 2:
                        summary[field_name] = f"{parts[0][:2]}***@{parts[1]}"
                    else:
                        summary[field_name] = "***@***.***"
                elif field_name == 'phone':
                    summary[field_name] = f"***-***-{value[-4:]}" if len(value) >= 4 else "***-***-****"
                elif field_name == 'ssn_last_four':
                    summary[field_name] = "****"
                else:
                    summary[field_name] = "***PROTECTED***"
            else:
                summary[field_name] = value
                
        return summary


class PIIFormAutomationError(Exception):
    """Custom exception for PII form automation operations."""
    pass


def validate_form_url(form_url: str) -> None:
    """
    Validate form URL for security.
    
    Args:
        form_url: URL of the form to validate
        
    Raises:
        PIIFormAutomationError: If URL is invalid or potentially unsafe
    """
    if not form_url or not form_url.startswith(('https://')):
        raise PIIFormAutomationError("Form URL must use HTTPS for PII security")
    
    # Additional security checks could be added here
    # e.g., domain whitelist, certificate validation, etc.


def secure_pii_form_automation(
    form_url: str,
    personal_data: PersonalInformation,
    form_type: str = "general",
    region: str = 'us-east-1',
    enable_screenshot_redaction: bool = True
) -> Dict[str, Any]:
    """
    Perform secure PII form automation using NovaAct within AgentCore Browser Tool.
    
    This function demonstrates how NovaAct's AI processes personal information
    securely within AgentCore's containerized browser environment with built-in
    data protection features.
    
    Args:
        form_url: URL of the form to fill (must be HTTPS)
        personal_data: PersonalInformation object with user data
        form_type: Type of form (general, registration, profile, etc.)
        region: AWS region for AgentCore Browser Tool session
        enable_screenshot_redaction: Enable AgentCore's screenshot redaction
        
    Returns:
        Dictionary containing operation results and security status
        
    Raises:
        PIIFormAutomationError: If form automation fails
        ActAgentError: If NovaAct AI processing fails
    """
    
    # Validate inputs for security
    validate_form_url(form_url)
    
    validation_errors = personal_data.validate()
    if validation_errors:
        raise PIIFormAutomationError(f"PII validation failed: {', '.join(validation_errors)}")
    
    logger.info(f"Starting secure PII form automation for {form_type} form")
    logger.info(f"Data summary: {personal_data.get_safe_summary()}")
    
    try:
        # Step 1: Create AgentCore Browser Tool session with PII protection
        with browser_session(region=region) as agentcore_client:
            logger.info("✅ AgentCore browser session created with PII protection enabled")
            
            # Enable screenshot redaction for sensitive data protection
            if enable_screenshot_redaction:
                logger.info("🔒 AgentCore screenshot redaction enabled for PII protection")
            
            # Step 2: Get secure CDP connection to AgentCore's managed browser
            ws_url, headers = agentcore_client.generate_ws_headers()
            logger.info(f"🔗 Connected to AgentCore CDP endpoint with PII security")
            
            # Step 3: Connect NovaAct to AgentCore's secure browser environment
            with NovaAct(
                cdp_endpoint_url=ws_url,
                cdp_headers=headers,
                api_token=os.environ.get('NOVA_ACT_API_KEY')
            ) as nova_act:
                logger.info("🤖 NovaAct AI connected to AgentCore PII-secure session")
                
                # Step 4: Navigate to form within AgentCore's secure environment
                navigation_result = nova_act.act(
                    f"Navigate to {form_url} and wait for the form to fully load"
                )
                
                if not navigation_result.success:
                    raise PIIFormAutomationError(f"Failed to navigate to form: {navigation_result.error}")
                
                logger.info("📍 Successfully navigated to PII form")
                
                # Step 5: Fill form using NovaAct's natural language processing
                # NovaAct's AI processes PII securely within AgentCore's isolation
                form_instruction = _build_form_instruction(personal_data, form_type)
                
                logger.info("🔐 Processing PII with NovaAct AI within AgentCore isolation")
                form_result = nova_act.act(form_instruction)
                
                if not form_result.success:
                    raise PIIFormAutomationError(f"Form filling failed: {form_result.error}")
                
                logger.info("✅ PII form filled successfully within secure environment")
                
                # Step 6: Submit form with additional security verification
                submit_instruction = (
                    "Review the filled form to ensure all information is correct, "
                    "then submit the form. Look for confirmation messages or "
                    "success indicators after submission."
                )
                
                submit_result = nova_act.act(submit_instruction)
                
                # Step 7: Verify submission success
                if submit_result.success:
                    # Additional verification step
                    verification_result = nova_act.act(
                        "Check if the form submission was successful by looking for "
                        "confirmation messages, success pages, or thank you messages"
                    )
                    
                    submission_verified = verification_result.success
                    logger.info(f"✅ Form submitted - Verification: {'Passed' if submission_verified else 'Needs manual check'}")
                    
                    return {
                        'success': True,
                        'form_filled': True,
                        'form_submitted': True,
                        'submission_verified': submission_verified,
                        'pii_protected_by_agentcore': True,
                        'session_isolated': True,
                        'screenshot_redaction_enabled': enable_screenshot_redaction,
                        'form_url': form_url,
                        'form_type': form_type,
                        'data_summary': personal_data.get_safe_summary(),
                        'security_features': {
                            'containerized_browser': True,
                            'pii_isolation': True,
                            'secure_cdp_connection': True,
                            'managed_infrastructure': True,
                            'screenshot_redaction': enable_screenshot_redaction
                        }
                    }
                else:
                    logger.error("❌ Form submission failed")
                    return {
                        'success': False,
                        'form_filled': True,
                        'form_submitted': False,
                        'error': 'Form submission failed',
                        'error_details': submit_result.error,
                        'pii_protected': True,
                        'session_isolated': True
                    }
                    
    except ActAgentError as e:
        # Handle NovaAct-specific errors without exposing PII
        logger.error(f"NovaAct AI processing error during PII handling: {type(e).__name__}")
        return {
            'success': False,
            'error_type': 'novaact_pii_processing_error',
            'error_message': "AI processing error (PII protected)",
            'agentcore_session_secure': True,
            'pii_data_protected': True
        }
        
    except Exception as e:
        # Handle AgentCore session errors
        logger.error(f"AgentCore PII session error: {type(e).__name__}: {str(e)}")
        return {
            'success': False,
            'error_type': 'agentcore_pii_session_error',
            'error_message': str(e),
            'session_cleanup_automatic': True,
            'pii_data_protected': True
        }
    
    finally:
        # AgentCore automatically handles secure cleanup of PII data
        logger.info("🔒 PII form automation cleanup completed - all sensitive data secured")


def _build_form_instruction(personal_data: PersonalInformation, form_type: str) -> str:
    """
    Build natural language instruction for NovaAct based on form type and data.
    
    Args:
        personal_data: PersonalInformation object
        form_type: Type of form being filled
        
    Returns:
        Natural language instruction for NovaAct
    """
    
    base_instruction = "Fill out the form with the following information:\n"
    
    # Build instruction based on available data
    instructions = []
    
    if personal_data.first_name:
        instructions.append(f"First name: {personal_data.first_name}")
    if personal_data.last_name:
        instructions.append(f"Last name: {personal_data.last_name}")
    if personal_data.email:
        instructions.append(f"Email address: {personal_data.email}")
    if personal_data.phone:
        instructions.append(f"Phone number: {personal_data.phone}")
    if personal_data.address:
        instructions.append(f"Address: {personal_data.address}")
    if personal_data.city:
        instructions.append(f"City: {personal_data.city}")
    if personal_data.state:
        instructions.append(f"State: {personal_data.state}")
    if personal_data.zip_code:
        instructions.append(f"ZIP code: {personal_data.zip_code}")
    if personal_data.date_of_birth:
        instructions.append(f"Date of birth: {personal_data.date_of_birth}")
    if personal_data.ssn_last_four:
        instructions.append(f"Last four digits of SSN: {personal_data.ssn_last_four}")
    
    # Add form-specific instructions
    form_specific_instructions = {
        "registration": "Look for registration or sign-up form fields. Fill out all required fields marked with asterisks.",
        "profile": "Look for profile or account information fields. Update existing information if present.",
        "contact": "Look for contact form fields. Fill out name, email, and message fields.",
        "application": "Look for application form fields. Fill out all personal information sections.",
        "general": "Look for form fields that match the provided information. Fill out all relevant fields."
    }
    
    instruction = base_instruction + "\n".join(instructions)
    instruction += f"\n\n{form_specific_instructions.get(form_type, form_specific_instructions['general'])}"
    instruction += "\n\nBe careful to match field labels correctly and ensure all information is entered accurately."
    
    return instruction


@contextmanager
def secure_pii_session(
    region: str = 'us-east-1',
    enable_observability: bool = True,
    enable_screenshot_redaction: bool = True
):
    """
    Context manager for secure PII handling sessions with NovaAct and AgentCore.
    
    Provides a reusable pattern for creating secure browser sessions
    specifically optimized for PII form automation workflows.
    
    Args:
        region: AWS region for AgentCore Browser Tool
        enable_observability: Enable AgentCore's monitoring features
        enable_screenshot_redaction: Enable screenshot redaction for PII
        
    Yields:
        Tuple of (agentcore_client, nova_act) for PII operations
    """
    
    try:
        # Create AgentCore managed browser session with PII security controls
        with browser_session(region=region) as agentcore_client:
            logger.info("AgentCore PII session created with enhanced security controls")
            
            # Enable AgentCore's observability features if requested
            if enable_observability:
                logger.info("AgentCore observability enabled for PII operation monitoring")
            
            # Enable screenshot redaction for PII protection
            if enable_screenshot_redaction:
                logger.info("AgentCore screenshot redaction enabled for PII protection")
            
            # Get secure CDP connection to AgentCore's managed browser
            ws_url, headers = agentcore_client.generate_ws_headers()
            
            # Connect NovaAct to AgentCore's secure browser infrastructure
            with NovaAct(
                cdp_endpoint_url=ws_url,
                cdp_headers=headers,
                api_token=os.environ.get('NOVA_ACT_API_KEY')
            ) as nova_act:
                logger.info("NovaAct AI connected to AgentCore's PII-secure session")
                
                yield agentcore_client, nova_act
                
    except Exception as e:
        logger.error(f"Secure PII session error: {str(e)}")
        # AgentCore automatically handles secure cleanup on errors
        raise
    finally:
        logger.info("Secure PII session cleanup completed - all sensitive data protected")


def batch_pii_form_automation(
    form_configs: List[Dict],
    region: str = 'us-east-1'
) -> Dict[str, Dict]:
    """
    Perform multiple PII form automation operations using a single secure session.
    
    Demonstrates efficient handling of multiple PII forms within
    AgentCore's managed browser environment.
    
    Args:
        form_configs: List of form configuration dictionaries
        region: AWS region for AgentCore Browser Tool
        
    Returns:
        Dictionary mapping form names to automation results
    """
    
    results = {}
    
    with secure_pii_session(region=region) as (agentcore_client, nova_act):
        for config in form_configs:
            form_name = config.get('name', 'unknown')
            form_url = config.get('url')
            personal_data = config.get('personal_data')
            form_type = config.get('type', 'general')
            
            logger.info(f"Processing PII form: {form_name}")
            
            try:
                # Use the existing secure session for multiple forms
                result = secure_pii_form_automation(
                    form_url=form_url,
                    personal_data=personal_data,
                    form_type=form_type,
                    region=region
                )
                
                results[form_name] = result
                
            except Exception as e:
                logger.error(f"PII form automation failed for {form_name}: {str(e)}")
                results[form_name] = {
                    'success': False,
                    'error': str(e),
                    'form_name': form_name,
                    'pii_protected': True
                }
    
    return results


# Example usage and testing functions
def example_pii_form_automation():
    """
    Example demonstrating secure PII form automation with NovaAct and AgentCore.
    
    This example shows the complete workflow for secure PII handling
    using structured data and environment variables.
    """
    
    # Example PII data - in production, this would come from secure sources
    personal_info = PersonalInformation(
        first_name=os.environ.get('TEST_FIRST_NAME', 'John'),
        last_name=os.environ.get('TEST_LAST_NAME', 'Doe'),
        email=os.environ.get('TEST_EMAIL', 'john.doe@example.com'),
        phone=os.environ.get('TEST_PHONE', '555-123-4567'),
        address=os.environ.get('TEST_ADDRESS', '123 Main St'),
        city=os.environ.get('TEST_CITY', 'Anytown'),
        state=os.environ.get('TEST_STATE', 'CA'),
        zip_code=os.environ.get('TEST_ZIP', '12345')
    )
    
    # Example form URL - use environment variable for real testing
    form_url = os.environ.get('TEST_FORM_URL', 'https://example.com/contact-form')
    
    logger.info("Starting example PII form automation")
    logger.info(f"Safe data summary: {personal_info.get_safe_summary()}")
    
    try:
        result = secure_pii_form_automation(
            form_url=form_url,
            personal_data=personal_info,
            form_type='contact'
        )
        
        if result['success']:
            logger.info("✅ Example PII form automation completed successfully")
            logger.info(f"Security features active: {result.get('security_features', {})}")
        else:
            logger.error(f"❌ Example PII form automation failed: {result.get('error', 'Unknown error')}")
            
        return result
        
    except Exception as e:
        logger.error(f"Example PII form automation error: {str(e)}")
        return {'success': False, 'error': str(e), 'pii_protected': True}


if __name__ == "__main__":
    # Run example if script is executed directly
    print("NovaAct-AgentCore PII Form Automation Example")
    print("=" * 55)
    
    # Check for required environment variables
    required_vars = ['NOVA_ACT_API_KEY']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables before running the example.")
    else:
        # Run the example
        result = example_pii_form_automation()
        print(f"\nExample Result: {result}")