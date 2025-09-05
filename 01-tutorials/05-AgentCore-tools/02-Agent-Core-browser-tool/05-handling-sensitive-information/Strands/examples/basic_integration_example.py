"""
Basic Integration Example: Strands with AgentCore Browser Tool

This example demonstrates the basic integration between Strands agents and
AgentCore Browser Tool, showcasing secure browser automation capabilities
and sensitive data handling.

Key Features Demonstrated:
- Creating and configuring AgentCore Browser Tool for Strands
- Secure browser session management
- Basic browser automation (navigate, click, fill forms)
- Sensitive data detection and masking
- Credential management integration
- Comprehensive audit logging

Requirements Addressed:
- 1.2: Secure credential management patterns
- 1.3: Proper data isolation and protection mechanisms
- 1.5: Browser automation methods that send commands to AgentCore Browser Tool
- 2.1: PII detection, masking, and classification in Strands workflows
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

# Strands framework imports
from strands import Agent
from strands.tools import tool
from strands_tools.browser.agent_core_browser import AgentCoreBrowser

# Add tools directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))

from agentcore_browser_tool import (
    AgentCoreBrowserTool,
    create_secure_browser_tool,
    create_authenticated_browser_tool
)

from sensitive_data_handler import (
    SensitiveDataHandler,
    create_secure_data_handler,
    PIIType,
    MaskingStrategy
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StrandsAgentCoreIntegrationExample:
    """
    Example class demonstrating Strands integration with AgentCore Browser Tool.
    
    This class shows how to use the custom Strands tools for secure browser
    automation and sensitive data handling in a production-ready manner.
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize the integration example.
        
        Args:
            session_id: Optional session ID for tracking
        """
        self.session_id = session_id or "strands-example-session"
        
        # Initialize browser tool
        self.browser_tool = create_secure_browser_tool(
            region="us-east-1",
            session_timeout=300,
            enable_observability=True,
            enable_screenshot_redaction=True
        )
        
        # Initialize sensitive data handler
        self.data_handler = create_secure_data_handler(
            session_id=self.session_id,
            agent_id="example-strands-agent",
            region="us-east-1",
            strict_mode=False
        )
        
        logger.info(f"StrandsAgentCoreIntegrationExample initialized: {self.session_id}")
    
    def demonstrate_basic_browser_automation(self) -> Dict[str, Any]:
        """
        Demonstrate basic browser automation capabilities.
        
        Returns:
            Dictionary containing demonstration results
        """
        
        logger.info("=== Demonstrating Basic Browser Automation ===")
        
        results = {
            'operations': [],
            'success': True,
            'errors': []
        }
        
        try:
            # Create browser session
            logger.info("1. Creating secure browser session...")
            result = self.browser_tool.create_session()
            results['operations'].append({
                'operation': 'create_session',
                'success': result.success,
                'data': result.data
            })
            
            if not result.success:
                results['success'] = False
                results['errors'].append(f"Session creation failed: {result.error}")
                return results
            
            # Navigate to a page
            logger.info("2. Navigating to example website...")
            result = self.browser_tool.navigate(
                url="https://example.com",
                wait_for_selector="body"
            )
            results['operations'].append({
                'operation': 'navigate',
                'success': result.success,
                'data': result.data
            })
            
            # Click an element
            logger.info("3. Clicking page element...")
            result = self.browser_tool.click(
                selector="a[href]",
                wait_timeout=5
            )
            results['operations'].append({
                'operation': 'click',
                'success': result.success,
                'data': result.data
            })
            
            # Fill a form (with sensitive data)
            logger.info("4. Filling form with sensitive data...")
            form_data = {
                'username': 'john.doe@example.com',
                'password': 'SecurePassword123!',
                'phone': '555-123-4567'
            }
            
            result = self.browser_tool.fill_form(
                form_data=form_data,
                form_selector="#login-form"
            )
            results['operations'].append({
                'operation': 'fill_form',
                'success': result.success,
                'data': result.data
            })
            
            # Extract data from page
            logger.info("5. Extracting data from page...")
            result = self.browser_tool.extract_data(
                selectors=["h1", ".content", "#main"],
                extract_type="text"
            )
            results['operations'].append({
                'operation': 'extract_data',
                'success': result.success,
                'data': result.data
            })
            
            # Get session metrics
            logger.info("6. Getting session metrics...")
            result = self.browser_tool.get_metrics()
            results['operations'].append({
                'operation': 'get_metrics',
                'success': result.success,
                'data': result.data
            })
            
            # Close session
            logger.info("7. Closing browser session...")
            result = self.browser_tool.close_session()
            results['operations'].append({
                'operation': 'close_session',
                'success': result.success,
                'data': result.data
            })
            
            logger.info("✅ Basic browser automation demonstration completed")
            
        except Exception as e:
            logger.error(f"Error in browser automation demonstration: {str(e)}")
            results['success'] = False
            results['errors'].append(str(e))
        
        return results
    
    def demonstrate_sensitive_data_handling(self) -> Dict[str, Any]:
        """
        Demonstrate sensitive data detection and handling capabilities.
        
        Returns:
            Dictionary containing demonstration results
        """
        
        logger.info("=== Demonstrating Sensitive Data Handling ===")
        
        results = {
            'pii_detections': [],
            'sanitization_results': [],
            'credential_operations': [],
            'audit_entries': [],
            'success': True,
            'errors': []
        }
        
        try:
            # Test data with various PII types
            test_data = [
                "Contact John Doe at john.doe@example.com or call 555-123-4567",
                "SSN: 123-45-6789, Credit Card: 4532-1234-5678-9012",
                "API Key: sk_test_1234567890abcdef, Password: MySecretPass123",
                "Address: 123 Main St, Anytown, ST 12345"
            ]
            
            # Detect PII in each test string
            logger.info("1. Detecting PII in test data...")
            for i, text in enumerate(test_data):
                detections = self.data_handler.detect_pii_in_text(text)
                results['pii_detections'].append({
                    'text_index': i,
                    'original_text': text,
                    'detections': detections
                })
                logger.info(f"   Text {i+1}: Found {len(detections)} PII instances")
            
            # Sanitize the test data
            logger.info("2. Sanitizing test data...")
            for i, text in enumerate(test_data):
                sanitized = self.data_handler.sanitize_text(text)
                results['sanitization_results'].append({
                    'text_index': i,
                    'original_text': text,
                    'sanitized_text': sanitized
                })
                logger.info(f"   Text {i+1}: Sanitized successfully")
            
            # Demonstrate credential management
            logger.info("3. Demonstrating credential management...")
            
            # Store credentials
            cred_success = self.data_handler.store_credentials(
                credential_id="example-login",
                username="testuser@example.com",
                password="SecurePassword123!"
            )
            results['credential_operations'].append({
                'operation': 'store_credentials',
                'success': cred_success,
                'credential_id': 'example-login'
            })
            
            # Retrieve credentials
            credentials = self.data_handler.get_credentials("example-login")
            results['credential_operations'].append({
                'operation': 'retrieve_credentials',
                'success': credentials is not None,
                'credential_id': 'example-login',
                'credentials_found': credentials is not None
            })
            
            # Get audit log
            logger.info("4. Retrieving audit log...")
            audit_entries = self.data_handler.get_audit_log()
            results['audit_entries'] = audit_entries
            logger.info(f"   Found {len(audit_entries)} audit entries")
            
            logger.info("✅ Sensitive data handling demonstration completed")
            
        except Exception as e:
            logger.error(f"Error in sensitive data handling demonstration: {str(e)}")
            results['success'] = False
            results['errors'].append(str(e))
        
        return results
    
    def demonstrate_authenticated_workflow(self) -> Dict[str, Any]:
        """
        Demonstrate authenticated browser workflow with credential injection.
        
        Returns:
            Dictionary containing demonstration results
        """
        
        logger.info("=== Demonstrating Authenticated Workflow ===")
        
        results = {
            'workflow_steps': [],
            'success': True,
            'errors': []
        }
        
        try:
            # Create authenticated browser tool
            auth_browser_tool = create_authenticated_browser_tool(
                username_field="email",
                password_field="password",
                login_url="https://example.com/login",
                login_button_selector="#login-button",
                success_indicator=".dashboard",
                region="us-east-1"
            )
            
            # Store credentials for the workflow
            logger.info("1. Storing credentials securely...")
            cred_success = self.data_handler.store_credentials(
                credential_id="workflow-login",
                username="workflow.user@example.com",
                password="WorkflowPassword123!"
            )
            
            results['workflow_steps'].append({
                'step': 'store_credentials',
                'success': cred_success,
                'description': 'Stored workflow credentials securely'
            })
            
            # Retrieve credentials for authentication
            logger.info("2. Retrieving credentials for authentication...")
            credentials = self.data_handler.get_credentials("workflow-login")
            
            if credentials:
                username, password = credentials
                
                # Perform authenticated login
                logger.info("3. Performing secure authentication...")
                result = auth_browser_tool.authenticate(
                    username=username,
                    password=password,
                    login_url="https://example.com/login"
                )
                
                results['workflow_steps'].append({
                    'step': 'authenticate',
                    'success': result.success,
                    'description': 'Performed secure authentication',
                    'data': result.data
                })
                
                # Navigate to protected page
                logger.info("4. Navigating to protected page...")
                result = auth_browser_tool.navigate(
                    url="https://example.com/dashboard",
                    wait_for_selector=".dashboard-content"
                )
                
                results['workflow_steps'].append({
                    'step': 'navigate_protected',
                    'success': result.success,
                    'description': 'Navigated to protected page',
                    'data': result.data
                })
                
                # Extract sensitive data from protected page
                logger.info("5. Extracting data from protected page...")
                result = auth_browser_tool.extract_data(
                    selectors=[".user-info", ".account-details"],
                    extract_type="text"
                )
                
                # Process extracted data for sensitive information
                if result.success and result.data:
                    processed_result = self.data_handler.process_tool_output(result)
                    
                    results['workflow_steps'].append({
                        'step': 'extract_and_sanitize',
                        'success': True,
                        'description': 'Extracted and sanitized sensitive data',
                        'original_data': result.data,
                        'processed_data': processed_result.data,
                        'pii_detected': processed_result.metadata.get('pii_detected', False)
                    })
                
                # Get final metrics
                logger.info("6. Getting workflow metrics...")
                result = auth_browser_tool.get_metrics()
                
                results['workflow_steps'].append({
                    'step': 'get_metrics',
                    'success': result.success,
                    'description': 'Retrieved workflow metrics',
                    'data': result.data
                })
                
                # Close session
                auth_browser_tool.close_session()
                
            else:
                results['success'] = False
                results['errors'].append("Failed to retrieve credentials for workflow")
            
            logger.info("✅ Authenticated workflow demonstration completed")
            
        except Exception as e:
            logger.error(f"Error in authenticated workflow demonstration: {str(e)}")
            results['success'] = False
            results['errors'].append(str(e))
        
        return results
    
    def run_complete_demonstration(self) -> Dict[str, Any]:
        """
        Run the complete integration demonstration.
        
        Returns:
            Dictionary containing all demonstration results
        """
        
        logger.info("🚀 Starting Complete Strands-AgentCore Integration Demonstration")
        logger.info("=" * 70)
        
        complete_results = {
            'session_id': self.session_id,
            'demonstrations': {},
            'overall_success': True,
            'summary': {}
        }
        
        # Run basic browser automation demonstration
        browser_results = self.demonstrate_basic_browser_automation()
        complete_results['demonstrations']['browser_automation'] = browser_results
        
        if not browser_results['success']:
            complete_results['overall_success'] = False
        
        # Run sensitive data handling demonstration
        data_results = self.demonstrate_sensitive_data_handling()
        complete_results['demonstrations']['sensitive_data_handling'] = data_results
        
        if not data_results['success']:
            complete_results['overall_success'] = False
        
        # Run authenticated workflow demonstration
        workflow_results = self.demonstrate_authenticated_workflow()
        complete_results['demonstrations']['authenticated_workflow'] = workflow_results
        
        if not workflow_results['success']:
            complete_results['overall_success'] = False
        
        # Generate summary
        complete_results['summary'] = {
            'browser_operations': len(browser_results.get('operations', [])),
            'pii_detections': sum(
                len(d['detections']) for d in data_results.get('pii_detections', [])
            ),
            'sanitization_operations': len(data_results.get('sanitization_results', [])),
            'credential_operations': len(data_results.get('credential_operations', [])),
            'workflow_steps': len(workflow_results.get('workflow_steps', [])),
            'audit_entries': len(data_results.get('audit_entries', [])),
            'total_errors': (
                len(browser_results.get('errors', [])) +
                len(data_results.get('errors', [])) +
                len(workflow_results.get('errors', []))
            )
        }
        
        # Final audit log
        final_audit = self.data_handler.get_audit_log()
        complete_results['final_audit_log'] = final_audit
        
        logger.info("=" * 70)
        if complete_results['overall_success']:
            logger.info("✅ Complete demonstration SUCCESSFUL")
        else:
            logger.info("❌ Complete demonstration completed with ERRORS")
        
        logger.info(f"📊 Summary: {complete_results['summary']}")
        logger.info("=" * 70)
        
        return complete_results


def main():
    """Main function to run the integration example."""
    
    print("Strands-AgentCore Browser Tool Integration Example")
    print("=" * 55)
    print()
    
    # Check for required environment variables
    required_vars = []  # Add any required environment variables here
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables before running the example.")
        return
    
    try:
        # Create and run the integration example
        example = StrandsAgentCoreIntegrationExample()
        results = example.run_complete_demonstration()
        
        # Print results summary
        print("\n" + "=" * 55)
        print("DEMONSTRATION RESULTS SUMMARY")
        print("=" * 55)
        
        print(f"Session ID: {results['session_id']}")
        print(f"Overall Success: {results['overall_success']}")
        print(f"Browser Operations: {results['summary']['browser_operations']}")
        print(f"PII Detections: {results['summary']['pii_detections']}")
        print(f"Sanitization Operations: {results['summary']['sanitization_operations']}")
        print(f"Credential Operations: {results['summary']['credential_operations']}")
        print(f"Workflow Steps: {results['summary']['workflow_steps']}")
        print(f"Audit Entries: {results['summary']['audit_entries']}")
        print(f"Total Errors: {results['summary']['total_errors']}")
        
        if results['overall_success']:
            print("\n✅ All demonstrations completed successfully!")
        else:
            print("\n⚠️ Some demonstrations completed with errors.")
            print("Check the detailed results for more information.")
        
        # Optionally save results to file
        import json
        results_file = "strands_agentcore_integration_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📄 Detailed results saved to: {results_file}")
        
    except Exception as e:
        print(f"\n❌ Error running integration example: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()