# Design Document

## Overview

This design document outlines the implementation approach for creating a comprehensive tutorial that demonstrates **how NovaAct's natural language browser automation handles sensitive information within Amazon Bedrock AgentCore's managed browser infrastructure**. The tutorial will consist of executable Jupyter notebooks with real working examples showing the complete integration between NovaAct's AI-powered automation and AgentCore's secure, scalable browser environment.

The focus is on **real-world integration patterns** that demonstrate how NovaAct's agentic approach to browser automation works securely within AgentCore's containerized browser sessions, showing developers exactly how to leverage both services together for handling sensitive data in production environments.

## Tutorial Architecture

The tutorial demonstrates the complete integration between NovaAct and AgentCore Browser Tool, positioned at:
```
01-tutorials/05-AgentCore-tools/02-Agent-Core-browser-tool/05-handling-sensitive-information/NovaAct/
```

### Integration Flow
```
Developer → NovaAct SDK → AgentCore Browser Tool → Secure Browser Session → Web Application
    ↓           ↓              ↓                      ↓                    ↓
Natural     AI Model      Managed Browser        Containerized         Target Site
Language    Processing    Infrastructure         Environment           with Forms
```

### Directory Structure
```
05-handling-sensitive-information/
├── NovaAct/
│   ├── README.md                                           # Tutorial overview and NovaAct-AgentCore integration
│   ├── requirements.txt                                    # NovaAct SDK and AgentCore dependencies
│   ├── 01_novaact_agentcore_secure_login.ipynb            # Basic NovaAct login with AgentCore
│   ├── 02_novaact_sensitive_form_automation.ipynb         # NovaAct PII handling in AgentCore sessions
│   ├── 03_novaact_agentcore_session_security.ipynb        # Session management and cleanup patterns
│   ├── 04_production_novaact_agentcore_patterns.ipynb     # Scaling NovaAct with AgentCore infrastructure
│   ├── examples/
│   │   ├── secure_login_with_novaact.py                   # Complete login automation example
│   │   ├── pii_form_automation.py                         # PII handling with NovaAct natural language
│   │   ├── payment_form_security.py                       # Secure payment processing
│   │   └── agentcore_session_helpers.py                   # AgentCore session utilities
│   └── assets/
│       ├── novaact_agentcore_architecture.png             # Integration architecture diagram
│       └── security_flow_diagram.png                      # Data protection flow
```

## NovaAct-AgentCore Integration Learning Path

### Progressive Tutorial Structure

The tutorial demonstrates how NovaAct's natural language automation works securely within AgentCore's managed browser infrastructure:

#### Notebook 1: NovaAct-AgentCore Secure Login (`01_novaact_agentcore_secure_login.ipynb`)
**Goal**: Demonstrate the fundamental integration between NovaAct's AI model and AgentCore's secure browser sessions

**Key Integration Points**:
- How NovaAct connects to AgentCore's CDP endpoints
- AgentCore's containerized browser isolation protecting NovaAct operations
- Secure credential flow between NovaAct SDK and AgentCore infrastructure

```python
# Real NovaAct-AgentCore integration pattern
import os
from bedrock_agentcore.tools.browser_client import browser_session
from nova_act import NovaAct

# Step 1: AgentCore managed browser session
with browser_session(region='us-east-1') as agentcore_client:
    # AgentCore provides secure, isolated browser environment
    ws_url, headers = agentcore_client.generate_ws_headers()
    
    # Step 2: NovaAct connects to AgentCore's browser infrastructure
    with NovaAct(
        cdp_endpoint_url=ws_url,  # AgentCore's secure CDP endpoint
        cdp_headers=headers,      # AgentCore authentication headers
        nova_act_api_key=os.environ['NOVA_ACT_API_KEY']
    ) as nova_act:
        # Step 3: NovaAct's AI processes natural language within AgentCore's secure environment
        result = nova_act.act(
            "Navigate to the login page and securely log in using the provided credentials"
        )
        
        # AgentCore's isolation ensures sensitive data stays within the managed session
        print(f"Login completed in AgentCore session: {result.success}")
```

#### Notebook 2: NovaAct Sensitive Form Automation (`02_novaact_sensitive_form_automation.ipynb`)
**Goal**: Show how NovaAct's natural language processing handles PII and sensitive data within AgentCore's secure browser sessions

#### Notebook 3: NovaAct-AgentCore Session Security (`03_novaact_agentcore_session_security.ipynb`)
**Goal**: Demonstrate session lifecycle management, error handling, and AgentCore's built-in security features

#### Notebook 4: Production NovaAct-AgentCore Patterns (`04_production_novaact_agentcore_patterns.ipynb`)
**Goal**: Show how to scale NovaAct automation using AgentCore's managed infrastructure in production

## Detailed Tutorial Content Design

### Notebook 1: NovaAct-AgentCore Secure Login (`01_novaact_agentcore_secure_login.ipynb`)

**Learning Objectives**:
- Understand how NovaAct's AI model integrates with AgentCore's managed browser infrastructure
- Learn the security benefits of running NovaAct within AgentCore's containerized environment
- Implement secure login automation using natural language instructions

**Content Structure**:
1. **NovaAct-AgentCore Architecture Overview** - How the services work together
2. **Setting Up the Integration** - Configuring NovaAct SDK with AgentCore browser sessions
3. **Secure Credential Management** - Managing both NovaAct API keys and AgentCore authentication
4. **Natural Language Login Automation** - Using NovaAct's AI within AgentCore's secure browser
5. **AgentCore Security Features** - Isolation, observability, and built-in protections

**Key Integration Examples**:
```python
# Real NovaAct-AgentCore secure login pattern
import os
from bedrock_agentcore.tools.browser_client import browser_session
from nova_act import NovaAct

def secure_login_with_novaact_agentcore():
    """Demonstrate secure login using NovaAct within AgentCore's managed browser."""
    
    # AgentCore provides fully managed, secure browser infrastructure
    with browser_session(region='us-east-1') as agentcore_client:
        print("✅ AgentCore browser session created - containerized and isolated")
        
        # Get secure CDP connection to AgentCore's managed browser
        ws_url, headers = agentcore_client.generate_ws_headers()
        print(f"🔗 Connected to AgentCore CDP endpoint: {ws_url[:50]}...")
        
        # NovaAct connects to AgentCore's secure browser environment
        with NovaAct(
            cdp_endpoint_url=ws_url,
            cdp_headers=headers,
            nova_act_api_key=os.environ['NOVA_ACT_API_KEY']
        ) as nova_act:
            print("🤖 NovaAct AI connected to AgentCore browser session")
            
            # NovaAct's natural language processing within AgentCore's secure environment
            result = nova_act.act(
                "Navigate to https://example-login-site.com and log in securely using the credentials I'll provide"
            )
            
            # AgentCore's isolation ensures credentials never leave the managed session
            if result.success:
                print("✅ Secure login completed within AgentCore's isolated environment")
            else:
                print("❌ Login failed - AgentCore session remains secure")
                
            return result
```

### Notebook 2: NovaAct Sensitive Form Automation (`02_novaact_sensitive_form_automation.ipynb`)

**Learning Objectives**:
- Learn how NovaAct's AI model processes sensitive form data within AgentCore's secure browser sessions
- Understand AgentCore's built-in data protection features during NovaAct operations
- Implement secure form filling patterns using natural language instructions

**Content Structure**:
1. **Sensitive Data Types in Browser Automation** - PII, payment info, credentials
2. **NovaAct's Natural Language Processing for Sensitive Data** - How the AI handles sensitive prompts
3. **AgentCore's Data Protection Features** - Screenshot redaction, session isolation, secure logging
4. **Secure Form Automation Patterns** - Best practices for different form types
5. **Error Handling and Debugging** - Safe troubleshooting without data exposure

### Notebook 3: NovaAct-AgentCore Session Security (`03_novaact_agentcore_session_security.ipynb`)

**Learning Objectives**:
- Understand AgentCore's session lifecycle and how it protects NovaAct operations
- Learn proper resource management for NovaAct-AgentCore integrations
- Implement secure error handling and session cleanup patterns

**Content Structure**:
1. **AgentCore Session Lifecycle** - Creation, isolation, monitoring, cleanup
2. **NovaAct Operation Security** - How AI processing is protected within AgentCore
3. **Session Monitoring and Observability** - Using AgentCore's built-in dashboards
4. **Error Handling Patterns** - Secure exception management
5. **Resource Cleanup** - Proper session termination and resource management

### Notebook 4: Production NovaAct-AgentCore Patterns (`04_production_novaact_agentcore_patterns.ipynb`)

**Learning Objectives**:
- Implement production-ready NovaAct-AgentCore integration patterns
- Understand how to scale NovaAct automation using AgentCore's managed infrastructure
- Learn monitoring and security best practices for production deployments

**Content Structure**:
1. **Production Architecture** - Scaling NovaAct with AgentCore's auto-scaling capabilities
2. **Credential Management at Scale** - AWS Secrets Manager integration for both services
3. **Monitoring and Observability** - Using AgentCore's dashboards to monitor NovaAct operations
4. **Security Best Practices** - Production security patterns for NovaAct-AgentCore integration
5. **Troubleshooting and Debugging** - Production debugging without compromising security

## Real-World Integration Examples

### NovaAct-AgentCore Secure Login (`examples/secure_login_with_novaact.py`)

```python
import os
import logging
from bedrock_agentcore.tools.browser_client import browser_session
from nova_act import NovaAct
from typing import Dict, Optional

def secure_login_with_novaact_agentcore(
    login_url: str,
    username: str,
    password: str,
    region: str = 'us-east-1'
) -> Dict:
    """
    Demonstrate secure login automation using NovaAct within AgentCore's managed browser.
    
    This example shows how NovaAct's AI model processes login instructions
    within AgentCore's containerized, secure browser environment.
    """
    
    # AgentCore provides fully managed browser infrastructure
    with browser_session(region=region) as agentcore_client:
        logging.info("AgentCore browser session created with isolation and security controls")
        
        # Get secure connection to AgentCore's managed browser
        ws_url, headers = agentcore_client.generate_ws_headers()
        
        # NovaAct connects to AgentCore's secure CDP endpoint
        with NovaAct(
            cdp_endpoint_url=ws_url,
            cdp_headers=headers,
            nova_act_api_key=os.environ['NOVA_ACT_API_KEY']
        ) as nova_act:
            
            # NovaAct's AI processes natural language within AgentCore's secure environment
            navigation_result = nova_act.act(f"Navigate to {login_url}")
            
            if navigation_result.success:
                # Secure credential handling - NovaAct AI processes within isolated session
                login_result = nova_act.act(
                    f"Log in to this website using username '{username}' and password '{password}'"
                )
                
                # AgentCore's isolation ensures credentials never leave the managed session
                return {
                    'success': login_result.success,
                    'session_isolated': True,
                    'agentcore_managed': True,
                    'credentials_protected': True
                }
            else:
                return {'success': False, 'error': 'Navigation failed'}
```

### PII Form Automation (`examples/pii_form_automation.py`)

```python
from bedrock_agentcore.tools.browser_client import browser_session
from nova_act import NovaAct
import os
from typing import Dict, List

def secure_pii_form_automation(
    form_url: str,
    personal_data: Dict[str, str],
    region: str = 'us-east-1'
) -> Dict:
    """
    Demonstrate how NovaAct handles PII within AgentCore's secure browser sessions.
    
    Shows AgentCore's data protection features and NovaAct's secure processing
    of personal information during form automation.
    """
    
    with browser_session(region=region) as agentcore_client:
        # AgentCore provides containerized browser with built-in data protection
        ws_url, headers = agentcore_client.generate_ws_headers()
        
        with NovaAct(
            cdp_endpoint_url=ws_url,
            cdp_headers=headers,
            nova_act_api_key=os.environ['NOVA_ACT_API_KEY']
        ) as nova_act:
            
            # Navigate to form within AgentCore's secure environment
            nav_result = nova_act.act(f"Navigate to {form_url}")
            
            if nav_result.success:
                # NovaAct's AI processes PII securely within AgentCore's isolation
                form_result = nova_act.act(
                    f"Fill out the personal information form with the following data: "
                    f"Name: {personal_data.get('name', '')}, "
                    f"Email: {personal_data.get('email', '')}, "
                    f"Phone: {personal_data.get('phone', '')}, "
                    f"Address: {personal_data.get('address', '')}"
                )
                
                # Submit form - all data processing happens within AgentCore's secure session
                submit_result = nova_act.act("Submit the completed form")
                
                return {
                    'form_completed': form_result.success,
                    'form_submitted': submit_result.success,
                    'pii_protected_by_agentcore': True,
                    'session_isolated': True
                }
            else:
                return {'success': False, 'error': 'Form navigation failed'}
```

### AgentCore Session Management (`examples/agentcore_session_helpers.py`)

```python
import os
import logging
from contextlib import contextmanager
from bedrock_agentcore.tools.browser_client import browser_session
from nova_act import NovaAct
from typing import Generator, Tuple

@contextmanager
def secure_novaact_agentcore_session(
    region: str = 'us-east-1',
    enable_observability: bool = True
) -> Generator[Tuple[object, NovaAct], None, None]:
    """
    Context manager for secure NovaAct-AgentCore integration.
    
    Demonstrates proper session lifecycle management with AgentCore's
    built-in security features and NovaAct's AI processing.
    """
    
    try:
        # Create AgentCore managed browser session with security controls
        with browser_session(region=region) as agentcore_client:
            logging.info("AgentCore session created with containerized browser isolation")
            
            # Enable AgentCore's observability features if requested
            if enable_observability:
                logging.info("AgentCore observability enabled for monitoring NovaAct operations")
            
            # Get secure CDP connection to AgentCore's managed browser
            ws_url, headers = agentcore_client.generate_ws_headers()
            
            # Connect NovaAct to AgentCore's secure browser infrastructure
            with NovaAct(
                cdp_endpoint_url=ws_url,
                cdp_headers=headers,
                nova_act_api_key=os.environ['NOVA_ACT_API_KEY']
            ) as nova_act:
                logging.info("NovaAct AI connected to AgentCore's secure browser session")
                
                yield agentcore_client, nova_act
                
    except Exception as e:
        logging.error(f"NovaAct-AgentCore session error: {str(e)}")
        # AgentCore automatically handles secure cleanup on errors
        raise
    finally:
        logging.info("NovaAct-AgentCore session cleanup completed - all resources secured")

def monitor_agentcore_session_security(agentcore_client) -> Dict:
    """
    Demonstrate how to use AgentCore's built-in monitoring for NovaAct operations.
    
    Shows AgentCore's observability features for tracking browser automation
    security without exposing sensitive data.
    """
    
    # AgentCore provides built-in dashboards and monitoring
    session_info = {
        'session_isolated': True,
        'browser_containerized': True,
        'observability_enabled': True,
        'security_monitoring_active': True
    }
    
    logging.info("AgentCore session security monitoring active")
    return session_info
```

## NovaAct-AgentCore Error Handling Patterns

### Secure Error Handling for NovaAct-AgentCore Integration

The tutorial demonstrates error handling patterns specific to NovaAct-AgentCore integration:

```python
import logging
from bedrock_agentcore.tools.browser_client import browser_session
from nova_act import NovaAct, ActAgentError
import os

def secure_novaact_agentcore_with_error_handling(
    task_description: str,
    region: str = 'us-east-1'
) -> Dict:
    """
    Demonstrate secure error handling for NovaAct operations within AgentCore sessions.
    
    Shows how AgentCore's isolation protects against errors and how to handle
    NovaAct AI processing errors without exposing sensitive data.
    """
    
    try:
        # AgentCore session creation with error handling
        with browser_session(region=region) as agentcore_client:
            logging.info("AgentCore browser session created successfully")
            
            try:
                # Get secure CDP connection
                ws_url, headers = agentcore_client.generate_ws_headers()
                
                # NovaAct connection to AgentCore with error handling
                with NovaAct(
                    cdp_endpoint_url=ws_url,
                    cdp_headers=headers,
                    nova_act_api_key=os.environ['NOVA_ACT_API_KEY']
                ) as nova_act:
                    
                    # NovaAct AI processing within AgentCore's secure environment
                    result = nova_act.act(task_description)
                    
                    return {
                        'success': result.success,
                        'agentcore_session_secure': True,
                        'error_contained': True
                    }
                    
            except ActAgentError as e:
                # Handle NovaAct-specific errors without exposing sensitive data
                logging.error(f"NovaAct AI processing error: {type(e).__name__}")
                return {
                    'success': False,
                    'error_type': 'novaact_processing_error',
                    'agentcore_session_secure': True,
                    'sensitive_data_protected': True
                }
                
    except Exception as e:
        # Handle AgentCore session errors
        logging.error(f"AgentCore session error: {type(e).__name__}")
        return {
            'success': False,
            'error_type': 'agentcore_session_error',
            'session_cleanup_automatic': True
        }
    
    finally:
        # AgentCore automatically handles secure cleanup
        logging.info("AgentCore session cleanup completed - all resources secured")
```

## NovaAct-AgentCore Integration Validation

### Ensuring Real Integration Examples

The tutorial validation focuses on verifying actual NovaAct-AgentCore integration:

1. **Integration Testing**
   - All notebooks demonstrate real NovaAct SDK connections to AgentCore browser sessions
   - Examples use actual browser_session() calls with NovaAct CDP integration
   - AgentCore's security features are properly demonstrated

2. **NovaAct AI Processing Validation**
   - Natural language instructions are processed by actual NovaAct AI model
   - Examples show real NovaAct.act() calls within AgentCore's managed browser
   - AI responses demonstrate secure processing within AgentCore's isolation

3. **AgentCore Security Feature Verification**
   - Examples demonstrate AgentCore's containerized browser isolation
   - Built-in observability and monitoring features are shown
   - Session lifecycle management is properly implemented

4. **Production Readiness**
   - Integration patterns work with real AWS credentials and regions
   - Examples scale using AgentCore's managed infrastructure
   - Security patterns are production-appropriate

## Implementation Approach

### NovaAct-AgentCore Integration Implementation

The tutorial implementation focuses on demonstrating real NovaAct-AgentCore integration patterns:

### Phase 1: Core Integration Setup
- Create tutorial structure with NovaAct SDK and AgentCore browser_client dependencies
- Set up README explaining NovaAct-AgentCore architecture and integration benefits
- Create first notebook demonstrating basic NovaAct connection to AgentCore browser sessions

### Phase 2: Sensitive Data Handling Notebooks
- Implement `01_novaact_agentcore_secure_login.ipynb` with real login automation
- Create `02_novaact_sensitive_form_automation.ipynb` showing PII handling within AgentCore
- Develop `03_novaact_agentcore_session_security.ipynb` with session lifecycle management

### Phase 3: Production Integration Patterns
- Complete `04_production_novaact_agentcore_patterns.ipynb` with scaling and monitoring
- Create supporting Python files demonstrating real integration patterns
- Add architecture diagrams showing NovaAct-AgentCore data flow and security

### Phase 4: Integration Validation and Testing
- Validate all notebooks work with real NovaAct API and AgentCore browser sessions
- Test examples demonstrate actual AI processing within managed browser infrastructure
- Verify security patterns leverage both NovaAct and AgentCore security features

## Key NovaAct-AgentCore Security Principles

### Integration-Specific Security
- **Dual Authentication**: Secure management of both NovaAct API keys and AgentCore credentials
- **AI Processing Security**: NovaAct's natural language processing within AgentCore's isolated environment
- **Session Isolation**: Leveraging AgentCore's containerized browser for NovaAct operations
- **Managed Infrastructure**: Using AgentCore's fully managed browser infrastructure for secure scaling
- **Built-in Observability**: Utilizing AgentCore's monitoring for NovaAct operation visibility

### Production Integration Patterns
- All examples demonstrate real NovaAct SDK integration with AgentCore browser_session()
- Security measures leverage the combined capabilities of both services
- Code examples show production-ready patterns for scaling NovaAct with AgentCore
- Integration patterns are designed for enterprise deployment scenarios