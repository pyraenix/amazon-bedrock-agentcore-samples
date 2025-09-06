# LlamaIndex AgentCore Browser Integration - API Reference

## Overview

The LlamaIndex AgentCore Browser Integration provides a comprehensive set of tools and classes for integrating LlamaIndex agents with Amazon Bedrock AgentCore's browser automation service. This integration enables intelligent agents to perform secure web browsing, content extraction, and interactive web tasks within AgentCore's VM-isolated environment.

## Table of Contents

- [Core Classes](#core-classes)
  - [LlamaIndexAgentCoreIntegration](#llamaindexagentcoreintegration)
  - [AgentCoreBrowserClient](#agentcorebrowserclient)
  - [ConfigurationManager](#configurationmanager)
- [LlamaIndex Tools](#llamaindex-tools)
  - [BrowserNavigationTool](#browsernavigationtool)
  - [TextExtractionTool](#textextractiontool)
  - [ScreenshotCaptureTool](#screenshotcapturetool)
  - [ElementClickTool](#elementclicktool)
  - [FormInteractionTool](#forminteractiontool)
  - [CaptchaDetectionTool](#captchadetectiontool)
- [Data Models](#data-models)
  - [BrowserResponse](#browserresponse)
  - [ElementSelector](#elementselector)
  - [SessionStatus](#sessionstatus)
- [Error Handling](#error-handling)
  - [Exception Hierarchy](#exception-hierarchy)
  - [Error Recovery](#error-recovery)
- [Configuration](#configuration)
  - [Configuration Options](#configuration-options)
  - [Environment Variables](#environment-variables)

---

## Core Classes

### LlamaIndexAgentCoreIntegration

The main entry point for integrating LlamaIndex agents with AgentCore browser capabilities.

#### Constructor

```python
LlamaIndexAgentCoreIntegration(
    config_path: Optional[str] = None,
    config_manager: Optional[ConfigurationManager] = None,
    aws_credentials: Optional[Dict[str, str]] = None,
    llm_model: Optional[str] = None,
    vision_model: Optional[str] = None
)
```

**Parameters:**
- `config_path` (str, optional): Path to YAML configuration file
- `config_manager` (ConfigurationManager, optional): Pre-configured configuration manager
- `aws_credentials` (dict, optional): AWS credentials dictionary with keys: `aws_access_key_id`, `aws_secret_access_key`, `region`
- `llm_model` (str, optional): Bedrock LLM model identifier (default: "anthropic.claude-3-sonnet-20240229-v1:0")
- `vision_model` (str, optional): Bedrock vision model identifier (default: "anthropic.claude-3-sonnet-20240229-v1:0")

#### Methods

##### `create_agent() -> ReActAgent`

Creates a LlamaIndex ReActAgent configured with browser tools.

**Returns:**
- `ReActAgent`: Configured LlamaIndex agent with browser capabilities

**Example:**
```python
integration = LlamaIndexAgentCoreIntegration()
agent = integration.create_agent()
response = await agent.achat("Navigate to https://example.com and extract the main content")
```

##### `get_browser_tools() -> List[BaseTool]`

Returns list of available browser tools for manual agent configuration.

**Returns:**
- `List[BaseTool]`: List of LlamaIndex tools for browser operations

##### `process_web_content(url: str, instructions: str = None) -> Dict[str, Any]`

High-level method for processing web content with intelligent agent orchestration.

**Parameters:**
- `url` (str): Target URL to process
- `instructions` (str, optional): Specific instructions for content processing

**Returns:**
- `Dict[str, Any]`: Processing results including extracted content and metadata

---

### AgentCoreBrowserClient

Low-level client for direct interaction with AgentCore browser service.

#### Constructor

```python
AgentCoreBrowserClient(
    config_manager: Optional[ConfigurationManager] = None,
    session_id: Optional[str] = None
)
```

**Parameters:**
- `config_manager` (ConfigurationManager, optional): Configuration manager instance
- `session_id` (str, optional): Existing session ID to reuse

#### Methods

##### `async create_session() -> str`

Creates a new browser session in AgentCore.

**Returns:**
- `str`: Session ID for the created browser session

**Raises:**
- `SessionError`: If session creation fails
- `AuthenticationError`: If AWS credentials are invalid

##### `async navigate(url: str, wait_for_load: bool = True, timeout: int = 30) -> BrowserResponse`

Navigates to the specified URL.

**Parameters:**
- `url` (str): Target URL
- `wait_for_load` (bool): Whether to wait for page load completion
- `timeout` (int): Maximum wait time in seconds

**Returns:**
- `BrowserResponse`: Navigation result with page information

**Raises:**
- `NavigationError`: If navigation fails
- `TimeoutError`: If operation times out

##### `async take_screenshot(element_selector: Optional[ElementSelector] = None) -> BrowserResponse`

Captures a screenshot of the current page or specific element.

**Parameters:**
- `element_selector` (ElementSelector, optional): Element to screenshot (full page if None)

**Returns:**
- `BrowserResponse`: Screenshot data in base64 format

##### `async extract_text(selector: Optional[ElementSelector] = None) -> BrowserResponse`

Extracts text content from page or specific elements.

**Parameters:**
- `selector` (ElementSelector, optional): Element selector for targeted extraction

**Returns:**
- `BrowserResponse`: Extracted text content

##### `async click_element(selector: ElementSelector, wait_for_response: bool = True) -> BrowserResponse`

Clicks on a web element.

**Parameters:**
- `selector` (ElementSelector): Element selector
- `wait_for_response` (bool): Whether to wait for page response after click

**Returns:**
- `BrowserResponse`: Click operation result

**Raises:**
- `ElementNotFoundError`: If element is not found

##### `async type_text(selector: ElementSelector, text: str, clear_first: bool = True) -> BrowserResponse`

Types text into an input element.

**Parameters:**
- `selector` (ElementSelector): Input element selector
- `text` (str): Text to type
- `clear_first` (bool): Whether to clear existing text first

**Returns:**
- `BrowserResponse`: Type operation result

##### `async close_session() -> BrowserResponse`

Closes the current browser session.

**Returns:**
- `BrowserResponse`: Session closure confirmation

---

### ConfigurationManager

Manages configuration settings for the integration.

#### Constructor

```python
ConfigurationManager(config_path: Optional[str] = None)
```

**Parameters:**
- `config_path` (str, optional): Path to YAML configuration file

#### Methods

##### `get_config() -> IntegrationConfig`

Returns the current configuration.

**Returns:**
- `IntegrationConfig`: Current configuration object

##### `validate_config() -> bool`

Validates the current configuration.

**Returns:**
- `bool`: True if configuration is valid

**Raises:**
- `ConfigurationError`: If configuration is invalid

---

## LlamaIndex Tools

### BrowserNavigationTool

LlamaIndex tool for web page navigation.

#### Tool Metadata
- **Name:** `navigate_browser`
- **Description:** Navigate to a URL using AgentCore browser tool

#### Input Schema
```python
{
    "url": str,  # Required: URL to navigate to
    "wait_for_load": bool,  # Optional: Wait for page load (default: True)
    "timeout": int  # Optional: Timeout in seconds (default: 30)
}
```

#### Output
```python
{
    "success": bool,
    "url": str,
    "title": str,
    "status_code": int,
    "load_time": float,
    "error": Optional[str]
}
```

#### Example Usage
```python
tool = BrowserNavigationTool(browser_client)
result = await tool.acall(url="https://example.com", wait_for_load=True)
```

---

### TextExtractionTool

LlamaIndex tool for extracting text content from web pages.

#### Tool Metadata
- **Name:** `extract_text`
- **Description:** Extract text content from web page or specific elements

#### Input Schema
```python
{
    "selector": Optional[str],  # CSS selector or XPath
    "clean_text": bool,  # Remove extra whitespace (default: True)
    "include_links": bool  # Include link URLs (default: False)
}
```

#### Output
```python
{
    "success": bool,
    "text": str,
    "word_count": int,
    "char_count": int,
    "links": Optional[List[str]],
    "error": Optional[str]
}
```

---

### ScreenshotCaptureTool

LlamaIndex tool for capturing screenshots.

#### Tool Metadata
- **Name:** `capture_screenshot`
- **Description:** Capture screenshot of current page or specific element

#### Input Schema
```python
{
    "element_selector": Optional[str],  # Element to screenshot
    "full_page": bool,  # Capture full page (default: True)
    "format": str  # Image format: 'png' or 'jpeg' (default: 'png')
}
```

#### Output
```python
{
    "success": bool,
    "image_data": str,  # Base64 encoded image
    "width": int,
    "height": int,
    "format": str,
    "error": Optional[str]
}
```

---

### ElementClickTool

LlamaIndex tool for clicking web elements.

#### Tool Metadata
- **Name:** `click_element`
- **Description:** Click on web elements using various selector methods

#### Input Schema
```python
{
    "selector": str,  # Required: Element selector
    "selector_type": str,  # 'css', 'xpath', 'text' (default: 'css')
    "wait_for_response": bool,  # Wait for page response (default: True)
    "double_click": bool  # Perform double click (default: False)
}
```

#### Output
```python
{
    "success": bool,
    "element_found": bool,
    "click_coordinates": Optional[Dict[str, int]],
    "page_changed": bool,
    "error": Optional[str]
}
```

---

### FormInteractionTool

LlamaIndex tool for form interactions.

#### Tool Metadata
- **Name:** `interact_with_form`
- **Description:** Fill forms, select dropdowns, and handle form elements

#### Input Schema
```python
{
    "form_data": Dict[str, Any],  # Field selector -> value mapping
    "submit": bool,  # Submit form after filling (default: False)
    "submit_selector": Optional[str]  # Submit button selector
}
```

#### Output
```python
{
    "success": bool,
    "fields_filled": List[str],
    "form_submitted": bool,
    "validation_errors": Optional[List[str]],
    "error": Optional[str]
}
```

---

### CaptchaDetectionTool

LlamaIndex tool for CAPTCHA detection and analysis.

#### Tool Metadata
- **Name:** `detect_captcha`
- **Description:** Detect and analyze CAPTCHAs on current page

#### Input Schema
```python
{
    "detection_strategy": str,  # 'dom', 'visual', 'comprehensive' (default: 'comprehensive')
    "analyze_type": bool  # Analyze CAPTCHA type (default: True)
}
```

#### Output
```python
{
    "captcha_detected": bool,
    "captcha_type": Optional[str],  # 'recaptcha', 'hcaptcha', 'text', etc.
    "confidence_score": float,
    "location": Optional[Dict[str, int]],
    "screenshot_data": Optional[str],
    "solving_required": bool,
    "error": Optional[str]
}
```

---

## Data Models

### BrowserResponse

Standard response format for all browser operations.

```python
@dataclass
class BrowserResponse:
    success: bool
    data: Dict[str, Any]
    error_message: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: Optional[str] = None
    operation_id: Optional[str] = None
```

### ElementSelector

Flexible element selector supporting multiple selection methods.

```python
@dataclass
class ElementSelector:
    value: str
    selector_type: str = "css"  # 'css', 'xpath', 'text', 'id', 'class'
    wait_timeout: int = 10
    multiple: bool = False
```

### SessionStatus

Browser session status enumeration.

```python
class SessionStatus(Enum):
    ACTIVE = "active"
    IDLE = "idle"
    CLOSED = "closed"
    ERROR = "error"
    EXPIRED = "expired"
```

---

## Error Handling

### Exception Hierarchy

```
AgentCoreBrowserError (base)
├── NavigationError
├── ElementNotFoundError
├── TimeoutError
├── SessionError
├── AuthenticationError
├── ServiceUnavailableError
├── CaptchaError
└── ConfigurationError
```

### Error Recovery

The integration includes automatic error recovery mechanisms:

- **Retry Logic**: Exponential backoff for transient failures
- **Session Recovery**: Automatic session recreation on expiration
- **Fallback Strategies**: Alternative approaches for failed operations

#### Retry Configuration

```python
@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
```

---

## Configuration

### Configuration Options

The integration supports YAML configuration files with the following structure:

```yaml
# AWS Configuration
aws:
  region: "us-east-1"
  access_key_id: "${AWS_ACCESS_KEY_ID}"
  secret_access_key: "${AWS_SECRET_ACCESS_KEY}"

# AgentCore Configuration
agentcore:
  browser_tool_endpoint: "https://agentcore.amazonaws.com"
  session_timeout: 300
  max_concurrent_sessions: 5

# Browser Configuration
browser:
  headless: true
  viewport_width: 1920
  viewport_height: 1080
  user_agent: "AgentCore-Browser/1.0"
  enable_javascript: true
  enable_images: true
  page_load_timeout: 30

# LlamaIndex Configuration
llamaindex:
  llm_model: "anthropic.claude-3-sonnet-20240229-v1:0"
  vision_model: "anthropic.claude-3-sonnet-20240229-v1:0"
  temperature: 0.1
  max_tokens: 4096

# Security Configuration
security:
  enable_pii_scrubbing: true
  log_sensitive_data: false
  session_encryption: true

# Monitoring Configuration
monitoring:
  enable_metrics: true
  log_level: "INFO"
  performance_tracking: true
```

### Environment Variables

The following environment variables are supported:

- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `AWS_DEFAULT_REGION`: AWS region
- `AGENTCORE_ENDPOINT`: AgentCore service endpoint
- `LLAMAINDEX_LOG_LEVEL`: Logging level
- `BROWSER_HEADLESS`: Enable headless mode (true/false)
- `SESSION_TIMEOUT`: Session timeout in seconds

---

## Best Practices

### Performance Optimization

1. **Session Reuse**: Reuse browser sessions for multiple operations
2. **Concurrent Operations**: Use async/await for parallel processing
3. **Resource Management**: Properly close sessions to free resources
4. **Caching**: Cache frequently accessed content

### Security Considerations

1. **Credential Management**: Use AWS IAM roles instead of hardcoded credentials
2. **Input Validation**: Sanitize all user inputs
3. **PII Protection**: Enable PII scrubbing for sensitive data
4. **Session Isolation**: Each session runs in isolated VM environment

### Error Handling

1. **Graceful Degradation**: Implement fallback strategies
2. **Logging**: Log errors with sufficient context
3. **Monitoring**: Set up alerts for critical failures
4. **Recovery**: Implement automatic recovery mechanisms

---

## Migration Guide

### From Selenium

```python
# Selenium
from selenium import webdriver
driver = webdriver.Chrome()
driver.get("https://example.com")
element = driver.find_element(By.CSS_SELECTOR, ".content")
text = element.text

# LlamaIndex AgentCore
integration = LlamaIndexAgentCoreIntegration()
agent = integration.create_agent()
response = await agent.achat(
    "Navigate to https://example.com and extract text from .content element"
)
```

### From Playwright

```python
# Playwright
from playwright.async_api import async_playwright
async with async_playwright() as p:
    browser = await p.chromium.launch()
    page = await browser.new_page()
    await page.goto("https://example.com")
    text = await page.text_content(".content")

# LlamaIndex AgentCore
integration = LlamaIndexAgentCoreIntegration()
client = integration.browser_client
await client.create_session()
await client.navigate("https://example.com")
response = await client.extract_text(ElementSelector(".content"))
```

---

## Support and Resources

- **Documentation**: [Full documentation](./README.md)
- **Examples**: [Example applications](./examples/)
- **Troubleshooting**: [Troubleshooting guide](./TROUBLESHOOTING.md)
- **FAQ**: [Frequently asked questions](./FAQ.md)