# Frequently Asked Questions (FAQ)

## General Questions

### Q: What is the LlamaIndex AgentCore Browser Integration?
**A:** This integration enables LlamaIndex agents to perform web browsing and automation tasks using Amazon Bedrock AgentCore's secure, VM-isolated browser service. It provides intelligent web interaction capabilities while leveraging AWS's managed infrastructure for security and scalability.

### Q: How does this differ from traditional browser automation tools like Selenium or Playwright?
**A:** Key differences include:
- **Managed Infrastructure**: No need to manage browser instances or drivers
- **VM Isolation**: Each session runs in isolated virtual machines for security
- **AI Integration**: Built-in integration with Bedrock vision models for CAPTCHA solving
- **Scalability**: Automatic scaling handled by AWS AgentCore service
- **Security**: Enterprise-grade security and compliance features

### Q: What are the main use cases for this integration?
**A:** Common use cases include:
- **Web Scraping**: Intelligent content extraction from dynamic websites
- **Form Automation**: Automated form filling and submission
- **CAPTCHA Handling**: AI-powered CAPTCHA detection and solving
- **Website Testing**: Automated testing of web applications
- **Data Collection**: Gathering information from multiple web sources
- **Monitoring**: Website monitoring and change detection

## Technical Questions

### Q: What Python version is required?
**A:** Python 3.12 is required. The integration is specifically designed and tested for Python 3.12 compatibility.

### Q: What AWS permissions are needed?
**A:** Required IAM permissions include:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock-agentcore:CreateBrowserSession",
                "bedrock-agentcore:InvokeBrowserTool",
                "bedrock-agentcore:CloseBrowserSession",
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "*"
        }
    ]
}
```

### Q: Can I use this integration without AWS credentials?
**A:** No, AWS credentials are required to access AgentCore services. However, you can use:
- IAM roles (recommended for EC2/Lambda)
- AWS credentials file
- Environment variables
- AWS SSO

### Q: How many concurrent browser sessions can I run?
**A:** The default limit is 5 concurrent sessions per account, but this can be increased by contacting AWS support. Configure in your settings:
```yaml
agentcore:
  max_concurrent_sessions: 5
```

### Q: What regions support AgentCore browser tool?
**A:** AgentCore browser tool is available in major AWS regions including:
- us-east-1 (N. Virginia)
- us-west-2 (Oregon)
- eu-west-1 (Ireland)
- ap-southeast-1 (Singapore)

Check the latest AWS documentation for current availability.

## Configuration Questions

### Q: How do I configure the integration?
**A:** You can configure using:

1. **YAML file** (recommended):
```yaml
aws:
  region: "us-east-1"
agentcore:
  session_timeout: 300
browser:
  headless: true
  viewport_width: 1920
  viewport_height: 1080
```

2. **Environment variables**:
```bash
export AWS_DEFAULT_REGION=us-east-1
export BROWSER_HEADLESS=true
```

3. **Programmatic configuration**:
```python
integration = LlamaIndexAgentCoreIntegration(
    aws_credentials={"region": "us-east-1"}
)
```

### Q: Can I customize the browser settings?
**A:** Yes, you can customize various browser settings:
```yaml
browser:
  headless: true
  viewport_width: 1920
  viewport_height: 1080
  user_agent: "Custom-Agent/1.0"
  enable_javascript: true
  enable_images: false  # Disable for faster loading
  page_load_timeout: 30
```

### Q: How do I handle different environments (dev/staging/prod)?
**A:** Use environment-specific configuration files:
```bash
# Development
python app.py --config config.dev.yaml

# Production
python app.py --config config.prod.yaml
```

Or environment variables:
```bash
export ENVIRONMENT=production
export AGENTCORE_ENDPOINT=https://prod.agentcore.amazonaws.com
```

## Usage Questions

### Q: How do I create a simple web scraping agent?
**A:** Here's a basic example:
```python
from integration import LlamaIndexAgentCoreIntegration

# Initialize integration
integration = LlamaIndexAgentCoreIntegration()
agent = integration.create_agent()

# Scrape content
response = await agent.achat(
    "Navigate to https://example.com and extract the main article text"
)
print(response.response)
```

### Q: How do I handle forms with the integration?
**A:** Use the FormInteractionTool:
```python
form_tool = FormInteractionTool(browser_client)
result = await form_tool.acall(
    form_data={
        "#username": "user@example.com",
        "#password": "password123",
        "#remember": True
    },
    submit=True,
    submit_selector="#login-button"
)
```

### Q: Can I take screenshots during automation?
**A:** Yes, use the ScreenshotCaptureTool:
```python
screenshot_tool = ScreenshotCaptureTool(browser_client)
result = await screenshot_tool.acall(
    element_selector=".main-content",  # Optional: specific element
    full_page=True
)

# Save screenshot
import base64
with open("screenshot.png", "wb") as f:
    f.write(base64.b64decode(result["image_data"]))
```

### Q: How do I handle CAPTCHAs?
**A:** The integration includes AI-powered CAPTCHA handling:
```python
captcha_tool = CaptchaDetectionTool(browser_client)
detection = await captcha_tool.acall()

if detection["captcha_detected"]:
    # Use vision model to solve
    solver = CaptchaSolvingWorkflow(vision_model)
    solution = await solver.solve_captcha(detection)
```

## Performance Questions

### Q: How can I optimize performance for large-scale scraping?
**A:** Performance optimization strategies:

1. **Disable unnecessary features**:
```yaml
browser:
  enable_images: false
  enable_css: false  # If styling not needed
```

2. **Use concurrent sessions**:
```python
import asyncio

async def scrape_urls(urls):
    tasks = [scrape_single_url(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results
```

3. **Implement session pooling**:
```python
class SessionPool:
    def __init__(self, pool_size=5):
        self.pool = []
        self.pool_size = pool_size
    
    async def get_session(self):
        if self.pool:
            return self.pool.pop()
        return await self.create_new_session()
```

### Q: What's the typical response time for browser operations?
**A:** Typical response times:
- **Navigation**: 2-5 seconds
- **Element interaction**: 0.5-2 seconds
- **Screenshot**: 1-3 seconds
- **Text extraction**: 0.5-1 second

Times vary based on page complexity and network conditions.

### Q: How do I handle memory usage in long-running applications?
**A:** Memory management strategies:

1. **Session cycling**:
```python
# Recreate session every 100 operations
if operation_count % 100 == 0:
    await client.close_session()
    await client.create_session()
```

2. **Resource cleanup**:
```python
try:
    # Perform operations
    pass
finally:
    await client.close_session()
```

## Error Handling Questions

### Q: What should I do if I get authentication errors?
**A:** Common solutions:
1. Verify AWS credentials are correct
2. Check IAM permissions
3. Ensure region is supported
4. Try refreshing credentials:
```python
import boto3
session = boto3.Session()
credentials = session.get_credentials()
credentials.refresh()
```

### Q: How do I handle network timeouts?
**A:** Configure appropriate timeouts:
```python
# Increase navigation timeout
await client.navigate(url, timeout=60)

# Configure global timeouts
config = {
    "browser": {
        "page_load_timeout": 60,
        "element_wait_timeout": 30
    }
}
```

### Q: What if an element is not found?
**A:** Use robust element selection:
```python
# Try multiple selectors
selectors = [
    ElementSelector("#submit-btn"),
    ElementSelector(".submit-button"),
    ElementSelector("Submit", selector_type="text")
]

for selector in selectors:
    try:
        await client.click_element(selector)
        break
    except ElementNotFoundError:
        continue
else:
    raise ElementNotFoundError("Submit button not found with any selector")
```

## Security Questions

### Q: Is my data secure when using AgentCore browser tool?
**A:** Yes, AgentCore provides enterprise-grade security:
- **VM Isolation**: Each session runs in isolated virtual machines
- **Data Encryption**: All data is encrypted in transit and at rest
- **No Data Persistence**: Browser data is not stored after session ends
- **Network Isolation**: Sessions are isolated from each other

### Q: Can I scrub sensitive information automatically?
**A:** Yes, enable PII scrubbing:
```yaml
security:
  enable_pii_scrubbing: true
  log_sensitive_data: false
```

Or programmatically:
```python
from privacy_manager import PrivacyManager

privacy = PrivacyManager()
clean_text = privacy.scrub_pii(extracted_text)
```

### Q: How are credentials managed?
**A:** Best practices for credential management:
1. Use IAM roles when possible
2. Store credentials in AWS Secrets Manager
3. Use environment variables for local development
4. Never hardcode credentials in source code

## Integration Questions

### Q: Can I use this with other LlamaIndex components?
**A:** Yes, the integration works with:
- **LlamaIndex Agents**: ReActAgent, OpenAIAgent
- **Document Processing**: Document loaders and processors
- **Vector Stores**: For storing extracted content
- **Query Engines**: For searching processed content

Example:
```python
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser

# Extract web content
content = await extract_web_content(url)

# Create documents
documents = [WebContentDocument(text=content, source_url=url)]

# Build index
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
```

### Q: Can I extend the integration with custom tools?
**A:** Yes, create custom tools by extending BaseTool:
```python
from llama_index.core.tools import BaseTool

class CustomBrowserTool(BaseTool):
    def __init__(self, browser_client):
        self.client = browser_client
        super().__init__(
            metadata=ToolMetadata(
                name="custom_operation",
                description="Custom browser operation"
            )
        )
    
    async def acall(self, **kwargs):
        # Custom implementation
        pass
```

### Q: How do I migrate from Selenium/Playwright?
**A:** Migration strategies:

1. **Identify current operations**:
   - Navigation → BrowserNavigationTool
   - Element interaction → ElementClickTool, FormInteractionTool
   - Content extraction → TextExtractionTool
   - Screenshots → ScreenshotCaptureTool

2. **Replace direct browser calls with agent instructions**:
```python
# Old Selenium approach
driver.get("https://example.com")
element = driver.find_element(By.CSS_SELECTOR, ".content")
text = element.text

# New LlamaIndex approach
response = await agent.achat(
    "Navigate to https://example.com and extract text from .content element"
)
```

## Troubleshooting Questions

### Q: Why is my agent not using browser tools?
**A:** Common causes:
1. Tools not properly registered with agent
2. Tool descriptions unclear to LLM
3. Agent prompt doesn't encourage tool usage

Solution:
```python
# Verify tool registration
tools = integration.get_browser_tools()
print([tool.metadata.name for tool in tools])

# Create agent with explicit tool usage
agent = ReActAgent.from_tools(
    tools=tools,
    llm=llm,
    verbose=True,
    system_prompt="You are a web browsing assistant. Use the available browser tools to navigate websites and extract information."
)
```

### Q: How do I debug browser operations?
**A:** Enable debug mode:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Take screenshots for debugging
async def debug_operation():
    await client.navigate(url)
    screenshot = await client.take_screenshot()
    
    # Save debug screenshot
    with open("debug.png", "wb") as f:
        f.write(base64.b64decode(screenshot.data["image_data"]))
```

### Q: What if the integration is slow?
**A:** Performance tuning:
1. Disable images and CSS if not needed
2. Use headless mode
3. Reduce viewport size
4. Implement connection pooling
5. Use async operations

```python
# Fast configuration
config = {
    "browser": {
        "headless": True,
        "enable_images": False,
        "viewport_width": 1280,
        "viewport_height": 720
    }
}
```

## Billing and Costs

### Q: How is usage billed?
**A:** Billing is based on:
- **Browser session time**: Charged per minute of active session
- **Bedrock model usage**: Charged per token for LLM and vision model calls
- **Data transfer**: Standard AWS data transfer rates

### Q: How can I optimize costs?
**A:** Cost optimization strategies:
1. **Close sessions promptly**: Don't leave sessions idle
2. **Use efficient prompts**: Reduce token usage
3. **Batch operations**: Process multiple URLs in single session
4. **Monitor usage**: Set up billing alerts

```python
# Efficient session management
async with SessionManager() as session:
    # Perform all operations
    results = await process_multiple_operations(session)
# Session automatically closed
```

## Getting Started

### Q: What's the quickest way to get started?
**A:** Follow these steps:

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure AWS credentials**:
```bash
aws configure
```

3. **Run simple example**:
```python
from integration import LlamaIndexAgentCoreIntegration

integration = LlamaIndexAgentCoreIntegration()
agent = integration.create_agent()

response = await agent.achat("Navigate to https://httpbin.org/html and extract the page title")
print(response.response)
```

### Q: Where can I find more examples?
**A:** Check the examples directory:
- `examples/basic_scraping.py`: Simple web scraping
- `examples/form_automation.py`: Form filling automation
- `examples/captcha_handling.py`: CAPTCHA solving workflow
- `examples/batch_processing.py`: Processing multiple URLs

### Q: How do I get support?
**A:** Support options:
1. **Documentation**: Check API reference and troubleshooting guide
2. **Examples**: Review example applications
3. **Community**: Join discussions and forums
4. **Issues**: Report bugs with detailed reproduction steps

For urgent issues, include:
- Error messages and stack traces
- Configuration (sanitized)
- Python and package versions
- Minimal reproduction example