# Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### Issue: ImportError for LlamaIndex components
```
ImportError: No module named 'llama_index.core'
```

**Solution:**
```bash
pip install llama-index-core
pip install llama-index-llms-bedrock
pip install llama-index-multi-modal-llms-bedrock
```

#### Issue: AWS SDK compatibility errors
```
botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

**Solution:**
1. Configure AWS credentials:
```bash
aws configure
```
2. Or set environment variables:
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```
3. Or use IAM roles for EC2/Lambda

### Configuration Issues

#### Issue: Configuration file not found
```
ConfigurationError: Configuration file not found: config.yaml
```

**Solution:**
1. Create configuration file from template:
```bash
cp config.example.yaml config.yaml
```
2. Or specify full path:
```python
integration = LlamaIndexAgentCoreIntegration(config_path="/full/path/to/config.yaml")
```

#### Issue: Invalid AWS region
```
ClientError: The security token included in the request is invalid
```

**Solution:**
1. Verify region in configuration:
```yaml
aws:
  region: "us-east-1"  # Use correct region
```
2. Check AgentCore service availability in your region

### Authentication Issues

#### Issue: AgentCore authentication failure
```
AuthenticationError: Failed to authenticate with AgentCore service
```

**Solution:**
1. Verify AWS credentials have required permissions:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock-agentcore:*"
            ],
            "Resource": "*"
        }
    ]
}
```

2. Check IAM role trust relationships for cross-account access

#### Issue: Session token expired
```
SessionError: Browser session has expired
```

**Solution:**
1. Implement session refresh:
```python
try:
    response = await client.navigate(url)
except SessionError:
    await client.create_session()
    response = await client.navigate(url)
```

2. Increase session timeout in configuration:
```yaml
agentcore:
  session_timeout: 600  # 10 minutes
```

### Browser Operation Issues

#### Issue: Navigation timeout
```
TimeoutError: Page navigation timed out after 30 seconds
```

**Solution:**
1. Increase timeout:
```python
await client.navigate(url, timeout=60)
```

2. Check network connectivity and target site availability

3. Use wait strategies:
```python
# Wait for specific element
await client.wait_for_element(ElementSelector(".content"))
```

#### Issue: Element not found
```
ElementNotFoundError: Element with selector '.button' not found
```

**Solution:**
1. Verify selector syntax:
```python
# CSS selector
selector = ElementSelector(".button", selector_type="css")

# XPath
selector = ElementSelector("//button[@class='submit']", selector_type="xpath")

# Text content
selector = ElementSelector("Submit", selector_type="text")
```

2. Wait for element to appear:
```python
selector = ElementSelector(".button", wait_timeout=10)
```

3. Check if element is in iframe:
```python
# Switch to iframe first
await client.switch_to_frame("iframe_name")
await client.click_element(selector)
```

#### Issue: CAPTCHA detection false positives
```
CaptchaError: CAPTCHA detected but none present
```

**Solution:**
1. Adjust detection sensitivity:
```python
tool = CaptchaDetectionTool(client)
result = await tool.acall(detection_strategy="dom")  # Less sensitive
```

2. Use visual confirmation:
```python
screenshot = await client.take_screenshot()
# Manual verification of screenshot
```

### Performance Issues

#### Issue: Slow page loading
```
TimeoutError: Page load exceeded timeout
```

**Solution:**
1. Optimize browser settings:
```yaml
browser:
  enable_images: false  # Disable images for faster loading
  enable_javascript: true  # Keep JS for functionality
  page_load_timeout: 60
```

2. Use selective content loading:
```python
# Navigate without waiting for full load
await client.navigate(url, wait_for_load=False)
# Wait for specific content
await client.wait_for_element(ElementSelector(".main-content"))
```

#### Issue: Memory usage growing over time
```
MemoryError: Browser session consuming excessive memory
```

**Solution:**
1. Implement session cycling:
```python
class SessionManager:
    def __init__(self, max_operations=100):
        self.operation_count = 0
        self.max_operations = max_operations
    
    async def perform_operation(self, operation):
        if self.operation_count >= self.max_operations:
            await self.client.close_session()
            await self.client.create_session()
            self.operation_count = 0
        
        result = await operation()
        self.operation_count += 1
        return result
```

2. Clear browser data periodically:
```python
await client.clear_cookies()
await client.clear_local_storage()
```

### Integration Issues

#### Issue: LlamaIndex agent not using browser tools
```
Agent response: I cannot browse the web or access external URLs
```

**Solution:**
1. Verify tool registration:
```python
integration = LlamaIndexAgentCoreIntegration()
tools = integration.get_browser_tools()
print(f"Available tools: {[tool.metadata.name for tool in tools]}")

agent = ReActAgent.from_tools(tools=tools, llm=llm)
```

2. Check tool descriptions:
```python
for tool in tools:
    print(f"Tool: {tool.metadata.name}")
    print(f"Description: {tool.metadata.description}")
```

#### Issue: Vision model not processing screenshots
```
Error: Vision model failed to analyze screenshot
```

**Solution:**
1. Verify vision model configuration:
```yaml
llamaindex:
  vision_model: "anthropic.claude-3-sonnet-20240229-v1:0"
```

2. Check image format:
```python
screenshot = await client.take_screenshot(format="png")  # Use PNG for better compatibility
```

3. Validate image data:
```python
import base64
try:
    image_data = base64.b64decode(screenshot.data["image_data"])
    print(f"Image size: {len(image_data)} bytes")
except Exception as e:
    print(f"Invalid image data: {e}")
```

### Network Issues

#### Issue: Connection refused to AgentCore service
```
ConnectionError: Failed to connect to AgentCore endpoint
```

**Solution:**
1. Check service endpoint:
```yaml
agentcore:
  browser_tool_endpoint: "https://agentcore.amazonaws.com"
```

2. Verify network connectivity:
```bash
curl -I https://agentcore.amazonaws.com/health
```

3. Check firewall/proxy settings:
```python
# Configure proxy if needed
import os
os.environ['HTTPS_PROXY'] = 'http://proxy.company.com:8080'
```

#### Issue: SSL certificate verification failed
```
SSLError: certificate verify failed
```

**Solution:**
1. Update certificates:
```bash
pip install --upgrade certifi
```

2. Configure SSL context (not recommended for production):
```python
import ssl
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE
```

### Debugging Tips

#### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('llamaindex_agentcore')
logger.setLevel(logging.DEBUG)
```

#### Capture Screenshots for Debugging
```python
async def debug_screenshot(client, operation_name):
    screenshot = await client.take_screenshot()
    with open(f"debug_{operation_name}_{datetime.now().isoformat()}.png", "wb") as f:
        f.write(base64.b64decode(screenshot.data["image_data"]))
```

#### Monitor Session State
```python
async def check_session_health(client):
    try:
        response = await client.get_current_url()
        print(f"Session active, current URL: {response.data.get('url')}")
        return True
    except SessionError:
        print("Session expired or invalid")
        return False
```

#### Validate Configuration
```python
def validate_setup():
    try:
        config_manager = ConfigurationManager()
        config = config_manager.get_config()
        print("✓ Configuration loaded successfully")
        
        # Test AWS credentials
        import boto3
        session = boto3.Session(
            aws_access_key_id=config.aws.access_key_id,
            aws_secret_access_key=config.aws.secret_access_key,
            region_name=config.aws.region
        )
        sts = session.client('sts')
        identity = sts.get_caller_identity()
        print(f"✓ AWS credentials valid: {identity['Arn']}")
        
        return True
    except Exception as e:
        print(f"✗ Setup validation failed: {e}")
        return False
```

## Performance Optimization

### Browser Settings Optimization
```yaml
# Fast browsing configuration
browser:
  headless: true
  enable_images: false
  enable_css: true
  enable_javascript: true
  page_load_timeout: 15
  viewport_width: 1280
  viewport_height: 720
```

### Concurrent Operations
```python
import asyncio

async def process_multiple_urls(urls):
    tasks = []
    for url in urls:
        task = process_single_url(url)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

### Resource Management
```python
class ResourceManager:
    def __init__(self):
        self.active_sessions = {}
    
    async def get_session(self, session_id=None):
        if session_id and session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        client = AgentCoreBrowserClient()
        session_id = await client.create_session()
        self.active_sessions[session_id] = client
        return client
    
    async def cleanup_all(self):
        for client in self.active_sessions.values():
            try:
                await client.close_session()
            except Exception as e:
                print(f"Error closing session: {e}")
        self.active_sessions.clear()
```

## Getting Help

### Log Collection
When reporting issues, include:

1. **Configuration** (sanitized):
```bash
# Remove sensitive data before sharing
cat config.yaml | sed 's/access_key_id:.*/access_key_id: [REDACTED]/'
```

2. **Error logs**:
```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)
```

3. **Environment information**:
```python
import sys
import platform
print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.architecture()}")

# Package versions
import pkg_resources
packages = ['llama-index-core', 'boto3', 'aiohttp']
for package in packages:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f"{package}: {version}")
    except pkg_resources.DistributionNotFound:
        print(f"{package}: Not installed")
```

### Support Channels
- **Documentation**: Check the full API reference
- **Examples**: Review example applications
- **Issues**: Report bugs with detailed reproduction steps
- **Community**: Join discussions for best practices

### Emergency Recovery
If the integration becomes completely unresponsive:

1. **Force cleanup**:
```python
import asyncio
from client import AgentCoreBrowserClient

async def emergency_cleanup():
    # This will attempt to close all sessions
    client = AgentCoreBrowserClient()
    try:
        await client.cleanup_all_sessions()
    except Exception as e:
        print(f"Cleanup error: {e}")

asyncio.run(emergency_cleanup())
```

2. **Reset configuration**:
```bash
cp config.example.yaml config.yaml
# Edit with your settings
```

3. **Restart with minimal configuration**:
```python
# Minimal working example
integration = LlamaIndexAgentCoreIntegration(
    aws_credentials={
        "aws_access_key_id": "your_key",
        "aws_secret_access_key": "your_secret",
        "region": "us-east-1"
    }
)
```