# LlamaIndex-AgentCore Browser Tool Integration

This package provides seamless integration between LlamaIndex framework and Amazon Bedrock AgentCore's built-in browser tool, enabling developers to build intelligent agents that can leverage AgentCore's enterprise-grade browser automation capabilities.

## Overview

The integration bridges LlamaIndex's agent intelligence with AgentCore's managed browser infrastructure:

- **LlamaIndex**: Provides intelligent orchestration, reasoning, and decision-making capabilities
- **AgentCore Browser Tool**: Provides secure, VM-isolated browser automation in managed cloud infrastructure
- **Integration**: Creates LlamaIndex tools that call AgentCore Browser Tool APIs for web automation tasks

## Features

- 🔒 **Enterprise Security**: VM-isolated browser sessions with enterprise-grade security controls
- 🤖 **AI-Powered**: Multi-modal AI capabilities for CAPTCHA solving and intelligent web interaction
- 📊 **Scalable**: Leverages AWS managed infrastructure for concurrent browser operations
- 🛠️ **Developer Friendly**: Simple LlamaIndex tool interface for complex browser automation
- 📈 **Observable**: Built-in logging, metrics, and debugging capabilities
- 🔧 **Configurable**: Flexible configuration for different environments and use cases

## Installation

### Development Setup (Python 3.12)

**1. Create Python 3.12 Virtual Environment:**
```bash
# Create virtual environment
python3.12 -m venv venv312

# Activate environment
source venv312/bin/activate  # On macOS/Linux
# or
venv312\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

**2. Or use the provided setup script:**
```bash
# Run environment setup (creates venv312 and installs dependencies)
python3.12 setup_env.py

# Activate environment using convenience script
source activate_env.sh
```

**3. Verify Installation:**
```bash
# Run comprehensive environment test
python test_environment.py

# Run Task 1 verification
python verify_task1.py
```

### Production Installation

```bash
pip install llamaindex-agentcore-browser-integration
```

### Requirements

- Python 3.12+
- AWS credentials configured
- Access to Amazon Bedrock AgentCore services

## Quick Start

```python
import asyncio
from llamaindex_agentcore_integration import LlamaIndexAgentCoreIntegration

async def main():
    # Initialize the integration
    integration = LlamaIndexAgentCoreIntegration(
        aws_credentials={
            "region": "us-east-1",
            "aws_access_key_id": "your-access-key",
            "aws_secret_access_key": "your-secret-key"
        }
    )
    
    # Process web content with AI-powered browser automation
    result = await integration.process_web_content("https://example.com")
    print(f"Processing result: {result}")
    
    # Clean up
    await integration.close()

# Run the example
asyncio.run(main())
```

## Configuration

### Environment Variables

```bash
# AWS Configuration
export AWS_DEFAULT_REGION=us-east-1
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key

# AgentCore Configuration
export AGENTCORE_BASE_URL=https://your-agentcore-endpoint
export AGENTCORE_BROWSER_TOOL_ENDPOINT=https://your-browser-tool-endpoint

# Browser Configuration
export BROWSER_HEADLESS=true
export BROWSER_VIEWPORT_WIDTH=1920
export BROWSER_VIEWPORT_HEIGHT=1080
export BROWSER_TIMEOUT=30

# LlamaIndex Models
export LLAMAINDEX_LLM_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
export LLAMAINDEX_VISION_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
```

### Configuration File

Create a `config.yaml` file:

```yaml
aws_credentials:
  region: us-east-1
  profile: default

agentcore_endpoints:
  base_url: https://your-agentcore-endpoint
  browser_tool_endpoint: https://your-browser-tool-endpoint

browser_config:
  headless: true
  viewport_width: 1920
  viewport_height: 1080
  timeout_seconds: 30
  enable_javascript: true
  enable_images: true

llm_model: anthropic.claude-3-sonnet-20240229-v1:0
vision_model: anthropic.claude-3-sonnet-20240229-v1:0
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   LlamaIndex    │    │   Integration    │    │   AgentCore         │
│   Agent         │◄──►│   Layer          │◄──►│   Browser Tool      │
│                 │    │                  │    │                     │
│ • Reasoning     │    │ • Tool Wrappers  │    │ • VM Isolation      │
│ • Planning      │    │ • Auth Handler   │    │ • Session Mgmt      │
│ • Orchestration │    │ • Response Parse │    │ • Security Controls │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
```

## Core Components

### Browser Client
- Handles communication with AgentCore browser tool service
- Manages authentication and session lifecycle
- Provides error handling and retry mechanisms

### LlamaIndex Tools
- `BrowserNavigationTool`: Navigate to URLs and handle page loading
- `CaptchaDetectionTool`: Detect and analyze CAPTCHAs using AI
- `ScreenshotCaptureTool`: Capture page or element screenshots
- `TextExtractionTool`: Extract clean text from web pages
- `FormInteractionTool`: Fill forms and interact with web elements
- `ElementClickTool`: Click buttons, links, and interactive elements

### Configuration Management
- Flexible configuration from files, environment variables, or code
- AWS credential management and validation
- Browser session configuration and optimization

## Development Status

This integration is currently under active development. The following features are planned:

- ✅ **Task 1**: Project structure and core interfaces (Complete)
- 🚧 **Task 2**: AgentCore browser client implementation (In Progress)
- 📋 **Task 3**: LlamaIndex tool implementations (Planned)
- 📋 **Task 4**: Multi-modal AI CAPTCHA solving (Planned)
- 📋 **Task 5**: Agent integration and workflows (Planned)
- 📋 **Task 6**: Document processing integration (Planned)
- 📋 **Task 7**: Security and compliance features (Planned)
- 📋 **Task 8**: Comprehensive testing suite (Planned)
- 📋 **Task 9**: Monitoring and observability (Planned)
- 📋 **Task 10**: Documentation and examples (Planned)
- 📋 **Task 11**: Package and deployment (Planned)

## Contributing

This project is part of the AWS AgentCore samples repository. Please refer to the main repository for contribution guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Check the [AgentCore documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html)
- Open an issue in the [GitHub repository](https://github.com/aws-samples/agentcore-samples/issues)
- Contact the AgentCore team

## Disclaimer

This is sample code for educational and demonstration purposes. For production use, please review and adapt the code according to your specific requirements and security policies.