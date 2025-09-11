# Browser Tool with NovaAct Framework

## Overview

This directory contains tutorials and examples for using Amazon Bedrock AgentCore Browser Tool with the NovaAct framework. NovaAct is designed for rapid prototyping and provides a simple, intuitive API that makes it perfect for beginners getting started with AI-powered browser automation.

## What is NovaAct?

NovaAct is a beginner-friendly agentic framework that excels at:

- **Simple API Design**: Intuitive interfaces that are easy to learn and use
- **Rapid Prototyping**: Quick setup and development cycles for testing ideas
- **Visual Understanding**: Built-in capabilities for processing screenshots and visual elements
- **Straightforward Integration**: Easy connection with AWS services and external APIs
- **Minimal Configuration**: Get started quickly with sensible defaults

## Integration with AgentCore Browser Tool

The combination of NovaAct and AgentCore Browser Tool provides:

- **Beginner-Friendly Automation**: Simple APIs for complex browser interactions
- **Secure Browser Sessions**: Enterprise-grade security with VM-level isolation
- **Visual Processing**: AI-powered understanding of web page content
- **Scalable Architecture**: Automatic scaling without complex configuration
- **Built-in Monitoring**: Easy-to-understand observability and debugging

## Tutorials Available

### 🚀 Getting Started
**Files**: 
- `01_getting_started-agentcore-browser-tool-with-nova-act.ipynb`
- `02_agentcore-browser-tool-live-view-with-nova-act.ipynb`

Learn the fundamentals of browser automation using NovaAct with AgentCore Browser Tool. These tutorials cover:

**Key Features**:
- Basic browser navigation and interaction
- Form filling and data extraction
- Screenshot analysis and visual understanding
- Interactive live view capabilities for debugging

**Prerequisites**: 
- Basic Python knowledge
- AWS account with Bedrock access
- [AgentCore Browser Tool Basics](../README.md)

### 🔐 Handling Sensitive Information
**Location**: `handling-sensitive-information/`

Discover how to safely handle sensitive data in browser automation scenarios using NovaAct's straightforward approach with AgentCore Browser Tool's security features.

**Key Features**:
- Secure form filling with sensitive data protection
- Basic authentication workflows
- Data masking and privacy preservation
- Simple audit trails and logging

**Prerequisites**:
- [AgentCore Identity](../../../03-AgentCore-identity/README.md)
- Completion of getting started tutorials above

## Getting Started

### Prerequisites

Before starting with NovaAct and AgentCore Browser Tool integration, ensure you have:

1. **AWS Account Setup**:
   - Access to Amazon Bedrock with appropriate model permissions
   - AgentCore Browser Tool service access
   - Basic IAM roles and policies configured

2. **Development Environment**:
   - Python 3.9+ with virtual environment capabilities
   - AWS CLI configured with appropriate permissions
   - Jupyter Notebook for interactive tutorials

3. **Framework Knowledge**:
   - Basic Python programming skills
   - Familiarity with Jupyter notebooks
   - No prior agentic framework experience required

### Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure AWS Credentials**:
   ```bash
   aws configure
   # or set environment variables
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   export AWS_DEFAULT_REGION=us-east-1
   ```

3. **Verify Setup**:
   ```bash
   python -c "import novaact; import agentcore_browser_tool; print('Setup complete!')"
   ```

## Architecture Overview

The NovaAct + AgentCore Browser Tool integration follows a simple, straightforward architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    NovaAct Agent Layer                     │
│        (Simple API, Visual Processing, Easy Setup)         │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                AgentCore Browser Tool Layer                 │
│     (Secure Browser Sessions, VM Isolation, Scaling)       │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Bedrock AI Layer                        │
│        (Vision Models, Text Analysis, Simple Decisions)     │
└─────────────────────────────────────────────────────────────┘
```

## Key Benefits

### Beginner-Friendly
- **Simple API**: Easy-to-understand methods and clear documentation
- **Quick Setup**: Minimal configuration required to get started
- **Clear Examples**: Step-by-step tutorials with detailed explanations
- **Gentle Learning Curve**: Progressive complexity from basic to advanced concepts

### Production Ready
- **Enterprise Security**: VM-level isolation and secure browser sessions
- **Automatic Scaling**: Browser sessions scale based on demand without configuration
- **Built-in Monitoring**: Easy-to-understand observability and debugging tools
- **Cost Effective**: Pay-per-use pricing model for browser sessions

### Visual Intelligence
- **Screenshot Analysis**: AI-powered understanding of web page content
- **Element Detection**: Automatic identification of interactive elements
- **Visual Feedback**: Live view capabilities for debugging and development
- **Multi-Modal Processing**: Combining text and visual information for decisions

## Learning Path

### Beginner Path (Recommended)
1. Start with `01_getting_started-agentcore-browser-tool-with-nova-act.ipynb`
2. Complete `02_agentcore-browser-tool-live-view-with-nova-act.ipynb`
3. Explore `handling-sensitive-information/` tutorials
4. Try building your own simple automation scripts

### Intermediate Path
1. Complete all NovaAct tutorials
2. Compare with other frameworks (Browser-use, Strands)
3. Implement custom solutions for specific use cases
4. Explore integration with other AgentCore components

### Advanced Path
1. Master NovaAct patterns and best practices
2. Build production-ready applications
3. Contribute to the NovaAct community
4. Mentor other beginners in the framework

## Tutorial Structure

Each tutorial in this directory follows a beginner-friendly approach:

### 📚 **Concept Introduction** (10-15 min)
- Clear explanation of what you'll learn
- Real-world context and use cases
- Prerequisites and setup verification

### 🔍 **Step-by-Step Implementation** (30-45 min)
- Detailed code examples with explanations
- Common pitfalls and how to avoid them
- Interactive exercises to reinforce learning

### 🧠 **Understanding the Results** (15-20 min)
- Analysis of what happened and why
- Debugging techniques and troubleshooting
- Best practices and optimization tips

### 🚀 **Next Steps** (5-10 min)
- Suggestions for further exploration
- Links to related tutorials and resources
- Ideas for applying concepts to your own projects

## Integration with AgentCore Ecosystem

This framework integrates seamlessly with other AgentCore components:

- **[AgentCore Runtime](../../../01-AgentCore-runtime/README.md)**: Deploy NovaAct agents to production
- **[AgentCore Memory](../../../04-AgentCore-memory/README.md)**: Add persistent memory to your agents
- **[AgentCore Identity](../../../03-AgentCore-identity/README.md)**: Secure authentication and authorization
- **[AgentCore Observability](../../../06-AgentCore-observability/README.md)**: Monitor and debug your agents
- **[AgentCore Gateway](../../../02-AgentCore-gateway/README.md)**: API integration and management

## Common Use Cases

### Web Data Extraction
- Scraping product information from e-commerce sites
- Gathering news articles and content
- Extracting contact information from business directories
- Monitoring competitor websites for changes

### Form Automation
- Filling out registration forms
- Submitting applications and requests
- Updating profile information across multiple sites
- Automating repetitive data entry tasks

### Testing and Monitoring
- Automated testing of web applications
- Monitoring website availability and performance
- Validating form submissions and user flows
- Checking for broken links and errors

## Support and Resources

- **NovaAct Documentation**: [Link to NovaAct framework docs]
- **AgentCore Browser Tool**: [Link to AgentCore Browser Tool documentation]
- **AWS Bedrock**: [Link to Bedrock documentation]
- **Community Support**: [Link to community forums]
- **Beginner's Guide**: [Link to comprehensive beginner resources]

## Troubleshooting

### Common Issues

**Installation Problems**:
- Ensure Python 3.9+ is installed
- Use virtual environments to avoid dependency conflicts
- Check AWS credentials are properly configured

**Browser Session Issues**:
- Verify AgentCore Browser Tool service access
- Check IAM permissions for Bedrock and browser tool services
- Review AWS region configuration

**Tutorial Execution Problems**:
- Ensure all prerequisites are met
- Check Jupyter notebook kernel is using correct Python environment
- Verify all required packages are installed

### Getting Help

1. **Check Prerequisites**: Ensure all setup steps are completed
2. **Review Error Messages**: Most errors include helpful guidance
3. **Consult Documentation**: Check framework and service documentation
4. **Community Support**: Ask questions in community forums
5. **Contact Support**: Reach out to AWS support for service-related issues

## Next Steps

1. **Complete Basic Tutorials**: Start with the getting started notebooks
2. **Explore Specializations**: Try the sensitive information handling tutorials
3. **Compare Frameworks**: Explore other framework directories to understand differences
4. **Build Projects**: Apply learned concepts to your own automation needs
5. **Join Community**: Connect with other developers and share your experiences

---

**Note**: This directory focuses on NovaAct-specific implementation patterns. For framework comparisons and alternative approaches, explore the other browser tool tutorials in the parent directory.