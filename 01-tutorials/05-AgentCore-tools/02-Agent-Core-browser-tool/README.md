# Amazon Bedrock AgentCore Browser Tool

## Overview

Amazon Bedrock AgentCore Browser Tool provides AI agents with a secure, fully managed way to interact with websites just like humans do. It allows agents to navigate web pages, fill out forms, and complete complex tasks without requiring developers to write and maintain custom automation scripts.

## How it works

A browser tool sandbox is a secure execution environment that enables AI agents to safely interact with web browsers. When a user makes a request, the Large Language Model (LLM) selects appropriate tools and translates commands. These commands are executed within a controlled sandbox environment containing a headless browser and hosted library server (using tools like Playwright). The sandbox provides isolation and security by containing web interactions within a restricted space, preventing unauthorized system access. The agent receives feedback through screenshots and can perform automated tasks while maintaining system security. This setup enables safe web automation for AI agents

![architecture local](../02-Agent-Core-browser-tool/images/browser-tool.png)

## Key Features

### Secure, Managed Web Interaction

Provides AI agents with a secure, fully managed way to interact with websites just like humans do, allowing navigation, form filling, and complex task completion without requiring custom automation scripts.

### Enterprise Security Features

Provides VM-level isolation with 1:1 mapping between user session and browser session, offering enterprise-grade security. Each browser session runs in an isolated sandbox environment to meet enterprise security needs

### Model-Agnostic Integration

Supports various AI models and frameworks while providing natural language abstractions for browser actions through tools like interact(), parse(), and discover(), making it particularly suitable for enterprise environments. The tool can execute browser commands from any library and supports various automation frameworks like Playwright and Puppeteer.

### Integration

Amazon Bedrock AgentCore Browser Tool integrates with other Amazon Bedrock AgentCore capabilities through a unified SDK, including:

- Amazon Bedrock AgentCore Runtime
- Amazon Bedrock AgentCore Identity
- Amazon Bedrock AgentCore Memory
- Amazon Bedrock AgentCore Observability

This integration aims to simplify the development process and provide a comprehensive platform for building, deploying, and managing AI agents, with powerful capabilities to perform browser based tasks.

### Use Cases

The Amazon Bedrock AgentCore Browser Tool is suitable for a wide range of applications, including:

- Web Navigation & Interaction
- Workflow Automation including filling forms

## Tutorials Overview

Our tutorials are organized by **agentic framework** to help you find all related content for your chosen framework in one place. Each framework directory contains comprehensive tutorials covering both basic browser automation and specialized applications like CAPTCHA handling and sensitive information management.

## Framework-Based Organization

### 🤖 [NovaAct Framework](01-browser-with-NovaAct/)
**Best for**: Beginners and rapid prototyping

- **Getting Started**: [Basic browser automation with NovaAct](01-browser-with-NovaAct/01_getting_started-agentcore-browser-tool-with-nova-act.ipynb)
- **Live View**: [Interactive browser sessions](01-browser-with-NovaAct/02_agentcore-browser-tool-live-view-with-nova-act.ipynb)
- **Sensitive Information**: [Secure data handling patterns](01-browser-with-NovaAct/handling-sensitive-information/)

### 🌐 [Browser-use Framework](02-browser-with-browserUse/)
**Best for**: AI-powered browser automation

- **Getting Started**: [Browser automation with Browser-use](02-browser-with-browserUse/getting_started-agentcore-browser-tool-with-browser-use.ipynb)
- **Live View**: [Interactive browser sessions](02-browser-with-browserUse/agentcore-browser-tool-live-view-with-browser-use.ipynb)
- **CAPTCHA Handling**: [AI-powered CAPTCHA detection and solving](02-browser-with-browserUse/captcha-handling/)
- **Sensitive Information**: [Secure data handling patterns](02-browser-with-browserUse/handling-sensitive-information/)

### 🔗 [Strands Framework](03-browser-with-Strands/)
**Best for**: Enterprise orchestration and complex workflows

- **CAPTCHA Handling**: [Enterprise-grade CAPTCHA handling with intelligent orchestration](03-browser-with-Strands/captcha-handling/) - **FEATURED!**
- **Sensitive Information**: [Enterprise security and compliance patterns](03-browser-with-Strands/handling-sensitive-information/)

### 📚 [LlamaIndex Framework](04-browser-with-LlamaIndex/)
**Best for**: RAG-enhanced browser automation and knowledge management

- **CAPTCHA Handling**: [RAG-powered CAPTCHA analysis and solving](04-browser-with-LlamaIndex/captcha-handling/)
- **Sensitive Information**: [Privacy-preserving data processing](04-browser-with-LlamaIndex/handling-sensitive-information/)

## Learning Paths

### 🚀 **Beginner Path** (Start Here)
1. Choose your preferred framework (NovaAct recommended for beginners)
2. Complete the "Getting Started" tutorial for your chosen framework
3. Explore "Live View" capabilities to understand interactive browser sessions
4. Try basic examples in specialized applications (CAPTCHA or sensitive information)

### 🔍 **Intermediate Path**
1. Complete tutorials for 2-3 different frameworks to understand their strengths
2. Explore advanced features in specialized applications
3. Implement custom solutions combining multiple frameworks
4. Focus on production-ready patterns and error handling

### 🤖 **Advanced Path**
1. Master all framework integrations and their unique capabilities
2. Build complex multi-step workflows with enterprise security
3. Implement custom tools and extensions for specific use cases
4. Deploy production systems with comprehensive monitoring

## Framework Comparison

| Framework | Best For | Complexity | Key Strengths |
|-----------|----------|------------|---------------|
| **NovaAct** | Beginners, Rapid Prototyping | Low | Simple API, Quick setup |
| **Browser-use** | AI-Powered Automation | Medium | Intelligent automation, Visual understanding |
| **Strands** | Enterprise Workflows | High | Orchestration, Error handling, Scalability |
| **LlamaIndex** | Knowledge Management | High | RAG capabilities, Multi-modal processing |

## Specialized Applications

### 🛡️ **CAPTCHA Handling**
Available in: Browser-use, Strands, LlamaIndex
- Multi-strategy CAPTCHA detection and solving
- AI-powered visual analysis using Bedrock models
- Enterprise-grade security and compliance
- Adaptive learning from solving experiences

### 🔐 **Sensitive Information Handling**
Available in: NovaAct, Browser-use, Strands, LlamaIndex
- Secure form filling with data protection
- Enterprise authentication workflows
- Privacy-preserving data processing
- Audit trails and compliance logging

## Prerequisites

Before starting any tutorial, ensure you have:

1. **AWS Account Setup**:
   - Access to Amazon Bedrock with appropriate model permissions
   - AgentCore Browser Tool service access
   - Proper IAM roles and policies configured

2. **Development Environment**:
   - Python 3.9+ with virtual environment capabilities
   - AWS CLI configured with appropriate permissions
   - Jupyter Notebook for interactive tutorials

3. **Framework Knowledge** (varies by chosen framework):
   - Basic understanding of your chosen agentic framework
   - Familiarity with AWS Bedrock models
   - Experience with Python and async programming

## Getting Started

1. **Choose Your Framework**: Review the framework comparison table above
2. **Set Up Prerequisites**: Ensure your AWS account and development environment are ready
3. **Start Learning**: Navigate to your chosen framework directory and begin with the README
4. **Progress Systematically**: Follow the recommended learning path for your experience level
5. **Explore Specializations**: Once comfortable with basics, explore CAPTCHA handling or sensitive information tutorials
