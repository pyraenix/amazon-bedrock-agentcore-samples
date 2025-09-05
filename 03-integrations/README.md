# AWS Bedrock AgentCore Integrations

This directory contains integrations between AWS Bedrock AgentCore components and various frameworks, platforms, and services.

## Directory Structure

```
03-integrations/
├── agentic-frameworks/              # Integrations with agentic frameworks
│   ├── strands-agents/             # Strands Agents integrations
│   ├── langchain/                  # LangChain integrations
│   ├── crewai/                     # CrewAI integrations
│   └── ...                         # Other agentic frameworks
├── bedrock-agentcore-browser-tool/ # Browser Tool specific integrations
│   └── agentic-frameworks/         # Browser Tool + Agentic Framework integrations
│       └── strands-agents/         # Strands + Browser Tool integration
├── bedrock-agent/                  # Bedrock Agent integrations
└── README.md                       # This file
```

## Integration Categories

### Bedrock AgentCore Browser Tool Integrations
**Location**: `bedrock-agentcore-browser-tool/`

Integrations specific to the Bedrock AgentCore Browser Tool component:
- **[Strands Agents](./bedrock-agentcore-browser-tool/agentic-frameworks/strands-agents/)** - Complete browser automation with AI agents

### General Agentic Framework Integrations  
**Location**: `agentic-frameworks/`

General integrations with agentic frameworks (not component-specific):
- **Strands Agents** - General Bedrock AgentCore integrations
- **LangChain** - LLM application integrations
- **CrewAI** - Multi-agent system integrations
- **AutoGen** - Conversational AI integrations

### Bedrock Agent Integrations
**Location**: `bedrock-agent/`

Integrations with AWS Bedrock Agents service.

## Integration Status

| Component | Framework | Status | Description |
|-----------|-----------|--------|-------------|
| Browser Tool | Strands Agents | ✅ Complete | AWS-hosted browser automation with AI agents |
| Browser Tool | LangChain | 🚧 Planned | Browser automation for LLM applications |
| Browser Tool | CrewAI | 🚧 Planned | Multi-agent web research workflows |
| General | Strands Agents | ✅ Available | General AgentCore tool integrations |

## Getting Started

1. **Choose Your Integration**: Navigate to the appropriate directory based on:
   - The Bedrock AgentCore component you want to use
   - The framework or platform you want to integrate with

2. **Follow Setup Instructions**: Each integration includes:
   - Detailed setup instructions
   - Requirements and prerequisites
   - Configuration examples
   - Test suites for verification

3. **Explore Examples**: Most integrations include:
   - Basic usage examples
   - Advanced use cases
   - Production deployment guidance

## Common Prerequisites

Most integrations require:
- **AWS Account**: With appropriate Bedrock service access
- **Python 3.10+**: For modern framework compatibility
- **AWS Credentials**: Properly configured for your environment
- **Internet Access**: For package installation and testing

## Contributing

To contribute a new integration:

1. **Choose the Right Location**: 
   - Component-specific: `{component-name}/`
   - Framework-specific: `agentic-frameworks/{framework-name}/`

2. **Follow the Structure**:
   - Include comprehensive documentation
   - Provide working examples and tests
   - Follow existing patterns and conventions

3. **Update Documentation**:
   - Add your integration to this README
   - Include status and description
   - Update relevant parent directory READMEs

## Support

For integration-specific support, refer to the documentation in each integration directory. For general questions about Bedrock AgentCore, see the main repository documentation.