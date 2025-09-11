# NovaAct-AgentCore Security Flow Diagram

## Data Protection Flow

```mermaid
sequenceDiagram
    participant DEV as Developer
    participant NOVA as NovaAct SDK
    participant CLIENT as AgentCore Client SDK
    participant GATEWAY as AgentCore Gateway
    participant CONTAINER as Secure Container
    participant BROWSER as Managed Browser
    participant AI as NovaAct AI Model
    participant TARGET as Target Website
    participant MONITOR as Security Monitor

    Note over DEV,MONITOR: Secure Session Initialization
    DEV->>CLIENT: Initialize browser_session()
    CLIENT->>GATEWAY: Authenticate with AWS credentials
    GATEWAY->>CONTAINER: Create isolated container
    CONTAINER->>BROWSER: Launch secure browser instance
    BROWSER->>CLIENT: Return CDP endpoint + headers
    
    Note over DEV,MONITOR: NovaAct Integration Setup
    DEV->>NOVA: Initialize NovaAct with CDP endpoint
    NOVA->>AI: Connect to AI model with API key
    AI->>BROWSER: Establish secure CDP connection
    MONITOR->>CONTAINER: Begin security monitoring
    
    Note over DEV,MONITOR: Sensitive Data Processing
    DEV->>NOVA: Send natural language instruction
    NOVA->>AI: Process instruction securely
    AI->>BROWSER: Execute browser actions via CDP
    BROWSER->>TARGET: Navigate to sensitive form
    TARGET->>BROWSER: Load form with PII fields
    
    Note over DEV,MONITOR: Data Protection in Action
    BROWSER->>MONITOR: Screenshot with redaction
    AI->>BROWSER: Fill sensitive fields
    BROWSER->>CONTAINER: Data stays within isolation
    CONTAINER->>MONITOR: Log actions without exposing data
    
    Note over DEV,MONITOR: Secure Completion
    BROWSER->>TARGET: Submit form securely
    TARGET->>BROWSER: Confirmation response
    BROWSER->>AI: Report success
    AI->>NOVA: Return result
    NOVA->>DEV: Success confirmation
    
    Note over DEV,MONITOR: Automatic Cleanup
    CLIENT->>CONTAINER: Terminate session
    CONTAINER->>BROWSER: Secure browser shutdown
    BROWSER->>MONITOR: Final security audit
    MONITOR->>GATEWAY: Session cleanup complete
```

## Security Layers Visualization

```mermaid
graph LR
    subgraph "Security Perimeter"
        subgraph "AgentCore Container Isolation"
            subgraph "Managed Browser Environment"
                subgraph "NovaAct AI Processing"
                    SENSITIVE[Sensitive Data Processing]
                end
                BROWSER_SEC[Browser Security Controls]
            end
            CONTAINER_SEC[Container Security]
        end
        AGENTCORE_SEC[AgentCore Security Layer]
    end
    
    subgraph "External Threats"
        MALWARE[Malware]
        DATA_LEAK[Data Leakage]
        UNAUTHORIZED[Unauthorized Access]
    end
    
    %% Security barriers
    AGENTCORE_SEC -.->|Blocks| MALWARE
    CONTAINER_SEC -.->|Prevents| DATA_LEAK
    BROWSER_SEC -.->|Stops| UNAUTHORIZED
    
    %% Data flow protection
    SENSITIVE --> BROWSER_SEC
    BROWSER_SEC --> CONTAINER_SEC
    CONTAINER_SEC --> AGENTCORE_SEC
    
    classDef security fill:#ffebee
    classDef threat fill:#f3e5f5
    classDef data fill:#e8f5e8
    
    class AGENTCORE_SEC,CONTAINER_SEC,BROWSER_SEC security
    class MALWARE,DATA_LEAK,UNAUTHORIZED threat
    class SENSITIVE data
```

## Security Control Points

### 1. Authentication Layer
- **NovaAct API Key**: Secure storage and transmission
- **AgentCore Credentials**: AWS IAM-based authentication
- **Session Tokens**: Temporary, scoped access tokens

### 2. Network Security
- **Encrypted Connections**: TLS encryption for all communications
- **VPC Isolation**: AgentCore runs in isolated VPC
- **CDP Security**: Secure Chrome DevTools Protocol connections

### 3. Container Isolation
- **Process Isolation**: Each session runs in separate container
- **Resource Limits**: Controlled CPU, memory, and network access
- **Filesystem Isolation**: Temporary, encrypted storage

### 4. Data Protection
- **Screenshot Redaction**: Automatic sensitive data masking
- **Memory Protection**: Secure memory handling and cleanup
- **Audit Logging**: Comprehensive logging without data exposure

### 5. AI Processing Security
- **Prompt Isolation**: NovaAct AI processing within secure boundaries
- **Model Security**: AI model runs with limited system access
- **Response Filtering**: Secure handling of AI-generated actions

## Threat Mitigation

| Threat | Mitigation | Implementation |
|--------|------------|----------------|
| Data Exfiltration | Container isolation + Network controls | AgentCore managed infrastructure |
| Credential Exposure | Secure storage + Temporary tokens | AWS Secrets Manager integration |
| Session Hijacking | Encrypted connections + Session validation | TLS + JWT tokens |
| Malware Injection | Isolated browser environment | Containerized browser instances |
| Unauthorized Access | Multi-layer authentication | NovaAct API + AgentCore IAM |
| Data Persistence | Automatic cleanup + Encrypted storage | Ephemeral containers |