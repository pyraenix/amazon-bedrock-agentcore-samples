# NovaAct-AgentCore Integration Architecture

## Architecture Diagram

```mermaid
graph TB
    subgraph "Developer Environment"
        DEV[Developer]
        SDK[NovaAct SDK]
        CLIENT[AgentCore Browser Client SDK]
    end
    
    subgraph "AWS Cloud - AgentCore Browser Tool"
        subgraph "AgentCore Managed Infrastructure"
            GATEWAY[AgentCore Gateway]
            RUNTIME[AgentCore Runtime]
            BROWSER[Containerized Browser]
            CDP[Chrome DevTools Protocol]
            MONITOR[Observability Dashboard]
        end
        
        subgraph "Security Layer"
            ISOLATION[Container Isolation]
            ENCRYPT[Data Encryption]
            AUDIT[Audit Logging]
            REDACT[Screenshot Redaction]
        end
    end
    
    subgraph "NovaAct AI Service"
        NOVA_API[NovaAct API]
        AI_MODEL[Natural Language AI Model]
        PROCESSOR[Instruction Processor]
    end
    
    subgraph "Target Web Applications"
        LOGIN[Login Forms]
        PII_FORMS[PII Forms]
        PAYMENT[Payment Forms]
        SENSITIVE[Sensitive Data]
    end
    
    %% Developer interactions
    DEV --> SDK
    DEV --> CLIENT
    
    %% SDK connections
    SDK --> NOVA_API
    CLIENT --> GATEWAY
    
    %% AgentCore internal flow
    GATEWAY --> RUNTIME
    RUNTIME --> BROWSER
    BROWSER --> CDP
    
    %% NovaAct AI processing
    NOVA_API --> AI_MODEL
    AI_MODEL --> PROCESSOR
    PROCESSOR --> CDP
    
    %% Security integration
    ISOLATION --> BROWSER
    ENCRYPT --> BROWSER
    AUDIT --> MONITOR
    REDACT --> MONITOR
    
    %% Browser to target applications
    BROWSER --> LOGIN
    BROWSER --> PII_FORMS
    BROWSER --> PAYMENT
    BROWSER --> SENSITIVE
    
    %% Styling
    classDef developer fill:#e1f5fe
    classDef agentcore fill:#f3e5f5
    classDef novaact fill:#e8f5e8
    classDef security fill:#fff3e0
    classDef target fill:#fce4ec
    
    class DEV,SDK,CLIENT developer
    class GATEWAY,RUNTIME,BROWSER,CDP,MONITOR agentcore
    class NOVA_API,AI_MODEL,PROCESSOR novaact
    class ISOLATION,ENCRYPT,AUDIT,REDACT security
    class LOGIN,PII_FORMS,PAYMENT,SENSITIVE target
```

## Integration Flow Description

1. **Developer Setup**: Developer uses both NovaAct SDK and AgentCore Browser Client SDK
2. **Secure Connection**: AgentCore Browser Client SDK establishes secure connection to managed browser infrastructure
3. **AI Integration**: NovaAct SDK connects to AgentCore's CDP endpoint for browser control
4. **Natural Language Processing**: NovaAct AI processes instructions within AgentCore's secure environment
5. **Containerized Execution**: All browser operations happen within AgentCore's isolated containers
6. **Security Monitoring**: AgentCore's built-in observability tracks all operations securely
7. **Target Interaction**: Secure browser interacts with web applications containing sensitive data

## Key Security Benefits

- **Container Isolation**: AgentCore's containerized browser prevents data leakage
- **Managed Infrastructure**: Fully managed browser environment with built-in security
- **AI Processing Security**: NovaAct's natural language processing within secure boundaries
- **Observability**: Real-time monitoring without exposing sensitive data
- **Automatic Cleanup**: Secure session termination and resource management