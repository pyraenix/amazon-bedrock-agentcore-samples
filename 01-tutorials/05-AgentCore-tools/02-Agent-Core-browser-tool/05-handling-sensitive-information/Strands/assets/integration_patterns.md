# Strands Integration Patterns with AgentCore Browser Tool

## Overview

This document provides comprehensive integration patterns for combining Strands agents with Amazon Bedrock AgentCore Browser Tool. These patterns demonstrate best practices for secure, scalable, and maintainable integrations that handle sensitive information in production environments.

## Core Integration Patterns

### 1. Single-Agent Secure Browser Pattern

The simplest integration pattern for basic secure browser automation:

```python
from strands_agents import Agent
from tools.strands_agentcore_session_helpers import StrandsAgentCoreClient
from tools.strands_security_policies import SecurityPolicy

class SecureBrowserAgent:
    """Single-agent pattern for secure browser automation."""
    
    def __init__(self, region: str, security_level: str = 'high'):
        self.client = StrandsAgentCoreClient(region=region)
        self.security_policy = SecurityPolicy.load(security_level)
        self.agent = self._create_secure_agent()
        
    def _create_secure_agent(self) -> Agent:
        """Create agent with security tools."""
        agent_config = {
            'name': 'secure_browser_agent',
            'tools': [
                'secure_browser_tool',
                'pii_detection_tool',
                'credential_injection_tool'
            ],
            'security_policy': self.security_policy,
            'audit_logging': True
        }
        
        return self.client.create_secure_agent(agent_config)
        
    def execute_secure_workflow(self, workflow_definition: dict) -> dict:
        """Execute workflow with security controls."""
        # Pre-execution security validation
        self._validate_workflow_security(workflow_definition)
        
        # Execute with monitoring
        with self.client.create_secure_session() as session:
            result = self.agent.execute_workflow(
                workflow_definition,
                session=session,
                security_context=self.security_policy.get_context()
            )
            
        # Post-execution cleanup and audit
        self._audit_workflow_execution(workflow_definition, result)
        
        return result

# Usage Example
agent = SecureBrowserAgent(region='us-east-1', security_level='maximum')

workflow = {
    'steps': [
        {'action': 'navigate', 'url': 'https://secure-portal.example.com'},
        {'action': 'inject_credentials', 'source': 'aws_secrets_manager'},
        {'action': 'extract_data', 'selector': '.sensitive-data', 'mask_pii': True}
    ]
}

result = agent.execute_secure_workflow(workflow)
```

### 2. Multi-Agent Coordination Pattern

For complex workflows requiring multiple specialized agents:

```python
from tools.strands_agentcore_session_helpers import SessionPool, MultiAgentOrchestrator
from strands_agents import Agent

class MultiAgentSecureWorkflow:
    """Multi-agent coordination pattern for complex workflows."""
    
    def __init__(self, max_concurrent_agents: int = 5):
        self.session_pool = SessionPool(
            max_sessions=max_concurrent_agents,
            region='us-east-1'
        )
        self.orchestrator = MultiAgentOrchestrator(self.session_pool)
        self.agents = self._create_specialized_agents()
        
    def _create_specialized_agents(self) -> dict:
        """Create specialized agents for different tasks."""
        return {
            'data_extractor': Agent.create({
                'name': 'data_extractor',
                'tools': ['secure_browser_tool', 'data_extraction_tool'],
                'specialization': 'web_data_extraction',
                'security_level': 'standard'
            }),
            'pii_analyzer': Agent.create({
                'name': 'pii_analyzer',
                'tools': ['pii_detection_tool', 'data_classification_tool'],
                'specialization': 'sensitive_data_analysis',
                'security_level': 'high'
            }),
            'compliance_validator': Agent.create({
                'name': 'compliance_validator',
                'tools': ['compliance_validation_tool', 'audit_tool'],
                'specialization': 'compliance_checking',
                'security_level': 'maximum'
            }),
            'report_generator': Agent.create({
                'name': 'report_generator',
                'tools': ['report_generation_tool', 'data_sanitization_tool'],
                'specialization': 'secure_reporting',
                'security_level': 'high'
            })
        }
        
    async def execute_coordinated_workflow(self, workflow_config: dict) -> dict:
        """Execute coordinated multi-agent workflow."""
        workflow_plan = {
            'extract_data': {
                'agent': 'data_extractor',
                'config': workflow_config['extraction'],
                'dependencies': []
            },
            'analyze_pii': {
                'agent': 'pii_analyzer',
                'config': workflow_config['pii_analysis'],
                'dependencies': ['extract_data']
            },
            'validate_compliance': {
                'agent': 'compliance_validator',
                'config': workflow_config['compliance'],
                'dependencies': ['analyze_pii']
            },
            'generate_report': {
                'agent': 'report_generator',
                'config': workflow_config['reporting'],
                'dependencies': ['validate_compliance']
            }
        }
        
        # Execute coordinated workflow
        results = await self.orchestrator.execute_workflow_plan(
            workflow_plan,
            agents=self.agents,
            isolation_level='high'
        )
        
        return results

# Usage Example
multi_agent_workflow = MultiAgentSecureWorkflow(max_concurrent_agents=4)

workflow_config = {
    'extraction': {
        'urls': ['https://portal1.example.com', 'https://portal2.example.com'],
        'credentials': 'aws_secrets_manager://prod/portal_creds'
    },
    'pii_analysis': {
        'confidence_threshold': 0.8,
        'industry_rules': 'healthcare'
    },
    'compliance': {
        'frameworks': ['hipaa', 'gdpr'],
        'audit_level': 'comprehensive'
    },
    'reporting': {
        'format': 'secure_pdf',
        'recipients': ['compliance@company.com']
    }
}

results = await multi_agent_workflow.execute_coordinated_workflow(workflow_config)
```

### 3. Multi-LLM Security Routing Pattern

Intelligent routing between different LLM models based on data sensitivity:

```python
from tools.strands_security_policies import MultiLLMSecurityManager, DataClassifier

class MultiLLMSecureAgent:
    """Agent with intelligent LLM routing based on data sensitivity."""
    
    def __init__(self, region: str):
        self.region = region
        self.llm_manager = MultiLLMSecurityManager(region)
        self.data_classifier = DataClassifier()
        self.model_configs = self._setup_model_configs()
        
    def _setup_model_configs(self) -> dict:
        """Configure different models for different security levels."""
        return {
            'maximum_security': {
                'model': 'anthropic.claude-3-sonnet-20240229-v1:0',
                'data_types': ['phi', 'pii', 'financial'],
                'compliance': ['hipaa', 'pci_dss'],
                'cost_tier': 'premium'
            },
            'high_security': {
                'model': 'anthropic.claude-3-haiku-20240307-v1:0',
                'data_types': ['pii', 'confidential'],
                'compliance': ['gdpr'],
                'cost_tier': 'standard'
            },
            'standard_security': {
                'model': 'meta.llama2-70b-chat-v1',
                'data_types': ['internal', 'business'],
                'compliance': ['basic'],
                'cost_tier': 'economical'
            },
            'basic_security': {
                'model': 'amazon.titan-text-express-v1',
                'data_types': ['public', 'general'],
                'compliance': [],
                'cost_tier': 'budget'
            }
        }
        
    def process_with_appropriate_model(self, content: str, context: dict) -> dict:
        """Process content with appropriate model based on sensitivity."""
        # Classify data sensitivity
        classification = self.data_classifier.classify(content, context)
        
        # Select appropriate model
        model_config = self._select_model_for_classification(classification)
        
        # Create agent with selected model
        agent = self._create_agent_with_model(model_config)
        
        # Process with security controls
        result = agent.process(
            content=content,
            context=context,
            security_level=classification.security_level,
            audit_logging=True
        )
        
        # Log routing decision
        self._log_routing_decision(classification, model_config, result)
        
        return result
        
    def _select_model_for_classification(self, classification: DataClassification) -> dict:
        """Select model based on data classification."""
        if classification.contains_phi or classification.contains_financial_data:
            return self.model_configs['maximum_security']
        elif classification.contains_pii:
            return self.model_configs['high_security']
        elif classification.sensitivity_level == 'internal':
            return self.model_configs['standard_security']
        else:
            return self.model_configs['basic_security']
            
    def _create_agent_with_model(self, model_config: dict) -> Agent:
        """Create agent configured for specific model."""
        agent_config = {
            'name': f'secure_agent_{model_config["cost_tier"]}',
            'llm_model': model_config['model'],
            'security_level': model_config.get('security_level', 'standard'),
            'compliance_requirements': model_config['compliance'],
            'tools': self._get_tools_for_security_level(model_config)
        }
        
        return Agent.create(agent_config)

# Usage Example
multi_llm_agent = MultiLLMSecureAgent(region='us-east-1')

# Process different types of content
healthcare_content = "Patient John Doe, SSN: 123-45-6789, has diabetes"
financial_content = "Credit card 4532-1234-5678-9012 charged $500"
general_content = "The weather is nice today"

# Each will be routed to appropriate model
healthcare_result = multi_llm_agent.process_with_appropriate_model(
    healthcare_content, 
    {'domain': 'healthcare', 'compliance': 'hipaa'}
)

financial_result = multi_llm_agent.process_with_appropriate_model(
    financial_content, 
    {'domain': 'financial', 'compliance': 'pci_dss'}
)

general_result = multi_llm_agent.process_with_appropriate_model(
    general_content, 
    {'domain': 'general'}
)
```

### 4. Session Pool Management Pattern

Efficient management of browser sessions for high-throughput operations:

```python
from tools.strands_agentcore_session_helpers import SessionPool, SessionHealthMonitor

class HighThroughputSecureProcessor:
    """High-throughput processing with session pool management."""
    
    def __init__(self, max_sessions: int = 20, region: str = 'us-east-1'):
        self.session_pool = SessionPool(
            max_sessions=max_sessions,
            region=region,
            health_check_interval=60,
            cleanup_interval=300
        )
        self.health_monitor = SessionHealthMonitor(self.session_pool)
        self.processing_queue = asyncio.Queue()
        
    async def start_processing(self):
        """Start high-throughput processing with session management."""
        # Start health monitoring
        self.health_monitor.start()
        
        # Start worker tasks
        workers = [
            asyncio.create_task(self._worker(f'worker_{i}'))
            for i in range(self.session_pool.max_sessions // 2)
        ]
        
        # Wait for all workers to complete
        await asyncio.gather(*workers)
        
    async def _worker(self, worker_id: str):
        """Worker task for processing items from queue."""
        while True:
            try:
                # Get work item
                work_item = await self.processing_queue.get()
                
                if work_item is None:  # Shutdown signal
                    break
                    
                # Get session from pool
                session = await self.session_pool.get_session()
                
                try:
                    # Process work item
                    result = await self._process_work_item(work_item, session)
                    
                    # Store result
                    await self._store_result(work_item, result)
                    
                finally:
                    # Return session to pool
                    await self.session_pool.return_session(session)
                    
                # Mark task as done
                self.processing_queue.task_done()
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                
    async def _process_work_item(self, work_item: dict, session: BrowserSession) -> dict:
        """Process individual work item with security controls."""
        # Create agent for this work item
        agent = Agent.create({
            'name': f'processor_{work_item["id"]}',
            'session': session,
            'security_level': work_item.get('security_level', 'standard'),
            'tools': ['secure_browser_tool', 'pii_detection_tool']
        })
        
        # Execute processing workflow
        workflow = self._create_workflow_for_item(work_item)
        result = await agent.execute_workflow(workflow)
        
        return result
        
    def add_work_item(self, item: dict):
        """Add work item to processing queue."""
        self.processing_queue.put_nowait(item)
        
    async def shutdown(self):
        """Graceful shutdown of processing system."""
        # Signal workers to stop
        for _ in range(self.session_pool.max_sessions // 2):
            await self.processing_queue.put(None)
            
        # Stop health monitoring
        self.health_monitor.stop()
        
        # Cleanup session pool
        await self.session_pool.cleanup_all_sessions()

# Usage Example
processor = HighThroughputSecureProcessor(max_sessions=20)

# Add work items
work_items = [
    {
        'id': f'item_{i}',
        'url': f'https://portal.example.com/page/{i}',
        'security_level': 'high' if i % 3 == 0 else 'standard',
        'data_extraction_rules': {...}
    }
    for i in range(100)
]

for item in work_items:
    processor.add_work_item(item)

# Start processing
await processor.start_processing()
```

### 5. Error Recovery and Resilience Pattern

Robust error handling and recovery mechanisms:

```python
from tools.strands_agentcore_session_helpers import SessionRecoveryManager
from strands_agents.exceptions import SecurityViolationError, SessionTimeoutError

class ResilientSecureAgent:
    """Agent with comprehensive error recovery and resilience."""
    
    def __init__(self, region: str, max_retries: int = 3):
        self.region = region
        self.max_retries = max_retries
        self.recovery_manager = SessionRecoveryManager(region)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=300
        )
        
    async def execute_with_resilience(self, workflow: dict, context: dict) -> dict:
        """Execute workflow with comprehensive error handling."""
        attempt = 0
        last_exception = None
        
        while attempt < self.max_retries:
            try:
                # Check circuit breaker
                if self.circuit_breaker.is_open():
                    raise CircuitBreakerOpenError("Circuit breaker is open")
                    
                # Execute workflow
                result = await self._execute_workflow_attempt(workflow, context, attempt)
                
                # Reset circuit breaker on success
                self.circuit_breaker.record_success()
                
                return result
                
            except SecurityViolationError as e:
                # Security violations should not be retried
                logger.error(f"Security violation: {e}")
                raise e
                
            except SessionTimeoutError as e:
                # Session timeout - try to recover
                logger.warning(f"Session timeout on attempt {attempt + 1}: {e}")
                await self._handle_session_timeout(context)
                last_exception = e
                
            except AgentCoreServiceError as e:
                # Service error - implement exponential backoff
                logger.warning(f"Service error on attempt {attempt + 1}: {e}")
                await self._handle_service_error(e, attempt)
                last_exception = e
                
            except Exception as e:
                # Unexpected error
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                self.circuit_breaker.record_failure()
                last_exception = e
                
            attempt += 1
            
            # Exponential backoff
            if attempt < self.max_retries:
                backoff_time = min(300, (2 ** attempt) + random.uniform(0, 1))
                await asyncio.sleep(backoff_time)
                
        # All retries exhausted
        self.circuit_breaker.record_failure()
        raise MaxRetriesExceededError(f"Failed after {self.max_retries} attempts") from last_exception
        
    async def _execute_workflow_attempt(self, workflow: dict, context: dict, attempt: int) -> dict:
        """Execute single workflow attempt."""
        # Create or recover session
        session = await self._get_or_create_session(context, attempt)
        
        try:
            # Create agent
            agent = Agent.create({
                'name': f'resilient_agent_attempt_{attempt}',
                'session': session,
                'security_level': context.get('security_level', 'standard'),
                'timeout': context.get('timeout', 300)
            })
            
            # Execute workflow with monitoring
            with self._create_execution_monitor(context) as monitor:
                result = await agent.execute_workflow(workflow)
                
            # Validate result
            self._validate_workflow_result(result, context)
            
            return result
            
        finally:
            # Cleanup session if needed
            await self._cleanup_session_if_needed(session, context)
            
    async def _handle_session_timeout(self, context: dict):
        """Handle session timeout with recovery."""
        # Invalidate current session
        await self.recovery_manager.invalidate_session(context.get('session_id'))
        
        # Clear session from context
        context.pop('session_id', None)
        
        # Log timeout event
        logger.info("Session timeout handled, will create new session on retry")
        
    async def _handle_service_error(self, error: AgentCoreServiceError, attempt: int):
        """Handle service errors with appropriate response."""
        if error.is_throttling_error():
            # Implement exponential backoff for throttling
            backoff_time = min(300, (2 ** attempt) * 2)
            logger.info(f"Throttling detected, backing off for {backoff_time}s")
            await asyncio.sleep(backoff_time)
            
        elif error.is_quota_exceeded():
            # Wait longer for quota reset
            logger.warning("Quota exceeded, waiting for reset")
            await asyncio.sleep(600)  # 10 minutes
            
        elif error.is_service_unavailable():
            # Service unavailable - shorter backoff
            logger.warning("Service unavailable, short backoff")
            await asyncio.sleep(30)

# Usage Example
resilient_agent = ResilientSecureAgent(region='us-east-1', max_retries=3)

workflow = {
    'steps': [
        {'action': 'navigate', 'url': 'https://unreliable-service.example.com'},
        {'action': 'extract_data', 'selector': '.important-data'},
        {'action': 'process_data', 'security_level': 'high'}
    ]
}

context = {
    'security_level': 'high',
    'timeout': 300,
    'compliance_mode': 'hipaa'
}

try:
    result = await resilient_agent.execute_with_resilience(workflow, context)
    print(f"Workflow completed successfully: {result}")
except MaxRetriesExceededError as e:
    print(f"Workflow failed after all retries: {e}")
except SecurityViolationError as e:
    print(f"Security violation detected: {e}")
```

### 6. Custom Security Tool Development Pattern

Pattern for developing custom security tools with Strands:

```python
from strands_agents.tools import BaseTool
from tools.strands_security_policies import SecurityPolicy

class CustomIndustrySecurityTool(BaseTool):
    """Custom security tool for industry-specific requirements."""
    
    def __init__(self, industry: str, compliance_frameworks: list):
        super().__init__(name=f"{industry}_security_tool")
        self.industry = industry
        self.compliance_frameworks = compliance_frameworks
        self.security_rules = self._load_industry_security_rules()
        self.validation_patterns = self._load_validation_patterns()
        
    def _load_industry_security_rules(self) -> dict:
        """Load industry-specific security rules."""
        rules = {}
        
        if self.industry == 'healthcare':
            rules.update({
                'phi_detection': True,
                'hipaa_compliance': True,
                'patient_consent_validation': True,
                'medical_record_encryption': True
            })
            
        elif self.industry == 'financial':
            rules.update({
                'pci_dss_compliance': True,
                'cardholder_data_protection': True,
                'fraud_detection': True,
                'transaction_monitoring': True
            })
            
        elif self.industry == 'legal':
            rules.update({
                'attorney_client_privilege': True,
                'document_confidentiality': True,
                'privilege_log_maintenance': True,
                'ethical_wall_enforcement': True
            })
            
        return rules
        
    def execute(self, content: str, context: dict) -> dict:
        """Execute industry-specific security validation."""
        validation_results = []
        
        # Apply industry-specific validations
        for rule_name, rule_enabled in self.security_rules.items():
            if rule_enabled:
                validation_result = self._apply_security_rule(
                    rule_name, content, context
                )
                validation_results.append(validation_result)
                
        # Consolidate results
        overall_result = self._consolidate_validation_results(validation_results)
        
        # Apply remediation if needed
        if overall_result.requires_remediation:
            remediated_content = self._apply_remediation(
                content, overall_result.violations
            )
        else:
            remediated_content = content
            
        return {
            'original_content': content,
            'remediated_content': remediated_content,
            'validation_results': validation_results,
            'overall_compliance': overall_result.is_compliant,
            'violations': overall_result.violations,
            'remediation_applied': overall_result.requires_remediation
        }
        
    def _apply_security_rule(self, rule_name: str, content: str, context: dict) -> ValidationResult:
        """Apply specific security rule."""
        if rule_name == 'phi_detection':
            return self._detect_phi(content, context)
        elif rule_name == 'pci_dss_compliance':
            return self._validate_pci_dss(content, context)
        elif rule_name == 'attorney_client_privilege':
            return self._validate_privilege(content, context)
        else:
            return ValidationResult(rule_name, True, "Rule not implemented")
            
    def _detect_phi(self, content: str, context: dict) -> ValidationResult:
        """Detect Protected Health Information."""
        phi_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{2}/\d{2}/\d{4}\b',  # Date of birth
            r'\b[A-Za-z]+ [A-Za-z]+ Medical Center\b',  # Medical facility names
            r'\bpatient\s+(?:id|number):\s*\d+\b'  # Patient IDs
        ]
        
        violations = []
        for pattern in phi_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                violations.extend([
                    PHIViolation(pattern=pattern, match=match, position=content.find(match))
                    for match in matches
                ])
                
        return ValidationResult(
            rule_name='phi_detection',
            is_compliant=len(violations) == 0,
            violations=violations,
            message=f"Found {len(violations)} PHI violations"
        )

# Usage Example - Healthcare Security Tool
healthcare_tool = CustomIndustrySecurityTool(
    industry='healthcare',
    compliance_frameworks=['hipaa', 'hitech']
)

# Register with agent
agent = Agent.create({
    'name': 'healthcare_compliance_agent',
    'tools': [healthcare_tool],
    'security_level': 'maximum'
})

# Test content
test_content = "Patient John Doe, SSN: 123-45-6789, visited General Hospital on 01/15/2024"

result = healthcare_tool.execute(test_content, {'domain': 'healthcare'})
print(f"Compliance check: {result['overall_compliance']}")
print(f"Violations found: {len(result['violations'])}")
```

## Integration Best Practices

### 1. Security-First Design

- **Principle of Least Privilege**: Grant minimal necessary permissions
- **Defense in Depth**: Implement multiple layers of security controls
- **Secure by Default**: Default configurations should be secure
- **Zero Trust Architecture**: Never trust, always verify

### 2. Performance Optimization

- **Session Reuse**: Implement efficient session pooling
- **Concurrent Processing**: Use async patterns for high throughput
- **Resource Management**: Proper cleanup and resource management
- **Caching Strategies**: Cache non-sensitive data appropriately

### 3. Error Handling and Resilience

- **Graceful Degradation**: Handle failures gracefully
- **Circuit Breaker Pattern**: Prevent cascade failures
- **Exponential Backoff**: Implement proper retry strategies
- **Comprehensive Logging**: Log all errors for debugging

### 4. Monitoring and Observability

- **Real-time Monitoring**: Monitor security events in real-time
- **Comprehensive Metrics**: Track performance and security metrics
- **Alerting**: Set up appropriate alerts for security events
- **Audit Trails**: Maintain comprehensive audit logs

### 5. Compliance and Governance

- **Industry Standards**: Follow industry-specific compliance requirements
- **Regular Audits**: Conduct regular security and compliance audits
- **Documentation**: Maintain comprehensive documentation
- **Training**: Ensure team is trained on security best practices

## Conclusion

These integration patterns provide a comprehensive foundation for building secure, scalable, and maintainable Strands-AgentCore Browser Tool integrations. Each pattern addresses specific use cases while maintaining security and compliance requirements. Choose the appropriate pattern based on your specific requirements and scale accordingly.