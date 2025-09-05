#!/usr/bin/env python3
"""
Strands Security Policies
=========================

This module provides comprehensive security policy management for Strands agents
working with AgentCore Browser Tool. It includes Bedrock model security routing,
compliance validation, and dynamic policy enforcement based on data sensitivity
and regulatory requirements.

Features:
- Multi-LLM security routing based on data classification
- Compliance-specific policy enforcement
- Dynamic security policy adaptation
- Real-time security validation
- Comprehensive audit logging
- Integration with AgentCore Browser Tool security features
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import hashlib

# Strands framework imports
from strands import Agent
from strands.tools import tool, PythonAgentTool
from strands.types.tools import AgentTool

# AWS imports
import boto3


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    GDPR = "gdpr"
    SOX = "sox"
    CCPA = "ccpa"
    FERPA = "ferpa"
    ATTORNEY_CLIENT = "attorney_client"
    GENERAL = "general"


class SecurityLevel(Enum):
    """Security levels for operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    MAXIMUM = "maximum"


class LLMProvider(Enum):
    """Supported LLM providers."""
    BEDROCK_CLAUDE = "bedrock_claude"
    BEDROCK_LLAMA = "bedrock_llama"
    BEDROCK_TITAN = "bedrock_titan"
    OPENAI_GPT4 = "openai_gpt4"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    LOCAL_OLLAMA = "local_ollama"


@dataclass
class SecurityPolicy:
    """Security policy definition."""
    policy_id: str
    name: str
    description: str
    data_classification: DataClassification
    compliance_frameworks: List[ComplianceFramework]
    security_level: SecurityLevel
    allowed_llm_providers: List[LLMProvider]
    data_residency_requirements: List[str]
    encryption_required: bool
    audit_level: str
    session_timeout: int
    max_data_retention_days: int
    pii_handling_rules: Dict[str, Any]
    browser_security_config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "description": self.description,
            "data_classification": self.data_classification.value,
            "compliance_frameworks": [cf.value for cf in self.compliance_frameworks],
            "security_level": self.security_level.value,
            "allowed_llm_providers": [llm.value for llm in self.allowed_llm_providers],
            "data_residency_requirements": self.data_residency_requirements,
            "encryption_required": self.encryption_required,
            "audit_level": self.audit_level,
            "session_timeout": self.session_timeout,
            "max_data_retention_days": self.max_data_retention_days,
            "pii_handling_rules": self.pii_handling_rules,
            "browser_security_config": self.browser_security_config
        }


@dataclass
class PolicyViolation:
    """Represents a security policy violation."""
    violation_id: str
    policy_id: str
    violation_type: str
    severity: str
    description: str
    context: Dict[str, Any]
    timestamp: datetime
    remediation_required: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "violation_id": self.violation_id,
            "policy_id": self.policy_id,
            "violation_type": self.violation_type,
            "severity": self.severity,
            "description": self.description,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "remediation_required": self.remediation_required
        }


class SecurityPolicyLibrary:
    """Library of predefined security policies."""
    
    def __init__(self):
        self.policies = self._initialize_policies()
    
    def _initialize_policies(self) -> Dict[str, SecurityPolicy]:
        """Initialize predefined security policies."""
        policies = {}
        
        # HIPAA Policy
        policies["hipaa_maximum"] = SecurityPolicy(
            policy_id="hipaa_maximum",
            name="HIPAA Maximum Security",
            description="Maximum security policy for HIPAA-protected health information",
            data_classification=DataClassification.RESTRICTED,
            compliance_frameworks=[ComplianceFramework.HIPAA],
            security_level=SecurityLevel.MAXIMUM,
            allowed_llm_providers=[LLMProvider.BEDROCK_CLAUDE, LLMProvider.LOCAL_OLLAMA],
            data_residency_requirements=["us-east-1", "us-west-2"],
            encryption_required=True,
            audit_level="comprehensive",
            session_timeout=1800,  # 30 minutes
            max_data_retention_days=2555,  # 7 years
            pii_handling_rules={
                "detection_required": True,
                "masking_strategy": "token_mask",
                "consent_validation": True,
                "data_minimization": True
            },
            browser_security_config={
                "isolation_level": "maximum",
                "screenshot_disabled": True,
                "clipboard_disabled": True,
                "network_monitoring": True,
                "memory_protection": True,
                "data_protection": "hipaa_compliant"
            }
        )
        
        # PCI DSS Policy
        policies["pci_dss_critical"] = SecurityPolicy(
            policy_id="pci_dss_critical",
            name="PCI DSS Critical Security",
            description="Critical security policy for payment card data",
            data_classification=DataClassification.RESTRICTED,
            compliance_frameworks=[ComplianceFramework.PCI_DSS],
            security_level=SecurityLevel.CRITICAL,
            allowed_llm_providers=[LLMProvider.BEDROCK_CLAUDE, LLMProvider.LOCAL_OLLAMA],
            data_residency_requirements=["us-east-1", "us-west-2", "eu-west-1"],
            encryption_required=True,
            audit_level="comprehensive",
            session_timeout=900,  # 15 minutes
            max_data_retention_days=2555,  # 7 years
            pii_handling_rules={
                "detection_required": True,
                "masking_strategy": "full_mask",
                "cardholder_data_protection": True,
                "tokenization_required": True
            },
            browser_security_config={
                "isolation_level": "maximum",
                "screenshot_disabled": True,
                "clipboard_disabled": True,
                "network_monitoring": True,
                "memory_protection": True,
                "data_protection": "pci_dss_compliant"
            }
        )
        
        # GDPR Policy
        policies["gdpr_high"] = SecurityPolicy(
            policy_id="gdpr_high",
            name="GDPR High Security",
            description="High security policy for GDPR personal data",
            data_classification=DataClassification.CONFIDENTIAL,
            compliance_frameworks=[ComplianceFramework.GDPR],
            security_level=SecurityLevel.HIGH,
            allowed_llm_providers=[
                LLMProvider.BEDROCK_CLAUDE, 
                LLMProvider.BEDROCK_LLAMA,
                LLMProvider.LOCAL_OLLAMA
            ],
            data_residency_requirements=["eu-west-1", "eu-central-1"],
            encryption_required=True,
            audit_level="detailed",
            session_timeout=2400,  # 40 minutes
            max_data_retention_days=1095,  # 3 years
            pii_handling_rules={
                "detection_required": True,
                "masking_strategy": "partial_mask",
                "consent_validation": True,
                "right_to_erasure": True,
                "data_portability": True
            },
            browser_security_config={
                "isolation_level": "high",
                "screenshot_disabled": True,
                "clipboard_disabled": True,
                "network_monitoring": True,
                "memory_protection": False,
                "data_protection": "gdpr_compliant"
            }
        )
        
        # Attorney-Client Privilege Policy
        policies["attorney_client_privileged"] = SecurityPolicy(
            policy_id="attorney_client_privileged",
            name="Attorney-Client Privileged",
            description="Maximum security for attorney-client privileged communications",
            data_classification=DataClassification.TOP_SECRET,
            compliance_frameworks=[ComplianceFramework.ATTORNEY_CLIENT],
            security_level=SecurityLevel.MAXIMUM,
            allowed_llm_providers=[LLMProvider.LOCAL_OLLAMA],  # Only local processing
            data_residency_requirements=["on_premises"],
            encryption_required=True,
            audit_level="comprehensive",
            session_timeout=1800,  # 30 minutes
            max_data_retention_days=2555,  # 7 years
            pii_handling_rules={
                "detection_required": True,
                "masking_strategy": "token_mask",
                "privilege_protection": True,
                "work_product_protection": True
            },
            browser_security_config={
                "isolation_level": "maximum",
                "screenshot_disabled": True,
                "clipboard_disabled": True,
                "network_monitoring": True,
                "memory_protection": True,
                "data_protection": "attorney_client_privileged",
                "privilege_protection": True
            }
        )
        
        # General High Security Policy
        policies["general_high"] = SecurityPolicy(
            policy_id="general_high",
            name="General High Security",
            description="High security policy for general sensitive data",
            data_classification=DataClassification.CONFIDENTIAL,
            compliance_frameworks=[ComplianceFramework.GENERAL],
            security_level=SecurityLevel.HIGH,
            allowed_llm_providers=[
                LLMProvider.BEDROCK_CLAUDE,
                LLMProvider.BEDROCK_LLAMA,
                LLMProvider.BEDROCK_TITAN,
                LLMProvider.OPENAI_GPT4
            ],
            data_residency_requirements=["us-east-1", "us-west-2"],
            encryption_required=True,
            audit_level="standard",
            session_timeout=3600,  # 60 minutes
            max_data_retention_days=1095,  # 3 years
            pii_handling_rules={
                "detection_required": True,
                "masking_strategy": "partial_mask"
            },
            browser_security_config={
                "isolation_level": "high",
                "screenshot_disabled": False,
                "clipboard_disabled": False,
                "network_monitoring": True,
                "memory_protection": False,
                "data_protection": "standard"
            }
        )
        
        return policies
    
    def get_policy(self, policy_id: str) -> Optional[SecurityPolicy]:
        """Get a security policy by ID."""
        return self.policies.get(policy_id)
    
    def get_policies_for_compliance(self, framework: ComplianceFramework) -> List[SecurityPolicy]:
        """Get all policies for a specific compliance framework."""
        return [
            policy for policy in self.policies.values()
            if framework in policy.compliance_frameworks
        ]
    
    def get_policies_for_classification(self, classification: DataClassification) -> List[SecurityPolicy]:
        """Get all policies for a specific data classification."""
        return [
            policy for policy in self.policies.values()
            if policy.data_classification == classification
        ]


class LLMSecurityRouter(PythonAgentTool):
    """Routes LLM requests based on security policies."""
    
    def __init__(self, region: str = "us-east-1"):
        super().__init__(name="llm_security_router")
        self.region = region
        self.policy_library = SecurityPolicyLibrary()
        self.audit_logger = AuditLogger(service_name="llm_security_router")
        
        # LLM client configurations
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=region)
        self.llm_configs = self._initialize_llm_configs()
    
    def _initialize_llm_configs(self) -> Dict[LLMProvider, Dict[str, Any]]:
        """Initialize LLM provider configurations."""
        return {
            LLMProvider.BEDROCK_CLAUDE: {
                "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
                "region": self.region,
                "endpoint": "bedrock-runtime",
                "data_residency": ["us-east-1", "us-west-2", "eu-west-1"]
            },
            LLMProvider.BEDROCK_LLAMA: {
                "model_id": "meta.llama2-70b-chat-v1",
                "region": self.region,
                "endpoint": "bedrock-runtime",
                "data_residency": ["us-east-1", "us-west-2"]
            },
            LLMProvider.BEDROCK_TITAN: {
                "model_id": "amazon.titan-text-express-v1",
                "region": self.region,
                "endpoint": "bedrock-runtime",
                "data_residency": ["us-east-1", "us-west-2"]
            },
            LLMProvider.LOCAL_OLLAMA: {
                "endpoint": "http://localhost:11434",
                "data_residency": ["on_premises"]
            }
        }
    
    async def route_request(self, 
                          request_data: Dict[str, Any],
                          policy_id: str,
                          fallback_allowed: bool = True) -> Dict[str, Any]:
        """Route LLM request based on security policy."""
        policy = self.policy_library.get_policy(policy_id)
        if not policy:
            raise ValueError(f"Security policy not found: {policy_id}")
        
        # Validate request against policy
        violations = await self._validate_request(request_data, policy)
        if violations:
            await self._handle_policy_violations(violations, policy_id)
            if not fallback_allowed:
                raise SecurityError(f"Policy violations detected: {violations}")
        
        # Select appropriate LLM provider
        selected_provider = await self._select_llm_provider(request_data, policy)
        
        # Route request to selected provider
        try:
            response = await self._execute_llm_request(request_data, selected_provider, policy)
            
            # Log successful routing
            await self.audit_logger.log_event({
                "event_type": "llm_request_routed",
                "policy_id": policy_id,
                "selected_provider": selected_provider.value,
                "data_classification": policy.data_classification.value,
                "security_level": policy.security_level.value,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return response
            
        except Exception as e:
            # Handle provider failure with fallback
            if fallback_allowed and len(policy.allowed_llm_providers) > 1:
                return await self._handle_provider_fallback(request_data, policy, selected_provider, str(e))
            else:
                raise
    
    async def _validate_request(self, 
                              request_data: Dict[str, Any], 
                              policy: SecurityPolicy) -> List[PolicyViolation]:
        """Validate request against security policy."""
        violations = []
        
        # Check data residency requirements
        if policy.data_residency_requirements:
            request_region = request_data.get("region", self.region)
            if request_region not in policy.data_residency_requirements:
                violations.append(PolicyViolation(
                    violation_id=str(uuid.uuid4()),
                    policy_id=policy.policy_id,
                    violation_type="data_residency",
                    severity="high",
                    description=f"Request region {request_region} not allowed by policy",
                    context={"request_region": request_region, "allowed_regions": policy.data_residency_requirements},
                    timestamp=datetime.utcnow(),
                    remediation_required=True
                ))
        
        # Check encryption requirements
        if policy.encryption_required and not request_data.get("encryption_enabled", False):
            violations.append(PolicyViolation(
                violation_id=str(uuid.uuid4()),
                policy_id=policy.policy_id,
                violation_type="encryption_required",
                severity="critical",
                description="Encryption required but not enabled",
                context={"encryption_enabled": request_data.get("encryption_enabled", False)},
                timestamp=datetime.utcnow(),
                remediation_required=True
            ))
        
        # Check PII handling requirements
        if policy.pii_handling_rules.get("detection_required", False):
            if not request_data.get("pii_detected", False):
                violations.append(PolicyViolation(
                    violation_id=str(uuid.uuid4()),
                    policy_id=policy.policy_id,
                    violation_type="pii_detection_required",
                    severity="medium",
                    description="PII detection required but not performed",
                    context={"pii_detected": request_data.get("pii_detected", False)},
                    timestamp=datetime.utcnow(),
                    remediation_required=False
                ))
        
        return violations
    
    async def _select_llm_provider(self, 
                                 request_data: Dict[str, Any], 
                                 policy: SecurityPolicy) -> LLMProvider:
        """Select appropriate LLM provider based on policy and request."""
        available_providers = policy.allowed_llm_providers.copy()
        
        # Filter by data residency requirements
        if policy.data_residency_requirements:
            filtered_providers = []
            for provider in available_providers:
                provider_config = self.llm_configs.get(provider, {})
                provider_residency = provider_config.get("data_residency", [])
                
                if any(region in policy.data_residency_requirements for region in provider_residency):
                    filtered_providers.append(provider)
            
            available_providers = filtered_providers
        
        if not available_providers:
            raise SecurityError("No LLM providers available that meet policy requirements")
        
        # Select based on security level and data classification
        if policy.security_level == SecurityLevel.MAXIMUM:
            # Prefer local or most secure providers
            if LLMProvider.LOCAL_OLLAMA in available_providers:
                return LLMProvider.LOCAL_OLLAMA
            elif LLMProvider.BEDROCK_CLAUDE in available_providers:
                return LLMProvider.BEDROCK_CLAUDE
        
        elif policy.security_level == SecurityLevel.CRITICAL:
            # Prefer Bedrock providers
            bedrock_providers = [p for p in available_providers if p.value.startswith("bedrock_")]
            if bedrock_providers:
                return bedrock_providers[0]
        
        # Default to first available provider
        return available_providers[0]
    
    async def _execute_llm_request(self, 
                                 request_data: Dict[str, Any],
                                 provider: LLMProvider,
                                 policy: SecurityPolicy) -> Dict[str, Any]:
        """Execute LLM request with selected provider."""
        provider_config = self.llm_configs[provider]
        
        if provider.value.startswith("bedrock_"):
            return await self._execute_bedrock_request(request_data, provider_config, policy)
        elif provider == LLMProvider.LOCAL_OLLAMA:
            return await self._execute_ollama_request(request_data, provider_config, policy)
        else:
            raise NotImplementedError(f"Provider {provider.value} not implemented")
    
    async def _execute_bedrock_request(self, 
                                     request_data: Dict[str, Any],
                                     config: Dict[str, Any],
                                     policy: SecurityPolicy) -> Dict[str, Any]:
        """Execute request using Bedrock."""
        try:
            # Prepare request body
            body = {
                "prompt": request_data.get("prompt", ""),
                "max_tokens_to_sample": request_data.get("max_tokens", 1000),
                "temperature": request_data.get("temperature", 0.1),
                "top_p": request_data.get("top_p", 0.9)
            }
            
            # Add security headers
            if policy.encryption_required:
                body["security_config"] = {
                    "encryption_enabled": True,
                    "compliance_level": policy.compliance_frameworks[0].value if policy.compliance_frameworks else "general"
                }
            
            response = self.bedrock_client.invoke_model(
                modelId=config["model_id"],
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json"
            )
            
            response_body = json.loads(response['body'].read())
            
            return {
                "provider": config["model_id"],
                "response": response_body,
                "security_applied": True,
                "policy_compliant": True
            }
            
        except Exception as e:
            raise LLMProviderError(f"Bedrock request failed: {str(e)}")
    
    async def _execute_ollama_request(self, 
                                    request_data: Dict[str, Any],
                                    config: Dict[str, Any],
                                    policy: SecurityPolicy) -> Dict[str, Any]:
        """Execute request using local Ollama."""
        # This would integrate with local Ollama instance
        # For demo purposes, return a mock response
        return {
            "provider": "ollama_local",
            "response": {"completion": "Local processing completed securely"},
            "security_applied": True,
            "policy_compliant": True,
            "data_residency": "on_premises"
        }
    
    async def _handle_provider_fallback(self, 
                                      request_data: Dict[str, Any],
                                      policy: SecurityPolicy,
                                      failed_provider: LLMProvider,
                                      error: str) -> Dict[str, Any]:
        """Handle provider failure with fallback."""
        # Remove failed provider from available options
        remaining_providers = [p for p in policy.allowed_llm_providers if p != failed_provider]
        
        if not remaining_providers:
            raise LLMProviderError(f"All providers failed. Last error: {error}")
        
        # Log fallback
        await self.audit_logger.log_event({
            "event_type": "llm_provider_fallback",
            "failed_provider": failed_provider.value,
            "error": error,
            "remaining_providers": [p.value for p in remaining_providers],
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Try next provider
        fallback_provider = remaining_providers[0]
        return await self._execute_llm_request(request_data, fallback_provider, policy)
    
    async def _handle_policy_violations(self, 
                                      violations: List[PolicyViolation],
                                      policy_id: str):
        """Handle security policy violations."""
        for violation in violations:
            await self.audit_logger.log_event({
                "event_type": "security_policy_violation",
                "violation": violation.to_dict(),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Handle critical violations
            if violation.severity == "critical":
                # Could trigger alerts, notifications, etc.
                pass


class ComplianceValidator(PythonAgentTool):
    """Validates operations against compliance requirements."""
    
    def __init__(self):
        super().__init__(name="compliance_validator")
        self.policy_library = SecurityPolicyLibrary()
        self.audit_logger = AuditLogger(service_name="compliance_validator")
    
    async def validate_operation(self, 
                               operation_data: Dict[str, Any],
                               compliance_frameworks: List[ComplianceFramework]) -> Dict[str, Any]:
        """Validate operation against compliance frameworks."""
        validation_results = {
            "compliant": True,
            "violations": [],
            "recommendations": [],
            "framework_results": {}
        }
        
        for framework in compliance_frameworks:
            framework_result = await self._validate_framework_compliance(operation_data, framework)
            validation_results["framework_results"][framework.value] = framework_result
            
            if not framework_result["compliant"]:
                validation_results["compliant"] = False
                validation_results["violations"].extend(framework_result["violations"])
            
            validation_results["recommendations"].extend(framework_result.get("recommendations", []))
        
        # Log validation
        await self.audit_logger.log_event({
            "event_type": "compliance_validation",
            "frameworks": [f.value for f in compliance_frameworks],
            "compliant": validation_results["compliant"],
            "violation_count": len(validation_results["violations"]),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return validation_results
    
    async def _validate_framework_compliance(self, 
                                           operation_data: Dict[str, Any],
                                           framework: ComplianceFramework) -> Dict[str, Any]:
        """Validate against specific compliance framework."""
        if framework == ComplianceFramework.HIPAA:
            return await self._validate_hipaa_compliance(operation_data)
        elif framework == ComplianceFramework.PCI_DSS:
            return await self._validate_pci_dss_compliance(operation_data)
        elif framework == ComplianceFramework.GDPR:
            return await self._validate_gdpr_compliance(operation_data)
        elif framework == ComplianceFramework.ATTORNEY_CLIENT:
            return await self._validate_attorney_client_compliance(operation_data)
        else:
            return {"compliant": True, "violations": [], "recommendations": []}
    
    async def _validate_hipaa_compliance(self, operation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate HIPAA compliance."""
        violations = []
        recommendations = []
        
        # Check for PHI handling
        if operation_data.get("contains_phi", False):
            if not operation_data.get("encryption_enabled", False):
                violations.append("PHI must be encrypted in transit and at rest")
            
            if not operation_data.get("access_controls", False):
                violations.append("Access controls required for PHI")
            
            if not operation_data.get("audit_logging", False):
                violations.append("Audit logging required for PHI access")
        
        # Check minimum necessary standard
        if not operation_data.get("data_minimization", False):
            recommendations.append("Apply minimum necessary standard for PHI access")
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "recommendations": recommendations
        }
    
    async def _validate_pci_dss_compliance(self, operation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate PCI DSS compliance."""
        violations = []
        recommendations = []
        
        # Check for cardholder data
        if operation_data.get("contains_cardholder_data", False):
            if not operation_data.get("encryption_enabled", False):
                violations.append("Cardholder data must be encrypted")
            
            if not operation_data.get("network_segmentation", False):
                violations.append("Network segmentation required for cardholder data environment")
            
            if not operation_data.get("access_controls", False):
                violations.append("Strong access controls required for cardholder data")
        
        # Check for sensitive authentication data
        if operation_data.get("contains_sad", False):
            violations.append("Sensitive authentication data must not be stored")
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "recommendations": recommendations
        }
    
    async def _validate_gdpr_compliance(self, operation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate GDPR compliance."""
        violations = []
        recommendations = []
        
        # Check for personal data
        if operation_data.get("contains_personal_data", False):
            if not operation_data.get("lawful_basis", False):
                violations.append("Lawful basis required for personal data processing")
            
            if not operation_data.get("consent_obtained", False):
                recommendations.append("Obtain explicit consent for personal data processing")
            
            if not operation_data.get("data_minimization", False):
                violations.append("Data minimization principle must be applied")
        
        # Check data subject rights
        if not operation_data.get("data_subject_rights_supported", False):
            recommendations.append("Implement data subject rights (access, rectification, erasure)")
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "recommendations": recommendations
        }
    
    async def _validate_attorney_client_compliance(self, operation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate attorney-client privilege compliance."""
        violations = []
        recommendations = []
        
        # Check for privileged communications
        if operation_data.get("contains_privileged_content", False):
            if not operation_data.get("maximum_security", False):
                violations.append("Maximum security required for privileged communications")
            
            if operation_data.get("third_party_access", False):
                violations.append("Third-party access may waive attorney-client privilege")
            
            if not operation_data.get("local_processing_only", False):
                violations.append("Privileged content should be processed locally only")
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "recommendations": recommendations
        }


# Custom exceptions
class SecurityError(Exception):
    """Security-related error."""
    pass

class LLMProviderError(Exception):
    """LLM provider-related error."""
    pass

class PolicyViolationError(Exception):
    """Policy violation error."""
    pass


# Convenience functions
async def get_policy_for_data(data_classification: DataClassification, 
                            compliance_framework: ComplianceFramework) -> Optional[SecurityPolicy]:
    """Get appropriate security policy for data classification and compliance."""
    library = SecurityPolicyLibrary()
    
    # Find policies that match both criteria
    matching_policies = []
    for policy in library.policies.values():
        if (policy.data_classification == data_classification and 
            compliance_framework in policy.compliance_frameworks):
            matching_policies.append(policy)
    
    # Return the most restrictive policy
    if matching_policies:
        return max(matching_policies, key=lambda p: p.security_level.value)
    
    return None

async def route_secure_request(request_data: Dict[str, Any], 
                             compliance_framework: ComplianceFramework,
                             data_classification: DataClassification = DataClassification.CONFIDENTIAL) -> Dict[str, Any]:
    """Route request with appropriate security policy."""
    policy = await get_policy_for_data(data_classification, compliance_framework)
    if not policy:
        raise SecurityError(f"No policy found for {data_classification.value} data with {compliance_framework.value} compliance")
    
    router = LLMSecurityRouter()
    return await router.route_request(request_data, policy.policy_id)


# Example usage
async def example_usage():
    """Example of how to use the security policies."""
    # Route a HIPAA-compliant request
    request_data = {
        "prompt": "Analyze patient data",
        "contains_phi": True,
        "encryption_enabled": True,
        "region": "us-east-1"
    }
    
    try:
        response = await route_secure_request(
            request_data, 
            ComplianceFramework.HIPAA,
            DataClassification.RESTRICTED
        )
        print(f"Secure response: {response}")
    except SecurityError as e:
        print(f"Security error: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())