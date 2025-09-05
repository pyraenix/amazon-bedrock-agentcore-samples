"""
Bedrock Model Security Router for Strands Integration

This module provides intelligent routing of Strands agent requests between different
Amazon Bedrock foundation models (Claude, Llama, Titan) based on data sensitivity
levels and security policies. It includes model-specific security policies,
intelligent fallback systems, and comprehensive audit trails.

Key Features:
- Routes Strands agent requests between different Bedrock models based on data sensitivity
- Model-specific security policies for different Bedrock foundation models
- Intelligent fallback system that maintains security levels when primary model fails
- Cross-model audit trail for tracking sensitive data across different Bedrock models
- Integration with Strands' multi-LLM capabilities

Requirements Addressed:
- 4.1: Bedrock model routing based on data sensitivity
- 4.2: Model-specific security policies for different Bedrock foundation models
- 4.3: Intelligent fallback system between Bedrock models that maintains security levels
"""

import os
import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager

# AWS SDK imports
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    boto3 = None
    ClientError = Exception
    NoCredentialsError = Exception

# Strands imports
try:
    from strands_agents.core.types import ToolResult
    from strands_agents.core.exceptions import ToolExecutionError
    from strands_agents.llm.base import BaseLLM
    from strands_agents.llm.bedrock import BedrockLLM
except ImportError:
    # Mock Strands imports for development/testing
    @dataclass
    class ToolResult:
        success: bool
        data: Any = None
        error: Optional[str] = None
        metadata: Dict[str, Any] = field(default_factory=dict)
    
    class ToolExecutionError(Exception):
        pass
    
    class BaseLLM:
        def __init__(self, model_id: str):
            self.model_id = model_id
    
    class BedrockLLM(BaseLLM):
        def __init__(self, model_id: str, region: str = "us-east-1"):
            super().__init__(model_id)
            self.region = region

# Import sensitive data handler for PII detection
try:
    from .sensitive_data_handler import SensitiveDataHandler, SensitivityLevel, PIIType
except ImportError:
    from sensitive_data_handler import SensitiveDataHandler, SensitivityLevel, PIIType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BedrockModel(Enum):
    """Enumeration of supported Bedrock foundation models."""
    CLAUDE_3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_3_OPUS = "anthropic.claude-3-opus-20240229-v1:0"
    CLAUDE_3_5_SONNET = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    LLAMA_3_8B = "meta.llama3-8b-instruct-v1:0"
    LLAMA_3_70B = "meta.llama3-70b-instruct-v1:0"
    TITAN_TEXT_EXPRESS = "amazon.titan-text-express-v1"
    TITAN_TEXT_LITE = "amazon.titan-text-lite-v1"
    COHERE_COMMAND_R = "cohere.command-r-v1:0"
    COHERE_COMMAND_R_PLUS = "cohere.command-r-plus-v1:0"


class SecurityTier(Enum):
    """Security tiers for model routing decisions."""
    PUBLIC = "public"           # No sensitive data
    INTERNAL = "internal"       # Internal business data
    CONFIDENTIAL = "confidential"  # Confidential business data
    RESTRICTED = "restricted"   # Highly sensitive data (PII, PHI, PCI)
    TOP_SECRET = "top_secret"   # Maximum security requirements


class ModelCapability(Enum):
    """Model capabilities for routing decisions."""
    GENERAL_PURPOSE = "general_purpose"
    REASONING = "reasoning"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    CREATIVE_WRITING = "creative_writing"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CLASSIFICATION = "classification"


@dataclass
class ModelSecurityPolicy:
    """Security policy configuration for a specific Bedrock model."""
    model: BedrockModel
    max_security_tier: SecurityTier
    allowed_pii_types: Set[PIIType]
    data_residency_regions: List[str]
    encryption_required: bool = True
    audit_level: str = "detailed"
    max_token_limit: int = 4096
    rate_limit_per_minute: int = 60
    
    # Compliance certifications
    hipaa_compliant: bool = False
    pci_dss_compliant: bool = False
    gdpr_compliant: bool = True
    sox_compliant: bool = False
    
    # Model-specific capabilities
    capabilities: Set[ModelCapability] = field(default_factory=set)
    
    # Cost and performance metrics
    cost_per_1k_tokens: float = 0.0
    avg_latency_ms: int = 1000
    
    def can_handle_security_tier(self, tier: SecurityTier) -> bool:
        """Check if model can handle the specified security tier."""
        tier_hierarchy = {
            SecurityTier.PUBLIC: 0,
            SecurityTier.INTERNAL: 1,
            SecurityTier.CONFIDENTIAL: 2,
            SecurityTier.RESTRICTED: 3,
            SecurityTier.TOP_SECRET: 4
        }
        
        return tier_hierarchy[tier] <= tier_hierarchy[self.max_security_tier]
    
    def can_handle_pii_types(self, pii_types: Set[PIIType]) -> bool:
        """Check if model can handle the specified PII types."""
        return pii_types.issubset(self.allowed_pii_types)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary representation."""
        return {
            'model': self.model.value,
            'max_security_tier': self.max_security_tier.value,
            'allowed_pii_types': [pii.value for pii in self.allowed_pii_types],
            'data_residency_regions': self.data_residency_regions,
            'encryption_required': self.encryption_required,
            'audit_level': self.audit_level,
            'max_token_limit': self.max_token_limit,
            'rate_limit_per_minute': self.rate_limit_per_minute,
            'compliance': {
                'hipaa_compliant': self.hipaa_compliant,
                'pci_dss_compliant': self.pci_dss_compliant,
                'gdpr_compliant': self.gdpr_compliant,
                'sox_compliant': self.sox_compliant
            },
            'capabilities': [cap.value for cap in self.capabilities],
            'performance': {
                'cost_per_1k_tokens': self.cost_per_1k_tokens,
                'avg_latency_ms': self.avg_latency_ms
            }
        }


@dataclass
class RoutingRequest:
    """Request for model routing decision."""
    request_id: str
    content: str
    session_id: str
    agent_id: Optional[str] = None
    
    # Security context
    detected_pii_types: Set[PIIType] = field(default_factory=set)
    security_tier: SecurityTier = SecurityTier.PUBLIC
    compliance_requirements: Set[str] = field(default_factory=set)
    
    # Routing preferences
    preferred_models: List[BedrockModel] = field(default_factory=list)
    required_capabilities: Set[ModelCapability] = field(default_factory=set)
    max_cost_per_1k_tokens: Optional[float] = None
    max_latency_ms: Optional[int] = None
    
    # Regional requirements
    required_regions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary representation."""
        return {
            'request_id': self.request_id,
            'content_length': len(self.content),
            'session_id': self.session_id,
            'agent_id': self.agent_id,
            'security_context': {
                'detected_pii_types': [pii.value for pii in self.detected_pii_types],
                'security_tier': self.security_tier.value,
                'compliance_requirements': list(self.compliance_requirements)
            },
            'routing_preferences': {
                'preferred_models': [model.value for model in self.preferred_models],
                'required_capabilities': [cap.value for cap in self.required_capabilities],
                'max_cost_per_1k_tokens': self.max_cost_per_1k_tokens,
                'max_latency_ms': self.max_latency_ms
            },
            'regional_requirements': self.required_regions
        }


@dataclass
class RoutingDecision:
    """Result of model routing decision."""
    request_id: str
    selected_model: BedrockModel
    fallback_models: List[BedrockModel]
    routing_reason: str
    security_validated: bool
    
    # Decision metadata
    decision_timestamp: datetime = field(default_factory=datetime.now)
    confidence_score: float = 1.0
    estimated_cost: float = 0.0
    estimated_latency_ms: int = 1000
    
    # Security context
    security_tier: SecurityTier = SecurityTier.PUBLIC
    pii_types_detected: Set[PIIType] = field(default_factory=set)
    compliance_validated: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary representation."""
        return {
            'request_id': self.request_id,
            'selected_model': self.selected_model.value,
            'fallback_models': [model.value for model in self.fallback_models],
            'routing_reason': self.routing_reason,
            'security_validated': self.security_validated,
            'decision_metadata': {
                'decision_timestamp': self.decision_timestamp.isoformat(),
                'confidence_score': self.confidence_score,
                'estimated_cost': self.estimated_cost,
                'estimated_latency_ms': self.estimated_latency_ms
            },
            'security_context': {
                'security_tier': self.security_tier.value,
                'pii_types_detected': [pii.value for pii in self.pii_types_detected],
                'compliance_validated': self.compliance_validated
            }
        }


@dataclass
class ModelUsageMetrics:
    """Usage metrics for model performance tracking."""
    model: BedrockModel
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_processed: int = 0
    total_cost: float = 0.0
    avg_latency_ms: float = 0.0
    
    # Security metrics
    sensitive_requests: int = 0
    pii_requests: int = 0
    compliance_violations: int = 0
    
    # Time tracking
    first_request: Optional[datetime] = None
    last_request: Optional[datetime] = None
    
    def add_request(
        self,
        success: bool,
        tokens: int,
        cost: float,
        latency_ms: int,
        has_pii: bool = False,
        is_sensitive: bool = False
    ):
        """Add a request to the metrics."""
        self.total_requests += 1
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.total_tokens_processed += tokens
        self.total_cost += cost
        
        # Update average latency
        if self.total_requests == 1:
            self.avg_latency_ms = latency_ms
        else:
            self.avg_latency_ms = (
                (self.avg_latency_ms * (self.total_requests - 1) + latency_ms) / 
                self.total_requests
            )
        
        if has_pii:
            self.pii_requests += 1
        
        if is_sensitive:
            self.sensitive_requests += 1
        
        # Update timestamps
        now = datetime.now()
        if self.first_request is None:
            self.first_request = now
        self.last_request = now
    
    def get_success_rate(self) -> float:
        """Get success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary representation."""
        return {
            'model': self.model.value,
            'request_metrics': {
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'success_rate': self.get_success_rate()
            },
            'performance_metrics': {
                'total_tokens_processed': self.total_tokens_processed,
                'total_cost': self.total_cost,
                'avg_latency_ms': self.avg_latency_ms
            },
            'security_metrics': {
                'sensitive_requests': self.sensitive_requests,
                'pii_requests': self.pii_requests,
                'compliance_violations': self.compliance_violations
            },
            'time_range': {
                'first_request': self.first_request.isoformat() if self.first_request else None,
                'last_request': self.last_request.isoformat() if self.last_request else None
            }
        }


class BedrockModelRouter:
    """
    Intelligent router for Strands agent requests across different Bedrock models.
    
    Routes requests between Claude, Llama, Titan, and other Bedrock models based on
    data sensitivity, security policies, compliance requirements, and performance
    characteristics. Provides intelligent fallback and comprehensive audit trails.
    """
    
    def __init__(
        self,
        region: str = "us-east-1",
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ):
        """
        Initialize Bedrock model router.
        
        Args:
            region: AWS region for Bedrock services
            session_id: Current session ID
            agent_id: Current agent ID
        """
        self.region = region
        self.session_id = session_id or f"strands-router-{uuid.uuid4().hex[:8]}"
        self.agent_id = agent_id
        
        # Initialize AWS Bedrock client
        if AWS_AVAILABLE:
            try:
                self.bedrock_client = boto3.client('bedrock-runtime', region_name=region)
                self.aws_available = True
                logger.info(f"Bedrock client initialized for region: {region}")
            except (NoCredentialsError, Exception) as e:
                logger.warning(f"Bedrock client not available: {str(e)}")
                self.bedrock_client = None
                self.aws_available = False
        else:
            logger.warning("AWS SDK not available - using mock mode")
            self.bedrock_client = None
            self.aws_available = False
        
        # Initialize sensitive data handler for PII detection
        self.sensitive_data_handler = SensitiveDataHandler(
            region=region,
            session_id=self.session_id,
            agent_id=self.agent_id
        )
        
        # Initialize model policies and metrics
        self.model_policies = self._initialize_model_policies()
        self.usage_metrics: Dict[BedrockModel, ModelUsageMetrics] = {
            model: ModelUsageMetrics(model) for model in BedrockModel
        }
        
        # Routing history for audit trail
        self.routing_history: List[RoutingDecision] = []
        
        logger.info(f"BedrockModelRouter initialized with {len(self.model_policies)} model policies")
    
    def _initialize_model_policies(self) -> Dict[BedrockModel, ModelSecurityPolicy]:
        """Initialize security policies for all supported Bedrock models."""
        
        policies = {}
        
        # Claude 3 Haiku - Fast, cost-effective, moderate security
        policies[BedrockModel.CLAUDE_3_HAIKU] = ModelSecurityPolicy(
            model=BedrockModel.CLAUDE_3_HAIKU,
            max_security_tier=SecurityTier.CONFIDENTIAL,
            allowed_pii_types={PIIType.EMAIL, PIIType.PHONE, PIIType.NAME},
            data_residency_regions=["us-east-1", "us-west-2", "eu-west-1"],
            encryption_required=True,
            audit_level="standard",
            max_token_limit=200000,
            rate_limit_per_minute=100,
            hipaa_compliant=False,
            pci_dss_compliant=False,
            gdpr_compliant=True,
            sox_compliant=False,
            capabilities={
                ModelCapability.GENERAL_PURPOSE,
                ModelCapability.SUMMARIZATION,
                ModelCapability.CLASSIFICATION
            },
            cost_per_1k_tokens=0.00025,
            avg_latency_ms=800
        )
        
        # Claude 3 Sonnet - Balanced performance and security
        policies[BedrockModel.CLAUDE_3_SONNET] = ModelSecurityPolicy(
            model=BedrockModel.CLAUDE_3_SONNET,
            max_security_tier=SecurityTier.RESTRICTED,
            allowed_pii_types={
                PIIType.EMAIL, PIIType.PHONE, PIIType.NAME, PIIType.ADDRESS,
                PIIType.DATE_OF_BIRTH, PIIType.USERNAME
            },
            data_residency_regions=["us-east-1", "us-west-2", "eu-west-1"],
            encryption_required=True,
            audit_level="detailed",
            max_token_limit=200000,
            rate_limit_per_minute=80,
            hipaa_compliant=True,
            pci_dss_compliant=False,
            gdpr_compliant=True,
            sox_compliant=True,
            capabilities={
                ModelCapability.GENERAL_PURPOSE,
                ModelCapability.REASONING,
                ModelCapability.ANALYSIS,
                ModelCapability.CODE_GENERATION
            },
            cost_per_1k_tokens=0.003,
            avg_latency_ms=1200
        )
        
        # Claude 3 Opus - Maximum capability and security
        policies[BedrockModel.CLAUDE_3_OPUS] = ModelSecurityPolicy(
            model=BedrockModel.CLAUDE_3_OPUS,
            max_security_tier=SecurityTier.TOP_SECRET,
            allowed_pii_types=set(PIIType),  # All PII types allowed
            data_residency_regions=["us-east-1", "us-west-2"],
            encryption_required=True,
            audit_level="comprehensive",
            max_token_limit=200000,
            rate_limit_per_minute=40,
            hipaa_compliant=True,
            pci_dss_compliant=True,
            gdpr_compliant=True,
            sox_compliant=True,
            capabilities=set(ModelCapability),  # All capabilities
            cost_per_1k_tokens=0.015,
            avg_latency_ms=2000
        )
        
        # Claude 3.5 Sonnet - Latest model with enhanced capabilities
        policies[BedrockModel.CLAUDE_3_5_SONNET] = ModelSecurityPolicy(
            model=BedrockModel.CLAUDE_3_5_SONNET,
            max_security_tier=SecurityTier.RESTRICTED,
            allowed_pii_types={
                PIIType.EMAIL, PIIType.PHONE, PIIType.NAME, PIIType.ADDRESS,
                PIIType.DATE_OF_BIRTH, PIIType.USERNAME, PIIType.IP_ADDRESS
            },
            data_residency_regions=["us-east-1", "us-west-2", "eu-west-1"],
            encryption_required=True,
            audit_level="detailed",
            max_token_limit=200000,
            rate_limit_per_minute=60,
            hipaa_compliant=True,
            pci_dss_compliant=True,
            gdpr_compliant=True,
            sox_compliant=True,
            capabilities={
                ModelCapability.GENERAL_PURPOSE,
                ModelCapability.REASONING,
                ModelCapability.ANALYSIS,
                ModelCapability.CODE_GENERATION,
                ModelCapability.CREATIVE_WRITING
            },
            cost_per_1k_tokens=0.003,
            avg_latency_ms=1000
        )
        
        # Llama 3 8B - Cost-effective, limited security
        policies[BedrockModel.LLAMA_3_8B] = ModelSecurityPolicy(
            model=BedrockModel.LLAMA_3_8B,
            max_security_tier=SecurityTier.INTERNAL,
            allowed_pii_types={PIIType.EMAIL, PIIType.NAME},
            data_residency_regions=["us-east-1", "us-west-2"],
            encryption_required=True,
            audit_level="basic",
            max_token_limit=8192,
            rate_limit_per_minute=120,
            hipaa_compliant=False,
            pci_dss_compliant=False,
            gdpr_compliant=True,
            sox_compliant=False,
            capabilities={
                ModelCapability.GENERAL_PURPOSE,
                ModelCapability.CODE_GENERATION
            },
            cost_per_1k_tokens=0.0003,
            avg_latency_ms=600
        )
        
        # Llama 3 70B - Higher capability, moderate security
        policies[BedrockModel.LLAMA_3_70B] = ModelSecurityPolicy(
            model=BedrockModel.LLAMA_3_70B,
            max_security_tier=SecurityTier.CONFIDENTIAL,
            allowed_pii_types={
                PIIType.EMAIL, PIIType.PHONE, PIIType.NAME, PIIType.USERNAME
            },
            data_residency_regions=["us-east-1", "us-west-2"],
            encryption_required=True,
            audit_level="standard",
            max_token_limit=8192,
            rate_limit_per_minute=60,
            hipaa_compliant=False,
            pci_dss_compliant=False,
            gdpr_compliant=True,
            sox_compliant=False,
            capabilities={
                ModelCapability.GENERAL_PURPOSE,
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION,
                ModelCapability.ANALYSIS
            },
            cost_per_1k_tokens=0.00265,
            avg_latency_ms=1500
        )
        
        # Titan Text Express - AWS native, good security
        policies[BedrockModel.TITAN_TEXT_EXPRESS] = ModelSecurityPolicy(
            model=BedrockModel.TITAN_TEXT_EXPRESS,
            max_security_tier=SecurityTier.RESTRICTED,
            allowed_pii_types={
                PIIType.EMAIL, PIIType.PHONE, PIIType.NAME, PIIType.ADDRESS,
                PIIType.USERNAME
            },
            data_residency_regions=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
            encryption_required=True,
            audit_level="detailed",
            max_token_limit=8192,
            rate_limit_per_minute=100,
            hipaa_compliant=True,
            pci_dss_compliant=True,
            gdpr_compliant=True,
            sox_compliant=True,
            capabilities={
                ModelCapability.GENERAL_PURPOSE,
                ModelCapability.SUMMARIZATION,
                ModelCapability.CLASSIFICATION
            },
            cost_per_1k_tokens=0.0008,
            avg_latency_ms=900
        )
        
        # Titan Text Lite - Lightweight, basic security
        policies[BedrockModel.TITAN_TEXT_LITE] = ModelSecurityPolicy(
            model=BedrockModel.TITAN_TEXT_LITE,
            max_security_tier=SecurityTier.INTERNAL,
            allowed_pii_types={PIIType.EMAIL, PIIType.NAME},
            data_residency_regions=["us-east-1", "us-west-2", "eu-west-1"],
            encryption_required=True,
            audit_level="basic",
            max_token_limit=4096,
            rate_limit_per_minute=150,
            hipaa_compliant=False,
            pci_dss_compliant=False,
            gdpr_compliant=True,
            sox_compliant=False,
            capabilities={
                ModelCapability.GENERAL_PURPOSE,
                ModelCapability.SUMMARIZATION
            },
            cost_per_1k_tokens=0.0003,
            avg_latency_ms=500
        )
        
        return policies
    
    def route_request(self, request: RoutingRequest) -> RoutingDecision:
        """
        Route a request to the most appropriate Bedrock model.
        
        Args:
            request: Routing request with content and requirements
            
        Returns:
            Routing decision with selected model and fallbacks
        """
        
        logger.info(f"Routing request {request.request_id} for session {request.session_id}")
        
        # Analyze content for PII and security requirements
        analyzed_request = self._analyze_request_security(request)
        
        # Find compatible models based on security requirements
        compatible_models = self._find_compatible_models(analyzed_request)
        
        if not compatible_models:
            raise ToolExecutionError(
                f"No compatible models found for security tier {analyzed_request.security_tier.value}"
            )
        
        # Select best model based on requirements and performance
        selected_model = self._select_optimal_model(analyzed_request, compatible_models)
        
        # Generate fallback chain
        fallback_models = self._generate_fallback_chain(analyzed_request, compatible_models, selected_model)
        
        # Create routing decision
        decision = RoutingDecision(
            request_id=analyzed_request.request_id,
            selected_model=selected_model,
            fallback_models=fallback_models,
            routing_reason=self._generate_routing_reason(analyzed_request, selected_model),
            security_validated=True,
            security_tier=analyzed_request.security_tier,
            pii_types_detected=analyzed_request.detected_pii_types,
            compliance_validated=self._validate_compliance(analyzed_request, selected_model)
        )
        
        # Add to routing history for audit trail
        self.routing_history.append(decision)
        
        logger.info(f"✅ Routed request {request.request_id} to {selected_model.value}")
        logger.info(f"Security tier: {analyzed_request.security_tier.value}")
        logger.info(f"PII types detected: {[pii.value for pii in analyzed_request.detected_pii_types]}")
        
        return decision
    
    def _analyze_request_security(self, request: RoutingRequest) -> RoutingRequest:
        """
        Analyze request content for PII and determine security requirements.
        
        Args:
            request: Original routing request
            
        Returns:
            Updated request with security analysis
        """
        
        # Detect PII in content
        pii_detections = self.sensitive_data_handler.pii_detector.detect_pii(request.content)
        
        # Extract PII types
        detected_pii_types = {detection.pii_type for detection in pii_detections}
        
        # Determine security tier based on PII types
        security_tier = self._determine_security_tier(detected_pii_types)
        
        # Update request with analysis results
        request.detected_pii_types = detected_pii_types
        request.security_tier = security_tier
        
        # Add compliance requirements based on PII types
        if PIIType.SSN in detected_pii_types or PIIType.MEDICAL_ID in detected_pii_types:
            request.compliance_requirements.add("HIPAA")
        
        if PIIType.CREDIT_CARD in detected_pii_types or PIIType.BANK_ACCOUNT in detected_pii_types:
            request.compliance_requirements.add("PCI_DSS")
        
        if any(pii in detected_pii_types for pii in [PIIType.EMAIL, PIIType.NAME, PIIType.ADDRESS]):
            request.compliance_requirements.add("GDPR")
        
        logger.info(f"Security analysis complete for request {request.request_id}")
        logger.info(f"Detected PII types: {[pii.value for pii in detected_pii_types]}")
        logger.info(f"Security tier: {security_tier.value}")
        
        return request
    
    def _determine_security_tier(self, pii_types: Set[PIIType]) -> SecurityTier:
        """
        Determine security tier based on detected PII types.
        
        Args:
            pii_types: Set of detected PII types
            
        Returns:
            Appropriate security tier
        """
        
        if not pii_types:
            return SecurityTier.PUBLIC
        
        # High-risk PII types require top secret handling
        high_risk_pii = {
            PIIType.SSN, PIIType.CREDIT_CARD, PIIType.BANK_ACCOUNT,
            PIIType.MEDICAL_ID, PIIType.PASSPORT, PIIType.DRIVER_LICENSE
        }
        
        if pii_types.intersection(high_risk_pii):
            return SecurityTier.TOP_SECRET
        
        # Moderate-risk PII types require restricted handling
        moderate_risk_pii = {
            PIIType.ADDRESS, PIIType.DATE_OF_BIRTH, PIIType.PHONE
        }
        
        if pii_types.intersection(moderate_risk_pii):
            return SecurityTier.RESTRICTED
        
        # Low-risk PII types require confidential handling
        low_risk_pii = {
            PIIType.EMAIL, PIIType.NAME, PIIType.USERNAME
        }
        
        if pii_types.intersection(low_risk_pii):
            return SecurityTier.CONFIDENTIAL
        
        # Technical identifiers require internal handling
        return SecurityTier.INTERNAL
    
    def _find_compatible_models(self, request: RoutingRequest) -> List[BedrockModel]:
        """
        Find models compatible with the request's security requirements.
        
        Args:
            request: Routing request with security analysis
            
        Returns:
            List of compatible models
        """
        
        compatible_models = []
        
        for model, policy in self.model_policies.items():
            # Check security tier compatibility
            if not policy.can_handle_security_tier(request.security_tier):
                continue
            
            # Check PII type compatibility
            if not policy.can_handle_pii_types(request.detected_pii_types):
                continue
            
            # Check compliance requirements
            if not self._check_compliance_compatibility(request, policy):
                continue
            
            # Check regional requirements
            if request.required_regions:
                if not any(region in policy.data_residency_regions for region in request.required_regions):
                    continue
            
            # Check capability requirements
            if request.required_capabilities:
                if not request.required_capabilities.issubset(policy.capabilities):
                    continue
            
            compatible_models.append(model)
        
        logger.info(f"Found {len(compatible_models)} compatible models")
        return compatible_models
    
    def _check_compliance_compatibility(self, request: RoutingRequest, policy: ModelSecurityPolicy) -> bool:
        """
        Check if model policy meets compliance requirements.
        
        Args:
            request: Routing request
            policy: Model security policy
            
        Returns:
            True if compatible, False otherwise
        """
        
        for requirement in request.compliance_requirements:
            if requirement == "HIPAA" and not policy.hipaa_compliant:
                return False
            elif requirement == "PCI_DSS" and not policy.pci_dss_compliant:
                return False
            elif requirement == "GDPR" and not policy.gdpr_compliant:
                return False
            elif requirement == "SOX" and not policy.sox_compliant:
                return False
        
        return True
    
    def _select_optimal_model(self, request: RoutingRequest, compatible_models: List[BedrockModel]) -> BedrockModel:
        """
        Select the optimal model from compatible options.
        
        Args:
            request: Routing request
            compatible_models: List of compatible models
            
        Returns:
            Selected optimal model
        """
        
        # Check for preferred models first
        for preferred_model in request.preferred_models:
            if preferred_model in compatible_models:
                logger.info(f"Selected preferred model: {preferred_model.value}")
                return preferred_model
        
        # Score models based on multiple criteria
        model_scores = {}
        
        for model in compatible_models:
            policy = self.model_policies[model]
            metrics = self.usage_metrics[model]
            
            score = 0.0
            
            # Performance score (30% weight)
            if request.max_latency_ms:
                latency_score = max(0, 1 - (policy.avg_latency_ms / request.max_latency_ms))
                score += latency_score * 0.3
            else:
                # Prefer faster models
                score += (1 - min(1, policy.avg_latency_ms / 3000)) * 0.3
            
            # Cost score (25% weight)
            if request.max_cost_per_1k_tokens:
                cost_score = max(0, 1 - (policy.cost_per_1k_tokens / request.max_cost_per_1k_tokens))
                score += cost_score * 0.25
            else:
                # Prefer cheaper models
                score += (1 - min(1, policy.cost_per_1k_tokens / 0.02)) * 0.25
            
            # Reliability score (25% weight)
            success_rate = metrics.get_success_rate() / 100
            score += success_rate * 0.25
            
            # Capability match score (20% weight)
            if request.required_capabilities:
                capability_match = len(request.required_capabilities.intersection(policy.capabilities)) / len(request.required_capabilities)
                score += capability_match * 0.2
            else:
                # Prefer models with more capabilities
                score += (len(policy.capabilities) / len(ModelCapability)) * 0.2
            
            model_scores[model] = score
        
        # Select model with highest score
        selected_model = max(model_scores.keys(), key=lambda m: model_scores[m])
        
        logger.info(f"Selected optimal model: {selected_model.value} (score: {model_scores[selected_model]:.3f})")
        return selected_model
    
    def _generate_fallback_chain(
        self,
        request: RoutingRequest,
        compatible_models: List[BedrockModel],
        selected_model: BedrockModel
    ) -> List[BedrockModel]:
        """
        Generate fallback chain for the selected model.
        
        Args:
            request: Routing request
            compatible_models: List of compatible models
            selected_model: Primary selected model
            
        Returns:
            List of fallback models in priority order
        """
        
        # Remove selected model from compatible models
        fallback_candidates = [m for m in compatible_models if m != selected_model]
        
        if not fallback_candidates:
            return []
        
        # Sort by reliability and performance
        fallback_models = sorted(
            fallback_candidates,
            key=lambda m: (
                self.usage_metrics[m].get_success_rate(),
                -self.model_policies[m].avg_latency_ms,
                -self.model_policies[m].cost_per_1k_tokens
            ),
            reverse=True
        )
        
        # Return top 3 fallback options
        return fallback_models[:3]
    
    def _generate_routing_reason(self, request: RoutingRequest, selected_model: BedrockModel) -> str:
        """
        Generate human-readable routing reason.
        
        Args:
            request: Routing request
            selected_model: Selected model
            
        Returns:
            Routing reason string
        """
        
        policy = self.model_policies[selected_model]
        reasons = []
        
        # Security-based reasons
        if request.security_tier != SecurityTier.PUBLIC:
            reasons.append(f"Security tier {request.security_tier.value} compatibility")
        
        if request.detected_pii_types:
            reasons.append(f"PII handling capability for {len(request.detected_pii_types)} types")
        
        if request.compliance_requirements:
            reasons.append(f"Compliance with {', '.join(request.compliance_requirements)}")
        
        # Performance-based reasons
        if request.max_latency_ms and policy.avg_latency_ms <= request.max_latency_ms:
            reasons.append(f"Low latency ({policy.avg_latency_ms}ms)")
        
        if request.max_cost_per_1k_tokens and policy.cost_per_1k_tokens <= request.max_cost_per_1k_tokens:
            reasons.append(f"Cost efficiency (${policy.cost_per_1k_tokens}/1k tokens)")
        
        # Capability-based reasons
        if request.required_capabilities:
            matching_caps = request.required_capabilities.intersection(policy.capabilities)
            if matching_caps:
                reasons.append(f"Required capabilities: {', '.join(cap.value for cap in matching_caps)}")
        
        if not reasons:
            reasons.append("Best overall match for requirements")
        
        return "; ".join(reasons)
    
    def _validate_compliance(self, request: RoutingRequest, selected_model: BedrockModel) -> bool:
        """
        Validate that selected model meets compliance requirements.
        
        Args:
            request: Routing request
            selected_model: Selected model
            
        Returns:
            True if compliant, False otherwise
        """
        
        policy = self.model_policies[selected_model]
        
        for requirement in request.compliance_requirements:
            if requirement == "HIPAA" and not policy.hipaa_compliant:
                return False
            elif requirement == "PCI_DSS" and not policy.pci_dss_compliant:
                return False
            elif requirement == "GDPR" and not policy.gdpr_compliant:
                return False
            elif requirement == "SOX" and not policy.sox_compliant:
                return False
        
        return True
    
    def execute_with_fallback(
        self,
        request: RoutingRequest,
        prompt: str,
        max_tokens: int = 1000
    ) -> ToolResult:
        """
        Execute request with automatic fallback on failure.
        
        Args:
            request: Routing request
            prompt: Prompt to send to model
            max_tokens: Maximum tokens to generate
            
        Returns:
            Tool result with response or error
        """
        
        # Get routing decision
        decision = self.route_request(request)
        
        # Try primary model first
        models_to_try = [decision.selected_model] + decision.fallback_models
        
        for i, model in enumerate(models_to_try):
            try:
                logger.info(f"Attempting execution with {model.value} (attempt {i+1})")
                
                # Execute with model
                result = self._execute_with_model(model, prompt, max_tokens, request)
                
                if result.success:
                    # Update metrics
                    self.usage_metrics[model].add_request(
                        success=True,
                        tokens=max_tokens,  # Approximate
                        cost=self.model_policies[model].cost_per_1k_tokens * (max_tokens / 1000),
                        latency_ms=self.model_policies[model].avg_latency_ms,
                        has_pii=bool(request.detected_pii_types),
                        is_sensitive=request.security_tier != SecurityTier.PUBLIC
                    )
                    
                    logger.info(f"✅ Successfully executed with {model.value}")
                    return result
                
            except Exception as e:
                logger.warning(f"Failed to execute with {model.value}: {str(e)}")
                
                # Update metrics
                self.usage_metrics[model].add_request(
                    success=False,
                    tokens=0,
                    cost=0,
                    latency_ms=0,
                    has_pii=bool(request.detected_pii_types),
                    is_sensitive=request.security_tier != SecurityTier.PUBLIC
                )
                
                # Continue to next model
                continue
        
        # All models failed
        error_msg = f"All models failed for request {request.request_id}"
        logger.error(error_msg)
        
        return ToolResult(
            success=False,
            error=error_msg,
            metadata={
                'request_id': request.request_id,
                'models_attempted': [m.value for m in models_to_try],
                'routing_decision': decision.to_dict()
            }
        )
    
    def _execute_with_model(
        self,
        model: BedrockModel,
        prompt: str,
        max_tokens: int,
        request: RoutingRequest
    ) -> ToolResult:
        """
        Execute request with specific Bedrock model.
        
        Args:
            model: Bedrock model to use
            prompt: Prompt to send
            max_tokens: Maximum tokens to generate
            request: Original request for context
            
        Returns:
            Tool result with response
        """
        
        if not self.aws_available:
            # Mock execution for testing
            return ToolResult(
                success=True,
                data={
                    'response': f"Mock response from {model.value}",
                    'model': model.value,
                    'tokens_used': max_tokens // 2,
                    'security_tier': request.security_tier.value
                },
                metadata={
                    'model': model.value,
                    'request_id': request.request_id,
                    'mock_execution': True
                }
            )
        
        try:
            # Prepare request body based on model type
            if model.value.startswith('anthropic.claude'):
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}]
                }
            elif model.value.startswith('meta.llama'):
                body = {
                    "prompt": prompt,
                    "max_gen_len": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            elif model.value.startswith('amazon.titan'):
                body = {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": max_tokens,
                        "temperature": 0.7,
                        "topP": 0.9
                    }
                }
            else:
                # Generic format
                body = {
                    "prompt": prompt,
                    "max_tokens": max_tokens
                }
            
            # Invoke model
            response = self.bedrock_client.invoke_model(
                modelId=model.value,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json"
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            # Extract generated text based on model type
            if model.value.startswith('anthropic.claude'):
                generated_text = response_body['content'][0]['text']
            elif model.value.startswith('meta.llama'):
                generated_text = response_body['generation']
            elif model.value.startswith('amazon.titan'):
                generated_text = response_body['results'][0]['outputText']
            else:
                generated_text = str(response_body)
            
            return ToolResult(
                success=True,
                data={
                    'response': generated_text,
                    'model': model.value,
                    'tokens_used': len(generated_text.split()),  # Approximate
                    'security_tier': request.security_tier.value
                },
                metadata={
                    'model': model.value,
                    'request_id': request.request_id,
                    'response_metadata': response.get('ResponseMetadata', {})
                }
            )
            
        except Exception as e:
            raise ToolExecutionError(f"Model execution failed: {str(e)}")
    
    def get_routing_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get routing history for audit trail.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of routing decisions
        """
        
        history = self.routing_history
        
        if limit:
            history = history[-limit:]
        
        return [decision.to_dict() for decision in history]
    
    def get_usage_metrics(self, model: Optional[BedrockModel] = None) -> Dict[str, Any]:
        """
        Get usage metrics for models.
        
        Args:
            model: Specific model to get metrics for (None for all)
            
        Returns:
            Usage metrics dictionary
        """
        
        if model:
            return self.usage_metrics[model].to_dict()
        
        return {
            'overall_metrics': {
                'total_requests': sum(m.total_requests for m in self.usage_metrics.values()),
                'total_cost': sum(m.total_cost for m in self.usage_metrics.values()),
                'avg_success_rate': sum(m.get_success_rate() for m in self.usage_metrics.values()) / len(self.usage_metrics)
            },
            'model_metrics': {
                model.value: metrics.to_dict() 
                for model, metrics in self.usage_metrics.items()
            }
        }
    
    def get_model_policies(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all model security policies.
        
        Returns:
            Dictionary of model policies
        """
        
        return {
            model.value: policy.to_dict()
            for model, policy in self.model_policies.items()
        }
    
    def update_model_policy(self, model: BedrockModel, policy: ModelSecurityPolicy) -> None:
        """
        Update security policy for a model.
        
        Args:
            model: Model to update
            policy: New security policy
        """
        
        self.model_policies[model] = policy
        logger.info(f"Updated security policy for {model.value}")
    
    def create_routing_request(
        self,
        content: str,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        **kwargs
    ) -> RoutingRequest:
        """
        Create a routing request with default values.
        
        Args:
            content: Content to route
            session_id: Session ID (uses router's if not provided)
            agent_id: Agent ID (uses router's if not provided)
            **kwargs: Additional routing parameters
            
        Returns:
            Configured routing request
        """
        
        return RoutingRequest(
            request_id=f"req-{uuid.uuid4().hex[:8]}",
            content=content,
            session_id=session_id or self.session_id,
            agent_id=agent_id or self.agent_id,
            **kwargs
        )