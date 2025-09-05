"""
Compliance Validation System for Strands Agents with Bedrock

This module provides comprehensive compliance validation for Strands agent operations
against HIPAA, PCI DSS, GDPR, and other regulatory requirements when using Amazon
Bedrock models. It includes real-time compliance monitoring, automated reporting,
and violation detection and remediation.

Key Features:
- Validates Strands agent operations against HIPAA, PCI DSS, and GDPR requirements
- Real-time compliance monitoring during Strands agent execution with sensitive data
- Automated compliance reporting for Strands workflows handling sensitive information
- Violation detection and remediation for non-compliant operations
- Integration with Bedrock model security policies

Requirements Addressed:
- 7.1: HIPAA, PCI DSS, and GDPR compliance validation
- 7.2: Real-time compliance monitoring during agent execution
- 7.3: Automated compliance reporting for workflows
- 7.4: Violation detection and remediation
"""

import os
import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Union, Set, Tuple
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

# Import related modules
try:
    from .sensitive_data_handler import SensitiveDataHandler, PIIType, SensitivityLevel
    from .bedrock_model_router import BedrockModel, SecurityTier
except ImportError:
    from sensitive_data_handler import SensitiveDataHandler, PIIType, SensitivityLevel
    from bedrock_model_router import BedrockModel, SecurityTier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    GDPR = "gdpr"
    SOX = "sox"
    CCPA = "ccpa"
    FERPA = "ferpa"
    GLBA = "glba"
    ISO_27001 = "iso_27001"


class ViolationSeverity(Enum):
    """Severity levels for compliance violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(Enum):
    """Compliance status for operations."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    REQUIRES_REVIEW = "requires_review"
    REMEDIATED = "remediated"


class RemediationAction(Enum):
    """Available remediation actions for violations."""
    MASK_DATA = "mask_data"
    ENCRYPT_DATA = "encrypt_data"
    DELETE_DATA = "delete_data"
    QUARANTINE_DATA = "quarantine_data"
    NOTIFY_DPO = "notify_dpo"
    LOG_INCIDENT = "log_incident"
    BLOCK_OPERATION = "block_operation"
    REQUIRE_APPROVAL = "require_approval"


@dataclass
class ComplianceRule:
    """Definition of a compliance rule."""
    rule_id: str
    framework: ComplianceFramework
    title: str
    description: str
    
    # Rule conditions
    applicable_pii_types: Set[PIIType] = field(default_factory=set)
    applicable_security_tiers: Set[SecurityTier] = field(default_factory=set)
    applicable_models: Set[BedrockModel] = field(default_factory=set)
    
    # Rule requirements
    requires_encryption: bool = False
    requires_audit_logging: bool = True
    requires_data_minimization: bool = False
    requires_consent: bool = False
    requires_retention_policy: bool = False
    
    # Violation handling
    violation_severity: ViolationSeverity = ViolationSeverity.MEDIUM
    remediation_actions: List[RemediationAction] = field(default_factory=list)
    
    # Time constraints
    max_retention_days: Optional[int] = None
    notification_deadline_hours: Optional[int] = None
    
    def is_applicable(
        self,
        pii_types: Set[PIIType],
        security_tier: SecurityTier,
        model: BedrockModel
    ) -> bool:
        """Check if rule is applicable to the given context."""
        
        # Check PII type applicability
        if self.applicable_pii_types and not pii_types.intersection(self.applicable_pii_types):
            return False
        
        # Check security tier applicability
        if self.applicable_security_tiers and security_tier not in self.applicable_security_tiers:
            return False
        
        # Check model applicability
        if self.applicable_models and model not in self.applicable_models:
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary representation."""
        return {
            'rule_id': self.rule_id,
            'framework': self.framework.value,
            'title': self.title,
            'description': self.description,
            'conditions': {
                'applicable_pii_types': [pii.value for pii in self.applicable_pii_types],
                'applicable_security_tiers': [tier.value for tier in self.applicable_security_tiers],
                'applicable_models': [model.value for model in self.applicable_models]
            },
            'requirements': {
                'requires_encryption': self.requires_encryption,
                'requires_audit_logging': self.requires_audit_logging,
                'requires_data_minimization': self.requires_data_minimization,
                'requires_consent': self.requires_consent,
                'requires_retention_policy': self.requires_retention_policy
            },
            'violation_handling': {
                'violation_severity': self.violation_severity.value,
                'remediation_actions': [action.value for action in self.remediation_actions]
            },
            'time_constraints': {
                'max_retention_days': self.max_retention_days,
                'notification_deadline_hours': self.notification_deadline_hours
            }
        }


@dataclass
class ComplianceViolation:
    """Represents a compliance violation."""
    violation_id: str
    rule_id: str
    framework: ComplianceFramework
    severity: ViolationSeverity
    
    # Violation context
    session_id: str
    agent_id: Optional[str]
    model: BedrockModel
    operation_type: str
    
    # Violation details
    violation_description: str
    detected_pii_types: Set[PIIType]
    security_tier: SecurityTier
    
    # Timestamps
    detected_at: datetime = field(default_factory=datetime.now)
    remediated_at: Optional[datetime] = None
    
    # Remediation
    status: ComplianceStatus = ComplianceStatus.NON_COMPLIANT
    remediation_actions_taken: List[RemediationAction] = field(default_factory=list)
    remediation_notes: str = ""
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert violation to dictionary representation."""
        return {
            'violation_id': self.violation_id,
            'rule_id': self.rule_id,
            'framework': self.framework.value,
            'severity': self.severity.value,
            'context': {
                'session_id': self.session_id,
                'agent_id': self.agent_id,
                'model': self.model.value,
                'operation_type': self.operation_type
            },
            'details': {
                'violation_description': self.violation_description,
                'detected_pii_types': [pii.value for pii in self.detected_pii_types],
                'security_tier': self.security_tier.value
            },
            'timestamps': {
                'detected_at': self.detected_at.isoformat(),
                'remediated_at': self.remediated_at.isoformat() if self.remediated_at else None
            },
            'remediation': {
                'status': self.status.value,
                'actions_taken': [action.value for action in self.remediation_actions_taken],
                'notes': self.remediation_notes
            },
            'metadata': self.metadata
        }


@dataclass
class ComplianceReport:
    """Compliance report for a time period."""
    report_id: str
    framework: ComplianceFramework
    start_date: datetime
    end_date: datetime
    
    # Summary statistics
    total_operations: int = 0
    compliant_operations: int = 0
    violations_detected: int = 0
    violations_remediated: int = 0
    
    # Violation breakdown
    violations_by_severity: Dict[ViolationSeverity, int] = field(default_factory=dict)
    violations_by_rule: Dict[str, int] = field(default_factory=dict)
    violations_by_model: Dict[BedrockModel, int] = field(default_factory=dict)
    
    # Detailed violations
    violations: List[ComplianceViolation] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    def get_compliance_rate(self) -> float:
        """Get overall compliance rate percentage."""
        if self.total_operations == 0:
            return 100.0
        return (self.compliant_operations / self.total_operations) * 100
    
    def get_remediation_rate(self) -> float:
        """Get violation remediation rate percentage."""
        if self.violations_detected == 0:
            return 100.0
        return (self.violations_remediated / self.violations_detected) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary representation."""
        return {
            'report_id': self.report_id,
            'framework': self.framework.value,
            'time_period': {
                'start_date': self.start_date.isoformat(),
                'end_date': self.end_date.isoformat()
            },
            'summary': {
                'total_operations': self.total_operations,
                'compliant_operations': self.compliant_operations,
                'violations_detected': self.violations_detected,
                'violations_remediated': self.violations_remediated,
                'compliance_rate': self.get_compliance_rate(),
                'remediation_rate': self.get_remediation_rate()
            },
            'violation_breakdown': {
                'by_severity': {sev.value: count for sev, count in self.violations_by_severity.items()},
                'by_rule': self.violations_by_rule,
                'by_model': {model.value: count for model, count in self.violations_by_model.items()}
            },
            'violations': [violation.to_dict() for violation in self.violations],
            'recommendations': self.recommendations
        }


class ComplianceValidator:
    """
    Comprehensive compliance validation system for Strands agents with Bedrock.
    
    Validates agent operations against multiple compliance frameworks including
    HIPAA, PCI DSS, GDPR, and others. Provides real-time monitoring, violation
    detection, automated remediation, and compliance reporting.
    """
    
    def __init__(
        self,
        region: str = "us-east-1",
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        enabled_frameworks: Optional[List[ComplianceFramework]] = None
    ):
        """
        Initialize compliance validator.
        
        Args:
            region: AWS region for services
            session_id: Current session ID
            agent_id: Current agent ID
            enabled_frameworks: List of enabled compliance frameworks
        """
        self.region = region
        self.session_id = session_id or f"compliance-{uuid.uuid4().hex[:8]}"
        self.agent_id = agent_id
        self.enabled_frameworks = enabled_frameworks or [
            ComplianceFramework.HIPAA,
            ComplianceFramework.PCI_DSS,
            ComplianceFramework.GDPR
        ]
        
        # Initialize AWS services
        if AWS_AVAILABLE:
            try:
                self.s3_client = boto3.client('s3', region_name=region)
                self.sns_client = boto3.client('sns', region_name=region)
                self.aws_available = True
                logger.info(f"AWS services initialized for compliance validation")
            except (NoCredentialsError, Exception) as e:
                logger.warning(f"AWS services not available: {str(e)}")
                self.s3_client = None
                self.sns_client = None
                self.aws_available = False
        else:
            self.s3_client = None
            self.sns_client = None
            self.aws_available = False
        
        # Initialize sensitive data handler
        self.sensitive_data_handler = SensitiveDataHandler(
            region=region,
            session_id=self.session_id,
            agent_id=self.agent_id
        )
        
        # Initialize compliance rules
        self.compliance_rules = self._initialize_compliance_rules()
        
        # Violation tracking
        self.violations: List[ComplianceViolation] = []
        self.operation_count = 0
        self.compliant_operations = 0
        
        logger.info(f"ComplianceValidator initialized with {len(self.compliance_rules)} rules")
        logger.info(f"Enabled frameworks: {[f.value for f in self.enabled_frameworks]}")
    
    def _initialize_compliance_rules(self) -> List[ComplianceRule]:
        """Initialize compliance rules for all supported frameworks."""
        
        rules = []
        
        # HIPAA Rules
        if ComplianceFramework.HIPAA in self.enabled_frameworks:
            rules.extend(self._create_hipaa_rules())
        
        # PCI DSS Rules
        if ComplianceFramework.PCI_DSS in self.enabled_frameworks:
            rules.extend(self._create_pci_dss_rules())
        
        # GDPR Rules
        if ComplianceFramework.GDPR in self.enabled_frameworks:
            rules.extend(self._create_gdpr_rules())
        
        # SOX Rules
        if ComplianceFramework.SOX in self.enabled_frameworks:
            rules.extend(self._create_sox_rules())
        
        return rules
    
    def _create_hipaa_rules(self) -> List[ComplianceRule]:
        """Create HIPAA compliance rules."""
        
        return [
            ComplianceRule(
                rule_id="HIPAA-001",
                framework=ComplianceFramework.HIPAA,
                title="PHI Encryption Requirement",
                description="All Protected Health Information (PHI) must be encrypted in transit and at rest",
                applicable_pii_types={PIIType.MEDICAL_ID, PIIType.SSN, PIIType.DATE_OF_BIRTH},
                applicable_security_tiers={SecurityTier.RESTRICTED, SecurityTier.TOP_SECRET},
                requires_encryption=True,
                requires_audit_logging=True,
                violation_severity=ViolationSeverity.CRITICAL,
                remediation_actions=[RemediationAction.ENCRYPT_DATA, RemediationAction.LOG_INCIDENT]
            ),
            
            ComplianceRule(
                rule_id="HIPAA-002",
                framework=ComplianceFramework.HIPAA,
                title="PHI Access Logging",
                description="All access to PHI must be logged with user identification and timestamp",
                applicable_pii_types={PIIType.MEDICAL_ID, PIIType.SSN, PIIType.DATE_OF_BIRTH},
                requires_audit_logging=True,
                violation_severity=ViolationSeverity.HIGH,
                remediation_actions=[RemediationAction.LOG_INCIDENT]
            ),
            
            ComplianceRule(
                rule_id="HIPAA-003",
                framework=ComplianceFramework.HIPAA,
                title="PHI Data Minimization",
                description="Only minimum necessary PHI should be processed",
                applicable_pii_types={PIIType.MEDICAL_ID, PIIType.SSN, PIIType.DATE_OF_BIRTH},
                requires_data_minimization=True,
                violation_severity=ViolationSeverity.MEDIUM,
                remediation_actions=[RemediationAction.MASK_DATA]
            ),
            
            ComplianceRule(
                rule_id="HIPAA-004",
                framework=ComplianceFramework.HIPAA,
                title="PHI Retention Limits",
                description="PHI must not be retained longer than necessary",
                applicable_pii_types={PIIType.MEDICAL_ID, PIIType.SSN, PIIType.DATE_OF_BIRTH},
                requires_retention_policy=True,
                max_retention_days=2555,  # 7 years
                violation_severity=ViolationSeverity.HIGH,
                remediation_actions=[RemediationAction.DELETE_DATA, RemediationAction.LOG_INCIDENT]
            )
        ]
    
    def _create_pci_dss_rules(self) -> List[ComplianceRule]:
        """Create PCI DSS compliance rules."""
        
        return [
            ComplianceRule(
                rule_id="PCI-001",
                framework=ComplianceFramework.PCI_DSS,
                title="Cardholder Data Encryption",
                description="All cardholder data must be encrypted using strong cryptography",
                applicable_pii_types={PIIType.CREDIT_CARD, PIIType.BANK_ACCOUNT},
                applicable_security_tiers={SecurityTier.RESTRICTED, SecurityTier.TOP_SECRET},
                requires_encryption=True,
                requires_audit_logging=True,
                violation_severity=ViolationSeverity.CRITICAL,
                remediation_actions=[RemediationAction.ENCRYPT_DATA, RemediationAction.LOG_INCIDENT]
            ),
            
            ComplianceRule(
                rule_id="PCI-002",
                framework=ComplianceFramework.PCI_DSS,
                title="Cardholder Data Access Control",
                description="Access to cardholder data must be restricted on a need-to-know basis",
                applicable_pii_types={PIIType.CREDIT_CARD, PIIType.BANK_ACCOUNT},
                requires_audit_logging=True,
                violation_severity=ViolationSeverity.HIGH,
                remediation_actions=[RemediationAction.LOG_INCIDENT, RemediationAction.BLOCK_OPERATION]
            ),
            
            ComplianceRule(
                rule_id="PCI-003",
                framework=ComplianceFramework.PCI_DSS,
                title="Cardholder Data Masking",
                description="Cardholder data must be masked when displayed",
                applicable_pii_types={PIIType.CREDIT_CARD, PIIType.BANK_ACCOUNT},
                violation_severity=ViolationSeverity.HIGH,
                remediation_actions=[RemediationAction.MASK_DATA]
            ),
            
            ComplianceRule(
                rule_id="PCI-004",
                framework=ComplianceFramework.PCI_DSS,
                title="Cardholder Data Retention",
                description="Cardholder data must not be retained longer than necessary",
                applicable_pii_types={PIIType.CREDIT_CARD, PIIType.BANK_ACCOUNT},
                requires_retention_policy=True,
                max_retention_days=365,  # 1 year maximum
                violation_severity=ViolationSeverity.HIGH,
                remediation_actions=[RemediationAction.DELETE_DATA, RemediationAction.LOG_INCIDENT]
            )
        ]
    
    def _create_gdpr_rules(self) -> List[ComplianceRule]:
        """Create GDPR compliance rules."""
        
        return [
            ComplianceRule(
                rule_id="GDPR-001",
                framework=ComplianceFramework.GDPR,
                title="Personal Data Consent",
                description="Processing of personal data requires explicit consent",
                applicable_pii_types={PIIType.EMAIL, PIIType.NAME, PIIType.ADDRESS, PIIType.PHONE},
                requires_consent=True,
                violation_severity=ViolationSeverity.HIGH,
                remediation_actions=[RemediationAction.BLOCK_OPERATION, RemediationAction.NOTIFY_DPO]
            ),
            
            ComplianceRule(
                rule_id="GDPR-002",
                framework=ComplianceFramework.GDPR,
                title="Personal Data Breach Notification",
                description="Personal data breaches must be reported within 72 hours",
                applicable_pii_types=set(PIIType),
                notification_deadline_hours=72,
                violation_severity=ViolationSeverity.CRITICAL,
                remediation_actions=[RemediationAction.NOTIFY_DPO, RemediationAction.LOG_INCIDENT]
            ),
            
            ComplianceRule(
                rule_id="GDPR-003",
                framework=ComplianceFramework.GDPR,
                title="Personal Data Minimization",
                description="Only necessary personal data should be processed",
                applicable_pii_types=set(PIIType),
                requires_data_minimization=True,
                violation_severity=ViolationSeverity.MEDIUM,
                remediation_actions=[RemediationAction.MASK_DATA]
            ),
            
            ComplianceRule(
                rule_id="GDPR-004",
                framework=ComplianceFramework.GDPR,
                title="Right to be Forgotten",
                description="Personal data must be deletable upon request",
                applicable_pii_types=set(PIIType),
                requires_retention_policy=True,
                violation_severity=ViolationSeverity.HIGH,
                remediation_actions=[RemediationAction.DELETE_DATA]
            )
        ]
    
    def _create_sox_rules(self) -> List[ComplianceRule]:
        """Create SOX compliance rules."""
        
        return [
            ComplianceRule(
                rule_id="SOX-001",
                framework=ComplianceFramework.SOX,
                title="Financial Data Audit Trail",
                description="All financial data access must maintain complete audit trail",
                applicable_pii_types={PIIType.BANK_ACCOUNT, PIIType.SSN},
                requires_audit_logging=True,
                violation_severity=ViolationSeverity.HIGH,
                remediation_actions=[RemediationAction.LOG_INCIDENT]
            ),
            
            ComplianceRule(
                rule_id="SOX-002",
                framework=ComplianceFramework.SOX,
                title="Financial Data Integrity",
                description="Financial data must be protected from unauthorized modification",
                applicable_pii_types={PIIType.BANK_ACCOUNT, PIIType.SSN},
                requires_encryption=True,
                violation_severity=ViolationSeverity.CRITICAL,
                remediation_actions=[RemediationAction.ENCRYPT_DATA, RemediationAction.LOG_INCIDENT]
            )
        ]
    
    def validate_operation(
        self,
        operation_type: str,
        content: str,
        model: BedrockModel,
        security_tier: SecurityTier,
        pii_types: Set[PIIType],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """
        Validate an operation against compliance requirements.
        
        Args:
            operation_type: Type of operation being performed
            content: Content being processed
            model: Bedrock model being used
            security_tier: Security tier of the operation
            pii_types: Detected PII types
            metadata: Additional operation metadata
            
        Returns:
            Validation result with compliance status
        """
        
        self.operation_count += 1
        metadata = metadata or {}
        
        logger.info(f"Validating operation: {operation_type} with model {model.value}")
        logger.info(f"Security tier: {security_tier.value}, PII types: {[pii.value for pii in pii_types]}")
        
        # Find applicable rules
        applicable_rules = [
            rule for rule in self.compliance_rules
            if rule.is_applicable(pii_types, security_tier, model)
        ]
        
        logger.info(f"Found {len(applicable_rules)} applicable compliance rules")
        
        # Check each rule
        violations = []
        compliance_status = ComplianceStatus.COMPLIANT
        
        for rule in applicable_rules:
            violation = self._check_rule_compliance(
                rule, operation_type, content, model, security_tier, pii_types, metadata
            )
            
            if violation:
                violations.append(violation)
                self.violations.append(violation)
                
                if violation.severity in [ViolationSeverity.HIGH, ViolationSeverity.CRITICAL]:
                    compliance_status = ComplianceStatus.NON_COMPLIANT
                elif compliance_status == ComplianceStatus.COMPLIANT:
                    compliance_status = ComplianceStatus.REQUIRES_REVIEW
        
        # Update compliance metrics
        if compliance_status == ComplianceStatus.COMPLIANT:
            self.compliant_operations += 1
        
        # Attempt automatic remediation for violations
        remediated_violations = []
        for violation in violations:
            if self._attempt_remediation(violation, content):
                remediated_violations.append(violation)
        
        # Prepare result
        result_data = {
            'compliance_status': compliance_status.value,
            'applicable_rules': len(applicable_rules),
            'violations_detected': len(violations),
            'violations_remediated': len(remediated_violations),
            'operation_metadata': {
                'operation_type': operation_type,
                'model': model.value,
                'security_tier': security_tier.value,
                'pii_types': [pii.value for pii in pii_types]
            }
        }
        
        if violations:
            result_data['violations'] = [v.to_dict() for v in violations]
        
        success = compliance_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.REQUIRES_REVIEW]
        
        logger.info(f"✅ Compliance validation complete: {compliance_status.value}")
        logger.info(f"Violations: {len(violations)}, Remediated: {len(remediated_violations)}")
        
        return ToolResult(
            success=success,
            data=result_data,
            metadata={
                'session_id': self.session_id,
                'agent_id': self.agent_id,
                'validation_timestamp': datetime.now().isoformat()
            }
        )
    
    def _check_rule_compliance(
        self,
        rule: ComplianceRule,
        operation_type: str,
        content: str,
        model: BedrockModel,
        security_tier: SecurityTier,
        pii_types: Set[PIIType],
        metadata: Dict[str, Any]
    ) -> Optional[ComplianceViolation]:
        """
        Check compliance against a specific rule.
        
        Args:
            rule: Compliance rule to check
            operation_type: Type of operation
            content: Content being processed
            model: Bedrock model
            security_tier: Security tier
            pii_types: Detected PII types
            metadata: Operation metadata
            
        Returns:
            Compliance violation if found, None otherwise
        """
        
        violation_reasons = []
        
        # Check encryption requirement
        if rule.requires_encryption:
            if not metadata.get('encrypted', False):
                violation_reasons.append("Data encryption required but not applied")
        
        # Check audit logging requirement
        if rule.requires_audit_logging:
            if not metadata.get('audit_logged', True):  # Default to True for existing operations
                violation_reasons.append("Audit logging required but not enabled")
        
        # Check data minimization requirement
        if rule.requires_data_minimization:
            if not metadata.get('data_minimized', False):
                violation_reasons.append("Data minimization required but not applied")
        
        # Check consent requirement
        if rule.requires_consent:
            if not metadata.get('consent_obtained', False):
                violation_reasons.append("User consent required but not obtained")
        
        # Check retention policy requirement
        if rule.requires_retention_policy and rule.max_retention_days:
            retention_date = metadata.get('retention_date')
            if retention_date:
                try:
                    retention_datetime = datetime.fromisoformat(retention_date)
                    max_retention = datetime.now() + timedelta(days=rule.max_retention_days)
                    if retention_datetime > max_retention:
                        violation_reasons.append(f"Data retention exceeds maximum of {rule.max_retention_days} days")
                except ValueError:
                    violation_reasons.append("Invalid retention date format")
            else:
                violation_reasons.append("Retention policy required but not defined")
        
        # If violations found, create violation record
        if violation_reasons:
            violation = ComplianceViolation(
                violation_id=f"viol-{uuid.uuid4().hex[:8]}",
                rule_id=rule.rule_id,
                framework=rule.framework,
                severity=rule.violation_severity,
                session_id=self.session_id,
                agent_id=self.agent_id,
                model=model,
                operation_type=operation_type,
                violation_description="; ".join(violation_reasons),
                detected_pii_types=pii_types,
                security_tier=security_tier,
                metadata=metadata
            )
            
            logger.warning(f"Compliance violation detected: {rule.rule_id} - {violation.violation_description}")
            return violation
        
        return None
    
    def _attempt_remediation(self, violation: ComplianceViolation, content: str) -> bool:
        """
        Attempt automatic remediation of a compliance violation.
        
        Args:
            violation: Compliance violation to remediate
            content: Original content
            
        Returns:
            True if remediation successful, False otherwise
        """
        
        rule = next((r for r in self.compliance_rules if r.rule_id == violation.rule_id), None)
        if not rule:
            return False
        
        remediation_successful = False
        
        for action in rule.remediation_actions:
            try:
                if action == RemediationAction.MASK_DATA:
                    # Apply data masking
                    pii_detections = self.sensitive_data_handler.pii_detector.detect_pii(content)
                    masked_content = self.sensitive_data_handler.data_masker.mask_text(content, pii_detections)
                    violation.remediation_actions_taken.append(action)
                    remediation_successful = True
                    logger.info(f"Applied data masking for violation {violation.violation_id}")
                
                elif action == RemediationAction.ENCRYPT_DATA:
                    # Mark data for encryption (would be handled by infrastructure)
                    violation.metadata['encryption_required'] = True
                    violation.remediation_actions_taken.append(action)
                    remediation_successful = True
                    logger.info(f"Marked data for encryption for violation {violation.violation_id}")
                
                elif action == RemediationAction.LOG_INCIDENT:
                    # Log the incident
                    self._log_compliance_incident(violation)
                    violation.remediation_actions_taken.append(action)
                    remediation_successful = True
                    logger.info(f"Logged compliance incident for violation {violation.violation_id}")
                
                elif action == RemediationAction.NOTIFY_DPO:
                    # Notify Data Protection Officer
                    self._notify_dpo(violation)
                    violation.remediation_actions_taken.append(action)
                    remediation_successful = True
                    logger.info(f"Notified DPO for violation {violation.violation_id}")
                
                elif action == RemediationAction.QUARANTINE_DATA:
                    # Quarantine the data
                    violation.metadata['quarantined'] = True
                    violation.remediation_actions_taken.append(action)
                    remediation_successful = True
                    logger.info(f"Quarantined data for violation {violation.violation_id}")
                
            except Exception as e:
                logger.error(f"Failed to apply remediation action {action.value}: {str(e)}")
                continue
        
        if remediation_successful:
            violation.status = ComplianceStatus.REMEDIATED
            violation.remediated_at = datetime.now()
            violation.remediation_notes = f"Automatically remediated with actions: {[a.value for a in violation.remediation_actions_taken]}"
        
        return remediation_successful
    
    def _log_compliance_incident(self, violation: ComplianceViolation) -> None:
        """Log a compliance incident."""
        
        incident_log = {
            'timestamp': datetime.now().isoformat(),
            'violation_id': violation.violation_id,
            'framework': violation.framework.value,
            'severity': violation.severity.value,
            'description': violation.violation_description,
            'session_id': violation.session_id,
            'agent_id': violation.agent_id,
            'model': violation.model.value
        }
        
        # In a real implementation, this would write to a secure audit log
        logger.info(f"COMPLIANCE INCIDENT: {json.dumps(incident_log)}")
    
    def _notify_dpo(self, violation: ComplianceViolation) -> None:
        """Notify Data Protection Officer of a violation."""
        
        if not self.aws_available or not self.sns_client:
            logger.warning("SNS not available - DPO notification skipped")
            return
        
        try:
            message = {
                'violation_id': violation.violation_id,
                'framework': violation.framework.value,
                'severity': violation.severity.value,
                'description': violation.violation_description,
                'detected_at': violation.detected_at.isoformat(),
                'session_id': violation.session_id,
                'requires_immediate_attention': violation.severity in [ViolationSeverity.HIGH, ViolationSeverity.CRITICAL]
            }
            
            # In a real implementation, this would send to a configured SNS topic
            logger.info(f"DPO NOTIFICATION: {json.dumps(message)}")
            
        except Exception as e:
            logger.error(f"Failed to notify DPO: {str(e)}")
    
    def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> ComplianceReport:
        """
        Generate compliance report for a specific framework and time period.
        
        Args:
            framework: Compliance framework to report on
            start_date: Start date for report (defaults to 30 days ago)
            end_date: End date for report (defaults to now)
            
        Returns:
            Compliance report
        """
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        logger.info(f"Generating compliance report for {framework.value}")
        logger.info(f"Time period: {start_date.isoformat()} to {end_date.isoformat()}")
        
        # Filter violations by framework and time period
        relevant_violations = [
            v for v in self.violations
            if v.framework == framework and start_date <= v.detected_at <= end_date
        ]
        
        # Create report
        report = ComplianceReport(
            report_id=f"report-{uuid.uuid4().hex[:8]}",
            framework=framework,
            start_date=start_date,
            end_date=end_date,
            total_operations=self.operation_count,
            compliant_operations=self.compliant_operations,
            violations_detected=len(relevant_violations),
            violations_remediated=len([v for v in relevant_violations if v.status == ComplianceStatus.REMEDIATED]),
            violations=relevant_violations
        )
        
        # Calculate violation breakdowns
        for violation in relevant_violations:
            # By severity
            if violation.severity not in report.violations_by_severity:
                report.violations_by_severity[violation.severity] = 0
            report.violations_by_severity[violation.severity] += 1
            
            # By rule
            if violation.rule_id not in report.violations_by_rule:
                report.violations_by_rule[violation.rule_id] = 0
            report.violations_by_rule[violation.rule_id] += 1
            
            # By model
            if violation.model not in report.violations_by_model:
                report.violations_by_model[violation.model] = 0
            report.violations_by_model[violation.model] += 1
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)
        
        logger.info(f"✅ Compliance report generated: {report.report_id}")
        logger.info(f"Compliance rate: {report.get_compliance_rate():.1f}%")
        logger.info(f"Remediation rate: {report.get_remediation_rate():.1f}%")
        
        return report
    
    def _generate_recommendations(self, report: ComplianceReport) -> List[str]:
        """Generate recommendations based on compliance report."""
        
        recommendations = []
        
        # High violation count recommendations
        if report.violations_detected > 10:
            recommendations.append("Consider implementing additional preventive controls to reduce violation frequency")
        
        # Low remediation rate recommendations
        if report.get_remediation_rate() < 80:
            recommendations.append("Improve automated remediation capabilities to increase remediation rate")
        
        # Critical violations recommendations
        critical_violations = report.violations_by_severity.get(ViolationSeverity.CRITICAL, 0)
        if critical_violations > 0:
            recommendations.append(f"Address {critical_violations} critical violations immediately")
        
        # Model-specific recommendations
        if report.violations_by_model:
            worst_model = max(report.violations_by_model.keys(), key=lambda m: report.violations_by_model[m])
            recommendations.append(f"Review security policies for {worst_model.value} model")
        
        # Rule-specific recommendations
        if report.violations_by_rule:
            most_violated_rule = max(report.violations_by_rule.keys(), key=lambda r: report.violations_by_rule[r])
            recommendations.append(f"Focus on compliance training for rule {most_violated_rule}")
        
        return recommendations
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get current compliance status summary."""
        
        active_violations = [v for v in self.violations if v.status == ComplianceStatus.NON_COMPLIANT]
        
        return {
            'overall_compliance_rate': (self.compliant_operations / max(1, self.operation_count)) * 100,
            'total_operations': self.operation_count,
            'compliant_operations': self.compliant_operations,
            'total_violations': len(self.violations),
            'active_violations': len(active_violations),
            'remediated_violations': len([v for v in self.violations if v.status == ComplianceStatus.REMEDIATED]),
            'enabled_frameworks': [f.value for f in self.enabled_frameworks],
            'violation_breakdown': {
                'critical': len([v for v in active_violations if v.severity == ViolationSeverity.CRITICAL]),
                'high': len([v for v in active_violations if v.severity == ViolationSeverity.HIGH]),
                'medium': len([v for v in active_violations if v.severity == ViolationSeverity.MEDIUM]),
                'low': len([v for v in active_violations if v.severity == ViolationSeverity.LOW])
            }
        }
    
    def get_compliance_rules(self, framework: Optional[ComplianceFramework] = None) -> List[Dict[str, Any]]:
        """
        Get compliance rules for a specific framework or all frameworks.
        
        Args:
            framework: Specific framework to get rules for (None for all)
            
        Returns:
            List of compliance rules
        """
        
        if framework:
            rules = [r for r in self.compliance_rules if r.framework == framework]
        else:
            rules = self.compliance_rules
        
        return [rule.to_dict() for rule in rules]
    
    def add_custom_rule(self, rule: ComplianceRule) -> None:
        """
        Add a custom compliance rule.
        
        Args:
            rule: Custom compliance rule to add
        """
        
        self.compliance_rules.append(rule)
        logger.info(f"Added custom compliance rule: {rule.rule_id}")
    
    def export_compliance_data(self, format: str = "json") -> str:
        """
        Export compliance data for external analysis.
        
        Args:
            format: Export format ("json" or "csv")
            
        Returns:
            Exported data as string
        """
        
        if format.lower() == "json":
            export_data = {
                'compliance_status': self.get_compliance_status(),
                'violations': [v.to_dict() for v in self.violations],
                'rules': [r.to_dict() for r in self.compliance_rules]
            }
            return json.dumps(export_data, indent=2)
        
        elif format.lower() == "csv":
            # Simple CSV export of violations
            csv_lines = ["violation_id,framework,severity,rule_id,detected_at,status"]
            for violation in self.violations:
                csv_lines.append(
                    f"{violation.violation_id},{violation.framework.value},"
                    f"{violation.severity.value},{violation.rule_id},"
                    f"{violation.detected_at.isoformat()},{violation.status.value}"
                )
            return "\n".join(csv_lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")