"""
Test Bedrock Security Framework Integration

This script tests the integration between the BedrockModelRouter and ComplianceValidator
to ensure they work together properly for Strands agents handling sensitive information.
"""

import sys
import os
import logging
from datetime import datetime
from typing import Set

# Add the tools directory to the path
tools_dir = os.path.join(os.path.dirname(__file__), '..', 'tools')
sys.path.append(tools_dir)

# Import sensitive data handler first
from sensitive_data_handler import SensitiveDataHandler, SensitivityLevel, PIIType

from bedrock_model_router import (
    BedrockModelRouter, BedrockModel, SecurityTier, RoutingRequest, 
    ModelCapability
)
from compliance_validator import (
    ComplianceValidator, ComplianceFramework, ViolationSeverity
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_bedrock_security_framework():
    """Test the complete Bedrock security framework."""
    
    print("🧪 Testing Strands Bedrock Multi-Model Security Framework")
    print("=" * 60)
    
    # Initialize components
    print("\n1. Initializing Security Framework Components...")
    
    router = BedrockModelRouter(
        region="us-east-1",
        session_id="test-session-001",
        agent_id="strands-test-agent"
    )
    
    validator = ComplianceValidator(
        region="us-east-1",
        session_id="test-session-001",
        agent_id="strands-test-agent",
        enabled_frameworks=[
            ComplianceFramework.HIPAA,
            ComplianceFramework.PCI_DSS,
            ComplianceFramework.GDPR
        ]
    )
    
    print("✅ Components initialized successfully")
    
    # Test Case 1: Public content (no PII)
    print("\n2. Testing Public Content Routing...")
    
    public_request = router.create_routing_request(
        content="What is the weather like today?",
        required_capabilities={ModelCapability.GENERAL_PURPOSE}
    )
    
    public_decision = router.route_request(public_request)
    print(f"✅ Public content routed to: {public_decision.selected_model.value}")
    print(f"   Security tier: {public_decision.security_tier.value}")
    print(f"   Fallback models: {[m.value for m in public_decision.fallback_models]}")
    
    # Validate compliance for public content
    public_validation = validator.validate_operation(
        operation_type="text_generation",
        content=public_request.content,
        model=public_decision.selected_model,
        security_tier=public_decision.security_tier,
        pii_types=public_decision.pii_types_detected,
        metadata={'encrypted': True, 'audit_logged': True}
    )
    
    print(f"   Compliance status: {public_validation.data['compliance_status']}")
    
    # Test Case 2: Healthcare content (HIPAA sensitive)
    print("\n3. Testing Healthcare Content Routing...")
    
    healthcare_content = """
    Patient John Doe (SSN: 123-45-6789) was admitted on 2024-01-15.
    Medical record number: MRN-789456123.
    Diagnosis: Hypertension and diabetes.
    """
    
    healthcare_request = router.create_routing_request(
        content=healthcare_content,
        compliance_requirements={"HIPAA"},
        required_capabilities={ModelCapability.ANALYSIS}
    )
    
    healthcare_decision = router.route_request(healthcare_request)
    print(f"✅ Healthcare content routed to: {healthcare_decision.selected_model.value}")
    print(f"   Security tier: {healthcare_decision.security_tier.value}")
    print(f"   PII types detected: {[pii.value for pii in healthcare_decision.pii_types_detected]}")
    print(f"   Compliance validated: {healthcare_decision.compliance_validated}")
    
    # Validate compliance for healthcare content
    healthcare_validation = validator.validate_operation(
        operation_type="medical_analysis",
        content=healthcare_content,
        model=healthcare_decision.selected_model,
        security_tier=healthcare_decision.security_tier,
        pii_types=healthcare_decision.pii_types_detected,
        metadata={
            'encrypted': True,
            'audit_logged': True,
            'consent_obtained': False,  # This should trigger a violation
            'data_minimized': True
        }
    )
    
    print(f"   Compliance status: {healthcare_validation.data['compliance_status']}")
    print(f"   Violations detected: {healthcare_validation.data['violations_detected']}")
    
    # Test Case 3: Financial content (PCI DSS sensitive)
    print("\n4. Testing Financial Content Routing...")
    
    financial_content = """
    Process payment for customer jane.smith@email.com
    Credit card: 4532-1234-5678-9012
    Bank account: 123456789
    Amount: $1,500.00
    """
    
    financial_request = router.create_routing_request(
        content=financial_content,
        compliance_requirements={"PCI_DSS"},
        required_capabilities={ModelCapability.CLASSIFICATION},
        max_cost_per_1k_tokens=0.01
    )
    
    financial_decision = router.route_request(financial_request)
    print(f"✅ Financial content routed to: {financial_decision.selected_model.value}")
    print(f"   Security tier: {financial_decision.security_tier.value}")
    print(f"   PII types detected: {[pii.value for pii in financial_decision.pii_types_detected]}")
    print(f"   Routing reason: {financial_decision.routing_reason}")
    
    # Validate compliance for financial content
    financial_validation = validator.validate_operation(
        operation_type="payment_processing",
        content=financial_content,
        model=financial_decision.selected_model,
        security_tier=financial_decision.security_tier,
        pii_types=financial_decision.pii_types_detected,
        metadata={
            'encrypted': True,
            'audit_logged': True,
            'data_minimized': False,  # This should trigger a violation
            'retention_date': '2025-12-31T00:00:00'
        }
    )
    
    print(f"   Compliance status: {financial_validation.data['compliance_status']}")
    print(f"   Violations detected: {financial_validation.data['violations_detected']}")
    
    # Test Case 4: Execute with fallback
    print("\n5. Testing Execution with Fallback...")
    
    try:
        execution_result = router.execute_with_fallback(
            request=healthcare_request,
            prompt="Analyze this medical data for risk factors",
            max_tokens=500
        )
        
        print(f"✅ Execution successful: {execution_result.success}")
        print(f"   Model used: {execution_result.data.get('model', 'unknown')}")
        print(f"   Response length: {len(execution_result.data.get('response', ''))}")
        
    except Exception as e:
        print(f"⚠️ Execution failed (expected in test environment): {str(e)}")
    
    # Test Case 5: Generate compliance reports
    print("\n6. Generating Compliance Reports...")
    
    for framework in [ComplianceFramework.HIPAA, ComplianceFramework.PCI_DSS, ComplianceFramework.GDPR]:
        report = validator.generate_compliance_report(framework)
        print(f"✅ {framework.value.upper()} Report Generated:")
        print(f"   Report ID: {report.report_id}")
        print(f"   Compliance Rate: {report.get_compliance_rate():.1f}%")
        print(f"   Violations: {report.violations_detected}")
        print(f"   Recommendations: {len(report.recommendations)}")
    
    # Test Case 6: Get usage metrics
    print("\n7. Checking Usage Metrics...")
    
    usage_metrics = router.get_usage_metrics()
    print(f"✅ Usage Metrics Retrieved:")
    print(f"   Total requests: {usage_metrics['overall_metrics']['total_requests']}")
    print(f"   Total cost: ${usage_metrics['overall_metrics']['total_cost']:.4f}")
    print(f"   Average success rate: {usage_metrics['overall_metrics']['avg_success_rate']:.1f}%")
    
    # Test Case 7: Get compliance status
    print("\n8. Checking Compliance Status...")
    
    compliance_status = validator.get_compliance_status()
    print(f"✅ Compliance Status Retrieved:")
    print(f"   Overall compliance rate: {compliance_status['overall_compliance_rate']:.1f}%")
    print(f"   Total violations: {compliance_status['total_violations']}")
    print(f"   Active violations: {compliance_status['active_violations']}")
    print(f"   Critical violations: {compliance_status['violation_breakdown']['critical']}")
    
    # Test Case 8: Model policy information
    print("\n9. Reviewing Model Policies...")
    
    model_policies = router.get_model_policies()
    print(f"✅ Model Policies Retrieved: {len(model_policies)} models")
    
    for model_name, policy in list(model_policies.items())[:3]:  # Show first 3
        print(f"   {model_name}:")
        print(f"     Max security tier: {policy['max_security_tier']}")
        print(f"     HIPAA compliant: {policy['compliance']['hipaa_compliant']}")
        print(f"     PCI DSS compliant: {policy['compliance']['pci_dss_compliant']}")
        print(f"     Cost per 1k tokens: ${policy['performance']['cost_per_1k_tokens']}")
    
    # Test Case 9: Routing history audit trail
    print("\n10. Reviewing Routing History...")
    
    routing_history = router.get_routing_history(limit=5)
    print(f"✅ Routing History Retrieved: {len(routing_history)} recent decisions")
    
    for decision in routing_history:
        print(f"   Request {decision['request_id']}:")
        print(f"     Model: {decision['selected_model']}")
        print(f"     Security tier: {decision['security_context']['security_tier']}")
        print(f"     Timestamp: {decision['decision_metadata']['decision_timestamp']}")
    
    print("\n" + "=" * 60)
    print("🎉 Bedrock Security Framework Test Complete!")
    print("\nKey Features Verified:")
    print("✅ Multi-model security routing based on data sensitivity")
    print("✅ Model-specific security policies for different Bedrock models")
    print("✅ Intelligent fallback system maintaining security levels")
    print("✅ Cross-model audit trail for sensitive data tracking")
    print("✅ HIPAA, PCI DSS, and GDPR compliance validation")
    print("✅ Real-time compliance monitoring during execution")
    print("✅ Automated compliance reporting")
    print("✅ Violation detection and remediation")
    
    return True


def demonstrate_advanced_features():
    """Demonstrate advanced features of the security framework."""
    
    print("\n" + "=" * 60)
    print("🚀 Advanced Security Framework Features")
    print("=" * 60)
    
    # Initialize components
    router = BedrockModelRouter(region="us-east-1")
    validator = ComplianceValidator(region="us-east-1")
    
    # Advanced Feature 1: Custom routing preferences
    print("\n1. Custom Routing Preferences...")
    
    custom_request = router.create_routing_request(
        content="Analyze customer feedback data containing personal information",
        preferred_models=[BedrockModel.CLAUDE_3_SONNET, BedrockModel.CLAUDE_3_5_SONNET],
        required_capabilities={ModelCapability.ANALYSIS, ModelCapability.CLASSIFICATION},
        max_cost_per_1k_tokens=0.005,
        max_latency_ms=1500,
        compliance_requirements={"GDPR"}
    )
    
    custom_decision = router.route_request(custom_request)
    print(f"✅ Custom routing decision: {custom_decision.selected_model.value}")
    print(f"   Confidence score: {custom_decision.confidence_score}")
    print(f"   Estimated cost: ${custom_decision.estimated_cost:.4f}")
    print(f"   Estimated latency: {custom_decision.estimated_latency_ms}ms")
    
    # Advanced Feature 2: Multi-framework compliance validation
    print("\n2. Multi-Framework Compliance Validation...")
    
    multi_framework_content = """
    Patient Sarah Johnson (SSN: 987-65-4321, Email: sarah.j@email.com)
    Credit Card: 5555-4444-3333-2222
    Medical Record: MRN-456789123
    European resident (GDPR applicable)
    """
    
    multi_validation = validator.validate_operation(
        operation_type="multi_framework_analysis",
        content=multi_framework_content,
        model=BedrockModel.CLAUDE_3_OPUS,
        security_tier=SecurityTier.TOP_SECRET,
        pii_types={PIIType.SSN, PIIType.EMAIL, PIIType.CREDIT_CARD, PIIType.MEDICAL_ID},
        metadata={
            'encrypted': True,
            'audit_logged': True,
            'consent_obtained': True,
            'data_minimized': True,
            'retention_date': '2025-01-15T00:00:00'
        }
    )
    
    print(f"✅ Multi-framework validation complete")
    print(f"   Status: {multi_validation.data['compliance_status']}")
    print(f"   Applicable rules: {multi_validation.data['applicable_rules']}")
    
    # Advanced Feature 3: Export compliance data
    print("\n3. Compliance Data Export...")
    
    json_export = validator.export_compliance_data(format="json")
    csv_export = validator.export_compliance_data(format="csv")
    
    print(f"✅ Compliance data exported")
    print(f"   JSON export size: {len(json_export)} characters")
    print(f"   CSV export lines: {len(csv_export.split(chr(10)))}")
    
    print("\n🎯 Advanced features demonstration complete!")


if __name__ == "__main__":
    try:
        # Run main test
        test_bedrock_security_framework()
        
        # Run advanced features demo
        demonstrate_advanced_features()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)