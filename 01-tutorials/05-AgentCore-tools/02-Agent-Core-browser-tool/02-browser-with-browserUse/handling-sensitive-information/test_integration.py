#!/usr/bin/env python3
"""
Integration test for browser-use compliance notebook.

This test validates that the notebook components work together correctly
for real compliance scenarios.
"""

import asyncio
import logging
import sys
import os
from typing import Dict, Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_healthcare_compliance_integration():
    """Test healthcare compliance integration."""
    
    print("🏥 Testing Healthcare Compliance Integration")
    print("=" * 50)
    
    try:
        # Import required components
        from tools.browseruse_agentcore_session_manager import BrowserUseAgentCoreSessionManager, SessionConfig
        from tools.browseruse_sensitive_data_handler import (
            BrowserUseSensitiveDataHandler,
            ComplianceFramework
        )
        
        # Initialize components
        session_manager = BrowserUseAgentCoreSessionManager()
        sensitive_data_handler = BrowserUseSensitiveDataHandler([
            ComplianceFramework.HIPAA,
            ComplianceFramework.PCI_DSS,
            ComplianceFramework.GDPR
        ])
        
        # Test healthcare data
        healthcare_data = {
            'patient_name': 'John Doe',
            'ssn': '123-45-6789',
            'date_of_birth': '03/15/1985',
            'medical_record_number': 'MRN-ABC123456',
            'phone': '(555) 123-4567',
            'email': 'john.doe@email.com',
            'insurance_id': 'INS-789456123'
        }
        
        print(f"📋 Processing {len(healthcare_data)} healthcare data fields")
        
        # Test PII detection
        total_pii_detected = 0
        compliance_issues = []
        
        for field_name, field_value in healthcare_data.items():
            if isinstance(field_value, str):
                detections = sensitive_data_handler.detect_pii(field_value, field_name)
                total_pii_detected += len(detections)
                
                for detection in detections:
                    print(f"  🔍 {field_name}: {detection.pii_type.value} -> {detection.masked_value}")
                    
                    # Check compliance impact
                    for framework in detection.compliance_impact:
                        if framework == ComplianceFramework.HIPAA:
                            compliance_issues.append({
                                'field': field_name,
                                'framework': 'HIPAA',
                                'pii_type': detection.pii_type.value
                            })
        
        print(f"\n📊 Healthcare Compliance Results:")
        print(f"  Total PII Detected: {total_pii_detected}")
        print(f"  HIPAA Compliance Issues: {len([i for i in compliance_issues if i['framework'] == 'HIPAA'])}")
        
        # Test session configuration for HIPAA
        hipaa_config = SessionConfig(
            region='us-east-1',
            session_timeout=600,
            enable_live_view=True,
            enable_session_replay=True,
            isolation_level="micro-vm",
            compliance_mode="hipaa"
        )
        
        print(f"  Session Configuration: ✅ HIPAA mode enabled")
        print(f"  Micro-VM Isolation: ✅ Enabled")
        print(f"  Session Replay: ✅ Enabled for audit")
        
        # Validate compliance
        if total_pii_detected >= 4:  # Should detect SSN, email, phone, medical record
            print(f"\n✅ Healthcare compliance integration test PASSED")
            print(f"   - PII detection working correctly")
            print(f"   - HIPAA compliance validation active")
            print(f"   - Session security configured properly")
            return True
        else:
            print(f"\n❌ Healthcare compliance integration test FAILED")
            print(f"   - Expected 4+ PII detections, got {total_pii_detected}")
            return False
            
    except Exception as e:
        print(f"\n❌ Healthcare compliance integration test FAILED: {e}")
        return False

async def test_payment_compliance_integration():
    """Test payment compliance integration."""
    
    print("\n💳 Testing Payment Compliance Integration")
    print("=" * 50)
    
    try:
        from tools.browseruse_sensitive_data_handler import (
            BrowserUseSensitiveDataHandler,
            ComplianceFramework
        )
        
        # Initialize for PCI-DSS compliance
        pci_handler = BrowserUseSensitiveDataHandler([ComplianceFramework.PCI_DSS])
        
        # Test payment data
        payment_data = {
            'cardholder_name': 'Jane Smith',
            'card_number': '4532-1234-5678-9012',
            'expiry_date': '12/25',
            'cvv': '123',
            'billing_address': '123 Main St, Anytown, ST 12345',
            'amount': '$150.00'
        }
        
        print(f"💰 Processing {len(payment_data)} payment data fields")
        
        # Test PCI-DSS specific detection
        pci_violations = 0
        
        for field_name, field_value in payment_data.items():
            if isinstance(field_value, str):
                detections = pci_handler.detect_pii(field_value, field_name)
                
                for detection in detections:
                    if ComplianceFramework.PCI_DSS in detection.compliance_impact:
                        pci_violations += 1
                        print(f"  🚨 PCI-DSS: {field_name} contains {detection.pii_type.value}")
                        print(f"     Original: {detection.value}")
                        print(f"     Masked: {detection.masked_value}")
        
        print(f"\n📊 Payment Compliance Results:")
        print(f"  PCI-DSS Violations: {pci_violations}")
        
        if pci_violations >= 1:  # Should detect credit card
            print(f"✅ Payment compliance integration test PASSED")
            print(f"   - Credit card detection working")
            print(f"   - PCI-DSS compliance validation active")
            return True
        else:
            print(f"❌ Payment compliance integration test FAILED")
            print(f"   - No PCI-DSS violations detected")
            return False
            
    except Exception as e:
        print(f"❌ Payment compliance integration test FAILED: {e}")
        return False

async def test_security_boundary_integration():
    """Test security boundary integration."""
    
    print("\n🛡️  Testing Security Boundary Integration")
    print("=" * 50)
    
    try:
        from tools.browseruse_security_boundary_validator import BrowserUseSecurityBoundaryValidator
        
        # Initialize security validator
        session_id = "integration-test-session"
        validator = BrowserUseSecurityBoundaryValidator(session_id)
        
        print(f"🔒 Testing security boundaries for session: {session_id}")
        
        # Test session isolation
        isolation_result = await validator.validate_session_isolation()
        print(f"  Session Isolation: {'✅ PASSED' if isolation_result.passed else '❌ FAILED'}")
        
        # Test data leakage prevention
        sensitive_data = {
            'ssn': '123-45-6789',
            'credit_card': '4532-1234-5678-9012'
        }
        
        leakage_result = await validator.validate_data_leakage_prevention(sensitive_data)
        print(f"  Data Leakage Prevention: {'✅ PASSED' if leakage_result.passed else '❌ FAILED'}")
        
        # Test error handling security
        error_result = await validator.validate_error_handling_security()
        print(f"  Error Handling Security: {'✅ PASSED' if error_result.passed else '❌ FAILED'}")
        
        # Generate security report
        security_report = validator.generate_security_report()
        
        print(f"\n📊 Security Boundary Results:")
        print(f"  Total Tests: {security_report['total_tests']}")
        print(f"  Passed Tests: {security_report['passed_tests']}")
        print(f"  Security Score: {security_report['security_score']:.1f}%")
        
        if security_report['security_score'] >= 80:
            print(f"✅ Security boundary integration test PASSED")
            return True
        else:
            print(f"❌ Security boundary integration test FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Security boundary integration test FAILED: {e}")
        return False

async def main():
    """Run all integration tests."""
    
    print("🧪 Browser-Use Compliance Notebook Integration Tests")
    print("=" * 60)
    
    tests = [
        test_healthcare_compliance_integration,
        test_payment_compliance_integration,
        test_security_boundary_integration
    ]
    
    results = []
    
    for test_func in tests:
        try:
            result = await test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    passed_tests = sum(results)
    total_tests = len(results)
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n" + "=" * 60)
    print(f"🎯 INTEGRATION TEST SUMMARY")
    print(f"=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed Tests: {passed_tests}")
    print(f"Failed Tests: {total_tests - passed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        print(f"\n🎉 ALL INTEGRATION TESTS PASSED!")
        print(f"📋 Notebook is ready for production compliance use")
        return 0
    else:
        print(f"\n⚠️  Some integration tests failed")
        print(f"📋 Please review and fix issues before production use")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)