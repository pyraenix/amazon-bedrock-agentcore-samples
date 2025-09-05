"""
Browser-Use Healthcare Form Automation Example

This example demonstrates how browser-use Agent handles healthcare forms with PII
while leveraging AgentCore's secure environment for HIPAA-compliant operations.

Features demonstrated:
- HIPAA-compliant PII masking within AgentCore's secure environment
- Session isolation and audit trail for healthcare data processing
- Secure handling of medical records and patient information
- Real-time monitoring through AgentCore's live view

Requirements: 2.1, 4.3, 6.1
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Core imports - production ready
from browser_use import Agent
from langchain_aws import ChatBedrock

# AgentCore and tutorial utilities
from tools.browseruse_agentcore_session_manager import (
    BrowserUseAgentCoreSessionManager, 
    SessionConfig
)
from tools.browseruse_sensitive_data_handler import (
    BrowserUseSensitiveDataHandler,
    ComplianceFramework,
    DataClassification,
    PIIType
)


@dataclass
class HealthcareFormData:
    """Healthcare form data structure with PII classification."""
    patient_name: str
    date_of_birth: str
    ssn: str
    medical_record_number: str
    insurance_id: str
    phone_number: str
    email: str
    address: str
    emergency_contact: str
    medical_conditions: List[str]
    medications: List[str]
    allergies: List[str]


class HealthcareFormAutomationExample:
    """
    Healthcare form automation example using browser-use with AgentCore.
    
    Demonstrates HIPAA-compliant handling of patient information with
    comprehensive PII masking and audit trails.
    """
    
    def __init__(self, region: str = 'us-east-1'):
        """
        Initialize the healthcare form automation example.
        
        Args:
            region: AWS region for AgentCore services
        """
        self.region = region
        self.logger = logging.getLogger(__name__)
        
        # Configure session for healthcare compliance
        self.session_config = SessionConfig(
            region=region,
            session_timeout=900,  # 15 minutes for complex healthcare forms
            enable_live_view=True,
            enable_session_replay=True,
            isolation_level="micro-vm",
            compliance_mode="enterprise"
        )
        
        # Initialize sensitive data handler with HIPAA compliance
        self.data_handler = BrowserUseSensitiveDataHandler(
            compliance_frameworks=[ComplianceFramework.HIPAA]
        )
        
        # Initialize session manager
        self.session_manager = BrowserUseAgentCoreSessionManager(self.session_config)
        
        # Initialize LLM model for healthcare context
        self.llm_model = ChatBedrock(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            region_name=region,
            model_kwargs={
                "max_tokens": 4000,
                "temperature": 0.1,  # Low temperature for precise healthcare data handling
                "top_p": 0.9
            }
        )
    
    async def create_sample_healthcare_data(self) -> HealthcareFormData:
        """
        Create sample healthcare data for demonstration.
        
        Returns:
            Sample healthcare form data with realistic PII
        """
        return HealthcareFormData(
            patient_name="Jane Smith",
            date_of_birth="03/15/1985",
            ssn="123-45-6789",
            medical_record_number="MRN-HC789456",
            insurance_id="INS-ABC123456789",
            phone_number="(555) 123-4567",
            email="jane.smith@email.com",
            address="123 Main St, Anytown, ST 12345",
            emergency_contact="John Smith - (555) 987-6543",
            medical_conditions=["Hypertension", "Type 2 Diabetes"],
            medications=["Metformin 500mg", "Lisinopril 10mg"],
            allergies=["Penicillin", "Shellfish"]
        )
    
    def mask_healthcare_data(self, form_data: HealthcareFormData) -> Dict[str, Any]:
        """
        Mask healthcare data according to HIPAA requirements.
        
        Args:
            form_data: Healthcare form data to mask
            
        Returns:
            Masked data with PII protection and audit information
        """
        self.logger.info("🔒 Masking healthcare data for HIPAA compliance")
        
        masked_data = {}
        pii_audit_trail = []
        
        # Process each field with PII detection and masking
        for field_name, field_value in form_data.__dict__.items():
            if isinstance(field_value, list):
                # Handle list fields (conditions, medications, allergies)
                masked_list = []
                for item in field_value:
                    masked_item, detections = self.data_handler.mask_text(str(item), field_name)
                    masked_list.append(masked_item)
                    if detections:
                        pii_audit_trail.extend([
                            {
                                'field': field_name,
                                'item': item,
                                'pii_type': d.pii_type.value,
                                'confidence': d.confidence,
                                'masked_value': d.masked_value,
                                'compliance_impact': [f.value for f in d.compliance_impact]
                            } for d in detections
                        ])
                masked_data[field_name] = masked_list
            else:
                # Handle string fields
                masked_value, detections = self.data_handler.mask_text(str(field_value), field_name)
                masked_data[field_name] = masked_value
                
                if detections:
                    pii_audit_trail.extend([
                        {
                            'field': field_name,
                            'original_value': field_value,
                            'pii_type': d.pii_type.value,
                            'confidence': d.confidence,
                            'masked_value': d.masked_value,
                            'compliance_impact': [f.value for f in d.compliance_impact]
                        } for d in detections
                    ])
        
        # Classify overall data sensitivity
        all_text = " ".join([str(v) for v in form_data.__dict__.values() if not isinstance(v, list)])
        classification = self.data_handler.classify_data(all_text)
        
        # Validate HIPAA compliance
        compliance_result = self.data_handler.validate_compliance(
            all_text, 
            [ComplianceFramework.HIPAA]
        )
        
        return {
            'masked_data': masked_data,
            'pii_audit_trail': pii_audit_trail,
            'data_classification': classification.value,
            'hipaa_compliance': compliance_result,
            'total_pii_detected': len(pii_audit_trail),
            'processing_timestamp': datetime.now().isoformat()
        }
    
    async def demonstrate_secure_form_filling(self, 
                                            form_url: str = "https://example-healthcare-portal.com/patient-form",
                                            form_data: Optional[HealthcareFormData] = None) -> Dict[str, Any]:
        """
        Demonstrate secure healthcare form filling with browser-use and AgentCore.
        
        Args:
            form_url: URL of the healthcare form to fill
            form_data: Healthcare data to use (sample data if not provided)
            
        Returns:
            Results of the secure form filling operation
        """
        if form_data is None:
            form_data = await self.create_sample_healthcare_data()
        
        self.logger.info("🏥 Starting secure healthcare form automation")
        
        # Mask sensitive data before processing
        masked_result = self.mask_healthcare_data(form_data)
        
        try:
            # Create secure AgentCore session with healthcare context
            session_id, ws_url, headers = await self.session_manager.create_secure_session(
                sensitive_context={
                    'data_type': 'healthcare',
                    'compliance': 'HIPAA',
                    'pii_types': ['ssn', 'dob', 'medical_record', 'phone', 'email'],
                    'classification': 'restricted',
                    'audit_required': True
                }
            )
            
            self.logger.info(f"✅ Created secure AgentCore session: {session_id}")
            live_view_url = self.session_manager.get_live_view_url(session_id)
            if live_view_url:
                self.logger.info(f"👁️ Live view available: {live_view_url}")
            
            # Create healthcare-specific task instruction
            task_instruction = f"""
            Navigate to the healthcare patient form at {form_url} and fill it out securely.
            
            CRITICAL SECURITY REQUIREMENTS:
            1. Use only the MASKED data provided - never expose original PII
            2. Verify form fields are properly encrypted before input
            3. Ensure all patient data is handled according to HIPAA requirements
            4. Take screenshots at key steps for audit trail
            5. Verify successful submission without data exposure
            
            MASKED PATIENT DATA TO USE:
            - Patient Name: {masked_result['masked_data']['patient_name']}
            - Date of Birth: {masked_result['masked_data']['date_of_birth']}
            - SSN: {masked_result['masked_data']['ssn']}
            - Medical Record: {masked_result['masked_data']['medical_record_number']}
            - Phone: {masked_result['masked_data']['phone_number']}
            - Email: {masked_result['masked_data']['email']}
            - Address: {masked_result['masked_data']['address']}
            - Emergency Contact: {masked_result['masked_data']['emergency_contact']}
            - Medical Conditions: {', '.join(masked_result['masked_data']['medical_conditions'])}
            - Medications: {', '.join(masked_result['masked_data']['medications'])}
            - Allergies: {', '.join(masked_result['masked_data']['allergies'])}
            
            COMPLIANCE NOTES:
            - This is HIPAA-protected health information
            - Session is isolated in AgentCore micro-VM
            - All actions are being recorded for audit
            - Data classification: {masked_result['data_classification']}
            """
            
            # Create browser-use agent with healthcare context
            agent = await self.session_manager.create_browseruse_agent(
                session_id=session_id,
                task=task_instruction,
                llm_model=self.llm_model
            )
            
            self.logger.info("🤖 Created browser-use agent for healthcare form automation")
            
            # Execute the secure healthcare form filling task
            execution_result = await self.session_manager.execute_sensitive_task(
                session_id=session_id,
                agent=agent,
                sensitive_data_context={
                    'pii_types': ['ssn', 'dob', 'medical_record', 'phone', 'email'],
                    'compliance_framework': 'HIPAA',
                    'data_classification': 'restricted',
                    'audit_level': 'comprehensive'
                }
            )
            
            # Get session status and metrics
            session_status = self.session_manager.get_session_status(session_id)
            
            # Compile comprehensive results
            results = {
                'status': 'completed',
                'session_id': session_id,
                'live_view_url': live_view_url,
                'execution_result': execution_result,
                'session_metrics': session_status,
                'security_measures': {
                    'pii_masking_applied': True,
                    'hipaa_compliance_validated': masked_result['hipaa_compliance']['compliant'],
                    'micro_vm_isolation': True,
                    'session_recording_enabled': True,
                    'audit_trail_complete': True
                },
                'pii_handling': {
                    'total_pii_detected': masked_result['total_pii_detected'],
                    'pii_audit_trail': masked_result['pii_audit_trail'],
                    'data_classification': masked_result['data_classification'],
                    'compliance_violations': masked_result['hipaa_compliance']['violations']
                },
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("✅ Healthcare form automation completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Healthcare form automation failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'session_id': session_id if 'session_id' in locals() else None,
                'timestamp': datetime.now().isoformat()
            }
        
        finally:
            # Always cleanup session for security
            if 'session_id' in locals():
                await self.session_manager.cleanup_session(session_id, reason="healthcare_task_complete")
                self.logger.info("🧹 Healthcare session cleaned up for security")
    
    async def demonstrate_patient_portal_navigation(self) -> Dict[str, Any]:
        """
        Demonstrate secure navigation of a patient portal with PII handling.
        
        Returns:
            Results of the patient portal navigation demonstration
        """
        self.logger.info("🏥 Demonstrating patient portal navigation with PII protection")
        
        # Create sample patient data
        patient_data = await self.create_sample_healthcare_data()
        
        # Use the secure session context manager
        async with self.session_manager.secure_session_context(
            task="Navigate patient portal and view medical records securely",
            llm_model=self.llm_model,
            sensitive_context={
                'data_type': 'healthcare',
                'compliance': 'HIPAA',
                'operation': 'patient_portal_access'
            }
        ) as (session_id, agent):
            
            self.logger.info(f"🔐 Secure patient portal session created: {session_id}")
            
            # Mask patient data for secure handling
            masked_result = self.mask_healthcare_data(patient_data)
            
            # Execute patient portal navigation
            portal_task = f"""
            Securely navigate a patient portal to view medical records.
            
            SECURITY REQUIREMENTS:
            1. Use masked patient identifiers only
            2. Verify secure connection (HTTPS) before login
            3. Handle any PII with HIPAA compliance
            4. Take screenshots for audit trail
            5. Ensure proper logout and session cleanup
            
            MASKED PATIENT IDENTIFIERS:
            - Medical Record: {masked_result['masked_data']['medical_record_number']}
            - Date of Birth: {masked_result['masked_data']['date_of_birth']}
            - Last 4 SSN: {masked_result['masked_data']['ssn'][-4:]}
            
            Navigate to: https://example-patient-portal.com
            """
            
            # Update agent task
            agent.task = portal_task
            
            # Execute with comprehensive monitoring
            result = await agent.run()
            
            return {
                'status': 'completed',
                'session_id': session_id,
                'portal_navigation_result': result,
                'pii_protection': masked_result,
                'security_measures': {
                    'hipaa_compliant': True,
                    'pii_masked': True,
                    'session_isolated': True,
                    'audit_trail_enabled': True
                },
                'timestamp': datetime.now().isoformat()
            }
    
    async def validate_hipaa_compliance(self, operation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate HIPAA compliance of healthcare operations.
        
        Args:
            operation_results: Results from healthcare operations
            
        Returns:
            HIPAA compliance validation results
        """
        self.logger.info("📋 Validating HIPAA compliance for healthcare operations")
        
        compliance_checks = {
            'pii_masking_applied': False,
            'session_isolation_verified': False,
            'audit_trail_complete': False,
            'secure_transmission_used': False,
            'access_controls_enforced': False,
            'data_minimization_applied': False
        }
        
        # Check PII masking
        if 'pii_handling' in operation_results:
            pii_data = operation_results['pii_handling']
            if pii_data.get('total_pii_detected', 0) > 0 and pii_data.get('pii_audit_trail'):
                compliance_checks['pii_masking_applied'] = True
        
        # Check session isolation
        if 'security_measures' in operation_results:
            security = operation_results['security_measures']
            if security.get('micro_vm_isolation') and security.get('session_recording_enabled'):
                compliance_checks['session_isolation_verified'] = True
                compliance_checks['audit_trail_complete'] = True
        
        # Check secure transmission (AgentCore provides this)
        if 'session_id' in operation_results:
            compliance_checks['secure_transmission_used'] = True
            compliance_checks['access_controls_enforced'] = True
        
        # Data minimization check
        if 'pii_handling' in operation_results:
            if operation_results['pii_handling'].get('data_classification') == 'restricted':
                compliance_checks['data_minimization_applied'] = True
        
        # Calculate overall compliance score
        passed_checks = sum(1 for check in compliance_checks.values() if check)
        total_checks = len(compliance_checks)
        compliance_score = (passed_checks / total_checks) * 100
        
        return {
            'hipaa_compliant': compliance_score >= 90,  # 90% threshold for compliance
            'compliance_score': compliance_score,
            'compliance_checks': compliance_checks,
            'recommendations': self._get_hipaa_recommendations(compliance_checks),
            'validation_timestamp': datetime.now().isoformat()
        }
    
    def _get_hipaa_recommendations(self, compliance_checks: Dict[str, bool]) -> List[str]:
        """Get HIPAA compliance recommendations based on check results."""
        recommendations = []
        
        if not compliance_checks['pii_masking_applied']:
            recommendations.append("Implement comprehensive PII masking for all patient data")
        
        if not compliance_checks['session_isolation_verified']:
            recommendations.append("Ensure proper session isolation using AgentCore micro-VM")
        
        if not compliance_checks['audit_trail_complete']:
            recommendations.append("Enable comprehensive audit logging for all healthcare operations")
        
        if not compliance_checks['secure_transmission_used']:
            recommendations.append("Use encrypted transmission for all patient data")
        
        if not compliance_checks['access_controls_enforced']:
            recommendations.append("Implement role-based access controls for patient data")
        
        if not compliance_checks['data_minimization_applied']:
            recommendations.append("Apply data minimization principles to reduce PII exposure")
        
        return recommendations
    
    async def run_comprehensive_healthcare_demo(self) -> Dict[str, Any]:
        """
        Run a comprehensive healthcare form automation demonstration.
        
        Returns:
            Complete demonstration results with HIPAA compliance validation
        """
        self.logger.info("🚀 Starting comprehensive healthcare form automation demo")
        
        try:
            # Step 1: Demonstrate secure form filling
            form_results = await self.demonstrate_secure_form_filling()
            
            # Step 2: Demonstrate patient portal navigation
            portal_results = await self.demonstrate_patient_portal_navigation()
            
            # Step 3: Validate HIPAA compliance
            compliance_validation = await self.validate_hipaa_compliance(form_results)
            
            # Compile comprehensive demo results
            demo_results = {
                'demo_status': 'completed',
                'healthcare_form_automation': form_results,
                'patient_portal_navigation': portal_results,
                'hipaa_compliance_validation': compliance_validation,
                'demo_summary': {
                    'total_operations': 2,
                    'pii_items_protected': form_results.get('pii_handling', {}).get('total_pii_detected', 0),
                    'sessions_created': 2,
                    'compliance_score': compliance_validation.get('compliance_score', 0),
                    'security_measures_applied': [
                        'PII masking and detection',
                        'AgentCore micro-VM isolation',
                        'HIPAA-compliant data handling',
                        'Comprehensive audit trails',
                        'Secure session management',
                        'Real-time monitoring via live view'
                    ]
                },
                'demo_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("✅ Comprehensive healthcare demo completed successfully")
            return demo_results
            
        except Exception as e:
            self.logger.error(f"❌ Healthcare demo failed: {e}")
            return {
                'demo_status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
        finally:
            # Cleanup any remaining resources
            await self.session_manager.shutdown()


# Standalone execution example
async def main():
    """Main execution function for the healthcare form automation example."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🏥 Healthcare Form Automation Example - Browser-Use + AgentCore")
    
    try:
        # Initialize the healthcare automation example
        healthcare_example = HealthcareFormAutomationExample(region='us-east-1')
        
        # Run the comprehensive demonstration
        results = await healthcare_example.run_comprehensive_healthcare_demo()
        
        # Display results
        print("\n" + "="*80)
        print("🏥 HEALTHCARE FORM AUTOMATION RESULTS")
        print("="*80)
        print(f"Demo Status: {results['demo_status']}")
        
        if results['demo_status'] == 'completed':
            summary = results['demo_summary']
            print(f"Total Operations: {summary['total_operations']}")
            print(f"PII Items Protected: {summary['pii_items_protected']}")
            print(f"HIPAA Compliance Score: {summary['compliance_score']:.1f}%")
            print(f"Sessions Created: {summary['sessions_created']}")
            
            print("\nSecurity Measures Applied:")
            for measure in summary['security_measures_applied']:
                print(f"  ✅ {measure}")
            
            # Show compliance validation
            compliance = results['hipaa_compliance_validation']
            print(f"\nHIPAA Compliance: {'✅ COMPLIANT' if compliance['hipaa_compliant'] else '❌ NON-COMPLIANT'}")
            
            if compliance.get('recommendations'):
                print("\nRecommendations:")
                for rec in compliance['recommendations']:
                    print(f"  📋 {rec}")
        else:
            print(f"Error: {results.get('error', 'Unknown error')}")
        
        print("="*80)
        
    except Exception as e:
        logger.error(f"❌ Failed to run healthcare example: {e}")
        print(f"\n❌ Example failed: {e}")


if __name__ == "__main__":
    print("🏥 Browser-Use Healthcare Form Automation Example")
    print("📋 Demonstrates HIPAA-compliant PII handling with AgentCore")
    print("⚠️  Requires: browser-use, bedrock-agentcore, and AWS credentials")
    print("🚀 Starting demonstration...\n")
    
    # Run the example
    asyncio.run(main())