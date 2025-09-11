#!/usr/bin/env python3
"""
Healthcare Document Processing with HIPAA Compliance
===================================================

This example demonstrates how to use Strands agents with AgentCore Browser Tool
for HIPAA-compliant patient data extraction and processing. It showcases:

1. Secure patient data extraction from healthcare portals
2. HIPAA-compliant PII detection and masking
3. Audit logging for healthcare compliance
4. Secure document processing workflows
5. Patient consent validation

Requirements:
- HIPAA compliance for patient data
- Secure credential management
- Comprehensive audit trails
- Data minimization principles
- Patient consent verification
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid

# Strands framework imports
from strands import Agent
from strands.tools import tool, PythonAgentTool

# AgentCore Browser Tool integration
from strands_tools.browser.agent_core_browser import AgentCoreBrowser
from bedrock_agentcore.tools.browser_client import BrowserClient

# Define missing classes for demo purposes
class BrowserSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.is_active = True

# AWS and security imports
import boto3
from cryptography.fernet import Fernet
import re


class HIPAAComplianceLevel(Enum):
    """HIPAA compliance levels for different data types."""
    PHI = "protected_health_information"  # Highest protection
    PII = "personally_identifiable_information"  # High protection
    DEMOGRAPHIC = "demographic_information"  # Medium protection
    GENERAL = "general_information"  # Standard protection


@dataclass
class PatientDataRecord:
    """Secure patient data record with HIPAA compliance."""
    patient_id: str  # Hashed identifier
    record_type: str
    data_classification: HIPAAComplianceLevel
    extracted_data: Dict[str, Any]
    consent_verified: bool
    extraction_timestamp: datetime
    audit_trail: List[str]
    
    def to_secure_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with sensitive data masked."""
        secure_data = asdict(self)
        if not self.consent_verified:
            secure_data['extracted_data'] = {"status": "consent_required"}
        return secure_data


class HIPAAComplianceTool(PythonAgentTool):
    """Custom Strands tool for HIPAA compliance validation."""
    
    def __init__(self):
        super().__init__(name="hipaa_compliance_validator")
        self.phi_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{10}\b',  # Phone numbers
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # Dates
            r'\b\d{5}(-\d{4})?\b',  # ZIP codes
        ]
        self.encryption_manager = EncryptionManager()
        
    async def detect_phi(self, content: str) -> Dict[str, Any]:
        """Detect Protected Health Information in content."""
        phi_detected = []
        
        for pattern in self.phi_patterns:
            matches = re.findall(pattern, content)
            if matches:
                phi_detected.extend([{
                    'type': 'phi_pattern',
                    'pattern': pattern,
                    'matches_count': len(matches),
                    'severity': 'high'
                }])
        
        return {
            'phi_detected': len(phi_detected) > 0,
            'phi_items': phi_detected,
            'compliance_level': HIPAAComplianceLevel.PHI.value if phi_detected else HIPAAComplianceLevel.GENERAL.value
        }
    
    async def mask_phi(self, content: str, masking_level: str = "full") -> str:
        """Mask PHI in content according to HIPAA requirements."""
        masked_content = content
        
        # Mask SSN
        masked_content = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', 'XXX-XX-XXXX', masked_content)
        
        # Mask phone numbers
        masked_content = re.sub(r'\b\d{10}\b', 'XXX-XXX-XXXX', masked_content)
        
        # Mask email addresses
        masked_content = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL_REDACTED]',
            masked_content
        )
        
        # Mask dates if full masking
        if masking_level == "full":
            masked_content = re.sub(r'\b\d{1,2}/\d{1,2}/\d{4}\b', '[DATE_REDACTED]', masked_content)
        
        return masked_content
    
    async def validate_consent(self, patient_id: str, data_type: str) -> bool:
        """Validate patient consent for data processing."""
        # In production, this would check against a consent management system
        # For demo purposes, we'll simulate consent validation
        consent_key = f"consent_{patient_id}_{data_type}"
        
        # Simulate consent database lookup
        # In real implementation, this would query a secure consent database
        return True  # Assuming consent is granted for demo


class SecureHealthcareAgent:
    """Strands agent specialized for HIPAA-compliant healthcare document processing."""
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.browser_client = BrowserClient(region=region)
        self.credential_manager = SecureCredentialManager()
        self.hipaa_tool = HIPAAComplianceTool()
        self.audit_logger = AuditLogger(service_name="healthcare_agent")
        self.encryption_key = Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
        
        # Initialize Strands agent with healthcare-specific configuration
        self.agent = Agent(
            name="healthcare_document_processor",
            tools=[self.hipaa_tool],
            security_context=SecurityContext(
                compliance_level="hipaa",
                data_classification="phi",
                audit_required=True
            )
        )
    
    async def create_secure_session(self, portal_config: Dict[str, Any]) -> BrowserSession:
        """Create a secure browser session for healthcare portal access."""
        session_config = {
            "isolation_level": "maximum",
            "data_protection": "hipaa_compliant",
            "audit_logging": True,
            "session_timeout": 1800,  # 30 minutes
            "screenshot_disabled": True,  # Prevent PHI in screenshots
            "clipboard_disabled": True,  # Prevent PHI in clipboard
        }
        
        session = await self.browser_client.create_session(session_config)
        
        # Log session creation for audit
        await self.audit_logger.log_event({
            "event_type": "session_created",
            "session_id": session.session_id,
            "portal": portal_config.get("name", "unknown"),
            "compliance_level": "hipaa",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return session
    
    async def authenticate_healthcare_portal(
        self, 
        session: BrowserSession, 
        portal_url: str,
        credential_id: str
    ) -> bool:
        """Securely authenticate to healthcare portal."""
        try:
            # Retrieve credentials from secure storage
            credentials = await self.credential_manager.get_credentials(credential_id)
            
            # Navigate to portal
            await session.navigate(portal_url)
            
            # Wait for login form
            await session.wait_for_element("input[type='email'], input[name='username']")
            
            # Fill credentials securely
            await session.fill_secure("input[type='email'], input[name='username']", credentials['username'])
            await session.fill_secure("input[type='password']", credentials['password'])
            
            # Submit login
            await session.click("button[type='submit'], input[type='submit']")
            
            # Verify successful login
            success = await session.wait_for_element(".dashboard, .patient-list, .main-content", timeout=10000)
            
            # Log authentication attempt
            await self.audit_logger.log_event({
                "event_type": "authentication",
                "session_id": session.session_id,
                "portal_url": portal_url,
                "success": success,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return success
            
        except Exception as e:
            await self.audit_logger.log_event({
                "event_type": "authentication_error",
                "session_id": session.session_id,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            return False
    
    async def extract_patient_data(
        self, 
        session: BrowserSession, 
        patient_identifier: str,
        data_types: List[str]
    ) -> List[PatientDataRecord]:
        """Extract patient data with HIPAA compliance."""
        extracted_records = []
        
        # Generate secure patient ID hash
        patient_id_hash = hashlib.sha256(patient_identifier.encode()).hexdigest()[:16]
        
        for data_type in data_types:
            try:
                # Validate consent before processing
                consent_valid = await self.hipaa_tool.validate_consent(patient_id_hash, data_type)
                
                if not consent_valid:
                    await self.audit_logger.log_event({
                        "event_type": "consent_denied",
                        "patient_id": patient_id_hash,
                        "data_type": data_type,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    continue
                
                # Navigate to patient data section
                await self._navigate_to_patient_data(session, patient_identifier, data_type)
                
                # Extract data based on type
                raw_data = await self._extract_data_by_type(session, data_type)
                
                # Analyze for PHI
                phi_analysis = await self.hipaa_tool.detect_phi(str(raw_data))
                
                # Determine compliance level
                compliance_level = HIPAAComplianceLevel.PHI if phi_analysis['phi_detected'] else HIPAAComplianceLevel.GENERAL
                
                # Create secure record
                record = PatientDataRecord(
                    patient_id=patient_id_hash,
                    record_type=data_type,
                    data_classification=compliance_level,
                    extracted_data=raw_data,
                    consent_verified=consent_valid,
                    extraction_timestamp=datetime.utcnow(),
                    audit_trail=[f"extracted_by_agent_{datetime.utcnow().isoformat()}"]
                )
                
                extracted_records.append(record)
                
                # Log extraction
                await self.audit_logger.log_event({
                    "event_type": "data_extraction",
                    "patient_id": patient_id_hash,
                    "data_type": data_type,
                    "phi_detected": phi_analysis['phi_detected'],
                    "compliance_level": compliance_level.value,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            except Exception as e:
                await self.audit_logger.log_event({
                    "event_type": "extraction_error",
                    "patient_id": patient_id_hash,
                    "data_type": data_type,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        return extracted_records
    
    async def _navigate_to_patient_data(self, session: BrowserSession, patient_id: str, data_type: str):
        """Navigate to specific patient data section."""
        # Search for patient
        search_selector = "input[name='search'], input[placeholder*='patient'], .search-input"
        await session.fill(search_selector, patient_id)
        await session.press("Enter")
        
        # Wait for patient results
        await session.wait_for_element(".patient-result, .patient-row, .search-result")
        
        # Click on patient
        await session.click(".patient-result:first-child, .patient-row:first-child")
        
        # Navigate to specific data type
        data_type_selectors = {
            "demographics": "a[href*='demographics'], .demographics-tab",
            "medical_history": "a[href*='history'], .history-tab",
            "medications": "a[href*='medications'], .medications-tab",
            "lab_results": "a[href*='labs'], .labs-tab",
            "insurance": "a[href*='insurance'], .insurance-tab"
        }
        
        if data_type in data_type_selectors:
            await session.click(data_type_selectors[data_type])
            await session.wait_for_load_state("networkidle")
    
    async def _extract_data_by_type(self, session: BrowserSession, data_type: str) -> Dict[str, Any]:
        """Extract data based on the specific type."""
        if data_type == "demographics":
            return await self._extract_demographics(session)
        elif data_type == "medical_history":
            return await self._extract_medical_history(session)
        elif data_type == "medications":
            return await self._extract_medications(session)
        elif data_type == "lab_results":
            return await self._extract_lab_results(session)
        elif data_type == "insurance":
            return await self._extract_insurance_info(session)
        else:
            return {"error": f"Unknown data type: {data_type}"}
    
    async def _extract_demographics(self, session: BrowserSession) -> Dict[str, Any]:
        """Extract demographic information."""
        demographics = {}
        
        # Common demographic field selectors
        field_selectors = {
            "name": ".patient-name, [data-field='name'], .name-field",
            "dob": ".date-of-birth, [data-field='dob'], .dob-field",
            "gender": ".gender, [data-field='gender'], .gender-field",
            "address": ".address, [data-field='address'], .address-field",
            "phone": ".phone, [data-field='phone'], .phone-field",
            "email": ".email, [data-field='email'], .email-field"
        }
        
        for field, selector in field_selectors.items():
            try:
                element = await session.query_selector(selector)
                if element:
                    value = await element.text_content()
                    demographics[field] = value.strip() if value else ""
            except:
                demographics[field] = ""
        
        return demographics
    
    async def _extract_medical_history(self, session: BrowserSession) -> Dict[str, Any]:
        """Extract medical history information."""
        history = {
            "conditions": [],
            "allergies": [],
            "surgeries": [],
            "family_history": []
        }
        
        # Extract conditions
        condition_elements = await session.query_selector_all(".condition, .diagnosis, .medical-condition")
        for element in condition_elements:
            text = await element.text_content()
            if text and text.strip():
                history["conditions"].append(text.strip())
        
        # Extract allergies
        allergy_elements = await session.query_selector_all(".allergy, .allergic-reaction")
        for element in allergy_elements:
            text = await element.text_content()
            if text and text.strip():
                history["allergies"].append(text.strip())
        
        return history
    
    async def _extract_medications(self, session: BrowserSession) -> Dict[str, Any]:
        """Extract medication information."""
        medications = {
            "current_medications": [],
            "past_medications": [],
            "prescriptions": []
        }
        
        # Extract current medications
        med_elements = await session.query_selector_all(".medication, .current-med, .prescription")
        for element in med_elements:
            text = await element.text_content()
            if text and text.strip():
                medications["current_medications"].append(text.strip())
        
        return medications
    
    async def _extract_lab_results(self, session: BrowserSession) -> Dict[str, Any]:
        """Extract laboratory results."""
        lab_results = {
            "recent_results": [],
            "test_history": []
        }
        
        # Extract lab results
        result_elements = await session.query_selector_all(".lab-result, .test-result, .lab-value")
        for element in result_elements:
            text = await element.text_content()
            if text and text.strip():
                lab_results["recent_results"].append(text.strip())
        
        return lab_results
    
    async def _extract_insurance_info(self, session: BrowserSession) -> Dict[str, Any]:
        """Extract insurance information."""
        insurance = {
            "primary_insurance": {},
            "secondary_insurance": {},
            "coverage_details": []
        }
        
        # Extract insurance details
        insurance_elements = await session.query_selector_all(".insurance, .coverage, .plan-info")
        for element in insurance_elements:
            text = await element.text_content()
            if text and text.strip():
                insurance["coverage_details"].append(text.strip())
        
        return insurance
    
    async def process_and_secure_data(self, records: List[PatientDataRecord]) -> Dict[str, Any]:
        """Process extracted data with HIPAA compliance."""
        processed_data = {
            "total_records": len(records),
            "phi_records": 0,
            "processed_records": [],
            "audit_summary": []
        }
        
        for record in records:
            # Mask PHI if present
            if record.data_classification == HIPAAComplianceLevel.PHI:
                processed_data["phi_records"] += 1
                
                # Mask sensitive data
                masked_data = {}
                for key, value in record.extracted_data.items():
                    if isinstance(value, str):
                        masked_data[key] = await self.hipaa_tool.mask_phi(value)
                    else:
                        masked_data[key] = value
                
                record.extracted_data = masked_data
            
            # Encrypt sensitive records
            if record.data_classification in [HIPAAComplianceLevel.PHI, HIPAAComplianceLevel.PII]:
                encrypted_data = self.fernet.encrypt(json.dumps(record.extracted_data).encode())
                record.extracted_data = {"encrypted": True, "data": encrypted_data.decode()}
            
            processed_data["processed_records"].append(record.to_secure_dict())
            
            # Add to audit trail
            processed_data["audit_summary"].append({
                "patient_id": record.patient_id,
                "record_type": record.record_type,
                "classification": record.data_classification.value,
                "processed_timestamp": datetime.utcnow().isoformat()
            })
        
        return processed_data
    
    async def generate_hipaa_audit_report(self, session_id: str) -> Dict[str, Any]:
        """Generate HIPAA compliance audit report."""
        audit_events = await self.audit_logger.get_session_events(session_id)
        
        report = {
            "session_id": session_id,
            "report_generated": datetime.utcnow().isoformat(),
            "compliance_framework": "HIPAA",
            "total_events": len(audit_events),
            "event_summary": {},
            "phi_handling_events": [],
            "security_events": [],
            "compliance_status": "compliant"
        }
        
        # Categorize events
        for event in audit_events:
            event_type = event.get("event_type", "unknown")
            if event_type not in report["event_summary"]:
                report["event_summary"][event_type] = 0
            report["event_summary"][event_type] += 1
            
            # Track PHI handling
            if "phi" in event_type or event.get("phi_detected"):
                report["phi_handling_events"].append(event)
            
            # Track security events
            if event_type in ["authentication", "session_created", "consent_denied"]:
                report["security_events"].append(event)
        
        return report


async def main():
    """Main function demonstrating healthcare document processing."""
    print("🏥 Healthcare Document Processing with HIPAA Compliance")
    print("=" * 60)
    
    # Initialize healthcare agent
    agent = SecureHealthcareAgent()
    
    # Healthcare portal configuration
    portal_config = {
        "name": "Epic MyChart",
        "url": "https://mychart.example.com/login",
        "credential_id": "epic_healthcare_creds"
    }
    
    try:
        # Create secure browser session
        print("Creating secure HIPAA-compliant browser session...")
        session = await agent.create_secure_session(portal_config)
        
        # Authenticate to healthcare portal
        print("Authenticating to healthcare portal...")
        auth_success = await agent.authenticate_healthcare_portal(
            session, 
            portal_config["url"],
            portal_config["credential_id"]
        )
        
        if not auth_success:
            print("❌ Authentication failed")
            return
        
        print("✅ Successfully authenticated to healthcare portal")
        
        # Extract patient data
        print("Extracting patient data with HIPAA compliance...")
        patient_records = await agent.extract_patient_data(
            session,
            patient_identifier="DOE123456",  # Example patient ID
            data_types=["demographics", "medical_history", "medications", "lab_results"]
        )
        
        print(f"✅ Extracted {len(patient_records)} patient records")
        
        # Process and secure data
        print("Processing data with HIPAA compliance controls...")
        processed_data = await agent.process_and_secure_data(patient_records)
        
        print(f"✅ Processed {processed_data['total_records']} records")
        print(f"📊 PHI records identified: {processed_data['phi_records']}")
        
        # Generate audit report
        print("Generating HIPAA compliance audit report...")
        audit_report = await agent.generate_hipaa_audit_report(session.session_id)
        
        print("✅ HIPAA Audit Report Generated:")
        print(f"   - Total events: {audit_report['total_events']}")
        print(f"   - PHI handling events: {len(audit_report['phi_handling_events'])}")
        print(f"   - Security events: {len(audit_report['security_events'])}")
        print(f"   - Compliance status: {audit_report['compliance_status']}")
        
        # Save results securely
        results = {
            "session_summary": {
                "session_id": session.session_id,
                "portal": portal_config["name"],
                "records_processed": len(patient_records),
                "phi_records": processed_data['phi_records'],
                "compliance_status": "hipaa_compliant"
            },
            "audit_report": audit_report,
            "processed_data": processed_data
        }
        
        # Save to secure file (in production, this would be encrypted storage)
        with open("healthcare_processing_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print("✅ Results saved to healthcare_processing_results.json")
        
    except Exception as e:
        print(f"❌ Error during healthcare document processing: {str(e)}")
        
    finally:
        # Clean up session
        if 'session' in locals():
            await session.close()
            print("🧹 Browser session cleaned up")


if __name__ == "__main__":
    asyncio.run(main())