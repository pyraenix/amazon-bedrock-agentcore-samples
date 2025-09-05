"""
Browser-Use PII Masking Integration

This module provides browser-use Agent extensions for automatic PII detection in web forms,
masking functions that work with browser-use's screenshot and DOM analysis, and validation
functions to ensure PII is properly masked in browser-use operations.

Integrates with AgentCore Browser Tool's micro-VM isolation for secure PII handling.
"""

import re
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import base64
from pathlib import Path

# Import from the same directory - use relative import without the dot
import sys
import os
sys.path.append(os.path.dirname(__file__))

from browseruse_sensitive_data_handler import (
    BrowserUseSensitiveDataHandler,
    PIIType,
    ComplianceFramework,
    DetectionResult,
    DataClassification
)


@dataclass
class BrowserElementPII:
    """PII detection result for browser elements."""
    element_id: Optional[str]
    element_type: str  # input, textarea, div, etc.
    element_name: Optional[str]
    element_value: str
    pii_detections: List[DetectionResult]
    masked_value: str
    xpath: Optional[str] = None
    css_selector: Optional[str] = None
    screenshot_region: Optional[Dict[str, int]] = None  # x, y, width, height


@dataclass
class FormPIIAnalysis:
    """Complete PII analysis of a web form."""
    form_id: Optional[str]
    form_action: Optional[str]
    form_method: str
    elements_with_pii: List[BrowserElementPII]
    total_pii_count: int
    highest_classification: DataClassification
    compliance_violations: List[Dict[str, Any]]
    masking_recommendations: List[str]


class BrowserUsePIIMasking:
    """
    Browser-use Agent extensions for automatic PII detection and masking.
    
    Provides integration with browser-use's screenshot and DOM analysis capabilities
    to detect, mask, and validate PII handling in web automation workflows.
    """
    
    def __init__(self, 
                 compliance_frameworks: Optional[List[ComplianceFramework]] = None,
                 enable_screenshot_analysis: bool = True,
                 enable_dom_analysis: bool = True):
        """
        Initialize the browser-use PII masking integration.
        
        Args:
            compliance_frameworks: Compliance frameworks to enforce
            enable_screenshot_analysis: Enable screenshot-based PII detection
            enable_dom_analysis: Enable DOM-based PII detection
        """
        self.logger = logging.getLogger(__name__)
        self.compliance_frameworks = compliance_frameworks or []
        self.enable_screenshot_analysis = enable_screenshot_analysis
        self.enable_dom_analysis = enable_dom_analysis
        
        # Initialize the core sensitive data handler
        self.pii_handler = BrowserUseSensitiveDataHandler(compliance_frameworks)
        
        # PII masking callbacks for browser-use integration
        self.pre_action_callbacks: List[Callable] = []
        self.post_action_callbacks: List[Callable] = []
        
        self.logger.info("Initialized browser-use PII masking integration")
    
    async def analyze_page_for_pii(self, 
                                 page_content: Dict[str, Any],
                                 screenshot_data: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Analyze a web page for PII in both DOM and screenshot.
        
        Args:
            page_content: Page content from browser-use (DOM, forms, inputs)
            screenshot_data: Optional screenshot data for visual analysis
            
        Returns:
            Comprehensive PII analysis results
        """
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'dom_analysis': None,
            'screenshot_analysis': None,
            'combined_results': None
        }
        
        # DOM-based PII analysis
        if self.enable_dom_analysis and 'forms' in page_content:
            dom_analysis = await self._analyze_dom_for_pii(page_content)
            analysis_results['dom_analysis'] = dom_analysis
        
        # Screenshot-based PII analysis
        if self.enable_screenshot_analysis and screenshot_data:
            screenshot_analysis = await self._analyze_screenshot_for_pii(screenshot_data)
            analysis_results['screenshot_analysis'] = screenshot_analysis
        
        # Combine results
        combined_results = self._combine_analysis_results(
            analysis_results['dom_analysis'],
            analysis_results['screenshot_analysis']
        )
        analysis_results['combined_results'] = combined_results
        
        self.logger.info(f"Page PII analysis completed: {combined_results.get('total_pii_count', 0)} items detected")
        return analysis_results
    
    async def _analyze_dom_for_pii(self, page_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze DOM content for PII in form fields and text content.
        
        Args:
            page_content: Page content with forms and elements
            
        Returns:
            DOM PII analysis results
        """
        forms_analysis = []
        total_pii_count = 0
        
        for form_data in page_content.get('forms', []):
            form_analysis = await self._analyze_form_for_pii(form_data)
            forms_analysis.append(form_analysis)
            total_pii_count += form_analysis.total_pii_count
        
        # Analyze other text content
        text_content_pii = []
        if 'text_content' in page_content:
            text_detections = self.pii_handler.detect_pii(page_content['text_content'])
            text_content_pii = text_detections
            total_pii_count += len(text_detections)
        
        return {
            'forms_analysis': forms_analysis,
            'text_content_pii': text_content_pii,
            'total_pii_count': total_pii_count,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    async def _analyze_form_for_pii(self, form_data: Dict[str, Any]) -> FormPIIAnalysis:
        """
        Analyze a specific form for PII in its fields.
        
        Args:
            form_data: Form data including fields and metadata
            
        Returns:
            Form PII analysis results
        """
        elements_with_pii = []
        total_pii_count = 0
        highest_classification = DataClassification.PUBLIC
        compliance_violations = []
        
        # Analyze form fields
        for field in form_data.get('fields', []):
            element_pii = await self._analyze_form_field_for_pii(field)
            
            if element_pii.pii_detections:
                elements_with_pii.append(element_pii)
                total_pii_count += len(element_pii.pii_detections)
                
                # Update highest classification
                field_classification = self._get_field_classification(element_pii.pii_detections)
                if self._is_higher_classification(field_classification, highest_classification):
                    highest_classification = field_classification
                
                # Check compliance violations
                field_violations = self._check_compliance_violations(element_pii.pii_detections)
                compliance_violations.extend(field_violations)
        
        # Generate masking recommendations
        masking_recommendations = self._generate_masking_recommendations(
            elements_with_pii, highest_classification, compliance_violations
        )
        
        return FormPIIAnalysis(
            form_id=form_data.get('id'),
            form_action=form_data.get('action'),
            form_method=form_data.get('method', 'GET'),
            elements_with_pii=elements_with_pii,
            total_pii_count=total_pii_count,
            highest_classification=highest_classification,
            compliance_violations=compliance_violations,
            masking_recommendations=masking_recommendations
        )
    
    async def _analyze_form_field_for_pii(self, field_data: Dict[str, Any]) -> BrowserElementPII:
        """
        Analyze a specific form field for PII.
        
        Args:
            field_data: Field data including type, name, value, etc.
            
        Returns:
            Element PII analysis results
        """
        field_value = str(field_data.get('value', ''))
        field_name = field_data.get('name', '')
        field_type = field_data.get('type', 'text')
        
        # Detect PII in field value
        pii_detections = self.pii_handler.detect_pii(field_value, field_name)
        
        # Create masked value - keep original value for validation purposes
        # The masked value shows what it SHOULD be, not what it currently is
        masked_value = field_value  # Start with original
        if pii_detections:
            masked_value, _ = self.pii_handler.mask_text(field_value, field_name)
        
        return BrowserElementPII(
            element_id=field_data.get('id'),
            element_type=field_type,
            element_name=field_name,
            element_value=field_value,  # This is the original unmasked value
            pii_detections=pii_detections,
            masked_value=masked_value,  # This is what it should be masked to
            xpath=field_data.get('xpath'),
            css_selector=field_data.get('css_selector'),
            screenshot_region=field_data.get('screenshot_region')
        )
    
    async def _analyze_screenshot_for_pii(self, screenshot_data: bytes) -> Dict[str, Any]:
        """
        Analyze screenshot for visible PII using OCR and pattern matching.
        
        Args:
            screenshot_data: Screenshot image data
            
        Returns:
            Screenshot PII analysis results
        """
        # Note: This is a placeholder for screenshot analysis
        # In a real implementation, you would use OCR libraries like pytesseract
        # to extract text from screenshots and then analyze for PII
        
        self.logger.info("Screenshot PII analysis (placeholder implementation)")
        
        return {
            'ocr_text_extracted': False,
            'pii_detected_in_screenshot': [],
            'visual_masking_required': [],
            'analysis_timestamp': datetime.now().isoformat(),
            'note': 'Screenshot analysis requires OCR implementation'
        }
    
    def _combine_analysis_results(self, 
                                dom_analysis: Optional[Dict[str, Any]], 
                                screenshot_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine DOM and screenshot analysis results.
        
        Args:
            dom_analysis: DOM-based PII analysis results
            screenshot_analysis: Screenshot-based PII analysis results
            
        Returns:
            Combined analysis results
        """
        total_pii_count = 0
        highest_classification = DataClassification.PUBLIC
        all_violations = []
        
        if dom_analysis:
            total_pii_count += dom_analysis.get('total_pii_count', 0)
            
            # Find highest classification from forms
            for form_analysis in dom_analysis.get('forms_analysis', []):
                if self._is_higher_classification(form_analysis.highest_classification, highest_classification):
                    highest_classification = form_analysis.highest_classification
                all_violations.extend(form_analysis.compliance_violations)
        
        if screenshot_analysis:
            # Add screenshot-based detections when implemented
            pass
        
        return {
            'total_pii_count': total_pii_count,
            'highest_classification': highest_classification,
            'compliance_violations': all_violations,
            'requires_secure_handling': highest_classification in [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED],
            'analysis_complete': True
        }
    
    def _get_field_classification(self, detections: List[DetectionResult]) -> DataClassification:
        """Get the highest classification level for detected PII."""
        if not detections:
            return DataClassification.PUBLIC
        
        high_sensitivity_types = {PIIType.SSN, PIIType.CREDIT_CARD, PIIType.MEDICAL_RECORD}
        medium_sensitivity_types = {PIIType.EMAIL, PIIType.PHONE, PIIType.DATE_OF_BIRTH}
        
        detected_types = {d.pii_type for d in detections}
        
        if detected_types & high_sensitivity_types:
            return DataClassification.RESTRICTED
        elif detected_types & medium_sensitivity_types:
            return DataClassification.CONFIDENTIAL
        else:
            return DataClassification.INTERNAL
    
    def _is_higher_classification(self, 
                                classification1: DataClassification, 
                                classification2: DataClassification) -> bool:
        """Check if classification1 is higher than classification2."""
        classification_order = {
            DataClassification.PUBLIC: 0,
            DataClassification.INTERNAL: 1,
            DataClassification.CONFIDENTIAL: 2,
            DataClassification.RESTRICTED: 3,
            DataClassification.TOP_SECRET: 4
        }
        
        return classification_order[classification1] > classification_order[classification2]
    
    def _check_compliance_violations(self, detections: List[DetectionResult]) -> List[Dict[str, Any]]:
        """Check for compliance violations in PII detections."""
        violations = []
        
        for detection in detections:
            for framework in self.compliance_frameworks:
                if framework in detection.compliance_impact and detection.confidence > 0.8:
                    violations.append({
                        'framework': framework.value,
                        'pii_type': detection.pii_type.value,
                        'confidence': detection.confidence,
                        'severity': 'high' if detection.confidence > 0.9 else 'medium'
                    })
        
        return violations
    
    def _generate_masking_recommendations(self, 
                                        elements_with_pii: List[BrowserElementPII],
                                        highest_classification: DataClassification,
                                        violations: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for PII masking."""
        recommendations = []
        
        if elements_with_pii:
            recommendations.append("Implement field-level PII masking")
            recommendations.append("Use AgentCore's secure input handling")
        
        if highest_classification == DataClassification.RESTRICTED:
            recommendations.extend([
                "Enable micro-VM isolation for form processing",
                "Implement comprehensive audit logging",
                "Use encrypted data transmission",
                "Enable session recording for compliance"
            ])
        
        if violations:
            frameworks = {v['framework'] for v in violations}
            for framework in frameworks:
                recommendations.append(f"Ensure {framework.upper()} compliance measures")
        
        return recommendations
    
    async def mask_form_inputs(self, 
                             form_data: Dict[str, Any],
                             masking_strategy: str = "partial") -> Dict[str, Any]:
        """
        Apply PII masking to form input data.
        
        Args:
            form_data: Form data to mask
            masking_strategy: Masking strategy ("full", "partial", "preserve_format")
            
        Returns:
            Masked form data
        """
        masked_form_data = form_data.copy()
        masking_log = []
        
        for field_name, field_value in form_data.items():
            if isinstance(field_value, str):
                # Detect PII in field value
                detections = self.pii_handler.detect_pii(field_value, field_name)
                
                if detections:
                    # Apply masking based on strategy
                    if masking_strategy == "full":
                        masked_value = "*" * len(field_value)
                    elif masking_strategy == "preserve_format":
                        masked_value, _ = self.pii_handler.mask_text(field_value, field_name)
                    else:  # partial
                        masked_value, _ = self.pii_handler.mask_text(field_value, field_name)
                    
                    masked_form_data[field_name] = masked_value
                    
                    masking_log.append({
                        'field_name': field_name,
                        'pii_types': [d.pii_type.value for d in detections],
                        'masking_applied': True,
                        'strategy': masking_strategy
                    })
        
        return {
            'masked_data': masked_form_data,
            'masking_log': masking_log,
            'timestamp': datetime.now().isoformat()
        }
    
    def register_pre_action_callback(self, callback: Callable) -> None:
        """Register a callback to be called before browser actions."""
        self.pre_action_callbacks.append(callback)
    
    def register_post_action_callback(self, callback: Callable) -> None:
        """Register a callback to be called after browser actions."""
        self.post_action_callbacks.append(callback)
    
    async def execute_pre_action_callbacks(self, action_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all pre-action callbacks."""
        results = []
        for callback in self.pre_action_callbacks:
            try:
                result = await callback(action_context)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Pre-action callback failed: {e}")
                results.append({'error': str(e)})
        
        return {'callback_results': results}
    
    async def execute_post_action_callbacks(self, action_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all post-action callbacks."""
        results = []
        for callback in self.post_action_callbacks:
            try:
                result = await callback(action_context)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Post-action callback failed: {e}")
                results.append({'error': str(e)})
        
        return {'callback_results': results}


class BrowserUsePIIValidator:
    """
    Validation functions to ensure PII is properly masked in browser-use operations.
    """
    
    def __init__(self, pii_masking: BrowserUsePIIMasking):
        """
        Initialize the PII validator.
        
        Args:
            pii_masking: PII masking instance to use for validation
        """
        self.logger = logging.getLogger(__name__)
        self.pii_masking = pii_masking
        self.validation_history: List[Dict[str, Any]] = []
    
    async def validate_page_pii_handling(self, 
                                       page_content: Dict[str, Any],
                                       expected_masking: bool = True) -> Dict[str, Any]:
        """
        Validate that PII is properly handled on a page.
        
        Args:
            page_content: Page content to validate
            expected_masking: Whether PII should be masked
            
        Returns:
            Validation results
        """
        validation_result = {
            'timestamp': datetime.now().isoformat(),
            'validation_passed': True,
            'issues_found': [],
            'recommendations': [],
            'pii_analysis': None
        }
        
        # Analyze page for PII
        pii_analysis = await self.pii_masking.analyze_page_for_pii(page_content)
        validation_result['pii_analysis'] = pii_analysis
        
        combined_results = pii_analysis.get('combined_results', {})
        total_pii_count = combined_results.get('total_pii_count', 0)
        
        if total_pii_count > 0:
            if expected_masking:
                # Check if PII is properly masked
                masking_issues = await self._check_pii_masking(pii_analysis)
                if masking_issues:
                    validation_result['validation_passed'] = False
                    validation_result['issues_found'].extend(masking_issues)
            else:
                # PII detected but masking not expected - flag as potential issue
                validation_result['issues_found'].append({
                    'type': 'unexpected_pii',
                    'message': f'Found {total_pii_count} PII items but masking not expected',
                    'severity': 'medium'
                })
        
        # Generate recommendations
        if not validation_result['validation_passed']:
            validation_result['recommendations'] = self._generate_validation_recommendations(
                validation_result['issues_found']
            )
        
        # Store validation history
        self.validation_history.append(validation_result)
        
        self.logger.info(f"PII validation completed: {'PASSED' if validation_result['validation_passed'] else 'FAILED'}")
        return validation_result
    
    async def _check_pii_masking(self, pii_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if detected PII is properly masked."""
        issues = []
        
        dom_analysis = pii_analysis.get('dom_analysis', {})
        
        # Check form fields
        for form_analysis in dom_analysis.get('forms_analysis', []):
            for element in form_analysis.elements_with_pii:
                # Check if PII is masked in the element value
                # If original value equals masked value, then no masking was applied
                if element.element_value == element.masked_value:
                    # No masking applied
                    issues.append({
                        'type': 'unmasked_pii',
                        'element_id': element.element_id,
                        'element_name': element.element_name,
                        'pii_types': [d.pii_type.value for d in element.pii_detections],
                        'message': 'PII detected but not masked',
                        'severity': 'high'
                    })
                else:
                    # Check for high-confidence PII that should be fully masked
                    for detection in element.pii_detections:
                        if detection.confidence > 0.9 and detection.pii_type in {PIIType.SSN, PIIType.CREDIT_CARD}:
                            # If the original value still appears in the masked value, it's insufficient
                            if detection.value in element.masked_value:
                                issues.append({
                                    'type': 'insufficient_masking',
                                    'element_id': element.element_id,
                                    'pii_type': detection.pii_type.value,
                                    'message': 'High-sensitivity PII not fully masked',
                                    'severity': 'critical'
                                })
        
        return issues
    
    def _generate_validation_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on validation issues."""
        recommendations = []
        
        issue_types = {issue['type'] for issue in issues}
        
        if 'unmasked_pii' in issue_types:
            recommendations.extend([
                "Implement automatic PII masking for form inputs",
                "Use AgentCore's secure input handling",
                "Enable field-level encryption for sensitive data"
            ])
        
        if 'insufficient_masking' in issue_types:
            recommendations.extend([
                "Increase masking strength for high-sensitivity PII",
                "Use full masking for SSN and credit card numbers",
                "Implement format-preserving encryption"
            ])
        
        if 'unexpected_pii' in issue_types:
            recommendations.extend([
                "Review data handling policies",
                "Implement PII detection in data pipelines",
                "Add compliance validation checks"
            ])
        
        return recommendations
    
    def get_validation_history(self) -> List[Dict[str, Any]]:
        """Get the validation history."""
        return self.validation_history.copy()
    
    def clear_validation_history(self) -> None:
        """Clear the validation history."""
        self.validation_history.clear()
        self.logger.info("Validation history cleared")


# Convenience functions for browser-use integration
async def analyze_browser_page_pii(page_content: Dict[str, Any],
                                 compliance_frameworks: Optional[List[ComplianceFramework]] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze a browser page for PII.
    
    Args:
        page_content: Page content from browser-use
        compliance_frameworks: Compliance frameworks to consider
        
    Returns:
        PII analysis results
    """
    pii_masking = BrowserUsePIIMasking(compliance_frameworks)
    return await pii_masking.analyze_page_for_pii(page_content)


async def mask_browser_form_data(form_data: Dict[str, Any],
                               compliance_frameworks: Optional[List[ComplianceFramework]] = None) -> Dict[str, Any]:
    """
    Convenience function to mask PII in browser form data.
    
    Args:
        form_data: Form data to mask
        compliance_frameworks: Compliance frameworks to consider
        
    Returns:
        Masked form data with logging
    """
    pii_masking = BrowserUsePIIMasking(compliance_frameworks)
    return await pii_masking.mask_form_inputs(form_data)


async def validate_browser_pii_handling(page_content: Dict[str, Any],
                                      compliance_frameworks: Optional[List[ComplianceFramework]] = None) -> Dict[str, Any]:
    """
    Convenience function to validate PII handling in browser operations.
    
    Args:
        page_content: Page content to validate
        compliance_frameworks: Compliance frameworks to consider
        
    Returns:
        Validation results
    """
    pii_masking = BrowserUsePIIMasking(compliance_frameworks)
    validator = BrowserUsePIIValidator(pii_masking)
    return await validator.validate_page_pii_handling(page_content)


# Example usage
if __name__ == "__main__":
    async def example_usage():
        """Example usage of browser-use PII masking integration."""
        
        # Example page content from browser-use
        sample_page_content = {
            'forms': [{
                'id': 'patient-form',
                'action': '/submit-patient-info',
                'method': 'POST',
                'fields': [
                    {
                        'id': 'ssn-field',
                        'name': 'patient_ssn',
                        'type': 'text',
                        'value': '123-45-6789',
                        'xpath': '//input[@name="patient_ssn"]'
                    },
                    {
                        'id': 'email-field',
                        'name': 'patient_email',
                        'type': 'email',
                        'value': 'john.doe@email.com',
                        'xpath': '//input[@name="patient_email"]'
                    }
                ]
            }],
            'text_content': 'Patient information form for medical records'
        }
        
        # Initialize PII masking with HIPAA compliance
        pii_masking = BrowserUsePIIMasking([ComplianceFramework.HIPAA])
        
        # Analyze page for PII
        analysis_results = await pii_masking.analyze_page_for_pii(sample_page_content)
        print("PII Analysis Results:")
        print(json.dumps(analysis_results, indent=2, default=str))
        
        # Validate PII handling
        validator = BrowserUsePIIValidator(pii_masking)
        validation_results = await validator.validate_page_pii_handling(sample_page_content)
        print("\nValidation Results:")
        print(json.dumps(validation_results, indent=2, default=str))
    
    # Run example
    asyncio.run(example_usage())