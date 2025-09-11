#!/usr/bin/env python3
"""
Strands PII Utilities
====================

This module provides comprehensive PII (Personally Identifiable Information) detection,
classification, and masking utilities specifically designed for Strands agents working
with AgentCore Browser Tool outputs. It includes advanced pattern recognition,
context-aware masking, and compliance-specific handling.

Features:
- Advanced PII detection with context awareness
- Multi-level masking strategies
- Compliance-specific PII handling (HIPAA, GDPR, PCI DSS)
- Real-time PII scanning of browser tool outputs
- Configurable sensitivity levels
- Audit logging for PII operations
"""

import re
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

# Strands framework imports
from strands import Agent
from strands.tools import tool

# Import AuditLogger from session helpers
from .strands_agentcore_session_helpers import AuditLogger


class EncryptionManager:
    """Simple encryption manager for PII operations."""
    
    def __init__(self):
        pass
    
    def encrypt_pii(self, value: str) -> str:
        """Encrypt PII value."""
        return hashlib.sha256(value.encode()).hexdigest()[:16]
    
    def decrypt_pii(self, encrypted_value: str) -> str:
        """Decrypt PII value (placeholder)."""
        return "[ENCRYPTED_PII]"

# Machine learning imports for advanced detection
try:
    import spacy
    from transformers import pipeline
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False


class PIIType(Enum):
    """Types of PII that can be detected."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    ADDRESS = "address"
    NAME = "name"
    DATE_OF_BIRTH = "date_of_birth"
    DRIVER_LICENSE = "driver_license"
    PASSPORT = "passport"
    BANK_ACCOUNT = "bank_account"
    IP_ADDRESS = "ip_address"
    URL = "url"
    MEDICAL_RECORD = "medical_record"
    FINANCIAL_ACCOUNT = "financial_account"


class SensitivityLevel(Enum):
    """Sensitivity levels for PII."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MaskingStrategy(Enum):
    """Strategies for masking PII."""
    FULL_MASK = "full_mask"  # Replace with X's or [REDACTED]
    PARTIAL_MASK = "partial_mask"  # Show first/last few characters
    HASH_MASK = "hash_mask"  # Replace with hash
    TOKEN_MASK = "token_mask"  # Replace with token like [EMAIL]
    PRESERVE_FORMAT = "preserve_format"  # Keep format but mask content


@dataclass
class SanitizationConfig:
    """Configuration for PII sanitization operations."""
    min_confidence_threshold: float = 0.8
    audit_sensitive_operations: bool = True
    enable_custom_patterns: bool = True
    strict_mode: bool = False
    masking_strategy: MaskingStrategy = MaskingStrategy.TOKEN_MASK
    compliance_level: str = "standard"


@dataclass
class PIIMatch:
    """Represents a detected PII match."""
    pii_type: PIIType
    value: str
    start_pos: int
    end_pos: int
    confidence: float
    sensitivity: SensitivityLevel
    context: str
    compliance_flags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "pii_type": self.pii_type.value,
            "value_hash": hashlib.sha256(self.value.encode()).hexdigest()[:16],
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": self.confidence,
            "sensitivity": self.sensitivity.value,
            "context": self.context[:50] + "..." if len(self.context) > 50 else self.context,
            "compliance_flags": self.compliance_flags
        }


@dataclass
class PIIAnalysisResult:
    """Result of PII analysis."""
    total_matches: int
    matches_by_type: Dict[str, int]
    matches_by_sensitivity: Dict[str, int]
    compliance_violations: List[str]
    risk_score: float
    matches: List[PIIMatch]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "total_matches": self.total_matches,
            "matches_by_type": self.matches_by_type,
            "matches_by_sensitivity": self.matches_by_sensitivity,
            "compliance_violations": self.compliance_violations,
            "risk_score": self.risk_score,
            "matches": [match.to_dict() for match in self.matches]
        }


class PIIPatternLibrary:
    """Library of PII detection patterns."""
    
    def __init__(self):
        self.patterns = {
            PIIType.EMAIL: [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            PIIType.PHONE: [
                r'\b\d{3}-\d{3}-\d{4}\b',  # US format
                r'\b\(\d{3}\)\s?\d{3}-\d{4}\b',  # (123) 456-7890
                r'\b\+\d{1,3}\s?\d{3,14}\b',  # International
                r'\b\d{10}\b'  # 10 digits
            ],
            PIIType.SSN: [
                r'\b\d{3}-\d{2}-\d{4}\b',  # XXX-XX-XXXX
                r'\b\d{9}\b'  # 9 digits
            ],
            PIIType.CREDIT_CARD: [
                r'\b4[0-9]{12}(?:[0-9]{3})?\b',  # Visa
                r'\b5[1-5][0-9]{14}\b',  # MasterCard
                r'\b3[47][0-9]{13}\b',  # American Express
                r'\b3[0-9]{4,}\b',  # Diners Club
                r'\b6(?:011|5[0-9]{2})[0-9]{12}\b',  # Discover
                r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b'  # Generic 16-digit
            ],
            PIIType.ADDRESS: [
                r'\b\d{1,5}\s[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd)\b',
                r'\b[A-Z]{2}\s?\d{5}(?:-\d{4})?\b'  # ZIP code
            ],
            PIIType.DATE_OF_BIRTH: [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
                r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'  # Month DD, YYYY
            ],
            PIIType.DRIVER_LICENSE: [
                r'\b[A-Z]{1,2}\d{6,8}\b',  # State format variations
                r'\bDL\s?#?\s?[A-Z0-9]{8,15}\b'
            ],
            PIIType.PASSPORT: [
                r'\b[A-Z]{2}\d{7}\b',  # US passport
                r'\bPassport\s?#?\s?[A-Z0-9]{6,9}\b'
            ],
            PIIType.BANK_ACCOUNT: [
                r'\b\d{8,17}\b',  # Account numbers
                r'\bAccount\s?#?\s?\d{8,17}\b'
            ],
            PIIType.IP_ADDRESS: [
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b',  # IPv4
                r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'  # IPv6
            ],
            PIIType.URL: [
                r'https?://[^\s<>"{}|\\^`\[\]]+',
                r'www\.[^\s<>"{}|\\^`\[\]]+'
            ]
        }
        
        # Context patterns that increase confidence
        self.context_patterns = {
            PIIType.EMAIL: [r'email', r'e-mail', r'contact', r'@'],
            PIIType.PHONE: [r'phone', r'tel', r'call', r'mobile', r'cell'],
            PIIType.SSN: [r'ssn', r'social security', r'tax id'],
            PIIType.CREDIT_CARD: [r'card', r'credit', r'payment', r'visa', r'mastercard'],
            PIIType.ADDRESS: [r'address', r'street', r'city', r'zip', r'postal'],
            PIIType.DATE_OF_BIRTH: [r'birth', r'dob', r'born', r'age'],
        }
        
        # Compliance mappings
        self.compliance_mappings = {
            PIIType.EMAIL: ["gdpr", "ccpa"],
            PIIType.PHONE: ["gdpr", "ccpa"],
            PIIType.SSN: ["hipaa", "pci_dss", "gdpr"],
            PIIType.CREDIT_CARD: ["pci_dss"],
            PIIType.ADDRESS: ["gdpr", "ccpa"],
            PIIType.DATE_OF_BIRTH: ["hipaa", "gdpr"],
            PIIType.MEDICAL_RECORD: ["hipaa"],
            PIIType.FINANCIAL_ACCOUNT: ["pci_dss", "gdpr"]
        }
        
        # Sensitivity mappings
        self.sensitivity_mappings = {
            PIIType.EMAIL: SensitivityLevel.MEDIUM,
            PIIType.PHONE: SensitivityLevel.MEDIUM,
            PIIType.SSN: SensitivityLevel.CRITICAL,
            PIIType.CREDIT_CARD: SensitivityLevel.CRITICAL,
            PIIType.ADDRESS: SensitivityLevel.HIGH,
            PIIType.DATE_OF_BIRTH: SensitivityLevel.HIGH,
            PIIType.DRIVER_LICENSE: SensitivityLevel.HIGH,
            PIIType.PASSPORT: SensitivityLevel.CRITICAL,
            PIIType.BANK_ACCOUNT: SensitivityLevel.CRITICAL,
            PIIType.IP_ADDRESS: SensitivityLevel.LOW,
            PIIType.URL: SensitivityLevel.LOW,
            PIIType.MEDICAL_RECORD: SensitivityLevel.CRITICAL,
            PIIType.FINANCIAL_ACCOUNT: SensitivityLevel.CRITICAL
        }


class AdvancedPIIDetector:
    """Advanced PII detector using NLP and ML techniques."""
    
    def __init__(self):
        self.nlp_available = ADVANCED_NLP_AVAILABLE
        if self.nlp_available:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.ner_pipeline = pipeline("ner", aggregation_strategy="simple")
            except:
                self.nlp_available = False
    
    async def detect_names(self, text: str) -> List[PIIMatch]:
        """Detect person names using NLP."""
        matches = []
        
        if not self.nlp_available:
            return matches
        
        try:
            # Use spaCy for named entity recognition
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    match = PIIMatch(
                        pii_type=PIIType.NAME,
                        value=ent.text,
                        start_pos=ent.start_char,
                        end_pos=ent.end_char,
                        confidence=0.8,
                        sensitivity=SensitivityLevel.HIGH,
                        context=text[max(0, ent.start_char-20):ent.end_char+20],
                        compliance_flags=["gdpr", "ccpa"]
                    )
                    matches.append(match)
            
            # Use transformer model for additional detection
            ner_results = self.ner_pipeline(text)
            for result in ner_results:
                if result['entity_group'] == 'PER':
                    match = PIIMatch(
                        pii_type=PIIType.NAME,
                        value=result['word'],
                        start_pos=result['start'],
                        end_pos=result['end'],
                        confidence=result['score'],
                        sensitivity=SensitivityLevel.HIGH,
                        context=text[max(0, result['start']-20):result['end']+20],
                        compliance_flags=["gdpr", "ccpa"]
                    )
                    matches.append(match)
        
        except Exception as e:
            logging.warning(f"Advanced name detection failed: {e}")
        
        return matches


class PIIDetector:
    """Main PII detection tool for Strands agents."""
    
    def __init__(self):
        self.pattern_library = PIIPatternLibrary()
        self.advanced_detector = AdvancedPIIDetector()
        self.audit_logger = AuditLogger(service_name="pii_detector")
        self.encryption_manager = EncryptionManager()
    
    async def analyze_text(self, 
                          text: str, 
                          compliance_context: List[str] = None,
                          sensitivity_threshold: SensitivityLevel = SensitivityLevel.LOW) -> PIIAnalysisResult:
        """Analyze text for PII content."""
        if compliance_context is None:
            compliance_context = ["general"]
        
        all_matches = []
        
        # Pattern-based detection
        for pii_type, patterns in self.pattern_library.patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    confidence = self._calculate_confidence(match, text, pii_type)
                    
                    if confidence > 0.5:  # Minimum confidence threshold
                        pii_match = PIIMatch(
                            pii_type=pii_type,
                            value=match.group(),
                            start_pos=match.start(),
                            end_pos=match.end(),
                            confidence=confidence,
                            sensitivity=self.pattern_library.sensitivity_mappings.get(pii_type, SensitivityLevel.MEDIUM),
                            context=text[max(0, match.start()-30):match.end()+30],
                            compliance_flags=self.pattern_library.compliance_mappings.get(pii_type, [])
                        )
                        all_matches.append(pii_match)
        
        # Advanced NLP-based detection
        nlp_matches = await self.advanced_detector.detect_names(text)
        all_matches.extend(nlp_matches)
        
        # Remove duplicates and filter by sensitivity
        filtered_matches = self._filter_and_deduplicate(all_matches, sensitivity_threshold)
        
        # Calculate analysis metrics
        analysis_result = self._calculate_analysis_metrics(filtered_matches, compliance_context)
        
        # Log analysis
        await self.audit_logger.log_event({
            "event_type": "pii_analysis_completed",
            "text_length": len(text),
            "total_matches": analysis_result.total_matches,
            "risk_score": analysis_result.risk_score,
            "compliance_context": compliance_context,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return analysis_result
    
    def _calculate_confidence(self, match, text: str, pii_type: PIIType) -> float:
        """Calculate confidence score for a PII match."""
        base_confidence = 0.7
        
        # Check for context clues
        context_patterns = self.pattern_library.context_patterns.get(pii_type, [])
        context_window = text[max(0, match.start()-50):match.end()+50].lower()
        
        context_boost = 0.0
        for pattern in context_patterns:
            if re.search(pattern, context_window):
                context_boost += 0.1
        
        # Validate specific PII types
        validation_boost = 0.0
        if pii_type == PIIType.CREDIT_CARD:
            if self._validate_credit_card(match.group()):
                validation_boost = 0.2
        elif pii_type == PIIType.EMAIL:
            if self._validate_email(match.group()):
                validation_boost = 0.1
        
        return min(1.0, base_confidence + context_boost + validation_boost)
    
    def _validate_credit_card(self, card_number: str) -> bool:
        """Validate credit card using Luhn algorithm."""
        def luhn_check(card_num):
            def digits_of(n):
                return [int(d) for d in str(n)]
            
            digits = digits_of(card_num.replace(' ', '').replace('-', ''))
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d*2))
            return checksum % 10 == 0
        
        return luhn_check(card_number)
    
    def _validate_email(self, email: str) -> bool:
        """Basic email validation."""
        return '@' in email and '.' in email.split('@')[-1]
    
    def _filter_and_deduplicate(self, 
                               matches: List[PIIMatch], 
                               sensitivity_threshold: SensitivityLevel) -> List[PIIMatch]:
        """Filter matches by sensitivity and remove duplicates."""
        # Define sensitivity order
        sensitivity_order = {
            SensitivityLevel.LOW: 0,
            SensitivityLevel.MEDIUM: 1,
            SensitivityLevel.HIGH: 2,
            SensitivityLevel.CRITICAL: 3
        }
        
        threshold_value = sensitivity_order[sensitivity_threshold]
        
        # Filter by sensitivity
        filtered = [m for m in matches if sensitivity_order[m.sensitivity] >= threshold_value]
        
        # Remove overlapping matches (keep highest confidence)
        deduplicated = []
        for match in sorted(filtered, key=lambda x: x.confidence, reverse=True):
            overlaps = False
            for existing in deduplicated:
                if (match.start_pos < existing.end_pos and 
                    match.end_pos > existing.start_pos):
                    overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(match)
        
        return deduplicated
    
    def _calculate_analysis_metrics(self, 
                                  matches: List[PIIMatch], 
                                  compliance_context: List[str]) -> PIIAnalysisResult:
        """Calculate analysis metrics and risk score."""
        matches_by_type = {}
        matches_by_sensitivity = {}
        compliance_violations = []
        
        for match in matches:
            # Count by type
            type_key = match.pii_type.value
            matches_by_type[type_key] = matches_by_type.get(type_key, 0) + 1
            
            # Count by sensitivity
            sens_key = match.sensitivity.value
            matches_by_sensitivity[sens_key] = matches_by_sensitivity.get(sens_key, 0) + 1
            
            # Check compliance violations
            for compliance_flag in match.compliance_flags:
                if compliance_flag in compliance_context:
                    violation = f"{match.pii_type.value}_in_{compliance_flag}_context"
                    if violation not in compliance_violations:
                        compliance_violations.append(violation)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(matches)
        
        return PIIAnalysisResult(
            total_matches=len(matches),
            matches_by_type=matches_by_type,
            matches_by_sensitivity=matches_by_sensitivity,
            compliance_violations=compliance_violations,
            risk_score=risk_score,
            matches=matches
        )
    
    def _calculate_risk_score(self, matches: List[PIIMatch]) -> float:
        """Calculate overall risk score based on matches."""
        if not matches:
            return 0.0
        
        # Weight by sensitivity
        sensitivity_weights = {
            SensitivityLevel.LOW: 0.1,
            SensitivityLevel.MEDIUM: 0.3,
            SensitivityLevel.HIGH: 0.6,
            SensitivityLevel.CRITICAL: 1.0
        }
        
        total_weight = sum(sensitivity_weights[match.sensitivity] * match.confidence for match in matches)
        max_possible_weight = len(matches) * 1.0  # Maximum if all were critical with 1.0 confidence
        
        return min(1.0, total_weight / max_possible_weight if max_possible_weight > 0 else 0.0)


class PIIMasker:
    """Tool for masking PII in text."""
    
    def __init__(self):
        self.audit_logger = AuditLogger(service_name="pii_masker")
    
    async def mask_text(self, 
                       text: str, 
                       matches: List[PIIMatch],
                       strategy: MaskingStrategy = MaskingStrategy.TOKEN_MASK) -> str:
        """Mask PII in text based on detected matches."""
        masked_text = text
        
        # Sort matches by position (reverse order to maintain positions)
        sorted_matches = sorted(matches, key=lambda x: x.start_pos, reverse=True)
        
        for match in sorted_matches:
            masked_value = self._apply_masking_strategy(match, strategy)
            masked_text = (masked_text[:match.start_pos] + 
                          masked_value + 
                          masked_text[match.end_pos:])
        
        # Log masking operation
        await self.audit_logger.log_event({
            "event_type": "pii_masking_applied",
            "original_length": len(text),
            "masked_length": len(masked_text),
            "matches_masked": len(matches),
            "strategy": strategy.value,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return masked_text
    
    def _apply_masking_strategy(self, match: PIIMatch, strategy: MaskingStrategy) -> str:
        """Apply specific masking strategy to a PII match."""
        value = match.value
        
        if strategy == MaskingStrategy.FULL_MASK:
            return 'X' * len(value)
        
        elif strategy == MaskingStrategy.PARTIAL_MASK:
            if len(value) <= 4:
                return 'X' * len(value)
            elif match.pii_type == PIIType.CREDIT_CARD:
                # Show last 4 digits
                return 'X' * (len(value) - 4) + value[-4:]
            elif match.pii_type == PIIType.EMAIL:
                # Show first char and domain
                parts = value.split('@')
                if len(parts) == 2:
                    return parts[0][0] + 'X' * (len(parts[0]) - 1) + '@' + parts[1]
            else:
                # Show first and last 2 characters
                return value[:2] + 'X' * (len(value) - 4) + value[-2:]
        
        elif strategy == MaskingStrategy.HASH_MASK:
            hash_value = hashlib.sha256(value.encode()).hexdigest()[:8]
            return f"[HASH:{hash_value}]"
        
        elif strategy == MaskingStrategy.TOKEN_MASK:
            token_map = {
                PIIType.EMAIL: "[EMAIL_REDACTED]",
                PIIType.PHONE: "[PHONE_REDACTED]",
                PIIType.SSN: "[SSN_REDACTED]",
                PIIType.CREDIT_CARD: "[CARD_REDACTED]",
                PIIType.ADDRESS: "[ADDRESS_REDACTED]",
                PIIType.NAME: "[NAME_REDACTED]",
                PIIType.DATE_OF_BIRTH: "[DOB_REDACTED]",
                PIIType.DRIVER_LICENSE: "[LICENSE_REDACTED]",
                PIIType.PASSPORT: "[PASSPORT_REDACTED]",
                PIIType.BANK_ACCOUNT: "[ACCOUNT_REDACTED]",
                PIIType.IP_ADDRESS: "[IP_REDACTED]",
                PIIType.URL: "[URL_REDACTED]"
            }
            return token_map.get(match.pii_type, "[PII_REDACTED]")
        
        elif strategy == MaskingStrategy.PRESERVE_FORMAT:
            if match.pii_type == PIIType.PHONE:
                # Preserve phone format
                if '-' in value:
                    return 'XXX-XXX-XXXX'
                elif '(' in value:
                    return '(XXX) XXX-XXXX'
                else:
                    return 'X' * len(value)
            elif match.pii_type == PIIType.SSN:
                return 'XXX-XX-XXXX'
            elif match.pii_type == PIIType.CREDIT_CARD:
                if ' ' in value:
                    return 'XXXX XXXX XXXX XXXX'
                elif '-' in value:
                    return 'XXXX-XXXX-XXXX-XXXX'
                else:
                    return 'X' * len(value)
            else:
                return 'X' * len(value)
        
        return value  # Fallback


class CompliancePIIHandler:
    """Compliance-specific PII handling."""
    
    def __init__(self, 
                 config: Optional[SanitizationConfig] = None,
                 region: str = "us-east-1",
                 session_id: Optional[str] = None,
                 agent_id: Optional[str] = None):

        self.config = config or SanitizationConfig()
        self.region = region
        self.session_id = session_id or f"session-{uuid.uuid4()}"
        self.agent_id = agent_id or f"agent-{uuid.uuid4()}"
        self.detector = PIIDetector()
        self.masker = PIIMasker()
        self.audit_logger = AuditLogger(service_name="compliance_pii_handler")
    
    async def process_for_hipaa(self, text: str) -> Dict[str, Any]:
        """Process text for HIPAA compliance."""
        analysis = await self.detector.analyze_text(
            text, 
            compliance_context=["hipaa"],
            sensitivity_threshold=SensitivityLevel.HIGH
        )
        
        # Use token masking for HIPAA
        masked_text = await self.masker.mask_text(
            text, 
            analysis.matches, 
            MaskingStrategy.TOKEN_MASK
        )
        
        return {
            "original_text": text,
            "masked_text": masked_text,
            "analysis": analysis.to_dict(),
            "compliance": "hipaa"
        }
    
    async def process_for_pci_dss(self, text: str) -> Dict[str, Any]:
        """Process text for PCI DSS compliance."""
        analysis = await self.detector.analyze_text(
            text, 
            compliance_context=["pci_dss"],
            sensitivity_threshold=SensitivityLevel.CRITICAL
        )
        
        # Use full masking for PCI DSS
        masked_text = await self.masker.mask_text(
            text, 
            analysis.matches, 
            MaskingStrategy.FULL_MASK
        )
        
        return {
            "original_text": text,
            "masked_text": masked_text,
            "analysis": analysis.to_dict(),
            "compliance": "pci_dss"
        }
    
    async def process_for_gdpr(self, text: str) -> Dict[str, Any]:
        """Process text for GDPR compliance."""
        analysis = await self.detector.analyze_text(
            text, 
            compliance_context=["gdpr"],
            sensitivity_threshold=SensitivityLevel.MEDIUM
        )
        
        # Use partial masking for GDPR (allows for data minimization)
        masked_text = await self.masker.mask_text(
            text, 
            analysis.matches, 
            MaskingStrategy.PARTIAL_MASK
        )
        
        return {
            "original_text": text,
            "masked_text": masked_text,
            "analysis": analysis.to_dict(),
            "compliance": "gdpr"
        }
    
    def detect_pii(self, text: str) -> List[PIIMatch]:
        """Detect PII in text (synchronous wrapper)."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._detect_pii_async(text))
                    result = future.result()
            else:
                result = loop.run_until_complete(self._detect_pii_async(text))
        except RuntimeError:
            # No event loop, create one
            result = asyncio.run(self._detect_pii_async(text))
        
        return result.matches
    
    async def _detect_pii_async(self, text: str) -> PIIAnalysisResult:
        """Async PII detection."""
        return await self.detector.analyze_text(
            text, 
            compliance_context=["general"],
            sensitivity_threshold=SensitivityLevel.LOW
        )
    
    def mask_sensitive_data(self, text: str):
        """Mask sensitive data in text (synchronous wrapper)."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._mask_sensitive_data_async(text))
                    result = future.result()
            else:
                result = loop.run_until_complete(self._mask_sensitive_data_async(text))
        except RuntimeError:
            result = asyncio.run(self._mask_sensitive_data_async(text))
        
        return result
    
    async def _mask_sensitive_data_async(self, text: str) -> str:
        """Async sensitive data masking."""
        analysis = await self.detector.analyze_text(text)
        masker = PIIMasker()
        return await masker.mask_text(text, analysis.matches, MaskingStrategy.TOKEN_MASK)


# Convenience functions for common use cases

async def scan_browser_output(output: str, compliance_context: str = "general") -> PIIAnalysisResult:
    """Scan browser tool output for PII."""
    detector = PIIDetector()
    return await detector.analyze_text(output, [compliance_context])

async def mask_browser_output(output: str, 
                            compliance_context: str = "general",
                            strategy: MaskingStrategy = MaskingStrategy.TOKEN_MASK) -> str:
    """Mask PII in browser tool output."""
    detector = PIIDetector()
    masker = PIIMasker()
    
    analysis = await detector.analyze_text(output, [compliance_context])
    return await masker.mask_text(output, analysis.matches, strategy)

async def process_sensitive_output(output: str, compliance_framework: str) -> Dict[str, Any]:
    """Process browser output for specific compliance framework."""
    handler = CompliancePIIHandler()
    
    if compliance_framework.lower() == "hipaa":
        return await handler.process_for_hipaa(output)
    elif compliance_framework.lower() == "pci_dss":
        return await handler.process_for_pci_dss(output)
    elif compliance_framework.lower() == "gdpr":
        return await handler.process_for_gdpr(output)
    else:
        # Default processing
        detector = PIIDetector()
        masker = PIIMasker()
        analysis = await detector.analyze_text(output)
        masked_text = await masker.mask_text(output, analysis.matches)
        
        return {
            "original_text": output,
            "masked_text": masked_text,
            "analysis": analysis.to_dict(),
            "compliance": "general"
        }


# Example usage
async def example_usage():
    """Example of how to use the PII utilities."""
    sample_text = """
    Contact John Doe at john.doe@example.com or call 555-123-4567.
    His SSN is 123-45-6789 and credit card is 4532-1234-5678-9012.
    Address: 123 Main Street, Anytown, NY 12345
    """
    
    # Detect PII
    detector = PIIDetector()
    analysis = await detector.analyze_text(sample_text, ["gdpr"])
    
    print(f"Found {analysis.total_matches} PII matches")
    print(f"Risk score: {analysis.risk_score}")
    
    # Mask PII
    masker = PIIMasker()
    masked_text = await masker.mask_text(sample_text, analysis.matches, MaskingStrategy.TOKEN_MASK)
    
    print(f"Masked text: {masked_text}")
    
    # Process for specific compliance
    result = await process_sensitive_output(sample_text, "gdpr")
    print(f"GDPR processed: {result['masked_text']}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())