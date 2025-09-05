"""
Strands CAPTCHA Handling Framework

This module provides a complete, consolidated implementation of the Strands CAPTCHA handling framework,
including all core components: detection, solving, orchestration, state management, and error handling.

Usage:
    from strands_captcha_framework import CaptchaHandlingAgent, CaptchaDetectionTool, CaptchaSolvingTool
    
    # Create and configure agent
    agent = CaptchaHandlingAgent()
    result = await agent.handle_captcha_workflow(page_url, task_description)
"""

import asyncio
import time
import json
import base64
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, replace
from enum import Enum
import logging

# Strands framework imports (these would be actual imports in real implementation)
try:
    from strands import Agent, Tool, ToolResult, ToolParameter, AgentConfig
    from strands.workflows import WorkflowContext
except ImportError:
    # Mock classes for tutorial purposes
    class Agent:
        def __init__(self, config=None): pass
        def get_tool(self, name): return None
        
    class Tool:
        def __init__(self): pass
        async def execute(self, **kwargs): return ToolResult(success=True, data={})
        
    class ToolResult:
        def __init__(self, success=True, data=None, error=None, error_code=None, message=None, metadata=None):
            self.success = success
            self.data = data or {}
            self.error = error
            self.error_code = error_code
            self.message = message
            self.metadata = metadata or {}
    
    class ToolParameter:
        def __init__(self, name, type, description, required=False, default=None):
            self.name = name
            self.type = type
            self.description = description
            self.required = required
            self.default = default
    
    class AgentConfig:
        def __init__(self, **kwargs): pass
    
    class WorkflowContext:
        def __init__(self, data=None):
            self._data = data or {}
        def get(self, key, default=None): return self._data.get(key, default)
        def set(self, key, value): self._data[key] = value
        def update(self, data): self._data.update(data)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CORE FRAMEWORK CLASSES AND ENUMS
# =============================================================================

class CaptchaType(Enum):
    """Enumeration of supported CAPTCHA types"""
    TEXT_CAPTCHA = "text_captcha"
    RECAPTCHA_V2 = "recaptcha_v2"
    RECAPTCHA_V3 = "recaptcha_v3"
    HCAPTCHA = "hcaptcha"
    IMAGE_SELECTION = "image_selection"
    MATHEMATICAL = "mathematical"
    TURNSTILE = "turnstile"
    FUNCAPTCHA = "funcaptcha"
    GENERIC = "generic"

class WorkflowPhase(Enum):
    """Enumeration of workflow phases"""
    INITIALIZATION = "initialization"
    DETECTION = "detection"
    ANALYSIS = "analysis"
    SOLUTION = "solution"
    SUBMISSION = "submission"
    VERIFICATION = "verification"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class CaptchaData:
    """Data structure for CAPTCHA information"""
    captcha_type: CaptchaType
    element_bounds: Dict[str, float]
    screenshot: Optional[str] = None
    selector: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None
    complexity_score: float = 0.0
    confidence: float = 0.0

@dataclass
class WorkflowState:
    """Comprehensive workflow state management"""
    workflow_id: str
    created_at: datetime
    current_phase: WorkflowPhase
    page_url: str
    task_description: str
    
    # Progress tracking
    completed_phases: List[WorkflowPhase]
    current_step_data: Dict[str, Any]
    error_history: List[Dict[str, Any]]
    
    # Service coordination
    agentcore_session_id: Optional[str] = None
    bedrock_model_sessions: Dict[str, Any] = None
    
    # Performance metrics
    performance_metrics: Dict[str, float] = None
    resource_usage: Dict[str, Any] = None
    
    # Recovery and retry
    retry_count: int = 0
    max_retries: int = 3
    recovery_checkpoints: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.bedrock_model_sessions is None:
            self.bedrock_model_sessions = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.resource_usage is None:
            self.resource_usage = {}
        if self.recovery_checkpoints is None:
            self.recovery_checkpoints = []

# =============================================================================
# EXCEPTION CLASSES
# =============================================================================

class CaptchaFrameworkError(Exception):
    """Base exception for CAPTCHA framework errors"""
    pass

class CaptchaDetectionError(CaptchaFrameworkError):
    """Exception raised during CAPTCHA detection"""
    pass

class CaptchaSolvingError(CaptchaFrameworkError):
    """Exception raised during CAPTCHA solving"""
    pass

class WorkflowError(CaptchaFrameworkError):
    """Exception raised during workflow execution"""
    pass

class StateManagementError(CaptchaFrameworkError):
    """Exception raised during state management operations"""
    pass

# =============================================================================
# CAPTCHA DETECTION TOOL
# =============================================================================

class CaptchaDetectionTool(Tool):
    """
    Strands tool for detecting CAPTCHAs using AgentCore Browser Tool.
    
    This tool implements comprehensive CAPTCHA detection with multiple strategies
    and validation mechanisms for high accuracy and low false positives.
    """
    
    name = "captcha_detector"
    description = "Detects various types of CAPTCHAs on web pages using AgentCore Browser Tool"
    version = "1.0.0"
    
    parameters = [
        ToolParameter(
            name="page_url",
            type="string",
            description="URL of the page to scan for CAPTCHAs",
            required=True
        ),
        ToolParameter(
            name="detection_strategy",
            type="string", 
            description="Detection strategy: 'comprehensive', 'recaptcha_focused', 'hcaptcha_focused'",
            required=False,
            default="comprehensive"
        ),
        ToolParameter(
            name="timeout",
            type="integer",
            description="Timeout in seconds for detection operation",
            required=False,
            default=30
        )
    ]
    
    def __init__(self, agentcore_client=None):
        super().__init__()
        self.agentcore = agentcore_client
        self.detection_strategies = self._initialize_detection_strategies()
        
    async def execute(self, **kwargs) -> ToolResult:
        """Execute CAPTCHA detection with comprehensive error handling"""
        
        try:
            page_url = kwargs.get('page_url')
            detection_strategy = kwargs.get('detection_strategy', 'comprehensive')
            timeout = kwargs.get('timeout', 30)
            
            if not page_url:
                return ToolResult(
                    success=False,
                    error="page_url parameter is required",
                    error_code="MISSING_PARAMETER"
                )
            
            logger.info(f"Starting CAPTCHA detection for {page_url} using {detection_strategy} strategy")
            
            detection_result = await asyncio.wait_for(
                self._execute_detection_workflow(page_url, detection_strategy),
                timeout=timeout
            )
            
            return ToolResult(
                success=True,
                data=detection_result,
                message=f"CAPTCHA detection completed for {page_url}",
                metadata={
                    'strategy_used': detection_strategy,
                    'execution_time': detection_result.get('execution_time'),
                    'captchas_found': len(detection_result.get('detected_captchas', []))
                }
            )
            
        except asyncio.TimeoutError:
            logger.error(f"CAPTCHA detection timed out after {timeout} seconds")
            return ToolResult(
                success=False,
                error=f"Detection operation timed out after {timeout} seconds",
                error_code="DETECTION_TIMEOUT"
            )
            
        except Exception as e:
            logger.error(f"CAPTCHA detection failed: {str(e)}")
            return ToolResult(
                success=False,
                error=f"Detection operation failed: {str(e)}",
                error_code="DETECTION_ERROR"
            )
    
    async def _execute_detection_workflow(self, page_url: str, strategy: str) -> Dict[str, Any]:
        """Execute the core detection workflow"""
        
        start_time = datetime.utcnow()
        
        # Simulate browser session creation and navigation
        # In real implementation, this would use AgentCore Browser Tool
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Execute detection strategy
        strategy_config = self.detection_strategies[strategy]
        detected_captchas = []
        
        for detector_config in strategy_config['detectors']:
            try:
                captcha_elements = await self._detect_captcha_type(detector_config)
                detected_captchas.extend(captcha_elements)
            except Exception as detector_error:
                logger.warning(f"Detector '{detector_config['name']}' failed: {detector_error}")
                continue
        
        # Validate and enrich detection results
        validated_captchas = []
        for captcha in detected_captchas:
            if await self._validate_captcha_element(captcha):
                # Simulate screenshot capture
                captcha['screenshot'] = await self._capture_captcha_screenshot(captcha)
                validated_captchas.append(captcha)
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            'detected_captchas': validated_captchas,
            'detection_strategy': strategy,
            'page_url': page_url,
            'execution_time': execution_time,
            'captcha_found': len(validated_captchas) > 0,
            'session_id': f"session_{int(time.time())}"
        }
    
    def _initialize_detection_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize detection strategies for different CAPTCHA types"""
        
        return {
            'comprehensive': {
                'description': 'Comprehensive scan for all CAPTCHA types',
                'detectors': [
                    {
                        'name': 'recaptcha_v2',
                        'selectors': [
                            "iframe[src*='recaptcha']",
                            ".g-recaptcha",
                            "#recaptcha-anchor"
                        ],
                        'validation_checks': ['visibility', 'size', 'interactability']
                    },
                    {
                        'name': 'hcaptcha',
                        'selectors': [
                            "iframe[src*='hcaptcha']",
                            ".h-captcha"
                        ],
                        'validation_checks': ['visibility', 'size', 'interactability']
                    },
                    {
                        'name': 'generic_image',
                        'selectors': [
                            "img[alt*='captcha' i]",
                            "img[src*='captcha' i]"
                        ],
                        'validation_checks': ['visibility', 'size', 'image_content']
                    }
                ]
            },
            'recaptcha_focused': {
                'description': 'Optimized for reCAPTCHA detection',
                'detectors': [
                    {
                        'name': 'recaptcha_v2_primary',
                        'selectors': [
                            "iframe[src*='recaptcha/api2/anchor']",
                            ".g-recaptcha"
                        ],
                        'validation_checks': ['visibility', 'size', 'interactability']
                    }
                ]
            },
            'hcaptcha_focused': {
                'description': 'Optimized for hCaptcha detection',
                'detectors': [
                    {
                        'name': 'hcaptcha_widget',
                        'selectors': [
                            "iframe[src*='hcaptcha.com']",
                            ".h-captcha"
                        ],
                        'validation_checks': ['visibility', 'size', 'widget_state']
                    }
                ]
            }
        }
    
    async def _detect_captcha_type(self, detector_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect specific CAPTCHA type using configured selectors"""
        
        detected_elements = []
        detector_name = detector_config['name']
        
        # Simulate element detection
        for selector in detector_config['selectors']:
            # In real implementation, this would query the DOM
            if 'recaptcha' in selector:
                detected_elements.append({
                    'detector': detector_name,
                    'selector': selector,
                    'captcha_type': CaptchaType.RECAPTCHA_V2,
                    'bounds': {'x': 100, 'y': 200, 'width': 300, 'height': 150},
                    'visible': True,
                    'confidence': 0.9
                })
            elif 'hcaptcha' in selector:
                detected_elements.append({
                    'detector': detector_name,
                    'selector': selector,
                    'captcha_type': CaptchaType.HCAPTCHA,
                    'bounds': {'x': 150, 'y': 250, 'width': 320, 'height': 180},
                    'visible': True,
                    'confidence': 0.85
                })
        
        return detected_elements
    
    async def _validate_captcha_element(self, captcha_info: Dict[str, Any]) -> bool:
        """Comprehensive validation of detected CAPTCHA element"""
        
        # Basic validation requirements
        if not captcha_info.get('visible', False):
            return False
        
        # Size validation
        bounds = captcha_info.get('bounds', {})
        if bounds.get('width', 0) < 10 or bounds.get('height', 0) < 10:
            return False
        
        # Confidence threshold
        if captcha_info.get('confidence', 0.0) < 0.7:
            return False
        
        return True
    
    async def _capture_captcha_screenshot(self, captcha_info: Dict[str, Any]) -> Optional[str]:
        """Capture screenshot of CAPTCHA element for AI analysis"""
        
        try:
            # Simulate screenshot capture
            await asyncio.sleep(0.05)  # Simulate capture time
            
            # Generate mock base64 screenshot data
            mock_screenshot = base64.b64encode(b"mock_screenshot_data").decode('utf-8')
            
            logger.debug(f"Captured screenshot for {captcha_info['detector']} CAPTCHA")
            return mock_screenshot
            
        except Exception as e:
            logger.warning(f"Screenshot capture failed: {e}")
            return None

# =============================================================================
# CAPTCHA SOLVING TOOL
# =============================================================================

class CaptchaSolvingTool(Tool):
    """
    Strands tool for solving CAPTCHAs using Bedrock AI models.
    
    This tool implements intelligent CAPTCHA solving with adaptive model selection,
    confidence scoring, and comprehensive error handling.
    """
    
    name = "captcha_solver"
    description = "Solves various types of CAPTCHAs using Bedrock AI models"
    version = "1.0.0"
    
    parameters = [
        ToolParameter(
            name="captcha_data",
            type="object",
            description="CAPTCHA detection data from captcha_detector tool",
            required=True
        ),
        ToolParameter(
            name="model_preference",
            type="string",
            description="Preferred model: 'claude-3-sonnet', 'claude-3-opus', 'auto'",
            required=False,
            default="auto"
        ),
        ToolParameter(
            name="confidence_threshold",
            type="float",
            description="Minimum confidence threshold for solution acceptance",
            required=False,
            default=0.7
        )
    ]
    
    def __init__(self, bedrock_client=None):
        super().__init__()
        self.bedrock = bedrock_client
        self.model_configs = self._initialize_model_configurations()
        self.prompt_templates = self._initialize_prompt_templates()
        
    async def execute(self, **kwargs) -> ToolResult:
        """Execute CAPTCHA solving with intelligent model selection"""
        
        try:
            captcha_data = kwargs.get('captcha_data')
            model_preference = kwargs.get('model_preference', 'auto')
            confidence_threshold = kwargs.get('confidence_threshold', 0.7)
            
            if not captcha_data or not isinstance(captcha_data, dict):
                return ToolResult(
                    success=False,
                    error="captcha_data parameter is required and must be a dictionary",
                    error_code="INVALID_CAPTCHA_DATA"
                )
            
            detected_captchas = captcha_data.get('detected_captchas', [])
            if not detected_captchas:
                return ToolResult(
                    success=False,
                    error="No CAPTCHAs found in detection data",
                    error_code="NO_CAPTCHAS_DETECTED"
                )
            
            logger.info(f"Starting CAPTCHA solving for {len(detected_captchas)} detected CAPTCHAs")
            
            # Solve each detected CAPTCHA
            solving_results = []
            for i, captcha in enumerate(detected_captchas):
                try:
                    solution_result = await self._solve_single_captcha(
                        captcha, model_preference, confidence_threshold
                    )
                    solving_results.append(solution_result)
                    
                except Exception as e:
                    logger.error(f"Failed to solve CAPTCHA {i}: {e}")
                    solving_results.append({
                        'success': False,
                        'error': str(e),
                        'captcha_index': i
                    })
            
            # Aggregate results
            successful_solutions = [r for r in solving_results if r.get('success', False)]
            
            return ToolResult(
                success=len(successful_solutions) > 0,
                data={
                    'solutions': solving_results,
                    'successful_count': len(successful_solutions),
                    'total_count': len(detected_captchas),
                    'success_rate': len(successful_solutions) / len(detected_captchas) if detected_captchas else 0
                },
                message=f"Solved {len(successful_solutions)}/{len(detected_captchas)} CAPTCHAs",
                metadata={
                    'confidence_threshold': confidence_threshold,
                    'model_preference': model_preference
                }
            )
            
        except Exception as e:
            logger.error(f"CAPTCHA solving failed: {e}")
            return ToolResult(
                success=False,
                error=f"CAPTCHA solving operation failed: {str(e)}",
                error_code="SOLVING_ERROR"
            )
    
    async def _solve_single_captcha(self, captcha_info: Dict[str, Any], 
                                  model_preference: str, confidence_threshold: float) -> Dict[str, Any]:
        """Solve a single CAPTCHA with adaptive strategy selection"""
        
        start_time = datetime.utcnow()
        
        try:
            # Classify CAPTCHA type and complexity
            captcha_classification = self._classify_captcha(captcha_info)
            
            # Select optimal model configuration
            if model_preference == 'auto':
                model_config = self._select_optimal_model(captcha_classification)
            else:
                model_config = self.model_configs.get(model_preference, 
                                                    self.model_configs['claude-3-sonnet'])
            
            # Prepare solving prompt
            solving_prompt = self._build_solving_prompt(captcha_info, captcha_classification)
            
            # Execute solving with primary strategy
            primary_result = await self._execute_model_solving(
                captcha_info, solving_prompt, model_config
            )
            
            # Validate solution confidence
            if primary_result['confidence'] >= confidence_threshold:
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                return {
                    'success': True,
                    'solution': primary_result['solution'],
                    'confidence': primary_result['confidence'],
                    'captcha_type': captcha_classification['type'],
                    'model_used': model_config['model_id'],
                    'execution_time': execution_time
                }
            else:
                # Retry with enhanced model if confidence is low
                enhanced_result = await self._retry_with_enhanced_model(
                    captcha_info, captcha_classification
                )
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                return {
                    'success': enhanced_result['confidence'] >= (confidence_threshold - 0.1),
                    'solution': enhanced_result['solution'],
                    'confidence': enhanced_result['confidence'],
                    'captcha_type': captcha_classification['type'],
                    'model_used': enhanced_result['model_used'],
                    'execution_time': execution_time,
                    'retry_attempted': True
                }
                
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time
            }
    
    def _initialize_model_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Initialize AI model configurations for different CAPTCHA types"""
        
        return {
            'claude-3-sonnet': {
                'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0',
                'focus': 'balanced_performance',
                'temperature': 0.1,
                'max_tokens': 1000
            },
            'claude-3-opus': {
                'model_id': 'anthropic.claude-3-opus-20240229-v1:0',
                'focus': 'complex_reasoning',
                'temperature': 0.05,
                'max_tokens': 1500
            }
        }
    
    def _initialize_prompt_templates(self) -> Dict[str, str]:
        """Initialize prompt templates for different CAPTCHA types"""
        
        return {
            'text_captcha': """
            Analyze this CAPTCHA image carefully. Look for:
            1. Text characters (letters, numbers, symbols)
            2. Distortions, noise, or overlapping elements
            3. Background patterns that might obscure text
            
            Extract the exact text shown. If uncertain about any character, 
            indicate your confidence level. Return only the text content.
            """,
            
            'image_selection': """
            This is an image selection CAPTCHA. Analyze each grid cell and:
            1. Identify the target object type from the instruction
            2. Look for partial objects at cell edges
            3. Consider lighting, angles, and image quality
            
            Select all cells containing the target object, including partial views.
            Provide confidence score for each selection.
            """,
            
            'recaptcha': """
            This is a reCAPTCHA challenge. Analyze the image and:
            1. Identify the challenge type (text, images, audio)
            2. Apply appropriate solving strategy
            3. Consider context and instructions carefully
            
            Provide the solution with confidence assessment.
            """
        }
    
    def _classify_captcha(self, captcha_info: Dict[str, Any]) -> Dict[str, Any]:
        """Classify CAPTCHA type and complexity"""
        
        captcha_type = captcha_info.get('captcha_type', CaptchaType.GENERIC)
        
        # Analyze complexity based on various factors
        complexity_score = 0.5  # Base complexity
        
        if captcha_type == CaptchaType.RECAPTCHA_V2:
            complexity_score = 0.8
        elif captcha_type == CaptchaType.HCAPTCHA:
            complexity_score = 0.7
        elif captcha_type == CaptchaType.TEXT_CAPTCHA:
            complexity_score = 0.6
        
        return {
            'type': captcha_type,
            'complexity_score': complexity_score,
            'estimated_difficulty': 'high' if complexity_score > 0.7 else 'medium' if complexity_score > 0.5 else 'low'
        }
    
    def _select_optimal_model(self, captcha_classification: Dict[str, Any]) -> Dict[str, Any]:
        """Select optimal AI model based on CAPTCHA complexity"""
        
        complexity_score = captcha_classification['complexity_score']
        
        if complexity_score > 0.7:
            return self.model_configs['claude-3-opus']  # Use most powerful model
        else:
            return self.model_configs['claude-3-sonnet']  # Use balanced model
    
    def _build_solving_prompt(self, captcha_info: Dict[str, Any], 
                            captcha_classification: Dict[str, Any]) -> str:
        """Build specialized prompt for CAPTCHA solving"""
        
        captcha_type = captcha_classification['type']
        
        if captcha_type == CaptchaType.TEXT_CAPTCHA:
            return self.prompt_templates['text_captcha']
        elif captcha_type == CaptchaType.IMAGE_SELECTION:
            return self.prompt_templates['image_selection']
        elif captcha_type in [CaptchaType.RECAPTCHA_V2, CaptchaType.RECAPTCHA_V3]:
            return self.prompt_templates['recaptcha']
        else:
            return self.prompt_templates['text_captcha']  # Default fallback
    
    async def _execute_model_solving(self, captcha_info: Dict[str, Any], 
                                   prompt: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute AI model solving with specified configuration"""
        
        # Simulate AI model processing
        await asyncio.sleep(0.2)  # Simulate model inference time
        
        # Mock solution based on CAPTCHA type
        captcha_type = captcha_info.get('captcha_type', CaptchaType.GENERIC)
        
        if captcha_type == CaptchaType.TEXT_CAPTCHA:
            solution = "MOCK123"
            confidence = 0.85
        elif captcha_type == CaptchaType.RECAPTCHA_V2:
            solution = {"action": "click_checkbox"}
            confidence = 0.9
        elif captcha_type == CaptchaType.HCAPTCHA:
            solution = {"selected_cells": [1, 3, 5]}
            confidence = 0.8
        else:
            solution = "generic_solution"
            confidence = 0.75
        
        return {
            'solution': solution,
            'confidence': confidence,
            'model_used': model_config['model_id']
        }
    
    async def _retry_with_enhanced_model(self, captcha_info: Dict[str, Any], 
                                       captcha_classification: Dict[str, Any]) -> Dict[str, Any]:
        """Retry solving with enhanced model configuration"""
        
        # Use the most powerful model for retry
        enhanced_config = self.model_configs['claude-3-opus']
        enhanced_prompt = self._build_solving_prompt(captcha_info, captcha_classification)
        
        result = await self._execute_model_solving(captcha_info, enhanced_prompt, enhanced_config)
        
        # Boost confidence slightly for enhanced model
        result['confidence'] = min(1.0, result['confidence'] + 0.1)
        
        return result

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

class WorkflowStateManager:
    """
    Comprehensive state management for CAPTCHA handling workflows.
    
    Manages workflow state across service boundaries with proper validation,
    recovery mechanisms, and audit trails.
    """
    
    def __init__(self):
        self.workflow_states = {}
        self.state_history = {}
        self.state_locks = {}
    
    async def initialize_workflow_state(self, workflow_id: str, initial_context: Dict[str, Any]) -> WorkflowState:
        """Initialize workflow state with comprehensive context tracking"""
        
        workflow_state = WorkflowState(
            workflow_id=workflow_id,
            created_at=datetime.utcnow(),
            current_phase=WorkflowPhase.INITIALIZATION,
            page_url=initial_context.get('page_url'),
            task_description=initial_context.get('task_description'),
            completed_phases=[],
            current_step_data={},
            error_history=[]
        )
        
        # Store state and create initial checkpoint
        self.workflow_states[workflow_id] = workflow_state
        await self._create_state_checkpoint(workflow_state, 'initialization')
        
        logger.info(f"Initialized workflow state: {workflow_id}")
        return workflow_state
    
    async def update_workflow_phase(self, workflow_id: str, new_phase: WorkflowPhase, 
                                  phase_data: Dict[str, Any]) -> WorkflowState:
        """Update workflow phase with proper state transition validation"""
        
        current_state = self.workflow_states.get(workflow_id)
        if not current_state:
            raise StateManagementError(f"Workflow state not found: {workflow_id}")
        
        # Validate phase transition
        if not self._is_valid_phase_transition(current_state.current_phase, new_phase):
            raise StateManagementError(
                f"Invalid phase transition from {current_state.current_phase} to {new_phase}"
            )
        
        # Create updated state
        updated_state = replace(
            current_state,
            current_phase=new_phase,
            completed_phases=current_state.completed_phases + [current_state.current_phase],
            current_step_data=phase_data
        )
        
        # Store updated state and create checkpoint
        self.workflow_states[workflow_id] = updated_state
        await self._create_state_checkpoint(updated_state, new_phase.value)
        
        logger.info(f"Updated workflow {workflow_id} to phase: {new_phase.value}")
        return updated_state
    
    async def handle_workflow_error(self, workflow_id: str, error: Exception, 
                                  recovery_strategy: str) -> WorkflowState:
        """Handle workflow errors with intelligent recovery strategies"""
        
        current_state = self.workflow_states.get(workflow_id)
        if not current_state:
            raise StateManagementError(f"Workflow state not found: {workflow_id}")
        
        # Record error in state history
        error_record = {
            'timestamp': datetime.utcnow(),
            'phase': current_state.current_phase.value,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'recovery_strategy': recovery_strategy
        }
        
        # Determine recovery action based on strategy
        if recovery_strategy == 'retry_current_phase':
            recovery_state = await self._prepare_phase_retry(current_state, error_record)
        elif recovery_strategy == 'rollback_to_checkpoint':
            recovery_state = await self._rollback_to_checkpoint(current_state, error_record)
        else:
            # Mark workflow as failed
            recovery_state = replace(
                current_state,
                current_phase=WorkflowPhase.FAILED,
                error_history=current_state.error_history + [error_record]
            )
        
        # Update state
        self.workflow_states[workflow_id] = recovery_state
        await self._create_state_checkpoint(recovery_state, f'error_recovery_{recovery_strategy}')
        
        logger.error(f"Handled error in workflow {workflow_id}: {recovery_strategy}")
        return recovery_state
    
    def _is_valid_phase_transition(self, from_phase: WorkflowPhase, to_phase: WorkflowPhase) -> bool:
        """Validate workflow phase transitions"""
        
        valid_transitions = {
            WorkflowPhase.INITIALIZATION: [WorkflowPhase.DETECTION],
            WorkflowPhase.DETECTION: [WorkflowPhase.ANALYSIS, WorkflowPhase.FAILED],
            WorkflowPhase.ANALYSIS: [WorkflowPhase.SOLUTION, WorkflowPhase.FAILED],
            WorkflowPhase.SOLUTION: [WorkflowPhase.SUBMISSION, WorkflowPhase.FAILED],
            WorkflowPhase.SUBMISSION: [WorkflowPhase.VERIFICATION, WorkflowPhase.FAILED],
            WorkflowPhase.VERIFICATION: [WorkflowPhase.COMPLETED, WorkflowPhase.FAILED],
            WorkflowPhase.FAILED: [WorkflowPhase.DETECTION]  # Allow retry from failed state
        }
        
        return to_phase in valid_transitions.get(from_phase, [])
    
    async def _create_state_checkpoint(self, workflow_state: WorkflowState, checkpoint_name: str):
        """Create state checkpoint for recovery purposes"""
        
        checkpoint_data = {
            'timestamp': datetime.utcnow(),
            'checkpoint_name': checkpoint_name,
            'state_snapshot': {
                'workflow_id': workflow_state.workflow_id,
                'current_phase': workflow_state.current_phase.value,
                'completed_phases': [p.value for p in workflow_state.completed_phases],
                'current_step_data': workflow_state.current_step_data.copy()
            }
        }
        
        # Store in state history
        if workflow_state.workflow_id not in self.state_history:
            self.state_history[workflow_state.workflow_id] = []
        
        self.state_history[workflow_state.workflow_id].append(checkpoint_data)
        
        # Keep only last 10 checkpoints to manage memory
        if len(self.state_history[workflow_state.workflow_id]) > 10:
            self.state_history[workflow_state.workflow_id] = \
                self.state_history[workflow_state.workflow_id][-10:]
    
    async def _prepare_phase_retry(self, current_state: WorkflowState, 
                                 error_record: Dict[str, Any]) -> WorkflowState:
        """Prepare state for phase retry"""
        
        # Increment retry count
        retry_count = current_state.retry_count + 1
        
        if retry_count > current_state.max_retries:
            return replace(
                current_state,
                current_phase=WorkflowPhase.FAILED,
                error_history=current_state.error_history + [error_record]
            )
        
        # Reset to previous phase for retry
        if current_state.completed_phases:
            retry_phase = current_state.completed_phases[-1]
        else:
            retry_phase = WorkflowPhase.DETECTION
        
        return replace(
            current_state,
            current_phase=retry_phase,
            retry_count=retry_count,
            error_history=current_state.error_history + [error_record]
        )
    
    async def _rollback_to_checkpoint(self, current_state: WorkflowState, 
                                    error_record: Dict[str, Any]) -> WorkflowState:
        """Rollback to previous checkpoint"""
        
        # Find the most recent successful checkpoint
        checkpoints = self.state_history.get(current_state.workflow_id, [])
        
        if not checkpoints:
            # No checkpoints available, mark as failed
            return replace(
                current_state,
                current_phase=WorkflowPhase.FAILED,
                error_history=current_state.error_history + [error_record]
            )
        
        # Get the last successful checkpoint
        last_checkpoint = checkpoints[-1]
        checkpoint_state = last_checkpoint['state_snapshot']
        
        return replace(
            current_state,
            current_phase=WorkflowPhase(checkpoint_state['current_phase']),
            current_step_data=checkpoint_state['current_step_data'],
            retry_count=current_state.retry_count + 1,
            error_history=current_state.error_history + [error_record]
        )

# =============================================================================
# ADVANCED STATE MANAGEMENT
# =============================================================================

class CaptchaWorkflowState(Enum):
    """States for CAPTCHA workflow progression"""
    INITIALIZED = "initialized"
    SERVICES_COORDINATING = "services_coordinating"
    SERVICES_READY = "services_ready"
    DETECTING_CAPTCHA = "detecting_captcha"
    CAPTCHA_DETECTED = "captcha_detected"
    ANALYZING_WITH_AI = "analyzing_with_ai"
    AI_ANALYSIS_COMPLETE = "ai_analysis_complete"
    SUBMITTING_SOLUTION = "submitting_solution"
    SOLUTION_SUBMITTED = "solution_submitted"
    VERIFYING_SUCCESS = "verifying_success"
    WORKFLOW_COMPLETE = "workflow_complete"
    WORKFLOW_FAILED = "workflow_failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"

class ServiceState(Enum):
    """States for individual services"""
    OFFLINE = "offline"
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class ServiceStatus:
    """Status information for a coordinated service"""
    service_name: str
    state: ServiceState
    last_activity: datetime
    current_operation: Optional[str] = None
    operation_start_time: Optional[datetime] = None
    error_message: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    health_score: float = 1.0
    
    def update_activity(self, operation: Optional[str] = None):
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()
        if operation:
            self.current_operation = operation
            self.operation_start_time = datetime.utcnow()
    
    def complete_operation(self, success: bool = True):
        """Mark current operation as complete"""
        if self.operation_start_time:
            duration = (datetime.utcnow() - self.operation_start_time).total_seconds()
            self.performance_metrics[f"{self.current_operation}_duration"] = duration
        
        self.current_operation = None
        self.operation_start_time = None
        self.state = ServiceState.READY if success else ServiceState.ERROR

@dataclass
class ServiceCoordinationContext:
    """Context for coordinating multiple services in CAPTCHA workflows"""
    agentcore_session_id: Optional[str] = None
    bedrock_model_id: Optional[str] = None
    browser_state: Dict[str, Any] = field(default_factory=dict)
    ai_analysis_state: Dict[str, Any] = field(default_factory=dict)
    coordination_metrics: Dict[str, float] = field(default_factory=dict)
    service_health: Dict[str, bool] = field(default_factory=dict)
    
    def update_service_health(self, service: str, healthy: bool):
        """Update health status of a coordinated service"""
        self.service_health[service] = healthy
        
    def get_healthy_services(self) -> List[str]:
        """Get list of currently healthy services"""
        return [service for service, healthy in self.service_health.items() if healthy]

class CaptchaStateManager:
    """Comprehensive state manager for CAPTCHA workflows across multiple services"""
    
    def __init__(self):
        # Service status tracking
        self.service_statuses: Dict[str, ServiceStatus] = {}
        
        # Workflow progress tracking
        self.workflow_progress: Dict[str, Any] = {}
        
        # Cross-service state synchronization
        self.sync_intervals: Dict[str, float] = defaultdict(lambda: 5.0)
        self.last_sync_times: Dict[str, datetime] = {}
        
        # State change listeners
        self.state_change_listeners: List[callable] = []
        
        # Performance tracking
        self.state_metrics = {
            'total_workflows': 0,
            'active_workflows': 0,
            'completed_workflows': 0,
            'failed_workflows': 0,
            'average_workflow_duration': 0.0,
            'state_sync_operations': 0,
            'state_conflicts_resolved': 0
        }
        
        print("🗃️ Strands CAPTCHA State Manager initialized")
    
    async def initialize_workflow_state(self, 
                                      workflow_id: str,
                                      captcha_data: Dict[str, Any],
                                      workflow_config: Dict[str, Any]) -> str:
        """Initialize state for a new CAPTCHA workflow"""
        
        print(f"🔄 Initializing workflow state: {workflow_id}")
        
        # Create initial workflow state
        initial_state = {
            'workflow_id': workflow_id,
            'state': CaptchaWorkflowState.INITIALIZED.value,
            'captcha_data': captcha_data,
            'workflow_config': workflow_config,
            'created_at': datetime.utcnow().isoformat(),
            'services': {},
            'progress': {
                'current_step': 'initialization',
                'steps_completed': [],
                'progress_percentage': 0.0
            },
            'metrics': {
                'start_time': time.time(),
                'service_coordination_time': 0.0,
                'detection_time': 0.0,
                'analysis_time': 0.0,
                'submission_time': 0.0
            }
        }
        
        # Initialize workflow progress tracking
        self.workflow_progress[workflow_id] = {
            'current_state': CaptchaWorkflowState.INITIALIZED,
            'progress_percentage': 0.0,
            'steps_completed': [],
            'steps_remaining': [
                'service_coordination',
                'captcha_detection',
                'ai_analysis',
                'solution_submission',
                'verification'
            ]
        }
        
        # Update metrics
        self.state_metrics['total_workflows'] += 1
        self.state_metrics['active_workflows'] += 1
        
        print(f"✅ Workflow state initialized: {workflow_id}")
        return workflow_id
    
    async def update_service_state(self, 
                                 workflow_id: str,
                                 service_name: str,
                                 service_state: ServiceState,
                                 operation: Optional[str] = None,
                                 metadata: Optional[Dict[str, Any]] = None):
        """Update state for a specific service"""
        
        print(f"🔧 Updating service state: {service_name} -> {service_state.value}")
        
        # Update service status
        if service_name not in self.service_statuses:
            self.service_statuses[service_name] = ServiceStatus(
                service_name=service_name,
                state=service_state,
                last_activity=datetime.utcnow()
            )
        else:
            self.service_statuses[service_name].state = service_state
            self.service_statuses[service_name].update_activity(operation)
    
    def get_state_metrics(self) -> Dict[str, Any]:
        """Get comprehensive state management metrics"""
        
        return {
            **self.state_metrics,
            'active_services': len([s for s in self.service_statuses.values() if s.state in [ServiceState.READY, ServiceState.BUSY]]),
            'total_services': len(self.service_statuses),
            'listeners_count': len(self.state_change_listeners)
        }

# =============================================================================
# ORCHESTRATION LAYER
# =============================================================================

class OrchestrationState(Enum):
    """States for CAPTCHA orchestration workflow"""
    INITIALIZING = "initializing"
    DETECTING = "detecting"
    ANALYZING = "analyzing"
    SOLVING = "solving"
    SUBMITTING = "submitting"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

class CaptchaOrchestrationAgent(Agent):
    """
    Strands agent that orchestrates AgentCore Browser Tool and Bedrock Vision Models
    for comprehensive CAPTCHA handling workflows
    """
    
    def __init__(self, config: AgentConfig, agentcore_client, region_name: str = 'us-east-1'):
        super().__init__(config)
        
        # Service clients
        self.agentcore = agentcore_client
        self.bedrock_vision = BedrockVisionClient(region_name)
        
        # State management
        self.state_manager = CaptchaStateManager()
        self.coordination_context = ServiceCoordinationContext()
        
        # Orchestration configuration
        self.orchestration_config = {
            'max_coordination_retries': 3,
            'service_timeout': 30.0,
            'state_sync_interval': 5.0,
            'performance_monitoring': True,
            'adaptive_routing': True
        }
        
        # Performance metrics
        self.performance_metrics = {
            'total_orchestrations': 0,
            'successful_orchestrations': 0,
            'average_coordination_time': 0.0,
            'service_utilization': {},
            'error_patterns': {}
        }
        
        print("🎭 Strands CAPTCHA Orchestration Agent initialized")
        print(f"   AgentCore Integration: ✅")
        print(f"   Bedrock Vision Integration: ✅")
        print(f"   State Management: ✅")
    
    async def orchestrate_captcha_workflow(self, 
                                         captcha_data: Dict[str, Any],
                                         workflow_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main orchestration method for CAPTCHA handling workflows
        
        Args:
            captcha_data: CAPTCHA detection data
            workflow_config: Configuration for workflow execution
            
        Returns:
            Complete orchestration results
        """
        orchestration_id = f"captcha_orch_{int(time.time())}"
        start_time = time.time()
        
        try:
            print(f"🎭 Starting CAPTCHA orchestration: {orchestration_id}")
            
            # Initialize orchestration state
            await self._initialize_orchestration_state(orchestration_id, captcha_data, workflow_config)
            
            # Create orchestration workflow
            workflow = CaptchaOrchestrationWorkflow(self, orchestration_id)
            
            # Execute coordinated workflow
            context = {
                'orchestration_id': orchestration_id,
                'captcha_data': captcha_data,
                'workflow_config': workflow_config or {},
                'start_time': start_time,
                'coordination_context': self.coordination_context
            }
            
            result_context = await workflow.execute(context)
            
            # Finalize orchestration
            final_result = await self._finalize_orchestration(orchestration_id, result_context)
            
            # Update performance metrics
            self._update_performance_metrics(orchestration_id, final_result, time.time() - start_time)
            
            return final_result
            
        except Exception as e:
            print(f"❌ Orchestration failed: {e}")
            await self._handle_orchestration_failure(orchestration_id, str(e))
            raise
    
    async def _initialize_orchestration_state(self, 
                                            orchestration_id: str,
                                            captcha_data: Dict[str, Any],
                                            workflow_config: Optional[Dict[str, Any]]):
        """Initialize state for orchestration workflow"""
        
        print(f"🔄 Initializing orchestration state: {orchestration_id}")
        
        # Initialize service coordination context
        self.coordination_context = ServiceCoordinationContext()
        
        # Health check services
        await self._perform_service_health_checks()
        
        print(f"✅ Orchestration state initialized")
    
    async def _perform_service_health_checks(self):
        """Perform health checks on coordinated services"""
        
        print("🏥 Performing service health checks...")
        
        # Check AgentCore Browser Tool
        try:
            agentcore_healthy = await self._check_agentcore_health()
            self.coordination_context.update_service_health('agentcore', agentcore_healthy)
            print(f"   AgentCore: {'✅' if agentcore_healthy else '❌'}")
        except Exception as e:
            self.coordination_context.update_service_health('agentcore', False)
            print(f"   AgentCore: ❌ ({e})")
        
        # Check Bedrock Vision Models
        try:
            bedrock_healthy = await self._check_bedrock_health()
            self.coordination_context.update_service_health('bedrock', bedrock_healthy)
            print(f"   Bedrock Vision: {'✅' if bedrock_healthy else '❌'}")
        except Exception as e:
            self.coordination_context.update_service_health('bedrock', False)
            print(f"   Bedrock Vision: ❌ ({e})")
        
        healthy_services = self.coordination_context.get_healthy_services()
        print(f"🏥 Health check complete. Healthy services: {healthy_services}")
    
    async def _check_agentcore_health(self) -> bool:
        """Check AgentCore Browser Tool health"""
        try:
            return self.agentcore is not None
        except:
            return False
    
    async def _check_bedrock_health(self) -> bool:
        """Check Bedrock Vision Models health"""
        try:
            return self.bedrock_vision.bedrock_available
        except:
            return False
    
    async def _finalize_orchestration(self, orchestration_id: str, result_context: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize orchestration and compile results"""
        
        print(f"🎯 Finalizing orchestration: {orchestration_id}")
        
        # Extract results from context
        final_result = {
            'orchestration_id': orchestration_id,
            'success': result_context.get('orchestration_success', False),
            'captcha_solved': result_context.get('captcha_solved', False),
            'solution_submitted': result_context.get('solution_submitted', False),
            'workflow_results': result_context.get('workflow_results', {}),
            'coordination_metrics': result_context.get('coordination_metrics', {}),
            'service_performance': result_context.get('service_performance', {}),
            'error_details': result_context.get('error_details'),
            'completed_at': datetime.utcnow().isoformat()
        }
        
        print(f"✅ Orchestration finalized: {'Success' if final_result['success'] else 'Failed'}")
        
        return final_result
    
    async def _handle_orchestration_failure(self, orchestration_id: str, error_message: str):
        """Handle orchestration failure"""
        
        print(f"💥 Handling orchestration failure: {orchestration_id}")
        
        failure_data = {
            'state': OrchestrationState.FAILED.value,
            'error_message': error_message,
            'failed_at': datetime.utcnow().isoformat(),
            'service_health': dict(self.coordination_context.service_health)
        }
    
    def _update_performance_metrics(self, orchestration_id: str, result: Dict[str, Any], duration: float):
        """Update performance metrics for orchestration"""
        
        self.performance_metrics['total_orchestrations'] += 1
        
        if result.get('success', False):
            self.performance_metrics['successful_orchestrations'] += 1
        
        # Update average coordination time
        total = self.performance_metrics['total_orchestrations']
        current_avg = self.performance_metrics['average_coordination_time']
        self.performance_metrics['average_coordination_time'] = ((current_avg * (total - 1)) + duration) / total
        
        print(f"📊 Performance metrics updated for {orchestration_id}")

class CaptchaOrchestrationWorkflow:
    """Strands workflow engine for coordinating multi-service CAPTCHA workflows"""
    
    def __init__(self, orchestration_agent: CaptchaOrchestrationAgent, orchestration_id: str):
        self.agent = orchestration_agent
        self.orchestration_id = orchestration_id
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the orchestration workflow"""
        
        try:
            # Step 1: Coordinate services
            context = await self.coordinate_services_step(context)
            
            # Step 2: Execute CAPTCHA detection
            context = await self.execute_captcha_detection_step(context)
            
            # Step 3: Coordinate AI analysis
            context = await self.coordinate_ai_analysis_step(context)
            
            # Step 4: Execute solution submission
            context = await self.execute_solution_submission_step(context)
            
            # Step 5: Verify coordination success
            context = await self.verify_coordination_success_step(context)
            
            return context
            
        except Exception as e:
            context['orchestration_success'] = False
            context['error_details'] = str(e)
            return context
    
    async def coordinate_services_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: Coordinate AgentCore and Bedrock services"""
        
        print(f"🔗 Step 1: Coordinating services for {self.orchestration_id}")
        
        try:
            # Verify service availability
            healthy_services = self.agent.coordination_context.get_healthy_services()
            
            if 'agentcore' not in healthy_services:
                raise Exception("AgentCore Browser Tool not available")
            
            if 'bedrock' not in healthy_services:
                raise Exception("Bedrock Vision Models not available")
            
            # Initialize service coordination
            coordination_start = time.time()
            
            # Prepare AgentCore session
            agentcore_session = await self._prepare_agentcore_session(context)
            context['agentcore_session'] = agentcore_session
            
            # Prepare Bedrock models
            bedrock_models = await self._prepare_bedrock_models(context)
            context['bedrock_models'] = bedrock_models
            
            coordination_time = time.time() - coordination_start
            context['service_coordination_time'] = coordination_time
            context['services_coordinated'] = True
            
            print(f"✅ Services coordinated successfully in {coordination_time:.2f}s")
            
        except Exception as e:
            context['services_coordinated'] = False
            context['coordination_error'] = str(e)
            print(f"❌ Service coordination failed: {e}")
        
        return context
    
    async def execute_captcha_detection_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Execute CAPTCHA detection using AgentCore"""
        
        print(f"🔍 Step 2: Executing CAPTCHA detection for {self.orchestration_id}")
        
        try:
            if not context.get('services_coordinated'):
                context['detection_skipped'] = True
                return context
            
            captcha_data = context.get('captcha_data')
            
            # Execute detection through AgentCore
            detection_start = time.time()
            
            # Simulate screenshot capture
            screenshot_data = b"mock_screenshot_data"  # In real implementation, use AgentCore
            
            # Enhanced detection analysis
            detection_result = {
                'screenshot_captured': True,
                'screenshot_size': len(screenshot_data) if screenshot_data else 0,
                'captcha_type': captcha_data.get('captcha_type', 'unknown'),
                'element_bounds': captcha_data.get('bounds', {}),
                'detection_confidence': 0.9,
                'agentcore_session_id': context.get('agentcore_session', {}).get('session_id')
            }
            
            detection_time = time.time() - detection_start
            
            context['detection_result'] = detection_result
            context['screenshot_data'] = screenshot_data
            context['detection_time'] = detection_time
            context['detection_success'] = True
            
            print(f"✅ CAPTCHA detection completed in {detection_time:.2f}s")
            
        except Exception as e:
            context['detection_success'] = False
            context['detection_error'] = str(e)
            print(f"❌ CAPTCHA detection failed: {e}")
        
        return context
    
    async def coordinate_ai_analysis_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Coordinate AI analysis using Bedrock Vision Models"""
        
        print(f"🧠 Step 3: Coordinating AI analysis for {self.orchestration_id}")
        
        try:
            if not context.get('detection_success'):
                context['analysis_skipped'] = True
                return context
            
            screenshot_data = context.get('screenshot_data')
            detection_result = context.get('detection_result')
            bedrock_models = context.get('bedrock_models')
            
            # Execute AI analysis through Bedrock
            analysis_start = time.time()
            
            # Determine optimal model based on CAPTCHA type
            captcha_type = detection_result.get('captcha_type', 'text_captcha')
            optimal_model = self._select_optimal_model(captcha_type, bedrock_models)
            
            # Perform AI analysis
            analysis_result = await self.agent.bedrock_vision.analyze_captcha_image(
                screenshot_data,
                CaptchaType.TEXT_CAPTCHA,  # Convert string to enum
                optimal_model
            )
            
            analysis_time = time.time() - analysis_start
            
            # Enhance analysis result with coordination metadata
            enhanced_result = {
                **analysis_result,
                'model_used': optimal_model,
                'analysis_time': analysis_time,
                'coordination_metadata': {
                    'agentcore_session': context.get('agentcore_session', {}).get('session_id'),
                    'bedrock_model': optimal_model,
                    'orchestration_id': self.orchestration_id
                }
            }
            
            context['analysis_result'] = enhanced_result
            context['analysis_time'] = analysis_time
            context['analysis_success'] = analysis_result.get('success', False)
            
            print(f"✅ AI analysis completed in {analysis_time:.2f}s using {optimal_model}")
            
        except Exception as e:
            context['analysis_success'] = False
            context['analysis_error'] = str(e)
            print(f"❌ AI analysis failed: {e}")
        
        return context
    
    async def execute_solution_submission_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Execute solution submission through AgentCore"""
        
        print(f"📤 Step 4: Executing solution submission for {self.orchestration_id}")
        
        try:
            if not context.get('analysis_success'):
                context['submission_skipped'] = True
                return context
            
            analysis_result = context.get('analysis_result')
            workflow_config = context.get('workflow_config', {})
            
            # Check if auto-submission is enabled
            auto_submit = workflow_config.get('auto_submit', True)
            
            if not auto_submit:
                context['submission_success'] = True
                context['submission_mode'] = 'manual'
                print("✅ Manual submission mode - analysis complete")
                return context
            
            # Execute submission through AgentCore
            submission_start = time.time()
            
            solution = analysis_result.get('solution', '')
            captcha_data = context.get('captcha_data')
            
            # Submit solution using AgentCore
            submission_result = await self._submit_solution_via_agentcore(
                solution, 
                captcha_data, 
                context.get('agentcore_session')
            )
            
            submission_time = time.time() - submission_start
            
            context['submission_result'] = submission_result
            context['submission_time'] = submission_time
            context['submission_success'] = submission_result.get('success', False)
            context['submission_mode'] = 'automatic'
            
            print(f"✅ Solution submission completed in {submission_time:.2f}s")
            
        except Exception as e:
            context['submission_success'] = False
            context['submission_error'] = str(e)
            print(f"❌ Solution submission failed: {e}")
        
        return context
    
    async def verify_coordination_success_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 5: Verify overall coordination success"""
        
        print(f"🔍 Step 5: Verifying coordination success for {self.orchestration_id}")
        
        try:
            # Compile coordination results
            coordination_success = (
                context.get('services_coordinated', False) and
                context.get('detection_success', False) and
                context.get('analysis_success', False) and
                context.get('submission_success', False)
            )
            
            # Calculate coordination metrics
            coordination_metrics = {
                'total_coordination_time': (
                    context.get('service_coordination_time', 0) +
                    context.get('detection_time', 0) +
                    context.get('analysis_time', 0) +
                    context.get('submission_time', 0)
                ),
                'service_utilization': {
                    'agentcore': context.get('detection_time', 0) + context.get('submission_time', 0),
                    'bedrock': context.get('analysis_time', 0)
                },
                'coordination_efficiency': self._calculate_coordination_efficiency(context)
            }
            
            # Compile workflow results
            workflow_results = {
                'captcha_solved': context.get('analysis_success', False),
                'solution_submitted': context.get('submission_success', False),
                'detection_result': context.get('detection_result', {}),
                'analysis_result': context.get('analysis_result', {}),
                'submission_result': context.get('submission_result', {})
            }
            
            context['orchestration_success'] = coordination_success
            context['coordination_metrics'] = coordination_metrics
            context['workflow_results'] = workflow_results
            
            print(f"✅ Coordination verification complete: {'Success' if coordination_success else 'Failed'}")
            
        except Exception as e:
            context['orchestration_success'] = False
            context['verification_error'] = str(e)
            print(f"❌ Coordination verification failed: {e}")
        
        return context
    
    async def _prepare_agentcore_session(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare AgentCore session for coordination"""
        
        session_data = {
            'session_id': f"agentcore_{self.orchestration_id}",
            'browser_config': {
                'headless': True,
                'timeout': 30
            },
            'coordination_mode': True,
            'created_at': datetime.utcnow().isoformat()
        }
        
        # Store session in coordination context
        self.agent.coordination_context.agentcore_session_id = session_data['session_id']
        
        return session_data
    
    async def _prepare_bedrock_models(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare Bedrock models for coordination"""
        
        models_config = {
            'primary_model': 'claude-3-sonnet',
            'fallback_model': 'claude-3-haiku',
            'vision_models': ['claude-3-sonnet', 'claude-3-opus'],
            'coordination_mode': True,
            'model_selection_strategy': 'adaptive'
        }
        
        # Store model config in coordination context
        self.agent.coordination_context.bedrock_model_id = models_config['primary_model']
        
        return models_config
    
    def _select_optimal_model(self, captcha_type: str, bedrock_models: Dict[str, Any]) -> str:
        """Select optimal Bedrock model based on CAPTCHA type"""
        
        model_preferences = {
            'text_captcha': 'claude-3-sonnet',
            'image_selection': 'claude-3-opus',
            'recaptcha': 'claude-3-sonnet',
            'math_captcha': 'claude-3-opus',
            'complex_visual': 'claude-3-opus'
        }
        
        return model_preferences.get(captcha_type, bedrock_models.get('primary_model', 'claude-3-sonnet'))
    
    async def _submit_solution_via_agentcore(self, 
                                           solution: str, 
                                           captcha_data: Dict[str, Any],
                                           agentcore_session: Dict[str, Any]) -> Dict[str, Any]:
        """Submit CAPTCHA solution via AgentCore Browser Tool"""
        
        try:
            # Simulate solution submission
            await asyncio.sleep(0.5)  # Simulate submission time
            
            return {
                'success': True,
                'solution_submitted': solution,
                'agentcore_session': agentcore_session.get('session_id')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'solution_attempted': solution
            }
    
    def _calculate_coordination_efficiency(self, context: Dict[str, Any]) -> float:
        """Calculate coordination efficiency score"""
        
        try:
            total_time = (
                context.get('service_coordination_time', 0) +
                context.get('detection_time', 0) +
                context.get('analysis_time', 0) +
                context.get('submission_time', 0)
            )
            
            # Baseline expected time for efficient coordination
            baseline_time = 10.0  # seconds
            
            if total_time <= baseline_time:
                return 1.0
            else:
                return max(0.1, baseline_time / total_time)
                
        except:
            return 0.5  # Default efficiency score

# =============================================================================
# END-TO-END INTEGRATION
# =============================================================================

class EndToEndCaptchaSolver(Tool):
    """
    Complete end-to-end CAPTCHA solving tool that integrates:
    - AgentCore Browser Tool for screenshots and interaction
    - Bedrock Vision models for AI analysis
    - Confidence scoring and solution validation
    """
    
    name = "end_to_end_captcha_solver"
    description = "Complete CAPTCHA solving workflow using AgentCore Browser Tool and Bedrock Vision models"
    
    def __init__(self, agentcore_client, region_name: str = 'us-east-1'):
        super().__init__()
        self.agentcore = agentcore_client
        
        # Initialize vision client
        self.vision_client = BedrockVisionClient(region_name)
        
        # Initialize component tools
        self.solver_tool = CaptchaSolvingTool(agentcore_client, self.vision_client)
        self.submission_tool = CaptchaSubmissionTool(agentcore_client)
        
        print("✅ End-to-End CAPTCHA Solver initialized")
    
    async def execute(self, captcha_data: Dict[str, Any], **kwargs) -> ToolResult:
        """
        Execute complete CAPTCHA solving workflow
        
        Args:
            captcha_data: CAPTCHA detection data
            **kwargs: Configuration options
            
        Returns:
            ToolResult with complete workflow results
        """
        try:
            workflow_start = time.time()
            print(f"🎯 Starting end-to-end CAPTCHA solving workflow...")
            
            # Extract configuration
            auto_submit = kwargs.get('auto_submit', True)
            confidence_threshold = kwargs.get('confidence_threshold', 0.7)
            max_retries = kwargs.get('max_retries', 2)
            
            # Phase 1: Solve CAPTCHA using AI
            print("🧠 Phase 1: AI-powered CAPTCHA analysis...")
            solving_result = await self.solver_tool.execute(captcha_data, **kwargs)
            
            if not solving_result.success:
                return ToolResult(
                    success=False,
                    error=solving_result.error,
                    message="CAPTCHA solving phase failed"
                )
            
            solution_data = solving_result.data
            confidence = solution_data.get('confidence', 0.0)
            
            print(f"✅ CAPTCHA solved with confidence: {confidence:.2f}")
            
            # Check confidence threshold
            if confidence < confidence_threshold:
                print(f"⚠️ Confidence {confidence:.2f} below threshold {confidence_threshold}")
                
                # Retry with different model if confidence is low
                if max_retries > 0:
                    print("🔄 Retrying with enhanced model configuration...")
                    
                    enhanced_kwargs = kwargs.copy()
                    enhanced_kwargs['model_config'] = {
                        'model_name': 'claude-3-opus',  # Use more powerful model
                        'temperature': 0.05
                    }
                    enhanced_kwargs['max_retries'] = max_retries - 1
                    
                    return await self.execute(captcha_data, **enhanced_kwargs)
                else:
                    return ToolResult(
                        success=False,
                        error=f"Confidence too low: {confidence:.2f}",
                        message="CAPTCHA confidence below acceptable threshold"
                    )
            
            # Phase 2: Submit solution if auto_submit is enabled
            submission_result = None
            if auto_submit and solution_data.get('submission_ready', False):
                print("📤 Phase 2: Submitting CAPTCHA solution...")
                
                submission_result = await self.submission_tool.execute(
                    solution_data=solution_data,
                    detection_data=captcha_data,
                    **kwargs
                )
                
                if submission_result.success:
                    print("✅ CAPTCHA solution submitted successfully")
                else:
                    print(f"❌ CAPTCHA submission failed: {submission_result.error}")
            
            # Compile complete results
            workflow_time = time.time() - workflow_start
            
            complete_result = {
                'workflow_success': True,
                'solution_data': solution_data,
                'submission_data': submission_result.data if submission_result and submission_result.success else None,
                'workflow_metrics': {
                    'total_time': workflow_time,
                    'confidence_score': confidence,
                    'auto_submitted': auto_submit and solution_data.get('submission_ready', False),
                    'submission_success': submission_result.success if submission_result else None
                },
                'captcha_metadata': {
                    'captcha_type': captcha_data.get('captcha_type'),
                    'model_used': solution_data.get('model_used'),
                    'processing_phases': ['analysis', 'submission'] if auto_submit else ['analysis']
                }
            }
            
            return ToolResult(
                success=True,
                data=complete_result,
                message=f"End-to-end CAPTCHA workflow completed in {workflow_time:.2f}s"
            )
            
        except Exception as e:
            print(f"❌ End-to-end CAPTCHA workflow failed: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                message=f"End-to-end workflow error: {e}"
            )


# =============================================================================
# BEDROCK VISION CLIENT
# =============================================================================

class BedrockVisionClient:
    """
    Specialized Bedrock client for vision-based CAPTCHA analysis.
    
    This client provides optimized interfaces for different Bedrock vision models
    with CAPTCHA-specific prompt engineering and response processing.
    """
    
    def __init__(self, region_name: str = 'us-east-1'):
        try:
            import boto3
            self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region_name)
            self.bedrock_available = True
        except ImportError:
            self.bedrock_available = False
            logger.warning("Boto3 not available - using mock Bedrock client")
        
        self.model_configs = {
            'claude-3-sonnet': {
                'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0',
                'max_tokens': 1000,
                'temperature': 0.1,
                'top_p': 0.9
            },
            'claude-3-opus': {
                'model_id': 'anthropic.claude-3-opus-20240229-v1:0',
                'max_tokens': 1500,
                'temperature': 0.05,
                'top_p': 0.95
            },
            'claude-3-haiku': {
                'model_id': 'anthropic.claude-3-haiku-20240307-v1:0',
                'max_tokens': 800,
                'temperature': 0.2,
                'top_p': 0.85
            }
        }
    
    async def analyze_captcha_image(self, image_data: str, captcha_type: CaptchaType, 
                                  model_preference: str = 'claude-3-sonnet') -> Dict[str, Any]:
        """Analyze CAPTCHA image using Bedrock vision models"""
        
        if not self.bedrock_available:
            return await self._mock_vision_analysis(image_data, captcha_type)
        
        model_config = self.model_configs.get(model_preference, self.model_configs['claude-3-sonnet'])
        
        # Build specialized prompt based on CAPTCHA type
        prompt = self._build_vision_prompt(captcha_type)
        
        try:
            # Prepare request payload
            request_payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": model_config['max_tokens'],
                "temperature": model_config['temperature'],
                "top_p": model_config['top_p'],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_data
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            }
            
            # Invoke Bedrock model
            response = self.bedrock_runtime.invoke_model(
                modelId=model_config['model_id'],
                body=json.dumps(request_payload)
            )
            
            # Process response
            response_body = json.loads(response['body'].read())
            
            return await self._process_vision_response(response_body, captcha_type)
            
        except Exception as e:
            logger.error(f"Bedrock vision analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'confidence': 0.0
            }
    
    def _build_vision_prompt(self, captcha_type: CaptchaType) -> str:
        """Build specialized prompts for different CAPTCHA types"""
        
        base_prompt = "You are an expert at analyzing CAPTCHA images. "
        
        if captcha_type == CaptchaType.TEXT_CAPTCHA:
            return base_prompt + """
Analyze this text-based CAPTCHA image and extract the exact text shown.

Instructions:
1. Look carefully at each character, including letters, numbers, and symbols
2. Account for distortions, rotations, and visual noise
3. Distinguish between similar characters (0/O, 1/l/I, 5/S, etc.)
4. Provide only the extracted text, no additional commentary
5. If uncertain about any character, provide your best interpretation

Respond with just the text content."""
        
        elif captcha_type == CaptchaType.IMAGE_SELECTION:
            return base_prompt + """
Analyze this image selection CAPTCHA and identify which grid squares contain the target object.

Instructions:
1. Read the instruction text to understand what object to find
2. Examine each grid square carefully
3. Look for partial objects at the edges of squares
4. Consider different angles, lighting, and image quality
5. Include squares with even partial views of the target object

Respond with a JSON array of grid positions (e.g., [1, 3, 5, 7] for a 3x3 grid numbered 1-9)."""
        
        elif captcha_type in [CaptchaType.RECAPTCHA_V2, CaptchaType.RECAPTCHA_V3]:
            return base_prompt + """
Analyze this reCAPTCHA challenge image.

Instructions:
1. Identify the type of challenge (text, image selection, audio, etc.)
2. If it's an image selection challenge, identify the target objects
3. If it's a text challenge, extract the text accurately
4. Consider the context and instructions provided
5. Provide a clear, actionable response

Respond with the appropriate solution based on the challenge type."""
        
        elif captcha_type == CaptchaType.MATHEMATICAL:
            return base_prompt + """
Analyze this mathematical CAPTCHA and solve the equation shown.

Instructions:
1. Identify all numbers and mathematical operators
2. Account for visual distortions that might affect number recognition
3. Solve the mathematical expression step by step
4. Provide only the numerical answer

Respond with just the calculated result."""
        
        else:
            return base_prompt + """
Analyze this CAPTCHA image and determine the appropriate response.

Instructions:
1. Identify the type of CAPTCHA challenge
2. Extract or identify the required information
3. Provide a clear, actionable response
4. If uncertain, provide your best interpretation

Respond with the solution to the CAPTCHA challenge."""
    
    async def _process_vision_response(self, response_body: Dict[str, Any], 
                                     captcha_type: CaptchaType) -> Dict[str, Any]:
        """Process Bedrock vision model response"""
        
        try:
            # Extract content from response
            content = response_body.get('content', [])
            if not content:
                return {'success': False, 'error': 'Empty response from model', 'confidence': 0.0}
            
            # Get text response
            text_response = content[0].get('text', '').strip()
            
            if not text_response:
                return {'success': False, 'error': 'No text in model response', 'confidence': 0.0}
            
            # Calculate confidence based on response characteristics
            confidence = self._calculate_response_confidence(text_response, captcha_type)
            
            # Process response based on CAPTCHA type
            if captcha_type == CaptchaType.IMAGE_SELECTION:
                try:
                    # Try to parse as JSON array
                    import re
                    json_match = re.search(r'\[([0-9,\s]+)\]', text_response)
                    if json_match:
                        selected_squares = [int(x.strip()) for x in json_match.group(1).split(',') if x.strip()]
                        solution = {'selected_squares': selected_squares}
                    else:
                        # Fallback: extract numbers from response
                        numbers = re.findall(r'\b\d+\b', text_response)
                        solution = {'selected_squares': [int(n) for n in numbers]}
                except:
                    solution = {'selected_squares': []}
                    confidence *= 0.5  # Reduce confidence for parsing errors
            
            elif captcha_type == CaptchaType.MATHEMATICAL:
                try:
                    # Extract numerical answer
                    import re
                    numbers = re.findall(r'-?\d+(?:\.\d+)?', text_response)
                    if numbers:
                        solution = float(numbers[-1])  # Take the last number as the answer
                    else:
                        solution = text_response
                        confidence *= 0.7
                except:
                    solution = text_response
                    confidence *= 0.5
            
            else:
                # For text and other CAPTCHAs, use the response directly
                solution = text_response
            
            return {
                'success': True,
                'solution': solution,
                'confidence': confidence,
                'raw_response': text_response,
                'model_used': response_body.get('model', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Error processing vision response: {e}")
            return {
                'success': False,
                'error': f"Response processing failed: {str(e)}",
                'confidence': 0.0
            }
    
    def _calculate_response_confidence(self, response: str, captcha_type: CaptchaType) -> float:
        """Calculate confidence score based on response characteristics"""
        
        base_confidence = 0.7
        
        # Length-based confidence adjustment
        if captcha_type == CaptchaType.TEXT_CAPTCHA:
            # Text CAPTCHAs typically have 4-8 characters
            if 3 <= len(response) <= 10:
                base_confidence += 0.1
            elif len(response) < 3 or len(response) > 15:
                base_confidence -= 0.2
        
        # Content quality indicators
        if response.isupper() or response.islower():
            base_confidence += 0.05  # Consistent case
        
        if any(char in response for char in ['?', 'unclear', 'uncertain']):
            base_confidence -= 0.3  # Model expressed uncertainty
        
        # Ensure confidence is within valid range
        return max(0.0, min(1.0, base_confidence))
    
    async def _mock_vision_analysis(self, image_data: str, captcha_type: CaptchaType) -> Dict[str, Any]:
        """Mock vision analysis for testing without Bedrock"""
        
        await asyncio.sleep(0.2)  # Simulate processing time
        
        mock_solutions = {
            CaptchaType.TEXT_CAPTCHA: "MOCK123",
            CaptchaType.IMAGE_SELECTION: {'selected_squares': [1, 3, 5]},
            CaptchaType.MATHEMATICAL: 42,
            CaptchaType.RECAPTCHA_V2: {'action': 'click_checkbox'},
            CaptchaType.HCAPTCHA: {'selected_squares': [2, 4, 6]}
        }
        
        solution = mock_solutions.get(captcha_type, "mock_solution")
        
        return {
            'success': True,
            'solution': solution,
            'confidence': 0.85,
            'raw_response': f"Mock analysis for {captcha_type.value}",
            'model_used': 'mock_model'
        }

# =============================================================================
# ENHANCED ERROR HANDLING
# =============================================================================

class CaptchaErrorHandler:
    """
    Comprehensive error handling system for CAPTCHA workflows.
    
    Provides intelligent error classification, recovery strategies,
    and detailed error reporting for production environments.
    """
    
    def __init__(self):
        self.error_history = []
        self.recovery_strategies = self._initialize_recovery_strategies()
        self.error_patterns = self._initialize_error_patterns()
    
    def _initialize_recovery_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize error recovery strategies"""
        
        return {
            'network_timeout': {
                'strategy': 'exponential_backoff',
                'max_retries': 3,
                'base_delay': 2.0,
                'max_delay': 30.0
            },
            'rate_limit': {
                'strategy': 'linear_backoff',
                'max_retries': 5,
                'base_delay': 10.0,
                'max_delay': 60.0
            },
            'captcha_detection_failed': {
                'strategy': 'alternative_detection',
                'max_retries': 2,
                'alternative_strategies': ['comprehensive', 'recaptcha_focused', 'hcaptcha_focused']
            },
            'ai_model_error': {
                'strategy': 'model_fallback',
                'max_retries': 2,
                'fallback_models': ['claude-3-sonnet', 'claude-3-haiku']
            },
            'low_confidence': {
                'strategy': 'enhanced_analysis',
                'max_retries': 1,
                'enhanced_model': 'claude-3-opus'
            },
            'browser_session_error': {
                'strategy': 'session_recreation',
                'max_retries': 2,
                'cleanup_required': True
            }
        }
    
    def _initialize_error_patterns(self) -> Dict[str, List[str]]:
        """Initialize error pattern recognition"""
        
        return {
            'network_timeout': [
                'timeout', 'connection timeout', 'read timeout',
                'network unreachable', 'connection refused'
            ],
            'rate_limit': [
                'rate limit', 'too many requests', '429',
                'quota exceeded', 'throttled'
            ],
            'authentication': [
                'unauthorized', '401', '403', 'access denied',
                'invalid credentials', 'authentication failed'
            ],
            'service_unavailable': [
                'service unavailable', '503', '502', '504',
                'bad gateway', 'gateway timeout'
            ],
            'captcha_not_found': [
                'captcha not found', 'no captcha detected',
                'element not found', 'selector not found'
            ],
            'ai_model_error': [
                'model error', 'inference failed', 'bedrock error',
                'model unavailable', 'token limit exceeded'
            ]
        }
    
    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle error with intelligent recovery strategy"""
        
        error_info = self._classify_error(error, context)
        
        # Record error in history
        error_record = {
            'timestamp': datetime.utcnow(),
            'error_type': error_info['type'],
            'error_message': str(error),
            'context': context,
            'classification': error_info
        }
        self.error_history.append(error_record)
        
        # Determine recovery strategy
        recovery_strategy = self.recovery_strategies.get(
            error_info['type'], 
            self.recovery_strategies['network_timeout']  # Default fallback
        )
        
        # Execute recovery strategy
        recovery_result = await self._execute_recovery_strategy(
            error_info, recovery_strategy, context
        )
        
        return {
            'error_classified': True,
            'error_type': error_info['type'],
            'recovery_strategy': recovery_strategy['strategy'],
            'recovery_result': recovery_result,
            'should_retry': recovery_result.get('should_retry', False),
            'retry_delay': recovery_result.get('retry_delay', 0),
            'modified_context': recovery_result.get('modified_context', context)
        }
    
    def _classify_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Classify error type for appropriate handling"""
        
        error_message = str(error).lower()
        error_type_name = type(error).__name__
        
        # Pattern-based classification
        for error_type, patterns in self.error_patterns.items():
            if any(pattern in error_message for pattern in patterns):
                return {
                    'type': error_type,
                    'confidence': 0.9,
                    'method': 'pattern_matching',
                    'matched_pattern': next(p for p in patterns if p in error_message)
                }
        
        # Exception type-based classification
        type_mappings = {
            'TimeoutError': 'network_timeout',
            'ConnectionError': 'network_timeout',
            'HTTPError': 'service_unavailable',
            'AuthenticationError': 'authentication',
            'RateLimitError': 'rate_limit',
            'CaptchaDetectionError': 'captcha_not_found',
            'CaptchaSolvingError': 'ai_model_error'
        }
        
        if error_type_name in type_mappings:
            return {
                'type': type_mappings[error_type_name],
                'confidence': 0.8,
                'method': 'exception_type',
                'exception_type': error_type_name
            }
        
        # Context-based classification
        if 'captcha_detection' in context.get('current_phase', ''):
            return {
                'type': 'captcha_detection_failed',
                'confidence': 0.7,
                'method': 'context_based'
            }
        elif 'ai_analysis' in context.get('current_phase', ''):
            return {
                'type': 'ai_model_error',
                'confidence': 0.7,
                'method': 'context_based'
            }
        
        # Default classification
        return {
            'type': 'unknown_error',
            'confidence': 0.5,
            'method': 'default',
            'original_error': error_type_name
        }
    
    async def _execute_recovery_strategy(self, error_info: Dict[str, Any], 
                                       strategy_config: Dict[str, Any], 
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the appropriate recovery strategy"""
        
        strategy_name = strategy_config['strategy']
        
        if strategy_name == 'exponential_backoff':
            return await self._exponential_backoff_strategy(strategy_config, context)
        
        elif strategy_name == 'linear_backoff':
            return await self._linear_backoff_strategy(strategy_config, context)
        
        elif strategy_name == 'alternative_detection':
            return await self._alternative_detection_strategy(strategy_config, context)
        
        elif strategy_name == 'model_fallback':
            return await self._model_fallback_strategy(strategy_config, context)
        
        elif strategy_name == 'enhanced_analysis':
            return await self._enhanced_analysis_strategy(strategy_config, context)
        
        elif strategy_name == 'session_recreation':
            return await self._session_recreation_strategy(strategy_config, context)
        
        else:
            return {
                'should_retry': False,
                'reason': f'Unknown recovery strategy: {strategy_name}'
            }
    
    async def _exponential_backoff_strategy(self, config: Dict[str, Any], 
                                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Implement exponential backoff recovery"""
        
        attempt = context.get('retry_attempt', 0)
        max_retries = config.get('max_retries', 3)
        
        if attempt >= max_retries:
            return {
                'should_retry': False,
                'reason': 'Maximum retries exceeded'
            }
        
        base_delay = config.get('base_delay', 2.0)
        max_delay = config.get('max_delay', 30.0)
        
        delay = min(base_delay * (2 ** attempt), max_delay)
        
        return {
            'should_retry': True,
            'retry_delay': delay,
            'modified_context': {
                **context,
                'retry_attempt': attempt + 1,
                'recovery_strategy': 'exponential_backoff'
            }
        }
    
    async def _alternative_detection_strategy(self, config: Dict[str, Any], 
                                            context: Dict[str, Any]) -> Dict[str, Any]:
        """Try alternative detection strategies"""
        
        current_strategy = context.get('detection_strategy', 'comprehensive')
        alternative_strategies = config.get('alternative_strategies', [])
        
        # Remove current strategy from alternatives
        available_alternatives = [s for s in alternative_strategies if s != current_strategy]
        
        if not available_alternatives:
            return {
                'should_retry': False,
                'reason': 'No alternative detection strategies available'
            }
        
        # Select next alternative strategy
        next_strategy = available_alternatives[0]
        
        return {
            'should_retry': True,
            'retry_delay': 1.0,
            'modified_context': {
                **context,
                'detection_strategy': next_strategy,
                'recovery_strategy': 'alternative_detection'
            }
        }
    
    async def _model_fallback_strategy(self, config: Dict[str, Any], 
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to alternative AI models"""
        
        current_model = context.get('model_preference', 'claude-3-sonnet')
        fallback_models = config.get('fallback_models', [])
        
        # Remove current model from fallbacks
        available_fallbacks = [m for m in fallback_models if m != current_model]
        
        if not available_fallbacks:
            return {
                'should_retry': False,
                'reason': 'No fallback models available'
            }
        
        # Select next fallback model
        next_model = available_fallbacks[0]
        
        return {
            'should_retry': True,
            'retry_delay': 2.0,
            'modified_context': {
                **context,
                'model_preference': next_model,
                'recovery_strategy': 'model_fallback'
            }
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        
        if not self.error_history:
            return {'total_errors': 0}
        
        # Count errors by type
        error_counts = {}
        for error in self.error_history:
            error_type = error['classification']['type']
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Calculate error rates over time
        recent_errors = [
            e for e in self.error_history 
            if (datetime.utcnow() - e['timestamp']).total_seconds() < 3600  # Last hour
        ]
        
        return {
            'total_errors': len(self.error_history),
            'recent_errors': len(recent_errors),
            'error_types': error_counts,
            'most_common_error': max(error_counts.items(), key=lambda x: x[1])[0] if error_counts else None,
            'error_rate_per_hour': len(recent_errors),
            'recovery_success_rate': self._calculate_recovery_success_rate()
        }
    
    def _calculate_recovery_success_rate(self) -> float:
        """Calculate the success rate of error recovery strategies"""
        
        recovery_attempts = [
            e for e in self.error_history 
            if 'recovery_strategy' in e.get('context', {})
        ]
        
        if not recovery_attempts:
            return 0.0
        
        # This would need to be tracked in actual implementation
        # For now, return a placeholder
        return 0.75  # 75% recovery success rate

# =============================================================================
# CORE CAPTCHA TOOLS
# =============================================================================

class CaptchaDetectionTool(Tool):
    """
    Tool for detecting CAPTCHAs on web pages using AgentCore Browser Tool
    """
    
    name = "captcha_detector"
    description = "Detects various types of CAPTCHAs on web pages"
    
    def __init__(self):
        super().__init__()
        
    async def execute(self, page_url: str, detection_strategy: str = 'comprehensive', 
                     timeout: int = 30, **kwargs) -> ToolResult:
        """
        Execute CAPTCHA detection on a web page
        
        Args:
            page_url: URL of the page to scan
            detection_strategy: Strategy to use for detection
            timeout: Maximum time to spend on detection
            
        Returns:
            ToolResult with detection results
        """
        
        try:
            print(f"🔍 Detecting CAPTCHAs on {page_url} using {detection_strategy} strategy")
            
            # Simulate detection process
            await asyncio.sleep(1.0)  # Simulate detection time
            
            # Mock detection results based on strategy
            detected_captchas = []
            
            if detection_strategy == 'recaptcha_focused':
                detected_captchas = [{
                    'captcha_type': 'recaptcha_v2',
                    'element_selector': '.g-recaptcha',
                    'bounds': {'x': 100, 'y': 200, 'width': 300, 'height': 150},
                    'confidence': 0.95
                }]
            elif detection_strategy == 'hcaptcha_focused':
                detected_captchas = [{
                    'captcha_type': 'hcaptcha',
                    'element_selector': '.h-captcha',
                    'bounds': {'x': 150, 'y': 250, 'width': 280, 'height': 140},
                    'confidence': 0.90
                }]
            elif detection_strategy == 'comprehensive':
                detected_captchas = [{
                    'captcha_type': 'text_captcha',
                    'element_selector': '#captcha-image',
                    'bounds': {'x': 200, 'y': 300, 'width': 200, 'height': 80},
                    'confidence': 0.85
                }]
            
            captcha_found = len(detected_captchas) > 0
            
            result_data = {
                'captcha_found': captcha_found,
                'detected_captchas': detected_captchas,
                'detection_strategy': detection_strategy,
                'page_url': page_url,
                'detection_time': 1.0
            }
            
            if captcha_found:
                print(f"✅ Found {len(detected_captchas)} CAPTCHA(s)")
            else:
                print("ℹ️ No CAPTCHAs detected")
            
            return ToolResult(
                success=True,
                data=result_data,
                message=f"Detection completed: {len(detected_captchas)} CAPTCHA(s) found"
            )
            
        except Exception as e:
            print(f"❌ CAPTCHA detection failed: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                message=f"Detection error: {e}"
            )


class CaptchaSolvingTool(Tool):
    """
    Tool for solving CAPTCHAs using Bedrock Vision Models
    """
    
    name = "captcha_solver"
    description = "Solves CAPTCHAs using AI vision models"
    
    def __init__(self, agentcore_client=None, vision_client=None):
        super().__init__()
        self.agentcore = agentcore_client
        self.vision_client = vision_client or BedrockVisionClient()
        
    async def execute(self, captcha_data: Dict[str, Any], model_preference: str = 'claude-3-sonnet',
                     confidence_threshold: float = 0.7, **kwargs) -> ToolResult:
        """
        Execute CAPTCHA solving using AI models
        
        Args:
            captcha_data: Data about detected CAPTCHAs
            model_preference: Preferred AI model to use
            confidence_threshold: Minimum confidence required
            
        Returns:
            ToolResult with solving results
        """
        
        try:
            print(f"🧠 Solving CAPTCHAs using {model_preference}")
            
            detected_captchas = captcha_data.get('detected_captchas', [])
            if not detected_captchas:
                return ToolResult(
                    success=False,
                    error="No CAPTCHAs to solve",
                    message="No CAPTCHA data provided"
                )
            
            solutions = []
            
            for captcha in detected_captchas:
                captcha_type_str = captcha.get('captcha_type', 'text_captcha')
                
                # Convert string to CaptchaType enum
                try:
                    captcha_type = CaptchaType(captcha_type_str)
                except ValueError:
                    captcha_type = CaptchaType.GENERIC
                
                # Simulate screenshot capture
                screenshot_data = base64.b64encode(b"mock_image_data").decode('utf-8')
                
                # Analyze with vision model
                analysis_result = await self.vision_client.analyze_captcha_image(
                    screenshot_data, captcha_type, model_preference
                )
                
                if analysis_result['success']:
                    solution = {
                        'captcha_id': captcha.get('element_selector', 'unknown'),
                        'captcha_type': captcha_type_str,
                        'solution': analysis_result['solution'],
                        'confidence': analysis_result['confidence'],
                        'model_used': analysis_result.get('model_used', model_preference),
                        'success': analysis_result['confidence'] >= confidence_threshold,
                        'submission_ready': True
                    }
                    solutions.append(solution)
                    
                    print(f"✅ Solved {captcha_type_str} with confidence {analysis_result['confidence']:.2f}")
                else:
                    print(f"❌ Failed to solve {captcha_type_str}: {analysis_result.get('error', 'Unknown error')}")
            
            successful_solutions = [s for s in solutions if s['success']]
            
            result_data = {
                'solutions': solutions,
                'successful_count': len(successful_solutions),
                'total_count': len(solutions),
                'model_used': model_preference,
                'confidence_threshold': confidence_threshold
            }
            
            return ToolResult(
                success=len(successful_solutions) > 0,
                data=result_data,
                message=f"Solved {len(successful_solutions)}/{len(solutions)} CAPTCHAs"
            )
            
        except Exception as e:
            print(f"❌ CAPTCHA solving failed: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                message=f"Solving error: {e}"
            )


class CaptchaSubmissionTool(Tool):
    """
    Tool for submitting CAPTCHA solutions using AgentCore Browser Tool
    """
    
    name = "captcha_submitter"
    description = "Submits CAPTCHA solutions to web forms"
    
    def __init__(self, agentcore_client=None):
        super().__init__()
        self.agentcore = agentcore_client
        
    async def execute(self, solution_data: Dict[str, Any], detection_data: Dict[str, Any],
                     **kwargs) -> ToolResult:
        """
        Execute CAPTCHA solution submission
        
        Args:
            solution_data: Data about the CAPTCHA solution
            detection_data: Original detection data
            
        Returns:
            ToolResult with submission results
        """
        
        try:
            print("📤 Submitting CAPTCHA solution")
            
            # Simulate submission process
            await asyncio.sleep(0.5)  # Simulate submission time
            
            # Mock submission success based on confidence
            confidence = solution_data.get('confidence', 0.0)
            submission_success = confidence > 0.75  # Higher confidence = higher success rate
            
            result_data = {
                'submitted': True,
                'success': submission_success,
                'solution': solution_data.get('solution', ''),
                'confidence': confidence,
                'submission_time': 0.5
            }
            
            if submission_success:
                print("✅ CAPTCHA solution accepted")
            else:
                print("❌ CAPTCHA solution rejected")
            
            return ToolResult(
                success=submission_success,
                data=result_data,
                message="Solution submitted" if submission_success else "Solution rejected"
            )
            
        except Exception as e:
            print(f"❌ CAPTCHA submission failed: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                message=f"Submission error: {e}"
            )

# =============================================================================
# MAIN CAPTCHA HANDLING AGENT
# =============================================================================

class CaptchaHandlingAgent(Agent):
    """
    Main Strands agent for orchestrating CAPTCHA handling workflows.
    
    This agent coordinates detection, solving, and submission across multiple
    services while maintaining intelligent decision-making and error recovery.
    """
    
    def __init__(self, config: AgentConfig = None):
        super().__init__(config)
        
        # Initialize components
        self.detection_tool = CaptchaDetectionTool()
        self.solving_tool = CaptchaSolvingTool()
        self.state_manager = WorkflowStateManager()
        
        # Configuration
        self.max_attempts = 3
        self.default_timeout = 60
        
        logger.info("Initialized CaptchaHandlingAgent")
    
    async def handle_captcha_workflow(self, page_url: str, task_description: str, 
                                    config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Orchestrate complete CAPTCHA handling workflow.
        
        Args:
            page_url: URL of the page to process
            task_description: Description of the task to perform
            config: Optional configuration parameters
            
        Returns:
            Dictionary containing workflow results and metadata
        """
        
        config = config or {}
        workflow_id = f"captcha_workflow_{int(time.time())}"
        
        # Initialize workflow state
        initial_context = {
            'page_url': page_url,
            'task_description': task_description,
            'config': config
        }
        
        workflow_state = await self.state_manager.initialize_workflow_state(
            workflow_id, initial_context
        )
        
        logger.info(f"🎯 Starting CAPTCHA workflow: {workflow_id}")
        logger.info(f"📋 Page: {page_url}")
        logger.info(f"📋 Task: {task_description}")
        
        try:
            # Phase 1: Detection
            detection_result = await self._orchestrate_detection_phase(workflow_state)
            
            if not detection_result.get('captcha_found', False):
                logger.info("✅ No CAPTCHA detected - proceeding with original task")
                return await self._execute_original_task(workflow_state)
            
            # Phase 2: AI-Powered Solution
            solution_result = await self._orchestrate_solution_phase(workflow_state, detection_result)
            
            # Phase 3: Submission and Verification
            submission_result = await self._orchestrate_submission_phase(workflow_state, solution_result)
            
            # Phase 4: Task Completion
            if submission_result.get('success', False):
                final_result = await self._execute_original_task(workflow_state)
                final_result['captcha_handled'] = True
                final_result['workflow_id'] = workflow_id
                return final_result
            else:
                return await self._handle_captcha_failure(workflow_state, submission_result)
                
        except Exception as e:
            logger.error(f"❌ Workflow orchestration failed: {e}")
            
            # Handle error through state manager
            error_recovery_state = await self.state_manager.handle_workflow_error(
                workflow_id, e, 'retry_current_phase'
            )
            
            return {
                'success': False,
                'error': str(e),
                'workflow_id': workflow_id,
                'workflow_phase': error_recovery_state.current_phase.value,
                'retry_count': error_recovery_state.retry_count
            }
    
    async def _orchestrate_detection_phase(self, workflow_state: WorkflowState) -> Dict[str, Any]:
        """Orchestrate CAPTCHA detection phase"""
        
        await self.state_manager.update_workflow_phase(
            workflow_state.workflow_id, 
            WorkflowPhase.DETECTION, 
            {'phase_start': datetime.utcnow()}
        )
        
        logger.info("🔍 Phase 1: CAPTCHA Detection")
        
        # Choose detection strategy based on page characteristics
        detection_strategy = await self._choose_detection_strategy(workflow_state.page_url)
        logger.info(f"📊 Selected detection strategy: {detection_strategy}")
        
        # Execute detection
        detection_result = await self.detection_tool.execute(
            page_url=workflow_state.page_url,
            detection_strategy=detection_strategy,
            timeout=30
        )
        
        if detection_result.success:
            logger.info(f"✅ CAPTCHA detection completed")
            
            # Update workflow state with detection results
            await self.state_manager.update_workflow_phase(
                workflow_state.workflow_id,
                WorkflowPhase.ANALYSIS,
                {
                    'detection_result': detection_result.data,
                    'detection_strategy': detection_strategy,
                    'phase_completed': datetime.utcnow()
                }
            )
            
            return detection_result.data
        else:
            logger.error(f"❌ CAPTCHA detection failed: {detection_result.error}")
            raise CaptchaDetectionError(f"Detection failed: {detection_result.error}")
    
    async def _orchestrate_solution_phase(self, workflow_state: WorkflowState, 
                                        detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate CAPTCHA solution phase using Bedrock AI"""
        
        await self.state_manager.update_workflow_phase(
            workflow_state.workflow_id,
            WorkflowPhase.SOLUTION,
            {'detection_data': detection_data, 'phase_start': datetime.utcnow()}
        )
        
        logger.info("🧠 Phase 2: AI-Powered CAPTCHA Solution")
        
        detected_captchas = detection_data.get('detected_captchas', [])
        if not detected_captchas:
            raise CaptchaSolvingError("No CAPTCHAs found in detection data")
        
        captcha_type = detected_captchas[0].get('captcha_type', 'unknown')
        
        # Choose AI model based on CAPTCHA complexity
        model_config = await self._choose_ai_model(captcha_type)
        logger.info(f"🤖 Selected AI model: {model_config}")
        
        # Execute solving
        solution_result = await self.solving_tool.execute(
            captcha_data=detection_data,
            model_preference=model_config,
            confidence_threshold=0.7
        )
        
        if solution_result.success:
            solutions = solution_result.data.get('solutions', [])
            successful_solutions = [s for s in solutions if s.get('success', False)]
            
            if successful_solutions:
                avg_confidence = sum(s.get('confidence', 0) for s in successful_solutions) / len(successful_solutions)
                logger.info(f"✅ CAPTCHA solution generated (avg confidence: {avg_confidence:.2f})")
                
                # Update workflow state with solution results
                await self.state_manager.update_workflow_phase(
                    workflow_state.workflow_id,
                    WorkflowPhase.SUBMISSION,
                    {
                        'solution_result': solution_result.data,
                        'model_used': model_config,
                        'phase_completed': datetime.utcnow()
                    }
                )
                
                return solution_result.data
            else:
                raise CaptchaSolvingError("No successful solutions generated")
        else:
            logger.error(f"❌ CAPTCHA solution failed: {solution_result.error}")
            raise CaptchaSolvingError(f"Solution failed: {solution_result.error}")
    
    async def _orchestrate_submission_phase(self, workflow_state: WorkflowState, 
                                          solution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate CAPTCHA solution submission phase"""
        
        await self.state_manager.update_workflow_phase(
            workflow_state.workflow_id,
            WorkflowPhase.VERIFICATION,
            {'solution_data': solution_data, 'phase_start': datetime.utcnow()}
        )
        
        logger.info("📤 Phase 3: Solution Submission")
        
        solutions = solution_data.get('solutions', [])
        successful_solutions = [s for s in solutions if s.get('success', False)]
        
        if not successful_solutions:
            return {'success': False, 'reason': 'no_valid_solutions'}
        
        # Simulate submission process
        await asyncio.sleep(0.2)  # Simulate submission time
        
        # Mock submission success based on confidence
        primary_solution = successful_solutions[0]
        confidence = primary_solution.get('confidence', 0.0)
        
        # Higher confidence = higher success probability
        submission_success = confidence > 0.75
        
        if submission_success:
            logger.info("✅ CAPTCHA solution submitted successfully")
            
            # Update workflow state to completed
            await self.state_manager.update_workflow_phase(
                workflow_state.workflow_id,
                WorkflowPhase.COMPLETED,
                {
                    'submission_result': {'success': True, 'verified': True},
                    'phase_completed': datetime.utcnow()
                }
            )
            
            return {'success': True, 'verified': True}
        else:
            logger.warning("⚠️ CAPTCHA submission not accepted")
            return {'success': False, 'reason': 'submission_rejected'}
    
    async def _choose_detection_strategy(self, page_url: str) -> str:
        """Choose optimal detection strategy based on page characteristics"""
        
        # Analyze page URL to determine likely CAPTCHA types
        if 'google.com' in page_url:
            return 'recaptcha_focused'
        elif 'cloudflare' in page_url:
            return 'comprehensive'  # Cloudflare uses various CAPTCHA types
        else:
            return 'comprehensive'
    
    async def _choose_ai_model(self, captcha_type: str) -> str:
        """Choose optimal AI model based on CAPTCHA type and complexity"""
        
        # Map CAPTCHA types to optimal models
        model_mapping = {
            CaptchaType.TEXT_CAPTCHA: 'claude-3-sonnet',
            CaptchaType.IMAGE_SELECTION: 'claude-3-sonnet',
            CaptchaType.RECAPTCHA_V2: 'claude-3-opus',  # More complex
            CaptchaType.RECAPTCHA_V3: 'claude-3-opus',  # More complex
            CaptchaType.HCAPTCHA: 'claude-3-sonnet'
        }
        
        return model_mapping.get(captcha_type, 'claude-3-sonnet')
    
    async def _execute_original_task(self, workflow_state: WorkflowState) -> Dict[str, Any]:
        """Execute the original task after CAPTCHA handling"""
        
        logger.info(f"🎯 Executing original task: {workflow_state.task_description}")
        
        # Simulate task execution
        await asyncio.sleep(0.1)
        
        return {
            'success': True,
            'task_completed': True,
            'task_description': workflow_state.task_description,
            'execution_time': 0.1,
            'workflow_id': workflow_state.workflow_id
        }
    
    async def _handle_captcha_failure(self, workflow_state: WorkflowState, 
                                    submission_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle CAPTCHA failure with appropriate recovery strategy"""
        
        if workflow_state.retry_count < self.max_attempts:
            logger.info(f"🔄 CAPTCHA failed - retrying (attempt {workflow_state.retry_count + 1})")
            
            # Wait before retry (exponential backoff)
            await asyncio.sleep(2 ** workflow_state.retry_count)
            
            # Retry the workflow
            return await self.handle_captcha_workflow(
                workflow_state.page_url,
                workflow_state.task_description
            )
        else:
            logger.error("❌ Maximum CAPTCHA attempts reached - workflow failed")
            return {
                'success': False,
                'reason': 'max_attempts_reached',
                'attempts': workflow_state.retry_count,
                'last_error': submission_result.get('reason'),
                'workflow_id': workflow_state.workflow_id
            }

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_captcha_agent(config: Dict[str, Any] = None) -> CaptchaHandlingAgent:
    """
    Factory function to create a configured CAPTCHA handling agent.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured CaptchaHandlingAgent instance
    """
    
    agent_config = AgentConfig()
    agent = CaptchaHandlingAgent(agent_config)
    
    if config:
        # Apply configuration settings
        if 'max_attempts' in config:
            agent.max_attempts = config['max_attempts']
        if 'default_timeout' in config:
            agent.default_timeout = config['default_timeout']
    
    return agent

async def validate_framework_setup() -> Dict[str, bool]:
    """
    Validate that the CAPTCHA framework is properly set up.
    
    Returns:
        Dictionary with validation results for each component
    """
    
    validation_results = {}
    
    try:
        # Test agent creation
        agent = create_captcha_agent()
        validation_results['agent_creation'] = True
        
        # Test detection tool
        detection_tool = CaptchaDetectionTool()
        validation_results['detection_tool'] = True
        
        # Test solving tool
        solving_tool = CaptchaSolvingTool()
        validation_results['solving_tool'] = True
        
        # Test state manager
        state_manager = WorkflowStateManager()
        validation_results['state_manager'] = True
        
        logger.info("✅ Framework validation completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Framework validation failed: {e}")
        validation_results['validation_error'] = str(e)
    
    return validation_results

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def example_usage():
    """Example usage of the Strands CAPTCHA handling framework"""
    
    # Create and configure agent
    agent = create_captcha_agent({
        'max_attempts': 3,
        'default_timeout': 60
    })
    
    # Handle CAPTCHA workflow
    result = await agent.handle_captcha_workflow(
        page_url="https://example.com/protected-page",
        task_description="Extract data from protected form"
    )
    
    if result['success']:
        print(f"✅ Workflow completed successfully!")
        print(f"📊 CAPTCHA handled: {result.get('captcha_handled', False)}")
        print(f"🎯 Task completed: {result.get('task_completed', False)}")
    else:
        print(f"❌ Workflow failed: {result.get('error', 'Unknown error')}")
    
    return result

    async def _execute_original_task(self, workflow_state: WorkflowState) -> Dict[str, Any]:
        """Execute the original task after CAPTCHA handling"""
        
        logger.info("🎯 Phase 4: Executing Original Task")
        
        # Simulate task execution
        await asyncio.sleep(0.5)
        
        return {
            'success': True,
            'task_completed': True,
            'task_description': workflow_state.task_description,
            'execution_time': time.time()
        }
    
    async def _handle_captcha_failure(self, workflow_state: WorkflowState, 
                                    submission_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle CAPTCHA failure scenarios"""
        
        logger.error("❌ CAPTCHA handling failed")
        
        # Update workflow state to failed
        await self.state_manager.update_workflow_phase(
            workflow_state.workflow_id,
            WorkflowPhase.FAILED,
            {
                'failure_reason': submission_result.get('reason', 'unknown'),
                'phase_completed': datetime.utcnow()
            }
        )
        
        return {
            'success': False,
            'captcha_handled': False,
            'failure_reason': submission_result.get('reason', 'unknown'),
            'workflow_id': workflow_state.workflow_id
        }
    
    async def _choose_detection_strategy(self, page_url: str) -> str:
        """Choose optimal detection strategy based on page characteristics"""
        
        # Simple heuristic based on URL patterns
        if 'google.com' in page_url:
            return 'recaptcha_focused'
        elif 'cloudflare' in page_url:
            return 'turnstile_focused'
        elif 'hcaptcha.com' in page_url:
            return 'hcaptcha_focused'
        else:
            return 'comprehensive'
    
    async def _choose_ai_model(self, captcha_type: str) -> str:
        """Choose optimal AI model based on CAPTCHA type"""
        
        model_mapping = {
            'text_captcha': 'claude-3-sonnet',
            'image_selection': 'claude-3-opus',
            'recaptcha_v2': 'claude-3-sonnet',
            'recaptcha_v3': 'claude-3-opus',
            'hcaptcha': 'claude-3-opus',
            'mathematical': 'claude-3-opus'
        }
        
        return model_mapping.get(captcha_type, 'claude-3-sonnet')


# =============================================================================
# WORKFLOW STATE MANAGER
# =============================================================================

class WorkflowStateManager:
    """
    Manages workflow state across CAPTCHA handling operations
    """
    
    def __init__(self):
        self.active_workflows: Dict[str, WorkflowState] = {}
        self.completed_workflows: Dict[str, WorkflowState] = {}
        
    async def initialize_workflow_state(self, workflow_id: str, 
                                      initial_context: Dict[str, Any]) -> WorkflowState:
        """Initialize a new workflow state"""
        
        workflow_state = WorkflowState(
            workflow_id=workflow_id,
            created_at=datetime.utcnow(),
            current_phase=WorkflowPhase.INITIALIZATION,
            page_url=initial_context['page_url'],
            task_description=initial_context['task_description'],
            completed_phases=[],
            current_step_data=initial_context,
            error_history=[]
        )
        
        self.active_workflows[workflow_id] = workflow_state
        return workflow_state
    
    async def update_workflow_phase(self, workflow_id: str, new_phase: WorkflowPhase, 
                                  step_data: Dict[str, Any]):
        """Update workflow to new phase"""
        
        if workflow_id in self.active_workflows:
            workflow_state = self.active_workflows[workflow_id]
            
            # Mark previous phase as completed
            if workflow_state.current_phase not in workflow_state.completed_phases:
                workflow_state.completed_phases.append(workflow_state.current_phase)
            
            # Update to new phase
            workflow_state.current_phase = new_phase
            workflow_state.current_step_data.update(step_data)
            
            # Move to completed if workflow is done
            if new_phase in [WorkflowPhase.COMPLETED, WorkflowPhase.FAILED]:
                self.completed_workflows[workflow_id] = workflow_state
                del self.active_workflows[workflow_id]
    
    async def handle_workflow_error(self, workflow_id: str, error: Exception, 
                                  recovery_strategy: str) -> WorkflowState:
        """Handle workflow error and apply recovery strategy"""
        
        if workflow_id in self.active_workflows:
            workflow_state = self.active_workflows[workflow_id]
            
            # Record error
            error_record = {
                'timestamp': datetime.utcnow(),
                'error': str(error),
                'phase': workflow_state.current_phase.value,
                'recovery_strategy': recovery_strategy
            }
            workflow_state.error_history.append(error_record)
            
            # Apply recovery strategy
            if recovery_strategy == 'retry_current_phase':
                workflow_state.retry_count += 1
                if workflow_state.retry_count <= workflow_state.max_retries:
                    logger.info(f"🔄 Retrying phase {workflow_state.current_phase.value} (attempt {workflow_state.retry_count})")
                else:
                    workflow_state.current_phase = WorkflowPhase.FAILED
                    logger.error(f"❌ Max retries exceeded for workflow {workflow_id}")
            
            return workflow_state
        
        raise ValueError(f"Workflow {workflow_id} not found")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_captcha_agent(config: Dict[str, Any] = None) -> CaptchaHandlingAgent:
    """
    Factory function to create a configured CAPTCHA handling agent
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured CaptchaHandlingAgent instance
    """
    
    agent_config = AgentConfig()
    
    if config:
        # Apply configuration overrides
        for key, value in config.items():
            setattr(agent_config, key, value)
    
    return CaptchaHandlingAgent(agent_config)


async def test_framework_integration():
    """
    Test the complete framework integration
    """
    
    print("🧪 Testing Strands CAPTCHA Framework Integration...")
    
    try:
        # Create agent
        agent = create_captcha_agent({
            'max_attempts': 2,
            'default_timeout': 30
        })
        
        # Test workflow
        result = await agent.handle_captcha_workflow(
            page_url="https://example.com/login",
            task_description="Login to user account",
            config={
                'auto_submit': True,
                'confidence_threshold': 0.7
            }
        )
        
        print(f"📊 Framework Test Results:")
        print(f"   Success: {result.get('success', False)}")
        print(f"   CAPTCHA Handled: {result.get('captcha_handled', False)}")
        print(f"   Task Completed: {result.get('task_completed', False)}")
        print(f"   Workflow ID: {result.get('workflow_id', 'N/A')}")
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"❌ Framework test failed: {e}")
        return False


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run framework integration test
    asyncio.run(test_framework_integration())