"""
Strands CAPTCHA Handling Examples

This module contains comprehensive examples demonstrating various aspects of the
Strands CAPTCHA handling framework, from basic usage to advanced enterprise patterns.

Examples are organized by complexity and use case:
- Basic Examples: Simple CAPTCHA detection and solving
- Intermediate Examples: Workflow orchestration and error handling  
- Advanced Examples: Enterprise deployment and production patterns
"""

import asyncio
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

# Import the main framework
from strands_captcha_framework import (
    CaptchaHandlingAgent, CaptchaDetectionTool, CaptchaSolvingTool,
    WorkflowStateManager, CaptchaType, WorkflowPhase, create_captcha_agent
)

# =============================================================================
# BASIC EXAMPLES
# =============================================================================

async def basic_captcha_detection_example():
    """
    Example 1: Basic CAPTCHA Detection
    
    Demonstrates how to use the CaptchaDetectionTool to detect CAPTCHAs
    on a web page with different detection strategies.
    """
    
    print("🔍 Example 1: Basic CAPTCHA Detection")
    print("=" * 50)
    
    # Create detection tool
    detection_tool = CaptchaDetectionTool()
    
    # Test different detection strategies
    test_pages = [
        ("https://example.com/recaptcha-page", "recaptcha_focused"),
        ("https://example.com/hcaptcha-page", "hcaptcha_focused"),
        ("https://example.com/mixed-captcha-page", "comprehensive")
    ]
    
    for page_url, strategy in test_pages:
        print(f"\n📄 Testing {page_url} with {strategy} strategy")
        
        try:
            result = await detection_tool.execute(
                page_url=page_url,
                detection_strategy=strategy,
                timeout=30
            )
            
            if result.success:
                captchas_found = len(result.data.get('detected_captchas', []))
                print(f"✅ Detection successful: {captchas_found} CAPTCHAs found")
                
                # Display detected CAPTCHA details
                for i, captcha in enumerate(result.data.get('detected_captchas', [])):
                    print(f"   CAPTCHA {i+1}: {captcha.get('captcha_type', 'unknown')} "
                          f"(confidence: {captcha.get('confidence', 0):.2f})")
            else:
                print(f"❌ Detection failed: {result.error}")
                
        except Exception as e:
            print(f"❌ Exception during detection: {e}")
    
    print("\n" + "=" * 50)

async def basic_captcha_solving_example():
    """
    Example 2: Basic CAPTCHA Solving
    
    Demonstrates how to use the CaptchaSolvingTool to solve detected CAPTCHAs
    using different AI models and confidence thresholds.
    """
    
    print("🧠 Example 2: Basic CAPTCHA Solving")
    print("=" * 50)
    
    # Create solving tool
    solving_tool = CaptchaSolvingTool()
    
    # Mock detection data for testing
    mock_detection_data = {
        'detected_captchas': [
            {
                'captcha_type': CaptchaType.TEXT_CAPTCHA,
                'bounds': {'x': 100, 'y': 200, 'width': 300, 'height': 150},
                'screenshot': 'mock_screenshot_data',
                'confidence': 0.9
            },
            {
                'captcha_type': CaptchaType.RECAPTCHA_V2,
                'bounds': {'x': 150, 'y': 250, 'width': 320, 'height': 180},
                'screenshot': 'mock_recaptcha_data',
                'confidence': 0.85
            }
        ]
    }
    
    # Test different model preferences and confidence thresholds
    test_configs = [
        ("auto", 0.7),
        ("claude-3-sonnet", 0.8),
        ("claude-3-opus", 0.6)
    ]
    
    for model_preference, confidence_threshold in test_configs:
        print(f"\n🤖 Testing with model: {model_preference}, threshold: {confidence_threshold}")
        
        try:
            result = await solving_tool.execute(
                captcha_data=mock_detection_data,
                model_preference=model_preference,
                confidence_threshold=confidence_threshold
            )
            
            if result.success:
                success_rate = result.data.get('success_rate', 0)
                successful_count = result.data.get('successful_count', 0)
                total_count = result.data.get('total_count', 0)
                
                print(f"✅ Solving successful: {successful_count}/{total_count} "
                      f"(success rate: {success_rate:.2%})")
                
                # Display solution details
                for i, solution in enumerate(result.data.get('solutions', [])):
                    if solution.get('success'):
                        print(f"   Solution {i+1}: {solution.get('solution')} "
                              f"(confidence: {solution.get('confidence', 0):.2f})")
            else:
                print(f"❌ Solving failed: {result.error}")
                
        except Exception as e:
            print(f"❌ Exception during solving: {e}")
    
    print("\n" + "=" * 50)

async def basic_agent_workflow_example():
    """
    Example 3: Basic Agent Workflow
    
    Demonstrates how to use the CaptchaHandlingAgent for complete
    end-to-end CAPTCHA handling workflows.
    """
    
    print("🎯 Example 3: Basic Agent Workflow")
    print("=" * 50)
    
    # Create CAPTCHA handling agent
    agent = create_captcha_agent({
        'max_attempts': 2,
        'default_timeout': 30
    })
    
    # Test different workflow scenarios
    test_scenarios = [
        {
            'page_url': 'https://example.com/simple-form',
            'task': 'Fill out contact form',
            'description': 'Simple form with text CAPTCHA'
        },
        {
            'page_url': 'https://example.com/protected-data',
            'task': 'Extract product information',
            'description': 'Data extraction with reCAPTCHA protection'
        },
        {
            'page_url': 'https://example.com/registration',
            'task': 'Complete user registration',
            'description': 'Registration form with hCaptcha'
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n📋 Scenario: {scenario['description']}")
        print(f"🌐 URL: {scenario['page_url']}")
        print(f"🎯 Task: {scenario['task']}")
        
        try:
            start_time = time.time()
            
            result = await agent.handle_captcha_workflow(
                page_url=scenario['page_url'],
                task_description=scenario['task']
            )
            
            execution_time = time.time() - start_time
            
            if result.get('success', False):
                print(f"✅ Workflow completed successfully in {execution_time:.2f}s")
                print(f"   CAPTCHA handled: {result.get('captcha_handled', False)}")
                print(f"   Task completed: {result.get('task_completed', False)}")
                print(f"   Workflow ID: {result.get('workflow_id', 'N/A')}")
            else:
                print(f"❌ Workflow failed: {result.get('error', 'Unknown error')}")
                print(f"   Reason: {result.get('reason', 'N/A')}")
                
        except Exception as e:
            print(f"❌ Exception during workflow: {e}")
    
    print("\n" + "=" * 50)

# =============================================================================
# INTERMEDIATE EXAMPLES
# =============================================================================

async def error_handling_example():
    """
    Example 4: Advanced Error Handling
    
    Demonstrates comprehensive error handling patterns including
    retry strategies, fallback mechanisms, and recovery procedures.
    """
    
    print("🛡️ Example 4: Advanced Error Handling")
    print("=" * 50)
    
    class RobustCaptchaAgent(CaptchaHandlingAgent):
        """Enhanced agent with advanced error handling capabilities"""
        
        async def handle_captcha_with_recovery(self, page_url: str, task_description: str):
            """Handle CAPTCHA with comprehensive error recovery"""
            
            max_attempts = 3
            recovery_strategies = ['retry_current_phase', 'rollback_to_checkpoint', 'alternative_approach']
            
            for attempt in range(max_attempts):
                for strategy in recovery_strategies:
                    try:
                        print(f"🔄 Attempt {attempt + 1} with strategy: {strategy}")
                        
                        result = await self.handle_captcha_workflow(page_url, task_description)
                        
                        if result.get('success', False):
                            print(f"✅ Success with strategy: {strategy}")
                            return result
                        else:
                            print(f"⚠️ Failed with strategy: {strategy} - {result.get('error', 'Unknown')}")
                            
                    except Exception as e:
                        print(f"❌ Exception with strategy {strategy}: {e}")
                        
                        # Apply recovery based on error type
                        if "timeout" in str(e).lower():
                            print("⏱️ Timeout detected - increasing timeout for next attempt")
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        elif "rate limit" in str(e).lower():
                            print("🚦 Rate limit detected - waiting before retry")
                            await asyncio.sleep(5 * (attempt + 1))
                        else:
                            print("🔧 Generic error - applying standard recovery")
                            await asyncio.sleep(1)
            
            return {
                'success': False,
                'error': 'All recovery strategies exhausted',
                'attempts': max_attempts,
                'strategies_tried': recovery_strategies
            }
    
    # Test error handling with various failure scenarios
    robust_agent = RobustCaptchaAgent()
    
    error_scenarios = [
        {
            'url': 'https://example.com/timeout-page',
            'task': 'Handle timeout scenario',
            'expected_error': 'timeout'
        },
        {
            'url': 'https://example.com/rate-limited-page',
            'task': 'Handle rate limiting',
            'expected_error': 'rate_limit'
        },
        {
            'url': 'https://example.com/complex-captcha',
            'task': 'Handle complex CAPTCHA',
            'expected_error': 'solving_failure'
        }
    ]
    
    for scenario in error_scenarios:
        print(f"\n🧪 Testing error scenario: {scenario['expected_error']}")
        print(f"🌐 URL: {scenario['url']}")
        
        try:
            result = await robust_agent.handle_captcha_with_recovery(
                scenario['url'], scenario['task']
            )
            
            if result.get('success', False):
                print(f"✅ Error scenario handled successfully")
            else:
                print(f"❌ Error scenario failed: {result.get('error', 'Unknown')}")
                print(f"   Attempts made: {result.get('attempts', 0)}")
                
        except Exception as e:
            print(f"❌ Unhandled exception: {e}")
    
    print("\n" + "=" * 50)

async def performance_optimization_example():
    """
    Example 5: Performance Optimization
    
    Demonstrates performance optimization techniques including
    caching, connection pooling, and parallel processing.
    """
    
    print("⚡ Example 5: Performance Optimization")
    print("=" * 50)
    
    class OptimizedCaptchaAgent(CaptchaHandlingAgent):
        """Agent with performance optimizations"""
        
        def __init__(self, config=None):
            super().__init__(config)
            
            # Initialize caches
            self.detection_cache = {}
            self.solution_cache = {}
            
            # Performance metrics
            self.metrics = {
                'cache_hits': 0,
                'cache_misses': 0,
                'total_requests': 0,
                'average_response_time': 0.0
            }
        
        async def handle_captcha_with_caching(self, page_url: str, task_description: str):
            """Handle CAPTCHA with intelligent caching"""
            
            start_time = time.time()
            self.metrics['total_requests'] += 1
            
            # Generate cache key
            cache_key = self._generate_cache_key(page_url, task_description)
            
            # Check cache first
            if cache_key in self.detection_cache:
                print(f"💾 Cache hit for {page_url}")
                self.metrics['cache_hits'] += 1
                
                cached_result = self.detection_cache[cache_key]
                
                # Validate cache freshness (5 minutes TTL)
                if time.time() - cached_result['timestamp'] < 300:
                    return cached_result['result']
                else:
                    print(f"⏰ Cache expired for {page_url}")
                    del self.detection_cache[cache_key]
            
            # Cache miss - execute workflow
            print(f"🔍 Cache miss for {page_url}")
            self.metrics['cache_misses'] += 1
            
            result = await self.handle_captcha_workflow(page_url, task_description)
            
            # Cache successful results
            if result.get('success', False):
                self.detection_cache[cache_key] = {
                    'result': result,
                    'timestamp': time.time()
                }
                print(f"💾 Cached result for {page_url}")
            
            # Update performance metrics
            execution_time = time.time() - start_time
            self.metrics['average_response_time'] = (
                (self.metrics['average_response_time'] * (self.metrics['total_requests'] - 1) + execution_time) /
                self.metrics['total_requests']
            )
            
            return result
        
        async def handle_multiple_captchas_parallel(self, requests: List[Dict[str, str]]):
            """Handle multiple CAPTCHA requests in parallel"""
            
            print(f"🚀 Processing {len(requests)} requests in parallel")
            
            # Create tasks for parallel execution
            tasks = []
            for i, request in enumerate(requests):
                task = asyncio.create_task(
                    self.handle_captcha_with_caching(
                        request['page_url'], 
                        request['task_description']
                    )
                )
                tasks.append(task)
            
            # Execute with timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=60
                )
                
                # Process results
                successful_results = []
                failed_results = []
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        failed_results.append({
                            'request_index': i,
                            'error': str(result)
                        })
                    elif result.get('success', False):
                        successful_results.append(result)
                    else:
                        failed_results.append({
                            'request_index': i,
                            'error': result.get('error', 'Unknown error')
                        })
                
                return {
                    'successful_count': len(successful_results),
                    'failed_count': len(failed_results),
                    'total_count': len(requests),
                    'success_rate': len(successful_results) / len(requests),
                    'successful_results': successful_results,
                    'failed_results': failed_results
                }
                
            except asyncio.TimeoutError:
                return {
                    'success': False,
                    'error': 'Parallel processing timed out',
                    'timeout': 60
                }
        
        def _generate_cache_key(self, page_url: str, task_description: str) -> str:
            """Generate cache key for request"""
            import hashlib
            
            key_data = f"{page_url}:{task_description}"
            return hashlib.md5(key_data.encode()).hexdigest()
        
        def get_performance_metrics(self) -> Dict[str, Any]:
            """Get current performance metrics"""
            
            cache_hit_rate = (
                self.metrics['cache_hits'] / 
                (self.metrics['cache_hits'] + self.metrics['cache_misses'])
                if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0
            )
            
            return {
                **self.metrics,
                'cache_hit_rate': cache_hit_rate,
                'cache_size': len(self.detection_cache)
            }
    
    # Test performance optimizations
    optimized_agent = OptimizedCaptchaAgent()
    
    # Test caching with repeated requests
    print("\n🧪 Testing caching performance")
    repeated_requests = [
        {'page_url': 'https://example.com/form1', 'task_description': 'Fill form 1'},
        {'page_url': 'https://example.com/form2', 'task_description': 'Fill form 2'},
        {'page_url': 'https://example.com/form1', 'task_description': 'Fill form 1'},  # Repeat
        {'page_url': 'https://example.com/form3', 'task_description': 'Fill form 3'},
        {'page_url': 'https://example.com/form2', 'task_description': 'Fill form 2'},  # Repeat
    ]
    
    for request in repeated_requests:
        result = await optimized_agent.handle_captcha_with_caching(
            request['page_url'], request['task_description']
        )
    
    # Display performance metrics
    metrics = optimized_agent.get_performance_metrics()
    print(f"\n📊 Performance Metrics:")
    print(f"   Total requests: {metrics['total_requests']}")
    print(f"   Cache hits: {metrics['cache_hits']}")
    print(f"   Cache misses: {metrics['cache_misses']}")
    print(f"   Cache hit rate: {metrics['cache_hit_rate']:.2%}")
    print(f"   Average response time: {metrics['average_response_time']:.3f}s")
    print(f"   Cache size: {metrics['cache_size']}")
    
    # Test parallel processing
    print(f"\n🧪 Testing parallel processing")
    parallel_requests = [
        {'page_url': f'https://example.com/page{i}', 'task_description': f'Task {i}'}
        for i in range(1, 6)
    ]
    
    parallel_result = await optimized_agent.handle_multiple_captchas_parallel(parallel_requests)
    
    print(f"📊 Parallel Processing Results:")
    print(f"   Successful: {parallel_result['successful_count']}")
    print(f"   Failed: {parallel_result['failed_count']}")
    print(f"   Success rate: {parallel_result['success_rate']:.2%}")
    
    print("\n" + "=" * 50)

async def retry_recovery_patterns_example():
    """
    Example 6: Retry and Recovery Patterns
    
    Demonstrates sophisticated retry and recovery patterns for
    handling various failure scenarios in CAPTCHA workflows.
    """
    
    print("🔄 Example 6: Retry and Recovery Patterns")
    print("=" * 50)
    
    class RetryRecoveryAgent(CaptchaHandlingAgent):
        """Agent with advanced retry and recovery patterns"""
        
        async def handle_captcha_with_adaptive_retry(self, page_url: str, task_description: str):
            """Handle CAPTCHA with adaptive retry strategies"""
            
            retry_strategies = [
                {'name': 'immediate_retry', 'delay': 0, 'max_attempts': 2},
                {'name': 'exponential_backoff', 'delay': 1, 'max_attempts': 3},
                {'name': 'linear_backoff', 'delay': 2, 'max_attempts': 2},
                {'name': 'jittered_retry', 'delay': 1, 'max_attempts': 2}
            ]
            
            for strategy in retry_strategies:
                print(f"\n🔧 Trying strategy: {strategy['name']}")
                
                for attempt in range(strategy['max_attempts']):
                    try:
                        print(f"   Attempt {attempt + 1}/{strategy['max_attempts']}")
                        
                        result = await self.handle_captcha_workflow(page_url, task_description)
                        
                        if result.get('success', False):
                            print(f"   ✅ Success with {strategy['name']} on attempt {attempt + 1}")
                            return result
                        else:
                            print(f"   ⚠️ Failed attempt {attempt + 1}: {result.get('error', 'Unknown')}")
                            
                            # Apply strategy-specific delay
                            if attempt < strategy['max_attempts'] - 1:  # Don't delay after last attempt
                                delay = self._calculate_retry_delay(strategy, attempt)
                                print(f"   ⏱️ Waiting {delay:.2f}s before retry")
                                await asyncio.sleep(delay)
                                
                    except Exception as e:
                        print(f"   ❌ Exception on attempt {attempt + 1}: {e}")
                        
                        if attempt < strategy['max_attempts'] - 1:
                            delay = self._calculate_retry_delay(strategy, attempt)
                            await asyncio.sleep(delay)
                
                print(f"   ❌ Strategy {strategy['name']} exhausted")
            
            return {
                'success': False,
                'error': 'All retry strategies exhausted',
                'strategies_tried': [s['name'] for s in retry_strategies]
            }
        
        def _calculate_retry_delay(self, strategy: Dict[str, Any], attempt: int) -> float:
            """Calculate retry delay based on strategy"""
            
            base_delay = strategy['delay']
            
            if strategy['name'] == 'immediate_retry':
                return 0
            elif strategy['name'] == 'exponential_backoff':
                return base_delay * (2 ** attempt)
            elif strategy['name'] == 'linear_backoff':
                return base_delay * (attempt + 1)
            elif strategy['name'] == 'jittered_retry':
                import random
                jitter = random.uniform(0.5, 1.5)
                return base_delay * jitter
            else:
                return base_delay
        
        async def handle_captcha_with_circuit_breaker(self, page_url: str, task_description: str):
            """Handle CAPTCHA with circuit breaker pattern"""
            
            # Circuit breaker state
            if not hasattr(self, 'circuit_breaker_state'):
                self.circuit_breaker_state = {
                    'failure_count': 0,
                    'last_failure_time': None,
                    'state': 'closed',  # closed, open, half_open
                    'failure_threshold': 3,
                    'recovery_timeout': 30  # seconds
                }
            
            cb_state = self.circuit_breaker_state
            
            # Check circuit breaker state
            if cb_state['state'] == 'open':
                # Check if recovery timeout has passed
                if (time.time() - cb_state['last_failure_time']) > cb_state['recovery_timeout']:
                    print("🔄 Circuit breaker transitioning to half-open")
                    cb_state['state'] = 'half_open'
                else:
                    print("⚡ Circuit breaker is open - failing fast")
                    return {
                        'success': False,
                        'error': 'Circuit breaker is open',
                        'circuit_breaker_state': cb_state['state']
                    }
            
            try:
                print(f"🔧 Executing with circuit breaker in {cb_state['state']} state")
                
                result = await self.handle_captcha_workflow(page_url, task_description)
                
                if result.get('success', False):
                    # Success - reset circuit breaker
                    if cb_state['state'] == 'half_open':
                        print("✅ Circuit breaker closing after successful recovery")
                        cb_state['state'] = 'closed'
                    
                    cb_state['failure_count'] = 0
                    return result
                else:
                    # Failure - increment counter
                    cb_state['failure_count'] += 1
                    cb_state['last_failure_time'] = time.time()
                    
                    if cb_state['failure_count'] >= cb_state['failure_threshold']:
                        print(f"⚡ Circuit breaker opening after {cb_state['failure_count']} failures")
                        cb_state['state'] = 'open'
                    
                    return result
                    
            except Exception as e:
                # Exception - treat as failure
                cb_state['failure_count'] += 1
                cb_state['last_failure_time'] = time.time()
                
                if cb_state['failure_count'] >= cb_state['failure_threshold']:
                    print(f"⚡ Circuit breaker opening after exception")
                    cb_state['state'] = 'open'
                
                return {
                    'success': False,
                    'error': str(e),
                    'circuit_breaker_state': cb_state['state']
                }
    
    # Test retry and recovery patterns
    retry_agent = RetryRecoveryAgent()
    
    # Test adaptive retry
    print("\n🧪 Testing adaptive retry strategies")
    result = await retry_agent.handle_captcha_with_adaptive_retry(
        'https://example.com/unreliable-page',
        'Handle unreliable CAPTCHA'
    )
    
    if result.get('success', False):
        print("✅ Adaptive retry succeeded")
    else:
        print(f"❌ Adaptive retry failed: {result.get('error', 'Unknown')}")
        print(f"   Strategies tried: {result.get('strategies_tried', [])}")
    
    # Test circuit breaker pattern
    print(f"\n🧪 Testing circuit breaker pattern")
    
    # Simulate multiple failures to trigger circuit breaker
    for i in range(5):
        print(f"\n🔄 Circuit breaker test {i + 1}")
        
        result = await retry_agent.handle_captcha_with_circuit_breaker(
            'https://example.com/failing-page',
            f'Circuit breaker test {i + 1}'
        )
        
        if result.get('success', False):
            print(f"✅ Circuit breaker test {i + 1} succeeded")
        else:
            print(f"❌ Circuit breaker test {i + 1} failed: {result.get('error', 'Unknown')}")
            print(f"   Circuit breaker state: {result.get('circuit_breaker_state', 'unknown')}")
    
    print("\n" + "=" * 50)

# =============================================================================
# ADVANCED EXAMPLES
# =============================================================================

async def enterprise_deployment_example():
    """
    Example 7: Enterprise Deployment Patterns
    
    Demonstrates enterprise-grade deployment patterns including
    monitoring, logging, security, and scalability features.
    """
    
    print("🏢 Example 7: Enterprise Deployment Patterns")
    print("=" * 50)
    
    class EnterpriseCaptchaAgent(CaptchaHandlingAgent):
        """Enterprise-grade CAPTCHA agent with comprehensive features"""
        
        def __init__(self, config=None):
            super().__init__(config)
            
            # Enterprise features
            self.audit_logger = self._setup_audit_logging()
            self.metrics_collector = self._setup_metrics_collection()
            self.security_manager = self._setup_security_manager()
            
        def _setup_audit_logging(self):
            """Setup comprehensive audit logging"""
            import logging
            
            audit_logger = logging.getLogger('captcha_audit')
            audit_logger.setLevel(logging.INFO)
            
            # Create audit log handler
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            audit_logger.addHandler(handler)
            
            return audit_logger
        
        def _setup_metrics_collection(self):
            """Setup metrics collection for monitoring"""
            return {
                'requests_total': 0,
                'requests_successful': 0,
                'requests_failed': 0,
                'captchas_detected': 0,
                'captchas_solved': 0,
                'average_response_time': 0.0,
                'error_rates': {},
                'security_events': 0
            }
        
        def _setup_security_manager(self):
            """Setup security management features"""
            return {
                'rate_limits': {},
                'blocked_ips': set(),
                'suspicious_patterns': [],
                'encryption_key': 'mock_encryption_key'
            }
        
        async def handle_captcha_enterprise(self, page_url: str, task_description: str, 
                                          user_context: Dict[str, Any] = None):
            """Handle CAPTCHA with enterprise security and monitoring"""
            
            user_context = user_context or {}
            request_id = f"req_{int(time.time() * 1000)}"
            
            # Security validation
            security_check = await self._validate_security(page_url, user_context)
            if not security_check['valid']:
                self.audit_logger.warning(
                    f"Security validation failed for request {request_id}: {security_check['reason']}"
                )
                return {
                    'success': False,
                    'error': 'Security validation failed',
                    'request_id': request_id
                }
            
            # Rate limiting
            rate_limit_check = await self._check_rate_limits(user_context.get('user_id', 'anonymous'))
            if not rate_limit_check['allowed']:
                self.audit_logger.warning(
                    f"Rate limit exceeded for request {request_id}"
                )
                return {
                    'success': False,
                    'error': 'Rate limit exceeded',
                    'request_id': request_id
                }
            
            # Start monitoring
            start_time = time.time()
            self.metrics_collector['requests_total'] += 1
            
            # Audit log start
            self.audit_logger.info(
                f"Starting CAPTCHA workflow - Request: {request_id}, "
                f"URL: {page_url}, User: {user_context.get('user_id', 'anonymous')}"
            )
            
            try:
                # Execute workflow with monitoring
                result = await self._execute_monitored_workflow(
                    page_url, task_description, request_id
                )
                
                # Update metrics
                execution_time = time.time() - start_time
                self._update_metrics(result, execution_time)
                
                # Audit log completion
                self.audit_logger.info(
                    f"Completed CAPTCHA workflow - Request: {request_id}, "
                    f"Success: {result.get('success', False)}, "
                    f"Duration: {execution_time:.3f}s"
                )
                
                return {
                    **result,
                    'request_id': request_id,
                    'execution_time': execution_time,
                    'security_validated': True
                }
                
            except Exception as e:
                # Update error metrics
                execution_time = time.time() - start_time
                self.metrics_collector['requests_failed'] += 1
                
                error_type = type(e).__name__
                if error_type not in self.metrics_collector['error_rates']:
                    self.metrics_collector['error_rates'][error_type] = 0
                self.metrics_collector['error_rates'][error_type] += 1
                
                # Audit log error
                self.audit_logger.error(
                    f"CAPTCHA workflow failed - Request: {request_id}, "
                    f"Error: {str(e)}, Duration: {execution_time:.3f}s"
                )
                
                return {
                    'success': False,
                    'error': str(e),
                    'request_id': request_id,
                    'execution_time': execution_time
                }
        
        async def _validate_security(self, page_url: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
            """Validate security requirements"""
            
            # Check blocked domains
            blocked_domains = ['malicious.com', 'spam.net']
            for domain in blocked_domains:
                if domain in page_url:
                    return {'valid': False, 'reason': f'Blocked domain: {domain}'}
            
            # Check user permissions
            user_id = user_context.get('user_id')
            if user_id and user_id in self.security_manager['blocked_ips']:
                return {'valid': False, 'reason': 'User blocked'}
            
            # Check for suspicious patterns
            if len(page_url) > 1000:  # Suspiciously long URL
                self.security_manager['suspicious_patterns'].append({
                    'type': 'long_url',
                    'url': page_url[:100] + '...',
                    'timestamp': datetime.utcnow()
                })
                return {'valid': False, 'reason': 'Suspicious URL pattern'}
            
            return {'valid': True, 'reason': 'Security validation passed'}
        
        async def _check_rate_limits(self, user_id: str) -> Dict[str, Any]:
            """Check rate limiting for user"""
            
            current_time = time.time()
            
            # Initialize rate limit tracking for user
            if user_id not in self.security_manager['rate_limits']:
                self.security_manager['rate_limits'][user_id] = {
                    'requests': [],
                    'window_start': current_time
                }
            
            user_limits = self.security_manager['rate_limits'][user_id]
            
            # Clean old requests (1-minute window)
            window_duration = 60  # seconds
            cutoff_time = current_time - window_duration
            user_limits['requests'] = [
                req_time for req_time in user_limits['requests'] 
                if req_time > cutoff_time
            ]
            
            # Check if under limit (10 requests per minute)
            max_requests = 10
            if len(user_limits['requests']) >= max_requests:
                return {'allowed': False, 'reason': 'Rate limit exceeded'}
            
            # Record this request
            user_limits['requests'].append(current_time)
            
            return {'allowed': True, 'remaining': max_requests - len(user_limits['requests'])}
        
        async def _execute_monitored_workflow(self, page_url: str, task_description: str, 
                                            request_id: str) -> Dict[str, Any]:
            """Execute workflow with comprehensive monitoring"""
            
            # Add monitoring context
            monitoring_context = {
                'request_id': request_id,
                'monitoring_enabled': True,
                'metrics_collection': True
            }
            
            # Execute base workflow
            result = await self.handle_captcha_workflow(page_url, task_description)
            
            # Collect additional metrics
            if result.get('captcha_handled', False):
                self.metrics_collector['captchas_detected'] += 1
                
                if result.get('success', False):
                    self.metrics_collector['captchas_solved'] += 1
            
            return result
        
        def _update_metrics(self, result: Dict[str, Any], execution_time: float):
            """Update performance metrics"""
            
            if result.get('success', False):
                self.metrics_collector['requests_successful'] += 1
            else:
                self.metrics_collector['requests_failed'] += 1
            
            # Update average response time
            total_requests = self.metrics_collector['requests_total']
            current_avg = self.metrics_collector['average_response_time']
            
            self.metrics_collector['average_response_time'] = (
                (current_avg * (total_requests - 1) + execution_time) / total_requests
            )
        
        def get_enterprise_metrics(self) -> Dict[str, Any]:
            """Get comprehensive enterprise metrics"""
            
            total_requests = self.metrics_collector['requests_total']
            
            return {
                'performance_metrics': {
                    'total_requests': total_requests,
                    'successful_requests': self.metrics_collector['requests_successful'],
                    'failed_requests': self.metrics_collector['requests_failed'],
                    'success_rate': (
                        self.metrics_collector['requests_successful'] / total_requests
                        if total_requests > 0 else 0
                    ),
                    'average_response_time': self.metrics_collector['average_response_time'],
                    'captchas_detected': self.metrics_collector['captchas_detected'],
                    'captchas_solved': self.metrics_collector['captchas_solved'],
                    'captcha_solve_rate': (
                        self.metrics_collector['captchas_solved'] / 
                        self.metrics_collector['captchas_detected']
                        if self.metrics_collector['captchas_detected'] > 0 else 0
                    )
                },
                'security_metrics': {
                    'security_events': self.metrics_collector['security_events'],
                    'blocked_users': len(self.security_manager['blocked_ips']),
                    'suspicious_patterns': len(self.security_manager['suspicious_patterns']),
                    'rate_limited_users': len(self.security_manager['rate_limits'])
                },
                'error_analysis': self.metrics_collector['error_rates']
            }
    
    # Test enterprise deployment
    enterprise_agent = EnterpriseCaptchaAgent()
    
    # Test with different user contexts
    test_users = [
        {'user_id': 'user_001', 'role': 'admin', 'permissions': ['captcha_solve']},
        {'user_id': 'user_002', 'role': 'user', 'permissions': ['basic_access']},
        {'user_id': 'anonymous', 'role': 'guest', 'permissions': []}
    ]
    
    test_scenarios = [
        'https://example.com/secure-form',
        'https://example.com/public-data',
        'https://example.com/protected-resource'
    ]
    
    print("\n🧪 Testing enterprise deployment scenarios")
    
    for user in test_users:
        for scenario_url in test_scenarios:
            print(f"\n👤 User: {user['user_id']} ({user['role']})")
            print(f"🌐 URL: {scenario_url}")
            
            result = await enterprise_agent.handle_captcha_enterprise(
                page_url=scenario_url,
                task_description=f"Enterprise task for {user['user_id']}",
                user_context=user
            )
            
            if result.get('success', False):
                print(f"✅ Enterprise workflow succeeded")
                print(f"   Request ID: {result.get('request_id', 'N/A')}")
                print(f"   Execution time: {result.get('execution_time', 0):.3f}s")
            else:
                print(f"❌ Enterprise workflow failed: {result.get('error', 'Unknown')}")
                print(f"   Request ID: {result.get('request_id', 'N/A')}")
    
    # Display enterprise metrics
    metrics = enterprise_agent.get_enterprise_metrics()
    
    print(f"\n📊 Enterprise Metrics Summary:")
    print(f"Performance:")
    perf = metrics['performance_metrics']
    print(f"   Total requests: {perf['total_requests']}")
    print(f"   Success rate: {perf['success_rate']:.2%}")
    print(f"   Average response time: {perf['average_response_time']:.3f}s")
    print(f"   CAPTCHA solve rate: {perf['captcha_solve_rate']:.2%}")
    
    print(f"Security:")
    sec = metrics['security_metrics']
    print(f"   Security events: {sec['security_events']}")
    print(f"   Blocked users: {sec['blocked_users']}")
    print(f"   Suspicious patterns: {sec['suspicious_patterns']}")
    
    if metrics['error_analysis']:
        print(f"Error Analysis:")
        for error_type, count in metrics['error_analysis'].items():
            print(f"   {error_type}: {count}")
    
    print("\n" + "=" * 50)

async def production_example():
    """
    Example 8: Production-Ready Implementation
    
    Demonstrates a complete production-ready CAPTCHA handling system
    with all enterprise features, monitoring, and best practices.
    """
    
    print("🚀 Example 8: Production-Ready Implementation")
    print("=" * 50)
    
    class ProductionCaptchaSystem:
        """Complete production-ready CAPTCHA handling system"""
        
        def __init__(self, config: Dict[str, Any]):
            self.config = config
            
            # Initialize core components
            self.agent = self._create_production_agent()
            self.monitoring = self._setup_monitoring()
            self.health_checker = self._setup_health_checks()
            
            # Production features
            self.load_balancer = self._setup_load_balancing()
            self.circuit_breakers = {}
            self.metrics_aggregator = self._setup_metrics_aggregation()
            
        def _create_production_agent(self):
            """Create production-configured agent"""
            
            agent_config = {
                'max_attempts': self.config.get('max_attempts', 3),
                'default_timeout': self.config.get('timeout', 60),
                'enable_caching': self.config.get('caching', True),
                'enable_monitoring': self.config.get('monitoring', True)
            }
            
            return create_captcha_agent(agent_config)
        
        def _setup_monitoring(self):
            """Setup comprehensive monitoring system"""
            
            return {
                'system_health': {
                    'status': 'healthy',
                    'last_check': datetime.utcnow(),
                    'uptime': 0,
                    'version': '1.0.0'
                },
                'performance_metrics': {
                    'requests_per_second': 0,
                    'average_latency': 0,
                    'error_rate': 0,
                    'throughput': 0
                },
                'resource_usage': {
                    'cpu_usage': 0,
                    'memory_usage': 0,
                    'disk_usage': 0,
                    'network_io': 0
                }
            }
        
        def _setup_health_checks(self):
            """Setup health check system"""
            
            return {
                'endpoints': [
                    {'name': 'agent_health', 'status': 'unknown', 'last_check': None},
                    {'name': 'detection_service', 'status': 'unknown', 'last_check': None},
                    {'name': 'solving_service', 'status': 'unknown', 'last_check': None},
                    {'name': 'database', 'status': 'unknown', 'last_check': None}
                ],
                'overall_status': 'unknown'
            }
        
        def _setup_load_balancing(self):
            """Setup load balancing for high availability"""
            
            return {
                'strategy': 'round_robin',
                'instances': [
                    {'id': 'instance_1', 'status': 'active', 'load': 0},
                    {'id': 'instance_2', 'status': 'active', 'load': 0},
                    {'id': 'instance_3', 'status': 'standby', 'load': 0}
                ],
                'current_instance': 0
            }
        
        def _setup_metrics_aggregation(self):
            """Setup metrics aggregation for monitoring"""
            
            return {
                'collection_interval': 60,  # seconds
                'retention_period': 86400,  # 24 hours
                'aggregated_metrics': [],
                'alerts': []
            }
        
        async def handle_production_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
            """Handle production CAPTCHA request with full monitoring"""
            
            request_id = request.get('request_id', f"prod_{int(time.time() * 1000)}")
            
            # Health check before processing
            health_status = await self.perform_health_check()
            if health_status['overall_status'] != 'healthy':
                return {
                    'success': False,
                    'error': 'System unhealthy',
                    'health_status': health_status,
                    'request_id': request_id
                }
            
            # Load balancing
            instance = self._select_instance()
            if not instance:
                return {
                    'success': False,
                    'error': 'No available instances',
                    'request_id': request_id
                }
            
            # Circuit breaker check
            circuit_key = f"instance_{instance['id']}"
            if not self._check_circuit_breaker(circuit_key):
                return {
                    'success': False,
                    'error': 'Circuit breaker open',
                    'instance': instance['id'],
                    'request_id': request_id
                }
            
            # Execute request with monitoring
            start_time = time.time()
            
            try:
                # Process CAPTCHA request
                result = await self.agent.handle_captcha_workflow(
                    page_url=request['page_url'],
                    task_description=request['task_description']
                )
                
                execution_time = time.time() - start_time
                
                # Update metrics
                await self._update_production_metrics(result, execution_time, instance)
                
                # Circuit breaker success
                self._record_circuit_breaker_success(circuit_key)
                
                return {
                    **result,
                    'request_id': request_id,
                    'instance_id': instance['id'],
                    'execution_time': execution_time,
                    'system_status': 'healthy'
                }
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Circuit breaker failure
                self._record_circuit_breaker_failure(circuit_key)
                
                # Update error metrics
                await self._update_error_metrics(e, execution_time, instance)
                
                return {
                    'success': False,
                    'error': str(e),
                    'request_id': request_id,
                    'instance_id': instance['id'],
                    'execution_time': execution_time
                }
        
        async def perform_health_check(self) -> Dict[str, Any]:
            """Perform comprehensive health check"""
            
            health_results = []
            
            for endpoint in self.health_checker['endpoints']:
                try:
                    # Simulate health check
                    await asyncio.sleep(0.01)  # Simulate check time
                    
                    # Mock health check results
                    if endpoint['name'] == 'agent_health':
                        status = 'healthy' if hasattr(self.agent, 'handle_captcha_workflow') else 'unhealthy'
                    else:
                        status = 'healthy'  # Mock other services as healthy
                    
                    endpoint['status'] = status
                    endpoint['last_check'] = datetime.utcnow()
                    
                    health_results.append(status == 'healthy')
                    
                except Exception as e:
                    endpoint['status'] = 'unhealthy'
                    endpoint['last_check'] = datetime.utcnow()
                    health_results.append(False)
            
            # Determine overall status
            overall_healthy = all(health_results)
            self.health_checker['overall_status'] = 'healthy' if overall_healthy else 'unhealthy'
            
            return {
                'overall_status': self.health_checker['overall_status'],
                'endpoints': self.health_checker['endpoints'],
                'check_time': datetime.utcnow()
            }
        
        def _select_instance(self) -> Optional[Dict[str, Any]]:
            """Select instance using load balancing strategy"""
            
            active_instances = [
                inst for inst in self.load_balancer['instances']
                if inst['status'] == 'active'
            ]
            
            if not active_instances:
                return None
            
            # Round robin selection
            if self.load_balancer['strategy'] == 'round_robin':
                current = self.load_balancer['current_instance']
                selected = active_instances[current % len(active_instances)]
                self.load_balancer['current_instance'] = (current + 1) % len(active_instances)
                return selected
            
            # Least loaded selection
            elif self.load_balancer['strategy'] == 'least_loaded':
                return min(active_instances, key=lambda x: x['load'])
            
            return active_instances[0]  # Fallback
        
        def _check_circuit_breaker(self, circuit_key: str) -> bool:
            """Check circuit breaker status"""
            
            if circuit_key not in self.circuit_breakers:
                self.circuit_breakers[circuit_key] = {
                    'state': 'closed',
                    'failure_count': 0,
                    'last_failure': None,
                    'success_count': 0
                }
            
            cb = self.circuit_breakers[circuit_key]
            
            if cb['state'] == 'open':
                # Check if recovery time has passed
                if cb['last_failure'] and (time.time() - cb['last_failure']) > 30:
                    cb['state'] = 'half_open'
                    return True
                return False
            
            return True
        
        def _record_circuit_breaker_success(self, circuit_key: str):
            """Record circuit breaker success"""
            
            if circuit_key in self.circuit_breakers:
                cb = self.circuit_breakers[circuit_key]
                cb['success_count'] += 1
                
                if cb['state'] == 'half_open' and cb['success_count'] >= 3:
                    cb['state'] = 'closed'
                    cb['failure_count'] = 0
        
        def _record_circuit_breaker_failure(self, circuit_key: str):
            """Record circuit breaker failure"""
            
            if circuit_key in self.circuit_breakers:
                cb = self.circuit_breakers[circuit_key]
                cb['failure_count'] += 1
                cb['last_failure'] = time.time()
                
                if cb['failure_count'] >= 5:
                    cb['state'] = 'open'
        
        async def _update_production_metrics(self, result: Dict[str, Any], 
                                           execution_time: float, instance: Dict[str, Any]):
            """Update production metrics"""
            
            # Update instance load
            instance['load'] += 1
            
            # Update system metrics
            perf = self.monitoring['performance_metrics']
            perf['requests_per_second'] += 1  # Simplified calculation
            perf['average_latency'] = (perf['average_latency'] + execution_time) / 2
            
            if result.get('success', False):
                perf['throughput'] += 1
            else:
                perf['error_rate'] += 0.01  # Simplified calculation
        
        async def _update_error_metrics(self, error: Exception, execution_time: float, 
                                      instance: Dict[str, Any]):
            """Update error metrics"""
            
            # Update instance load (even for errors)
            instance['load'] += 1
            
            # Update error rate
            perf = self.monitoring['performance_metrics']
            perf['error_rate'] += 0.05  # Simplified calculation
            
            # Create alert if error rate is high
            if perf['error_rate'] > 0.1:  # 10% error rate threshold
                alert = {
                    'type': 'high_error_rate',
                    'message': f'Error rate exceeded threshold: {perf["error_rate"]:.2%}',
                    'timestamp': datetime.utcnow(),
                    'severity': 'warning'
                }
                self.metrics_aggregator['alerts'].append(alert)
        
        def get_production_status(self) -> Dict[str, Any]:
            """Get comprehensive production status"""
            
            return {
                'system_info': {
                    'version': self.monitoring['system_health']['version'],
                    'status': self.monitoring['system_health']['status'],
                    'uptime': time.time() - self.monitoring['system_health'].get('start_time', time.time())
                },
                'health_status': self.health_checker,
                'performance_metrics': self.monitoring['performance_metrics'],
                'load_balancer_status': self.load_balancer,
                'circuit_breaker_status': {
                    circuit_key: {
                        'state': cb['state'],
                        'failure_count': cb['failure_count']
                    }
                    for circuit_key, cb in self.circuit_breakers.items()
                },
                'active_alerts': self.metrics_aggregator['alerts'][-5:]  # Last 5 alerts
            }
    
    # Test production system
    production_config = {
        'max_attempts': 3,
        'timeout': 60,
        'caching': True,
        'monitoring': True,
        'load_balancing': True
    }
    
    production_system = ProductionCaptchaSystem(production_config)
    
    # Simulate production requests
    print("\n🧪 Testing production system")
    
    production_requests = [
        {
            'request_id': f'prod_req_{i}',
            'page_url': f'https://production.example.com/page{i}',
            'task_description': f'Production task {i}'
        }
        for i in range(1, 6)
    ]
    
    # Process requests
    for request in production_requests:
        print(f"\n📋 Processing production request: {request['request_id']}")
        
        result = await production_system.handle_production_request(request)
        
        if result.get('success', False):
            print(f"✅ Production request succeeded")
            print(f"   Instance: {result.get('instance_id', 'N/A')}")
            print(f"   Execution time: {result.get('execution_time', 0):.3f}s")
        else:
            print(f"❌ Production request failed: {result.get('error', 'Unknown')}")
            print(f"   Instance: {result.get('instance_id', 'N/A')}")
    
    # Display production status
    status = production_system.get_production_status()
    
    print(f"\n📊 Production System Status:")
    print(f"System Info:")
    sys_info = status['system_info']
    print(f"   Version: {sys_info['version']}")
    print(f"   Status: {sys_info['status']}")
    print(f"   Uptime: {sys_info['uptime']:.1f}s")
    
    print(f"Performance:")
    perf = status['performance_metrics']
    print(f"   Requests/sec: {perf['requests_per_second']}")
    print(f"   Average latency: {perf['average_latency']:.3f}s")
    print(f"   Error rate: {perf['error_rate']:.2%}")
    print(f"   Throughput: {perf['throughput']}")
    
    print(f"Load Balancer:")
    lb = status['load_balancer_status']
    for instance in lb['instances']:
        print(f"   {instance['id']}: {instance['status']} (load: {instance['load']})")
    
    if status['active_alerts']:
        print(f"Active Alerts:")
        for alert in status['active_alerts']:
            print(f"   {alert['type']}: {alert['message']}")
    
    print("\n" + "=" * 50)

# =============================================================================
# MAIN EXAMPLE RUNNER
# =============================================================================

async def run_all_examples():
    """Run all examples in sequence"""
    
    print("🚀 Strands CAPTCHA Framework Examples")
    print("=" * 60)
    print("Running comprehensive examples from basic to advanced...")
    print("=" * 60)
    
    examples = [
        ("Basic CAPTCHA Detection", basic_captcha_detection_example),
        ("Basic CAPTCHA Solving", basic_captcha_solving_example),
        ("Basic Agent Workflow", basic_agent_workflow_example),
        ("Error Handling Patterns", error_handling_example),
        ("Performance Optimization", performance_optimization_example),
        ("Retry and Recovery Patterns", retry_recovery_patterns_example),
        ("Enterprise Deployment", enterprise_deployment_example),
        ("Production Implementation", production_example)
    ]
    
    for name, example_func in examples:
        try:
            print(f"\n🎯 Running: {name}")
            await example_func()
            print(f"✅ Completed: {name}")
        except Exception as e:
            print(f"❌ Failed: {name} - {e}")
        
        # Small delay between examples
        await asyncio.sleep(0.5)
    
    print("\n" + "=" * 60)
    print("🎉 All examples completed!")
    print("=" * 60)

if __name__ == "__main__":
    # Run all examples
    asyncio.run(run_all_examples())