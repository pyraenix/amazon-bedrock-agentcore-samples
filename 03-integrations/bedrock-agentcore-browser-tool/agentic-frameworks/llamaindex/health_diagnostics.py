"""
Health checks and diagnostics for AgentCore browser tool integration.

This module provides health check endpoints, diagnostic tools, system status
reporting, and automated recovery mechanisms for the integration.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List, Callable, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import defaultdict

from interfaces import IBrowserClient, BrowserResponse
from exceptions import AgentCoreBrowserError, BrowserErrorType
from monitoring import MetricsCollector, ErrorTracker, BrowserOperationLogger


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of system components."""
    BROWSER_CLIENT = "browser_client"
    AGENTCORE_API = "agentcore_api"
    SESSION_MANAGER = "session_manager"
    AUTHENTICATION = "authentication"
    NETWORK = "network"
    MEMORY = "memory"
    PERFORMANCE = "performance"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: ComponentType
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    duration_ms: float
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'component': self.component.value,
            'status': self.status.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'duration_ms': self.duration_ms,
            'error': self.error
        }


@dataclass
class SystemStatus:
    """Overall system status."""
    overall_status: HealthStatus
    components: List[HealthCheckResult]
    timestamp: datetime
    uptime_seconds: float
    total_operations: int
    error_rate: float
    performance_metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'overall_status': self.overall_status.value,
            'components': [comp.to_dict() for comp in self.components],
            'timestamp': self.timestamp.isoformat(),
            'uptime_seconds': self.uptime_seconds,
            'total_operations': self.total_operations,
            'error_rate': self.error_rate,
            'performance_metrics': self.performance_metrics
        }


class HealthChecker:
    """Performs health checks on system components."""
    
    def __init__(self, 
                 browser_client: IBrowserClient,
                 timeout_seconds: int = 30):
        """
        Initialize health checker.
        
        Args:
            browser_client: Browser client to check
            timeout_seconds: Timeout for health checks
        """
        self.browser_client = browser_client
        self.timeout_seconds = timeout_seconds
        self.logger = BrowserOperationLogger("health_checker")
        self._start_time = datetime.now(timezone.utc)
    
    async def check_browser_client(self) -> HealthCheckResult:
        """Check browser client health."""
        start_time = time.time()
        
        try:
            # Test basic client functionality
            if not hasattr(self.browser_client, 'create_session'):
                raise AttributeError("Browser client missing required methods")
            
            # Check if client is properly configured
            config_status = "configured" if hasattr(self.browser_client, '_endpoints') else "not_configured"
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component=ComponentType.BROWSER_CLIENT,
                status=HealthStatus.HEALTHY,
                message="Browser client is functional",
                details={
                    'configuration_status': config_status,
                    'client_type': type(self.browser_client).__name__
                },
                timestamp=datetime.now(timezone.utc),
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component=ComponentType.BROWSER_CLIENT,
                status=HealthStatus.CRITICAL,
                message="Browser client check failed",
                details={},
                timestamp=datetime.now(timezone.utc),
                duration_ms=duration_ms,
                error=str(e)
            )
    
    async def check_agentcore_api(self) -> HealthCheckResult:
        """Check AgentCore API connectivity."""
        start_time = time.time()
        
        try:
            # Try to create a test session
            session_id = await asyncio.wait_for(
                self.browser_client.create_session(),
                timeout=self.timeout_seconds
            )
            
            # Clean up test session
            if session_id:
                try:
                    await self.browser_client.close_session(session_id)
                except Exception:
                    pass  # Ignore cleanup errors
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component=ComponentType.AGENTCORE_API,
                status=HealthStatus.HEALTHY,
                message="AgentCore API is accessible",
                details={
                    'test_session_created': bool(session_id),
                    'response_time_ms': duration_ms
                },
                timestamp=datetime.now(timezone.utc),
                duration_ms=duration_ms
            )
            
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component=ComponentType.AGENTCORE_API,
                status=HealthStatus.CRITICAL,
                message="AgentCore API timeout",
                details={'timeout_seconds': self.timeout_seconds},
                timestamp=datetime.now(timezone.utc),
                duration_ms=duration_ms,
                error="API request timed out"
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            status = HealthStatus.WARNING if "test_mode" in str(e).lower() else HealthStatus.CRITICAL
            
            return HealthCheckResult(
                component=ComponentType.AGENTCORE_API,
                status=status,
                message="AgentCore API check failed",
                details={
                    'error_type': type(e).__name__
                },
                timestamp=datetime.now(timezone.utc),
                duration_ms=duration_ms,
                error=str(e)
            )
    
    async def check_authentication(self) -> HealthCheckResult:
        """Check authentication status."""
        start_time = time.time()
        
        try:
            # Check if credentials are configured
            has_credentials = hasattr(self.browser_client, '_aws_credentials') and \
                            self.browser_client._aws_credentials is not None
            
            # Check if endpoints are configured
            has_endpoints = hasattr(self.browser_client, '_endpoints') and \
                          self.browser_client._endpoints is not None
            
            duration_ms = (time.time() - start_time) * 1000
            
            if has_credentials and has_endpoints:
                status = HealthStatus.HEALTHY
                message = "Authentication is configured"
            elif has_endpoints:
                status = HealthStatus.WARNING
                message = "Endpoints configured but credentials missing"
            else:
                status = HealthStatus.CRITICAL
                message = "Authentication not configured"
            
            return HealthCheckResult(
                component=ComponentType.AUTHENTICATION,
                status=status,
                message=message,
                details={
                    'has_credentials': has_credentials,
                    'has_endpoints': has_endpoints
                },
                timestamp=datetime.now(timezone.utc),
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component=ComponentType.AUTHENTICATION,
                status=HealthStatus.CRITICAL,
                message="Authentication check failed",
                details={},
                timestamp=datetime.now(timezone.utc),
                duration_ms=duration_ms,
                error=str(e)
            )
    
    async def check_performance(self, 
                              metrics_collector: Optional[MetricsCollector] = None) -> HealthCheckResult:
        """Check system performance metrics."""
        start_time = time.time()
        
        try:
            details = {}
            status = HealthStatus.HEALTHY
            message = "Performance metrics normal"
            
            if metrics_collector:
                overall_metrics = metrics_collector.get_overall_metrics()
                error_summary = metrics_collector.get_error_summary()
                
                details.update({
                    'total_operations': overall_metrics.total_operations,
                    'success_rate': (overall_metrics.successful_operations / 
                                   max(overall_metrics.total_operations, 1)) * 100,
                    'average_duration_ms': overall_metrics.average_duration_ms,
                    'error_rate': overall_metrics.error_rate * 100,
                    'top_errors': dict(list(error_summary.items())[:5])
                })
                
                # Determine status based on metrics
                if overall_metrics.error_rate > 0.5:  # 50% error rate
                    status = HealthStatus.CRITICAL
                    message = "High error rate detected"
                elif overall_metrics.error_rate > 0.2:  # 20% error rate
                    status = HealthStatus.WARNING
                    message = "Elevated error rate"
                elif overall_metrics.average_duration_ms > 10000:  # 10 seconds
                    status = HealthStatus.WARNING
                    message = "Slow response times"
            else:
                details['metrics_collector'] = 'not_available'
                status = HealthStatus.WARNING
                message = "Performance metrics not available"
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component=ComponentType.PERFORMANCE,
                status=status,
                message=message,
                details=details,
                timestamp=datetime.now(timezone.utc),
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component=ComponentType.PERFORMANCE,
                status=HealthStatus.CRITICAL,
                message="Performance check failed",
                details={},
                timestamp=datetime.now(timezone.utc),
                duration_ms=duration_ms,
                error=str(e)
            )
    
    async def run_all_checks(self, 
                           metrics_collector: Optional[MetricsCollector] = None) -> List[HealthCheckResult]:
        """Run all health checks."""
        checks = [
            self.check_browser_client(),
            self.check_authentication(),
            self.check_agentcore_api(),
            self.check_performance(metrics_collector)
        ]
        
        results = await asyncio.gather(*checks, return_exceptions=True)
        
        # Convert exceptions to error results
        health_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                health_results.append(HealthCheckResult(
                    component=list(ComponentType)[i],
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(result)}",
                    details={},
                    timestamp=datetime.now(timezone.utc),
                    duration_ms=0.0,
                    error=str(result)
                ))
            else:
                health_results.append(result)
        
        return health_results


class DiagnosticTools:
    """Tools for diagnosing integration issues."""
    
    def __init__(self, 
                 browser_client: IBrowserClient,
                 metrics_collector: Optional[MetricsCollector] = None,
                 error_tracker: Optional[ErrorTracker] = None):
        """
        Initialize diagnostic tools.
        
        Args:
            browser_client: Browser client to diagnose
            metrics_collector: Metrics collector for analysis
            error_tracker: Error tracker for analysis
        """
        self.browser_client = browser_client
        self.metrics_collector = metrics_collector
        self.error_tracker = error_tracker
        self.logger = BrowserOperationLogger("diagnostic_tools")
    
    async def diagnose_connection_issues(self) -> Dict[str, Any]:
        """Diagnose connection and network issues."""
        diagnosis = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'connection_tests': {},
            'recommendations': []
        }
        
        try:
            # Test basic connectivity
            start_time = time.time()
            try:
                session_id = await asyncio.wait_for(
                    self.browser_client.create_session(),
                    timeout=10
                )
                connection_time = (time.time() - start_time) * 1000
                
                diagnosis['connection_tests']['session_creation'] = {
                    'success': True,
                    'duration_ms': connection_time,
                    'session_id': session_id
                }
                
                # Clean up
                if session_id:
                    await self.browser_client.close_session(session_id)
                
                if connection_time > 5000:  # 5 seconds
                    diagnosis['recommendations'].append(
                        "Slow connection detected - check network latency"
                    )
                
            except asyncio.TimeoutError:
                diagnosis['connection_tests']['session_creation'] = {
                    'success': False,
                    'error': 'timeout',
                    'duration_ms': 10000
                }
                diagnosis['recommendations'].extend([
                    "Connection timeout - check network connectivity",
                    "Verify AgentCore service availability",
                    "Consider increasing timeout values"
                ])
                
            except Exception as e:
                diagnosis['connection_tests']['session_creation'] = {
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__
                }
                
                if "authentication" in str(e).lower():
                    diagnosis['recommendations'].append(
                        "Authentication issue - verify AWS credentials"
                    )
                elif "endpoint" in str(e).lower():
                    diagnosis['recommendations'].append(
                        "Endpoint configuration issue - verify AgentCore endpoints"
                    )
                else:
                    diagnosis['recommendations'].append(
                        f"Connection error: {str(e)}"
                    )
            
            # Check configuration
            config_issues = []
            if not hasattr(self.browser_client, '_endpoints') or not self.browser_client._endpoints:
                config_issues.append("AgentCore endpoints not configured")
            
            if not hasattr(self.browser_client, '_aws_credentials') or not self.browser_client._aws_credentials:
                config_issues.append("AWS credentials not configured")
            
            diagnosis['connection_tests']['configuration'] = {
                'issues': config_issues,
                'valid': len(config_issues) == 0
            }
            
            if config_issues:
                diagnosis['recommendations'].extend([
                    f"Configuration issue: {issue}" for issue in config_issues
                ])
            
        except Exception as e:
            diagnosis['connection_tests']['error'] = str(e)
            diagnosis['recommendations'].append(f"Diagnostic error: {str(e)}")
        
        return diagnosis
    
    async def diagnose_performance_issues(self) -> Dict[str, Any]:
        """Diagnose performance-related issues."""
        diagnosis = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'performance_analysis': {},
            'recommendations': []
        }
        
        try:
            if self.metrics_collector:
                overall_metrics = self.metrics_collector.get_overall_metrics()
                
                diagnosis['performance_analysis'] = {
                    'total_operations': overall_metrics.total_operations,
                    'success_rate': overall_metrics.successful_operations / max(overall_metrics.total_operations, 1),
                    'error_rate': overall_metrics.error_rate,
                    'average_duration_ms': overall_metrics.average_duration_ms,
                    'min_duration_ms': overall_metrics.min_duration_ms,
                    'max_duration_ms': overall_metrics.max_duration_ms
                }
                
                # Analyze performance issues
                if overall_metrics.average_duration_ms > 10000:  # 10 seconds
                    diagnosis['recommendations'].append(
                        "High average response time - consider optimizing operations"
                    )
                
                if overall_metrics.error_rate > 0.2:  # 20%
                    diagnosis['recommendations'].append(
                        "High error rate - investigate common failure patterns"
                    )
                
                if overall_metrics.max_duration_ms > 60000:  # 1 minute
                    diagnosis['recommendations'].append(
                        "Some operations taking very long - check for timeouts"
                    )
                
                # Get recent operations for analysis
                recent_ops = self.metrics_collector.get_recent_operations(50)
                if recent_ops:
                    slow_ops = [op for op in recent_ops if op.duration_ms and op.duration_ms > 10000]
                    failed_ops = [op for op in recent_ops if not op.success]
                    
                    diagnosis['performance_analysis']['recent_analysis'] = {
                        'total_recent': len(recent_ops),
                        'slow_operations': len(slow_ops),
                        'failed_operations': len(failed_ops)
                    }
                    
                    if slow_ops:
                        slow_types = defaultdict(int)
                        for op in slow_ops:
                            slow_types[op.operation_type] += 1
                        diagnosis['performance_analysis']['slow_operation_types'] = dict(slow_types)
                    
                    if failed_ops:
                        error_types = defaultdict(int)
                        for op in failed_ops:
                            if op.error_type:
                                error_types[op.error_type] += 1
                        diagnosis['performance_analysis']['common_errors'] = dict(error_types)
            
            else:
                diagnosis['performance_analysis']['error'] = 'Metrics collector not available'
                diagnosis['recommendations'].append(
                    'Enable metrics collection for performance analysis'
                )
            
        except Exception as e:
            diagnosis['performance_analysis']['error'] = str(e)
            diagnosis['recommendations'].append(f"Performance analysis error: {str(e)}")
        
        return diagnosis
    
    async def diagnose_error_patterns(self) -> Dict[str, Any]:
        """Diagnose error patterns and trends."""
        diagnosis = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'error_analysis': {},
            'recommendations': []
        }
        
        try:
            if self.error_tracker:
                error_summary = self.error_tracker.get_error_summary()
                
                diagnosis['error_analysis'] = {
                    'total_errors': error_summary['total_errors'],
                    'error_rate_per_minute': error_summary['error_rate_per_minute'],
                    'error_types': error_summary['error_types'],
                    'operation_types': error_summary['operation_types'],
                    'time_window_minutes': error_summary['time_window_minutes']
                }
                
                # Analyze error patterns
                if error_summary['error_rate_per_minute'] > 5:
                    diagnosis['recommendations'].append(
                        "High error rate detected - investigate root cause"
                    )
                
                # Analyze most common errors
                if error_summary['error_types']:
                    most_common_error = max(error_summary['error_types'].items(), key=lambda x: x[1])
                    diagnosis['error_analysis']['most_common_error'] = {
                        'type': most_common_error[0],
                        'count': most_common_error[1]
                    }
                    
                    # Provide specific recommendations based on error type
                    error_type = most_common_error[0].lower()
                    if 'timeout' in error_type:
                        diagnosis['recommendations'].append(
                            "Frequent timeouts - increase timeout values or check performance"
                        )
                    elif 'session' in error_type:
                        diagnosis['recommendations'].append(
                            "Session issues - check session management and lifecycle"
                        )
                    elif 'element' in error_type:
                        diagnosis['recommendations'].append(
                            "Element selection issues - verify selectors and page loading"
                        )
                    elif 'network' in error_type or 'service' in error_type:
                        diagnosis['recommendations'].append(
                            "Network/service issues - check connectivity and service status"
                        )
            
            else:
                diagnosis['error_analysis']['error'] = 'Error tracker not available'
                diagnosis['recommendations'].append(
                    'Enable error tracking for error pattern analysis'
                )
            
        except Exception as e:
            diagnosis['error_analysis']['error'] = str(e)
            diagnosis['recommendations'].append(f"Error analysis error: {str(e)}")
        
        return diagnosis
    
    async def run_comprehensive_diagnosis(self) -> Dict[str, Any]:
        """Run comprehensive system diagnosis."""
        diagnosis = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'system_diagnosis': {}
        }
        
        try:
            # Run all diagnostic tests
            connection_diagnosis = await self.diagnose_connection_issues()
            performance_diagnosis = await self.diagnose_performance_issues()
            error_diagnosis = await self.diagnose_error_patterns()
            
            diagnosis['system_diagnosis'] = {
                'connection': connection_diagnosis,
                'performance': performance_diagnosis,
                'errors': error_diagnosis
            }
            
            # Aggregate recommendations
            all_recommendations = []
            all_recommendations.extend(connection_diagnosis.get('recommendations', []))
            all_recommendations.extend(performance_diagnosis.get('recommendations', []))
            all_recommendations.extend(error_diagnosis.get('recommendations', []))
            
            diagnosis['aggregated_recommendations'] = list(set(all_recommendations))
            
            # Determine overall health
            has_critical_issues = any([
                'timeout' in str(connection_diagnosis).lower(),
                'authentication' in str(connection_diagnosis).lower(),
                error_diagnosis.get('error_analysis', {}).get('error_rate_per_minute', 0) > 10
            ])
            
            diagnosis['overall_health'] = 'critical' if has_critical_issues else 'healthy'
            
        except Exception as e:
            diagnosis['system_diagnosis']['error'] = str(e)
            diagnosis['overall_health'] = 'unknown'
        
        return diagnosis


class SystemStatusReporter:
    """Reports system status and provides dashboard capabilities."""
    
    def __init__(self,
                 health_checker: HealthChecker,
                 metrics_collector: Optional[MetricsCollector] = None,
                 error_tracker: Optional[ErrorTracker] = None):
        """
        Initialize system status reporter.
        
        Args:
            health_checker: Health checker instance
            metrics_collector: Metrics collector instance
            error_tracker: Error tracker instance
        """
        self.health_checker = health_checker
        self.metrics_collector = metrics_collector
        self.error_tracker = error_tracker
        self.logger = BrowserOperationLogger("status_reporter")
        self._start_time = datetime.now(timezone.utc)
    
    async def get_system_status(self) -> SystemStatus:
        """Get comprehensive system status."""
        # Run health checks
        health_results = await self.health_checker.run_all_checks(self.metrics_collector)
        
        # Determine overall status
        overall_status = HealthStatus.HEALTHY
        for result in health_results:
            if result.status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
                break
            elif result.status == HealthStatus.WARNING and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.WARNING
        
        # Get performance metrics
        performance_metrics = {}
        total_operations = 0
        error_rate = 0.0
        
        if self.metrics_collector:
            overall_metrics = self.metrics_collector.get_overall_metrics()
            total_operations = overall_metrics.total_operations
            error_rate = overall_metrics.error_rate
            
            performance_metrics = {
                'total_operations': overall_metrics.total_operations,
                'successful_operations': overall_metrics.successful_operations,
                'failed_operations': overall_metrics.failed_operations,
                'average_duration_ms': overall_metrics.average_duration_ms,
                'min_duration_ms': overall_metrics.min_duration_ms,
                'max_duration_ms': overall_metrics.max_duration_ms,
                'error_rate': overall_metrics.error_rate,
                'operations_per_second': overall_metrics.operations_per_second
            }
        
        # Calculate uptime
        uptime_seconds = (datetime.now(timezone.utc) - self._start_time).total_seconds()
        
        return SystemStatus(
            overall_status=overall_status,
            components=health_results,
            timestamp=datetime.now(timezone.utc),
            uptime_seconds=uptime_seconds,
            total_operations=total_operations,
            error_rate=error_rate,
            performance_metrics=performance_metrics
        )
    
    async def generate_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report."""
        system_status = await self.get_system_status()
        
        report = {
            'report_timestamp': datetime.now(timezone.utc).isoformat(),
            'system_status': system_status.to_dict(),
            'detailed_metrics': {},
            'error_analysis': {},
            'recommendations': []
        }
        
        # Add detailed metrics if available
        if self.metrics_collector:
            report['detailed_metrics'] = {
                'operations_by_type': {},
                'recent_operations': []
            }
            
            # Get metrics by operation type
            for op_type in ['navigate', 'screenshot', 'extract_text', 'click_element']:
                type_metrics = self.metrics_collector.get_metrics_by_type(op_type)
                if type_metrics.total_operations > 0:
                    report['detailed_metrics']['operations_by_type'][op_type] = {
                        'total': type_metrics.total_operations,
                        'success_rate': type_metrics.successful_operations / type_metrics.total_operations,
                        'average_duration_ms': type_metrics.average_duration_ms,
                        'error_rate': type_metrics.error_rate
                    }
            
            # Get recent operations
            recent_ops = self.metrics_collector.get_recent_operations(10)
            report['detailed_metrics']['recent_operations'] = [
                {
                    'operation_id': op.operation_id,
                    'operation_type': op.operation_type,
                    'success': op.success,
                    'duration_ms': op.duration_ms,
                    'timestamp': op.start_time.isoformat(),
                    'error_type': op.error_type
                }
                for op in recent_ops
            ]
        
        # Add error analysis if available
        if self.error_tracker:
            error_summary = self.error_tracker.get_error_summary()
            report['error_analysis'] = error_summary
            
            # Generate recommendations based on errors
            if error_summary['total_errors'] > 0:
                if error_summary['error_rate_per_minute'] > 5:
                    report['recommendations'].append(
                        "High error rate detected - immediate investigation recommended"
                    )
                
                # Analyze error types
                for error_type, count in error_summary['error_types'].items():
                    if count > 3:  # More than 3 occurrences
                        report['recommendations'].append(
                            f"Frequent {error_type} errors - check related functionality"
                        )
        
        # Add general recommendations based on system status
        if system_status.overall_status == HealthStatus.CRITICAL:
            report['recommendations'].append(
                "System in critical state - immediate attention required"
            )
        elif system_status.overall_status == HealthStatus.WARNING:
            report['recommendations'].append(
                "System showing warning signs - monitoring recommended"
            )
        
        return report


class AutoRecoveryManager:
    """Manages automated recovery and self-healing mechanisms."""
    
    def __init__(self,
                 browser_client: IBrowserClient,
                 health_checker: HealthChecker,
                 metrics_collector: Optional[MetricsCollector] = None,
                 error_tracker: Optional[ErrorTracker] = None):
        """
        Initialize auto-recovery manager.
        
        Args:
            browser_client: Browser client to manage
            health_checker: Health checker for monitoring
            metrics_collector: Metrics collector for analysis
            error_tracker: Error tracker for analysis
        """
        self.browser_client = browser_client
        self.health_checker = health_checker
        self.metrics_collector = metrics_collector
        self.error_tracker = error_tracker
        self.logger = BrowserOperationLogger("auto_recovery")
        
        self._recovery_actions: Dict[ComponentType, Callable] = {
            ComponentType.BROWSER_CLIENT: self._recover_browser_client,
            ComponentType.AGENTCORE_API: self._recover_agentcore_api,
            ComponentType.SESSION_MANAGER: self._recover_session_manager,
            ComponentType.AUTHENTICATION: self._recover_authentication
        }
        
        self._recovery_history: List[Dict[str, Any]] = []
        self._recovery_lock = threading.Lock()
    
    async def check_and_recover(self) -> Dict[str, Any]:
        """Check system health and perform recovery if needed."""
        recovery_report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'health_check_results': [],
            'recovery_actions': [],
            'overall_result': 'no_action_needed'
        }
        
        try:
            # Run health checks
            health_results = await self.health_checker.run_all_checks(self.metrics_collector)
            recovery_report['health_check_results'] = [result.to_dict() for result in health_results]
            
            # Identify components needing recovery
            critical_components = [
                result for result in health_results 
                if result.status == HealthStatus.CRITICAL
            ]
            
            if critical_components:
                recovery_report['overall_result'] = 'recovery_attempted'
                
                # Attempt recovery for each critical component
                for component_result in critical_components:
                    recovery_action = await self._attempt_recovery(component_result)
                    recovery_report['recovery_actions'].append(recovery_action)
                
                # Re-run health checks after recovery
                post_recovery_results = await self.health_checker.run_all_checks(self.metrics_collector)
                recovery_report['post_recovery_health'] = [result.to_dict() for result in post_recovery_results]
                
                # Determine if recovery was successful
                still_critical = [
                    result for result in post_recovery_results 
                    if result.status == HealthStatus.CRITICAL
                ]
                
                if len(still_critical) < len(critical_components):
                    recovery_report['overall_result'] = 'partial_recovery'
                if not still_critical:
                    recovery_report['overall_result'] = 'full_recovery'
            
        except Exception as e:
            recovery_report['error'] = str(e)
            recovery_report['overall_result'] = 'recovery_failed'
            self.logger.log_operation_error(
                operation_id="auto_recovery",
                operation_type="system_recovery",
                error=e
            )
        
        # Record recovery attempt
        with self._recovery_lock:
            self._recovery_history.append(recovery_report)
            # Keep only last 100 recovery attempts
            if len(self._recovery_history) > 100:
                self._recovery_history.pop(0)
        
        return recovery_report
    
    async def _attempt_recovery(self, component_result: HealthCheckResult) -> Dict[str, Any]:
        """Attempt recovery for a specific component."""
        recovery_action = {
            'component': component_result.component.value,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action_taken': 'none',
            'success': False,
            'error': None
        }
        
        try:
            recovery_func = self._recovery_actions.get(component_result.component)
            if recovery_func:
                result = await recovery_func(component_result)
                recovery_action.update(result)
            else:
                recovery_action['error'] = f"No recovery action defined for {component_result.component.value}"
            
        except Exception as e:
            recovery_action['error'] = str(e)
            recovery_action['success'] = False
        
        return recovery_action
    
    async def _recover_browser_client(self, component_result: HealthCheckResult) -> Dict[str, Any]:
        """Recover browser client issues."""
        try:
            # Try to reinitialize client configuration
            if hasattr(self.browser_client, '_load_configuration'):
                self.browser_client._load_configuration()
            
            return {
                'action_taken': 'reinitialize_configuration',
                'success': True
            }
            
        except Exception as e:
            return {
                'action_taken': 'reinitialize_configuration',
                'success': False,
                'error': str(e)
            }
    
    async def _recover_agentcore_api(self, component_result: HealthCheckResult) -> Dict[str, Any]:
        """Recover AgentCore API connectivity issues."""
        try:
            # Close any existing sessions and try to create a new one
            if hasattr(self.browser_client, 'session_id') and self.browser_client.session_id:
                await self.browser_client.close_session()
            
            # Wait a bit before retrying
            await asyncio.sleep(2)
            
            # Try to create a new session
            session_id = await self.browser_client.create_session()
            
            return {
                'action_taken': 'reset_session',
                'success': bool(session_id),
                'new_session_id': session_id
            }
            
        except Exception as e:
            return {
                'action_taken': 'reset_session',
                'success': False,
                'error': str(e)
            }
    
    async def _recover_session_manager(self, component_result: HealthCheckResult) -> Dict[str, Any]:
        """Recover session management issues."""
        try:
            # Force close all sessions and reset state
            if hasattr(self.browser_client, 'session_id'):
                self.browser_client.session_id = None
            
            if hasattr(self.browser_client, '_session_status'):
                from interfaces import SessionStatus
                self.browser_client._session_status = SessionStatus.CLOSED
            
            return {
                'action_taken': 'reset_session_state',
                'success': True
            }
            
        except Exception as e:
            return {
                'action_taken': 'reset_session_state',
                'success': False,
                'error': str(e)
            }
    
    async def _recover_authentication(self, component_result: HealthCheckResult) -> Dict[str, Any]:
        """Recover authentication issues."""
        try:
            # Try to reload credentials
            if hasattr(self.browser_client, 'config_manager'):
                aws_creds = self.browser_client.config_manager.get_aws_credentials()
                self.browser_client._aws_credentials = aws_creds if isinstance(aws_creds, dict) else aws_creds.to_dict()
            
            return {
                'action_taken': 'reload_credentials',
                'success': True
            }
            
        except Exception as e:
            return {
                'action_taken': 'reload_credentials',
                'success': False,
                'error': str(e)
            }
    
    def get_recovery_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get history of recovery attempts."""
        with self._recovery_lock:
            return self._recovery_history[-limit:]
    
    def clear_recovery_history(self):
        """Clear recovery history."""
        with self._recovery_lock:
            self._recovery_history.clear()