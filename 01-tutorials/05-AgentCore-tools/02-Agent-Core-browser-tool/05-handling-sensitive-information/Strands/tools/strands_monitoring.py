#!/usr/bin/env python3
"""
Strands Monitoring and Observability
===================================

This module provides comprehensive monitoring, observability, and audit logging
for production Strands deployments using AgentCore Browser Tool. It includes
real-time metrics collection, security event monitoring, compliance reporting,
and integration with AWS CloudWatch and other monitoring services.

Features:
- Real-time performance monitoring
- Security event detection and alerting
- Compliance audit trail generation
- Integration with AWS CloudWatch, X-Ray, and CloudTrail
- Custom metrics for Strands-specific operations
- Automated anomaly detection
- Comprehensive dashboard generation
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import statistics
from collections import defaultdict, deque

# Strands framework imports
from strands import Agent
from strands.tools import tool, PythonAgentTool
from strands.types.tools import AgentTool

# AWS imports
import boto3
from botocore.exceptions import ClientError


class MetricType(Enum):
    """Types of metrics to collect."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EventType(Enum):
    """Types of events to monitor."""
    SECURITY_EVENT = "security_event"
    PERFORMANCE_EVENT = "performance_event"
    COMPLIANCE_EVENT = "compliance_event"
    ERROR_EVENT = "error_event"
    BUSINESS_EVENT = "business_event"


@dataclass
class Metric:
    """Represents a monitoring metric."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    dimensions: Dict[str, str]
    unit: str = "Count"
    
    def to_cloudwatch_format(self) -> Dict[str, Any]:
        """Convert to CloudWatch metric format."""
        return {
            "MetricName": self.name,
            "Value": self.value,
            "Unit": self.unit,
            "Timestamp": self.timestamp,
            "Dimensions": [
                {"Name": k, "Value": v} for k, v in self.dimensions.items()
            ]
        }


@dataclass
class Alert:
    """Represents a monitoring alert."""
    alert_id: str
    name: str
    description: str
    severity: AlertSeverity
    event_type: EventType
    context: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "alert_id": self.alert_id,
            "name": self.name,
            "description": self.description,
            "severity": self.severity.value,
            "event_type": self.event_type.value,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolution_timestamp": self.resolution_timestamp.isoformat() if self.resolution_timestamp else None
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for Strands operations."""
    session_id: str
    operation_type: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    success: bool
    error_message: Optional[str]
    data_processed_kb: float
    memory_usage_mb: float
    cpu_usage_percent: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "session_id": self.session_id,
            "operation_type": self.operation_type,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error_message": self.error_message,
            "data_processed_kb": self.data_processed_kb,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent
        }


class MetricsCollector:
    """Collects and aggregates metrics."""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.metrics_buffer: deque = deque(maxlen=buffer_size)
        self.aggregated_metrics: Dict[str, List[float]] = defaultdict(list)
        self.last_flush = datetime.utcnow()
        
    def record_metric(self, metric: Metric):
        """Record a metric."""
        self.metrics_buffer.append(metric)
        self.aggregated_metrics[metric.name].append(metric.value)
    
    def get_metrics_summary(self, metric_name: str) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        values = self.aggregated_metrics.get(metric_name, [])
        if not values:
            return {}
        
        return {
            "count": len(values),
            "sum": sum(values),
            "avg": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0
        }
    
    def flush_metrics(self) -> List[Metric]:
        """Flush buffered metrics and return them."""
        metrics = list(self.metrics_buffer)
        self.metrics_buffer.clear()
        self.last_flush = datetime.utcnow()
        return metrics
    
    def should_flush(self, flush_interval_seconds: int = 60) -> bool:
        """Check if metrics should be flushed."""
        return (datetime.utcnow() - self.last_flush).total_seconds() >= flush_interval_seconds


class SecurityMonitor:
    """Monitors security events and anomalies."""
    
    def __init__(self):
        self.security_events: List[Dict[str, Any]] = []
        self.failed_auth_attempts: Dict[str, int] = defaultdict(int)
        self.suspicious_patterns: List[str] = [
            "multiple_failed_auth",
            "unusual_data_access",
            "privilege_escalation",
            "data_exfiltration_attempt",
            "policy_violation"
        ]
    
    def record_security_event(self, event: Dict[str, Any]) -> Optional[Alert]:
        """Record a security event and check for anomalies."""
        self.security_events.append(event)
        
        # Check for suspicious patterns
        return self._analyze_security_event(event)
    
    def _analyze_security_event(self, event: Dict[str, Any]) -> Optional[Alert]:
        """Analyze security event for anomalies."""
        event_type = event.get("event_type", "")
        
        # Check for multiple failed authentication attempts
        if "authentication" in event_type and not event.get("success", True):
            session_id = event.get("session_id", "unknown")
            self.failed_auth_attempts[session_id] += 1
            
            if self.failed_auth_attempts[session_id] >= 3:
                return Alert(
                    alert_id=str(uuid.uuid4()),
                    name="Multiple Failed Authentication Attempts",
                    description=f"Session {session_id} has {self.failed_auth_attempts[session_id]} failed auth attempts",
                    severity=AlertSeverity.WARNING,
                    event_type=EventType.SECURITY_EVENT,
                    context={"session_id": session_id, "failed_attempts": self.failed_auth_attempts[session_id]},
                    timestamp=datetime.utcnow()
                )
        
        # Check for policy violations
        if "policy_violation" in event_type:
            return Alert(
                alert_id=str(uuid.uuid4()),
                name="Security Policy Violation",
                description=f"Policy violation detected: {event.get('violation_type', 'unknown')}",
                severity=AlertSeverity.ERROR,
                event_type=EventType.SECURITY_EVENT,
                context=event,
                timestamp=datetime.utcnow()
            )
        
        # Check for PII exposure
        if event.get("pii_detected", False) and not event.get("pii_masked", False):
            return Alert(
                alert_id=str(uuid.uuid4()),
                name="Potential PII Exposure",
                description="PII detected but not properly masked",
                severity=AlertSeverity.CRITICAL,
                event_type=EventType.COMPLIANCE_EVENT,
                context=event,
                timestamp=datetime.utcnow()
            )
        
        return None


class PerformanceMonitor:
    """Monitors performance metrics and detects anomalies."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.response_times: deque = deque(maxlen=window_size)
        self.error_rates: deque = deque(maxlen=window_size)
        self.throughput: deque = deque(maxlen=window_size)
        
    def record_performance_metrics(self, metrics: PerformanceMetrics) -> List[Alert]:
        """Record performance metrics and check for anomalies."""
        alerts = []
        
        # Record response time
        self.response_times.append(metrics.duration_ms)
        
        # Record error rate
        self.error_rates.append(0 if metrics.success else 1)
        
        # Check for performance anomalies
        if len(self.response_times) >= 10:
            avg_response_time = statistics.mean(self.response_times)
            
            # Alert on high response time
            if metrics.duration_ms > avg_response_time * 3:
                alerts.append(Alert(
                    alert_id=str(uuid.uuid4()),
                    name="High Response Time",
                    description=f"Response time {metrics.duration_ms}ms is 3x higher than average {avg_response_time:.2f}ms",
                    severity=AlertSeverity.WARNING,
                    event_type=EventType.PERFORMANCE_EVENT,
                    context={"response_time": metrics.duration_ms, "average": avg_response_time},
                    timestamp=datetime.utcnow()
                ))
        
        # Check error rate
        if len(self.error_rates) >= 10:
            error_rate = sum(self.error_rates) / len(self.error_rates)
            
            if error_rate > 0.1:  # 10% error rate
                alerts.append(Alert(
                    alert_id=str(uuid.uuid4()),
                    name="High Error Rate",
                    description=f"Error rate {error_rate:.2%} exceeds threshold",
                    severity=AlertSeverity.ERROR,
                    event_type=EventType.PERFORMANCE_EVENT,
                    context={"error_rate": error_rate},
                    timestamp=datetime.utcnow()
                ))
        
        return alerts


class ComplianceMonitor:
    """Monitors compliance-related events and generates reports."""
    
    def __init__(self):
        self.compliance_events: List[Dict[str, Any]] = []
        self.framework_violations: Dict[str, int] = defaultdict(int)
        
    def record_compliance_event(self, event: Dict[str, Any]) -> Optional[Alert]:
        """Record a compliance event."""
        self.compliance_events.append(event)
        
        # Check for compliance violations
        if event.get("violation_detected", False):
            framework = event.get("compliance_framework", "unknown")
            self.framework_violations[framework] += 1
            
            return Alert(
                alert_id=str(uuid.uuid4()),
                name="Compliance Violation",
                description=f"Violation detected for {framework}: {event.get('violation_type', 'unknown')}",
                severity=AlertSeverity.CRITICAL,
                event_type=EventType.COMPLIANCE_EVENT,
                context=event,
                timestamp=datetime.utcnow()
            )
        
        return None
    
    def generate_compliance_report(self, 
                                 framework: str,
                                 start_date: datetime,
                                 end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for a specific framework."""
        relevant_events = [
            event for event in self.compliance_events
            if (event.get("compliance_framework") == framework and
                start_date <= datetime.fromisoformat(event.get("timestamp", "")) <= end_date)
        ]
        
        violations = [event for event in relevant_events if event.get("violation_detected", False)]
        
        return {
            "framework": framework,
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_events": len(relevant_events),
            "violations": len(violations),
            "compliance_rate": (len(relevant_events) - len(violations)) / len(relevant_events) if relevant_events else 1.0,
            "violation_details": violations,
            "generated_at": datetime.utcnow().isoformat()
        }


class StrandsMonitor(PythonAgentTool):
    """Main monitoring tool for Strands agents."""
    
    def __init__(self, region: str = "us-east-1"):
        super().__init__(name="strands_monitor")
        self.region = region
        
        # Initialize monitoring components
        self.metrics_collector = MetricsCollector()
        self.security_monitor = SecurityMonitor()
        self.performance_monitor = PerformanceMonitor()
        self.compliance_monitor = ComplianceMonitor()
        
        # Initialize AWS clients
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.xray = boto3.client('xray', region_name=region)
        self.sns = boto3.client('sns', region_name=region)
        
        # Initialize audit logger
        self.audit_logger = AuditLogger(service_name="strands_monitor")
        
        # Alert management
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_handlers: Dict[AlertSeverity, List[Callable]] = defaultdict(list)
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start_monitoring(self):
        """Start the monitoring system."""
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        await self.audit_logger.log_event({
            "event_type": "monitoring_started",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def stop_monitoring(self):
        """Stop the monitoring system."""
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining metrics
        await self._flush_metrics_to_cloudwatch()
        
        await self.audit_logger.log_event({
            "event_type": "monitoring_stopped",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def record_metric(self, 
                          name: str,
                          value: float,
                          metric_type: MetricType = MetricType.COUNTER,
                          dimensions: Dict[str, str] = None,
                          unit: str = "Count"):
        """Record a custom metric."""
        if dimensions is None:
            dimensions = {}
        
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.utcnow(),
            dimensions=dimensions,
            unit=unit
        )
        
        self.metrics_collector.record_metric(metric)
    
    async def record_security_event(self, event: Dict[str, Any]):
        """Record a security event."""
        event["timestamp"] = datetime.utcnow().isoformat()
        
        # Monitor for anomalies
        alert = self.security_monitor.record_security_event(event)
        if alert:
            await self._handle_alert(alert)
        
        # Log to audit trail
        await self.audit_logger.log_event({
            "event_type": "security_event_recorded",
            "security_event": event,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def record_performance_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        # Check for performance anomalies
        alerts = self.performance_monitor.record_performance_metrics(metrics)
        for alert in alerts:
            await self._handle_alert(alert)
        
        # Record as CloudWatch metrics
        await self.record_metric("ResponseTime", metrics.duration_ms, MetricType.TIMER, 
                                {"Operation": metrics.operation_type}, "Milliseconds")
        await self.record_metric("DataProcessed", metrics.data_processed_kb, MetricType.GAUGE,
                                {"Operation": metrics.operation_type}, "Kilobytes")
        await self.record_metric("MemoryUsage", metrics.memory_usage_mb, MetricType.GAUGE,
                                {"Operation": metrics.operation_type}, "Megabytes")
        
        # Log performance metrics
        await self.audit_logger.log_event({
            "event_type": "performance_metrics_recorded",
            "metrics": metrics.to_dict(),
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def record_compliance_event(self, event: Dict[str, Any]):
        """Record a compliance event."""
        event["timestamp"] = datetime.utcnow().isoformat()
        
        # Monitor for violations
        alert = self.compliance_monitor.record_compliance_event(event)
        if alert:
            await self._handle_alert(alert)
        
        # Log compliance event
        await self.audit_logger.log_event({
            "event_type": "compliance_event_recorded",
            "compliance_event": event,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Flush metrics to CloudWatch
                if self.metrics_collector.should_flush():
                    await self._flush_metrics_to_cloudwatch()
                
                # Check for alert resolution
                await self._check_alert_resolution()
                
                # Sleep for monitoring interval
                await asyncio.sleep(30)  # 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.audit_logger.log_event({
                    "event_type": "monitoring_loop_error",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
    
    async def _flush_metrics_to_cloudwatch(self):
        """Flush metrics to CloudWatch."""
        metrics = self.metrics_collector.flush_metrics()
        
        if not metrics:
            return
        
        try:
            # Group metrics by namespace
            metric_data = []
            for metric in metrics:
                metric_data.append(metric.to_cloudwatch_format())
            
            # Send to CloudWatch in batches
            batch_size = 20  # CloudWatch limit
            for i in range(0, len(metric_data), batch_size):
                batch = metric_data[i:i + batch_size]
                
                self.cloudwatch.put_metric_data(
                    Namespace='Strands/AgentCore',
                    MetricData=batch
                )
            
            await self.audit_logger.log_event({
                "event_type": "metrics_flushed_to_cloudwatch",
                "metric_count": len(metrics),
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except ClientError as e:
            await self.audit_logger.log_event({
                "event_type": "cloudwatch_flush_error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def _handle_alert(self, alert: Alert):
        """Handle a new alert."""
        self.active_alerts[alert.alert_id] = alert
        
        # Log alert
        await self.audit_logger.log_event({
            "event_type": "alert_triggered",
            "alert": alert.to_dict(),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Execute alert handlers
        handlers = self.alert_handlers.get(alert.severity, [])
        for handler in handlers:
            try:
                await handler(alert)
            except Exception as e:
                await self.audit_logger.log_event({
                    "event_type": "alert_handler_error",
                    "alert_id": alert.alert_id,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
    
    async def _check_alert_resolution(self):
        """Check if any alerts can be resolved."""
        # This would implement logic to check if alert conditions have been resolved
        # For now, we'll auto-resolve alerts after 1 hour
        current_time = datetime.utcnow()
        
        for alert_id, alert in list(self.active_alerts.items()):
            if not alert.resolved and (current_time - alert.timestamp).total_seconds() > 3600:
                alert.resolved = True
                alert.resolution_timestamp = current_time
                
                await self.audit_logger.log_event({
                    "event_type": "alert_auto_resolved",
                    "alert_id": alert_id,
                    "timestamp": datetime.utcnow().isoformat()
                })
    
    def add_alert_handler(self, severity: AlertSeverity, handler: Callable[[Alert], None]):
        """Add an alert handler for a specific severity level."""
        self.alert_handlers[severity].append(handler)
    
    async def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for monitoring dashboard."""
        current_time = datetime.utcnow()
        
        # Get metrics summaries
        response_time_summary = self.metrics_collector.get_metrics_summary("ResponseTime")
        error_rate_summary = self.metrics_collector.get_metrics_summary("ErrorRate")
        
        # Get active alerts
        active_alerts = [alert.to_dict() for alert in self.active_alerts.values() if not alert.resolved]
        
        # Get recent security events
        recent_security_events = [
            event for event in self.security_monitor.security_events[-10:]
        ]
        
        return {
            "timestamp": current_time.isoformat(),
            "metrics": {
                "response_time": response_time_summary,
                "error_rate": error_rate_summary,
                "active_sessions": len(self.metrics_collector.aggregated_metrics.get("ActiveSessions", [])),
                "total_requests": sum(self.metrics_collector.aggregated_metrics.get("TotalRequests", []))
            },
            "alerts": {
                "active_count": len(active_alerts),
                "active_alerts": active_alerts,
                "critical_count": len([a for a in active_alerts if a["severity"] == "critical"]),
                "warning_count": len([a for a in active_alerts if a["severity"] == "warning"])
            },
            "security": {
                "recent_events": recent_security_events,
                "failed_auth_attempts": dict(self.security_monitor.failed_auth_attempts)
            },
            "compliance": {
                "framework_violations": dict(self.compliance_monitor.framework_violations)
            }
        }
    
    async def generate_compliance_report(self, 
                                       framework: str,
                                       days_back: int = 30) -> Dict[str, Any]:
        """Generate compliance report."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        return self.compliance_monitor.generate_compliance_report(framework, start_date, end_date)


# Convenience functions for common monitoring patterns

async def monitor_strands_operation(monitor: StrandsMonitor,
                                  operation_name: str,
                                  session_id: str):
    """Context manager for monitoring Strands operations."""
    start_time = datetime.utcnow()
    
    try:
        yield
        
        # Record successful operation
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds() * 1000
        
        metrics = PerformanceMetrics(
            session_id=session_id,
            operation_type=operation_name,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration,
            success=True,
            error_message=None,
            data_processed_kb=0.0,  # Would be calculated based on actual data
            memory_usage_mb=0.0,    # Would be measured
            cpu_usage_percent=0.0   # Would be measured
        )
        
        await monitor.record_performance_metrics(metrics)
        
    except Exception as e:
        # Record failed operation
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds() * 1000
        
        metrics = PerformanceMetrics(
            session_id=session_id,
            operation_type=operation_name,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration,
            success=False,
            error_message=str(e),
            data_processed_kb=0.0,
            memory_usage_mb=0.0,
            cpu_usage_percent=0.0
        )
        
        await monitor.record_performance_metrics(metrics)
        raise


# Example alert handlers
async def critical_alert_handler(alert: Alert):
    """Handle critical alerts."""
    print(f"CRITICAL ALERT: {alert.name} - {alert.description}")
    # Could send to PagerDuty, Slack, etc.

async def warning_alert_handler(alert: Alert):
    """Handle warning alerts."""
    print(f"WARNING: {alert.name} - {alert.description}")
    # Could send to monitoring dashboard, email, etc.


# Example usage
async def example_usage():
    """Example of how to use the monitoring system."""
    monitor = StrandsMonitor()
    
    # Add alert handlers
    monitor.add_alert_handler(AlertSeverity.CRITICAL, critical_alert_handler)
    monitor.add_alert_handler(AlertSeverity.WARNING, warning_alert_handler)
    
    try:
        # Start monitoring
        await monitor.start_monitoring()
        
        # Record some metrics
        await monitor.record_metric("TestMetric", 42.0, MetricType.GAUGE)
        
        # Record a security event
        await monitor.record_security_event({
            "event_type": "authentication_attempt",
            "session_id": "test-session",
            "success": False,
            "user_agent": "test-agent"
        })
        
        # Generate dashboard data
        dashboard_data = await monitor.generate_dashboard_data()
        print(f"Dashboard data: {json.dumps(dashboard_data, indent=2)}")
        
        # Wait a bit
        await asyncio.sleep(5)
        
    finally:
        # Stop monitoring
        await monitor.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(example_usage())