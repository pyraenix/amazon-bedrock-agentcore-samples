#!/usr/bin/env python3
"""
LlamaIndex Monitoring and Observability Integration

This module provides comprehensive monitoring and observability capabilities for LlamaIndex
applications integrated with AgentCore Browser Tool. Includes performance monitoring,
security event tracking, compliance reporting, and operational insights.

Key Features:
- Performance metrics collection and analysis
- Security event monitoring and alerting
- Compliance tracking and reporting
- Operational health monitoring
- Custom metrics and dashboards
- Integration with AWS CloudWatch and other monitoring systems
- Real-time alerting and notifications

Requirements: 1.2, 2.5, 4.2, 5.4
"""

import os
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from collections import defaultdict, deque
from contextlib import contextmanager
import threading
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics to collect"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EventType(Enum):
    """Types of events to monitor"""
    PERFORMANCE = "performance"
    SECURITY = "security"
    ERROR = "error"
    COMPLIANCE = "compliance"
    OPERATIONAL = "operational"


@dataclass
class Metric:
    """Represents a monitoring metric"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = "count"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary"""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "unit": self.unit
        }


@dataclass
class Alert:
    """Represents a monitoring alert"""
    alert_id: str
    name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            "alert_id": self.alert_id,
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "resolved": self.resolved,
            "resolution_timestamp": self.resolution_timestamp.isoformat() if self.resolution_timestamp else None
        }


@dataclass
class MonitoringEvent:
    """Represents a monitoring event"""
    event_id: str
    event_type: EventType
    source: str
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    severity: AlertSeverity = AlertSeverity.INFO
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source": self.source,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "severity": self.severity.value
        }


class MetricsCollector:
    """Collects and aggregates metrics"""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.metrics_buffer: deque = deque(maxlen=buffer_size)
        self.aggregated_metrics: Dict[str, List[Metric]] = defaultdict(list)
        self._lock = threading.Lock()
        
        logger.info("Metrics collector initialized")
    
    def record_metric(self, name: str, value: Union[int, float], metric_type: MetricType = MetricType.COUNTER,
                     tags: Optional[Dict[str, str]] = None, unit: str = "count"):
        """Record a metric"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.utcnow(),
            tags=tags or {},
            unit=unit
        )
        
        with self._lock:
            self.metrics_buffer.append(metric)
            self.aggregated_metrics[name].append(metric)
            
            # Keep only recent metrics for aggregation
            if len(self.aggregated_metrics[name]) > 100:
                self.aggregated_metrics[name] = self.aggregated_metrics[name][-100:]
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        self.record_metric(name, value, MetricType.COUNTER, tags)
    
    def set_gauge(self, name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None, unit: str = "count"):
        """Set a gauge metric"""
        self.record_metric(name, value, MetricType.GAUGE, tags, unit)
    
    def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timer metric"""
        self.record_metric(name, duration, MetricType.TIMER, tags, "seconds")
    
    def get_metrics(self, name: Optional[str] = None, since: Optional[datetime] = None) -> List[Metric]:
        """Get metrics by name and time range"""
        with self._lock:
            if name:
                metrics = self.aggregated_metrics.get(name, [])
            else:
                metrics = list(self.metrics_buffer)
            
            if since:
                metrics = [m for m in metrics if m.timestamp >= since]
            
            return metrics
    
    def get_metric_summary(self, name: str, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        metrics = self.get_metrics(name, since)
        
        if not metrics:
            return {"name": name, "count": 0}
        
        values = [m.value for m in metrics]
        
        return {
            "name": name,
            "count": len(values),
            "sum": sum(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "latest": values[-1] if values else None,
            "timestamp_range": {
                "start": min(m.timestamp for m in metrics).isoformat(),
                "end": max(m.timestamp for m in metrics).isoformat()
            }
        }


class PerformanceMonitor:
    """Monitors performance metrics for LlamaIndex operations"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_timers: Dict[str, float] = {}
        
        logger.info("Performance monitor initialized")
    
    @contextmanager
    def timer(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations"""
        start_time = time.time()
        timer_id = f"{operation_name}_{uuid4().hex[:8]}"
        self.active_timers[timer_id] = start_time
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            self.metrics_collector.record_timer(
                f"{operation_name}_duration",
                duration,
                tags
            )
            
            self.active_timers.pop(timer_id, None)
    
    def record_query_performance(self, query_time: float, query_type: str = "standard",
                                success: bool = True, tags: Optional[Dict[str, str]] = None):
        """Record query performance metrics"""
        base_tags = {"query_type": query_type, "success": str(success)}
        if tags:
            base_tags.update(tags)
        
        self.metrics_collector.record_timer("query_duration", query_time, base_tags)
        self.metrics_collector.increment_counter("queries_total", 1, base_tags)
        
        if success:
            self.metrics_collector.increment_counter("queries_successful", 1, base_tags)
        else:
            self.metrics_collector.increment_counter("queries_failed", 1, base_tags)
    
    def record_extraction_performance(self, extraction_time: float, url: str, success: bool = True,
                                    documents_extracted: int = 0, tags: Optional[Dict[str, str]] = None):
        """Record web extraction performance metrics"""
        base_tags = {"success": str(success), "domain": self._extract_domain(url)}
        if tags:
            base_tags.update(tags)
        
        self.metrics_collector.record_timer("extraction_duration", extraction_time, base_tags)
        self.metrics_collector.increment_counter("extractions_total", 1, base_tags)
        self.metrics_collector.set_gauge("documents_extracted", documents_extracted, base_tags)
        
        if success:
            self.metrics_collector.increment_counter("extractions_successful", 1, base_tags)
        else:
            self.metrics_collector.increment_counter("extractions_failed", 1, base_tags)
    
    def record_pii_detection_performance(self, detection_time: float, pii_count: int, document_size: int,
                                       tags: Optional[Dict[str, str]] = None):
        """Record PII detection performance metrics"""
        self.metrics_collector.record_timer("pii_detection_duration", detection_time, tags)
        self.metrics_collector.set_gauge("pii_instances_detected", pii_count, tags)
        self.metrics_collector.set_gauge("document_size_processed", document_size, tags, "bytes")
    
    def record_session_metrics(self, session_id: str, session_age: float, requests_count: int,
                             success_rate: float, tags: Optional[Dict[str, str]] = None):
        """Record browser session metrics"""
        base_tags = {"session_id": session_id}
        if tags:
            base_tags.update(tags)
        
        self.metrics_collector.set_gauge("session_age", session_age, base_tags, "seconds")
        self.metrics_collector.set_gauge("session_requests", requests_count, base_tags)
        self.metrics_collector.set_gauge("session_success_rate", success_rate, base_tags, "percentage")
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return "unknown"


class SecurityMonitor:
    """Monitors security events and compliance"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.security_events: List[MonitoringEvent] = []
        self.threat_indicators: Dict[str, int] = defaultdict(int)
        
        logger.info("Security monitor initialized")
    
    def record_pii_detection_event(self, document_id: str, pii_types: List[str], pii_count: int,
                                  source: str = "pii_detector"):
        """Record PII detection security event"""
        event = MonitoringEvent(
            event_id=uuid4().hex,
            event_type=EventType.SECURITY,
            source=source,
            message=f"PII detected in document {document_id}",
            timestamp=datetime.utcnow(),
            metadata={
                "document_id": document_id,
                "pii_types": pii_types,
                "pii_count": pii_count
            },
            severity=AlertSeverity.WARNING if pii_count > 5 else AlertSeverity.INFO
        )
        
        self.security_events.append(event)
        self.metrics_collector.increment_counter("security_events_total", 1, {"type": "pii_detection"})
        
        if pii_count > 10:
            self.metrics_collector.increment_counter("security_violations", 1, {"type": "high_pii_count"})
    
    def record_authentication_event(self, success: bool, url: str, user_id: Optional[str] = None):
        """Record authentication security event"""
        event = MonitoringEvent(
            event_id=uuid4().hex,
            event_type=EventType.SECURITY,
            source="authentication",
            message=f"Authentication {'successful' if success else 'failed'} for {url}",
            timestamp=datetime.utcnow(),
            metadata={
                "success": success,
                "url": url,
                "user_id": user_id
            },
            severity=AlertSeverity.WARNING if not success else AlertSeverity.INFO
        )
        
        self.security_events.append(event)
        
        if not success:
            self.threat_indicators["failed_auth"] += 1
            self.metrics_collector.increment_counter("authentication_failures", 1, {"domain": self._extract_domain(url)})
        else:
            self.metrics_collector.increment_counter("authentication_successes", 1, {"domain": self._extract_domain(url)})
    
    def record_data_access_event(self, resource: str, action: str, user_id: Optional[str] = None,
                               sensitive_data: bool = False):
        """Record data access security event"""
        event = MonitoringEvent(
            event_id=uuid4().hex,
            event_type=EventType.SECURITY,
            source="data_access",
            message=f"Data access: {action} on {resource}",
            timestamp=datetime.utcnow(),
            metadata={
                "resource": resource,
                "action": action,
                "user_id": user_id,
                "sensitive_data": sensitive_data
            },
            severity=AlertSeverity.WARNING if sensitive_data else AlertSeverity.INFO
        )
        
        self.security_events.append(event)
        
        tags = {"action": action, "sensitive": str(sensitive_data)}
        self.metrics_collector.increment_counter("data_access_events", 1, tags)
    
    def record_compliance_violation(self, violation_type: str, description: str, severity: AlertSeverity = AlertSeverity.ERROR):
        """Record compliance violation"""
        event = MonitoringEvent(
            event_id=uuid4().hex,
            event_type=EventType.COMPLIANCE,
            source="compliance_checker",
            message=f"Compliance violation: {violation_type}",
            timestamp=datetime.utcnow(),
            metadata={
                "violation_type": violation_type,
                "description": description
            },
            severity=severity
        )
        
        self.security_events.append(event)
        self.metrics_collector.increment_counter("compliance_violations", 1, {"type": violation_type})
    
    def get_security_summary(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get security events summary"""
        if since is None:
            since = datetime.utcnow() - timedelta(hours=24)
        
        recent_events = [e for e in self.security_events if e.timestamp >= since]
        
        event_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for event in recent_events:
            event_counts[event.event_type.value] += 1
            severity_counts[event.severity.value] += 1
        
        return {
            "total_events": len(recent_events),
            "event_types": dict(event_counts),
            "severity_distribution": dict(severity_counts),
            "threat_indicators": dict(self.threat_indicators),
            "time_range": {
                "start": since.isoformat(),
                "end": datetime.utcnow().isoformat()
            }
        }
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return "unknown"


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Dict[str, Any]] = []
        self.notification_handlers: List[Callable] = []
        
        logger.info("Alert manager initialized")
    
    def add_alert_rule(self, name: str, condition: Callable[[MetricsCollector], bool],
                      severity: AlertSeverity, message_template: str):
        """Add alert rule"""
        rule = {
            "name": name,
            "condition": condition,
            "severity": severity,
            "message_template": message_template
        }
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {name}")
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add notification handler"""
        self.notification_handlers.append(handler)
    
    def check_alerts(self):
        """Check all alert rules and trigger alerts if needed"""
        for rule in self.alert_rules:
            try:
                if rule["condition"](self.metrics_collector):
                    self._trigger_alert(rule)
            except Exception as e:
                logger.error(f"Error checking alert rule {rule['name']}: {str(e)}")
    
    def _trigger_alert(self, rule: Dict[str, Any]):
        """Trigger an alert"""
        alert_id = f"{rule['name']}_{int(time.time())}"
        
        # Check if similar alert is already active
        existing_alert = None
        for alert in self.active_alerts.values():
            if alert.name == rule["name"] and not alert.resolved:
                existing_alert = alert
                break
        
        if existing_alert:
            logger.debug(f"Alert {rule['name']} already active, skipping")
            return
        
        alert = Alert(
            alert_id=alert_id,
            name=rule["name"],
            severity=rule["severity"],
            message=rule["message_template"],
            timestamp=datetime.utcnow()
        )
        
        self.active_alerts[alert_id] = alert
        
        # Send notifications
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error sending notification: {str(e)}")
        
        logger.warning(f"Alert triggered: {alert.name} - {alert.message}")
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_timestamp = datetime.utcnow()
            logger.info(f"Alert resolved: {alert.name}")


class CloudWatchIntegration:
    """Integration with AWS CloudWatch for metrics and logging"""
    
    def __init__(self, region: str, namespace: str = "LlamaIndex/AgentCore"):
        self.region = region
        self.namespace = namespace
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.logs_client = boto3.client('logs', region_name=region)
        
        logger.info(f"CloudWatch integration initialized for region {region}")
    
    def send_metrics(self, metrics: List[Metric]):
        """Send metrics to CloudWatch"""
        try:
            metric_data = []
            
            for metric in metrics:
                dimensions = [
                    {"Name": key, "Value": value} 
                    for key, value in metric.tags.items()
                ]
                
                metric_data.append({
                    "MetricName": metric.name,
                    "Value": metric.value,
                    "Unit": self._convert_unit(metric.unit),
                    "Timestamp": metric.timestamp,
                    "Dimensions": dimensions
                })
            
            # Send in batches of 20 (CloudWatch limit)
            for i in range(0, len(metric_data), 20):
                batch = metric_data[i:i+20]
                
                self.cloudwatch.put_metric_data(
                    Namespace=self.namespace,
                    MetricData=batch
                )
            
            logger.debug(f"Sent {len(metrics)} metrics to CloudWatch")
            
        except ClientError as e:
            logger.error(f"Failed to send metrics to CloudWatch: {str(e)}")
    
    def send_log_event(self, log_group: str, log_stream: str, event: MonitoringEvent):
        """Send log event to CloudWatch Logs"""
        try:
            self.logs_client.put_log_events(
                logGroupName=log_group,
                logStreamName=log_stream,
                logEvents=[{
                    "timestamp": int(event.timestamp.timestamp() * 1000),
                    "message": json.dumps(event.to_dict())
                }]
            )
            
        except ClientError as e:
            logger.error(f"Failed to send log event to CloudWatch: {str(e)}")
    
    def _convert_unit(self, unit: str) -> str:
        """Convert unit to CloudWatch format"""
        unit_mapping = {
            "count": "Count",
            "seconds": "Seconds",
            "bytes": "Bytes",
            "percentage": "Percent"
        }
        return unit_mapping.get(unit, "Count")


class LlamaIndexMonitor:
    """Main monitoring class for LlamaIndex applications"""
    
    def __init__(self, region: str = "us-east-1", enable_cloudwatch: bool = False):
        self.region = region
        self.metrics_collector = MetricsCollector()
        self.performance_monitor = PerformanceMonitor(self.metrics_collector)
        self.security_monitor = SecurityMonitor(self.metrics_collector)
        self.alert_manager = AlertManager(self.metrics_collector)
        
        # CloudWatch integration
        self.cloudwatch_integration = None
        if enable_cloudwatch:
            try:
                self.cloudwatch_integration = CloudWatchIntegration(region)
            except Exception as e:
                logger.warning(f"Failed to initialize CloudWatch integration: {str(e)}")
        
        # Background monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._setup_default_alerts()
        
        logger.info("LlamaIndex monitor initialized")
    
    def _setup_default_alerts(self):
        """Setup default alert rules"""
        # High error rate alert
        def high_error_rate(metrics: MetricsCollector) -> bool:
            failed_queries = metrics.get_metric_summary("queries_failed", datetime.utcnow() - timedelta(minutes=5))
            total_queries = metrics.get_metric_summary("queries_total", datetime.utcnow() - timedelta(minutes=5))
            
            if total_queries["count"] > 10:
                error_rate = failed_queries["count"] / total_queries["count"]
                return error_rate > 0.1  # 10% error rate
            return False
        
        self.alert_manager.add_alert_rule(
            "high_query_error_rate",
            high_error_rate,
            AlertSeverity.WARNING,
            "High query error rate detected (>10%)"
        )
        
        # High PII detection alert
        def high_pii_detection(metrics: MetricsCollector) -> bool:
            pii_events = metrics.get_metric_summary("security_events_total", datetime.utcnow() - timedelta(minutes=10))
            return pii_events["count"] > 50
        
        self.alert_manager.add_alert_rule(
            "high_pii_detection",
            high_pii_detection,
            AlertSeverity.ERROR,
            "High number of PII detection events"
        )
    
    async def start_monitoring(self, interval: int = 60):
        """Start background monitoring"""
        self._monitoring_task = asyncio.create_task(self._monitoring_loop(interval))
        logger.info("Background monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Background monitoring stopped")
    
    async def _monitoring_loop(self, interval: int):
        """Background monitoring loop"""
        while True:
            try:
                # Check alerts
                self.alert_manager.check_alerts()
                
                # Send metrics to CloudWatch if enabled
                if self.cloudwatch_integration:
                    recent_metrics = self.metrics_collector.get_metrics(
                        since=datetime.utcnow() - timedelta(seconds=interval)
                    )
                    if recent_metrics:
                        self.cloudwatch_integration.send_metrics(recent_metrics)
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(interval)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        now = datetime.utcnow()
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)
        
        return {
            "performance": {
                "queries_last_hour": self.metrics_collector.get_metric_summary("queries_total", last_hour),
                "query_duration_last_hour": self.metrics_collector.get_metric_summary("query_duration", last_hour),
                "extractions_last_hour": self.metrics_collector.get_metric_summary("extractions_total", last_hour),
                "extraction_duration_last_hour": self.metrics_collector.get_metric_summary("extraction_duration", last_hour)
            },
            "security": self.security_monitor.get_security_summary(last_day),
            "alerts": {
                "active_alerts": len([a for a in self.alert_manager.active_alerts.values() if not a.resolved]),
                "total_alerts_today": len([a for a in self.alert_manager.active_alerts.values() 
                                         if a.timestamp >= now - timedelta(days=1)])
            },
            "system": {
                "uptime": "N/A",  # Would need to track application start time
                "last_updated": now.isoformat()
            }
        }


# Utility functions and decorators

def monitor_performance(monitor: LlamaIndexMonitor, operation_name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator to monitor function performance"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            with monitor.performance_monitor.timer(operation_name, tags):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            with monitor.performance_monitor.timer(operation_name, tags):
                return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def log_security_event(monitor: LlamaIndexMonitor, event_type: str):
    """Decorator to log security events"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                monitor.security_monitor.record_data_access_event(
                    resource=func.__name__,
                    action=event_type,
                    sensitive_data=True
                )
                return result
            except Exception as e:
                monitor.security_monitor.record_compliance_violation(
                    violation_type="operation_failure",
                    description=f"Failed to execute {func.__name__}: {str(e)}",
                    severity=AlertSeverity.ERROR
                )
                raise
        
        def sync_wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                monitor.security_monitor.record_data_access_event(
                    resource=func.__name__,
                    action=event_type,
                    sensitive_data=True
                )
                return result
            except Exception as e:
                monitor.security_monitor.record_compliance_violation(
                    violation_type="operation_failure",
                    description=f"Failed to execute {func.__name__}: {str(e)}",
                    severity=AlertSeverity.ERROR
                )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Example usage
async def example_monitoring():
    """Example of monitoring usage"""
    
    # Initialize monitor
    monitor = LlamaIndexMonitor("us-east-1", enable_cloudwatch=True)
    
    # Add custom notification handler
    def email_notification(alert: Alert):
        print(f"EMAIL ALERT: {alert.name} - {alert.message}")
    
    monitor.alert_manager.add_notification_handler(email_notification)
    
    # Start background monitoring
    await monitor.start_monitoring(interval=30)
    
    try:
        # Simulate some operations
        for i in range(10):
            # Record some metrics
            monitor.performance_monitor.record_query_performance(
                query_time=0.5 + i * 0.1,
                success=i < 8,  # Simulate some failures
                tags={"query_type": "test"}
            )
            
            monitor.security_monitor.record_pii_detection_event(
                document_id=f"doc_{i}",
                pii_types=["email", "phone"],
                pii_count=i % 5
            )
            
            await asyncio.sleep(1)
        
        # Get dashboard data
        dashboard = monitor.get_dashboard_data()
        print(json.dumps(dashboard, indent=2))
        
    finally:
        await monitor.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(example_monitoring())