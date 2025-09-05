#!/usr/bin/env python3
"""
AgentCore Session Helpers for LlamaIndex Integration

This module provides utility functions and classes for managing AgentCore Browser Tool
sessions when integrated with LlamaIndex agents. Includes session pooling, health monitoring,
lifecycle management, and security controls.

Key Features:
- Session pool management for concurrent LlamaIndex operations
- Session health monitoring and automatic recovery
- Secure session lifecycle management
- Resource cleanup and optimization
- Error handling and retry mechanisms
- Audit logging for compliance

Requirements: 1.2, 2.5, 4.2, 5.4
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from contextlib import asynccontextmanager
import weakref
from uuid import uuid4

# Import for AgentCore Browser Tool (not Browser Client)
from bedrock_agentcore.tools.browser_client import browser_session

# Configure logging
logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Browser session states"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    EXPIRED = "expired"
    CLOSED = "closed"





class SessionPriority(Enum):
    """Session priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SessionMetrics:
    """Session performance and usage metrics"""
    session_id: str
    created_at: datetime
    last_used: datetime
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    network_usage: float = 0.0
    
    def update_request_metrics(self, success: bool, response_time: float):
        """Update request metrics"""
        self.total_requests += 1
        self.last_used = datetime.utcnow()
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Update average response time
        if self.total_requests == 1:
            self.average_response_time = response_time
        else:
            self.average_response_time = (
                (self.average_response_time * (self.total_requests - 1) + response_time) 
                / self.total_requests
            )
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def age(self) -> timedelta:
        """Calculate session age"""
        return datetime.utcnow() - self.created_at
    
    @property
    def idle_time(self) -> timedelta:
        """Calculate idle time"""
        return datetime.utcnow() - self.last_used


@dataclass
class SessionConfig:
    """Configuration for browser sessions"""
    max_idle_time: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    max_session_age: timedelta = field(default_factory=lambda: timedelta(hours=2))
    max_requests_per_session: int = 1000
    enable_screenshots: bool = True
    enable_network_monitoring: bool = True
    enable_performance_monitoring: bool = True
    timeout_seconds: int = 30
    retry_attempts: int = 3
    security_level: str = "high"
    audit_logging: bool = True


class ManagedBrowserSession:
    """Managed browser session with health monitoring and lifecycle management"""
    
    def __init__(self, session: Any, session_id: str, config: SessionConfig):
        self.session = session
        self.session_id = session_id
        self.config = config
        self.state = SessionState.INITIALIZING
        self.metrics = SessionMetrics(session_id=session_id, created_at=datetime.utcnow(), last_used=datetime.utcnow())
        self.priority = SessionPriority.MEDIUM
        self.tags: Dict[str, str] = {}
        self.cleanup_callbacks: List[Callable] = []
        self._lock = asyncio.Lock()
        
        logger.info(f"Created managed session {session_id}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.release()
    
    async def acquire(self):
        """Acquire session for use"""
        async with self._lock:
            if self.state == SessionState.CLOSED:
                raise RuntimeError(f"Session {self.session_id} is closed")
            
            if self.state == SessionState.ERROR:
                await self._recover_session()
            
            self.state = SessionState.BUSY
            logger.debug(f"Acquired session {self.session_id}")
    
    async def release(self):
        """Release session after use"""
        async with self._lock:
            if self.state == SessionState.BUSY:
                self.state = SessionState.IDLE
                self.metrics.last_used = datetime.utcnow()
                logger.debug(f"Released session {self.session_id}")
    
    async def execute_with_retry(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                start_time = datetime.utcnow()
                result = await operation(*args, **kwargs)
                end_time = datetime.utcnow()
                
                response_time = (end_time - start_time).total_seconds()
                self.metrics.update_request_metrics(True, response_time)
                
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Session {self.session_id} operation failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
                self.metrics.update_request_metrics(False, 0.0)
        
        self.state = SessionState.ERROR
        raise last_exception
    
    async def health_check(self) -> bool:
        """Perform session health check"""
        try:
            # Check if session is still responsive
            await self.session.evaluate("document.readyState")
            
            # Check session age and usage
            if self.metrics.age > self.config.max_session_age:
                logger.info(f"Session {self.session_id} exceeded max age")
                return False
            
            if self.metrics.total_requests > self.config.max_requests_per_session:
                logger.info(f"Session {self.session_id} exceeded max requests")
                return False
            
            if self.metrics.idle_time > self.config.max_idle_time:
                logger.info(f"Session {self.session_id} exceeded max idle time")
                return False
            
            # Check success rate
            if self.metrics.total_requests > 10 and self.metrics.success_rate < 0.5:
                logger.warning(f"Session {self.session_id} has low success rate: {self.metrics.success_rate}")
                return False
            
            self.state = SessionState.ACTIVE
            return True
            
        except Exception as e:
            logger.error(f"Health check failed for session {self.session_id}: {str(e)}")
            self.state = SessionState.ERROR
            return False
    
    async def _recover_session(self):
        """Attempt to recover session from error state"""
        try:
            logger.info(f"Attempting to recover session {self.session_id}")
            
            # Try to navigate to a simple page to test responsiveness
            await self.session.navigate("about:blank")
            
            self.state = SessionState.ACTIVE
            logger.info(f"Successfully recovered session {self.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to recover session {self.session_id}: {str(e)}")
            self.state = SessionState.ERROR
            raise
    
    async def close(self):
        """Close session and cleanup resources"""
        async with self._lock:
            if self.state != SessionState.CLOSED:
                try:
                    # Execute cleanup callbacks
                    for callback in self.cleanup_callbacks:
                        try:
                            await callback()
                        except Exception as e:
                            logger.warning(f"Cleanup callback failed: {str(e)}")
                    
                    # Close the browser session
                    await self.session.close()
                    self.state = SessionState.CLOSED
                    
                    logger.info(f"Closed session {self.session_id}")
                    
                except Exception as e:
                    logger.error(f"Error closing session {self.session_id}: {str(e)}")
    
    def add_cleanup_callback(self, callback: Callable):
        """Add cleanup callback"""
        self.cleanup_callbacks.append(callback)
    
    def add_tag(self, key: str, value: str):
        """Add tag to session"""
        self.tags[key] = value
    
    def get_tag(self, key: str) -> Optional[str]:
        """Get tag value"""
        return self.tags.get(key)


class SessionPool:
    """Pool of managed browser sessions for LlamaIndex operations"""
    
    def __init__(self, region: str, max_sessions: int = 10, config: Optional[SessionConfig] = None):
        self.region = region
        self.max_sessions = max_sessions
        self.config = config or SessionConfig()
        self.sessions: Dict[str, ManagedBrowserSession] = {}
        self.available_sessions: asyncio.Queue = asyncio.Queue()
        self.browser_client = BrowserClient(region=region)
        self._lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info(f"Initialized session pool with max {max_sessions} sessions")
    
    async def start(self):
        """Start the session pool"""
        # Start background tasks
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Session pool started")
    
    async def stop(self):
        """Stop the session pool and cleanup all sessions"""
        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Close all sessions
        async with self._lock:
            for session in list(self.sessions.values()):
                await session.close()
            self.sessions.clear()
        
        logger.info("Session pool stopped")
    
    @asynccontextmanager
    async def get_session(self, priority: SessionPriority = SessionPriority.MEDIUM, tags: Optional[Dict[str, str]] = None):
        """Get a session from the pool"""
        session = await self._acquire_session(priority, tags)
        try:
            yield session
        finally:
            await self._release_session(session)
    
    async def _acquire_session(self, priority: SessionPriority, tags: Optional[Dict[str, str]]) -> ManagedBrowserSession:
        """Acquire a session from the pool"""
        # Try to get an available session
        try:
            session_id = await asyncio.wait_for(self.available_sessions.get(), timeout=5.0)
            session = self.sessions.get(session_id)
            
            if session and await session.health_check():
                session.priority = priority
                if tags:
                    for key, value in tags.items():
                        session.add_tag(key, value)
                
                await session.acquire()
                return session
            else:
                # Session is unhealthy, remove it
                if session:
                    await self._remove_session(session_id)
        
        except asyncio.TimeoutError:
            pass
        
        # Create new session if pool not full
        async with self._lock:
            if len(self.sessions) < self.max_sessions:
                return await self._create_session(priority, tags)
        
        # Wait for available session
        session_id = await self.available_sessions.get()
        session = self.sessions[session_id]
        
        if await session.health_check():
            session.priority = priority
            if tags:
                for key, value in tags.items():
                    session.add_tag(key, value)
            
            await session.acquire()
            return session
        else:
            await self._remove_session(session_id)
            return await self._acquire_session(priority, tags)
    
    async def _release_session(self, session: ManagedBrowserSession):
        """Release a session back to the pool"""
        await session.release()
        await self.available_sessions.put(session.session_id)
    
    async def _create_session(self, priority: SessionPriority, tags: Optional[Dict[str, str]]) -> ManagedBrowserSession:
        """Create a new managed session"""
        session_id = f"session_{uuid4().hex[:8]}"
        
        # Create browser session
        browser_session = await self.browser_client.create_session()
        
        # Create managed session
        managed_session = ManagedBrowserSession(browser_session, session_id, self.config)
        managed_session.priority = priority
        
        if tags:
            for key, value in tags.items():
                managed_session.add_tag(key, value)
        
        # Add to pool
        self.sessions[session_id] = managed_session
        
        await managed_session.acquire()
        
        logger.info(f"Created new session {session_id}")
        return managed_session
    
    async def _remove_session(self, session_id: str):
        """Remove session from pool"""
        async with self._lock:
            session = self.sessions.pop(session_id, None)
            if session:
                await session.close()
                logger.info(f"Removed session {session_id} from pool")
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                unhealthy_sessions = []
                for session_id, session in self.sessions.items():
                    if not await session.health_check():
                        unhealthy_sessions.append(session_id)
                
                # Remove unhealthy sessions
                for session_id in unhealthy_sessions:
                    await self._remove_session(session_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {str(e)}")
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
                expired_sessions = []
                for session_id, session in self.sessions.items():
                    if (session.metrics.idle_time > self.config.max_idle_time or 
                        session.metrics.age > self.config.max_session_age):
                        expired_sessions.append(session_id)
                
                # Remove expired sessions
                for session_id in expired_sessions:
                    await self._remove_session(session_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {str(e)}")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        total_sessions = len(self.sessions)
        active_sessions = sum(1 for s in self.sessions.values() if s.state == SessionState.ACTIVE)
        idle_sessions = sum(1 for s in self.sessions.values() if s.state == SessionState.IDLE)
        error_sessions = sum(1 for s in self.sessions.values() if s.state == SessionState.ERROR)
        
        total_requests = sum(s.metrics.total_requests for s in self.sessions.values())
        successful_requests = sum(s.metrics.successful_requests for s in self.sessions.values())
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "idle_sessions": idle_sessions,
            "error_sessions": error_sessions,
            "available_slots": self.max_sessions - total_sessions,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0.0
        }


class SessionManager:
    """High-level session manager for LlamaIndex-AgentCore integration"""
    
    def __init__(self, region: str, pool_config: Optional[Dict[str, Any]] = None):
        self.region = region
        self.pools: Dict[str, SessionPool] = {}
        self.default_pool_config = pool_config or {}
        
        logger.info("Session manager initialized")
    
    async def get_pool(self, pool_name: str = "default", **config) -> SessionPool:
        """Get or create a session pool"""
        if pool_name not in self.pools:
            pool_config = {**self.default_pool_config, **config}
            max_sessions = pool_config.pop("max_sessions", 10)
            session_config = SessionConfig(**pool_config)
            
            pool = SessionPool(self.region, max_sessions, session_config)
            await pool.start()
            
            self.pools[pool_name] = pool
            logger.info(f"Created session pool '{pool_name}'")
        
        return self.pools[pool_name]
    
    @asynccontextmanager
    async def session(self, pool_name: str = "default", priority: SessionPriority = SessionPriority.MEDIUM, 
                     tags: Optional[Dict[str, str]] = None, **pool_config):
        """Get a session from specified pool"""
        pool = await self.get_pool(pool_name, **pool_config)
        async with pool.get_session(priority, tags) as session:
            yield session
    
    async def close_all_pools(self):
        """Close all session pools"""
        for pool_name, pool in self.pools.items():
            await pool.stop()
            logger.info(f"Closed session pool '{pool_name}'")
        
        self.pools.clear()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all pools"""
        return {name: pool.get_pool_stats() for name, pool in self.pools.items()}


# Utility functions for common session operations

async def create_secure_session(region: str, security_config: Optional[Dict[str, Any]] = None):
    """
    Create a secure session using AgentCore Browser Tool.
    
    This function provides access to the managed AgentCore Browser Tool service,
    which runs in a secure, isolated, containerized environment.
    
    Args:
        region: AWS region for the AgentCore Browser Tool
        security_config: Additional security configuration
        
    Returns:
        Context manager for AgentCore Browser Tool session
    """
    config = SessionConfig(
        security_level="high",
        audit_logging=True,
        enable_network_monitoring=True,
        enable_performance_monitoring=True,
        **(security_config or {})
    )
    
    # Return the browser_session context manager for AgentCore Browser Tool
    # This connects to the managed Browser Tool service (not a client)
    return browser_session(region=region)


async def execute_with_session_retry(session_manager: SessionManager, operation: Callable, 
                                   max_retries: int = 3, pool_name: str = "default") -> Any:
    """Execute operation with session retry logic"""
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            async with session_manager.session(pool_name) as session:
                return await session.execute_with_retry(operation)
        
        except Exception as e:
            last_exception = e
            logger.warning(f"Session operation failed (attempt {attempt + 1}): {str(e)}")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
    
    raise last_exception


def log_session_audit(session: ManagedBrowserSession, action: str, details: Optional[Dict[str, Any]] = None):
    """Log session audit event"""
    audit_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": session.session_id,
        "action": action,
        "state": session.state.value,
        "metrics": {
            "total_requests": session.metrics.total_requests,
            "success_rate": session.metrics.success_rate,
            "age_seconds": session.metrics.age.total_seconds(),
            "idle_seconds": session.metrics.idle_time.total_seconds()
        },
        "tags": session.tags,
        "details": details or {}
    }
    
    logger.info(f"Session audit: {audit_entry}")


# Example usage and testing functions

async def example_session_usage():
    """Example of how to use the session management utilities"""
    
    # Initialize session manager
    session_manager = SessionManager("us-east-1")
    
    try:
        # Use session with automatic management
        async with session_manager.session(priority=SessionPriority.HIGH, 
                                         tags={"purpose": "llamaindex", "task": "web_scraping"}) as session:
            
            # Perform operations with retry
            async def navigate_operation():
                await session.session.navigate("https://example.com")
                return await session.session.evaluate("document.title")
            
            title = await session.execute_with_retry(navigate_operation)
            print(f"Page title: {title}")
            
            # Log audit event
            log_session_audit(session, "page_navigation", {"url": "https://example.com", "title": title})
        
        # Get pool statistics
        stats = session_manager.get_all_stats()
        print(f"Pool stats: {stats}")
        
    finally:
        # Cleanup
        await session_manager.close_all_pools()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_session_usage())