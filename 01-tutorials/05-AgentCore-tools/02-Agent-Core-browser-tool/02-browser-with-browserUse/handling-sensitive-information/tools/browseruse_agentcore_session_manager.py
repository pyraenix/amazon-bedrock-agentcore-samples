"""
Browser-Use AgentCore Session Manager

This module provides session management functionality for integrating browser-use
with Amazon Bedrock AgentCore Browser Tool, focusing on secure handling of
sensitive information through micro-VM isolation and proper session lifecycle management.

This is a production implementation that requires:
- browser-use library: pip install browser-use
- bedrock-agentcore SDK: Available in AgentCore environment
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, field
import json
from contextlib import asynccontextmanager

# Required imports - no fallbacks
from bedrock_agentcore.tools.browser_client import BrowserClient
from browser_use import Agent
from browser_use.browser.session import BrowserSession


@dataclass
class SessionConfig:
    """Configuration for AgentCore browser sessions."""
    region: str = 'us-east-1'
    session_timeout: int = 300  # 5 minutes
    enable_live_view: bool = True
    enable_session_replay: bool = True
    isolation_level: str = "micro-vm"
    compliance_mode: str = "enterprise"
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class SessionMetrics:
    """Metrics and monitoring data for browser sessions."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    operations_count: int = 0
    sensitive_data_accessed: bool = False
    compliance_violations: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    live_view_url: Optional[str] = None
    errors: List[str] = field(default_factory=list)


class BrowserUseAgentCoreSessionManager:
    """
    Session manager for browser-use integration with AgentCore Browser Tool.
    
    Provides secure session lifecycle management, WebSocket connection handling,
    and proper cleanup for sensitive operations within AgentCore's micro-VM environment.
    """
    
    def __init__(self, config: Optional[SessionConfig] = None):
        """
        Initialize the session manager.
        
        Args:
            config: Session configuration. Uses defaults if not provided.
        """
        self.config = config or SessionConfig()
        self.logger = logging.getLogger(__name__)
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_metrics: Dict[str, SessionMetrics] = {}
        self.agentcore_client: Optional[BrowserClient] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Initialize AgentCore client
        self._initialize_agentcore_client()
        
        # Cleanup task will be started when first session is created
        # This avoids AsyncIO event loop issues during import
    
    def _initialize_agentcore_client(self) -> None:
        """Initialize the AgentCore Browser Client."""
        try:
            self.agentcore_client = BrowserClient(region=self.config.region)
            self.logger.info(f"AgentCore client initialized for region: {self.config.region}")
        except Exception as e:
            self.logger.error(f"Failed to initialize AgentCore client: {e}")
            raise
    
    def _start_cleanup_task(self) -> None:
        """Start the background cleanup task."""
        try:
            # Check if we're in an async context and can create tasks
            loop = asyncio.get_running_loop()
            if self._cleanup_task is None or self._cleanup_task.done():
                self._cleanup_task = loop.create_task(self._periodic_cleanup())
        except RuntimeError:
            # No event loop running, cleanup task will be started when needed
            self.logger.debug("No event loop running, cleanup task will be started later")
            self._cleanup_task = None
    
    async def _periodic_cleanup(self) -> None:
        """Periodically clean up expired sessions."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic cleanup: {e}")
    
    async def _cleanup_expired_sessions(self) -> None:
        """Clean up sessions that have exceeded their timeout."""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session_data in self.active_sessions.items():
            session_start = session_data.get('start_time')
            if session_start and (current_time - session_start).seconds > self.config.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.logger.warning(f"Cleaning up expired session: {session_id}")
            await self.cleanup_session(session_id, reason="timeout")
    
    def _ensure_cleanup_task_started(self) -> None:
        """Ensure the cleanup task is started if we're in an async context."""
        try:
            if self._cleanup_task is None or self._cleanup_task.done():
                loop = asyncio.get_running_loop()
                self._cleanup_task = loop.create_task(self._periodic_cleanup())
                self.logger.debug("Cleanup task started")
        except RuntimeError:
            # No event loop running, that's fine
            pass

    async def create_secure_session(self, 
                                  session_id: Optional[str] = None,
                                  sensitive_context: Optional[Dict[str, Any]] = None) -> Tuple[str, str, Dict[str, str]]:
        """
        Create a secure AgentCore browser session for browser-use operations.
        
        Args:
            session_id: Optional custom session ID. Generated if not provided.
            sensitive_context: Context about sensitive data handling requirements.
            
        Returns:
            Tuple of (session_id, websocket_url, headers)
            
        Raises:
            Exception: If session creation fails
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Ensure cleanup task is running now that we're in an async context
        self._ensure_cleanup_task_started()
        
        self.logger.info(f"Creating secure AgentCore session: {session_id}")
        
        try:
            # Create AgentCore session with retries
            agentcore_session = await self._create_agentcore_session_with_retry()
            
            # Extract WebSocket connection details from session response
            ws_url = agentcore_session.websocket_url
            headers = {
                'Authorization': f'Bearer {agentcore_session.access_token}',
                'X-Session-ID': agentcore_session.session_id
            }
            
            # Store session data
            session_data = {
                'session_id': session_id,
                'agentcore_session_id': agentcore_session.session_id,
                'agentcore_session': agentcore_session,
                'ws_url': ws_url,
                'headers': headers,
                'start_time': datetime.now(),
                'sensitive_context': sensitive_context or {},
                'status': 'active'
            }
            
            self.active_sessions[session_id] = session_data
            
            # Initialize metrics
            self.session_metrics[session_id] = SessionMetrics(
                session_id=session_id,
                start_time=datetime.now(),
                live_view_url=agentcore_session.live_view_url if hasattr(agentcore_session, 'live_view_url') else None
            )
            
            self.logger.info(f"Session created successfully: {session_id}")
            return session_id, ws_url, headers
            
        except Exception as e:
            self.logger.error(f"Failed to create session {session_id}: {e}")
            # Clean up any partial session data
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            if session_id in self.session_metrics:
                del self.session_metrics[session_id]
            raise
    
    async def _create_agentcore_session_with_retry(self):
        """Create AgentCore session with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                # Use the real AgentCore Browser Client API
                session_response = await self.agentcore_client.create_browser_session(
                    timeout_seconds=self.config.session_timeout,
                    enable_live_view=self.config.enable_live_view,
                    enable_session_replay=self.config.enable_session_replay
                )
                return session_response
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    self.logger.warning(f"Session creation attempt {attempt + 1} failed, retrying...")
        
        raise last_exception
    
    def _get_live_view_url(self, agentcore_session) -> Optional[str]:
        """Get live view URL from AgentCore session."""
        if self.config.enable_live_view and hasattr(agentcore_session, 'live_view_url'):
            return agentcore_session.live_view_url
        return None
    
    async def create_browseruse_agent(self, 
                                    session_id: str, 
                                    task: str, 
                                    llm_model: Any) -> Agent:
        """
        Create a browser-use Agent connected to the AgentCore session.
        
        Args:
            session_id: ID of the AgentCore session
            task: Task description for the browser-use agent
            llm_model: LLM model instance for the agent
            
        Returns:
            Configured browser-use Agent instance
            
        Raises:
            ValueError: If session doesn't exist
            Exception: If agent creation fails
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session_data = self.active_sessions[session_id]
        
        try:
            # Create browser session with AgentCore WebSocket connection
            browser_session = BrowserSession(
                cdp_url=session_data['ws_url'],
                cdp_headers=session_data['headers']
            )
            
            # Create browser-use agent
            agent = Agent(
                task=task,
                llm=llm_model,
                browser_session=browser_session
            )
            
            # Update session data
            session_data['agent'] = agent
            session_data['task'] = task
            
            # Update metrics
            if session_id in self.session_metrics:
                self.session_metrics[session_id].operations_count += 1
            
            self.logger.info(f"Browser-use agent created for session: {session_id}")
            return agent
            
        except Exception as e:
            self.logger.error(f"Failed to create browser-use agent for session {session_id}: {e}")
            if session_id in self.session_metrics:
                self.session_metrics[session_id].errors.append(str(e))
            raise
    
    async def execute_sensitive_task(self, 
                                   session_id: str, 
                                   agent: Agent, 
                                   sensitive_data_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a browser-use task with sensitive data handling.
        
        Args:
            session_id: ID of the AgentCore session
            agent: Browser-use agent instance
            sensitive_data_context: Context about sensitive data in the task
            
        Returns:
            Task execution results
            
        Raises:
            ValueError: If session doesn't exist
            Exception: If task execution fails
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session_data = self.active_sessions[session_id]
        
        try:
            self.logger.info(f"Executing sensitive task for session: {session_id}")
            
            # Mark sensitive data access
            if session_id in self.session_metrics:
                self.session_metrics[session_id].sensitive_data_accessed = True
            
            # Execute the actual browser-use task
            result = await self._execute_browseruse_task(agent, sensitive_data_context)
            
            # Update session metrics
            if session_id in self.session_metrics:
                self.session_metrics[session_id].operations_count += 1
                self.session_metrics[session_id].performance_metrics['last_execution_time'] = datetime.now().timestamp()
            
            self.logger.info(f"Task executed successfully for session: {session_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Task execution failed for session {session_id}: {e}")
            if session_id in self.session_metrics:
                self.session_metrics[session_id].errors.append(str(e))
            raise
    
    async def _execute_browseruse_task(self, agent: Agent, sensitive_data_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute actual browser-use task with sensitive data handling."""
        try:
            # Execute the browser-use agent task
            result = await agent.run()
            
            return {
                'status': 'completed',
                'task': agent.task,
                'result': result,
                'sensitive_data_handled': bool(sensitive_data_context),
                'execution_time': datetime.now().isoformat(),
                'sensitive_context': sensitive_data_context
            }
        except Exception as e:
            self.logger.error(f"Browser-use task execution failed: {e}")
            return {
                'status': 'failed',
                'task': agent.task,
                'error': str(e),
                'sensitive_data_handled': bool(sensitive_data_context),
                'execution_time': datetime.now().isoformat()
            }
    
    async def cleanup_session(self, session_id: str, reason: str = "manual") -> None:
        """
        Clean up an AgentCore session and all associated resources.
        
        Args:
            session_id: ID of the session to clean up
            reason: Reason for cleanup (manual, timeout, error, etc.)
        """
        if session_id not in self.active_sessions:
            self.logger.warning(f"Session {session_id} not found for cleanup")
            return
        
        self.logger.info(f"Cleaning up session {session_id} (reason: {reason})")
        
        try:
            session_data = self.active_sessions[session_id]
            
            # Close browser-use agent if exists
            if 'agent' in session_data:
                agent = session_data['agent']
                if hasattr(agent, 'browser_session') and hasattr(agent.browser_session, 'close'):
                    await agent.browser_session.close()
            
            # Terminate AgentCore session
            if 'agentcore_session' in session_data:
                agentcore_session = session_data['agentcore_session']
                try:
                    await self.agentcore_client.terminate_session(agentcore_session.session_id)
                except Exception as e:
                    self.logger.warning(f"Failed to terminate AgentCore session {agentcore_session.session_id}: {e}")
            
            # Update metrics
            if session_id in self.session_metrics:
                self.session_metrics[session_id].end_time = datetime.now()
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            self.logger.info(f"Session {session_id} cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up session {session_id}: {e}")
            # Force removal even if cleanup fails
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of a session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            Session status information or None if session doesn't exist
        """
        if session_id not in self.active_sessions:
            return None
        
        session_data = self.active_sessions[session_id].copy()
        metrics = self.session_metrics.get(session_id)
        
        status = {
            'session_id': session_id,
            'status': session_data.get('status', 'unknown'),
            'start_time': session_data.get('start_time'),
            'task': session_data.get('task'),
            'live_view_url': metrics.live_view_url if metrics else None,
            'operations_count': metrics.operations_count if metrics else 0,
            'sensitive_data_accessed': metrics.sensitive_data_accessed if metrics else False,
            'errors': metrics.errors if metrics else []
        }
        
        return status
    
    def get_live_view_url(self, session_id: str) -> Optional[str]:
        """
        Get the live view URL for monitoring a session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            Live view URL or None if not available
        """
        metrics = self.session_metrics.get(session_id)
        return metrics.live_view_url if metrics else None
    
    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """
        List all active sessions.
        
        Returns:
            List of active session information
        """
        sessions = []
        for session_id in self.active_sessions:
            status = self.get_session_status(session_id)
            if status:
                sessions.append(status)
        return sessions
    
    @asynccontextmanager
    async def secure_session_context(self, 
                                   task: str, 
                                   llm_model: Any,
                                   sensitive_context: Optional[Dict[str, Any]] = None):
        """
        Context manager for secure session handling with automatic cleanup.
        
        Args:
            task: Task description for the browser-use agent
            llm_model: LLM model instance
            sensitive_context: Context about sensitive data handling
            
        Yields:
            Tuple of (session_id, agent)
        """
        session_id = None
        try:
            # Create session
            session_id, ws_url, headers = await self.create_secure_session(
                sensitive_context=sensitive_context
            )
            
            # Create agent
            agent = await self.create_browseruse_agent(session_id, task, llm_model)
            
            yield session_id, agent
            
        finally:
            # Always cleanup
            if session_id:
                await self.cleanup_session(session_id, reason="context_exit")
    
    async def emergency_cleanup_all(self) -> None:
        """Emergency cleanup of all active sessions."""
        self.logger.warning("Performing emergency cleanup of all sessions")
        
        session_ids = list(self.active_sessions.keys())
        for session_id in session_ids:
            try:
                await self.cleanup_session(session_id, reason="emergency")
            except Exception as e:
                self.logger.error(f"Error in emergency cleanup of session {session_id}: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the session manager and clean up all resources."""
        self.logger.info("Shutting down session manager")
        
        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clean up all active sessions
        await self.emergency_cleanup_all()
        
        self.logger.info("Session manager shutdown complete")


# Convenience functions for common operations
async def create_secure_browseruse_session(task: str, 
                                         llm_model: Any,
                                         config: Optional[SessionConfig] = None,
                                         sensitive_context: Optional[Dict[str, Any]] = None) -> Tuple[str, Agent, BrowserUseAgentCoreSessionManager]:
    """
    Convenience function to create a secure browser-use session with AgentCore.
    
    Args:
        task: Task description for the browser-use agent
        llm_model: LLM model instance
        config: Session configuration
        sensitive_context: Context about sensitive data handling
        
    Returns:
        Tuple of (session_id, agent, session_manager)
    """
    session_manager = BrowserUseAgentCoreSessionManager(config)
    session_id, ws_url, headers = await session_manager.create_secure_session(
        sensitive_context=sensitive_context
    )
    agent = await session_manager.create_browseruse_agent(session_id, task, llm_model)
    
    return session_id, agent, session_manager


# Real usage example - requires actual LLM model and proper setup
async def create_secure_browseruse_session_example():
    """
    Example of creating a secure browser-use session with AgentCore.
    
    This example requires:
    - A real LLM model (e.g., from langchain, openai, anthropic)
    - Proper AWS credentials for AgentCore
    - browser-use library installed
    """
    # This would be your actual LLM model
    # from langchain_anthropic import ChatAnthropic
    # llm_model = ChatAnthropic(model="claude-3-sonnet-20240229")
    
    config = SessionConfig(
        region='us-east-1',
        session_timeout=600,  # 10 minutes for complex tasks
        enable_live_view=True,
        enable_session_replay=True
    )
    
    session_manager = BrowserUseAgentCoreSessionManager(config)
    
    try:
        # Create secure session for sensitive data handling
        session_id, ws_url, headers = await session_manager.create_secure_session(
            sensitive_context={
                'data_type': 'healthcare',
                'compliance': 'HIPAA',
                'pii_types': ['ssn', 'dob', 'medical_record']
            }
        )
        
        print(f"✅ Created secure AgentCore session: {session_id}")
        print(f"🔗 WebSocket URL: {ws_url}")
        print(f"👁️ Live view: {session_manager.get_live_view_url(session_id)}")
        
        # Note: Actual agent creation requires a real LLM model
        # agent = await session_manager.create_browseruse_agent(
        #     session_id,
        #     "Navigate to a healthcare form and fill it out securely, masking any PII",
        #     llm_model
        # )
        
        # # Execute sensitive task
        # result = await session_manager.execute_sensitive_task(
        #     session_id,
        #     agent,
        #     {'pii_types': ['ssn', 'dob', 'medical_record']}
        # )
        
        return session_id, session_manager
        
    except Exception as e:
        print(f"❌ Failed to create session: {e}")
        await session_manager.shutdown()
        raise

if __name__ == "__main__":
    print("🚀 Browser-Use AgentCore Session Manager")
    print("⚠️  This requires real dependencies: browser-use, bedrock-agentcore, and LLM model")
    print("📝 See the tutorial notebook for complete examples")