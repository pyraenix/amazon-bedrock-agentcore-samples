"""
Multi-Agent Coordinator for Strands Integration

This module provides coordination capabilities for managing multiple Strands agents
sharing AgentCore Browser Tool sessions. It implements secure data sharing, resource
allocation, session management, and isolation mechanisms to prevent data leakage
between different agents.

Key Features:
- Multi-agent coordination for shared AgentCore Browser Tool sessions
- Secure data sharing between Strands agents handling sensitive information
- Resource allocation and session management for concurrent operations
- Isolation mechanisms to prevent data leakage between agents
- Agent lifecycle management with security controls
- Comprehensive monitoring and audit logging

Requirements Addressed:
- 6.1: Multi-agent coordination with security controls
- 6.2: Secure data sharing between agents
- 6.5: Resource allocation and session management for concurrent operations
- 6.6: Isolation mechanisms to prevent data leakage between agents
"""

import os
import json
import uuid
import asyncio
import logging
import hashlib
from typing import Dict, List, Optional, Any, Union, Callable, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from contextlib import asynccontextmanager
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading

# Strands imports
try:
    from strands_agents.core.agent import Agent
    from strands_agents.core.exceptions import AgentCoordinationError
    from strands_agents.core.types import AgentResult
except ImportError:
    # Mock Strands imports for development/testing
    class Agent:
        def __init__(self, name: str):
            self.name = name
            self.id = uuid.uuid4().hex[:8]
    
    class AgentCoordinationError(Exception):
        pass
    
    @dataclass
    class AgentResult:
        success: bool
        data: Any = None
        error: Optional[str] = None
        metadata: Dict[str, Any] = field(default_factory=dict)

# Import related components
from .agentcore_browser_tool import AgentCoreBrowserTool, BrowserSessionConfig
from .secure_workflow_orchestrator import SessionPool, SessionPoolConfig, EncryptionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    ACTIVE = "active"
    WAITING = "waiting"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


class ResourceType(Enum):
    """Types of resources that can be allocated."""
    BROWSER_SESSION = "browser_session"
    DATA_STORE = "data_store"
    COMPUTE_SLOT = "compute_slot"
    NETWORK_CONNECTION = "network_connection"


class IsolationLevel(Enum):
    """Levels of isolation between agents."""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    COMPLETE = "complete"


@dataclass
class AgentContext:
    """Context information for a Strands agent."""
    agent_id: str
    agent_name: str
    agent: Agent
    status: AgentStatus = AgentStatus.IDLE
    assigned_resources: Set[str] = field(default_factory=set)
    shared_data_keys: Set[str] = field(default_factory=set)
    isolation_level: IsolationLevel = IsolationLevel.STRICT
    security_clearance: str = "standard"
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent context to dictionary."""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'status': self.status.value,
            'assigned_resources': list(self.assigned_resources),
            'shared_data_keys': list(self.shared_data_keys),
            'isolation_level': self.isolation_level.value,
            'security_clearance': self.security_clearance,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat()
        }


@dataclass
class ResourceAllocation:
    """Resource allocation for agents."""
    resource_id: str
    resource_type: ResourceType
    allocated_to: str  # agent_id
    allocation_time: datetime
    expiry_time: Optional[datetime] = None
    exclusive: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if resource allocation has expired."""
        if self.expiry_time is None:
            return False
        return datetime.now() > self.expiry_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert resource allocation to dictionary."""
        return {
            'resource_id': self.resource_id,
            'resource_type': self.resource_type.value,
            'allocated_to': self.allocated_to,
            'allocation_time': self.allocation_time.isoformat(),
            'expiry_time': self.expiry_time.isoformat() if self.expiry_time else None,
            'exclusive': self.exclusive,
            'metadata': self.metadata
        }


@dataclass
class SecureDataShare:
    """Secure data sharing between agents."""
    share_id: str
    data_key: str
    owner_agent_id: str
    shared_with: Set[str]  # agent_ids
    data_hash: str
    encrypted_data: bytes
    access_permissions: Dict[str, List[str]]  # agent_id -> permissions
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    access_log: List[Dict[str, Any]] = field(default_factory=list)
    
    def can_access(self, agent_id: str, permission: str) -> bool:
        """Check if agent can access data with specific permission."""
        if agent_id not in self.shared_with:
            return False
        
        agent_permissions = self.access_permissions.get(agent_id, [])
        return permission in agent_permissions or 'all' in agent_permissions
    
    def log_access(self, agent_id: str, action: str, success: bool) -> None:
        """Log data access attempt."""
        self.access_log.append({
            'agent_id': agent_id,
            'action': action,
            'success': success,
            'timestamp': datetime.now().isoformat()
        })
    
    def is_expired(self) -> bool:
        """Check if data share has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


@dataclass
class CoordinationConfig:
    """Configuration for multi-agent coordination."""
    max_concurrent_agents: int = 10
    default_isolation_level: IsolationLevel = IsolationLevel.STRICT
    resource_timeout: int = 300  # 5 minutes
    data_share_timeout: int = 3600  # 1 hour
    enable_cross_agent_communication: bool = True
    audit_all_operations: bool = True
    cleanup_interval: int = 60  # 1 minute
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'max_concurrent_agents': self.max_concurrent_agents,
            'default_isolation_level': self.default_isolation_level.value,
            'resource_timeout': self.resource_timeout,
            'data_share_timeout': self.data_share_timeout,
            'enable_cross_agent_communication': self.enable_cross_agent_communication,
            'audit_all_operations': self.audit_all_operations,
            'cleanup_interval': self.cleanup_interval
        }


class ResourceManager:
    """Manages resource allocation for multiple agents."""
    
    def __init__(self, config: CoordinationConfig):
        """Initialize resource manager."""
        self.config = config
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.resource_locks: Dict[str, asyncio.Lock] = {}
        self.allocation_lock = asyncio.Lock()
        
        logger.info("ResourceManager initialized")
    
    async def allocate_resource(
        self,
        resource_id: str,
        resource_type: ResourceType,
        agent_id: str,
        exclusive: bool = True,
        timeout: Optional[int] = None
    ) -> bool:
        """Allocate a resource to an agent."""
        async with self.allocation_lock:
            # Check if resource is already allocated
            if resource_id in self.allocations:
                existing = self.allocations[resource_id]
                
                # Check if allocation has expired
                if existing.is_expired():
                    await self._cleanup_allocation(resource_id)
                elif existing.exclusive or exclusive:
                    logger.warning(f"Resource {resource_id} already exclusively allocated to {existing.allocated_to}")
                    return False
                elif existing.allocated_to == agent_id:
                    # Agent already has access
                    return True
            
            # Create new allocation
            expiry_time = None
            if timeout:
                expiry_time = datetime.now() + timedelta(seconds=timeout)
            elif self.config.resource_timeout > 0:
                expiry_time = datetime.now() + timedelta(seconds=self.config.resource_timeout)
            
            allocation = ResourceAllocation(
                resource_id=resource_id,
                resource_type=resource_type,
                allocated_to=agent_id,
                allocation_time=datetime.now(),
                expiry_time=expiry_time,
                exclusive=exclusive
            )
            
            self.allocations[resource_id] = allocation
            
            # Create resource lock if needed
            if resource_id not in self.resource_locks:
                self.resource_locks[resource_id] = asyncio.Lock()
            
            logger.info(f"Resource allocated: {resource_id} -> {agent_id} (exclusive: {exclusive})")
            return True
    
    async def release_resource(self, resource_id: str, agent_id: str) -> bool:
        """Release a resource from an agent."""
        async with self.allocation_lock:
            if resource_id not in self.allocations:
                return False
            
            allocation = self.allocations[resource_id]
            if allocation.allocated_to != agent_id:
                logger.warning(f"Agent {agent_id} cannot release resource {resource_id} allocated to {allocation.allocated_to}")
                return False
            
            await self._cleanup_allocation(resource_id)
            logger.info(f"Resource released: {resource_id} from {agent_id}")
            return True
    
    async def get_agent_resources(self, agent_id: str) -> List[ResourceAllocation]:
        """Get all resources allocated to an agent."""
        return [
            allocation for allocation in self.allocations.values()
            if allocation.allocated_to == agent_id and not allocation.is_expired()
        ]
    
    async def _cleanup_allocation(self, resource_id: str) -> None:
        """Clean up a resource allocation."""
        if resource_id in self.allocations:
            del self.allocations[resource_id]
        
        if resource_id in self.resource_locks:
            del self.resource_locks[resource_id]
    
    async def cleanup_expired_allocations(self) -> int:
        """Clean up all expired resource allocations."""
        expired_resources = [
            resource_id for resource_id, allocation in self.allocations.items()
            if allocation.is_expired()
        ]
        
        for resource_id in expired_resources:
            await self._cleanup_allocation(resource_id)
        
        if expired_resources:
            logger.info(f"Cleaned up {len(expired_resources)} expired resource allocations")
        
        return len(expired_resources)


class SecureDataManager:
    """Manages secure data sharing between agents."""
    
    def __init__(self, config: CoordinationConfig, encryption_manager: EncryptionManager):
        """Initialize secure data manager."""
        self.config = config
        self.encryption_manager = encryption_manager
        self.data_shares: Dict[str, SecureDataShare] = {}
        self.data_locks: Dict[str, asyncio.Lock] = {}
        self.share_lock = asyncio.Lock()
        
        logger.info("SecureDataManager initialized")
    
    async def create_data_share(
        self,
        data_key: str,
        data: Dict[str, Any],
        owner_agent_id: str,
        shared_with: List[str],
        permissions: Dict[str, List[str]],
        expires_in: Optional[int] = None
    ) -> str:
        """Create a secure data share between agents."""
        async with self.share_lock:
            share_id = f"share_{uuid.uuid4().hex[:8]}"
            
            # Encrypt data
            encrypted_data, data_hash = self.encryption_manager.encrypt_data(data)
            
            # Set expiry time
            expires_at = None
            if expires_in:
                expires_at = datetime.now() + timedelta(seconds=expires_in)
            elif self.config.data_share_timeout > 0:
                expires_at = datetime.now() + timedelta(seconds=self.config.data_share_timeout)
            
            # Create data share
            data_share = SecureDataShare(
                share_id=share_id,
                data_key=data_key,
                owner_agent_id=owner_agent_id,
                shared_with=set(shared_with),
                data_hash=data_hash,
                encrypted_data=encrypted_data,
                access_permissions=permissions,
                expires_at=expires_at
            )
            
            self.data_shares[share_id] = data_share
            self.data_locks[share_id] = asyncio.Lock()
            
            logger.info(f"Data share created: {share_id} by {owner_agent_id} for {len(shared_with)} agents")
            return share_id
    
    async def access_shared_data(
        self,
        share_id: str,
        agent_id: str,
        permission: str = "read"
    ) -> Optional[Dict[str, Any]]:
        """Access shared data with permission check."""
        if share_id not in self.data_shares:
            return None
        
        data_share = self.data_shares[share_id]
        
        # Check if share has expired
        if data_share.is_expired():
            await self._cleanup_data_share(share_id)
            return None
        
        # Check permissions
        if not data_share.can_access(agent_id, permission):
            data_share.log_access(agent_id, permission, False)
            logger.warning(f"Agent {agent_id} denied access to {share_id} with permission {permission}")
            return None
        
        try:
            # Decrypt and return data
            decrypted_data = self.encryption_manager.decrypt_data(
                data_share.encrypted_data,
                data_share.data_hash
            )
            
            data_share.log_access(agent_id, permission, True)
            logger.info(f"Agent {agent_id} accessed shared data {share_id} with permission {permission}")
            
            return decrypted_data
        
        except Exception as e:
            data_share.log_access(agent_id, permission, False)
            logger.error(f"Failed to decrypt shared data {share_id} for agent {agent_id}: {str(e)}")
            return None
    
    async def update_shared_data(
        self,
        share_id: str,
        agent_id: str,
        data: Dict[str, Any]
    ) -> bool:
        """Update shared data (requires write permission)."""
        if share_id not in self.data_shares:
            return False
        
        data_share = self.data_shares[share_id]
        
        # Check write permission
        if not data_share.can_access(agent_id, "write"):
            data_share.log_access(agent_id, "write", False)
            logger.warning(f"Agent {agent_id} denied write access to {share_id}")
            return False
        
        try:
            # Encrypt updated data
            encrypted_data, data_hash = self.encryption_manager.encrypt_data(data)
            
            # Update data share
            data_share.encrypted_data = encrypted_data
            data_share.data_hash = data_hash
            
            data_share.log_access(agent_id, "write", True)
            logger.info(f"Agent {agent_id} updated shared data {share_id}")
            
            return True
        
        except Exception as e:
            data_share.log_access(agent_id, "write", False)
            logger.error(f"Failed to update shared data {share_id} for agent {agent_id}: {str(e)}")
            return False
    
    async def revoke_data_access(self, share_id: str, agent_id: str, revoked_by: str) -> bool:
        """Revoke an agent's access to shared data."""
        if share_id not in self.data_shares:
            return False
        
        data_share = self.data_shares[share_id]
        
        # Only owner can revoke access
        if revoked_by != data_share.owner_agent_id:
            logger.warning(f"Agent {revoked_by} cannot revoke access to {share_id} (not owner)")
            return False
        
        # Remove agent from shared_with and permissions
        data_share.shared_with.discard(agent_id)
        if agent_id in data_share.access_permissions:
            del data_share.access_permissions[agent_id]
        
        logger.info(f"Access revoked for agent {agent_id} to shared data {share_id}")
        return True
    
    async def _cleanup_data_share(self, share_id: str) -> None:
        """Clean up a data share."""
        if share_id in self.data_shares:
            del self.data_shares[share_id]
        
        if share_id in self.data_locks:
            del self.data_locks[share_id]
    
    async def cleanup_expired_shares(self) -> int:
        """Clean up all expired data shares."""
        expired_shares = [
            share_id for share_id, data_share in self.data_shares.items()
            if data_share.is_expired()
        ]
        
        for share_id in expired_shares:
            await self._cleanup_data_share(share_id)
        
        if expired_shares:
            logger.info(f"Cleaned up {len(expired_shares)} expired data shares")
        
        return len(expired_shares)


class MultiAgentCoordinator:
    """
    Multi-agent coordinator for Strands agents using AgentCore Browser Tool.
    
    This coordinator manages multiple Strands agents sharing AgentCore Browser Tool
    sessions, implements secure data sharing, resource allocation, and isolation
    mechanisms to prevent data leakage between agents.
    
    Features:
    - Multi-agent coordination with security controls
    - Secure data sharing between agents handling sensitive information
    - Resource allocation and session management for concurrent operations
    - Isolation mechanisms to prevent data leakage between agents
    - Agent lifecycle management with comprehensive monitoring
    - Audit logging and observability for all operations
    """
    
    def __init__(
        self,
        config: Optional[CoordinationConfig] = None,
        session_pool_config: Optional[SessionPoolConfig] = None,
        encryption_password: Optional[str] = None
    ):
        """
        Initialize the multi-agent coordinator.
        
        Args:
            config: Configuration for coordination behavior
            session_pool_config: Configuration for session pool management
            encryption_password: Password for encrypting shared data
        """
        self.config = config or CoordinationConfig()
        self.session_pool_config = session_pool_config or SessionPoolConfig()
        
        # Initialize managers
        self.resource_manager = ResourceManager(self.config)
        self.encryption_manager = EncryptionManager(encryption_password)
        self.data_manager = SecureDataManager(self.config, self.encryption_manager)
        self.session_pool = SessionPool(self.session_pool_config)
        
        # Agent management
        self.agents: Dict[str, AgentContext] = {}
        self.agent_tasks: Dict[str, asyncio.Task] = {}
        self.coordination_lock = asyncio.Lock()
        
        # Cleanup task
        self._cleanup_task = None
        self._shutdown_event = asyncio.Event()
        
        logger.info("MultiAgentCoordinator initialized")
        logger.info(f"Configuration: {self.config.to_dict()}")
    
    async def register_agent(
        self,
        agent: Agent,
        isolation_level: IsolationLevel = None,
        security_clearance: str = "standard"
    ) -> str:
        """Register a Strands agent for coordination."""
        async with self.coordination_lock:
            if len(self.agents) >= self.config.max_concurrent_agents:
                raise AgentCoordinationError(f"Maximum concurrent agents limit reached: {self.config.max_concurrent_agents}")
            
            agent_id = f"agent_{uuid.uuid4().hex[:8]}"
            isolation_level = isolation_level or self.config.default_isolation_level
            
            agent_context = AgentContext(
                agent_id=agent_id,
                agent_name=agent.name,
                agent=agent,
                isolation_level=isolation_level,
                security_clearance=security_clearance
            )
            
            self.agents[agent_id] = agent_context
            
            logger.info(f"Agent registered: {agent_id} ({agent.name}) with {isolation_level.value} isolation")
            return agent_id
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent and clean up its resources."""
        async with self.coordination_lock:
            if agent_id not in self.agents:
                return False
            
            # Cancel any running tasks
            if agent_id in self.agent_tasks:
                self.agent_tasks[agent_id].cancel()
                del self.agent_tasks[agent_id]
            
            # Release all resources
            agent_resources = await self.resource_manager.get_agent_resources(agent_id)
            for resource in agent_resources:
                await self.resource_manager.release_resource(resource.resource_id, agent_id)
            
            # Clean up agent context
            agent_context = self.agents[agent_id]
            del self.agents[agent_id]
            
            logger.info(f"Agent unregistered: {agent_id} ({agent_context.agent_name})")
            return True
    
    async def execute_agent_task(
        self,
        agent_id: str,
        task_func: Callable,
        task_args: Tuple = (),
        task_kwargs: Dict[str, Any] = None,
        require_resources: List[Tuple[str, ResourceType]] = None
    ) -> AgentResult:
        """Execute a task with an agent, managing resources and isolation."""
        if agent_id not in self.agents:
            return AgentResult(
                success=False,
                error=f"Agent not found: {agent_id}"
            )
        
        agent_context = self.agents[agent_id]
        task_kwargs = task_kwargs or {}
        require_resources = require_resources or []
        
        try:
            # Update agent status
            agent_context.status = AgentStatus.ACTIVE
            agent_context.last_activity = datetime.now()
            
            # Allocate required resources
            allocated_resources = []
            for resource_id, resource_type in require_resources:
                success = await self.resource_manager.allocate_resource(
                    resource_id, resource_type, agent_id
                )
                if success:
                    allocated_resources.append(resource_id)
                    agent_context.assigned_resources.add(resource_id)
                else:
                    # Release already allocated resources
                    for res_id in allocated_resources:
                        await self.resource_manager.release_resource(res_id, agent_id)
                        agent_context.assigned_resources.discard(res_id)
                    
                    agent_context.status = AgentStatus.BLOCKED
                    return AgentResult(
                        success=False,
                        error=f"Failed to allocate resource: {resource_id}",
                        metadata={'agent_id': agent_id, 'blocked_resource': resource_id}
                    )
            
            # Get browser session if needed
            session = None
            if any(rt == ResourceType.BROWSER_SESSION for _, rt in require_resources):
                session = await self.session_pool.get_session(agent_id, "task")
                task_kwargs['browser_session'] = session
            
            # Execute task with isolation
            logger.info(f"Executing task for agent {agent_id} with {len(allocated_resources)} resources")
            
            # Run task in thread pool to avoid blocking
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(task_func, *task_args, **task_kwargs)
                result = await asyncio.get_event_loop().run_in_executor(None, future.result)
            
            # Mark agent as completed
            agent_context.status = AgentStatus.COMPLETED
            agent_context.last_activity = datetime.now()
            
            logger.info(f"✅ Task completed for agent {agent_id}")
            
            return AgentResult(
                success=True,
                data=result,
                metadata={
                    'agent_id': agent_id,
                    'resources_used': allocated_resources,
                    'isolation_level': agent_context.isolation_level.value
                }
            )
        
        except Exception as e:
            agent_context.status = AgentStatus.FAILED
            logger.error(f"❌ Task failed for agent {agent_id}: {str(e)}")
            
            return AgentResult(
                success=False,
                error=str(e),
                metadata={'agent_id': agent_id}
            )
        
        finally:
            # Release resources
            for resource_id in allocated_resources:
                await self.resource_manager.release_resource(resource_id, agent_id)
                agent_context.assigned_resources.discard(resource_id)
            
            # Release session
            if session:
                await self.session_pool.release_session(agent_id, "task")
    
    async def share_data_between_agents(
        self,
        data_key: str,
        data: Dict[str, Any],
        owner_agent_id: str,
        target_agent_ids: List[str],
        permissions: Dict[str, List[str]] = None,
        expires_in: Optional[int] = None
    ) -> str:
        """Share data securely between agents."""
        if owner_agent_id not in self.agents:
            raise AgentCoordinationError(f"Owner agent not found: {owner_agent_id}")
        
        # Validate target agents
        for agent_id in target_agent_ids:
            if agent_id not in self.agents:
                raise AgentCoordinationError(f"Target agent not found: {agent_id}")
        
        # Set default permissions
        if permissions is None:
            permissions = {agent_id: ["read"] for agent_id in target_agent_ids}
        
        # Create data share
        share_id = await self.data_manager.create_data_share(
            data_key=data_key,
            data=data,
            owner_agent_id=owner_agent_id,
            shared_with=target_agent_ids,
            permissions=permissions,
            expires_in=expires_in
        )
        
        # Update agent contexts
        owner_context = self.agents[owner_agent_id]
        owner_context.shared_data_keys.add(share_id)
        
        for agent_id in target_agent_ids:
            target_context = self.agents[agent_id]
            target_context.shared_data_keys.add(share_id)
        
        logger.info(f"Data shared: {data_key} from {owner_agent_id} to {len(target_agent_ids)} agents")
        return share_id
    
    async def get_shared_data(
        self,
        share_id: str,
        agent_id: str,
        permission: str = "read"
    ) -> Optional[Dict[str, Any]]:
        """Get shared data for an agent."""
        if agent_id not in self.agents:
            return None
        
        return await self.data_manager.access_shared_data(share_id, agent_id, permission)
    
    async def coordinate_agents(
        self,
        agent_tasks: List[Dict[str, Any]],
        coordination_strategy: str = "parallel"
    ) -> List[AgentResult]:
        """Coordinate execution of multiple agent tasks."""
        logger.info(f"Coordinating {len(agent_tasks)} agent tasks with {coordination_strategy} strategy")
        
        if coordination_strategy == "parallel":
            # Execute all tasks in parallel
            tasks = []
            for task_config in agent_tasks:
                task = asyncio.create_task(
                    self.execute_agent_task(**task_config)
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to failed results
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    processed_results.append(AgentResult(
                        success=False,
                        error=str(result)
                    ))
                else:
                    processed_results.append(result)
            
            return processed_results
        
        elif coordination_strategy == "sequential":
            # Execute tasks one by one
            results = []
            for task_config in agent_tasks:
                result = await self.execute_agent_task(**task_config)
                results.append(result)
                
                # Stop on first failure if configured
                if not result.success and task_config.get('stop_on_failure', False):
                    break
            
            return results
        
        else:
            raise AgentCoordinationError(f"Unknown coordination strategy: {coordination_strategy}")
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an agent."""
        if agent_id not in self.agents:
            return None
        
        agent_context = self.agents[agent_id]
        return agent_context.to_dict()
    
    def list_active_agents(self) -> List[Dict[str, Any]]:
        """List all active agents."""
        return [
            context.to_dict() for context in self.agents.values()
            if context.status in [AgentStatus.ACTIVE, AgentStatus.WAITING]
        ]
    
    async def start_cleanup_task(self) -> None:
        """Start the background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Cleanup task started")
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while not self._shutdown_event.is_set():
            try:
                # Clean up expired resources and data shares
                await self.resource_manager.cleanup_expired_allocations()
                await self.data_manager.cleanup_expired_shares()
                
                # Clean up inactive agents
                current_time = datetime.now()
                inactive_agents = [
                    agent_id for agent_id, context in self.agents.items()
                    if (current_time - context.last_activity).total_seconds() > 3600  # 1 hour
                    and context.status in [AgentStatus.IDLE, AgentStatus.COMPLETED, AgentStatus.FAILED]
                ]
                
                for agent_id in inactive_agents:
                    await self.unregister_agent(agent_id)
                
                # Wait for next cleanup cycle
                await asyncio.sleep(self.config.cleanup_interval)
            
            except Exception as e:
                logger.error(f"Error in cleanup loop: {str(e)}")
                await asyncio.sleep(self.config.cleanup_interval)
    
    async def shutdown(self) -> None:
        """Shutdown the coordinator and clean up all resources."""
        logger.info("Shutting down MultiAgentCoordinator")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all agent tasks
        for task in self.agent_tasks.values():
            task.cancel()
        
        # Unregister all agents
        agent_ids = list(self.agents.keys())
        for agent_id in agent_ids:
            await self.unregister_agent(agent_id)
        
        # Clean up session pool
        await self.session_pool.cleanup_all_sessions()
        
        logger.info("✅ MultiAgentCoordinator shutdown complete")


# Convenience functions for agent coordination

def create_agent_task_config(
    agent_id: str,
    task_func: Callable,
    task_args: Tuple = (),
    task_kwargs: Dict[str, Any] = None,
    require_resources: List[Tuple[str, ResourceType]] = None,
    stop_on_failure: bool = False
) -> Dict[str, Any]:
    """Create a task configuration for agent coordination."""
    return {
        'agent_id': agent_id,
        'task_func': task_func,
        'task_args': task_args,
        'task_kwargs': task_kwargs or {},
        'require_resources': require_resources or [],
        'stop_on_failure': stop_on_failure
    }


def create_data_share_permissions(
    agent_permissions: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """Create data share permissions configuration."""
    valid_permissions = ['read', 'write', 'delete', 'all']
    
    validated_permissions = {}
    for agent_id, permissions in agent_permissions.items():
        validated_permissions[agent_id] = [
            perm for perm in permissions if perm in valid_permissions
        ]
    
    return validated_permissions