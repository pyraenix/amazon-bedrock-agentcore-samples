# Performance Optimization and Best Practices Guide

## Overview

This guide provides comprehensive strategies for optimizing the performance of your LlamaIndex AgentCore Browser Integration applications. It covers configuration optimization, resource management, scaling strategies, and monitoring best practices.

## Table of Contents

1. [Configuration Optimization](#configuration-optimization)
2. [Resource Management](#resource-management)
3. [Concurrency and Scaling](#concurrency-and-scaling)
4. [Caching Strategies](#caching-strategies)
5. [Error Handling and Resilience](#error-handling-and-resilience)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Cost Optimization](#cost-optimization)
8. [Security Best Practices](#security-best-practices)
9. [Production Deployment](#production-deployment)
10. [Troubleshooting Performance Issues](#troubleshooting-performance-issues)

## Configuration Optimization

### Browser Configuration

Optimize browser settings for better performance:

```yaml
# config.yaml - Performance-optimized configuration
browser:
  # Disable unnecessary features for faster loading
  headless: true
  enable_images: false      # Disable images unless needed
  enable_css: true          # Keep CSS for layout
  enable_javascript: true   # Usually needed for modern sites
  enable_plugins: false     # Disable browser plugins
  enable_flash: false       # Disable Flash
  
  # Optimize viewport for faster rendering
  viewport_width: 1280      # Smaller than default for faster rendering
  viewport_height: 720
  
  # Aggressive timeout settings
  page_load_timeout: 15     # Reduced from default 30s
  element_wait_timeout: 5   # Reduced from default 10s
  script_timeout: 10        # Timeout for JavaScript execution
  
  # Network optimization
  user_agent: "FastScraper/1.0"
  accept_language: "en-US"  # Single language for consistency
  
  # Memory optimization
  max_memory_usage: 512     # Limit memory usage (MB)
  clear_cache_frequency: 100 # Clear cache every N operations
```

### LlamaIndex Configuration

Optimize LLM settings for performance:

```yaml
llamaindex:
  # Use faster models for simple tasks
  llm_model: "anthropic.claude-3-haiku-20240307-v1:0"  # Faster than Sonnet
  vision_model: "anthropic.claude-3-haiku-20240307-v1:0"
  
  # Optimize token usage
  max_tokens: 2048          # Reduce for faster responses
  temperature: 0.0          # Deterministic responses
  
  # Batch processing settings
  batch_size: 5             # Process multiple requests together
  request_timeout: 30       # Timeout for LLM requests
```

### AWS Configuration

Optimize AWS settings:

```yaml
aws:
  region: "us-east-1"       # Use closest region
  
agentcore:
  # Session management
  session_timeout: 300      # 5 minutes - balance between reuse and resource usage
  max_concurrent_sessions: 10
  session_pool_size: 5      # Pre-create sessions for faster startup
  
  # Request optimization
  request_timeout: 45       # Timeout for AgentCore requests
  retry_attempts: 2         # Reduce retries for faster failure handling
  retry_delay: 1.0          # Shorter delay between retries
```

## Resource Management

### Memory Management

Implement efficient memory usage patterns:

```python
"""
Memory-efficient browser client implementation.
"""
import gc
import psutil
from typing import Optional
from contextlib import asynccontextmanager

class MemoryOptimizedBrowserClient:
    """Browser client with memory optimization."""
    
    def __init__(self, memory_limit_mb: int = 1024):
        self.memory_limit = memory_limit_mb * 1024 * 1024  # Convert to bytes
        self.operation_count = 0
        self.cleanup_frequency = 50  # Cleanup every N operations
    
    async def perform_operation(self, operation_func, *args, **kwargs):
        """Perform operation with memory monitoring."""
        # Check memory usage before operation
        if self._should_cleanup():
            await self._cleanup_resources()
        
        try:
            result = await operation_func(*args, **kwargs)
            self.operation_count += 1
            return result
        finally:
            # Force garbage collection periodically
            if self.operation_count % 10 == 0:
                gc.collect()
    
    def _should_cleanup(self) -> bool:
        """Check if cleanup is needed."""
        process = psutil.Process()
        memory_usage = process.memory_info().rss
        
        return (
            memory_usage > self.memory_limit or
            self.operation_count % self.cleanup_frequency == 0
        )
    
    async def _cleanup_resources(self):
        """Clean up resources to free memory."""
        # Clear browser cache
        await self.clear_cache()
        
        # Clear cookies and local storage
        await self.clear_cookies()
        await self.clear_local_storage()
        
        # Force garbage collection
        gc.collect()
        
        print(f"Memory cleanup performed after {self.operation_count} operations")

# Context manager for automatic resource cleanup
@asynccontextmanager
async def managed_browser_session(client):
    """Context manager for automatic session cleanup."""
    session_id = None
    try:
        session_id = await client.create_session()
        yield client
    finally:
        if session_id:
            await client.close_session()
            gc.collect()
```

### Session Pooling

Implement session pooling for better resource utilization:

```python
"""
Session pool implementation for better resource management.
"""
import asyncio
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class PooledSession:
    """Represents a pooled browser session."""
    session_id: str
    client: 'AgentCoreBrowserClient'
    created_at: datetime
    last_used: datetime
    operation_count: int = 0
    
    @property
    def age(self) -> timedelta:
        return datetime.now() - self.created_at
    
    @property
    def idle_time(self) -> timedelta:
        return datetime.now() - self.last_used

class BrowserSessionPool:
    """Efficient browser session pool."""
    
    def __init__(self, 
                 min_size: int = 2,
                 max_size: int = 10,
                 max_age_minutes: int = 30,
                 max_idle_minutes: int = 5):
        self.min_size = min_size
        self.max_size = max_size
        self.max_age = timedelta(minutes=max_age_minutes)
        self.max_idle = timedelta(minutes=max_idle_minutes)
        
        self.available_sessions: List[PooledSession] = []
        self.in_use_sessions: List[PooledSession] = []
        self._lock = asyncio.Lock()
    
    async def get_session(self) -> PooledSession:
        """Get a session from the pool."""
        async with self._lock:
            # Clean up expired sessions first
            await self._cleanup_expired_sessions()
            
            # Try to get an available session
            if self.available_sessions:
                session = self.available_sessions.pop(0)
                session.last_used = datetime.now()
                self.in_use_sessions.append(session)
                return session
            
            # Create new session if under limit
            if len(self.in_use_sessions) < self.max_size:
                session = await self._create_new_session()
                self.in_use_sessions.append(session)
                return session
            
            # Wait for a session to become available
            while not self.available_sessions:
                await asyncio.sleep(0.1)
            
            session = self.available_sessions.pop(0)
            session.last_used = datetime.now()
            self.in_use_sessions.append(session)
            return session
    
    async def return_session(self, session: PooledSession):
        """Return a session to the pool."""
        async with self._lock:
            if session in self.in_use_sessions:
                self.in_use_sessions.remove(session)
                
                # Check if session is still valid
                if (session.age < self.max_age and 
                    session.operation_count < 1000):  # Limit operations per session
                    self.available_sessions.append(session)
                else:
                    await self._close_session(session)
    
    async def _create_new_session(self) -> PooledSession:
        """Create a new pooled session."""
        from client import AgentCoreBrowserClient
        
        client = AgentCoreBrowserClient()
        session_id = await client.create_session()
        
        return PooledSession(
            session_id=session_id,
            client=client,
            created_at=datetime.now(),
            last_used=datetime.now()
        )
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        current_time = datetime.now()
        
        # Clean up idle available sessions
        self.available_sessions = [
            s for s in self.available_sessions
            if current_time - s.last_used < self.max_idle
        ]
        
        # Clean up old sessions
        expired_sessions = [
            s for s in self.available_sessions
            if s.age > self.max_age
        ]
        
        for session in expired_sessions:
            await self._close_session(session)
            self.available_sessions.remove(session)
    
    async def _close_session(self, session: PooledSession):
        """Close a session and clean up resources."""
        try:
            await session.client.close_session()
        except Exception as e:
            print(f"Error closing session {session.session_id}: {e}")
    
    async def close_all(self):
        """Close all sessions in the pool."""
        all_sessions = self.available_sessions + self.in_use_sessions
        
        for session in all_sessions:
            await self._close_session(session)
        
        self.available_sessions.clear()
        self.in_use_sessions.clear()

# Usage example
async def example_with_session_pool():
    pool = BrowserSessionPool(min_size=3, max_size=10)
    
    try:
        # Get session from pool
        session = await pool.get_session()
        
        # Use session for operations
        await session.client.navigate("https://example.com")
        result = await session.client.extract_text()
        
        # Return session to pool
        await pool.return_session(session)
        
    finally:
        await pool.close_all()
```

## Concurrency and Scaling

### Optimal Concurrency Patterns

Implement efficient concurrent processing:

```python
"""
Optimized concurrent processing patterns.
"""
import asyncio
import time
from typing import List, Callable, Any
from dataclasses import dataclass

@dataclass
class ProcessingMetrics:
    """Metrics for processing performance."""
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    total_time: float
    average_time_per_task: float
    throughput: float  # tasks per second

class ConcurrentProcessor:
    """Optimized concurrent task processor."""
    
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.metrics = []
    
    async def process_batch(self, 
                          tasks: List[Any],
                          processor_func: Callable,
                          batch_size: int = None) -> ProcessingMetrics:
        """Process tasks in optimized batches."""
        if batch_size is None:
            batch_size = min(len(tasks), self.max_concurrent * 2)
        
        start_time = time.time()
        all_results = []
        
        # Process in batches to avoid overwhelming the system
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await self._process_batch_concurrent(batch, processor_func)
            all_results.extend(batch_results)
            
            # Small delay between batches to prevent rate limiting
            if i + batch_size < len(tasks):
                await asyncio.sleep(0.1)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        successful = sum(1 for r in all_results if r.get('success', False))
        failed = len(all_results) - successful
        
        metrics = ProcessingMetrics(
            total_tasks=len(tasks),
            successful_tasks=successful,
            failed_tasks=failed,
            total_time=total_time,
            average_time_per_task=total_time / len(tasks) if tasks else 0,
            throughput=len(tasks) / total_time if total_time > 0 else 0
        )
        
        self.metrics.append(metrics)
        return metrics
    
    async def _process_batch_concurrent(self, 
                                      batch: List[Any],
                                      processor_func: Callable) -> List[Any]:
        """Process a batch of tasks concurrently."""
        async def bounded_process(task):
            async with self.semaphore:
                try:
                    return await processor_func(task)
                except Exception as e:
                    return {"success": False, "error": str(e), "task": task}
        
        return await asyncio.gather(
            *[bounded_process(task) for task in batch],
            return_exceptions=True
        )
    
    def get_performance_summary(self) -> dict:
        """Get performance summary across all batches."""
        if not self.metrics:
            return {}
        
        total_tasks = sum(m.total_tasks for m in self.metrics)
        total_successful = sum(m.successful_tasks for m in self.metrics)
        total_time = sum(m.total_time for m in self.metrics)
        avg_throughput = sum(m.throughput for m in self.metrics) / len(self.metrics)
        
        return {
            "total_tasks_processed": total_tasks,
            "overall_success_rate": total_successful / total_tasks if total_tasks > 0 else 0,
            "total_processing_time": total_time,
            "average_throughput": avg_throughput,
            "batches_processed": len(self.metrics)
        }

# Adaptive concurrency based on performance
class AdaptiveConcurrentProcessor(ConcurrentProcessor):
    """Processor that adapts concurrency based on performance."""
    
    def __init__(self, initial_concurrent: int = 3, max_concurrent: int = 20):
        super().__init__(initial_concurrent)
        self.initial_concurrent = initial_concurrent
        self.max_concurrent_limit = max_concurrent
        self.performance_history = []
    
    async def process_with_adaptation(self, 
                                    tasks: List[Any],
                                    processor_func: Callable) -> ProcessingMetrics:
        """Process tasks with adaptive concurrency."""
        current_concurrent = self.initial_concurrent
        best_throughput = 0
        best_concurrent = current_concurrent
        
        # Test different concurrency levels with small batches
        test_batch_size = min(20, len(tasks) // 4)
        test_tasks = tasks[:test_batch_size]
        
        for concurrent_level in [3, 5, 8, 12, 16]:
            if concurrent_level > self.max_concurrent_limit:
                break
            
            self.max_concurrent = concurrent_level
            self.semaphore = asyncio.Semaphore(concurrent_level)
            
            metrics = await self.process_batch(test_tasks, processor_func, test_batch_size)
            
            if metrics.throughput > best_throughput:
                best_throughput = metrics.throughput
                best_concurrent = concurrent_level
            
            print(f"Concurrency {concurrent_level}: {metrics.throughput:.2f} tasks/sec")
        
        # Use optimal concurrency for remaining tasks
        self.max_concurrent = best_concurrent
        self.semaphore = asyncio.Semaphore(best_concurrent)
        
        remaining_tasks = tasks[test_batch_size:]
        if remaining_tasks:
            final_metrics = await self.process_batch(remaining_tasks, processor_func)
            print(f"Using optimal concurrency {best_concurrent}: {final_metrics.throughput:.2f} tasks/sec")
            return final_metrics
        
        return metrics
```

## Caching Strategies

### Content Caching

Implement intelligent content caching:

```python
"""
Intelligent caching system for web content.
"""
import hashlib
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class CacheEntry:
    """Represents a cached content entry."""
    key: str
    content: str
    metadata: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_minutes: int
    
    @property
    def is_expired(self) -> bool:
        return datetime.now() - self.created_at > timedelta(minutes=self.ttl_minutes)
    
    @property
    def age_minutes(self) -> float:
        return (datetime.now() - self.created_at).total_seconds() / 60

class IntelligentCache:
    """Intelligent caching system with TTL and LRU eviction."""
    
    def __init__(self, db_path: str = "content_cache.db", max_entries: int = 10000):
        self.db_path = db_path
        self.max_entries = max_entries
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for caching."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                content TEXT,
                metadata TEXT,
                created_at TEXT,
                last_accessed TEXT,
                access_count INTEGER,
                ttl_minutes INTEGER
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed)")
        conn.commit()
        conn.close()
    
    def _generate_key(self, url: str, operation: str, params: Dict[str, Any] = None) -> str:
        """Generate cache key from URL and operation parameters."""
        key_data = {
            "url": url,
            "operation": operation,
            "params": params or {}
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, url: str, operation: str, params: Dict[str, Any] = None) -> Optional[CacheEntry]:
        """Get cached content if available and not expired."""
        key = self._generate_key(url, operation, params)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT * FROM cache_entries WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        entry = CacheEntry(
            key=row[0],
            content=row[1],
            metadata=json.loads(row[2]),
            created_at=datetime.fromisoformat(row[3]),
            last_accessed=datetime.fromisoformat(row[4]),
            access_count=row[5],
            ttl_minutes=row[6]
        )
        
        if entry.is_expired:
            self._delete_entry(key)
            return None
        
        # Update access statistics
        self._update_access(key)
        return entry
    
    def set(self, url: str, operation: str, content: str, 
            metadata: Dict[str, Any] = None, ttl_minutes: int = 60):
        """Cache content with specified TTL."""
        key = self._generate_key(url, operation, metadata)
        now = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        
        # Insert or replace cache entry
        conn.execute("""
            INSERT OR REPLACE INTO cache_entries 
            (key, content, metadata, created_at, last_accessed, access_count, ttl_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            key,
            content,
            json.dumps(metadata or {}),
            now.isoformat(),
            now.isoformat(),
            1,
            ttl_minutes
        ))
        
        conn.commit()
        conn.close()
        
        # Cleanup old entries if cache is full
        self._cleanup_if_needed()
    
    def _update_access(self, key: str):
        """Update access statistics for cache entry."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE cache_entries 
            SET last_accessed = ?, access_count = access_count + 1
            WHERE key = ?
        """, (datetime.now().isoformat(), key))
        conn.commit()
        conn.close()
    
    def _delete_entry(self, key: str):
        """Delete cache entry."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
        conn.commit()
        conn.close()
    
    def _cleanup_if_needed(self):
        """Clean up cache if it exceeds maximum entries."""
        conn = sqlite3.connect(self.db_path)
        
        # Count current entries
        cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
        count = cursor.fetchone()[0]
        
        if count > self.max_entries:
            # Remove oldest accessed entries (LRU)
            entries_to_remove = count - self.max_entries + 100  # Remove extra for buffer
            conn.execute("""
                DELETE FROM cache_entries WHERE key IN (
                    SELECT key FROM cache_entries 
                    ORDER BY last_accessed ASC 
                    LIMIT ?
                )
            """, (entries_to_remove,))
        
        # Remove expired entries
        expired_cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
        conn.execute("""
            DELETE FROM cache_entries 
            WHERE datetime(created_at) < datetime(?)
        """, (expired_cutoff,))
        
        conn.commit()
        conn.close()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        conn = sqlite3.connect(self.db_path)
        
        # Basic stats
        cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
        total_entries = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT AVG(access_count) FROM cache_entries")
        avg_access_count = cursor.fetchone()[0] or 0
        
        cursor = conn.execute("""
            SELECT AVG((julianday('now') - julianday(created_at)) * 24 * 60) 
            FROM cache_entries
        """)
        avg_age_minutes = cursor.fetchone()[0] or 0
        
        # Hit rate calculation (would need separate tracking)
        cursor = conn.execute("SELECT SUM(access_count) FROM cache_entries")
        total_accesses = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "total_entries": total_entries,
            "average_access_count": round(avg_access_count, 2),
            "average_age_minutes": round(avg_age_minutes, 2),
            "total_accesses": total_accesses,
            "cache_utilization": min(total_entries / self.max_entries, 1.0)
        }

# Usage with browser client
class CachedBrowserClient:
    """Browser client with intelligent caching."""
    
    def __init__(self, cache: IntelligentCache):
        self.cache = cache
        self.cache_hits = 0
        self.cache_misses = 0
    
    async def cached_extract_text(self, url: str, selector: str = None) -> str:
        """Extract text with caching."""
        # Check cache first
        cache_entry = self.cache.get(url, "extract_text", {"selector": selector})
        
        if cache_entry:
            self.cache_hits += 1
            print(f"Cache hit for {url} (age: {cache_entry.age_minutes:.1f}m)")
            return cache_entry.content
        
        # Cache miss - perform actual extraction
        self.cache_misses += 1
        print(f"Cache miss for {url} - extracting...")
        
        # Simulate actual extraction (replace with real implementation)
        content = await self._actual_extract_text(url, selector)
        
        # Cache the result
        self.cache.set(
            url, 
            "extract_text", 
            content,
            {"selector": selector},
            ttl_minutes=30  # Cache for 30 minutes
        )
        
        return content
    
    async def _actual_extract_text(self, url: str, selector: str = None) -> str:
        """Actual text extraction implementation."""
        # This would be replaced with real AgentCore browser client call
        import time
        await asyncio.sleep(1)  # Simulate network delay
        return f"Extracted content from {url} with selector {selector}"
    
    def get_cache_performance(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }
```

## Error Handling and Resilience

### Circuit Breaker Pattern

Implement circuit breaker for resilient operations:

```python
"""
Circuit breaker pattern for resilient browser operations.
"""
import asyncio
import time
from enum import Enum
from typing import Callable, Any, Optional
from dataclasses import dataclass

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open" # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5      # Failures before opening
    recovery_timeout: int = 60      # Seconds before trying half-open
    success_threshold: int = 3      # Successes to close from half-open
    timeout: int = 30               # Operation timeout

class CircuitBreaker:
    """Circuit breaker for browser operations."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.operation_count = 0
    
    async def call(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation through circuit breaker."""
        self.operation_count += 1
        
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time < self.config.recovery_timeout:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")
            else:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
        
        try:
            # Execute operation with timeout
            result = await asyncio.wait_for(
                operation(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Handle success
            await self._on_success()
            return result
            
        except Exception as e:
            await self._on_failure()
            raise e
    
    async def _on_success(self):
        """Handle successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                print("Circuit breaker CLOSED - service recovered")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0  # Reset failure count on success
    
    async def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            print("Circuit breaker OPEN - service still failing")
        elif (self.state == CircuitState.CLOSED and 
              self.failure_count >= self.config.failure_threshold):
            self.state = CircuitState.OPEN
            print(f"Circuit breaker OPEN - {self.failure_count} failures")
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "operation_count": self.operation_count,
            "last_failure_time": self.last_failure_time
        }

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass

# Resilient browser client with circuit breaker
class ResilientBrowserClient:
    """Browser client with circuit breaker and retry logic."""
    
    def __init__(self):
        self.circuit_breaker = CircuitBreaker(CircuitBreakerConfig())
        self.retry_config = {
            "max_attempts": 3,
            "base_delay": 1.0,
            "max_delay": 10.0,
            "exponential_base": 2.0
        }
    
    async def resilient_navigate(self, url: str) -> dict:
        """Navigate with circuit breaker and retry logic."""
        return await self.circuit_breaker.call(self._navigate_with_retry, url)
    
    async def _navigate_with_retry(self, url: str) -> dict:
        """Navigate with exponential backoff retry."""
        last_exception = None
        
        for attempt in range(self.retry_config["max_attempts"]):
            try:
                # Simulate navigation (replace with actual implementation)
                result = await self._actual_navigate(url)
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.retry_config["max_attempts"] - 1:
                    delay = min(
                        self.retry_config["base_delay"] * 
                        (self.retry_config["exponential_base"] ** attempt),
                        self.retry_config["max_delay"]
                    )
                    
                    print(f"Navigation attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    print(f"All navigation attempts failed: {e}")
        
        raise last_exception
    
    async def _actual_navigate(self, url: str) -> dict:
        """Actual navigation implementation."""
        # Simulate potential failure
        import random
        if random.random() < 0.3:  # 30% failure rate for demo
            raise Exception(f"Navigation failed for {url}")
        
        return {"success": True, "url": url, "title": f"Page: {url}"}
```

## Monitoring and Observability

### Performance Monitoring

Implement comprehensive performance monitoring:

```python
"""
Performance monitoring and metrics collection.
"""
import time
import asyncio
from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class OperationMetrics:
    """Metrics for a single operation."""
    operation_name: str
    start_time: float
    end_time: float
    success: bool
    error_message: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def timestamp(self) -> str:
        return datetime.fromtimestamp(self.start_time).isoformat()

class PerformanceMonitor:
    """Comprehensive performance monitoring system."""
    
    def __init__(self):
        self.metrics: List[OperationMetrics] = []
        self.active_operations: Dict[str, float] = {}
        self.operation_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
    
    def start_operation(self, operation_name: str, operation_id: str = None) -> str:
        """Start tracking an operation."""
        if operation_id is None:
            operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        
        self.active_operations[operation_id] = time.time()
        self.operation_counts[operation_name] = self.operation_counts.get(operation_name, 0) + 1
        
        return operation_id
    
    def end_operation(self, operation_id: str, operation_name: str, 
                     success: bool = True, error_message: str = None,
                     metadata: Dict[str, Any] = None):
        """End tracking an operation."""
        if operation_id not in self.active_operations:
            return
        
        start_time = self.active_operations.pop(operation_id)
        end_time = time.time()
        
        if not success:
            self.error_counts[operation_name] = self.error_counts.get(operation_name, 0) + 1
        
        metrics = OperationMetrics(
            operation_name=operation_name,
            start_time=start_time,
            end_time=end_time,
            success=success,
            error_message=error_message,
            metadata=metadata or {}
        )
        
        self.metrics.append(metrics)
    
    def get_performance_summary(self, operation_name: str = None) -> Dict[str, Any]:
        """Get performance summary for operations."""
        if operation_name:
            relevant_metrics = [m for m in self.metrics if m.operation_name == operation_name]
        else:
            relevant_metrics = self.metrics
        
        if not relevant_metrics:
            return {"error": "No metrics available"}
        
        durations = [m.duration for m in relevant_metrics]
        successful_ops = [m for m in relevant_metrics if m.success]
        failed_ops = [m for m in relevant_metrics if not m.success]
        
        return {
            "operation_name": operation_name or "all_operations",
            "total_operations": len(relevant_metrics),
            "successful_operations": len(successful_ops),
            "failed_operations": len(failed_ops),
            "success_rate": len(successful_ops) / len(relevant_metrics),
            "average_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "total_duration": sum(durations),
            "operations_per_second": len(relevant_metrics) / sum(durations) if sum(durations) > 0 else 0
        }
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """Get detailed error analysis."""
        error_details = {}
        
        for metric in self.metrics:
            if not metric.success and metric.error_message:
                error_type = type(metric.error_message).__name__ if hasattr(metric.error_message, '__class__') else "Unknown"
                
                if error_type not in error_details:
                    error_details[error_type] = {
                        "count": 0,
                        "operations": [],
                        "first_occurrence": metric.timestamp,
                        "last_occurrence": metric.timestamp
                    }
                
                error_details[error_type]["count"] += 1
                error_details[error_type]["operations"].append(metric.operation_name)
                error_details[error_type]["last_occurrence"] = metric.timestamp
        
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_rate": sum(self.error_counts.values()) / len(self.metrics) if self.metrics else 0,
            "error_breakdown": self.error_counts,
            "error_details": error_details
        }
    
    def export_metrics(self, filename: str = None) -> str:
        """Export metrics to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_metrics_{timestamp}.json"
        
        export_data = {
            "summary": self.get_performance_summary(),
            "error_analysis": self.get_error_analysis(),
            "detailed_metrics": [
                {
                    "operation_name": m.operation_name,
                    "timestamp": m.timestamp,
                    "duration": m.duration,
                    "success": m.success,
                    "error_message": m.error_message,
                    "metadata": m.metadata
                }
                for m in self.metrics
            ],
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filename

# Context manager for automatic operation tracking
class MonitoredOperation:
    """Context manager for automatic operation monitoring."""
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str, metadata: Dict[str, Any] = None):
        self.monitor = monitor
        self.operation_name = operation_name
        self.metadata = metadata or {}
        self.operation_id = None
    
    async def __aenter__(self):
        self.operation_id = self.monitor.start_operation(self.operation_name)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        error_message = str(exc_val) if exc_val else None
        
        self.monitor.end_operation(
            self.operation_id,
            self.operation_name,
            success=success,
            error_message=error_message,
            metadata=self.metadata
        )
        
        return False  # Don't suppress exceptions

# Usage example
async def monitored_browser_operations():
    """Example of using performance monitoring."""
    monitor = PerformanceMonitor()
    
    # Monitor individual operations
    async with MonitoredOperation(monitor, "page_navigation", {"url": "https://example.com"}):
        await asyncio.sleep(1)  # Simulate navigation
    
    async with MonitoredOperation(monitor, "text_extraction", {"selector": ".content"}):
        await asyncio.sleep(0.5)  # Simulate extraction
    
    # Get performance summary
    summary = monitor.get_performance_summary()
    print(f"Performance Summary: {json.dumps(summary, indent=2)}")
    
    # Export metrics
    filename = monitor.export_metrics()
    print(f"Metrics exported to: {filename}")
```

## Cost Optimization

### AWS Cost Management

Strategies for optimizing AWS costs:

```python
"""
Cost optimization strategies for AgentCore usage.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class CostMetrics:
    """Cost tracking metrics."""
    session_minutes: float
    llm_tokens_used: int
    vision_model_calls: int
    api_requests: int
    estimated_cost_usd: float

class CostOptimizer:
    """Cost optimization manager."""
    
    def __init__(self):
        # AWS pricing (approximate, check current pricing)
        self.pricing = {
            "session_per_minute": 0.01,      # AgentCore session cost
            "llm_token_per_1k": 0.003,       # Bedrock LLM tokens
            "vision_call": 0.01,             # Vision model call
            "api_request": 0.0001            # API request cost
        }
        
        self.cost_metrics = CostMetrics(0, 0, 0, 0, 0.0)
        self.session_start_times: Dict[str, datetime] = {}
    
    def start_session_tracking(self, session_id: str):
        """Start tracking session time for cost calculation."""
        self.session_start_times[session_id] = datetime.now()
    
    def end_session_tracking(self, session_id: str):
        """End session tracking and update costs."""
        if session_id in self.session_start_times:
            start_time = self.session_start_times.pop(session_id)
            duration_minutes = (datetime.now() - start_time).total_seconds() / 60
            
            self.cost_metrics.session_minutes += duration_minutes
            session_cost = duration_minutes * self.pricing["session_per_minute"]
            self.cost_metrics.estimated_cost_usd += session_cost
    
    def track_llm_usage(self, tokens_used: int):
        """Track LLM token usage."""
        self.cost_metrics.llm_tokens_used += tokens_used
        token_cost = (tokens_used / 1000) * self.pricing["llm_token_per_1k"]
        self.cost_metrics.estimated_cost_usd += token_cost
    
    def track_vision_usage(self, calls: int = 1):
        """Track vision model usage."""
        self.cost_metrics.vision_model_calls += calls
        vision_cost = calls * self.pricing["vision_call"]
        self.cost_metrics.estimated_cost_usd += vision_cost
    
    def track_api_request(self, requests: int = 1):
        """Track API request usage."""
        self.cost_metrics.api_requests += requests
        api_cost = requests * self.pricing["api_request"]
        self.cost_metrics.estimated_cost_usd += api_cost
    
    def get_cost_breakdown(self) -> Dict[str, Any]:
        """Get detailed cost breakdown."""
        session_cost = self.cost_metrics.session_minutes * self.pricing["session_per_minute"]
        llm_cost = (self.cost_metrics.llm_tokens_used / 1000) * self.pricing["llm_token_per_1k"]
        vision_cost = self.cost_metrics.vision_model_calls * self.pricing["vision_call"]
        api_cost = self.cost_metrics.api_requests * self.pricing["api_request"]
        
        return {
            "total_estimated_cost": self.cost_metrics.estimated_cost_usd,
            "breakdown": {
                "session_cost": session_cost,
                "llm_cost": llm_cost,
                "vision_cost": vision_cost,
                "api_cost": api_cost
            },
            "usage_metrics": {
                "session_minutes": self.cost_metrics.session_minutes,
                "llm_tokens": self.cost_metrics.llm_tokens_used,
                "vision_calls": self.cost_metrics.vision_model_calls,
                "api_requests": self.cost_metrics.api_requests
            },
            "cost_per_unit": {
                "cost_per_minute": session_cost / self.cost_metrics.session_minutes if self.cost_metrics.session_minutes > 0 else 0,
                "cost_per_token": llm_cost / self.cost_metrics.llm_tokens_used if self.cost_metrics.llm_tokens_used > 0 else 0,
                "cost_per_vision_call": self.pricing["vision_call"],
                "cost_per_api_request": self.pricing["api_request"]
            }
        }
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get cost optimization recommendations."""
        recommendations = []
        breakdown = self.get_cost_breakdown()
        
        # Session optimization
        if breakdown["breakdown"]["session_cost"] > breakdown["total_estimated_cost"] * 0.5:
            recommendations.append(
                "Session costs are high - consider shorter sessions and better session pooling"
            )
        
        # LLM optimization
        if breakdown["breakdown"]["llm_cost"] > breakdown["total_estimated_cost"] * 0.3:
            recommendations.append(
                "LLM costs are significant - consider using more efficient prompts and caching responses"
            )
        
        # Vision model optimization
        if breakdown["breakdown"]["vision_cost"] > breakdown["total_estimated_cost"] * 0.2:
            recommendations.append(
                "Vision model usage is high - cache vision analysis results and batch image processing"
            )
        
        # General recommendations
        if self.cost_metrics.session_minutes > 0:
            avg_session_length = self.cost_metrics.session_minutes / len(self.session_start_times) if self.session_start_times else self.cost_metrics.session_minutes
            if avg_session_length > 10:  # 10 minutes
                recommendations.append(
                    f"Average session length is {avg_session_length:.1f} minutes - consider breaking into shorter sessions"
                )
        
        return recommendations

# Cost-aware browser client
class CostAwareBrowserClient:
    """Browser client with cost tracking and optimization."""
    
    def __init__(self):
        self.cost_optimizer = CostOptimizer()
        self.current_session_id = None
        self.operation_count = 0
        self.cost_threshold_usd = 10.0  # Alert threshold
    
    async def create_session(self) -> str:
        """Create session with cost tracking."""
        session_id = f"session_{int(datetime.now().timestamp())}"
        self.current_session_id = session_id
        self.cost_optimizer.start_session_tracking(session_id)
        return session_id
    
    async def close_session(self):
        """Close session and update cost tracking."""
        if self.current_session_id:
            self.cost_optimizer.end_session_tracking(self.current_session_id)
            self.current_session_id = None
    
    async def navigate(self, url: str) -> dict:
        """Navigate with cost tracking."""
        self.cost_optimizer.track_api_request()
        
        # Simulate navigation
        await asyncio.sleep(0.5)
        
        # Track estimated token usage for LLM processing
        estimated_tokens = len(url) * 2  # Rough estimate
        self.cost_optimizer.track_llm_usage(estimated_tokens)
        
        self.operation_count += 1
        await self._check_cost_threshold()
        
        return {"success": True, "url": url}
    
    async def extract_text_with_vision(self, screenshot_data: bytes) -> str:
        """Extract text using vision model with cost tracking."""
        self.cost_optimizer.track_vision_usage()
        self.cost_optimizer.track_api_request()
        
        # Simulate vision processing
        await asyncio.sleep(1)
        
        # Estimate token usage for vision model response
        estimated_tokens = 500  # Vision models typically use more tokens
        self.cost_optimizer.track_llm_usage(estimated_tokens)
        
        await self._check_cost_threshold()
        
        return "Extracted text from image"
    
    async def _check_cost_threshold(self):
        """Check if cost threshold is exceeded."""
        current_cost = self.cost_optimizer.cost_metrics.estimated_cost_usd
        
        if current_cost > self.cost_threshold_usd:
            print(f"⚠️ Cost threshold exceeded: ${current_cost:.2f}")
            
            # Get optimization recommendations
            recommendations = self.cost_optimizer.get_optimization_recommendations()
            print("💡 Cost optimization recommendations:")
            for rec in recommendations:
                print(f"   • {rec}")
    
    def get_cost_report(self) -> Dict[str, Any]:
        """Get comprehensive cost report."""
        breakdown = self.cost_optimizer.get_cost_breakdown()
        recommendations = self.cost_optimizer.get_optimization_recommendations()
        
        return {
            "cost_breakdown": breakdown,
            "optimization_recommendations": recommendations,
            "operations_performed": self.operation_count,
            "cost_per_operation": breakdown["total_estimated_cost"] / self.operation_count if self.operation_count > 0 else 0,
            "report_timestamp": datetime.now().isoformat()
        }

# Usage example
async def cost_aware_operations():
    """Example of cost-aware browser operations."""
    client = CostAwareBrowserClient()
    
    try:
        # Create session
        session_id = await client.create_session()
        print(f"Created session: {session_id}")
        
        # Perform operations
        await client.navigate("https://example.com")
        await client.navigate("https://httpbin.org/html")
        
        # Simulate vision processing
        fake_screenshot = b"fake_image_data"
        await client.extract_text_with_vision(fake_screenshot)
        
        # Get cost report
        report = client.get_cost_report()
        print(f"Cost Report: {json.dumps(report, indent=2)}")
        
    finally:
        await client.close_session()

if __name__ == "__main__":
    asyncio.run(cost_aware_operations())
```

This comprehensive performance guide provides practical strategies for optimizing your LlamaIndex AgentCore Browser Integration applications. Implement these patterns based on your specific use case and performance requirements.