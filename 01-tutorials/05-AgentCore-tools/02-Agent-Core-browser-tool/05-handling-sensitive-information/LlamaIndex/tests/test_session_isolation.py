"""
Session Isolation Tests for LlamaIndex-AgentCore Integration

Tests to verify session isolation for AgentCore browser sessions with LlamaIndex.
Requirements: 1.3, 4.1
"""

import pytest
import asyncio
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock
import multiprocessing
import queue

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'examples'))

from agentcore_session_helpers import SessionManager, SessionPool
from agentcore_browser_loader import AgentCoreBrowserLoader

try:
    from bedrock_agentcore.tools.browser_client import BrowserSession
except ImportError:
    # Mock for testing
    class BrowserSession:
        def __init__(self, session_id: str):
            self.session_id = session_id
            self.cookies = {}
            self.local_storage = {}
            self.session_storage = {}


class TestSessionDataIsolation:
    """Test that session data is properly isolated between sessions."""
    
    def setup_method(self):
        """Set up test environment."""
        self.session_manager = SessionManager()
        self.session_pool = SessionPool()
        
    def test_basic_session_isolation(self):
        """Test basic isolation between different sessions."""
        # Create multiple sessions with different data
        session_data = [
            {"user_id": "user1", "role": "admin", "department": "IT"},
            {"user_id": "user2", "role": "user", "department": "HR"},
            {"user_id": "user3", "role": "manager", "department": "Finance"}
        ]
        
        session_ids = []
        for data in session_data:
            session_id = self.session_manager.create_session(data)
            session_ids.append(session_id)
            
        # Verify each session has only its own data
        for i, session_id in enumerate(session_ids):
            retrieved_data = self.session_manager.get_session_data(session_id)
            
            assert retrieved_data["user_id"] == session_data[i]["user_id"]
            assert retrieved_data["role"] == session_data[i]["role"]
            assert retrieved_data["department"] == session_data[i]["department"]
            
            # Verify it doesn't contain other sessions' data
            for j, other_data in enumerate(session_data):
                if i != j:
                    assert retrieved_data["user_id"] != other_data["user_id"]
                    
    def test_session_modification_isolation(self):
        """Test that modifying one session doesn't affect others."""
        # Create two sessions
        session1_id = self.session_manager.create_session({"counter": 0, "name": "session1"})
        session2_id = self.session_manager.create_session({"counter": 0, "name": "session2"})
        
        # Modify session1
        self.session_manager.update_session_data(session1_id, {"counter": 10, "modified": True})
        
        # Verify session2 is unaffected
        session2_data = self.session_manager.get_session_data(session2_id)
        assert session2_data["counter"] == 0
        assert session2_data["name"] == "session2"
        assert "modified" not in session2_data
        
        # Verify session1 has the changes
        session1_data = self.session_manager.get_session_data(session1_id)
        assert session1_data["counter"] == 10
        assert session1_data["modified"] is True
        
    def test_session_memory_isolation(self):
        """Test that session memory spaces are isolated."""
        # Create sessions with large data structures
        large_data1 = {"data": list(range(1000)), "session": "session1"}
        large_data2 = {"data": list(range(1000, 2000)), "session": "session2"}
        
        session1_id = self.session_manager.create_session(large_data1)
        session2_id = self.session_manager.create_session(large_data2)
        
        # Verify data integrity
        retrieved1 = self.session_manager.get_session_data(session1_id)
        retrieved2 = self.session_manager.get_session_data(session2_id)
        
        assert retrieved1["data"] == list(range(1000))
        assert retrieved2["data"] == list(range(1000, 2000))
        assert retrieved1["session"] == "session1"
        assert retrieved2["session"] == "session2"
        
    def test_session_cleanup_isolation(self):
        """Test that cleaning up one session doesn't affect others."""
        # Create multiple sessions
        session_ids = []
        for i in range(5):
            session_id = self.session_manager.create_session({"index": i})
            session_ids.append(session_id)
            
        # Clean up middle session
        middle_session = session_ids[2]
        self.session_manager.cleanup_session(middle_session)
        
        # Verify middle session is gone
        assert self.session_manager.get_session(middle_session) is None
        
        # Verify other sessions remain
        for i, session_id in enumerate(session_ids):
            if i != 2:  # Skip the cleaned up session
                session_data = self.session_manager.get_session_data(session_id)
                assert session_data["index"] == i


class TestBrowserSessionIsolation:
    """Test browser-specific session isolation."""
    
    def setup_method(self):
        """Set up test environment."""
        self.session_manager = SessionManager()
        
    def test_cookie_isolation(self):
        """Test that cookies are isolated between browser sessions."""
        # Create two browser sessions
        session1_id = self.session_manager.create_browser_session()
        session2_id = self.session_manager.create_browser_session()
        
        # Set different cookies for each session
        self.session_manager.set_session_cookie(session1_id, "auth_token", "token1_abc123")
        self.session_manager.set_session_cookie(session1_id, "user_pref", "theme_dark")
        
        self.session_manager.set_session_cookie(session2_id, "auth_token", "token2_def456")
        self.session_manager.set_session_cookie(session2_id, "user_pref", "theme_light")
        
        # Verify cookies are isolated
        session1_cookies = self.session_manager.get_session_cookies(session1_id)
        session2_cookies = self.session_manager.get_session_cookies(session2_id)
        
        assert session1_cookies["auth_token"] == "token1_abc123"
        assert session1_cookies["user_pref"] == "theme_dark"
        
        assert session2_cookies["auth_token"] == "token2_def456"
        assert session2_cookies["user_pref"] == "theme_light"
        
        # Verify cross-contamination doesn't occur
        assert session1_cookies["auth_token"] != session2_cookies["auth_token"]
        assert session1_cookies["user_pref"] != session2_cookies["user_pref"]
        
    def test_local_storage_isolation(self):
        """Test that local storage is isolated between sessions."""
        session1_id = self.session_manager.create_browser_session()
        session2_id = self.session_manager.create_browser_session()
        
        # Set local storage data
        self.session_manager.set_local_storage(session1_id, "user_data", '{"name": "User1"}')
        self.session_manager.set_local_storage(session1_id, "preferences", '{"lang": "en"}')
        
        self.session_manager.set_local_storage(session2_id, "user_data", '{"name": "User2"}')
        self.session_manager.set_local_storage(session2_id, "preferences", '{"lang": "es"}')
        
        # Verify isolation
        session1_storage = self.session_manager.get_local_storage(session1_id)
        session2_storage = self.session_manager.get_local_storage(session2_id)
        
        assert '"name": "User1"' in session1_storage["user_data"]
        assert '"name": "User2"' in session2_storage["user_data"]
        assert '"lang": "en"' in session1_storage["preferences"]
        assert '"lang": "es"' in session2_storage["preferences"]
        
    def test_session_storage_isolation(self):
        """Test that session storage is isolated between sessions."""
        session1_id = self.session_manager.create_browser_session()
        session2_id = self.session_manager.create_browser_session()
        
        # Set session storage data
        self.session_manager.set_session_storage(session1_id, "temp_data", "session1_temp")
        self.session_manager.set_session_storage(session2_id, "temp_data", "session2_temp")
        
        # Verify isolation
        session1_temp = self.session_manager.get_session_storage(session1_id, "temp_data")
        session2_temp = self.session_manager.get_session_storage(session2_id, "temp_data")
        
        assert session1_temp == "session1_temp"
        assert session2_temp == "session2_temp"
        assert session1_temp != session2_temp
        
    def test_browser_state_isolation(self):
        """Test that browser state (URL, history, etc.) is isolated."""
        session1_id = self.session_manager.create_browser_session()
        session2_id = self.session_manager.create_browser_session()
        
        # Navigate to different URLs
        self.session_manager.navigate_session(session1_id, "https://example1.com")
        self.session_manager.navigate_session(session2_id, "https://example2.com")
        
        # Add to history
        self.session_manager.navigate_session(session1_id, "https://example1.com/page1")
        self.session_manager.navigate_session(session2_id, "https://example2.com/page2")
        
        # Verify current URLs are different
        session1_url = self.session_manager.get_current_url(session1_id)
        session2_url = self.session_manager.get_current_url(session2_id)
        
        assert "example1.com/page1" in session1_url
        assert "example2.com/page2" in session2_url
        
        # Verify history is isolated
        session1_history = self.session_manager.get_session_history(session1_id)
        session2_history = self.session_manager.get_session_history(session2_id)
        
        assert any("example1.com" in url for url in session1_history)
        assert any("example2.com" in url for url in session2_history)
        assert not any("example2.com" in url for url in session1_history)
        assert not any("example1.com" in url for url in session2_history)


class TestConcurrentSessionAccess:
    """Test session isolation under concurrent access."""
    
    def setup_method(self):
        """Set up test environment."""
        self.session_manager = SessionManager()
        self.results = {}
        self.errors = []
        
    def test_concurrent_session_creation(self):
        """Test that concurrent session creation maintains isolation."""
        def create_session_worker(worker_id, results_dict):
            try:
                session_data = {"worker_id": worker_id, "timestamp": time.time()}
                session_id = self.session_manager.create_session(session_data)
                
                # Verify session data
                retrieved_data = self.session_manager.get_session_data(session_id)
                results_dict[worker_id] = {
                    "session_id": session_id,
                    "worker_id": retrieved_data["worker_id"],
                    "success": True
                }
            except Exception as e:
                results_dict[worker_id] = {"success": False, "error": str(e)}
                
        # Create multiple threads
        threads = []
        results = {}
        
        for i in range(10):
            thread = threading.Thread(target=create_session_worker, args=(i, results))
            threads.append(thread)
            thread.start()
            
        # Wait for all threads
        for thread in threads:
            thread.join()
            
        # Verify all sessions were created successfully
        assert len(results) == 10
        assert all(result["success"] for result in results.values())
        
        # Verify each worker got its own session
        worker_ids = [result["worker_id"] for result in results.values()]
        assert len(set(worker_ids)) == 10  # All unique
        
        # Verify session IDs are unique
        session_ids = [result["session_id"] for result in results.values()]
        assert len(set(session_ids)) == 10  # All unique
        
    def test_concurrent_session_access(self):
        """Test concurrent access to different sessions."""
        # Create sessions first
        session_ids = []
        for i in range(5):
            session_id = self.session_manager.create_session({"counter": 0, "session_num": i})
            session_ids.append(session_id)
            
        def access_session_worker(session_id, session_num, results_dict):
            try:
                # Perform multiple operations on the session
                for j in range(10):
                    # Read current data
                    data = self.session_manager.get_session_data(session_id)
                    current_counter = data["counter"]
                    
                    # Update counter
                    self.session_manager.update_session_data(
                        session_id, 
                        {"counter": current_counter + 1, "session_num": session_num}
                    )
                    
                    time.sleep(0.01)  # Small delay to increase chance of race conditions
                    
                # Final verification
                final_data = self.session_manager.get_session_data(session_id)
                results_dict[session_num] = {
                    "final_counter": final_data["counter"],
                    "session_num": final_data["session_num"],
                    "success": True
                }
            except Exception as e:
                results_dict[session_num] = {"success": False, "error": str(e)}
                
        # Start concurrent access
        threads = []
        results = {}
        
        for i, session_id in enumerate(session_ids):
            thread = threading.Thread(target=access_session_worker, args=(session_id, i, results))
            threads.append(thread)
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join()
            
        # Verify results
        assert len(results) == 5
        assert all(result["success"] for result in results.values())
        
        # Each session should have its own counter and session number
        for i in range(5):
            assert results[i]["final_counter"] == 10
            assert results[i]["session_num"] == i
            
    def test_concurrent_session_cleanup(self):
        """Test that concurrent session cleanup maintains isolation."""
        # Create many sessions
        session_ids = []
        for i in range(20):
            session_id = self.session_manager.create_session({"index": i})
            session_ids.append(session_id)
            
        def cleanup_worker(session_ids_to_cleanup, results_list):
            try:
                for session_id in session_ids_to_cleanup:
                    self.session_manager.cleanup_session(session_id)
                    time.sleep(0.001)  # Small delay
                results_list.append({"success": True})
            except Exception as e:
                results_list.append({"success": False, "error": str(e)})
                
        # Split sessions into groups for concurrent cleanup
        group1 = session_ids[:10]
        group2 = session_ids[10:]
        
        results = []
        
        thread1 = threading.Thread(target=cleanup_worker, args=(group1, results))
        thread2 = threading.Thread(target=cleanup_worker, args=(group2, results))
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        # Verify cleanup succeeded
        assert len(results) == 2
        assert all(result["success"] for result in results)
        
        # Verify all sessions are cleaned up
        for session_id in session_ids:
            assert self.session_manager.get_session(session_id) is None


class TestSessionPoolIsolation:
    """Test session pool isolation and management."""
    
    def setup_method(self):
        """Set up test environment."""
        self.session_pool = SessionPool(max_sessions=10)
        
    def test_session_pool_isolation(self):
        """Test that session pool maintains isolation between sessions."""
        # Get multiple sessions from pool
        sessions = []
        for i in range(5):
            session = self.session_pool.get_session(f"user_{i}")
            sessions.append(session)
            
        # Set different data in each session
        for i, session in enumerate(sessions):
            session.set_data("user_id", f"user_{i}")
            session.set_data("counter", i * 10)
            
        # Verify isolation
        for i, session in enumerate(sessions):
            assert session.get_data("user_id") == f"user_{i}"
            assert session.get_data("counter") == i * 10
            
            # Verify it doesn't have other sessions' data
            for j in range(5):
                if i != j:
                    assert session.get_data("user_id") != f"user_{j}"
                    assert session.get_data("counter") != j * 10
                    
    def test_session_pool_reuse_isolation(self):
        """Test that reused sessions from pool are properly isolated."""
        # Get a session and set data
        session1 = self.session_pool.get_session("user1")
        session1.set_data("sensitive_data", "secret_info_123")
        
        # Return session to pool
        self.session_pool.return_session(session1)
        
        # Get a new session (might be the same instance)
        session2 = self.session_pool.get_session("user2")
        
        # Verify previous data is not accessible
        assert session2.get_data("sensitive_data") is None
        
        # Set new data
        session2.set_data("new_data", "different_info_456")
        
        # Verify isolation
        assert session2.get_data("new_data") == "different_info_456"
        
    def test_session_pool_concurrent_access(self):
        """Test concurrent access to session pool maintains isolation."""
        def worker(worker_id, results_dict):
            try:
                # Get session from pool
                session = self.session_pool.get_session(f"worker_{worker_id}")
                
                # Set worker-specific data
                session.set_data("worker_id", worker_id)
                session.set_data("data", f"worker_{worker_id}_data")
                
                # Simulate work
                time.sleep(0.1)
                
                # Verify data integrity
                assert session.get_data("worker_id") == worker_id
                assert session.get_data("data") == f"worker_{worker_id}_data"
                
                # Return session
                self.session_pool.return_session(session)
                
                results_dict[worker_id] = {"success": True}
            except Exception as e:
                results_dict[worker_id] = {"success": False, "error": str(e)}
                
        # Start concurrent workers
        threads = []
        results = {}
        
        for i in range(8):  # Less than pool size to avoid blocking
            thread = threading.Thread(target=worker, args=(i, results))
            threads.append(thread)
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join()
            
        # Verify all workers succeeded
        assert len(results) == 8
        assert all(result["success"] for result in results.values())


class TestSessionSecurityIsolation:
    """Test security-related session isolation."""
    
    def setup_method(self):
        """Set up test environment."""
        self.session_manager = SessionManager()
        
    def test_credential_isolation_between_sessions(self):
        """Test that credentials are isolated between sessions."""
        # Create sessions with different credentials
        creds1 = {"username": "user1", "password": "secret1", "api_key": "key1"}
        creds2 = {"username": "user2", "password": "secret2", "api_key": "key2"}
        
        session1_id = self.session_manager.create_session(creds1)
        session2_id = self.session_manager.create_session(creds2)
        
        # Verify credential isolation
        session1_creds = self.session_manager.get_session_credentials(session1_id)
        session2_creds = self.session_manager.get_session_credentials(session2_id)
        
        assert session1_creds["username"] == "user1"
        assert session1_creds["password"] == "secret1"
        assert session2_creds["username"] == "user2"
        assert session2_creds["password"] == "secret2"
        
        # Verify no cross-contamination
        assert session1_creds["password"] != session2_creds["password"]
        assert session1_creds["api_key"] != session2_creds["api_key"]
        
    def test_session_token_isolation(self):
        """Test that session tokens are isolated."""
        # Create sessions with tokens
        session1_id = self.session_manager.create_session({"token": "token_abc_123"})
        session2_id = self.session_manager.create_session({"token": "token_def_456"})
        
        # Verify token isolation
        token1 = self.session_manager.get_session_token(session1_id)
        token2 = self.session_manager.get_session_token(session2_id)
        
        assert token1 == "token_abc_123"
        assert token2 == "token_def_456"
        assert token1 != token2
        
    def test_session_permission_isolation(self):
        """Test that session permissions are isolated."""
        # Create sessions with different permissions
        session1_id = self.session_manager.create_session({
            "permissions": ["read", "write", "admin"]
        })
        session2_id = self.session_manager.create_session({
            "permissions": ["read"]
        })
        
        # Verify permission isolation
        perms1 = self.session_manager.get_session_permissions(session1_id)
        perms2 = self.session_manager.get_session_permissions(session2_id)
        
        assert "admin" in perms1
        assert "admin" not in perms2
        assert "write" in perms1
        assert "write" not in perms2
        assert "read" in both perms1 and perms2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])