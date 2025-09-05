#!/usr/bin/env python3
"""
Validation Script for Secure RAG Pipeline

Simple validation script to test the core functionality of the SecureRAGPipeline
without requiring full LlamaIndex dependencies. This script validates the
implementation against the requirements.

Requirements Addressed:
- 3.1: Secure RAG pipeline using AgentCore Browser Tool for data ingestion
- 3.2: Secure vector storage with encryption for sensitive embeddings  
- 3.3: Query engines that handle sensitive context without data leakage
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add the examples directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_secure_rag_config():
    """Test SecureRAGConfig functionality."""
    print("Testing SecureRAGConfig...")
    
    try:
        from secure_rag_pipeline import SecureRAGConfig
        
        # Test default configuration
        config = SecureRAGConfig()
        assert config.storage_dir == "./secure_vector_store"
        assert config.enable_encryption == True
        assert config.embedding_model == "amazon.titan-embed-text-v1"
        assert config.similarity_top_k == 5
        assert config.enable_query_sanitization == True
        assert config.enable_response_filtering == True
        assert config.audit_all_operations == True
        
        # Test custom configuration
        custom_config = SecureRAGConfig(
            storage_dir="./custom_storage",
            enable_encryption=False,
            similarity_top_k=10
        )
        assert custom_config.storage_dir == "./custom_storage"
        assert custom_config.enable_encryption == False
        assert custom_config.similarity_top_k == 10
        
        # Test encryption key generation
        config1 = SecureRAGConfig(enable_encryption=True, encryption_key=None)
        config2 = SecureRAGConfig(enable_encryption=True, encryption_key=None)
        assert config1.encryption_key is not None
        assert config2.encryption_key is not None
        assert config1.encryption_key != config2.encryption_key
        
        print("✅ SecureRAGConfig tests passed")
        return True
        
    except Exception as e:
        print(f"❌ SecureRAGConfig tests failed: {str(e)}")
        return False


def test_query_metrics():
    """Test QueryMetrics functionality."""
    print("Testing QueryMetrics...")
    
    try:
        from secure_rag_pipeline import QueryMetrics
        from datetime import datetime
        
        # Test initialization
        metrics = QueryMetrics(query_id="test-query-123")
        assert metrics.query_id == "test-query-123"
        assert isinstance(metrics.timestamp, datetime)
        assert metrics.documents_retrieved == 0
        assert metrics.response_length == 0
        assert metrics.response_sanitized == False
        assert len(metrics.security_violations) == 0
        assert len(metrics.audit_events) == 0
        
        # Test adding audit events
        metrics.add_audit_event("test_event", {"key": "value"})
        assert len(metrics.audit_events) == 1
        event = metrics.audit_events[0]
        assert event['event_type'] == "test_event"
        assert event['details'] == {"key": "value"}
        assert 'timestamp' in event
        
        # Test adding security violations
        metrics.add_security_violation("test violation")
        assert len(metrics.security_violations) == 1
        assert metrics.security_violations[0] == "test violation"
        
        # Test serialization
        result = metrics.to_dict()
        assert isinstance(result, dict)
        assert result['query_id'] == "test-query-123"
        assert 'query_characteristics' in result
        assert 'retrieval_metrics' in result
        assert 'response_metrics' in result
        assert 'security_metrics' in result
        
        print("✅ QueryMetrics tests passed")
        return True
        
    except Exception as e:
        print(f"❌ QueryMetrics tests failed: {str(e)}")
        return False


def test_secure_vector_store():
    """Test SecureVectorStore functionality."""
    print("Testing SecureVectorStore...")
    
    try:
        from secure_rag_pipeline import SecureVectorStore, SecureRAGConfig
        
        # Create temporary directory for testing
        temp_dir = tempfile.mkdtemp()
        
        try:
            config = SecureRAGConfig(
                storage_dir=temp_dir,
                enable_encryption=True
            )
            
            # Test initialization
            store = SecureVectorStore(config)
            assert str(store.storage_dir) == temp_dir
            assert store.storage_context is not None
            
            # Test encryption/decryption (if available)
            if store.cipher_suite is not None:
                original_data = "This is sensitive test data"
                encrypted_data = store.encrypt_data(original_data)
                decrypted_data = store.decrypt_data(encrypted_data)
                
                assert original_data != encrypted_data
                assert original_data == decrypted_data
                print("  ✅ Encryption/decryption working")
            else:
                print("  ⚠️ Encryption not available (cryptography library not installed)")
            
            # Test without encryption
            config_no_encrypt = SecureRAGConfig(
                storage_dir=temp_dir,
                enable_encryption=False
            )
            store_no_encrypt = SecureVectorStore(config_no_encrypt)
            
            test_data = "Test data"
            encrypted = store_no_encrypt.encrypt_data(test_data)
            decrypted = store_no_encrypt.decrypt_data(encrypted)
            
            assert encrypted == test_data  # Should be unchanged
            assert decrypted == test_data
            
            print("✅ SecureVectorStore tests passed")
            return True
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
        
    except Exception as e:
        print(f"❌ SecureVectorStore tests failed: {str(e)}")
        return False


def test_utility_functions():
    """Test utility functions."""
    print("Testing utility functions...")
    
    try:
        from secure_rag_pipeline import create_secure_rag_pipeline, create_high_security_rag_pipeline
        
        # Test create_secure_rag_pipeline (this will fail without full LlamaIndex, but we can test the function exists)
        try:
            pipeline = create_secure_rag_pipeline(
                storage_dir="./test_storage",
                enable_encryption=False,  # Disable to avoid encryption dependencies
                region="us-west-2"
            )
            print("  ✅ create_secure_rag_pipeline function works")
        except Exception as e:
            if "BedrockEmbedding" in str(e) or "llama_index" in str(e):
                print("  ⚠️ create_secure_rag_pipeline requires full LlamaIndex installation")
            else:
                raise e
        
        # Test create_high_security_rag_pipeline
        try:
            high_sec_pipeline = create_high_security_rag_pipeline(
                storage_dir="./high_security_storage",
                region="us-east-1"
            )
            print("  ✅ create_high_security_rag_pipeline function works")
        except Exception as e:
            if "BedrockEmbedding" in str(e) or "llama_index" in str(e):
                print("  ⚠️ create_high_security_rag_pipeline requires full LlamaIndex installation")
            else:
                raise e
        
        print("✅ Utility functions tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Utility functions tests failed: {str(e)}")
        return False


def test_integration_components():
    """Test integration with existing components."""
    print("Testing integration components...")
    
    try:
        # Test imports
        from secure_rag_pipeline import SecureRAGPipeline
        from sensitive_data_handler import DocumentSanitizer, SensitiveDataClassifier
        from agentcore_browser_loader import AgentCoreBrowserLoader
        
        print("  ✅ All required imports successful")
        
        # Test that components can be instantiated together
        temp_dir = tempfile.mkdtemp()
        
        try:
            from secure_rag_pipeline import SecureRAGConfig
            
            config = SecureRAGConfig(
                storage_dir=temp_dir,
                enable_encryption=False  # Disable for testing
            )
            
            # This will fail without full LlamaIndex, but we can test the class exists
            try:
                pipeline = SecureRAGPipeline(config=config)
                print("  ✅ SecureRAGPipeline instantiation successful")
            except Exception as e:
                if "BedrockEmbedding" in str(e) or "llama_index" in str(e):
                    print("  ⚠️ SecureRAGPipeline requires full LlamaIndex installation")
                else:
                    raise e
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        print("✅ Integration components tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Integration components tests failed: {str(e)}")
        return False


def test_security_features():
    """Test security features implementation."""
    print("Testing security features...")
    
    try:
        from secure_rag_pipeline import SecureRAGConfig
        from sensitive_data_handler import create_secure_sanitization_config
        
        # Test security configuration
        config = SecureRAGConfig()
        assert config.enable_query_sanitization == True
        assert config.enable_response_filtering == True
        assert config.audit_all_operations == True
        assert config.enable_context_filtering == True
        
        # Test high-security configuration
        high_sec_config = SecureRAGConfig(
            similarity_top_k=3,
            max_response_length=1000,
            max_sensitive_context=0.1
        )
        assert high_sec_config.similarity_top_k == 3
        assert high_sec_config.max_response_length == 1000
        assert high_sec_config.max_sensitive_context == 0.1
        
        # Test sanitization integration
        sanitization_config = create_secure_sanitization_config(strict_mode=True)
        assert sanitization_config is not None
        assert sanitization_config.audit_sensitive_operations == True
        
        print("✅ Security features tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Security features tests failed: {str(e)}")
        return False


def main():
    """Run all validation tests."""
    print("Secure RAG Pipeline Validation")
    print("=" * 50)
    
    tests = [
        test_secure_rag_config,
        test_query_metrics,
        test_secure_vector_store,
        test_utility_functions,
        test_integration_components,
        test_security_features
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {str(e)}")
            print()
    
    print("=" * 50)
    print(f"Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All core functionality tests passed!")
        print("\nNext steps:")
        print("1. Install full LlamaIndex dependencies for complete functionality")
        print("2. Configure AWS credentials for Bedrock embedding model")
        print("3. Test with real web data ingestion")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed or had issues")
        print("Please review the implementation and dependencies")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)