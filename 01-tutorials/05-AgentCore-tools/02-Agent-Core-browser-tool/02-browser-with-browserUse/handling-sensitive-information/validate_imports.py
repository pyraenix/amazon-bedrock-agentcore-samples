#!/usr/bin/env python3
"""
Simple validation script to test imports and basic functionality
of the browser-use sensitive information detection modules.
"""

import sys
import os

# Add the tools directory to the path
tools_path = os.path.join(os.path.dirname(__file__), 'tools')
sys.path.insert(0, tools_path)

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        from browseruse_sensitive_data_handler import (
            BrowserUseSensitiveDataHandler,
            PIIType,
            ComplianceFramework,
            DataClassification
        )
        print("✓ browseruse_sensitive_data_handler imported successfully")
    except Exception as e:
        print(f"✗ Failed to import browseruse_sensitive_data_handler: {e}")
        return False
    
    try:
        from browseruse_pii_masking import (
            BrowserUsePIIMasking,
            BrowserUsePIIValidator
        )
        print("✓ browseruse_pii_masking imported successfully")
    except Exception as e:
        print(f"✗ Failed to import browseruse_pii_masking: {e}")
        return False
    
    try:
        from browseruse_credential_handling import (
            BrowserUseCredentialHandler,
            CredentialType
        )
        print("✓ browseruse_credential_handling imported successfully")
    except Exception as e:
        print(f"✗ Failed to import browseruse_credential_handling: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of the modules."""
    print("\nTesting basic functionality...")
    
    try:
        from browseruse_sensitive_data_handler import (
            BrowserUseSensitiveDataHandler,
            ComplianceFramework,
            PIIType
        )
        
        # Test PII detection
        handler = BrowserUseSensitiveDataHandler([ComplianceFramework.HIPAA])
        test_text = "Patient SSN: 123-45-6789, Email: john@example.com"
        detections = handler.detect_pii(test_text)
        
        print(f"✓ PII detection found {len(detections)} items")
        
        # Test masking
        masked_text, _ = handler.mask_text(test_text)
        print(f"✓ Text masking: '{test_text}' -> '{masked_text}'")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_credential_handling():
    """Test credential handling functionality."""
    print("\nTesting credential handling...")
    
    try:
        from browseruse_credential_handling import (
            BrowserUseCredentialHandler,
            CredentialType
        )
        
        # Test credential handler initialization
        handler = BrowserUseCredentialHandler(session_id="test-session")
        print("✓ Credential handler initialized")
        
        # Test credential metadata
        credentials = handler.list_credentials()
        print(f"✓ Listed {len(credentials)} credentials")
        
        return True
        
    except Exception as e:
        print(f"✗ Credential handling test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("Browser-Use Sensitive Information Detection - Import Validation")
    print("=" * 60)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test basic functionality
    if not test_basic_functionality():
        success = False
    
    # Test credential handling
    if not test_credential_handling():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All validation tests passed!")
        return 0
    else:
        print("✗ Some validation tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())