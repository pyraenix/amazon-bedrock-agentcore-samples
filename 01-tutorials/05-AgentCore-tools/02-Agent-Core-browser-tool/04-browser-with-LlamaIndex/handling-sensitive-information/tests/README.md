# LlamaIndex-AgentCore Security Testing Suite

This directory contains comprehensive security validation tests for the LlamaIndex-AgentCore Browser Tool integration. The test suite validates credential isolation, PII masking, session isolation, audit trail completeness, and overall security compliance.

## Test Structure

### Core Test Files

- **`test_security_validation.py`** - Main security validation test suite with comprehensive integration tests
- **`test_credential_security.py`** - Focused tests for credential isolation, encryption, and secure handling
- **`test_pii_masking.py`** - Comprehensive PII detection, masking, and sanitization tests
- **`test_session_isolation.py`** - Session isolation verification tests for AgentCore browser sessions
- **`test_audit_trail.py`** - Audit trail completeness tests for sensitive operations

### Test Categories

#### 1. Credential Security Tests (`test_credential_security.py`)
- **Credential Encryption**: Tests encryption/decryption of credentials at rest
- **Credential Isolation**: Verifies credentials are isolated between sessions and processes
- **Credential Logging**: Ensures credentials never appear in logs or error messages
- **Credential Validation**: Tests credential format validation and security checks

#### 2. PII Masking Tests (`test_pii_masking.py`)
- **PII Detection**: Tests detection of emails, phones, SSNs, credit cards, names, addresses
- **PII Masking**: Validates proper masking while preserving document context
- **Document Sanitization**: Tests LlamaIndex document-level sanitization
- **Data Classification**: Tests sensitivity level and compliance classification

#### 3. Session Isolation Tests (`test_session_isolation.py`)
- **Session Data Isolation**: Verifies session data is isolated between different sessions
- **Browser Session Isolation**: Tests cookie, local storage, and session storage isolation
- **Concurrent Session Access**: Validates isolation under concurrent operations
- **Session Pool Isolation**: Tests session pool management and reuse isolation

#### 4. Audit Trail Tests (`test_audit_trail.py`)
- **Audit Log Generation**: Ensures all sensitive operations are logged
- **Audit Log Integrity**: Tests tamper detection and log chain integrity
- **Audit Log Completeness**: Validates all required fields and metadata
- **Compliance Reporting**: Tests GDPR, HIPAA, and SOX compliance reporting

#### 5. Integration Tests (`test_security_validation.py`)
- **End-to-End Security Workflow**: Complete workflow from session creation to cleanup
- **Security Under Error Conditions**: Validates security is maintained during errors
- **Concurrent Security Operations**: Tests security under concurrent operations
- **Performance Impact**: Measures security overhead on operations

## Running Tests

### Prerequisites

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov pytest-xdist pytest-timeout
   ```

2. **Environment Setup**:
   ```bash
   # Copy and configure environment variables
   cp .env.example .env
   # Edit .env with your AWS credentials and configuration
   ```

### Running Individual Test Suites

```bash
# Run all security tests
pytest -v

# Run specific test categories
pytest -v -m security
pytest -v -m pii
pytest -v -m credentials
pytest -v -m sessions
pytest -v -m audit

# Run specific test files
pytest -v test_credential_security.py
pytest -v test_pii_masking.py
pytest -v test_session_isolation.py
pytest -v test_audit_trail.py

# Run with coverage
pytest -v --cov=../examples --cov-report=html

# Run in parallel
pytest -v -n auto
```

### Using the Test Runner Script

The `run_security_tests.py` script provides a comprehensive test execution interface:

```bash
# Run all tests with HTML report
python ../run_security_tests.py --output-format html

# Run only unit tests with coverage
python ../run_security_tests.py --test-type unit --coverage

# Run tests in parallel with all report formats
python ../run_security_tests.py --parallel --output-format all

# Run specific test markers
python ../run_security_tests.py --markers "security and not slow"
```

### Using the Integration Validator

The `validate_security_integration.py` script provides end-to-end validation:

```bash
# Run complete validation
python ../validate_security_integration.py

# Run specific validation suites
python ../validate_security_integration.py --test-suite security
python ../validate_security_integration.py --test-suite performance
python ../validate_security_integration.py --test-suite compliance

# Verbose output
python ../validate_security_integration.py --verbose
```

## Test Configuration

### Pytest Configuration (`pytest.ini`)

The test suite uses pytest with the following key configurations:

- **Coverage**: Minimum 80% coverage requirement
- **Timeout**: 300-second timeout for long-running tests
- **Markers**: Organized test categorization
- **Logging**: Comprehensive logging configuration
- **Warnings**: Filtered warnings for cleaner output

### Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.security` - Security-related tests
- `@pytest.mark.pii` - PII handling tests
- `@pytest.mark.credentials` - Credential security tests
- `@pytest.mark.sessions` - Session isolation tests
- `@pytest.mark.audit` - Audit trail tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.compliance` - Compliance tests
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.e2e` - End-to-end tests

## Test Data and Fixtures

### Mock Data

Tests use realistic but safe mock data:

```python
# Example PII test data
test_pii_content = """
Patient: John Doe
Email: john.doe@hospital.com
Phone: 555-123-4567
SSN: 123-45-6789
"""

# Example credential test data
test_credentials = {
    "username": "test_user",
    "password": "secure_password_123!",
    "api_key": "sk-test-key-abcdef123456"
}
```

### Test Fixtures

Common fixtures are provided for:

- Session managers
- Browser loaders
- Data handlers
- Security auditors
- Mock AgentCore components

## Expected Test Results

### Passing Tests

When all tests pass, you should see:

```
================================ test session starts ================================
collected 45 items

test_credential_security.py::TestCredentialEncryption::test_credential_encryption_at_rest PASSED
test_credential_security.py::TestCredentialIsolation::test_session_credential_isolation PASSED
test_pii_masking.py::TestPIIDetection::test_email_detection PASSED
test_pii_masking.py::TestPIIMasking::test_email_masking PASSED
test_session_isolation.py::TestSessionDataIsolation::test_basic_session_isolation PASSED
test_audit_trail.py::TestAuditLogGeneration::test_session_creation_audit PASSED

================================ 45 passed in 12.34s ================================
```

### Test Coverage

The test suite aims for >80% code coverage:

```
Name                                    Stmts   Miss  Cover
-----------------------------------------------------------
examples/agentcore_browser_loader.py     156     12    92%
examples/sensitive_data_handler.py       134      8    94%
examples/secure_rag_pipeline.py          98      6    94%
examples/agentcore_session_helpers.py    145     15    90%
examples/llamaindex_monitoring.py        87      7    92%
-----------------------------------------------------------
TOTAL                                    620     48    92%
```

## Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Ensure examples directory is in Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/examples"
   ```

2. **AWS Credential Issues**:
   ```bash
   # Check AWS credentials
   aws sts get-caller-identity
   
   # Or set environment variables
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   export AWS_REGION=us-east-1
   ```

3. **Missing Dependencies**:
   ```bash
   # Install all test dependencies
   pip install -r requirements.txt
   pip install pytest pytest-cov pytest-xdist pytest-timeout junit-xml
   ```

4. **Test Timeouts**:
   ```bash
   # Increase timeout for slow tests
   pytest -v --timeout=600
   ```

### Debug Mode

For debugging failing tests:

```bash
# Run with verbose output and no capture
pytest -v -s --tb=long

# Run specific failing test
pytest -v -s test_credential_security.py::TestCredentialEncryption::test_credential_encryption_at_rest

# Run with pdb debugger
pytest -v --pdb
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Security Tests

on: [push, pull_request]

jobs:
  security-tests:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
        
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-xdist
        
    - name: Run security tests
      run: |
        cd tests
        python ../run_security_tests.py --test-type all --output-format junit
        
    - name: Upload test results
      uses: actions/upload-artifact@v2
      with:
        name: test-results
        path: test_report_*.xml
```

## Security Considerations

### Test Data Security

- All test data uses mock/synthetic values
- No real credentials or PII in test files
- Test databases are isolated and ephemeral
- Audit logs are cleaned after tests

### Test Environment Isolation

- Tests run in isolated environments
- No network calls to production systems
- Mock AgentCore services for testing
- Temporary files are cleaned up

### Compliance Testing

The test suite validates compliance with:

- **GDPR**: Data subject rights, consent management
- **HIPAA**: PHI access logging, audit trails
- **SOX**: Financial data access controls
- **PCI DSS**: Credit card data handling

## Contributing

### Adding New Tests

1. **Follow naming conventions**: `test_*.py` files, `Test*` classes, `test_*` methods
2. **Use appropriate markers**: Add `@pytest.mark.*` decorators
3. **Include docstrings**: Document test purpose and requirements
4. **Mock external dependencies**: Use mocks for AgentCore services
5. **Clean up resources**: Ensure tests clean up after themselves

### Test Quality Guidelines

- **Isolation**: Tests should not depend on each other
- **Deterministic**: Tests should produce consistent results
- **Fast**: Unit tests should complete quickly
- **Comprehensive**: Cover both positive and negative cases
- **Readable**: Clear test names and documentation

## Support

For issues with the security test suite:

1. Check the troubleshooting section above
2. Review test logs and error messages
3. Ensure all dependencies are installed
4. Verify environment configuration
5. Run tests with verbose output for debugging

The security test suite is designed to provide comprehensive validation of the LlamaIndex-AgentCore integration's security features, ensuring production-ready security standards are met.