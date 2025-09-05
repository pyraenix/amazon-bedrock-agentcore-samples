# Browser-Use with AgentCore Browser Tool - Test Suite

This directory contains comprehensive tests for validating the integration between browser-use and Amazon Bedrock AgentCore Browser Tool for secure handling of sensitive information.

## Test Categories

### Unit Tests
- `test_pii_masking.py` - Tests for PII detection and masking functionality
- `test_credential_security.py` - Tests for secure credential handling
- `test_session_isolation.py` - Tests for AgentCore session isolation
- `test_audit_trail.py` - Tests for audit trail and logging functionality

### Integration Tests
- `validate_security_integration.py` - Validates security integration between browser-use and AgentCore
- `compliance_validation.py` - Tests compliance with HIPAA, PCI-DSS, GDPR requirements
- `performance_validation.py` - Performance and scalability tests

### Security Tests
- `run_security_tests.py` - Comprehensive security test runner
- Security boundary validation
- Data leakage prevention tests
- Session isolation verification

## Running Tests

### Prerequisites
1. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. Configure AWS credentials for AgentCore Browser Tool access

### Running Individual Test Suites

```bash
# Run PII masking tests
python -m pytest tests/test_pii_masking.py -v

# Run credential security tests
python -m pytest tests/test_credential_security.py -v

# Run session isolation tests
python -m pytest tests/test_session_isolation.py -v

# Run audit trail tests
python -m pytest tests/test_audit_trail.py -v
```

### Running Integration Tests

```bash
# Run security integration validation
python tests/validate_security_integration.py

# Run compliance validation
python tests/compliance_validation.py

# Run performance validation
python tests/performance_validation.py
```

### Running Security Tests

```bash
# Run comprehensive security test suite
python tests/run_security_tests.py
```

### Running All Tests

```bash
# Run complete test suite
python -m pytest tests/ -v --tb=short

# Run with coverage report
python -m pytest tests/ --cov=tools --cov=examples --cov-report=html
```

## Test Configuration

Tests use the configuration file `test_config.json` which contains:
- Test environment settings
- Mock data configurations
- Security test parameters
- Performance benchmarks

## Test Data

The `tutorial_data/` directory contains:
- Mock sensitive data for testing
- Sample forms and documents
- Test credentials (non-functional)
- Compliance validation datasets

## Continuous Integration

Tests are designed to run in CI/CD environments with:
- Automated security scanning
- Compliance validation
- Performance regression testing
- Integration verification

## Security Considerations

- All test data is synthetic and non-sensitive
- Tests validate security boundaries without exposing real data
- Credential tests use mock authentication systems
- Session isolation is verified in controlled environments