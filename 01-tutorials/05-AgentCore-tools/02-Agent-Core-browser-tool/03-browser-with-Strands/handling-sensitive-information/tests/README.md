# Strands-AgentCore Security Test Suite

This directory contains comprehensive security validation tests for the Strands agents integration with Amazon Bedrock AgentCore Browser Tool. The test suite ensures that all security controls, compliance requirements, and performance standards are met.

## Test Structure

### Unit Tests
- `test_credential_security.py` - Validates credential handling and security
- `test_pii_masking.py` - Tests PII detection and masking functionality
- `test_session_isolation.py` - Verifies session isolation between agents
- `test_audit_trail.py` - Ensures comprehensive audit logging

### Integration Tests
- `validate_security_integration.py` - End-to-end security validation
- `performance_validation.py` - Performance testing under security constraints
- `compliance_validation.py` - Regulatory compliance validation

### Test Runner
- `run_security_tests.py` - Main test runner for all security tests

## Quick Start

### Run All Tests
```bash
python run_security_tests.py
```

### Run Specific Test Categories
```bash
# Unit tests only
python run_security_tests.py --category unit

# Integration tests only
python run_security_tests.py --category integration

# Performance tests only
python run_security_tests.py --category performance

# Compliance tests only
python run_security_tests.py --category compliance
```

### Run with Custom Configuration
```bash
python run_security_tests.py --config test_config.json --verbose
```

## Test Categories

### 1. Credential Security Tests (`test_credential_security.py`)

**Purpose**: Verify that credentials are never exposed in logs, outputs, or memory.

**Key Tests**:
- Credential masking in logs
- Secure credential injection
- Credential rotation security
- Memory cleanup after use
- Audit trail without credential exposure

**Requirements Covered**: 8.1, 8.2, 8.3, 8.4

### 2. PII Masking Tests (`test_pii_masking.py`)

**Purpose**: Validate PII detection, classification, and masking throughout workflows.

**Key Tests**:
- PII detection accuracy
- Format-preserving masking
- Sensitivity classification
- Real-time masking during workflows
- Performance under load

**Requirements Covered**: 8.1, 8.2, 8.3, 8.4

### 3. Session Isolation Tests (`test_session_isolation.py`)

**Purpose**: Ensure complete isolation between different agent sessions.

**Key Tests**:
- Data isolation between sessions
- Memory isolation
- Network isolation
- Concurrent session handling
- Session cleanup verification

**Requirements Covered**: 8.1, 8.2, 8.3, 8.4

### 4. Audit Trail Tests (`test_audit_trail.py`)

**Purpose**: Verify comprehensive audit logging for compliance.

**Key Tests**:
- Complete audit trail coverage
- Log integrity protection
- Compliance reporting
- Real-time monitoring
- Retention policy enforcement

**Requirements Covered**: 8.1, 8.2, 8.3, 8.4

### 5. Security Integration Validation (`validate_security_integration.py`)

**Purpose**: End-to-end validation of the complete security integration.

**Key Tests**:
- Secure session lifecycle
- Complete credential security flow
- PII handling pipeline
- Multi-agent isolation
- Error handling security

**Requirements Covered**: 8.1, 8.2, 8.3, 8.4, 8.5

### 6. Performance Validation (`performance_validation.py`)

**Purpose**: Ensure security controls don't significantly impact performance.

**Key Tests**:
- Session creation performance
- Credential injection performance
- PII processing performance
- Memory usage patterns
- Concurrent operation performance

**Requirements Covered**: 8.1, 8.2, 8.3, 8.4, 8.5

### 7. Compliance Validation (`compliance_validation.py`)

**Purpose**: Verify regulatory compliance across different industry scenarios.

**Key Tests**:
- HIPAA compliance validation
- PCI DSS compliance validation
- GDPR compliance validation
- SOX compliance validation
- Industry-specific scenarios

**Requirements Covered**: 8.1, 8.2, 8.3, 8.4, 8.5

## Configuration

### Test Configuration File (`test_config.json`)

The test suite uses a comprehensive configuration file that defines:

- **Integration Settings**: Validation levels, security standards, test timeouts
- **Performance Thresholds**: Acceptable performance limits for various operations
- **Compliance Standards**: Regulatory requirements to validate
- **Security Controls**: Specific security features to test
- **Test Execution**: Parallel execution, retry policies, reporting options

### Key Configuration Sections

```json
{
  "integration": {
    "validation_level": "COMPREHENSIVE",
    "security_standards": ["HIPAA", "PCI_DSS", "GDPR", "SOX"]
  },
  "performance": {
    "thresholds": {
      "session_creation_ms": 200,
      "credential_injection_ms": 100,
      "pii_masking_ms": 75,
      "audit_logging_ms": 50
    }
  },
  "compliance": {
    "standards": ["HIPAA", "PCI_DSS", "GDPR", "SOX"],
    "validation_level": "STRICT"
  }
}
```

## Test Results and Reporting

### Output Files

The test suite generates several output files:

- `security_test_results_[timestamp].json` - Detailed test results in JSON format
- `security_test_report_[timestamp].md` - Human-readable test report
- `security_integration_validation_results.json` - Integration test results
- `performance_validation_results.json` - Performance test results
- `compliance_validation_results.json` - Compliance test results

### Report Contents

Each report includes:

- **Executive Summary**: Overall status and scores
- **Test Statistics**: Pass/fail counts, execution times
- **Security Score**: Overall security rating (0.0-1.0)
- **Performance Metrics**: Response times, throughput, resource usage
- **Compliance Status**: Regulatory compliance validation results
- **Critical Issues**: Security violations requiring immediate attention
- **Recommendations**: Actionable improvement suggestions

## Security Standards Validated

### HIPAA (Health Insurance Portability and Accountability Act)
- PHI (Protected Health Information) protection
- Administrative, physical, and technical safeguards
- Audit controls and access management
- Data integrity and transmission security

### PCI DSS (Payment Card Industry Data Security Standard)
- Cardholder data protection
- Strong access controls
- Network security monitoring
- Regular security testing

### GDPR (General Data Protection Regulation)
- Data subject rights implementation
- Privacy by design principles
- Lawful basis for processing
- Data breach notification procedures

### SOX (Sarbanes-Oxley Act)
- Internal controls over financial reporting
- Audit trail requirements
- Data retention policies
- Change management controls

## Performance Benchmarks

### Acceptable Performance Thresholds

| Operation | Threshold | Description |
|-----------|-----------|-------------|
| Session Creation | < 200ms | Time to create secure session |
| Credential Injection | < 100ms | Time to securely inject credentials |
| PII Masking | < 75ms | Time to detect and mask PII |
| Audit Logging | < 50ms | Time to log security events |
| Memory Usage | < 512MB | Maximum memory per session |

### Scalability Requirements

- **Concurrent Sessions**: Support up to 20 concurrent sessions
- **Performance Degradation**: < 2x slowdown under full load
- **Memory Growth**: < 100MB growth during extended operations
- **Response Time Consistency**: < 30% variance in response times

## Troubleshooting

### Common Issues

1. **Test Timeouts**
   - Increase timeout values in configuration
   - Check system resources and performance
   - Verify network connectivity

2. **Mock Import Errors**
   - Ensure all required dependencies are installed
   - Check Python path configuration
   - Verify mock implementations are available

3. **Performance Test Failures**
   - Review system load during testing
   - Adjust performance thresholds if needed
   - Check for resource constraints

4. **Compliance Test Issues**
   - Verify compliance configuration is correct
   - Check that all required controls are implemented
   - Review audit evidence collection

### Debug Mode

Run tests with verbose output for debugging:

```bash
python run_security_tests.py --verbose
```

### Individual Test Execution

Run specific test files for focused debugging:

```bash
python -m pytest test_credential_security.py -v
python -m pytest test_pii_masking.py -v
python -m pytest test_session_isolation.py -v
python -m pytest test_audit_trail.py -v
```

## Dependencies

### Required Python Packages

```bash
pip install pytest
pip install asyncio
pip install psutil
pip install statistics
```

### Mock Dependencies

The test suite includes comprehensive mocking for:
- Strands framework components
- AgentCore Browser Tool integration
- AWS services (Secrets Manager, CloudWatch)
- Security and compliance validation tools

## Contributing

When adding new tests:

1. Follow the existing test structure and naming conventions
2. Include comprehensive mocking for external dependencies
3. Add appropriate assertions for security validation
4. Update configuration files with new test parameters
5. Document new test cases in this README

## Security Considerations

- Test data uses synthetic PII and credentials only
- No real sensitive data is used in testing
- All test outputs are sanitized to prevent data exposure
- Test environments should be isolated from production systems

## Support

For issues with the test suite:

1. Check the troubleshooting section above
2. Review test logs for specific error messages
3. Verify configuration settings match your environment
4. Ensure all dependencies are properly installed

## License

This test suite is part of the Strands-AgentCore integration tutorial and follows the same licensing terms as the main project.