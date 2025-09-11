
# Strands-AgentCore Security Test Suite Report

**Test Run ID:** security_test_1756958106
**Start Time:** 2025-09-03T20:55:06.209029
**End Time:** 2025-09-03T20:55:08.414488
**Duration:** 2.21 seconds
**Overall Status:** FAILED

## Summary Scores
- **Security Score:** 0.18/1.00
- **Performance Score:** 0.85/1.00
- **Compliance Score:** 0.96/1.00

## Test Statistics
- **Total Tests:** 102
- **Passed Tests:** 18
- **Failed Tests:** 82
- **Skipped Tests:** 2

## Test Categories

### Unit Tests
- **Status:** FAILED
- **Tests:** 83 total, 1 passed, 82 failed

### Integration Tests
- **Status:** SKIPPED
- **Tests:** 0 total, 0 passed, 0 failed

### Performance Tests
- **Status:** PASSED
- **Tests:** 5 total, 4 passed, 0 failed
- **Performance Score:** 0.85

### Compliance Tests
- **Status:** PASSED
- **Tests:** 3 total, 3 passed, 0 failed
- **Compliance Score:** 0.96

### Security Validation
- **Status:** PASSED
- **Tests:** 6 total, 6 passed, 0 failed

### End To End Tests
- **Status:** PASSED
- **Tests:** 5 total, 4 passed, 0 failed

## Critical Issues (4)
- Unit test failures in test_credential_security.py: 18 tests failed
- Unit test failures in test_pii_masking.py: 22 tests failed
- Unit test failures in test_session_isolation.py: 18 tests failed
- Unit test failures in test_audit_trail.py: 24 tests failed

## Warnings (3)
- Performance test framework not fully available
- Compliance test framework not fully available
- End-to-end test framework partially mocked

## Recommendations
- Consider implementing full performance test suite
- Implement full compliance validation suite
- Implement full end-to-end test scenarios

## Conclusion
Security tests failed. Critical security issues must be addressed before deployment.
