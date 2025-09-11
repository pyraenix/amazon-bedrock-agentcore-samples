#!/usr/bin/env python3
"""
Test runner for Strands CAPTCHA tutorial validation and performance benchmarks.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False
    except FileNotFoundError:
        print("❌ FAILED - Command not found")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    required_packages = [
        "pytest",
        "psutil",
        "asyncio"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def run_tutorial_validation_tests():
    """Run tutorial validation tests."""
    test_files = [
        "tests/test_tutorial_validation.py",
        "tests/test_captcha_sites.py", 
        "tests/test_error_handling.py"
    ]
    
    results = []
    
    for test_file in test_files:
        if Path(test_file).exists():
            command = ["python", "-m", "pytest", test_file, "-v", "--tb=short"]
            success = run_command(command, f"Tutorial Validation: {test_file}")
            results.append((test_file, success))
        else:
            print(f"⚠️  Test file not found: {test_file}")
            results.append((test_file, False))
    
    return results

def run_performance_benchmarks():
    """Run performance benchmark tests."""
    test_file = "tests/test_performance_benchmarks.py"
    
    if Path(test_file).exists():
        command = ["python", "-m", "pytest", test_file, "-v", "--tb=short", "-s"]
        success = run_command(command, f"Performance Benchmarks: {test_file}")
        return [(test_file, success)]
    else:
        print(f"⚠️  Performance test file not found: {test_file}")
        return [(test_file, False)]

def run_code_quality_checks():
    """Run code quality checks."""
    results = []
    
    # Check if tutorial modules can be imported
    tutorial_modules = [
        "strands_captcha_integration.py",
        "strands_captcha_solver.py",
        "strands_captcha_vision.py",
        "strands_orchestration_layer.py",
        "strands_error_handling.py"
    ]
    
    print("\nChecking tutorial module imports...")
    for module_file in tutorial_modules:
        if Path(module_file).exists():
            module_name = module_file.replace('.py', '')
            try:
                # Try to import the module
                command = ["python", "-c", f"import {module_name}; print('✅ {module_name} imported successfully')"]
                success = run_command(command, f"Import check: {module_name}")
                results.append((module_file, success))
            except Exception as e:
                print(f"❌ Failed to import {module_name}: {e}")
                results.append((module_file, False))
        else:
            print(f"⚠️  Module file not found: {module_file}")
            results.append((module_file, False))
    
    return results

def generate_test_report(validation_results, performance_results, quality_results):
    """Generate a comprehensive test report."""
    print(f"\n{'='*80}")
    print("STRANDS CAPTCHA TUTORIAL TEST REPORT")
    print(f"{'='*80}")
    
    # Summary
    total_tests = len(validation_results) + len(performance_results) + len(quality_results)
    passed_tests = sum(1 for _, success in validation_results + performance_results + quality_results if success)
    
    print(f"\nSUMMARY:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Failed: {total_tests - passed_tests}")
    print(f"  Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    # Detailed results
    print(f"\nTUTORIAL VALIDATION TESTS:")
    for test_file, success in validation_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {test_file}")
    
    print(f"\nPERFORMANCE BENCHMARK TESTS:")
    for test_file, success in performance_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {test_file}")
    
    print(f"\nCODE QUALITY CHECKS:")
    for module_file, success in quality_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {module_file}")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    if passed_tests == total_tests:
        print("  🎉 All tests passed! Tutorial is ready for use.")
    else:
        print("  🔧 Some tests failed. Please review the failures above.")
        print("  📝 Check that all tutorial modules are properly implemented.")
        print("  🔍 Verify that mock dependencies are correctly configured.")
        print("  ⚡ Review performance benchmarks for optimization opportunities.")
    
    return passed_tests == total_tests

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run Strands CAPTCHA tutorial tests")
    parser.add_argument("--validation", action="store_true", help="Run validation tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--quality", action="store_true", help="Run quality checks only")
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    
    args = parser.parse_args()
    
    # Default to all tests if no specific test type is specified
    if not any([args.validation, args.performance, args.quality]):
        args.all = True
    
    print("Strands CAPTCHA Tutorial Test Runner")
    print("====================================")
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        return 1
    
    validation_results = []
    performance_results = []
    quality_results = []
    
    # Run selected test suites
    if args.validation or args.all:
        validation_results = run_tutorial_validation_tests()
    
    if args.performance or args.all:
        performance_results = run_performance_benchmarks()
    
    if args.quality or args.all:
        quality_results = run_code_quality_checks()
    
    # Generate report
    all_passed = generate_test_report(validation_results, performance_results, quality_results)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())