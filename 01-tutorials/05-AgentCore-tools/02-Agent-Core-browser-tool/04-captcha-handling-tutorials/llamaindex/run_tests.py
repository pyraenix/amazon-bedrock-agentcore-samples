#!/usr/bin/env python3
"""
Test runner script for LlamaIndex CAPTCHA integration tutorial.

This script runs all test scenarios and provides comprehensive validation
of the tutorial components and examples.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import json


class TestRunner:
    """Test runner for LlamaIndex CAPTCHA tutorial validation."""
    
    def __init__(self, test_dir: str = "tests"):
        self.test_dir = Path(test_dir)
        self.results = {}
        self.start_time = None
        self.end_time = None
        
    def run_all_tests(self, verbose: bool = False, coverage: bool = False) -> Dict[str, Any]:
        """Run all test suites and return results."""
        print("🚀 Starting LlamaIndex CAPTCHA Tutorial Test Suite")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Test suites to run
        test_suites = [
            ("Tutorial Validation", "test_tutorial_validation.py"),
            ("CAPTCHA Site Testing", "test_captcha_sites.py"),
            ("Error Handling", "test_error_handling.py")
        ]
        
        for suite_name, test_file in test_suites:
            print(f"\n📋 Running {suite_name}...")
            result = self._run_test_suite(test_file, verbose, coverage)
            self.results[suite_name] = result
            
            if result["success"]:
                print(f"✅ {suite_name}: PASSED ({result['duration']:.2f}s)")
            else:
                print(f"❌ {suite_name}: FAILED ({result['duration']:.2f}s)")
                
        self.end_time = time.time()
        
        # Run additional validations
        self._validate_examples()
        self._validate_notebook()
        
        return self._generate_summary()
        
    def _run_test_suite(self, test_file: str, verbose: bool, coverage: bool) -> Dict[str, Any]:
        """Run a specific test suite."""
        test_path = self.test_dir / test_file
        
        if not test_path.exists():
            return {
                "success": False,
                "error": f"Test file {test_file} not found",
                "duration": 0,
                "output": ""
            }
            
        # Build pytest command
        cmd = ["python3", "-m", "pytest", str(test_path)]
        
        if verbose:
            cmd.append("-v")
            
        if coverage:
            cmd.extend(["--cov=../", "--cov-report=term-missing"])
            
        # Add timeout
        cmd.extend(["--timeout=300"])
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.test_dir.parent,
                timeout=600  # 10 minute timeout
            )
            
            duration = time.time() - start_time
            
            return {
                "success": result.returncode == 0,
                "duration": duration,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return {
                "success": False,
                "error": "Test suite timed out",
                "duration": duration,
                "output": ""
            }
        except Exception as e:
            duration = time.time() - start_time
            return {
                "success": False,
                "error": str(e),
                "duration": duration,
                "output": ""
            }
            
    def _validate_examples(self) -> None:
        """Validate that example scripts can be imported."""
        print("\n📁 Validating Example Scripts...")
        
        examples_dir = self.test_dir.parent / "examples"
        
        if not examples_dir.exists():
            print("⚠️  Examples directory not found")
            return
            
        example_files = list(examples_dir.glob("*.py"))
        
        for example_file in example_files:
            try:
                # Try to compile the file
                with open(example_file, 'r') as f:
                    code = f.read()
                    
                compile(code, str(example_file), 'exec')
                print(f"✅ {example_file.name}: Syntax OK")
                
            except SyntaxError as e:
                print(f"❌ {example_file.name}: Syntax Error - {e}")
            except Exception as e:
                print(f"⚠️  {example_file.name}: Warning - {e}")
                
    def _validate_notebook(self) -> None:
        """Validate the main tutorial notebook."""
        print("\n📓 Validating Tutorial Notebook...")
        
        notebook_path = self.test_dir.parent / "llamaindex-captcha.ipynb"
        
        if not notebook_path.exists():
            print("⚠️  Tutorial notebook not found")
            return
            
        try:
            with open(notebook_path, 'r') as f:
                notebook_data = json.load(f)
                
            # Basic validation
            if 'cells' not in notebook_data:
                print("❌ Notebook: Missing cells")
                return
                
            cell_count = len(notebook_data['cells'])
            code_cells = sum(1 for cell in notebook_data['cells'] if cell.get('cell_type') == 'code')
            markdown_cells = sum(1 for cell in notebook_data['cells'] if cell.get('cell_type') == 'markdown')
            
            print(f"✅ Notebook: {cell_count} total cells ({code_cells} code, {markdown_cells} markdown)")
            
            # Check for key content
            all_source = []
            for cell in notebook_data['cells']:
                if 'source' in cell:
                    all_source.extend(cell['source'])
                    
            combined_source = ''.join(all_source)
            
            key_terms = ['CAPTCHA', 'LlamaIndex', 'Bedrock', 'AgentCore']
            missing_terms = [term for term in key_terms if term not in combined_source]
            
            if missing_terms:
                print(f"⚠️  Notebook: Missing key terms: {missing_terms}")
            else:
                print("✅ Notebook: Contains all key terms")
                
        except json.JSONDecodeError:
            print("❌ Notebook: Invalid JSON format")
        except Exception as e:
            print(f"❌ Notebook: Validation error - {e}")
            
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test summary report."""
        total_duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        passed_suites = sum(1 for result in self.results.values() if result["success"])
        total_suites = len(self.results)
        
        summary = {
            "total_duration": total_duration,
            "total_suites": total_suites,
            "passed_suites": passed_suites,
            "failed_suites": total_suites - passed_suites,
            "success_rate": passed_suites / total_suites if total_suites > 0 else 0,
            "results": self.results
        }
        
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Duration: {total_duration:.2f} seconds")
        print(f"Test Suites: {passed_suites}/{total_suites} passed")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        
        if summary['success_rate'] == 1.0:
            print("🎉 All tests passed! Tutorial is ready for use.")
        else:
            print("⚠️  Some tests failed. Please review the results above.")
            
        return summary
        
    def run_specific_test(self, test_name: str, verbose: bool = False) -> Dict[str, Any]:
        """Run a specific test suite."""
        print(f"🎯 Running specific test: {test_name}")
        
        test_file = f"test_{test_name}.py"
        result = self._run_test_suite(test_file, verbose, False)
        
        if result["success"]:
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
            if result["error"]:
                print(f"Error: {result['error']}")
                
        return result
        
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance and benchmark tests."""
        print("⚡ Running Performance Tests...")
        
        # This would run performance-specific tests
        # For now, just run regular tests with timing
        return self.run_all_tests(verbose=True, coverage=False)


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="LlamaIndex CAPTCHA Tutorial Test Runner")
    
    parser.add_argument(
        "--test",
        choices=["all", "tutorial", "sites", "errors", "performance"],
        default="all",
        help="Which test suite to run"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Enable coverage reporting"
    )
    
    parser.add_argument(
        "--output",
        help="Output file for test results (JSON format)"
    )
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    runner = TestRunner()
    
    # Run tests based on selection
    if args.test == "all":
        results = runner.run_all_tests(args.verbose, args.coverage)
    elif args.test == "performance":
        results = runner.run_performance_tests()
    else:
        test_mapping = {
            "tutorial": "tutorial_validation",
            "sites": "captcha_sites", 
            "errors": "error_handling"
        }
        results = runner.run_specific_test(test_mapping[args.test], args.verbose)
        
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")
        
    # Exit with appropriate code
    if isinstance(results, dict) and results.get("success_rate", 0) == 1.0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()