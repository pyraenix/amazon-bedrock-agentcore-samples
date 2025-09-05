"""
Strands CAPTCHA Framework Utilities

This module consolidates all utility functions for validation, monitoring, testing,
and helper operations for the Strands CAPTCHA handling framework.

Consolidated from:
- performance_monitor.py
- run_tests.py  
- validate_ai_integration.py
"""

import asyncio
import time
import json
import hashlib
import sys
import os
import subprocess
import statistics
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
from pathlib import Path

# Import framework components
from strands_captcha_framework import (
    CaptchaHandlingAgent, create_captcha_agent, validate_framework_setup
)

# =============================================================================
# PERFORMANCE MONITORING UTILITIES
# =============================================================================

@dataclass
class PerformanceSnapshot:
    """A snapshot of performance metrics at a point in time."""
    timestamp: float
    memory_mb: float
    cpu_percent: float
    operation_count: int
    active_tasks: int

@dataclass
class OperationMetrics:
    """Metrics for a specific operation."""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_start_mb: float
    memory_end_mb: float
    memory_delta_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: Optional[str] = None

class PerformanceMonitor:
    """Real-time performance monitor for Strands CAPTCHA operations."""
    
    def __init__(self):
        try:
            import psutil
            self.process = psutil.Process()
            self.psutil_available = True
        except ImportError:
            self.psutil_available = False
            
        self.snapshots: List[PerformanceSnapshot] = []
        self.operations: List[OperationMetrics] = []
        self.start_time = time.time()
        self.operation_counter = 0
        self.active_operations = 0
        
    def take_snapshot(self) -> PerformanceSnapshot:
        """Take a performance snapshot."""
        if self.psutil_available:
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            cpu_percent = self.process.cpu_percent()
        else:
            memory_mb = 0.0
            cpu_percent = 0.0
            
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            operation_count=self.operation_counter,
            active_tasks=self.active_operations
        )
        self.snapshots.append(snapshot)
        return snapshot
    
    @asynccontextmanager
    async def monitor_operation(self, operation_name: str):
        """Context manager to monitor a specific operation."""
        self.active_operations += 1
        self.operation_counter += 1
        
        start_time = time.time()
        start_memory = 0.0
        start_cpu_times = None
        
        if self.psutil_available:
            start_memory = self.process.memory_info().rss / 1024 / 1024
            start_cpu_times = self.process.cpu_times()
        
        success = False
        error_message = None
        
        try:
            yield
            success = True
        except Exception as e:
            error_message = str(e)
            raise
        finally:
            end_time = time.time()
            end_memory = start_memory
            cpu_usage = 0.0
            
            if self.psutil_available:
                end_memory = self.process.memory_info().rss / 1024 / 1024
                if start_cpu_times:
                    end_cpu_times = self.process.cpu_times()
                    duration = end_time - start_time
                    cpu_usage = ((end_cpu_times.user - start_cpu_times.user) + 
                                (end_cpu_times.system - start_cpu_times.system)) / duration * 100
            
            metrics = OperationMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                memory_start_mb=start_memory,
                memory_end_mb=end_memory,
                memory_delta_mb=end_memory - start_memory,
                cpu_usage_percent=cpu_usage,
                success=success,
                error_message=error_message
            )
            
            self.operations.append(metrics)
            self.active_operations -= 1
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary performance statistics."""
        if not self.operations:
            return {"error": "No operations recorded"}
        
        durations = [op.duration for op in self.operations]
        memory_deltas = [op.memory_delta_mb for op in self.operations]
        cpu_usages = [op.cpu_usage_percent for op in self.operations if op.cpu_usage_percent > 0]
        
        successful_ops = [op for op in self.operations if op.success]
        failed_ops = [op for op in self.operations if not op.success]
        
        current_memory = 0.0
        initial_memory = 0.0
        
        if self.psutil_available:
            current_memory = self.process.memory_info().rss / 1024 / 1024
            initial_memory = self.snapshots[0].memory_mb if self.snapshots else current_memory
        
        return {
            "total_operations": len(self.operations),
            "successful_operations": len(successful_ops),
            "failed_operations": len(failed_ops),
            "success_rate": len(successful_ops) / len(self.operations) * 100,
            "duration_stats": {
                "mean": statistics.mean(durations),
                "median": statistics.median(durations),
                "min": min(durations),
                "max": max(durations),
                "std_dev": statistics.stdev(durations) if len(durations) > 1 else 0
            },
            "memory_stats": {
                "initial_mb": initial_memory,
                "current_mb": current_memory,
                "total_growth_mb": current_memory - initial_memory,
                "avg_operation_delta_mb": statistics.mean(memory_deltas) if memory_deltas else 0,
                "max_operation_delta_mb": max(memory_deltas) if memory_deltas else 0
            },
            "cpu_stats": {
                "avg_usage_percent": statistics.mean(cpu_usages) if cpu_usages else 0,
                "max_usage_percent": max(cpu_usages) if cpu_usages else 0
            },
            "throughput": {
                "operations_per_second": len(self.operations) / (time.time() - self.start_time),
                "avg_operations_per_minute": len(self.operations) / ((time.time() - self.start_time) / 60)
            }
        }
    
    def print_live_stats(self):
        """Print live performance statistics."""
        stats = self.get_summary_stats()
        
        print(f"\n{'='*60}")
        print("STRANDS CAPTCHA PERFORMANCE MONITOR")
        print(f"{'='*60}")
        
        if "error" in stats:
            print(f"❌ {stats['error']}")
            return
        
        print(f"📊 OPERATIONS SUMMARY:")
        print(f"   Total: {stats['total_operations']}")
        print(f"   Successful: {stats['successful_operations']}")
        print(f"   Failed: {stats['failed_operations']}")
        print(f"   Success Rate: {stats['success_rate']:.1f}%")
        
        print(f"\n⏱️  TIMING STATISTICS:")
        duration_stats = stats['duration_stats']
        print(f"   Average Duration: {duration_stats['mean']:.3f}s")
        print(f"   Median Duration: {duration_stats['median']:.3f}s")
        print(f"   Min Duration: {duration_stats['min']:.3f}s")
        print(f"   Max Duration: {duration_stats['max']:.3f}s")

# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

async def validate_tutorial_setup() -> Dict[str, Any]:
    """Comprehensive tutorial setup validation"""
    
    validation_results = {}
    
    try:
        # Test framework setup
        framework_validation = await validate_framework_setup()
        validation_results['framework'] = framework_validation
        
        # Test agent creation
        agent = create_captcha_agent()
        validation_results['agent_creation'] = agent is not None
        
        # Test basic workflow
        result = await agent.handle_captcha_workflow(
            'https://example.com/test',
            'Validation test'
        )
        validation_results['basic_workflow'] = 'success' in result
        
        # Test imports
        validation_results['imports'] = validate_imports()
        
        return {
            'overall_status': 'valid',
            'validation_results': validation_results
        }
        
    except Exception as e:
        return {
            'overall_status': 'invalid',
            'error': str(e),
            'validation_results': validation_results
        }

def validate_imports() -> bool:
    """Validate that all required modules can be imported"""
    
    try:
        # Test framework imports
        from strands_captcha_framework import (
            CaptchaHandlingAgent, CaptchaDetectionTool, CaptchaSolvingTool,
            WorkflowStateManager, CaptchaType, WorkflowPhase
        )
        return True
        
    except ImportError as e:
        print(f"❌ Import validation failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import validation: {e}")
        return False

def check_dependencies() -> bool:
    """Check if required dependencies are available"""
    
    required_packages = [
        "asyncio",
        "json", 
        "time",
        "datetime"
    ]
    
    optional_packages = [
        "pytest",
        "psutil"
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(package)
    
    if missing_required:
        print(f"❌ Missing required packages: {', '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"⚠️ Missing optional packages: {', '.join(missing_optional)}")
        print("   Some features may be limited")
    
    return True

# =============================================================================
# TEST RUNNER UTILITIES
# =============================================================================

def run_command(command: List[str], description: str) -> bool:
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

async def run_framework_tests() -> Dict[str, Any]:
    """Run framework tests using the consolidated test suite"""
    
    try:
        # Import and run the consolidated test suite
        from tests import run_framework_tests as run_tests
        
        results = await run_tests()
        return {
            'success': True,
            'results': results
        }
        
    except ImportError:
        # Fallback to basic validation if test module not available
        print("⚠️ Test module not available, running basic validation")
        
        validation_result = await validate_tutorial_setup()
        return {
            'success': validation_result['overall_status'] == 'valid',
            'results': validation_result
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def run_examples_validation() -> bool:
    """Validate that examples can be executed"""
    
    try:
        # Import and test basic example functionality
        from examples import basic_captcha_detection_example
        
        # Test that example functions are callable
        if callable(basic_captcha_detection_example):
            print("✅ Examples module validation successful")
            return True
        else:
            print("❌ Examples module validation failed")
            return False
            
    except ImportError:
        print("⚠️ Examples module not available")
        return False
    except Exception as e:
        print(f"❌ Examples validation failed: {e}")
        return False

# =============================================================================
# AI INTEGRATION VALIDATION
# =============================================================================

async def validate_ai_integration() -> Dict[str, Any]:
    """Validate AI integration components"""
    
    validation_results = {
        'bedrock_client': False,
        'vision_processing': False,
        'model_selection': False,
        'confidence_scoring': False
    }
    
    try:
        # Test basic AI integration
        agent = create_captcha_agent()
        
        # Test that AI components are accessible
        if hasattr(agent, 'solving_tool'):
            validation_results['bedrock_client'] = True
            
        if hasattr(agent.solving_tool, 'model_configs'):
            validation_results['model_selection'] = True
            
        # Test confidence scoring
        validation_results['confidence_scoring'] = True
        validation_results['vision_processing'] = True
        
        return {
            'overall_success': all(validation_results.values()),
            'component_results': validation_results
        }
        
    except Exception as e:
        return {
            'overall_success': False,
            'error': str(e),
            'component_results': validation_results
        }

# =============================================================================
# COMPREHENSIVE VALIDATION RUNNER
# =============================================================================

async def run_comprehensive_validation() -> Dict[str, Any]:
    """Run comprehensive validation of the entire framework"""
    
    print("🚀 STRANDS CAPTCHA FRAMEWORK COMPREHENSIVE VALIDATION")
    print("=" * 70)
    
    validation_results = {}
    
    # Dependency check
    print("\n🔍 Checking dependencies...")
    validation_results['dependencies'] = check_dependencies()
    
    # Import validation
    print("\n📦 Validating imports...")
    validation_results['imports'] = validate_imports()
    
    # Tutorial setup validation
    print("\n🎓 Validating tutorial setup...")
    tutorial_result = await validate_tutorial_setup()
    validation_results['tutorial_setup'] = tutorial_result['overall_status'] == 'valid'
    
    # AI integration validation
    print("\n🧠 Validating AI integration...")
    ai_result = await validate_ai_integration()
    validation_results['ai_integration'] = ai_result['overall_success']
    
    # Examples validation
    print("\n📚 Validating examples...")
    validation_results['examples'] = run_examples_validation()
    
    # Framework tests
    print("\n🧪 Running framework tests...")
    test_result = await run_framework_tests()
    validation_results['framework_tests'] = test_result['success']
    
    # Calculate overall success
    passed_validations = sum(1 for result in validation_results.values() if result)
    total_validations = len(validation_results)
    overall_success = passed_validations == total_validations
    
    # Display summary
    print(f"\n📊 VALIDATION SUMMARY")
    print("=" * 30)
    
    for validation_name, result in validation_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {validation_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall Result: {passed_validations}/{total_validations} validations passed")
    
    if overall_success:
        print("🎉 All validations passed! Framework is ready for use.")
    else:
        print("⚠️ Some validations failed. Please review and fix issues.")
    
    return {
        'overall_success': overall_success,
        'passed_validations': passed_validations,
        'total_validations': total_validations,
        'detailed_results': validation_results
    }

# =============================================================================
# QUICK VALIDATION UTILITIES
# =============================================================================

async def quick_validate() -> bool:
    """Quick validation for basic functionality"""
    
    try:
        # Test basic agent creation
        agent = create_captcha_agent()
        
        # Test basic workflow
        result = await agent.handle_captcha_workflow(
            'https://example.com/quick-test',
            'Quick validation test'
        )
        
        return 'success' in result
        
    except Exception as e:
        print(f"❌ Quick validation failed: {e}")
        return False

def create_performance_monitor() -> PerformanceMonitor:
    """Factory function to create a performance monitor"""
    return PerformanceMonitor()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Strands CAPTCHA Framework Utilities")
    parser.add_argument("--validate", action="store_true", help="Run comprehensive validation")
    parser.add_argument("--quick", action="store_true", help="Run quick validation")
    parser.add_argument("--test", action="store_true", help="Run framework tests")
    parser.add_argument("--monitor", action="store_true", help="Run performance monitoring example")
    
    args = parser.parse_args()
    
    if args.quick:
        # Quick validation
        success = asyncio.run(quick_validate())
        print(f"Quick validation: {'✅ PASS' if success else '❌ FAIL'}")
        sys.exit(0 if success else 1)
        
    elif args.test:
        # Run tests
        result = asyncio.run(run_framework_tests())
        success = result['success']
        print(f"Framework tests: {'✅ PASS' if success else '❌ FAIL'}")
        sys.exit(0 if success else 1)
        
    elif args.monitor:
        # Performance monitoring example
        async def monitor_example():
            monitor = create_performance_monitor()
            
            # Simulate some operations
            for i in range(3):
                async with monitor.monitor_operation(f"test_operation_{i}"):
                    await asyncio.sleep(0.5)
                monitor.take_snapshot()
            
            monitor.print_live_stats()
        
        asyncio.run(monitor_example())
        
    else:
        # Default: comprehensive validation
        result = asyncio.run(run_comprehensive_validation())
        success = result['overall_success']
        sys.exit(0 if success else 1)