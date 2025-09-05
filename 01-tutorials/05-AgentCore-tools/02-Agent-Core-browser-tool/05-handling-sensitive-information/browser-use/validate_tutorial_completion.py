#!/usr/bin/env python3
"""
Browser-Use AgentCore Tutorial Completion Validation Script

This script validates that users have successfully completed the browser-use with
AgentCore Browser Tool sensitive information handling tutorial.
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import nbformat
from nbformat import v4 as nbf
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TutorialCompletionValidator:
    """Validates completion of browser-use with AgentCore tutorial."""
    
    def __init__(self):
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "tutorial_completion": {},
            "notebook_analysis": {},
            "example_execution": {},
            "test_results": {},
            "issues": [],
            "recommendations": [],
            "completion_score": 0
        }
        
        # Define expected tutorial components
        self.expected_notebooks = [
            "browseruse_agentcore_secure_connection_tutorial.ipynb",
            "browseruse_pii_masking_tutorial.ipynb", 
            "browseruse_compliance_audit_tutorial.ipynb",
            "browseruse_production_deployment_tutorial.ipynb"
        ]
        
        self.expected_examples = [
            "examples/healthcare_form_automation.py",
            "examples/financial_form_security.py",
            "examples/credential_management.py"
        ]
        
        self.expected_tests = [
            "tests/test_browseruse_agentcore_integration.py",
            "tests/test_browseruse_sensitive_data_handling.py",
            "tests/test_browseruse_security_boundaries.py"
        ]
    
    def check_notebook_exists_and_executed(self, notebook_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if notebook exists and has been executed."""
        
        notebook_info = {
            "exists": False,
            "executed": False,
            "cell_count": 0,
            "executed_cells": 0,
            "has_outputs": False,
            "last_modified": None,
            "errors": []
        }
        
        try:
            path = Path(notebook_path)
            if not path.exists():
                notebook_info["errors"].append(f"Notebook {notebook_path} not found")
                return False, notebook_info
            
            notebook_info["exists"] = True
            notebook_info["last_modified"] = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
            
            # Read and analyze notebook
            with open(path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            notebook_info["cell_count"] = len(nb.cells)
            
            executed_cells = 0
            has_outputs = False
            
            for cell in nb.cells:
                if cell.cell_type == 'code':
                    # Check if cell has been executed (has execution_count)
                    if cell.get('execution_count') is not None:
                        executed_cells += 1
                    
                    # Check if cell has outputs
                    if cell.get('outputs'):
                        has_outputs = True
            
            notebook_info["executed_cells"] = executed_cells
            notebook_info["has_outputs"] = has_outputs
            notebook_info["executed"] = executed_cells > 0 and has_outputs
            
            return True, notebook_info
            
        except Exception as e:
            notebook_info["errors"].append(f"Error analyzing notebook {notebook_path}: {str(e)}")
            return False, notebook_info
    
    def validate_notebook_content(self, notebook_path: str) -> Dict[str, Any]:
        """Validate notebook content for tutorial-specific requirements."""
        
        content_validation = {
            "has_agentcore_imports": False,
            "has_browser_use_imports": False,
            "has_sensitive_data_handling": False,
            "has_session_management": False,
            "has_security_validation": False,
            "key_concepts_covered": [],
            "errors": []
        }
        
        try:
            path = Path(notebook_path)
            if not path.exists():
                content_validation["errors"].append(f"Notebook {notebook_path} not found")
                return content_validation
            
            with open(path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # Analyze notebook content
            all_source = ""
            for cell in nb.cells:
                if cell.cell_type == 'code':
                    all_source += cell.source + "\n"
            
            # Check for key imports and concepts
            if "bedrock_agentcore" in all_source or "BrowserClient" in all_source:
                content_validation["has_agentcore_imports"] = True
                content_validation["key_concepts_covered"].append("AgentCore Integration")
            
            if "browser_use" in all_source or "Agent" in all_source:
                content_validation["has_browser_use_imports"] = True
                content_validation["key_concepts_covered"].append("Browser-Use Framework")
            
            if "mask_pii" in all_source or "sensitive" in all_source.lower():
                content_validation["has_sensitive_data_handling"] = True
                content_validation["key_concepts_covered"].append("Sensitive Data Handling")
            
            if "create_session" in all_source or "session_id" in all_source:
                content_validation["has_session_management"] = True
                content_validation["key_concepts_covered"].append("Session Management")
            
            if "compliance" in all_source.lower() or "security" in all_source.lower():
                content_validation["has_security_validation"] = True
                content_validation["key_concepts_covered"].append("Security & Compliance")
            
        except Exception as e:
            content_validation["errors"].append(f"Error validating content for {notebook_path}: {str(e)}")
        
        return content_validation
    
    async def validate_notebooks(self) -> bool:
        """Validate all tutorial notebooks."""
        logger.info("Validating tutorial notebooks...")
        
        all_notebooks_valid = True
        
        for notebook in self.expected_notebooks:
            logger.info(f"Checking notebook: {notebook}")
            
            exists, info = self.check_notebook_exists_and_executed(notebook)
            content_validation = self.validate_notebook_content(notebook)
            
            notebook_result = {
                "exists": exists,
                "info": info,
                "content": content_validation,
                "score": 0
            }
            
            # Calculate score for this notebook
            score = 0
            if exists:
                score += 20  # Base score for existence
                
                if info["executed"]:
                    score += 30  # Bonus for execution
                
                if info["executed_cells"] > info["cell_count"] * 0.5:
                    score += 20  # Bonus for executing most cells
                
                # Content validation scoring
                content_score = sum([
                    10 if content_validation["has_agentcore_imports"] else 0,
                    10 if content_validation["has_browser_use_imports"] else 0,
                    10 if content_validation["has_sensitive_data_handling"] else 0,
                    5 if content_validation["has_session_management"] else 0,
                    5 if content_validation["has_security_validation"] else 0
                ])
                score += content_score
            
            notebook_result["score"] = min(score, 100)  # Cap at 100
            
            self.validation_results["notebook_analysis"][notebook] = notebook_result
            
            if not exists or not info["executed"]:
                all_notebooks_valid = False
                self.validation_results["issues"].append(f"Notebook {notebook} not completed")
        
        return all_notebooks_valid
    
    async def validate_examples(self) -> bool:
        """Validate tutorial examples."""
        logger.info("Validating tutorial examples...")
        
        all_examples_valid = True
        
        for example in self.expected_examples:
            logger.info(f"Checking example: {example}")
            
            example_path = Path(example)
            example_result = {
                "exists": example_path.exists(),
                "executable": False,
                "has_main_function": False,
                "imports_valid": False,
                "score": 0
            }
            
            if example_path.exists():
                try:
                    with open(example_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for key components
                    if "def main(" in content or "if __name__ == '__main__':" in content:
                        example_result["has_main_function"] = True
                    
                    if ("bedrock_agentcore" in content or "browser_use" in content):
                        example_result["imports_valid"] = True
                    
                    # Try to validate syntax
                    try:
                        compile(content, example_path, 'exec')
                        example_result["executable"] = True
                    except SyntaxError:
                        example_result["executable"] = False
                    
                    # Calculate score
                    score = 25  # Base score for existence
                    if example_result["has_main_function"]:
                        score += 25
                    if example_result["imports_valid"]:
                        score += 25
                    if example_result["executable"]:
                        score += 25
                    
                    example_result["score"] = score
                    
                except Exception as e:
                    example_result["score"] = 10  # Minimal score for existence
                    self.validation_results["issues"].append(f"Error analyzing example {example}: {str(e)}")
            else:
                all_examples_valid = False
                self.validation_results["issues"].append(f"Example {example} not found")
            
            self.validation_results["example_execution"][example] = example_result
        
        return all_examples_valid
    
    async def validate_tests(self) -> bool:
        """Validate tutorial tests."""
        logger.info("Validating tutorial tests...")
        
        all_tests_valid = True
        
        for test_file in self.expected_tests:
            logger.info(f"Checking test: {test_file}")
            
            test_path = Path(test_file)
            test_result = {
                "exists": test_path.exists(),
                "has_test_functions": False,
                "imports_valid": False,
                "executable": False,
                "score": 0
            }
            
            if test_path.exists():
                try:
                    with open(test_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for test functions
                    if "def test_" in content or "async def test_" in content:
                        test_result["has_test_functions"] = True
                    
                    # Check for required imports
                    if ("pytest" in content or "unittest" in content) and ("bedrock_agentcore" in content or "browser_use" in content):
                        test_result["imports_valid"] = True
                    
                    # Try to validate syntax
                    try:
                        compile(content, test_path, 'exec')
                        test_result["executable"] = True
                    except SyntaxError:
                        test_result["executable"] = False
                    
                    # Calculate score
                    score = 25  # Base score for existence
                    if test_result["has_test_functions"]:
                        score += 25
                    if test_result["imports_valid"]:
                        score += 25
                    if test_result["executable"]:
                        score += 25
                    
                    test_result["score"] = score
                    
                except Exception as e:
                    test_result["score"] = 10  # Minimal score for existence
                    self.validation_results["issues"].append(f"Error analyzing test {test_file}: {str(e)}")
            else:
                all_tests_valid = False
                self.validation_results["issues"].append(f"Test file {test_file} not found")
            
            self.validation_results["test_results"][test_file] = test_result
        
        return all_tests_valid
    
    def check_tutorial_artifacts(self) -> bool:
        """Check for tutorial completion artifacts."""
        logger.info("Checking tutorial completion artifacts...")
        
        artifacts = {
            "setup_report_exists": Path("setup_report.json").exists(),
            "integration_report_exists": Path("integration_validation_report.json").exists(),
            "logs_directory_exists": Path("logs").exists(),
            "assets_directory_exists": Path("assets").exists(),
            "readme_exists": Path("README.md").exists()
        }
        
        self.validation_results["tutorial_completion"]["artifacts"] = artifacts
        
        # Check if user has created any custom configurations
        custom_configs = {
            "custom_env_file": Path(".env").exists() and Path(".env").stat().st_size > 0,
            "custom_examples": len(list(Path("examples").glob("*.py"))) > len(self.expected_examples) if Path("examples").exists() else False,
            "additional_notebooks": len(list(Path(".").glob("*.ipynb"))) > len(self.expected_notebooks)
        }
        
        self.validation_results["tutorial_completion"]["custom_work"] = custom_configs
        
        return all(artifacts.values())
    
    def calculate_overall_completion_score(self) -> int:
        """Calculate overall tutorial completion score."""
        
        # Notebook scores (40% of total)
        notebook_scores = [result["score"] for result in self.validation_results["notebook_analysis"].values()]
        avg_notebook_score = sum(notebook_scores) / len(notebook_scores) if notebook_scores else 0
        notebook_contribution = (avg_notebook_score * 0.4)
        
        # Example scores (25% of total)
        example_scores = [result["score"] for result in self.validation_results["example_execution"].values()]
        avg_example_score = sum(example_scores) / len(example_scores) if example_scores else 0
        example_contribution = (avg_example_score * 0.25)
        
        # Test scores (25% of total)
        test_scores = [result["score"] for result in self.validation_results["test_results"].values()]
        avg_test_score = sum(test_scores) / len(test_scores) if test_scores else 0
        test_contribution = (avg_test_score * 0.25)
        
        # Artifacts (10% of total)
        artifacts = self.validation_results["tutorial_completion"].get("artifacts", {})
        artifact_score = (sum(artifacts.values()) / len(artifacts) * 100) if artifacts else 0
        artifact_contribution = (artifact_score * 0.1)
        
        total_score = notebook_contribution + example_contribution + test_contribution + artifact_contribution
        
        self.validation_results["completion_score"] = int(total_score)
        
        return int(total_score)
    
    def generate_completion_report(self) -> Dict[str, Any]:
        """Generate comprehensive completion report."""
        
        # Calculate completion score
        completion_score = self.calculate_overall_completion_score()
        
        # Determine completion level
        if completion_score >= 90:
            completion_level = "🏆 EXCELLENT - Tutorial Mastered"
        elif completion_score >= 75:
            completion_level = "🎯 GOOD - Tutorial Completed"
        elif completion_score >= 60:
            completion_level = "📚 FAIR - Partial Completion"
        elif completion_score >= 40:
            completion_level = "⚠️ NEEDS WORK - Basic Understanding"
        else:
            completion_level = "❌ INCOMPLETE - Needs Attention"
        
        self.validation_results["completion_level"] = completion_level
        
        # Generate recommendations based on score
        if completion_score < 90:
            if any(not result["info"]["executed"] for result in self.validation_results["notebook_analysis"].values()):
                self.validation_results["recommendations"].append("Execute all notebook cells to complete the tutorials")
            
            if any(not result["exists"] for result in self.validation_results["example_execution"].values()):
                self.validation_results["recommendations"].append("Review and run all example scripts")
            
            if any(not result["exists"] for result in self.validation_results["test_results"].values()):
                self.validation_results["recommendations"].append("Complete the testing exercises")
        
        return self.validation_results
    
    def print_completion_report(self):
        """Print formatted completion report."""
        
        print("\n" + "="*80)
        print("🎓 BROWSER-USE WITH AGENTCORE - TUTORIAL COMPLETION VALIDATION")
        print("="*80)
        
        print(f"\n📊 COMPLETION SUMMARY:")
        print(f"   Completion Level: {self.validation_results['completion_level']}")
        print(f"   Overall Score: {self.validation_results['completion_score']}/100")
        
        print(f"\n📚 NOTEBOOK ANALYSIS:")
        for notebook, result in self.validation_results["notebook_analysis"].items():
            status = "✅" if result["exists"] and result["info"]["executed"] else "❌"
            print(f"   {status} {notebook}: {result['score']}/100")
            if result["content"]["key_concepts_covered"]:
                print(f"      Concepts: {', '.join(result['content']['key_concepts_covered'])}")
        
        print(f"\n🔧 EXAMPLE VALIDATION:")
        for example, result in self.validation_results["example_execution"].items():
            status = "✅" if result["exists"] and result["executable"] else "❌"
            print(f"   {status} {example}: {result['score']}/100")
        
        print(f"\n🧪 TEST VALIDATION:")
        for test, result in self.validation_results["test_results"].items():
            status = "✅" if result["exists"] and result["has_test_functions"] else "❌"
            print(f"   {status} {test}: {result['score']}/100")
        
        artifacts = self.validation_results["tutorial_completion"].get("artifacts", {})
        if artifacts:
            print(f"\n📁 TUTORIAL ARTIFACTS:")
            for artifact, exists in artifacts.items():
                status = "✅" if exists else "❌"
                print(f"   {status} {artifact.replace('_', ' ').title()}")
        
        if self.validation_results["issues"]:
            print(f"\n⚠️ ISSUES FOUND:")
            for i, issue in enumerate(self.validation_results["issues"], 1):
                print(f"   {i}. {issue}")
        
        if self.validation_results["recommendations"]:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(self.validation_results["recommendations"], 1):
                print(f"   {i}. {rec}")
        
        print(f"\n🎯 NEXT STEPS:")
        if self.validation_results["completion_score"] >= 90:
            print("   🏆 Congratulations! You've mastered the tutorial!")
            print("   🚀 You're ready to implement browser-use with AgentCore in production")
            print("   📖 Consider exploring advanced AgentCore features")
        elif self.validation_results["completion_score"] >= 75:
            print("   🎯 Great job! You've completed the tutorial successfully")
            print("   🔍 Review any remaining issues for deeper understanding")
            print("   🚀 You're ready to start building with browser-use and AgentCore")
        else:
            print("   📚 Continue working through the tutorial materials")
            print("   🔧 Focus on the areas identified in the recommendations")
            print("   🔄 Run this validation again after making progress")
        
        print("\n" + "="*80)
    
    def save_completion_report(self, filename: str = "tutorial_completion_report.json"):
        """Save completion report to JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.validation_results, f, indent=2)
            logger.info(f"Completion report saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save completion report: {str(e)}")

async def main():
    """Main tutorial completion validation function."""
    
    print("🎓 Starting Browser-Use with AgentCore tutorial completion validation...")
    
    validator = TutorialCompletionValidator()
    
    # Run all validation checks
    await validator.validate_notebooks()
    await validator.validate_examples()
    await validator.validate_tests()
    validator.check_tutorial_artifacts()
    
    # Generate and display report
    validator.generate_completion_report()
    validator.print_completion_report()
    
    # Save report
    validator.save_completion_report()
    
    # Return success status based on completion score
    return validator.validation_results["completion_score"] >= 75

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Tutorial completion validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Tutorial completion validation failed with error: {str(e)}")
        logger.error(f"Validation error traceback: {traceback.format_exc()}")
        sys.exit(1)