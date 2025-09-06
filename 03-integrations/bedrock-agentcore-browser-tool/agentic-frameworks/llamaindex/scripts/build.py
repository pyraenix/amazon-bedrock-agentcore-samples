#!/usr/bin/env python3
"""
Build script for llamaindex-agentcore-browser-integration package.

This script handles building distribution packages, running tests,
and preparing releases.
"""

import sys
import subprocess
import shutil
import os
import argparse
from pathlib import Path
from typing import List, Optional
import json
import tempfile

class BuildError(Exception):
    """Custom exception for build errors."""
    pass

class PackageBuilder:
    """Package builder for distribution."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.root_dir = Path(__file__).parent.parent
        self.dist_dir = self.root_dir / "dist"
        self.build_dir = self.root_dir / "build"
        
    def log(self, message: str, level: str = "INFO") -> None:
        """Log build messages."""
        if self.verbose or level in ["ERROR", "WARNING"]:
            print(f"[{level}] {message}")
    
    def clean(self) -> None:
        """Clean build artifacts."""
        self.log("Cleaning build artifacts...")
        
        dirs_to_clean = [
            self.dist_dir,
            self.build_dir,
            self.root_dir / "*.egg-info",
            self.root_dir / "__pycache__",
            self.root_dir / ".pytest_cache",
            self.root_dir / ".mypy_cache",
            self.root_dir / ".coverage",
            self.root_dir / "htmlcov"
        ]
        
        for dir_path in dirs_to_clean:
            if dir_path.exists():
                if dir_path.is_dir():
                    shutil.rmtree(dir_path)
                    self.log(f"Removed directory: {dir_path}")
                else:
                    # Handle glob patterns
                    for path in self.root_dir.glob(dir_path.name):
                        if path.is_dir():
                            shutil.rmtree(path)
                        else:
                            path.unlink()
                        self.log(f"Removed: {path}")
    
    def run_tests(self) -> bool:
        """Run test suite."""
        self.log("Running test suite...")
        
        try:
            cmd = [
                sys.executable, "-m", "pytest",
                "tests/",
                "--cov=llamaindex_agentcore_integration",
                "--cov-report=html",
                "--cov-report=term-missing",
                "--junitxml=test-results.xml",
                "-v"
            ]
            
            if not self.verbose:
                cmd.append("--quiet")
            
            result = subprocess.run(
                cmd,
                cwd=self.root_dir,
                check=True
            )
            
            self.log("All tests passed")
            return True
            
        except subprocess.CalledProcessError as e:
            self.log(f"Tests failed: {e}", "ERROR")
            return False
    
    def run_linting(self) -> bool:
        """Run code linting and formatting checks."""
        self.log("Running code quality checks...")
        
        checks = [
            # Black formatting check
            ([sys.executable, "-m", "black", "--check", "."], "Black formatting"),
            # isort import sorting check
            ([sys.executable, "-m", "isort", "--check-only", "."], "Import sorting"),
            # Flake8 linting
            ([sys.executable, "-m", "flake8", "."], "Flake8 linting"),
            # MyPy type checking
            ([sys.executable, "-m", "mypy", "llamaindex_agentcore_integration"], "Type checking"),
            # Ruff linting
            ([sys.executable, "-m", "ruff", "check", "."], "Ruff linting"),
        ]
        
        all_passed = True
        
        for cmd, description in checks:
            try:
                self.log(f"Running {description}...")
                subprocess.run(
                    cmd,
                    cwd=self.root_dir,
                    check=True,
                    capture_output=not self.verbose
                )
                self.log(f"{description} passed")
                
            except subprocess.CalledProcessError as e:
                self.log(f"{description} failed: {e}", "ERROR")
                all_passed = False
        
        return all_passed
    
    def run_security_checks(self) -> bool:
        """Run security checks."""
        self.log("Running security checks...")
        
        checks = [
            # Bandit security linting
            ([sys.executable, "-m", "bandit", "-r", "llamaindex_agentcore_integration"], "Bandit security scan"),
            # Safety dependency vulnerability check
            ([sys.executable, "-m", "safety", "check"], "Safety vulnerability scan"),
        ]
        
        all_passed = True
        
        for cmd, description in checks:
            try:
                self.log(f"Running {description}...")
                subprocess.run(
                    cmd,
                    cwd=self.root_dir,
                    check=True,
                    capture_output=not self.verbose
                )
                self.log(f"{description} passed")
                
            except subprocess.CalledProcessError as e:
                self.log(f"{description} failed: {e}", "WARNING")
                # Don't fail build on security warnings, just log them
        
        return all_passed
    
    def build_wheel(self) -> Path:
        """Build wheel distribution."""
        self.log("Building wheel distribution...")
        
        try:
            subprocess.run(
                [sys.executable, "-m", "build", "--wheel"],
                cwd=self.root_dir,
                check=True,
                capture_output=not self.verbose
            )
            
            # Find the built wheel
            wheel_files = list(self.dist_dir.glob("*.whl"))
            if not wheel_files:
                raise BuildError("No wheel file found after build")
            
            wheel_path = wheel_files[0]
            self.log(f"Wheel built: {wheel_path}")
            return wheel_path
            
        except subprocess.CalledProcessError as e:
            raise BuildError(f"Failed to build wheel: {e}")
    
    def build_sdist(self) -> Path:
        """Build source distribution."""
        self.log("Building source distribution...")
        
        try:
            subprocess.run(
                [sys.executable, "-m", "build", "--sdist"],
                cwd=self.root_dir,
                check=True,
                capture_output=not self.verbose
            )
            
            # Find the built sdist
            sdist_files = list(self.dist_dir.glob("*.tar.gz"))
            if not sdist_files:
                raise BuildError("No source distribution found after build")
            
            sdist_path = sdist_files[0]
            self.log(f"Source distribution built: {sdist_path}")
            return sdist_path
            
        except subprocess.CalledProcessError as e:
            raise BuildError(f"Failed to build source distribution: {e}")
    
    def validate_distributions(self, wheel_path: Path, sdist_path: Path) -> bool:
        """Validate built distributions."""
        self.log("Validating distributions...")
        
        try:
            # Check wheel with twine
            subprocess.run(
                [sys.executable, "-m", "twine", "check", str(wheel_path)],
                check=True,
                capture_output=not self.verbose
            )
            
            # Check sdist with twine
            subprocess.run(
                [sys.executable, "-m", "twine", "check", str(sdist_path)],
                check=True,
                capture_output=not self.verbose
            )
            
            self.log("Distribution validation passed")
            return True
            
        except subprocess.CalledProcessError as e:
            self.log(f"Distribution validation failed: {e}", "ERROR")
            return False
    
    def test_installation(self, wheel_path: Path) -> bool:
        """Test installation of built wheel in clean environment."""
        self.log("Testing installation in clean environment...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Create virtual environment
                venv_dir = Path(temp_dir) / "test_venv"
                subprocess.run(
                    [sys.executable, "-m", "venv", str(venv_dir)],
                    check=True,
                    capture_output=not self.verbose
                )
                
                # Get python executable in venv
                if sys.platform == "win32":
                    venv_python = venv_dir / "Scripts" / "python.exe"
                else:
                    venv_python = venv_dir / "bin" / "python"
                
                # Install wheel
                subprocess.run(
                    [str(venv_python), "-m", "pip", "install", str(wheel_path)],
                    check=True,
                    capture_output=not self.verbose
                )
                
                # Test import
                subprocess.run(
                    [str(venv_python), "-c", 
                     "import llamaindex_agentcore_integration; print('Import successful')"],
                    check=True,
                    capture_output=not self.verbose
                )
                
                self.log("Installation test passed")
                return True
                
            except subprocess.CalledProcessError as e:
                self.log(f"Installation test failed: {e}", "ERROR")
                return False
    
    def generate_build_info(self, wheel_path: Path, sdist_path: Path) -> None:
        """Generate build information file."""
        build_info = {
            "wheel": {
                "path": str(wheel_path),
                "size": wheel_path.stat().st_size,
                "name": wheel_path.name
            },
            "sdist": {
                "path": str(sdist_path),
                "size": sdist_path.stat().st_size,
                "name": sdist_path.name
            },
            "build_timestamp": subprocess.run(
                ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                capture_output=True,
                text=True
            ).stdout.strip(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform
        }
        
        build_info_path = self.dist_dir / "build_info.json"
        with open(build_info_path, "w") as f:
            json.dump(build_info, f, indent=2)
        
        self.log(f"Build info saved: {build_info_path}")
    
    def build(self, 
              run_tests: bool = True,
              run_linting: bool = True,
              run_security: bool = True,
              test_install: bool = True) -> None:
        """Run complete build process."""
        try:
            self.log("Starting build process...")
            
            # Clean previous builds
            self.clean()
            
            # Run quality checks
            if run_linting:
                if not self.run_linting():
                    raise BuildError("Code quality checks failed")
            
            if run_security:
                self.run_security_checks()  # Don't fail on security warnings
            
            if run_tests:
                if not self.run_tests():
                    raise BuildError("Tests failed")
            
            # Build distributions
            wheel_path = self.build_wheel()
            sdist_path = self.build_sdist()
            
            # Validate distributions
            if not self.validate_distributions(wheel_path, sdist_path):
                raise BuildError("Distribution validation failed")
            
            # Test installation
            if test_install:
                if not self.test_installation(wheel_path):
                    raise BuildError("Installation test failed")
            
            # Generate build info
            self.generate_build_info(wheel_path, sdist_path)
            
            self.log("Build completed successfully!")
            self._print_build_summary(wheel_path, sdist_path)
            
        except BuildError as e:
            self.log(f"Build failed: {e}", "ERROR")
            sys.exit(1)
        except Exception as e:
            self.log(f"Unexpected error during build: {e}", "ERROR")
            sys.exit(1)
    
    def _print_build_summary(self, wheel_path: Path, sdist_path: Path) -> None:
        """Print build summary."""
        print("\n" + "="*60)
        print("BUILD COMPLETE!")
        print("="*60)
        print(f"\nBuilt distributions:")
        print(f"  Wheel: {wheel_path}")
        print(f"  Source: {sdist_path}")
        print(f"\nTo install:")
        print(f"  pip install {wheel_path}")
        print(f"\nTo upload to PyPI:")
        print(f"  twine upload {self.dist_dir}/*")
        print("="*60)

def main():
    """Main build entry point."""
    parser = argparse.ArgumentParser(
        description="Build llamaindex-agentcore-browser-integration package"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests"
    )
    parser.add_argument(
        "--skip-linting",
        action="store_true",
        help="Skip code quality checks"
    )
    parser.add_argument(
        "--skip-security",
        action="store_true",
        help="Skip security checks"
    )
    parser.add_argument(
        "--skip-install-test",
        action="store_true",
        help="Skip installation test"
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Only clean build artifacts"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    builder = PackageBuilder(verbose=args.verbose)
    
    if args.clean_only:
        builder.clean()
        return
    
    builder.build(
        run_tests=not args.skip_tests,
        run_linting=not args.skip_linting,
        run_security=not args.skip_security,
        test_install=not args.skip_install_test
    )

if __name__ == "__main__":
    main()