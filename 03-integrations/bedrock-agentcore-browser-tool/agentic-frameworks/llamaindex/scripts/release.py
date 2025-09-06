#!/usr/bin/env python3
"""
Release automation script for llamaindex-agentcore-browser-integration.

This script handles version bumping, changelog generation, git tagging,
and package publishing.
"""

import sys
import subprocess
import re
import argparse
from pathlib import Path
from typing import Tuple, Optional, List
import json
from datetime import datetime

class ReleaseError(Exception):
    """Custom exception for release errors."""
    pass

class ReleaseManager:
    """Manages package releases."""
    
    def __init__(self, verbose: bool = False, dry_run: bool = False):
        self.verbose = verbose
        self.dry_run = dry_run
        self.root_dir = Path(__file__).parent.parent
        self.version_file = self.root_dir / "llamaindex_agentcore_integration" / "_version.py"
        self.changelog_file = self.root_dir / "CHANGELOG.md"
        
    def log(self, message: str, level: str = "INFO") -> None:
        """Log release messages."""
        prefix = "[DRY RUN] " if self.dry_run else ""
        if self.verbose or level in ["ERROR", "WARNING"]:
            print(f"{prefix}[{level}] {message}")
    
    def get_current_version(self) -> str:
        """Get current version from version file."""
        if not self.version_file.exists():
            raise ReleaseError(f"Version file not found: {self.version_file}")
        
        with open(self.version_file, "r") as f:
            content = f.read()
        
        match = re.search(r'__version__ = ["\']([^"\']+)["\']', content)
        if not match:
            raise ReleaseError("Could not find version in version file")
        
        return match.group(1)
    
    def parse_version(self, version: str) -> Tuple[int, int, int]:
        """Parse version string into components."""
        match = re.match(r'^(\d+)\.(\d+)\.(\d+)(?:-.*)?$', version)
        if not match:
            raise ReleaseError(f"Invalid version format: {version}")
        
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    
    def bump_version(self, current_version: str, bump_type: str) -> str:
        """Bump version based on type."""
        major, minor, patch = self.parse_version(current_version)
        
        if bump_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump_type == "minor":
            minor += 1
            patch = 0
        elif bump_type == "patch":
            patch += 1
        else:
            raise ReleaseError(f"Invalid bump type: {bump_type}")
        
        return f"{major}.{minor}.{patch}"
    
    def update_version_file(self, new_version: str) -> None:
        """Update version in version file."""
        if not self.version_file.exists():
            raise ReleaseError(f"Version file not found: {self.version_file}")
        
        with open(self.version_file, "r") as f:
            content = f.read()
        
        # Update version
        content = re.sub(
            r'__version__ = ["\'][^"\']+["\']',
            f'__version__ = "{new_version}"',
            content
        )
        
        # Update version info tuple
        major, minor, patch = self.parse_version(new_version)
        content = re.sub(
            r'__version_info__ = \([^)]+\)',
            f'__version_info__ = ({major}, {minor}, {patch})',
            content
        )
        
        # Update build date
        build_date = datetime.now().strftime("%Y-%m-%d")
        content = re.sub(
            r'BUILD_DATE = ["\'][^"\']+["\']',
            f'BUILD_DATE = "{build_date}"',
            content
        )
        
        if not self.dry_run:
            with open(self.version_file, "w") as f:
                f.write(content)
        
        self.log(f"Updated version to {new_version}")
    
    def get_git_commits_since_tag(self, tag: str) -> List[str]:
        """Get git commits since last tag."""
        try:
            result = subprocess.run(
                ["git", "log", f"{tag}..HEAD", "--oneline"],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.root_dir
            )
            return [line.strip() for line in result.stdout.split('\n') if line.strip()]
        except subprocess.CalledProcessError:
            return []
    
    def get_latest_tag(self) -> Optional[str]:
        """Get latest git tag."""
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.root_dir
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None
    
    def generate_changelog_entry(self, version: str, commits: List[str]) -> str:
        """Generate changelog entry from commits."""
        date = datetime.now().strftime("%Y-%m-%d")
        
        entry = f"\n## [{version}] - {date}\n\n"
        
        # Categorize commits
        features = []
        fixes = []
        docs = []
        other = []
        
        for commit in commits:
            commit_lower = commit.lower()
            if any(keyword in commit_lower for keyword in ["feat:", "feature:", "add:"]):
                features.append(commit)
            elif any(keyword in commit_lower for keyword in ["fix:", "bug:", "patch:"]):
                fixes.append(commit)
            elif any(keyword in commit_lower for keyword in ["doc:", "docs:", "readme:"]):
                docs.append(commit)
            else:
                other.append(commit)
        
        if features:
            entry += "### Added\n"
            for commit in features:
                entry += f"- {commit}\n"
            entry += "\n"
        
        if fixes:
            entry += "### Fixed\n"
            for commit in fixes:
                entry += f"- {commit}\n"
            entry += "\n"
        
        if docs:
            entry += "### Documentation\n"
            for commit in docs:
                entry += f"- {commit}\n"
            entry += "\n"
        
        if other:
            entry += "### Other\n"
            for commit in other:
                entry += f"- {commit}\n"
            entry += "\n"
        
        return entry
    
    def update_changelog(self, version: str) -> None:
        """Update changelog with new version."""
        latest_tag = self.get_latest_tag()
        
        if latest_tag:
            commits = self.get_git_commits_since_tag(latest_tag)
        else:
            # First release, get all commits
            try:
                result = subprocess.run(
                    ["git", "log", "--oneline"],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=self.root_dir
                )
                commits = [line.strip() for line in result.stdout.split('\n') if line.strip()]
            except subprocess.CalledProcessError:
                commits = []
        
        if not commits:
            self.log("No commits found for changelog", "WARNING")
            return
        
        new_entry = self.generate_changelog_entry(version, commits)
        
        if self.changelog_file.exists():
            with open(self.changelog_file, "r") as f:
                existing_content = f.read()
            
            # Insert new entry after the header
            lines = existing_content.split('\n')
            header_end = 0
            for i, line in enumerate(lines):
                if line.startswith('## '):
                    header_end = i
                    break
            
            new_content = '\n'.join(lines[:header_end]) + new_entry + '\n'.join(lines[header_end:])
        else:
            new_content = f"# Changelog\n\nAll notable changes to this project will be documented in this file.\n{new_entry}"
        
        if not self.dry_run:
            with open(self.changelog_file, "w") as f:
                f.write(new_content)
        
        self.log(f"Updated changelog with {len(commits)} commits")
    
    def check_git_status(self) -> None:
        """Check git status for uncommitted changes."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.root_dir
            )
            
            if result.stdout.strip():
                raise ReleaseError(
                    "Git working directory is not clean. "
                    "Please commit or stash changes before releasing."
                )
        except subprocess.CalledProcessError as e:
            raise ReleaseError(f"Failed to check git status: {e}")
    
    def create_git_tag(self, version: str) -> None:
        """Create git tag for release."""
        tag_name = f"v{version}"
        
        try:
            if not self.dry_run:
                # Commit version changes
                subprocess.run(
                    ["git", "add", str(self.version_file), str(self.changelog_file)],
                    check=True,
                    cwd=self.root_dir
                )
                
                subprocess.run(
                    ["git", "commit", "-m", f"Release {version}"],
                    check=True,
                    cwd=self.root_dir
                )
                
                # Create tag
                subprocess.run(
                    ["git", "tag", "-a", tag_name, "-m", f"Release {version}"],
                    check=True,
                    cwd=self.root_dir
                )
            
            self.log(f"Created git tag: {tag_name}")
            
        except subprocess.CalledProcessError as e:
            raise ReleaseError(f"Failed to create git tag: {e}")
    
    def build_package(self) -> None:
        """Build package for release."""
        self.log("Building package...")
        
        try:
            if not self.dry_run:
                subprocess.run(
                    [sys.executable, "scripts/build.py", "--verbose"],
                    check=True,
                    cwd=self.root_dir
                )
            
            self.log("Package built successfully")
            
        except subprocess.CalledProcessError as e:
            raise ReleaseError(f"Failed to build package: {e}")
    
    def publish_package(self, repository: str = "pypi") -> None:
        """Publish package to repository."""
        self.log(f"Publishing package to {repository}...")
        
        try:
            if not self.dry_run:
                cmd = [sys.executable, "-m", "twine", "upload"]
                
                if repository == "testpypi":
                    cmd.extend(["--repository", "testpypi"])
                
                cmd.append("dist/*")
                
                subprocess.run(cmd, check=True, cwd=self.root_dir)
            
            self.log(f"Package published to {repository}")
            
        except subprocess.CalledProcessError as e:
            raise ReleaseError(f"Failed to publish package: {e}")
    
    def push_to_remote(self) -> None:
        """Push commits and tags to remote."""
        self.log("Pushing to remote repository...")
        
        try:
            if not self.dry_run:
                # Push commits
                subprocess.run(
                    ["git", "push"],
                    check=True,
                    cwd=self.root_dir
                )
                
                # Push tags
                subprocess.run(
                    ["git", "push", "--tags"],
                    check=True,
                    cwd=self.root_dir
                )
            
            self.log("Pushed to remote repository")
            
        except subprocess.CalledProcessError as e:
            raise ReleaseError(f"Failed to push to remote: {e}")
    
    def release(self, 
                bump_type: str,
                repository: str = "pypi",
                skip_build: bool = False,
                skip_publish: bool = False,
                skip_push: bool = False) -> None:
        """Run complete release process."""
        try:
            self.log("Starting release process...")
            
            # Check git status
            if not self.dry_run:
                self.check_git_status()
            
            # Get current version and bump
            current_version = self.get_current_version()
            new_version = self.bump_version(current_version, bump_type)
            
            self.log(f"Releasing version {new_version} (was {current_version})")
            
            # Update version file
            self.update_version_file(new_version)
            
            # Update changelog
            self.update_changelog(new_version)
            
            # Create git tag
            self.create_git_tag(new_version)
            
            # Build package
            if not skip_build:
                self.build_package()
            
            # Publish package
            if not skip_publish:
                self.publish_package(repository)
            
            # Push to remote
            if not skip_push:
                self.push_to_remote()
            
            self.log(f"Release {new_version} completed successfully!")
            self._print_release_summary(new_version, repository)
            
        except ReleaseError as e:
            self.log(f"Release failed: {e}", "ERROR")
            sys.exit(1)
        except Exception as e:
            self.log(f"Unexpected error during release: {e}", "ERROR")
            sys.exit(1)
    
    def _print_release_summary(self, version: str, repository: str) -> None:
        """Print release summary."""
        print("\n" + "="*60)
        print("RELEASE COMPLETE!")
        print("="*60)
        print(f"\nVersion: {version}")
        print(f"Repository: {repository}")
        print(f"Git tag: v{version}")
        print(f"\nNext steps:")
        print("- Verify package on PyPI")
        print("- Update documentation if needed")
        print("- Announce release")
        print("="*60)

def main():
    """Main release entry point."""
    parser = argparse.ArgumentParser(
        description="Release llamaindex-agentcore-browser-integration package"
    )
    parser.add_argument(
        "bump_type",
        choices=["major", "minor", "patch"],
        help="Version bump type"
    )
    parser.add_argument(
        "--repository",
        choices=["pypi", "testpypi"],
        default="pypi",
        help="Repository to publish to (default: pypi)"
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip building package"
    )
    parser.add_argument(
        "--skip-publish",
        action="store_true",
        help="Skip publishing package"
    )
    parser.add_argument(
        "--skip-push",
        action="store_true",
        help="Skip pushing to remote"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    manager = ReleaseManager(verbose=args.verbose, dry_run=args.dry_run)
    manager.release(
        bump_type=args.bump_type,
        repository=args.repository,
        skip_build=args.skip_build,
        skip_publish=args.skip_publish,
        skip_push=args.skip_push
    )

if __name__ == "__main__":
    main()