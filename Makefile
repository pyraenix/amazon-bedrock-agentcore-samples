# Tutorial Validation Framework Makefile

.PHONY: help install uninstall check test clean lint format docs

# Default target
help:
	@echo "Tutorial Validation Framework - Available Commands:"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  install     - Install the validation framework"
	@echo "  uninstall   - Uninstall the validation framework"
	@echo "  check       - Check installation status"
	@echo ""
	@echo "Development:"
	@echo "  test        - Run unit tests"
	@echo "  lint        - Run code linting"
	@echo "  format      - Format code with black"
	@echo "  clean       - Clean temporary files"
	@echo ""
	@echo "Validation:"
	@echo "  validate    - Run validation on all frameworks"
	@echo "  dry-run     - Show what would be executed"
	@echo "  diagnose    - Run system diagnostics"
	@echo ""
	@echo "Configuration:"
	@echo "  config      - Generate default configuration"
	@echo "  check-config - Validate configuration file"
	@echo ""

# Installation targets
install:
	@echo "Installing Tutorial Validation Framework..."
	python3 setup_validation_framework.py --install --verbose

uninstall:
	@echo "Uninstalling Tutorial Validation Framework..."
	python3 setup_validation_framework.py --uninstall --verbose

check:
	@echo "Checking installation status..."
	python3 setup_validation_framework.py --check

# Development targets
test:
	@echo "Running unit tests..."
	python3 -m pytest tutorial_validation_framework/testing/test_*.py -v

lint:
	@echo "Running code linting..."
	python3 -m flake8 tutorial_validation_framework/ --max-line-length=100
	python3 -m pylint tutorial_validation_framework/ --disable=C0114,C0115,C0116

format:
	@echo "Formatting code..."
	python3 -m black tutorial_validation_framework/ --line-length=100
	python3 -m isort tutorial_validation_framework/

clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

# Validation targets
validate:
	@echo "Running validation on all frameworks..."
	python3 -m tutorial_validation_framework.cli --all

validate-parallel:
	@echo "Running validation in parallel..."
	python3 -m tutorial_validation_framework.cli --all --parallel

validate-sequential:
	@echo "Running validation sequentially..."
	python3 -m tutorial_validation_framework.cli --all --sequential

dry-run:
	@echo "Dry run - showing execution plan..."
	python3 -m tutorial_validation_framework.cli --all --dry-run

diagnose:
	@echo "Running system diagnostics..."
	python3 setup_validation_framework.py --diagnose --verbose

# Configuration targets
config:
	@echo "Generating default configuration..."
	python3 -m tutorial_validation_framework.cli --generate-config config/validation_config.yaml

check-config:
	@echo "Validating configuration..."
	python3 setup_validation_framework.py --validate-config config/validation_config.yaml

# Framework-specific targets
validate-novaact:
	@echo "Validating NovaAct framework..."
	python3 -m tutorial_validation_framework.cli --frameworks novaact

validate-browseruse:
	@echo "Validating BrowserUse framework..."
	python3 -m tutorial_validation_framework.cli --frameworks browseruse

validate-strands:
	@echo "Validating Strands framework..."
	python3 -m tutorial_validation_framework.cli --frameworks strands

validate-llamaindex:
	@echo "Validating LlamaIndex framework..."
	python3 -m tutorial_validation_framework.cli --frameworks llamaindex

# CI/CD targets
ci-test:
	@echo "Running CI tests..."
	python3 -m tutorial_validation_framework.cli --all --ci-mode --output-format junit

ci-validate:
	@echo "Running CI validation..."
	python3 -m tutorial_validation_framework.cli --all --ci-mode --timeout 15 --fail-fast

# Utility targets
list-frameworks:
	@echo "Available frameworks:"
	python3 -m tutorial_validation_framework.cli --list-frameworks

interactive:
	@echo "Starting interactive mode..."
	python3 -m tutorial_validation_framework.cli --interactive

# Documentation targets
docs:
	@echo "Generating documentation..."
	@echo "Documentation generation not implemented yet"

# Quick setup for new users
quick-setup: install config
	@echo ""
	@echo "Quick setup completed!"
	@echo "Run 'make dry-run' to test your setup"
	@echo "Run 'make validate' to start validation"

# Development setup
dev-setup: install
	@echo "Installing development dependencies..."
	pip3 install pytest flake8 pylint black isort
	@echo "Development setup completed!"