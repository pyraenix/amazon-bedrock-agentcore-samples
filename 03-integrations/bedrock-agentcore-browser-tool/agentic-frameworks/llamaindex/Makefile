# Makefile for llamaindex-agentcore-browser-integration

.PHONY: help install install-dev install-prod clean test test-unit test-integration lint format type-check security-check build release docs serve-docs

# Default target
help:
	@echo "Available targets:"
	@echo "  install          - Install package in development mode"
	@echo "  install-dev      - Install with development dependencies"
	@echo "  install-prod     - Install with production dependencies"
	@echo "  clean            - Clean build artifacts and cache"
	@echo "  test             - Run all tests"
	@echo "  test-unit        - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  lint             - Run all linting checks"
	@echo "  format           - Format code with black and isort"
	@echo "  type-check       - Run type checking with mypy"
	@echo "  security-check   - Run security checks"
	@echo "  build            - Build distribution packages"
	@echo "  release          - Create a new release"
	@echo "  docs             - Build documentation"
	@echo "  serve-docs       - Serve documentation locally"

# Python executable
PYTHON := python3.12
PIP := $(PYTHON) -m pip

# Installation targets
install:
	$(PYTHON) install.py --environment=dev --verbose

install-dev:
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	pre-commit install

install-prod:
	$(PIP) install -r requirements-prod.txt
	$(PIP) install .

# Clean targets
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# Testing targets
test:
	$(PYTHON) -m pytest tests/ -v --cov=llamaindex_agentcore_integration --cov-report=html --cov-report=term-missing

test-unit:
	$(PYTHON) -m pytest tests/ -v -m "unit" --cov=llamaindex_agentcore_integration

test-integration:
	$(PYTHON) -m pytest tests/ -v -m "integration" --cov=llamaindex_agentcore_integration

test-performance:
	$(PYTHON) -m pytest tests/ -v -m "performance" --benchmark-only

test-security:
	$(PYTHON) -m pytest tests/ -v -m "security"

# Code quality targets
lint: format type-check security-check
	@echo "All linting checks completed"

format:
	@echo "Formatting code..."
	$(PYTHON) -m black .
	$(PYTHON) -m isort .
	$(PYTHON) -m ruff check . --fix

type-check:
	@echo "Running type checks..."
	$(PYTHON) -m mypy llamaindex_agentcore_integration

security-check:
	@echo "Running security checks..."
	$(PYTHON) -m bandit -r llamaindex_agentcore_integration
	$(PYTHON) -m safety check

# Build targets
build:
	$(PYTHON) scripts/build.py --verbose

build-wheel:
	$(PYTHON) -m build --wheel

build-sdist:
	$(PYTHON) -m build --sdist

# Release targets
release-patch:
	$(PYTHON) scripts/release.py patch --verbose

release-minor:
	$(PYTHON) scripts/release.py minor --verbose

release-major:
	$(PYTHON) scripts/release.py major --verbose

release-dry-run:
	$(PYTHON) scripts/release.py patch --dry-run --verbose

# Documentation targets
docs:
	@echo "Building documentation..."
	cd docs && $(PYTHON) -m sphinx -b html . _build/html

serve-docs:
	@echo "Serving documentation at http://localhost:8000"
	cd docs/_build/html && $(PYTHON) -m http.server 8000

# Development targets
dev-setup: install-dev
	@echo "Setting up development environment..."
	cp config.template.yaml config.yaml
	@echo "Development environment ready!"
	@echo "Please edit config.yaml with your settings"

pre-commit-run:
	pre-commit run --all-files

pre-commit-update:
	pre-commit autoupdate

# Docker targets
docker-build:
	docker build -t llamaindex-agentcore-integration .

docker-run:
	docker run -it --rm llamaindex-agentcore-integration

# Utility targets
check-deps:
	$(PIP) list --outdated

update-deps:
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -r requirements-dev.txt

verify-install:
	$(PYTHON) -c "import llamaindex_agentcore_integration; print('Installation verified!')"

# Environment targets
create-venv:
	$(PYTHON) -m venv venv312
	@echo "Virtual environment created. Activate with:"
	@echo "source venv312/bin/activate  # Linux/Mac"
	@echo "venv312\\Scripts\\activate     # Windows"

# Benchmarking
benchmark:
	$(PYTHON) -m pytest tests/ --benchmark-only --benchmark-sort=mean

# Coverage targets
coverage:
	$(PYTHON) -m pytest tests/ --cov=llamaindex_agentcore_integration --cov-report=html
	@echo "Coverage report generated in htmlcov/"

coverage-xml:
	$(PYTHON) -m pytest tests/ --cov=llamaindex_agentcore_integration --cov-report=xml

# Profiling
profile:
	$(PYTHON) -m cProfile -o profile.stats examples/performance_test.py
	$(PYTHON) -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Quick development workflow
quick-test: format test-unit

# Full CI workflow
ci: clean install-dev lint test build

# Local development workflow
dev: clean dev-setup quick-test
	@echo "Development workflow completed!"

# Production deployment check
prod-check: clean install-prod test security-check build
	@echo "Production deployment check completed!"