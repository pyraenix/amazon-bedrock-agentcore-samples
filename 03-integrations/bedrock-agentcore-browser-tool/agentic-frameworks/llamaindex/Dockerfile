# Multi-stage Dockerfile for llamaindex-agentcore-browser-integration
# Optimized for production deployment

# Build stage
FROM python:3.12-slim as builder

# Set build arguments
ARG BUILD_ENV=production
ARG PYTHON_VERSION=3.12

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements*.txt ./
COPY pyproject.toml setup.py MANIFEST.in ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements-prod.txt

# Copy source code
COPY llamaindex_agentcore_integration/ ./llamaindex_agentcore_integration/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY docs/ ./docs/
COPY examples/ ./examples/

# Install the package
RUN pip install -e .

# Production stage
FROM python:3.12-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LLAMAINDEX_AGENTCORE_ENV=production
ENV PATH="/app/.local/bin:$PATH"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create app directory and set ownership
WORKDIR /app
RUN chown -R appuser:appuser /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder --chown=appuser:appuser /app ./

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data /app/tmp \
    && chown -R appuser:appuser /app/logs /app/data /app/tmp

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import llamaindex_agentcore_integration; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "llamaindex_agentcore_integration"]

# Development stage
FROM builder as development

# Install development dependencies
RUN pip install -r requirements-dev.txt

# Install additional development tools
RUN pip install ipython jupyter

# Set development environment
ENV LLAMAINDEX_AGENTCORE_ENV=development
ENV PYTHONPATH=/app

# Expose port for development server
EXPOSE 8000

# Development command
CMD ["python", "-m", "llamaindex_agentcore_integration", "--debug"]

# Testing stage
FROM development as testing

# Copy test files
COPY tests/ ./tests/
COPY pytest.ini ./

# Run tests
RUN python -m pytest tests/ -v

# Default to production stage
FROM production