"""Version information for llamaindex-agentcore-browser-integration."""

__version__ = "0.1.0"
__version_info__ = (0, 1, 0)

# Version components
MAJOR = 0
MINOR = 1
PATCH = 0

# Build metadata
BUILD_DATE = "2025-01-02"
BUILD_COMMIT = "main"

# Compatibility information
PYTHON_REQUIRES = ">=3.12,<3.13"
LLAMAINDEX_MIN_VERSION = "0.11.0"
AGENTCORE_API_VERSION = "v1"

def get_version() -> str:
    """Get the current version string."""
    return __version__

def get_version_info() -> tuple[int, int, int]:
    """Get version info as tuple."""
    return __version_info__

def is_compatible_python(version_info: tuple[int, int]) -> bool:
    """Check if Python version is compatible."""
    major, minor = version_info
    return major == 3 and minor >= 12

def get_build_info() -> dict[str, str]:
    """Get build information."""
    return {
        "version": __version__,
        "build_date": BUILD_DATE,
        "build_commit": BUILD_COMMIT,
        "python_requires": PYTHON_REQUIRES,
        "llamaindex_min_version": LLAMAINDEX_MIN_VERSION,
        "agentcore_api_version": AGENTCORE_API_VERSION
    }