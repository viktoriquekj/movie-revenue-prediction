"""
Utility functions for preprocessing, evaluation, and shared helpers.
"""

__all__ = []

# Always make the paths module available (no heavy deps here)
from . import paths  # noqa: F401
__all__.append("paths")

# Try to import functions, but don't break if heavy deps (like xgboost) fail.
try:
    from . import functions  # noqa: F401
    __all__.append("functions")
except Exception:
    # This allows light-weight uses (like importing paths) in environments
    # where xgboost or other compiled dependencies are not installed correctly.
    pass
