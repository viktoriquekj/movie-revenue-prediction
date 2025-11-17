"""
movie_revenue_prediction

Package containing code for the Movie Revenue Prediction project:
- data fetching (TMDB API)
- feature engineering
- model training (simple models, neural networks, ensembles)
- deployment and prediction pipelines
"""

# Do NOT import subpackages at the top level here.
# This avoids eager imports and circular import issues.

__all__ = ["api", "features", "models", "utils", "deployment"]

# Optional: a simple version string
__version__ = "0.1.0"
