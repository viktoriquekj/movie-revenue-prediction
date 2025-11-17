# src/movie_revenue_prediction/utils/paths.py
from pathlib import Path

# This file is located at: project_root/src/movie_revenue_prediction/utils/paths.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
CURATED_DIR = DATA_DIR / "curated"
RAW_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
RESULTS_DIR = PROJECT_ROOT / "results"
