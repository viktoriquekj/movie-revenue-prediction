"""
Deployment utilities for Mixed Model C (cyc: EN + RF + XGB + NN_SWA).

This file:
- Loads the trained base models and meta-learner.
- Reapplies the same preprocessing as in training:
  * make_features_from_artifacts (top-k lists)
  * winsorization (saved thresholds)
  * time-aware priors (directors, lead_cast, composers)
  * cyclical time features
- Produces predictions for new data (e.g. 2025 movies).
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from movie_revenue_prediction.utils.paths import ARTIFACTS_DIR

from tensorflow import keras

from movie_revenue_prediction.features.build_features import make_features_from_artifacts

from movie_revenue_prediction.utils.functions import (
    list_columns_to_pipe,
    apply_winsor_from_thresholds,
    apply_timeaware_prior,
    add_time_cyclical,
)
from movie_revenue_prediction.models.ensemble_model import predict_with_ensemble_multiX
# -------------------------------------------------------------------
# Paths / config
# -------------------------------------------------------------------

ENSEMBLE_DIR: Path = ARTIFACTS_DIR / "ensemble_C"
PREPROC_DIR: Path = ENSEMBLE_DIR / "preprocessing"


# -------------------------------------------------------------------
# 1. Load all artifacts needed for Mixed Model C
# -------------------------------------------------------------------
def load_model_C_artifacts(ensemble_dir: Path | None = None):
    """
    Load everything needed for Mixed Model C (cyc: EN + RF + XGB + NN_SWA).

    Returns:
      - base_pipes_C: dict[name -> fitted Pipeline], with keys:
          {"EN", "RF", "XGB", "NN_SWA"}
      - meta_C:       meta-learner (e.g. Ridge) with model_names_ set
      - feat_artifacts: dict used by make_features_from_artifacts
      - winsor_thresholds: dict[col -> (ql, qh)]
      - feature_cols_cyc: list of cyclical feature columns (final training order)
      - priors: dict with "directors", "lead_cast", "composers" prior tables
    """
    if ensemble_dir is None:
        ensemble_dir = ENSEMBLE_DIR
    ensemble_dir = Path(ensemble_dir)

    base_dir = ensemble_dir / "base_models"
    preproc_dir = ensemble_dir / "preprocessing"

    # --- base models ---
    en_pipe  = joblib.load(base_dir / "EN_cyc.pkl")
    rf_pipe  = joblib.load(base_dir / "RF_cyc.pkl")
    xgb_pipe = joblib.load(base_dir / "XGB_cyc.pkl")

    # NN: load skeleton pipeline + Keras model weights
    nn_pipe_skel = joblib.load(base_dir / "NN_SWA_cyc_pipe.pkl")
    nn_model     = keras.models.load_model(base_dir / "NN_SWA_cyc.keras")
    nn_pipe_skel.named_steps["nn"].model_ = nn_model

    # Use training-time model names as keys
    base_pipes_C = {
        "EN":     en_pipe,
        "RF":     rf_pipe,
        "XGB":    xgb_pipe,
        "NN_SWA": nn_pipe_skel,
    }

    # --- meta-learner ---
    meta_C = joblib.load(ensemble_dir / "meta_learner.pkl")
    with (ensemble_dir / "meta_info.json").open("r") as f:
        meta_info = json.load(f)
    meta_C.model_names_ = meta_info["model_names"]

    # --- preprocessing artifacts ---
    with (preproc_dir / "topk_lists.json").open("r") as f:
        feat_artifacts = json.load(f)

    with (preproc_dir / "winsor_thresholds.json").open("r") as f:
        winsor_thresholds = json.load(f)

    with (preproc_dir / "feature_cols_cyc.json").open("r") as f:
        feature_cols_cyc = json.load(f)

    prior_dir  = pd.read_parquet(preproc_dir / "prior_directors.parquet")
    prior_cast = pd.read_parquet(preproc_dir / "prior_lead_cast.parquet")
    prior_comp = pd.read_parquet(preproc_dir / "prior_composers.parquet")

    priors = {
        "directors": prior_dir,
        "lead_cast": prior_cast,
        "composers": prior_comp,
    }

    return base_pipes_C, meta_C, feat_artifacts, winsor_thresholds, feature_cols_cyc, priors

# def load_model_C_artifacts(ensemble_dir: str = ENSEMBLE_DIR):
#     """
#     Load everything needed for Mixed Model C (cyc: EN + RF + XGB + NN_SWA).

#     Returns:
#       - base_pipes_C: dict[name -> fitted Pipeline], with keys:
#           {"EN", "RF", "XGB", "NN_SWA"}
#       - meta_C:       meta-learner (e.g. Ridge) with model_names_ set
#       - feat_artifacts: dict used by make_features_from_artifacts
#       - winsor_thresholds: dict[col -> (ql, qh)]
#       - feature_cols_cyc: list of cyclical feature columns (final training order)
#       - priors: dict with "directors", "lead_cast", "composers" prior tables
#     """

#     base_dir   = os.path.join(ensemble_dir, "base_models")
#     preproc_dir = os.path.join(ensemble_dir, "preprocessing")

#     # --- base models ---
#     en_pipe  = joblib.load(os.path.join(base_dir, "EN_cyc.pkl"))
#     rf_pipe  = joblib.load(os.path.join(base_dir, "RF_cyc.pkl"))
#     xgb_pipe = joblib.load(os.path.join(base_dir, "XGB_cyc.pkl"))

#     # NN: load skeleton pipeline + Keras model weights
#     nn_pipe_skel = joblib.load(os.path.join(base_dir, "NN_SWA_cyc_pipe.pkl"))
#     nn_model     = keras.models.load_model(os.path.join(base_dir, "NN_SWA_cyc.keras"))
#     nn_pipe_skel.named_steps["nn"].model_ = nn_model

#     # Use training-time model names as keys
#     base_pipes_C = {
#         "EN":     en_pipe,
#         "RF":     rf_pipe,
#         "XGB":    xgb_pipe,
#         "NN_SWA": nn_pipe_skel,
#     }

#     # --- meta-learner ---
#     meta_C = joblib.load(os.path.join(ensemble_dir, "meta_learner.pkl"))
#     with open(os.path.join(ensemble_dir, "meta_info.json"), "r") as f:
#         meta_info = json.load(f)
#     # Ensure the order of base models matches training
#     meta_C.model_names_ = meta_info["model_names"]

#     # --- preprocessing artifacts ---
#     with open(os.path.join(preproc_dir, "topk_lists.json"), "r") as f:
#         feat_artifacts = json.load(f)

#     with open(os.path.join(preproc_dir, "winsor_thresholds.json"), "r") as f:
#         winsor_thresholds = json.load(f)

#     with open(os.path.join(preproc_dir, "feature_cols_cyc.json"), "r") as f:
#         feature_cols_cyc = json.load(f)

#     prior_dir  = pd.read_parquet(os.path.join(preproc_dir, "prior_directors.parquet"))
#     prior_cast = pd.read_parquet(os.path.join(preproc_dir, "prior_lead_cast.parquet"))
#     prior_comp = pd.read_parquet(os.path.join(preproc_dir, "prior_composers.parquet"))

#     priors = {
#         "directors": prior_dir,
#         "lead_cast": prior_cast,
#         "composers": prior_comp,
#     }

#     return base_pipes_C, meta_C, feat_artifacts, winsor_thresholds, feature_cols_cyc, priors

# -------------------------------------------------------------------
# 2. Preprocess new data exactly like training
# -------------------------------------------------------------------

def preprocess_for_2025(
    df_2025_raw: pd.DataFrame,
    feat_artifacts: dict,
    winsor_thresholds: dict,
    feature_cols_cyc: list[str],
    priors: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reproduce the training-time preprocessing for new data:

      1. Ensure multi-label columns are pipe-separated.
      2. Feature engineering via make_features_from_artifacts(df, feat_artifacts)
         (uses saved top-k for genres, keywords, etc.)
      3. Winsorization using saved thresholds (same as training).
      4. Time-aware priors for directors, lead_cast, composers.
      5. Build base X from x_* columns (dropping x_weekofyear, x_quarter).
      6. Apply add_time_cyclical(X_base) to create cyclical features.
      7. Align columns to feature_cols_cyc (same order as training).

    Returns:
      - X_2025_cyc: 2D array/DataFrame with cyclical features for Mixed Model C
      - df_2025_feat: full feature DataFrame with all x_* features + priors
    """

    df = df_2025_raw.copy()

    # 1) Make sure multi-label columns are in pipe format (idempotent)
    df = list_columns_to_pipe(
        df,
        [
            "genres",
            "production_countries",
            "spoken_languages",
            "keywords",
            "directors",
            "lead_cast",
            "lead_cast_genders",
            "composers",
        ],
    )

    # 2) Features with saved top-k lists
    df_feat = make_features_from_artifacts(df, feat_artifacts)

    # 3) Winsorization with saved thresholds
    df_feat = apply_winsor_from_thresholds(df_feat, winsor_thresholds, inplace=False)

    # 4) Apply time-aware priors
    df_feat = apply_timeaware_prior(
        df_feat,
        priors["directors"],
        entity_col="directors",
        year_col="x_year",
        multi_label=False,
    )
    df_feat = apply_timeaware_prior(
        df_feat,
        priors["lead_cast"],
        entity_col="lead_cast",
        year_col="x_year",
        multi_label=True,
    )
    df_feat = apply_timeaware_prior(
        df_feat,
        priors["composers"],
        entity_col="composers",
        year_col="x_year",
        multi_label=True,
    )

    # 5) Base feature matrix: all x_* except the dropped correlation ones
    to_drop_corr = ["x_weekofyear", "x_quarter"]
    base_features = [
        c for c in df_feat.columns
        if c.startswith("x_") and c not in to_drop_corr
    ]
    X_2025_base = df_feat[base_features].copy()

    # 6) Add cyclical time features (same as during training)
    X_2025_cyc = add_time_cyclical(X_2025_base)

    # 7) Align to training cyclical feature list
    X_2025_cyc = X_2025_cyc.reindex(columns=feature_cols_cyc, fill_value=0)

    return X_2025_cyc, df_feat

# -------------------------------------------------------------------
# 3. High-level prediction function for df_2025
# -------------------------------------------------------------------

def predict_2025_with_model_C(df_2025_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point for notebooks / jobs.

    Input:
      - df_2025_raw: raw 2025 dataframe (same schema as df_union / df_all)

    Output:
      - df_out: df_2025 with engineered features + predictions:
          * y_pred_log_revenue_C
          * y_pred_revenue_C
    """

    # 1) Load artifacts (models + preprocessing info)
    (
        base_pipes_C,
        meta_C,
        feat_artifacts,
        winsor_thresholds,
        feature_cols_cyc,
        priors,
    ) = load_model_C_artifacts()

    # 2) Preprocess 2025 data
    X_2025_cyc, df_2025_feat = preprocess_for_2025(
        df_2025_raw,
        feat_artifacts=feat_artifacts,
        winsor_thresholds=winsor_thresholds,
        feature_cols_cyc=feature_cols_cyc,
        priors=priors,
    )

    # 3) Build model_to_X mapping (all models use cyclical features)
    model_names = (
        list(meta_C.model_names_)
        if hasattr(meta_C, "model_names_")
        else list(base_pipes_C.keys())
    )
    model_to_X_2025 = {name: X_2025_cyc for name in model_names}

    # 4) Predict log-revenue with ensemble
    y_pred_log = predict_with_ensemble_multiX(
        fitted_full=base_pipes_C,
        meta=meta_C,
        model_to_X=model_to_X_2025,
    )

    # 5) Back-transform to real revenue
    y_pred = np.expm1(y_pred_log)

    df_out = df_2025_feat.copy()
    df_out["y_pred_log_revenue_C"] = y_pred_log
    df_out["y_pred_revenue_C"] = y_pred

    return df_out

# -------------------------------------------------------------------
# 4. Optional CLI entry point for a quick local run
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Example: reproduce your df_2025 extraction here so you can run:
    #   python deployment.py
    from movie_revenue_prediction.utils.functions import list_columns_to_pipe

    # Load the curated dataset
    df_all = pd.read_csv("data/curated/all_ids_dataset.csv")

    # Ensure multi-label columns are in pipe format
    df_all = list_columns_to_pipe(
        df_all,
        [
            "genres",
            "production_countries",
            "spoken_languages",
            "keywords",
            "directors",
            "lead_cast",
            "lead_cast_genders",
            "composers",
        ],
    )

    # Flag collections
    df_all["is_in_collection"] = np.where(
        df_all["collection_name"].fillna("").str.strip() != "", 1, 0
    )

    # Parse dates and derive year
    df_all["release_date"] = pd.to_datetime(df_all["release_date"], errors="coerce")
    df_all["release_year"] = df_all["release_date"].dt.year

    # Same basic sanity filter as training
    df_all = df_all[
        (df_all["release_year"].between(2017, 2025, inclusive="both"))
        & (df_all["budget"] > 100)
        & (df_all["revenue"] > 100)
    ].copy()

    # Extract 2025 subset
    df_2025 = df_all[df_all["release_year"] == 2025].copy()
    print("2025 movies:", df_2025.shape)

    df_2025_scored = predict_2025_with_model_C(df_2025)
    out_path = "data/curated/df_2025_predictions_model_C.csv"
    df_2025_scored.to_csv(out_path, index=False)
    print(f"Saved 2025 predictions to: {out_path}")
    print(df_2025_scored[
        ["id", "title", "release_date", "budget", "revenue",
         "y_pred_log_revenue_C", "y_pred_revenue_C"]
    ].head())
