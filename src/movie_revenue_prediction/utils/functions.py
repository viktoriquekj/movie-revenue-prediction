from __future__ import annotations

import re
import pandas as pd
import numpy as np
import json, os
from movie_revenue_prediction.utils.paths import ARTIFACTS_DIR

import matplotlib.pyplot as plt
from datetime import datetime

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Optional, Dict, Any

from scikeras.wrappers import KerasRegressor

from movie_revenue_prediction.api.tmdb_client import TMDBClient  



# --------- Internal helpers ---------

def list_columns_to_pipe(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Convert values like:
        ['Action', 'Crime']      (real list)
        "['Action', 'Crime']"    (stringified list)
    into:
        'Action|Crime'

    - Real list/tuple/set/ndarray → joined with "|"
    - Stringified list "[A, B]" → parsed and joined with "|"
    - Already pipe strings → kept as-is
    - Empty-ish → np.nan
    """
    df_out = df.copy()

    def _to_pipe(x):
        # --- Real list-like values first ---
        if isinstance(x, (list, tuple, set, np.ndarray)):
            if isinstance(x, np.ndarray):
                x = x.tolist()
            cleaned = [str(v).strip() for v in x if str(v).strip()]
            return "|".join(cleaned) if cleaned else np.nan

        # --- String values (may be stringified lists) ---
        if isinstance(x, str):
            s = x.strip()
            if s == "" or s.lower() in ("none", "nan", "[]"):
                return np.nan

            # Stringified list: "['Action', 'Crime']"
            if s.startswith("[") and s.endswith("]"):
                inner = s[1:-1]
                parts = [
                    part.strip().strip("'\"")
                    for part in inner.split(",")
                    if part.strip().strip("'\"")
                ]
                return "|".join(parts) if parts else np.nan

            # Already pipe-delimited → leave as is
            if "|" in s:
                return s

            # Just a single string label
            return s

        # --- Missing-like / other scalars ---
        if x is None:
            return np.nan
        try:
            if pd.isna(x):
                return np.nan
        except Exception:
            pass

        s = str(x).strip()
        return s if s else np.nan

    for col in columns:
        if col in df_out.columns:
            df_out[col] = df_out[col].apply(_to_pipe)
        else:
            print(f"[warn] Column '{col}' not found — skipped.")

    return df_out



def winsorize_log_features(
    df,
    cols,
    q_low=0.01,
    q_high=None,          # <- None means "do NOT cap the upper tail"
    inplace=False,
    verbose=True,
):
    """
    Winsorize (cap) specified log-transformed numeric columns at given quantile thresholds.
    Quantiles are computed from the input DataFrame (typically your TRAIN set).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset (usually the training subset).
    cols : list of str
        List of column names to winsorize (e.g. ['x_budget_log', 'y_log_revenue']).
    q_low : float
        Lower quantile to clip at (default 1%).
    q_high : float or None
        Upper quantile to clip at. If None, no upper clipping is applied.
    inplace : bool
        If True, modifies df in place. Otherwise, returns a new DataFrame.
    verbose : bool
        If True, prints summary statistics.

    Returns
    -------
    df_winsor : pd.DataFrame
        Winsorized DataFrame (unless inplace=True).
    thresholds : dict
        Dictionary of applied thresholds {col: (low, high)}.
        high will be None if q_high is None.
    """
    df_target = df if inplace else df.copy()
    thresholds = {}

    for col in cols:
        if col not in df_target.columns:
            print(f"Warning: {col} not in DataFrame — skipped.")
            continue

        # Compute quantile bounds
        ql = df_target[col].quantile(q_low) if q_low is not None else None
        qh = df_target[col].quantile(q_high) if q_high is not None else None

        thresholds[col] = (
            float(ql) if ql is not None else None,
            float(qh) if qh is not None else None,
        )

        # Clip column
        df_target[col] = df_target[col].clip(
            lower=ql if ql is not None else None,
            upper=qh if qh is not None else None,
        )

        if verbose:
            n = len(df_target)
            below = ((df[col] < ql).sum() if ql is not None else 0)
            above = ((df[col] > qh).sum() if qh is not None else 0)

            msg = f"{col}: "
            if ql is not None:
                msg += f"capped {below} rows ({100*below/n:.2f}%) below {ql:.3f}"
            if qh is not None:
                msg += f", capped {above} rows ({100*above/n:.2f}%) above {qh:.3f}"
            print(msg)

    return (None if inplace else df_target), thresholds

def apply_winsor_from_thresholds(df, thresholds, inplace=False):
    df_target = df if inplace else df.copy()
    for col, (low, high) in thresholds.items():
        if col not in df_target.columns:
            continue
        df_target[col] = df_target[col].clip(
            lower=low if low is not None else None,
            upper=high if high is not None else None,
        )
    return df_target


def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", "_", s)
    return re.sub(r"_+", "_", s)

def _split_multi(val):
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        return [p.strip() for p in val.split("|") if p.strip()]
    return []

def build_timeaware_prior_table(
    train_df: pd.DataFrame,
    entity_col: str,          # e.g. "directors" or "lead_cast"
    year_col: str,            # e.g. "x_year"
    target_col: str,          # e.g. "revenue" (winsorized already)
    all_years: list[int] | None = None,  # full year range to cover (e.g., df_features['x_year'].unique())
    multi_label: bool = False,
    m: float = 5.0            # smoothing strength
) -> pd.DataFrame:
    """
    Build a table of smoothed average(target) up to previous year for each entity.
    Uses TRAIN ONLY (pass the train subset).
    """
    td = train_df[[entity_col, year_col, target_col]].dropna(subset=[year_col]).copy()
    td[year_col] = td[year_col].astype(int)

    # explode if multi-label (one row per entity token)
    if multi_label:
        td = td.explode(entity_col)
        td[entity_col] = td[entity_col].apply(lambda x: _split_multi(x)[0] if isinstance(x, list) else x)
        td = td.explode(entity_col)
        td[entity_col] = td[entity_col].astype(str).str.strip()
    else:
        # ensure single labels are strings
        td[entity_col] = td[entity_col].astype(str).str.strip()

    # Remove empty entities
    td = td[td[entity_col].astype(bool)]

    # aggregate by (entity, year): sum & count
    yearly = (
        td.groupby([entity_col, year_col])[target_col]
          .agg(sum_y='sum', n='count')
          .reset_index()
    )

    # ensure a continuous year grid per entity and forward-fill later
    if all_years is None:
        all_years = sorted(yearly[year_col].unique().tolist())
    else:
        all_years = sorted(set(int(y) for y in all_years))

    # build complete grid
    entities = yearly[entity_col].unique()
    grid = (
        pd.MultiIndex.from_product([entities, all_years], names=[entity_col, year_col])
        .to_frame(index=False)
    )
    yearly = grid.merge(yearly, on=[entity_col, year_col], how="left").fillna({'sum_y':0.0, 'n':0})

    # cumulative sums per entity, then shift by 1 year to get "up to previous year"
    yearly = yearly.sort_values([entity_col, year_col])
    yearly['cum_sum'] = yearly.groupby(entity_col)['sum_y'].cumsum()
    yearly['cum_n']   = yearly.groupby(entity_col)['n'].cumsum()

    # previous-year stats
    yearly['prev_sum'] = yearly.groupby(entity_col)['cum_sum'].shift(1).fillna(0.0)
    yearly['prev_n']   = yearly.groupby(entity_col)['cum_n'].shift(1).fillna(0.0)

    global_mean = float(train_df[target_col].mean())

    # Bayesian smoothing toward global mean
    # prior_mean = (prev_sum + m*global_mean) / (prev_n + m)
    yearly['prior_mean'] = (yearly['prev_sum'] + m*global_mean) / (yearly['prev_n'] + m)

    # output column name
    out_col = f"x_{_slug(entity_col)}_avg_revenue_prevyear"
    prior_table = yearly[[entity_col, year_col, 'prior_mean']].rename(columns={'prior_mean': out_col})

    return prior_table  # merge/apply this to any split

def apply_timeaware_prior(
    df: pd.DataFrame,
    prior_table: pd.DataFrame,
    entity_col: str,
    year_col: str,
    out_col: str | None = None,
    multi_label: bool = False,
    fallback_value: float | None = None
) -> pd.DataFrame:
    """
    Apply a precomputed (entity, year) prior to a dataframe (Train/Val/Test).
    If multi_label, averages the prior over all tokens in the cell.
    """
    out_df = df.copy()
    out_col = out_col or f"x_{_slug(entity_col)}_avg_revenue_prevyear"

    # dict for quick lookup
    key = prior_table.set_index([entity_col, year_col]).iloc[:, 0]
    lookup = key.to_dict()

    if fallback_value is None:
        # global fallback if not provided
        # (use overall mean of the prior table)
        fallback_value = float(prior_table.iloc[:, -1].mean())

    def _row_lookup_single(ent, yr):
        return lookup.get((str(ent).strip(), int(yr)), fallback_value)

    if not multi_label:
        out_df[out_col] = [
            _row_lookup_single(ent, yr) if pd.notna(yr) else fallback_value
            for ent, yr in zip(out_df[entity_col], out_df[year_col])
        ]
    else:
        # average across tokens in this row
        vals = []
        for ent_cell, yr in zip(out_df[entity_col], out_df[year_col]):
            tokens = _split_multi(ent_cell)
            if not tokens or pd.isna(yr):
                vals.append(fallback_value)
                continue
            scores = [lookup.get((str(t).strip(), int(yr)), fallback_value) for t in tokens]
            vals.append(float(np.mean(scores)) if scores else fallback_value)
        out_df[out_col] = vals

    return out_df


def high_corr_pairs(df, threshold=0.9):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={'level_0': 'Feature_1', 'level_1': 'Feature_2', 0: 'Corr'})
        .query('Corr > @threshold')
        .sort_values('Corr', ascending=False)
    )
    return pairs


# ---------- Helpers for training----------
def add_time_cyclical(X, month_col="x_month", weekday_col="x_weekday"):
    Xc = X.copy()
    if month_col in Xc.columns:
        Xc["x_month_sin"] = np.sin(2*np.pi * (Xc[month_col] / 12.0))
        Xc["x_month_cos"] = np.cos(2*np.pi * (Xc[month_col] / 12.0))
    if weekday_col in Xc.columns:
        Xc["x_wday_sin"] = np.sin(2*np.pi * (Xc[weekday_col] / 7.0))
        Xc["x_wday_cos"] = np.cos(2*np.pi * (Xc[weekday_col] / 7.0))
    return Xc

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
                          greater_is_better=False)



def chronosort_for_tscv(X, y, year_col="x_year"):
    """
    Sort X,y by year_col chronologically while preserving the original index labels.
    Drops rows where y is NA and keeps X,y aligned by index.

    - X_sorted.index is a permutation of X.index (sorted by year_col)
    - y_sorted.index is aligned to X_sorted.index
    """
    if year_col not in X.columns:
        raise KeyError(f"{year_col} not in X columns")

    # Sort by year_col; keep original index
    X_sorted = X.sort_values(year_col)
    y_sorted = y.reindex(X_sorted.index)

    # Drop rows with missing y, keeping index alignment
    mask = y_sorted.notna()
    X_sorted = X_sorted.loc[mask]
    y_sorted = y_sorted.loc[mask]

    return X_sorted, y_sorted


def eval_on_val(fitted, X_val_, y_val_):
    y_pred = fitted.predict(X_val_)
    return {
        "RMSE": rmse(y_val_, y_pred),
        "MAE":  mean_absolute_error(y_val_, y_pred),
        "R2":   r2_score(y_val_, y_pred),
    }



# --------- Save best params (generic) ---------

def save_best_params(model_name, search_obj, feature_cols,
                     target="y_log_revenue",
                     drop_corr=None,
                     cv_desc="TimeSeriesSplit(n_splits=5)",
                     out_dir="artifacts",
                     filename=None,
                     extra_meta=None):
    """
    Save best parameters and metadata for reproducibility.
    - model_name: str, e.g., "ElasticNet", "RandomForest", "XGBoost"
    - search_obj: fitted RandomizedSearchCV (has .best_params_)
    - feature_cols: list of feature column names used to train/validate
    - drop_corr: list of dropped features (optional)
    - filename: optional fixed filename; default is auto-named
    - extra_meta: dict with anything else you want to track
    """
    meta = {
        "model": model_name,
        "best_params": search_obj.best_params_,
        "cv": cv_desc,
        "target": target,
        "drop_corr": drop_corr or [],
        "n_candidates": getattr(search_obj, "n_iter", None),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "features": list(feature_cols),
        "scorer": "RMSE(log) lower is better",
        "cv_best_rmse_log": float(-search_obj.best_score_) if hasattr(search_obj, "best_score_") else None,
    }
    if extra_meta:
        meta.update(extra_meta)

    os.makedirs(out_dir, exist_ok=True)
    if filename is None:
        safe_name = model_name.lower().replace(" ", "_")
        filename = f"{safe_name}_best_params.json"
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print(f"Saved params to {path}")
    return path


def load_best_params_json(short_name: str, folder: str | os.PathLike = ARTIFACTS_DIR) -> dict:
    """
    Load best model parameters from a JSON file inside artifacts/.

    Parameters
    ----------
    short_name : str
        e.g. 'elasticnet_cyc_best_params'
        The function will load artifacts/<short_name>.json

    folder : str or Path, default ARTIFACTS_DIR
        Base directory where the JSON is stored. Automatically resolved
        to the project's artifacts/ folder.

    Returns
    -------
    dict
        The contents of the JSON file as a dictionary.
    """
    path = os.path.join(folder, f"{short_name}.json")

    with open(path, "r") as f:
        return json.load(f)


def make_elasticnet_from_saved(saved: dict) -> Pipeline:
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", ElasticNet(max_iter=10_000, random_state=42))
    ])
    pipe.set_params(**saved["best_params"])
    return pipe

def make_rf_from_saved(saved: dict) -> Pipeline:
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    pipe.set_params(**saved["best_params"])
    return pipe

def make_xgb_from_saved(saved: dict) -> Pipeline:
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("xgb", XGBRegressor(
            random_state=42, n_estimators=500,
            tree_method="hist", objective="reg:squarederror",
            eval_metric="rmse", n_jobs=-1
        ))
    ])
    pipe.set_params(**saved["best_params"])
    return pipe

# def make_nn_from_saved(saved: dict, n_features: int, step_name="nn") -> Pipeline:
#     # Build KerasRegressor and set saved params (keys start with 'nn__...')
#     reg = KerasRegressor(
#         model=build_mlp,
#         model__n_features=n_features,
#         # sensible defaults; will be overridden by saved best_params
#         epochs=40, batch_size=32, verbose=0
#     )
#     pipe = Pipeline([
#         ("imp", SimpleImputer(strategy="median")),
#         ("scaler", StandardScaler()),
#         (step_name, reg)
#     ])
#     # Map saved keys directly (they include 'nn__...')
#     pipe.set_params(**saved["best_params"])
#     return pipe



def plot_nn_training_history(history: dict, title: str = "NN Training (per epoch)"):
    if not history:
        print("No history captured."); return

    def _plot_pair(train_key, val_key, ylabel, subtitle):
        if train_key in history or val_key in history:
            plt.figure(figsize=(10,4))
            if train_key in history: plt.plot(history[train_key], label=f"train_{ylabel}")
            if val_key in history: plt.plot(history[val_key],   label=f"val_{ylabel}")
            plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(f"{title} - {subtitle}")
            plt.legend(); plt.show()

    _plot_pair("loss", "val_loss", "Loss", "Loss")
    # RMSE may appear as 'rmse' (we named it), so this stays:
    _plot_pair("rmse", "val_rmse", "RMSE", "RMSE")
    # MAE
    _plot_pair("mae", "val_mae", "MAE", "MAE")
    

def select_features_from_saved(X: pd.DataFrame, saved: dict) -> pd.DataFrame:
    """
    Subset X to the exact feature set used when the model was tuned.
    Expects 'features' key in the saved JSON produced by save_best_params.

    Raises if any expected feature is missing in the current X.
    """
    feats = saved.get("features", None)
    if feats is None:
        # No feature list stored -> just return X as-is
        return X

    missing = [f for f in feats if f not in X.columns]
    if missing:
        raise KeyError(
            f"Current X is missing features expected by saved model. "
            f"Example missing: {missing[:10]}"
        )

    # Keep only those features, in the same order
    return X[feats]
