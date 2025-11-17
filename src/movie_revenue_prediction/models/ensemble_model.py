from __future__ import annotations

import pandas as pd
import numpy as np
from movie_revenue_prediction.models.nn_model import _nn_final_callbacks
from movie_revenue_prediction.utils.functions import chronosort_for_tscv
from sklearn.linear_model import  Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone, BaseEstimator, TransformerMixin

# from scikeras.wrappers import KerasRegressor
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers, regularizers, callbacks


def fit_meta_learner(oof_df: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0) -> tuple[Ridge, pd.Index]:
    """
    Fit Ridge on rows where all OOF columns are present (no NaNs).
    Align y_train to the OOF index produced by time-aware CV.
    Returns: (fitted_meta_model, used_index)

    Changes:
    - Reindex y_train to oof_df.index to avoid 'Unalignable boolean Series' errors.
    - Store the OOF column order on the meta model as `model_names_`
      so prediction stacking uses the exact same order.
    """
    # Align y to OOF index
    y_aligned = y_train.reindex(oof_df.index)

    # Keep only rows where OOF is complete and y is present
    mask = oof_df.notna().all(axis=1) & y_aligned.notna()

    X_meta = oof_df.loc[mask].values
    y_meta = y_aligned.loc[mask].values

    meta = Ridge(alpha=alpha, random_state=42)
    meta.fit(X_meta, y_meta)

    # Remember the stacking order used to train the meta-learner
    meta.model_names_ = list(oof_df.columns)

    return meta, oof_df.index[mask]


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        # IMPORTANT: do not copy/modify; keep the same reference for sklearn.clone
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.cols]


def generate_oof_preds_timeaware_multiX(
    base_pipes: dict,
    model_to_X: dict,
    y_train: pd.Series,
    n_splits: int = 5,
    nn_step_name: str = "nn",
    year_col: str = "x_year",
) -> tuple[pd.DataFrame, dict]:
    """
    Time-aware OOF generator for models that use DIFFERENT feature matrices.

    Parameters
    ----------
    base_pipes : dict[str, Pipeline]
        Mapping model_name -> sklearn Pipeline (EN, RF, XGB, NN, ...).
    model_to_X : dict[str, pd.DataFrame]
        Mapping model_name -> feature matrix for that model (e.g. X_train_cyc or X_train).
        All Xs must share the same index as y_train and contain `year_col`.
    y_train : pd.Series
        Target aligned index-wise with the X matrices.
    n_splits : int
        Number of TimeSeriesSplit folds.
    nn_step_name : str
        Name of the NN step inside the pipeline (default: "nn").
    year_col : str
        Column name containing the year used for chronosort.

    Returns
    -------
    oof_df : pd.DataFrame
        OOF predictions (one column per model), indexed like the sorted y.
    fitted_full : dict[str, Pipeline]
        Each base model refit on the FULL sorted train set (for val/test).
    """
    # -------- consistency checks --------
    model_names = list(base_pipes.keys())
    if set(model_names) != set(model_to_X.keys()):
        raise ValueError(
            "base_pipes and model_to_X must have the same keys. "
            f"base_pipes: {list(base_pipes.keys())}, model_to_X: {list(model_to_X.keys())}"
        )

    # -------- reference X for chronosort + splits --------
    ref_name = model_names[0]
    X_ref = model_to_X[ref_name]

    X_ref_sorted, y_sorted = chronosort_for_tscv(X_ref, y_train, year_col=year_col)

    # Align all other Xs to the sorted index
    model_to_X_sorted = {}
    for name in model_names:
        Xm = model_to_X[name]
        # index alignment is critical here
        model_to_X_sorted[name] = Xm.loc[X_ref_sorted.index]

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Initialize OOF containers
    oof = {name: pd.Series(index=y_sorted.index, dtype=float) for name in model_names}

    # -------- build OOF via positional splits --------
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_ref_sorted), start=1):
        print(f"Fold {fold}/{n_splits}")
        y_tr = y_sorted.iloc[tr_idx]

        for name in model_names:
            pipe = base_pipes[name]
            X_tr = model_to_X_sorted[name].iloc[tr_idx]
            X_va = model_to_X_sorted[name].iloc[va_idx]

            model = clone(pipe)

            fit_params = {}
            if name.lower().startswith("nn") or name.upper().startswith("SWA"):
                fit_params = {
                    f"{nn_step_name}__callbacks": _nn_final_callbacks(),
                    f"{nn_step_name}__validation_split": 0.15,
                    f"{nn_step_name}__verbose": 0,
                }

            model.fit(X_tr, y_tr, **fit_params)
            preds = model.predict(X_va)
            preds = np.asarray(preds).ravel()

            # positional assignment → indices here are 0..n-1 relative to y_sorted
            oof[name].iloc[va_idx] = preds

        # SMALL DEBUG (optional): show how many non-null OOF per model after each fold
        # for m in model_names:
        #     print(f"  after fold {fold}, non-null in {m}: {oof[m].notna().sum()}")

    # -------- construct OOF DataFrame --------
    oof_df = pd.DataFrame({name: oof[name] for name in model_names}, index=y_sorted.index)

    # -------- refit each model on FULL sorted train --------
    fitted_full = {}
    for name in model_names:
        pipe = base_pipes[name]
        X_full = model_to_X_sorted[name]
        model = clone(pipe)

        fit_params = {}
        if name.lower().startswith("nn") or name.upper().startswith("SWA"):
            fit_params = {
                f"{nn_step_name}__callbacks": _nn_final_callbacks(),
                f"{nn_step_name}__validation_split": 0.15,
                f"{nn_step_name}__verbose": 0,
            }

        model.fit(X_full, y_sorted, **fit_params)
        fitted_full[name] = model

    return oof_df, fitted_full



def predict_with_ensemble_multiX(
    fitted_full: dict,
    meta: Ridge,
    model_to_X: dict
) -> np.ndarray:
    """
    Predict with an ensemble where each base model uses its own feature matrix.

    Parameters
    ----------
    fitted_full : dict[str, Pipeline]
        Base models fitted on the FULL train (keys match meta.model_names_).
    meta : Ridge
        Trained meta-learner with attribute `model_names_` from fit_meta_learner.
    model_to_X : dict[str, pd.DataFrame]
        Mapping model_name -> feature matrix to use for that model on this split
        (e.g. X_val, X_val_cyc, X_test, ...).

    Returns
    -------
    preds : np.ndarray of shape (n_samples,)
        Final ensemble predictions.
    """
    if hasattr(meta, "model_names_"):
        model_names = list(meta.model_names_)
    else:
        model_names = list(fitted_full.keys())

    preds_list = []
    for name in model_names:
        if name not in fitted_full:
            raise KeyError(
                f"Base model '{name}' not found in fitted_full. "
                f"Available: {list(fitted_full.keys())}"
            )
        if name not in model_to_X:
            raise KeyError(
                f"Feature matrix for base model '{name}' not found in model_to_X. "
                f"Available: {list(model_to_X.keys())}"
            )
        model = fitted_full[name]
        Xm = model_to_X[name]
        preds = model.predict(Xm)
        preds_list.append(np.asarray(preds).ravel())

    stack = np.column_stack(preds_list)
    return meta.predict(stack)
