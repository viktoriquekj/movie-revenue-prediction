from __future__ import annotations

import numpy as np

from movie_revenue_prediction.utils.functions import chronosort_for_tscv, eval_on_val

from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor



def fit_elasticnet_cv(X, y, tscv, param_dist, n_iter=30, verbose=0):
    """
    Fit ElasticNet using RandomizedSearchCV with time-aware CV.
    Returns the fitted search object.
    """
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   ElasticNet(max_iter=10_000, random_state=42))
    ])

    rmse_scorer = make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
                              greater_is_better=False)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=rmse_scorer,
        cv=tscv,
        n_jobs=-1,
        random_state=42,
        verbose=verbose
    )
    search.fit(X, y)
    return search

### Random Forest

def fit_rf_cv(X, y, param_dist, n_iter=40, n_splits=5, verbose=1):
    """Median-impute -> RF; time-aware CV with RMSE scoring."""
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf",  RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    rmse_scorer = make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
                              greater_is_better=False)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=rmse_scorer,
        cv=tscv,
        n_jobs=-1,
        random_state=42,
        verbose=verbose
    )
    search.fit(X, y)
    return search



### XGBoost

def fit_xgb_cv(X, y, param_dist, n_iter=50, n_splits=5, verbose=1):
    """Median-impute -> XGBRegressor; time-aware CV with RMSE scoring."""
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("xgb", XGBRegressor(
            random_state=42, n_estimators=500,
            tree_method="hist", objective="reg:squarederror",
            eval_metric="rmse", n_jobs=-1
        ))
    ])
    rmse_scorer = make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
                              greater_is_better=False)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=rmse_scorer,
        cv=tscv,
        n_jobs=-1,
        random_state=42,
        verbose=verbose
    )
    search.fit(X, y)
    return search
# =========================
# ElasticNet
# =========================
def run_elasticnet_experiment(X_train, y_train, X_val, y_val, version_name="ElasticNet Base"):
    """Train and evaluate ElasticNet with given feature set.
    Returns: best_model, best_params, metrics_dict, search_obj
    """
    print(f"\n Running {version_name} ...")

    # Sort chronologically for time-series CV
    X_sorted, y_sorted = chronosort_for_tscv(X_train, y_train, year_col="x_year")

    # Parameter search space
    param_dist = {
        "model__alpha": np.logspace(-3, 1, 30),
        "model__l1_ratio": np.linspace(0.0, 1.0, 20)
    }

    # Time-based cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Fit model
    search = fit_elasticnet_cv(X_sorted, y_sorted, tscv, param_dist, n_iter=30, verbose=1)

    best_model = search.best_estimator_
    val_metrics = eval_on_val(best_model, X_val, y_val)

    print(f"Best Params: {search.best_params_}")
    print(f"CV Best RMSE (log): {-search.best_score_:.4f}")
    print(f"Validation → RMSE: {val_metrics['RMSE']:.4f}, "
          f"MAE: {val_metrics['MAE']:.4f}, R²: {val_metrics['R2']:.4f}")

    return best_model, search.best_params_, val_metrics, search


# =========================
# Random Forest
# =========================
def run_rf_experiment(X_train_, y_train_, X_val_, y_val_, version_name="RF Base"):
    """Train and evaluate Random Forest with given feature set.
    Returns: best_model, best_params, metrics_dict, search_obj
    """
    print(f"\n Running {version_name} ...")
    X_sorted, y_sorted = chronosort_for_tscv(X_train_, y_train_, year_col="x_year")

    # NOTE: removed 'auto' to avoid InvalidParameterError in newer sklearn
    rf_params = {
        "rf__n_estimators": [300, 500, 800],
        "rf__max_depth": [None, 8, 12, 16, 22],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
        "rf__max_features": ["sqrt", "log2", None, 0.3, 0.5, 0.7]
    }

    search = fit_rf_cv(X_sorted, y_sorted, rf_params, n_iter=40, verbose=1)
    best_model = search.best_estimator_
    metrics = eval_on_val(best_model, X_val_, y_val_)

    print("Best Params:", search.best_params_)
    print(f"CV Best RMSE (log): {-search.best_score_:.4f}")
    print(f"Validation → RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, R²: {metrics['R2']:.4f}")

    return best_model, search.best_params_, metrics, search


# =========================
# XGBoost
# =========================
def run_xgb_experiment(X_train_, y_train_, X_val_, y_val_, version_name="XGB Base"):
    """Train and evaluate XGBoost with given feature set.
    Returns: best_model, best_params, metrics_dict, search_obj
    """
    try:
        _ = XGBRegressor  # ensure imported
    except NameError:
        print(f"\nSkipping {version_name}: xgboost not installed.")
        return None, None, {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan}, None

    print(f"\n⚡ Running {version_name} ...")
    X_sorted, y_sorted = chronosort_for_tscv(X_train_, y_train_, year_col="x_year")

    xgb_params = {
        "xgb__max_depth": [4, 6, 8, 10],
        "xgb__learning_rate": np.logspace(-3, -1, 6),  # 0.001..0.1
        "xgb__subsample": [0.6, 0.8, 1.0],
        "xgb__colsample_bytree": [0.6, 0.8, 1.0],
        "xgb__min_child_weight": [1, 5, 10],
        "xgb__reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "xgb__reg_lambda": [0.5, 1.0, 2.0]
    }

    search = fit_xgb_cv(X_sorted, y_sorted, xgb_params, n_iter=50, verbose=1)
    best_model = search.best_estimator_
    metrics = eval_on_val(best_model, X_val_, y_val_)

    print("Best Params:", search.best_params_)
    print(f"CV Best RMSE (log): {-search.best_score_:.4f}")
    print(f"Validation → RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, R²: {metrics['R2']:.4f}")

    return best_model, search.best_params_, metrics, search
