# Movie Revenue Prediction (TMDB Data)

A complete, end-to-end machine learning pipeline that predicts **movie box-office revenue before release** using TMDB metadata, engineered features, and a stacked ensemble model.

Designed following **MaggieInData’s Portfolio Guide** to highlight both  
**business value**, **engineering rigor**, and **model interpretability**.

**Note:** This project uses the TMDB API but is **not endorsed or certified** by TMDB.

---

## Project Overview

Studios and streamers rely on early revenue forecasts to drive:

- P&A (Print & Advertising) allocation  
- Greenlighting and ROI decisions  
- Slate planning and scenario modeling  

This project builds a **reproducible forecasting system** that predicts  
**log(revenue)** using pre-release attributes (budget, cast, genres, dates, certifications, etc.).

The pipeline includes:

- TMDB API ingestion  
- Time-aware feature engineering  
- Multiple ML baselines (ElasticNet, RF, XGBoost, NN)  
- Final **stacked ensemble (Model C)**  
- Deployment workflow for **upcoming 2025 releases**  
- SHAP-based feature importance analysis  

For business framing and architecture:
- `docs/business_context.md`  
- `docs/architecture.md`  
- `docs/conclusions.md` (dashboard-style)

---

## Project Structure

movie-revenue-prediction/
├── src/
│ └── movie_revenue_prediction/
│ ├── api/ # TMDB client (data collection)
│ ├── features/ # Feature engineering, priors, cyclical time features
│ ├── models/ # ML models, NN, ensemble stacking
│ ├── utils/ # Paths, metrics, helpers
│ └── deployment/ # Production prediction pipeline
│
├── notebooks/
│ ├── 01_exploration_modeling.ipynb
│ ├── 02_deployment_2025_demo.ipynb
│ └── 03_shap_feature_analysis.ipynb
│
├── data/
│ ├── raw/ # Raw TMDB pulls (gitignored)
│ ├── curated/ # Train/val/test parquet splits (gitignored)
│ └── results/ # Local intermediates (gitignored)
│
├── artifacts/
│ ├── *_best_params.json # Hyperparameters for ElasticNet/RF/XGB/NN
│ └── ensemble_C/ # Final stacked ensemble artifacts
│ ├── ensemble_manifest.json
│ ├── meta_info.json
│ └── preprocessing/ + base_models/ # Binary weights (gitignored)
│
├── results/
│ ├── metrics/ # metrics_summary.csv, rmse_by_year.csv
│ ├── predictions/ # oof_predictions.csv, df_all_predictions.csv
│ ├── final_results/ # df_2025_predictions.csv
│ └── plots/ # Evaluation + feature importance
│
├── docs/ # Business + architecture + conclusions
│
├── requirements.txt
├── pyproject.toml
└── README.md


---

## 🧠 Feature Engineering

Key engineered features include:

- **Log-budget**, runtime, collection/franchise flag  
- **Historical priors** for directors, lead cast, composers  
- **Multi-hot encodings**: genres, keywords, production countries  
- **Cyclical temporal features**: month, weekday, release season  
- **Leakage-free chronological splits (2017–2024)**  
- **Winsorization + imputers** for robust model input  

Feature generation is implemented in:
`src/movie_revenue_prediction/features/build_features.py`

---

## Models

- **ElasticNet**  
- **RandomForest**  
- **XGBoost**  
- **MLP Neural Network (with SWA)**  
- **Stacked Ensemble (Model C)** ← **Final production model**

Artifacts stored in:  
`artifacts/ensemble_C/`
Evaluation metrics stored in:  
`results/metrics/`

---

## Deployment (2025 Forecast Demo)

Predict revenue for unreleased 2025 movies:

```python
from movie_revenue_prediction.deployment.pipeline import predict_with_model_C

df = predict_with_model_C("data/curated/tmdb_2025.csv")
``` 

Output saved to:
`results/final_results/df_2025_predictions.csv`

Notebook demo:
`notebooks/02_deployment_2025_demo.ipynb`

## Dashboard (Coming Soon)

An interactive dashboard (Tableau or Power BI) will include:

### Page 1 — EDA on Historical Dataset
- Revenue distribution  
- Genre, certification, and language breakdowns  
- Budget vs revenue relationships  
- Seasonal and calendar-based insights  

### Page 2 — Prediction Analysis
- Predicted vs actual revenue (log-scale and raw scale)  
- Residual distributions and error diagnostics  
- RMSE by year  
- Feature importance (SHAP)  
- Distribution of 2025 revenue forecasts  

Dashboard link: [to be added]

---

## Environment Notes (SHAP Requires a Separate Environment)

Most of the project (training, feature engineering, evaluation, deployment) runs in the main environment:

```bash
pip install -r requirements.txt
pip install -e .
``` 

However, the notebook `03_shap_feature_analysis.ipynb` uses SHAP, which requires:

- numpy >= 2

- avoiding TensorFlow and XGBoost due to binary incompatibilities on macOS

To run SHAP, create a dedicated lightweight environment: 

```bash
conda create -n movie_shap_env python=3.11 -y
conda activate movie_shap_env

pip install "numpy>=2"
pip install shap>=0.50
pip install pandas scikit-learn matplotlib joblib

``` 
Use this environment only for interpretability work. 

## Key Conclusions (Short Version)

- Budget, franchise status, and runtime are the strongest predictors of movie revenue, based on SHAP feature importance analysis.

- Temporal features such as month and weekday meaningfully capture release-strategy effects.

- Ensemble Model C explains approximately 86% of the variance in 2024 revenue and reduces error by over 60% compared to a naive baseline.

- Performance remains stable when applied to 2025 forward predictions, though results should be interpreted carefully because 2025 revenues are incomplete.

Full conclusions are available in:
`docs/conclusions.md`

## Installation 

```bash
pip install -r requirements.txt
pip install -e .
```

Example import:

```python
from movie_revenue_prediction.features.build_features import make_features_timeaware_splits
``` 

## Attribution

This project uses data obtained through the TMDB API.
It is not endorsed or certified by TMDB.
Please refer to TMDB's terms of service for permitted usage.

## Future Work

- Publish full dashboard

- Integrate automated retraining

- Deploy inference pipeline on Databricks or AWS

- Add model drift monitoring and expanded interpretability


