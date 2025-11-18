# Movie Revenue Prediction – Recruiter Overview

This project predicts **movie box-office revenue before release** using metadata from the TMDB API.  
It demonstrates end-to-end machine learning capability: **data ingestion**, **feature engineering**, **time-aware modeling**, **model comparison**, and a **production-ready ensemble pipeline**.

## What the project shows
- Full ML workflow: ingestion → features → modeling → evaluation → deployment  
- Custom engineered features (historical priors, genre one-hot encodings, cyclical date features)  
- Multiple models (ElasticNet, RandomForest, XGBoost, Neural Network) combined into a **stacked ensemble**  
- Strong performance: **~86% variance explained** on the true 2024 hold-out test  
- Clean, modular, production-style Python package

## Project deliverables
- **Ensemble production model** + inference pipeline  
- **2025 movie revenue forecasts**  
- **Model evaluation reports** (RMSE, MAE, R², drift checks)  
- **Feature importance (SHAP) analysis**  
- **Architecture diagrams and business context**  
- Dashboard (Tableau/Power BI) – *link will be added*

## Why this project matters
Movie revenue forecasting is a noisy real-world problem.  
This project highlights the ability to:

- build predictive systems under uncertainty  
- engineer meaningful features from unstructured metadata  
- validate models correctly using **chronological splits**  
- design modular, reproducible, industry-standard ML code  

It is both **business-relevant** and **technically robust**, making it a strong portfolio example.

## What I Learned

- Developed a fully reproducible ML pipeline with modular code, feature engineering, and deployment logic.
- Applied advanced techniques: winsorization, cyclical time features, historical priors, and SHAP for explainability.
- Learned that strong linear and ensemble baselines often outperform neural networks on smaller, noisy datasets.
- Gained experience collecting and cleaning data from the TMDB API and handling real-world data quality issues.

## What Did Not Work

- Winsorizing revenue reduced model ability to predict extreme blockbuster outcomes.
- Combining models trained on different feature sets (base vs cyclical) added complexity without improving accuracy.
- Neural networks overfit easily and did not outperform simpler models.
- High-cardinality features (keywords, companies) added noise despite top-K filtering.

## Future Improvements

- Test alternative data splits (e.g., random or rolling windows) to reduce temporal drift.
- Try stronger models such as LightGBM, CatBoost, or tabular transformers.
- Add richer features: NLP embeddings from plot descriptions, collaboration networks, or market inflation adjustments.
- Model tail behavior explicitly (e.g., quantile regression) to improve blockbuster prediction accuracy.
