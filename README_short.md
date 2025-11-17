# Movie Revenue Prediction – Recruiter Overview

This project predicts **movie box-office revenue before release** using metadata from the TMDB API.  
It demonstrates end-to-end machine learning capability: **data ingestion**, **feature engineering**, **time-aware modeling**, **model comparison**, and a **production-ready ensemble pipeline**.

## What the project shows
- Full ML workflow: ingestion → features → modeling → evaluation → deployment  
- Custom engineered features (historical priors, genre encodings, cyclical date features)  
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
