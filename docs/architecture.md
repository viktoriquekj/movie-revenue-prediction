# Project Architecture

This document provides a visual overview of the system components and how data flows through the movie revenue prediction pipeline.

---

## 1. High-level End-to-End Pipeline

```mermaid
flowchart LR
    A[TMDB IDs<br/>data/raw/ids.csv] --> B[TMDB API Client<br/>tmdb_client.py]
    B --> C[Curated Datasets<br/>data/curated/tmdb_2017_2024.csv<br/>data/curated/tmdb_2025.csv]

    C --> D[Feature Engineering<br/>build_features.py]
    D --> E[Model Training<br/>simple_models.py<br/>nn_model.py<br/>ensemble_model.py]
    E --> F[Artifacts<br/>artifacts/model_params<br/>artifacts/ensemble_C]

    C --> G[Deployment Pipeline<br/>deployment/pipeline.py]
    F --> G
    G --> H[Predictions for 2025 Movies<br/>data/results_2025/df_2025.csv]
flowchart TD
    A[Curated Training Data<br/>2017–2024] --> B[Feature Engineering<br/>Time-aware, priors, multi-hot, cyclical]
    B --> C[Chronological CV Splits]

    C --> D1[ElasticNet<br/>simple_models.py]
    C --> D2[RandomForest<br/>simple_models.py]
    C --> D3[XGBoost<br/>simple_models.py]
    C --> D4[Neural Network (MLP + SWA)<br/>nn_model.py]

    D1 --> E[Out-of-Fold Predictions]
    D2 --> E
    D3 --> E
    D4 --> E

    E --> F[Meta-learner / Ensemble Model C<br/>ensemble_model.py]

    F --> G[Saved Artifacts<br/>artifacts/ensemble_C]
flowchart TD
    A[Curated 2025 Data<br/>tmdb_2025.csv] --> B[Load Artifacts<br/>priors, encoders, topK lists, thresholds<br/>artifacts/ensemble_C]
    B --> C[Feature Engineering for 2025<br/>build_features.py + utils/functions.py]

    C --> D1[ElasticNet Prediction]
    C --> D2[RandomForest Prediction]
    C --> D3[XGBoost Prediction]
    C --> D4[Neural Network Prediction]

    D1 --> E[Stack Base Predictions]
    D2 --> E
    D3 --> E
    D4 --> E

    E --> F[Ensemble Meta-learner<br/>Model C]
    F --> G[Final Revenue Predictions<br/>data/results_2025/df_2025.csv]

