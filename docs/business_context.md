# Business Context and Problem Definition

This document outlines the business framing, stakeholder impact, and operational requirements for the Movie Revenue Prediction project. It is intended for hiring managers, product teams, and non-technical stakeholders who want to understand the value and applicability of the system. 


## 1. Introduction

The film industry depends heavily on early financial forecasting to allocate production resources, plan marketing budgets, and make distribution decisions. Forecasting theatrical revenue before a movie is released is challenging due to:

- incomplete or noisy early data

- variability in audience reception

- performance differences across markets and genres

- limited comparable historical records

This project investigates how several machine-learning models — Elastic Net, Random Forest, XGBoost, Neural Networks, and a stacked Ensemble model — perform on the task of predicting log-transformed movie revenue prior to release.

The purpose is two-fold:

1. Build an accurate predictive model for future revenue.

2. Compare the interpretability, robustness, and generalization performance of different modeling approaches.

The work combines classical statistical learning methods with modern machine-learning techniques to produce a realistic forecasting system. 


## 2. Executive Summary

A streaming studio or theatrical distributor seeks early revenue indicators for upcoming films. This project provides a model that predicts log(revenue) from pre-release TMDB metadata, offering an evidence-based approach to financial and marketing decisions.

**Success Criteria**

- Lower RMSE on log-revenue compared to a naive baseline

- Improved accuracy of forecasts generated 2–6 months before release

- Stable performance across production years

**Primary Stakeholders**

- Finance: forecasting, ROI, cashflow management

- Marketing: print and advertising (P&A) budget optimization

- Content/Production: greenlighting and slate planning

**Key Business Decision**
Early prioritization of titles for marketing and distribution resources. 


## 3. Business Problem

Finance and Marketing teams commit significant spend well before theatrical release:

- prints and advertising (P&A) budgets

- localization and distribution

- partnerships and promotional activities

- release-date strategy

However, current forecasting approaches are often heuristic, inconsistent, and difficult to scale. There is a need for a systematic, data-driven forecasting tool that can support decision-making during the pre-release period. 


## 4. Goals of the Forecasting System
### **Primary Goal**

Predict worldwide theatrical revenue before release using features available at least 30 days prior to release.

### **Deliverables**

- A reproducible machine-learning pipeline

- Properly engineered features without data leakage

- Model evaluation framework and comparison suite

- A high-performing ensemble model for deployment

- Forecasts for unreleased films (2025 dataset provided as an example)

### **Secondary Modeling Goals**

- Compare GLMs with tree-based models and neural networks

- Quantify the benefit of ensemble stacking

- Assess year-over-year model stability

- Provide explainability outputs (feature importance, partial dependency, etc.) 


## 5. Constraints and Realities

Forecasting early-stage film performance involves several challenges.

### **Data Constraints**

- Missing or inconsistent early metadata (e.g., budget, certifications)

- Heavy-tailed revenue distribution requiring transformation

- Cyclical release-date effects (seasonality)

- Sparse historical priors for underrepresented directors or cast members

### **Methodological Constraints**

- Avoid data leakage (e.g., popularity or vote counts updated after early screenings)

- Use chronological splits instead of random cross-validation

- Maintain transparency for validation by non-technical stakeholders

### **Operational Constraints**

- Forecast coverage: at least 95% of upcoming titles

- Pipeline runtime: under 5 minutes on a standard laptop

- Year-over-year RMSE(log) drift:

    - ≤15% without retraining

    - ≤5% with quarterly retraining

- Adoption proxy: Use in at least one budgeting cycle 


## 6. Stakeholders and Their Needs
### **Finance**

- Early revenue projections for cashflow modeling and ROI scenarios

- Ability to run optimistic and conservative forecast variants

### **Marketing**

- Allocation of P&A budgets

- Prioritizing high-ROI titles

- Adjusting campaign intensity based on projected outcomes

### **Content / Greenlighting**

- Benchmarking new projects against historical analogs

- Informing production investment and slate diversification

### **Data Science / Analytics**

- Reproducible model training processes

- Monitoring drift and stability

- Providing explainability and auditability 


## 7. Operational Flow

1. Collect movie IDs from TMDB.

2. Fetch metadata using the TMDB API client.

3. Apply feature engineering pipelines.

4. Train individual models.

5. Generate out-of-fold predictions for ensemble training.

6. Train the ensemble meta-learner.

7. Package preprocessing artifacts and model assets.

8. Run predictions on unseen datasets (e.g., 2025 movies).

9. Export results to `data/results_2025/`. 


## 8. Key Success Metrics
### **Modeling Metrics**

- RMSE on log-revenue (validation and out-of-time splits)

- Stability across production years

- Performance uplift relative to a baseline

### **Business Metrics**

- Increased accuracy and earlier availability of forecasts

- Improved allocation of marketing budgets

- Reduced reliance on subjective heuristics 


## 9. Outputs

The system produces:

- Predicted log-revenue (and converted revenue) for unreleased titles

- A complete forecast dataset for 2025 (`df_2025.csv`)

- Model artifacts including:

    - historical priors

    - top-K token lists

    - base model weights

    - ensemble meta-model

Stored in:

- `artifacts/ensemble_C/` (model components)

- `data/results_2025/` (forecast outputs) 


## 10. Summary

This project demonstrates how machine-learning methods can be used to forecast movie revenue under realistic operational constraints. By combining classical regression, modern tree-based models, neural networks, and ensemble learning, the system achieves strong predictive performance while maintaining reproducibility and business interpretability.

The goal is to provide a robust forecasting pipeline that aligns with how studios and distributors plan promotional budgets, financial models, and release strategies.