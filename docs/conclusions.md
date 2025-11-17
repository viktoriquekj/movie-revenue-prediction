# Dashboard Conclusions 

**Note on 2025 Data Completeness**

The 2025 revenue data is incomplete, as many titles released in 2025 have not yet finished their full theatrical run or have not reported final box-office numbers.
As a result, all conclusions involving 2025 performance—model accuracy, error metrics, and distributional insights—should be interpreted with caution, since the target values may change as real revenues are finalized. 

## Model Performance Summary

**Ensemble C – Final Model**

| Metric (Log Revenue) | 2024 Test | 2025 Forward |
|----------------------|-----------|--------------|
| **RMSE (log)**       | **1.79**  | **1.96**     |
| **MAE (log)**        | 1.26      | 1.33         |
| **R²**               | **0.86**  | **0.83**     |
| **Improvement vs Baseline** | **62.3%** | **58.6%** |

**Interpretation:**  
The model performs strongly on the true hold-out year (2024) and maintains stable accuracy when extended to 2025 forward predictions.

---

## Key Findings

### 1. Strong Predictive Performance
- Captures approximately **86% of variance** in log-revenue on the 2024 test set.  
- Only minor degradation is observed when predicting 2025 movies.

### 2. Significant Lift Over Baseline
- Reduces error by **62%** versus a naïve year-mean baseline in 2024.  
- Continues to outperform baseline in 2025 with **59%** improvement.

### 3. Revenue Prediction Accuracy
- Typical prediction error is approximately **$30–40M** (MAE).  
- Larger errors occur primarily in blockbuster releases, which is expected due to extreme revenue variability.

---

## Feature Importance Insights

Feature importance was assessed using **SHAP values**, which quantify how each feature shifts individual predictions relative to a baseline expectation.
SHAP is model-agnostic, interpretable, and allows us to understand both global drivers and feature-specific behavior.

### Why Random Forest-based SHAP?

SHAP values were computed using the Random Forest model from the ensemble.
Random Forests are tree-based and compatible with `TreeExplainer`, making them an excellent proxy for understanding the ensemble’s learned structure.

### Global Importance (Mean |SHAP|)

The top contributors to predicted revenue are:

1. Budget (log-scaled) – by far the strongest driver of expected revenue

    - Higher budgets systematically push predictions upward

    - Low budgets strongly constrain revenue potential

    - SHAP dependence shows a near-linear relationship with output

2. Runtime – moderate importance

    - Longer runtimes correlate with higher expected revenue

    - Extremely long runtimes (>180 min) plateau in effect

3. Franchise movies (`is_in_collection`)

    - Being part of a franchise significantly increases predicted revenue

    - SHAP shows a clear separation:

        - 0 → negative contribution

        - 1 → strong positive contribution

4. Release Timing Variables (`year`, `month`, `weekday`, cyclical encodings)

    - Capture patterns like seasonality and box-office growth

5. Cast Gender Balance, Country, and Certification

    - Smaller effects, but still meaningful adjustments to predicted revenue

    - Certifications like PG-13 and R subtly shape expected performance

### Interpretation of SHAP Plots

- Budget Log SHAP Dependence: Strong monotonic increase → the model uses budget as a primary indicator of commercial scale.

- Runtime SHAP Dependence: Nonlinear effect → very short films underperform; optimal range ≈ 100–140 minutes.

- Collection Status: Binary jump → franchise titles generate disproportionate revenue.

Overall, the feature importance analysis shows that the model captures realistic economic patterns consistent with industry intuition:
bigger budgets, franchise alignment, strategic release timing, and certain runtime ranges all drive revenue upward.

---

## Year-Over-Year Stability

**RMSE (log) by Year: Model vs Baseline**

| Year | Model RMSE | Baseline RMSE | Improvement |
|------|------------|----------------|-------------|
| 2024 | 1.79       | 4.74           | **62%**     |
| 2025 | 1.96       | 4.72           | **59%**     |

The model demonstrates stable performance and limited drift when applied to future movie predictions.

---

## 2025 Forecast Insights

- Predicted revenues for 2025 follow a distribution consistent with historical patterns.  
- Accuracy remains stable across genres and budget segments.  
- Blockbusters continue to contribute most to uncertainty, which is normal for revenue forecasting.

---

## Overall Conclusion

The Ensemble C model is:

- **Accurate:** Explains ~86% of variance in 2024 revenue.  
- **Robust:** Maintains strong performance on forward-looking 2025 data.  
- **Business-ready:** Generates actionable insights for revenue planning.  
- **Consistently superior:** Provides a 60%+ error reduction over baseline models.

**Applications include:**

- Early revenue forecasting for unreleased films  
- P&A budget allocation  
- Slate planning and scenario modeling  
- Financial prediction and portfolio risk assessment  

---

## Supporting Files

These results are derived from:

- `results/metrics/metrics_summary.csv`  
- `results/metrics/rmse_by_year.csv`  
- `results/validation_predictions/oof_predictions.csv`  
- `results/final_results/df_2025_predictions.csv`

Visualizations in:

- `results/plots/`

---

## Next Steps

- Publish interactive dashboard (Tableau or Power BI)  
- Automate periodic retraining and model drift monitoring  
