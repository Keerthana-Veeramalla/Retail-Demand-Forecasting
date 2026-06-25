# Retail Demand Forecasting & Explainability Dashboard

Forecasts daily store sales using the Rossmann Store Sales dataset (Kaggle), comparing a Linear Regression baseline against XGBoost, validated with proper time-based splitting (not random shuffling), and explained with SHAP. Deployed as an interactive Streamlit dashboard.

## Problem

Retail businesses need to forecast demand to manage stock and staffing. Given a store's history (sales, promotions, holidays, competition), predict future daily sales — and explain *why* the model predicts what it predicts.

## Dataset

[Rossmann Store Sales — Kaggle](https://www.kaggle.com/c/rossmann-store-sales)
Historical daily sales for 1,115 European drug stores (Jan 2013 – Jul 2015), ~1M rows, including promotions, holidays, store type, and competition distance.

Download `train.csv` and `store.csv` from Kaggle and place them in the `data/` folder before running anything.

## Approach

1. **Feature engineering** (`feature_engineering.py`)
   - Date/seasonality features: day-of-week, month, week-of-year, weekend flag, holiday flags
   - Lag features: sales 1, 7, and 14 days ago (per store, no cross-store leakage)
   - Rolling averages: 7-day and 30-day rolling mean sales (shifted to avoid leaking the current day)
   - Categorical encoding for store type, assortment, promo interval

2. **Modeling** (`train_model.py`)
   - Baseline: Linear Regression (scaled features)
   - Main model: XGBoost Regressor
   - **Time-based train/test split** — last 42 days held out as test set; never randomly shuffled, since random splitting on time-series data leaks future information into training
   - **Walk-forward validation** — 3 rolling folds to confirm the model is stable across different time windows, not just lucky on one split
   - Metrics: RMSE and MAPE (regression metrics, not classification accuracy/F1)

3. **Explainability + Dashboard** (`app.py`)
   - SHAP TreeExplainer shows which features push a prediction up or down for a given store/day
   - Streamlit dashboard: pick a store, see sales history, predicted-vs-actual on the holdout period, an N-day forward forecast, and the SHAP breakdown

## How to run

```bash
pip install -r requirements.txt

# 1. Place train.csv and store.csv (from Kaggle) into the data/ folder

# 2. Build features
python feature_engineering.py

# 3. Train models and see evaluation metrics in the console
python train_model.py

# 4. Launch the dashboard
streamlit run app.py
```

## Results

*(Fill in after running on the real dataset — these numbers will differ from any test run)*

| Model | RMSE | MAPE |
|---|---|---|
| Linear Regression (baseline) | — | — |
| XGBoost | — | — |

Walk-forward validation (3 folds): *(paste your fold-by-fold RMSE/MAPE here)*

## Key design decisions

- **Time-based split over random split**: random k-fold cross-validation on time-series data lets the model "see the future" during training, which inflates accuracy in a way that won't hold up in production. Splitting by date avoids this.
- **Per-store grouping for lag features**: lag/rolling features are computed within each store's own history (`groupby("Store")`), so one store's data never leaks into another's.
- **SHAP over generic feature importance**: SHAP shows the direction and magnitude of each feature's effect on a *specific* prediction, not just a global ranking — closer to what a business stakeholder actually wants to know ("why is the forecast high this week?").

## Tech stack

Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit, Plotly
