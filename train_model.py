"""
train_model.py
--------------
Trains a baseline Linear Regression and an XGBoost model on the
engineered features, using a TIME-BASED train/test split (never
random shuffle for time series!), then compares RMSE and MAPE.

Saves the trained XGBoost model + feature list to disk for the
Streamlit app to load later.

Run after feature_engineering.py.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb

from feature_engineering import FEATURES, TARGET


def time_based_split(df, test_days=42):
    """
    Split by DATE, not randomly. The last `test_days` days become the
    test set; everything before that is training data. This avoids
    leaking future information into the model, which a random split
    would do.
    """
    df = df.sort_values("Date")
    cutoff_date = df["Date"].max() - pd.Timedelta(days=test_days)
    train_df = df[df["Date"] <= cutoff_date]
    test_df = df[df["Date"] > cutoff_date]
    return train_df, test_df


def evaluate(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    print(f"{label:20s} | RMSE: {rmse:10.2f} | MAPE: {mape:6.2f}%")
    return rmse, mape


def walk_forward_validation(df, n_splits=3, test_days=42):
    """
    Walk-forward validation: repeatedly train on everything up to time T,
    test on the next `test_days`, then roll T forward. This proves model
    stability across different time windows, not just one lucky split.
    """
    df = df.sort_values("Date")

    results = []
    for i in range(n_splits, 0, -1):
        cutoff = df["Date"].max() - pd.Timedelta(days=test_days * i)
        train_window = df[df["Date"] <= cutoff]
        test_window = df[
            (df["Date"] > cutoff) & (df["Date"] <= cutoff + pd.Timedelta(days=test_days))
        ]
        if len(train_window) == 0 or len(test_window) == 0:
            continue

        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, random_state=42
        )
        model.fit(train_window[FEATURES], train_window[TARGET])
        preds = model.predict(test_window[FEATURES])
        rmse, mape = evaluate(test_window[TARGET], preds, f"Fold ending {cutoff.date()}")
        results.append({"cutoff": cutoff, "rmse": rmse, "mape": mape})

    return pd.DataFrame(results)


def main():
    print("Loading processed features...")
    df = pd.read_csv("data/processed_features.csv", parse_dates=["Date"])

    train_df, test_df = time_based_split(df, test_days=42)
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")
    print(f"Train date range: {train_df['Date'].min().date()} -> {train_df['Date'].max().date()}")
    print(f"Test date range:  {test_df['Date'].min().date()} -> {test_df['Date'].max().date()}")

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    # ---- Baseline: Linear Regression ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)

    print("\n--- Hold-out Test Set Results ---")
    evaluate(y_test, lr_preds, "Linear Regression")

    # ---- Main model: XGBoost ----
    xgb_model = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    evaluate(y_test, xgb_preds, "XGBoost")

    # ---- Walk-forward validation (stability check) ----
    print("\n--- Walk-Forward Validation (3 rolling folds) ---")
    wf_results = walk_forward_validation(df, n_splits=3, test_days=42)
    print(wf_results)

    # ---- Save model + scaler + feature list for the Streamlit app ----
    joblib.dump(xgb_model, "xgb_model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(FEATURES, "features.joblib")
    print("\nSaved xgb_model.joblib, scaler.joblib, features.joblib")
    print("You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
