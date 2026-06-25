"""
feature_engineering.py
-----------------------
Loads raw Rossmann train.csv + store.csv, cleans them, and builds
time-series features: lag features, rolling averages, and
date/seasonality features.

Run this first. It produces 'data/processed_features.csv', which
train_model.py consumes.
"""

import pandas as pd
import numpy as np


def load_raw_data(train_path="data/train.csv", store_path="data/store.csv"):
    """Load the raw Kaggle files."""
    train = pd.read_csv(train_path, parse_dates=["Date"], low_memory=False)
    store = pd.read_csv(store_path)
    df = train.merge(store, on="Store", how="left")
    return df


def basic_clean(df):
    """Drop closed days (Sales=0 because store was shut) and sort by time."""
    df = df[df["Open"] == 1].copy()
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    # Fill missing competition / promo columns sensibly
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
        df["CompetitionDistance"].median()
    )
    for col in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
                "Promo2SinceWeek", "Promo2SinceYear"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    df["PromoInterval"] = df["PromoInterval"].fillna("None")

    return df


def add_date_features(df):
    """Day-of-week, month, year, weekend flag, and holiday flags."""
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"] = df["Date"].dt.dayofweek  # 0=Monday
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

    # StateHoliday comes as 'a','b','c','0' -> encode as flag
    df["IsStateHoliday"] = (df["StateHoliday"] != "0").astype(int)
    df["IsSchoolHoliday"] = df["SchoolHoliday"].astype(int)

    return df


def add_lag_and_rolling_features(df, lags=(1, 7, 14), windows=(7, 30)):
    """
    Per-store lag features and rolling averages.
    IMPORTANT: grouped by Store so we never leak one store's history into another's.
    """
    df = df.sort_values(["Store", "Date"])

    for lag in lags:
        df[f"sales_lag_{lag}"] = df.groupby("Store")["Sales"].shift(lag)

    for window in windows:
        df[f"sales_roll_mean_{window}"] = (
            df.groupby("Store")["Sales"]
            .shift(1)  # shift first so the current day's actual sale isn't included
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(drop=True)
        )

    return df


def encode_categoricals(df):
    """One-hot / simple encode store type, assortment, promo interval."""
    df["StoreType"] = df["StoreType"].astype("category").cat.codes
    df["Assortment"] = df["Assortment"].astype("category").cat.codes
    df["PromoInterval"] = df["PromoInterval"].astype("category").cat.codes
    return df


def build_feature_table(train_path="data/train.csv", store_path="data/store.csv"):
    df = load_raw_data(train_path, store_path)
    df = basic_clean(df)
    df = add_date_features(df)
    df = add_lag_and_rolling_features(df)
    df = encode_categoricals(df)

    # Drop rows where lag features are NaN (the very first few days per store)
    feature_cols = [c for c in df.columns if c.startswith("sales_lag_") or c.startswith("sales_roll_")]
    df = df.dropna(subset=feature_cols)

    return df


FEATURES = [
    "Store", "DayOfWeek", "Promo", "StoreType", "Assortment",
    "CompetitionDistance", "Promo2", "IsStateHoliday", "IsSchoolHoliday",
    "IsWeekend", "Month", "WeekOfYear",
    "sales_lag_1", "sales_lag_7", "sales_lag_14",
    "sales_roll_mean_7", "sales_roll_mean_30",
]
TARGET = "Sales"


if __name__ == "__main__":
    print("Loading and engineering features... this can take a minute on the full dataset.")
    df = build_feature_table()
    out_path = "data/processed_features.csv"
    df.to_csv(out_path, index=False)
    print(f"Done. Saved {len(df):,} rows x {len(df.columns)} columns to {out_path}")
    print("Feature columns used for modeling:", FEATURES)
