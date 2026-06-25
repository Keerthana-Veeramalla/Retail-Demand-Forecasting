"""
app.py
------
Streamlit dashboard: pick a store, see historical sales, a forecast
for the next N days, and a SHAP explanation of what's driving the
prediction.

Run with: streamlit run app.py
(Run feature_engineering.py and train_model.py first so the model
files exist.)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go

from feature_engineering import FEATURES, TARGET

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")

st.title("📈 Retail Demand Forecasting Dashboard")
st.caption("Rossmann Store Sales — XGBoost forecast with SHAP explainability")


@st.cache_resource
def load_artifacts():
    model = joblib.load("xgb_model.joblib")
    features = joblib.load("features.joblib")
    return model, features


@st.cache_data
def load_data():
    df = pd.read_csv("data/processed_features.csv", parse_dates=["Date"])
    return df


model, FEATURE_COLS = load_artifacts()
df = load_data()

# ---- Sidebar controls ----
st.sidebar.header("Controls")
store_ids = sorted(df["Store"].unique())
selected_store = st.sidebar.selectbox("Select Store", store_ids)
forecast_days = st.sidebar.slider("Days to forecast", min_value=7, max_value=42, value=14, step=7)

store_df = df[df["Store"] == selected_store].sort_values("Date")

st.subheader(f"Store {selected_store} — Recent Sales History")
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(
    x=store_df["Date"].tail(180), y=store_df["Sales"].tail(180),
    mode="lines", name="Actual Sales", line=dict(color="#2E86AB")
))
fig_hist.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10),
                        xaxis_title="Date", yaxis_title="Sales")
st.plotly_chart(fig_hist, use_container_width=True)

# ---- Hold-out predictions vs actual (model sanity check) ----
st.subheader("Model Performance: Predicted vs Actual (Last 42 Days)")
recent = store_df.tail(42).copy()
recent["Predicted"] = model.predict(recent[FEATURE_COLS])

fig_compare = go.Figure()
fig_compare.add_trace(go.Scatter(x=recent["Date"], y=recent["Sales"],
                                  mode="lines+markers", name="Actual", line=dict(color="#2E86AB")))
fig_compare.add_trace(go.Scatter(x=recent["Date"], y=recent["Predicted"],
                                  mode="lines+markers", name="Predicted", line=dict(color="#E63946", dash="dash")))
fig_compare.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10),
                           xaxis_title="Date", yaxis_title="Sales")
st.plotly_chart(fig_compare, use_container_width=True)

rmse = np.sqrt(np.mean((recent["Sales"] - recent["Predicted"]) ** 2))
mape = np.mean(np.abs((recent["Sales"] - recent["Predicted"]) / recent["Sales"])) * 100
col1, col2 = st.columns(2)
col1.metric("RMSE (last 42 days)", f"{rmse:,.0f}")
col2.metric("MAPE (last 42 days)", f"{mape:.2f}%")

# ---- Forward forecast (iterative, using model's own predictions as new lags) ----
st.subheader(f"Forecast: Next {forecast_days} Days")

future_rows = []
history = store_df.copy().reset_index(drop=True)

for i in range(forecast_days):
    last_row = history.iloc[-1]
    next_date = last_row["Date"] + pd.Timedelta(days=1)

    new_row = last_row.copy()
    new_row["Date"] = next_date
    new_row["DayOfWeek"] = next_date.dayofweek
    new_row["IsWeekend"] = int(next_date.dayofweek >= 5)
    new_row["Month"] = next_date.month
    new_row["WeekOfYear"] = next_date.isocalendar()[1]

    # Roll lag features forward using recent actual/predicted sales
    new_row["sales_lag_1"] = history["Sales"].iloc[-1]
    new_row["sales_lag_7"] = history["Sales"].iloc[-7] if len(history) >= 7 else history["Sales"].iloc[-1]
    new_row["sales_lag_14"] = history["Sales"].iloc[-14] if len(history) >= 14 else history["Sales"].iloc[-1]
    new_row["sales_roll_mean_7"] = history["Sales"].tail(7).mean()
    new_row["sales_roll_mean_30"] = history["Sales"].tail(30).mean()

    pred = model.predict(pd.DataFrame([new_row[FEATURE_COLS]]))[0]
    new_row["Sales"] = pred  # feed prediction back in as if it were actual, for next iteration's lags

    future_rows.append({"Date": next_date, "Forecast": pred})
    history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

future_df = pd.DataFrame(future_rows)

fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=store_df["Date"].tail(60), y=store_df["Sales"].tail(60),
                                   mode="lines", name="Recent Actual", line=dict(color="#2E86AB")))
fig_forecast.add_trace(go.Scatter(x=future_df["Date"], y=future_df["Forecast"],
                                   mode="lines+markers", name="Forecast", line=dict(color="#F4A261")))
fig_forecast.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10),
                            xaxis_title="Date", yaxis_title="Sales")
st.plotly_chart(fig_forecast, use_container_width=True)

st.dataframe(future_df.style.format({"Forecast": "{:,.0f}"}), use_container_width=True)

# ---- SHAP explainability ----
st.subheader("Why this forecast? (SHAP feature importance)")

with st.spinner("Computing SHAP values..."):
    explainer = shap.TreeExplainer(model)
    sample = recent[FEATURE_COLS].tail(1)
    shap_values = explainer.shap_values(sample)

shap_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "shap_value": shap_values[0]
}).sort_values("shap_value", key=abs, ascending=False)

fig_shap = go.Figure(go.Bar(
    x=shap_df["shap_value"], y=shap_df["feature"], orientation="h",
    marker_color=["#E63946" if v < 0 else "#2A9D8F" for v in shap_df["shap_value"]]
))
fig_shap.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10),
                        xaxis_title="Impact on prediction (Sales)", yaxis_title="")
st.plotly_chart(fig_shap, use_container_width=True)

st.caption("Green = pushes the prediction up. Red = pushes it down. "
           "Based on the most recent day's feature values for this store.")
