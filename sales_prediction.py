# file: app.py

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import plotly.graph_objects as go


# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="ML Dashboard", layout="wide")
st.title("📊 Sales Prediction ML Dashboard")


# ----------------------------
# FILE UPLOAD
# ----------------------------
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

model_name = st.sidebar.selectbox(
    "Select Model for Prediction Plot",
    ["Linear Regression", "Random Forest", "XGBoost"]
)

run = st.sidebar.button("🚀 Train Models")


# ----------------------------
# MODELS
# ----------------------------
def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
    }


# ----------------------------
# METRICS
# ----------------------------
def evaluate(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def cross_validate_model(model, X, y):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2"
        }
    )

    return {
        "MAE": (-scores["test_mae"].mean(), scores["test_mae"].std()),
        "RMSE": (-scores["test_rmse"].mean(), scores["test_rmse"].std()),
        "R2": (scores["test_r2"].mean(), scores["test_r2"].std()),
    }


# ----------------------------
# PLOT
# ----------------------------
def plot_model(y_true, preds, title):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_true,
        y=preds,
        mode="markers",
        name="Predictions",
        hovertemplate="Actual: %{x}<br>Predicted: %{y}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=[y_true.min(), y_true.max()],
        y=[y_true.min(), y_true.max()],
        mode="lines",
        name="Ideal",
        line=dict(color="red")
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Actual Sales",
        yaxis_title="Predicted Sales",
        template="plotly_white"
    )

    return fig


# ----------------------------
# MAIN APP
# ----------------------------
if file:
    df = pd.read_csv(file)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    st.subheader("📄 Full Dataset")
    st.dataframe(df, use_container_width=True)

    if run:
        X = df[["TV", "Radio", "Newspaper"]]
        y = df["Sales"]

        models = get_models()

        results = {}
        cv_results = {}
        predictions = {}

        selected_model = models[model_name]

        # Train all models
        for name, model in models.items():
            model.fit(X, y)
            preds = model.predict(X)

            results[name] = evaluate(y, preds)
            cv_results[name] = cross_validate_model(model, X, y)
            predictions[name] = preds

        # ---------------- UI ----------------
        tab1, tab2, tab3 = st.tabs([
            "📊 Model Comparison",
            "🔁 Cross Validation",
            "📈 Selected Model Plot"
        ])

        # ---------------- COMPARISON ----------------
        with tab1:
            st.subheader("All Models Performance")

            st.dataframe(pd.DataFrame(results).T, use_container_width=True)

        # ---------------- CV ----------------
        with tab2:
            st.subheader("5-Fold Cross Validation")

            cv_table = {}

            for model, m in cv_results.items():
                cv_table[model] = {
                    "MAE": f"{m['MAE'][0]:.3f} ± {m['MAE'][1]:.3f}",
                    "RMSE": f"{m['RMSE'][0]:.3f} ± {m['RMSE'][1]:.3f}",
                    "R2": f"{m['R2'][0]:.3f} ± {m['R2'][1]:.3f}",
                }

            st.dataframe(pd.DataFrame(cv_table).T, use_container_width=True)

        # ---------------- SELECTED MODEL PLOT ----------------
        with tab3:
            st.subheader(f"Prediction Plot: {model_name}")

            st.plotly_chart(
                plot_model(y, predictions[model_name], model_name),
                use_container_width=True
            )

else:
    st.info("👈 Upload CSV to start")