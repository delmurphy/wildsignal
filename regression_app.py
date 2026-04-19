import sys
import os
sys.path.append(os.path.abspath("src"))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

from utils import simulate_future, forecast_biodiversity, simulate_with_uncertainty_regression

# -----------------------------
# Load data
# -----------------------------
full_df = pd.read_parquet('Data/Processed/full_df.parquet')
full_df = full_df.rename(columns={'residual_z': 'biodiversity_z'})

# -----------------------------
# Load model bundle (REGRESSION)
# -----------------------------
@st.cache_resource
def load_model_bundle():
    return joblib.load("models/production_model_regression.pkl")

bundle = load_model_bundle()

model = bundle['model']
features = bundle['features']

# -----------------------------
# Climate scenarios
# -----------------------------
SCENARIOS = {
    "best case": {
        "temp increase": 1.5,
        "precipitation change": 0.1
    },
    "middle of the road": {
        "temp increase": 4,
        "precipitation change": 0.2
    },
    "business as usual": {
        "temp increase": 8,
        "precipitation change": 0.3
    }
}

# -----------------------------
# UI
# -----------------------------
st.title("🌿 Biodiversity Forecast (Z-score)")

st.write("Simulate biodiversity changes under climate scenarios.")

# State selection
states = full_df['state'].unique()
selected_state = st.selectbox("Select Bundesland", states)

# Scenario selection
selected_scenario = st.radio(
    "Select Climate Scenario",
    list(SCENARIOS.keys())
)

# Simulation length
sim_length = st.slider(
    "Simulation horizon (years)",
    min_value=10, max_value=100, value=50
)

# Number of simulations
n_runs = st.slider(
    "Number of simulations (uncertainty)",
    min_value=10, max_value=200, value=50
)

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict"):

    st.subheader("Running Monte Carlo simulation...")

    # -----------------------------
    # HISTORICAL DATA
    # -----------------------------
    historic_yearly = (
        full_df[full_df["state"] == selected_state]
        .groupby("year")["biodiversity_z"]
        .mean()
        .reset_index()
    )

    last_hist_year = historic_yearly["year"].max()

    # -----------------------------
    # RUN SIMULATION (REGRESSION)
    # -----------------------------
    result = simulate_with_uncertainty_regression(
        df=full_df,
        state=selected_state,
        scenario=selected_scenario,
        features=features,
        model=model,
        sim_length=sim_length,
        n_runs=n_runs
    )

    # align future years after historical
    result["year"] = result["year"] + last_hist_year

    # -----------------------------
    # SUMMARY METRICS
    # -----------------------------
    avg_future = result["mean"].mean()
    trend = result["mean"].iloc[-1] - result["mean"].iloc[0]

    st.subheader("Results")

    st.write(f"**Average predicted biodiversity z-score:** {avg_future:.2f}")
    st.write(f"**Change over period:** {trend:.2f}")

    if avg_future < 0:
        st.warning("⚠️ Biodiversity expected to decline below baseline.")
    else:
        st.success("✅ Biodiversity remains near or above baseline.")

    # -----------------------------
    # PLOT
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    # HISTORICAL
    ax.plot(
        historic_yearly["year"],
        historic_yearly["biodiversity_z"],
        marker="o",
        label="Observed (historical)",
        color="black"
    )

    # PREDICTED MEAN
    ax.plot(
        result["year"],
        result["mean"],
        color="green",
        label="Predicted mean"
    )

    # UNCERTAINTY BAND
    ax.fill_between(
        result["year"],
        result["lower"],
        result["upper"],
        color="green",
        alpha=0.2,
        label="Uncertainty (±1 std)"
    )

    # ZERO LINE (important for z-scores)
    ax.axhline(0, linestyle="--", color="grey", alpha=0.7)

    # FORECAST SPLIT
    ax.axvline(
        last_hist_year,
        linestyle="--",
        color="grey",
        label="Forecast start"
    )

    # LABELS
    ax.set_xlabel("Year")
    ax.set_ylabel("Biodiversity (z-score)")
    ax.set_title(f"{selected_state} — biodiversity trajectory")

    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # -----------------------------
    # OPTIONAL TABLE
    # -----------------------------
    with st.expander("Show simulation data"):
        st.dataframe(result)