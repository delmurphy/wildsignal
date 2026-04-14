# %% [markdown]
# # 7. Streamlit app for classification model
# 
# 🔹 A. Exploration (historical data) - transparency + insight
# 
# Users can:
# 
# - Select state
# - Select time range  
# - View:
#     - biodiversity observations
#     - anomalies (your target)
#     - weather variables
# 
# 🔹 B. Prediction (model) - real-world application
# 
# Users can:
# 
# - Input:
#     - state
#     - month/year (or future scenario)
#     - weather conditions (different climate change scenarios)
# - Output:
#     - probability of biodiversity anomaly
#     - classification (shock / no shock)

# %% [markdown]
# *Note - run multiple simulations to calculate uncertainty around predictions*
# for i in range(100):
#     simulate weather
#     predict anomalies
#     count anomalies
# 
# mean +- std across simulations

# %%
import sys
import os
sys.path.append(os.path.abspath("src")) 

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from utils import simulate_future, forecast_bioshocks, simulate_with_uncertainty

# %%
# Load data
full_df = pd.read_parquet('Data/Processed/full_df.parquet')


# %%
# -----------------------------
# Load model bundle
# -----------------------------
@st.cache_resource
def load_model_bundle():
    return joblib.load("models/production_model.pkl")

bundle = load_model_bundle()

model = bundle['model']
features = bundle['features']
threshold = bundle['threshold']

# -----------------------------
# Import functions
# -----------------------------

from utils import simulate_future, forecast_bioshocks

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
        "temp increase": 7,
        "precipitation change": 0.3
    }
}
# -----------------------------
# UI
# -----------------------------
st.title("🌿 Biodiversity Anomaly Forecast")

st.write("Simulate biodiversity anomalies under climate scenarios.")

# State selection
state_list = full_df['state'].unique()
states = state_list
selected_state = st.selectbox("Select Bundesland", states)

# Scenario selection
selected_scenario = st.radio(
    "Select Climate Scenario",
    list(SCENARIOS.keys())
)

scenario = selected_scenario

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict"):

    st.subheader("Running Monte Carlo simulation...")

    historic_yearly = (
    full_df[full_df["state"] == selected_state]
    .groupby("year")["biodiversity_anomaly_sensitive"]
    .sum()
    .reset_index()
    .rename(columns={"biodiversity_anomaly_sensitive": "n_anomalies"})
)

    # 1. run uncertainty simulation
    result = simulate_with_uncertainty(
        df = full_df,
        state=selected_state,
        scenario=scenario,
        features = features,
        model=model,
        threshold = threshold,
        n_runs=100
    )

    last_hist_year = historic_yearly["year"].max()

    result["year"] = result["year"] + last_hist_year

    total = result["mean_anomalies"].sum()

    # -----------------------------
    # Output text
    # -----------------------------
    st.subheader("Results")

    st.write(
        f"**Expected anomaly months (10 years): {total:.1f} ± {result['std_anomalies'].sum():.1f}**"
    )

    st.write(
        f"Average per year: {total/10:.1f}"
    )

    # -----------------------------
    # Plot with uncertainty band
    # -----------------------------
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10,5))

    # -----------------------------
    # HISTORICAL DATA
    # -----------------------------
    ax.plot(
        historic_yearly["year"],
        historic_yearly["n_anomalies"],
        marker="o",
        label="Observed (historical)",
        color="black"
    )

    # -----------------------------
    # FORECAST MEAN
    # -----------------------------
    ax.plot(
        result["year"],
        result["mean_anomalies"],
        label="Predicted mean",
        color="blue"
    )

    # -----------------------------
    # UNCERTAINTY BAND
    # -----------------------------
    ax.fill_between(
        result["year"],
        result["lower"],
        result["upper"],
        color="blue",
        alpha=0.2,
        label="Prediction uncertainty (±1 std)"
    )

    # -----------------------------
    # STYLE
    # -----------------------------
    ax.axvline(
        historic_yearly["year"].max(),
        linestyle="--",
        color="grey",
        label="Forecast start"
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Biodiversity anomaly months")
    ax.set_title(f"{selected_state} — historical vs forecast")

    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # -----------------------------
    # Optional table
    # -----------------------------
    with st.expander("Show simulation results"):
        st.dataframe(result)

# if st.button("Predict"):

#     st.subheader("Running simulation...")

#     # 1. simulate future climate + features
#     future_df = simulate_future(
#         df=full_df.loc[full_df['state']==selected_state],
#         features = features,
#         scenario=scenario
#     )

#     # 2. predict anomalies
#     preds = forecast_bioshocks(
#         df = future_df,
#         model = model,
#         features = features,
#         threshold = threshold
#     )

#     # -----------------------------
#     # 3. Aggregate per year
#     # -----------------------------
#     yearly = (
#         preds.groupby("year")["biodiversity_anomaly"]
#         .sum()
#         .reset_index()
#     )

#     yearly.rename(
#         columns={"biodiversity_anomaly": "n_anomalies"},
#         inplace=True
#     )

#     total = yearly["n_anomalies"].sum()

#     # -----------------------------
#     # 4. Output text
#     # -----------------------------
#     st.subheader("Results")

#     st.write(
#         f"**Total predicted anomaly months (10 years): {int(total)}**"
#     )

#     st.write(
#         f"Average per year: {total/10:.1f}"
#     )

#     # -----------------------------
#     # 5. Plot
#     # -----------------------------
#     fig, ax = plt.subplots()

#     ax.plot(
#         yearly["year"],
#         yearly["n_anomalies"],
#         marker="o"
#     )

#     ax.set_xlabel("Year (future offset)")
#     ax.set_ylabel("Number of anomaly months")
#     ax.set_title(f"{selected_state} — {selected_scenario}")

#     ax.grid(True)

#     st.pyplot(fig)

#     # -----------------------------
#     # 6. Optional: raw table
#     # -----------------------------
#     with st.expander("Show yearly data"):
#         st.dataframe(yearly)


