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
import json
import plotly.express as px
from streamlit_plotly_events import plotly_events
import unicodedata
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
    return joblib.load("models/production_model_full_classifier.pkl")

bundle = load_model_bundle()

model = bundle['model']
features = bundle['features']
threshold = bundle['threshold']*2

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
        "temp increase": 8,
        "precipitation change": 0.3
    }
}
# -----------------------------
# UI
# -----------------------------
# ---------------------------------------
# TITLE
# ---------------------------------------
st.title("🌿 Biodiversity Anomaly Forecast")
st.write("Click a Bundesland on the map to select it.")

st.write("Simulate biodiversity anomalies under climate scenarios.")

# State selection dropdown
# state_list = full_df['state'].unique()
# states = state_list
# selected_state = st.selectbox("Select Bundesland", states)


import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import unicodedata
import re
from shapely.geometry import shape, mapping

# ---------------------------------------
# LOAD DATA
# ---------------------------------------
full_df = pd.read_parquet("Data/Processed/full_df.parquet")

# ---------------------------------------
# NORMALIZATION
# ---------------------------------------
def norm(x):
    x = unicodedata.normalize("NFKC", str(x))
    x = re.sub(r"\s+", " ", x)
    return x.strip().lower()

# ---------------------------------------
# LOAD GEOJSON
# ---------------------------------------
with open("Data/Raw/2_hoch.geo.json", "r", encoding="utf-8") as f:
    geojson = json.load(f)

# ---------------------------------------
# GEOMETRY SAFETY (minimal)
# ---------------------------------------
def fix_geometry(geom):
    if not geom.is_valid:
        geom = geom.buffer(0)
    geom = geom.simplify(0.0002, preserve_topology=True)
    return geom if not geom.is_empty else None

clean_features = []
for f in geojson["features"]:
    geom = shape(f["geometry"])
    geom = fix_geometry(geom)

    if geom is None:
        continue

    f["geometry"] = mapping(geom)
    f["id"] = f["properties"]["id"]
    clean_features.append(f)

geojson["features"] = clean_features

# ---------------------------------------
# LOOKUP TABLES
# ---------------------------------------
state_lookup = {
    norm(f["properties"]["name"]): f["properties"]["id"]
    for f in geojson["features"]
}

reverse_lookup = {v: k for k, v in state_lookup.items()}

# ---------------------------------------
# DATA PREP
# ---------------------------------------
df_map = full_df.copy()
df_map["state_norm"] = df_map["state"].apply(norm)
df_map["state_id"] = df_map["state_norm"].map(state_lookup)

df_map = df_map.dropna(subset=["state_id"])
df_map = df_map.groupby(["state", "state_id"]).size().reset_index(name="value")

df_map["state_id"] = df_map["state_id"].astype(str)

# ---------------------------------------
# SESSION STATE
# ---------------------------------------
if "selected_state" not in st.session_state:
    st.session_state.selected_state = None



# ---------------------------------------
# BASE MAP
# ---------------------------------------
fig = go.Figure()

# base layer
fig.add_trace(
    go.Choroplethmapbox(
        geojson=geojson,
        locations=df_map["state_id"],
        featureidkey="properties.id",
        z=df_map["value"],
        colorscale=[[0, "#eeeeee"], [1, "#ff7f0e"]],
        marker_line_width=0.8,
        marker_line_color="white",
        hovertext=df_map["state"],
        hoverinfo="text"
    )
)

# ---------------------------------------
# SELECTED STATE HIGHLIGHT
# ---------------------------------------
if st.session_state.selected_state:
    selected = df_map[df_map["state_id"] == st.session_state.selected_state]

    fig.add_trace(
        go.Choroplethmapbox(
            geojson=geojson,
            locations=selected["state_id"],
            featureidkey="properties.id",
            z=[1] * len(selected),
            colorscale=[[0, "red"], [1, "red"]],
            marker_line_width=2,
            marker_line_color="black",
            showscale=False,
            hoverinfo="skip"
        )
    )

# ---------------------------------------
# LAYOUT
# ---------------------------------------
fig.update_layout(
    mapbox=dict(
        style="carto-positron",
        zoom=4.5,
        center=dict(lat=51.0, lon=10.0),
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    clickmode="event+select"
)

# ---------------------------------------
# RENDER WITH CLICK EVENTS
# ---------------------------------------
event = st.plotly_chart(
    fig,
    use_container_width=True,
    key="map",
    on_select="rerun"
)

# ---------------------------------------
# CLICK HANDLING (THIS IS THE KEY FIX)
# ---------------------------------------
if event and hasattr(event, "selection") and event.selection:
    try:
        clicked_id = event.selection["points"][0]["location"]
        st.session_state.selected_state = clicked_id
    except Exception:
        pass

# ---------------------------------------
# UI OUTPUT
# ---------------------------------------
st.subheader("Selected Bundesland")

if st.session_state.selected_state:
    st.success(reverse_lookup.get(st.session_state.selected_state, "Unknown"))
else:
    st.info("Click a state on the map")

# selected_state = st.session_state.selected_state

state_id_to_name = (
    df_map[["state_id", "state"]]
    .dropna()
    .drop_duplicates()
    .set_index("state_id")["state"]
    .to_dict()
)

state_id = st.session_state.selected_state

selected_state = state_id_to_name.get(state_id)


# Scenario selection
selected_scenario = st.radio(
    "Select Climate Scenario",
    list(SCENARIOS.keys())
)

scenario = selected_scenario

# Simulation length
sim_length = st.slider(
    "How long should the simulation run for?",
    min_value=10, max_value=100, value=50
)

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
        f"**Expected anomaly months ({sim_length} years): {total:.1f} ± {result['std_anomalies'].sum():.1f}**"
    )

    st.write(
        f"Average per year: {total/sim_length:.1f}"
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


