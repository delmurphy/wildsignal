import sys
import os
sys.path.append(os.path.abspath("src"))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import json
import unicodedata
import re
from shapely.geometry import shape, mapping
import plotly.graph_objects as go

from utils import simulate_with_uncertainty_regression

# -----------------------------
# Load data
# -----------------------------
full_df = pd.read_parquet('Data/Processed/full_df.parquet')
full_df = full_df.rename(columns={'turnover_residual_z': 'biodiversity_z'})

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
st.title("🌿 WildSignal Biodiversity Forecast")

st.caption(
    "This chart shows how ecosystem dynamics are expected to change relative to recent conditions under different IPCC climate change projections. "
    "Higher values indicate more ecological change (not necessarily better biodiversity), "
    "while lower values indicate more stable ecosystems."
)

st.write("Simulate biodiversity changes under climate scenarios.")


# -----------------------------
# State selection
# -----------------------------

### Function to normalise state names in df and geojson
def norm(x):
    x = unicodedata.normalize("NFKC", str(x))
    x = re.sub(r"\s+", " ", x)
    return x.strip().lower()

### Load geojson
with open("Data/Raw/2_hoch.geo.json", "r", encoding="utf-8") as f:
    geojson = json.load(f)

### Lookup tables for geojson 'name' and 'id'
state_lookup = {
    norm(f["properties"]["name"]): f["properties"]["id"]
    for f in geojson["features"]
}

reverse_lookup = {v: k for k, v in state_lookup.items()}

### prep data
df_map = full_df.copy()
df_map["state_norm"] = df_map["state"].apply(norm)
df_map["state_id"] = df_map["state_norm"].map(state_lookup)

df_map = df_map.dropna(subset=["state_id"])
df_map = df_map.groupby(["state", "state_id"]).size().reset_index(name="value")

df_map["state_id"] = df_map["state_id"].astype(str)

# clear selected state
if "selected_state" not in st.session_state:
    st.session_state.selected_state = None


### Create colour scale for states
# unique states
state_ids = df_map["state_id"].unique()

# generate evenly spaced greens
greens = plt.cm.Greens(np.linspace(0.5, 0.75, len(state_ids)))

# convert to hex
greens_hex = [
    "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
    for r, g, b, _ in greens
]

# map each state_id -> color index
state_color_map = dict(zip(state_ids, range(len(state_ids))))

# assign numeric index for coloring
df_map["color_id"] = df_map["state_id"].map(state_color_map)

colorscale = []
n = len(greens_hex)

for i, color in enumerate(greens_hex):
    colorscale.append([i/n, color])
    colorscale.append([(i+1)/n, color])

### Create base map
fig = go.Figure()

# base layer
fig.add_trace(
    go.Choroplethmapbox(
        geojson=geojson,
        locations=df_map["state_id"],
        featureidkey="properties.id",
        # z=df_map["value"],
        # colorscale=[[0, "#eeeeee"], [1, "#ff7f0e"]],
        z=df_map["color_id"],
        colorscale=colorscale,
        marker_line_width=0.8,
        marker_line_color="white",
        hovertext=df_map["state"],
        hovertemplate="<b>%{hovertext}</b><extra></extra>",
        hoverinfo="text",
        showscale = False
    )
)

### Highlight selected state

if st.session_state.selected_state:
    selected = df_map[df_map["state_id"] == st.session_state.selected_state]

    fig.add_trace(
        go.Choroplethmapbox(
            geojson=geojson,
            locations=selected["state_id"],
            featureidkey="properties.id",
            z=[1] * len(selected),
            colorscale=[[0, "#d62728"], [1, "#d62728"]],
            marker_line_width=2,
            marker_line_color="black",
            showscale=False,
            hoverinfo="skip"
        )
    )

### map layout
fig.update_layout(
    mapbox=dict(
        style="carto-positron",
        zoom=4.5,
        center=dict(lat=51.0, lon=10.0),
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    clickmode="event+select"
)

### click events
event = st.plotly_chart(
    fig,
    use_container_width=True,
    key="map",
    on_select="rerun"
)

if event and hasattr(event, "selection") and event.selection:
    try:
        clicked_id = event.selection["points"][0]["location"]
        st.session_state.selected_state = clicked_id
    except Exception:
        pass

### UI output
st.subheader("Selected Bundesland")

if st.session_state.selected_state:
    st.success(reverse_lookup.get(st.session_state.selected_state, "Unknown"))
else:
    st.info("Click a state on the map")


state_id_to_name = (
    df_map[["state_id", "state"]]
    .dropna()
    .drop_duplicates()
    .set_index("state_id")["state"]
    .to_dict()
)

state_id = st.session_state.selected_state

selected_state = state_id_to_name.get(state_id)


# -----------------------------
# Scenario selection
# -----------------------------
selected_scenario = st.radio(
    "Select Climate Scenario",
    list(SCENARIOS.keys())
)

st.caption("Scenario definitions follow simplified IPCC-style warming pathways.")


with st.expander("What do these scenarios mean?"):
    st.markdown("""
    **Best case**: Strong climate mitigation consistent with Paris Agreement (~1.5°C warming by 2100).

    **Middle of the road**: Moderate emissions with partial mitigation (~3–4°C warming by 2100).

    **Business as usual**: High emissions trajectory with limited climate policy (~6–8°C warming by 2100).
    """)



# -----------------------------
# Simulation length
# -----------------------------
sim_length = st.slider(
    "Simulation horizon (years)",
    min_value=10, max_value=100, value=50
)

n_runs = st.slider(
    "Number of simulations (estimate uncertainty of results)",
    min_value=50,
    max_value=200,
    value=50,
    step=50
)

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict"):

    st.subheader("Running Monte Carlo simulation...")

    # -----------------------------
    # RECENT BASELINE (last 5 years)
    # -----------------------------
    BASELINE_YEARS = 5

    historic_yearly = (
        full_df[full_df["state"] == selected_state]
        .groupby("year")["biodiversity_z"]
        .mean()
        .reset_index()
    )

    last_hist_year = historic_yearly["year"].max()

    baseline = (
        historic_yearly[historic_yearly["year"] >= last_hist_year - BASELINE_YEARS]
        ["biodiversity_z"]
        .mean()
    )

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

    # convert to change relative to baseline
    result["delta"] = result["mean"] - baseline
    result["lower_delta"] = result["lower"] - baseline
    result["upper_delta"] = result["upper"] - baseline

    # -----------------------------
    # SUMMARY METRICS
    # -----------------------------
    avg_future = result["delta"].mean()
    trend = result["delta"].iloc[-1] - result["delta"].iloc[0]

    st.subheader("Results")

    st.write(f"**Change over period:** {trend:.2f}")

    if trend > 0:
        st.warning("⚠️ Ecological change is expected to increase relative to today.")
    else:
        st.success("✅ Ecosystem remains relatively stable compared to today.")

    # -----------------------------
    # PLOT
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    # color based on direction
    line_color = "red" if result["delta"].mean() > 0 else "blue"

    ax.plot(
        result["year"],
        result["delta"],
        color=line_color,
        linewidth=2,
        label="Projected change"
)


    # UNCERTAINTY BAND
    ax.fill_between(
        result["year"],
        result["lower_delta"],
        result["upper_delta"],
        color=line_color,
        alpha=0.2,
        label="Uncertainty"
    )

    # Annotate y axis
    ax.set_yticks([])            # remove ticks
    ax.set_yticklabels([])       # remove labels

    # Get y-limits for positioning
    ymin, ymax = ax.get_ylim()

    # --- Top: more unstable ---
    ax.annotate(
        "More unstable ↑",
        xy=(1.02, ymax-20), xycoords=('axes fraction', 'data'),
        xytext=(1.02, ymax),
        textcoords=('axes fraction', 'data'),
        ha='left', va='bottom',
        fontsize=14,
    )

    # --- Bottom: more stable ---
    ax.annotate(
        "More stable ↓",
        xy=(1.02, ymin+20), xycoords=('axes fraction', 'data'),
        xytext=(1.02, ymin),
        textcoords=('axes fraction', 'data'),
        ha='left', va='top',
        fontsize=14,
    )

    # LABELS
    ax.set_xlabel("Year")
    ax.set_ylabel("Biodiversity Instability")
    ax.set_title(f"{selected_state} — projected ecological change")

    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # -----------------------------
    # OPTIONAL TABLE
    # -----------------------------
    with st.expander("Show simulation data"):
        st.dataframe(result)


st.caption("""
    NB! Disclaimer: This application is a simplified exploratory modelling tool based on coarse-resolution biodiversity and climate data. 
    Results should be interpreted as illustrative rather than predictive.

    The model does not account for land-use change, conservation policy, species interactions, extreme ecological tipping points, or other complex ecological processes.

    Outputs reflect climate-related signals under simplified assumptions and are intended for demonstration purposes only.
""")