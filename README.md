# WildSignal 

Final project for WBS data science bootcamp. A machine learning model that predicts biodiversity (in)stability in response to extreme weather events in Germany.

# 🌿 Biodiversity and Climate Change in Germany

## Overview

This project explores how biodiversity patterns in Germany may respond to changes in climate conditions over time. It combines large-scale species observation data with historical and simulated weather data to model ecological change across German federal states (Bundesländer).

A Streamlit web application allows users to explore historical trends and simulate future biodiversity under different climate scenarios.

---

## 🧠 Key Idea

Instead of directly measuring biodiversity as simple species counts, this project focuses on:

> **Ecological turnover** — how much the composition of species in a region changes over time.

This helps reduce bias from uneven observation effort and provides a more dynamic view of ecosystem change.

---

## 📊 Data Sources

- **GBIF (Global Biodiversity Information Facility)**  
  Species occurrence records across Germany (2004–2024)

- **Open-Meteo API**  
  Historical daily weather data per German state (2004-2024)

---

### 📦 Data Availability

Due to the large size of the raw GBIF and weather datasets, the repository does not include full raw data files.

Instead, a processed and aggregated dataset (`full_df.parquet`) is provided, which contains the final modelling-ready features used in analysis and the Streamlit application.

Raw data can be accessed directly via the GBIF and OpenMeteo APIs if required.

---

## ⚙️ Methodology

### 1. Data Processing
- Aggregated GBIF observations and weather data by state, year, and month
- Filtered rare species and low-frequency records
- Controlled for observation effort using statistical adjustments

### 2. Biodiversity Metric
- Constructed species presence lists per region and time period
- Computed ecological similarity using:
  - Jaccard similarity
  - Turnover (1 - Jaccard)
- Standardised values into residual/z-score form

### 3. Modelling Approach
- Machine learning models (XGBoost regression)
- Features derived from climate variables:
  - Temperature extremes
  - Drought/Heatwave/Flood risk indices
  - Seasonal weather patterns
  - Lagged & Rolling average weather features

### 4. Future Simulation
- Monte Carlo simulation of future climate scenarios:
  - Best case (~1.5°C warming)
  - Middle of the road (~3–4°C warming)
  - Business as usual (~6–8°C warming)
- Uncertainty estimated via repeated simulations

---

## 📈 Web Application

The project includes an interactive **Streamlit dashboard** where users can:

- Select German states via an interactive map
- Simulate future ecological change under different climate scenarios
- Visualise uncertainty in predictions

---

## ⚠️ Limitations

This is a **simplified research prototype**, and results should be interpreted accordingly.

Key limitations:

- Coarse spatial resolution (state-level aggregation)
- Sampling bias in GBIF observation data
- No explicit modelling of:
  - land-use change
  - conservation policy
  - species interactions
  - ecological tipping points
- Climate effects are isolated from other real-world drivers

This tool is intended for **exploratory and educational purposes**, not precise forecasting.

---

## 🧪 Tech Stack

- Python
- Pandas / NumPy
- DuckDB
- Scikit-learn / XGBoost
- GeoPandas / Shapely
- Plotly / Matplotlib
- Streamlit

---

## 🚀 Running the App Locally

1. Clone the repo  
git clone https://github.com/delmurphy/wildsignal.git
cd wildsignal  

2. Create the environment  
conda env create -f environment.yml  

3. Activate the environment  
conda activate wildsignal  

4. Run the streamlit app  
streamlit run app.py


## 📊 Presentation

A PDF presentation summarising the project, methodology, and key findings is included in this repository:

- `WildSignal.pdf`

It provides a visual overview of the analysis, modelling approach, and main results.


## 📁 Project Structure
.
├── regression_app.py
├── Data/
│   ├── Raw/
│   └── Processed/
├── models/
├── src/
│   └── utils.py
├── environment.yml
├── WildSignal.pdf
└── README.md


---

## 📌 Key Insight

This project highlights that:

Changes in ecological systems are not always about “more or less biodiversity,” but about **how stable or dynamic ecosystems are under changing environmental conditions.**

---

## 📜 Disclaimer

This project is a simplified modelling exercise using aggregated ecological and climate data. It does not account for land-use change, ecological tipping points, or policy interventions, and should not be interpreted as a predictive ecological forecast.