# SpaceX Fuel Burn Anomaly Visualizer

A Streamlit web app that visualizes rocket **fuel burn rate** and flags **anomalies** by comparing actual burn against an expected (nominal) curve.

## Features
- Upload CSV or use the bundled sample.
- Expected curve via robust rolling smoothing.
- Dual anomaly logic: **Z-score** + optional **IsolationForest**.
- Stage-aware analysis and optional event markers.
- Plotly interactive charts + downloadable anomaly table.

## Input CSV
Minimum columns:
- `time_s` (seconds since T0)
- Either `burn_rate_kg_s` **or** `fuel_mass_kg` (app will derive burn rate from fuel mass)

Optional columns:
- `stage` (e.g., 1, 2)
- `event` (string labels like MECO, SECO)

See `data/sample_spacex_telemetry.csv` for a working example.

## Quickstart

```bash
# 1) Create & activate a virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).

## Notes
- Tweak the **smoothing window**, **z-score threshold**, and **IsolationForest** contamination in the sidebar.
- If you provide `fuel_mass_kg` instead of `burn_rate_kg_s`, the app derives burn rate as `-d(mass)/dt`.

---

Built with ❤️ using Streamlit + Plotly + scikit-learn.