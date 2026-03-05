# 🏧 ATM Intelligence Demand Forecasting
**FA-2 | Data Mining | Year 1 | FinTrust Bank Ltd.**

## About
This interactive Streamlit dashboard analyses synthetic ATM transaction 
data for FinTrust Bank. It transforms raw data into actionable insights 
by applying core data mining techniques across four stages.

## What the App Does

### Stage 3 — Exploratory Data Analysis (EDA)
Visualises withdrawal and deposit distributions, time-based trends, 
holiday and event impacts, weather effects, competitor density, 
and feature correlations through histograms, box plots, line charts, 
bar charts, scatter plots, and a correlation heatmap.

### Stage 4 — Clustering
Groups ATMs into meaningful demand clusters (High, Steady, Low) 
using K-Means clustering with standardised features. 
Includes Elbow Method and Silhouette Score charts to justify K selection.

### Stage 5 — Anomaly Detection
Detects unusual withdrawal spikes using Z-Score and IQR methods. 
Highlights anomalies linked to holidays and special events, 
helping managers anticipate and respond to cash shortages.

### Stage 6 — Interactive Planner
Allows filtering by ATM ID, location, day, and time of day 
to explore demand patterns for any specific segment, 
with actionable management recommendations.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`
3. Open browser at: `http://localhost:8501`

## Requirements
- Python 3.8+
- streamlit
- pandas
- numpy
- plotly
- scikit-learn

## Conclusion
The dashboard successfully demonstrates how data mining techniques 
can transform raw ATM transaction data into operational intelligence. 
Clustering reveals that Urban ATMs drive the highest demand and need 
daily replenishment, while Rural ATMs can operate on bi-weekly cycles. 
Anomaly detection confirms that holidays and special events cause 
significant withdrawal spikes, requiring proactive cash pre-stocking. 
Together, these insights provide FinTrust Bank managers with a 
reproducible, data-driven tool to optimise ATM cash management, 
reduce outage risk, and cut unnecessary logistics costs.

---
*All data is synthetic and generated for demonstration purposes only.*
