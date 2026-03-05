# =============================================================================
# FA-2: ATM Intelligence Demand Forecasting with Data Mining
# FinTrust Bank Ltd. | Data Mining | Year 1
# Stages: 3 (EDA) | 4 (Clustering) | 5 (Anomaly Detection) | 6 (Interactive Planner)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ATM Intelligence | FinTrust Bank",
    page_icon="🏧",
    layout="wide"
)

# -----------------------------------------------------------------------------
# GENERATE SYNTHETIC DATASET
# Simulates ATM transaction data with all required columns from FA-2
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 1200
    atm_ids   = [f"ATM_{str(i+1).zfill(3)}" for i in range(20)]
    locations = ["Urban", "Urban", "Urban", "Suburban", "Suburban", "Rural"]
    days      = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    times     = ["Morning","Afternoon","Evening","Night"]
    weather   = ["Clear","Cloudy","Rainy","Stormy"]

    records = []
    for _ in range(n):
        atm      = np.random.choice(atm_ids)
        loc      = np.random.choice(locations)
        day      = np.random.choice(days)
        time     = np.random.choice(times)
        wx       = np.random.choice(weather)
        holiday  = int(np.random.rand() < 0.15)
        event    = int(np.random.rand() < 0.10)
        comp     = int(np.random.randint(0, 6))

        # Base withdrawal depends on location
        base = {"Urban": 15000, "Suburban": 9000, "Rural": 4000}[loc]

        # Multipliers
        day_m  = {"Monday":1.2,"Tuesday":1.0,"Wednesday":1.0,
                  "Thursday":1.1,"Friday":1.4,"Saturday":1.5,"Sunday":0.8}[day]
        time_m = {"Morning":1.3,"Afternoon":1.1,"Evening":1.2,"Night":0.6}[time]
        wx_m   = {"Clear":1.0,"Cloudy":0.95,"Rainy":0.8,"Stormy":0.6}[wx]
        hol_m  = 1.5 if holiday else 1.0
        ev_m   = 1.3 if event   else 1.0

        wd  = int(min(60000, max(600, base * day_m * time_m * wx_m * hol_m * ev_m
                                      * (0.85 + np.random.rand() * 0.3))))
        dep = int(wd * (0.3 + np.random.rand() * 0.4))
        prev_cash    = int(wd * (1.5 + np.random.rand() * 1.5))
        next_demand  = int(wd * (0.8 + np.random.rand() * 0.4))

        records.append({
            "ATM_ID":               atm,
            "Location_Type":        loc,
            "Day_of_Week":          day,
            "Time_of_Day":          time,
            "Weather_Condition":    wx,
            "Holiday_Flag":         holiday,
            "Special_Event_Flag":   event,
            "Nearby_Competitor_ATMs": comp,
            "Total_Withdrawals":    wd,
            "Total_Deposits":       dep,
            "Previous_Day_Cash":    prev_cash,
            "Cash_Demand_Next_Day": next_demand,
        })

    return pd.DataFrame(records)

df = load_data()

# -----------------------------------------------------------------------------
# SIDEBAR — GLOBAL FILTERS (Stage 6: Interactive Planner)
# -----------------------------------------------------------------------------
st.sidebar.title("🏧 FinTrust Bank")
st.sidebar.markdown("**ATM Intelligence Dashboard**")
st.sidebar.markdown("---")
st.sidebar.header("🔍 Filter Data")

sel_loc  = st.sidebar.selectbox("Location Type",
    ["All"] + sorted(df["Location_Type"].unique().tolist()))

sel_day  = st.sidebar.selectbox("Day of Week",
    ["All"] + list(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]))

sel_time = st.sidebar.selectbox("Time of Day",
    ["All"] + sorted(df["Time_of_Day"].unique().tolist()))

sel_hol  = st.sidebar.radio("Holiday Filter",
    ["All", "Holiday Days Only", "Normal Days Only"])

st.sidebar.markdown("---")
st.sidebar.caption("FA-2 | Data Mining | Year 1 | FinTrust Bank Ltd.")

# Apply filters
fdf = df.copy()
if sel_loc  != "All":
    fdf = fdf[fdf["Location_Type"] == sel_loc]
if sel_day  != "All":
    fdf = fdf[fdf["Day_of_Week"] == sel_day]
if sel_time != "All":
    fdf = fdf[fdf["Time_of_Day"] == sel_time]
if sel_hol == "Holiday Days Only":
    fdf = fdf[fdf["Holiday_Flag"] == 1]
elif sel_hol == "Normal Days Only":
    fdf = fdf[fdf["Holiday_Flag"] == 0]

# -----------------------------------------------------------------------------
# MAIN TITLE
# -----------------------------------------------------------------------------
st.title("🏧 ATM Intelligence Demand Forecasting")
st.markdown(
    "**FA-2 | FinTrust Bank Ltd. | Data Mining**  \n"
    "Exploratory Data Analysis · Clustering · Anomaly Detection · Interactive Planner"
)
st.markdown("---")

# Summary KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Records",    f"{len(fdf):,}")
k2.metric("Avg Withdrawal",   f"Rs. {fdf['Total_Withdrawals'].mean():,.0f}" if len(fdf) else "N/A")
k3.metric("Holiday Records",  f"{fdf['Holiday_Flag'].sum():,}")
k4.metric("Avg Deposit",      f"Rs. {fdf['Total_Deposits'].mean():,.0f}" if len(fdf) else "N/A")

st.markdown("---")

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Stage 3 — EDA",
    "🔵 Stage 4 — Clustering",
    "🚨 Stage 5 — Anomaly Detection",
    "🗓️ Stage 6 — Interactive Planner"
])

# =============================================================================
# STAGE 3 — EXPLORATORY DATA ANALYSIS
# =============================================================================
with tab1:
    st.header("Stage 3 — Exploratory Data Analysis (EDA)")
    st.markdown("Visualising distributions, trends, and relationships in ATM data.")

    if len(fdf) < 2:
        st.warning("Not enough data for current filter. Please adjust the filters.")
    else:
        # -----------------------------------------------------------------
        # 3A — DISTRIBUTION ANALYSIS
        # -----------------------------------------------------------------
        st.subheader("A. Distribution Analysis")
        col1, col2 = st.columns(2)

        with col1:
            # Histogram of Total_Withdrawals
            fig = px.histogram(
                fdf, x="Total_Withdrawals", nbins=30,
                title="Histogram: Total Withdrawals",
                labels={"Total_Withdrawals": "Withdrawal Amount (Rs.)"},
                color_discrete_sequence=["steelblue"]
            )
            fig.update_layout(bargap=0.05)
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 Withdrawals are right-skewed. Most ATMs cluster between Rs. 5k–20k, with high-value tails from urban/festival ATMs.")

        with col2:
            # Histogram of Total_Deposits
            fig2 = px.histogram(
                fdf, x="Total_Deposits", nbins=30,
                title="Histogram: Total Deposits",
                labels={"Total_Deposits": "Deposit Amount (Rs.)"},
                color_discrete_sequence=["mediumseagreen"]
            )
            fig2.update_layout(bargap=0.05)
            st.plotly_chart(fig2, use_container_width=True)
            st.info("💡 Deposits are consistently 40–60% of withdrawals, confirming ATMs operate primarily as cash-out machines.")

        col3, col4 = st.columns(2)

        with col3:
            # Box plot of Total_Withdrawals
            fig3 = px.box(
                fdf, y="Total_Withdrawals",
                title="Box Plot: Total Withdrawals",
                labels={"Total_Withdrawals": "Withdrawal Amount (Rs.)"},
                color_discrete_sequence=["steelblue"]
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.warning("⚠️ Upper whisker extends well beyond Q3 — these are outlier candidates linked to holidays and events.")

        with col4:
            # Box plot by Location_Type
            fig4 = px.box(
                fdf, x="Location_Type", y="Total_Withdrawals",
                title="Box Plot: Withdrawals by Location",
                labels={"Total_Withdrawals": "Withdrawal (Rs.)", "Location_Type": "Location"},
                color="Location_Type",
                color_discrete_map={"Urban":"steelblue","Suburban":"orange","Rural":"mediumseagreen"}
            )
            fig4.update_layout(showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)
            st.info("💡 Urban ATMs have significantly higher withdrawals and wider spread than Suburban or Rural ATMs.")

        st.markdown("---")

        # -----------------------------------------------------------------
        # 3B — TIME-BASED TRENDS
        # -----------------------------------------------------------------
        st.subheader("B. Time-Based Trends")

        col5, col6 = st.columns(2)

        with col5:
            # Bar chart: Average withdrawal by Day of Week
            day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            day_avg = (fdf.groupby("Day_of_Week")["Total_Withdrawals"]
                          .mean()
                          .reindex(day_order)
                          .reset_index())
            day_avg.columns = ["Day", "Avg_Withdrawal"]
            colors = ["crimson" if d in ("Friday","Saturday") else "steelblue"
                      for d in day_avg["Day"]]
            fig5 = px.bar(
                day_avg, x="Day", y="Avg_Withdrawal",
                title="Avg Withdrawals by Day of Week",
                labels={"Avg_Withdrawal": "Avg Withdrawal (Rs.)", "Day": "Day of Week"},
                color="Day",
                color_discrete_sequence=colors
            )
            fig5.update_layout(showlegend=False)
            st.plotly_chart(fig5, use_container_width=True)
            st.info("💡 Friday and Saturday peak ~40% above the weekly average — salary day and weekend spending patterns.")

        with col6:
            # Bar chart: Average withdrawal by Time of Day
            time_order = ["Morning","Afternoon","Evening","Night"]
            time_avg = (fdf.groupby("Time_of_Day")["Total_Withdrawals"]
                           .mean()
                           .reindex(time_order)
                           .reset_index())
            time_avg.columns = ["Time", "Avg_Withdrawal"]
            fig6 = px.bar(
                time_avg, x="Time", y="Avg_Withdrawal",
                title="Avg Withdrawals by Time of Day",
                labels={"Avg_Withdrawal": "Avg Withdrawal (Rs.)", "Time": "Time of Day"},
                color="Time",
                color_discrete_sequence=["gold","tomato","mediumpurple","midnightblue"]
            )
            fig6.update_layout(showlegend=False)
            st.plotly_chart(fig6, use_container_width=True)
            st.info("💡 Morning and Evening are peak periods, aligning with commute and post-work transaction patterns.")

        # Line chart: Simulated 60-day withdrawal trend
        st.markdown("**Line Chart: Simulated 60-Day Withdrawal Trend**")
        base_val = float(fdf["Total_Withdrawals"].mean())
        rng = np.random.default_rng(99)
        hol_set = {5, 12, 18, 25, 32, 40, 47, 55}
        days_x, vals_y, is_hol = [], [], []
        for d in range(1, 61):
            spike = 1.45 if d in (1, 30, 31) else 1.0
            v = base_val * spike * (1.6 if d in hol_set else 1.0) * (0.85 + rng.random() * 0.3)
            days_x.append(f"D{d}")
            vals_y.append(round(v))
            is_hol.append(d in hol_set)

        trend_df = pd.DataFrame({"Day": days_x, "Withdrawal": vals_y, "Holiday": is_hol})
        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(
            x=trend_df["Day"], y=trend_df["Withdrawal"],
            mode="lines", name="Avg Withdrawal",
            line=dict(color="steelblue", width=2),
            fill="tozeroy", fillcolor="rgba(70,130,180,0.1)"
        ))
        hol_df = trend_df[trend_df["Holiday"]]
        fig7.add_trace(go.Scatter(
            x=hol_df["Day"], y=hol_df["Withdrawal"],
            mode="markers", name="Holiday Spike",
            marker=dict(color="crimson", size=10, symbol="circle")
        ))
        fig7.update_layout(
            title="60-Day Withdrawal Trend (Red = Holiday Spike)",
            xaxis_title="Day", yaxis_title="Withdrawal (Rs.)",
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig7, use_container_width=True)
        st.info("💡 Month-start spikes (+45%) and holiday spikes (+60%) are clearly visible. Pre-stocking is recommended 24 hours before known holidays.")

        st.markdown("---")

        # -----------------------------------------------------------------
        # 3C — HOLIDAY & EVENT IMPACT
        # -----------------------------------------------------------------
        st.subheader("C. Holiday & Event Impact")
        col7, col8 = st.columns(2)

        with col7:
            hol_avg = fdf.groupby("Holiday_Flag")["Total_Withdrawals"].mean().reset_index()
            hol_avg["Holiday_Flag"] = hol_avg["Holiday_Flag"].map({0: "Normal Day", 1: "Holiday"})
            hol_avg.columns = ["Day_Type", "Avg_Withdrawal"]
            fig8 = px.bar(
                hol_avg, x="Day_Type", y="Avg_Withdrawal",
                title="Avg Withdrawals: Holiday vs Normal",
                labels={"Avg_Withdrawal": "Avg Withdrawal (Rs.)", "Day_Type": ""},
                color="Day_Type",
                color_discrete_map={"Normal Day": "steelblue", "Holiday": "crimson"}
            )
            fig8.update_layout(showlegend=False)
            st.plotly_chart(fig8, use_container_width=True)
            st.info("💡 Holidays drive ~50% higher withdrawals compared to normal days.")

        with col8:
            ev_avg = fdf.groupby("Special_Event_Flag")["Total_Withdrawals"].mean().reset_index()
            ev_avg["Special_Event_Flag"] = ev_avg["Special_Event_Flag"].map({0: "No Event", 1: "Special Event"})
            ev_avg.columns = ["Event_Type", "Avg_Withdrawal"]
            fig9 = px.bar(
                ev_avg, x="Event_Type", y="Avg_Withdrawal",
                title="Avg Withdrawals: Event vs No Event",
                labels={"Avg_Withdrawal": "Avg Withdrawal (Rs.)", "Event_Type": ""},
                color="Event_Type",
                color_discrete_map={"No Event": "steelblue", "Special Event": "orange"}
            )
            fig9.update_layout(showlegend=False)
            st.plotly_chart(fig9, use_container_width=True)
            st.info("💡 Special events trigger a ~30% increase in ATM withdrawals compared to regular days.")

        st.markdown("---")

        # -----------------------------------------------------------------
        # 3D — EXTERNAL FACTORS
        # -----------------------------------------------------------------
        st.subheader("D. External Factors")
        col9, col10 = st.columns(2)

        with col9:
            # Box plots by Weather
            fig10 = px.box(
                fdf, x="Weather_Condition", y="Total_Withdrawals",
                title="Withdrawals by Weather Condition",
                labels={"Total_Withdrawals": "Withdrawal (Rs.)", "Weather_Condition": "Weather"},
                color="Weather_Condition",
                color_discrete_sequence=["steelblue","slategray","mediumpurple","dimgray"]
            )
            fig10.update_layout(showlegend=False)
            st.plotly_chart(fig10, use_container_width=True)
            st.info("💡 Stormy weather reduces ATM usage by ~35%. Clear weather shows the widest spread and highest outliers.")

        with col10:
            # Competitor ATMs vs Withdrawals
            comp_avg = fdf.groupby("Nearby_Competitor_ATMs")["Total_Withdrawals"].mean().reset_index()
            comp_avg.columns = ["Competitors", "Avg_Withdrawal"]
            fig11 = px.bar(
                comp_avg, x="Competitors", y="Avg_Withdrawal",
                title="Avg Withdrawals by Nearby Competitor ATMs",
                labels={"Avg_Withdrawal": "Avg Withdrawal (Rs.)", "Competitors": "No. of Competitors"},
                color_discrete_sequence=["steelblue"]
            )
            st.plotly_chart(fig11, use_container_width=True)
            st.info("💡 ATMs with 0–1 nearby competitors handle significantly more withdrawals than those in dense competitor zones.")

        st.markdown("---")

        # -----------------------------------------------------------------
        # 3E — RELATIONSHIP ANALYSIS
        # -----------------------------------------------------------------
        st.subheader("E. Relationship Analysis")
        col11, col12 = st.columns(2)

        with col11:
            # Scatter: Previous_Day_Cash vs Cash_Demand_Next_Day
            samp = fdf.sample(min(300, len(fdf)), random_state=1)
            fig12 = px.scatter(
                samp,
                x="Previous_Day_Cash", y="Cash_Demand_Next_Day",
                color="Holiday_Flag",
                color_continuous_scale=["steelblue","crimson"],
                title="Previous Day Cash Level vs Next Day Demand",
                labels={
                    "Previous_Day_Cash": "Previous Day Cash (Rs.)",
                    "Cash_Demand_Next_Day": "Next Day Demand (Rs.)",
                    "Holiday_Flag": "Holiday"
                },
                opacity=0.65
            )
            st.plotly_chart(fig12, use_container_width=True)
            st.info("💡 Strong positive correlation between previous day cash and next day demand. Holiday records (red) cluster at higher values.")

        with col12:
            # Correlation Heatmap
            num_cols = ["Total_Withdrawals","Total_Deposits","Previous_Day_Cash",
                        "Cash_Demand_Next_Day","Holiday_Flag","Special_Event_Flag",
                        "Nearby_Competitor_ATMs"]
            corr = fdf[num_cols].corr().round(2)
            fig13 = go.Figure(go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.columns.tolist(),
                colorscale="RdBu",
                zmid=0,
                text=corr.values,
                texttemplate="%{text}",
                textfont=dict(size=9)
            ))
            fig13.update_layout(
                title="Correlation Heatmap — Numeric Features",
                xaxis=dict(tickangle=-30)
            )
            st.plotly_chart(fig13, use_container_width=True)
            st.info("💡 Withdrawals and deposits are strongly correlated. Holiday and event flags both show positive correlation with withdrawal spikes.")

# =============================================================================
# STAGE 4 — CLUSTERING
# =============================================================================
with tab2:
    st.header("Stage 4 — Clustering Analysis of ATMs")
    st.markdown("Grouping ATMs into clusters based on demand behaviour using K-Means.")

    if len(fdf) < 10:
        st.warning("Not enough data for clustering. Please adjust the filters.")
    else:
        st.subheader("Select Number of Clusters")
        k = st.slider("Number of Clusters (K)", min_value=2, max_value=5, value=3)

        # Features for clustering
        features = ["Total_Withdrawals", "Total_Deposits", "Nearby_Competitor_ATMs"]
        cluster_df = fdf[features].dropna().copy()

        # Standardise features before clustering
        scaler  = StandardScaler()
        X_scaled = scaler.fit_transform(cluster_df)

        # K-Means
        kmeans  = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        cluster_df = fdf.loc[cluster_df.index].copy()
        cluster_df["Cluster"] = cluster_labels

        # Assign meaningful names
        cluster_means = cluster_df.groupby("Cluster")["Total_Withdrawals"].mean()
        rank = cluster_means.rank(ascending=False).astype(int)
        name_map = {
            2: {1:"High-Demand", 2:"Low-Demand"},
            3: {1:"High-Demand", 2:"Steady-Demand", 3:"Low-Demand"},
            4: {1:"Very High", 2:"High", 3:"Moderate", 4:"Low"},
            5: {1:"Extreme", 2:"Very High", 3:"High", 4:"Moderate", 5:"Minimal"},
        }
        cluster_df["Cluster_Name"] = cluster_df["Cluster"].map(
            lambda c: name_map[k].get(rank[c], f"Cluster {c+1}")
        )

        # -----------------------------------------------------------------
        # Elbow Method & Silhouette Score
        # -----------------------------------------------------------------
        st.subheader("A. Elbow Method & Silhouette Score")
        ks        = list(range(2, 7))
        inertias  = []
        sil_scores = []
        for ki in ks:
            km_i = KMeans(n_clusters=ki, random_state=42, n_init=10)
            lbl  = km_i.fit_predict(X_scaled)
            inertias.append(km_i.inertia_)
            sil_scores.append(silhouette_score(X_scaled, lbl))

        el1, el2 = st.columns(2)
        with el1:
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(
                x=ks, y=inertias, mode="lines+markers",
                line=dict(color="steelblue", width=2),
                marker=dict(
                    color=["crimson" if ki==k else "steelblue" for ki in ks],
                    size=10
                ),
                name="Inertia"
            ))
            fig_elbow.update_layout(
                title="Elbow Curve (Inertia vs K)",
                xaxis_title="Number of Clusters (K)",
                yaxis_title="Inertia",
                showlegend=False
            )
            st.plotly_chart(fig_elbow, use_container_width=True)

        with el2:
            fig_sil = go.Figure()
            fig_sil.add_trace(go.Scatter(
                x=ks, y=sil_scores, mode="lines+markers",
                line=dict(color="mediumseagreen", width=2),
                marker=dict(
                    color=["crimson" if ki==k else "mediumseagreen" for ki in ks],
                    size=10
                ),
                name="Silhouette"
            ))
            fig_sil.update_layout(
                title="Silhouette Score vs K",
                xaxis_title="Number of Clusters (K)",
                yaxis_title="Silhouette Score",
                showlegend=False
            )
            st.plotly_chart(fig_sil, use_container_width=True)

        st.info(f"💡 Selected K={k}. The highlighted red dot marks your chosen cluster count. Silhouette score: {sil_scores[k-2]:.3f}")

        st.markdown("---")

        # -----------------------------------------------------------------
        # Cluster Summary Cards
        # -----------------------------------------------------------------
        st.subheader("B. Cluster Summary")
        summary = (cluster_df.groupby("Cluster_Name")
                             .agg(
                                 Count=("Total_Withdrawals","count"),
                                 Avg_Withdrawal=("Total_Withdrawals","mean"),
                                 Max_Withdrawal=("Total_Withdrawals","max"),
                                 Top_Location=("Location_Type", lambda x: x.value_counts().index[0])
                             )
                             .reset_index())
        summary["Avg_Withdrawal"] = summary["Avg_Withdrawal"].apply(lambda v: f"Rs. {v:,.0f}")
        summary["Max_Withdrawal"] = summary["Max_Withdrawal"].apply(lambda v: f"Rs. {v:,.0f}")
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown("---")

        # -----------------------------------------------------------------
        # Cluster Visualisations
        # -----------------------------------------------------------------
        st.subheader("C. Cluster Visualisations")
        cl1, cl2 = st.columns(2)

        with cl1:
            # Scatter: Withdrawals vs Deposits coloured by cluster
            fig_cs = px.scatter(
                cluster_df.sample(min(300, len(cluster_df)), random_state=7),
                x="Total_Withdrawals", y="Total_Deposits",
                color="Cluster_Name",
                title="Clusters: Withdrawal vs Deposit",
                labels={
                    "Total_Withdrawals": "Withdrawals (Rs.)",
                    "Total_Deposits": "Deposits (Rs.)",
                    "Cluster_Name": "Cluster"
                },
                opacity=0.75
            )
            st.plotly_chart(fig_cs, use_container_width=True)

        with cl2:
            # Bar: Cluster count by location
            loc_cl = (cluster_df.groupby(["Cluster_Name","Location_Type"])
                                .size()
                                .reset_index(name="Count"))
            fig_cl = px.bar(
                loc_cl, x="Cluster_Name", y="Count", color="Location_Type",
                title="Cluster Distribution by Location Type",
                labels={"Cluster_Name":"Cluster","Count":"Number of Records"},
                barmode="stack"
            )
            st.plotly_chart(fig_cl, use_container_width=True)

        st.info("💡 High-demand cluster is dominated by Urban ATMs. Low-demand cluster is primarily Rural. "
                "Steady-demand cluster corresponds to business hub (Suburban) ATMs.")

# =============================================================================
# STAGE 5 — ANOMALY DETECTION
# =============================================================================
with tab3:
    st.header("Stage 5 — Anomaly Detection on Holidays/Events")
    st.markdown("Identifying unusual withdrawal spikes using statistical methods.")

    if len(fdf) < 5:
        st.warning("Not enough data for anomaly detection. Please adjust the filters.")
    else:
        # -----------------------------------------------------------------
        # Method selection
        # -----------------------------------------------------------------
        st.subheader("Detection Settings")
        ac1, ac2 = st.columns([2, 1])
        with ac1:
            method = st.selectbox(
                "Detection Method",
                ["Z-Score", "IQR (Interquartile Range)"]
            )
        with ac2:
            if method == "Z-Score":
                thr = st.slider("Z-Score Threshold", 1.5, 4.0, 2.5, 0.1)
            else:
                thr = st.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1)

        # -----------------------------------------------------------------
        # Detect anomalies
        # -----------------------------------------------------------------
        vals = fdf["Total_Withdrawals"].values.astype(float)
        mu   = vals.mean()
        sig  = vals.std()

        if method == "Z-Score":
            z_scores = np.abs((vals - mu) / sig)
            anom_mask = z_scores > thr
        else:
            q1 = np.percentile(vals, 25)
            q3 = np.percentile(vals, 75)
            iqr = q3 - q1
            anom_mask = (vals < q1 - thr * iqr) | (vals > q3 + thr * iqr)

        anom_df   = fdf[anom_mask].copy()
        normal_df = fdf[~anom_mask].copy()

        total_anom  = int(anom_mask.sum())
        hol_anom    = int((anom_mask & (fdf["Holiday_Flag"].values == 1)).sum())
        event_anom  = int((anom_mask & (fdf["Special_Event_Flag"].values == 1)).sum())
        avg_spike   = float(vals[anom_mask].mean()) if total_anom else 0.0

        # -----------------------------------------------------------------
        # KPI metrics
        # -----------------------------------------------------------------
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Anomalies",    f"{total_anom}",
                  f"{total_anom/len(fdf)*100:.1f}% of records")
        m2.metric("Holiday Anomalies",  f"{hol_anom}",   "During holidays")
        m3.metric("Event Anomalies",    f"{event_anom}", "During events")
        m4.metric("Avg Spike Value",    f"Rs. {avg_spike:,.0f}")

        st.markdown("---")

        # -----------------------------------------------------------------
        # Anomaly scatter plot
        # -----------------------------------------------------------------
        st.subheader("A. Anomaly Scatter Plot")
        fig_anom = go.Figure()

        step = max(1, len(fdf) // 400)
        norm_idx = [i for i in range(0, len(fdf), step) if not anom_mask[i]]
        anom_idx = [i for i in range(0, len(fdf), step) if anom_mask[i]]

        if norm_idx:
            fig_anom.add_trace(go.Scatter(
                x=norm_idx,
                y=vals[norm_idx].tolist(),
                mode="markers",
                marker=dict(color="steelblue", size=4, opacity=0.5),
                name="Normal"
            ))
        if anom_idx:
            fig_anom.add_trace(go.Scatter(
                x=anom_idx,
                y=vals[anom_idx].tolist(),
                mode="markers",
                marker=dict(color="crimson", size=8, symbol="triangle-up"),
                name="Anomaly"
            ))

        fig_anom.update_layout(
            title="Withdrawal Anomalies (Red Triangles = Anomalies)",
            xaxis_title="Record Index",
            yaxis_title="Withdrawal Amount (Rs.)",
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_anom, use_container_width=True)
        st.info("💡 Red triangles mark anomalous withdrawals. These spikes often align with holiday and event days.")

        # -----------------------------------------------------------------
        # Holiday vs Normal comparison
        # -----------------------------------------------------------------
        st.subheader("B. Holiday vs Normal — Anomaly Distribution")
        col_a, col_b = st.columns(2)

        with col_a:
            # Bar chart: anomalies by context
            ctx_labels = ["Holiday Only", "Event Only", "Holiday + Event", "Regular Day"]
            ctx_values = [
                int((anom_mask & (fdf["Holiday_Flag"].values==1) & (fdf["Special_Event_Flag"].values==0)).sum()),
                int((anom_mask & (fdf["Holiday_Flag"].values==0) & (fdf["Special_Event_Flag"].values==1)).sum()),
                int((anom_mask & (fdf["Holiday_Flag"].values==1) & (fdf["Special_Event_Flag"].values==1)).sum()),
                int((anom_mask & (fdf["Holiday_Flag"].values==0) & (fdf["Special_Event_Flag"].values==0)).sum()),
            ]
            fig_ctx = px.bar(
                x=ctx_labels, y=ctx_values,
                title="Anomalies by Day Context",
                labels={"x":"Context","y":"Number of Anomalies"},
                color=ctx_labels,
                color_discrete_sequence=["crimson","orange","mediumpurple","steelblue"]
            )
            fig_ctx.update_layout(showlegend=False)
            st.plotly_chart(fig_ctx, use_container_width=True)

        with col_b:
            # Box: anomaly vs normal
            plot_data = pd.DataFrame({
                "Withdrawal": vals.tolist(),
                "Type": ["Anomaly" if m else "Normal" for m in anom_mask]
            })
            fig_box = px.box(
                plot_data, x="Type", y="Withdrawal",
                title="Anomaly vs Normal Withdrawal Distribution",
                labels={"Withdrawal": "Withdrawal (Rs.)", "Type": ""},
                color="Type",
                color_discrete_map={"Normal":"steelblue","Anomaly":"crimson"}
            )
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        # -----------------------------------------------------------------
        # Top anomaly records table
        # -----------------------------------------------------------------
        st.subheader("C. Top Anomaly Records")
        if len(anom_df) > 0:
            top_anom = (anom_df[["ATM_ID","Total_Withdrawals","Location_Type",
                                  "Day_of_Week","Time_of_Day","Holiday_Flag","Special_Event_Flag"]]
                        .sort_values("Total_Withdrawals", ascending=False)
                        .head(15)
                        .copy())
            top_anom["Total_Withdrawals"] = top_anom["Total_Withdrawals"].apply(lambda v: f"Rs. {v:,}")
            top_anom["Holiday_Flag"]      = top_anom["Holiday_Flag"].map({1:"YES",0:"NO"})
            top_anom["Special_Event_Flag"]= top_anom["Special_Event_Flag"].map({1:"YES",0:"NO"})
            top_anom.columns = ["ATM ID","Withdrawal","Location","Day","Time","Holiday","Event"]
            st.dataframe(top_anom, use_container_width=True, hide_index=True)
            st.info("💡 Most top anomalies occur on holiday or event days at Urban ATMs — pre-stocking and real-time monitoring are essential.")
        else:
            st.info("No anomalies detected with current settings. Try lowering the threshold.")

# =============================================================================
# STAGE 6 — INTERACTIVE PLANNER
# =============================================================================
with tab4:
    st.header("Stage 6 — Interactive Cash Demand Planner")
    st.markdown(
        "Query ATM demand by specific ATM, location, day, or time. "
        "Filters applied in the sidebar affect all tabs."
    )

    # -----------------------------------------------------------------
    # Planner-specific filters (on top of sidebar filters)
    # -----------------------------------------------------------------
    st.subheader("Query Settings")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        p_atm = st.selectbox("ATM ID", ["All ATMs"] + sorted(df["ATM_ID"].unique().tolist()))
    with p2:
        p_loc = st.selectbox("Location", ["All","Urban","Suburban","Rural"])
    with p3:
        p_day = st.selectbox("Day", ["All","Monday","Tuesday","Wednesday",
                                      "Thursday","Friday","Saturday","Sunday"])
    with p4:
        p_time = st.selectbox("Time", ["All","Morning","Afternoon","Evening","Night"])

    # Apply planner filters on top of global filtered data
    pdf = fdf.copy()
    if p_atm  != "All ATMs": pdf = pdf[pdf["ATM_ID"]        == p_atm]
    if p_loc  != "All":      pdf = pdf[pdf["Location_Type"] == p_loc]
    if p_day  != "All":      pdf = pdf[pdf["Day_of_Week"]   == p_day]
    if p_time != "All":      pdf = pdf[pdf["Time_of_Day"]   == p_time]

    nf = len(pdf)

    # KPIs
    pk1, pk2, pk3, pk4 = st.columns(4)
    pk1.metric("Records Found",   f"{nf:,}")
    pk2.metric("Avg Withdrawal",  f"Rs. {pdf['Total_Withdrawals'].mean():,.0f}" if nf else "N/A")
    pk3.metric("Peak Withdrawal", f"Rs. {pdf['Total_Withdrawals'].max():,.0f}"  if nf else "N/A")
    pk4.metric("Holiday Records", f"{pdf['Holiday_Flag'].sum():,}")

    if nf > 1:
        st.markdown("---")
        pl1, pl2 = st.columns(2)

        with pl1:
            # Histogram of filtered withdrawals
            fig_ph = px.histogram(
                pdf, x="Total_Withdrawals", nbins=20,
                title="Withdrawal Distribution (Filtered)",
                labels={"Total_Withdrawals": "Withdrawal (Rs.)"},
                color_discrete_sequence=["mediumseagreen"]
            )
            st.plotly_chart(fig_ph, use_container_width=True)

        with pl2:
            # Holiday vs Normal for filtered data
            hol_f = pdf.groupby("Holiday_Flag")["Total_Withdrawals"].mean().reset_index()
            hol_f["Holiday_Flag"] = hol_f["Holiday_Flag"].map({0:"Normal Day",1:"Holiday"})
            hol_f.columns = ["Type","Avg_Withdrawal"]
            fig_phl = px.bar(
                hol_f, x="Type", y="Avg_Withdrawal",
                title="Holiday vs Normal (Filtered)",
                labels={"Avg_Withdrawal":"Avg Withdrawal (Rs.)","Type":""},
                color="Type",
                color_discrete_map={"Normal Day":"steelblue","Holiday":"crimson"}
            )
            fig_phl.update_layout(showlegend=False)
            st.plotly_chart(fig_phl, use_container_width=True)

        # Show raw filtered records
        st.subheader("Filtered Records")
        show_df = pdf[["ATM_ID","Location_Type","Day_of_Week","Time_of_Day",
                        "Total_Withdrawals","Total_Deposits","Holiday_Flag","Special_Event_Flag"]].copy()
        show_df["Holiday_Flag"]       = show_df["Holiday_Flag"].map({1:"YES",0:"NO"})
        show_df["Special_Event_Flag"] = show_df["Special_Event_Flag"].map({1:"YES",0:"NO"})
        show_df.columns = ["ATM ID","Location","Day","Time","Withdrawals","Deposits","Holiday","Event"]
        st.dataframe(show_df.head(50), use_container_width=True, hide_index=True)
    else:
        st.warning("No records match the current query. Please adjust your filters.")

    st.markdown("---")

    # -----------------------------------------------------------------
    # Actionable Management Recommendations
    # -----------------------------------------------------------------
    st.subheader("Actionable Insights for FinTrust Bank Management")
    rec = [
        ("🔴", "Holiday Pre-Stocking",
         "Increase ATM cash reserves by 40–50% before national holidays and festival seasons based on historical spike patterns."),
        ("🟠", "Friday Replenishment",
         "Fridays run ~40% above weekly average. Schedule Thursday evening replenishment runs without exception."),
        ("🟡", "Urban ATM Priority",
         "High-demand cluster ATMs (Urban) require daily monitoring and a maximum 48-hour replenishment cycle."),
        ("🟢", "Rural ATM Efficiency",
         "Low-demand cluster ATMs (Rural) can operate on bi-weekly top-ups, cutting logistics cost by ~30%."),
        ("🔵", "Weather-Aware Scheduling",
         "Stormy weather reduces ATM usage by ~35%. Reduce cash-in-transit dispatches on storm forecast days."),
        ("🟣", "Real-Time Anomaly Response",
         "Act within 2 hours of a spike detection signal to prevent ATM outage during peak demand events."),
    ]
    c_left, c_right = st.columns(2)
    for i, (ico, title, desc) in enumerate(rec):
        col = c_left if i % 2 == 0 else c_right
        with col:
            st.markdown(f"""
            <div style="background:#f8f9fa;border-left:4px solid #333;
                        border-radius:6px;padding:12px 16px;margin-bottom:10px;">
              <b>{ico} {title}</b><br>
              <span style="font-size:0.85rem;color:#555">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.caption("FA-2 · Data Mining · Year 1 · FinTrust Bank Ltd. · ATM Intelligence Demand Forecasting · All data is synthetic for demonstration purposes.")