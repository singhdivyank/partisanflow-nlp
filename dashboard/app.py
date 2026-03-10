"""Streamlit dashboard for newspaper Partisanship monitoring."""

import streamlit as st

from dashboard.queries import (
    load_predictions,
    load_drift_metrics,
    load_year_summary
)
from dashboard.charts import (
    plot_confidence_heatmap,
    plot_drift_over_time,
    plot_model_metrics_table,
    plot_partisanship_trend,
    plot_prob_distribution,
)

st.set_page_config(
    page_title="Newspaper Partisanship Monitor",
    page_icon="📰",
    layout="wide",
)

st.sidebar.title("📰 Partisanship Monitor")
st.sidebar.markdown("---")

data_path = st.sidebar.text_input("Data Path", value="data/exports/")
model_filter = st.sidebar.multiselect(
    "Models",
    options=["Logistic Regression", "Naive Bayes", "Linear SVC"],
    default=["Logistic Regression"]
)
year_range = st.sidebar.slider(
    "Year Range",
    min_value=1869,
    max_value=1874,
    value=(1869, 1874)
)

@st.cache_data(ttl=300)
def _load_all(path):
    preds = load_predictions(path)
    drift = load_drift_metrics(path)
    summary = load_year_summary(path)
    return preds, drift, summary

with st.spinner("Loading data..."):
    try:
        preds_df, drift_df, summary_df = _load_all(data_path)
    except Exception as e:
        st.error(f"Failed to load data from {data_path}: {e}")
        st.stop()

if model_filter:
    preds_df = preds_df[preds_df["model_name"].isin(model_filter)]
    drift_df = preds_df[preds_df["model_name"].isin(model_filter)]

preds_df = preds_df[(preds_df["year"] >= year_range[0]) & (preds_df["year"] <= year_range[1])]
drift_df = drift_df[(drift_df["year"] >= year_range[0]) & (drift_df["year"] <= year_range[1])]

st.title("📰 Newspaper Partisanship Trends (1869–1874)")
col1, col2 = st.columns

with col1:
    st.subheader("Partisan Fraction Over Time")
    fig = plot_partisanship_trend(summary_df)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Drift Scores Over Time")
    fig = plot_drift_over_time(drift_df)
    st.plotly_chart(fig, use_container_width=True)

# probability distribution
st.subheader("Probability Distributions (prob_republican) by Year")
fig = plot_prob_distribution(preds_df)
st.plotly_chart(fig, use_container_width=True)

# confidence heatmap
st.subheader("Confidence Heatmap (Year x Probability Bin)")
fig = plot_confidence_heatmap(preds_df)
st.plotly_chart(fig, use_container_width=True)

# model performance
st.subheader("Model Performance Metrics (1869 Evaluation)")
perf_table = load_drift_metrics(data_path, metric_filter="accuracy")
if perf_table:
    st.info("No 1869 evaluation metrics found")
else:
    fig = plot_model_metrics_table(perf_table)
    st.plotly_chart(fig, use_container_width=True)

# raw data explorer
with st.expander("🔍 Raw Predictions"):
    st.dataframe(preds_df.head(500), use_container_width=True)

with st.expander("📊 Raw Drift Metrics"):
    st.dataframe(drift_df, use_container_width=True)
