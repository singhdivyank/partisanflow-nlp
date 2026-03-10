"""
Plotly chart builders for streamlit dashboard. 
Each function accepts a Pandas DataFrame and returns a plotly Figure.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils.constants import (
    DRIFT_PSI_PROB1, 
    PSI_CRITICAL_THRESHOLD, 
    PSI_WARNING_THRESHOLD
)

def plot_confidence_heatmap(preds_df: pd.DataFrame) -> go.Figure:
    """Represent year x probability bin with row-normalised density"""

    bins = np.linspace(0.0, 1.0, 11)
    labels = [f"{bin:.1f}" for bin in bins[:-1]]
    
    preds_df = preds_df.dropna(subset=["prob_republican"]).copy()
    preds_df["bin"] = pd.cut(preds_df["prob_republican"], bins=bins, labels=labels, include_lowest=True)

    heat_map = preds_df.groupby(["year", "bin"]).size().reset_index(name="count")
    pivot = heat_map.pivot(index="year", columns="bin", values="count").fillna(0)
    pivot = pivot.div(pivot.sum(axis=1), axis=0)

    fig = go.Figure(
        data=go.heatmap(
            z=pivot.values,
            x=pivot.columns.to_list(),
            y=pivot.index.to_list(),
            colorscale="Blues",
            colorbar=dict(title="Density"),
            zmin=0,
            zmax=1
        )
    )
    fig.update_layout(
        xaxis_title="Probability Bin",
        yaxis_title="Year",
        template="plotly_white"
    )
    
    return fig

def plot_drift_over_time(dfrift_df: pd.DataFrame) -> go.Figure:
    """Line chart of PSI and KL divergence over inference years."""

    psi_df = dfrift_df[dfrift_df["metric_name"] == DRIFT_PSI_PROB1].copy()
    fig = go.Figure()
    for model in psi_df["model_name"].unique():
        sub = psi_df[psi_df["model_name"] == model].sort_values("year")
        fig.add_trace(
            go.scatter(
                x=sub["year"],
                y=sub["metric_value"],
                model="lines+markers",
                name=f"{model} (PSI)"
            )
        )
    
    # threshold lines
    fig.add_hline(
        y=PSI_CRITICAL_THRESHOLD,
        line_dash="dash",
        line_color="red",
        annotation_text="Critical Threshold"
    )
    fig.add_hline(
        y=PSI_WARNING_THRESHOLD,
        line_dash="dash",
        line_color="orange",
        annotation_text="Warning Threshold"
    )

def plot_model_metrics_table(metrics_df: pd.DataFrame) -> go.Figure:
    """Table for model performance metrics on 1869 evaluation set."""

    pivot = metrics_df.pivot_table(
        index="model_name",
        columns="metric_name",
        values="metric_value",
    ).reset_index().round(4)

    fig = go.Figure(
        data=[go.Table(
            header=dict(
                values=list(pivot.columns),
                fill_color="#1f77b4",
                font=dict(color="white", size=12),
                align="left"
            ),
            cells=dict(
                values=[pivot[col] for col in pivot.columns],
                fill_color="lavender",
                align="left"
            ),
        )]
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    return fig

def plot_partisanship_trend(summary_df: pd.DataFrame) -> go.Figure:
    """Line chart of partisan fraction per year for republican data, with ci bands"""

    fig = go.Figure()

    for model in summary_df["model_name"].unique():
        summary_df = summary_df[summary_df["model_name"] == model].sort_values("year")
        fig.add_trace(
            go.Scatter(
                x=summary_df["year"],
                y=summary_df["partisan_fraction"],
                name=model,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=summary_df.get("ci_upper", summary_df["partisan_fraction"]),
                    arrayminus=summary_df["partisan_fraction"] - summary_df.get("ci_lower", summary_df["partisan_fraction"]),
                    visible=True
                ) if "ci_upper" in summary_df.columns else None
            )
        )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Fraction Predicted Partisan",
        yaxis=dict(range=[0, 1]),
        legend_title="Model",
        hovermode="x unified",
        template="plotly_white"
    )
    return fig

def plot_prob_distribution(preds_df: pd.DataFrame) -> go.Figure:
    """Histogram of republican probabilities for each year"""

    preds_df = preds_df.dropna(subset=["prob_republican"])

    fig = px.histogram(
        preds_df,
        x="prob_republican",
        color="year",
        facet_col="model_name",
        nbins=30,
        barmode="overlay",
        opacity=0.6,
        labels={"proc_republican": "P(republican)", "year": "Year"},
        template="plotly_white"
    )
    fig.update_layout(bargap=0.5)
    return fig
