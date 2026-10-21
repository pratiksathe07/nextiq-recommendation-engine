"""
NextIQ — Advanced Commercial Analytics & Customer Segmentation Module
Computes RFM (Recency, Frequency, Monetary) segmentation, Pareto analysis,
and temporal buying patterns for business intelligence.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import (
    BRAND_ACCENT,
    BRAND_DARK,
    BRAND_PRIMARY,
    BRAND_SECONDARY,
    BRAND_SUCCESS,
    BRAND_WARNING,
    COL_CUSTOMER_ID,
    COL_DESCRIPTION,
    COL_INVOICE_DATE,
    COL_INVOICE_NO,
    COL_QUANTITY,
    COL_REVENUE,
    COL_STOCK_CODE,
    PLOTLY_TEMPLATE,
)


def compute_rfm_segments(sales_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes Recency, Frequency, Monetary customer value metrics and assigns behavioral segments.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - rfm_df: Customer-level table with R, F, M values and assigned segments.
            - segment_summary: Summary table of customer count and total revenue by segment.
    """
    valid = sales_df.dropna(subset=[COL_CUSTOMER_ID]).copy()
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()

    snapshot_date = valid[COL_INVOICE_DATE].max() + pd.Timedelta(days=1)

    rfm = (
        valid.groupby(COL_CUSTOMER_ID)
        .agg(
            recency_days=(COL_INVOICE_DATE, lambda x: (snapshot_date - x.max()).days),
            frequency_invoices=(COL_INVOICE_NO, "nunique"),
            monetary_value=(COL_REVENUE, "sum"),
        )
        .reset_index()
    )

    rfm["monetary_value"] = rfm["monetary_value"].round(2)

    # Compute quantile scores (1 to 4)
    # For Recency, lower days is better -> invert labels
    rfm["r_score"] = pd.qcut(rfm["recency_days"].rank(method="first"), q=4, labels=[4, 3, 2, 1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency_invoices"].rank(method="first"), q=4, labels=[1, 2, 3, 4]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary_value"].rank(method="first"), q=4, labels=[1, 2, 3, 4]).astype(int)

    def assign_segment(row) -> str:
        r = row["r_score"]
        f = row["f_score"]
        m = row["m_score"]

        if r >= 3 and f >= 3 and m >= 3:
            return "Champions"
        elif r >= 3 and f >= 2:
            return "Loyal Customers"
        elif r >= 3 and f == 1:
            return "Promising New"
        elif r == 2 and f >= 2:
            return "Needs Attention"
        elif r <= 2 and f >= 3:
            return "At Risk"
        elif r == 1 and f <= 2:
            return "Hibernating / Lost"
        else:
            return "Standard Customers"

    rfm["segment"] = rfm.apply(assign_segment, axis=1)

    # Segment summary
    summary = (
        rfm.groupby("segment")
        .agg(
            customer_count=(COL_CUSTOMER_ID, "count"),
            total_revenue=("monetary_value", "sum"),
            avg_recency=("recency_days", "mean"),
            avg_frequency=("frequency_invoices", "mean"),
            avg_monetary=("monetary_value", "mean"),
        )
        .reset_index()
    )
    summary["avg_recency"] = summary["avg_recency"].round(1)
    summary["avg_frequency"] = summary["avg_frequency"].round(1)
    summary["avg_monetary"] = summary["avg_monetary"].round(2)
    summary["total_revenue"] = summary["total_revenue"].round(2)
    summary = summary.sort_values("total_revenue", ascending=False).reset_index(drop=True)

    return rfm, summary


def plot_rfm_segments(segment_summary: pd.DataFrame) -> go.Figure:
    """Plots customer segment counts and revenue share."""
    fig = px.treemap(
        segment_summary,
        path=["segment"],
        values="total_revenue",
        color="avg_monetary",
        color_continuous_scale="RdBu_r",
        title="<b>Customer Value Segmentation (RFM Portfolio Treemap)</b>",
        hover_data=["customer_count", "avg_recency", "avg_frequency"],
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def get_pareto_analysis(sales_df: pd.DataFrame) -> Tuple[go.Figure, float, int]:
    """
    Computes product revenue concentration (Pareto 80/20 rule).
    Returns (Figure, pct_products_for_80_rev, total_products).
    """
    prod_rev = (
        sales_df.groupby(COL_STOCK_CODE)[COL_REVENUE]
        .sum()
        .reset_index()
        .sort_values(COL_REVENUE, ascending=False)
        .reset_index(drop=True)
    )

    total_revenue = prod_rev[COL_REVENUE].sum()
    prod_rev["cum_rev"] = prod_rev[COL_REVENUE].cumsum()
    prod_rev["cum_pct_rev"] = (prod_rev["cum_rev"] / total_revenue) * 100
    prod_rev["cum_pct_products"] = ((prod_rev.index + 1) / len(prod_rev)) * 100

    # Find product percentage corresponding to 80% revenue
    cutoff_row = prod_rev[prod_rev["cum_pct_rev"] >= 80.0].iloc[0] if not prod_rev.empty else None
    pct_for_80 = round(float(cutoff_row["cum_pct_products"]), 1) if cutoff_row is not None else 20.0
    total_products = len(prod_rev)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=prod_rev["cum_pct_products"],
            y=prod_rev["cum_pct_rev"],
            mode="lines",
            name="Cumulative Revenue %",
            line=dict(color=BRAND_PRIMARY, width=3),
        )
    )

    # 80/20 Reference Lines
    fig.add_hline(y=80, line_dash="dash", line_color=BRAND_WARNING, annotation_text="80% Revenue Cutoff")
    fig.add_vline(x=pct_for_80, line_dash="dash", line_color=BRAND_ACCENT, annotation_text=f"{pct_for_80}% Catalog")

    fig.update_layout(
        title=f"<b>Product Revenue Concentration (Pareto Curve: {pct_for_80}% of Catalog = 80% Revenue)</b>",
        template=PLOTLY_TEMPLATE,
        xaxis_title="Cumulative % of Catalog Products",
        yaxis_title="Cumulative % of Total Revenue",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig, pct_for_80, total_products


def plot_weekday_hour_activity(sales_df: pd.DataFrame) -> go.Figure:
    """Visualizes order intensity by day of week and hour of day."""
    df = sales_df.copy()
    df["Hour"] = df[COL_INVOICE_DATE].dt.hour
    df["DayOfWeek"] = df[COL_INVOICE_DATE].dt.day_name()

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"]

    matrix = (
        df.groupby(["DayOfWeek", "Hour"])[COL_INVOICE_NO]
        .nunique()
        .unstack(fill_value=0)
        .reindex(day_order)
    )

    fig = px.imshow(
        matrix,
        labels=dict(x="Hour of Day (UTC)", y="Day of Week", color="Order Count"),
        x=matrix.columns,
        y=matrix.index,
        color_continuous_scale="Viridis",
        title="<b>Commercial Ordering Intensity Heatmap (Day of Week vs Hour)</b>",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig
