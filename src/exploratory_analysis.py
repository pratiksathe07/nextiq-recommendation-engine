"""
NextIQ — Exploratory Data Analysis (EDA) & Interactive Visualization Module
Computes macro statistics, distributions, and generates high-impact Plotly visualizations.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    BRAND_ACCENT,
    BRAND_DARK,
    BRAND_PRIMARY,
    BRAND_SECONDARY,
    BRAND_SUCCESS,
    BRAND_WARNING,
    COL_COUNTRY,
    COL_CUSTOMER_ID,
    COL_DESCRIPTION,
    COL_INVOICE_DATE,
    COL_INVOICE_NO,
    COL_QUANTITY,
    COL_REVENUE,
    COL_STOCK_CODE,
    COL_UNIT_PRICE,
    CURRENCY_SYMBOL,
    PLOTLY_TEMPLATE,
)
from src.data_processing import CleaningReport


def calculate_kpis(sales_df: pd.DataFrame, report: Optional[CleaningReport] = None) -> Dict[str, Any]:
    """
    Computes primary key performance indicators from valid transaction records.
    """
    total_rev = sales_df[COL_REVENUE].sum() if COL_REVENUE in sales_df.columns else 0.0
    total_qty = int(sales_df[COL_QUANTITY].sum()) if COL_QUANTITY in sales_df.columns else 0
    unique_inv = int(sales_df[COL_INVOICE_NO].nunique()) if COL_INVOICE_NO in sales_df.columns else 0
    unique_cust = int(sales_df[COL_CUSTOMER_ID].dropna().nunique()) if COL_CUSTOMER_ID in sales_df.columns else 0
    unique_items = int(sales_df[COL_STOCK_CODE].nunique()) if COL_STOCK_CODE in sales_df.columns else 0

    avg_order_value = round(total_rev / unique_inv, 2) if unique_inv > 0 else 0.0
    avg_basket_units = round(total_qty / unique_inv, 1) if unique_inv > 0 else 0.0

    retention_pct = 0.0
    if report and report.raw_rows > 0:
        retention_pct = round((report.clean_sales_rows / report.raw_rows) * 100, 2)

    return {
        "total_revenue": round(total_rev, 2),
        "total_quantity": total_qty,
        "unique_invoices": unique_inv,
        "unique_customers": unique_cust,
        "unique_items": unique_items,
        "avg_order_value": avg_order_value,
        "avg_basket_units": avg_basket_units,
        "currency": CURRENCY_SYMBOL,
        "retention_pct": retention_pct,
    }


def plot_monthly_sales_trend(sales_df: pd.DataFrame) -> go.Figure:
    """
    Generates a dual-axis chart showing monthly revenue and order volume.
    """
    df = sales_df.copy()
    df["YearMonth"] = df[COL_INVOICE_DATE].dt.to_period("M").astype(str)

    monthly = (
        df.groupby("YearMonth")
        .agg(
            revenue=(COL_REVENUE, "sum"),
            invoices=(COL_INVOICE_NO, "nunique"),
            quantity=(COL_QUANTITY, "sum"),
        )
        .reset_index()
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Revenue Bar Chart
    fig.add_trace(
        go.Bar(
            x=monthly["YearMonth"],
            y=monthly["revenue"],
            name=f"Revenue ({CURRENCY_SYMBOL})",
            marker_color=BRAND_PRIMARY,
            opacity=0.85,
            hovertemplate=f"<b>%{{x}}</b><br>Revenue: {CURRENCY_SYMBOL}%{{y:,.2f}}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Invoices Line Chart
    fig.add_trace(
        go.Scatter(
            x=monthly["YearMonth"],
            y=monthly["invoices"],
            name="Distinct Invoices",
            mode="lines+markers",
            line=dict(color=BRAND_ACCENT, width=3),
            marker=dict(size=7, color=BRAND_ACCENT),
            hovertemplate="<b>%{x}</b><br>Orders: %{y:,}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="<b>Monthly Revenue and Order Volume Trends</b>",
        template=PLOTLY_TEMPLATE,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_yaxes(title_text=f"Revenue ({CURRENCY_SYMBOL})", secondary_y=False, showgrid=True)
    fig.update_yaxes(title_text="Total Orders", secondary_y=True, showgrid=False)

    return fig


def plot_top_products(sales_df: pd.DataFrame, by: str = "revenue", top_n: int = 10) -> go.Figure:
    """
    Plots the top N products ranked by chosen dimension.
    Supported 'by': 'revenue', 'quantity', 'customers', 'invoices'.
    """
    metric_map = {
        "revenue": (COL_REVENUE, "sum", f"Revenue ({CURRENCY_SYMBOL})", BRAND_PRIMARY, f"{CURRENCY_SYMBOL}%{{x:,.2f}}"),
        "quantity": (COL_QUANTITY, "sum", "Units Purchased", BRAND_SECONDARY, "%{x:,}"),
        "customers": (COL_CUSTOMER_ID, "nunique", "Distinct Customers", BRAND_ACCENT, "%{x:,}"),
        "invoices": (COL_INVOICE_NO, "nunique", "Distinct Orders", BRAND_SUCCESS, "%{x:,}"),
    }

    col, agg_fn, title_label, bar_color, hover_fmt = metric_map.get(
        by, (COL_REVENUE, "sum", f"Revenue ({CURRENCY_SYMBOL})", BRAND_PRIMARY, f"{CURRENCY_SYMBOL}%{{x:,.2f}}")
    )

    grouped = (
        sales_df.groupby([COL_STOCK_CODE, COL_DESCRIPTION])
        .agg(value=(col, agg_fn))
        .reset_index()
        .sort_values("value", ascending=True)
        .tail(top_n)
    )

    # Create descriptive label
    grouped["Display"] = grouped[COL_DESCRIPTION].str[:32] + " (" + grouped[COL_STOCK_CODE] + ")"

    fig = go.Figure(
        go.Bar(
            x=grouped["value"],
            y=grouped["Display"],
            orientation="h",
            marker=dict(
                color=grouped["value"],
                colorscale="Viridis",
                showscale=False,
            ),
            hovertemplate=f"<b>%{{y}}</b><br>{title_label}: {hover_fmt}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"<b>Top {top_n} Products by {title_label}</b>",
        template=PLOTLY_TEMPLATE,
        xaxis_title=title_label,
        yaxis_title="",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_customer_purchase_distribution(model_df: pd.DataFrame) -> go.Figure:
    """
    Visualizes the customer purchase frequency distribution (power law / long-tail).
    """
    cust_orders = model_df.groupby(COL_CUSTOMER_ID)[COL_INVOICE_NO].nunique().reset_index(name="order_count")

    # Clip for clean visualization of long tail
    cust_orders["binned_orders"] = cust_orders["order_count"].clip(upper=25)

    fig = px.histogram(
        cust_orders,
        x="binned_orders",
        nbins=25,
        title="<b>Customer Purchase Frequency Distribution (Pareto Long-Tail)</b>",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[BRAND_PRIMARY],
    )
    fig.update_layout(
        xaxis_title="Orders per Customer (Capped at 25+)",
        yaxis_title="Customer Count",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_country_sales_distribution(sales_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Visualizes revenue contribution by customer country of origin.
    """
    country_rev = (
        sales_df.groupby(COL_COUNTRY)[COL_REVENUE]
        .sum()
        .reset_index()
        .sort_values(COL_REVENUE, ascending=False)
    )

    top_countries = country_rev.head(top_n).copy()
    top_countries["Display"] = top_countries[COL_COUNTRY]

    fig = px.bar(
        top_countries,
        x=COL_COUNTRY,
        y=COL_REVENUE,
        title=f"<b>Top {top_n} Geographic Markets by Revenue ({CURRENCY_SYMBOL})</b>",
        template=PLOTLY_TEMPLATE,
        color=COL_REVENUE,
        color_continuous_scale="Purples",
    )
    fig.update_layout(
        xaxis_title="Country",
        yaxis_title=f"Total Revenue ({CURRENCY_SYMBOL})",
        coloraxis_showscale=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_traces(hovertemplate=f"<b>%{{x}}</b><br>Revenue: {CURRENCY_SYMBOL}%{{y:,.2f}}<extra></extra>")
    return fig


def plot_data_cleaning_funnel(report: CleaningReport) -> go.Figure:
    """
    Visualizes the data attrition and filtration funnel from raw to model-ready records.
    """
    stages = [
        "Raw Ingested Records",
        "Exact Duplicates Removed",
        "Cancellations & Returns Excluded",
        "Invalid Price / Qty Filtered",
        "Admin Service Codes Excluded",
        "Clean Commercial Sales",
        "Model-Ready Customer Transactions",
    ]

    values = [
        report.raw_rows,
        report.duplicates_removed,
        report.cancellations_identified,
        report.invalid_price_or_qty_rows,
        report.admin_stock_code_rows,
        report.clean_sales_rows,
        report.clean_model_rows,
    ]

    colors = [
        BRAND_PRIMARY,
        BRAND_WARNING,
        "#EF4444",  # Red
        "#F97316",  # Orange
        "#8B5CF6",  # Purple
        BRAND_ACCENT,
        BRAND_SUCCESS,
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=stages,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v:,}" for v in values],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Data Processing Pipeline Filtration Audit</b>",
        template=PLOTLY_TEMPLATE,
        xaxis_title="Record Count",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig
