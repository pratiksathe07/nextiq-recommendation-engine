"""
NextIQ — Intelligent Personalized Recommendation Engine
Streamlit Interactive Dashboard & Recommender Application
"""

import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    ASSETS_DIR,
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
    DEFAULT_HYBRID_WEIGHTS,
    DEFAULT_TOP_K,
    RAW_CSV_PATH,
    SAMPLE_DATA_PATH,
)
from src.analytics import (
    compute_rfm_segments,
    get_pareto_analysis,
    plot_rfm_segments,
    plot_weekday_hour_activity,
)
from src.content_based import ContentBasedRecommender
from src.data_loader import get_dataset_summary, load_raw_data
from src.data_processing import (
    CleaningReport,
    build_interaction_matrix,
    clean_transactions,
    get_item_catalog,
)
from src.evaluation import (
    chronological_train_test_split,
    run_full_benchmark,
)
from src.exploratory_analysis import (
    calculate_kpis,
    plot_country_sales_distribution,
    plot_customer_purchase_distribution,
    plot_data_cleaning_funnel,
    plot_monthly_sales_trend,
    plot_top_products,
)
from src.hybrid_recommender import HybridRecommender
from src.recommender import (
    ItemItemCollaborativeRecommender,
    PopularityRecommender,
    UserUserCollaborativeRecommender,
)

# ----------------------------------------------------------------------------
# Page Configuration & Styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="NextIQ — Recommendation Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Custom CSS
css_path = ASSETS_DIR / "style.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Cached Data Ingestion & Model Resource Management
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_clean_data(selected_path_str: str):
    """Loads raw data, executes full cleaning pipeline, and returns dataframes and audit report."""
    raw_df = load_raw_data(selected_path_str)
    raw_summary = get_dataset_summary(raw_df)
    sales_df, model_df, report = clean_transactions(raw_df)
    catalog = get_item_catalog(sales_df)
    return raw_df, raw_summary, sales_df, model_df, report, catalog


@st.cache_resource(show_spinner=False)
def fit_recommendation_models(model_df: pd.DataFrame, catalog_df: pd.DataFrame):
    """Fits and caches all 5 recommendation algorithms to avoid retraining on every interaction."""
    pop_recommender = PopularityRecommender(ranking_metric="customers").fit(model_df, catalog_df)
    item_recommender = ItemItemCollaborativeRecommender().fit(model_df, catalog_df)
    user_recommender = UserUserCollaborativeRecommender().fit(model_df, catalog_df)
    content_recommender = ContentBasedRecommender().fit(model_df, catalog_df)
    hybrid_recommender = HybridRecommender().fit(model_df, catalog_df)
    return {
        "Popularity": pop_recommender,
        "Item-Item CF": item_recommender,
        "User-User CF": user_recommender,
        "Content-Based": content_recommender,
        "Hybrid": hybrid_recommender,
    }


@st.cache_data(show_spinner=False)
def get_cached_benchmark_results(model_df: pd.DataFrame, catalog_df: pd.DataFrame):
    """Runs chronological offline benchmark evaluation."""
    train_df, test_df, _ = chronological_train_test_split(model_df, test_ratio=0.20)
    results_df, results_dict = run_full_benchmark(train_df, test_df, catalog_df, max_eval_users=100)
    return results_df, results_dict


# ----------------------------------------------------------------------------
# Sidebar Navigation & Controls
# ----------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="margin-bottom: 1.25rem;">
        <h2 style="margin: 0; font-weight: 800; color: #4F46E5; letter-spacing: -0.5px;">⚡ NextIQ</h2>
        <p style="margin: 0; font-size: 0.85rem; color: #64748B; font-weight: 500;">Predict Preferences. Recommend Smarter.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Dataset Selection
available_sources = {}
if RAW_CSV_PATH.exists():
    available_sources["Full UCI Dataset (541k records)"] = str(RAW_CSV_PATH)
if SAMPLE_DATA_PATH.exists():
    available_sources["Verified Sample Dataset (5k records)"] = str(SAMPLE_DATA_PATH)

if not available_sources:
    st.error("No dataset found. Please run `python data/download_data.py --generate-sample`.")
    st.stop()

dataset_choice = st.sidebar.selectbox(
    "Active Dataset Source",
    options=list(available_sources.keys()),
    index=0,
    help="Switch between the full 541k UCI dataset and the lightweight sample.",
)
active_data_path = available_sources[dataset_choice]

# Page Navigation
nav_options = [
    "📊 Overview & Data Quality",
    "🎯 Personalized Recommendations",
    "🔍 Similar Products Discovery",
    "🌟 Popular Products Ranking",
    "🧪 Offline Model Evaluation",
    "📈 Commercial Business Insights",
    "ℹ️ Architecture & Documentation",
]
selected_page = st.sidebar.radio("Navigation", nav_options, index=0)

if st.sidebar.button("🔄 Reload / Refresh Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Antigravity Machine Learning Portfolio Project\nAuthor: NextIQ Team")


# ----------------------------------------------------------------------------
# Load Active Data
# ----------------------------------------------------------------------------
with st.spinner("Processing transaction data pipeline..."):
    raw_df, raw_summary, sales_df, model_df, report, catalog_df = load_and_clean_data(active_data_path)

catalog_lookup = dict(zip(catalog_df[COL_STOCK_CODE], catalog_df[COL_DESCRIPTION]))


# ----------------------------------------------------------------------------
# PAGE 1: OVERVIEW & DATA QUALITY
# ----------------------------------------------------------------------------
if selected_page == "📊 Overview & Data Quality":
    st.markdown('<div class="brand-title">NextIQ — Intelligent Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-tagline">Predict Preferences. Recommend Smarter. — Operational Overview & Data Health</div>', unsafe_allow_html=True)

    # Status Pill
    st.markdown(
        f'<span class="status-pill-success">✓ Data Pipeline Active: {len(sales_df):,} Clean Transactions Loaded</span> &nbsp; '
        f'<span class="status-pill-info">Source: {Path(active_data_path).name}</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    # High-Level KPI Cards
    kpis = calculate_kpis(sales_df, report)
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Valid Revenue</div>
                <div class="kpi-value">{kpis['currency']}{kpis['total_revenue']:,.2f}</div>
                <div class="kpi-subtext">Commercial gross sales</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Units Purchased</div>
                <div class="kpi-value">{kpis['total_quantity']:,}</div>
                <div class="kpi-subtext">Net merchandise items</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Invoices</div>
                <div class="kpi-value">{kpis['unique_invoices']:,}</div>
                <div class="kpi-subtext">Avg basket: {kpis['currency']}{kpis['avg_order_value']:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Unique Customers</div>
                <div class="kpi-value">{kpis['unique_customers']:,}</div>
                <div class="kpi-subtext">Identified purchasing accounts</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c5:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Catalog Products</div>
                <div class="kpi-value">{kpis['unique_items']:,}</div>
                <div class="kpi-subtext">Distinct StockCodes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c6:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Data Retention</div>
                <div class="kpi-value">{kpis['retention_pct']}%</div>
                <div class="kpi-subtext">{report.clean_sales_rows:,} / {report.raw_rows:,} rows</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualizations Row: Sales Trend & Funnel
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        fig_trend = plot_monthly_sales_trend(sales_df)
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_right:
        fig_funnel = plot_data_cleaning_funnel(report)
        st.plotly_chart(fig_funnel, use_container_width=True)

    # Secondary Row: Matrix Sparsity and Raw Ingestion Audit
    s_col1, s_col2 = st.columns([1, 1])

    with s_col1:
        st.subheader("Interaction Matrix Sparsity Analysis")
        st.write(
            f"The customer-item interaction matrix contains **{report.unique_customers:,} customers** and "
            f"**{report.unique_items:,} catalog items**, yielding **{report.unique_customers * report.unique_items:,} possible pairs**."
        )
        sparsity_pct = report.sparsity_ratio * 100
        st.metric("Matrix Sparsity", f"{sparsity_pct:.2f}%", help="Percentage of customer-item pairs with 0 interactions.")
        st.info(
            "High matrix sparsity (>99%) is characteristic of e-commerce retail. "
            "NextIQ leverages compressed sparse row (CSR) matrices and cosine indexing to compute similarities efficiently in memory."
        )

    with s_col2:
        st.subheader("Data Cleaning Audit Table")
        audit_table = pd.DataFrame([
            {"Stage / Rule": "Raw Ingested Records", "Count": report.raw_rows, "Impact": "Baseline"},
            {"Stage / Rule": "Exact Duplicate Removal", "Count": report.duplicates_removed, "Impact": "Deduplicated"},
            {"Stage / Rule": "Cancellations / Returns (Invoice 'C' & Qty ≤ 0)", "Count": report.cancellations_identified, "Impact": "Excluded from purchases"},
            {"Stage / Rule": "Invalid Prices / Quantities (Price ≤ 0, Qty ≤ 0)", "Count": report.invalid_price_or_qty_rows, "Impact": "Filtered"},
            {"Stage / Rule": "Administrative Codes (POST, D, M, etc.)", "Count": report.admin_stock_code_rows, "Impact": "Non-merchandise excluded"},
            {"Stage / Rule": "Clean Commercial Transactions", "Count": report.clean_sales_rows, "Impact": "Ready for sales analytics"},
            {"Stage / Rule": "Model-Ready Collaborative Pairs", "Count": report.clean_model_rows, "Impact": "Ready for recommender models"},
        ])
        st.dataframe(audit_table, use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# PAGE 2: PERSONALIZED RECOMMENDATIONS
# ----------------------------------------------------------------------------
elif selected_page == "🎯 Personalized Recommendations":
    st.markdown('<div class="brand-title">Personalized Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-tagline">Multi-model personalization engine with transparent explanations</div>', unsafe_allow_html=True)

    with st.spinner("Initializing recommendation models..."):
        models = fit_recommendation_models(model_df, catalog_df)

    active_customers = sorted(model_df[COL_CUSTOMER_ID].unique())

    # User Control Panel
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 1.2, 1.0])

    with ctrl_col1:
        # Customer selector with sample option and custom input
        st.write("<b>Customer Selection</b>", unsafe_allow_html=True)
        selected_mode = st.radio("Selector Mode", ["Pick Active Customer", "Test Custom / Cold-Start Customer ID"], horizontal=True)

        if selected_mode == "Pick Active Customer":
            chosen_customer = st.selectbox(
                "Choose Customer ID",
                options=active_customers,
                index=0,
                help="Select from customers with known purchase history.",
            )
        else:
            chosen_customer = st.text_input(
                "Enter Customer ID (Try 'NEW_GUEST' or '99999' for cold start)",
                value="NEW_GUEST_USER",
                help="Entering an unseen customer ID will test cold-start fallback handling.",
            ).strip()

    with ctrl_col2:
        st.write("<b>Algorithm Selection</b>", unsafe_allow_html=True)
        algo_choice = st.selectbox(
            "Recommendation Model",
            options=["Hybrid", "Item-Item CF", "User-User CF", "Content-Based", "Popularity"],
            index=0,
            help="Choose the algorithm to generate candidate ranking.",
        )

        top_k = st.slider("Top N Recommendations", min_value=5, max_value=20, value=10, step=5)
        exclude_prior = st.checkbox("Exclude previously purchased items", value=True)

    with ctrl_col3:
        if algo_choice == "Hybrid":
            st.write("<b>Hybrid Signal Weights</b>", unsafe_allow_html=True)
            w_cf = st.slider("Item-Item CF Weight", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
            w_cb = st.slider("Content-Based Weight", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
            w_pop = st.slider("Popularity Weight", min_value=0.0, max_value=1.0, value=0.15, step=0.05)
        else:
            st.write("<b>Model Configuration</b>", unsafe_allow_html=True)
            st.info(f"Using {algo_choice} algorithm with default parameters.")

    st.write("---")

    # Customer History Display
    customer_history_df = model_df[model_df[COL_CUSTOMER_ID] == chosen_customer]
    is_cold_start = customer_history_df.empty

    if is_cold_start:
        st.warning(
            f"⚠️ **Cold-Start Customer Detected**: Customer ID `{chosen_customer}` has no recorded historical purchases. "
            "NextIQ will automatically engage the **Popularity Fallback Engine** to provide relevant catalog recommendations."
        )
    else:
        with st.expander(f"🛒 Customer `{chosen_customer}` Purchase History ({len(customer_history_df)} transactions)", expanded=False):
            hist_view = customer_history_df[[COL_STOCK_CODE, COL_DESCRIPTION, COL_QUANTITY, COL_UNIT_PRICE, COL_REVENUE, COL_INVOICE_DATE]]
            st.dataframe(hist_view.sort_values(COL_INVOICE_DATE, ascending=False), use_container_width=True, hide_index=True)

    # Generate Recommendations
    with st.spinner("Generating recommendations..."):
        if algo_choice == "Hybrid":
            # Instantiate custom-weighted hybrid if changed
            active_model = HybridRecommender(cf_weight=w_cf, content_weight=w_cb, pop_weight=w_pop)
            active_model.cf_recommender = models["Item-Item CF"]
            active_model.content_recommender = models["Content-Based"]
            active_model.popularity_recommender = models["Popularity"]
            active_model.catalog_map = models["Item-Item CF"].catalog_map
            active_model.customer_history = models["Popularity"].customer_history
            active_model.is_fitted = True
        else:
            active_model = models[algo_choice]

        recommendations = active_model.recommend(
            customer_id=chosen_customer,
            top_n=top_k,
            exclude_purchased=exclude_prior,
        )

    # Render Recommendations
    st.subheader(f"Top {len(recommendations)} Recommendations for Customer `{chosen_customer}`")

    if not recommendations:
        st.warning("No recommendations could be generated for the given parameters.")
    else:
        # Display recommendations in a responsive 2-column grid
        r_col1, r_col2 = st.columns(2)
        for i, item in enumerate(recommendations):
            col_target = r_col1 if (i % 2 == 0) else r_col2
            with col_target:
                st.markdown(
                    f"""
                    <div class="rec-card">
                        <div class="rec-rank">Rank #{item.rank} &nbsp;•&nbsp; <span class="rec-badge">{item.strategy}</span></div>
                        <div class="rec-title">{item.description}</div>
                        <div class="rec-meta">StockCode: <b>{item.stock_code}</b> &nbsp;|&nbsp; Predicted Score: <b>{item.score:.4f}</b></div>
                        <div class="rec-explanation">💡 <b>Why recommended:</b> {item.explanation}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ----------------------------------------------------------------------------
# PAGE 3: SIMILAR PRODUCTS DISCOVERY
# ----------------------------------------------------------------------------
elif selected_page == "🔍 Similar Products Discovery":
    st.markdown('<div class="brand-title">Similar Products Discovery</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-tagline">Item-to-Item Collaborative vs Content Similarity Comparison</div>', unsafe_allow_html=True)

    with st.spinner("Loading product similarity index..."):
        models = fit_recommendation_models(model_df, catalog_df)
        item_cf = models["Item-Item CF"]
        content_model = models["Content-Based"]

    # Catalog dropdown options with description and StockCode
    catalog_sorted = catalog_df.sort_values("total_quantity", ascending=False).reset_index(drop=True)
    product_options = [
        f"{row[COL_STOCK_CODE]} — {row[COL_DESCRIPTION]} ({CURRENCY_SYMBOL}{row['avg_unit_price']:.2f})"
        for _, row in catalog_sorted.iterrows()
    ]

    selected_product_str = st.selectbox(
        "Select Target Product to Analyze",
        options=product_options,
        index=0,
        help="Search and pick an item to inspect similar recommendations.",
    )
    target_stock_code = selected_product_str.split(" — ")[0]
    target_meta = catalog_sorted[catalog_sorted[COL_STOCK_CODE] == target_stock_code].iloc[0]

    # Target Product Summary Card
    st.markdown(
        f"""
        <div class="kpi-card" style="margin-bottom: 1.5rem;">
            <div class="kpi-label">Selected Target Product</div>
            <div class="kpi-value" style="font-size: 1.35rem;">{target_meta[COL_DESCRIPTION]}</div>
            <div class="kpi-subtext" style="color: #475569; margin-top: 0.5rem;">
                StockCode: <b>{target_stock_code}</b> &nbsp;|&nbsp; 
                Avg Unit Price: <b>{CURRENCY_SYMBOL}{target_meta['avg_unit_price']:.2f}</b> &nbsp;|&nbsp; 
                Total Units Sold: <b>{int(target_meta['total_quantity']):,}</b> &nbsp;|&nbsp; 
                Total Orders: <b>{int(target_meta['total_invoices']):,}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_n_sim = st.slider("Number of Similar Products to Display", min_value=3, max_value=15, value=6)

    cf_sims = item_cf.get_similar_items(target_stock_code, top_n=top_n_sim)
    cb_sims = content_model.get_similar_items(target_stock_code, top_n=top_n_sim)

    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        st.subheader("🤝 Frequently Co-Purchased Products (Item-Item CF)")
        st.caption("Based on empirical customer market-basket co-occurrence patterns.")
        if not cf_sims:
            st.info("No co-purchase neighbors found with sufficient interaction history.")
        else:
            for rank, (code, desc, score) in enumerate(cf_sims, start=1):
                st.markdown(
                    f"""
                    <div class="rec-card" style="border-left-color: #4F46E5;">
                        <div class="rec-rank">Rank #{rank} &nbsp;•&nbsp; Co-Purchase Similarity: <b>{score:.4f}</b></div>
                        <div class="rec-title">{desc}</div>
                        <div class="rec-meta">StockCode: <b>{code}</b></div>
                        <div class="rec-explanation">Observed in the same shopping baskets as '{target_meta[COL_DESCRIPTION][:25]}...'.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with sim_col2:
        st.subheader("📝 Textually Similar Products (Content-Based TF-IDF)")
        st.caption("Based on lexical and attribute overlap in product descriptions.")
        if not cb_sims:
            st.info("No textually similar items found in the description vocabulary.")
        else:
            for rank, (code, desc, score) in enumerate(cb_sims, start=1):
                st.markdown(
                    f"""
                    <div class="rec-card" style="border-left-color: #06B6D4;">
                        <div class="rec-rank" style="color: #06B6D4;">Rank #{rank} &nbsp;•&nbsp; Text Similarity: <b>{score:.4f}</b></div>
                        <div class="rec-title">{desc}</div>
                        <div class="rec-meta">StockCode: <b>{code}</b></div>
                        <div class="rec-explanation">Shares key textual description attributes and product keywords.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.write("---")
    st.info(
        "💡 **Key Insight:** Notice how Collaborative Filtering surfaces complementary items that customers buy together "
        "(e.g., teacups and matching saucers), while Content-Based Filtering surfaces substitutable items with similar names "
        "(e.g., other colors or variations of the same product). Combining both produces optimal hybrid recommendations!"
    )


# ----------------------------------------------------------------------------
# PAGE 4: POPULAR PRODUCTS RANKING
# ----------------------------------------------------------------------------
elif selected_page == "🌟 Popular Products Ranking":
    st.markdown('<div class="brand-title">Popular Products Ranking</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-tagline">Catalog demand baselines across distinct commercial dimensions</div>', unsafe_allow_html=True)

    ranking_strategy = st.radio(
        "Ranking Dimension",
        [
            "Distinct Customer Count (Broadest Appeal)",
            "Units Purchased (Total Volume)",
            "Distinct Invoices (Order Frequency)",
            "Commercial Revenue (Gross Value)",
        ],
        horizontal=True,
    )

    by_map = {
        "Distinct Customer Count (Broadest Appeal)": ("customers", "Distinct Customers"),
        "Units Purchased (Total Volume)": ("quantity", "Units Purchased"),
        "Distinct Invoices (Order Frequency)": ("invoices", "Distinct Invoices"),
        "Commercial Revenue (Gross Value)": ("revenue", f"Revenue ({CURRENCY_SYMBOL})"),
    }
    by_key, by_label = by_map[ranking_strategy]

    top_n_pop = st.slider("Top N Products to Display", min_value=5, max_value=30, value=10, step=5)

    fig_pop = plot_top_products(sales_df, by=by_key, top_n=top_n_pop)
    st.plotly_chart(fig_pop, use_container_width=True)

    # Catalog Table with filter
    st.subheader("Searchable Catalog Volume Table")
    search_query = st.text_input("Filter Catalog by Keyword (e.g. 'HEART', 'BAG', 'TEA')", value="").strip().upper()

    filtered_catalog = catalog_df.copy()
    if search_query:
        filtered_catalog = filtered_catalog[
            filtered_catalog[COL_DESCRIPTION].str.contains(search_query, na=False)
            | filtered_catalog[COL_STOCK_CODE].str.contains(search_query, na=False)
        ]

    st.dataframe(
        filtered_catalog.sort_values("total_revenue", ascending=False).head(50),
        use_container_width=True,
        hide_index=True,
    )


# ----------------------------------------------------------------------------
# PAGE 5: OFFLINE MODEL EVALUATION
# ----------------------------------------------------------------------------
elif selected_page == "🧪 Offline Model Evaluation":
    st.markdown('<div class="brand-title">Offline Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-tagline">Rigorous Chronological Holdout Benchmark Across 5 Recommendation Models</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="kpi-card" style="margin-bottom: 1.25rem;">
            <div class="kpi-label">Evaluation Protocol Summary</div>
            <div style="font-size: 0.95rem; color: #334155; line-height: 1.5;">
                • <b>Zero Temporal Leakage:</b> Strictly splits historical transactions chronologically. 
                All training events strictly precede the held-out test evaluation window.<br>
                • <b>Discovery Task:</b> Previously purchased items are strictly excluded from recommendations, measuring the system's ability to discover <i>new</i> relevant items.<br>
                • <b>Fair Comparison:</b> Popularity, Item-Item CF, User-User CF, Content-Based, and Hybrid models are evaluated on the identical test cohort.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Computing offline benchmark metrics across test interactions..."):
        benchmark_df, benchmark_dict = get_cached_benchmark_results(model_df, catalog_df)

    # Highlight Best Performers
    best_recall_10_model = benchmark_df.loc[benchmark_df["Recall@10"].idxmax(), "Algorithm"]
    best_ndcg_model = benchmark_df.loc[benchmark_df["NDCG@10"].idxmax(), "Algorithm"]
    best_coverage_model = benchmark_df.loc[benchmark_df["Catalog Coverage"].idxmax(), "Algorithm"]

    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Best Recall@10</div>
                <div class="kpi-value" style="font-size: 1.4rem; color: #4F46E5;">{best_recall_10_model}</div>
                <div class="kpi-subtext">Highest proportion of test items captured</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Best NDCG@10</div>
                <div class="kpi-value" style="font-size: 1.4rem; color: #10B981;">{best_ndcg_model}</div>
                <div class="kpi-subtext">Highest rank-discounted relevance</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Highest Catalog Coverage</div>
                <div class="kpi-value" style="font-size: 1.4rem; color: #06B6D4;">{best_coverage_model}</div>
                <div class="kpi-subtext">Widest catalog diversity recommended</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("<br>", unsafe_allow_html=True)
    st.subheader("Offline Benchmark Results Comparison Table")
    st.dataframe(
        benchmark_df.style.highlight_max(
            subset=["Recall@5", "Recall@10", "Precision@5", "Precision@10", "NDCG@10", "Catalog Coverage", "Diversity"],
            color="#E0E7FF",
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Interactive Bar Charts for Metrics
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_recall = px.bar(
            benchmark_df,
            x="Algorithm",
            y=["Recall@5", "Recall@10"],
            barmode="group",
            title="<b>Recall Comparison (@5 and @10)</b>",
            color_discrete_sequence=["#4F46E5", "#06B6D4"],
            template="plotly_white",
        )
        fig_recall.update_layout(margin=dict(l=40, r=40, t=60, b=40))
        st.plotly_chart(fig_recall, use_container_width=True)

    with chart_col2:
        fig_ndcg = px.bar(
            benchmark_df,
            x="Algorithm",
            y="NDCG@10",
            title="<b>Ranking Quality (NDCG@10)</b>",
            color="NDCG@10",
            color_continuous_scale="Purples",
            template="plotly_white",
        )
        fig_ndcg.update_layout(coloraxis_showscale=False, margin=dict(l=40, r=40, t=60, b=40))
        st.plotly_chart(fig_ndcg, use_container_width=True)

    # Coverage vs Diversity chart
    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        fig_cov = px.bar(
            benchmark_df,
            x="Algorithm",
            y="Catalog Coverage",
            title="<b>Catalog Coverage (%)</b>",
            color_discrete_sequence=["#10B981"],
            template="plotly_white",
        )
        fig_cov.update_layout(margin=dict(l=40, r=40, t=60, b=40))
        st.plotly_chart(fig_cov, use_container_width=True)

    with chart_col4:
        fig_div = px.bar(
            benchmark_df,
            x="Algorithm",
            y="Diversity",
            title="<b>Recommendation Diversity (Intra-List Dissimilarity)</b>",
            color_discrete_sequence=["#F59E0B"],
            template="plotly_white",
        )
        fig_div.update_layout(margin=dict(l=40, r=40, t=60, b=40))
        st.plotly_chart(fig_div, use_container_width=True)

    st.write("---")
    st.markdown(
        """
        > [!IMPORTANT]
        > **Methodological Disclaimer & Real-World Limitations**:  
        > Offline metrics (Recall@K, NDCG@K) measure predictive accuracy on historical held-out transactions. 
        > They do **not** directly prove customer satisfaction, click-through rates, or revenue uplift. 
        > In real e-commerce systems, offline evaluation is a filter to prune poor candidates before deploying online A/B tests.
        """
    )


# ----------------------------------------------------------------------------
# PAGE 6: COMMERCIAL BUSINESS INSIGHTS
# ----------------------------------------------------------------------------
elif selected_page == "📈 Commercial Business Insights":
    st.markdown('<div class="brand-title">Commercial Business Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-tagline">Advanced Customer Segmentation & Market Basket Analytics</div>', unsafe_allow_html=True)

    # Date and Country Filters
    with st.expander("🔍 Interactive Data Filters", expanded=True):
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            countries = ["All Countries"] + sorted(sales_df[COL_COUNTRY].dropna().unique().tolist())
            selected_country = st.selectbox("Filter by Country", countries, index=0)
        with f_col2:
            min_date = sales_df[COL_INVOICE_DATE].min().date()
            max_date = sales_df[COL_INVOICE_DATE].max().date()
            date_range = st.date_input("Filter Date Range", value=(min_date, max_date))

    # Apply Filters
    filtered_df = sales_df.copy()
    if selected_country != "All Countries":
        filtered_df = filtered_df[filtered_df[COL_COUNTRY] == selected_country]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
        filtered_df = filtered_df[(filtered_df[COL_INVOICE_DATE] >= start_dt) & (filtered_df[COL_INVOICE_DATE] <= end_dt)]

    st.markdown(f"**Filtered Dataset:** {len(filtered_df):,} transactions ({CURRENCY_SYMBOL}{filtered_df[COL_REVENUE].sum():,.2f} revenue)")

    # Geographic Breakdown & Pareto Curve
    b_col1, b_col2 = st.columns(2)

    with b_col1:
        fig_country = plot_country_sales_distribution(filtered_df, top_n=10)
        st.plotly_chart(fig_country, use_container_width=True)

    with b_col2:
        fig_pareto, pareto_pct, total_prods = get_pareto_analysis(filtered_df)
        st.plotly_chart(fig_pareto, use_container_width=True)

    # RFM Customer Value Segmentation
    st.subheader("Customer Value Segmentation (RFM Analysis)")
    rfm_df, rfm_summary = compute_rfm_segments(filtered_df)

    if not rfm_summary.empty:
        rfm_col1, rfm_col2 = st.columns([1.1, 0.9])
        with rfm_col1:
            fig_rfm = plot_rfm_segments(rfm_summary)
            st.plotly_chart(fig_rfm, use_container_width=True)
        with rfm_col2:
            st.dataframe(rfm_summary, use_container_width=True, hide_index=True)

    # Heatmap of Shopping Hours
    st.subheader("Shopping Intensity Heatmap")
    fig_heatmap = plot_weekday_hour_activity(filtered_df)
    st.plotly_chart(fig_heatmap, use_container_width=True)


# ----------------------------------------------------------------------------
# PAGE 7: ABOUT NEXTIQ & ARCHITECTURE
# ----------------------------------------------------------------------------
elif selected_page == "ℹ️ Architecture & Documentation":
    st.markdown('<div class="brand-title">About NextIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-tagline">Intelligent Personalized Recommendation Engine — Architecture & Documentation</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏗️ System Architecture",
        "📐 Algorithms & Math",
        "📂 Dataset & Ethics",
        "🚀 Future Extensions",
    ])

    with tab1:
        st.markdown(
            """
            ### NextIQ Architectural Overview
            NextIQ is engineered using a modular, decoupled, local-first Python architecture:
            
            ```
            NextIQ Pipeline
            ├── 1. Ingestion & Schema Validation (src/data_loader.py)
            │      └── Checks columns, formats, encodings, null values
            ├── 2. Data Cleaning & Implicit Feedback (src/data_processing.py)
            │      └── Identifies cancellations, drops duplicates, builds sparse interaction matrix
            ├── 3. Recommendation Models (src/recommender.py, src/content_based.py, src/hybrid_recommender.py)
            │      ├── Popularity Recommender (Customers / Units / Invoices)
            │      ├── Item-Item Collaborative Filtering (Cosine similarity over item-user vectors)
            │      ├── User-User Collaborative Filtering (Nearest neighbor user aggregation)
            │      ├── Content-Based Filtering (TF-IDF vectorizer over product descriptions)
            │      └── Weighted Hybrid Engine (Min-max score fusion + Popularity fallback)
            ├── 4. Chronological Offline Evaluation (src/evaluation.py)
            │      └── Strict temporal holdout (Recall@K, Precision@K, NDCG@K, Coverage, Diversity)
            └── 5. Presentation Layer (app.py)
                   └── Streamlit interactive UI, Plotly charts, responsive KPI dashboards
            ```
            """
        )

    with tab2:
        st.markdown(
            r"""
            ### Algorithmic Formulations
            
            #### 1. Item-Item Collaborative Filtering
            Given customer-item interaction matrix $R \in \mathbb{R}^{U \times I}$, item similarity is computed as:
            $$\text{Sim}(i, j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}$$
            For a target user $u$, candidate items $i \notin H_u$ are scored as:
            $$\hat{r}_{ui} = \sum_{j \in H_u} \text{Sim}(i, j) \cdot r_{uj}$$
            
            #### 2. Content-Based TF-IDF Matching
            Textual descriptions are preprocessed into term-frequency inverse-document-frequency vectors:
            $$\text{tf-idf}(t, d, D) = \text{tf}(t, d) \times \log \left( \frac{1 + |D|}{1 + |\{d \in D : t \in d\}|} \right) + 1$$
            The customer preference profile $\mathbf{p}_u$ is constructed as a weighted centroid of their historical item vectors.
            
            #### 3. Weighted Hybrid Fusion
            Scores from distinct components are min-max normalized to $[0, 1]$ before linear combination:
            $$S_{\text{hybrid}}(u, i) = w_{\text{cf}} \cdot \tilde{S}_{\text{cf}}(u, i) + w_{\text{cb}} \cdot \tilde{S}_{\text{cb}}(u, i) + w_{\text{pop}} \cdot \tilde{S}_{\text{pop}}(i)$$
            where $w_{\text{cf}} + w_{\text{cb}} + w_{\text{pop}} = 1.0$.
            """
        )

    with tab3:
        st.markdown(
            """
            ### Primary Dataset Attribution
            - **Dataset Name**: UCI Machine Learning Repository — Online Retail Dataset
            - **Donor**: Dr. Daqing Chen, School of Engineering, London South Bank University, London, UK.
            - **URL**: [https://archive.ics.uci.edu/dataset/352/online-retail](https://archive.ics.uci.edu/dataset/352/online-retail)
            - **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
            - **Scope**: Historical transactions from a UK-based registered non-store online retail company between 01/12/2010 and 09/12/2011.
            - **Privacy**: Customer IDs are anonymized numerical codes; no personally identifiable names or payment card details exist in the dataset.
            """
        )

    with tab4:
        st.markdown(
            """
            ### Future Roadmap: Streaming Domain Extension
            NextIQ's recommendation interfaces (`BaseRecommender`, `fit`, `recommend`, `evaluate`) are modularly designed 
            to support streaming domains (Movies, TV Shows, Music):
            
            - **MovieLens Integration**: The user-item matrix abstraction seamlessly maps from `(CustomerID, StockCode, Quantity)` 
              to `(UserID, MovieID, Rating)`.
            - **Content Extension**: Movie metadata (genres, directors, plot summaries) can substitute product descriptions.
            - **Separate Deliverable**: The streaming extension will be introduced in a future release with its own dedicated dataset, 
              processing pipeline, and automated test suite without contaminating the retail model.
            """
        )
