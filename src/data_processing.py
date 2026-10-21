"""
NextIQ — Data Preprocessing and Interaction Modeling Pipeline
Cleans raw retail transaction records, filters anomalies, isolates cancellations/returns,
and constructs model-ready implicit feedback interaction matrices.
"""

from dataclasses import asdict, dataclass
import logging
from typing import Dict, Optional, Set, Tuple
import numpy as np
import pandas as pd

from config import (
    ADMIN_STOCK_CODES,
    COL_COUNTRY,
    COL_CUSTOMER_ID,
    COL_DESCRIPTION,
    COL_INVOICE_DATE,
    COL_INVOICE_NO,
    COL_QUANTITY,
    COL_REVENUE,
    COL_STOCK_CODE,
    COL_UNIT_PRICE,
    MIN_CUSTOMER_INTERACTIONS,
    MIN_ITEM_INTERACTIONS,
    MIN_PURCHASE_QUANTITY,
    MIN_UNIT_PRICE,
)

logger = logging.getLogger("NextIQ.DataProcessing")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@dataclass
class CleaningReport:
    """Detailed audit metrics capturing data reduction at each cleaning stage."""
    raw_rows: int
    duplicates_removed: int
    cancellations_identified: int
    missing_customer_rows: int
    invalid_price_or_qty_rows: int
    admin_stock_code_rows: int
    clean_sales_rows: int
    clean_model_rows: int
    unique_customers: int
    unique_items: int
    total_valid_revenue: float
    total_valid_quantity: int
    sparsity_ratio: float

    def to_dict(self) -> Dict:
        return asdict(self)


def clean_transactions(
    df: pd.DataFrame,
    admin_codes: Optional[Set[str]] = None,
    drop_missing_customers_for_model: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    """
    Executes the multi-stage cleaning pipeline on raw retail transaction data.
    
    Args:
        df: Raw DataFrame matching the UCI schema.
        admin_codes: Optional set of non-merchandise stock codes to exclude.
        drop_missing_customers_for_model: Whether to require valid CustomerID in model-ready data.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
            - clean_sales_df: Valid commercial transactions (including guest checkouts for macro sales analysis).
            - model_ready_df: Valid transactions with customer IDs for personalized recommendation algorithms.
            - report: Comprehensive audit metrics.
    """
    if df is None or df.empty:
        raise ValueError("Cannot process empty or None DataFrame.")

    raw_rows = len(df)
    logger.info(f"Initiating data cleaning pipeline for {raw_rows:,} records.")

    if admin_codes is None:
        admin_codes = ADMIN_STOCK_CODES

    # Step 1: Standardize datatypes and identifiers
    cleaned = df.copy()
    cleaned[COL_INVOICE_NO] = cleaned[COL_INVOICE_NO].astype(str).str.strip()
    cleaned[COL_STOCK_CODE] = cleaned[COL_STOCK_CODE].astype(str).str.strip().str.upper()
    cleaned[COL_DESCRIPTION] = cleaned[COL_DESCRIPTION].fillna("UNKNOWN PRODUCT").astype(str).str.strip().str.upper()
    cleaned[COL_INVOICE_DATE] = pd.to_datetime(cleaned[COL_INVOICE_DATE], errors="coerce")
    cleaned[COL_QUANTITY] = pd.to_numeric(cleaned[COL_QUANTITY], errors="coerce").fillna(0).astype(int)
    cleaned[COL_UNIT_PRICE] = pd.to_numeric(cleaned[COL_UNIT_PRICE], errors="coerce").fillna(0.0)

    # Standardize CustomerID: strip trailing .0 if ingested as float string, preserve empty as NaN
    cleaned[COL_CUSTOMER_ID] = (
        cleaned[COL_CUSTOMER_ID]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    cleaned.loc[cleaned[COL_CUSTOMER_ID].isin(["nan", "None", "", "NULL"]), COL_CUSTOMER_ID] = np.nan

    # Step 2: Remove exact duplicate records
    pre_dup_count = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    duplicates_removed = pre_dup_count - len(cleaned)
    logger.info(f"Removed {duplicates_removed:,} exact duplicate rows.")

    # Step 3: Identify cancellations and returns
    # UCI documentation: Invoice numbers starting with 'C' denote cancellations.
    # Quantity < 0 also indicates returned merchandise.
    is_cancellation = (
        cleaned[COL_INVOICE_NO].str.startswith("C", na=False)
        | cleaned[COL_INVOICE_NO].str.startswith("c", na=False)
        | (cleaned[COL_QUANTITY] <= 0)
    )
    cancellations_count = int(is_cancellation.sum())
    logger.info(f"Identified {cancellations_count:,} cancellation and return transactions.")

    # Filter out cancellations for positive purchase modeling
    positive_transactions = cleaned[~is_cancellation].copy()

    # Step 4: Identify invalid prices and non-positive quantities
    valid_price_mask = positive_transactions[COL_UNIT_PRICE] >= MIN_UNIT_PRICE
    valid_qty_mask = positive_transactions[COL_QUANTITY] >= MIN_PURCHASE_QUANTITY
    invalid_price_or_qty = ~(valid_price_mask & valid_qty_mask)
    invalid_price_or_qty_count = int(invalid_price_or_qty.sum())
    valid_commercial = positive_transactions[valid_price_mask & valid_qty_mask].copy()

    # Step 5: Filter administrative non-product codes (Postage, Bank charges, manual fees)
    is_admin = valid_commercial[COL_STOCK_CODE].isin(admin_codes)
    admin_count = int(is_admin.sum())
    valid_commercial = valid_commercial[~is_admin].copy()

    # Step 6: Compute commercial line-item revenue
    valid_commercial[COL_REVENUE] = valid_commercial[COL_QUANTITY] * valid_commercial[COL_UNIT_PRICE]

    clean_sales_rows = len(valid_commercial)
    logger.info(f"Clean commercial sales dataset contains {clean_sales_rows:,} records.")

    # Step 7: Filter for model-ready collaborative interactions
    missing_customer_mask = valid_commercial[COL_CUSTOMER_ID].isna()
    missing_customer_count = int(missing_customer_mask.sum())

    if drop_missing_customers_for_model:
        model_ready_df = valid_commercial[~missing_customer_mask].copy()
    else:
        model_ready_df = valid_commercial.copy()

    clean_model_rows = len(model_ready_df)
    unique_cust = int(model_ready_df[COL_CUSTOMER_ID].nunique())
    unique_itm = int(model_ready_df[COL_STOCK_CODE].nunique())

    # Calculate matrix sparsity
    if unique_cust > 0 and unique_itm > 0:
        total_possible = unique_cust * unique_itm
        observed_pairs = len(model_ready_df.groupby([COL_CUSTOMER_ID, COL_STOCK_CODE]))
        sparsity_ratio = 1.0 - (observed_pairs / total_possible)
    else:
        sparsity_ratio = 0.0

    total_valid_revenue = float(round(valid_commercial[COL_REVENUE].sum(), 2))
    total_valid_quantity = int(valid_commercial[COL_QUANTITY].sum())

    report = CleaningReport(
        raw_rows=raw_rows,
        duplicates_removed=duplicates_removed,
        cancellations_identified=cancellations_count,
        missing_customer_rows=missing_customer_count,
        invalid_price_or_qty_rows=invalid_price_or_qty_count,
        admin_stock_code_rows=admin_count,
        clean_sales_rows=clean_sales_rows,
        clean_model_rows=clean_model_rows,
        unique_customers=unique_cust,
        unique_items=unique_itm,
        total_valid_revenue=total_valid_revenue,
        total_valid_quantity=total_valid_quantity,
        sparsity_ratio=round(sparsity_ratio, 6),
    )

    logger.info(
        f"Cleaning Complete: {raw_rows:,} raw -> {clean_sales_rows:,} sales rows -> "
        f"{clean_model_rows:,} model rows ({unique_cust:,} customers, {unique_itm:,} items, "
        f"sparsity: {sparsity_ratio * 100:.2f}%)."
    )

    return valid_commercial, model_ready_df, report


def build_interaction_matrix(
    df: pd.DataFrame,
    min_customer_interactions: int = MIN_CUSTOMER_INTERACTIONS,
    min_item_interactions: int = MIN_ITEM_INTERACTIONS,
    value_col: str = "log_quantity",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregates repeated transactions into customer-product interaction records.
    Computes implicit feedback weights (total quantity, order frequency, log quantity, binary purchase).
    
    Args:
        df: Cleaned model-ready transactions with CustomerID and StockCode.
        min_customer_interactions: Minimum interactions required for a customer.
        min_item_interactions: Minimum interactions required for an item.
        value_col: Column to use for matrix weights ('log_quantity', 'total_quantity', 'frequency', 'binary').
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - aggregated_interactions_df: Table of (CustomerID, StockCode, metrics).
            - user_item_pivot: Customer-item pivot table (Customer x Item).
    """
    if df.empty:
        raise ValueError("Cannot build interaction matrix from an empty DataFrame.")

    # Aggregate by customer and product
    agg = (
        df.groupby([COL_CUSTOMER_ID, COL_STOCK_CODE])
        .agg(
            total_quantity=(COL_QUANTITY, "sum"),
            invoice_frequency=(COL_INVOICE_NO, "nunique"),
            total_spend=(COL_REVENUE, "sum"),
            first_purchase=(COL_INVOICE_DATE, "min"),
            last_purchase=(COL_INVOICE_DATE, "max"),
        )
        .reset_index()
    )

    # Compute implicit feedback representations
    agg["binary"] = 1.0
    agg["log_quantity"] = np.log1p(agg["total_quantity"])
    agg["frequency"] = agg["invoice_frequency"].astype(float)

    # Filter infrequent customers and items to reduce noise if requested
    if min_customer_interactions > 1:
        cust_counts = agg.groupby(COL_CUSTOMER_ID)[COL_STOCK_CODE].count()
        valid_custs = cust_counts[cust_counts >= min_customer_interactions].index
        agg = agg[agg[COL_CUSTOMER_ID].isin(valid_custs)].copy()

    if min_item_interactions > 1:
        item_counts = agg.groupby(COL_STOCK_CODE)[COL_CUSTOMER_ID].count()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        agg = agg[agg[COL_STOCK_CODE].isin(valid_items)].copy()

    # Create pivot representation
    pivot = agg.pivot_table(
        index=COL_CUSTOMER_ID,
        columns=COL_STOCK_CODE,
        values=value_col,
        fill_value=0.0,
    )

    logger.info(
        f"Built interaction matrix: {pivot.shape[0]:,} customers x {pivot.shape[1]:,} items "
        f"(using '{value_col}' weighting)."
    )

    return agg, pivot


def get_item_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a canonical product catalog mapping StockCode to its most descriptive name,
    average unit price, and primary category metadata.
    """
    # Exclude administrative codes
    items = df[~df[COL_STOCK_CODE].isin(ADMIN_STOCK_CODES)].copy()

    # Pick the most frequent non-empty description for each StockCode
    valid_desc = items[items[COL_DESCRIPTION] != "UNKNOWN PRODUCT"]
    if not valid_desc.empty:
        desc_counts = valid_desc.groupby([COL_STOCK_CODE, COL_DESCRIPTION]).size().reset_index(name="count")
        best_desc = desc_counts.sort_values([COL_STOCK_CODE, "count"], ascending=[True, False]).drop_duplicates(
            subset=[COL_STOCK_CODE]
        )[[COL_STOCK_CODE, COL_DESCRIPTION]]
    else:
        best_desc = pd.DataFrame(columns=[COL_STOCK_CODE, COL_DESCRIPTION])

    # Compute pricing and volume summary
    summary = (
        items.groupby(COL_STOCK_CODE)
        .agg(
            avg_unit_price=(COL_UNIT_PRICE, "mean"),
            total_quantity=(COL_QUANTITY, "sum"),
            total_revenue=(COL_REVENUE, "sum") if COL_REVENUE in items.columns else (COL_QUANTITY, "count"),
            distinct_customers=(COL_CUSTOMER_ID, "nunique"),
            total_invoices=(COL_INVOICE_NO, "nunique"),
        )
        .reset_index()
    )

    catalog = summary.merge(best_desc, on=COL_STOCK_CODE, how="left")
    catalog[COL_DESCRIPTION] = catalog[COL_DESCRIPTION].fillna("UNKNOWN PRODUCT")
    catalog["avg_unit_price"] = catalog["avg_unit_price"].round(2)
    catalog["total_revenue"] = catalog["total_revenue"].round(2)

    return catalog
