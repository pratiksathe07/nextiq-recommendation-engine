"""
Unit Tests for NextIQ Data Preprocessing Module
Tests deduplication, cancellation filtering, price/quantity checks,
administrative code filtering, and implicit-feedback matrix construction.
"""

import pandas as pd
import numpy as np
import pytest

from config import (
    COL_COUNTRY,
    COL_CUSTOMER_ID,
    COL_DESCRIPTION,
    COL_INVOICE_DATE,
    COL_INVOICE_NO,
    COL_QUANTITY,
    COL_REVENUE,
    COL_STOCK_CODE,
    COL_UNIT_PRICE,
)
from src.data_processing import (
    CleaningReport,
    build_interaction_matrix,
    clean_transactions,
    get_item_catalog,
)


@pytest.fixture
def raw_test_df() -> pd.DataFrame:
    """Constructs a representative DataFrame with normal sales, cancellations, duplicates, and edge cases."""
    return pd.DataFrame({
        COL_INVOICE_NO: [
            "536365",       # Valid 1
            "536365",       # Valid 2
            "536365",       # Duplicate of Valid 2
            "C536366",      # Cancellation (Invoice prefix C)
            "536367",       # Negative quantity (Return)
            "536368",       # Zero price (Damaged / Administrative)
            "536369",       # Postage admin code
            "536370",       # Missing customer ID (Guest checkout)
            "536371",       # Valid customer 2
            "536372",       # Valid customer 1 repeat purchase
        ],
        COL_STOCK_CODE: [
            "85123A",
            "71053",
            "71053",        # Duplicate
            "85123A",
            "84406B",
            "22752",
            "POST",         # Admin code
            "22633",
            "22632",
            "85123A",
        ],
        COL_DESCRIPTION: [
            "WHITE HANGING HEART T-LIGHT HOLDER",
            "WHITE METAL LANTERN",
            "WHITE METAL LANTERN",
            "WHITE HANGING HEART T-LIGHT HOLDER",
            "CREAM CUPID HEARTS COAT HANGER",
            "SET 7 BABUSHKA NESTING BOXES",
            "POSTAGE",
            "HAND WARMER UNION JACK",
            "HAND WARMER RED POLKA DOT",
            "WHITE HANGING HEART T-LIGHT HOLDER",
        ],
        COL_QUANTITY: [6, 12, 12, -6, -2, 1, 1, 4, 10, 3],
        COL_INVOICE_DATE: [
            "2010-12-01 08:26:00",
            "2010-12-01 08:26:00",
            "2010-12-01 08:26:00",
            "2010-12-01 08:28:00",
            "2010-12-01 08:34:00",
            "2010-12-01 09:00:00",
            "2010-12-01 09:15:00",
            "2010-12-01 09:30:00",
            "2010-12-01 09:45:00",
            "2010-12-01 10:00:00",
        ],
        COL_UNIT_PRICE: [2.55, 3.39, 3.39, 2.55, 2.75, 0.0, 18.00, 1.85, 1.85, 2.55],
        COL_CUSTOMER_ID: [
            "17850",
            "17850",
            "17850",
            "17850",
            "13047",
            "13047",
            "13047",
            None,           # Guest checkout
            "12583",
            "17850",
        ],
        COL_COUNTRY: ["United Kingdom"] * 10,
    })


def test_clean_transactions_deduplication(raw_test_df):
    """Verifies that exact duplicates are properly dropped and audited."""
    sales_df, model_df, report = clean_transactions(raw_test_df)
    assert report.duplicates_removed == 1
    assert report.raw_rows == 10


def test_clean_transactions_cancellations(raw_test_df):
    """Verifies that both 'C' prefix invoices and negative quantities are excluded from positive purchases."""
    sales_df, model_df, report = clean_transactions(raw_test_df)
    # C536366 and 536367 (Quantity -2) should both be identified as cancellations
    assert report.cancellations_identified == 2
    assert (sales_df[COL_QUANTITY] > 0).all()
    assert (~sales_df[COL_INVOICE_NO].str.startswith("C")).all()


def test_clean_transactions_zero_price_and_admin(raw_test_df):
    """Verifies that zero unit prices and administrative codes are filtered."""
    sales_df, model_df, report = clean_transactions(raw_test_df)
    assert (sales_df[COL_UNIT_PRICE] > 0).all()
    assert "POST" not in sales_df[COL_STOCK_CODE].values
    assert report.admin_stock_code_rows >= 1


def test_clean_transactions_missing_customers(raw_test_df):
    """Verifies that missing CustomerID is kept in sales_df but filtered in model_df."""
    sales_df, model_df, report = clean_transactions(raw_test_df)
    # Sales DF should include guest checkout
    assert sales_df[COL_CUSTOMER_ID].isna().any()
    # Model DF must NOT contain any NaN customer ID
    assert model_df[COL_CUSTOMER_ID].notna().all()


def test_clean_transactions_revenue_calculation(raw_test_df):
    """Verifies that line-item revenue is accurately computed as Quantity * UnitPrice."""
    sales_df, model_df, report = clean_transactions(raw_test_df)
    expected_rev = sales_df[COL_QUANTITY] * sales_df[COL_UNIT_PRICE]
    assert np.allclose(sales_df[COL_REVENUE], expected_rev)


def test_build_interaction_matrix(raw_test_df):
    """Verifies implicit-feedback interaction matrix aggregation."""
    _, model_df, _ = clean_transactions(raw_test_df)
    agg, pivot = build_interaction_matrix(
        model_df,
        min_customer_interactions=1,
        min_item_interactions=1,
        value_col="log_quantity",
    )
    assert not agg.empty
    assert "total_quantity" in agg.columns
    assert "invoice_frequency" in agg.columns
    assert "log_quantity" in agg.columns
    assert "binary" in agg.columns

    # Verify Customer 17850 has StockCode 85123A with repeated quantity (6 + 3 = 9)
    cust_17850_item = agg[(agg[COL_CUSTOMER_ID] == "17850") & (agg[COL_STOCK_CODE] == "85123A")]
    assert len(cust_17850_item) == 1
    assert cust_17850_item["total_quantity"].iloc[0] == 9
    assert cust_17850_item["invoice_frequency"].iloc[0] == 2
    assert cust_17850_item["log_quantity"].iloc[0] == pytest.approx(np.log1p(9), rel=1e-3)


def test_get_item_catalog(raw_test_df):
    """Verifies that item catalog correctly aggregates pricing and descriptions."""
    sales_df, _, _ = clean_transactions(raw_test_df)
    catalog = get_item_catalog(sales_df)
    assert not catalog.empty
    assert COL_STOCK_CODE in catalog.columns
    assert COL_DESCRIPTION in catalog.columns
    assert "avg_unit_price" in catalog.columns
    # Check that POST is excluded from product catalog
    assert "POST" not in catalog[COL_STOCK_CODE].values
