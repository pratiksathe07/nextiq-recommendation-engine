"""
Unit Tests for NextIQ Data Loader Module
Tests schema validation, path resolution, and ingestion error handling.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from config import (
    COL_COUNTRY,
    COL_CUSTOMER_ID,
    COL_DESCRIPTION,
    COL_INVOICE_DATE,
    COL_INVOICE_NO,
    COL_QUANTITY,
    COL_STOCK_CODE,
    COL_UNIT_PRICE,
    REQUIRED_COLUMNS,
    SAMPLE_DATA_PATH,
)
from src.data_loader import (
    DataLoadingError,
    get_dataset_summary,
    load_raw_data,
    resolve_data_path,
    validate_schema,
)


@pytest.fixture
def valid_mock_df() -> pd.DataFrame:
    """Provides a minimal valid synthetic retail DataFrame."""
    return pd.DataFrame({
        COL_INVOICE_NO: ["536365", "536366"],
        COL_STOCK_CODE: ["85123A", "71053"],
        COL_DESCRIPTION: ["WHITE HANGING HEART T-LIGHT HOLDER", "WHITE METAL LANTERN"],
        COL_QUANTITY: [6, 12],
        COL_INVOICE_DATE: ["2010-12-01 08:26:00", "2010-12-01 08:28:00"],
        COL_UNIT_PRICE: [2.55, 3.39],
        COL_CUSTOMER_ID: ["17850", "17850"],
        COL_COUNTRY: ["United Kingdom", "United Kingdom"],
    })


def test_validate_schema_success(valid_mock_df):
    """Verifies that a conformant DataFrame passes schema validation."""
    is_valid, issues = validate_schema(valid_mock_df)
    assert is_valid is True
    assert len(issues) == 0


def test_validate_schema_empty():
    """Verifies that an empty DataFrame fails schema validation with descriptive message."""
    empty_df = pd.DataFrame()
    is_valid, issues = validate_schema(empty_df)
    assert is_valid is False
    assert any("empty" in issue.lower() for issue in issues)


def test_validate_schema_missing_column(valid_mock_df):
    """Verifies that dropping a required column triggers validation failure."""
    incomplete_df = valid_mock_df.drop(columns=[COL_CUSTOMER_ID])
    is_valid, issues = validate_schema(incomplete_df)
    assert is_valid is False
    assert any(COL_CUSTOMER_ID in issue for issue in issues)


def test_validate_schema_none():
    """Verifies that None DataFrame fails schema validation."""
    is_valid, issues = validate_schema(None)
    assert is_valid is False
    assert any("none" in issue.lower() for issue in issues)


def test_resolve_data_path_custom_existing(tmp_path):
    """Verifies that resolve_data_path prioritizes custom valid path."""
    fake_csv = tmp_path / "custom_data.csv"
    fake_csv.write_text("InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country\n")
    resolved = resolve_data_path(fake_csv)
    assert resolved == fake_csv.resolve()


def test_resolve_data_path_nonexistent():
    """Verifies that resolve_data_path raises FileNotFoundError with download guidance when missing."""
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_data_path("non_existent_path_xyz_123.csv")
    assert "does not exist" in str(exc_info.value)
    assert "https://archive.ics.uci.edu" in str(exc_info.value)


def test_get_dataset_summary(valid_mock_df):
    """Verifies dataset summary calculation."""
    summary = get_dataset_summary(valid_mock_df)
    assert summary["total_rows"] == 2
    assert summary["total_columns"] == 8
    assert summary["unique_invoices"] == 2
    assert summary["unique_customers"] == 1
    assert summary["unique_items"] == 2
    assert "memory_mb" in summary


def test_load_raw_data_sample():
    """Verifies loading the generated verified sample dataset."""
    assert SAMPLE_DATA_PATH.exists(), "Sample dataset must exist for this test"
    df = load_raw_data(SAMPLE_DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 1000
    for col in REQUIRED_COLUMNS:
        assert col in df.columns
