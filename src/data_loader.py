"""
NextIQ — Data Loader Module
Handles ingestion, path resolution, and schema validation for UCI Online Retail transactions.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from config import (
    COL_CUSTOMER_ID,
    COL_DESCRIPTION,
    COL_INVOICE_DATE,
    COL_INVOICE_NO,
    COL_QUANTITY,
    COL_STOCK_CODE,
    COL_UNIT_PRICE,
    RAW_CSV_PATH,
    RAW_DATA_PATH,
    REQUIRED_COLUMNS,
    SAMPLE_DATA_PATH,
)

logger = logging.getLogger("NextIQ.DataLoader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class DataLoadingError(Exception):
    """Custom exception raised when data ingestion or schema validation fails."""
    pass


def resolve_data_path(custom_path: Optional[str | Path] = None) -> Path:
    """
    Resolves the dataset file path checking user overrides, raw CSV, raw Excel, and sample fallback.
    If an explicit custom_path is provided, it must exist.
    """
    if custom_path:
        p = Path(custom_path)
        if p.is_file():
            logger.info(f"Resolved custom dataset location: {p.resolve()}")
            return p.resolve()
        else:
            msg = (
                f"Specified dataset path '{custom_path}' does not exist!\n"
                "Please verify the file path or download the UCI Online Retail dataset:\n"
                "Source: https://archive.ics.uci.edu/dataset/352/online-retail"
            )
            logger.error(msg)
            raise FileNotFoundError(msg)

    candidates = [RAW_CSV_PATH, RAW_DATA_PATH, SAMPLE_DATA_PATH]
    for path in candidates:
        if path.is_file():
            logger.info(f"Resolved default dataset location: {path.resolve()}")
            return path.resolve()

    # If none found, provide an informative error
    searched_paths = "\n".join([f" - {p}" for p in candidates])
    msg = (
        "No retail dataset found! Checked default paths:\n"
        f"{searched_paths}\n\n"
        "Please download the UCI Online Retail dataset or run data/download_data.py to download/generate data:\n"
        "Source: https://archive.ics.uci.edu/dataset/352/online-retail\n"
        f"Save it as '{RAW_CSV_PATH}' or '{RAW_DATA_PATH}'."
    )
    logger.error(msg)
    raise FileNotFoundError(msg)


def load_raw_data(file_path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Loads raw transaction data from CSV or Excel file.
    
    Args:
        file_path: Optional path to data file. If None, resolves automatically.
        
    Returns:
        pd.DataFrame: Raw ingested transaction data.
        
    Raises:
        FileNotFoundError: If no data file can be located.
        DataLoadingError: If the file is unreadable or corrupted.
    """
    resolved_path = resolve_data_path(file_path)
    suffix = resolved_path.suffix.lower()

    logger.info(f"Loading raw data from {resolved_path} (Format: {suffix})...")

    try:
        if suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(
                resolved_path,
                dtype={
                    COL_INVOICE_NO: str,
                    COL_STOCK_CODE: str,
                    COL_CUSTOMER_ID: str,
                },
                engine="openpyxl" if suffix == ".xlsx" else None,
            )
        elif suffix == ".csv":
            # Attempt utf-8, fallback to latin1 / iso-8859-1 for international characters
            try:
                df = pd.read_csv(
                    resolved_path,
                    dtype={
                        COL_INVOICE_NO: str,
                        COL_STOCK_CODE: str,
                        COL_CUSTOMER_ID: str,
                    },
                    encoding="utf-8",
                    low_memory=False,
                )
            except UnicodeDecodeError:
                logger.warning("UTF-8 decoding failed. Retrying with ISO-8859-1 encoding.")
                df = pd.read_csv(
                    resolved_path,
                    dtype={
                        COL_INVOICE_NO: str,
                        COL_STOCK_CODE: str,
                        COL_CUSTOMER_ID: str,
                    },
                    encoding="ISO-8859-1",
                    low_memory=False,
                )
        else:
            raise DataLoadingError(f"Unsupported file format '{suffix}'. Supported formats: .csv, .xlsx, .xls")

    except Exception as e:
        if isinstance(e, (FileNotFoundError, DataLoadingError)):
            raise
        raise DataLoadingError(f"Failed to read file '{resolved_path}': {str(e)}") from e

    logger.info(f"Loaded {len(df):,} raw records with {df.shape[1]} columns.")
    
    # Run schema validation
    is_valid, issues = validate_schema(df)
    if not is_valid:
        issue_str = "\n".join([f" - {iss}" for iss in issues])
        raise DataLoadingError(f"Schema validation failed for '{resolved_path}':\n{issue_str}")

    return df


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates DataFrame against the expected UCI Online Retail schema.
    
    Args:
        df: DataFrame to validate.
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list of detected validation issues).
    """
    issues: List[str] = []

    if df is None:
        return False, ["DataFrame is None."]

    if df.empty:
        return False, ["DataFrame contains 0 rows (empty dataset)."]

    # Check for missing required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    # Check for non-null records in key identifiers
    if COL_INVOICE_NO in df.columns and df[COL_INVOICE_NO].dropna().empty:
        issues.append(f"Column '{COL_INVOICE_NO}' has 100% missing values.")

    if COL_STOCK_CODE in df.columns and df[COL_STOCK_CODE].dropna().empty:
        issues.append(f"Column '{COL_STOCK_CODE}' has 100% missing values.")

    is_valid = len(issues) == 0
    return is_valid, issues


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes summary metrics for loaded raw dataset.
    
    Args:
        df: Raw or loaded DataFrame.
        
    Returns:
        Dict[str, Any]: Summary metadata dictionary.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    missing_counts = df.isnull().sum().to_dict()
    missing_percentages = {col: round((cnt / total_rows) * 100, 2) for col, cnt in missing_counts.items()}
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    unique_invoices = df[COL_INVOICE_NO].nunique() if COL_INVOICE_NO in df.columns else 0
    unique_items = df[COL_STOCK_CODE].nunique() if COL_STOCK_CODE in df.columns else 0
    unique_customers = df[COL_CUSTOMER_ID].dropna().nunique() if COL_CUSTOMER_ID in df.columns else 0

    date_min = None
    date_max = None
    if COL_INVOICE_DATE in df.columns:
        dates = pd.to_datetime(df[COL_INVOICE_DATE], errors="coerce").dropna()
        if not dates.empty:
            date_min = str(dates.min())
            date_max = str(dates.max())

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "memory_mb": memory_mb,
        "unique_invoices": unique_invoices,
        "unique_items": unique_items,
        "unique_customers": unique_customers,
        "date_min": date_min,
        "date_max": date_max,
        "missing_counts": missing_counts,
        "missing_percentages": missing_percentages,
    }
