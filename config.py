"""
NextIQ — Intelligent Personalized Recommendation Engine
Centralized Configuration Module
"""

import os
from pathlib import Path
from typing import Dict, List, Set

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "online_retail.xlsx"
RAW_CSV_PATH = DATA_DIR / "online_retail.csv"
SAMPLE_DATA_PATH = DATA_DIR / "sample_online_retail.csv"
ASSETS_DIR = BASE_DIR / "assets"
REPORTS_DIR = BASE_DIR / "reports"

# Schema & Column Names (UCI Online Retail standard)
COL_INVOICE_NO = "InvoiceNo"
COL_STOCK_CODE = "StockCode"
COL_DESCRIPTION = "Description"
COL_QUANTITY = "Quantity"
COL_INVOICE_DATE = "InvoiceDate"
COL_UNIT_PRICE = "UnitPrice"
COL_CUSTOMER_ID = "CustomerID"
COL_COUNTRY = "Country"
COL_REVENUE = "Revenue"

REQUIRED_COLUMNS: List[str] = [
    COL_INVOICE_NO,
    COL_STOCK_CODE,
    COL_DESCRIPTION,
    COL_QUANTITY,
    COL_INVOICE_DATE,
    COL_UNIT_PRICE,
    COL_CUSTOMER_ID,
    COL_COUNTRY,
]

EXPECTED_DTYPES: Dict[str, str] = {
    COL_INVOICE_NO: "object",
    COL_STOCK_CODE: "object",
    COL_DESCRIPTION: "object",
    COL_QUANTITY: "int64",
    COL_INVOICE_DATE: "datetime64[ns]",
    COL_UNIT_PRICE: "float64",
    COL_CUSTOMER_ID: "object",
    COL_COUNTRY: "object",
}

# Administrative / Service Stock Codes (non-merchandise to exclude from product recommendations)
ADMIN_STOCK_CODES: Set[str] = {
    "POST",        # Postage
    "D",           # Discount
    "M",           # Manual
    "BANK CHARGES",
    "PADS",
    "DOT",         # DOTCOM POSTAGE
    "CRUK",        # CRUK Commission
    "AMAZONFEE",
    "S",           # Sample
}

# Business Cleaning Thresholds
MIN_UNIT_PRICE: float = 0.001
MIN_PURCHASE_QUANTITY: int = 1
MIN_CUSTOMER_INTERACTIONS: int = 2
MIN_ITEM_INTERACTIONS: int = 2
CURRENCY_SYMBOL: str = "₹"  # Indian Rupee (₹)

# Recommendation Model Parameters
DEFAULT_TOP_K: int = 10
CONTENT_MAX_FEATURES: int = 1500
CONTENT_NGRAM_RANGE = (1, 2)
NEIGHBORHOOD_SIZE: int = 20

# Hybrid Default Weights (Sum to 1.0)
DEFAULT_HYBRID_WEIGHTS: Dict[str, float] = {
    "item_item_cf": 0.50,
    "content_based": 0.35,
    "popularity": 0.15,
}

# Offline Evaluation Parameters
EVALUATION_SPLIT_RATIO: float = 0.20  # Chronological holdout ratio
EVALUATION_K_VALUES: List[int] = [5, 10, 20]
RANDOM_SEED: int = 42

# UI Branding & Color Palette
BRAND_PRIMARY = "#4F46E5"    # Indigo 600
BRAND_SECONDARY = "#7C3AED"  # Violet 600
BRAND_ACCENT = "#06B6D4"     # Cyan 500
BRAND_SUCCESS = "#10B981"    # Emerald 500
BRAND_WARNING = "#F59E0B"    # Amber 500
BRAND_DARK = "#0F172A"       # Slate 900
BRAND_LIGHT = "#F8FAFC"      # Slate 50
PLOTLY_TEMPLATE = "plotly_white"
