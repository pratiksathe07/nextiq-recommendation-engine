"""
Unit Tests for NextIQ Offline Model Evaluation
Tests metric formulas (Recall@K, Precision@K, NDCG@K, Coverage, Diversity),
temporal splitting correctness, and prevention of future-data leakage.
"""

import math
import pandas as pd
import pytest

from config import COL_CUSTOMER_ID, COL_INVOICE_DATE, COL_STOCK_CODE
from src.evaluation import (
    catalog_coverage,
    chronological_train_test_split,
    intra_list_diversity,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_recall_at_k():
    """Verifies hand-calculated Recall@K."""
    actual = {"ITEM_1", "ITEM_2", "ITEM_3", "ITEM_4"}
    recommended = ["ITEM_1", "ITEM_5", "ITEM_2", "ITEM_6", "ITEM_7"]

    # At K=1: hit {'ITEM_1'} out of 4 -> 0.25
    assert recall_at_k(recommended, actual, k=1) == pytest.approx(0.25)
    # At K=3: hits {'ITEM_1', 'ITEM_2'} out of 4 -> 0.50
    assert recall_at_k(recommended, actual, k=3) == pytest.approx(0.50)
    # At K=5: hits {'ITEM_1', 'ITEM_2'} out of 4 -> 0.50
    assert recall_at_k(recommended, actual, k=5) == pytest.approx(0.50)
    # Empty actual -> 0.0
    assert recall_at_k(recommended, set(), k=5) == 0.0


def test_precision_at_k():
    """Verifies hand-calculated Precision@K."""
    actual = {"ITEM_1", "ITEM_2"}
    recommended = ["ITEM_1", "ITEM_5", "ITEM_2", "ITEM_6"]

    # At K=1: 1 hit in 1 rec -> 1.0
    assert precision_at_k(recommended, actual, k=1) == pytest.approx(1.0)
    # At K=2: 1 hit in 2 recs -> 0.5
    assert precision_at_k(recommended, actual, k=2) == pytest.approx(0.5)
    # At K=4: 2 hits in 4 recs -> 0.5
    assert precision_at_k(recommended, actual, k=4) == pytest.approx(0.5)
    # No hits
    assert precision_at_k(["ITEM_X", "ITEM_Y"], actual, k=2) == 0.0


def test_ndcg_at_k():
    """Verifies NDCG@K ranking discounting."""
    actual = {"A"}
    
    # Perfect top ranking: hit at rank 1
    # DCG = 1 / log2(2) = 1.0; IDCG = 1 / log2(2) = 1.0 -> NDCG = 1.0
    assert ndcg_at_k(["A", "B", "C"], actual, k=3) == pytest.approx(1.0)

    # Sub-optimal ranking: hit at rank 2
    # DCG = 1 / log2(3) = ~0.6309; IDCG = 1.0 -> NDCG = ~0.6309
    expected_ndcg = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert ndcg_at_k(["B", "A", "C"], actual, k=3) == pytest.approx(expected_ndcg, rel=1e-3)

    # Hit at rank 3
    expected_ndcg_3 = (1.0 / math.log2(4)) / (1.0 / math.log2(2))
    assert ndcg_at_k(["B", "C", "A"], actual, k=3) == pytest.approx(expected_ndcg_3, rel=1e-3)

    # Zero hits
    assert ndcg_at_k(["X", "Y", "Z"], actual, k=3) == 0.0


def test_catalog_coverage():
    """Verifies catalog coverage computation."""
    catalog = {"A", "B", "C", "D", "E"}
    recs = [
        ["A", "B"],
        ["B", "C"],
    ]
    # Unique recommended: A, B, C (3 items out of 5) -> 60%
    assert catalog_coverage(recs, catalog) == pytest.approx(0.60)
    # Empty recommendations -> 0.0
    assert catalog_coverage([], catalog) == 0.0


def test_intra_list_diversity():
    """Verifies intra-list diversity distance calculation."""
    sim_lookup = {
        ("A", "B"): 0.8,
        ("A", "C"): 0.1,
        ("B", "C"): 0.1,
    }
    # Pairwise distances:
    # dist(A, B) = 1 - 0.8 = 0.2
    # dist(A, C) = 1 - 0.1 = 0.9
    # dist(B, C) = 1 - 0.1 = 0.9
    # mean distance = (0.2 + 0.9 + 0.9) / 3 = 2.0 / 3 = 0.6667
    div = intra_list_diversity(["A", "B", "C"], sim_lookup)
    assert div == pytest.approx(2.0 / 3.0, rel=1e-3)


def test_chronological_train_test_split_leakage_prevention():
    """
    CRITICAL TEST: Verifies that future purchases NEVER leak into training interactions.
    All train InvoiceDate <= cutoff < all test InvoiceDate.
    """
    dates = pd.date_range("2011-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        COL_INVOICE_DATE: dates,
        COL_CUSTOMER_ID: ["U1", "U1", "U2", "U2", "U1", "U2", "U3", "U1", "U2", "U3"],
        COL_STOCK_CODE: [f"ITEM_{i}" for i in range(10)],
    })

    train_df, test_df, eligible_users = chronological_train_test_split(df, test_ratio=0.30)

    # Assert no future date in training
    max_train_date = train_df[COL_INVOICE_DATE].max()
    min_test_date = test_df[COL_INVOICE_DATE].min()
    assert max_train_date <= min_test_date, "Temporal leakage detected: max train date exceeds min test date!"

    # Assert eligible users are in both train and test
    for user in eligible_users:
        assert user in train_df[COL_CUSTOMER_ID].values
        assert user in test_df[COL_CUSTOMER_ID].values
