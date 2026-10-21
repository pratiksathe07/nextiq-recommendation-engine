"""
Unit Tests for NextIQ Hybrid Recommendation Engine
Tests multi-signal score fusion, weight normalization, and fallback execution.
"""

import numpy as np
import pandas as pd
import pytest

from config import (
    COL_CUSTOMER_ID,
    COL_DESCRIPTION,
    COL_INVOICE_DATE,
    COL_INVOICE_NO,
    COL_QUANTITY,
    COL_STOCK_CODE,
    COL_UNIT_PRICE,
)
from src.hybrid_recommender import HybridRecommender, min_max_normalize


@pytest.fixture
def hybrid_test_df() -> pd.DataFrame:
    """Provides a synthetic interaction DataFrame for hybrid testing."""
    records = [
        ("INV1", "A1", "VINTAGE HEART LIGHT", 5, "2011-01-01", 2.5, "CUST1"),
        ("INV1", "A2", "RETRO STAR LIGHT", 4, "2011-01-01", 2.5, "CUST1"),
        ("INV2", "A1", "VINTAGE HEART LIGHT", 3, "2011-01-02", 2.5, "CUST2"),
        ("INV2", "B1", "METAL FLOWER POT", 2, "2011-01-02", 4.0, "CUST2"),
        ("INV3", "A2", "RETRO STAR LIGHT", 6, "2011-01-03", 2.5, "CUST3"),
        ("INV3", "B2", "CERAMIC GARDEN POT", 1, "2011-01-03", 4.0, "CUST3"),
    ]
    df = pd.DataFrame(
        records,
        columns=[
            COL_INVOICE_NO,
            COL_STOCK_CODE,
            COL_DESCRIPTION,
            COL_QUANTITY,
            COL_INVOICE_DATE,
            COL_UNIT_PRICE,
            COL_CUSTOMER_ID,
        ],
    )
    df["log_quantity"] = np.log1p(df[COL_QUANTITY])
    return df


def test_min_max_normalize():
    """Verifies score min-max scaling to [0, 1]."""
    scores = {"item1": 10.0, "item2": 20.0, "item3": 30.0}
    norm = min_max_normalize(scores)
    assert norm["item1"] == pytest.approx(0.0)
    assert norm["item2"] == pytest.approx(0.5)
    assert norm["item3"] == pytest.approx(1.0)

    # Constant scores should normalize to 1.0
    constant_scores = {"item1": 5.0, "item2": 5.0}
    norm_const = min_max_normalize(constant_scores)
    assert norm_const["item1"] == 1.0
    assert norm_const["item2"] == 1.0


def test_hybrid_recommender_weights():
    """Verifies that custom raw weights are correctly normalized to sum to 1.0."""
    model = HybridRecommender(cf_weight=2.0, content_weight=2.0, pop_weight=1.0)
    assert model.weights["item_item_cf"] == pytest.approx(0.4)
    assert model.weights["content_based"] == pytest.approx(0.4)
    assert model.weights["popularity"] == pytest.approx(0.2)
    assert sum(model.weights.values()) == pytest.approx(1.0)


def test_hybrid_recommender_recommend(hybrid_test_df):
    """Verifies hybrid recommendations for an active customer."""
    model = HybridRecommender(cf_weight=0.5, content_weight=0.3, pop_weight=0.2).fit(hybrid_test_df)

    # CUST2 bought A1 and B1; recommend for CUST2
    recs = model.recommend(customer_id="CUST2", top_n=2, exclude_purchased=True)
    assert len(recs) > 0
    codes = [r.stock_code for r in recs]
    assert "A1" not in codes
    assert "B1" not in codes
    assert "Hybrid" in recs[0].strategy
    assert len(recs[0].explanation) > 0


def test_hybrid_recommender_cold_start(hybrid_test_df):
    """Verifies hybrid fallback to popularity for completely unseen customer."""
    model = HybridRecommender().fit(hybrid_test_df)
    recs = model.recommend(customer_id="NON_EXISTENT_USER", top_n=3)
    assert len(recs) == 3
    assert "Cold-start fallback" in recs[0].explanation
