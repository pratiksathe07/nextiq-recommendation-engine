"""
Unit Tests for NextIQ Recommendation Algorithms
Tests Popularity, Item-Item CF, User-User CF, Content-Based CF,
exclusion of purchased items, and cold-start fallback logic.
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
from src.content_based import ContentBasedRecommender
from src.recommender import (
    ItemItemCollaborativeRecommender,
    PopularityRecommender,
    UserUserCollaborativeRecommender,
)


@pytest.fixture
def synthetic_interactions() -> pd.DataFrame:
    """
    Creates a small, controlled transaction dataset with known co-purchases:
    - User U1 bought: ITEM_A, ITEM_B, ITEM_C
    - User U2 bought: ITEM_A, ITEM_B
    - User U3 bought: ITEM_C, ITEM_D
    - User U4 bought: ITEM_D
    - Cold-start User U_NEW has 0 history
    """
    records = [
        # U1
        ("INV1", "ITEM_A", "HEART LIGHT HOLDER", 5, "2011-01-01", 2.5, "U1"),
        ("INV1", "ITEM_B", "STAR LIGHT HOLDER", 3, "2011-01-01", 2.5, "U1"),
        ("INV2", "ITEM_C", "VINTAGE COAT HANGER", 2, "2011-01-02", 3.0, "U1"),
        # U2
        ("INV3", "ITEM_A", "HEART LIGHT HOLDER", 4, "2011-01-03", 2.5, "U2"),
        ("INV3", "ITEM_B", "STAR LIGHT HOLDER", 4, "2011-01-03", 2.5, "U2"),
        # U3
        ("INV4", "ITEM_C", "VINTAGE COAT HANGER", 6, "2011-01-04", 3.0, "U3"),
        ("INV4", "ITEM_D", "ANTIQUE METAL LANTERN", 2, "2011-01-04", 4.0, "U3"),
        # U4
        ("INV5", "ITEM_D", "ANTIQUE METAL LANTERN", 1, "2011-01-05", 4.0, "U4"),
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


def test_popularity_recommender(synthetic_interactions):
    """Verifies popularity ranking by customers, quantity, and invoices."""
    pop_cust = PopularityRecommender(ranking_metric="customers").fit(synthetic_interactions)
    recs = pop_cust.recommend(top_n=3, exclude_purchased=False)
    assert len(recs) == 3
    # ITEM_A, ITEM_B, ITEM_C each have 2 distinct customers; ITEM_D has 2
    assert recs[0].stock_code in ["ITEM_A", "ITEM_B", "ITEM_C", "ITEM_D"]
    assert recs[0].score > 0
    assert "Popular baseline" in recs[0].explanation

    # Exclude purchased for U2 (bought A and B) -> should recommend C or D
    recs_u2 = pop_cust.recommend(customer_id="U2", top_n=2, exclude_purchased=True)
    rec_codes = [r.stock_code for r in recs_u2]
    assert "ITEM_A" not in rec_codes
    assert "ITEM_B" not in rec_codes


def test_item_item_collaborative_recommender(synthetic_interactions):
    """Verifies item-item CF similarity and recommendations."""
    model = ItemItemCollaborativeRecommender().fit(synthetic_interactions)

    # ITEM_A and ITEM_B are co-purchased by U1 and U2 -> strong similarity
    sim_items = model.get_similar_items("ITEM_A", top_n=2)
    assert len(sim_items) > 0
    top_sim_code = sim_items[0][0]
    assert top_sim_code == "ITEM_B"
    assert sim_items[0][2] > 0.0

    # User U2 bought ITEM_A and ITEM_B; model should recommend ITEM_C (co-bought with ITEM_A/B by U1)
    recs = model.recommend(customer_id="U2", top_n=2, exclude_purchased=True)
    assert len(recs) > 0
    codes = [r.stock_code for r in recs]
    assert "ITEM_A" not in codes
    assert "ITEM_B" not in codes
    assert "ITEM_C" in codes


def test_item_item_cold_start(synthetic_interactions):
    """Verifies cold-start fallback when customer has no history."""
    model = ItemItemCollaborativeRecommender().fit(synthetic_interactions)
    recs = model.recommend(customer_id="UNKNOWN_CUSTOMER_999", top_n=3)
    assert len(recs) == 3
    assert "Cold-start fallback" in recs[0].explanation


def test_user_user_collaborative_recommender(synthetic_interactions):
    """Verifies user-user CF neighbor finding and recommendations."""
    model = UserUserCollaborativeRecommender(neighborhood_size=2).fit(synthetic_interactions)

    # For U2 (bought A and B), nearest neighbor is U1 (bought A, B, C)
    recs = model.recommend(customer_id="U2", top_n=2, exclude_purchased=True)
    assert len(recs) > 0
    codes = [r.stock_code for r in recs]
    assert "ITEM_A" not in codes
    assert "ITEM_B" not in codes
    assert "ITEM_C" in codes


def test_content_based_recommender(synthetic_interactions):
    """Verifies content-based TF-IDF description similarity."""
    model = ContentBasedRecommender().fit(synthetic_interactions)

    # ITEM_A ("HEART LIGHT HOLDER") and ITEM_B ("STAR LIGHT HOLDER") share words "LIGHT HOLDER"
    sim_items = model.get_similar_items("ITEM_A", top_n=2)
    assert len(sim_items) > 0
    assert sim_items[0][0] == "ITEM_B"
    assert sim_items[0][2] > 0.0

    # Recommendations for U4 (bought ITEM_D "ANTIQUE METAL LANTERN")
    recs = model.recommend(customer_id="U4", top_n=2, exclude_purchased=True)
    assert len(recs) > 0
    rec_codes = [r.stock_code for r in recs]
    assert "ITEM_D" not in rec_codes
