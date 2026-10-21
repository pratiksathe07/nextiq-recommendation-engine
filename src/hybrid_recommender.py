"""
NextIQ — Hybrid Recommendation Engine
Combines Collaborative Filtering, Content-Based Similarity, and Popularity signals
using normalized score fusion and dynamic cold-start fallbacks.
"""

import logging
from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd

from config import DEFAULT_HYBRID_WEIGHTS, DEFAULT_TOP_K
from src.content_based import ContentBasedRecommender
from src.recommender import (
    BaseRecommender,
    ItemItemCollaborativeRecommender,
    PopularityRecommender,
    RecommendationItem,
)

logger = logging.getLogger("NextIQ.HybridRecommender")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalizes a dictionary of scores into the [0, 1] interval."""
    if not scores:
        return {}
    vals = list(scores.values())
    min_val = min(vals)
    max_val = max(vals)
    rng = max_val - min_val
    if rng <= 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - min_val) / rng for k, v in scores.items()}


class HybridRecommender(BaseRecommender):
    """
    Weighted Hybrid Recommender integrating:
    1. Item-Item Collaborative Filtering (co-purchase behavioral signal)
    2. Content-Based Filtering (product description attribute signal)
    3. Popularity Ranking (unbiased catalog volume prior)
    """

    def __init__(
        self,
        cf_weight: float = DEFAULT_HYBRID_WEIGHTS["item_item_cf"],
        content_weight: float = DEFAULT_HYBRID_WEIGHTS["content_based"],
        pop_weight: float = DEFAULT_HYBRID_WEIGHTS["popularity"],
    ):
        super().__init__(name="HybridRecommender")
        self.raw_weights = {
            "item_item_cf": cf_weight,
            "content_based": content_weight,
            "popularity": pop_weight,
        }
        # Normalize weights so they sum to 1.0
        total_w = sum(self.raw_weights.values())
        if total_w <= 0:
            raise ValueError("Sum of hybrid weights must be strictly positive.")
        self.weights = {k: v / total_w for k, v in self.raw_weights.items()}

        self.cf_recommender = ItemItemCollaborativeRecommender()
        self.content_recommender = ContentBasedRecommender()
        self.popularity_recommender = PopularityRecommender(ranking_metric="customers")
        self.customer_history: Dict[str, Set[str]] = {}

    def fit(self, interactions_df: pd.DataFrame, catalog_df: Optional[pd.DataFrame] = None) -> "HybridRecommender":
        if interactions_df.empty:
            raise ValueError("Cannot fit HybridRecommender on empty DataFrame.")

        logger.info("Fitting HybridRecommender sub-models...")
        self.popularity_recommender.fit(interactions_df, catalog_df)
        self.cf_recommender.fit(interactions_df, catalog_df)
        self.content_recommender.fit(interactions_df, catalog_df)

        self.catalog_map = self.cf_recommender.catalog_map
        self.customer_history = self.popularity_recommender.customer_history

        self.is_fitted = True
        logger.info(
            f"Fitted HybridRecommender successfully with weights: "
            f"CF={self.weights['item_item_cf']:.2f}, "
            f"Content={self.weights['content_based']:.2f}, "
            f"Pop={self.weights['popularity']:.2f}"
        )
        return self

    def recommend(
        self,
        customer_id: str,
        top_n: int = DEFAULT_TOP_K,
        exclude_purchased: bool = True,
    ) -> List[RecommendationItem]:
        if not self.is_fitted:
            raise RuntimeError("HybridRecommender must be fitted before generating recommendations.")

        cid = str(customer_id)
        history = self.customer_history.get(cid, set())

        # Cold start handling
        if cid not in self.customer_history or not history:
            logger.info(f"Cold-start customer '{cid}': falling back to PopularityRecommender.")
            fallback_items = self.popularity_recommender.recommend(
                customer_id=cid, top_n=top_n, exclude_purchased=exclude_purchased
            )
            for item in fallback_items:
                item.explanation = f"Cold-start fallback: {item.explanation}"
                item.strategy = "Popularity (Hybrid Cold-Start)"
            return fallback_items

        # Retrieve a broader candidate pool from each individual component
        candidate_pool_size = max(top_n * 4, 40)

        cf_candidates = self.cf_recommender.recommend(
            customer_id=cid, top_n=candidate_pool_size, exclude_purchased=exclude_purchased
        )
        content_candidates = self.content_recommender.recommend(
            customer_id=cid, top_n=candidate_pool_size, exclude_purchased=exclude_purchased
        )
        pop_candidates = self.popularity_recommender.recommend(
            customer_id=cid, top_n=candidate_pool_size, exclude_purchased=exclude_purchased
        )

        # Build raw score maps
        cf_scores = {item.stock_code: item.score for item in cf_candidates}
        content_scores = {item.stock_code: item.score for item in content_candidates}
        pop_scores = {item.stock_code: item.score for item in pop_candidates}

        # Min-max normalize each signal dictionary
        norm_cf = min_max_normalize(cf_scores)
        norm_content = min_max_normalize(content_scores)
        norm_pop = min_max_normalize(pop_scores)

        # Union of all candidate item keys
        all_candidates = set(norm_cf.keys()) | set(norm_content.keys()) | set(norm_pop.keys())
        if exclude_purchased:
            all_candidates = all_candidates - history

        # Compute combined weighted score
        w_cf = self.weights["item_item_cf"]
        w_cnt = self.weights["content_based"]
        w_pop = self.weights["popularity"]

        combined_scores: Dict[str, float] = {}
        for code in all_candidates:
            s_cf = norm_cf.get(code, 0.0)
            s_cnt = norm_content.get(code, 0.0)
            s_pop = norm_pop.get(code, 0.0)
            combined = (w_cf * s_cf) + (w_cnt * s_cnt) + (w_pop * s_pop)
            combined_scores[code] = combined

        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        recommendations: List[RecommendationItem] = []
        for rank, (code, score) in enumerate(sorted_items, start=1):
            s_cf = norm_cf.get(code, 0.0)
            s_cnt = norm_content.get(code, 0.0)
            s_pop = norm_pop.get(code, 0.0)

            # Build transparent multi-factor explanation
            signals = []
            if s_cf > 0.15:
                signals.append("co-purchase history")
            if s_cnt > 0.15:
                signals.append("product description similarity")
            if s_pop > 0.30:
                signals.append("high catalog popularity")

            if signals:
                signal_str = " + ".join(signals)
                explanation = f"Hybrid recommendation balancing {signal_str}."
            else:
                explanation = "Hybrid recommendation combining collaborative and catalog signals."

            recommendations.append(
                RecommendationItem(
                    stock_code=code,
                    description=self._get_description(code),
                    score=round(score, 4),
                    rank=rank,
                    explanation=explanation,
                    strategy="Hybrid (CF + Content + Popularity)",
                )
            )

        return recommendations
