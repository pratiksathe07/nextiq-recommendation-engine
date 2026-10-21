"""
NextIQ — Content-Based Recommendation Engine
Constructs TF-IDF textual representation of product descriptions,
builds customer preference profiles, and computes catalog content similarity.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    COL_CUSTOMER_ID,
    COL_DESCRIPTION,
    COL_QUANTITY,
    COL_STOCK_CODE,
    CONTENT_MAX_FEATURES,
    CONTENT_NGRAM_RANGE,
    DEFAULT_TOP_K,
)
from src.recommender import BaseRecommender, PopularityRecommender, RecommendationItem

logger = logging.getLogger("NextIQ.ContentBased")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def preprocess_text(text: Optional[str]) -> str:
    """Cleans and standardizes raw product descriptions for TF-IDF vectorization."""
    if text is None or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class ContentBasedRecommender(BaseRecommender):
    """
    Content-Based Filtering recommender.
    Leverages TF-IDF vectorization of product descriptions to compute product-to-product
    content similarity and build customer text-interest profiles.
    """

    def __init__(
        self,
        max_features: int = CONTENT_MAX_FEATURES,
        ngram_range: Tuple[int, int] = CONTENT_NGRAM_RANGE,
    ):
        super().__init__(name="ContentBasedRecommender")
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.content_similarity_matrix: Optional[np.ndarray] = None
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}
        self.customer_history: Dict[str, Dict[str, float]] = {}
        self.fallback_recommender: Optional[PopularityRecommender] = None

    def fit(self, interactions_df: pd.DataFrame, catalog_df: Optional[pd.DataFrame] = None) -> "ContentBasedRecommender":
        if interactions_df.empty:
            raise ValueError("Cannot fit ContentBasedRecommender on empty DataFrame.")

        # Fallback recommender for cold start
        self.fallback_recommender = PopularityRecommender(ranking_metric="customers")
        self.fallback_recommender.fit(interactions_df, catalog_df)

        # Build comprehensive catalog
        if catalog_df is not None and not catalog_df.empty:
            catalog = catalog_df.copy()
            self.catalog_map = dict(zip(catalog[COL_STOCK_CODE], catalog[COL_DESCRIPTION]))
        else:
            catalog = (
                interactions_df.dropna(subset=[COL_STOCK_CODE])
                .drop_duplicates(subset=[COL_STOCK_CODE])[[COL_STOCK_CODE, COL_DESCRIPTION]]
                .copy()
            )
            self.catalog_map = dict(zip(catalog[COL_STOCK_CODE], catalog[COL_DESCRIPTION]))

        catalog[COL_STOCK_CODE] = catalog[COL_STOCK_CODE].astype(str)
        catalog["clean_text"] = catalog[COL_DESCRIPTION].astype(str).apply(preprocess_text)

        # Fallback for empty descriptions
        catalog.loc[catalog["clean_text"] == "", "clean_text"] = "unlabeled merchandise item"

        unique_catalog_items = catalog.drop_duplicates(subset=[COL_STOCK_CODE]).reset_index(drop=True)
        self.item_to_idx = {row[COL_STOCK_CODE]: i for i, row in unique_catalog_items.iterrows()}
        self.idx_to_item = {i: item for item, i in self.item_to_idx.items()}

        logger.info(f"Vectorizing descriptions for {len(unique_catalog_items):,} products using TF-IDF...")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words="english",
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(unique_catalog_items["clean_text"]).toarray()

        logger.info("Computing product description cosine similarity matrix...")
        self.content_similarity_matrix = cosine_similarity(self.tfidf_matrix, dense_output=True)
        np.fill_diagonal(self.content_similarity_matrix, 0.0)

        # Cache customer histories
        clean_df = interactions_df.dropna(subset=[COL_CUSTOMER_ID, COL_STOCK_CODE]).copy()
        clean_df[COL_CUSTOMER_ID] = clean_df[COL_CUSTOMER_ID].astype(str)
        clean_df[COL_STOCK_CODE] = clean_df[COL_STOCK_CODE].astype(str)

        if COL_QUANTITY in clean_df.columns:
            agg = clean_df.groupby([COL_CUSTOMER_ID, COL_STOCK_CODE])[COL_QUANTITY].sum().reset_index()
            agg["weight"] = np.log1p(agg[COL_QUANTITY])
        else:
            agg = clean_df.groupby([COL_CUSTOMER_ID, COL_STOCK_CODE]).size().reset_index(name="weight")
            agg["weight"] = 1.0

        for u, item, w in zip(agg[COL_CUSTOMER_ID], agg[COL_STOCK_CODE], agg["weight"]):
            if u not in self.customer_history:
                self.customer_history[u] = {}
            self.customer_history[u][item] = float(w)

        self.is_fitted = True
        logger.info("Fitted ContentBasedRecommender successfully.")
        return self

    def get_similar_items(self, stock_code: str, top_n: int = 5) -> List[Tuple[str, str, float]]:
        """
        Retrieves top N items with the most similar textual descriptions.
        """
        if not self.is_fitted or self.content_similarity_matrix is None:
            raise RuntimeError("Model is not fitted.")

        stock_code = str(stock_code)
        if stock_code not in self.item_to_idx:
            logger.warning(f"StockCode '{stock_code}' not in Content vocabulary.")
            return []

        item_idx = self.item_to_idx[stock_code]
        sim_scores = self.content_similarity_matrix[item_idx]

        top_indices = np.argsort(sim_scores)[::-1][:top_n]
        results = []
        for idx in top_indices:
            score = float(sim_scores[idx])
            if score > 0.0:
                code = self.idx_to_item[idx]
                results.append((code, self._get_description(code), round(score, 4)))
        return results

    def recommend(
        self,
        customer_id: str,
        top_n: int = DEFAULT_TOP_K,
        exclude_purchased: bool = True,
    ) -> List[RecommendationItem]:
        if not self.is_fitted or self.tfidf_matrix is None:
            raise RuntimeError("Model must be fitted before generating recommendations.")

        cid = str(customer_id)
        if cid not in self.customer_history or not self.customer_history[cid]:
            logger.info(f"Cold-start customer '{cid}': falling back to PopularityRecommender.")
            fallback_items = self.fallback_recommender.recommend(customer_id=cid, top_n=top_n, exclude_purchased=exclude_purchased)
            for item in fallback_items:
                item.explanation = f"Cold-start fallback: {item.explanation}"
                item.strategy = "Popularity (Cold-Start Fallback)"
            return fallback_items

        user_history = self.customer_history[cid]
        purchased_items = set(user_history.keys())

        # Construct weighted customer profile in TF-IDF space
        profile_vector = np.zeros(self.tfidf_matrix.shape[1])
        total_weight = 0.0

        for item, weight in user_history.items():
            if item in self.item_to_idx:
                item_idx = self.item_to_idx[item]
                profile_vector += weight * self.tfidf_matrix[item_idx]
                total_weight += weight

        if total_weight > 0:
            profile_vector /= total_weight
            norm = np.linalg.norm(profile_vector)
            if norm > 0:
                profile_vector /= norm

        # Compute cosine similarity between customer profile and all catalog items
        scores = cosine_similarity(profile_vector.reshape(1, -1), self.tfidf_matrix).flatten()

        if exclude_purchased:
            for item in purchased_items:
                if item in self.item_to_idx:
                    scores[self.item_to_idx[item]] = -1.0

        ranked_indices = np.argsort(scores)[::-1]
        recommendations: List[RecommendationItem] = []
        rank = 1

        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0.0:
                break

            code = self.idx_to_item[idx]
            explanation = "Product description matches key themes in your historical purchases."

            recommendations.append(
                RecommendationItem(
                    stock_code=code,
                    description=self._get_description(code),
                    score=round(score, 4),
                    rank=rank,
                    explanation=explanation,
                    strategy="Content-Based Filtering",
                )
            )
            rank += 1
            if len(recommendations) >= top_n:
                break

        # Backfill if necessary
        if len(recommendations) < top_n:
            needed = top_n - len(recommendations)
            existing_codes = {item.stock_code for item in recommendations} | (purchased_items if exclude_purchased else set())
            fallback_items = self.fallback_recommender.recommend(customer_id=cid, top_n=top_n * 2, exclude_purchased=exclude_purchased)
            for fb in fallback_items:
                if fb.stock_code not in existing_codes:
                    fb.rank = rank
                    fb.explanation = f"Catalog backfill: {fb.explanation}"
                    fb.strategy = "Popularity (Backfill)"
                    recommendations.append(fb)
                    existing_codes.add(fb.stock_code)
                    rank += 1
                    if len(recommendations) >= top_n:
                        break

        return recommendations
