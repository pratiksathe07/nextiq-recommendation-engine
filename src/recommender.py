"""
NextIQ — Core Collaborative Filtering & Popularity Recommendation Engines
Implements Popularity, Item-Item CF, and User-User CF with sparse matrix optimizations,
cold-start fallbacks, and transparent human-readable explanations.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import logging
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    COL_CUSTOMER_ID,
    COL_DESCRIPTION,
    COL_INVOICE_NO,
    COL_QUANTITY,
    COL_STOCK_CODE,
    DEFAULT_TOP_K,
    NEIGHBORHOOD_SIZE,
)

logger = logging.getLogger("NextIQ.Recommender")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@dataclass
class RecommendationItem:
    """Represents a single ranked product recommendation with metadata and rationale."""
    stock_code: str
    description: str
    score: float
    rank: int
    explanation: str
    strategy: str

    def to_dict(self) -> Dict:
        return asdict(self)


class BaseRecommender(ABC):
    """Abstract base class for NextIQ recommendation algorithms."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.catalog_map: Dict[str, str] = {}  # StockCode -> Description

    @abstractmethod
    def fit(self, interactions_df: pd.DataFrame, catalog_df: Optional[pd.DataFrame] = None) -> "BaseRecommender":
        """Fits the recommender on historical training interactions."""
        pass

    @abstractmethod
    def recommend(
        self,
        customer_id: str,
        top_n: int = DEFAULT_TOP_K,
        exclude_purchased: bool = True,
    ) -> List[RecommendationItem]:
        """Generates ranked recommendations for a given customer."""
        pass

    def _get_description(self, stock_code: str) -> str:
        """Looks up canonical description from catalog, with fallback."""
        return self.catalog_map.get(stock_code, "PRODUCT " + str(stock_code))


class PopularityRecommender(BaseRecommender):
    """
    Non-personalized baseline recommending top products ranked by:
    - 'customers': Distinct customer count (broadest appeal)
    - 'quantity': Total units purchased (volume demand)
    - 'invoices': Number of distinct orders (order frequency)
    """

    def __init__(self, ranking_metric: str = "customers"):
        super().__init__(name="PopularityRecommender")
        if ranking_metric not in ["customers", "quantity", "invoices"]:
            raise ValueError(f"Unsupported ranking metric '{ranking_metric}'. Choose from: 'customers', 'quantity', 'invoices'")
        self.ranking_metric = ranking_metric
        self.ranked_items: List[Tuple[str, float, int]] = []
        self.customer_history: Dict[str, Set[str]] = {}

    def fit(self, interactions_df: pd.DataFrame, catalog_df: Optional[pd.DataFrame] = None) -> "PopularityRecommender":
        if interactions_df.empty:
            raise ValueError("Cannot fit PopularityRecommender on empty DataFrame.")

        # Update catalog map
        if catalog_df is not None and not catalog_df.empty:
            self.catalog_map = dict(zip(catalog_df[COL_STOCK_CODE], catalog_df[COL_DESCRIPTION]))
        elif COL_DESCRIPTION in interactions_df.columns:
            valid_desc = interactions_df.dropna(subset=[COL_DESCRIPTION])
            self.catalog_map = dict(zip(valid_desc[COL_STOCK_CODE], valid_desc[COL_DESCRIPTION]))

        # Cache customer purchase histories
        self.customer_history = (
            interactions_df.groupby(COL_CUSTOMER_ID)[COL_STOCK_CODE]
            .apply(set)
            .to_dict()
        )

        # Aggregate popularity metrics
        grouped = interactions_df.groupby(COL_STOCK_CODE).agg(
            customers=(COL_CUSTOMER_ID, "nunique"),
            quantity=(COL_QUANTITY, "sum") if COL_QUANTITY in interactions_df.columns else (COL_STOCK_CODE, "count"),
            invoices=(COL_INVOICE_NO, "nunique") if COL_INVOICE_NO in interactions_df.columns else (COL_STOCK_CODE, "count"),
        ).reset_index()

        sort_col = self.ranking_metric
        grouped = grouped.sort_values(sort_col, ascending=False).reset_index(drop=True)

        # Normalize score to 0..1 range
        max_val = grouped[sort_col].max()
        if max_val > 0:
            grouped["score"] = grouped[sort_col] / max_val
        else:
            grouped["score"] = 1.0

        self.ranked_items = [
            (row[COL_STOCK_CODE], float(row["score"]), int(row[sort_col]))
            for _, row in grouped.iterrows()
        ]

        self.is_fitted = True
        logger.info(f"Fitted PopularityRecommender on {len(self.ranked_items):,} items (metric: '{self.ranking_metric}').")
        return self

    def recommend(
        self,
        customer_id: Optional[str] = None,
        top_n: int = DEFAULT_TOP_K,
        exclude_purchased: bool = True,
    ) -> List[RecommendationItem]:
        if not self.is_fitted:
            raise RuntimeError("PopularityRecommender must be fitted before generating recommendations.")

        history = self.customer_history.get(str(customer_id), set()) if customer_id else set()
        recommendations: List[RecommendationItem] = []
        rank = 1

        for stock_code, score, raw_count in self.ranked_items:
            if exclude_purchased and stock_code in history:
                continue

            metric_labels = {
                "customers": f"Purchased by {raw_count:,} unique customers",
                "quantity": f"Top volume seller ({raw_count:,} units sold)",
                "invoices": f"Ordered across {raw_count:,} separate orders",
            }
            explanation = f"Popular baseline: {metric_labels[self.ranking_metric]} across the catalog."

            recommendations.append(
                RecommendationItem(
                    stock_code=stock_code,
                    description=self._get_description(stock_code),
                    score=round(score, 4),
                    rank=rank,
                    explanation=explanation,
                    strategy="Popularity",
                )
            )
            rank += 1
            if len(recommendations) >= top_n:
                break

        return recommendations


class ItemItemCollaborativeRecommender(BaseRecommender):
    """
    Item-to-Item Collaborative Filtering using cosine similarity on sparse customer-item interactions.
    Recommends products similar to items in the customer's purchase history.
    Includes cold-start fallback to PopularityRecommender.
    """

    def __init__(self, neighborhood_size: int = NEIGHBORHOOD_SIZE, weighting: str = "log_quantity"):
        super().__init__(name="ItemItemCollaborativeRecommender")
        self.neighborhood_size = neighborhood_size
        self.weighting = weighting
        self.item_similarity_matrix: Optional[np.ndarray] = None
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}
        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.interaction_matrix: Optional[csr_matrix] = None
        self.customer_history: Dict[str, Dict[str, float]] = {}  # Customer -> {Item: Weight}
        self.fallback_recommender: Optional[PopularityRecommender] = None

    def fit(self, interactions_df: pd.DataFrame, catalog_df: Optional[pd.DataFrame] = None) -> "ItemItemCollaborativeRecommender":
        if interactions_df.empty:
            raise ValueError("Cannot fit ItemItemCollaborativeRecommender on empty DataFrame.")

        # Fit fallback recommender for cold start
        self.fallback_recommender = PopularityRecommender(ranking_metric="customers")
        self.fallback_recommender.fit(interactions_df, catalog_df)

        if catalog_df is not None and not catalog_df.empty:
            self.catalog_map = dict(zip(catalog_df[COL_STOCK_CODE], catalog_df[COL_DESCRIPTION]))
        elif COL_DESCRIPTION in interactions_df.columns:
            valid_desc = interactions_df.dropna(subset=[COL_DESCRIPTION])
            self.catalog_map = dict(zip(valid_desc[COL_STOCK_CODE], valid_desc[COL_DESCRIPTION]))

        # Filter out NaN customers
        clean_df = interactions_df.dropna(subset=[COL_CUSTOMER_ID, COL_STOCK_CODE]).copy()
        clean_df[COL_CUSTOMER_ID] = clean_df[COL_CUSTOMER_ID].astype(str)
        clean_df[COL_STOCK_CODE] = clean_df[COL_STOCK_CODE].astype(str)

        # Aggregate interactions
        if "log_quantity" in clean_df.columns:
            agg = clean_df[[COL_CUSTOMER_ID, COL_STOCK_CODE, "log_quantity"]].drop_duplicates()
            agg = agg.rename(columns={"log_quantity": "weight"})
        elif COL_QUANTITY in clean_df.columns:
            agg = clean_df.groupby([COL_CUSTOMER_ID, COL_STOCK_CODE])[COL_QUANTITY].sum().reset_index()
            agg["weight"] = np.log1p(agg[COL_QUANTITY])
        else:
            agg = clean_df.groupby([COL_CUSTOMER_ID, COL_STOCK_CODE]).size().reset_index(name="weight")
            agg["weight"] = 1.0

        unique_users = sorted(agg[COL_CUSTOMER_ID].unique())
        unique_items = sorted(agg[COL_STOCK_CODE].unique())

        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.item_to_idx = {item: i for i, item in enumerate(unique_items)}
        self.idx_to_item = {i: item for item, i in self.item_to_idx.items()}

        rows = agg[COL_CUSTOMER_ID].map(self.user_to_idx).values
        cols = agg[COL_STOCK_CODE].map(self.item_to_idx).values
        data = agg["weight"].values

        # User-Item interaction matrix (N_users x N_items)
        self.interaction_matrix = csr_matrix((data, (rows, cols)), shape=(len(unique_users), len(unique_items)))

        # Build Customer History lookup
        for u, item, w in zip(agg[COL_CUSTOMER_ID], agg[COL_STOCK_CODE], agg["weight"]):
            if u not in self.customer_history:
                self.customer_history[u] = {}
            self.customer_history[u][item] = float(w)

        # Compute Item-Item Cosine Similarity matrix (Item x Item)
        # Transpose to Item x User
        item_user_matrix = self.interaction_matrix.T.tocsr()
        logger.info(f"Computing Item-Item similarity matrix for {len(unique_items):,} catalog products...")
        self.item_similarity_matrix = cosine_similarity(item_user_matrix, dense_output=True)
        # Zero out diagonal self-similarity
        np.fill_diagonal(self.item_similarity_matrix, 0.0)

        self.is_fitted = True
        logger.info(f"Fitted ItemItemCollaborativeRecommender successfully.")
        return self

    def get_similar_items(self, stock_code: str, top_n: int = 5) -> List[Tuple[str, str, float]]:
        """
        Retrieves top N items most similar to a target item based on co-purchase patterns.
        Returns: List of (StockCode, Description, similarity_score).
        """
        if not self.is_fitted or self.item_similarity_matrix is None:
            raise RuntimeError("Model is not fitted.")

        stock_code = str(stock_code)
        if stock_code not in self.item_to_idx:
            logger.warning(f"StockCode '{stock_code}' not in Item-Item vocabulary.")
            return []

        item_idx = self.item_to_idx[stock_code]
        sim_scores = self.item_similarity_matrix[item_idx]

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
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generating recommendations.")

        cid = str(customer_id)
        # Cold start handling: customer unseen or has 0 interactions
        if cid not in self.user_to_idx or cid not in self.customer_history or not self.customer_history[cid]:
            logger.info(f"Cold-start customer '{cid}': falling back to PopularityRecommender.")
            fallback_items = self.fallback_recommender.recommend(customer_id=cid, top_n=top_n, exclude_purchased=exclude_purchased)
            for item in fallback_items:
                item.explanation = f"Cold-start fallback: {item.explanation}"
                item.strategy = "Popularity (Cold-Start Fallback)"
            return fallback_items

        user_history = self.customer_history[cid]
        purchased_items = set(user_history.keys())

        # Vector of user interaction weights across all items (1 x N_items)
        user_vector = np.zeros(len(self.item_to_idx))
        for item, weight in user_history.items():
            if item in self.item_to_idx:
                user_vector[self.item_to_idx[item]] = weight

        # Score candidates: UserVector @ ItemSimilarityMatrix (1 x N_items)
        scores = user_vector @ self.item_similarity_matrix

        if exclude_purchased:
            for item in purchased_items:
                if item in self.item_to_idx:
                    scores[self.item_to_idx[item]] = -1.0

        # Sort candidate items by predicted score
        ranked_indices = np.argsort(scores)[::-1]
        recommendations: List[RecommendationItem] = []
        rank = 1

        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0.0:
                break  # No meaningful similarity signal left

            code = self.idx_to_item[idx]
            
            # Find the historical item that contributed most to this recommendation
            top_source_item = None
            max_contribution = -1.0
            for hist_item, hist_weight in user_history.items():
                if hist_item in self.item_to_idx:
                    hist_idx = self.item_to_idx[hist_item]
                    sim = self.item_similarity_matrix[hist_idx, idx]
                    contribution = sim * hist_weight
                    if contribution > max_contribution:
                        max_contribution = contribution
                        top_source_item = hist_item

            if top_source_item:
                source_desc = self._get_description(top_source_item)[:28]
                explanation = f"Frequently co-purchased with '{source_desc}' ({top_source_item}) in your history."
            else:
                explanation = "Similar to products in your historical purchase basket."

            recommendations.append(
                RecommendationItem(
                    stock_code=code,
                    description=self._get_description(code),
                    score=round(score, 4),
                    rank=rank,
                    explanation=explanation,
                    strategy="Item-Item Collaborative Filtering",
                )
            )
            rank += 1
            if len(recommendations) >= top_n:
                break

        # If sparse history yielded fewer than top_n recommendations, backfill with popularity
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


class UserUserCollaborativeRecommender(BaseRecommender):
    """
    User-to-User Collaborative Filtering.
    Finds nearest neighbors for the active user via cosine similarity and aggregates their purchases.
    Avoids dense N x N matrix instantiation to keep memory usage minimal.
    """

    def __init__(self, neighborhood_size: int = NEIGHBORHOOD_SIZE):
        super().__init__(name="UserUserCollaborativeRecommender")
        self.neighborhood_size = neighborhood_size
        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}
        self.user_item_matrix: Optional[csr_matrix] = None
        self.customer_history: Dict[str, Set[str]] = {}
        self.fallback_recommender: Optional[PopularityRecommender] = None

    def fit(self, interactions_df: pd.DataFrame, catalog_df: Optional[pd.DataFrame] = None) -> "UserUserCollaborativeRecommender":
        if interactions_df.empty:
            raise ValueError("Cannot fit UserUserCollaborativeRecommender on empty DataFrame.")

        # Fallback for cold start
        self.fallback_recommender = PopularityRecommender(ranking_metric="customers")
        self.fallback_recommender.fit(interactions_df, catalog_df)

        if catalog_df is not None and not catalog_df.empty:
            self.catalog_map = dict(zip(catalog_df[COL_STOCK_CODE], catalog_df[COL_DESCRIPTION]))
        elif COL_DESCRIPTION in interactions_df.columns:
            valid_desc = interactions_df.dropna(subset=[COL_DESCRIPTION])
            self.catalog_map = dict(zip(valid_desc[COL_STOCK_CODE], valid_desc[COL_DESCRIPTION]))

        clean_df = interactions_df.dropna(subset=[COL_CUSTOMER_ID, COL_STOCK_CODE]).copy()
        clean_df[COL_CUSTOMER_ID] = clean_df[COL_CUSTOMER_ID].astype(str)
        clean_df[COL_STOCK_CODE] = clean_df[COL_STOCK_CODE].astype(str)

        # Implicit feedback weights
        if "log_quantity" in clean_df.columns:
            agg = clean_df[[COL_CUSTOMER_ID, COL_STOCK_CODE, "log_quantity"]].drop_duplicates()
            agg = agg.rename(columns={"log_quantity": "weight"})
        elif COL_QUANTITY in clean_df.columns:
            agg = clean_df.groupby([COL_CUSTOMER_ID, COL_STOCK_CODE])[COL_QUANTITY].sum().reset_index()
            agg["weight"] = np.log1p(agg[COL_QUANTITY])
        else:
            agg = clean_df.groupby([COL_CUSTOMER_ID, COL_STOCK_CODE]).size().reset_index(name="weight")
            agg["weight"] = 1.0

        unique_users = sorted(agg[COL_CUSTOMER_ID].unique())
        unique_items = sorted(agg[COL_STOCK_CODE].unique())

        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.item_to_idx = {item: i for i, item in enumerate(unique_items)}
        self.idx_to_item = {i: item for item, i in self.item_to_idx.items()}

        rows = agg[COL_CUSTOMER_ID].map(self.user_to_idx).values
        cols = agg[COL_STOCK_CODE].map(self.item_to_idx).values
        data = agg["weight"].values

        self.user_item_matrix = csr_matrix((data, (rows, cols)), shape=(len(unique_users), len(unique_items)))
        self.customer_history = clean_df.groupby(COL_CUSTOMER_ID)[COL_STOCK_CODE].apply(set).to_dict()

        self.is_fitted = True
        logger.info(f"Fitted UserUserCollaborativeRecommender on {len(unique_users):,} users and {len(unique_items):,} items.")
        return self

    def recommend(
        self,
        customer_id: str,
        top_n: int = DEFAULT_TOP_K,
        exclude_purchased: bool = True,
    ) -> List[RecommendationItem]:
        if not self.is_fitted or self.user_item_matrix is None:
            raise RuntimeError("Model must be fitted before generating recommendations.")

        cid = str(customer_id)
        if cid not in self.user_to_idx or cid not in self.customer_history:
            logger.info(f"Cold-start customer '{cid}': falling back to PopularityRecommender.")
            fallback_items = self.fallback_recommender.recommend(customer_id=cid, top_n=top_n, exclude_purchased=exclude_purchased)
            for item in fallback_items:
                item.explanation = f"Cold-start fallback: {item.explanation}"
                item.strategy = "Popularity (Cold-Start Fallback)"
            return fallback_items

        user_idx = self.user_to_idx[cid]
        purchased_items = self.customer_history.get(cid, set())

        # Efficiently compute similarities between this user and all other users
        user_row = self.user_item_matrix.getrow(user_idx)
        user_similarities = cosine_similarity(user_row, self.user_item_matrix).flatten()
        user_similarities[user_idx] = 0.0  # Zero out self

        # Identify top-K neighbor users
        neighbor_indices = np.argsort(user_similarities)[::-1][: self.neighborhood_size]
        neighbor_weights = user_similarities[neighbor_indices]

        # Filter neighbors with positive similarity
        valid_mask = neighbor_weights > 0.0
        neighbor_indices = neighbor_indices[valid_mask]
        neighbor_weights = neighbor_weights[valid_mask]

        if len(neighbor_indices) == 0:
            # No similar neighbors found; fallback to popularity
            fallback_items = self.fallback_recommender.recommend(customer_id=cid, top_n=top_n, exclude_purchased=exclude_purchased)
            for item in fallback_items:
                item.explanation = f"Insufficient neighbor overlap fallback: {item.explanation}"
                item.strategy = "Popularity (Neighbor Fallback)"
            return fallback_items

        # Aggregate neighbor item ratings: neighbor_weights @ neighbor_rows
        neighbor_matrix = self.user_item_matrix[neighbor_indices]
        # Predicted item scores (1 x N_items)
        scores = neighbor_matrix.T.dot(neighbor_weights)

        # Normalize by sum of neighbor weights
        sum_weights = neighbor_weights.sum()
        if sum_weights > 0:
            scores = scores / sum_weights

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
            explanation = (
                f"Frequently bought by {len(neighbor_indices)} customers with shopping patterns similar to yours."
            )

            recommendations.append(
                RecommendationItem(
                    stock_code=code,
                    description=self._get_description(code),
                    score=round(score, 4),
                    rank=rank,
                    explanation=explanation,
                    strategy="User-User Collaborative Filtering",
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
