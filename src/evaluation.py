"""
NextIQ — Offline Recommendation Model Evaluation Pipeline
Implements rigorous chronological train/test splitting, ranking metrics
(Recall@K, Precision@K, NDCG@K, Catalog Coverage, Diversity), and benchmark runners.
"""

from dataclasses import asdict, dataclass
import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    COL_CUSTOMER_ID,
    COL_INVOICE_DATE,
    COL_STOCK_CODE,
    DEFAULT_TOP_K,
    EVALUATION_K_VALUES,
    EVALUATION_SPLIT_RATIO,
)
from src.content_based import ContentBasedRecommender
from src.hybrid_recommender import HybridRecommender
from src.recommender import (
    BaseRecommender,
    ItemItemCollaborativeRecommender,
    PopularityRecommender,
    UserUserCollaborativeRecommender,
)

logger = logging.getLogger("NextIQ.Evaluation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ============================================================================
# Core Metric Functions
# ============================================================================

def recall_at_k(recommended: List[str], actual: Set[str], k: int = 10) -> float:
    """
    Computes Recall@K: proportion of relevant test items captured in top K recommendations.
    Recall@K = |Recommended[:K] ∩ Actual| / |Actual|
    """
    if not actual or k <= 0:
        return 0.0
    rec_k = set(recommended[:k])
    hits = len(rec_k & actual)
    return hits / len(actual)


def precision_at_k(recommended: List[str], actual: Set[str], k: int = 10) -> float:
    """
    Computes Precision@K: proportion of top K recommendations that are relevant.
    Precision@K = |Recommended[:K] ∩ Actual| / K
    """
    if k <= 0:
        return 0.0
    rec_k = set(recommended[:k])
    hits = len(rec_k & actual)
    return hits / k


def ndcg_at_k(recommended: List[str], actual: Set[str], k: int = 10) -> float:
    """
    Computes Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance.
    DCG@K = sum(rel_i / log2(i + 1))
    IDCG@K = sum(1 / log2(i + 1)) for i in 1..min(k, |Actual|)
    """
    if not actual or k <= 0:
        return 0.0

    rec_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k):
        if item in actual:
            # i is 0-indexed, so position is (i + 1), denominator is log2(position + 1) = log2(i + 2)
            dcg += 1.0 / math.log2(i + 2)

    # Ideal DCG
    ideal_hits = min(k, len(actual))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def catalog_coverage(all_recommendations: List[List[str]], catalog_items: Set[str]) -> float:
    """
    Computes Catalog Coverage: fraction of unique catalog items recommended at least once.
    Coverage = |Union of all recommendations| / |Catalog|
    """
    if not catalog_items:
        return 0.0
    recommended_set: Set[str] = set()
    for rec_list in all_recommendations:
        recommended_set.update(rec_list)
    return len(recommended_set & catalog_items) / len(catalog_items)


def intra_list_diversity(
    recommendations: List[str],
    similarity_lookup: Dict[Tuple[str, str], float],
) -> float:
    """
    Computes intra-list diversity as average pairwise distance (1 - similarity)
    among recommended items.
    """
    n = len(recommendations)
    if n <= 1:
        return 0.0

    pairwise_distances = []
    for i in range(n):
        for j in range(i + 1, n):
            item_a = recommendations[i]
            item_b = recommendations[j]
            # Lookup similarity, default to 0.0 if unknown
            sim = similarity_lookup.get((item_a, item_b), similarity_lookup.get((item_b, item_a), 0.0))
            dist = max(0.0, 1.0 - sim)
            pairwise_distances.append(dist)

    return float(np.mean(pairwise_distances)) if pairwise_distances else 0.0


# ============================================================================
# Chronological Train / Test Splitter
# ============================================================================

def chronological_train_test_split(
    df: pd.DataFrame,
    test_ratio: float = EVALUATION_SPLIT_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, Set[str]]:
    """
    Splits transactions chronologically so future purchases never leak into training.
    
    Uses a global temporal cutoff determined by quantile of InvoiceDate.
    - Training data: all records where InvoiceDate <= cutoff.
    - Test data: all records where InvoiceDate > cutoff.
    - Eligible users: users with >= 1 purchase in training AND >= 1 purchase in test.
    """
    if df.empty:
        raise ValueError("Cannot split empty DataFrame.")

    clean_df = df.dropna(subset=[COL_CUSTOMER_ID, COL_STOCK_CODE, COL_INVOICE_DATE]).copy()
    clean_df[COL_INVOICE_DATE] = pd.to_datetime(clean_df[COL_INVOICE_DATE])
    clean_df = clean_df.sort_values(COL_INVOICE_DATE).reset_index(drop=True)

    # Determine global chronological cutoff
    cutoff_idx = int(len(clean_df) * (1.0 - test_ratio))
    cutoff_date = clean_df[COL_INVOICE_DATE].iloc[cutoff_idx]

    train_df = clean_df[clean_df[COL_INVOICE_DATE] <= cutoff_date].copy()
    test_df = clean_df[clean_df[COL_INVOICE_DATE] > cutoff_date].copy()

    logger.info(
        f"Chronological split at {cutoff_date}: "
        f"Train={len(train_df):,} records ({train_df[COL_INVOICE_DATE].min()} to {train_df[COL_INVOICE_DATE].max()}), "
        f"Test={len(test_df):,} records ({test_df[COL_INVOICE_DATE].min()} to {test_df[COL_INVOICE_DATE].max()})."
    )

    # Identify eligible users (active in train and has held-out interactions in test)
    train_users = set(train_df[COL_CUSTOMER_ID].unique())
    test_users = set(test_df[COL_CUSTOMER_ID].unique())
    eligible_users = train_users & test_users

    logger.info(
        f"Found {len(eligible_users):,} eligible evaluation customers "
        f"(active in both train and holdout periods)."
    )

    return train_df, test_df, eligible_users


# ============================================================================
# Benchmark Runner
# ============================================================================

@dataclass
class ModelEvaluationResult:
    """Aggregated evaluation metrics for a single recommendation algorithm."""
    model_name: str
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    precision_at_10: float
    ndcg_at_10: float
    catalog_coverage: float
    recommendation_diversity: float
    evaluated_users: int
    test_interactions_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_recommender(
    model: BaseRecommender,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    eligible_users: Set[str],
    catalog_items: Set[str],
    similarity_lookup: Dict[Tuple[str, str], float],
    max_eval_users: int = 150,
) -> ModelEvaluationResult:
    """
    Evaluates a single model against held-out test interactions.
    """
    # Group test ground truth by customer
    test_ground_truth = (
        test_df[test_df[COL_CUSTOMER_ID].isin(eligible_users)]
        .groupby(COL_CUSTOMER_ID)[COL_STOCK_CODE]
        .apply(set)
        .to_dict()
    )

    eval_user_list = list(test_ground_truth.keys())
    if len(eval_user_list) > max_eval_users:
        # Deterministic sample for reproducibility and speed
        rng = np.random.RandomState(42)
        eval_user_list = sorted(list(rng.choice(eval_user_list, size=max_eval_users, replace=False)))

    recalls_5 = []
    recalls_10 = []
    precisions_5 = []
    precisions_10 = []
    ndcgs_10 = []
    diversities = []
    all_recs = []
    total_test_interactions = 0

    for user in eval_user_list:
        actual_items = test_ground_truth[user]
        total_test_interactions += len(actual_items)

        # Generate top 10 recommendations, excluding items already purchased in training
        recs = model.recommend(customer_id=user, top_n=10, exclude_purchased=True)
        rec_codes = [r.stock_code for r in recs]
        all_recs.append(rec_codes)

        recalls_5.append(recall_at_k(rec_codes, actual_items, k=5))
        recalls_10.append(recall_at_k(rec_codes, actual_items, k=10))
        precisions_5.append(precision_at_k(rec_codes, actual_items, k=5))
        precisions_10.append(precision_at_k(rec_codes, actual_items, k=10))
        ndcgs_10.append(ndcg_at_k(rec_codes, actual_items, k=10))
        diversities.append(intra_list_diversity(rec_codes, similarity_lookup))

    coverage = catalog_coverage(all_recs, catalog_items)

    return ModelEvaluationResult(
        model_name=model.name,
        recall_at_5=round(float(np.mean(recalls_5)), 4),
        recall_at_10=round(float(np.mean(recalls_10)), 4),
        precision_at_5=round(float(np.mean(precisions_5)), 4),
        precision_at_10=round(float(np.mean(precisions_10)), 4),
        ndcg_at_10=round(float(np.mean(ndcgs_10)), 4),
        catalog_coverage=round(coverage, 4),
        recommendation_diversity=round(float(np.mean(diversities)), 4),
        evaluated_users=len(eval_user_list),
        test_interactions_count=total_test_interactions,
    )


def run_full_benchmark(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    catalog_df: Optional[pd.DataFrame] = None,
    max_eval_users: int = 150,
) -> Tuple[pd.DataFrame, Dict[str, ModelEvaluationResult]]:
    """
    Executes a complete offline evaluation comparing:
    1. Popularity Baseline
    2. Item-Item Collaborative Filtering
    3. User-User Collaborative Filtering
    4. Content-Based Filtering
    5. Hybrid Recommender
    """
    logger.info("Starting Full Offline Recommendation Benchmark...")

    train_users = set(train_df[COL_CUSTOMER_ID].dropna().unique())
    test_users = set(test_df[COL_CUSTOMER_ID].dropna().unique())
    eligible_users = train_users & test_users
    catalog_items = set(train_df[COL_STOCK_CODE].dropna().unique())

    # Build similarity lookup for diversity calculation using Item-Item CF
    logger.info("Pre-fitting models for evaluation...")
    pop_model = PopularityRecommender(ranking_metric="customers").fit(train_df, catalog_df)
    item_cf_model = ItemItemCollaborativeRecommender().fit(train_df, catalog_df)
    user_cf_model = UserUserCollaborativeRecommender().fit(train_df, catalog_df)
    content_model = ContentBasedRecommender().fit(train_df, catalog_df)
    hybrid_model = HybridRecommender().fit(train_df, catalog_df)

    # Build pairwise similarity dictionary from Item-Item CF for diversity metric
    similarity_lookup: Dict[Tuple[str, str], float] = {}
    if item_cf_model.item_similarity_matrix is not None:
        items = list(item_cf_model.item_to_idx.keys())
        # Sample or limit to avoid explosive dict
        matrix = item_cf_model.item_similarity_matrix
        for i, code_a in enumerate(items):
            # Record non-zero similarities
            non_zeros = np.where(matrix[i] > 0.05)[0]
            for j in non_zeros:
                if i < j:
                    similarity_lookup[(code_a, items[j])] = float(matrix[i, j])

    models = [pop_model, item_cf_model, user_cf_model, content_model, hybrid_model]
    results: Dict[str, ModelEvaluationResult] = {}
    rows = []

    for m in models:
        logger.info(f"Evaluating {m.name}...")
        res = evaluate_recommender(
            model=m,
            train_df=train_df,
            test_df=test_df,
            eligible_users=eligible_users,
            catalog_items=catalog_items,
            similarity_lookup=similarity_lookup,
            max_eval_users=max_eval_users,
        )
        results[m.name] = res
        rows.append(res.to_dict())

    results_df = pd.DataFrame(rows)
    # Format readable column names
    results_df = results_df.rename(
        columns={
            "model_name": "Algorithm",
            "recall_at_5": "Recall@5",
            "recall_at_10": "Recall@10",
            "precision_at_5": "Precision@5",
            "precision_at_10": "Precision@10",
            "ndcg_at_10": "NDCG@10",
            "catalog_coverage": "Catalog Coverage",
            "recommendation_diversity": "Diversity",
            "evaluated_users": "Users Evaluated",
            "test_interactions_count": "Test Interactions",
        }
    )

    logger.info("Offline Benchmark Evaluation Complete.")
    return results_df, results
