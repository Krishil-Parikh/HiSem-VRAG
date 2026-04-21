"""
Video Popularity Module - Simulated Likes & Subscribers for Dataset Videos

Provides simulated engagement metrics (likes, dislikes, subscribers, views)
for the 20-video YouCook2 mini dataset and a weighted re-ranking function
that blends semantic similarity scores with popularity signals.

This module is ADDITIVE only — it does NOT modify any existing retrieval
or re-ranking logic.  It applies a lightweight popularity boost AFTER
all other scoring is finalised.
"""

import json
import random
import math
from pathlib import Path
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Paths (same root as pipeline.py)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
IMPLEMENTATION_DIR = BASE_DIR.parent / "Implementation" / "mini_dataset_youcook2"
IMPLEMENTATION_MEMORY = IMPLEMENTATION_DIR / "video_memory"

# Persistent file – generated once, then loaded from disk on subsequent runs
POPULARITY_DATA_PATH = BASE_DIR / "data" / "video_popularity.json"


# ---------------------------------------------------------------------------
# Simulated popularity data for the curated 20-video dataset
# ---------------------------------------------------------------------------
# Each video gets a deterministic but realistic set of engagement numbers.
# We seed the RNG with the video_id hash so values are stable across restarts.
# ---------------------------------------------------------------------------

def _simulate_one_video(video_id: str) -> Dict[str, Any]:
    """Generate realistic engagement stats for a single video."""
    seed = abs(hash(video_id)) % (2**31)
    rng = random.Random(seed)

    # Subscribers of the channel (10K – 2M range, log-distributed)
    subscribers = int(10 ** rng.uniform(4.0, 6.3))

    # Views (typically 2x-50x subscriber count, with long tail)
    views = int(subscribers * rng.uniform(2, 50))

    # Like ratio (85-99 %)
    like_ratio = rng.uniform(0.85, 0.99)
    total_reactions = int(views * rng.uniform(0.02, 0.12))
    likes = int(total_reactions * like_ratio)
    dislikes = total_reactions - likes

    return {
        "video_id": video_id,
        "likes": likes,
        "dislikes": dislikes,
        "subscribers": subscribers,
        "views": views,
        "like_ratio": round(like_ratio, 4),
        "engagement_rate": round(total_reactions / max(views, 1), 4),
    }


def _generate_all_popularity() -> Dict[str, Dict[str, Any]]:
    """Discover all dataset videos and simulate popularity for each."""
    video_ids: List[str] = []

    if IMPLEMENTATION_MEMORY.exists():
        for entry in IMPLEMENTATION_MEMORY.iterdir():
            if entry.is_dir() and (entry / "video_memory.json").exists():
                video_ids.append(entry.name)

    video_ids.sort()
    popularity: Dict[str, Dict[str, Any]] = {}
    for vid in video_ids:
        popularity[vid] = _simulate_one_video(vid)

    return popularity


def load_popularity(force_regenerate: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Load (or generate + persist) the popularity data.

    Returns a dict keyed by video_id with engagement stats.
    """
    if not force_regenerate and POPULARITY_DATA_PATH.exists():
        with open(POPULARITY_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    popularity = _generate_all_popularity()

    # Persist so values are stable
    POPULARITY_DATA_PATH.parent.mkdir(exist_ok=True, parents=True)
    with open(POPULARITY_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(popularity, f, indent=2)
    print(f"[Popularity] Generated engagement data for {len(popularity)} videos -> {POPULARITY_DATA_PATH}")

    return popularity


# ---------------------------------------------------------------------------
# Normalised popularity score  (0 → 1)
# ---------------------------------------------------------------------------

def _compute_popularity_score(stats: Dict[str, Any]) -> float:
    """
    Combine likes, subscribers, views, and engagement into a single
    normalised popularity score in [0, 1].

    Formula (weighted blend):
        0.35 * norm_likes  +  0.25 * norm_subscribers  +
        0.20 * norm_views  +  0.20 * norm_engagement

    Each component is log-scaled and then min-max normalised across the
    full dataset so comparisons are fair.
    """
    # This will be called after global normalisation, so individual
    # raw values are already turned into 0-1 floats stored in _norm_cache.
    return stats.get("_normalised_score", 0.5)


_norm_cache: Optional[Dict[str, float]] = None


def _build_normalised_scores(popularity: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """Build normalised popularity scores for all videos."""
    global _norm_cache
    if _norm_cache is not None:
        return _norm_cache

    if not popularity:
        _norm_cache = {}
        return _norm_cache

    # Collect raw values
    all_likes = []
    all_subs = []
    all_views = []
    all_eng = []

    for stats in popularity.values():
        all_likes.append(math.log1p(stats["likes"]))
        all_subs.append(math.log1p(stats["subscribers"]))
        all_views.append(math.log1p(stats["views"]))
        all_eng.append(stats["engagement_rate"])

    def _minmax(vals):
        lo, hi = min(vals), max(vals)
        span = hi - lo if hi != lo else 1.0
        return [(v - lo) / span for v in vals]

    n_likes = _minmax(all_likes)
    n_subs = _minmax(all_subs)
    n_views = _minmax(all_views)
    n_eng = _minmax(all_eng)

    scores: Dict[str, float] = {}
    for i, vid in enumerate(popularity.keys()):
        score = (
            0.35 * n_likes[i]
            + 0.25 * n_subs[i]
            + 0.20 * n_views[i]
            + 0.20 * n_eng[i]
        )
        scores[vid] = round(score, 6)

    _norm_cache = scores
    return _norm_cache


def get_popularity_score(video_id: str, popularity: Dict[str, Dict[str, Any]]) -> float:
    """Return the normalised popularity score for a video (0-1)."""
    scores = _build_normalised_scores(popularity)
    return scores.get(video_id, 0.5)  # default mid-range for unknown videos


# ---------------------------------------------------------------------------
# Popularity-weighted re-ranking
# ---------------------------------------------------------------------------

# Weight given to popularity in the final blend (rest goes to similarity)
POPULARITY_WEIGHT = 0.10  #  10 % popularity, 90 % similarity


def rerank_with_popularity(
    results: List[Dict[str, Any]],
    popularity: Dict[str, Dict[str, Any]],
    weight: float = POPULARITY_WEIGHT,
    score_key: str = "score",
) -> List[Dict[str, Any]]:
    """
    Apply a lightweight popularity re-ranking on top of already-scored results.

    final_score = (1 - weight) * similarity_score  +  weight * popularity_score

    The results list is re-sorted in-place by the new ``final_score`` and the
    original similarity score is preserved in ``original_score``.

    Args:
        results:    list of evidence / display-scene dicts (must have *score_key*)
        popularity: full popularity dict from load_popularity()
        weight:     blend weight for popularity (default 0.10)
        score_key:  key in each dict that holds the current similarity score

    Returns:
        The same list, re-sorted, with ``popularity_score`` and ``final_score``
        fields injected into each dict.
    """
    if not results or not popularity:
        return results

    for item in results:
        vid = item.get("video_id", "")
        sim_score = item.get(score_key, 0.0)
        pop_score = get_popularity_score(vid, popularity)

        item["original_score"] = sim_score
        item["popularity_score"] = pop_score
        item[score_key] = round((1 - weight) * sim_score + weight * pop_score, 6)

    results.sort(key=lambda x: x.get(score_key, 0), reverse=True)
    return results


def get_video_popularity(video_id: str) -> Optional[Dict[str, Any]]:
    """Convenience: return the full popularity dict for one video."""
    popularity = load_popularity()
    return popularity.get(video_id)
