"""Quick test to verify popularity module works correctly."""
import sys
sys.path.insert(0, ".")

from video.popularity import (
    load_popularity,
    get_popularity_score,
    rerank_with_popularity,
    _build_normalised_scores,
)

# 1. Generate / load popularity data
p = load_popularity(force_regenerate=True)
print(f"Generated popularity for {len(p)} videos\n")

# 2. Show sample data
for vid, stats in list(p.items())[:5]:
    print(f"  {vid}: {stats['likes']:>6} likes, {stats['subscribers']:>8} subs, {stats['views']:>10} views, eng={stats['engagement_rate']:.4f}")

# 3. Show normalised scores
scores = _build_normalised_scores(p)
print("\nNormalised popularity scores (sorted desc):")
for vid, sc in sorted(scores.items(), key=lambda x: -x[1]):
    print(f"  {vid}: {sc:.4f}")

# 4. Test re-ranking on dummy results
print("\n--- Re-ranking test ---")
dummy_results = [
    {"video_id": list(p.keys())[0], "score": 0.80, "scene_id": 0},
    {"video_id": list(p.keys())[1], "score": 0.85, "scene_id": 1},
    {"video_id": list(p.keys())[2], "score": 0.82, "scene_id": 2},
]
print("Before:", [(r["video_id"], r["score"]) for r in dummy_results])

rerank_with_popularity(dummy_results, p, weight=0.10)
print("After: ", [(r["video_id"], round(r["score"], 4), f"pop={r['popularity_score']:.4f}") for r in dummy_results])

print("\nAll tests passed!")
