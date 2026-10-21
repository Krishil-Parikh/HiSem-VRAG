# ablation_query_generator.py
"""
Query Generator for Ablation Studies

Generates 1600+ queries from all available videos for comprehensive evaluation.
- Iterates through all videos in video_memory/
- Generates multiple queries per video from topic annotations
- Uses LLM-based query cleaning for realistic test data
- Caches queries to JSON for reproducibility
"""

import os
import json
import random
from typing import List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import from existing modules
from phase_5_query_pipeline_openrouter import (
    MEMORY_ROOT,
    call_openrouter,
)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "ablation_results")
os.makedirs(CACHE_DIR, exist_ok=True)

QUERIES_CACHE_PATH = os.path.join(CACHE_DIR, "evaluation_queries.json")

# Query generation settings
QUERIES_PER_VIDEO = 4  # Target: 460 videos × 4 = 1840 queries
MIN_TOPIC_WORDS = 5  # Minimum words in topic text
MIN_QUERY_LENGTH = 15  # Minimum characters in cleaned query
LLM_WORKERS = 8  # Parallel LLM calls for query cleaning


# ══════════════════════════════════════════════════════════════════════════════
# LLM-BASED QUERY CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def generate_natural_query(transcript_snippet: str) -> str:
    """
    Use LLM to generate a natural, specific human-like question 
    based on a transcript snippet.
    """
    try:
        system_prompt = (
            "You are a home cook looking for specific moments in cooking videos. "
            "Given a transcript snippet, write ONE natural, conversational question that a user "
            "would actually type into a search bar to find this exact moment. "
            "\nRules:\n"
            "1. Be specific (mention ingredients/tools from the snippet).\n"
            "2. Be concise (12-20 words).\n"
            "3. Use varied starts: 'How do I...', 'What's the best way to...', "
            "'Can you show me...', 'When do I add...', 'How long should I...'.\n"
            "4. Output ONLY the question text. No preamble."
        )
        
        prompt = f'Transcript snippet: "{transcript_snippet}"\nNatural Question:'
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = call_openrouter(
            messages=messages,
            model="meta-llama/llama-3.1-8b-instruct",
            temperature=0.7,
            max_tokens=60
        )
        
        cleaned = response.strip().strip('"').strip("'")
        
        # Chatter detection
        chatter_keywords = [
            "however", "i don't see", "input text", "rewritten", "here is",
            "please provide", "as an ai", "unfortunately", "i'd be happy"
        ]
        low_cleaned = cleaned.lower()
        if any(kw in low_cleaned for kw in chatter_keywords):
            return ""
        
        # Validate it looks like a question
        if len(cleaned) > 10 and (
            "?" in cleaned or 
            cleaned.lower().startswith(("how", "what", "where", "can", "show", "when", "why"))
        ):
            return cleaned
        
        return ""
        
    except Exception as e:
        print(f"[WARN] Query generation failed: {e}")
        return ""


def generate_query_from_topic(topic: Dict, video_id: str) -> Dict:
    """
    Generate a single query from a topic annotation.
    Returns query dict with ground truth, or None if cleaning fails.
    """
    text = topic.get("text", "")
    
    # Skip short or noisy topics
    if not text or len(text.split()) < MIN_TOPIC_WORDS:
        return None
    
    # Quality filter: check alphanumeric density
    alnum_count = sum(1 for c in text if c.isalnum())
    if alnum_count / (len(text) + 1) < 0.5:
        return None
    
    # Clean with LLM
    query = generate_natural_query(text)
    
    if not query or len(query) < MIN_QUERY_LENGTH:
        return None
    
    return {
        "query": query,
        "ground_truth": {
            "video_id": video_id,
            "topic_id": topic.get("topic_id"),
            "start_time": topic.get("start_time", 0),
            "end_time": topic.get("end_time", 0),
            "original_text": text[:200]
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# BATCH QUERY GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_queries_from_video(video_id: str, max_queries: int = QUERIES_PER_VIDEO) -> List[Dict]:
    """
    Generate queries from a single video's topic annotations.
    """
    topics_path = os.path.join(MEMORY_ROOT, video_id, "topics.json")
    if not os.path.exists(topics_path):
        return []
    
    try:
        with open(topics_path, 'r', encoding='utf-8') as f:
            topics = json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load topics for {video_id}: {e}")
        return []
    
    # Filter valid topics
    valid_topics = [t for t in topics if t.get("text") and len(t["text"].split()) >= MIN_TOPIC_WORDS]
    
    if not valid_topics:
        return []
    
    # Shuffle and take subset
    random.shuffle(valid_topics)
    selected_topics = valid_topics[:max_queries]
    
    queries = []
    seen_queries = set()
    
    for topic in selected_topics:
        result = generate_query_from_topic(topic, video_id)
        
        if result and result["query"] not in seen_queries:
            seen_queries.add(result["query"])
            queries.append(result)
    
    return queries


def generate_all_queries(
    max_videos: int = None,
    queries_per_video: int = QUERIES_PER_VIDEO,
    use_cache: bool = True,
    parallel: bool = True
) -> List[Dict]:
    """
    Generate queries from all available videos.
    
    Args:
        max_videos: Limit number of videos (None = all)
        queries_per_video: Max queries per video
        use_cache: If True, load from cache if exists
        parallel: If True, use parallel processing
    
    Returns:
        List of query dicts with ground truth
    """
    # Check cache
    if use_cache and os.path.exists(QUERIES_CACHE_PATH):
        print(f"[INFO] Loading queries from cache: {QUERIES_CACHE_PATH}")
        with open(QUERIES_CACHE_PATH, 'r', encoding='utf-8') as f:
            cached = json.load(f)
        print(f"[INFO] Loaded {len(cached['queries'])} cached queries")
        return cached["queries"]
    
    # Get all video directories
    video_dirs = [
        d for d in os.listdir(MEMORY_ROOT)
        if os.path.isdir(os.path.join(MEMORY_ROOT, d))
        and os.path.exists(os.path.join(MEMORY_ROOT, d, "topics.json"))
    ]
    
    print(f"[INFO] Found {len(video_dirs)} videos with topics")
    
    if max_videos:
        random.shuffle(video_dirs)
        video_dirs = video_dirs[:max_videos]
        print(f"[INFO] Limited to {len(video_dirs)} videos")
    
    all_queries = []
    
    if parallel:
        # Parallel query generation
        print(f"[INFO] Generating queries in parallel ({LLM_WORKERS} workers)...")
        
        with ThreadPoolExecutor(max_workers=LLM_WORKERS) as executor:
            futures = {
                executor.submit(generate_queries_from_video, vid_id, queries_per_video): vid_id
                for vid_id in video_dirs
            }
            
            completed = 0
            for future in as_completed(futures):
                vid_id = futures[future]
                try:
                    queries = future.result()
                    all_queries.extend(queries)
                except Exception as e:
                    print(f"[ERROR] Failed for {vid_id}: {e}")
                
                completed += 1
                if completed % 50 == 0:
                    print(f"  Progress: {completed}/{len(video_dirs)} videos, {len(all_queries)} queries")
    else:
        # Sequential generation
        for i, vid_id in enumerate(video_dirs):
            queries = generate_queries_from_video(vid_id, queries_per_video)
            all_queries.extend(queries)
            
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(video_dirs)} videos, {len(all_queries)} queries")
    
    print(f"[INFO] Generated {len(all_queries)} total queries")
    
    # Save to cache
    cache_data = {
        "generated_at": datetime.now().isoformat(),
        "num_videos": len(video_dirs),
        "queries_per_video": queries_per_video,
        "total_queries": len(all_queries),
        "queries": all_queries
    }
    
    with open(QUERIES_CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Saved queries to: {QUERIES_CACHE_PATH}")
    
    return all_queries


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_queries(queries: List[Dict]) -> Dict[str, Any]:
    """
    Validate generated queries.
    """
    stats = {
        "total_queries": len(queries),
        "unique_videos": len(set(q["ground_truth"]["video_id"] for q in queries)),
        "avg_query_length": sum(len(q["query"]) for q in queries) / len(queries) if queries else 0,
        "min_query_length": min(len(q["query"]) for q in queries) if queries else 0,
        "max_query_length": max(len(q["query"]) for q in queries) if queries else 0,
    }
    
    # Check for duplicates
    unique_queries = set(q["query"] for q in queries)
    stats["duplicate_queries"] = len(queries) - len(unique_queries)
    
    # Check query patterns
    patterns = {
        "how": 0,
        "what": 0,
        "when": 0,
        "where": 0,
        "can": 0,
        "other": 0
    }
    
    for q in queries:
        query_lower = q["query"].lower()
        matched = False
        for pattern in ["how", "what", "when", "where", "can"]:
            if query_lower.startswith(pattern):
                patterns[pattern] += 1
                matched = True
                break
        if not matched:
            patterns["other"] += 1
    
    stats["query_patterns"] = patterns
    
    return stats


def print_validation_report(stats: Dict[str, Any]):
    """Print validation report."""
    print("\n" + "="*60)
    print("QUERY VALIDATION REPORT")
    print("="*60)
    
    print(f"\nSUMMARY:")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Unique videos: {stats['unique_videos']}")
    print(f"  Queries per video: {stats['total_queries'] / stats['unique_videos']:.1f}")
    
    print(f"\nQUERY LENGTH:")
    print(f"  Average: {stats['avg_query_length']:.1f} chars")
    print(f"  Min: {stats['min_query_length']} chars")
    print(f"  Max: {stats['max_query_length']} chars")
    
    print(f"\nQUERY PATTERNS:")
    for pattern, count in stats["query_patterns"].items():
        pct = count / stats["total_queries"] * 100 if stats["total_queries"] > 0 else 0
        print(f"  {pattern}: {count} ({pct:.1f}%)")
    
    print(f"\nDUPLICATES: {stats['duplicate_queries']}")
    
    print("="*60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def sample_queries(queries: List[Dict], n: int = 100, seed: int = 42) -> List[Dict]:
    """
    Sample n queries for quick testing.
    """
    random.seed(seed)
    if n >= len(queries):
        return queries
    return random.sample(queries, n)


def stratified_sample(queries: List[Dict], n: int = 100, seed: int = 42) -> List[Dict]:
    """
    Stratified sampling: ensure queries from different videos.
    """
    random.seed(seed)
    
    # Group by video
    by_video = {}
    for q in queries:
        vid = q["ground_truth"]["video_id"]
        if vid not in by_video:
            by_video[vid] = []
        by_video[vid].append(q)
    
    # Sample proportionally
    sampled = []
    videos = list(by_video.keys())
    random.shuffle(videos)
    
    queries_per_video = max(1, n // len(videos))
    
    for vid in videos:
        video_queries = by_video[vid]
        take = min(queries_per_video, len(video_queries))
        sampled.extend(random.sample(video_queries, take))
        
        if len(sampled) >= n:
            break
    
    return sampled[:n]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate evaluation queries for ablation studies")
    parser.add_argument("--validate", action="store_true", help="Only validate cached queries")
    parser.add_argument("--regenerate", action="store_true", help="Force regeneration (ignore cache)")
    parser.add_argument("--max-videos", type=int, default=None, help="Limit number of videos")
    parser.add_argument("--queries-per-video", type=int, default=QUERIES_PER_VIDEO, help="Queries per video")
    parser.add_argument("--sample", type=int, default=None, help="Sample N queries for testing")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("ABLATION STUDY QUERY GENERATOR")
    print("="*70 + "\n")
    
    if args.validate:
        # Just validate existing cache
        if not os.path.exists(QUERIES_CACHE_PATH):
            print("[ERROR] No cached queries found. Run without --validate first.")
            exit(1)
        
        with open(QUERIES_CACHE_PATH, 'r', encoding='utf-8') as f:
            cached = json.load(f)
        
        queries = cached["queries"]
        stats = validate_queries(queries)
        print_validation_report(stats)
        
        # Show sample queries
        print("\nSAMPLE QUERIES:")
        print("-" * 60)
        for q in queries[:10]:
            print(f"  Q: {q['query']}")
            print(f"     → Video: {q['ground_truth']['video_id']}, "
                  f"Time: {q['ground_truth']['start_time']:.1f}-{q['ground_truth']['end_time']:.1f}s")
            print()
    
    else:
        # Generate queries
        queries = generate_all_queries(
            max_videos=args.max_videos,
            queries_per_video=args.queries_per_video,
            use_cache=not args.regenerate,
            parallel=True
        )
        
        # Validate
        stats = validate_queries(queries)
        print_validation_report(stats)
        
        # Sample if requested
        if args.sample:
            sampled = stratified_sample(queries, n=args.sample)
            sample_path = os.path.join(CACHE_DIR, f"sample_{args.sample}_queries.json")
            with open(sample_path, 'w', encoding='utf-8') as f:
                json.dump({"queries": sampled}, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Saved {len(sampled)} sample queries to: {sample_path}")
