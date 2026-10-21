# phase_6_evaluation.py
"""
PHASE 6: Evaluation Module with Proper Recall@K

Evaluates the Video-RAG pipeline with:
- Recall@K for Topics (K=1,3,5) - time overlap based
- Recall@K for Scenes (K=3,5,10) - time overlap based  
- Recall@K for Videos (K=1,3) - video-level retrieval
- QA accuracy (LLM-as-judge)
- VLM calls per query
- Latency metrics
- Oracle upper bounds

Outputs results to CSV for analysis.
"""
import os
import json
import csv
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field
import requests
import time

# Import the query pipeline
from phase_5_query_pipeline_openrouter import (
    TEXT_EMBEDDER,
    call_openrouter,
    ANSWER_MODEL,
    load_global_index,
    retrieve_global_topics_faiss,
    load_video_data,
    retrieve_scenes,
    select_keyframes_with_embeddings,
    parallel_vlm_calls,
    decompose_query,
    get_siglip_text_emb,
    load_siglip,
    MIN_TOPIC_SCORE,
    TOP_VIDEOS,
    TOP_TOPICS_PER_VIDEO,
)

# ====================== CONFIG ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_ROOT = os.path.join(BASE_DIR, "video_memory")
EVAL_OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation_results")
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

# Recall@K settings
TOPIC_K_VALUES = [1, 3, 5]
SCENE_K_VALUES = [3, 5, 10]
VIDEO_K_VALUES = [1, 3]
OVERLAP_THRESHOLD = 0.3  # 30% overlap required for relevance


# ====================== TIME OVERLAP FUNCTIONS ======================
def time_overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    """Check if two time ranges overlap at all."""
    return not (a_end < b_start or a_start > b_end)


def compute_overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """
    Compute overlap ratio using Overlap Coefficient logic.
    Intersection / min(duration_a, duration_b).
    This handles segments of different scales (scenes vs topics).
    """
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
    duration_a = a_end - a_start
    duration_b = b_end - b_start
    min_duration = min(duration_a, duration_b)
    
    if min_duration <= 0:
        return 0.0
    
    return intersection / min_duration


def is_relevant(retrieved_start: float, retrieved_end: float, 
                gt_start: float, gt_end: float, 
                threshold: float = OVERLAP_THRESHOLD) -> bool:
    """Check if retrieved segment is relevant to ground truth based on overlap ratio."""
    ratio = compute_overlap_ratio(retrieved_start, retrieved_end, gt_start, gt_end)
    return ratio >= threshold


# ====================== RECALL@K COMPUTATION ======================
def recall_at_k_topics(retrieved_topics: List[Dict], gt_start: float, gt_end: float, k: int) -> int:
    """
    Compute Recall@K for topics.
    Returns 1 if any of top-K topics overlaps with ground truth, 0 otherwise.
    """
    topk = retrieved_topics[:k]
    for t in topk:
        t_start = t.get("start_time", 0)
        t_end = t.get("end_time", 0)
        if is_relevant(t_start, t_end, gt_start, gt_end):
            return 1
    return 0


def recall_at_k_scenes(retrieved_scenes: List[Dict], gt_start: float, gt_end: float, k: int) -> int:
    """
    Compute Recall@K for scenes.
    Returns 1 if any of top-K scenes overlaps with ground truth, 0 otherwise.
    """
    topk = retrieved_scenes[:k]
    for s in topk:
        # Handle both nested and flat scene structures
        scene = s.get("scene", s)
        s_start = scene.get("start_time", 0)
        s_end = scene.get("end_time", 0)
        if is_relevant(s_start, s_end, gt_start, gt_end):
            return 1
    return 0


# ====================== QUERY CLEANING (OPTIONAL) ======================
USE_LLM_QUERY_CLEANING = True  # Set to True to enable LLM-based query cleaning


def generate_natural_query(transcript_snippet: str) -> str:
    """
    Use LLM to generate a natural, specific human-like question based on a transcript snippet.
    This replaces robotic 'summaries' with varied, realistic 'how-to' queries.
    """
    if not USE_LLM_QUERY_CLEANING:
        return transcript_snippet
    
    try:
        system_prompt = (
            "You are a home cook looking for specific moments in cooking videos. "
            "Given a transcript snippet, write ONE natural, conversational question that a user "
            "would actually type into a search bar to find this exact moment. "
            "\nRules:\n"
            "1. Be specific (mention ingredients/tools from the snippet).\n"
            "2. Be concise (6-12 words).\n"
            "3. Use varied starts: 'How do I...', 'What's the best way to...', 'Can you show me...', 'When do I add...'.\n"
            "4. Output ONLY the question text. No preamble."
        )
        
        prompt = f"Transcript snippet: \"{transcript_snippet}\"\nNatural Question:"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = call_openrouter(
            messages=messages,
            model="meta-llama/llama-3.1-8b-instruct",
            temperature=0.7,  # Higher temperature for variety
            max_tokens=60
        )
        
        cleaned = response.strip().strip('"').strip("'")
        
        # Chatter Detection
        chatter_keywords = ["however", "i don't see", "input text", "rewritten", "here is", 
                           "please provide", "as an ai", "unfortunately"]
        low_cleaned = cleaned.lower()
        if any(kw in low_cleaned for kw in chatter_keywords):
            return ""
        
        if len(cleaned) > 10 and ("?" in cleaned or cleaned.lower().startswith(("how", "what", "where", "can", "show"))):
            return cleaned
        
        return ""
        
    except Exception as e:
        print(f"[WARN] Natural query generation failed: {e}")
        return ""


# ====================== PSEUDO-QUERY GENERATION ======================
def generate_pseudo_queries_from_video(video_id: str, max_queries: int = 5) -> List[Dict]:
    """
    Generate pseudo-queries from YouCook2 step annotations.
    Each step becomes a query with its time range as ground truth.
    """
    topics_path = os.path.join(MEMORY_ROOT, video_id, "topics.json")
    if not os.path.exists(topics_path):
        return []
    
    with open(topics_path, 'r', encoding='utf-8') as f:
        topics = json.load(f)
    
    queries = []
    seen_queries = set()
    
    # Shuffle topics to get variety if we have more than max_queries
    import random
    valid_topics = [t for t in topics if t.get("text") and len(t["text"].split()) > 5]
    random.shuffle(valid_topics)
    
    for topic in valid_topics:
        if len(queries) >= max_queries:
            break
        
        text = topic.get("text", "")
        
        # Quality Filter: check alphanumeric density to avoid music/noise segments
        alnum_count = sum(1 for c in text if c.isalnum())
        if alnum_count / (len(text) + 1) < 0.5:
            continue
        
        # Clean with LLM
        query = generate_natural_query(text)
        
        # Skip if cleaning failed or produced noise
        if not query or query in seen_queries or len(query) < 15:
            continue
        
        seen_queries.add(query)
        
        queries.append({
            "query": query,
            "ground_truth": {
                "video_id": video_id,
                "topic_id": topic.get("topic_id"),
                "start_time": topic.get("start_time", 0),
                "end_time": topic.get("end_time", 0),
                "original_text": text[:200]
            }
        })
    
    return queries


def generate_evaluation_dataset(num_videos: int = 10, queries_per_video: int = 2) -> List[Dict]:
    """Generate evaluation dataset from multiple videos."""
    all_queries = []
    
    video_dirs = [d for d in os.listdir(MEMORY_ROOT) 
                  if os.path.isdir(os.path.join(MEMORY_ROOT, d))
                  and os.path.exists(os.path.join(MEMORY_ROOT, d, "topics.json"))]
    
    # Shuffle and take subset
    import random
    random.shuffle(video_dirs)
    
    for vid_id in video_dirs[:num_videos]:
        queries = generate_pseudo_queries_from_video(vid_id, queries_per_video)
        all_queries.extend(queries)
    
    return all_queries


# ====================== QUERY METRICS DATACLASS ======================
@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query_id: int
    query: str
    timestamp: str
    
    # Ground truth
    gt_video_id: str = ""
    gt_start_time: float = 0.0
    gt_end_time: float = 0.0
    
    # Latency
    total_latency_sec: float = 0.0
    
    # Retrieval counts
    topics_retrieved: int = 0
    scenes_retrieved: int = 0
    vlm_calls: int = 0
    
    # Scores
    top_topic_score: float = 0.0
    
    # Recall@K for Videos (new!)
    recall_at_1_video: int = 0
    recall_at_3_video: int = 0
    
    # Recall@K for Topics
    recall_at_1_topics: int = 0
    recall_at_3_topics: int = 0
    recall_at_5_topics: int = 0
    
    # Recall@K for Scenes (within GT video only)
    recall_at_3_scenes: int = 0
    recall_at_5_scenes: int = 0
    recall_at_10_scenes: int = 0
    
    # Oracle upper bound (max achievable scene recall if topic is correct)
    oracle_scene_recall: int = 0
    
    # Answer
    answer: str = ""


# ====================== EVALUATION LOGGER ======================
class EvaluationLogger:
    """Logs evaluation metrics to CSV."""
    
    def __init__(self, run_name: str = None):
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(EVAL_OUTPUT_DIR, f"eval_{self.run_name}.csv")
        self.metrics: List[QueryMetrics] = []
        self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV with headers."""
        headers = [
            "query_id", "query", "timestamp",
            "gt_video_id", "gt_start_time", "gt_end_time",
            "total_latency_sec",
            "topics_retrieved", "scenes_retrieved", "vlm_calls",
            "top_topic_score",
            "recall@1_video", "recall@3_video",  # Video-level recall
            "recall@1_topics", "recall@3_topics", "recall@5_topics",
            "recall@3_scenes", "recall@5_scenes", "recall@10_scenes",
            "oracle_scene_recall",
            "answer_preview"
        ]
        
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        
        print(f"[EVAL] CSV initialized: {self.csv_path}")
    
    def log(self, metrics: QueryMetrics):
        """Log a single query's metrics."""
        self.metrics.append(metrics)
        
        row = [
            metrics.query_id,
            metrics.query[:80],
            metrics.timestamp,
            metrics.gt_video_id,
            f"{metrics.gt_start_time:.1f}",
            f"{metrics.gt_end_time:.1f}",
            f"{metrics.total_latency_sec:.2f}",
            metrics.topics_retrieved,
            metrics.scenes_retrieved,
            metrics.vlm_calls,
            f"{metrics.top_topic_score:.4f}",
            metrics.recall_at_1_video,
            metrics.recall_at_3_video,
            metrics.recall_at_1_topics,
            metrics.recall_at_3_topics,
            metrics.recall_at_5_topics,
            metrics.recall_at_3_scenes,
            metrics.recall_at_5_scenes,
            metrics.recall_at_10_scenes,
            metrics.oracle_scene_recall,
            metrics.answer[:100] if metrics.answer else ""
        ]
        
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def get_summary(self) -> Dict[str, float]:
        """Compute summary statistics including Recall@K."""
        if not self.metrics:
            return {}
        
        n = len(self.metrics)
        
        summary = {
            "total_queries": n,
            
            # Latency
            "avg_latency_sec": np.mean([m.total_latency_sec for m in self.metrics]),
            "p50_latency_sec": np.percentile([m.total_latency_sec for m in self.metrics], 50),
            "p95_latency_sec": np.percentile([m.total_latency_sec for m in self.metrics], 95),
            
            # Retrieval counts
            "avg_topics_retrieved": np.mean([m.topics_retrieved for m in self.metrics]),
            "avg_scenes_retrieved": np.mean([m.scenes_retrieved for m in self.metrics]),
            "avg_vlm_calls": np.mean([m.vlm_calls for m in self.metrics]),
            
            # Scores
            "avg_top_topic_score": np.mean([m.top_topic_score for m in self.metrics]),
            
            # ═══════════════════════════════════════════════
            # RECALL@K METRICS (the important ones!)
            # ═══════════════════════════════════════════════
            # Video-level recall (new!)
            "recall@1_video": np.mean([m.recall_at_1_video for m in self.metrics]),
            "recall@3_video": np.mean([m.recall_at_3_video for m in self.metrics]),
            
            # Topic-level recall
            "recall@1_topics": np.mean([m.recall_at_1_topics for m in self.metrics]),
            "recall@3_topics": np.mean([m.recall_at_3_topics for m in self.metrics]),
            "recall@5_topics": np.mean([m.recall_at_5_topics for m in self.metrics]),
            
            # Scene-level recall (within GT video only)
            "recall@3_scenes": np.mean([m.recall_at_3_scenes for m in self.metrics]),
            "recall@5_scenes": np.mean([m.recall_at_5_scenes for m in self.metrics]),
            "recall@10_scenes": np.mean([m.recall_at_10_scenes for m in self.metrics]),
            
            # Oracle upper bound
            "oracle_scene_recall": np.mean([m.oracle_scene_recall for m in self.metrics]),
        }
        
        return summary
    
    def print_summary(self):
        """Print summary to console."""
        summary = self.get_summary()
        
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY ({len(self.metrics)} queries)")
        print(f"{'='*60}")
        
        print(f"\nLATENCY:")
        print(f"  Average: {summary.get('avg_latency_sec', 0):.2f}s")
        print(f"  P50: {summary.get('p50_latency_sec', 0):.2f}s")
        print(f"  P95: {summary.get('p95_latency_sec', 0):.2f}s")
        
        print(f"\nRETRIEVAL COUNTS:")
        print(f"  Avg topics: {summary.get('avg_topics_retrieved', 0):.1f}")
        print(f"  Avg scenes: {summary.get('avg_scenes_retrieved', 0):.1f}")
        print(f"  Avg VLM calls: {summary.get('avg_vlm_calls', 0):.1f}")
        
        print(f"\nRECALL@K VIDEO (is GT video in top-K?):")
        print(f"  Recall@1: {summary.get('recall@1_video', 0):.1%}")
        print(f"  Recall@3: {summary.get('recall@3_video', 0):.1%}")
        
        print(f"\nRECALL@K TOPICS (time-overlap based):")
        print(f"  Recall@1: {summary.get('recall@1_topics', 0):.1%}")
        print(f"  Recall@3: {summary.get('recall@3_topics', 0):.1%}")
        print(f"  Recall@5: {summary.get('recall@5_topics', 0):.1%}")
        
        print(f"\nRECALL@K SCENES (within GT video only):")
        print(f"  Recall@3: {summary.get('recall@3_scenes', 0):.1%}")
        print(f"  Recall@5: {summary.get('recall@5_scenes', 0):.1%}")
        print(f"  Recall@10: {summary.get('recall@10_scenes', 0):.1%}")
        
        print(f"\nORACLE UPPER BOUND:")
        print(f"  Max scene recall (if topic correct): {summary.get('oracle_scene_recall', 0):.1%}")
        
        print(f"\nResults saved to: {self.csv_path}")
        print(f"{'='*60}\n")
        
        return summary


# ====================== CORE EVALUATION FUNCTION ======================
def evaluate_single_query(query: str, ground_truth: Dict, query_id: int = 0) -> QueryMetrics:
    """
    Run query through pipeline and compute all metrics including Recall@K.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_time = time.time()
    
    gt_video = ground_truth.get("video_id", "")
    gt_start = ground_truth.get("start_time", 0)
    gt_end = ground_truth.get("end_time", 0)
    
    # Embed query
    q_emb = TEXT_EMBEDDER.encode(query)
    
    # ===== 1. Retrieval (Main Query + Subquestions) =====
    sub_questions = decompose_query(query)
    print(f"  [PIPELINE] Sub-questions: {sub_questions}")
    
    # Collect all topics from all sub-queries
    all_retrieved_topics = {}  # key: (video_id, topic_id), value: topic dict
    
    # Retrieval tasks: main query + subquestions
    retrieval_tasks = [("Main Query", query)] + [("Subquestion", sq) for sq in sub_questions]
    
    for label, q_text in retrieval_tasks:
        emb = TEXT_EMBEDDER.encode(q_text)
        retrieved = retrieve_global_topics_faiss(emb)
        
        for t in retrieved:
            key = (t["video_id"], t["topic_id"])
            if key not in all_retrieved_topics or t["score"] > all_retrieved_topics[key]["score"]:
                all_retrieved_topics[key] = t
    
    # Merge and sort
    global_topics = list(all_retrieved_topics.values())
    global_topics.sort(key=lambda x: x["score"], reverse=True)
    
    # Re-rank
    for i, t in enumerate(global_topics):
        t["rank"] = i + 1
    
    # ===== 2. Vision Embeddings for Keyframe selection =====
    # Load SigLIP embedding for the query
    siglip_q_emb = get_siglip_text_emb(query)
    
    # Collect all retrieved scenes
    all_scenes = []
    
    # Filter topics and get scenes
    valid_topics = [t for t in global_topics if t["score"] >= MIN_TOPIC_SCORE]
    
    video_topics = {}
    for t in valid_topics:
        vid = t["video_id"]
        if vid not in video_topics:
            video_topics[vid] = []
        if len(video_topics[vid]) < TOP_TOPICS_PER_VIDEO:
            video_topics[vid].append(t)
    
    top_video_ids = list(video_topics.keys())[:TOP_VIDEOS]
    
    vlm_calls = 0
    
    for vid_id in top_video_ids:
        video_data = load_video_data(vid_id)
        if not video_data:
            continue
        
        # Load unified memory to get hierarchical structure
        memory_data = video_data.get("memory", {})
        topics_data = memory_data.get("topics", [])
        
        # Track frame offset across topics
        topic_frame_offsets = {}
        current_offset = 0
        manifest = video_data.get("manifest", [])
        
        for m_topic in manifest:
            m_topic_id = m_topic["topic_id"]
            topic_frame_offsets[m_topic_id] = current_offset
            current_offset += m_topic.get("frame_count", 0)
        
        # Filter topics for this video that were retrieved
        retrieved_ids = [t["topic_id"] for t in video_topics[vid_id]]
        
        for topic in topics_data:
            if topic["topic_id"] not in retrieved_ids:
                continue
            
            topic_id = topic["topic_id"]
            topic_offset = topic_frame_offsets.get(topic_id, 0)
            
            scored_scenes = retrieve_scenes(topic, video_data, query_emb=siglip_q_emb, topic_frame_offset=topic_offset)
            
            # Store scenes with full time info
            for s_item in scored_scenes:
                scene = s_item["scene"]
                all_scenes.append({
                    "video_id": vid_id,
                    "scene": scene,
                    "score": s_item["score"],
                    "start_time": scene.get("start_time", 0),
                    "end_time": scene.get("end_time", 0)
                })
                
                # Count VLM calls using semantic selection
                keyframes = select_keyframes_with_embeddings(
                    scene, siglip_q_emb, video_data, scene_frame_offset=topic_offset
                )
                vlm_calls += len(keyframes)
    
    total_latency = time.time() - start_time
    
    # ═══════════════════════════════════════════════
    # COMPUTE RECALL@K
    # ═══════════════════════════════════════════════
    
    # --- Video-level Recall@K (new!) ---
    # Check if GT video is in top-K retrieved videos
    retrieved_video_ids = [t["video_id"] for t in global_topics]
    
    # Get unique videos in order of first appearance
    seen = set()
    unique_videos = []
    for vid in retrieved_video_ids:
        if vid not in seen:
            seen.add(vid)
            unique_videos.append(vid)
    
    r1_video = 1 if gt_video in unique_videos[:1] else 0
    r3_video = 1 if gt_video in unique_videos[:3] else 0
    
    # --- Recall@K for Topics ---
    r1_topics = recall_at_k_topics(global_topics, gt_start, gt_end, k=1)
    r3_topics = recall_at_k_topics(global_topics, gt_start, gt_end, k=3)
    r5_topics = recall_at_k_topics(global_topics, gt_start, gt_end, k=5)
    
    # --- Recall@K for Scenes (WITHIN GT VIDEO ONLY - CRITICAL FIX!) ---
    # Filter scenes to only those from the ground-truth video
    gt_scenes = [s for s in all_scenes if s["video_id"] == gt_video]
    
    # GLOBAL SORT: Sort ground truth scenes by score descending before Recall@K
    gt_scenes.sort(key=lambda x: x["score"], reverse=True)
    
    r3_scenes = recall_at_k_scenes(gt_scenes, gt_start, gt_end, k=3)
    r5_scenes = recall_at_k_scenes(gt_scenes, gt_start, gt_end, k=5)
    r10_scenes = recall_at_k_scenes(gt_scenes, gt_start, gt_end, k=10)
    
    # --- Oracle upper bound ---
    # What's the max achievable scene recall if we had perfect topic retrieval?
    oracle_scene = recall_at_k_scenes(gt_scenes, gt_start, gt_end, k=len(gt_scenes)) if gt_scenes else 0
    
    # Top topic score
    top_score = global_topics[0]["score"] if global_topics else 0.0
    
    return QueryMetrics(
        query_id=query_id,
        query=query,
        timestamp=timestamp,
        gt_video_id=gt_video,
        gt_start_time=gt_start,
        gt_end_time=gt_end,
        total_latency_sec=total_latency,
        topics_retrieved=len(global_topics),
        scenes_retrieved=len(all_scenes),
        vlm_calls=vlm_calls,
        top_topic_score=top_score,
        recall_at_1_video=r1_video,
        recall_at_3_video=r3_video,
        recall_at_1_topics=r1_topics,
        recall_at_3_topics=r3_topics,
        recall_at_5_topics=r5_topics,
        recall_at_3_scenes=r3_scenes,
        recall_at_5_scenes=r5_scenes,
        recall_at_10_scenes=r10_scenes,
        oracle_scene_recall=oracle_scene,
        answer=""  # Skip answer generation for speed in batch eval
    )


def run_evaluation_batch(queries: List[Dict], run_name: str = None) -> EvaluationLogger:
    """
    Run evaluation on a batch of queries with ground truth.
    
    queries: List of {"query": str, "ground_truth": {"video_id": str, "start_time": float, "end_time": float}}
    """
    logger = EvaluationLogger(run_name)
    
    print(f"\n{'='*60}")
    print(f"STARTING EVALUATION: {len(queries)} queries")
    print(f"{'='*60}\n")
    
    # Load index once
    load_global_index()
    
    for i, q in enumerate(queries):
        query = q.get("query", "")
        ground_truth = q.get("ground_truth", {})
        
        print(f"[{i+1}/{len(queries)}] {query[:50]}...")
        print(f"    GT: video={ground_truth.get('video_id', '?')}, time={ground_truth.get('start_time', 0):.1f}-{ground_truth.get('end_time', 0):.1f}s")
        
        metrics = evaluate_single_query(query, ground_truth, i)
        logger.log(metrics)
        
        print(f"    [OK] {metrics.total_latency_sec:.1f}s | R@1={metrics.recall_at_1_topics} R@3={metrics.recall_at_3_topics} R@5={metrics.recall_at_5_topics}")
    
    # Print summary
    summary = logger.print_summary()
    
    # Save summary JSON
    summary_path = os.path.join(EVAL_OUTPUT_DIR, f"summary_{logger.run_name}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    
    return logger


# ====================== MAIN ======================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("PHASE 6: EVALUATION WITH RECALL@K")
    print("="*70)
    
    # Generate pseudo-queries from YouCook2 videos
    print("\nGenerating pseudo-queries from video annotations...")
    queries = generate_evaluation_dataset(num_videos=10, queries_per_video=2)
    
    print(f"Generated {len(queries)} queries")
    for i, q in enumerate(queries[:5]):
        print(f"  {i+1}. {q['query'][:60]}...")
    if len(queries) > 5:
        print(f"  ... and {len(queries)-5} more")
    
    # Run evaluation
    run_evaluation_batch(queries, run_name="recall_eval")
