# ablation_pipeline.py
"""
Configurable Query Pipeline for Ablation Studies

This module provides a configurable version of the query pipeline that supports
all ablation variants. It does NOT modify any existing files.

Key Features:
- Hierarchical vs Flat retrieval toggle
- Query decomposition toggle
- Semantic vs Heuristic scene ranking
- Semantic vs Position-based keyframe selection
- Temporal smoothing toggle
- Configurable thresholds
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import time
from scipy.ndimage import gaussian_filter1d

# Import from existing pipeline (read-only)
from phase_5_query_pipeline_openrouter import (
    TEXT_EMBEDDER,
    load_global_index,
    retrieve_global_topics_faiss,
    load_video_data,
    get_siglip_text_emb,
    safe_cos_sim,
    decompose_query,
    parallel_vlm_calls,
    call_openrouter,
    ANSWER_MODEL,
    MEMORY_ROOT,
    BASE_DIR,
)

from ablation_config import (
    AblationConfig,
    SceneRankingMode,
    KeyframeMode,
    get_config,
)


# ══════════════════════════════════════════════════════════════════════════════
# FLAT RETRIEVAL (Ablation 1)
# ══════════════════════════════════════════════════════════════════════════════

_flat_scene_index = None
_flat_scene_metadata = None


def build_flat_scene_index():
    """
    Build a flat FAISS index over ALL scenes across all videos.
    This is used when hierarchical retrieval is disabled.
    
    Returns cached index if already built.
    """
    global _flat_scene_index, _flat_scene_metadata
    
    if _flat_scene_index is not None:
        return _flat_scene_index, _flat_scene_metadata
    
    import faiss
    
    print("[ABLATION] Building flat scene index...")
    
    all_embeddings = []
    all_metadata = []
    
    # Iterate through all videos
    video_dirs = [d for d in os.listdir(MEMORY_ROOT)
                  if os.path.isdir(os.path.join(MEMORY_ROOT, d))]
    
    for vid_id in video_dirs:
        vid_dir = os.path.join(MEMORY_ROOT, vid_id)
        
        # Load scenes
        scenes_path = os.path.join(vid_dir, "scenes.json")
        if not os.path.exists(scenes_path):
            continue
        
        with open(scenes_path, 'r', encoding='utf-8') as f:
            scenes_data = json.load(f)
        
        # Load frame embeddings
        emb_path = os.path.join(vid_dir, "frame_embeddings.npy")
        if not os.path.exists(emb_path):
            continue
        
        frame_emb = np.load(emb_path)
        
        # Process each topic's scenes
        for topic_data in scenes_data:
            scenes = topic_data.get("scenes", [])
            
            for scene in scenes:
                frame_ids = scene.get("frame_ids", [])
                if not frame_ids:
                    continue
                
                # Compute scene embedding as mean of frame embeddings
                valid_ids = [fid for fid in frame_ids if fid < len(frame_emb)]
                if not valid_ids:
                    continue
                
                scene_emb = np.mean(frame_emb[valid_ids], axis=0)
                
                all_embeddings.append(scene_emb)
                all_metadata.append({
                    "video_id": vid_id,
                    "topic_id": topic_data.get("topic_id"),
                    "scene_id": scene.get("scene_id"),
                    "start_time": scene.get("start_time", 0),
                    "end_time": scene.get("end_time", 0),
                    "frame_count": scene.get("frame_count", 0),
                    "frame_paths": scene.get("frame_paths", []),
                    "frame_ids": frame_ids,
                })
    
    if not all_embeddings:
        print("[WARN] No scenes found for flat index!")
        return None, None
    
    # Build FAISS index
    embeddings = np.array(all_embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    
    dim = embeddings.shape[1]
    _flat_scene_index = faiss.IndexFlatIP(dim)
    _flat_scene_index.add(embeddings)
    _flat_scene_metadata = all_metadata
    
    print(f"[ABLATION] Flat scene index built: {len(all_metadata)} scenes")
    
    return _flat_scene_index, _flat_scene_metadata


def retrieve_scenes_flat(query_emb: np.ndarray, top_k: int = 24) -> List[Dict]:
    """
    Flat retrieval: Get top-K scenes directly from all videos.
    No hierarchical topic filtering.
    """
    import faiss
    
    index, metadata = build_flat_scene_index()
    
    if index is None:
        return []
    
    # Normalize query
    query_vec = query_emb.astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(query_vec)
    
    # Search
    scores, indices = index.search(query_vec, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        
        meta = metadata[idx]
        results.append({
            "video_id": meta["video_id"],
            "topic_id": meta["topic_id"],
            "scene": {
                "scene_id": meta["scene_id"],
                "start_time": meta["start_time"],
                "end_time": meta["end_time"],
                "frame_count": meta["frame_count"],
                "frame_paths": meta["frame_paths"],
                "frame_ids": meta["frame_ids"],
            },
            "score": float(score),
            "start_time": meta["start_time"],
            "end_time": meta["end_time"],
        })
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SCENE RANKING VARIANTS (Ablation 3)
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_scenes_semantic(
    topic: Dict, 
    video_data: Dict, 
    query_emb: np.ndarray,
    topic_frame_offset: int,
    config: AblationConfig
) -> List[Dict]:
    """
    Semantic scene retrieval using cosine similarity.
    """
    scenes = topic.get("scenes", [])
    if not scenes:
        scenes_data = video_data.get("scenes", [])
        topic_id = topic.get("topic_id")
        for s in scenes_data:
            if s.get("topic_id") == topic_id:
                scenes = s.get("scenes", [])
                break
    
    if not scenes:
        return []
    
    frame_emb = video_data.get("frame_emb")
    scored = []
    
    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    
    for i, scene in enumerate(scenes):
        frame_ids = scene.get("frame_ids", [])
        scene_scores = []
        
        for fid in frame_ids:
            actual_idx = topic_frame_offset + fid
            if actual_idx < len(frame_emb):
                f_vec = frame_emb[actual_idx]
                scene_scores.append(float(np.dot(f_vec, q_norm.flatten())))
        
        if scene_scores:
            max_s = max(scene_scores)
            mean_s = float(np.mean(sorted(scene_scores, reverse=True)[:3]))
            score = (0.7 * max_s + 0.3 * mean_s) + (min(0.05, len(scene_scores) * 0.002))
        else:
            score = 0.0
        
        if score >= config.min_scene_score:
            scored.append({"scene": scene, "score": score, "rank": i + 1})
    
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:config.top_scenes_per_topic]


def retrieve_scenes_heuristic(
    topic: Dict, 
    video_data: Dict,
    config: AblationConfig
) -> List[Dict]:
    """
    Heuristic scene retrieval based on position and frame count.
    No embedding similarity computation.
    """
    scenes = topic.get("scenes", [])
    if not scenes:
        scenes_data = video_data.get("scenes", [])
        topic_id = topic.get("topic_id")
        for s in scenes_data:
            if s.get("topic_id") == topic_id:
                scenes = s.get("scenes", [])
                break
    
    if not scenes:
        return []
    
    scored = []
    
    for i, scene in enumerate(scenes):
        # Heuristic score: position bonus + frame count bonus
        position_score = 1.0 - (i * 0.05)  # Earlier scenes preferred
        frame_count = scene.get("frame_count", 0)
        frame_score = min(1.0, frame_count / 20.0)  # More frames = more content
        
        score = (position_score + frame_score) / 2.0
        
        if score >= config.min_scene_score:
            scored.append({"scene": scene, "score": score, "rank": i + 1})
    
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:config.top_scenes_per_topic]


# ══════════════════════════════════════════════════════════════════════════════
# KEYFRAME SELECTION VARIANTS (Ablation 4 & 5)
# ══════════════════════════════════════════════════════════════════════════════

def select_keyframes_position(scene: Dict, config: AblationConfig) -> List[Tuple[str, float]]:
    """
    Position-based keyframe selection: First, Middle, Last.
    """
    frame_paths = scene.get("frame_paths", [])
    if not frame_paths:
        return []
    
    n = len(frame_paths)
    if n >= 3:
        indices = [0, n // 2, n - 1][:config.top_frames_per_scene]
    else:
        indices = list(range(min(n, config.top_frames_per_scene)))
    
    return [(frame_paths[i], 0.5) for i in indices]


def select_keyframes_semantic(
    scene: Dict,
    query_emb: np.ndarray,
    video_data: Dict,
    scene_frame_offset: int,
    config: AblationConfig,
    apply_smoothing: bool = True
) -> List[Tuple[str, float]]:
    """
    Semantic keyframe selection with optional temporal smoothing.
    """
    frame_paths = scene.get("frame_paths", [])
    frame_ids = scene.get("frame_ids", [])
    
    if not frame_paths:
        return []
    
    frame_emb = video_data.get("frame_emb")
    
    if frame_emb is None or len(frame_ids) == 0:
        return select_keyframes_position(scene, config)
    
    # Get embeddings for frames in this scene
    scene_embeddings = []
    valid_frame_paths = []
    
    for fid, fp in zip(frame_ids, frame_paths):
        actual_idx = scene_frame_offset + fid
        if actual_idx < len(frame_emb):
            scene_embeddings.append(frame_emb[actual_idx])
            valid_frame_paths.append(fp)
    
    if not scene_embeddings:
        return select_keyframes_position(scene, config)
    
    scene_embeddings = np.array(scene_embeddings, dtype=np.float32)
    
    # Compute cos(q, vi) for each frame
    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    
    try:
        if scene_embeddings.shape[1] != q_norm.shape[0]:
            raw_scores = np.zeros(len(scene_embeddings))
        else:
            raw_scores = np.dot(scene_embeddings, q_norm.flatten())
    except Exception:
        raw_scores = np.zeros(len(scene_embeddings))
    
    # Apply temporal smoothing if enabled
    if apply_smoothing and config.temporal_smoothing_sigma > 0 and len(raw_scores) >= 3:
        smoothed_scores = gaussian_filter1d(raw_scores, sigma=config.temporal_smoothing_sigma)
    else:
        smoothed_scores = raw_scores
    
    # Sort and return top-K
    frame_score_pairs = list(zip(valid_frame_paths, smoothed_scores.tolist()))
    frame_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    return frame_score_pairs[:config.top_frames_per_scene]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ABLATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_ablation_pipeline(
    query: str, 
    config: AblationConfig,
    skip_vlm: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run the query pipeline with the specified ablation configuration.
    
    Args:
        query: User query string
        config: AblationConfig specifying which features to enable/disable
        skip_vlm: If True, skip VLM calls (for evaluation speed)
        verbose: If True, print detailed progress
    
    Returns:
        Dict with retrieved topics, scenes, and metrics
    """
    start_time = time.time()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"ABLATION PIPELINE: {config.name}")
        print(f"{'='*60}")
        print(f"Query: {query}")
    
    # ═══════════════════════════════════════════════════════════
    # STEP 1: Query Decomposition (Ablation 2)
    # ═══════════════════════════════════════════════════════════
    if config.use_decomposition:
        sub_questions = decompose_query(query)
        if verbose:
            print(f"Sub-questions: {sub_questions}")
    else:
        sub_questions = [query]
        if verbose:
            print("Decomposition DISABLED")
    
    # ═══════════════════════════════════════════════════════════
    # STEP 2: Retrieval (Ablation 1: Hierarchical vs Flat)
    # ═══════════════════════════════════════════════════════════
    
    # Get SigLIP embedding for scene/frame scoring
    siglip_q_emb = get_siglip_text_emb(query)
    
    all_topics = []
    all_scenes = []
    vlm_call_count = 0
    
    if config.use_hierarchical:
        # ─────────────────────────────────────────────────────
        # HIERARCHICAL RETRIEVAL (Baseline)
        # ─────────────────────────────────────────────────────
        if verbose:
            print("Retrieval: HIERARCHICAL")
        
        # Collect topics from all sub-queries
        all_retrieved_topics = {}
        
        for sq in sub_questions:
            sq_emb = TEXT_EMBEDDER.encode(sq)
            retrieved = retrieve_global_topics_faiss(sq_emb, top_k=config.top_k_global)
            
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
        
        all_topics = global_topics
        
        # Filter by threshold and group by video
        valid_topics = [t for t in global_topics if t["score"] >= config.min_topic_score]
        
        video_topics = {}
        for t in valid_topics:
            vid = t["video_id"]
            if vid not in video_topics:
                video_topics[vid] = []
            if len(video_topics[vid]) < config.top_topics_per_video:
                video_topics[vid].append(t)
        
        top_video_ids = list(video_topics.keys())[:config.top_videos]
        
        # ─────────────────────────────────────────────────────
        # Scene Retrieval (within topics)
        # ─────────────────────────────────────────────────────
        for vid_id in top_video_ids:
            video_data = load_video_data(vid_id)
            if not video_data:
                continue
            
            memory_data = video_data.get("memory", {})
            topics_data = memory_data.get("topics", [])
            
            # Track frame offsets
            topic_frame_offsets = {}
            current_offset = 0
            manifest = video_data.get("manifest", [])
            
            for m_topic in manifest:
                m_topic_id = m_topic["topic_id"]
                topic_frame_offsets[m_topic_id] = current_offset
                current_offset += m_topic.get("frame_count", 0)
            
            retrieved_ids = [t["topic_id"] for t in video_topics[vid_id]]
            
            for topic in topics_data:
                if topic["topic_id"] not in retrieved_ids:
                    continue
                
                topic_id = topic["topic_id"]
                topic_offset = topic_frame_offsets.get(topic_id, 0)
                
                # Scene retrieval (Ablation 3)
                if config.scene_ranking_mode == SceneRankingMode.SEMANTIC:
                    scored_scenes = retrieve_scenes_semantic(
                        topic, video_data, siglip_q_emb, topic_offset, config
                    )
                else:
                    scored_scenes = retrieve_scenes_heuristic(topic, video_data, config)
                
                # Process scenes
                for s_item in scored_scenes:
                    scene = s_item["scene"]
                    
                    all_scenes.append({
                        "video_id": vid_id,
                        "topic_id": topic_id,
                        "scene": scene,
                        "score": s_item["score"],
                        "start_time": scene.get("start_time", 0),
                        "end_time": scene.get("end_time", 0),
                    })
                    
                    # Keyframe selection (Ablation 4 & 5)
                    if config.keyframe_mode == KeyframeMode.POSITION:
                        keyframes = select_keyframes_position(scene, config)
                    elif config.keyframe_mode == KeyframeMode.SEMANTIC:
                        keyframes = select_keyframes_semantic(
                            scene, siglip_q_emb, video_data, topic_offset, config,
                            apply_smoothing=False
                        )
                    else:  # SEMANTIC_SMOOTHED
                        keyframes = select_keyframes_semantic(
                            scene, siglip_q_emb, video_data, topic_offset, config,
                            apply_smoothing=True
                        )
                    
                    vlm_call_count += len(keyframes)
    
    else:
        # ─────────────────────────────────────────────────────
        # FLAT RETRIEVAL (Ablation 1)
        # ─────────────────────────────────────────────────────
        if verbose:
            print("Retrieval: FLAT (no hierarchical)")
        
        # Direct scene retrieval without topic filtering
        flat_scenes = retrieve_scenes_flat(siglip_q_emb, top_k=24)
        
        # Create pseudo-topics from retrieved scenes
        video_topic_set = set()
        for s in flat_scenes:
            video_topic_set.add((s["video_id"], s["topic_id"]))
        
        # Build topic list for metrics
        for vid_id, topic_id in video_topic_set:
            all_topics.append({
                "video_id": vid_id,
                "topic_id": topic_id,
                "score": 0.5,  # Placeholder score
                "start_time": 0,
                "end_time": 0,
            })
        
        all_scenes = flat_scenes
        vlm_call_count = len(flat_scenes) * config.top_frames_per_scene
    
    # ═══════════════════════════════════════════════════════════
    # STEP 3: Compute Metrics
    # ═══════════════════════════════════════════════════════════
    latency = time.time() - start_time
    
    # Sort scenes by score for consistent Recall@K
    all_scenes.sort(key=lambda x: x["score"], reverse=True)
    
    # Get unique videos in retrieval order
    seen_videos = set()
    retrieved_videos = []
    for t in all_topics:
        vid = t["video_id"]
        if vid not in seen_videos:
            seen_videos.add(vid)
            retrieved_videos.append(vid)
    
    result = {
        "query": query,
        "config_name": config.name,
        "sub_questions": sub_questions,
        "retrieved_topics": all_topics,
        "retrieved_scenes": all_scenes,
        "retrieved_videos": retrieved_videos,
        "vlm_calls": vlm_call_count,
        "latency_sec": latency,
    }
    
    if verbose:
        print(f"\nResults:")
        print(f"  Topics: {len(all_topics)}")
        print(f"  Scenes: {len(all_scenes)}")
        print(f"  VLM calls: {vlm_call_count}")
        print(f"  Latency: {latency:.2f}s")
    
    return result


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION HELPER
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_ablation_query(
    query: str,
    ground_truth: Dict,
    config: AblationConfig
) -> Dict[str, Any]:
    """
    Run ablation pipeline and compute all metrics for a single query.
    
    Returns dict with all recall metrics, latency, etc.
    """
    # Run pipeline
    result = run_ablation_pipeline(query, config, skip_vlm=True, verbose=False)
    
    gt_video = ground_truth.get("video_id", "")
    gt_start = ground_truth.get("start_time", 0)
    gt_end = ground_truth.get("end_time", 0)
    
    # ─────────────────────────────────────────────────────────
    # Video Recall@K
    # ─────────────────────────────────────────────────────────
    retrieved_videos = result["retrieved_videos"]
    r1_video = 1 if gt_video in retrieved_videos[:1] else 0
    r3_video = 1 if gt_video in retrieved_videos[:3] else 0
    
    # ─────────────────────────────────────────────────────────
    # Topic Recall@K (time-overlap based)
    # ─────────────────────────────────────────────────────────
    def is_relevant(r_start, r_end, g_start, g_end, threshold=0.3):
        intersection = max(0, min(r_end, g_end) - max(r_start, g_start))
        min_duration = min(r_end - r_start, g_end - g_start)
        if min_duration <= 0:
            return False
        return (intersection / min_duration) >= threshold
    
    topics = result["retrieved_topics"]
    r1_topics = 0
    r3_topics = 0
    r5_topics = 0
    
    for k, rk in [(1, "r1"), (3, "r3"), (5, "r5")]:
        for t in topics[:k]:
            if t["video_id"] == gt_video:
                t_start = t.get("start_time", 0)
                t_end = t.get("end_time", 0)
                if is_relevant(t_start, t_end, gt_start, gt_end):
                    if k == 1:
                        r1_topics = 1
                    elif k == 3:
                        r3_topics = 1
                    else:
                        r5_topics = 1
                    break
    
    # ─────────────────────────────────────────────────────────
    # Scene Recall@K (within GT video only)
    # ─────────────────────────────────────────────────────────
    scenes = result["retrieved_scenes"]
    gt_scenes = [s for s in scenes if s.get("video_id") == gt_video]
    gt_scenes.sort(key=lambda x: x["score"], reverse=True)
    
    r3_scenes = 0
    r5_scenes = 0
    r10_scenes = 0
    
    for k, rk in [(3, "r3"), (5, "r5"), (10, "r10")]:
        for s in gt_scenes[:k]:
            scene = s.get("scene", s)
            s_start = scene.get("start_time", s.get("start_time", 0))
            s_end = scene.get("end_time", s.get("end_time", 0))
            if is_relevant(s_start, s_end, gt_start, gt_end):
                if k == 3:
                    r3_scenes = 1
                elif k == 5:
                    r5_scenes = 1
                else:
                    r10_scenes = 1
                break
    
    # Oracle scene recall
    oracle_scene = 0
    for s in gt_scenes:
        scene = s.get("scene", s)
        s_start = scene.get("start_time", s.get("start_time", 0))
        s_end = scene.get("end_time", s.get("end_time", 0))
        if is_relevant(s_start, s_end, gt_start, gt_end):
            oracle_scene = 1
            break
    
    return {
        "query": query,
        "config_name": config.name,
        "gt_video_id": gt_video,
        "gt_start": gt_start,
        "gt_end": gt_end,
        "latency_sec": result["latency_sec"],
        "topics_retrieved": len(topics),
        "scenes_retrieved": len(scenes),
        "vlm_calls": result["vlm_calls"],
        "top_topic_score": topics[0]["score"] if topics else 0.0,
        "recall_at_1_video": r1_video,
        "recall_at_3_video": r3_video,
        "recall_at_1_topics": r1_topics,
        "recall_at_3_topics": r3_topics,
        "recall_at_5_topics": r5_topics,
        "recall_at_3_scenes": r3_scenes,
        "recall_at_5_scenes": r5_scenes,
        "recall_at_10_scenes": r10_scenes,
        "oracle_scene_recall": oracle_scene,
    }


if __name__ == "__main__":
    # Quick test
    from ablation_config import ABLATION_CONFIGS
    
    print("\n" + "="*70)
    print("ABLATION PIPELINE TEST")
    print("="*70)
    
    # Load index
    load_global_index()
    
    test_query = "How do I prepare the dressing for a salad?"
    
    for name, config in list(ABLATION_CONFIGS.items())[:3]:
        print(f"\n{'─'*50}")
        print(f"Testing: {name}")
        print(f"{'─'*50}")
        
        result = run_ablation_pipeline(test_query, config, verbose=True)
