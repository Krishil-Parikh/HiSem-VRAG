# phase_5_query_pipeline_openrouter.py
"""
OPTIMIZED Query Pipeline with:
1. FAISS-based global topic index for O(log N) cross-video retrieval
2. Hard limits on topics/scenes/frames
3. Early stopping by confidence threshold
4. Parallel VLM calls
5. VLM caching
"""
import os
import json
import requests
import numpy as np
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import torch
from PIL import Image
from typing import List, Dict, Any, Tuple
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import faiss
from scipy.ndimage import gaussian_filter1d  # For temporal smoothing
# ====================== CONFIG ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_ROOT = os.path.join(BASE_DIR, "video_memory")
INDEX_PATH = os.path.join(MEMORY_ROOT, "global_topic_index.faiss")
METADATA_PATH = os.path.join(MEMORY_ROOT, "global_topic_metadata.json")
# ────────────────────────────────────────────────
#   OPENROUTER API KEYS  (read from .env / environment, never hard-coded)
# ────────────────────────────────────────────────
import itertools
import threading
import base64
import time
import requests

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass  # python-dotenv is optional; plain environment variables still work

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def _load_openrouter_keys() -> List[str]:
    # OPENROUTER_API_KEYS (comma-separated, round-robin) or a single OPENROUTER_API_KEY
    raw = os.environ.get("OPENROUTER_API_KEYS") or os.environ.get("OPENROUTER_API_KEY", "")
    return [k.strip() for k in raw.split(",") if k.strip()]

OPENROUTER_KEYS = _load_openrouter_keys()
_or_cycle = itertools.cycle(OPENROUTER_KEYS) if OPENROUTER_KEYS else None
_or_lock = threading.Lock()

def get_next_openrouter_key() -> str:
    if _or_cycle is None:
        raise RuntimeError(
            "No OpenRouter API key found. Copy .env.example to .env and set OPENROUTER_API_KEY."
        )
    with _or_lock:
        return next(_or_cycle)

# Models (paper, Table "Models used in the VRAG pipeline"); override via .env if needed
DECOMPOSE_MODEL = os.environ.get("VRAG_LLM_MODEL", "meta-llama/llama-3.1-8b-instruct")
ANSWER_MODEL    = os.environ.get("VRAG_LLM_MODEL", "meta-llama/llama-3.1-8b-instruct")
VLM_MODEL       = os.environ.get("VRAG_VLM_MODEL", "qwen/qwen2.5-vl-72b-instruct")
# Embedders
TEXT_EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
VISION_MODEL_NAME = "google/siglip-so400m-patch14-384"  # Same as used in phase_3_2
_siglip_model = None
_siglip_processor = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ═══════════════════════════════════════════════════
# PIPELINE CONFIGURATION (Tunable)
# ═══════════════════════════════════════════════════
CONFIG = {
    "TOP_K_GLOBAL": 5,
    "TOP_VIDEOS": 3,
    "TOP_TOPICS_PER_VIDEO": 3,
    "TOP_SCENES_PER_TOPIC": 3,
    "TOP_FRAMES_PER_SCENE": 2,
    "MIN_TOPIC_SCORE": 0.7,           # tuned for the latency-recall trade-off (paper, Sec. 4.2)
    "MIN_SCENE_SCORE": 0.05,
    "TEMPORAL_SMOOTHING_SIGMA": 1.5,
    "VLM_WORKERS": 4
}
MIN_TOPIC_SCORE = CONFIG["MIN_TOPIC_SCORE"]
TOP_VIDEOS = CONFIG["TOP_VIDEOS"]
TOP_TOPICS_PER_VIDEO = CONFIG["TOP_TOPICS_PER_VIDEO"]
VLM_WORKERS = CONFIG["VLM_WORKERS"]
TOP_K_GLOBAL = CONFIG["TOP_K_GLOBAL"]
TOP_SCENES_PER_TOPIC = CONFIG["TOP_SCENES_PER_TOPIC"]
TOP_FRAMES_PER_SCENE = CONFIG["TOP_FRAMES_PER_SCENE"]
MIN_SCENE_SCORE = CONFIG["MIN_SCENE_SCORE"]
TEMPORAL_SMOOTHING_SIGMA = CONFIG["TEMPORAL_SMOOTHING_SIGMA"]

# Max VLM calls = 3 × 2 × 2 × 2 = 24 (optimized but allows more keyframes)
# ====================== GLOBAL INDEX LOADING ======================
_global_index = None
_global_metadata = None
def load_global_index():
    """Load the pre-built FAISS index and metadata."""
    global _global_index, _global_metadata
    
    if _global_index is not None:
        return _global_index, _global_metadata
    
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        print("[WARN] Global index not found. Run phase_5_0_build_global_index.py first!")
        return None, None
    
    _global_index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        _global_metadata = json.load(f)
    
    print(f"[INFO] Loaded global index with {_global_index.ntotal} topics")
    return _global_index, _global_metadata

def load_siglip():
    """Lazy load SigLIP model for text-to-image similarity."""
    global _siglip_model, _siglip_processor
    if _siglip_model is None:
        from transformers import AutoProcessor, AutoModel
        print(f"[INFO] Loading SigLIP model {VISION_MODEL_NAME} for keyframe selection...")
        _siglip_processor = AutoProcessor.from_pretrained(VISION_MODEL_NAME)
        _siglip_model = AutoModel.from_pretrained(VISION_MODEL_NAME).to(DEVICE)
        _siglip_model.eval()
    return _siglip_model, _siglip_processor

def get_siglip_text_emb(text: str) -> np.ndarray:
    """Embed text using SigLIP for comparison with frames."""
    model, processor = load_siglip()
    inputs = processor(text=[text], padding="max_length", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features.cpu().numpy().flatten()
# ====================== HELPER: SAFE COSINE SIMILARITY ======================
def safe_cos_sim(query_emb, target_emb):
    """Compute cosine similarity ensuring both tensors are float32."""
    if isinstance(query_emb, np.ndarray):
        q_t = torch.from_numpy(query_emb.astype(np.float32))
    else:
        q_t = query_emb.float()
    
    if q_t.dim() == 1:
        q_t = q_t.unsqueeze(0)
    
    if isinstance(target_emb, np.ndarray):
        t_t = torch.from_numpy(target_emb.astype(np.float32))
    else:
        t_t = target_emb.float()
    
    if t_t.dim() == 1:
        t_t = t_t.unsqueeze(0)
    
    return util.cos_sim(q_t, t_t)
# ====================== OPENROUTER LLM / VLM HELPERS ======================
def call_openrouter(messages: List[Dict], model: str, temperature: float = 0.4,
                    max_tokens: int = 800, retries: int = 3) -> str:
    """Chat-completion call to OpenRouter (OpenAI-compatible). Returns text, or "" on failure.

    `messages` may contain plain-text content or multimodal content (text + image_url parts).
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(retries):
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {get_next_openrouter_key()}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2 ** attempt)  # rate limited / transient server error
                continue
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            return (text or "").strip()
        except RuntimeError:
            raise  # missing API key: fail loudly instead of retrying
        except Exception as e:
            print(f"OpenRouter exception ({model}): {e}")
            time.sleep(2 ** attempt)
    return ""
def decompose_query(query: str) -> List[str]:
    """Decompose complex queries into sub-questions with the text LLM."""
    prompt = (
        "Decompose the following complex cooking query into up to 4 sequential, specific sub-queries. "
        "Return a numbered list (1., 2., ...), one sub-query per line.\n"
        f"Query: {query}"
    )
    messages = [{"role": "user", "content": prompt}]
    text = call_openrouter(messages, DECOMPOSE_MODEL, temperature=0.2, max_tokens=150)
    if not text:
        return [query]
    lines = [line.strip() for line in text.split("\n") if line.strip() and line.strip()[0].isdigit()]
    subs = [line.split(".", 1)[1].strip() if "." in line else line for line in lines]
    return subs[:4] or [query]
# ====================== VLM CACHE ======================
def get_vlm_cache_path(video_id: str, frame_path: str) -> str:
    vid_dir = os.path.join(MEMORY_ROOT, video_id, "vlm_cache")
    os.makedirs(vid_dir, exist_ok=True)
    frame_basename = os.path.basename(frame_path).replace(".jpg", ".json")
    return os.path.join(vid_dir, frame_basename)
def describe_frame_with_vlm(frame_path: str, context: str = "", video_id: str = None) -> Dict:
    """VLM call with caching. Returns dict with frame and description."""
    result = {"frame": os.path.basename(frame_path), "description": "", "cached": False}
    
    if not os.path.exists(frame_path):
        result["description"] = f"[frame not found]"
        return result
    
    # Check cache
    if video_id:
        cache_path = get_vlm_cache_path(video_id, frame_path)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                    result["description"] = cached.get("description", "")
                    result["cached"] = True
                    return result
            except:
                pass
    prompt = f"Describe this cooking video frame concisely. Focus on: ingredients, tools, actions, cooking stage. {context}"

    try:
        with open(frame_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }]
        desc = call_openrouter(messages, VLM_MODEL, temperature=0.3, max_tokens=200)
        if not desc:
            desc = "[VLM blocked or empty response]"
    except Exception as e:
        print(f"OpenRouter VLM error: {e}")
        desc = ""
        
    result["description"] = desc
    
    # Cache result
    if video_id and desc and not desc.startswith("["):
        cache_path = get_vlm_cache_path(video_id, frame_path)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({"description": desc}, f)
        except:
            pass
    
    return result
def parallel_vlm_calls(frame_tasks: List[Tuple[str, str, str]]) -> List[Dict]:
    """
    Execute VLM calls in parallel.
    frame_tasks: list of (frame_path, context, video_id)
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=CONFIG["VLM_WORKERS"]) as executor:
        futures = {
            executor.submit(describe_frame_with_vlm, fp, ctx, vid): (fp, ctx, vid)
            for fp, ctx, vid in frame_tasks
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                fp, ctx, vid = futures[future]
                results.append({"frame": os.path.basename(fp), "description": f"[error: {e}]", "cached": False})
    
    return results
# ====================== DATA LOADING ======================
def load_video_data(video_id: str) -> Dict:
    """Load video memory data."""
    vid_dir = os.path.join(MEMORY_ROOT, video_id)
    if not os.path.exists(vid_dir):
        return {}
    data = {"video_id": video_id, "vid_dir": vid_dir}
    
    # Load core project files
    for fname, key in [("topics.json", "topics"), ("scenes.json", "scenes"), ("video_memory.json", "memory")]:
        fpath = os.path.join(vid_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                data[key] = json.load(f)
                
    # Load manifest from the correct frames/ root (definitive source for embedding order)
    manifest_path = os.path.join(BASE_DIR, "frames", video_id, "frames_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            data["manifest"] = json.load(f)
    
    for fname, key in [("frame_embeddings.npy", "frame_emb"), ("transcript_embeddings.npy", "transcript_emb")]:
        fpath = os.path.join(vid_dir, fname)
        if os.path.exists(fpath):
            data[key] = np.load(fpath)
    
    return data
# ====================== FAISS CROSS-VIDEO RETRIEVAL ======================
def retrieve_global_topics_faiss(query_emb: np.ndarray, top_k: int = None) -> List[Dict]:
    """
    O(log N) cross-video retrieval using FAISS.
    Returns top-K topics from across all videos.
    """
    if top_k is None:
        top_k = CONFIG["TOP_K_GLOBAL"]
    index, metadata = load_global_index()
    
    if index is None:
        print("[WARN] FAISS index not loaded, falling back to slow retrieval")
        return []
    
    # Normalize query for cosine similarity
    query_vec = query_emb.astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(query_vec)
    
    # Search
    scores, indices = index.search(query_vec, top_k)
    
    results = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0:
            continue
        meta = metadata[idx]
        results.append({
            "rank": i + 1,
            "score": float(score),
            "video_id": meta["video_id"],
            "topic_id": meta["topic_id"],
            "start_time": meta["start_time"],
            "end_time": meta["end_time"],
            "text_preview": meta["text_preview"]
        })
    
    return results
# ====================== SCENE RETRIEVAL ======================
def retrieve_scenes(topic: Dict, video_data: Dict, query_emb: np.ndarray = None, topic_frame_offset: int = 0) -> List[Dict]:
    """
    Retrieve top scenes from a topic.
    If query_emb is provided, uses SEMANTIC SIMILARITY (from Flowchart 3 logic).
    Otherwise falls back to heuristic scoring.
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

    for i, scene in enumerate(scenes):
        score = 0.0
        
        if query_emb is not None and frame_emb is not None:
            # SEMANTIC SCORING: Find best matching frame in scene
            frame_ids = scene.get("frame_ids", [])
            scene_scores = []
            
            # Use normalized query
            q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
            
            for fid in frame_ids:
                actual_idx = topic_frame_offset + fid
                if actual_idx < len(frame_emb):
                    f_vec = frame_emb[actual_idx]
                    # f_vec is PRE-NORMALIZED on disk (Phase 3.2), don't re-normalize!
                    scene_scores.append(float(np.dot(f_vec, q_norm.flatten())))
            
            if scene_scores:
                # HYBRID SCORING: Prefers strong matches but considers scene consistency
                max_s = max(scene_scores)
                mean_s = float(np.mean(sorted(scene_scores, reverse=True)[:3]))
                score = (0.7 * max_s + 0.3 * mean_s) + (min(0.05, len(scene_scores) * 0.002))
            else:
                # Fallback to heuristic if no frames matched
                score = (1.0 - (i * 0.05) + min(1.0, scene.get("frame_count", 0) / 20.0)) / 2.0
        else:
            # ORIGINAL HEURISTIC FALLBACK
            position_score = 1.0 - (i * 0.05)
            frame_score = min(1.0, scene.get("frame_count", 0) / 20.0)
            score = (position_score + frame_score) / 2.0
        
        if score >= CONFIG["MIN_SCENE_SCORE"]:
            scored.append({"scene": scene, "score": score, "rank": i + 1})
    
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:CONFIG["TOP_SCENES_PER_TOPIC"]]
# ====================== KEYFRAME SELECTION (EMBEDDING-BASED) ======================
def select_keyframes_with_embeddings(
    scene: Dict, 
    query_emb: np.ndarray, 
    video_data: Dict,
    scene_frame_offset: int = 0
) -> List[Tuple[str, float]]:
    """
    Select keyframes using embedding similarity + temporal smoothing.
    
    Implements Flowchart 3:
    1. Get frame embeddings for scene
    2. Compute cos(q, vi) for each frame
    3. Apply temporal smoothing (Gaussian filter)
    4. Select top-K frames by smoothed score
    
    Returns: List of (frame_path, score) tuples
    """
    frame_paths = scene.get("frame_paths", [])
    frame_ids = scene.get("frame_ids", [])
    
    if not frame_paths:
        return []
    
    # Get frame embeddings from video data
    frame_emb = video_data.get("frame_emb")
    
    if frame_emb is None or len(frame_ids) == 0:
        # Fallback to position-based if no embeddings
        return [(fp, 0.5) for fp in frame_paths[:CONFIG["TOP_FRAMES_PER_SCENE"]]]
    
    # Get embeddings for frames in this scene
    # frame_ids are relative to the topic, need to adjust with offset
    scene_embeddings = []
    valid_frame_paths = []
    
    for i, (fid, fp) in enumerate(zip(frame_ids, frame_paths)):
        # Compute actual index in the video's frame_embeddings array
        actual_idx = scene_frame_offset + fid
        if actual_idx < len(frame_emb):
            scene_embeddings.append(frame_emb[actual_idx])
            valid_frame_paths.append(fp)
    
    if not scene_embeddings:
        return [(fp, 0.5) for fp in frame_paths[:CONFIG["TOP_FRAMES_PER_SCENE"]]]
    
    scene_embeddings = np.array(scene_embeddings, dtype=np.float32)
    
    # ═══════════════════════════════════════════════
    # STEP 1: Compute cos(q, vi) for each frame
    # ═══════════════════════════════════════════════
    # Normalize query embedding
    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    
    # scene_embeddings are PRE-NORMALIZED on disk (Phase 3.2), don't re-normalize!
    # scores = cos(q, vi) = query . embeddings
    try:
        # Check alignment
        if scene_embeddings.shape[1] != q_norm.shape[0]:
            print(f"  [WARN] Embedding mismatch: frame={scene_embeddings.shape[1]}, query={q_norm.shape[0]}. Falling back to scores=0.")
            raw_scores = np.zeros(len(scene_embeddings))
        else:
            raw_scores = np.dot(scene_embeddings, q_norm.flatten())
    except Exception as e:
        print(f"  [ERROR] np.dot failed: {e}")
        raw_scores = np.zeros(len(scene_embeddings))
    
    # ═══════════════════════════════════════════════
    # STEP 2: Temporal Smoothing (Gaussian filter)
    # ═══════════════════════════════════════════════
    if len(raw_scores) >= 3:
        smoothed_scores = gaussian_filter1d(raw_scores, sigma=CONFIG["TEMPORAL_SMOOTHING_SIGMA"])
    else:
        smoothed_scores = raw_scores
    
    # ═══════════════════════════════════════════════
    # STEP 3: Top-K Frame Selection by Score
    # ═══════════════════════════════════════════════
    frame_score_pairs = list(zip(valid_frame_paths, smoothed_scores.tolist()))
    frame_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-K frames with their scores
    return frame_score_pairs[:CONFIG["TOP_FRAMES_PER_SCENE"]]


def select_keyframes(scene: Dict) -> List[str]:
    """Legacy fallback: Select keyframes from a scene (position-based)."""
    frame_paths = scene.get("frame_paths", [])
    if not frame_paths:
        return []
    
    n = len(frame_paths)
    if n >= 3:
        # First, middle, last - but limit by TOP_FRAMES_PER_SCENE
        indices = [0, n // 2, n - 1][:CONFIG["TOP_FRAMES_PER_SCENE"]]
    else:
        indices = list(range(min(n, CONFIG["TOP_FRAMES_PER_SCENE"])))
    
    return [frame_paths[i] for i in indices]
# ====================== MAIN PIPELINE ======================
def run_query_pipeline(query: str):
    """
    FULL Query Pipeline (Flowchart-aligned):
    1. Query decomposition into subquestions
    2. Retrieve topics for EACH subquestion (merged)
    3. Scene retrieval with temporal relations
    4. Embedding-based keyframe selection with temporal smoothing
    5. Parallel VLM calls
    6. Evidence fusion with scene relations
    7. Answer generation
    """
    start_time = datetime.now()
    
    print(f"\n{'='*70}")
    print(f"OPTIMIZED QUERY PIPELINE – {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    print(f"QUESTION: {query}\n")
    
    # ===== 5.1 Query Decomposition =====
    print("╔══ 5.1 QUERY DECOMPOSITION ══╗")
    sub_questions = decompose_query(query)
    for i, sq in enumerate(sub_questions, 1):
        print(f"  {i}. {sq}")
    print()
    
    # ===== 5.1.1 SIGLIP EMBEDDING (for keyframe selection) =====
    # We need the query in SigLIP's latent space to compare with frame embeddings
    print("╔══ 5.1.1 SIGLIP EMBEDDING ══╗")
    siglip_q_emb = get_siglip_text_emb(query)
    print(f"  Query embedded in SigLIP space (dim={len(siglip_q_emb)})\n")
    
    # ===== 5.2 SUBQUESTION-BASED TOPIC RETRIEVAL =====
    print("╔══ 5.2 SUBQUESTION-BASED TOPIC RETRIEVAL ══╗")
    
    # Embed main query
    main_q_emb = TEXT_EMBEDDER.encode(query)
    
    # Collect topics from all subquestions + main query
    all_retrieved_topics = {}
    
    # First, retrieve for main query
    main_topics = retrieve_global_topics_faiss(main_q_emb)
    for t in main_topics:
        key = (t["video_id"], t["topic_id"])
        if key not in all_retrieved_topics:
            all_retrieved_topics[key] = t
        else:
            # Keep highest score
            if t["score"] > all_retrieved_topics[key]["score"]:
                all_retrieved_topics[key] = t
    
    print(f"  Main query: {len(main_topics)} topics")
    
    # Now retrieve for each subquestion
    subq_embeddings = []
    for i, sq in enumerate(sub_questions):
        sq_emb = TEXT_EMBEDDER.encode(sq)
        subq_embeddings.append(sq_emb)
        sq_topics = retrieve_global_topics_faiss(sq_emb, top_k=CONFIG["TOP_K_GLOBAL"])
        print(f"  Subquestion {i+1}: {len(sq_topics)} topics")
        
        for t in sq_topics:
            key = (t["video_id"], t["topic_id"])
            if key not in all_retrieved_topics:
                all_retrieved_topics[key] = t
            else:
                # Keep highest score across all queries
                if t["score"] > all_retrieved_topics[key]["score"]:
                    all_retrieved_topics[key] = t
    
    # Merge and sort by score
    global_topics = list(all_retrieved_topics.values())
    global_topics.sort(key=lambda x: x["score"], reverse=True)
    
    # Re-rank
    for i, t in enumerate(global_topics):
        t["rank"] = i + 1
    
    print(f"\n  MERGED: {len(global_topics)} unique topics from all subquestions")
    
    if not global_topics:
        print("  [ERROR] No topics found. Build index first!")
        return
    
    print(f"\n  *** TOP MERGED TOPICS ***")
    for t in global_topics[:10]:
        status = "[OK]" if t["score"] >= MIN_TOPIC_SCORE else "[FAIL]"
        print(f"  [{status}] Rank {t['rank']}: {t['video_id']} Topic {t['topic_id']} (score: {t['score']:.4f})")
        print(f"      {t['text_preview'][:100]}...")
    
    # Filter by threshold and group by video
    valid_topics = [t for t in global_topics if t["score"] >= MIN_TOPIC_SCORE]
    print(f"\n  Topics passing threshold ({MIN_TOPIC_SCORE}): {len(valid_topics)}")
    
    # Group by video, limit per video
    video_topics = {}
    for t in valid_topics:
        vid = t["video_id"]
        if vid not in video_topics:
            video_topics[vid] = []
        if len(video_topics[vid]) < CONFIG["TOP_TOPICS_PER_VIDEO"]:
            video_topics[vid].append(t)
    
    # Limit videos
    top_video_ids = list(video_topics.keys())[:CONFIG["TOP_VIDEOS"]]
    print(f"  Processing {len(top_video_ids)} videos: {top_video_ids}")
    
    # Collect all evidence
    all_evidence = {"topics": [], "scenes": [], "visual_desc": []}
    
    # Collect all VLM tasks for parallel execution
    vlm_tasks = []  # (frame_path, context, video_id, topic_id, scene_id)
    
    for vid_id in top_video_ids:
        print(f"\n{'─'*50}")
        print(f"VIDEO: {vid_id}")
        print(f"{'─'*50}")
        
        video_data = load_video_data(vid_id)
        if not video_data:
            print(f"  [ERROR] Could not load video data")
            continue
        
        # Load unified memory to get hierarchical structure and offsets
        memory_data = video_data.get("memory", {})
        topics_data = memory_data.get("topics", [])
        
        # Track frame offset across topics using the definitive manifest
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
            topic_score = next(t["score"] for t in video_topics[vid_id] if t["topic_id"] == topic_id)
            
            print(f"\n  ╠══ TOPIC {topic_id} (score: {topic_score:.4f}) ══╣")
            
            # Add topic evidence with relation info
            all_evidence["topics"].append({
                "video_id": vid_id,
                "topic_id": topic_id,
                "score": topic_score,
                "text": topic.get("text", ""),
                "start_time": topic.get("start_time"),
                "end_time": topic.get("end_time"),
                "prev_topic_id": topic.get("prev_topic_id"),
                "next_topic_id": topic.get("next_topic_id")
            })
            
            # 5.3 Scene Retrieval (SEMANTIC UPGRADE)
            print(f"    ╠══ 5.3 SEMANTIC SCENE RETRIEVAL ══╣")
            # Now passing siglip_q_emb for semantic ranking
            topic_offset = topic_frame_offsets.get(topic_id, 0)
            scored_scenes = retrieve_scenes(topic, video_data, query_emb=siglip_q_emb, topic_frame_offset=topic_offset)
            print(f"    Retrieved {len(scored_scenes)} highly relevant scenes")
            
            for s_item in scored_scenes:
                scene = s_item["scene"]
                scene_id = scene.get("scene_id", "?")
                
                print(f"       Scene {scene_id} ({scene.get('start_time', 0):.1f}s - {scene.get('end_time', 0):.1f}s)")
                
                # Add scene evidence with TEMPORAL RELATIONS (Flowchart requirement)
                all_evidence["scenes"].append({
                    "video_id": vid_id,
                    "topic_id": topic_id,
                    "scene_id": scene_id,
                    "start_time": scene.get("start_time"),
                    "end_time": scene.get("end_time"),
                    "frame_count": scene.get("frame_count"),
                    "prev_scene_id": scene.get("prev_scene_id"),
                    "next_scene_id": scene.get("next_scene_id")
                })
                
                # 5.4 Keyframe Selection (EMBEDDING-BASED + SMOOTHING)
                # This uses the new logic from Flowchart 3
                keyframe_pairs = select_keyframes_with_embeddings(
                    scene, 
                    siglip_q_emb,  # Use SigLIP embedding here!
                    video_data, 
                    scene_frame_offset=topic_offset
                )
                
                for fp, score in keyframe_pairs:
                    context = f"Topic {topic_id}, Scene {scene_id} (vis-score: {score:.2f})"
                    vlm_tasks.append((fp, context, vid_id, topic_id, scene_id))
    # ===== 5.5 PARALLEL VLM CALLS =====
    print(f"\n{'='*70}")
    print(f"╔══ 5.5 VLM CALLS ({len(vlm_tasks)} frames, {VLM_WORKERS} workers) ══╗")
    print(f"{'='*70}")
    
    if vlm_tasks:
        # Extract just the args for parallel_vlm_calls
        vlm_args = [(fp, ctx, vid) for fp, ctx, vid, _, _ in vlm_tasks]
        vlm_results = parallel_vlm_calls(vlm_args)
        
        # Map results back
        for i, result in enumerate(vlm_results):
            fp, ctx, vid, topic_id, scene_id = vlm_tasks[i]
            cache_status = "[CACHED]" if result.get("cached") else "[NEW]"
            print(f"  {cache_status} {result['frame']}: {result['description'][:80]}...")
            
            if result["description"] and not result["description"].startswith("["):
                all_evidence["visual_desc"].append({
                    "video_id": vid,
                    "topic_id": topic_id,
                    "scene_id": scene_id,
                    "frame": result["frame"],
                    "description": result["description"]
                })
    # ===== 5.6 Answer Generation =====
    print(f"\n{'='*70}")
    print("╔══ 5.6 ANSWER GENERATION ══╗")
    print(f"{'='*70}")
    
    print(f"\nEvidence Summary:")
    print(f"  Topics: {len(all_evidence['topics'])}")
    print(f"  Scenes: {len(all_evidence['scenes'])}")
    print(f"  Visual: {len(all_evidence['visual_desc'])}")
    
    context_str = json.dumps(all_evidence, indent=2, ensure_ascii=False)
    
    # Truncate if too long
    if len(context_str) > 10000:
        context_str = context_str[:10000] + "\n... [truncated]"
    
    print(f"\nEvidence JSON ({len(context_str)} chars):")
    print("─" * 40)
    print(context_str[:3000] + ("..." if len(context_str) > 3000 else ""))
    print("─" * 40)
    messages = [
        {"role": "system", "content": "You are a cooking assistant. Answer using ONLY the provided evidence. Be specific and cite timestamps when possible."},
        {"role": "user", "content": f"Evidence:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"}
    ]
    answer = call_openrouter(messages, ANSWER_MODEL, temperature=0.35, max_tokens=600)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n{'='*70}")
    print("FINAL ANSWER")
    print(f"{'='*70}")
    print(answer)
    print(f"{'='*70}")
    print(f"\nTotal time: {duration:.1f} seconds")
    print(f"{'='*70}\n")
    return {
        "query": query,
        "sub_questions": sub_questions,
        "evidence": all_evidence,
        "answer": answer,
        "duration_seconds": duration
    }
# ────────────────────────────────────────────────
if __name__ == "__main__":
    # Load index at startup
    load_global_index()
    
    # Test query
    sample_query = "How do you prepare the dressing for the salad?"
    run_query_pipeline(sample_query)