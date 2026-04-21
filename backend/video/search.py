import os
import faiss
import json
import requests
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.ndimage import gaussian_filter1d

# Popularity re-ranking (additive — does not touch existing logic)
from .popularity import load_popularity, rerank_with_popularity

# Re-use paths from pipeline
from .pipeline import (
    IMPLEMENTATION_MEMORY, 
    MEMORY_ROOT, 
    get_text_embedder, 
    get_vision_model, 
    DEVICE
)

GLOBAL_INDEX_PATH = IMPLEMENTATION_MEMORY / "global_topic_index.faiss"
GLOBAL_METADATA_PATH = IMPLEMENTATION_MEMORY / "global_topic_metadata.json"

# OpenRouter Config
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "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")
OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/yourusername/vrag",
    "X-Title": "VideoRAG Dashboard",
}

DECOMPOSE_MODEL = "meta-llama/llama-3.1-8b-instruct"
ANSWER_MODEL    = "meta-llama/llama-3.1-8b-instruct"
VLM_MODEL       = "qwen/qwen2.5-vl-72b-instruct"

# Search Parameters
MIN_TOPIC_SCORE = 0.25
TOP_K_GLOBAL = 15
TOP_VIDEOS = 3
TOP_TOPICS_PER_VIDEO = 3
TOP_SCENES_PER_TOPIC = 3
VLM_WORKERS = 4

_index = None
_metadata = None

def load_global_search():
    global _index, _metadata
    if _index is not None:
        return _index, _metadata
        
    if not GLOBAL_INDEX_PATH.exists() or not GLOBAL_METADATA_PATH.exists():
        return None, None
        
    _index = faiss.read_index(str(GLOBAL_INDEX_PATH))
    with open(GLOBAL_METADATA_PATH, 'r', encoding='utf-8') as f:
        _metadata = json.load(f)
    return _index, _metadata

def call_openrouter(messages: List[Dict], model: str, temperature: float = 0.4, max_tokens: int = 800) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=OPENROUTER_HEADERS,
            json=payload,
            timeout=60
        )
        if response.status_code != 200:
            return ""
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""

def decompose_query(query: str) -> List[str]:
    messages = [
        {"role": "system", "content": "Break down questions into 2-3 specific sub-questions for video retrieval. Output only a numbered list."},
        {"role": "user", "content": f"Question: {query}"}
    ]
    text = call_openrouter(messages, DECOMPOSE_MODEL, temperature=0.3, max_tokens=200)
    if not text: return [query]
    lines = [line.strip() for line in text.split("\n") if line.strip() and line.strip()[0].isdigit()]
    subs = [line.split(".", 1)[1].strip() if "." in line else line for line in lines]
    return subs[:3] or [query]

def get_siglip_text_emb(text: str) -> np.ndarray:
    processor, model = get_vision_model()
    inputs = processor(text=[text], padding="max_length", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features.cpu().numpy().flatten()

def search_global_topics(query: str, top_k: int = 5, video_ids: list[str] = None):
    index, metadata = load_global_search()
    if index is None: return []
        
    embedder = get_text_embedder()
    query_emb = embedder.encode(query).astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(query_emb)
    
    search_k = max(top_k * 5, 20) if video_ids else top_k
    scores, indices = index.search(query_emb, search_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0: continue
        meta = metadata[idx]
        if video_ids and meta["video_id"] not in video_ids: continue
            
        results.append({
            "video_id": meta["video_id"],
            "topic_id": meta["topic_id"],
            "score": float(score),
            "text": meta.get("text_preview", ""),
            "start_time": meta.get("start_time"),
            "end_time": meta.get("end_time")
        })
        if len(results) >= top_k: break
    return results

def describe_frame_with_vlm(frame_path: str, context: str = "") -> str:
    if not os.path.exists(frame_path): return "[frame not found]"
    
    import base64
    with open(frame_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")
        
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": f"Describe this cooking video frame concisely. Focus on: ingredients, tools, actions. Context: {context}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}}
        ]}
    ]
    return call_openrouter(messages, VLM_MODEL, temperature=0.3, max_tokens=150)

def run_full_rag_pipeline(query: str, video_ids: list[str] = None):
    # 1. Decompose Query
    sub_questions = decompose_query(query)
    queries = [query] + sub_questions
    
    # 2. Get Vision Embedding for Scene Ranking
    siglip_q_emb = get_siglip_text_emb(query)
    q_norm = siglip_q_emb / (np.linalg.norm(siglip_q_emb) + 1e-8)
    
    # 3. Retrieve Merged Topics
    embedder = get_text_embedder()
    all_hits = {}
    
    # Search for main query and subquestions
    for q in queries:
        hits = search_global_topics(q, top_k=5, video_ids=video_ids)
        for h in hits:
            key = (h["video_id"], h["topic_id"])
            if key not in all_hits or h["score"] > all_hits[key]["score"]:
                all_hits[key] = h
                
    sorted_topics = sorted(all_hits.values(), key=lambda x: x["score"], reverse=True)
    print(f"[RAG] Found {len(sorted_topics)} potential topics across all sub-queries")
    valid_topics = [t for t in sorted_topics if t["score"] >= MIN_TOPIC_SCORE][:TOP_K_GLOBAL]
    print(f"[RAG] {len(valid_topics)} topics passed score threshold {MIN_TOPIC_SCORE}")
    
    metrics = {
        "num_queries_formed": len(queries),
        "queries": queries,
        "num_videos_retrieved": 0,
        "num_topics_retrieved": len(valid_topics),
        "num_scenes_retrieved": 0,
        "num_frames_retrieved": 0,
        "scene_similarities": []
    }
    
    if not valid_topics:
        if sorted_topics:
            print(f"[RAG] Top score was {sorted_topics[0]['score']:.4f} (below threshold)")
        return {"answer": "I couldn't find any relevant clips in the database for your query.", "evidence": [], "metrics": metrics, "display_scenes": []}

    # Group and load data
    video_groups = {}
    for t in valid_topics:
        vid = t["video_id"]
        if vid not in video_groups: video_groups[vid] = []
        if len(video_groups[vid]) < TOP_TOPICS_PER_VIDEO:
            video_groups[vid].append(t)
            
    metrics["num_videos_retrieved"] = len(video_groups)
    
    final_evidence = []
    vlm_tasks = [] # (frame_path, context, evidence_idx)
    
    for vid_id, topics in list(video_groups.items())[:TOP_VIDEOS]:
        # Load video memory
        mem_path = Path(MEMORY_ROOT) / vid_id / "video_memory.json"
        if not mem_path.exists():
            mem_path = IMPLEMENTATION_MEMORY / vid_id / "video_memory.json"
        if not mem_path.exists(): continue
        
        with open(mem_path, 'r', encoding='utf-8') as f:
            video_mem = json.load(f)
            
        emb_path = Path(MEMORY_ROOT) / vid_id / "frame_embeddings.npy"
        if not emb_path.exists():
            emb_path = IMPLEMENTATION_MEMORY / vid_id / "frame_embeddings.npy"
        frame_emb = np.load(emb_path) if emb_path.exists() else None
        
        # Calculate offsets
        topic_offsets = {}
        curr = 0
        for t_mem in video_mem["topics"]:
            topic_offsets[t_mem["topic_id"]] = curr
            curr += sum(s["frame_count"] for s in t_mem["scenes"])
            
        retrieved_topic_ids = [t["topic_id"] for t in topics]
        
        for t_mem in video_mem["topics"]:
            if t_mem["topic_id"] not in retrieved_topic_ids: continue
            
            # Semantic Scene Retrieval
            scene_scores = []
            topic_offset = topic_offsets.get(t_mem["topic_id"], 0)
            
            for scene in t_mem["scenes"]:
                score = 0.5 # fallback
                if frame_emb is not None:
                    # Find best frame in scene
                    f_ids = scene.get("frame_ids", [])
                    s_embs = []
                    for fid in f_ids:
                        idx = topic_offset + fid
                        if idx < len(frame_emb): s_embs.append(frame_emb[idx])
                    
                    if s_embs:
                        s_embs = np.array(s_embs)
                        f_scores = np.dot(s_embs, q_norm)
                        score = float(np.max(f_scores))
                
                scene_scores.append((scene, score))
            
            scene_scores.sort(key=lambda x: x[1], reverse=True)
            
            for scene, s_score in scene_scores[:TOP_SCENES_PER_TOPIC]:
                evidence_item = {
                    "video_id": vid_id,
                    "topic_id": t_mem["topic_id"],
                    "scene_id": scene["scene_id"],
                    "text": t_mem["text"],
                    "start_time": scene["start_time"],
                    "end_time": scene["end_time"],
                    "score": s_score,
                    "frame_url": "", # To be filled
                    "visual_description": "" 
                }
                
                # Pick best frame for VLM and display
                if scene.get("frame_paths"):
                    fp = scene["frame_paths"][len(scene["frame_paths"])//2]
                    # Convert implementation path to absolute if needed
                    full_fp = Path(fp)
                    if not full_fp.is_absolute():
                        p1 = Path(MEMORY_ROOT) / vid_id / fp
                        p2 = IMPLEMENTATION_MEMORY / vid_id / fp
                        full_fp = p1 if p1.exists() else (p2 if p2.exists() else full_fp)
                    
                    if full_fp.exists():
                        idx = len(final_evidence)
                        vlm_tasks.append((str(full_fp), t_mem["text"][:100], idx))
                        evidence_item["frame_url"] = str(full_fp)
                
                final_evidence.append(evidence_item)

    # 4. Parallel VLM
    metrics["num_frames_retrieved"] = len(vlm_tasks)
    if vlm_tasks:
        with ThreadPoolExecutor(max_workers=VLM_WORKERS) as executor:
            futures = {executor.submit(describe_frame_with_vlm, task[0], task[1]): task[2] for task in vlm_tasks}
            for future in as_completed(futures):
                idx = futures[future]
                final_evidence[idx]["visual_description"] = future.result()

    # 5. Build Display Scenes & Final Answer Generation
    metrics["num_scenes_retrieved"] = len(final_evidence)
    metrics["scene_similarities"] = [{"video_id": e["video_id"], "scene_id": e["scene_id"], "score": e["score"]} for e in final_evidence]
    
    display_scenes = []
    grouped_evidence = {}
    for ev in final_evidence:
        vid = ev["video_id"]
        if vid not in grouped_evidence:
            grouped_evidence[vid] = []
        grouped_evidence[vid].append(ev)
        
    for vid, evs in grouped_evidence.items():
        # deduplicate by scene_id and topic_id combined
        unique_scenes_map = {}
        for e in evs:
            key = f"{e['topic_id']}_{e['scene_id']}"
            if key not in unique_scenes_map or e["score"] > unique_scenes_map[key]["score"]:
                unique_scenes_map[key] = e
        unique_scenes = list(unique_scenes_map.values())
        
        if len(unique_scenes) >= 4:
            min_start = min(e["start_time"] for e in unique_scenes)
            max_end = max(e["end_time"] for e in unique_scenes)
            unique_scenes.sort(key=lambda x: x["start_time"])
            combined_text = " | ".join([e["text"] for e in unique_scenes])
            combined_visual = " | ".join([e["visual_description"] for e in unique_scenes if e["visual_description"]])
            
            best_scene = max(unique_scenes, key=lambda x: x["score"])
            display_scenes.append({
                "video_id": vid,
                "type": "combined",
                "topic_id": best_scene["topic_id"],
                "scene_ids": [e["scene_id"] for e in unique_scenes],
                "start_time": min_start,
                "end_time": max_end,
                "text": combined_text,
                "visual_description": combined_visual,
                "frame_url": best_scene.get("frame_url", ""),
                "score": best_scene["score"]
            })
        else:
            for e in unique_scenes:
                display_scenes.append({
                    "video_id": vid,
                    "type": "single",
                    "topic_id": e["topic_id"],
                    "scene_ids": [e["scene_id"]],
                    "start_time": e["start_time"],
                    "end_time": e["end_time"],
                    "text": e["text"],
                    "visual_description": e["visual_description"],
                    "frame_url": e.get("frame_url", ""),
                    "score": e["score"]
                })

    context_str = json.dumps([{
        "video": e["video_id"],
        "start": f"{e['start_time']:.1f}s",
        "description": e["text"],
        "visual": e["visual_description"]
    } for e in final_evidence], indent=2)

    messages = [
        {"role": "system", "content": "You are a specialized cooking assistant. Answer the user's question using the provided video evidence. Be precise about steps and tools mentioned. Cite video IDs and timestamps."},
        {"role": "user", "content": f"Evidence:\n{context_str}\n\nQuestion: {query}"}
    ]
    print(f"[RAG] Final evidence count: {len(final_evidence)}")
    answer = call_openrouter(messages, ANSWER_MODEL, temperature=0.3)
    
    # =====================================================================
    # POPULARITY RE-RANKING  (additive step — runs AFTER everything else)
    # Applies a small weighted boost based on simulated likes / subscribers.
    # Original similarity scores are preserved in "original_score".
    # =====================================================================
    popularity_data = load_popularity()
    rerank_with_popularity(final_evidence, popularity_data, weight=0.10, score_key="score")
    rerank_with_popularity(display_scenes, popularity_data, weight=0.10, score_key="score")
    print(f"[RAG] Popularity re-ranking applied (weight=0.10)")
    
    # Refresh metrics with post-popularity scores
    metrics["scene_similarities"] = [
        {"video_id": e["video_id"], "scene_id": e["scene_id"], "score": e["score"]}
        for e in final_evidence
    ]
    
    return {
        "answer": answer or "I found relevant clips but failed to generate a summary. Please check the evidence below.",
        "evidence": final_evidence,
        "metrics": metrics,
        "display_scenes": display_scenes
    }
