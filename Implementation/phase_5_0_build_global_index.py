# phase_5_0_build_global_index.py
"""
Build a global FAISS index over ALL topics across ALL videos.
This makes cross-video retrieval O(log N) instead of O(N × topics).
Run once after building video memory, or whenever new videos are added.
"""
import os
import json
import numpy as np
import faiss
from pathlib import Path
from datetime import datetime
# ====================== CONFIG ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_ROOT = os.path.join(BASE_DIR, "video_memory")
INDEX_PATH = os.path.join(MEMORY_ROOT, "global_topic_index.faiss")
METADATA_PATH = os.path.join(MEMORY_ROOT, "global_topic_metadata.json")
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dimension
def build_global_topic_index():
    """
    Build a FAISS index containing all topic embeddings from all videos.
    Also saves metadata mapping index position → (video_id, topic_id).
    """
    print(f"\n{'='*60}")
    print(f"Building Global Topic Index – {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    all_embeddings = []
    all_metadata = []
    
    # Scan all video directories
    video_dirs = [d for d in os.listdir(MEMORY_ROOT)
                  if os.path.isdir(os.path.join(MEMORY_ROOT, d))]
    
    print(f"Found {len(video_dirs)} video directories")
    
    for vid_id in video_dirs:
        topics_path = os.path.join(MEMORY_ROOT, vid_id, "topics.json")
        
        if not os.path.exists(topics_path):
            continue
        
        try:
            with open(topics_path, 'r', encoding='utf-8') as f:
                topics = json.load(f)
        except:
            continue
        
        for topic in topics:
            if "representative_embedding" not in topic or not topic["representative_embedding"]:
                continue
            
            emb = np.array(topic["representative_embedding"], dtype=np.float32)
            
            # Verify dimension
            if emb.shape[0] != EMBEDDING_DIM:
                print(f"  [WARN] {vid_id} topic {topic.get('topic_id')} has dim {emb.shape[0]}, expected {EMBEDDING_DIM}")
                continue
            
            all_embeddings.append(emb)
            all_metadata.append({
                "video_id": vid_id,
                "topic_id": topic.get("topic_id"),
                "start_time": topic.get("start_time"),
                "end_time": topic.get("end_time"),
                "text_preview": topic.get("text", "")[:200]
            })
    
    if not all_embeddings:
        print("ERROR: No valid topic embeddings found!")
        return
    
    print(f"\nCollected {len(all_embeddings)} topic embeddings from {len(set(m['video_id'] for m in all_metadata))} videos")
    
    # Stack into matrix
    emb_matrix = np.vstack(all_embeddings).astype(np.float32)
    print(f"Embedding matrix shape: {emb_matrix.shape}")
    
    # Normalize for cosine similarity
    faiss.normalize_L2(emb_matrix)
    
    # Build FAISS index
    # Using IndexFlatIP for exact inner product (cosine after normalization)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(emb_matrix)
    
    print(f"\nFAISS index built with {index.ntotal} vectors")
    
    # Save index
    faiss.write_index(index, INDEX_PATH)
    print(f"Index saved to: {INDEX_PATH}")
    
    # Save metadata
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2)
    print(f"Metadata saved to: {METADATA_PATH}")
    
    print(f"\n{'='*60}")
    print("Global index build complete!")
    print(f"{'='*60}\n")
    
    return index, all_metadata
if __name__ == "__main__":
    build_global_topic_index()