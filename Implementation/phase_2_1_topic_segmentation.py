# phase_2_1_topic_segmentation.py
import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
# ====================== CONFIG ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNK_JSON_DIR = os.path.join(BASE_DIR, "chunked_transcripts_json")
EMBED_NPY_DIR   = os.path.join(BASE_DIR, "transcript_embeddings")
TOPICS_DIR      = os.path.join(BASE_DIR, "topics_json")
os.makedirs(TOPICS_DIR, exist_ok=True)
COSINE_THRESHOLD = 0.68          # tune: 0.65–0.75 common range; lower = more topics
MIN_TOPIC_DURATION_SEC = 60.0    # enforce ≥ 60s per topic
MAX_WORKERS = 12                 # very fast operation, can be high
# ====================== HELPERS ======================
def segment_one_video(args):
    """
    Perform topic segmentation for one video using chunk embeddings
    args: (chunk_json_path, embed_npy_path)
    Returns: (stem, success, message)
    """
    chunk_json_path, embed_npy_path = args
    stem = Path(chunk_json_path).stem
    # Output path
    topic_path = os.path.join(TOPICS_DIR, f"{stem}.json")
    if os.path.exists(topic_path):
        try:
            with open(topic_path, 'r') as f:
                topics = json.load(f)
            if isinstance(topics, list) and len(topics) > 0:
                return stem, True, f"already segmented ({len(topics)} topics)"
        except:
            pass
    try:
        # Load chunks (timestamps + text)
        with open(chunk_json_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        if not chunks:
            return stem, False, "no chunks"
        # Load embeddings
        embeddings = np.load(embed_npy_path)
        if len(embeddings) != len(chunks):
            return stem, False, f"embedding count mismatch ({len(embeddings)} vs {len(chunks)} chunks)"
        # Compute cosine similarities between consecutive chunks
        similarities = []
        for i in range(1, len(embeddings)):
            sim = np.dot(embeddings[i-1], embeddings[i]) / (
                np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
            )
            similarities.append(sim)
        # Detect boundaries: low similarity → boundary after chunk i
        boundaries = [0]  # start with first chunk
        for i, sim in enumerate(similarities):
            if sim < COSINE_THRESHOLD:
                boundaries.append(i + 1)  # boundary after chunk i
        boundaries.append(len(chunks))  # end
        # Build initial topics
        initial_topics = []
        for start_idx, end_idx in zip(boundaries[:-1], boundaries[1:]):
            topic_chunks = chunks[start_idx:end_idx]
            topic_start = topic_chunks[0]["start"]
            topic_end   = topic_chunks[-1]["end"]
            topic_dur   = topic_end - topic_start
            topic_text = " ".join(c["text"] for c in topic_chunks)
            initial_topics.append({
                "start_time": topic_start,
                "end_time": topic_end,
                "duration": round(topic_dur, 2),
                "chunk_ids": list(range(start_idx, end_idx)),
                "text": topic_text.strip()
            })
        # Enforce min duration: merge short topics with previous
        topics = []
        current = initial_topics[0].copy()
        for next_topic in initial_topics[1:]:
            if current["duration"] < MIN_TOPIC_DURATION_SEC:
                # Merge with next (greedy backward merge)
                current["end_time"] = next_topic["end_time"]
                current["duration"] = current["end_time"] - current["start_time"]
                current["chunk_ids"].extend(next_topic["chunk_ids"])
                current["text"] += " " + next_topic["text"]
            else:
                topics.append(current)
                current = next_topic.copy()
        # Add last one (if long enough or forced)
        if current["duration"] >= MIN_TOPIC_DURATION_SEC or not topics:
            topics.append(current)
        else:
            # If last is too short, merge to previous
            if topics:
                topics[-1]["end_time"] = current["end_time"]
                topics[-1]["duration"] = topics[-1]["end_time"] - topics[-1]["start_time"]
                topics[-1]["chunk_ids"].extend(current["chunk_ids"])
                topics[-1]["text"] += " " + current["text"]
        # Assign topic_id and optional average embedding
        for idx, topic in enumerate(topics):
            topic["topic_id"] = idx
            # Optional: average embedding (useful for later retrieval)
            topic_chunk_embs = embeddings[topic["chunk_ids"]]
            topic["representative_embedding"] = np.mean(topic_chunk_embs, axis=0).tolist()
        # Save
        with open(topic_path, 'w', encoding='utf-8') as f:
            json.dump(topics, f, ensure_ascii=False, indent=2)
        return stem, True, f"segmented → {len(topics)} topics (avg duration {np.mean([t['duration'] for t in topics]):.1f}s)"
    except Exception as e:
        return stem, False, f"error: {str(e)[:180]}"
# ====================== MAIN ======================
if __name__ == "__main__":
    start_time = datetime.now()
    print(f"\n=== Phase 2.1 — Topic Segmentation ({start_time.strftime('%Y-%m-%d %H:%M:%S')}) ===\n")
    print(f"Cosine threshold: {COSINE_THRESHOLD} | Min topic duration: {MIN_TOPIC_DURATION_SEC}s\n")
    chunk_jsons = [
        os.path.join(CHUNK_JSON_DIR, f)
        for f in os.listdir(CHUNK_JSON_DIR)
        if f.endswith('.json')
    ]
    embed_npy_paths = {
        Path(f).stem: os.path.join(EMBED_NPY_DIR, f)
        for f in os.listdir(EMBED_NPY_DIR)
        if f.endswith('.npy')
    }
    # Only process videos that have both chunk json and embeddings
    valid_pairs = []
    for jpath in chunk_jsons:
        stem = Path(jpath).stem
        npy_path = embed_npy_paths.get(stem)
        if npy_path and os.path.exists(npy_path):
            valid_pairs.append((jpath, npy_path))
    print(f"Found {len(valid_pairs)} videos with both chunks and embeddings")
    success = 0
    failed = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(segment_one_video, pair) for pair in valid_pairs]
        for future in tqdm(as_completed(futures), total=len(valid_pairs), desc="Topic Segmentation"):
            name, ok, msg = future.result()
            if ok:
                success += 1
                tqdm.write(f"OK   {name}: {msg}")
            else:
                failed.append((name, msg))
                tqdm.write(f"FAIL {name}: {msg}")
    duration = datetime.now() - start_time
    print(f"\n=== Phase 2.1 Complete ===")
    print(f"Duration: {duration}")
    print(f"Succeeded: {success} / {len(valid_pairs)}")
    if failed:
        print(f"Failed: {len(failed)}")
        print("First few failures:")
        for n, m in failed[:5]:
            print(f"  - {n}: {m}")
    topic_count = len([f for f in os.listdir(TOPICS_DIR) if f.endswith('.json')])
    print(f"\nTopic segmentations saved: {topic_count} files → {TOPICS_DIR}")
    print("\nNext possible steps:")
    print("  • Inspect a few topics.json files — check if # topics and durations look reasonable")
    print("  • Tune COSINE_THRESHOLD or MIN_TOPIC_DURATION_SEC if needed")
    print("  • Proceed to Phase 3 — Visual processing inside each topic (frame extraction + embedding)\n")