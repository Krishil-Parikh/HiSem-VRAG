# phase_1_3_chunk_transcripts.py
import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch  # ← Added this import (fixes the NameError)


# ====================== CONFIG ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASR_DIR = os.path.join(BASE_DIR, "asr_transcripts_json")
CHUNKED_DIR = os.path.join(BASE_DIR, "chunked_transcripts_json")


os.makedirs(CHUNKED_DIR, exist_ok=True)


EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # fast & effective; ~80MB download
MIN_CHUNK_SEC = 20.0
MAX_CHUNK_SEC = 40.0
SIM_THRESHOLD = 0.75                  # cosine sim > this → merge adjacent chunks
MAX_WORKERS = 8                       # CPU-friendly (can be higher if you have many cores)


# ====================== HELPERS ======================
def chunk_and_merge_one(args):
    """
    Process one ASR JSON → create merged 20-40s chunks
    args: (json_path, embedder)
    Returns: (stem_name, success, message)
    """
    json_path, embedder = args
    stem = Path(json_path).stem
    out_path = os.path.join(CHUNKED_DIR, f"{stem}.json")


    if os.path.exists(out_path):
        try:
            with open(out_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            if isinstance(chunks, list) and len(chunks) > 0:
                return stem, True, f"already chunked ({len(chunks)} chunks)"
        except:
            pass  # corrupt → re-process


    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)


        if not segments or not isinstance(segments, list):
            return stem, False, "empty or invalid transcript"


        # Sort segments by start time (safety)
        segments.sort(key=lambda x: x.get("start", 0))


        # Step 1: Greedy initial chunking by time
        chunks = []
        current = {"texts": [], "starts": [], "ends": [], "orig_ids": []}
        current_dur = 0.0


        for i, seg in enumerate(segments):
            seg_text = seg.get("text", "").strip()
            if not seg_text:
                continue


            seg_dur = seg["end"] - seg["start"]
            current["texts"].append(seg_text)
            current["starts"].append(seg["start"])
            current["ends"].append(seg["end"])
            current["orig_ids"].append(i)
            current_dur += seg_dur


            # Close chunk if over max or (over min and not last segment)
            if current_dur >= MAX_CHUNK_SEC or (current_dur >= MIN_CHUNK_SEC and i < len(segments) - 1):
                chunk_text = " ".join(current["texts"])
                chunks.append({
                    "chunk_id": len(chunks),
                    "text": chunk_text,
                    "start": min(current["starts"]),
                    "end": max(current["ends"]),
                    "original_segment_ids": current["orig_ids"]
                })
                current = {"texts": [], "starts": [], "ends": [], "orig_ids": []}
                current_dur = 0.0


        # Last remaining chunk
        if current["texts"]:
            chunk_text = " ".join(current["texts"])
            chunks.append({
                "chunk_id": len(chunks),
                "text": chunk_text,
                "start": min(current["starts"]),
                "end": max(current["ends"]),
                "original_segment_ids": current["orig_ids"]
            })


        # Step 2: Merge adjacent chunks if semantically similar
        if len(chunks) <= 1:
            final_chunks = chunks
        else:
            chunk_texts = [c["text"] for c in chunks]
            embeddings = embedder.encode(chunk_texts, convert_to_tensor=True, show_progress_bar=False)
            final_chunks = []
            current_merged = chunks[0].copy()


            for i in range(1, len(chunks)):
                sim = util.cos_sim(embeddings[i-1], embeddings[i]).item()
                if sim >= SIM_THRESHOLD:
                    # Merge
                    current_merged["text"] += " " + chunks[i]["text"]
                    current_merged["end"] = max(current_merged["end"], chunks[i]["end"])
                    current_merged["original_segment_ids"].extend(chunks[i]["original_segment_ids"])
                else:
                    final_chunks.append(current_merged)
                    current_merged = chunks[i].copy()


            final_chunks.append(current_merged)


            # Re-assign chunk_ids
            for idx, chunk in enumerate(final_chunks):
                chunk["chunk_id"] = idx


        # Save result
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(final_chunks, f, ensure_ascii=False, indent=2)


        return stem, True, f"chunked & merged → {len(final_chunks)} final chunks"


    except Exception as e:
        return stem, False, f"error: {str(e)[:180]}"



# ====================== MAIN ======================
if __name__ == "__main__":
    start_time = datetime.now()
    print(f"\n=== Phase 1.3 — Transcript Chunking & Merging ({start_time.strftime('%Y-%m-%d %H:%M:%S')}) ===\n")
    print(f"Model: {EMBEDDING_MODEL} | Threshold: {SIM_THRESHOLD} | Min/Max: {MIN_CHUNK_SEC}/{MAX_CHUNK_SEC}s\n")


    print("Loading sentence-transformer model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
    print(f"Embedder loaded on: {device}")


    json_files = [os.path.join(ASR_DIR, f) for f in os.listdir(ASR_DIR) if f.endswith('.json')]


    print(f"Found {len(json_files)} ASR JSON files to process")


    success = 0
    failed = []


    # Prepare args list for parallel execution
    tasks = [(jpath, embedder) for jpath in json_files]


    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(chunk_and_merge_one, task) for task in tasks]


        for future in tqdm(as_completed(futures), total=len(json_files), desc="Chunking"):
            name, ok, msg = future.result()
            if ok:
                success += 1
                tqdm.write(f"OK   {name}: {msg}")
            else:
                failed.append((name, msg))
                tqdm.write(f"FAIL {name}: {msg}")


    duration = datetime.now() - start_time
    print(f"\n=== Phase 1.3 Complete ===")
    print(f"Duration: {duration}")
    print(f"Succeeded: {success} / {len(json_files)}")
    if failed:
        print(f"Failed: {len(failed)}")
        print("First few failures:")
        for n, m in failed[:5]:
            print(f"  - {n}: {m}")


    chunk_count = len([f for f in os.listdir(CHUNKED_DIR) if f.endswith('.json')])
    print(f"\nChunked & merged transcripts saved: {chunk_count} files → {CHUNKED_DIR}")
    print("\nNext step: Phase 1.4 — Embed these chunks (transcript_embeddings.npy per video)")