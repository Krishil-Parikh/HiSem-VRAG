# phase_1_4_embed_chunks.py
import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch


# ====================== CONFIG ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKED_DIR = os.path.join(BASE_DIR, "chunked_transcripts_json")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "transcript_embeddings")


os.makedirs(EMBEDDINGS_DIR, exist_ok=True)


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_WORKERS = 8                       # CPU/GPU friendly
BATCH_SIZE = 32                       # for embedder.encode()


# ====================== HELPERS ======================
def embed_one_video(args):
    """
    Embed all chunks for one video → save as .npy
    args: (json_path, embedder)
    Returns: (stem, success, message)
    """
    json_path, embedder = args
    stem = Path(json_path).stem
    npy_path = os.path.join(EMBEDDINGS_DIR, f"{stem}.npy")


    if os.path.exists(npy_path):
        try:
            emb = np.load(npy_path)
            if emb.ndim == 2 and emb.shape[1] == 384:  # MiniLM dim
                return stem, True, f"already embedded ({emb.shape[0]} chunks)"
        except:
            pass  # corrupt → re-embed


    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)


        if not chunks or not isinstance(chunks, list):
            return stem, False, "empty or invalid chunks"


        # Extract texts
        texts = [chunk["text"].strip() for chunk in chunks if chunk.get("text", "").strip()]


        if not texts:
            return stem, False, "no valid text in chunks"


        # Embed in batch
        embeddings = embedder.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # good for cosine later
        )


        # Save
        np.save(npy_path, embeddings)


        return stem, True, f"embedded → {embeddings.shape[0]} chunks, shape {embeddings.shape}"


    except Exception as e:
        return stem, False, f"error: {str(e)[:180]}"



# ====================== MAIN ======================
if __name__ == "__main__":
    start_time = datetime.now()
    print(f"\n=== Phase 1.4 — Embed Transcript Chunks ({start_time.strftime('%Y-%m-%d %H:%M:%S')}) ===\n")
    print(f"Model: {EMBEDDING_MODEL} | Output shape per video: (num_chunks, 384)\n")


    print("Loading sentence-transformer model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
    print(f"Embedder loaded on: {device}")


    json_files = [
        os.path.join(CHUNKED_DIR, f)
        for f in os.listdir(CHUNKED_DIR)
        if f.endswith('.json')
    ]


    print(f"Found {len(json_files)} chunked JSON files")


    success = 0
    failed = []


    tasks = [(jpath, embedder) for jpath in json_files]


    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(embed_one_video, task) for task in tasks]


        for future in tqdm(as_completed(futures), total=len(json_files), desc="Embedding"):
            name, ok, msg = future.result()
            if ok:
                success += 1
                tqdm.write(f"OK   {name}: {msg}")
            else:
                failed.append((name, msg))
                tqdm.write(f"FAIL {name}: {msg}")


    duration = datetime.now() - start_time
    print(f"\n=== Phase 1.4 Complete ===")
    print(f"Duration: {duration}")
    print(f"Succeeded: {success} / {len(json_files)}")
    if failed:
        print(f"Failed: {len(failed)}")
        print("First few failures:")
        for n, m in failed[:5]:
            print(f"  - {n}: {m}")


    npy_count = len([f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith('.npy')])
    print(f"\nEmbeddings saved: {npy_count} .npy files → {EMBEDDINGS_DIR}")
    print("\nPhase 1 (Audio + ASR backbone) finished!")
    print("Next logical step: Phase 2 — Topic segmentation using these embeddings")
    print("  (cosine similarity between adjacent chunks + threshold + min duration enforcement)\n")