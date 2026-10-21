# phase_3_2_embed_frames.py
import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModel
from concurrent.futures import ThreadPoolExecutor, as_completed
# ====================== CONFIG ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_ROOT_DIR = os.path.join(BASE_DIR, "frames")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "frame_embeddings")
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
VISION_MODEL_NAME = "google/siglip-so400m-patch14-384"  # Strong 2026-era CLIP-like; 1152-dim
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32                       # adjust down if OOM (16–64 typical)
MAX_WORKERS = 4                       # lower due to GPU memory usage
# ====================== HELPERS ======================
def load_model_and_processor():
    print(f"Loading {VISION_MODEL_NAME} on {DEVICE}...")
    processor = AutoProcessor.from_pretrained(VISION_MODEL_NAME)
    model = AutoModel.from_pretrained(VISION_MODEL_NAME).to(DEVICE)
    model.eval()
    return processor, model
def embed_frames_in_batch(images: list[Image.Image], processor, model):
    """Batch-embed list of PIL images"""
    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    return outputs.cpu().numpy()  # shape: (batch, dim)
def embed_one_video(args):
    """
    Embed all extracted frames for one video
    args: (stem, manifest_path)
    """
    stem, manifest_path = args
    npy_path = os.path.join(EMBEDDINGS_DIR, f"{stem}.npy")
    if os.path.exists(npy_path):
        try:
            embs = np.load(npy_path)
            if embs.ndim == 2 and embs.shape[1] in [768, 1024, 1152]:  # common dims
                return stem, True, f"already embedded ({embs.shape[0]} frames)"
        except:
            pass
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifests = json.load(f)
        all_embeddings = []
        all_timestamps = []  # optional, for reference
        for topic_manifest in manifests:
            topic_frames = []
            for rel_path in topic_manifest["frame_paths"]:
                full_path = os.path.join(BASE_DIR, rel_path)
                if os.path.exists(full_path):
                    img = Image.open(full_path).convert("RGB")
                    topic_frames.append(img)
                else:
                    print(f"Warning: missing frame {full_path}")
            if not topic_frames:
                continue
            # Embed in batches
            for i in range(0, len(topic_frames), BATCH_SIZE):
                batch_imgs = topic_frames[i:i + BATCH_SIZE]
                batch_embs = embed_frames_in_batch(batch_imgs, processor, model)
                all_embeddings.append(batch_embs)
            all_timestamps.extend(topic_manifest["timestamps"])
        if not all_embeddings:
            return stem, False, "no frames found"
        # Concatenate all
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        # Normalize (good for cosine similarity later)
        norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        all_embeddings /= np.maximum(norms, 1e-8)
        np.save(npy_path, all_embeddings)
        return stem, True, f"embedded {all_embeddings.shape[0]} frames → shape {all_embeddings.shape}"
    except Exception as e:
        return stem, False, f"error: {str(e)[:180]}"
# ====================== MAIN ======================
if __name__ == "__main__":
    start_time = datetime.now()
    print(f"\n=== Phase 3.2 — Frame Embedding ({start_time.strftime('%Y-%m-%d %H:%M:%S')}) ===\n")
    print(f"Vision model: {VISION_MODEL_NAME}")
    print(f"Device: {DEVICE} | Batch size: {BATCH_SIZE}\n")
    processor, model = load_model_and_processor()
    # Find all manifests
    manifests = []
    for vid_dir in os.listdir(FRAMES_ROOT_DIR):
        manifest_path = os.path.join(FRAMES_ROOT_DIR, vid_dir, "frames_manifest.json")
        if os.path.exists(manifest_path):
            stem = vid_dir
            manifests.append((stem, manifest_path))
    print(f"Found {len(manifests)} videos with frame manifests")
    success = 0
    failed = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(embed_one_video, task) for task in manifests]
        for future in tqdm(as_completed(futures), total=len(manifests), desc="Frame Embedding"):
            name, ok, msg = future.result()
            if ok:
                success += 1
                tqdm.write(f"OK   {name}: {msg}")
            else:
                failed.append((name, msg))
                tqdm.write(f"FAIL {name}: {msg}")
    duration = datetime.now() - start_time
    print(f"\n=== Phase 3.2 Complete ===")
    print(f"Duration: {duration}")
    print(f"Succeeded: {success} / {len(manifests)}")
    if failed:
        print(f"Failed: {len(failed)}")
        print("First few failures:")
        for n, m in failed[:5]:
            print(f"  - {n}: {m}")
    npy_count = len([f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith('.npy')])
    total_embs = 0
    for f in os.listdir(EMBEDDINGS_DIR):
        if f.endswith('.npy'):
            try:
                shape = np.load(os.path.join(EMBEDDINGS_DIR, f)).shape
                total_embs += shape[0]
            except:
                pass
    print(f"\nFrame embeddings saved: {npy_count} .npy files → {EMBEDDINGS_DIR}")
    print(f"Total frame embeddings: ~{total_embs:,}")
    print("\nNext: Phase 3.3 — Scene clustering inside each topic (KMeans / HDBSCAN on frame embeddings)")
    print("   • We'll load per-topic frames/embeddings using the manifest")
    print("   • Cluster → define scenes with start/end + frame ids\n")