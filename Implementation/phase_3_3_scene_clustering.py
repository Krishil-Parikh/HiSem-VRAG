# phase_3_3_scene_clustering.py
import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm
import hdbscan
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor, as_completed
# ====================== CONFIG ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_ROOT_DIR = os.path.join(BASE_DIR, "frames")
FRAME_EMB_DIR   = os.path.join(BASE_DIR, "frame_embeddings")
SCENES_DIR      = os.path.join(BASE_DIR, "scenes_json")
os.makedirs(SCENES_DIR, exist_ok=True)
CLUSTER_METHOD = "hdbscan"          # "hdbscan" or "kmeans"
HDBSCAN_MIN_CLUSTER_SIZE = 5        # min frames per cluster (scene)
HDBSCAN_MIN_SAMPLES      = 3        # controls noise sensitivity
KMEANS_N_CLUSTERS_FACTOR = 0.15     # rough: ~15% of frames as clusters (tune)
MAX_WORKERS = 8
# ====================== HELPERS ======================
def cluster_frames(embeddings: np.ndarray, timestamps: list[float], frame_paths: list[str], method: str = "hdbscan"):
    """
    Cluster frame embeddings → return list of scene dicts
    """
    if len(embeddings) < 3:
        # Too few frames → single scene
        return [{
            "scene_id": 0,
            "start_time": timestamps[0] if timestamps else 0.0,
            "end_time": timestamps[-1] if timestamps else 0.0,
            "frame_count": len(embeddings),
            "frame_ids": list(range(len(embeddings))),
            "frame_paths": frame_paths,
            "cluster_label": 0
        }]
    if method == "hdbscan":
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=HDBSCAN_MIN_SAMPLES,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(embeddings)
    else:  # kmeans fallback
        n_clusters = max(2, int(len(embeddings) * KMEANS_N_CLUSTERS_FACTOR))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
    # Build scenes
    unique_labels = sorted(set(labels) - {-1})  # ignore noise (-1) in HDBSCAN
    scenes = []
    for label in unique_labels:
        mask = (labels == label)
        scene_timestamps = np.array(timestamps)[mask]
        scene_paths = [frame_paths[i] for i in range(len(mask)) if mask[i]]
        if len(scene_timestamps) == 0:
            continue
        scenes.append({
            "scene_id": len(scenes),
            "cluster_label": int(label),
            "start_time": float(scene_timestamps.min()),
            "end_time": float(scene_timestamps.max()),
            "frame_count": int(mask.sum()),
            "frame_ids": np.where(mask)[0].tolist(),
            "frame_paths": scene_paths
        })
    # Sort scenes by start time
    scenes.sort(key=lambda x: x["start_time"])
    # Re-number scene_id sequentially
    for i, s in enumerate(scenes):
        s["scene_id"] = i
    return scenes
def process_one_topic_group(args):
    """
    Cluster scenes for one video (all topics)
    args: (stem, manifest_path, emb_npy_path)
    """
    stem, manifest_path, emb_npy_path = args
    out_path = os.path.join(SCENES_DIR, f"{stem}.json")
    if os.path.exists(out_path):
        try:
            with open(out_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                total_scenes = sum(len(t.get("scenes", [])) for t in data)
                return stem, True, f"already clustered ({total_scenes} scenes total)"
        except:
            pass
    try:
        all_emb = np.load(emb_npy_path)
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifests = json.load(f)
        video_scenes = []  # list of topic-level scene results
        frame_offset = 0
        for topic_idx, topic_manifest in enumerate(manifests):
            topic_frames_count = topic_manifest["frame_count"]
            if topic_frames_count == 0:
                video_scenes.append({
                    "topic_id": topic_manifest["topic_id"],
                    "scenes": []
                })
                frame_offset += topic_frames_count
                continue
            topic_emb = all_emb[frame_offset : frame_offset + topic_frames_count]
            topic_ts  = topic_manifest["timestamps"]
            topic_paths = [os.path.join(BASE_DIR, p) for p in topic_manifest["frame_paths"]]
            topic_scenes = cluster_frames(
                topic_emb,
                topic_ts,
                topic_paths,
                method=CLUSTER_METHOD
            )
            video_scenes.append({
                "topic_id": topic_manifest["topic_id"],
                "start_time": topic_manifest["start_time"],
                "end_time": topic_manifest["end_time"],
                "scenes": topic_scenes
            })
            frame_offset += topic_frames_count
        # Save per-video scenes
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(video_scenes, f, ensure_ascii=False, indent=2)
        total_scenes = sum(len(t["scenes"]) for t in video_scenes)
        return stem, True, f"clustered → {total_scenes} scenes across {len(video_scenes)} topics"
    except Exception as e:
        return stem, False, f"error: {str(e)[:180]}"
# ====================== MAIN ======================
if __name__ == "__main__":
    start_time = datetime.now()
    print(f"\n=== Phase 3.3 — Scene Clustering inside Topics ({start_time.strftime('%Y-%m-%d %H:%M:%S')}) ===\n")
    print(f"Method: {CLUSTER_METHOD.upper()} | Min cluster size: {HDBSCAN_MIN_CLUSTER_SIZE if CLUSTER_METHOD == 'hdbscan' else 'auto'}\n")
    manifests = []
    for vid_dir in os.listdir(FRAMES_ROOT_DIR):
        manifest_path = os.path.join(FRAMES_ROOT_DIR, vid_dir, "frames_manifest.json")
        emb_path = os.path.join(FRAME_EMB_DIR, f"{vid_dir}.npy")
        if os.path.exists(manifest_path) and os.path.exists(emb_path):
            manifests.append((vid_dir, manifest_path, emb_path))
    print(f"Found {len(manifests)} videos ready for scene clustering")
    success = 0
    failed = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_one_topic_group, m) for m in manifests]
        for future in tqdm(as_completed(futures), total=len(manifests), desc="Scene Clustering"):
            name, ok, msg = future.result()
            if ok:
                success += 1
                tqdm.write(f"OK   {name}: {msg}")
            else:
                failed.append((name, msg))
                tqdm.write(f"FAIL {name}: {msg}")
    duration = datetime.now() - start_time
    print(f"\n=== Phase 3.3 Complete ===")
    print(f"Duration: {duration}")
    print(f"Succeeded: {success} / {len(manifests)}")
    if failed:
        print(f"Failed: {len(failed)}")
        print("First few:")
        for n, m in failed[:5]:
            print(f"  - {n}: {m}")
    scene_files = len([f for f in os.listdir(SCENES_DIR) if f.endswith('.json')])
    print(f"\nScenes saved: {scene_files} files → {SCENES_DIR}")
    print("\nNext: Phase 4 — Build hierarchical video memory store")
    print("   • topics.json (already have)")
    print("   • scenes.json (just created)")
    print("   • frame_embeddings.npy (per video)")
    print("   • transcript_embeddings.npy (per video)")
    print(" → Combine into structured video_memory/ folder\n")