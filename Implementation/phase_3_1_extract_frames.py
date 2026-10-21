# phase_3_1_extract_frames.py
import os
import json
import cv2
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
# ====================== CONFIG ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_VIDEO_DIR = os.path.join(BASE_DIR, "raw_full_videos")     # your full .mp4 files
TOPICS_DIR    = os.path.join(BASE_DIR, "topics_json")
FRAMES_DIR    = os.path.join(BASE_DIR, "frames")               # new folder
os.makedirs(FRAMES_DIR, exist_ok=True)
FPS_TARGET = 1.0                      # 1 frame per second
MAX_WORKERS = 6                       # adjust based on disk I/O & CPU
# ====================== HELPERS ======================
def extract_frames_for_topic(video_path: str, topic: dict, output_topic_dir: str) -> dict:
    """
    Extract frames at ~1 FPS within topic time range
    Returns manifest: {"frame_paths": [...], "timestamps": [...], "topic_id": int}
    """
    start_sec = topic["start_time"]
    end_sec   = topic["end_time"]
    topic_id  = topic["topic_id"]
    topic_dir = os.path.join(output_topic_dir, f"topic_{topic_id:03d}")
    os.makedirs(topic_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "cannot open video"}
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video <= 0:
        fps_video = 30.0  # fallback
    frame_interval = int(round(fps_video / FPS_TARGET))  # e.g. every 30th frame if 30 FPS
    frame_paths = []
    timestamps = []
    current_sec = 0.0
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_sec = frame_idx / fps_video
        if current_sec >= end_sec:
            break
        if current_sec >= start_sec and frame_idx % frame_interval == 0:
            ts_str = f"{current_sec:06.2f}"
            frame_path = os.path.join(topic_dir, f"frame_{ts_str}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            timestamps.append(round(current_sec, 2))
        frame_idx += 1
    cap.release()
    return {
        "topic_id": topic_id,
        "start_time": start_sec,
        "end_time": end_sec,
        "frame_count": len(frame_paths),
        "timestamps": timestamps,
        "frame_paths": [os.path.relpath(p, BASE_DIR) for p in frame_paths]  # relative for portability
    }
def process_one_video(args):
    """
    Process all topics for one video
    args: (stem, topics_list)
    """
    stem, topics = args
    video_path = os.path.join(RAW_VIDEO_DIR, f"{stem}.mp4")
    if not os.path.exists(video_path):
        return stem, False, "video file missing"
    video_topic_dir = os.path.join(FRAMES_DIR, stem)
    os.makedirs(video_topic_dir, exist_ok=True)
    manifests = []
    success_count = 0
    for topic in topics:
        manifest = extract_frames_for_topic(video_path, topic, video_topic_dir)
        if "error" not in manifest:
            manifests.append(manifest)
            success_count += 1
        else:
            return stem, False, f"failed topic {topic['topic_id']}: {manifest['error']}"
    # Save manifest per video
    manifest_path = os.path.join(video_topic_dir, "frames_manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifests, f, indent=2)
    return stem, True, f"extracted {sum(m['frame_count'] for m in manifests)} frames across {len(topics)} topics"
# ====================== MAIN ======================
if __name__ == "__main__":
    start_time = datetime.now()
    print(f"\n=== Phase 3.1 — Frame Extraction (1 FPS per topic) ({start_time.strftime('%Y-%m-%d %H:%M:%S')}) ===\n")
    topics_jsons = [
        os.path.join(TOPICS_DIR, f)
        for f in os.listdir(TOPICS_DIR)
        if f.endswith('.json')
    ]
    print(f"Found {len(topics_jsons)} topic JSON files")
    tasks = []
    for jpath in topics_jsons:
        stem = Path(jpath).stem
        with open(jpath, 'r', encoding='utf-8') as f:
            topics = json.load(f)
        if topics:
            tasks.append((stem, topics))
    print(f"Processing {len(tasks)} videos with topics")
    success = 0
    failed = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_one_video, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Frame Extraction"):
            name, ok, msg = future.result()
            if ok:
                success += 1
                tqdm.write(f"OK   {name}: {msg}")
            else:
                failed.append((name, msg))
                tqdm.write(f"FAIL {name}: {msg}")
    duration = datetime.now() - start_time
    print(f"\n=== Phase 3.1 Complete ===")
    print(f"Duration: {duration}")
    print(f"Succeeded: {success} / {len(tasks)}")
    if failed:
        print(f"Failed: {len(failed)}")
        print("First few failures:")
        for n, m in failed[:5]:
            print(f"  - {n}: {m}")
    # Quick stats
    total_frames = 0
    for vid_dir in os.listdir(FRAMES_DIR):
        if os.path.isdir(os.path.join(FRAMES_DIR, vid_dir)):
            for root, _, files in os.walk(os.path.join(FRAMES_DIR, vid_dir)):
                total_frames += len([f for f in files if f.endswith('.jpg')])
    print(f"\nTotal frames extracted: ~{total_frames:,}")
    print(f"Frames root: {FRAMES_DIR}")
    print("\nNext: Phase 3.2 — Embed frames with CLIP / ViT / similar")
    print("   • We can use open-clip or huggingface transformers for ViT/CLIP embeddings")
    print("   • Output: frame_embeddings/{youtube_id}.npy (all frames) or per-topic .npy\n")