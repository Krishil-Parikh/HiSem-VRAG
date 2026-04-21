"""
Video Processing Pipeline - EXACT same logic as Implementation phases 1-4

This module contains the same processing steps from:
- phase_1_1: Extract WAV audio
- phase_1_2: WhisperX ASR transcription  
- phase_1_3: Chunk transcripts (20-40s + similarity merge)
- phase_1_4: Embed chunks (MiniLM)
- phase_2_1: Topic segmentation (cosine threshold)
- phase_3_1: Extract frames (1 FPS)
- phase_3_2: Embed frames (SigLIP)
- phase_3_3: Scene clustering (HDBSCAN)
- phase_4: Build memory store
"""

import os
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional
import numpy as np
import cv2
from PIL import Image
import torch
from tqdm import tqdm

# Lazy imports for heavy models (loaded on first use)
_whisper_model = None
_align_model = None
_align_metadata = None
_text_embedder = None
_vision_processor = None
_vision_model = None

# ====================== CONFIG ======================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
MEMORY_ROOT = BASE_DIR / "video_memory"

# Pre-processed dataset from Implementation (read-only)
IMPLEMENTATION_DIR = BASE_DIR.parent / "Implementation" / "mini_dataset_youcook2"
IMPLEMENTATION_MEMORY = IMPLEMENTATION_DIR / "video_memory"

# Create directories
for d in [DATA_DIR, UPLOADS_DIR, MEMORY_ROOT]:
    d.mkdir(exist_ok=True, parents=True)

# Model configs (same as Implementation)
WHISPER_MODEL_NAME = "large-v3"
TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VISION_MODEL_NAME = "google/siglip-so400m-patch14-384"

# Processing params (same as Implementation)
MIN_CHUNK_SEC = 20.0
MAX_CHUNK_SEC = 40.0
SIM_THRESHOLD = 0.75
COSINE_THRESHOLD = 0.68
MIN_TOPIC_DURATION_SEC = 60.0
FPS_TARGET = 1.0
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ====================== MODEL LOADING ======================
def get_whisper_models():
    """Load WhisperX models (lazy loading)"""
    global _whisper_model, _align_model, _align_metadata
    
    if _whisper_model is None:
        import whisperx
        
        # PyTorch 2.6+ security fix
        _orig_torch_load = torch.load
        def _monkey_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return _orig_torch_load(*args, **kwargs)
        torch.load = _monkey_torch_load
        
        compute_type = "float16" if DEVICE == "cuda" else "float32"
        
        print(f"Loading WhisperX model '{WHISPER_MODEL_NAME}' on {DEVICE}...")
        _whisper_model = whisperx.load_model(
            WHISPER_MODEL_NAME,
            device=DEVICE,
            compute_type=compute_type,
            language="en",
            vad_method="silero"
        )
        
        print("Loading alignment model...")
        _align_model, _align_metadata = whisperx.load_align_model(
            language_code="en",
            device=DEVICE
        )
    
    return _whisper_model, _align_model, _align_metadata


def get_text_embedder():
    """Load sentence transformer (lazy loading)"""
    global _text_embedder
    
    if _text_embedder is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading text embedder '{TEXT_EMBEDDING_MODEL}' on {DEVICE}...")
        _text_embedder = SentenceTransformer(TEXT_EMBEDDING_MODEL, device=DEVICE)
    
    return _text_embedder


def get_vision_model():
    """Load vision model (lazy loading)"""
    global _vision_processor, _vision_model
    
    if _vision_model is None:
        from transformers import AutoProcessor, AutoModel
        print(f"Loading vision model '{VISION_MODEL_NAME}' on {DEVICE}...")
        _vision_processor = AutoProcessor.from_pretrained(VISION_MODEL_NAME)
        _vision_model = AutoModel.from_pretrained(VISION_MODEL_NAME).to(DEVICE)
        _vision_model.eval()
    
    return _vision_processor, _vision_model


# ====================== PHASE 1.1: EXTRACT AUDIO ======================
def extract_audio_wav(video_path: Path, output_dir: Path, progress_callback: Optional[Callable] = None) -> Path:
    """
    Extract 16kHz mono WAV from video using ffmpeg
    Same as phase_1_1_full_videos_and_wav.py
    """
    if progress_callback:
        progress_callback("Extracting audio from video...")
    
    wav_path = output_dir / "audio.wav"
    
    # Find ffmpeg - try shutil.which first, then common locations
    import shutil as sh
    ffmpeg_path = sh.which('ffmpeg')
    if not ffmpeg_path:
        # Try common Windows locations
        common_paths = [
            r'C:\ProgramData\chocolatey\bin\ffmpeg.exe',
            r'C:\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
        ]
        for p in common_paths:
            if Path(p).exists():
                ffmpeg_path = p
                break
    
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg and add to PATH.")
    
    cmd = [
        ffmpeg_path, '-y',
        '-i', str(video_path),
        '-ar', '16000',
        '-ac', '1',
        '-c:a', 'pcm_s16le',
        str(wav_path)
    ]
    
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if not wav_path.exists():
        raise RuntimeError("Failed to extract audio")
    
    return wav_path


# ====================== PHASE 1.2: ASR TRANSCRIPTION ======================
def transcribe_whisperx(wav_path: Path, output_dir: Path, progress_callback: Optional[Callable] = None) -> Path:
    """
    Transcribe audio using WhisperX + Silero VAD
    Same as phase_1_2_asr_whisperx.py
    """
    import whisperx
    
    if progress_callback:
        progress_callback("Transcribing audio with WhisperX...")
    
    whisper_model, align_model, align_metadata = get_whisper_models()
    
    # Load audio
    audio = whisperx.load_audio(str(wav_path))
    
    # Transcribe
    result = whisper_model.transcribe(audio, batch_size=8, language="en")
    
    # Align
    aligned_result = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        DEVICE,
        return_char_alignments=False
    )
    
    # Clean segments
    segments = []
    for seg in aligned_result.get("segments", []):
        if all(k in seg for k in ["text", "start", "end"]):
            segments.append({
                "text": seg["text"].strip(),
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2)
            })
    
    # Save
    asr_path = output_dir / "asr_transcript.json"
    with open(asr_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    
    return asr_path


# ====================== PHASE 1.3: CHUNK TRANSCRIPTS ======================
def chunk_transcripts(asr_path: Path, output_dir: Path, progress_callback: Optional[Callable] = None) -> Path:
    """
    Chunk transcripts into 20-40s windows + similarity merge
    Same as phase_1_3_chunk_transcripts.py
    """
    from sentence_transformers import util
    
    if progress_callback:
        progress_callback("Chunking transcripts...")
    
    embedder = get_text_embedder()
    
    with open(asr_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    
    if not segments:
        raise ValueError("Empty transcript")
    
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
    if len(chunks) > 1:
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embedder.encode(chunk_texts, convert_to_tensor=True, show_progress_bar=False)
        final_chunks = []
        current_merged = chunks[0].copy()
        
        for i in range(1, len(chunks)):
            sim = util.cos_sim(embeddings[i-1], embeddings[i]).item()
            if sim >= SIM_THRESHOLD:
                current_merged["text"] += " " + chunks[i]["text"]
                current_merged["end"] = max(current_merged["end"], chunks[i]["end"])
                current_merged["original_segment_ids"].extend(chunks[i]["original_segment_ids"])
            else:
                final_chunks.append(current_merged)
                current_merged = chunks[i].copy()
        
        final_chunks.append(current_merged)
        
        for idx, chunk in enumerate(final_chunks):
            chunk["chunk_id"] = idx
    else:
        final_chunks = chunks
    
    # Save
    chunks_path = output_dir / "chunked_transcript.json"
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(final_chunks, f, ensure_ascii=False, indent=2)
    
    return chunks_path


# ====================== PHASE 1.4: EMBED CHUNKS ======================
def embed_chunks(chunks_path: Path, output_dir: Path, progress_callback: Optional[Callable] = None) -> Path:
    """
    Embed transcript chunks using MiniLM
    Same as phase_1_4_embed_chunks.py
    """
    if progress_callback:
        progress_callback("Embedding transcript chunks...")
    
    embedder = get_text_embedder()
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    texts = [chunk["text"].strip() for chunk in chunks if chunk.get("text", "").strip()]
    
    if not texts:
        raise ValueError("No valid text in chunks")
    
    embeddings = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Save
    emb_path = output_dir / "transcript_embeddings.npy"
    np.save(emb_path, embeddings)
    
    return emb_path


# ====================== PHASE 2.1: TOPIC SEGMENTATION ======================
def segment_topics(chunks_path: Path, emb_path: Path, output_dir: Path, progress_callback: Optional[Callable] = None) -> Path:
    """
    Segment into topics using cosine similarity threshold
    Same as phase_2_1_topic_segmentation.py
    """
    if progress_callback:
        progress_callback("Segmenting topics...")
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    embeddings = np.load(emb_path)
    
    if len(embeddings) != len(chunks):
        raise ValueError(f"Embedding count mismatch ({len(embeddings)} vs {len(chunks)} chunks)")
    
    # Compute cosine similarities between consecutive chunks
    similarities = []
    for i in range(1, len(embeddings)):
        sim = np.dot(embeddings[i-1], embeddings[i]) / (
            np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
        )
        similarities.append(sim)
    
    # Detect boundaries
    boundaries = [0]
    for i, sim in enumerate(similarities):
        if sim < COSINE_THRESHOLD:
            boundaries.append(i + 1)
    boundaries.append(len(chunks))
    
    # Build initial topics
    initial_topics = []
    for start_idx, end_idx in zip(boundaries[:-1], boundaries[1:]):
        topic_chunks = chunks[start_idx:end_idx]
        topic_start = topic_chunks[0]["start"]
        topic_end = topic_chunks[-1]["end"]
        topic_dur = topic_end - topic_start
        topic_text = " ".join(c["text"] for c in topic_chunks)
        initial_topics.append({
            "start_time": topic_start,
            "end_time": topic_end,
            "duration": round(topic_dur, 2),
            "chunk_ids": list(range(start_idx, end_idx)),
            "text": topic_text.strip()
        })
    
    # Enforce min duration
    topics = []
    current = initial_topics[0].copy()
    for next_topic in initial_topics[1:]:
        if current["duration"] < MIN_TOPIC_DURATION_SEC:
            current["end_time"] = next_topic["end_time"]
            current["duration"] = current["end_time"] - current["start_time"]
            current["chunk_ids"].extend(next_topic["chunk_ids"])
            current["text"] += " " + next_topic["text"]
        else:
            topics.append(current)
            current = next_topic.copy()
    
    if current["duration"] >= MIN_TOPIC_DURATION_SEC or not topics:
        topics.append(current)
    else:
        if topics:
            topics[-1]["end_time"] = current["end_time"]
            topics[-1]["duration"] = topics[-1]["end_time"] - topics[-1]["start_time"]
            topics[-1]["chunk_ids"].extend(current["chunk_ids"])
            topics[-1]["text"] += " " + current["text"]
    
    # Assign topic_id and representative embedding
    for idx, topic in enumerate(topics):
        topic["topic_id"] = idx
        topic_chunk_embs = embeddings[topic["chunk_ids"]]
        topic["representative_embedding"] = np.mean(topic_chunk_embs, axis=0).tolist()
    
    # Save
    topics_path = output_dir / "topics.json"
    with open(topics_path, 'w', encoding='utf-8') as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)
    
    return topics_path


# ====================== PHASE 3.1: EXTRACT FRAMES ======================
def extract_frames(video_path: Path, topics_path: Path, output_dir: Path, progress_callback: Optional[Callable] = None) -> Path:
    """
    Extract frames at 1 FPS within topic time ranges
    Same as phase_3_1_extract_frames.py
    """
    if progress_callback:
        progress_callback("Extracting frames...")
    
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    with open(topics_path, 'r', encoding='utf-8') as f:
        topics = json.load(f)
    
    manifests = []
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video <= 0:
        fps_video = 30.0
    
    frame_interval = int(round(fps_video / FPS_TARGET))
    
    for topic in topics:
        start_sec = topic["start_time"]
        end_sec = topic["end_time"]
        topic_id = topic["topic_id"]
        
        topic_dir = frames_dir / f"topic_{topic_id:03d}"
        topic_dir.mkdir(exist_ok=True)
        
        frame_paths = []
        timestamps = []
        
        # Seek to start
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
        
        frame_idx = int(start_sec * fps_video)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            current_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            if current_sec >= end_sec:
                break
            
            if current_sec >= start_sec and frame_idx % frame_interval == 0:
                ts_str = f"{current_sec:06.2f}"
                frame_path = topic_dir / f"frame_{ts_str}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(str(frame_path.relative_to(output_dir)))
                timestamps.append(round(current_sec, 2))
            
            frame_idx += 1
        
        manifests.append({
            "topic_id": topic_id,
            "start_time": start_sec,
            "end_time": end_sec,
            "frame_count": len(frame_paths),
            "timestamps": timestamps,
            "frame_paths": frame_paths
        })
    
    cap.release()
    
    # Save manifest
    manifest_path = output_dir / "frames_manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifests, f, indent=2)
    
    return manifest_path


# ====================== PHASE 3.2: EMBED FRAMES ======================
def embed_frames(manifest_path: Path, output_dir: Path, progress_callback: Optional[Callable] = None) -> Path:
    """
    Embed frames using SigLIP vision model
    Same as phase_3_2_embed_frames.py
    """
    if progress_callback:
        progress_callback("Embedding frames with SigLIP...")
    
    processor, model = get_vision_model()
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifests = json.load(f)
    
    all_embeddings = []
    
    for topic_manifest in manifests:
        topic_frames = []
        for rel_path in topic_manifest["frame_paths"]:
            full_path = output_dir / rel_path
            if full_path.exists():
                img = Image.open(full_path).convert("RGB")
                topic_frames.append(img)
        
        if not topic_frames:
            continue
        
        # Embed in batches
        batch_size = 32
        for i in range(0, len(topic_frames), batch_size):
            batch_imgs = topic_frames[i:i + batch_size]
            inputs = processor(images=batch_imgs, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
            all_embeddings.append(outputs.cpu().numpy())
    
    if not all_embeddings:
        raise ValueError("No frames found")
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # Normalize
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    all_embeddings /= np.maximum(norms, 1e-8)
    
    # Save
    emb_path = output_dir / "frame_embeddings.npy"
    np.save(emb_path, all_embeddings)
    
    return emb_path


# ====================== PHASE 3.3: SCENE CLUSTERING ======================
def cluster_scenes(manifest_path: Path, frame_emb_path: Path, output_dir: Path, progress_callback: Optional[Callable] = None) -> Path:
    """
    Cluster frames into scenes using HDBSCAN
    Same as phase_3_3_scene_clustering.py
    """
    import hdbscan
    
    if progress_callback:
        progress_callback("Clustering scenes...")
    
    all_emb = np.load(frame_emb_path)
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifests = json.load(f)
    
    video_scenes = []
    frame_offset = 0
    
    for topic_manifest in manifests:
        topic_frames_count = topic_manifest["frame_count"]
        
        if topic_frames_count == 0:
            video_scenes.append({
                "topic_id": topic_manifest["topic_id"],
                "scenes": []
            })
            continue
        
        topic_emb = all_emb[frame_offset:frame_offset + topic_frames_count]
        topic_ts = topic_manifest["timestamps"]
        topic_paths = topic_manifest["frame_paths"]
        
        # Cluster
        if len(topic_emb) < 3:
            scenes = [{
                "scene_id": 0,
                "start_time": topic_ts[0] if topic_ts else 0.0,
                "end_time": topic_ts[-1] if topic_ts else 0.0,
                "frame_count": len(topic_emb),
                "frame_ids": list(range(len(topic_emb))),
                "frame_paths": topic_paths,
                "cluster_label": 0
            }]
        else:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                min_samples=HDBSCAN_MIN_SAMPLES,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            labels = clusterer.fit_predict(topic_emb)
            
            unique_labels = sorted(set(labels) - {-1})
            scenes = []
            
            for label in unique_labels:
                mask = (labels == label)
                scene_timestamps = np.array(topic_ts)[mask]
                scene_paths = [topic_paths[i] for i in range(len(mask)) if mask[i]]
                
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
            
            scenes.sort(key=lambda x: x["start_time"])
            for i, s in enumerate(scenes):
                s["scene_id"] = i
        
        video_scenes.append({
            "topic_id": topic_manifest["topic_id"],
            "start_time": topic_manifest["start_time"],
            "end_time": topic_manifest["end_time"],
            "scenes": scenes
        })
        
        frame_offset += topic_frames_count
    
    # Save
    scenes_path = output_dir / "scenes.json"
    with open(scenes_path, 'w', encoding='utf-8') as f:
        json.dump(video_scenes, f, ensure_ascii=False, indent=2)
    
    return scenes_path


# ====================== PHASE 4: BUILD MEMORY STORE ======================
def build_memory_store(output_dir: Path, progress_callback: Optional[Callable] = None) -> Path:
    """
    Build unified hierarchical memory structure
    Same as phase_4_build_memory_store.py
    """
    if progress_callback:
        progress_callback("Building memory store...")
    
    topics_path = output_dir / "topics.json"
    scenes_path = output_dir / "scenes.json"
    
    with open(topics_path, 'r', encoding='utf-8') as f:
        topics = json.load(f)
    
    with open(scenes_path, 'r', encoding='utf-8') as f:
        scenes_per_topic = json.load(f)
    
    video_id = output_dir.name
    
    video_memory = {
        "video_id": video_id,
        "num_topics": len(topics),
        "topics": []
    }
    
    for i, topic in enumerate(topics):
        topic_copy = topic.copy()
        topic_copy["prev_topic_id"] = i - 1 if i > 0 else None
        topic_copy["next_topic_id"] = i + 1 if i < len(topics) - 1 else None
        
        matching_scenes = next((t["scenes"] for t in scenes_per_topic if t["topic_id"] == topic["topic_id"]), [])
        topic_copy["scenes"] = []
        
        for j, scene in enumerate(matching_scenes):
            scene_copy = scene.copy()
            scene_copy["prev_scene_id"] = j - 1 if j > 0 else None
            scene_copy["next_scene_id"] = j + 1 if j < len(matching_scenes) - 1 else None
            scene_copy["topic_id"] = topic["topic_id"]
            topic_copy["scenes"].append(scene_copy)
        
        video_memory["topics"].append(topic_copy)
    
    # Save unified json
    memory_path = output_dir / "video_memory.json"
    with open(memory_path, 'w', encoding='utf-8') as f:
        json.dump(video_memory, f, ensure_ascii=False, indent=2)
    
    return memory_path


# ====================== FULL PIPELINE ======================
def process_video_full_pipeline(
    video_path: Path,
    video_id: str,
    progress_callback: Optional[Callable] = None
) -> dict:
    """
    Run the complete video processing pipeline (phases 1-4)
    
    Args:
        video_path: Path to the uploaded video file
        video_id: Unique identifier for this video
        progress_callback: Optional callback for progress updates
    
    Returns:
        dict with processing results and paths
    """
    output_dir = MEMORY_ROOT / video_id
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Copy video to output dir
    video_dest = output_dir / f"source{video_path.suffix}"
    shutil.copy(video_path, video_dest)
    
    results = {"video_id": video_id, "output_dir": str(output_dir)}
    
    try:
        # Phase 1.1: Extract audio
        wav_path = extract_audio_wav(video_dest, output_dir, progress_callback)
        print("Phase 1.1 completed: ", wav_path)
        results["phase_1_1"] = "completed"
        
        # Phase 1.2: ASR
        asr_path = transcribe_whisperx(wav_path, output_dir, progress_callback)
        print("Phase 1.2 completed: ", asr_path)
        results["phase_1_2"] = "completed"
        
        # Phase 1.3: Chunk transcripts
        chunks_path = chunk_transcripts(asr_path, output_dir, progress_callback)
        print("Phase 1.3 completed: ", chunks_path)
        results["phase_1_3"] = "completed"
        
        # Phase 1.4: Embed chunks
        chunk_emb_path = embed_chunks(chunks_path, output_dir, progress_callback)
        print("Phase 1.4 completed: ", chunk_emb_path)
        results["phase_1_4"] = "completed"
        
        # Phase 2.1: Topic segmentation
        topics_path = segment_topics(chunks_path, chunk_emb_path, output_dir, progress_callback)
        print("Phase 2.1 completed: ", topics_path)
        results["phase_2_1"] = "completed"
        
        # Phase 3.1: Extract frames
        manifest_path = extract_frames(video_dest, topics_path, output_dir, progress_callback)
        print("Phase 3.1 completed: ", manifest_path)
        results["phase_3_1"] = "completed"
        
        # Phase 3.2: Embed frames
        frame_emb_path = embed_frames(manifest_path, output_dir, progress_callback)
        print("Phase 3.2 completed: ", frame_emb_path)
        results["phase_3_2"] = "completed"
        
        # Phase 3.3: Cluster scenes
        scenes_path = cluster_scenes(manifest_path, frame_emb_path, output_dir, progress_callback)
        print("Phase 3.3 completed: ", scenes_path)
        results["phase_3_3"] = "completed"
        
        # Phase 4: Build memory store
        memory_path = build_memory_store(output_dir, progress_callback)
        print("Phase 4 completed: ", memory_path)
        results["phase_4"] = "completed"
        
        results["status"] = "completed"
        results["memory_path"] = str(memory_path)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        results["status"] = "failed"
        results["error"] = str(e)
    
    return results
