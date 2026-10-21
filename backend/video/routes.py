"""
Video Routes - Upload, Processing, Status
Jobs are queued and processed by worker.py (run separately)
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import json
import uuid
import shutil
from pathlib import Path
from datetime import datetime

from .pipeline import MEMORY_ROOT, UPLOADS_DIR, IMPLEMENTATION_MEMORY
from .search import search_global_topics

router = APIRouter()

# Job status storage
JOBS_FILE = Path(__file__).parent.parent / "data" / "jobs.json"
BASE_URL = "http://localhost:8000"  # For rewriting frame URLs


def load_jobs() -> dict:
    """Load jobs from JSON file"""
    if JOBS_FILE.exists():
        with open(JOBS_FILE, "r") as f:
            return json.load(f)
    return {"jobs": {}}


def save_jobs(data: dict):
    """Save jobs to JSON file"""
    JOBS_FILE.parent.mkdir(exist_ok=True, parents=True)
    with open(JOBS_FILE, "w") as f:
        json.dump(data, f, indent=2)


# Response models
class UploadResponse(BaseModel):
    video_id: str
    status: str
    message: str


class VideoStatus(BaseModel):
    video_id: str
    status: str
    current_phase: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


class VideoInfo(BaseModel):
    video_id: str
    status: str
    has_memory: bool

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    video_ids: Optional[list[str]] = None

class QueryResult(BaseModel):
    video_id: str
    topic_id: int
    score: float
    text: str
    start_time: float
    end_time: float

class EvidenceItem(BaseModel):
    video_id: str
    topic_id: int
    scene_id: int
    text: str
    start_time: float
    end_time: float
    score: float
    frame_url: str
    visual_description: str
    original_score: Optional[float] = None
    popularity_score: Optional[float] = None

class SceneSimilarity(BaseModel):
    video_id: str
    scene_id: int
    score: float

class QueryMetrics(BaseModel):
    num_queries_formed: int
    queries: list[str]
    num_videos_retrieved: int
    num_topics_retrieved: int
    num_scenes_retrieved: int
    num_frames_retrieved: int
    scene_similarities: list[SceneSimilarity]

class DisplaySceneItem(BaseModel):
    video_id: str
    type: str
    topic_id: int = 0
    scene_ids: list[int]
    start_time: float
    end_time: float
    text: str
    visual_description: str
    frame_url: str
    score: float
    original_score: Optional[float] = None
    popularity_score: Optional[float] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    evidence: list[EvidenceItem]
    metrics: QueryMetrics
    display_scenes: list[DisplaySceneItem]


@router.post("/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file and queue for processing.
    Processing is handled by worker.py (run separately).
    Use /status/{video_id} to check progress.
    """
    # Validate file type
    allowed_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate unique video ID
    video_id = str(uuid.uuid4())[:8] + "_" + Path(file.filename).stem[:20]
    video_id = video_id.replace(" ", "_")
    
    # Save uploaded file
    UPLOADS_DIR.mkdir(exist_ok=True, parents=True)
    upload_path = UPLOADS_DIR / f"{video_id}{file_ext}"
    
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create job entry (worker.py will pick it up)
    data = load_jobs()
    data["jobs"][video_id] = {
        "video_id": video_id,
        "filename": file.filename,
        "status": "queued",
        "current_phase": "Waiting for worker...",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    save_jobs(data)
    
    return UploadResponse(
        video_id=video_id,
        status="queued",
        message="Video uploaded. Run worker.py to start processing."
    )


@router.get("/status/{video_id}", response_model=VideoStatus)
async def get_video_status(video_id: str):
    """Get the processing status of a video"""
    data = load_jobs()
    
    if video_id not in data["jobs"]:
        raise HTTPException(status_code=404, detail="Video not found")
    
    job = data["jobs"][video_id]
    
    return VideoStatus(
        video_id=video_id,
        status=job["status"],
        current_phase=job.get("current_phase"),
        error=job.get("error"),
        created_at=job["created_at"],
        updated_at=job["updated_at"]
    )


@router.get("/list")
async def list_videos():
    """List all videos and their status (uploaded + pre-processed dataset)"""
    data = load_jobs()
    
    videos = []
    
    # 1. User-uploaded videos
    for video_id, job in data.get("jobs", {}).items():
        memory_dir = MEMORY_ROOT / video_id
        has_memory = (memory_dir / "video_memory.json").exists()
        
        videos.append({
            "video_id": video_id,
            "filename": job.get("filename", "unknown"),
            "status": job["status"],
            "has_memory": has_memory,
            "created_at": job["created_at"],
            "source": "uploaded"
        })
    
    # 2. Pre-processed dataset videos from Implementation
    uploaded_ids = set(data.get("jobs", {}).keys())
    if IMPLEMENTATION_MEMORY.exists():
        for entry in IMPLEMENTATION_MEMORY.iterdir():
            if entry.is_dir() and (entry / "video_memory.json").exists():
                vid = entry.name
                if vid not in uploaded_ids:  # avoid duplicates
                    videos.append({
                        "video_id": vid,
                        "filename": vid,
                        "status": "completed",
                        "has_memory": True,
                        "created_at": "2026-01-01T00:00:00",
                        "source": "dataset"
                    })
    
    # Sort: uploaded first (newest), then dataset
    videos.sort(key=lambda x: (0 if x["source"] == "uploaded" else 1, x["created_at"]), reverse=False)
    # Re-sort so uploaded are newest-first at top, dataset alphabetical after
    uploaded = sorted([v for v in videos if v["source"] == "uploaded"], key=lambda x: x["created_at"], reverse=True)
    dataset = sorted([v for v in videos if v["source"] == "dataset"], key=lambda x: x["video_id"])
    
    return {"videos": uploaded + dataset}


@router.get("/memory/{video_id}")
async def get_video_memory(video_id: str):
    """Get the processed video memory structure (checks uploaded + dataset)"""
    # Check user uploads first, then Implementation dataset
    memory_path = MEMORY_ROOT / video_id / "video_memory.json"
    if not memory_path.exists() and IMPLEMENTATION_MEMORY.exists():
        memory_path = IMPLEMENTATION_MEMORY / video_id / "video_memory.json"
    
    if not memory_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Video memory not found. Video may still be processing."
        )
    
    with open(memory_path, 'r', encoding='utf-8') as f:
        memory = json.load(f)
    
    return memory


@router.get("/topics/{video_id}")
async def get_video_topics(video_id: str):
    """Get rich topic/scene data with API-relative frame URLs"""
    memory_path = MEMORY_ROOT / video_id / "video_memory.json"
    is_dataset = False
    
    if not memory_path.exists() and IMPLEMENTATION_MEMORY.exists():
        memory_path = IMPLEMENTATION_MEMORY / video_id / "video_memory.json"
        is_dataset = True
    
    if not memory_path.exists():
        raise HTTPException(status_code=404, detail="Video memory not found")
        
    with open(memory_path, 'r', encoding='utf-8') as f:
        memory = json.load(f)
        
    # Rewrite frame paths to API URLs
    for topic in memory.get("topics", []):
        topic_id = topic.get("topic_id")
        for scene in topic.get("scenes", []):
            if "frame_paths" in scene:
                new_paths = []
                for fp in scene["frame_paths"]:
                    # Extract filename from path (handles both Windows and POSIX)
                    fname = Path(fp).name
                    # URL format: /api/video/frame/{video_id}/{topic_id}/{filename}
                    new_paths.append(f"{BASE_URL}/api/video/frame/{video_id}/{topic_id}/{fname}")
                scene["frame_urls"] = new_paths
                
    return memory

@router.get("/frame/{video_id}/{topic_id}/{filename}")
async def get_frame(video_id: str, topic_id: int, filename: str):
    """Serve a specific frame image"""
    # Try user uploads first
    topic_str = f"topic_{topic_id:03d}"
    frame_path = MEMORY_ROOT / video_id / "frames" / topic_str / filename
    
    if not frame_path.exists():
        # Try implementation dataset
        # Implementation folder structure: frames/{video_id}/topic_{idx}/{filename}
        implementation_frame = IMPLEMENTATION_MEMORY.parent / "frames" / video_id / topic_str / filename
        if implementation_frame.exists():
            frame_path = implementation_frame
            
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")
        
    return FileResponse(frame_path)

@router.delete("/{video_id}")
async def delete_video(video_id: str):
    """Delete a video and all its processed data (dataset videos are protected)"""
    # Block deletion of dataset videos
    if IMPLEMENTATION_MEMORY.exists() and (IMPLEMENTATION_MEMORY / video_id).exists():
        raise HTTPException(
            status_code=403,
            detail="Cannot delete pre-processed dataset videos."
        )
    
    # Remove from jobs
    data = load_jobs()
    if video_id in data["jobs"]:
        del data["jobs"][video_id]
        save_jobs(data)
    
    # Remove upload
    for ext in [".mp4", ".mkv", ".avi", ".mov", ".webm"]:
        upload_path = UPLOADS_DIR / f"{video_id}{ext}"
        if upload_path.exists():
            upload_path.unlink()
    
    # Remove memory folder
    memory_dir = MEMORY_ROOT / video_id
    if memory_dir.exists():
        shutil.rmtree(memory_dir)
    
    return {"message": f"Video {video_id} deleted successfully"}

@router.post("/query", response_model=QueryResponse)
async def global_query(request: QueryRequest):
    """Search globally across all videos and return rich RAG results"""
    from .search import run_full_rag_pipeline
    
    result = run_full_rag_pipeline(request.query, video_ids=request.video_ids)
    
    # Rewrite evidence frame URLs to API routes
    for item in result["evidence"]:
        if item["frame_url"]:
            fname = Path(item["frame_url"]).name
            item["frame_url"] = f"{BASE_URL}/api/video/frame/{item['video_id']}/{item['topic_id']}/{fname}"
            
    for item in result.get("display_scenes", []):
        if item.get("frame_url"):
            fname = Path(item["frame_url"]).name
            topic_id = item.get("topic_id", 0) if isinstance(item, dict) else item.topic_id
            # Python dictionaries don't have .topic_id if it's a dict, but pydantic model serialization might be different
            # It's returning a dict from search.py!
            topic_id = item.get("topic_id", 0)
            item["frame_url"] = f"{BASE_URL}/api/video/frame/{item['video_id']}/{topic_id}/{fname}"
            
    return QueryResponse(
        query=request.query,
        answer=result["answer"],
        evidence=result["evidence"],
        metrics=result.get("metrics", {}),
        display_scenes=result.get("display_scenes", [])
    )

@router.get("/stream/{video_id}")
async def stream_video(video_id: str):
    """Stream video file for frontend playback"""
    for ext in [".mp4", ".mkv", ".avi", ".mov", ".webm"]:
        upload_path = UPLOADS_DIR / f"{video_id}{ext}"
        if upload_path.exists():
            return FileResponse(upload_path)
            
    dataset_path = IMPLEMENTATION_MEMORY.parent / "raw_full_videos" / f"{video_id}.mp4"
    if dataset_path.exists():
        return FileResponse(dataset_path)
        
    raise HTTPException(status_code=404, detail="Video file not found")


# ===========================================================================
# POPULARITY ENDPOINTS  (additive — does not touch any existing routes)
# ===========================================================================

@router.get("/popularity")
async def get_all_popularity():
    """Return simulated likes / subscribers / engagement for all dataset videos"""
    from .popularity import load_popularity
    popularity = load_popularity()
    return {"popularity": popularity, "count": len(popularity)}


@router.get("/popularity/{video_id}")
async def get_video_popularity_endpoint(video_id: str):
    """Return popularity data for a single video"""
    from .popularity import load_popularity
    popularity = load_popularity()
    if video_id not in popularity:
        raise HTTPException(status_code=404, detail="Popularity data not found for this video")
    return popularity[video_id]
