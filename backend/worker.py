"""
Video Processing Worker - Runs as a separate process
Watches for queued jobs and processes them
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from video.pipeline import process_video_full_pipeline, MEMORY_ROOT, UPLOADS_DIR

JOBS_FILE = Path(__file__).parent / "data" / "jobs.json"


def load_jobs():
    if JOBS_FILE.exists():
        with open(JOBS_FILE, "r") as f:
            return json.load(f)
    return {"jobs": {}}


def save_jobs(data):
    JOBS_FILE.parent.mkdir(exist_ok=True, parents=True)
    with open(JOBS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def update_status(video_id, status, phase=None, error=None):
    data = load_jobs()
    if video_id in data["jobs"]:
        data["jobs"][video_id]["status"] = status
        data["jobs"][video_id]["updated_at"] = datetime.now().isoformat()
        if phase:
            data["jobs"][video_id]["current_phase"] = phase
        if error:
            data["jobs"][video_id]["error"] = error
        save_jobs(data)


def find_upload_file(video_id):
    for ext in [".mp4", ".mkv", ".avi", ".mov", ".webm"]:
        path = UPLOADS_DIR / f"{video_id}{ext}"
        if path.exists():
            return path
    return None


def process_one_job(video_id):
    """Process a single video job"""
    print(f"\n{'='*60}")
    print(f"Processing: {video_id}")
    print(f"{'='*60}\n")
    
    video_path = find_upload_file(video_id)
    if not video_path:
        update_status(video_id, "failed", error="Upload file not found")
        return
    
    def progress_callback(message):
        print(f"  [{video_id}] {message}")
        update_status(video_id, "processing", phase=message)
    
    try:
        update_status(video_id, "processing", phase="Starting pipeline...")
        
        result = process_video_full_pipeline(
            video_path=video_path,
            video_id=video_id,
            progress_callback=progress_callback
        )
        
        if result.get("status") == "completed":
            print(f"\n✓ {video_id} completed successfully!")
            update_status(video_id, "completed", phase="Done!")
        else:
            print(f"\n✗ {video_id} failed: {result.get('error')}")
            update_status(video_id, "failed", error=result.get("error", "Unknown error"))
    
    except Exception as e:
        import traceback
        print(f"\n✗ {video_id} error: {e}")
        traceback.print_exc()
        update_status(video_id, "failed", error=str(e))


def main():
    print("="*60)
    print("Video Processing Worker")
    print("="*60)
    print(f"Jobs file: {JOBS_FILE}")
    print(f"Uploads dir: {UPLOADS_DIR}")
    print(f"Memory root: {MEMORY_ROOT}")
    print("Watching for queued jobs... (Ctrl+C to stop)")
    print()
    
    while True:
        try:
            data = load_jobs()
            
            # Find queued jobs
            for video_id, job in data.get("jobs", {}).items():
                if job.get("status") == "queued":
                    process_one_job(video_id)
            
            # Sleep before checking again
            time.sleep(2)
        
        except KeyboardInterrupt:
            print("\nWorker stopped.")
            break
        except Exception as e:
            print(f"Worker error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
