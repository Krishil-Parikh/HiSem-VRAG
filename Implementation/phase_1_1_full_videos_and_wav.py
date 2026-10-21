# phase_1_1_full_videos_and_wav.py
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import yt_dlp


# ====================== CONFIG ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


RAW_FULL_VIDEO_DIR = os.path.join(BASE_DIR, "raw_full_videos")
FULL_WAV_DIR       = os.path.join(BASE_DIR, "audio_full_wav")


os.makedirs(RAW_FULL_VIDEO_DIR, exist_ok=True)
os.makedirs(FULL_WAV_DIR,       exist_ok=True)


MAX_WORKERS_DOWNLOAD = 5     # start conservative – YouTube can throttle hard
MAX_WORKERS_EXTRACT  = 8


# ====================== HELPERS ======================
def get_unique_videos():
    print("Loading YouCook2 (val + test)...")
    ds_val  = load_dataset("lmms-lab/YouCook2", split="val")
    ds_test = load_dataset("lmms-lab/YouCook2", split="test")
    
    df_val  = ds_val.to_pandas()
    df_test = ds_test.to_pandas()
    df      = pd.concat([df_val, df_test], ignore_index=True)
    
    unique = df.drop_duplicates(subset=['youtube_id', 'video_url'])[['youtube_id', 'video_url']]
    print(f"→ {len(unique)} unique full videos to attempt")
    return unique



def download_full_and_extract_mp3_temp(row):
    """
    Mimics your original script: download video + auto-extract mp3 via postprocessor
    We do this because it worked for you on segments.
    """
    youtube_id = row['youtube_id']
    url        = row['video_url']
    
    video_path = os.path.join(RAW_FULL_VIDEO_DIR, f"{youtube_id}.mp4")
    mp3_temp   = os.path.join(RAW_FULL_VIDEO_DIR, f"{youtube_id}.mp3")  # temp location
    
    # Skip if video already good
    if os.path.exists(video_path) and os.path.getsize(video_path) > 2_000_000:
        return youtube_id, True, "video already exists"
    
    ydl_opts = {
        'format': '18',                   # same as your working script
        'outtmpl': video_path,
        'ignoreerrors': True,
        'quiet': True,
        'no_warnings': True,
        'continuedl': True,
        'retries': 10,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'keepvideo': True,
        'postprocessor_args': [
            '-ar', '16000'
        ],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Check results
        video_ok = os.path.exists(video_path) and os.path.getsize(video_path) > 2_000_000
        mp3_ok   = os.path.exists(mp3_temp)
        
        if video_ok:
            return youtube_id, True, "video+mp3 success"
        else:
            return youtube_id, False, "no video file after download"
            
    except Exception as e:
        return youtube_id, False, str(e)[:140]



def convert_mp3_to_wav(youtube_id):
    mp3_path = os.path.join(RAW_FULL_VIDEO_DIR, f"{youtube_id}.mp3")
    wav_path = os.path.join(FULL_WAV_DIR,       f"{youtube_id}.wav")
    
    if os.path.exists(wav_path):
        return youtube_id, True, "wav already exists"
    
    if not os.path.exists(mp3_path):
        return youtube_id, False, "no mp3 to convert"
    
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', mp3_path,
            '-ar', '16000',
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            wav_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if os.path.exists(wav_path):
            return youtube_id, True, f"wav created ({os.path.getsize(wav_path)/1e6:.1f} MB)"
        else:
            return youtube_id, False, "ffmpeg did not create wav"
    except Exception as e:
        return youtube_id, False, str(e)[:120]



# ====================== MAIN ======================
if __name__ == "__main__":
    print(f"\n=== Phase 1.1 – Full videos + WAV ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ===\n")
    
    unique_df = get_unique_videos()
    
    # ── 1. Download full videos + temp mp3 (parallel) ──
    print(f"\nDownloading full videos ({len(unique_df)} attempts) ...\n")
    
    success_dl = 0
    fail_dl = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_DOWNLOAD) as ex:
        futures = [ex.submit(download_full_and_extract_mp3_temp, row) for _, row in unique_df.iterrows()]
        
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Download"):
            yid, ok, msg = fut.result()
            if ok:
                success_dl += 1
                tqdm.write(f"OK  {yid}  {msg}")
            else:
                fail_dl.append((yid, msg))
                tqdm.write(f"×   {yid}  {msg}")
    
    print(f"\nDownload phase: {success_dl} succeeded | {len(fail_dl)} failed")
    
    # ── 2. Convert temp .mp3 → .wav (parallel) ──
    print(f"\nConverting mp3 → 16kHz mono wav ...\n")
    
    mp3_files = [f for f in os.listdir(RAW_FULL_VIDEO_DIR) if f.endswith('.mp3')]
    success_wav = 0
    fail_wav = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_EXTRACT) as ex:
        futures = [ex.submit(convert_mp3_to_wav, fname.replace('.mp3', '')) for fname in mp3_files]
        
        for fut in tqdm(as_completed(futures), total=len(futures), desc="MP3 → WAV"):
            yid, ok, msg = fut.result()
            if ok:
                success_wav += 1
                tqdm.write(f"OK  {yid}  {msg}")
            else:
                fail_wav.append((yid, msg))
                tqdm.write(f"×   {yid}  {msg}")
    
    print(f"\nWAV conversion: {success_wav} succeeded | {len(fail_wav)} failed")
    
    # Final summary
    wav_count = len([f for f in os.listdir(FULL_WAV_DIR) if f.endswith('.wav')])
    print(f"\nFinished Phase 1.1")
    print(f"  Full videos  →  {RAW_FULL_VIDEO_DIR}  ({len(os.listdir(RAW_FULL_VIDEO_DIR))} files)")
    print(f"  Full-length WAV audio →  {FULL_WAV_DIR}  ({wav_count} files)")
    print("Ready for Phase 1.2 (ASR)\n")