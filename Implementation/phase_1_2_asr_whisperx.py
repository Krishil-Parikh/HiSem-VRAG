# phase_1_2_asr_whisperx.py
import os
import json
from pathlib import Path
from datetime import datetime
import torch
from tqdm import tqdm
import whisperx
from concurrent.futures import ThreadPoolExecutor, as_completed

# PyTorch 2.6+ security fix: monkeypatch torch.load to force weights_only=False
# This is required because even with safe_globals, some dependencies fail to load.
# IMPORTANT: Only do this if you trust your model sources (OpenAI/WhisperX)
import torch
_orig_torch_load = torch.load
def _monkey_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _monkey_torch_load

# ====================== CONFIG ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WAV_DIR = os.path.join(BASE_DIR, "audio_full_wav")
ASR_OUTPUT_DIR = os.path.join(BASE_DIR, "asr_transcripts_json")

# Fix for CUDA DLLs on Windows (WhisperX / CTranslate2)
try:
    import site
    site_pkgs = site.getsitepackages()
    for sp in site_pkgs:
        nvidia_path = os.path.join(sp, "nvidia")
        if os.path.exists(nvidia_path):
            for root, dirs, files in os.walk(nvidia_path):
                if "bin" in dirs or "lib" in dirs:
                    for dname in ["bin", "lib"]:
                        lib_path = os.path.join(root, dname)
                        if os.path.isdir(lib_path):
                            os.environ["PATH"] = lib_path + os.pathsep + os.environ["PATH"]
except Exception:
    pass

os.makedirs(ASR_OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "large-v3"           # or "large-v3-turbo" for faster but slightly lower quality

# Forcing GPU as requested
if torch.cuda.is_available():
    DEVICE = "cuda"
    print("[OK] CUDA is available. Using GPU.")
else:
    DEVICE = "cpu"
    print("[FAIL] WARNING: CUDA not available. Falling back to CPU.")
    print("   To use GPU, ensure you have:")
    print("   1. NVIDIA Drivers installed")
    print("   2. CUDA Toolkit installed")
    print("   3. Torch installed with CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu121")

BATCH_SIZE = 8                    # lower to 4 or 2 if you get OOM errors
COMPUTE_TYPE = "float16" if DEVICE.startswith("cuda") else "float32"

MAX_WORKERS = 2                   # 1–4 depending on GPU memory and number of GPUs
                                          # Increase only if you have strong hardware

# ====================== HELPERS ======================
# Load models globally once
print(f"Loading WhisperX model '{MODEL_NAME}' on {DEVICE}...")
whisper_model = whisperx.load_model(
    MODEL_NAME,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    language="en",              # YouCook2 is English
    vad_method="silero"         # VAD method moved here
)

print(f"Loading alignment model for 'en'...")
align_model, align_metadata = whisperx.load_align_model(
    language_code="en",
    device=DEVICE
)

def transcribe_one_wav(wav_path: str) -> tuple:
    """
    Transcribe one .wav file using WhisperX + Silero VAD
    Returns: (filename_stem, success: bool, message: str)
    """
    wav_name = Path(wav_path).stem
    json_path = os.path.join(ASR_OUTPUT_DIR, f"{wav_name}.json")

    # Skip if valid JSON already exists
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0 and all("text" in item and "start" in item for item in data):
                return wav_name, True, f"already done ({len(data)} segments)"
        except Exception:
            # corrupt file → will re-transcribe
            pass

    try:
        # Load audio
        audio = whisperx.load_audio(wav_path)

        # Transcribe
        result = whisper_model.transcribe(
            audio,
            batch_size=BATCH_SIZE,
            language="en"
        )

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

        # Save result
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        num_segments = len(segments)
        msg = f"success → {num_segments} segments, saved to {json_path}"
        return wav_name, True, msg

    except Exception as e:
        error_msg = str(e).replace("\n", " ").strip()[:220]
        return wav_name, False, f"failed: {error_msg}"


# ====================== MAIN ======================
if __name__ == "__main__":
    start_time = datetime.now()
    print(f"\n=== Phase 1.2 — WhisperX ASR (started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}) ===\n")
    print(f"Model       : openai/whisper-{MODEL_NAME}")
    print(f"Device      : {DEVICE}")
    print(f"Batch size  : {BATCH_SIZE}")
    print(f"VAD method  : silero (no pyannote)")
    print(f"Max workers : {MAX_WORKERS}\n")

    wav_files = [
        os.path.join(WAV_DIR, f)
        for f in os.listdir(WAV_DIR)
        if f.lower().endswith(('.wav', '.wave'))
    ]

    print(f"Found {len(wav_files)} .wav files")

    success_count = 0
    failed = []

    # Process sequentially - GPU models are NOT thread-safe
    for wav_path in tqdm(wav_files, desc="ASR Progress", unit="file"):
        try:
            wav_name, success, message = transcribe_one_wav(wav_path)
            if success:
                success_count += 1
                tqdm.write(f"OK   {wav_name}: {message}")
            else:
                failed.append((wav_name, message))
                tqdm.write(f"FAIL {wav_name}: {message}")
        except Exception as e:
            wav_name = Path(wav_path).stem
            failed.append((wav_name, str(e)[:200]))
            tqdm.write(f"FAIL {wav_name}: {e}")

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n=== Finished Phase 1.2 ===")
    print(f"Duration         : {duration}")
    print(f"Succeeded        : {success_count} / {len(wav_files)}")
    print(f"Failed           : {len(failed)}")

    if failed:
        print("\nFailed files:")
        for name, msg in failed:
            print(f"  • {name}: {msg}")

    json_count = sum(1 for f in os.listdir(ASR_OUTPUT_DIR) if f.endswith('.json'))
    print(f"\nTranscripts saved: {json_count} files in")
    print(f"  → {ASR_OUTPUT_DIR}")
    print("\nNext step: Phase 1.3 — Chunk transcripts into 20–40s windows + similarity merging\n")