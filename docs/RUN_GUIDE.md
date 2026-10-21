# Run guide

Step-by-step instructions for reproducing the pipeline, the evaluation and the ablations of **HiSem-VRAG**, and for running the demo web application. For the method and results, see the [README](../README.md).

## Contents

1. [Requirements](#1-requirements)
2. [Get the code](#2-get-the-code)
3. [Python environment](#3-python-environment)
4. [API keys](#4-api-keys)
5. [Offline pipeline (build the hierarchical memory)](#5-offline-pipeline-build-the-hierarchical-memory)
6. [Online retrieval, evaluation and ablations](#6-online-retrieval-evaluation-and-ablations)
7. [Demo web application](#7-demo-web-application)
8. [Troubleshooting](#8-troubleshooting)
9. [Reproducibility notes](#9-reproducibility-notes)

All commands for the research pipeline are run **from the `Implementation/` folder**. Every script resolves its input and output folders relative to its own location, so the working directory matters less than keeping the scripts together.

---

## 1. Requirements

| Need | Details |
|---|---|
| Python | 3.11 (the version the code was developed and tested with) |
| GPU | An NVIDIA GPU is strongly recommended for ASR (WhisperX large-v3) and SigLIP embeddings. The code was developed on a single consumer GPU (CUDA 12.1, fp16). Without a GPU the scripts fall back to the CPU and become very slow. |
| ffmpeg | Must be installed and on your `PATH` (used for audio extraction). Check with `ffmpeg -version`. |
| Disk | Tens of gigabytes. Videos, 16 kHz audio and 1 FPS frames for hundreds of videos dominate. |
| Internet | Needed to download videos from YouTube, model weights from Hugging Face, and for OpenRouter API calls. |
| OpenRouter key | Needed for query decomposition, VLM captioning, answer generation, pseudo-query generation and the QA evaluation. Create one at <https://openrouter.ai/keys>. |
| Node.js 18+ | Only for the demo frontend. |

---

## 2. Get the code

```bash
git clone https://github.com/Krishil-Parikh/HiSem-VRAG.git
cd HiSem-VRAG/Implementation
```

---

## 3. Python environment

**Windows (PowerShell)**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3.11 -m venv venv
source venv/bin/activate
```

Install PyTorch first, using the build that matches your machine, then the remaining packages:

```bash
# NVIDIA GPU, CUDA 12.1
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# or CPU only
# pip install torch==2.5.1 torchaudio==2.5.1

pip install -r requirements.txt
```

`requirements.txt` pins the exact versions the pipeline was run with. `faiss-cpu` is used for the global index. The first run of each model downloads its weights from Hugging Face (WhisperX large-v3, all-MiniLM-L6-v2, `google/siglip-so400m-patch14-384`).

---

## 4. API keys

All keys are read from environment variables, loaded from a `.env` file. Real keys must never be committed (`.env` is git-ignored).

```bash
cp .env.example .env          # Windows: copy .env.example .env
```

Edit `.env` and set your key:

```
OPENROUTER_API_KEY=your-openrouter-api-key
```

Optional settings (see `.env.example`):

| Variable | Purpose | Default (matches the paper) |
|---|---|---|
| `OPENROUTER_API_KEYS` | Several keys, comma-separated, used round-robin to spread rate limits. Takes precedence over `OPENROUTER_API_KEY`. | unset |
| `VRAG_LLM_MODEL` | Text LLM for query decomposition and answers | `meta-llama/llama-3.1-8b-instruct` |
| `VRAG_VLM_MODEL` | Vision-language model for keyframe captions | `qwen/qwen2.5-vl-72b-instruct` |
| `VRAG_QA_MODEL` | Model that writes QA pairs in phase 7 | `meta-llama/llama-3.1-8b-instruct` |
| `VRAG_JUDGE_MODEL` | LLM judge in phase 7 (the paper uses Gemma-4-31B-IT) | `google/gemma-4-31b-it` |

Model identifiers on OpenRouter change over time. If a request fails with a "model not found" error, look up the current identifier at <https://openrouter.ai/models> and set the matching variable in `.env`.

---

## 5. Offline pipeline (build the hierarchical memory)

Run the phases **in order**. Each phase reads the previous phase's output folder (created next to the scripts).

| # | Command | Reads | Writes |
|---|---|---|---|
| 1.1 | `python phase_1_1_full_videos_and_wav.py` | YouCook2 metadata (`lmms-lab/YouCook2`, val + test, from Hugging Face) | `raw_full_videos/` (MP4), `audio_full_wav/` (16 kHz mono WAV) |
| 1.2 | `python phase_1_2_asr_whisperx.py` | `audio_full_wav/` | `asr_transcripts_json/` |
| 1.3 | `python phase_1_3_chunk_transcripts.py` | `asr_transcripts_json/` | `chunked_transcripts_json/` |
| 1.4 | `python phase_1_4_embed_chunks.py` | `chunked_transcripts_json/` | `transcript_embeddings/` |
| 2.1 | `python phase_2_1_topic_segmentation.py` | chunks + embeddings | `topics_json/` |
| 3.1 | `python phase_3_1_extract_frames.py` | `raw_full_videos/`, `topics_json/` | `frames/` (1 FPS JPEGs + manifest per video) |
| 3.2 | `python phase_3_2_embed_frames.py` | `frames/` | `frame_embeddings/` |
| 3.3 | `python phase_3_3_scene_clustering.py` | `frames/`, `frame_embeddings/` | `scenes_json/` |
| 4 | `python phase_4_build_memory_store.py` | topics, scenes, both embedding sets | `video_memory/<video_id>/` |
| 5.0 | `python phase_5_0_build_global_index.py` | `video_memory/` | `video_memory/global_topic_index.faiss` and `global_topic_metadata.json` |

Notes:

- **Phase 1.1** downloads the videos with `yt-dlp`. Some YouTube videos have been removed or are blocked, so you will end up with fewer videos than the dataset lists (the paper starts from 596 downloaded videos and ends with 463 usable ones). Failed downloads are reported and skipped.
- **Phases 1.2, 1.3 and 3.2** skip items that already have an output file, so you can stop and resume.
- **Phase 1.2** loads WhisperX large-v3 on the GPU in fp16. If you run out of GPU memory, lower `BATCH_SIZE` (default 8) or `MAX_WORKERS` (default 2) at the top of the script.
- **Phase 3.2** loads SigLIP so400m. Lower `BATCH_SIZE` (default 32) if you run out of GPU memory.
- The offline hyperparameters (chunk window 20 to 40 s, merge threshold 0.75, topic threshold 0.68, minimum topic length 60 s, HDBSCAN `min_cluster_size`=5 and `min_samples`=3, 1 FPS) are constants at the top of each script and already match the paper.

After phase 5.0 you have a complete memory store: one folder per video under `video_memory/` plus the global topic index.

**Expected folder layout** after running phases 1.1 to 5.0 (inside `Implementation/`):

```
raw_full_videos/            <video_id>.mp4
audio_full_wav/             <video_id>.wav
asr_transcripts_json/       <video_id>.json
chunked_transcripts_json/   <video_id>.json
transcript_embeddings/      <video_id>.npy
topics_json/                <video_id>.json
frames/                     <video_id>/ ... frames_manifest.json and one folder per topic
frame_embeddings/           <video_id>.npy
scenes_json/                <video_id>.json
video_memory/
  <video_id>/               video_memory.json, topics.json, scenes.json, frame_embeddings.npy, ...
  global_topic_index.faiss
  global_topic_metadata.json
```

If a later phase reports a missing file, compare this layout with what is on disk to find the phase that did not finish. Video counts generally fall from one stage to the next as items are filtered out. For reference, the paper's funnel goes 596 unique videos, 520 transcribed, 460 chunked and 463 in the final memory store.

---

## 6. Online retrieval, evaluation and ablations

These steps call the OpenRouter API, so make sure `.env` is set up.

### 6.1 Ask a question (phase 5)

```bash
python phase_5_query_pipeline_openrouter.py
```

This loads the global index and runs one sample query (`"How do you prepare the dressing for the salad?"`), printing the decomposed sub-questions, retrieved topics and scenes, selected keyframes, VLM captions, the evidence JSON and the final answer. To use your own question, edit `sample_query` at the bottom of the file, or import `run_query_pipeline` from your own script:

```python
from phase_5_query_pipeline_openrouter import load_global_index, run_query_pipeline

load_global_index()
result = run_query_pipeline("When do I add the potatoes to the biryani mix?")
print(result["answer"])
```

The retrieval defaults at the top of the file (`CONFIG`) are the paper's values: `TOP_K_GLOBAL`=5, `TOP_VIDEOS`=3, `TOP_TOPICS_PER_VIDEO`=3, `TOP_SCENES_PER_TOPIC`=3, `TOP_FRAMES_PER_SCENE`=2, `MIN_TOPIC_SCORE`=0.7, `MIN_SCENE_SCORE`=0.05, `TEMPORAL_SMOOTHING_SIGMA`=1.5, `VLM_WORKERS`=4.

VLM captions are cached per frame under `video_memory/<video_id>/vlm_cache/`, so repeated runs are cheaper.

### 6.2 Retrieval evaluation (phase 6)

```bash
python phase_6_evaluation.py
```

Phase 6 generates pseudo-queries from the video transcripts with the LLM, runs the full pipeline on each, and computes Recall@K at video, topic and scene level plus latency. Results are written to `evaluation_results/` as a per-query CSV (`eval_<run_name>.csv`) and a summary JSON (`summary_<run_name>.json`).

The default run is deliberately small (10 videos, 2 queries each) so you can check your setup quickly. For a larger evaluation, edit the call in the `__main__` block at the bottom of `phase_6_evaluation.py`:

```python
queries = generate_evaluation_dataset(num_videos=463, queries_per_video=4)
```

### 6.3 Ablation study (Table 10 of the paper)

First generate the pseudo-queries, then run the ablations:

```bash
python ablation_query_generator.py                 # builds and caches ablation_results/evaluation_queries.json
python ablation_runner.py --list                   # show all configurations
python ablation_runner.py --main                   # base + the main ablations
python ablation_runner.py --threshold              # 27-configuration threshold sweep
python ablation_runner.py --all                    # everything
```

Useful options:

| Option | Meaning |
|---|---|
| `--ablation <name>` | Run one configuration, for example `flat_retrieval` or `no_decomposition` |
| `--queries N` | Use a stratified sample of N queries instead of all of them (good for a dry run) |
| `--regenerate` | Ignore the cached queries and generate new ones |
| `--from-csv <file>` | Reuse queries (and baseline metrics) from an existing evaluation CSV |

Per-query CSVs (`ablation_<name>.csv`) and summary JSONs (`main_ablations_summary.json`, and so on) are written to `ablation_results/`. The configuration names map to the paper's IDs as follows:

| Paper ID | `--ablation` name |
|---|---|
| Base | `baseline` |
| A1 | `flat_retrieval` |
| A2 | `no_decomposition` |
| A3 | `heuristic_ranking` |
| A4 | `position_keyframes` |
| A5 | `semantic_keyframes_no_smooth` |
| A6 | `no_smoothing` |

The defaults in `ablation_config.py` use the paper's tuned retrieval parameters, and the threshold sweep varies `MIN_TOPIC_SCORE` (0.1, 0.3, 0.5), `MIN_SCENE_SCORE` (0.01, 0.05, 0.1) and `TOP_SCENES_PER_TOPIC` (3, 5, 8).

### 6.4 End-to-end QA quality (phase 7, Table 11 of the paper)

Phase 7 has two steps:

```bash
# 1. Generate question-answer pairs from video transcripts
python phase_7_qa_generation_eval.py --step generate --num_videos 10 --qa_per_video 5
```

This writes `unverified_qa.json`. The paper evaluates on **human-verified** pairs: review the file, remove or fix bad pairs, and save the result as `verified_qa.json` (a JSON list of objects with `question`, `answer`, `video_id` and `type`). If `verified_qa.json` exists it is used, otherwise the unverified file is used.

```bash
# 2. Compare no-retrieval, flat retrieval and VRAG, and score the answers
python phase_7_qa_generation_eval.py --step evaluate
```

Each answer is scored with ROUGE-L, BERTScore F1 and an LLM judge (1 to 5). The judge defaults to the OpenRouter model in `VRAG_JUDGE_MODEL`. The script writes a CSV (`phase_7_eval_results.csv`), an HTML report (`phase_7_eval_report.html`) and a checkpoint file.

| Option | Meaning |
|---|---|
| `--qa_model <slug>` | OpenRouter model used to write the QA pairs |
| `--judge_model <slug>` | OpenRouter model used as the judge |
| `--use-ollama` | Use a local Ollama model as the judge instead of OpenRouter (then `--judge_model auto` picks one) |
| `--resume` | Continue from the checkpoint, skipping finished questions |
| `--workers N` | Parallel evaluation workers (default 1; raise carefully to respect API rate limits) |
| `--step list-models` | List local Ollama models (only relevant with `--use-ollama`) |

---

## 7. Demo web application

The demo has a FastAPI backend (`backend/`) and a Next.js frontend (`frontend/`). It indexes uploaded videos with the same hierarchical design and lets you chat with them. **Unlike the research pipeline, the demo backend calls the Gemini API**, so it needs Gemini keys (not an OpenRouter key).

### 7.1 Backend

```bash
cd backend
python -m venv venv
# activate the venv (see Section 3), install PyTorch for your machine, then:
pip install -r requirements.txt

cp .env.example .env            # Windows: copy .env.example .env
# edit .env and set GEMINI_API_KEYS=key1,key2   (comma-separated, used round-robin)

python main.py                  # API on http://localhost:8000
```

Uploaded videos are queued as jobs and processed by a separate worker process. In a second terminal (same virtual environment):

```bash
cd backend
python worker.py
```

### 7.2 Frontend

```bash
cd frontend
npm install
npm run dev                     # app on http://localhost:3000
```

The frontend talks to the backend at `http://localhost:8000` (set in `frontend/lib/api.ts`), and the backend allows requests from `http://localhost:3000`. Create an account on the sign-up page, then upload a video from the dashboard.

### 7.3 Browsing the research dataset in the demo (optional)

The backend can also list and search videos that were pre-processed by the research pipeline. It looks for them in `Implementation/mini_dataset_youcook2/`, using the same folder layout the pipeline produces:

```
Implementation/mini_dataset_youcook2/
  video_memory/        copy of Implementation/video_memory/ (including the global FAISS index)
  raw_full_videos/     the MP4 files, used for playback
  frames/              the extracted frames, used for thumbnails
```

Copy or move those three folders from `Implementation/` into `Implementation/mini_dataset_youcook2/` and restart the backend.

---

## 8. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `RuntimeError: No OpenRouter API key found` | `.env` is missing or not in `Implementation/`. Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`. |
| `[WARN] GEMINI_API_KEYS is not set` (demo backend) | Copy `backend/.env.example` to `backend/.env` and set `GEMINI_API_KEYS`. |
| HTTP 400 or 404 from OpenRouter mentioning the model | The model slug changed. Pick a current one at <https://openrouter.ai/models> and set `VRAG_LLM_MODEL`, `VRAG_VLM_MODEL`, `VRAG_QA_MODEL` or `VRAG_JUDGE_MODEL`. |
| HTTP 429 from OpenRouter | Rate limited. The client retries with back-off. Use several keys in `OPENROUTER_API_KEYS` or lower `VLM_WORKERS` in `CONFIG`. |
| `Global index not found. Run phase_5_0_build_global_index.py first!` | Phases 1 to 5.0 have not all been run, or `video_memory/` is missing. |
| CUDA out-of-memory in phase 1.2 or 3.2 | Lower `BATCH_SIZE` and `MAX_WORKERS` at the top of the script. |
| `ffmpeg` not found | Install ffmpeg and make sure it is on your `PATH`. |
| Many downloads fail in phase 1.1 | Videos removed from YouTube cannot be recovered. Update `yt-dlp` (`pip install -U yt-dlp`) if downloads fail for every video. |
| cuDNN or CUDA DLL errors on Windows in phase 1.2 | The script adds the `nvidia` libraries bundled with your venv to `PATH`. Make sure the venv contains the CUDA runtime wheels that came with PyTorch. |

---

## 9. Reproducibility notes

Re-running the pipeline will give numbers close to, but not necessarily identical to, those in the paper. The main reasons:

- **Query sets are not included.** The 1,103-query evaluation set and the 427 human-verified QA pairs are not part of this repository. Phase 6 and `ablation_query_generator.py` generate fresh pseudo-queries, and phase 7 generates fresh QA pairs that you review yourself, so absolute scores will differ from the paper's.
- **LLM output is not deterministic.** Query decomposition (temperature 0.2), VLM captions (0.3), answer generation (0.35) and pseudo-query generation (0.7) all go through hosted models. Different runs, providers and model versions can give different text. VLM captions are cached per frame under `video_memory/<video_id>/vlm_cache/`, so a second run reuses them.
- **Model versions change.** OpenRouter model identifiers and the models behind them are updated over time. Record the identifiers you used (`VRAG_LLM_MODEL`, `VRAG_VLM_MODEL`, `VRAG_JUDGE_MODEL`) alongside your results.
- **Sampling is seeded where it is in your control.** The query samplers in `ablation_query_generator.py` default to seed 42, so `--queries N` selects the same subset for the same query file.
- **Retrieval itself is exact.** The global index is a flat inner-product FAISS index, so for a fixed set of embeddings the retrieved topics and scenes are deterministic.
- **Latency depends on your setup.** End-to-end latency includes network round trips to the API provider. It will vary with your connection, the provider's load and your GPU for the local models.
- **Library versions are pinned** in `requirements.txt`. Using different versions of PyTorch, WhisperX or transformers can change ASR output and embeddings slightly.

When you report results from a re-run, state the number of videos that survived preprocessing, the models you used and the date, since all three affect the outcome.
