# Implementation

Research code for the paper *Hierarchical Semantic Retrieval for Long-Form Video Question Answering* (VRAG). The method, results and animated figures are in the [top-level README](../README.md). Detailed run instructions are in [docs/RUN_GUIDE.md](../docs/RUN_GUIDE.md).

Run everything from this folder.

| Phase | Script | Purpose |
|---|---|---|
| 1.1 | `phase_1_1_full_videos_and_wav.py` | Download YouCook2 videos and extract 16 kHz audio |
| 1.2 | `phase_1_2_asr_whisperx.py` | ASR with WhisperX large-v3 |
| 1.3 | `phase_1_3_chunk_transcripts.py` | 20 to 40 s transcript chunks with similarity merging |
| 1.4 | `phase_1_4_embed_chunks.py` | Chunk embeddings (all-MiniLM-L6-v2) |
| 2.1 | `phase_2_1_topic_segmentation.py` | Topic segmentation |
| 3.1 | `phase_3_1_extract_frames.py` | Frame extraction at 1 FPS |
| 3.2 | `phase_3_2_embed_frames.py` | Frame embeddings (SigLIP so400m) |
| 3.3 | `phase_3_3_scene_clustering.py` | Scene clustering (HDBSCAN) |
| 4 | `phase_4_build_memory_store.py` | Hierarchical memory store |
| 5.0 | `phase_5_0_build_global_index.py` | Global FAISS topic index |
| 5 | `phase_5_query_pipeline_openrouter.py` | Query-time hierarchical retrieval and answer generation (OpenRouter) |
| 6 | `phase_6_evaluation.py` | Recall@K evaluation and latency |
| 7 | `phase_7_qa_generation_eval.py` | QA generation, ROUGE-L, BERTScore and LLM-judge evaluation |
| Ablations | `ablation_config.py`, `ablation_pipeline.py`, `ablation_query_generator.py`, `ablation_runner.py` | Ablation study (Base, A1 to A6) and threshold sweep |

Setup in three lines:

```bash
pip install -r requirements.txt     # after installing PyTorch and ffmpeg
cp .env.example .env                # then add your OPENROUTER_API_KEY
python phase_1_1_full_videos_and_wav.py
```
