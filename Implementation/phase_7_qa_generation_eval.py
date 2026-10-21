#!/usr/bin/env python3
"""
Phase 7: QA Generation & Evaluation Pipeline
- Generates QA pairs via OpenRouter
- Runs 3 approaches: Baseline A (LLM Only), Baseline B (Flat Retrieval), VRAG Pipeline
- Evaluates with ROUGE-L, BERTScore, and an LLM judge (OpenRouter by default, Ollama optional)
- Outputs CSV + HTML report
"""

import os
import json
import csv
import time
import argparse
import traceback
import requests
import html as html_module
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Optional rich output ──────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    console = Console()
    RICH = True
except ImportError:
    RICH = False
    class Console:
        def print(self, *a, **kw): print(*a)
        def rule(self, *a, **kw): print("─" * 60)
        def log(self, *a, **kw): print(*a)
    console = Console()

# ── Core deps ─────────────────────────────────────────────────────────────────
try:
    import evaluate
except ImportError:
    print("[WARN] evaluate not found. Run: pip install evaluate rouge-score bert-score")

# Import VRAG Pipeline and OpenRouter functions
try:
    from phase_5_query_pipeline_openrouter import (
        run_query_pipeline,
        call_openrouter,
        ANSWER_MODEL,
        OPENROUTER_KEYS,
        retrieve_global_topics_faiss,
        TEXT_EMBEDDER,
        load_global_index,
    )
except ImportError:
    print("[WARN] Could not import phase_5_query_pipeline_openrouter. Ensure it's in the same directory.")
    run_query_pipeline = None

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR            = Path(__file__).parent
MEMORY_ROOT         = BASE_DIR / "video_memory"
QA_UNVERIFIED_PATH  = BASE_DIR / "unverified_qa.json"
QA_VERIFIED_PATH    = BASE_DIR / "verified_qa.json"
EVAL_RESULTS_CSV    = BASE_DIR / "phase_7_eval_results.csv"
EVAL_RESULTS_HTML   = BASE_DIR / "phase_7_eval_report.html"
CHECKPOINT_PATH     = BASE_DIR / "phase_7_checkpoint.json"

# ── Ollama config ─────────────────────────────────────────────────────────────
# ── OpenRouter model defaults (override via .env) ─────────────────────────────
DEFAULT_QA_MODEL    = os.environ.get("VRAG_QA_MODEL",    "meta-llama/llama-3.1-8b-instruct")
DEFAULT_JUDGE_MODEL = os.environ.get("VRAG_JUDGE_MODEL", "google/gemma-4-31b-it")  # paper: Gemma-4-31B-IT judge
OLLAMA_BASE_URL     = "http://localhost:11434"
# Models ordered by preference for your RTX 4060 8GB + 32GB RAM setup
OLLAMA_PREFERRED    = [
    "mistral:7b-instruct-q4_K_M",   # ~4.1 GB VRAM – fits on 4060 cleanly
    "mistral",                        # default tag fallback
    "llama3.1:8b-instruct-q4_K_M",   # ~4.7 GB – also fits
    "llama3.1:8b",
    "llama3:8b",
    "gemma2:9b-instruct-q4_K_M",     # ~5.2 GB
    "phi3:mini",                      # tiny fallback
]


# ═══════════════════════════════════════════════════════════════════════════════
# OLLAMA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def ollama_list_models() -> List[str]:
    """Return names of all models currently pulled in Ollama."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def ollama_detect_judge() -> Optional[str]:
    """Pick the best available Ollama model for judging."""
    available = ollama_list_models()
    if not available:
        return None
    for preferred in OLLAMA_PREFERRED:
        for avail in available:
            # match by prefix so "mistral:7b-instruct-q4_K_M" matches "mistral:7b-…"
            if avail.startswith(preferred.split(":")[0]):
                return avail
    # fallback: just return the first available
    return available[0]


def ollama_generate(model: str, prompt: str, temperature: float = 0.0,
                    retries: int = 3) -> str:
    """Call Ollama /api/generate with retry logic."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    for attempt in range(retries):
        try:
            # Print so the user knows it's actively calling Ollama
            print(f"  [DEBUG] Sending request to Ollama: {model}...")
            r = requests.post(f"{OLLAMA_BASE_URL}/api/generate",
                              json=payload, timeout=120)
            r.raise_for_status()
            ans = r.json().get("response", "").strip()
            print(f"  [DEBUG] Ollama response: {ans[:100]}...")
            return ans
        except requests.exceptions.ConnectionError:
            if attempt == 0:
                console.print("[yellow]Ollama not reachable – is it running? (`ollama serve`)[/yellow]"
                              if RICH else "[WARN] Ollama not reachable.")
            time.sleep(2 ** attempt)
        except Exception as e:
            console.print(f"[red]Ollama error (attempt {attempt+1}): {e}[/red]"
                          if RICH else f"[WARN] Ollama error: {e}")
            time.sleep(2 ** attempt)
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSCRIPT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_transcript(video_id: str) -> str:
    topics_path = MEMORY_ROOT / video_id / "topics.json"
    if not topics_path.exists():
        return ""
    with open(topics_path, encoding="utf-8") as f:
        topics = json.load(f)
    parts = [t["text"] for t in topics if t.get("text")]
    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# QA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

QA_GENERATION_PROMPT = """You are building a rigorous benchmark for long-form video question answering.

Transcript (excerpt):
{transcript}

Generate exactly {n} diverse, high-quality question-answer pairs. Aim for variety:
  • At least 1 question requiring multi-hop temporal reasoning (e.g., "What happened AFTER X?")
  • At least 1 question about a specific detail or name
  • At least 1 causal question ("Why did…?")
  • At least 1 procedural/sequence question ("In what order…?")
  • Remaining questions can be open-ended factual

Rules:
  - Every question must be answerable solely from the transcript
  - Answers should be complete sentences (1-3 sentences)
  - No yes/no questions
  - No trivial or overly broad questions

Return ONLY valid JSON, no markdown, no explanation:
[
  {{"question": "...", "answer": "...", "type": "temporal|detail|causal|procedural|factual"}},
  ...
]
"""

def generate_qa_pairs(transcript: str, model_name: str, n: int = 5) -> List[Dict]:
    # Trim transcript to avoid token limits but keep enough context
    trimmed = transcript[:6000] if len(transcript) > 6000 else transcript
    prompt = QA_GENERATION_PROMPT.format(transcript=trimmed, n=n)

    for attempt in range(3):
        try:
            text = call_openrouter([{"role": "user", "content": prompt}], model_name,
                                   temperature=0.7, max_tokens=2000)
            if not text:
                raise ValueError("Empty response from OpenRouter")
            text = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            pairs = json.loads(text)
            # Validate structure
            return [p for p in pairs if "question" in p and "answer" in p]
        except json.JSONDecodeError:
            console.print(f"[yellow]JSON parse failed (attempt {attempt+1}), retrying…[/yellow]"
                          if RICH else f"[WARN] JSON parse fail attempt {attempt+1}")
            time.sleep(2)
        except Exception as e:
            console.print(f"[red]QA generation error: {e}[/red]" if RICH else f"[ERR] {e}")
            time.sleep(2)
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# LLM JUDGE  (OpenRouter by default, Ollama optional)
# ═══════════════════════════════════════════════════════════════════════════════

JUDGE_PROMPT = """You are a strict expert evaluator for a video question-answering system.

Rate the GENERATED ANSWER against the GROUND TRUTH on a 1–5 scale:

1 – Completely wrong or irrelevant
2 – Partially correct, missing key facts
3 – Mostly correct, minor errors or vagueness
4 – Correct and covers main points
5 – Perfect match in meaning, detail, and correctness

Question:        {question}
Ground Truth:    {gt_answer}
Generated Answer:{gen_answer}

Reply with ONLY a single integer (1, 2, 3, 4, or 5). Nothing else."""

def llm_judge_ollama(model: str, question: str, gt: str, gen: str) -> int:
    prompt = JUDGE_PROMPT.format(question=question, gt_answer=gt, gen_answer=gen)
    raw = ollama_generate(model, prompt, temperature=0.0)
    for char in raw:
        if char.isdigit():
            return min(max(int(char), 1), 5)
    return 0

def llm_judge_openrouter(model_name: str, question: str, gt: str, gen: str) -> int:
    prompt = JUDGE_PROMPT.format(question=question, gt_answer=gt, gen_answer=gen)
    try:
        text = call_openrouter([{"role": "user", "content": prompt}], model_name,
                               temperature=0.0, max_tokens=8)
        for char in text:
            if char.isdigit():
                return min(max(int(char), 1), 5)
    except Exception as e:
        console.print(f"[red]OpenRouter judge error: {e}[/red]" if RICH else f"[ERR] {e}")
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINES
# ═══════════════════════════════════════════════════════════════════════════════

def run_baseline_llm_only(query: str) -> str:
    """Baseline A: LLM Only (no retrieval)"""
    messages = [
        {"role": "system", "content": "You are a cooking assistant. Answer the question based on your general knowledge."},
        {"role": "user", "content": f"Question: {query}\n\nAnswer:"}
    ]
    ans = call_openrouter(messages, ANSWER_MODEL, temperature=0.35, max_tokens=600)
    return ans if ans else "No answer generated."

def run_baseline_flat_retrieval(query: str, top_k: int = 5) -> str:
    """Baseline B: Flat text retrieval (no hierarchy, no vision)"""
    load_global_index()
    q_emb = TEXT_EMBEDDER.encode(query)
    topics = retrieve_global_topics_faiss(q_emb, top_k=top_k)
    
    evidence = []
    for t in topics:
        evidence.append(f"Video {t['video_id']}, Time {t['start_time']}s-{t['end_time']}s: {t['text_preview']}")
        
    context_str = "\n".join(evidence)
    
    messages = [
        {"role": "system", "content": "You are a cooking assistant. Answer using ONLY the provided evidence. Be specific and cite timestamps when possible."},
        {"role": "user", "content": f"Evidence:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"}
    ]
    ans = call_openrouter(messages, ANSWER_MODEL, temperature=0.35, max_tokens=600)
    return ans if ans else "No answer generated."


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT  (resume on crash)
# ═══════════════════════════════════════════════════════════════════════════════

def load_checkpoint() -> List[Dict]:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_checkpoint(results: List[Dict]):
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# HTML REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def score_color(score: float, max_score: float = 1.0) -> str:
    pct = score / max_score
    if pct >= 0.75: return "#22c55e"
    if pct >= 0.50: return "#f59e0b"
    return "#ef4444"


def generate_html_report(results: List[Dict]):
    # Compute averages
    def avg(key): return sum(r.get(key, 0) for r in results) / max(len(results), 1)
    
    avg_r_llm = avg('rougeL_llm')
    avg_r_flat = avg('rougeL_flat')
    avg_r_vrag = avg('rougeL_vrag')
    
    avg_b_llm = avg('bertscore_f1_llm')
    avg_b_flat = avg('bertscore_f1_flat')
    avg_b_vrag = avg('bertscore_f1_vrag')
    
    avg_j_llm = avg('judge_llm')
    avg_j_flat = avg('judge_flat')
    avg_j_vrag = avg('judge_vrag')

    rows = ""
    for i, r in enumerate(results, 1):
        jc = score_color(r.get("judge_vrag", 0), 5)
        rc = score_color(r.get("rougeL_vrag", 0))
        bc = score_color(r.get("bertscore_f1_vrag", 0))
        rows += f"""
        <tr>
          <td>{i}</td>
          <td class="vid">{html_module.escape(r['video_id'])}</td>
          <td>{html_module.escape(r['question'])}</td>
          <td class="ans">{html_module.escape(r['gt_answer'])}</td>
          <td class="ans">{html_module.escape(r.get('ans_vrag', ''))}</td>
          <td style="color:{rc};font-weight:700">{r.get('rougeL_vrag', 0):.3f}</td>
          <td style="color:{bc};font-weight:700">{r.get('bertscore_f1_vrag', 0):.3f}</td>
          <td style="color:{jc};font-weight:700">{r.get('judge_vrag', 0)}/5</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Phase 7 Evaluation Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem; }}
  h1 {{ font-size: 1.8rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.25rem; }}
  .meta {{ color: #64748b; font-size: 0.85rem; margin-bottom: 2rem; }}
  .summary {{ display: flex; gap: 1.5rem; margin-bottom: 2rem; flex-wrap: wrap; }}
  .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px;
           padding: 1.25rem 2rem; min-width: 180px; text-align: center; }}
  .card .label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em;
                  color: #94a3b8; margin-bottom: 0.5rem; }}
  .card .value {{ font-size: 1.5rem; font-weight: 800; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  thead {{ background: #1e293b; position: sticky; top: 0; }}
  th {{ padding: 0.75rem 0.6rem; text-align: left; color: #94a3b8;
        font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;
        border-bottom: 1px solid #334155; }}
  td {{ padding: 0.65rem 0.6rem; border-bottom: 1px solid #1e293b; vertical-align: top; }}
  tr:hover td {{ background: #1e293b55; }}
  .vid {{ color: #60a5fa; font-size: 0.75rem; font-family: monospace; }}
  .ans {{ max-width: 280px; line-height: 1.4; color: #cbd5e1; }}
  .scroll {{ overflow-x: auto; border-radius: 12px; border: 1px solid #334155; margin-top: 1rem; }}
</style>
</head>
<body>
<h1>Phase 7 — VRAG Evaluation Report</h1>
<p class="meta">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} &nbsp;|&nbsp; {len(results)} QA pairs evaluated</p>

<h3>Averages Comparison</h3>
<div class="summary">
  <div class="card"><div class="label">ROUGE-L</div>
    <div class="value">LLM: <span style="color:{score_color(avg_r_llm)}">{avg_r_llm:.3f}</span><br>
    Flat: <span style="color:{score_color(avg_r_flat)}">{avg_r_flat:.3f}</span><br>
    VRAG: <span style="color:{score_color(avg_r_vrag)}">{avg_r_vrag:.3f}</span></div></div>
  <div class="card"><div class="label">BERTScore F1</div>
    <div class="value">LLM: <span style="color:{score_color(avg_b_llm)}">{avg_b_llm:.3f}</span><br>
    Flat: <span style="color:{score_color(avg_b_flat)}">{avg_b_flat:.3f}</span><br>
    VRAG: <span style="color:{score_color(avg_b_vrag)}">{avg_b_vrag:.3f}</span></div></div>
  <div class="card"><div class="label">LLM Judge / 5</div>
    <div class="value">LLM: <span style="color:{score_color(avg_j_llm, 5)}">{avg_j_llm:.2f}</span><br>
    Flat: <span style="color:{score_color(avg_j_flat, 5)}">{avg_j_flat:.2f}</span><br>
    VRAG: <span style="color:{score_color(avg_j_vrag, 5)}">{avg_j_vrag:.2f}</span></div></div>
</div>

<div class="scroll">
<table>
  <thead><tr>
    <th>#</th><th>Video ID</th><th>Question</th>
    <th>Ground Truth</th><th>Generated (VRAG)</th>
    <th>ROUGE-L (VRAG)</th><th>BERTScore (VRAG)</th><th>Judge (VRAG)</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>
</div>
</body>
</html>"""

    with open(EVAL_RESULTS_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    console.print(f"[green]HTML report saved → {EVAL_RESULTS_HTML}[/green]"
                  if RICH else f"[OK] HTML report: {EVAL_RESULTS_HTML}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP: GENERATE
# ═══════════════════════════════════════════════════════════════════════════════

def step_generate(num_videos: int, model_name: str, qa_per_video: int):
    if not OPENROUTER_KEYS:
        console.print("[bold red][ERROR] OPENROUTER_API_KEY not set (see .env.example).[/bold red]"
                      if RICH else "[ERROR] OPENROUTER_API_KEY not set (see .env.example).")
        return

    if not MEMORY_ROOT.exists():
        console.print(f"[red]video_memory directory not found at {MEMORY_ROOT}[/red]"
                      if RICH else f"[ERR] {MEMORY_ROOT} not found")
        return

    video_dirs = [d for d in MEMORY_ROOT.iterdir() if d.is_dir()]
    console.print(f"Found [cyan]{len(video_dirs)}[/cyan] video directories."
                  if RICH else f"[INFO] {len(video_dirs)} videos found.")

    all_qa: List[Dict] = []
    processed = 0

    iter_list = video_dirs[:num_videos] if len(video_dirs) >= num_videos else video_dirs
    if RICH:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      BarColumn(), TextColumn("{task.completed}/{task.total}"),
                      TimeElapsedColumn(), console=console) as progress:
            task = progress.add_task("Generating QA pairs…", total=len(iter_list))
            for vid_dir in iter_list:
                vid = vid_dir.name
                transcript = extract_transcript(vid)
                if not transcript or len(transcript) < 100:
                    progress.advance(task)
                    continue
                progress.update(task, description=f"[cyan]{vid[:30]}…[/cyan]")
                pairs = generate_qa_pairs(transcript, model_name, n=qa_per_video)
                for qa in pairs:
                    qa["video_id"] = vid
                all_qa.extend(pairs)
                processed += 1
                progress.advance(task)
                time.sleep(1.5)
    else:
        for vid_dir in iter_list:
            vid = vid_dir.name
            transcript = extract_transcript(vid)
            if not transcript or len(transcript) < 100:
                continue
            print(f"  Generating for {vid}…")
            pairs = generate_qa_pairs(transcript, model_name, n=qa_per_video)
            for qa in pairs:
                qa["video_id"] = vid
            all_qa.extend(pairs)
            processed += 1
            time.sleep(1.5)

    with open(QA_UNVERIFIED_PATH, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, indent=2, ensure_ascii=False)

    console.print(f"\n[bold green][OK] Generated {len(all_qa)} QA pairs from {processed} videos.[/bold green]"
                  if RICH else f"[OK] {len(all_qa)} QA pairs saved.")
    console.print(f"  Review: [yellow]{QA_UNVERIFIED_PATH}[/yellow]"
                  if RICH else f"  Review: {QA_UNVERIFIED_PATH}")
    console.print(f"  Then copy/rename to: [yellow]{QA_VERIFIED_PATH}[/yellow] and run --step evaluate"
                  if RICH else f"  Then rename to {QA_VERIFIED_PATH} and run --step evaluate")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP: EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════

def step_evaluate(judge_model_arg: str, use_ollama: bool, resume: bool, workers: int):
    if run_query_pipeline is None:
        console.print("[red][ERROR] run_query_pipeline not importable.[/red]"
                      if RICH else "[ERROR] run_query_pipeline not importable.")
        return

    # ── Judge selection ───────────────────────────────────────────────────────
    openrouter_ready = bool(OPENROUTER_KEYS)
    ollama_model  = None
    judge_backend = "none"

    if use_ollama:
        available_models = ollama_list_models()
        ollama_model = ollama_detect_judge() if judge_model_arg == "auto" else judge_model_arg
        
        if ollama_model:
            # Validate that the requested model is actually downloaded
            if judge_model_arg != "auto" and ollama_model not in available_models:
                console.print(f"[bold red][ERROR] You requested Ollama judge model '{ollama_model}', but it is NOT downloaded in your Ollama![/bold red]" if RICH else f"[ERROR] Model '{ollama_model}' not found in Ollama.")
                console.print(f"Available Ollama models: {', '.join(available_models) if available_models else 'None'}")
                console.print("Please pull the model first (e.g. 'ollama pull qwen2.5:14b') or pass a valid model.")
                return
                
            judge_backend = "ollama"
            console.print(f"[green]Ollama judge:[/green] [cyan]{ollama_model}[/cyan]"
                          if RICH else f"[INFO] Ollama judge: {ollama_model}")
        else:
            console.print("[yellow]No Ollama models found – falling back to the OpenRouter judge.[/yellow]"
                          if RICH else "[WARN] No Ollama models, falling back to OpenRouter.")
            use_ollama = False

    if not use_ollama:
        if OPENROUTER_KEYS:
            judge_backend = "openrouter"
            console.print(f"[green]OpenRouter judge:[/green] [cyan]{judge_model_arg}[/cyan]"
                          if RICH else f"[INFO] OpenRouter judge: {judge_model_arg}")
        else:
            console.print("[red][WARN] No judge available (no OPENROUTER_API_KEY set).[/red]"
                          if RICH else "[WARN] No judge available.")

    # ── Load metrics ──────────────────────────────────────────────────────────
    console.print("Loading evaluation metrics…" if RICH else "[INFO] Loading metrics…")
    rouge     = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    # ── Load QA dataset ───────────────────────────────────────────────────────
    qa_path = QA_VERIFIED_PATH if QA_VERIFIED_PATH.exists() else QA_UNVERIFIED_PATH
    if not qa_path.exists():
        console.print(f"[red][ERROR] No QA file at {qa_path}. Run --step generate first.[/red]"
                      if RICH else f"[ERROR] {qa_path} not found.")
        return
    with open(qa_path, encoding="utf-8") as f:
        qa_dataset: List[Dict] = json.load(f)
    console.print(f"Loaded [cyan]{len(qa_dataset)}[/cyan] QA pairs from [dim]{qa_path}[/dim]"
                  if RICH else f"[INFO] {len(qa_dataset)} QA pairs loaded.")

    # ── Resume from checkpoint ────────────────────────────────────────────────
    done_results: List[Dict] = []
    done_questions: set      = set()
    if resume and CHECKPOINT_PATH.exists():
        done_results   = load_checkpoint()
        done_questions = {r["question"] for r in done_results}
        console.print(f"[yellow]Resuming – {len(done_results)} already done.[/yellow]"
                      if RICH else f"[INFO] Resuming from {len(done_results)} done.")
        qa_dataset = [q for q in qa_dataset if q["question"] not in done_questions]

    # ── Evaluation loop ───────────────────────────────────────────────────────
    results: List[Dict] = list(done_results)

    def evaluate_single(qa: Dict) -> Dict:
        q  = qa["question"]
        gt = qa["answer"]
        vid= qa["video_id"]

        # Run Baselines
        ans_llm = run_baseline_llm_only(q)
        ans_flat = run_baseline_flat_retrieval(q)

        # Run VRAG pipeline
        try:
            pipeline_result = run_query_pipeline(q)
            ans_vrag = pipeline_result.get("answer", "") or "No answer generated."
        except Exception as e:
            ans_vrag = f"Pipeline error: {e}"

        def get_metrics(gen):
            try:
                rl = rouge.compute(predictions=[gen], references=[gt])["rougeL"]
            except Exception:
                rl = 0.0

            try:
                bf1 = bertscore.compute(predictions=[gen], references=[gt], lang="en")["f1"][0]
            except Exception:
                bf1 = 0.0

            if judge_backend == "ollama":
                js = llm_judge_ollama(ollama_model, q, gt, gen)
            elif judge_backend == "openrouter":
                js = llm_judge_openrouter(judge_model_arg, q, gt, gen)
            else:
                js = 0
            
            return round(rl, 4), round(bf1, 4), js

        rl_llm, bf1_llm, j_llm = get_metrics(ans_llm)
        rl_flat, bf1_flat, j_flat = get_metrics(ans_flat)
        rl_vrag, bf1_vrag, j_vrag = get_metrics(ans_vrag)

        return {
            "video_id":        vid,
            "question":        q,
            "gt_answer":       gt,
            
            "ans_llm":         ans_llm,
            "rougeL_llm":      rl_llm,
            "bertscore_f1_llm":bf1_llm,
            "judge_llm":       j_llm,
            
            "ans_flat":        ans_flat,
            "rougeL_flat":     rl_flat,
            "bertscore_f1_flat":bf1_flat,
            "judge_flat":      j_flat,

            "ans_vrag":        ans_vrag,
            "rougeL_vrag":     rl_vrag,
            "bertscore_f1_vrag":bf1_vrag,
            "judge_vrag":      j_vrag,
        }

    if RICH:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      BarColumn(), TextColumn("{task.completed}/{task.total}"),
                      TimeElapsedColumn(), console=console) as progress:
            task = progress.add_task("Evaluating…", total=len(qa_dataset))

            # Use workers=1 for sequential (safer with metrics state); >1 for pipeline calls
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(evaluate_single, qa): qa for qa in qa_dataset}
                for future in as_completed(futures):
                    try:
                        res = future.result()
                        results.append(res)
                        save_checkpoint(results)
                        progress.update(task, advance=1,
                                        description=f"[cyan]{res['video_id'][:25]}[/cyan] "
                                                    f"VRAG-R:{res['rougeL_vrag']:.2f} "
                                                    f"VRAG-J:{res['judge_vrag']}/5")
                    except Exception:
                        progress.advance(task)
                        traceback.print_exc()
    else:
        for i, qa in enumerate(qa_dataset, 1):
            print(f"[{i}/{len(qa_dataset)}] {qa['video_id']} | {qa['question'][:60]}…")
            try:
                res = evaluate_single(qa)
                results.append(res)
                save_checkpoint(results)
                print(f"  → LLM  | ROUGE-L:{res['rougeL_llm']:.3f} BErt:{res['bertscore_f1_llm']:.3f} Judge:{res['judge_llm']}/5")
                print(f"  → Flat | ROUGE-L:{res['rougeL_flat']:.3f} BErt:{res['bertscore_f1_flat']:.3f} Judge:{res['judge_flat']}/5")
                print(f"  → VRAG | ROUGE-L:{res['rougeL_vrag']:.3f} BErt:{res['bertscore_f1_vrag']:.3f} Judge:{res['judge_vrag']}/5")
            except Exception:
                traceback.print_exc()

    if not results:
        console.print("[red]No results to save.[/red]" if RICH else "[ERR] No results.")
        return

    # ── Save CSV ──────────────────────────────────────────────────────────────
    with open(EVAL_RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # ── Aggregates ────────────────────────────────────────────────────────────
    def avg(key): return sum(r.get(key, 0) for r in results) / len(results)
    
    avg_r_llm = avg('rougeL_llm')
    avg_r_flat = avg('rougeL_flat')
    avg_r_vrag = avg('rougeL_vrag')
    
    avg_b_llm = avg('bertscore_f1_llm')
    avg_b_flat = avg('bertscore_f1_flat')
    avg_b_vrag = avg('bertscore_f1_vrag')
    
    avg_j_llm = avg('judge_llm')
    avg_j_flat = avg('judge_flat')
    avg_j_vrag = avg('judge_vrag')

    # ── HTML report ───────────────────────────────────────────────────────────
    generate_html_report(results)

    # ── Clean checkpoint ──────────────────────────────────────────────────────
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    # ── Summary ───────────────────────────────────────────────────────────────
    if RICH:
        console.rule("[bold]Final Results[/bold]")
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim")
        table.add_column("LLM Only", justify="right")
        table.add_column("Flat Retrieval", justify="right")
        table.add_column("VRAG Pipeline", justify="right")
        
        table.add_row("Avg ROUGE-L", f"{avg_r_llm:.4f}", f"{avg_r_flat:.4f}", f"{avg_r_vrag:.4f}")
        table.add_row("Avg BERTScore F1", f"{avg_b_llm:.4f}", f"{avg_b_flat:.4f}", f"{avg_b_vrag:.4f}")
        table.add_row("Avg LLM Judge", f"{avg_j_llm:.2f} / 5", f"{avg_j_flat:.2f} / 5", f"{avg_j_vrag:.2f} / 5")
        
        console.print(table)
        console.print(f"\n[dim]Total Pairs Evaluated: {len(results)}[/dim]")
        console.print(f"[green][OK] CSV  → {EVAL_RESULTS_CSV}[/green]")
        console.print(f"[green][OK] HTML → {EVAL_RESULTS_HTML}[/green]")
    else:
        print("\n── Final Averages ──")
        print("               LLM Only | Flat Ret | VRAG")
        print(f"ROUGE-L      : {avg_r_llm:.4f}   | {avg_r_flat:.4f}   | {avg_r_vrag:.4f}")
        print(f"BERTScore F1 : {avg_b_llm:.4f}   | {avg_b_flat:.4f}   | {avg_b_vrag:.4f}")
        print(f"LLM Judge    : {avg_j_llm:.2f}     | {avg_j_flat:.2f}     | {avg_j_vrag:.2f}")
        print(f"CSV  → {EVAL_RESULTS_CSV}")
        print(f"HTML → {EVAL_RESULTS_HTML}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP: LIST OLLAMA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def step_list_models():
    models = ollama_list_models()
    if not models:
        console.print("[red]No Ollama models found (is Ollama running?)[/red]"
                      if RICH else "[INFO] No Ollama models found.")
        console.print("Suggested pull commands for your RTX 4060 (8 GB VRAM):"
                      if RICH else "Suggested models:")
        suggestions = [
            ("mistral:7b-instruct-q4_K_M", "~4.1 GB VRAM – recommended"),
            ("llama3.1:8b-instruct-q4_K_M", "~4.7 GB VRAM – strong reasoning"),
            ("phi3:mini",                    "~2.3 GB VRAM – fastest"),
        ]
        for name, note in suggestions:
            console.print(f"  ollama pull {name}   [dim]# {note}[/dim]"
                          if RICH else f"  ollama pull {name}  ({note})")
    else:
        console.print("[bold]Available Ollama models:[/bold]" if RICH else "Available Ollama models:")
        for m in models:
            console.print(f"  [cyan]{m}[/cyan]" if RICH else f"  {m}")
        best = ollama_detect_judge()
        console.print(f"\n[green]→ Auto-selected judge: {best}[/green]"
                      if RICH else f"\nAuto-selected judge: {best}")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 7 – QA Generation & Evaluation Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--step", choices=["generate", "evaluate", "list-models"], required=True,
        help=(
            "generate     – Create QA pairs from video transcripts (OpenRouter)\n"
            "evaluate     – Run VRAG pipeline + compute metrics + generate report\n"
            "list-models  – Show available Ollama models (only needed with --use-ollama)"
        ),
    )
    parser.add_argument("--num_videos",   type=int, default=10,
                        help="Videos to generate QA for (default: 10)")
    parser.add_argument("--qa_per_video", type=int, default=5,
                        help="QA pairs per video (default: 5)")
    parser.add_argument("--qa_model",     type=str, default=DEFAULT_QA_MODEL,
                        help=f"OpenRouter model for QA generation (default: {DEFAULT_QA_MODEL})")
    parser.add_argument("--judge_model",  type=str, default=None,
                        help=(
                            f"OpenRouter judge model (default: {DEFAULT_JUDGE_MODEL}).\n"
                            "With --use-ollama: 'auto' (detect best) or an exact Ollama model name."
                        ))
    parser.add_argument("--use-ollama",   action="store_true",
                        help="Use a local Ollama model as the judge instead of OpenRouter")
    parser.add_argument("--resume",       action="store_true",
                        help="Resume evaluation from checkpoint (skips already-done pairs)")
    parser.add_argument("--workers",      type=int, default=1,
                        help="Parallel workers for evaluation (default: 1; increase carefully)")

    args = parser.parse_args()

    if args.step == "generate":
        step_generate(args.num_videos, args.qa_model, args.qa_per_video)
    elif args.step == "evaluate":
        step_evaluate(
            judge_model_arg = args.judge_model or ("auto" if args.use_ollama else DEFAULT_JUDGE_MODEL),
            use_ollama      = args.use_ollama,
            resume          = args.resume,
            workers         = args.workers,
        )
    elif args.step == "list-models":
        step_list_models()