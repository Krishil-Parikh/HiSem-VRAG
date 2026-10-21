# ablation_runner.py
"""
Ablation Study Runner for Video-RAG Pipeline

Main entry point for running all ablation experiments.
Supports:
- Running individual ablations
- Running all 6 ablation studies
- Running threshold sensitivity sweep (27 configs)
- Parallel execution across queries
- CSV/JSON output for analysis
- Loading queries from existing evaluation CSV
"""

import os
import json
import csv
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import argparse

# Import ablation modules
from ablation_config import (
    AblationConfig,
    ABLATION_CONFIGS,
    THRESHOLD_CONFIGS,
    get_config,
    get_all_configs,
    list_ablations,
)
from ablation_pipeline import (
    run_ablation_pipeline,
    evaluate_ablation_query,
)
from ablation_query_generator import (
    generate_all_queries,
    sample_queries,
    stratified_sample,
)
from phase_5_query_pipeline_openrouter import load_global_index


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "ablation_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Parallel execution
EVAL_WORKERS = 4  # Parallel query evaluation


# ══════════════════════════════════════════════════════════════════════════════
# LOAD QUERIES FROM EXISTING CSV
# ══════════════════════════════════════════════════════════════════════════════

def load_queries_from_csv(csv_path: str) -> Tuple[List[Dict], Dict]:
    """
    Load queries and ground truth from an existing evaluation CSV.
    Also extracts baseline (hierarchical) metrics if present.
    
    Args:
        csv_path: Path to CSV (e.g., eval_massive_eval_1600.csv)
    
    Returns:
        (queries, baseline_metrics) where:
        - queries: List of dicts with 'query' and 'ground_truth'
        - baseline_metrics: Dict of aggregated baseline metrics (for comparison)
    """
    queries = []
    baseline_results = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Extract query and ground truth
            query_dict = {
                "query": row.get("query", ""),
                "ground_truth": {
                    "video_id": row.get("gt_video_id", ""),
                    "start_time": float(row.get("gt_start_time", 0)),
                    "end_time": float(row.get("gt_end_time", 0)),
                }
            }
            queries.append(query_dict)
            
            # Extract baseline metrics
            try:
                baseline_results.append({
                    "latency_sec": float(row.get("total_latency_sec", 0)),
                    "topics_retrieved": int(row.get("topics_retrieved", 0)),
                    "scenes_retrieved": int(row.get("scenes_retrieved", 0)),
                    "vlm_calls": int(row.get("vlm_calls", 0)),
                    "recall_at_1_video": int(row.get("recall@1_video", 0)),
                    "recall_at_3_video": int(row.get("recall@3_video", 0)),
                    "recall_at_1_topics": int(row.get("recall@1_topics", 0)),
                    "recall_at_3_topics": int(row.get("recall@3_topics", 0)),
                    "recall_at_5_topics": int(row.get("recall@5_topics", 0)),
                    "recall_at_3_scenes": int(row.get("recall@3_scenes", 0)),
                    "recall_at_5_scenes": int(row.get("recall@5_scenes", 0)),
                    "recall_at_10_scenes": int(row.get("recall@10_scenes", 0)),
                    "oracle_scene_recall": int(row.get("oracle_scene_recall", 0)),
                })
            except (ValueError, KeyError):
                pass
    
    # Compute aggregated baseline metrics
    baseline_metrics = {}
    if baseline_results:
        baseline_metrics = {
            "num_queries": len(baseline_results),
            "avg_latency_sec": np.mean([r["latency_sec"] for r in baseline_results]),
            "avg_topics_retrieved": np.mean([r["topics_retrieved"] for r in baseline_results]),
            "avg_scenes_retrieved": np.mean([r["scenes_retrieved"] for r in baseline_results]),
            "avg_vlm_calls": np.mean([r["vlm_calls"] for r in baseline_results]),
            "recall_at_1_video": np.mean([r["recall_at_1_video"] for r in baseline_results]),
            "recall_at_3_video": np.mean([r["recall_at_3_video"] for r in baseline_results]),
            "recall_at_1_topics": np.mean([r["recall_at_1_topics"] for r in baseline_results]),
            "recall_at_3_topics": np.mean([r["recall_at_3_topics"] for r in baseline_results]),
            "recall_at_5_topics": np.mean([r["recall_at_5_topics"] for r in baseline_results]),
            "recall_at_3_scenes": np.mean([r["recall_at_3_scenes"] for r in baseline_results]),
            "recall_at_5_scenes": np.mean([r["recall_at_5_scenes"] for r in baseline_results]),
            "recall_at_10_scenes": np.mean([r["recall_at_10_scenes"] for r in baseline_results]),
            "oracle_scene_recall": np.mean([r["oracle_scene_recall"] for r in baseline_results]),
        }
    
    print(f"[INFO] Loaded {len(queries)} queries from {csv_path}")
    if baseline_metrics:
        print(f"[INFO] Baseline metrics extracted:")
        print(f"       R@3 Topics: {baseline_metrics['recall_at_3_topics']:.1%}")
        print(f"       R@3 Scenes: {baseline_metrics['recall_at_3_scenes']:.1%}")
        print(f"       Avg Latency: {baseline_metrics['avg_latency_sec']:.2f}s")
    
    return queries, baseline_metrics


# ══════════════════════════════════════════════════════════════════════════════
# METRICS DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AblationMetrics:
    """Aggregated metrics for an ablation run."""
    ablation_name: str
    description: str
    num_queries: int
    
    # Latency
    avg_latency_sec: float = 0.0
    p50_latency_sec: float = 0.0
    p95_latency_sec: float = 0.0
    
    # Retrieval counts
    avg_topics_retrieved: float = 0.0
    avg_scenes_retrieved: float = 0.0
    avg_vlm_calls: float = 0.0
    
    # Video Recall@K
    recall_at_1_video: float = 0.0
    recall_at_3_video: float = 0.0
    
    # Topic Recall@K
    recall_at_1_topics: float = 0.0
    recall_at_3_topics: float = 0.0
    recall_at_5_topics: float = 0.0
    
    # Scene Recall@K (within GT video)
    recall_at_3_scenes: float = 0.0
    recall_at_5_scenes: float = 0.0
    recall_at_10_scenes: float = 0.0
    
    # Oracle
    oracle_scene_recall: float = 0.0
    
    # Config params (for threshold sweep)
    min_topic_score: float = 0.0
    min_scene_score: float = 0.0
    top_scenes_per_topic: int = 0


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE ABLATION RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_single_ablation(
    ablation_name: str,
    queries: List[Dict],
    save_csv: bool = True,
    verbose: bool = False
) -> AblationMetrics:
    """
    Run a single ablation configuration on all queries.
    
    Args:
        ablation_name: Name of the ablation (from ABLATION_CONFIGS or THRESHOLD_CONFIGS)
        queries: List of query dicts with ground_truth
        save_csv: If True, save per-query results to CSV
        verbose: If True, print detailed progress
    
    Returns:
        AblationMetrics with aggregated results
    """
    config = get_config(ablation_name)
    
    print(f"\n{'═'*70}")
    print(f"ABLATION: {ablation_name}")
    print(f"Description: {config.description}")
    print(f"Queries: {len(queries)}")
    print(f"{'═'*70}\n")
    
    # Initialize CSV
    csv_path = os.path.join(RESULTS_DIR, f"ablation_{ablation_name}.csv")
    csv_headers = [
        "query_id", "query", "gt_video_id", "gt_start", "gt_end",
        "latency_sec", "topics_retrieved", "scenes_retrieved", "vlm_calls",
        "top_topic_score",
        "recall@1_video", "recall@3_video",
        "recall@1_topics", "recall@3_topics", "recall@5_topics",
        "recall@3_scenes", "recall@5_scenes", "recall@10_scenes",
        "oracle_scene_recall"
    ]
    
    if save_csv:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
    
    # Collect all results
    all_results = []
    
    start_time = time.time()
    
    for i, q in enumerate(queries):
        query = q.get("query", "")
        gt = q.get("ground_truth", {})
        
        if verbose or (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(queries)}] {query[:50]}...")
        
        try:
            result = evaluate_ablation_query(query, gt, config)
            all_results.append(result)
            
            # Write to CSV
            if save_csv:
                row = [
                    i,
                    query[:80],
                    result["gt_video_id"],
                    f"{result['gt_start']:.1f}",
                    f"{result['gt_end']:.1f}",
                    f"{result['latency_sec']:.2f}",
                    result["topics_retrieved"],
                    result["scenes_retrieved"],
                    result["vlm_calls"],
                    f"{result['top_topic_score']:.4f}",
                    result["recall_at_1_video"],
                    result["recall_at_3_video"],
                    result["recall_at_1_topics"],
                    result["recall_at_3_topics"],
                    result["recall_at_5_topics"],
                    result["recall_at_3_scenes"],
                    result["recall_at_5_scenes"],
                    result["recall_at_10_scenes"],
                    result["oracle_scene_recall"],
                ]
                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
        
        except Exception as e:
            print(f"[ERROR] Query {i} failed: {e}")
            continue
    
    total_time = time.time() - start_time
    
    # Compute aggregated metrics
    if not all_results:
        print("[WARN] No results collected!")
        return AblationMetrics(ablation_name, config.description, 0)
    
    latencies = [r["latency_sec"] for r in all_results]
    
    metrics = AblationMetrics(
        ablation_name=ablation_name,
        description=config.description,
        num_queries=len(all_results),
        
        avg_latency_sec=np.mean(latencies),
        p50_latency_sec=np.percentile(latencies, 50),
        p95_latency_sec=np.percentile(latencies, 95),
        
        avg_topics_retrieved=np.mean([r["topics_retrieved"] for r in all_results]),
        avg_scenes_retrieved=np.mean([r["scenes_retrieved"] for r in all_results]),
        avg_vlm_calls=np.mean([r["vlm_calls"] for r in all_results]),
        
        recall_at_1_video=np.mean([r["recall_at_1_video"] for r in all_results]),
        recall_at_3_video=np.mean([r["recall_at_3_video"] for r in all_results]),
        
        recall_at_1_topics=np.mean([r["recall_at_1_topics"] for r in all_results]),
        recall_at_3_topics=np.mean([r["recall_at_3_topics"] for r in all_results]),
        recall_at_5_topics=np.mean([r["recall_at_5_topics"] for r in all_results]),
        
        recall_at_3_scenes=np.mean([r["recall_at_3_scenes"] for r in all_results]),
        recall_at_5_scenes=np.mean([r["recall_at_5_scenes"] for r in all_results]),
        recall_at_10_scenes=np.mean([r["recall_at_10_scenes"] for r in all_results]),
        
        oracle_scene_recall=np.mean([r["oracle_scene_recall"] for r in all_results]),
        
        min_topic_score=config.min_topic_score,
        min_scene_score=config.min_scene_score,
        top_scenes_per_topic=config.top_scenes_per_topic,
    )
    
    # Print summary
    print(f"\n{'─'*50}")
    print(f"RESULTS: {ablation_name}")
    print(f"{'─'*50}")
    print(f"  Queries: {metrics.num_queries}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Avg latency: {metrics.avg_latency_sec:.2f}s")
    print(f"")
    print(f"  VIDEO RECALL:")
    print(f"    R@1: {metrics.recall_at_1_video:.1%}")
    print(f"    R@3: {metrics.recall_at_3_video:.1%}")
    print(f"")
    print(f"  TOPIC RECALL:")
    print(f"    R@1: {metrics.recall_at_1_topics:.1%}")
    print(f"    R@3: {metrics.recall_at_3_topics:.1%}")
    print(f"    R@5: {metrics.recall_at_5_topics:.1%}")
    print(f"")
    print(f"  SCENE RECALL (within GT video):")
    print(f"    R@3: {metrics.recall_at_3_scenes:.1%}")
    print(f"    R@5: {metrics.recall_at_5_scenes:.1%}")
    print(f"    R@10: {metrics.recall_at_10_scenes:.1%}")
    print(f"    Oracle: {metrics.oracle_scene_recall:.1%}")
    print(f"")
    print(f"  CSV: {csv_path}")
    print(f"{'─'*50}\n")
    
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# BATCH ABLATION RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def run_main_ablations(queries: List[Dict], save_results: bool = True) -> Dict[str, AblationMetrics]:
    """
    Run all 6 main ablation studies.
    """
    print("\n" + "═"*70)
    print("RUNNING MAIN ABLATIONS (6)")
    print("═"*70 + "\n")
    
    results = {}
    
    for name in ABLATION_CONFIGS.keys():
        metrics = run_single_ablation(name, queries, save_csv=save_results)
        results[name] = metrics
    
    if save_results:
        # Save summary JSON
        summary = {name: asdict(m) for name, m in results.items()}
        summary_path = os.path.join(RESULTS_DIR, "main_ablations_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n[INFO] Summary saved to: {summary_path}")
    
    return results


def run_threshold_sweep(queries: List[Dict], save_results: bool = True) -> Dict[str, AblationMetrics]:
    """
    Run threshold sensitivity study (27 configurations).
    """
    print("\n" + "═"*70)
    print("RUNNING THRESHOLD SENSITIVITY SWEEP (27 configs)")
    print("═"*70 + "\n")
    
    results = {}
    
    for name in THRESHOLD_CONFIGS.keys():
        metrics = run_single_ablation(name, queries, save_csv=save_results)
        results[name] = metrics
    
    if save_results:
        # Save summary JSON
        summary = {name: asdict(m) for name, m in results.items()}
        summary_path = os.path.join(RESULTS_DIR, "threshold_sweep_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n[INFO] Summary saved to: {summary_path}")
    
    return results


def run_all_ablations(queries: List[Dict]) -> Dict[str, AblationMetrics]:
    """
    Run all ablation studies (6 main + 27 threshold = 33 total).
    """
    print("\n" + "═"*70)
    print("RUNNING ALL ABLATIONS (33 total)")
    print("═"*70 + "\n")
    
    # Run main ablations
    main_results = run_main_ablations(queries, save_results=True)
    
    # Run threshold sweep
    threshold_results = run_threshold_sweep(queries, save_results=True)
    
    # Combine
    all_results = {**main_results, **threshold_results}
    
    # Save combined summary
    summary = {name: asdict(m) for name, m in all_results.items()}
    summary_path = os.path.join(RESULTS_DIR, "all_ablations_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[INFO] All results saved to: {RESULTS_DIR}")
    
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_comparison_table(results: Dict[str, AblationMetrics]) -> str:
    """
    Generate a markdown comparison table for ablation results.
    """
    lines = []
    lines.append("# Ablation Study Results\n")
    lines.append("| Ablation | R@3 Topics | R@3 Scenes | R@5 Scenes | Latency (s) | VLM Calls |")
    lines.append("|----------|------------|------------|------------|-------------|-----------|")
    
    for name, m in results.items():
        lines.append(
            f"| {name} | {m.recall_at_3_topics:.1%} | {m.recall_at_3_scenes:.1%} | "
            f"{m.recall_at_5_scenes:.1%} | {m.avg_latency_sec:.2f} | {m.avg_vlm_calls:.1f} |"
        )
    
    return "\n".join(lines)


def generate_threshold_heatmap_data(results: Dict[str, AblationMetrics]) -> Dict:
    """
    Generate data for threshold sensitivity heatmap visualization.
    """
    # Filter to threshold configs only
    threshold_results = {k: v for k, v in results.items() if k.startswith("threshold_")}
    
    data = {
        "min_topic_score": [],
        "min_scene_score": [],
        "top_scenes_per_topic": [],
        "recall_at_3_scenes": [],
        "latency_sec": [],
        "vlm_calls": [],
    }
    
    for name, m in threshold_results.items():
        data["min_topic_score"].append(m.min_topic_score)
        data["min_scene_score"].append(m.min_scene_score)
        data["top_scenes_per_topic"].append(m.top_scenes_per_topic)
        data["recall_at_3_scenes"].append(m.recall_at_3_scenes)
        data["latency_sec"].append(m.avg_latency_sec)
        data["vlm_calls"].append(m.avg_vlm_calls)
    
    return data


# ══════════════════════════════════════════════════════════════════════════════
# RUN ABLATIONS (SKIP BASELINE)
# ══════════════════════════════════════════════════════════════════════════════

def run_ablations_skip_baseline(
    queries: List[Dict],
    baseline_metrics: Dict,
    save_results: bool = True
) -> Dict[str, AblationMetrics]:
    """
    Run ablation studies, skipping baseline since it's already computed.
    Uses baseline_metrics from existing CSV.
    """
    print("\n" + "═"*70)
    print("RUNNING ABLATIONS (SKIPPING BASELINE - ALREADY IN CSV)")
    print("═"*70 + "\n")
    
    # Create baseline AblationMetrics from existing data
    baseline_am = AblationMetrics(
        ablation_name="baseline",
        description="Hierarchical retrieval (from existing CSV)",
        num_queries=baseline_metrics.get("num_queries", len(queries)),
        avg_latency_sec=baseline_metrics.get("avg_latency_sec", 0),
        avg_topics_retrieved=baseline_metrics.get("avg_topics_retrieved", 0),
        avg_scenes_retrieved=baseline_metrics.get("avg_scenes_retrieved", 0),
        avg_vlm_calls=baseline_metrics.get("avg_vlm_calls", 0),
        recall_at_1_video=baseline_metrics.get("recall_at_1_video", 0),
        recall_at_3_video=baseline_metrics.get("recall_at_3_video", 0),
        recall_at_1_topics=baseline_metrics.get("recall_at_1_topics", 0),
        recall_at_3_topics=baseline_metrics.get("recall_at_3_topics", 0),
        recall_at_5_topics=baseline_metrics.get("recall_at_5_topics", 0),
        recall_at_3_scenes=baseline_metrics.get("recall_at_3_scenes", 0),
        recall_at_5_scenes=baseline_metrics.get("recall_at_5_scenes", 0),
        recall_at_10_scenes=baseline_metrics.get("recall_at_10_scenes", 0),
        oracle_scene_recall=baseline_metrics.get("oracle_scene_recall", 0),
    )
    
    results = {"baseline": baseline_am}
    
    print(f"[INFO] Using baseline metrics from CSV:")
    print(f"       R@3 Topics: {baseline_am.recall_at_3_topics:.1%}")
    print(f"       R@3 Scenes: {baseline_am.recall_at_3_scenes:.1%}")
    print(f"       Avg Latency: {baseline_am.avg_latency_sec:.2f}s\n")
    
    # Run remaining ablations (skip baseline)
    ablations_to_run = [name for name in ABLATION_CONFIGS.keys() if name != "baseline"]
    
    print(f"[INFO] Running {len(ablations_to_run)} remaining ablations...")
    
    for name in ablations_to_run:
        metrics = run_single_ablation(name, queries, save_csv=save_results)
        results[name] = metrics
    
    if save_results:
        # Save summary JSON
        summary = {name: asdict(m) for name, m in results.items()}
        summary_path = os.path.join(RESULTS_DIR, "main_ablations_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n[INFO] Summary saved to: {summary_path}")
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run ablation studies for Video-RAG pipeline")
    
    parser.add_argument("--ablation", type=str, default=None,
                        help="Run specific ablation (e.g., 'baseline', 'flat_retrieval')")
    parser.add_argument("--all", action="store_true",
                        help="Run all ablations (6 main + 27 threshold)")
    parser.add_argument("--main", action="store_true",
                        help="Run only main 6 ablations")
    parser.add_argument("--threshold", action="store_true",
                        help="Run only threshold sweep (27 configs)")
    
    parser.add_argument("--queries", type=int, default=None,
                        help="Number of queries to use (default: all)")
    parser.add_argument("--regenerate", action="store_true",
                        help="Regenerate queries (ignore cache)")
    
    parser.add_argument("--from-csv", type=str, default=None,
                        help="Load queries from existing evaluation CSV (skips baseline)")
    
    parser.add_argument("--list", action="store_true",
                        help="List available ablation configurations")
    
    args = parser.parse_args()
    
    # List ablations
    if args.list:
        print("\n" + "="*70)
        print("AVAILABLE ABLATION CONFIGURATIONS")
        print("="*70 + "\n")
        
        print("MAIN ABLATIONS (7):")
        for name, cfg in ABLATION_CONFIGS.items():
            print(f"  • {name}: {cfg.description}")
        
        print(f"\nTHRESHOLD SWEEP (27):")
        print(f"  • threshold_<mts>_<mss>_<tsp>")
        print(f"    mts = MIN_TOPIC_SCORE: [0.1, 0.3, 0.5]")
        print(f"    mss = MIN_SCENE_SCORE: [0.01, 0.05, 0.1]")
        print(f"    tsp = TOP_SCENES_PER_TOPIC: [3, 5, 8]")
        
        print()
        return
    
    print("\n" + "═"*70)
    print("VIDEO-RAG ABLATION STUDY RUNNER")
    print("═"*70 + "\n")
    
    # Load global index
    print("[INFO] Loading global index...")
    load_global_index()
    
    # Load queries - either from CSV or generate
    baseline_metrics = {}
    
    if args.from_csv:
        print(f"[INFO] Loading queries from: {args.from_csv}")
        all_queries, baseline_metrics = load_queries_from_csv(args.from_csv)
    else:
        print("[INFO] Loading/generating queries...")
        all_queries = generate_all_queries(use_cache=not args.regenerate)
    
    if args.queries:
        queries = stratified_sample(all_queries, n=args.queries)
        print(f"[INFO] Using {len(queries)} sampled queries")
    else:
        queries = all_queries
        print(f"[INFO] Using all {len(queries)} queries")
    
    # Run ablations
    if args.ablation:
        # Single ablation
        run_single_ablation(args.ablation, queries)
    
    elif args.all:
        # All ablations
        if baseline_metrics:
            # Skip baseline, use metrics from CSV
            results = run_ablations_skip_baseline(queries, baseline_metrics)
            # Also run threshold sweep
            threshold_results = run_threshold_sweep(queries)
            results.update(threshold_results)
        else:
            results = run_all_ablations(queries)
        
        # Generate comparison table
        table = generate_comparison_table(results)
        table_path = os.path.join(RESULTS_DIR, "comparison_table.md")
        with open(table_path, 'w') as f:
            f.write(table)
        print(f"\n[INFO] Comparison table saved to: {table_path}")
    
    elif args.main:
        # Main ablations only
        if baseline_metrics:
            run_ablations_skip_baseline(queries, baseline_metrics)
        else:
            run_main_ablations(queries)
    
    elif args.threshold:
        # Threshold sweep only
        run_threshold_sweep(queries)
    
    else:
        # Default: run main ablations (skip baseline if from CSV)
        print("[INFO] No ablation specified. Running main ablations.")
        print("[TIP] Use --all for full suite, --ablation <name> for specific, --list to see options")
        if baseline_metrics:
            run_ablations_skip_baseline(queries, baseline_metrics)
        else:
            run_main_ablations(queries)


if __name__ == "__main__":
    main()

