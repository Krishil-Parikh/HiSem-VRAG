# ablation_config.py
"""
Ablation Study Configuration for Video-RAG Pipeline

Defines all ablation variants and their configurations.
This file is used by ablation_pipeline.py and ablation_runner.py.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
from enum import Enum


class SceneRankingMode(Enum):
    SEMANTIC = "semantic"  # max cos(q, frame) per scene
    HEURISTIC = "heuristic"  # position + frame count


class KeyframeMode(Enum):
    SEMANTIC_SMOOTHED = "semantic_smoothed"  # Top-K by cos sim + Gaussian smoothing
    SEMANTIC = "semantic"  # Top-K by cos sim only
    POSITION = "position"  # First, middle, last


@dataclass
class AblationConfig:
    """Configuration for a single ablation variant."""
    name: str
    description: str
    
    # ══════════════════════════════════════════════════════════════
    # Ablation 1: Hierarchical vs Flat Retrieval
    # ══════════════════════════════════════════════════════════════
    use_hierarchical: bool = True  # False = flat retrieval
    
    # ══════════════════════════════════════════════════════════════
    # Ablation 2: Query Decomposition
    # ══════════════════════════════════════════════════════════════
    use_decomposition: bool = True  # False = use original query only
    
    # ══════════════════════════════════════════════════════════════
    # Ablation 3: Scene Ranking Strategy
    # ══════════════════════════════════════════════════════════════
    scene_ranking_mode: SceneRankingMode = SceneRankingMode.SEMANTIC
    
    # ══════════════════════════════════════════════════════════════
    # Ablation 4 & 5: Keyframe Selection + Temporal Smoothing
    # ══════════════════════════════════════════════════════════════
    keyframe_mode: KeyframeMode = KeyframeMode.SEMANTIC_SMOOTHED
    temporal_smoothing_sigma: float = 1.5  # 0 = no smoothing
    
    # ══════════════════════════════════════════════════════════════
    # Ablation 6: Threshold Sensitivity
    # ══════════════════════════════════════════════════════════════
    min_topic_score: float = 0.7
    min_scene_score: float = 0.05
    top_scenes_per_topic: int = 3
    
    # Fixed parameters (not ablated); values follow the paper's online retrieval table
    top_k_global: int = 5
    top_videos: int = 3
    top_topics_per_video: int = 3
    top_frames_per_scene: int = 2
    vlm_workers: int = 4


# ══════════════════════════════════════════════════════════════════════════════
# PREDEFINED ABLATION CONFIGURATIONS
# ══════════════════════════════════════════════════════════════════════════════

ABLATION_CONFIGS: Dict[str, AblationConfig] = {
    # ───────────────────────────────────────────────────────────────
    # BASELINE (Full Hierarchical Pipeline)
    # ───────────────────────────────────────────────────────────────
    "baseline": AblationConfig(
        name="baseline",
        description="Full hierarchical pipeline with all features enabled",
        use_hierarchical=True,
        use_decomposition=True,
        scene_ranking_mode=SceneRankingMode.SEMANTIC,
        keyframe_mode=KeyframeMode.SEMANTIC_SMOOTHED,
        temporal_smoothing_sigma=1.5,
        min_topic_score=0.7,
        min_scene_score=0.05,
        top_scenes_per_topic=3,
    ),
    
    # ───────────────────────────────────────────────────────────────
    # ABLATION 1: Flat Retrieval (No Hierarchical)
    # ───────────────────────────────────────────────────────────────
    "flat_retrieval": AblationConfig(
        name="flat_retrieval",
        description="Flat single-stage retrieval over all scenes",
        use_hierarchical=False,  # ← ABLATED
        use_decomposition=True,
        scene_ranking_mode=SceneRankingMode.SEMANTIC,
        keyframe_mode=KeyframeMode.SEMANTIC_SMOOTHED,
        temporal_smoothing_sigma=1.5,
        min_topic_score=0.7,
        min_scene_score=0.05,
        top_scenes_per_topic=3,
    ),
    
    # ───────────────────────────────────────────────────────────────
    # ABLATION 2: No Query Decomposition
    # ───────────────────────────────────────────────────────────────
    "no_decomposition": AblationConfig(
        name="no_decomposition",
        description="Use original query only, no sub-questions",
        use_hierarchical=True,
        use_decomposition=False,  # ← ABLATED
        scene_ranking_mode=SceneRankingMode.SEMANTIC,
        keyframe_mode=KeyframeMode.SEMANTIC_SMOOTHED,
        temporal_smoothing_sigma=1.5,
        min_topic_score=0.7,
        min_scene_score=0.05,
        top_scenes_per_topic=3,
    ),
    
    # ───────────────────────────────────────────────────────────────
    # ABLATION 3: Heuristic Scene Ranking
    # ───────────────────────────────────────────────────────────────
    "heuristic_ranking": AblationConfig(
        name="heuristic_ranking",
        description="Order scenes by position/frame-count instead of cosine similarity",
        use_hierarchical=True,
        use_decomposition=True,
        scene_ranking_mode=SceneRankingMode.HEURISTIC,  # ← ABLATED
        keyframe_mode=KeyframeMode.SEMANTIC_SMOOTHED,
        temporal_smoothing_sigma=1.5,
        min_topic_score=0.7,
        min_scene_score=0.05,
        top_scenes_per_topic=3,
    ),
    
    # ───────────────────────────────────────────────────────────────
    # ABLATION 4a: Position-Based Keyframes
    # ───────────────────────────────────────────────────────────────
    "position_keyframes": AblationConfig(
        name="position_keyframes",
        description="Select first/middle/last frames instead of semantic selection",
        use_hierarchical=True,
        use_decomposition=True,
        scene_ranking_mode=SceneRankingMode.SEMANTIC,
        keyframe_mode=KeyframeMode.POSITION,  # ← ABLATED
        temporal_smoothing_sigma=1.5,
        min_topic_score=0.7,
        min_scene_score=0.05,
        top_scenes_per_topic=3,
    ),
    
    # ───────────────────────────────────────────────────────────────
    # ABLATION 4b: Semantic Keyframes (no smoothing)
    # ───────────────────────────────────────────────────────────────
    "semantic_keyframes_no_smooth": AblationConfig(
        name="semantic_keyframes_no_smooth",
        description="Semantic keyframe selection without temporal smoothing",
        use_hierarchical=True,
        use_decomposition=True,
        scene_ranking_mode=SceneRankingMode.SEMANTIC,
        keyframe_mode=KeyframeMode.SEMANTIC,  # ← ABLATED (no smoothing)
        temporal_smoothing_sigma=0.0,
        min_topic_score=0.7,
        min_scene_score=0.05,
        top_scenes_per_topic=3,
    ),
    
    # ───────────────────────────────────────────────────────────────
    # ABLATION 5: No Temporal Smoothing
    # ───────────────────────────────────────────────────────────────
    "no_smoothing": AblationConfig(
        name="no_smoothing",
        description="Disable Gaussian temporal smoothing for frame scores",
        use_hierarchical=True,
        use_decomposition=True,
        scene_ranking_mode=SceneRankingMode.SEMANTIC,
        keyframe_mode=KeyframeMode.SEMANTIC,  # Still semantic but no smooth
        temporal_smoothing_sigma=0.0,  # ← ABLATED
        min_topic_score=0.7,
        min_scene_score=0.05,
        top_scenes_per_topic=3,
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLD SENSITIVITY SWEEP (Ablation 6)
# ══════════════════════════════════════════════════════════════════════════════

THRESHOLD_SWEEP_VALUES = {
    "min_topic_score": [0.1, 0.3, 0.5],
    "min_scene_score": [0.01, 0.05, 0.1],
    "top_scenes_per_topic": [3, 5, 8],
}


def generate_threshold_sweep_configs() -> Dict[str, AblationConfig]:
    """
    Generate all 27 (3×3×3) threshold combinations.
    Returns dict with keys like 'threshold_0.1_0.01_3'.
    """
    configs = {}
    
    for mts in THRESHOLD_SWEEP_VALUES["min_topic_score"]:
        for mss in THRESHOLD_SWEEP_VALUES["min_scene_score"]:
            for tsp in THRESHOLD_SWEEP_VALUES["top_scenes_per_topic"]:
                name = f"threshold_{mts}_{mss}_{tsp}"
                configs[name] = AblationConfig(
                    name=name,
                    description=f"Threshold sweep: topic={mts}, scene={mss}, scenes/topic={tsp}",
                    use_hierarchical=True,
                    use_decomposition=True,
                    scene_ranking_mode=SceneRankingMode.SEMANTIC,
                    keyframe_mode=KeyframeMode.SEMANTIC_SMOOTHED,
                    temporal_smoothing_sigma=1.5,
                    min_topic_score=mts,
                    min_scene_score=mss,
                    top_scenes_per_topic=tsp,
                )
    
    return configs


# Generate threshold sweep configs on import
THRESHOLD_CONFIGS = generate_threshold_sweep_configs()


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_all_configs() -> Dict[str, AblationConfig]:
    """Get all ablation configs (6 main + 27 threshold sweep)."""
    all_configs = dict(ABLATION_CONFIGS)
    all_configs.update(THRESHOLD_CONFIGS)
    return all_configs


def get_config(name: str) -> AblationConfig:
    """Get a specific ablation config by name."""
    all_configs = get_all_configs()
    if name not in all_configs:
        raise ValueError(f"Unknown ablation config: {name}. Available: {list(all_configs.keys())}")
    return all_configs[name]


def list_ablations() -> List[str]:
    """List all available ablation names."""
    return list(get_all_configs().keys())


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║           ABLATION STUDY CONFIGURATIONS                       ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")
    
    print("MAIN ABLATIONS (6):")
    print("─" * 60)
    for name, cfg in ABLATION_CONFIGS.items():
        print(f"  • {name}: {cfg.description}")
    
    print(f"\nTHRESHOLD SWEEP CONFIGS ({len(THRESHOLD_CONFIGS)}):")
    print("─" * 60)
    print(f"  {len(THRESHOLD_CONFIGS)} configurations from:")
    print(f"    MIN_TOPIC_SCORE: {THRESHOLD_SWEEP_VALUES['min_topic_score']}")
    print(f"    MIN_SCENE_SCORE: {THRESHOLD_SWEEP_VALUES['min_scene_score']}")
    print(f"    TOP_SCENES_PER_TOPIC: {THRESHOLD_SWEEP_VALUES['top_scenes_per_topic']}")
    
    print(f"\nTOTAL CONFIGS: {len(get_all_configs())}")
