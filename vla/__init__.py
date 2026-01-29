"""VLA optimization: caching, temporal skipping, and token merging."""

from .vla_cache import VLACache, get_layer_mask_schedule, quick_frame_similarity
from .vla_model import VLACacheModel, create_vla_cache_model
from .temporal_vla import TemporalVLA, TemporalConfig, TemporalStats, FrameDecision
from .token_merging import (
    HierarchicalTokenMerger,
    TokenMergingConfig,
    TokenMerger,
    SaliencyScorer,
    HierarchicalPooler,
)
from .efficient_vla import EfficientVLAPipeline, EfficientVLAConfig
from .speculative import (
    SpeculativePruningConfig,
    SpeculativeTokenSignals,
    SpeculativePruner,
)

__all__ = [
    # VLA Cache (original)
    "VLACache",
    "VLACacheModel",
    "create_vla_cache_model",
    "get_layer_mask_schedule",
    "quick_frame_similarity",
    # Temporal skipping
    "TemporalVLA",
    "TemporalConfig",
    "TemporalStats",
    "FrameDecision",
    # Token merging
    "HierarchicalTokenMerger",
    "TokenMergingConfig",
    "TokenMerger",
    "SaliencyScorer",
    "HierarchicalPooler",
    # Combined pipeline
    "EfficientVLAPipeline",
    "EfficientVLAConfig",
    # Speculative pruning
    "SpeculativePruningConfig",
    "SpeculativeTokenSignals",
    "SpeculativePruner",
]
