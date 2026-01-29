"""Target-guided speculative pruning for lossless draft verification.

This module adds a smarter pruning path that keeps tokens likely to be
validated by the target model. It uses target-model signals (confidence,
agreement, entropy, optional attention) to prune draft tokens instead of
blindly dropping a fixed ratio.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class SpeculativePruningConfig:
    """Configuration for target-guided speculative pruning."""

    keep_ratio: float = 0.5
    min_keep_ratio: float = 0.2
    max_keep_ratio: float = 0.9
    agreement_weight: float = 0.35
    target_confidence_weight: float = 0.35
    draft_confidence_weight: float = 0.2
    attention_weight: float = 0.1
    entropy_weight: float = 0.2
    min_target_confidence: float = 0.15


@dataclass
class SpeculativeTokenSignals:
    """Optional extra signals for pruning decisions."""

    attention_scores: Optional[np.ndarray] = None
    target_entropies: Optional[np.ndarray] = None


class SpeculativePruner:
    """Compute keep masks using target-model verification signals."""

    def __init__(self, config: SpeculativePruningConfig = None):
        self.config = config or SpeculativePruningConfig()

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(logits)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def _entropy(self, probs: np.ndarray) -> np.ndarray:
        safe = np.clip(probs, 1e-8, 1.0)
        return -np.sum(safe * np.log(safe), axis=-1)

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        if scores is None:
            return None
        min_val = np.min(scores)
        max_val = np.max(scores)
        if np.isclose(max_val, min_val):
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

    def _top_token(self, logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs = self._softmax(logits)
        top_ids = np.argmax(probs, axis=-1)
        top_scores = np.max(probs, axis=-1)
        return top_ids, top_scores

    def score_tokens(
        self,
        draft_logits: np.ndarray,
        target_logits: np.ndarray,
        signals: Optional[SpeculativeTokenSignals] = None,
    ) -> np.ndarray:
        """Score token positions by likelihood of target acceptance.

        Returns scores in [0, 1], higher means more important to keep.
        """
        signals = signals or SpeculativeTokenSignals()

        draft_ids, draft_conf = self._top_token(draft_logits)
        target_ids, target_conf = self._top_token(target_logits)

        agreement = (draft_ids == target_ids).astype(np.float32)
        agreement = self._normalize(agreement)

        target_conf = self._normalize(target_conf)
        draft_conf = self._normalize(draft_conf)

        target_probs = self._softmax(target_logits)
        entropies = self._entropy(target_probs)
        entropies = self._normalize(entropies)
        entropy_signal = 1.0 - entropies

        attention = self._normalize(signals.attention_scores)
        if attention is None:
            attention = np.zeros_like(entropy_signal)

        if signals.target_entropies is not None:
            entropy_signal = 1.0 - self._normalize(signals.target_entropies)

        score = (
            self.config.agreement_weight * agreement
            + self.config.target_confidence_weight * target_conf
            + self.config.draft_confidence_weight * draft_conf
            + self.config.attention_weight * attention
            + self.config.entropy_weight * entropy_signal
        )

        return np.clip(score, 0.0, 1.0)

    def keep_mask(
        self,
        draft_logits: np.ndarray,
        target_logits: np.ndarray,
        signals: Optional[SpeculativeTokenSignals] = None,
    ) -> np.ndarray:
        """Return a boolean keep mask for token positions."""
        scores = self.score_tokens(draft_logits, target_logits, signals)

        target_conf = self._top_token(target_logits)[1]
        keep_min = target_conf >= self.config.min_target_confidence

        keep_ratio = np.clip(
            self.config.keep_ratio,
            self.config.min_keep_ratio,
            self.config.max_keep_ratio,
        )
        keep_count = max(1, int(round(keep_ratio * len(scores))))

        top_indices = np.argsort(scores)[-keep_count:]
        keep = np.zeros_like(scores, dtype=bool)
        keep[top_indices] = True

        return keep & keep_min
