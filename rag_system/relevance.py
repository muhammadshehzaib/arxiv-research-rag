"""Reranker relevance scores are ranking signals, not answer probabilities."""

import math
import os


# Bump when scoring changes so cached answers cannot retain obsolete source scores.
SCORING_VERSION = 2
THRESHOLD_ENV_KEYS = {
    "cohere": "REFUSAL_THRESHOLD_COHERE",
    "sentence-transformers": "REFUSAL_THRESHOLD_SENTENCE_TRANSFORMERS",
    "flashrank": "REFUSAL_THRESHOLD_FLASHRANK",
}


def finite_raw_score(score):
    """Return a JSON-safe numeric score, or None when the provider did not score it."""
    try:
        value = float(score)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def normalize_relevance_score(score, provider):
    """Keep bounded provider scores; sigmoid only explicit CrossEncoder logits.

    No transformation here calibrates the probability that an answer is correct.
    Missing, invalid, and unsupported scores fail closed with zero relevance.
    """
    value = finite_raw_score(score)
    if value is None:
        return 0.0
    if provider == "sentence-transformers":
        # Stable sigmoid, including very large positive/negative logits.
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)
    if provider in ("cohere", "flashrank") and 0.0 <= value <= 1.0:
        return value
    return 0.0


def get_relevance_threshold(provider):
    """Use the provider override, then the legacy global setting, then 0.70.

    These are heuristic relevance cutoffs and need validation on each model/corpus.
    """
    env_key = THRESHOLD_ENV_KEYS.get(provider)
    value = os.getenv(env_key) if env_key else None
    if value is None:
        value = os.getenv("REFUSAL_THRESHOLD", "0.70")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("Reranker relevance thresholds must be greater than 0 and at most 1.")
    return threshold
