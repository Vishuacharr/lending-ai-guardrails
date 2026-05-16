"""
Evaluation metrics for mortgage AI outputs.

Implements faithfulness, relevance, and accuracy scoring using
rule-based and lightweight string-matching techniques — no LLM calls.
"""

from __future__ import annotations

import re
import math
from collections import Counter
from typing import Any


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _tf_idf_similarity(a: str, b: str) -> float:
    """Compute cosine similarity between two strings using TF-IDF vectors."""
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    vocab = set(tokens_a) | set(tokens_b)
    if not vocab:
        return 0.0

    def tf(tokens: list[str], term: str) -> float:
        count = tokens.count(term)
        return count / len(tokens) if tokens else 0.0

    def idf(term: str) -> float:
        docs_with_term = sum(1 for tokens in [tokens_a, tokens_b] if term in tokens)
        return math.log((2 + 1) / (docs_with_term + 1)) + 1

    def vec(tokens: list[str]) -> dict[str, float]:
        return {term: tf(tokens, term) * idf(term) for term in vocab}

    va, vb = vec(tokens_a), vec(tokens_b)
    dot = sum(va[t] * vb[t] for t in vocab)
    mag_a = math.sqrt(sum(v**2 for v in va.values()))
    mag_b = math.sqrt(sum(v**2 for v in vb.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return round(dot / (mag_a * mag_b), 4)


def _field_overlap(extracted: dict[str, Any], ground_truth: dict[str, Any]) -> float:
    """
    Fraction of ground-truth fields correctly matched in extracted output.
    Numeric fields allow 1% tolerance; string fields use exact match.
    """
    if not ground_truth:
        return 1.0
    correct = 0
    for key, expected in ground_truth.items():
        actual = extracted.get(key)
        if actual is None:
            continue
        if isinstance(expected, (int, float)):
            try:
                tol = abs(expected) * 0.01
                if abs(float(actual) - float(expected)) <= max(tol, 0.01):
                    correct += 1
            except (ValueError, TypeError):
                pass
        else:
            if str(actual).strip().lower() == str(expected).strip().lower():
                correct += 1
    return round(correct / len(ground_truth), 4)


def score_faithfulness(
    ai_output: str,
    source_context: str,
) -> float:
    """
    Faithfulness: how well the AI output is grounded in the source document.

    Uses TF-IDF cosine similarity between AI output and the source context
    it was derived from. A hallucinated answer will diverge from source vocabulary.

    Returns a score in [0.0, 1.0].
    """
    if not source_context.strip():
        return 0.0
    return _tf_idf_similarity(ai_output, source_context)


def score_relevance(
    ai_output: str,
    query: str,
) -> float:
    """
    Relevance: how well the AI output addresses the query or task.

    Uses TF-IDF cosine similarity between AI output and the original query.

    Returns a score in [0.0, 1.0].
    """
    if not query.strip():
        return 0.0
    return _tf_idf_similarity(ai_output, query)


def score_accuracy(
    extracted_fields: dict[str, Any],
    ground_truth_fields: dict[str, Any],
) -> float:
    """
    Accuracy: fraction of ground-truth fields correctly extracted.

    Numeric fields tolerate ±1% difference; strings require exact match.

    Returns a score in [0.0, 1.0].
    """
    return _field_overlap(extracted_fields, ground_truth_fields)


def compute_composite_score(
    faithfulness: float,
    relevance: float,
    accuracy: float,
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> float:
    """
    Weighted composite of all three metrics.

    Default weights: faithfulness 40%, relevance 30%, accuracy 30%.
    Mortgage decisions weight faithfulness highest because hallucinations
    carry the largest regulatory risk.

    Returns a score in [0.0, 1.0].
    """
    wf, wr, wa = weights
    return round(wf * faithfulness + wr * relevance + wa * accuracy, 4)
