"""
Decision router for mortgage AI outputs.

Routes each evaluation result to one of three outcomes:
  - AUTO_APPROVE  : confidence > HIGH_THRESHOLD (0.85)
  - HUMAN_REVIEW  : confidence between LOW_THRESHOLD (0.60) and HIGH_THRESHOLD
  - REJECT        : confidence < LOW_THRESHOLD (0.60)

Thresholds are configurable at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


HIGH_THRESHOLD: float = 0.85
LOW_THRESHOLD: float = 0.60


class RoutingDecision(str, Enum):
    AUTO_APPROVE = "AUTO_APPROVE"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    REJECT = "REJECT"


@dataclass
class RouteResult:
    """Outcome of the routing step."""

    decision: RoutingDecision
    confidence: float
    reason: str
    high_threshold: float = HIGH_THRESHOLD
    low_threshold: float = LOW_THRESHOLD

    @property
    def label(self) -> str:
        return self.decision.value

    def __repr__(self) -> str:
        return f"RouteResult({self.decision.value}, confidence={self.confidence:.3f})"


def route(
    confidence: float,
    flags: list[str] | None = None,
    high_threshold: float = HIGH_THRESHOLD,
    low_threshold: float = LOW_THRESHOLD,
) -> RouteResult:
    """
    Route an evaluation to AUTO_APPROVE, HUMAN_REVIEW, or REJECT.

    Args:
        confidence: Combined confidence score in [0.0, 1.0].
        flags: Warning messages from confidence scoring (used in reason).
        high_threshold: Auto-approve cutoff (default 0.85).
        low_threshold: Reject cutoff (default 0.60).

    Returns:
        RouteResult with the routing decision and a human-readable reason.
    """
    flags = flags or []
    flag_summary = "; ".join(flags[:3]) if flags else "none"

    if confidence >= high_threshold:
        return RouteResult(
            decision=RoutingDecision.AUTO_APPROVE,
            confidence=confidence,
            reason=(
                f"Confidence {confidence:.3f} exceeds auto-approve threshold "
                f"{high_threshold}. Flags: {flag_summary}"
            ),
            high_threshold=high_threshold,
            low_threshold=low_threshold,
        )
    elif confidence >= low_threshold:
        return RouteResult(
            decision=RoutingDecision.HUMAN_REVIEW,
            confidence=confidence,
            reason=(
                f"Confidence {confidence:.3f} is in human-review range "
                f"[{low_threshold}, {high_threshold}). Flags: {flag_summary}"
            ),
            high_threshold=high_threshold,
            low_threshold=low_threshold,
        )
    else:
        return RouteResult(
            decision=RoutingDecision.REJECT,
            confidence=confidence,
            reason=(
                f"Confidence {confidence:.3f} below rejection threshold "
                f"{low_threshold}. Flags: {flag_summary}"
            ),
            high_threshold=high_threshold,
            low_threshold=low_threshold,
        )
