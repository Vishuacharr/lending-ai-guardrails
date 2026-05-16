"""Core evaluation engine package."""

from core.evaluator import EvaluationHarness, EvaluationResult
from core.metrics import score_faithfulness, score_relevance, score_accuracy, compute_composite_score
from core.confidence import compute_confidence, ConfidenceResult
from core.router import route, RoutingDecision, RouteResult
from core.audit_logger import AuditLogger, AuditRecord

__all__ = [
    "EvaluationHarness",
    "EvaluationResult",
    "score_faithfulness",
    "score_relevance",
    "score_accuracy",
    "compute_composite_score",
    "compute_confidence",
    "ConfidenceResult",
    "route",
    "RoutingDecision",
    "RouteResult",
    "AuditLogger",
    "AuditRecord",
]
