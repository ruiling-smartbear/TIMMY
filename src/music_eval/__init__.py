"""Explainable evaluation for generated music outputs."""

from music_eval.audio import AudioData
from music_eval.evaluator import evaluate_entry
from music_eval.models import Dropout, EvaluationResult, Finding, ManifestEntry
from music_eval.plugins.base import AnalysisConfig, MetricOutput, MetricPlugin

__all__ = [
    "AnalysisConfig",
    "AudioData",
    "Dropout",
    "EvaluationResult",
    "Finding",
    "ManifestEntry",
    "MetricOutput",
    "MetricPlugin",
    "evaluate_entry",
]
__version__ = "0.6.0"
