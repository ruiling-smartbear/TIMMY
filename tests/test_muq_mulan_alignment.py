from __future__ import annotations

from pathlib import Path

import numpy as np

from music_eval.audio import AudioData
from music_eval.models import ManifestEntry
from music_eval.plugins.base import AnalysisConfig
from music_eval.plugins.muq_mulan_alignment import MuQMuLanAlignmentMetric


class FakeBackend:
    device_name = "fake"

    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float64)
        self.offset = 0
        self.calls = []

    def score(self, audio_windows, sample_rate, texts):
        self.calls.append((len(audio_windows), sample_rate, tuple(texts)))
        start = self.offset
        self.offset += len(audio_windows)
        return self.scores[start : self.offset]


def _audio(seconds: float, sample_rate: int = 10) -> AudioData:
    samples = np.zeros((round(seconds * sample_rate), 2), dtype=np.float64)
    samples.setflags(write=False)
    return AudioData(samples, sample_rate, 2, 2)


def test_muq_mulan_reports_margin_rank_windows_and_provenance():
    backend = FakeBackend([[0.8, 0.2], [0.6, 0.7], [0.9, 0.1]])
    metric = MuQMuLanAlignmentMetric(
        backend_factory=lambda: backend,
        window_seconds=10,
        batch_windows=2,
    )
    entry = ManifestEntry(
        id="song",
        audio=Path("song.wav"),
        prompt="bright synthwave",
        negative_prompts=("ambient drone",),
    )

    output = metric.evaluate(entry, _audio(25), None, AnalysisConfig())

    assert [call[0] for call in backend.calls] == [2, 1]
    assert output.metrics["track"]["positive_similarity"] == 0.74
    assert output.metrics["track"]["margin"] == 0.36
    assert output.metrics["track"]["positive_rank"] == 1
    assert output.metrics["windows"][1]["positive_rank"] == 2
    assert output.metrics["weights_license"] == "CC-BY-NC-4.0"
    assert output.metrics["precision"] == "float32"


def test_muq_mulan_requires_prompt():
    metric = MuQMuLanAlignmentMetric(
        backend_factory=lambda: FakeBackend([[0.5]])
    )

    try:
        metric.evaluate(
            ManifestEntry(id="song", audio=Path("song.wav")),
            _audio(1),
            None,
            AnalysisConfig(),
        )
    except ValueError as error:
        assert "requires a nonempty manifest prompt" in str(error)
    else:
        raise AssertionError("prompt-free MuQ-MuLan evaluation should fail")
