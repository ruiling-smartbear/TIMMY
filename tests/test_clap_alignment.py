from __future__ import annotations

from pathlib import Path

import numpy as np

from music_eval.audio import AudioData
from music_eval.models import ManifestEntry
from music_eval.plugins.base import AnalysisConfig
from music_eval.plugins.clap_alignment import ClapAlignmentMetric


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


def test_clap_reports_hard_negative_margin_rank_and_windows():
    backend = FakeBackend(
        [
            [0.8, 0.2, 0.3],
            [0.7, 0.2, 0.3],
            [0.6, 0.2, 0.3],
        ]
    )
    metric = ClapAlignmentMetric(
        backend_factory=lambda: backend,
        window_seconds=10,
        batch_windows=2,
    )
    entry = ManifestEntry(
        id="song",
        audio=Path("song.wav"),
        prompt="bright synthwave",
        negative_prompts=("ambient drone", "acoustic blues"),
    )

    output = metric.evaluate(entry, _audio(25), None, AnalysisConfig())

    assert [call[0] for call in backend.calls] == [2, 1]
    assert output.metrics["track"]["positive_similarity"] == 0.72
    assert output.metrics["track"]["best_negative_similarity"] == 0.3
    assert output.metrics["track"]["margin"] == 0.42
    assert output.metrics["track"]["positive_rank"] == 1
    assert output.metrics["track"]["positive_top1"] is True
    assert len(output.metrics["windows"]) == 3
    assert output.metrics["windows"][2]["end_seconds"] == 25.0
    assert output.metrics["summary"]["positive_similarity"]["minimum"] == 0.6


def test_clap_rank_exposes_when_negative_prompt_wins():
    backend = FakeBackend([[0.2, 0.5, 0.1]])
    metric = ClapAlignmentMetric(backend_factory=lambda: backend)
    entry = ManifestEntry(
        id="song",
        audio=Path("song.wav"),
        prompt="rock",
        negative_prompts=("ambient", "jazz"),
    )

    output = metric.evaluate(entry, _audio(1), None, AnalysisConfig())

    assert output.metrics["track"]["positive_rank"] == 2
    assert output.metrics["track"]["positive_top1"] is False
    assert output.metrics["track"]["margin"] == -0.3


def test_clap_without_negatives_reports_similarity_without_fake_margin():
    backend = FakeBackend([[0.4]])
    metric = ClapAlignmentMetric(backend_factory=lambda: backend)
    entry = ManifestEntry(id="song", audio=Path("song.wav"), prompt="solo piano")

    output = metric.evaluate(entry, _audio(1), None, AnalysisConfig())

    assert output.metrics["track"]["positive_rank"] == 1
    assert output.metrics["track"]["margin"] is None
    assert "margin" not in output.metrics["summary"]


def test_clap_requires_prompt():
    metric = ClapAlignmentMetric(backend_factory=lambda: FakeBackend([[0.5]]))

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
        raise AssertionError("prompt-free CLAP evaluation should fail")


def test_clap_rejects_wrong_backend_shape():
    metric = ClapAlignmentMetric(backend_factory=lambda: FakeBackend([[0.5, 0.2]]))
    entry = ManifestEntry(id="song", audio=Path("song.wav"), prompt="piano")

    try:
        metric.evaluate(entry, _audio(1), None, AnalysisConfig())
    except ValueError as error:
        assert "expected (1, 1)" in str(error)
    else:
        raise AssertionError("wrong similarity shape should fail")
