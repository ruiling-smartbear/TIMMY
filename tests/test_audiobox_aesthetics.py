from __future__ import annotations

from pathlib import Path

import numpy as np

from music_eval.audio import AudioData
from music_eval.models import ManifestEntry
from music_eval.plugins.audiobox_aesthetics import AudioboxAestheticsMetric
from music_eval.plugins.base import AnalysisConfig


class FakePredictor:
    def __init__(self) -> None:
        self.calls = 0
        self.seen = 0

    def forward(self, batch):
        self.calls += 1
        results = []
        for _item in batch:
            index = self.seen
            results.append(
                {
                    "CE": index + 1.0,
                    "CU": index + 2.0,
                    "PC": index + 3.0,
                    "PQ": index + 4.0,
                }
            )
            self.seen += 1
        return results


def _audio(seconds: float, sample_rate: int = 10) -> AudioData:
    samples = np.zeros((round(seconds * sample_rate), 2), dtype=np.float64)
    samples.setflags(write=False)
    return AudioData(samples, sample_rate, 2, 2)


def test_audiobox_preserves_window_scores_and_duration_weighted_mean():
    predictor = FakePredictor()
    metric = AudioboxAestheticsMetric(
        predictor_factory=lambda: predictor,
        tensor_factory=lambda samples: samples,
        window_seconds=10,
        batch_windows=2,
    )

    output = metric.evaluate(
        ManifestEntry(id="song", audio=Path("song.wav")),
        _audio(25),
        None,
        AnalysisConfig(),
    )

    assert predictor.calls == 2
    assert len(output.metrics["windows"]) == 3
    assert output.metrics["windows"][2]["end_seconds"] == 25.0
    assert output.metrics["track_scores"]["CE"] == 1.8
    assert output.metrics["axis_summary"]["CE"]["minimum"] == 1.0
    assert output.metrics["axis_summary"]["CE"]["worst_window_start_seconds"] == 0.0
    assert output.metrics["preprocessing"]["model_sample_rate_hz"] == 16000


def test_audiobox_reuses_one_predictor_across_entries():
    created = []

    def create_predictor():
        predictor = FakePredictor()
        created.append(predictor)
        return predictor

    metric = AudioboxAestheticsMetric(
        predictor_factory=create_predictor,
        tensor_factory=lambda samples: samples,
    )
    entry = ManifestEntry(id="song", audio=Path("song.wav"))

    metric.evaluate(entry, _audio(1), None, AnalysisConfig())
    metric.evaluate(entry, _audio(1), None, AnalysisConfig())

    assert len(created) == 1
    assert created[0].calls == 2


def test_audiobox_rejects_incomplete_predictor_output():
    class IncompletePredictor:
        def forward(self, batch):
            return [{"CE": 1.0} for _item in batch]

    metric = AudioboxAestheticsMetric(
        predictor_factory=IncompletePredictor,
        tensor_factory=lambda samples: samples,
    )

    try:
        metric.evaluate(
            ManifestEntry(id="song", audio=Path("song.wav")),
            _audio(1),
            None,
            AnalysisConfig(),
        )
    except ValueError as error:
        assert "invalid CU score" in str(error)
    else:
        raise AssertionError("invalid predictor output should fail")
