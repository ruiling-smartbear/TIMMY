from __future__ import annotations

from typing import Any

from music_eval.audio import AudioData
from music_eval.metrics import summarize_signal
from music_eval.models import Finding, ManifestEntry
from music_eval.plugins.base import AnalysisConfig, MetricOutput


class IntegrityMetric:
    name = "integrity"

    def evaluate(
        self,
        entry: ManifestEntry,
        candidate: AudioData,
        reference: AudioData | None,
        config: AnalysisConfig,
    ) -> MetricOutput:
        del reference
        signal_metrics, dropouts = summarize_signal(
            candidate.samples,
            candidate.sample_rate,
            silence_dbfs=config.silence_dbfs,
            window_seconds=config.window_seconds,
            minimum_dropout_seconds=config.minimum_dropout_seconds,
            clip_threshold=config.clip_threshold,
        )
        metrics: dict[str, Any] = signal_metrics
        metrics.update(
            {
                "sample_rate": candidate.sample_rate,
                "channels": candidate.channels,
                "frames": candidate.frames,
                "duration_seconds": candidate.duration_seconds,
                "sample_width_bytes": candidate.sample_width_bytes,
                "sample_format": candidate.sample_format,
            }
        )
        findings: list[Finding] = []
        expected = entry.expectations
        if candidate.frames == 0:
            findings.append(Finding("empty_audio", "failure", "audio contains no frames"))
        elif float(metrics["peak_dbfs"]) <= config.silence_dbfs:
            findings.append(
                Finding(
                    "all_silent",
                    "failure",
                    f"entire audio is at or below {config.silence_dbfs:.1f} dBFS",
                )
            )
        if dropouts:
            findings.append(
                Finding(
                    "dropout_detected",
                    "warning",
                    f"detected {len(dropouts)} silent region(s) lasting at least "
                    f"{config.minimum_dropout_seconds:.2f}s",
                )
            )
        if float(metrics["clipped_sample_ratio"]) > 0.0:
            findings.append(
                Finding(
                    "clipping_detected",
                    "warning",
                    f"{float(metrics['clipped_sample_ratio']):.4%} of samples are near full scale",
                )
            )
        dc_offsets = metrics["dc_offset"]
        if isinstance(dc_offsets, list) and any(
            abs(float(value)) > 0.05 for value in dc_offsets
        ):
            findings.append(
                Finding("dc_offset", "warning", "one or more channels exceed 0.05 DC offset")
            )

        if expected.duration_seconds is not None:
            error = abs(candidate.duration_seconds - expected.duration_seconds)
            metrics["duration_error_seconds"] = error
            if error > expected.duration_tolerance_seconds:
                findings.append(
                    Finding(
                        "duration_mismatch",
                        "failure",
                        f"duration {candidate.duration_seconds:.3f}s differs from expected "
                        f"{expected.duration_seconds:.3f}s by {error:.3f}s",
                    )
                )
        if expected.sample_rate is not None and candidate.sample_rate != expected.sample_rate:
            findings.append(
                Finding(
                    "sample_rate_mismatch",
                    "failure",
                    f"sample rate {candidate.sample_rate} does not match expected "
                    f"{expected.sample_rate}",
                )
            )
        if expected.channels is not None and candidate.channels != expected.channels:
            findings.append(
                Finding(
                    "channel_mismatch",
                    "failure",
                    f"channel count {candidate.channels} does not match expected "
                    f"{expected.channels}",
                )
            )
        gated_metrics = (
            ("clipped_sample_ratio", expected.max_clipped_ratio, "clipping_limit"),
            (
                "silent_window_ratio",
                expected.max_silent_window_ratio,
                "silence_ratio_limit",
            ),
            ("longest_dropout_seconds", expected.max_dropout_seconds, "dropout_limit"),
        )
        for metric_name, limit, code in gated_metrics:
            if limit is not None and float(metrics[metric_name]) > limit:
                findings.append(
                    Finding(
                        code,
                        "failure",
                        f"{metric_name}={float(metrics[metric_name]):.6g} exceeds limit "
                        f"{limit:.6g}",
                    )
                )
        return MetricOutput(metrics=metrics, findings=findings, dropouts=dropouts)
