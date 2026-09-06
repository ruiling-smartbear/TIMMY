from __future__ import annotations

import json
import math
import shutil
import wave
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from music_eval.audio import AudioData, read_wav

MAD_FIDELITY_SIGMAS = tuple(round(step * 0.02, 2) for step in range(11))


def _sigma_text(sigma: float) -> str:
    """Shortest decimal that round-trips ``sigma``, so distinct levels never collide."""
    return np.format_float_positional(float(sigma), trim="0")


def _sigma_slug(sigma: float) -> str:
    return _sigma_text(sigma).replace(".", "p")


def _write_pcm16(path: Path, samples: NDArray[np.float64], sample_rate: int) -> None:
    integers = np.rint(np.clip(samples, -1.0, 32767.0 / 32768.0) * 32768.0).astype(
        "<i2"
    )
    with wave.open(str(path), "wb") as output:
        output.setnchannels(samples.shape[1])
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(integers.tobytes())


def _validate_sigmas(sigmas: tuple[float, ...]) -> None:
    if not sigmas:
        raise ValueError("at least one noise sigma is required")
    if len(set(sigmas)) != len(sigmas):
        raise ValueError("noise sigmas must be unique")
    if any(not math.isfinite(sigma) or sigma < 0 for sigma in sigmas):
        raise ValueError("noise sigmas must be finite and nonnegative")
    if tuple(sorted(sigmas)) != sigmas:
        raise ValueError("noise sigmas must be in increasing order")
    slugs = [_sigma_slug(sigma) for sigma in sigmas]
    if len(set(slugs)) != len(slugs):
        raise ValueError(f"noise sigmas must map to unique file name slugs, got {slugs}")


def _perturb(
    audio: AudioData, sigma: float, generator: np.random.Generator
) -> tuple[NDArray[np.float64], float]:
    noisy = audio.samples + generator.normal(0.0, sigma, size=audio.samples.shape)
    clipped_ratio = float(np.mean((noisy < -1.0) | (noisy > 32767.0 / 32768.0)))
    return np.asarray(noisy, dtype=np.float64), clipped_ratio


def prepare_fidelity_perturbations(
    sources: list[Path],
    output_dir: Path,
    *,
    sigmas: tuple[float, ...] = MAD_FIDELITY_SIGMAS,
    seed: int = 20260906,
) -> dict[str, Any]:
    """Create deterministic Gaussian-noise ordering fixtures.

    The default levels match the fidelity perturbation strengths reported by
    Huang et al. The output is a local diagnostic fixture, not a reproduction
    of their corpus-scale experiment.
    """

    if not sources:
        raise ValueError("at least one source WAV is required")
    _validate_sigmas(sigmas)
    if output_dir.exists():
        raise ValueError(f"output already exists: {output_dir}")
    for source in sources:
        if not source.is_file():
            raise ValueError(f"source WAV does not exist: {source}")

    audio_dir = output_dir / "audio"
    reference_dir = output_dir / "reference"
    audio_dir.mkdir(parents=True)
    reference_dir.mkdir()
    cases = []
    manifest_lines = []
    seed_sequence = np.random.SeedSequence(seed)
    children = iter(seed_sequence.spawn(len(sources) * len(sigmas)))
    for source_index, source in enumerate(sources):
        decoded = read_wav(source)
        reference_name = f"{source_index:04d}-{source.name}"
        reference_path = reference_dir / reference_name
        shutil.copyfile(source, reference_path)
        for sigma in sigmas:
            child = next(children)
            perturbed, clipped_ratio = _perturb(
                decoded, sigma, np.random.default_rng(child)
            )
            candidate_name = (
                f"{source_index:04d}-{source.stem}-noise-{_sigma_slug(sigma)}.wav"
            )
            candidate_path = audio_dir / candidate_name
            _write_pcm16(candidate_path, perturbed, decoded.sample_rate)
            case_id = f"{source_index:04d}-{source.stem}-noise-{_sigma_slug(sigma)}"
            manifest_lines.append(
                json.dumps(
                    {
                        "id": case_id,
                        "audio": str(candidate_path.relative_to(output_dir)),
                        "reference": str(reference_path.relative_to(output_dir)),
                        "labels": {
                            "condition": "gaussian-noise",
                            "noise_sigma": _sigma_text(sigma),
                        },
                    },
                    separators=(",", ":"),
                )
            )
            cases.append(
                {
                    "id": case_id,
                    "source": str(source.resolve()),
                    "noise_sigma": sigma,
                    "clipped_sample_ratio_before_pcm_encoding": clipped_ratio,
                    "spawn_key": list(child.spawn_key),
                }
            )

    (output_dir / "manifest.jsonl").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )
    provenance = {
        "schema_version": 1,
        "protocol": "mad-fidelity-gaussian-noise-strengths",
        "claim": "metric sensitivity to increasing additive Gaussian noise",
        "seed": seed,
        "noise_sigmas": list(sigmas),
        "encoding": "PCM signed 16-bit WAV; samples outside PCM range are clipped",
        "cases": cases,
        "limitation": (
            "This fixture generator matches the reported noise strengths only. It does not "
            "reproduce the paper's FMA-Pop corpus size, embeddings, or validation results."
        ),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance
