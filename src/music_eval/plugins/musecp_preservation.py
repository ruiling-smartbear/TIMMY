from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from music_eval.audio import AudioData
from music_eval.models import Finding, ManifestEntry
from music_eval.plugins.base import AnalysisConfig, MetricOutput

MUSECP_VERSION = "0.3.0"
MetricFunction = Callable[[str, str], object]


def _run_structural_score(original: str, edited: str) -> object:
    program = (
        "import json, sys; "
        "from musecpeval import structural_score; "
        "print(json.dumps(structural_score(sys.argv[1], sys.argv[2]), allow_nan=False))"
    )
    with tempfile.TemporaryDirectory(prefix="timmy-musecp-structure-") as work_dir:
        work_path = Path(work_dir)
        original_copy = work_path / "reference" / "input.wav"
        edited_copy = work_path / "candidate" / "input.wav"
        original_copy.parent.mkdir()
        edited_copy.parent.mkdir()
        shutil.copyfile(original, original_copy)
        shutil.copyfile(edited, edited_copy)
        completed = subprocess.run(
            [sys.executable, "-c", program, str(original_copy), str(edited_copy)],
            cwd=work_dir,
            check=False,
            stdout=subprocess.PIPE,
            text=True,
        )
    if completed.returncode:
        raise RuntimeError(
            f"MuseCPEval structural_score subprocess exited {completed.returncode}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("MuseCPEval structural_score returned invalid JSON") from error


def _load_functions() -> dict[str, MetricFunction]:
    try:
        module = import_module("musecpeval")
    except ImportError as error:
        raise RuntimeError(
            "MuseCPEval is optional; install music-eval[editing] in a dedicated environment"
        ) from error
    names = {
        "harmony_tonality": "harmony_score",
        "rhythm_meter": "rhythm_score",
        "melodic_content": "melody_score",
        "timbre_texture": "timbre_score",
    }
    functions = {
        group: getattr(module, function_name) for group, function_name in names.items()
    }
    functions["structural_form"] = _run_structural_score
    return functions


def _package_version() -> str:
    try:
        return version("musecpeval")
    except PackageNotFoundError:
        return "unknown"


def _json_value(value: object, path: str = "result") -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"MuseCPEval returned non-finite value at {path}")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise ValueError(f"MuseCPEval returned unsupported value at {path}: {type(value).__name__}")


class MuseCPPreservationMetric:
    """Expose MuseCPEval's independent music-context-preservation facets."""

    name = "musecp_preservation"

    def __init__(
        self,
        *,
        function_loader: Callable[[], dict[str, MetricFunction]] = _load_functions,
    ) -> None:
        self._function_loader = function_loader
        self._functions: dict[str, MetricFunction] | None = None

    def evaluate(
        self,
        entry: ManifestEntry,
        candidate: AudioData,
        reference: AudioData | None,
        config: AnalysisConfig,
    ) -> MetricOutput:
        del candidate, config
        if reference is None or entry.reference is None:
            return MetricOutput(
                metrics={"available": False, "reason": "reference_required"},
                findings=[
                    Finding(
                        "reference_required",
                        "warning",
                        "MuseCP preservation needs an original/reference WAV",
                    )
                ],
            )
        if self._functions is None:
            self._functions = self._function_loader()

        expected = {
            "harmony_tonality",
            "rhythm_meter",
            "structural_form",
            "melodic_content",
            "timbre_texture",
        }
        if set(self._functions) != expected:
            raise ValueError("MuseCPEval adapter must provide all five metric families")

        original_path = str(entry.reference)
        edited_path = str(entry.audio)
        facets = {
            group: _json_value(function(original_path, edited_path), group)
            for group, function in self._functions.items()
        }
        return MetricOutput(
            metrics={
                "available": True,
                "implementation": "musecpeval",
                "required_version": MUSECP_VERSION,
                "package_version": _package_version(),
                "original": original_path,
                "edited": edited_path,
                "facets": facets,
                "interpretation": (
                    "Facet scores are preserved separately. They measure context preservation, "
                    "not whether the requested edit succeeded inside its target region."
                ),
            }
        )
