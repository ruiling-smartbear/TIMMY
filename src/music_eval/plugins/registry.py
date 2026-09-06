from __future__ import annotations

from importlib.metadata import entry_points

from music_eval.plugins.audiobox_aesthetics import AudioboxAestheticsMetric
from music_eval.plugins.base import MetricPlugin
from music_eval.plugins.clap_alignment import ClapAlignmentMetric
from music_eval.plugins.integrity import IntegrityMetric
from music_eval.plugins.pairwise import PairwiseFidelityMetric

BUILTINS: dict[str, type[MetricPlugin]] = {
    "audiobox_aesthetics": AudioboxAestheticsMetric,
    "clap_alignment": ClapAlignmentMetric,
    "integrity": IntegrityMetric,
    "pairwise": PairwiseFidelityMetric,
}


def available_plugins() -> list[str]:
    external = [entry.name for entry in entry_points(group="music_eval.metrics")]
    return sorted(set(BUILTINS) | set(external))


def resolve_plugins(names: list[str] | None = None) -> list[MetricPlugin]:
    requested = names or ["integrity", "pairwise"]
    if len(set(requested)) != len(requested):
        raise ValueError("metric plugin names must be unique")
    external = {
        entry.name: entry for entry in entry_points(group="music_eval.metrics")
    }
    resolved: list[MetricPlugin] = []
    for name in requested:
        if name in BUILTINS:
            plugin = BUILTINS[name]()
        elif name in external:
            loaded = external[name].load()
            plugin = loaded() if isinstance(loaded, type) else loaded
        else:
            raise ValueError(
                f"unknown metric plugin '{name}'; available: {', '.join(available_plugins())}"
            )
        if getattr(plugin, "name", None) != name or not callable(
            getattr(plugin, "evaluate", None)
        ):
            raise ValueError(
                f"metric plugin '{name}' must expose matching name and callable evaluate"
            )
        resolved.append(plugin)
    return resolved
