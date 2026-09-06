# Metric plugin API

A metric plugin supplies a unique `name` and an `evaluate` method:

```python
from music_eval import MetricOutput


class ExampleMetric:
    name = "example"

    def evaluate(self, entry, candidate, reference, config):
        return MetricOutput(metrics={"score": 0.75})
```

Plugins receive the parsed manifest entry, decoded candidate, optional decoded
reference, and shared analysis configuration. They return namespaced metrics,
findings, and optional dropout spans.

An external package registers the plugin in its own `pyproject.toml`:

```toml
[project.entry-points."music_eval.metrics"]
example = "example_package:ExampleMetric"
```

After installation:

```bash
music-eval list-metrics
music-eval evaluate manifest.jsonl --metrics integrity,example
```

## Plugin rules

- Do not mutate audio arrays or manifest entries.
- Include checkpoint and preprocessing identity in learned-metric output.
- Do not download checkpoints during module import.
- Treat an unavailable optional dependency as an actionable plugin error.
- Keep raw evidence in `metrics`; use findings only for documented policy.
- Do not write into another plugin's namespace.

Plugin exceptions become attributed failures so batch evaluation can finish
without allowing CI to pass.

## Bundled optional learned metrics

`audiobox_aesthetics` is bundled but imports neither PyTorch nor Meta's package
until evaluation begins. Install it with `music-eval[audiobox]`. Its output
records the checkpoint, package version, preprocessing contract, duration-
weighted CE/CU/PC/PQ track scores, per-axis summaries, and every 10-second
window. The plugin does not create findings or gates from learned scores.

`clap_alignment` is installed with `music-eval[clap]`. It reads the manifest's
positive `prompt` and optional `negative_prompts`, then reports per-window and
duration-weighted cosine similarity, hard-negative margin, and positive rank.
The implementation records its checkpoint, Transformers version, device, and
preprocessing contract. Missing prompts and unavailable dependencies are
attributed plugin failures; low learned scores remain ungated evidence.

`temporal_consistency` has no optional dependencies. It provides local spectral
features, transition distributions, first-to-last drift, non-local repetition,
and ending evidence. It intentionally produces no findings: a repeated chorus
and a collapsed generation can look alike at this level, so downstream policy
must use genre applicability and human or higher-level structural evidence.
