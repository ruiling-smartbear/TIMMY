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
reference, and shared analysis configuration. They return a flat metrics
dictionary, which the evaluator stores under the plugin's `name`, plus
findings and optional dropout spans.

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

Plugin exceptions become attributed failures so batch evaluation can finish
without allowing CI to pass.

## Bundled metrics

The bundled plugins, their extras, preprocessing and report fields are
described in [metrics.md](metrics.md).
