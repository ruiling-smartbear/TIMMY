# Contributing

Contributions should preserve the separation between measurable evidence and
project-specific policy.

## Setup

```bash
python -m pip install -e '.[test]'
ruff check .
mypy src/music_eval
pytest
```

## Metric contributions

A new metric should document:

- exactly what it measures and does not measure;
- preprocessing, model, and checkpoint identity;
- valid audio lengths, sample rates, and alignment assumptions;
- expected runtime and hardware;
- calibration evidence before any default failure threshold;
- synthetic or redistributable fixtures that cover failure behavior.

Do not add a learned metric to the required core dependency set. Package it as
an optional extra or external entry-point plugin.

## Tests

Regression tests must demonstrate that the proposed failure is caught. Audio
fixtures should be generated during the test or use an explicitly documented,
redistributable license. Do not commit copyrighted commercial music.
