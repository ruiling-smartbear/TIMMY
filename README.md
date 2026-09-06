<p align="center">
  <img src="https://raw.githubusercontent.com/ruiling-smartbear/TIMMY/main/assets/timmy-logo-horizontal.png" alt="TIMMY" width="620">
</p>

# TIMMY

**Transparent, Interpretable, Modular Metrics for Your music.**

[![test](https://github.com/ruiling-smartbear/TIMMY/actions/workflows/test.yml/badge.svg)](https://github.com/ruiling-smartbear/TIMMY/actions/workflows/test.yml)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

TIMMY (`music-eval` on the command line) evaluates generated music from a
JSONL manifest and reports each kind of evidence separately: is the file
intact, does the serving path reproduce the reference, does the audio match
its prompt, how does it score on learned aesthetics, what does a listening
panel prefer. There is no overall score. Every number keeps its source, and
the only pass/fail decisions are the ones you write into the manifest.

The core needs numpy and nothing else. Learned metrics are optional extras
that never download a checkpoint on import.

## Quick start

```bash
python -m pip install -e .
python examples/generate_demo.py
music-eval evaluate examples/manifest.jsonl --output demo-report
```

The demo writes one clean tone and one tone with a one-second hole. The second
case declares `"max_dropout_seconds": 0.5`, so `music-eval` exits 1 and the
report names the finding:

```text
evaluated 2 file(s): 1 passed, 0 warnings, 1 failed
- failure · integrity:dropout_limit: longest_dropout_seconds=1.00003 exceeds limit 0.5
```

Each run writes `report.json` (stable schema, `schema_version` inside),
`report.md` for pull requests, and a standalone `report.html`.

## A manifest

One JSON object per line; paths are relative to the manifest file.

```json
{"id":"jpop-001-seed9","audio":"outputs/jpop-001-seed9.wav","reference":"references/jpop-001-seed9.wav","prompt":"Bright J-pop with female vocals and electric guitar","negative_prompts":["dark ambient drone","acoustic Delta blues"],"seed":9,"labels":{"genre":"j-pop","mood":["bright","energetic"],"vocals":"female","model":"candidate-v2"},"expectations":{"duration_seconds":16.0,"sample_rate":32000,"channels":2,"max_dropout_seconds":1.0,"min_reference_waveform_correlation":0.99}}
```

`labels` can hold any dimension with one or more values; the report groups
results by every one of them, so a strong average cannot hide a genre that
fails. `expectations` are the gates. Anything not declared there is reported
as evidence or, for a few signal defects, as a warning. The full contract is
in [docs/manifest.md](docs/manifest.md).

## Metrics

| Metric | What it measures | Extra install |
|---|---|---|
| `integrity` | format, levels, DC offset, clipping, silence, sample-accurate dropouts, stereo correlation, every expectation gate | none |
| `pairwise` | duration and level deltas, exact PCM match, aligned correlation and NRMSE against a reference run | none |
| `temporal_consistency` | windowed spectral drift, non-local repetition, ending shape | none |
| `production_quality` | BS.1770 loudness, loudness range, true-peak estimate, dynamics, bandwidth, click candidates | `[production]` |
| `audiobox_aesthetics` | Meta Audiobox Aesthetics CE/CU/PC/PQ per 10 s window | `[audiobox]` |
| `clap_alignment` | prompt similarity with hard-negative margin and rank (LAION CLAP) | `[clap]` |
| `muq_mulan_alignment` | the same view from MuQ-MuLan; weights are CC BY-NC | `[muq]` |
| `musecp_preservation` | edit-context preservation in five musical facets (MuseCPEval) | `[editing]` |

```bash
music-eval evaluate manifest.jsonl --metrics integrity,temporal_consistency,production_quality
music-eval list-metrics
```

Field names, preprocessing and what each metric does not claim are in
[docs/metrics.md](docs/metrics.md). Third-party metrics plug in through an
entry point ([docs/plugin-api.md](docs/plugin-api.md)).

## On real output

[examples/minimax-music3](examples/minimax-music3/README.md) runs the offline
metrics on the five MiniMax Music 3 reference takes published in the
SGLang-Omni cookbook:

| Take | Status | Duration | LUFS | LRA | Est. true peak | Clicks/s |
|---|---|---:|---:|---:|---:|---:|
| lo-fi hip-hop | warning | 30.02 s | -13.5 | 7.7 | +0.37 dBTP | 6.9 |
| J-pop | fail | 26.23 s | -16.0 | 9.9 | -1.62 dBTP | 0.2 |
| synthwave | pass | 30.02 s | -18.9 | 6.8 | -1.13 dBTP | 0.7 |
| acoustic folk | pass | 30.02 s | -14.3 | 9.4 | -0.23 dBTP | 0.2 |
| orchestral | warning | 30.02 s | -17.6 | 16.4 | +0.01 dBTP | 0.1 |

The J-pop take fails only because it is shorter than the 30 s the manifest
expects. The 6.9 clicks per second on the lo-fi track are the vinyl crackle
its caption asked for, and the 16 LU loudness range on the orchestral take is
the "build from restrained to triumphant" in its caption; the same numbers on
the folk ballad would be defects. That is why those metrics report and never
gate.

## Workflows

**Serving fidelity.** Put the reference run's WAV in `reference` and gate on
`min_reference_waveform_correlation` or `max_reference_nrmse`. This is the
check for "did the new build produce the same audio", not a similarity score
for independently sampled music.

**Benchmark suites.** `init-suite` writes a fixed manifest (eight genres, two
prompts each, your seeds) that any model runner can fill;
`init-longform-study` builds matched 30 s / 2 min / 4 min / 8 min ladders.
See [docs/suites.md](docs/suites.md).

**Blind listening studies.** `init-listening-study` turns a comparison
manifest into a model-blind A/B page with seeded order, hidden repeat trials
and a private organizer key, with criteria presets taken from MusicPrefs and
SongEval or your own list; `serve-listening-study` collects responses;
`analyze-listening-study` reports preference rates, Bradley-Terry strengths
with rater-level bootstrap intervals, repeat and inter-rater agreement, and a
side-bias check. See [docs/listening-studies.md](docs/listening-studies.md).

**Corpus distribution.** `compare-distributions` takes precomputed embeddings
from one pinned encoder and reports Frechet distance, Kernel Audio Distance,
diversity, nearest-neighbour distances and k-NN precision/recall/density/
coverage, overall and per label. See
[docs/distribution-evaluation.md](docs/distribution-evaluation.md).

**Metric meta-evaluation.** `prepare-fidelity-perturbations` writes a
Gaussian-noise ladder with provenance; `analyze-metric-ordering` checks
whether a metric's scores follow the ladder (Kendall tau-b, ordered-pair
accuracy). See [docs/metric-meta-evaluation.md](docs/metric-meta-evaluation.md).

**Music editing.** `musecp_preservation` scores how much of the original an
edit kept, facet by facet, through a version-pinned MuseCPEval adapter. See
[docs/model-support.md](docs/model-support.md).

## Status

Version 0.8.0. The signal metrics, reports, listening-study tooling,
distribution statistics and meta-evaluation are tested on every push
(Python 3.10 to 3.13). The learned-metric adapters are tested with injected
fakes plus a real MuseCPEval install; no pinned-checkpoint campaign on a
public dataset has been run yet. What is implemented, what is validated and
what is only proposed is tracked in the
[RFC](docs/rfcs/0001-timmy-evidence-framework.md) and the
[qualification record](docs/qualification.md); sources for every cited method
are in the [source ledger](docs/research/source-ledger.md).

## Documentation

[Architecture](docs/architecture.md) ·
[Manifest](docs/manifest.md) ·
[Metrics](docs/metrics.md) ·
[Plugin API](docs/plugin-api.md) ·
[Suites](docs/suites.md) ·
[Listening studies](docs/listening-studies.md) ·
[Distribution evaluation](docs/distribution-evaluation.md) ·
[Metric meta-evaluation](docs/metric-meta-evaluation.md) ·
[Model support](docs/model-support.md) ·
[Datasets](docs/datasets.md) ·
[Music-theory research notes](docs/music-theory-research.md) ·
[Evaluation landscape](docs/music-evaluation-landscape.md) ·
[Changelog](CHANGELOG.md)

## Development

```bash
python -m pip install -e '.[test]'
ruff check .
mypy src/music_eval
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for what a new metric has to document.

## License and citation

Apache-2.0. Cite TIMMY with [CITATION.cff](CITATION.cff) and cite the
underlying method, model and dataset papers listed in the
[source ledger](docs/research/source-ledger.md); this project's citation does
not replace theirs.
