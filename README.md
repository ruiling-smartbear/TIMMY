# music-eval

`music-eval` is a reproducible, explainable evaluation framework for generated
music. It keeps technical integrity, serving fidelity, prompt adherence,
learned quality estimates, distribution metrics, and human preference as
separate evidence instead of hiding them behind one universal score.

The current release is a lightweight, offline foundation. It validates PCM WAV
outputs, compares deterministic candidate/reference pairs, stratifies results
by musical attributes, and offers optional Meta Audiobox Aesthetics and CLAP
alignment plugins alongside a boundary for future MuQ, FAD, and serving
integrations.

## Principles

- **No unexplained overall score.** Every number retains its meaning and source.
- **Explicit policy.** A signal anomaly is diagnostic unless the manifest sets
  a failure threshold.
- **Stratified results.** Genre, mood, instrumentation, vocals, duration, model,
  backend, and other dimensions are reported independently.
- **Reproducible inputs.** Prompt, seed, candidate, reference, labels, and
  expectations live in a versionable JSONL manifest.
- **Small core, optional models.** The base package does not download a neural
  checkpoint or require a GPU.

## Install

```bash
python -m pip install -e .
```

## Evaluate existing outputs

Create `manifest.jsonl`; paths are resolved relative to that file:

```json
{"id":"jpop-001-seed9","audio":"outputs/jpop-001-seed9.wav","reference":"references/jpop-001-seed9.wav","prompt":"Bright J-pop with female vocals and electric guitar","negative_prompts":["dark ambient drone","acoustic Delta blues"],"seed":9,"labels":{"genre":"j-pop","mood":["bright","energetic"],"instruments":["electric guitar","drums"],"vocals":"female","model":"candidate-v2"},"expectations":{"duration_seconds":16.0,"duration_tolerance_seconds":0.25,"sample_rate":32000,"channels":2,"max_dropout_seconds":1.0,"min_reference_waveform_correlation":0.99}}
```

Run the evaluator:

```bash
music-eval evaluate manifest.jsonl --output report
```

It writes:

- `report/report.json`: stable, machine-readable schema;
- `report/report.md`: pull-request and experiment summary;
- `report/report.html`: standalone human report.

The process exits nonzero for decode failures and explicit policy violations.
Ungated clipping, DC offset, and dropout observations are warnings. Use
`--fail-on-warning` only when a project intentionally wants that policy.

## Start the standard genre suite

The built-in `generative-music-v1` suite contains two prompts in each of eight
genres: J-pop, orchestral, synthwave, acoustic folk, lo-fi hip-hop, rock, blues,
and ambient. Two default seeds produce 32 independently reported cases.

```bash
music-eval init-suite benchmark.jsonl \
  --suite generative-music-v1 \
  --seeds 9 17 \
  --duration-seconds 16 \
  --sample-rate 32000 \
  --channels 2
```

The generated manifest points to `outputs/<case>-seed<seed>.wav`. A model runner
can fill those paths without coupling its dependencies to this project.

```bash
music-eval list-suites
music-eval list-metrics
```

## Built-in metrics

### `integrity`

- format, sample rate, channels, frame count, and duration;
- RMS and peak dBFS;
- per-channel level and DC offset;
- clipped-sample ratio;
- silent-window ratio and contiguous dropout regions;
- stereo correlation;
- explicit metadata and signal gates.

### `pairwise`

When a manifest includes `reference`, the evaluator reports duration and RMS
deltas plus exact PCM equality. Waveform correlation and NRMSE are available
only when sample rate and tensor shape match. These aligned-waveform measures
are useful for deterministic serving parity; they are not valid general
perceptual metrics for independently sampled music.

Select plugins explicitly when needed:

```bash
music-eval evaluate manifest.jsonl --metrics integrity,pairwise
```

Third-party packages can register plugins through the `music_eval.metrics`
Python entry-point group. See [Plugin API](docs/plugin-api.md).

### `audiobox_aesthetics` (optional)

Install Meta's open no-reference aesthetic predictor separately so the base
package stays small and never downloads a checkpoint implicitly:

```bash
python -m pip install -e '.[audiobox]'
music-eval evaluate manifest.jsonl \
  --metrics integrity,audiobox_aesthetics \
  --output report
```

The plugin reports Content Enjoyment (CE), Content Usefulness (CU), Production
Complexity (PC), and Production Quality (PQ). It follows the official 16 kHz
mono, non-overlapping 10-second window preprocessing. In addition to the
duration-weighted track score, it preserves every window score, the 10th
percentile, minimum, and worst-window location. These learned estimates are
diagnostic evidence, not pass/fail policy or a universal music-quality score.

### `clap_alignment` (optional)

CLAP measures whether the audio matches its positive prompt. When the manifest
provides `negative_prompts`, this plugin additionally reports the best negative,
positive-minus-negative margin, and positive rank:

```bash
python -m pip install -e '.[clap]'
music-eval evaluate manifest.jsonl \
  --metrics integrity,clap_alignment \
  --output report
```

It uses `laion/clap-htsat-unfused`, high-quality resampling to 48 kHz mono, and
non-overlapping 10-second windows. Reports retain track-level duration-weighted
cosine similarities and every window so a strong opening cannot conceal later
prompt drift. Raw similarity has no universal pass threshold; hard-negative
margin and within-case rank are the more interpretable comparisons.

## Grouped evidence

Labels are multi-valued. One sample may belong to `genre=synthwave`,
`mood=[dark, nostalgic]`, and `instruments=[synthesizer, electronic drums]`.
Reports calculate counts, pass rate, mean duration, mean RMS, silence, dropout,
and finding frequencies for every value in every dimension. A strong aggregate
result therefore cannot hide a genre-specific failure.

## Demo

The demo creates one healthy tone and one tone with an injected one-second
dropout:

```bash
python examples/generate_demo.py
music-eval evaluate examples/manifest.jsonl --output demo-report
```

The second case deliberately violates its explicit dropout limit, so a nonzero
exit code proves the gate catches the injected defect.

## Documentation

- [Architecture](docs/architecture.md)
- [Manifest contract](docs/manifest.md)
- [Metric plugin API](docs/plugin-api.md)
- [Contributing](CONTRIBUTING.md)

## Roadmap

1. Aligned serving-fidelity and repeatability adapters.
2. Independent music-text alignment with MuQ-MuLan.
3. Additional per-sample learned quality metrics such as MuQ-Eval.
4. Corpus-level distribution metrics such as FAD/MAD.
5. A/B listening studies and model-serving adapters.

Each layer will remain visible and independently disableable. Learned metrics
will report model/checkpoint identity and will not become aesthetic ground
truth by default.

## Development

```bash
python -m pip install -e '.[test]'
ruff check .
mypy src/music_eval
pytest
```

## License

Apache-2.0.
