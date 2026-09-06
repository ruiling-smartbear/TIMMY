# music-eval

`music-eval` is a reproducible, explainable evaluation framework for generated
music. It keeps technical integrity, serving fidelity, prompt adherence,
learned quality estimates, distribution metrics, and human preference as
separate evidence instead of hiding them behind one universal score.

The current release is a lightweight, offline foundation. It validates PCM WAV
outputs, compares deterministic candidate/reference pairs, stratifies results
by musical attributes, and offers optional Meta Audiobox Aesthetics, CLAP, and
MuQ-MuLan alignment plugins alongside a boundary for future FAD and serving
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

Install learned-model extras in a dedicated virtual environment. Their
upstream packages include mutually versioned PyTorch, Torchaudio, and
Torchvision dependencies, so pip may otherwise upgrade an existing ML stack.

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

### `muq_mulan_alignment` (optional, non-commercial weights)

MuQ-MuLan provides a second, music-specific audio/text alignment view. It uses
Tencent AI Lab's official `OpenMuQ/MuQ-MuLan-large` checkpoint in float32,
resamples to 24 kHz mono, and reports the same window, hard-negative margin,
and rank evidence as CLAP:

```bash
python -m pip install -e '.[muq]'
music-eval evaluate manifest.jsonl \
  --metrics integrity,muq_mulan_alignment \
  --output report
```

The MuQ source code is MIT licensed, but the published model weights are
CC-BY-NC 4.0. Do not use those weights for commercial evaluation without
separate permission. As with CLAP, raw similarity has no universal threshold.

### `temporal_consistency`

This CPU-only metric exposes changes across a full track without downloading a
model. It reports two-second sliding-window RMS, spectral centroid, bandwidth,
95% rolloff, flatness, and zero-crossing rate; adjacent transition size;
first-to-last spectral drift; non-local near-duplicate evidence; and ending
level/discontinuity evidence.

```bash
music-eval evaluate manifest.jsonl \
  --metrics integrity,temporal_consistency \
  --output report
```

Near-duplicate ratios are deliberately not failures: repeated choruses,
minimalism, drones, and genuine generation collapse can produce similar signal
patterns. Genre-aware policy and listening evidence must interpret the result.

### `production_quality` (optional)

This layer measures delivery and mastering evidence separately from musical
taste:

- BS.1770-4 integrated loudness and EBU loudness range via `pyloudnorm`;
- sample peak, crest factor, 400 ms block-level dynamics, and per-channel peak;
- a clearly labeled 4× polyphase true-peak estimate;
- spectral rolloff and energy above 15/18/20 kHz;
- possible click/pop locations found from robust sample discontinuities.

```bash
python -m pip install -e '.[production]'
music-eval evaluate manifest.jsonl \
  --metrics integrity,production_quality \
  --output report
```

The true-peak result is diagnostic and is not represented as a certified EBU
meter. Click candidates and restricted bandwidth can be intentional musical
choices, so this plugin does not fail samples by default. LRA values from tracks
shorter than 60 seconds are retained but explicitly marked unstable, following
EBU guidance.

## Human blind listening studies

Objective metrics do not replace listening. The study builder creates a
model-blind A/B experiment with deterministic randomization and hidden repeat
trials for within-rater reliability. Start with a JSONL comparison manifest:

```json
{"id":"rock-01","prompt":"Raw garage rock with a memorable chorus","audio_a":"outputs/model-a-rock.wav","system_a":"model-a","audio_b":"outputs/model-b-rock.wav","system_b":"model-b","labels":{"genre":"rock"}}
```

Build and serve the study:

```bash
music-eval init-listening-study comparisons.jsonl \
  --output listening-study --title "Model A vs Model B" \
  --seed 20260906 --repeat-fraction 0.1
music-eval serve-listening-study listening-study --port 8000
```

The public directory contains sanitized PCM audio under hashed names,
randomized A/B positions, and a standalone browser interface. The builder does
not copy system fields, labels, or repeat markers into it. Organizers must still
keep their own title and prompt text neutral. The private answer key is written
next to the directory as `listening-study.organizer.json`; never upload or
serve it with the public study.

Responses submitted through the local server appear under
`listening-study/responses/`. Without a server, the page offers a response JSON
download instead. In-progress answers are stored locally in the browser and
restored after an accidental refresh. Analyze one or more response files with:

```bash
music-eval analyze-listening-study listening-study.organizer.json \
  listening-study/responses/*.json --output listening-report \
  --bootstrap-samples 1000 --bootstrap-seed 20260906
```

The report keeps overall preference, prompt alignment, musicality/structure,
and production quality separate. Each criterion includes raw preference rate,
Bradley–Terry strength, rater-level 95% bootstrap intervals, hidden-repeat and
cross-rater agreement, and an A/B side-choice diagnostic. See
[Blind listening studies](docs/listening-studies.md) for design and sampling
requirements.

## Corpus-level distribution comparison

Per-track scores can miss mode collapse: a model may produce several polished
tracks while covering only a narrow part of the requested music distribution.
`compare-distributions` consumes a JSONL file of precomputed audio embeddings
from one pinned model/checkpoint:

```json
{"id":"ref-rock-01","system":"reference","embedding":[0.12,-0.41,0.08],"embedding_model":"my-audio-encoder","checkpoint":"sha256:...","labels":{"genre":"rock"}}
```

```bash
music-eval compare-distributions embeddings.jsonl \
  --reference-system reference --neighbors 3 \
  --output distribution-report
```

For every candidate system, reports include Fréchet distance in the declared
embedding space, within-system cosine diversity, nearest-reference distances,
and k-NN precision/recall/density/coverage when the sample count is sufficient.
The same comparison is repeated by genre and every other supplied label when
both sides have at least two examples.

The command intentionally accepts embeddings rather than silently choosing or
downloading an encoder. It rejects mixed checkpoints and dimensions. Its
Fréchet result is **not automatically FAD**; that name is justified only when a
specific audio embedding and preprocessing protocol defines it. See
[Distribution evaluation](docs/distribution-evaluation.md).

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
- [Model support matrix](docs/model-support.md)
- [Blind listening studies](docs/listening-studies.md)
- [Distribution evaluation](docs/distribution-evaluation.md)
- [Contributing](CONTRIBUTING.md)

## Roadmap

1. Aligned serving-fidelity and repeatability adapters.
2. Independent music-text alignment with MuQ-MuLan.
3. Additional per-sample learned quality metrics such as MuQ-Eval.
4. Higher-level beat, harmony, vocal, and section analysis.
5. Calibrated encoder adapters for named FAD/MAD protocols.
6. Prospective power-planning helpers and prompt-level uncertainty intervals.

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
