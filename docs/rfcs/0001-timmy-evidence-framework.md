# RFC 0001: TIMMY as an evidence framework for generated music

- Status: Draft for implementation and external review
- Author: Ruilin Gao
- Last updated: 2026-09-06

## Summary

TIMMY stands for **Transparent, Interpretable, Modular Metrics for Your
music**. It is an independent evaluation framework, not a universal music
quality score. Every result names the claim, protocol, input revision,
implementation, checkpoint, aggregation rule, uncertainty, and known blind
spots that give the number meaning.

The initial package remains named `music-eval` and keeps the `music-eval` CLI
to avoid needless compatibility churn. TIMMY is the project identity.

## Why this exists

Generated-music systems make heterogeneous claims: clean audio, prompt
alignment, long-form coherence, faithful local editing, useful stems, vocal
intelligibility, creative control, and listener preference. No single metric
validly identifies all of them. Current research instead supports a portfolio
of evidence:

- per-track integrity and production diagnostics;
- prompt/audio alignment with hard negatives;
- learned, multidimensional aesthetic estimates;
- corpus-level fidelity and diversity;
- music-context preservation for edits;
- ordered perturbation tests that audit the metrics themselves;
- blinded human preference and workflow studies.

## Decisions

### 1. Preserve axes instead of manufacturing an overall score

TIMMY will not average CE, prompt similarity, KAD, loudness, musical form, and
listener preference. These quantities have different units, populations, and
failure modes. Reports may show a claim-specific decision only when the
benchmark author supplies a visible policy.

### 2. Require provenance for every learned or corpus metric

Reports retain package version, checkpoint identity, preprocessing, windowing,
aggregation, reference corpus, and relevant license restrictions. Scores from
different encoders or references are not silently compared.

### 3. Audit metrics with controlled orderings

The `analyze-metric-ordering` command measures whether a metric follows a
known perturbation or control order using Kendall tau-b and ordered-pair
accuracy. Ties remain visible. A high score proves sensitivity only to that
specific intervention; it does not establish perceptual validity.

### 4. Reuse authoritative implementations through thin adapters

KAD follows the estimator in `kadtk`. Music-edit context preservation calls
the pinned MIT-licensed MuseCPEval package. TIMMY validates and records their
outputs but does not fork the mathematical definitions. Optional model stacks
live in isolated extras because their dependencies and licenses differ.

### 5. Treat time as first-class evidence

Track means can hide a strong opening and a broken ending. Learned alignment
and aesthetic plugins retain window scores. Temporal diagnostics retain local
transitions, first-to-last drift, repetition evidence, and ending behavior.
TIMMY now generates matched 30-second/two-/four-/eight-minute prompt-and-seed
manifests. This controls the experiment but does not turn a short-window
average into evidence of eight-minute musical form; listener-calibrated
degradation analysis remains a separate qualification step.

### 6. Keep human evaluation honest

TIMMY currently supports model-blind randomized A/B studies with hidden repeat
trials, rater-level bootstrap intervals, cross-rater agreement, and side-bias
diagnostics. It will use the term MUSHRA only for protocols that actually meet
the relevant ITU-R requirements, including the appropriate references and
anchors.

## Evidence matrix

| Claim | Implemented evidence | Required supplement | Status |
| --- | --- | --- | --- |
| file/runtime integrity | PCM validation, duration, clipping, silence, dropout | serving adapter for model-specific parity | implemented |
| deterministic serving fidelity | exact PCM, correlation, NRMSE for aligned files | none when alignment assumptions hold | implemented |
| production quality | BS.1770 loudness, dynamics, bandwidth, click candidates | trained ears for artistic intent | implemented |
| prompt adherence | CLAP and MuQ-MuLan, windows, hard-negative margin/rank | prompt-specific listener rating | implemented, optional |
| no-reference aesthetics | Audiobox CE/CU/PC/PQ with worst windows | domain and culture validation | implemented, optional |
| corpus fidelity/diversity | Fréchet embedding distance, KAD, PRDC, diversity | pinned encoder/reference and sufficient N | implemented |
| edit context preservation | MuseCPEval five-facet adapter | target-region success and seam audibility | implemented, optional |
| temporal consistency | transitions, drift, repetition and ending evidence | musical-form and long-duration qualification | implemented |
| matched long-form design | prompt/seed matched duration ladders with stable grouping | generated outputs, degradation analysis, blinded structure ratings | implemented protocol |
| metric validity | ordered perturbation meta-evaluation | human-preference correlation and held-out perturbations | implemented |
| listener preference | blinded A/B, reliability, intervals, side bias | recruitment/power plan and preregistered policy | implemented |
| vocal intelligibility | none | separator + ASR/PER adapter and human phrasing study | proposed |
| local edit success | preservation only | declared interval, instruction success, seam tests | proposed |
| workflow/agency | none | event log and controlled user study | proposed |
| provenance/originality | none | C2PA inspection and separate similarity/memorization protocol | proposed |

## Testing standard

Every new method must include:

1. a positive fixture it should recognize;
2. a negative or counterexample it should reject or distinguish;
3. malformed/non-finite dependency output handling;
4. provenance fields in machine-readable output;
5. documentation of what a passing result cannot prove.

Heavy third-party checkpoints may be mocked in unit tests, but qualification
must separately run the pinned real model on public or releasable fixtures.

## Upstream contribution strategy

TIMMY should mature as its own Apache-2.0 project. Runtime repositories such as
SGLang-Omni should not inherit all optional checkpoints and MIR dependencies.
After the schema and CLI stabilize, an upstream RFC can propose a small,
optional adapter that exports generated artifacts and attaches a TIMMY report
to model qualification. Any such PR should include only the adapter and a
pinned example protocol, not vendor the evaluation framework into serving.

## Non-goals

- deciding whether music is objectively good;
- inferring therapeutic benefit from affect features;
- treating provenance metadata as proof that content is true or human-made;
- using waveform correlation for independently sampled generations;
- using non-commercial checkpoints in a commercial service without permission;
- calling a listening page MUSHRA when it does not implement the standard.

## Open questions

- Which named encoder/reference pairs should TIMMY qualify for FAD, KAD, and MAD?
- What minimum prompt and sample counts are affordable without unstable rankings?
- Should region-aware editing be a TIMMY-native protocol around MuseCPEval or an
  upstream contribution to that project?
- Which long-form boundaries should be listener-calibrated by genre and duration?

The working notes this RFC was distilled from are in
[research/report-source.md](../research/report-source.md).
