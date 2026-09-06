# Architecture

`music-eval` separates orchestration, metrics, policy, and presentation.

```text
JSONL manifest
    │
    ├── candidate/reference WAV decoding
    │
    ├── metric plugins
    │     ├── integrity, pairwise
    │     ├── temporal_consistency, production_quality
    │     ├── learned: audiobox_aesthetics, clap_alignment, muq_mulan_alignment
    │     └── musecp_preservation (edit context)
    │
    ├── explicit expectation gates
    │
    ├── per-sample results
    │
    ├── arbitrary label-dimension aggregation
    │
    └── JSON / Markdown / HTML reports

Pairwise listening JSONL
    │
    ├── PCM rewrite + anonymous content-hash filenames
    ├── seeded trial order and A/B assignment
    ├── public browser study ──→ validated response JSON
    └── private organizer key ──→ Bradley–Terry + reliability reports

Embedding JSONL (one pinned space)
    │
    ├── provenance + dimension validation
    ├── global candidate/reference distribution comparison
    ├── label-stratified comparisons
    └── JSON / Markdown / HTML corpus reports

Metric scores at ordered perturbation levels
    │
    ├── prepare-fidelity-perturbations: Gaussian-noise ladder + provenance
    └── analyze-metric-ordering: Kendall tau-b, ordered pairs, ties
```

## Core boundary

The core owns strict manifest parsing, WAV decoding (PCM and IEEE float), plugin discovery and
isolation, finding severity, grouping, exit policy, and report schemas. It does
not own model serving, checkpoint downloads, large embedding models, or an
overall music-quality score. Those belong in optional adapters and plugins.

The human-listening path is a parallel subsystem rather than a metric plugin.
Its public study package and private organizer key are separate artifacts so
the page cannot decode system identities. The response collector can validate
answers using public trial IDs and criteria, but only offline analysis receives
the organizer key.

## Evidence and policy

A metric reports evidence. An expectation turns selected evidence into a gate.
Finding a one-second silent interval is a warning by default because silence
can be intentional. A manifest containing `"max_dropout_seconds": 0.5` makes
that observation a failure for that specific benchmark contract.

This division prevents a hidden global threshold from declaring ambient music
invalid while still allowing a serving integration to guard against known
dropout regressions.

## Metric namespaces

A plugin returns a flat dictionary; the evaluator stores it under the plugin's
`name`, so every plugin owns one namespace in `result.metrics`:

```json
{
  "metrics": {
    "integrity": {"duration_seconds": 16.0},
    "pairwise": {"reference_pcm_exact": true},
    "clap_alignment": {"track": {"positive_similarity": 0.41, "margin": 0.23}}
  }
}
```

Two plugins cannot overwrite each other's fields, and findings carry their
plugin source.

## Failure isolation

A candidate or declared reference that cannot be decoded is a core failure. A
metric-plugin exception becomes an attributed `plugin_error` result, allowing a
large experiment to finish while keeping the process exit code nonzero.

## Versioning

The JSON report contains `schema_version`. Backward-incompatible report changes
must increment it. Manifest parsing rejects unknown fields so misspelled policy
keys cannot silently disable a gate.
