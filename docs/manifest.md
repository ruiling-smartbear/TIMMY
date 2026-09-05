# Manifest contract

The manifest is newline-delimited JSON. Every nonempty line is one independent
evaluation case.

## Entry fields

| Field | Required | Meaning |
|---|---:|---|
| `id` | yes | Unique, nonempty case identifier |
| `audio` | yes | Candidate PCM WAV path, relative to the manifest |
| `prompt` | no | Generation prompt |
| `negative_prompts` | no | Hard negatives for future alignment plugins |
| `seed` | no | Integer generation seed |
| `reference` | no | Aligned reference WAV path |
| `labels` | no | Arbitrary dimensions mapped to one or more values |
| `expectations` | no | Explicit failure policy |

Label dimensions and values are case-folded and whitespace-trimmed. Values may
be one string or a list. Duplicate values are removed.

## Expectations

| Field | Range | Gate |
|---|---:|---|
| `duration_seconds` | ≥ 0 | Expected candidate duration |
| `duration_tolerance_seconds` | ≥ 0 | Allowed duration error; default 0.25 |
| `sample_rate` | positive integer | Exact sample rate |
| `channels` | positive integer | Exact channel count |
| `max_clipped_ratio` | [0, 1] | Maximum near-full-scale sample ratio |
| `max_silent_window_ratio` | [0, 1] | Maximum fraction of silent windows |
| `max_dropout_seconds` | ≥ 0 | Maximum contiguous detected silence |
| `max_reference_duration_delta_seconds` | ≥ 0 | Reference duration difference |
| `min_reference_waveform_correlation` | [-1, 1] | Aligned correlation |
| `max_reference_nrmse` | ≥ 0 | Aligned waveform NRMSE |

Waveform correlation and NRMSE require matching sample rate, frame count,
channel count, and non-constant audio. Requesting these gates with incompatible
files fails explicitly rather than skipping the comparison.

Unknown fields are rejected. A typo such as `expectation` instead of
`expectations` must not create a false green.
