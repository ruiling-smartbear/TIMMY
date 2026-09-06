# Metrics

Every metric writes into its own namespace under `results[].metrics` and never
into another plugin's. Metrics report evidence; the only pass/fail decisions
come from the manifest's `expectations` (see [manifest.md](manifest.md)) and
from decode or plugin errors. `music-eval list-metrics` prints the names the
installed package knows about, including third-party plugins registered
through the `music_eval.metrics` entry-point group ([plugin-api.md](plugin-api.md)).

| Metric | Measures | Extra dependency | Findings |
|---|---|---|---|
| `integrity` | format, levels, DC offset, clipping, silence, dropouts, stereo correlation | none | warnings, plus every expectation gate |
| `pairwise` | deltas and exact/aligned agreement against a reference | none | reference gates |
| `temporal_consistency` | windowed spectral drift, repetition, ending | none | none |
| `production_quality` | loudness, true peak, dynamics, bandwidth, clicks | `[production]` | none |
| `audiobox_aesthetics` | learned CE/CU/PC/PQ per 10 s window | `[audiobox]` | none |
| `clap_alignment` | prompt similarity with hard-negative margin and rank | `[clap]` | none; a missing prompt is a plugin error |
| `muq_mulan_alignment` | same, music-specific model, non-commercial weights | `[muq]` | none; a missing prompt is a plugin error |
| `musecp_preservation` | edit-context preservation in five facets | `[editing]` | warning when the case has no reference |

## `integrity`

Runs on every case. Reports `sample_rate`, `channels`, `frames`,
`duration_seconds`, `sample_width_bytes`, `sample_format` (`pcm` or `float`),
`rms_dbfs`, `peak_dbfs`, `dc_offset` per channel, `clipped_sample_ratio`,
`silent_window_ratio`, `longest_dropout_seconds`, `channel_rms_dbfs`, and
`stereo_correlation` for two-channel audio. Every dropout is listed under
`results[].dropouts` with sample-accurate `start_seconds` and `end_seconds`.

Without expectations, clipping, DC offset, silence and dropouts produce
warnings; `--fail-on-warning` turns them into failures. With expectations,
each declared limit becomes a failure finding with the measured value in the
message.

The signal thresholds are `evaluate` options: `--silence-dbfs` (-50),
`--window-seconds` (0.25, used only for `silent_window_ratio`),
`--minimum-dropout-seconds` (0.5) and `--clip-threshold` (0.999).

## `pairwise`

Active when a case declares `reference`. Reports
`reference_duration_delta_seconds`, `reference_rms_delta_db` (null when either
side is empty), `reference_pcm_exact`, and, only when sample rate and shape
match, `reference_waveform_correlation` and `reference_nrmse`. These aligned
measures answer one question: did this serving path produce the same audio as
the reference run? They say nothing about two independently sampled pieces of
music, so do not use them as a general similarity score.

## `temporal_consistency`

CPU only. Cuts the mono mix into 2-second windows with a 1-second hop and
reports per window `rms_dbfs`, `spectral_centroid_hz`, `spectral_bandwidth_hz`,
`spectral_rolloff_95_hz`, `spectral_flatness` and `zero_crossing_rate`. Each
window also gets a spectral profile: mean power in 64 log-spaced bands from
20 Hz to 20 kHz, log-compressed with an 80 dB range below the loudest band,
mean-removed and unit-normalized. Cosine similarity between profiles drives:

- `transitions` and `transition_summary`: adjacent-window RMS and centroid
  jumps and spectral similarity;
- `first_to_last`: drift between the opening and closing windows;
- `nonlocal_repetition`: similarity of window pairs at least 8 s apart,
  the share of windows with a near-duplicate (similarity >= 0.995), and the
  pair count;
- `ending`: level of the last 250 ms, its change against the previous 250 ms,
  and the absolute value of the final sample.

A silent window has no spectral direction; its similarities are null and are
left out of the summaries. Near-duplicate ratios are never findings: a
repeated chorus, a drone and a collapsed generation can all score high.

## `production_quality`

Install `music-eval[production]` (`pyloudnorm`, `scipy`). Reports
`loudness.integrated_lufs` (ITU-R BS.1770-4) and `loudness.loudness_range_lu`
(EBU Tech 3342) through pyloudnorm, `sample_peak_dbfs`, a 4x polyphase
`estimated_true_peak_dbtp` that is a diagnostic, not a certified meter,
`crest_factor_db`, 400 ms `block_rms_dbfs` percentiles, per-channel true peaks,
`spectral` roll-off and energy ratios above 15, 18 and 20 kHz, and
`impulsive_discontinuities` (sample-difference outliers with their timestamps).

`loudness_range_stable` is false below 60 s. Tech 3342 warns that very short
programmes can produce misleadingly high LRA without naming a threshold; the
60 s cut is this project's conservative choice. The plugin never gates: a
click is a fault in a ballad and a feature in lo-fi, and loudness targets
belong to the delivery platform, not the evaluator.

## `audiobox_aesthetics`

Install `music-eval[audiobox]`. Uses Meta's Audiobox Aesthetics predictor
(checkpoint recorded under `checkpoint`) on 16 kHz mono, non-overlapping
10-second windows, and reports duration-weighted `track_scores` for CE, CU, PC
and PQ, an `axis_summary` with the p10, minimum and worst-window position, and
every window. The package is imported at evaluation time, never at import.

## `clap_alignment` and `muq_mulan_alignment`

Install `music-eval[clap]` or `music-eval[muq]`. Both read the case's `prompt`
and optional `negative_prompts`, resample to the model's rate (48 kHz for
`laion/clap-htsat-unfused`, 24 kHz for `OpenMuQ/MuQ-MuLan-large`), score
10-second windows and report `track.positive_similarity`,
`track.negative_similarities`, `track.best_negative_similarity`,
`track.margin` (positive minus best negative), `track.positive_rank`, a
`summary` over windows, and each window. Raw similarity has no universal
threshold; margin and rank against hard negatives are the comparisons that
carry information. The MuQ code is MIT-licensed, but the published MuQ-MuLan
weights are CC BY-NC 4.0, which the report records.

## `musecp_preservation`

Install `music-eval[editing]` (`musecpeval[structure]==0.3.0`). For an edit
case, `reference` is the original and `audio` the edited track; the plugin
returns MuseCPEval's harmony/tonality, rhythm/meter, structural form,
melody/motif and timbre/texture facets as separate groups with the package
version. Preservation of context is not evidence that the requested edit
succeeded; that needs region-level and listening evidence.
