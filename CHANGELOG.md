# Changelog

## 0.8.0 (2026-09-06)

Correctness fixes that change reported numbers, so this is a minor bump.

Fixed

- Dropouts are now located on 10 ms frames and refined to the exact sample.
  Before, a silent region was only counted in whole 0.25 s windows aligned to
  the file start, so a half-second gap starting at 0.10 s reported 0.0 s and
  passed its gate while the same gap at 0.25 s failed it.
- `report.json` refuses non-finite numbers; the pairwise level delta for an
  empty candidate or reference is `null` instead of a bare `NaN` token.
- The temporal-consistency spectral profile averages 64 log-spaced bands with
  an 80 dB range instead of sampling single FFT bins, so two windows of the
  same texture now look alike. A silent window has no spectral direction and
  its similarities are `null` rather than counted as maximum drift.
- Bradley-Terry ranks are dense (tied strengths share a rank) and each ranking
  row carries `separated`, true when the maximum-likelihood estimate does not
  exist because some system never lost or never won; the report marks those
  rows.
- Listening-study files are written and read as UTF-8 regardless of locale;
  the responses directory is hidden by resolved path; passing `study.json` as
  the organizer key is rejected with a hint; a `null` note becomes an empty
  string and a non-string `submitted_at` is rejected.
- A `null` `duration_tolerance_seconds` is rejected at manifest load instead
  of surfacing as a plugin error; an explicit empty plugin list runs no
  plugins rather than the defaults.
- Noise sigmas such as 0.001 and 0.002 no longer collapse to the same file
  name; slugs use the shortest round-trip decimal and collisions are refused.
  Default sigma labels change spelling (`0.20` becomes `0.2`).
- Temporal consistency and production quality floor levels at -120 dB like
  the integrity metric, so digital silence reads the same everywhere.
- Report cells escape `|`, so a system named `cand|v2` no longer breaks the
  distribution table.

Added

- `init-listening-study --criteria-preset` with criteria sets from
  published protocols: `musicprefs` (fidelity, musicality), `songeval` (five
  aesthetic dimensions) and `songeval-instrumental`; the study page shows a
  one-line definition under each criterion and `study.json` records them.
- `music-eval --version`.

Changed

- WAV files are read by a small RIFF parser instead of the `wave` module:
  PCM 8 to 32 bit and IEEE float 32/64, plain or `WAVE_FORMAT_EXTENSIBLE`,
  identically on Python 3.10 through 3.13. Integrity reports `sample_format`.
- `results[].audio` and `results[].reference` are the paths as written in
  the manifest, not resolved absolute paths, so reports are portable.
- The LRA note cites what EBU Tech 3342 says (very short programmes can give
  misleadingly high values) and states the 60 s cut as this project's own.
- Bradley-Terry fitting is vectorized; the 1000-sample bootstrap on a
  10-rater study takes about 2 s instead of 36 s.
- New `docs/metrics.md` and `docs/suites.md`; the README is a summary.
- `examples/minimax-music3`: the offline metrics on five real generated
  takes, with the committed report.

## 0.7.0 (2026-09-06)

First public version: integrity and pairwise metrics, temporal consistency,
production quality, Audiobox Aesthetics, CLAP and MuQ-MuLan alignment,
MuseCPEval adapter, blind listening studies with Bradley-Terry analysis,
corpus distribution comparison with KAD, metric meta-evaluation, benchmark
suites and long-form ladders.
