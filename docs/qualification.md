# Qualification record

This page distinguishes deterministic tests, dependency smoke tests, and
research-scale validation. A green unit test is not automatically a qualified
perceptual metric.

## 2026-09-06 local qualification

| Layer | Evidence | Result | Remaining limitation |
| --- | --- | --- | --- |
| base Python package | full pytest suite, strict mypy, ruff, diff whitespace check | pass | macOS/Python 3.13 local environment only |
| documentation topology | every relative Markdown link resolves | pass | does not test rendered layout in every client |
| external citations | HTTP audit of every unique URL in README/docs | pass after replacing retired librosa URLs and removing a dead LeVo repository link | servers may later move or block automated requests |
| KAD implementation | compared TIMMY NumPy result with official `kadtk` commit `8bda71d87f55cc7f8a8e38d3d4cfbae902ff78b1` on the same synthetic embeddings | `-20.9491786502` vs `-20.9491844177`; absolute delta `5.77e-06` | validates statistic implementation, not a named audio encoder campaign |
| MuseCPEval installation | fresh virtual environment installed `musecpeval[structure]==0.3.0` | pass | large dependency stack; dedicated environment recommended |
| MuseCPEval adapter | real `music-eval evaluate` identity-pair run against MuseCPEval 0.3.0 | all five facet groups returned; adapter report passed; structural subprocess left no source-side artifacts after input isolation | synthetic pure tone is a wiring smoke test, not editing validity |
| MuseCPEval edge behavior | short pure tone produced empty-beat/segment warnings and timbre numerical warnings | warnings observed; output remained finite; TIMMY rejects non-finite values | real musical edits are still required for research qualification |
| learned checkpoints | dependency-injected unit tests for Audiobox, CLAP and MuQ-MuLan adapters | pass | pinned real-checkpoint dataset campaigns have not yet been run in this record |
| long-form experiment design | manifest invariant tests over matched prompt/seed duration ladders | pass | no generator outputs or human structure ratings have been collected |

## Test interpretation

- Unit fixtures prove parsing, aggregation, error propagation, invariants, and
  report shape.
- Cross-implementation tests prove that a statistic matches a selected upstream
  implementation within a declared numerical tolerance.
- A dependency smoke test proves installation and wiring, not scientific
  validity.
- Research qualification requires a versioned dataset, real checkpoints,
  sample-size analysis, stratified results, and human evidence appropriate to
  the claim.

## Required next campaigns

1. Song Describer CLAP versus MuQ-MuLan hard-negative alignment.
2. License-filtered FMA-Pop KAD/FAD/MAD comparison with sample-size curves.
3. MusicPrefs system-order correlation, contingent on explicit usage terms.
4. MuseCPBench or separately licensed edit pairs across all five facets plus
   target-region success and seam listening.
5. Fill the implemented matched 30-second, two-, four-, and eight-minute
   manifests, then estimate onset-of-degradation curves and run blinded
   structure ratings.
