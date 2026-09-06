# MiniMax Music 3 reference outputs

Five real generated tracks run through the offline metrics. The audio comes from
the SGLang-Omni cookbook for MiniMax Music 3 (`docs/cookbook/minimax_music3.md`
in `sgl-project/sglang-omni`), which publishes one 30-second reference output
per prompt. `manifest.jsonl` carries the cookbook captions as prompts, the seeds
1 to 5, genre and vocal labels, and the expectation that every take is 30 s of
32 kHz stereo.

```bash
python examples/minimax-music3/fetch_audio.py
music-eval evaluate examples/minimax-music3/manifest.jsonl \
  --metrics integrity,temporal_consistency,production_quality \
  --output examples/minimax-music3/report
```

The committed `report/` is the output of that command (music-eval 0.8.0,
pyloudnorm 0.2.0, scipy 1.17.1). The exit code is 1 because one take fails a
gate.

## What the report says

| Take | Status | Duration | LUFS | LRA | Est. true peak | Clicks/s | Findings |
|---|---|---:|---:|---:|---:|---:|---|
| 00_lofi_hiphop | warning | 30.02 s | -13.5 | 7.7 | +0.37 dBTP | 6.9 | clipping_detected |
| 01_jpop_bright | fail | 26.23 s | -16.0 | 9.9 | -1.62 dBTP | 0.2 | duration_mismatch |
| 02_synthwave_moody | pass | 30.02 s | -18.9 | 6.8 | -1.13 dBTP | 0.7 | |
| 03_acoustic_folk | pass | 30.02 s | -14.3 | 9.4 | -0.23 dBTP | 0.2 | |
| 04_orchestral_epic | warning | 30.02 s | -17.6 | 16.4 | +0.01 dBTP | 0.1 | clipping_detected |

The J-pop take is the only failure: 26.2 s against a 30 s expectation. The
take simply ends early, which says nothing about how it sounds, but a serving
pipeline that promises fixed-length output needs to know.

Two takes touch full scale. The lo-fi track peaks at +0.37 dBTP and 0.011 % of
its samples sit at the rails; that is a mastering problem in any genre. The
orchestral take clips a single sample while its loudness
range of 16.4 LU is exactly what the caption asked for, "dynamic build from
restrained to triumphant"; the same LRA on the lo-fi track would be a warning
sign. Its tail is 11 dB louder than the 250 ms before it and the last sample is
still at 0.03, so the piece is cut at the 30 s limit rather than ended.

The click detector reports 6.9 candidates per second on the lo-fi track and
under one per second everywhere else. The caption asked for "soft vinyl
crackle", so those impulses are the requested texture, not a decoder fault.
This is why `production_quality` never gates: the same number is a defect in
the folk ballad and a feature in lo-fi.

Every take carries a small DC offset, between -0.0002 and +0.0012, positive on
nine of the ten channels. It is 60 dB down and inaudible, but it is systematic
across seeds and genres, which points at the decoder rather than the material.

Loudness lands between -13.5 and -18.9 LUFS. Streaming platforms normalize to
around -14 LUFS, so the lo-fi and folk takes would be turned down and the
synthwave take turned up; none of that is a quality judgement, it is context
for the true-peak numbers above. Note that `loudness_range_stable` is false
for all five: EBU Tech 3342 warns that very short programmes can give
misleadingly high LRA, and this report marks anything under 60 s as
indicative only.

Temporal consistency finds no near-duplicate windows in any take, with the
highest non-local similarity (0.968) in the loop-based lo-fi track, as expected.
First-to-last spectral similarity is lowest for J-pop (0.33), the take with the
most arrangement change over its length.

## What this example does not show

There is no reference audio, so the serving-fidelity comparison (`pairwise`)
is empty here; that layer needs a second run of the same seed. There is no
prompt-alignment score because the learned plugins need model checkpoints that
this example does not download. And five takes are a smoke test of the
reporting, not an evaluation of the model.
