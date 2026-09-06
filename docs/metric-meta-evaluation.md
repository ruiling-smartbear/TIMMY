# Metric ordering meta-evaluation

An automatic evaluator should notice the failure it claims to measure. TIMMY
implements the ordered-degradation test used in *Aligning Text-to-Music
Evaluation with Human Preferences*: construct conditions with increasing
degradation, score each level, and measure whether the metric's ordering agrees
with the known order using Kendall's tau.

This command analyzes precomputed scores so it can test built-in TIMMY plugins,
external evaluators, human ratings, or future metrics without coupling their
dependencies to the core package.

TIMMY can generate deterministic local fixtures for the paper's reported
fidelity noise strengths:

```bash
music-eval prepare-fidelity-perturbations clean/*.wav \
  --output fidelity-audit --seed 20260906
music-eval evaluate fidelity-audit/manifest.jsonl \
  --metrics integrity,production_quality,audiobox_aesthetics \
  --output fidelity-scores
```

`fidelity-audit/provenance.json` records every source, child seed, sigma, and
the sample ratio that exceeded the PCM range before encoding. The outputs are
16-bit PCM because TIMMY's dependency-free core writes WAV directly. This
introduces quantization and possible clipping, both of which must remain part
of the protocol rather than being hidden.

```json
{"id":"noise-0-a","condition":"white-noise","metric":"audiobox-pq","level":0.0,"score":7.2,"expected_direction":"decrease"}
{"id":"noise-1-a","condition":"white-noise","metric":"audiobox-pq","level":0.02,"score":6.8,"expected_direction":"decrease"}
{"id":"noise-2-a","condition":"white-noise","metric":"audiobox-pq","level":0.04,"score":5.9,"expected_direction":"decrease"}
```

```bash
music-eval analyze-metric-ordering ordering.jsonl --output ordering-report
```

Replicates may share the same condition, metric, and level. TIMMY averages
replicates within a level and retains their sample count, population standard
deviation, minimum, and maximum. It reports Kendall tau-b, strict ordered-pair
accuracy, and score ties. Ties are not counted as successes.

`level` must encode increasing degradation strength. Set
`expected_direction=increase` for distances such as KAD and `decrease` for
quality estimates such as Audiobox PQ. The sign is normalized before ordering
statistics are calculated.

## Protocols

The paper's four synthetic protocols, each with K = 11 levels (Table 4 in its
Appendix B):

- fidelity: Gaussian noise with standard deviation in
  {0, 0.02, 0.04, ..., 0.18, 0.2};
- musicality: perturb a subset of Lakh MIDI notes by [-6, 6] semitones in
  pitch and [-0.2, 0.2] s in onset and offset, with the note perturbation
  probability in {0, 0.05, 0.1, ..., 0.45, 0.5};
- context: generate 30-second MusicGen-Small clips in blocks of
  {1, 2, ..., 10, 15} s, each block seeing only the previous block as context;
- diversity: generate 5,000 clips from prompt pools of
  {1, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 2500} unique prompts.

`prepare-fidelity-perturbations` reproduces the first ladder exactly (its
default sigmas are that list; `--sigmas` sets another increasing list); the
other three need MIDI rendering, a generator and a prompt corpus and are not
bundled.

Reproducing those headline results also requires the named corpora, generator,
embedding backbone, layer, pooling rule, reference distribution, and 5,000
clips per level used by the paper. A local smoke test with one audio file should
be labeled as such rather than presented as a reproduction.

TIMMY may add diagnostic perturbations—truncation, looping, clicks, bandwidth
loss, section shuffling, wrong instruments—but each is a separate condition,
not silently labeled as one of the paper's four protocols.

High tau proves sensitivity to that controlled perturbation family. It does not
prove agreement with listeners, robustness to metric gaming, or validity for a
different genre, culture, duration, encoder, or production style.

## References

- [Huang et al., *Aligning Text-to-Music Evaluation with Human Preferences*](https://arxiv.org/abs/2503.16669)
- [Official MAD implementation](https://github.com/i-need-sleep/mad)
- [MusicPrefs dataset](https://huggingface.co/datasets/i-need-sleep/musicprefs)
