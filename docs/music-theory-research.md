# Music-theory evidence from rendered audio

## Decision

Do not ship a single opaque “music theory score”. Rendered music permits a
mixture of music-information-retrieval estimates with very different failure
modes. The implementation should expose three independently selectable layers:

1. `rhythm_tonality`: a lightweight, deterministic baseline over the mixture;
2. `beat_structure`: optional learned beat, downbeat, and section inference;
3. `melodic_transcription`: optional source separation and note transcription.

A future `music_theory` preset may select those plugins, but it must not merge
their outputs into one number. None of these layers decides whether a song is
good. Non-diatonic harmony, rubato, free meter, drones, and intentional
repetition are valid music rather than defects.

## Why the problem is difficult

A waveform does not identify which source produced a frequency, which pulse is
the intended metrical level, or whether a repeated section is a chorus rather
than copied failure. Half/double-tempo ambiguity, relative major/minor keys,
detuning, percussion leakage, dense mixes, and source-separation artifacts are
normal operating conditions.

This is visible in the upstream tools themselves:

- Spotify Basic Pitch supports polyphonic, instrument-agnostic transcription,
  but its documentation says it works best on one instrument at a time.
- Essentia labels its HPCP chord detector experimental and prone to errors.
- Beat This! warns that evaluation on its training datasets can be unfairly
  optimistic and documents which checkpoints are suitable for held-out tests.

Consequently every inferred section needs provenance, an applicability state,
and native confidence or an explicitly labeled uncalibrated strength. An
estimate without enough evidence should be `insufficient_evidence`, not zero.

## Proposed plugin layers

### 1. `rhythm_tonality` — portable CPU baseline

Optional dependencies: `librosa`, `scipy`, and `soxr`. No model checkpoint.

Rhythm evidence:

- onset envelope and onset density;
- global tempo plus explicit half- and double-tempo alternatives;
- beat locations and inter-beat-interval coefficient of variation;
- local tempogram median, spread, and drift;
- beat-synchronous onset alignment;
- `insufficient_evidence` for silence, low-onset material, or too few beats.

Tonal evidence:

- tuning offset in cents;
- harmonic/percussive separation before tonal analysis;
- CQT chroma and beat-synchronous chroma;
- global key candidates from declared key profiles, including top-two margin;
- local key-profile trajectory and tonal-centroid drift;
- chroma entropy and pitch-class occupancy.

The key-profile result is a candidate ranking, not ground truth. A single-note
signal can identify a pitch class but cannot establish key or mode, so the
plugin must withhold a key when support or top-two margin is insufficient.

This layer should not attempt chord labels or melody notes. A hand-written
template matcher can be useful for debugging synthetic fixtures, but it is not
strong enough to describe arbitrary mastered music as a product feature.

### 2. `beat_structure` — learned full-mix analysis

Recommended backends:

- Beat This! `final0` for beat and downbeat timestamps. The package is
  pip-installable, can fall back to CPU, and its code and published weights are
  MIT licensed.
- All-In-One Music Structure Analyzer for tempo, beats, downbeats, functional
  boundaries, and labels such as intro, verse, chorus, bridge, and outro. Its
  code is MIT licensed.

Keep both checkpoint identities in the report. Do not silently substitute the
lightweight baseline for a missing learned backend. For short clips, report
structure as `not_applicable`; functional form needs enough musical context.

Derived evidence may include section count, duration distribution, boundary
confidence, repeated-label recurrence, and whether the ending is covered by a
section. “Too many” sections or a missing chorus must not become generic
failures because valid musical forms differ by genre.

### 3. `melodic_transcription` — expensive and source-aware

Recommended pipeline:

1. separate vocals, drums, bass, and other stems with Demucs;
2. run Basic Pitch on a selected tonal stem or a user-supplied isolated stem;
3. retain every note event with pitch, start, end, confidence, and pitch bend;
4. derive pitch range, voiced duration, interval distribution, phrase contour,
   note density, and repetition evidence;
5. compare with an annotation only when one is supplied.

Basic Pitch is Apache-2.0 and Demucs code is MIT. The report still needs exact
model/checkpoint provenance and must state that separated stems are estimates.
Running a polyphonic transcriber directly on the complete mix should be allowed
only as an explicitly degraded mode.

Do not add automatic “out-of-key note ratio” as a quality gate. Chromatic notes,
borrowed chords, blue notes, ornaments, and key changes would all create false
failures. It may be diagnostic only after a key hypothesis has adequate support.

### Harmony

Harmony is the least suitable component for the first release. A later
`harmony` backend can expose chord intervals and strengths from Essentia,
Chordino, or a separately qualified learned recognizer. It must preserve `N`
(no chord), support at least major/minor and common seventh reductions, and
publish both the original vocabulary and the reduction used for comparison.

No harmony backend should be promoted until it is evaluated with weighted chord
symbol recall on held-out, licensed audio. The upstream Essentia warning makes
unit tests on synthesized triads insufficient qualification.

## Result contract

Each analysis section should follow the same shape:

```json
{
  "status": "analyzed | insufficient_evidence | not_applicable | unavailable",
  "backend": "beat_this",
  "checkpoint": "final0",
  "version": "...",
  "preprocessing": {"sample_rate_hz": 22050, "channels": "mono"},
  "confidence_kind": "native | calibrated | heuristic | unavailable",
  "estimates": {},
  "events": [],
  "limitations": []
}
```

Example top-level namespace:

```json
{
  "rhythm": {
    "status": "analyzed",
    "tempo_bpm": 117.45,
    "tempo_alternatives_bpm": [58.73, 234.91],
    "beat_times_seconds": [1.02, 1.51, 2.02],
    "beat_interval_cv": 0.03
  },
  "tonality": {
    "status": "analyzed",
    "key_candidates": [
      {"key": "C major", "strength": 0.81},
      {"key": "A minor", "strength": 0.77}
    ],
    "top_two_margin": 0.04,
    "key": null,
    "reason": "top candidates are ambiguous"
  }
}
```

Raw time-series can become large. JSON should retain events and compact local
summaries; optional NumPy artifacts may hold full chroma, tempogram, and pitch
matrices with a hash recorded in JSON.

## Targets and reference annotations

Free-form prompt text is not a music-theory annotation. Add an optional,
strictly parsed manifest section rather than guessing exact targets from prose:

```json
{
  "theory": {
    "expected_tempo_bpm": 120,
    "tempo_tolerance_bpm": 5,
    "expected_meter": "4/4",
    "expected_key": "C major",
    "annotations": "annotations/song.jams"
  }
}
```

JAMS is preferable to inventing event formats. Relevant namespaces include
beats, key/mode, chords, pitch contours, and structural segments. A reference
annotation enables established `mir_eval` measures:

- beat F-measure and continuity measures;
- key weighted score;
- chord weighted accuracy under declared vocabulary reductions;
- melody voicing recall/false alarm, raw pitch/chroma, and overall accuracy;
- structure boundary and segment-label measures.

Only explicit target tolerances can create pass/fail findings. Unreferenced
estimates remain evidence.

## Validation strategy

### Unit and metamorphic tests

- exact click tracks at several tempi, including tempo changes and silence;
- synthesized major/minor chords, inversions, transposition, and detuning;
- monophonic melodies with known note events, rests, vibrato, and glissando;
- AABA/verse-chorus synthetic sections with controlled boundaries;
- invariance checks for gain, stereo pan, sample rate, leading silence, and
  harmless encoding changes;
- mutation checks proving that swapped pitch classes, deleted beats, shifted
  boundaries, and shortened notes are detected.

Synthetic tests validate implementation and invariants, not real-world accuracy.
For example, a 120 BPM synthetic click track produced approximately 117.45 BPM
with `librosa`'s default frame resolution in a local spike, so exact equality is
an invalid test oracle.

### Corpus qualification

Use held-out annotated corpora for each task and report dataset revision,
license, exclusions, and confidence intervals. Do not vendor copyrighted audio
into this repository. Benchmark runners should accept user-provided corpus
roots. Guard against training/test overlap; Beat This! explicitly documents
this risk for its checkpoints.

Qualification should compare at least one portable and one learned backend,
stratified by genre, vocals, instrumentation density, duration, and production
style. Report coverage and abstention rates alongside accuracy: a system that
withholds every difficult case is not useful.

### Human validation

Add a listening-study question only for claims that remain perceptual, such as
“more convincing musicality and structure.” Measure whether automatic evidence
tracks pairwise preferences, but do not train a universal aggregate score from
the same study used for final reporting.

## Implementation order

1. Add the strict `theory` manifest schema and JAMS sidecar boundary.
2. Implement `rhythm_tonality` with dependency injection and synthetic tests.
3. Add report panels for estimates, confidence, abstentions, and local traces.
4. Add `mir_eval` reference scoring and metamorphic tests.
5. Integrate Beat This! and qualify beat/downbeat behavior on held-out audio.
6. Integrate All-In-One for functional structure.
7. Add Demucs + Basic Pitch as a separate expensive plugin.
8. Evaluate a harmony backend before deciding whether it is shippable.

For SGLang-Omni, land this later as a separate optional benchmark integration,
not as a runtime dependency or required pull-request check.

## Primary references

- [librosa beat tracking](https://librosa.org/doc/0.10.2/generated/librosa.beat.beat_track.html),
  [CQT chroma](https://librosa.org/doc/0.10.2/generated/librosa.feature.chroma_cqt.html),
  and [pYIN](https://librosa.org/doc/0.10.2/generated/librosa.pyin.html)
- [Essentia RhythmExtractor2013](https://essentia.upf.edu/reference/std_RhythmExtractor2013.html),
  [KeyExtractor](https://essentia.upf.edu/reference/std_KeyExtractor.html), and
  [ChordsDetection](https://essentia.upf.edu/reference/std_ChordsDetection.html)
- [Beat This! official implementation](https://github.com/CPJKU/beat_this)
  and [ISMIR 2024 paper](https://arxiv.org/abs/2407.21658)
- [All-In-One Music Structure Analyzer](https://github.com/mir-aidj/all-in-one)
- [Spotify Basic Pitch](https://github.com/spotify/basic-pitch) and
  [ICASSP 2022 paper](https://arxiv.org/abs/2203.09893)
- [Demucs v4](https://github.com/facebookresearch/demucs)
- [JAMS](https://github.com/marl/jams) and
  [`mir_eval` task metrics](https://mir-eval.readthedocs.io/latest/)
