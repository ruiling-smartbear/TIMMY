# Music-AI evaluation landscape and product ideas

## Executive conclusion

The field does not have one accepted definition of “good generated music”. The
useful research direction is therefore not a larger opaque quality model. It is
an **evidence system** that tests a concrete claim under a concrete protocol.

The strongest opportunity for `music-eval` is to connect three communities
that currently speak different languages:

1. product teams make claims such as “eight-minute coherence”, “better genre
   mashups”, “faithful covers”, and “local edits”;
2. MIR research provides task-specific measurements for rhythm, melody,
   harmony, structure, separation, retrieval, and transcription;
3. HCI and computational-creativity research evaluates control, agency,
   exploration, workflow, and human outcomes rather than only the final WAV.

The framework should preserve those layers separately. A generated track can
be technically clean but musically dull, prompt-aligned but difficult to edit,
or aesthetically strong while copying a training example. No scalar average
can explain those cases.

## What Suno's public material reveals

Suno's public posts are product announcements, not reproducible evaluation
reports. They describe capabilities but do not disclose evaluation sets,
sample counts, baselines, annotator protocols, confidence intervals, or failure
rates. That is not evidence that the claims are false; it means an independent
evaluator must first translate each claim into a falsifiable protocol.

The v4 announcement claims cleaner audio, sharper lyrics, more dynamic song
structures, remastering, covers, and reusable vocal/style “Personas”. The v4.5
announcement adds closer genre following and genre mashups, richer voices,
better prompt interpretation, covers that retain more melodic detail, faster
generation, songs up to eight minutes, fuller mixes, and reduced degradation
and shimmer. Suno Studio adds stem generation, multitrack editing, BPM and
pitch control, and audio/MIDI export. v5.5 adds verified voice identity,
catalog-conditioned custom models, and learned personal taste. Suno's safety
page additionally says generated downloads carry C2PA Content Credentials and
describes similarity checks for uploaded content. Provenance and originality
therefore belong in the evaluation contract, not only in platform policy.

Those claims imply a much better benchmark than “generate 100 songs and run
one aesthetic scorer”:

| Product claim | Testable protocol | Evidence to preserve |
| --- | --- | --- |
| cleaner audio, less shimmer | inject and detect controlled artifacts; compare generations blindly | artifact curves, worst windows, listener preference |
| sharper lyrics | separate vocals, transcribe, and compare phonemes or lyrics | PER/CER, intelligibility ratings, separation provenance |
| dynamic song structure | long-window section and novelty analysis | boundaries, repeated sections, drift, human form rating |
| closer genre following | positive prompt plus nearby hard-negative genres | positive rank, margins, confusion by genre and culture |
| coherent genre mashups | require evidence for both genres, not merely either one | per-genre scores, balance, listener “fusion vs collage” rating |
| faithful covers | preserve content while intentionally changing style | melody/lyrics preservation and style-change axes |
| reusable persona or voice | same identity across different songs without copying content | speaker similarity, musical diversity, privacy/memorization checks |
| eight-minute coherence | score overlapping windows and section relations through time | degradation-onset curve, not only a track average |
| local section replacement | align before/after audio around a declared edit interval | inside-target success, outside preservation, boundary seams |
| stem generation/export | evaluate every stem and their reconstruction together | bleed, additivity, timing, phase, stem-specific adherence |
| BPM/pitch controls | paired counterfactual generations changing one control | target response, monotonicity, unintended changes |
| personalized taste | predict held-out choices for the same user | calibration, cold-start curve, diversity, cross-user leakage |
| responsible provenance | remove, transcode, and rewrap generated files | C2PA presence, survival, issuer identity, tamper result |

This “claim-to-evidence” translation can itself become a first-class feature:
every benchmark stores a claim, protocol, dataset revision, metrics, thresholds,
known blind spots, and human evidence in a machine-readable claim card.

## What the university work adds

### Carnegie Mellon: evaluate the act of creating

CMU's Generative Creativity Lab frames its goal as augmenting human creativity
and productivity. Two projects show why output-only evaluation is incomplete:

- An experiment with 140 musically trained participants found the AI-assisted
  group was slower, used fewer notes, and produced melodies that a separate
  group judged less creative. The study measured creativity, enjoyment, and
  musicality, not just technical correctness.
- AMUSE accepts musical and non-musical inspirations, permits editing and
  reuse, and was studied in terms of perceived agency and creativity.
- VibeClip starts from a video's image and transcript, helps non-experts express
  a desired “vibe”, and presents several candidates in the video context.

The implementation lesson is to add a **creative-session record** alongside
the audio manifest:

```json
{
  "session_id": "session-17",
  "task": "score a 30-second video",
  "events": [
    {"t": 0.0, "type": "prompt", "text": "calm but expectant"},
    {"t": 14.2, "type": "candidate_generated", "candidate_id": "c1"},
    {"t": 38.9, "type": "edit", "region_seconds": [12.0, 18.0]},
    {"t": 71.3, "type": "accepted", "candidate_id": "c3"}
  ]
}
```

From that record, report time to first usable result, time to acceptance,
candidate count, fraction reused, prompt-revision burden, abandoned branches,
exploration diversity, and user-rated agency. More generations are not
automatically better: they may indicate exploration or an inability to steer.

### USC: music quality changes through time and modality

USC's work on music and emotion relates musical features—including dynamics,
register, rhythm, harmony, timbre, contrast, and complexity—to subjective and
physiological response. Its genre work jointly models chord and lyric content,
which is a reminder that a genre label is not purely an audio-text embedding
problem.

This suggests two plugins:

1. `affect_trajectory`: compare time-varying arousal/valence estimates with a
   declared target curve and retain event-level changes such as crescendos,
   entrances, and contrast;
2. `multimodal_style`: preserve separate evidence from audio, lyrics, harmony,
   instrumentation, and optional video instead of treating a single CLAP
   similarity as genre ground truth.

### Stanford CCRMA and Music, Brain & Health: interaction and target curves

CCRMA presentations organize current generative-audio work around multitrack
generation, assistive creation tools, multimodal representation, and
text-queried source separation. Related work on musical machines emphasizes
human-AI ensembles and real-time neural audio processing.

Stanford's Music, Brain & Health Lab argues for protocols that are standardized
yet adaptable to the listener. Its proposed music trajectories explicitly use
tempo, spectral brightness, and harmonic tension to follow desired affect and
arousal through time.

That yields two additional benchmark families:

- `interactive_performance`: control-to-audio latency, jitter, beat alignment,
  response to gestures, continuity during control changes, cancellation, and
  recovery after overload;
- `target_curve`: deviation from an explicit time-varying target while also
  reporting personal fit, calibration, and within-listener repeatability.

A therapeutic claim would require clinical validation and must never be
inferred from an affect model alone. The target-curve machinery is useful more
broadly for film scoring, games, exercise, meditation, and adaptive playlists.

## What workshops say the field is becoming

The 2025 AIMC theme was “The Artist in the Loop”. Its call highlights search
and discovery, control and differentiation, and benefits beyond the musical
output. It accepts papers, systems, performances, and workshops because a live
instrument or creative practice cannot be evaluated like a static benchmark.

The NeurIPS 2025 AI for Music workshop similarly spans analysis, creation,
performance, production, retrieval, education, therapy, and ethical or legal
effects. Its program is unusually informative for a roadmap:

- rigorous creativity evaluation;
- multi-track information dynamics and rhythmic synchronization;
- low-latency controllable generation;
- interactive performance systems “in the wild”;
- expressive singing pitch control;
- gesture-driven human-in-the-loop generation;
- music plagiarism, fingerprinting, membership inference, unlearning, voice
  protection, watermarking, and geographic/cultural bias;
- soundtrack retrieval and video-to-music generation;
- source separation, multitrack transcription, and DAW integration.

The message is consistent: an evaluation tool should support **works and
workflows**, not only leaderboards.

## Plugin and experiment ideas

### 1. `claim_card` — turn marketing prose into a reproducible experiment

Store the claim, model revision, generation settings, prompt suite, expected
failure modes, metrics, human-study questions, and acceptance policy. Generate
an evidence table showing `supported`, `contradicted`, or
`insufficient_evidence`; never infer “supported” merely because a command ran.

This is small to implement and gives every later plugin a coherent purpose.

### 2. `counterfactual_control` — did one knob change only what it promised?

Generate matched pairs with the same seed and prompt while changing one
attribute: tempo 90→120 BPM, major→minor, acoustic→electric, no vocals→vocals,
or calm→tense. Report:

- target response and direction;
- monotonicity over three or more control levels;
- leakage into unrelated dimensions;
- repeatability over seeds;
- failure and refusal rate.

This is stronger than evaluating unrelated generations because it isolates a
causal intervention. It also directly tests Suno-style prompt helpers, sliders,
and BPM/pitch controls.

### 3. `edit_locality` — test generative editing rather than generation

Inputs are the original audio, edited audio, requested interval, and edit
instruction. Align the two tracks, then report:

- success inside the requested interval;
- preservation before and after it;
- spectral, loudness, phase, beat, and reverb-tail discontinuities at both
  boundaries;
- timing displacement and duration changes outside the edit;
- perceptual A/B ratings focused on seam visibility.

Metamorphic fixtures can insert known clean and broken joins. This is one of
the clearest unoccupied niches in existing open music evaluation stacks.

### 4. `cover_invariance` — two objectives that must not be averaged

A cover is successful only when it preserves the requested musical identity
and changes the requested style. Report separate axes for:

- melody/contour, lyrics, section order, and timing preservation;
- timbre, instrumentation, groove, genre, and production transformation.

Also detect the two degenerate solutions: copying the original and producing
an unrelated but stylistically plausible song.

### 5. `longform_curve` — find where a track begins to fail

Score overlapping windows at several scales and produce a hazard-like curve
for the onset of degradation. Track prompt alignment, aesthetic estimates,
artifacts, loudness, tempo, key candidates, instrumentation, vocal identity,
section recurrence, and duplicate/collapse evidence.

Compare 30-second, two-minute, four-minute, and eight-minute generations with
matched prompts. Report survival to a declared quality boundary as well as
mean scores. This directly tests a long-form claim without allowing a polished
opening to conceal a broken ending.

### 6. `stem_quality` — evaluate a generative workstation

Without isolated references, still test:

- reconstruction error between the mixture and summed stems;
- residual energy and stem additivity;
- pairwise stem dependence and likely bleed;
- beat/onset synchronization;
- prompt and instrument-label alignment per stem;
- silence, clipping, bandwidth, and artifacts per stem;
- MIDI/audio onset and pitch agreement when both are exported.

With references, add SI-SDR and established source-separation measures. A
single score should not combine separation fidelity and musical usefulness.

### 7. `creative_process` — was the person actually helped?

Build study templates for time to satisfactory output, number of revisions,
idea diversity, perceived control, authorship, effort, learning, and final
preference. Compare at least manual, one-shot AI, and editable AI conditions.

Use task-specific outcomes: a soundtrack study should rate fit with the video;
a songwriter study should retain the final composition and the path used to
reach it. Randomize conditions and keep judges of final outputs separate from
participants who created them where possible.

### 8. `memorization_novelty` — distinguish originality from surface variety

Combine nearest-neighbor audio embeddings, robust fingerprints, melody/motif
overlap, lyric overlap, and within-prompt diversity. Evaluate retrieval against
a licensed comparison corpus and include exact-match canaries where legally
and ethically appropriate.

The result is evidence of similarity, not a legal conclusion. Report which
representation produced every match. Add opt-out/unlearning regression cases
and membership-inference audits only when the benchmark has legitimate access
to the relevant training-status labels.

### 9. `affect_trajectory` and `video_fit`

Accept a target arousal/valence curve or timestamped video events. Compare the
generated track's local trajectory, musical boundaries, beats, and salient
events with that target. Human raters should judge narrative fit in context,
because feature alignment does not establish emotional success.

### 10. `metric_redteam` — evaluate the evaluators

Create a library of controlled degradations:

- clipping, bandwidth loss, shimmer, codec damage, clicks, and dropouts;
- truncated endings, loop collapse, repeated sections, shuffled sections;
- beat shifts, tempo drift, pitch shifts, detuning, and wrong meter;
- wrong instrument, swapped genre, removed vocals, incorrect lyrics;
- exact copying, near-copying, and diversity collapse.

Every metric declares which perturbations it should detect, ignore, or abstain
on. Run dose-response tests and correlate metric deltas with blinded human
preferences. This guards against adding impressive-looking models that do not
measure the claims users care about.

### 11. `provenance` — verify origin claims without judging the music

Inspect C2PA or other declared credentials, issuer identity, generation model,
timestamps, and asset hashes. Test whether credentials survive common export,
transcode, trim, and metadata-copy workflows, and whether tampering is reported
clearly. Keep “credential missing”, “credential invalid”, and “not generated by
this issuer” distinct; none proves that audio was human-made.

## A practical benchmark layout

Use independent tracks rather than a universal ranking:

| Track | Core question | Suggested evidence |
| --- | --- | --- |
| fidelity | is the file technically usable? | integrity, production, artifacts, worst windows |
| musical content | what rhythm, melody, harmony, and form is present? | MIR events, annotations, abstentions |
| semantic adherence | did it follow the request? | CLAP/MuQ-MuLan, hard negatives, human ratings |
| control | did a requested intervention work locally and specifically? | paired counterfactual and locality tests |
| long form | when and how does quality drift? | multiscale trajectories and degradation onset |
| multitrack | are stems separable, aligned, and useful? | reconstruction, bleed, onset/pitch agreement |
| creativity | is it varied and non-derivative? | diversity, motif/fingerprint evidence, human novelty |
| workflow | did it help a person reach a goal? | session logs, time, revisions, agency, task success |
| interaction | can it respond musically in real time? | latency, jitter, synchronization, recovery |
| responsibility | whose identity/data/style is involved? | consent, provenance, leakage, bias, opt-out tests |

Results should be stratified by genre, culture, language, vocals, duration,
instrument density, model, and generation mode. “World music” is not an
acceptable residual category; genre taxonomies and annotator populations must
be documented.

## Recommended implementation order

### P0: differentiating foundation

1. Add a claim-card schema and evidence-status report.
2. Add a session/event schema for human-in-the-loop experiments.
3. Implement `counterfactual_control` for pre-generated matched pairs.
4. Implement `edit_locality` with synthetic seam and preservation tests.
5. Extend temporal reporting into `longform_curve` with multiscale windows.
6. Add `metric_redteam` fixtures and dose-response contracts.

These mostly reuse the existing core and avoid committing to another large
learned model.

### P1: music-specific depth

1. Implement the proposed `rhythm_tonality` plugin.
2. Add reference-aware lyric PER and transcription provenance.
3. Add `stem_quality` and optional reference separation metrics.
4. Add cover invariance and video/event-fit protocols.
5. Qualify one named embedding for a reproducible FAD/MAD protocol.

### P2: research-grade extensions

1. Melody, chord, downbeat, and functional-structure backends.
2. Affect-trajectory calibration and personalized target curves.
3. Originality, fingerprint, opt-out, and membership-inference studies.
4. Interactive-performance instrumentation and DAW adapters.
5. Longitudinal human studies of agency, learning, and creative diversity.

## Research ideas with paper potential

1. **ClaimBench for generative music.** Collect public model claims and publish
   falsifiable, versioned protocols rather than another model leaderboard.
2. **Edit locality as a perceptual benchmark.** Establish automatic boundary
   measures plus a listening protocol for section replacement, stem inpainting,
   and continuation.
3. **Counterfactual controllability.** Measure target response, monotonicity,
   and collateral change under isolated musical controls.
4. **Long-form quality survival.** Replace a track average with the probability
   of remaining acceptable through time and compare failure modes by genre.
5. **Metric red teaming for music.** Test whether learned evaluators react
   correctly to controlled musical, semantic, and production perturbations.
6. **Output quality versus creative utility.** Combine blind final-output
   judgments with process logs to ask whether a system that scores higher
   actually helps creators work faster, explore more, or feel more agency.
7. **Fusion or collage?** Create a genre-mashup benchmark that measures the
   presence, balance, and interaction of two styles rather than single-label
   classification.
8. **Personalization without collapse.** Measure held-out user preference,
   diversity, novelty, and cross-user privacy together.

The strongest first publication is likely the combination of ClaimBench,
counterfactual control, edit locality, and metric red teaming. It is coherent,
does not require training a foundation model, and addresses a visible gap
between product announcements and current academic leaderboards.

## Primary sources

- [Suno v4](https://about.suno.com/blog/v4),
  [Suno v4.5](https://about.suno.com/blog/introducing-v4-5),
  [Suno v5.5](https://about.suno.com/blog/v5-5), and
  [Suno Studio](https://about.suno.com/blog/suno-studio), and
  [Suno safety and Content Credentials](https://about.suno.com/safety)
- [CMU Generative Creativity Lab](https://gclef-cmu.org/),
  [CMU creativity experiment](https://www.cmu.edu/news/stories/archives/2026/january/as-ai-generated-music-advances-humans-still-lead-in-creativity-cmu-research-finds),
  [AMUSE coverage](https://www.cs.cmu.edu/news/2025/ai-augments-creativity), and
  [VibeClip coverage](https://hcii.cmu.edu/news/designing-for-vibes-text-to-music)
- [USC SAIL music and emotion](https://viterbischool.usc.edu/news/2019/10/why-music-makes-us-feel-according-to-ai/)
  and [joint chord/lyric genre modeling](https://viterbischool.usc.edu/news/2019/08/ai-predicts-musical-genre-by-identifying-surprising-patterns-in-hit-songs/)
- [Stanford CCRMA: Generative AI for Music and Audio](https://ccrma.stanford.edu/events/generative-ai-music-and-audio),
  [Musically Intelligent Machines](https://ccrma.stanford.edu/events/juhan-nam-my-journey-toward-musically-intelligent-machines), and
  [Music, Brain & Health Lab research](https://musicbrainhealthlab.stanford.edu/research)
- [AIMC 2025 call: Artist in the Loop](https://aimusiccreativity.org/2025/call.html)
  and [NeurIPS 2025 AI for Music workshop](https://aiformusicworkshop.github.io/neurips2025/)
- [ISMIR dataset resources](https://ismir.net/resources/datasets/),
  [MIREX](https://music-ir.org/mirex/wiki/MIREX_HOME), and
  [`mir_eval`](https://mir-eval.readthedocs.io/latest/)
- [Aligning Text-to-Music Evaluation with Human Preferences](https://arxiv.org/abs/2503.16669)
