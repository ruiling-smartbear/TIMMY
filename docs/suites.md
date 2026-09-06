# Benchmark suites and long-form ladders

A suite is a manifest whose prompts, seeds and expectations are fixed in
advance, so that two model runs or two serving builds produce comparable
reports. `init-suite` writes one; a model runner fills in the audio paths.

## The built-in genre suite

`generative-music-v1` holds two prompts in each of eight genres: J-pop,
orchestral, synthwave, acoustic folk, lo-fi hip-hop, rock, blues and ambient.
With two seeds that is 32 independently reported cases.

```bash
music-eval init-suite benchmark.jsonl \
  --suite generative-music-v1 \
  --seeds 9 17 \
  --duration-seconds 16 \
  --sample-rate 32000 \
  --channels 2
```

Each row points at `outputs/<case>-seed<seed>.wav`, carries the prompt, its
hard negatives, genre/mood/instrument/vocal labels, and expectations for the
requested duration, sample rate and channel count. Generate the audio with any
runner, then:

```bash
music-eval evaluate benchmark.jsonl --output report
```

`music-eval list-suites` prints the available suite names; `--force`
overwrites an existing manifest.

## Matched long-form ladders

Claims about long-form generation need matched controls. `init-longform-study`
keeps each suite prompt and seed fixed and varies only the requested duration,
by default 30 seconds, two minutes, four minutes and eight minutes
(`--durations` overrides the ladder):

```bash
music-eval init-longform-study longform.jsonl \
  --case-ids orchestral-epic ambient-drone \
  --seeds 9 17 \
  --audio-directory outputs
```

Every row records a stable `longform_group` and its target duration, so the
report can be read as one curve per prompt and seed. The window-preserving
metrics (`temporal_consistency`, `clap_alignment`, `audiobox_aesthetics`)
report every window, which is what a degradation-over-time analysis needs.
The ladder is the experiment design only; whether a model keeps its form at
eight minutes is a question for listening tests and structure ratings on the
generated audio.
