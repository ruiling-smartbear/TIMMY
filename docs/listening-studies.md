# Blind listening studies

The listening layer answers preference questions that waveform statistics and
learned predictors cannot settle. It is pairwise by design: raters compare two
outputs generated from the same prompt rather than assigning an uncalibrated
absolute score.

## Experimental contract

- Use the same prompt, requested duration, seed policy, and post-processing for
  both systems in a comparison.
- Do not encode system identity in prompts, public filenames, page text, or
  metadata exposed by the study package.
- Randomize left/right position and trial order with a recorded seed.
- Sample prompts across the genres and conditions named in the benchmark plan;
  do not select only favorable generations after listening.
- Choose rater codes that are anonymous but stable within a study.
- Use headphones and one reasonably quiet environment where possible.
- Treat repeat-trial agreement as measurement quality evidence, not as a reason
  to silently delete a rater. State any exclusion policy before collecting data.

The public package and organizer key are deliberately separated. The public
`study.json` contains only anonymous trial IDs, prompt text, and hashed audio
paths. Labels remain private as well because a careless `model` or `backend`
label could disclose identity. The sibling `.organizer.json` file maps A/B
positions to systems, retains labels for later stratification, and identifies
repeated source cases. Analysis requires the private key.

## Criteria

The defaults deliberately remain separate:

- `overall_preference`: which sample the listener would choose overall;
- `prompt_alignment`: which better realizes the requested content;
- `musicality_structure`: which has more convincing composition and temporal
  development;
- `production_quality`: which has cleaner, more coherent sound and mix.

A tie is a first-class answer. Forcing a random A/B selection invents evidence.

## Analysis

The report shows two complementary summaries. Preference rate counts a tie as
half a win. Bradley–Terry estimation fits a global latent ordering from the
pairwise outcomes and reports centered log strength plus the implied win
probability against an average system. These numbers are meaningful only when
the comparison graph connects the systems and the sample is sufficiently
large; they are not universal quality scores.

Repeat reliability compares the winning *system*, not the displayed A/B side,
across hidden duplicate trials. This is essential because duplicate trials may
reverse sides. A study with no repeats reports reliability as unmeasured.

The current implementation reports point estimates. Before using a study for a
publication claim, add an a priori power plan and confidence intervals or a
rater-level bootstrap appropriate to the sampling design.
