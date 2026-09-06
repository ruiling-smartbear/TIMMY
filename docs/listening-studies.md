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
`study.json` contains the title, the criteria and their definitions, anonymous
trial IDs, prompt text, and hashed audio paths. Labels remain private as well because a careless `model` or `backend`
label could disclose identity. The sibling `.organizer.json` file maps A/B
positions to systems, retains labels for later stratification, and identifies
repeated source cases. Analysis requires the private key.

## Criteria

The default set (`--criteria-preset timmy`, also used when nothing is given)
keeps four questions apart:

- `overall_preference`: which sample the listener would choose overall;
- `prompt_alignment`: which better realizes the requested content;
- `musicality_structure`: which has more convincing composition and temporal
  development;
- `production_quality`: which has cleaner, more coherent sound and mix.

A tie is a first-class answer. Forcing a random A/B selection invents evidence.

### Presets from published protocols

Different questions need different criteria, and a criteria set that a paper
already validated is easier to compare against than a home-made one.
`init-listening-study --criteria-preset` offers:

| Preset | Criteria | Source | Use it when |
|---|---|---|---|
| `timmy` | the four above | this project | prompts and mastering matter as much as the music |
| `musicprefs` | `fidelity`, `musicality` | Huang et al. 2025, MusicPrefs (arXiv 2503.16669): 2,520 pairwise judgments over seven systems, one preference per axis | you want the protocol that MAD was validated against, or a quick two-axis system comparison |
| `songeval` | `overall_coherence`, `memorability`, `vocal_naturalness`, `structure_clarity`, `overall_musicality` | Yao et al. 2025, SongEval (arXiv 2505.10793): 2,399 full songs rated by 16 trained annotators | you need to know *why* one system wins on full-length songs with vocals |
| `songeval-instrumental` | SongEval without `vocal_naturalness` | derived from the above | the same, for instrumental material |

Two differences from the papers are deliberate and recorded in the study
file. SongEval rated each song in isolation on a 1 to 5 scale; TIMMY keeps its
five dimensions but asks them pairwise, so the result is a preference per
dimension, not a MOS. MusicPrefs collected ties and then discarded them;
TIMMY keeps ties as half-wins. The one-line definition raters see under each
criterion (`criteria_definitions` in `study.json`) paraphrases the papers'
descriptions; `--criteria` accepts any custom list, which gets no definition
unless the name matches a known one.

## Analysis

The report shows two complementary summaries. Preference rate counts a tie as
half a win. Bradley–Terry estimation fits a global latent ordering from the
pairwise outcomes and reports centered log strength plus the implied win
probability against an average system. The fit scores a tie as half a win for
each side, so a tie contributes half of a win likelihood term in each
direction, matching the preference-rate convention. Each ranking row carries a
`separated` flag that is true when the maximum-likelihood estimate does not
exist because some system or group of systems never lost or never won; the
strengths shown are then a lightly ridge-regularized estimate whose magnitudes
reflect the regularizer rather than the evidence, so read them as an ordering
only (the report marks such rows with an asterisk). These numbers are
meaningful only when the comparison graph connects the systems and the sample
is sufficiently large; they are not universal quality scores.

Repeat reliability compares the winning *system*, not the displayed A/B side,
across hidden duplicate trials. This is essential because duplicate trials may
reverse sides. A study with no repeats reports reliability as unmeasured.

When at least two raters are available, the analyzer reports percentile 95%
intervals from a seeded rater-level bootstrap. Whole raters—not individual
answers—are resampled, preserving the dependence among one person's answers.
These intervals capture rater-sampling uncertainty only; they do not capture
uncertainty from prompt or corpus selection.

The report also includes exact pairwise agreement between raters, the fraction
of trials with unanimous choices, and raw A/B-side selection. A large side
imbalance is a diagnostic to inspect the randomization and interface; it is not
automatically proof of bias because finite randomization can place a stronger
system on one side more often.

Before using a study for a publication claim, define the exclusion policy and
power/sample-size plan in advance. Bootstrap intervals cannot repair a biased
prompt sample or an underpowered experimental design.
