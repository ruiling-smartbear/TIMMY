# Dataset selection and licensing

TIMMY does not download a benchmark corpus implicitly. Audio rights, access,
prompt coverage, and reference construction are part of the experiment, not an
implementation detail. This is an engineering inventory, not legal advice.

## Vetted candidates

| Dataset | Best use | Scale | Published terms | TIMMY decision |
| --- | --- | ---: | --- | --- |
| [Song Describer](https://github.com/mulab-mir/song-describer-dataset) | music-text alignment and hard-negative prompts | 706 tracks, about 1,100 captions | dataset CC BY-SA 4.0; repository MIT | preferred small public alignment qualification set; retain attribution/share-alike obligations |
| [MusicCaps](https://huggingface.co/datasets/google/MusicCaps/blob/main/README.md) | short music-text retrieval/alignment | 5,521 captions/YouTube segment IDs | dataset card says CC BY-SA 4.0 | use metadata after checking source-audio availability and platform terms; do not vendor audio |
| [MTG-Jamendo](https://mtg.github.io/mtg-jamendo-dataset/) | genre/instrument/mood strata and reference embeddings | 55,000+ tracks in common subsets | metadata CC BY-NC-SA 4.0; per-track CC licenses; project states non-commercial research use | research-only qualification with each audio license preserved |
| [FMA](https://github.com/mdeff/fma) / FMA-Pop | corpus reference for FAD/MAD-style studies | up to 106,574 tracks; FMA-Pop is a selected subset | code MIT, metadata CC BY 4.0, audio licenses vary by track | reproduce named papers only with the exact subset and track-level licenses; never call the whole corpus uniformly licensed |
| [MusicPrefs](https://huggingface.co/datasets/i-need-sleep/musicprefs) | metric-to-human-preference validation | 5,040 generated clips and 2,520 pairwise judgments reported by the paper | no explicit license was visible in the audited dataset card | do not redistribute or bundle; seek clarification before release/commercial use |
| [SongEval](https://huggingface.co/datasets/ASLP-lab/SongEval) | full-song, five-axis aesthetic calibration | 2,399 songs, over 140 hours, 16 expert raters reported | access is public; terms must be inspected at download time | candidate research qualification set, not bundled until license and complete split are recorded |
| [MAESTRO](https://magenta.withgoogle.com/datasets/maestro) | piano MIDI/audio timing, pitch and controlled musicality perturbations | about 200 hours depending on version | CC BY-NC-SA 4.0 | research-only symbolic/audio fixtures; always record dataset version |
| [MUSDB18](https://sigsep.github.io/datasets/musdb.html) | stem reconstruction and source-separation reference metrics | 150 multitrack songs | parser code is MIT; dataset access/terms are separate | suitable for qualified stem protocols after access terms are recorded; do not infer data rights from parser license |

## Minimum dataset record

Every serious TIMMY campaign should store:

- dataset name, version, source URL, retrieval date, and content digest;
- audio-level license/attribution when licenses vary;
- inclusion/exclusion rules and failed downloads;
- prompt, genre, language, vocal, duration, and demographic/cultural strata;
- reference/candidate sampling and any post-listening exclusions;
- preprocessing, resampling, clipping, segmentation, and normalization;
- which artifacts may be redistributed with the report.

## Recommended first qualification campaign

1. Use Song Describer for a small public CLAP/MuQ-MuLan alignment comparison,
   with nearby genre/instrument captions as hard negatives.
2. Use a license-filtered FMA-Pop manifest for named corpus metrics. Keep the
   encoder/checkpoint/reference fixed and report sample-size sensitivity.
3. Use MusicPrefs only for a non-redistributed metric-ranking study until its
   dataset license is explicit.
4. Use synthetic audio and MIDI exclusively for deterministic regression
   fixtures. These fixtures verify implementation behavior; they are not
   evidence of listener validity.

## Rejected shortcuts

- “Public download” is not a license.
- A repository's software license does not automatically license its audio.
- Missing YouTube segments must be reported, not silently replaced.
- Generated examples selected after listening cannot support an unbiased model
  comparison.
- A research-only or non-commercial dataset cannot qualify commercial use.
