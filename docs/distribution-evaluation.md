# Distribution evaluation

Single-track evaluators answer whether one output is intact, aligned, polished,
or preferred. They cannot show whether a generator covers the reference
distribution or repeatedly emits the same few patterns. This command treats a
system's outputs as a corpus in one declared audio-embedding space.

## Input contract

The input is JSONL with these fields:

| Field | Required | Meaning |
|---|---:|---|
| `id` | yes | Unique sample identifier |
| `system` | yes | Candidate or reference corpus name |
| `embedding` | yes | Non-empty finite numeric vector |
| `embedding_model` | yes | Encoder architecture/protocol identity |
| `checkpoint` | yes | Exact checkpoint revision or digest |
| `labels` | no | String or string-list strata such as genre and mood |

Every record must have the same embedding dimension, model identity, and
checkpoint. Mixed provenance is rejected because distances across incompatible
spaces are meaningless. Generation prompt/seed policy, clip duration,
preprocessing, and corpus sampling should be versioned beside the manifest.

## Reported evidence

- **Fréchet embedding distance:** Gaussian mean/covariance distance. Lower is
  closer in this embedding space. It is sensitive to sample size and does not
  become Fréchet Audio Distance merely because the inputs represent audio.
- **Candidate diversity:** mean pairwise cosine distance inside the candidate
  set. Low diversity can indicate collapse, but can also reflect a deliberately
  narrow test stratum.
- **Candidate → reference nearest distance:** a fidelity-style diagnostic;
  lower means each generated sample lies near some reference sample.
- **Reference → candidate nearest distance:** a coverage-style diagnostic;
  lower means reference regions have nearby generated examples.
- **k-NN precision, recall, density, coverage:** non-parametric manifold
  evidence. These values are omitted rather than fabricated when either corpus
  has at most `k` examples.

No default pass/fail threshold is applied. Results should be compared using the
same encoder, checkpoint, sampling plan, corpus size, and label distribution.
Genre-level results are usually more actionable than one aggregate distance.

## Sampling requirements

Use at least dozens—and preferably hundreds—of independently generated clips
per system for a serious distribution claim. Balance prompts and seeds across
systems. Do not select candidate outputs after listening. A candidate and its
reference corpus should use comparable durations and preprocessing. Report the
sample count beside every metric; more data reduces but does not eliminate
embedding-model bias.
