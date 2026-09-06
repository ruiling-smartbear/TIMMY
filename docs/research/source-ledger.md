# TIMMY research source ledger

This ledger separates source statements from TIMMY's engineering inferences.
URLs point to papers, official repositories, standards bodies, or the
organization that conducted the work.

| ID | Primary source | Supported fact | TIMMY use | Caveat |
| --- | --- | --- | --- | --- |
| MAD-2025 | [paper](https://arxiv.org/abs/2503.16669), [official code](https://github.com/i-need-sleep/mad), [MusicPrefs](https://huggingface.co/datasets/i-need-sleep/musicprefs) | MAD evaluates metrics with fidelity, musicality, context, and diversity perturbations (K = 11 levels each, Table 4) and validates against 2,520 pairwise fidelity/musicality preferences | ordered metric meta-evaluation; the fidelity ladder in `prepare-fidelity-perturbations`; `musicprefs` listening-study criteria preset | sensitivity to synthetic orderings is not general validity |
| KAD-2025 | [paper](https://arxiv.org/abs/2502.15602), [official code](https://github.com/YoonjinXD/kadtk) | KAD is a kernel/MMD-based two-sample audio distance | exact estimator over supplied pinned embeddings | finite-sample estimates can be negative; reference determines bandwidth |
| FADtk | [official repository](https://github.com/microsoft/fadtk) | FAD depends on a named embedding and reference protocol; toolkit supports several encoders | do not label arbitrary embedding Fréchet distance as FAD | score comparability requires identical protocol |
| Audiobox | [paper](https://arxiv.org/abs/2502.05139), [model card](https://huggingface.co/facebook/audiobox-aesthetics/blob/main/README.md) | 562 hours of professionally rated audio; CE, CU, PC and PQ axes; model/data terms stated by Meta | optional four-axis windowed scorer | learned ratings are domain-dependent estimates |
| MuQ | [official repository](https://github.com/tencent-ailab/MuQ) | MuQ-MuLan supplies music/audio-text embeddings; repository documents 24 kHz and weight license | optional prompt alignment and hard-negative ranking | published weights are CC-BY-NC 4.0 |
| CLAP | [official repository](https://github.com/LAION-AI/CLAP) | contrastive language/audio representations support retrieval and similarity | optional independent alignment view | checkpoint and training-data terms must be checked separately |
| SongEval | [paper](https://arxiv.org/abs/2505.10793) | 2,399 full songs, over 140 hours, 16 professional annotators, five aesthetic dimensions rated 1 to 5 in isolation | `songeval` listening-study criteria preset (dimensions reused pairwise); motivates full-song multidimensional human calibration | benchmark publication does not make its axes universal; the paper's scale is absolute, TIMMY's judgment is pairwise |
| SongBench | [official repository](https://github.com/Tencent/SongBench) | song evaluation covers vocal, instrument, melody, structure, arrangement, mixing and musicality | candidate future multidimensional adapter | repository terms are academic/non-commercial |
| MuseCP | [paper](https://arxiv.org/abs/2512.14629), [official code](https://github.com/Yashvishe13/MuseCPEval) | evaluates harmony, rhythm, structure, melody and timbre preservation in music edits | pinned optional adapter | preservation is not target-edit success; structure extra has difficult dependencies |
| MusicWeaver | [paper](https://arxiv.org/abs/2509.21714) | introduces long-range structure coherence and edit fidelity around a beat-aligned plan | informs future region/plan-aware protocols | model-specific plan metrics are not yet a general implementation |
| YuE | [paper](https://arxiv.org/abs/2503.08638) | open long-form song generation reports multiple objective and human measures | supports portfolio rather than single-score design | exact protocol must be pinned before comparison |
| LeVo | [paper](https://arxiv.org/abs/2506.07520) | uses separated-vocal phoneme error and MuQ-MuLan; explains why word/character errors may be inaudible | motivates vocal PER adapter | the previously published official repository URL returned 404 during the 2026-09-06 audit; separation and ASR errors must remain visible |
| MUSHRA | [ITU-R BS.1534-3](https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1534-3-201510-I%21%21PDF-E.pdf) | specifies a multi-stimulus subjective method with anchors and reference requirements | reserve the name for compliant studies | generic A/B tests are not MUSHRA |
| BS.1116 | [ITU-R recommendation](https://www.itu.int/rec/R-REC-BS.1116-3-201502-I/en) | method for subjective assessment of small audio impairments | possible artifact-listening protocol | not a general creativity study |
| EBU-R128 | [EBU recommendation](https://tech.ebu.ch/publications/r128) | loudness normalization recommendation based on loudness and permitted peak | production-quality provenance | TIMMY's oversampled peak is not a certified meter |
| C2PA | [official specification](https://spec.c2pa.org/specifications/specifications/2.3/explainer/Explainer.html) | credentials convey signed provenance and tamper information | future provenance inspection | does not prove semantic truth or human authorship |
| CMU-creativity | [CMU report](https://www.cmu.edu/news/stories/archives/2026/january/as-ai-generated-music-advances-humans-still-lead-in-creativity-cmu-research-finds) | study with 140 musically trained participants evaluated process and outputs | motivates agency/workflow evidence | one study does not determine every creative workflow |
| AIMC-2025 | [official call](https://aimusiccreativity.org/2025/call.html) | “Artist in the Loop” includes control, discovery and benefits beyond output | motivates session-level protocols | workshop themes are agenda evidence, not metric validation |
| NeurIPS-AI-Music | [official workshop](https://aiformusicworkshop.github.io/neurips2025/) | covers evaluation, interaction, copyright/privacy, performance and production | roadmap coverage audit | workshop scope is not a benchmark standard |

## Explicit inferences

- Product claims should be translated into claim-specific, falsifiable
  protocols. This is TIMMY's design inference, not a quoted standard.
- A portfolio of separately reported metrics is safer than an opaque average
  because the sources evaluate different constructs and populations.
- TIMMY should remain independent from serving runtimes and integrate through
  optional artifact/report adapters. This is a maintenance decision.
- Region-aware edit success and boundary audibility remain gaps even after
  MuseCPEval context-preservation scores are available.
