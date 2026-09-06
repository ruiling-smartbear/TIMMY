# Model support

“Supported” means the repository contains a real adapter to the publisher's
checkpoint and inference package. A backend protocol or a manifest field alone
does not count as model support.

| Model | Status | Role | Runtime | License note |
|---|---|---|---|---|
| Meta Audiobox Aesthetics | integrated | no-reference CE/CU/PC/PQ estimates | `music-eval[audiobox]` | inspect upstream code and weights terms |
| LAION CLAP HTSAT unfused | integrated | prompt/audio alignment and hard-negative rank | `music-eval[clap]` | inspect upstream checkpoint terms |
| OpenMuQ MuQ-MuLan-large | integrated | music-specific prompt/audio alignment and hard-negative rank | `music-eval[muq]` | weights are CC-BY-NC 4.0 |
| MuQ-Eval | planned | learned music-quality estimate | none | no adapter is shipped yet |
| FAD embedding backend | planned | corpus-level distribution distance | user-provided embeddings only | no encoder is bundled yet |
| Suno / Udio | not bundled | systems under evaluation | export WAV files into the manifest | proprietary generators, not evaluator checkpoints |

All learned scores remain evidence rather than automatic pass/fail gates. A
release should combine them with signal diagnostics, corpus-level comparison,
and a blinded listening study. The generic manifest accepts WAV outputs from
any generator; that does not imply that the generator's private API or model is
integrated into this package.
