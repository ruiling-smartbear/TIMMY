# Model support

“Supported” means the repository contains a real adapter to the publisher's
checkpoint and inference package. A backend protocol or a manifest field alone
does not count as model support.

| Model | Status | Role | Runtime | License note |
|---|---|---|---|---|
| Meta Audiobox Aesthetics | integrated | no-reference CE/CU/PC/PQ estimates | `music-eval[audiobox]` | model card states CC-BY 4.0; preserve attribution |
| LAION CLAP HTSAT unfused | integrated | prompt/audio alignment and hard-negative rank | `music-eval[clap]` | source is CC0; verify the selected checkpoint and training-data terms separately |
| OpenMuQ MuQ-MuLan-large | integrated | music-specific prompt/audio alignment and hard-negative rank | `music-eval[muq]` | weights are CC-BY-NC 4.0 |
| MuseCPEval 0.3.0 | integrated | edit-context preservation across harmony, rhythm, structure, melody and timbre | `music-eval[editing]` | MIT; structure pulls the separate `msaf` dependency stack |
| SongEval-style song-quality predictor | planned | learned overall song-quality estimate | none | no adapter is shipped yet |
| KAD statistic | integrated | corpus-level kernel distance | user-provided embeddings only | encoder/checkpoint/reference define the protocol |
| FAD embedding backend | planned | named corpus-level distribution distance | user-provided embeddings only | no named encoder adapter is bundled yet |
| Suno / Udio | not bundled | systems under evaluation | export WAV files into the manifest | proprietary generators, not evaluator checkpoints |

All learned scores remain evidence rather than automatic pass/fail gates. A
release should combine them with signal diagnostics, corpus-level comparison,
and a blinded listening study. The generic manifest accepts WAV outputs from
any generator; that does not imply that the generator's private API or model is
integrated into this package.
