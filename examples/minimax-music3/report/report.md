# Music evaluation report

Total: **5** · Passed: **2** · Warnings: **2** · Failed: **1**

| ID | Genre | Status | Duration | RMS | Longest dropout | First↔Last | Repeat | LUFS | LRA | Est. dBTP | Findings |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 00_lofi_hiphop | lo-fi hip-hop | warning | 30.023s | -16.06 dBFS | 0.000s | 0.828 | 0.0% | -13.50 | 7.73 | 0.37 | integrity:clipping_detected |
| 01_jpop_bright | j-pop | fail | 26.227s | -19.17 dBFS | 0.000s | 0.325 | 0.0% | -16.01 | 9.87 | -1.62 | integrity:duration_mismatch |
| 02_synthwave_moody | synthwave | pass | 30.023s | -21.06 dBFS | 0.000s | 0.729 | 0.0% | -18.89 | 6.83 | -1.13 | — |
| 03_acoustic_folk | acoustic folk | pass | 30.023s | -17.41 dBFS | 0.000s | 0.743 | 0.0% | -14.27 | 9.35 | -0.23 | — |
| 04_orchestral_epic | orchestral | warning | 30.023s | -20.06 dBFS | 0.000s | 0.690 | 0.0% | -17.59 | 16.40 | 0.01 | integrity:clipping_detected |

## By backend

| Value | Samples | Pass | Warning | Fail | Pass rate |
|---|---:|---:|---:|---:|---:|
| sglang-omni | 5 | 2 | 2 | 1 | 40.0% |

## By genre

| Value | Samples | Pass | Warning | Fail | Pass rate |
|---|---:|---:|---:|---:|---:|
| acoustic folk | 1 | 1 | 0 | 0 | 100.0% |
| j-pop | 1 | 0 | 0 | 1 | 0.0% |
| lo-fi hip-hop | 1 | 0 | 1 | 0 | 0.0% |
| orchestral | 1 | 0 | 1 | 0 | 0.0% |
| synthwave | 1 | 1 | 0 | 0 | 100.0% |

## By model

| Value | Samples | Pass | Warning | Fail | Pass rate |
|---|---:|---:|---:|---:|---:|
| minimax-music3 | 5 | 2 | 2 | 1 | 40.0% |

## By tempo

| Value | Samples | Pass | Warning | Fail | Pass rate |
|---|---:|---:|---:|---:|---:|
| fast | 1 | 0 | 0 | 1 | 0.0% |
| medium | 2 | 1 | 1 | 0 | 50.0% |
| slow | 2 | 1 | 1 | 0 | 50.0% |

## By vocals

| Value | Samples | Pass | Warning | Fail | Pass rate |
|---|---:|---:|---:|---:|---:|
| instrumental | 1 | 1 | 0 | 0 | 100.0% |
| vocal | 4 | 1 | 2 | 1 | 25.0% |

## 00_lofi_hiphop

- **warning · integrity:clipping_detected:** 0.0106% of samples are near full scale

## 01_jpop_bright

- **failure · integrity:duration_mismatch:** duration 26.227s differs from expected 30.000s by 3.773s

## 04_orchestral_epic

- **warning · integrity:clipping_detected:** 0.0001% of samples are near full scale
