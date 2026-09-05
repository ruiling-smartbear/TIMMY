from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
SAMPLE_RATE = 32000


def write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = np.round(np.clip(samples, -1.0, 32767 / 32768) * 32768).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(np.column_stack((pcm, pcm)).tobytes())


time = np.arange(4 * SAMPLE_RATE) / SAMPLE_RATE
healthy = 0.2 * np.sin(2 * np.pi * 220 * time)
dropout = healthy.copy()
dropout[SAMPLE_RATE : 2 * SAMPLE_RATE] = 0.0
write_wav(ROOT / "healthy.wav", healthy)
write_wav(ROOT / "dropout.wav", dropout)

entries = [
    {
        "id": "healthy-demo",
        "audio": "healthy.wav",
        "prompt": "steady 220 Hz demonstration tone",
        "labels": {"genre": "demo", "condition": "healthy"},
        "expectations": {
            "duration_seconds": 4.0,
            "sample_rate": SAMPLE_RATE,
            "channels": 2,
            "max_dropout_seconds": 0.0,
        },
    },
    {
        "id": "dropout-demo",
        "audio": "dropout.wav",
        "prompt": "demonstration tone with an injected defect",
        "labels": {"genre": "demo", "condition": "injected-dropout"},
        "expectations": {
            "duration_seconds": 4.0,
            "sample_rate": SAMPLE_RATE,
            "channels": 2,
            "max_dropout_seconds": 0.5,
        },
    },
]
(ROOT / "manifest.jsonl").write_text(
    "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
)
print(f"wrote demo inputs under {ROOT}")
