"""Download the five MiniMax Music 3 reference outputs used by this example.

The files are the reference outputs that the SGLang-Omni cookbook for MiniMax
Music 3 links (docs/cookbook/minimax_music3.md in sgl-project/sglang-omni).
They are generated audio published by that project, not part of this
repository; run this script once and the manifest resolves.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

FILES = {
    "00_lofi_hiphop.wav": (
        "https://github.com/user-attachments/files/30963578/00_lofi_hiphop.wav",
        "31e1c58b1eec04216c33998bb37d73daf5decfea54bf6ba3d40cb6fb4a9fc1b1",
    ),
    "01_jpop_bright.wav": (
        "https://github.com/user-attachments/files/30963575/01_jpop_bright.wav",
        "567b426a4f794365c22918a71cd7e9507f6c88a998e8ee82e9e0875dec290161",
    ),
    "02_synthwave_moody.wav": (
        "https://github.com/user-attachments/files/30963577/02_synthwave_moody.wav",
        "12888b2723bebd370bf5ecd2022aa4127c30b664a7b1cbae68bc0e094292cbbf",
    ),
    "03_acoustic_folk.wav": (
        "https://github.com/user-attachments/files/30963579/03_acoustic_folk.wav",
        "9d73e4c22284734834fc18e568d659f1b36560dad55b385bc589ac609eda1cd1",
    ),
    "04_orchestral_epic.wav": (
        "https://github.com/user-attachments/files/30963576/04_orchestral_epic.wav",
        "e809ec54b7a85715582b4e3917b2aaf3b8947c0bdcfff07856f4ca5fe31af773",
    ),
}
AUDIO_DIR = Path(__file__).parent / "audio"


def main() -> int:
    AUDIO_DIR.mkdir(exist_ok=True)
    failures = 0
    for name, (url, expected) in FILES.items():
        target = AUDIO_DIR / name
        if not target.exists():
            print(f"downloading {name}")
            with urllib.request.urlopen(url) as response:
                target.write_bytes(response.read())
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != expected:
            print(f"{name}: sha256 {digest} does not match {expected}", file=sys.stderr)
            failures += 1
        else:
            print(f"{name}: ok")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
