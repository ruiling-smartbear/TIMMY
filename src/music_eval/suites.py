from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SuiteCase:
    id: str
    prompt: str
    genre: str
    mood: tuple[str, ...]
    instruments: tuple[str, ...]
    vocals: str
    negative_genres: tuple[str, ...]


@dataclass(frozen=True)
class SuiteDefinition:
    name: str
    description: str
    cases: tuple[SuiteCase, ...]


GENERATIVE_MUSIC_V1 = SuiteDefinition(
    name="generative-music-v1",
    description="Eight genres, two prompts per genre, designed for stratified evaluation.",
    cases=(
        SuiteCase(
            "jpop-bright-band",
            "Bright J-pop with energetic female vocals, electric guitar, and a catchy chorus",
            "j-pop",
            ("bright", "energetic"),
            ("electric guitar", "drums"),
            "female",
            ("ambient", "blues", "orchestral"),
        ),
        SuiteCase(
            "jpop-city-night",
            "Polished modern J-pop for a neon city night, melodic bass and crisp drums",
            "j-pop",
            ("uplifting", "night"),
            ("synthesizer", "bass", "drums"),
            "unspecified",
            ("acoustic folk", "blues", "orchestral"),
        ),
        SuiteCase(
            "orchestral-epic",
            "Epic orchestral score with sweeping strings, brass, and cinematic percussion",
            "orchestral",
            ("epic", "dramatic"),
            ("strings", "brass", "percussion"),
            "instrumental",
            ("lo-fi hip-hop", "blues", "synthwave"),
        ),
        SuiteCase(
            "orchestral-intimate",
            "Intimate chamber orchestra with expressive strings and gentle woodwinds",
            "orchestral",
            ("intimate", "gentle"),
            ("strings", "woodwinds"),
            "instrumental",
            ("rock", "synthwave", "lo-fi hip-hop"),
        ),
        SuiteCase(
            "synthwave-drive",
            "Driving synthwave with analog arpeggios, gated drums, and a dark retro mood",
            "synthwave",
            ("dark", "driving", "retro"),
            ("synthesizer", "electronic drums"),
            "instrumental",
            ("acoustic folk", "orchestral", "blues"),
        ),
        SuiteCase(
            "synthwave-dream",
            "Dreamy synthwave with warm pads, a steady electronic beat, and nostalgic melody",
            "synthwave",
            ("dreamy", "nostalgic"),
            ("synthesizer", "electronic drums"),
            "instrumental",
            ("rock", "blues", "acoustic folk"),
        ),
        SuiteCase(
            "folk-warm",
            "Warm acoustic folk with fingerpicked guitar, soft violin, and natural vocals",
            "acoustic folk",
            ("warm", "organic"),
            ("acoustic guitar", "violin"),
            "unspecified",
            ("synthwave", "orchestral", "lo-fi hip-hop"),
        ),
        SuiteCase(
            "folk-morning",
            "Gentle acoustic folk for a quiet morning with guitar and light hand percussion",
            "acoustic folk",
            ("gentle", "calm"),
            ("acoustic guitar", "hand percussion"),
            "instrumental",
            ("rock", "synthwave", "orchestral"),
        ),
        SuiteCase(
            "lofi-study",
            "Relaxed lo-fi hip-hop study beat with dusty drums, mellow keys, and vinyl texture",
            "lo-fi hip-hop",
            ("relaxed", "mellow"),
            ("drums", "electric piano"),
            "instrumental",
            ("orchestral", "rock", "j-pop"),
        ),
        SuiteCase(
            "lofi-rain",
            "Late-night lo-fi hip-hop with soft piano chords, restrained beat, and rainy ambience",
            "lo-fi hip-hop",
            ("late-night", "calm"),
            ("piano", "drums"),
            "instrumental",
            ("orchestral", "rock", "blues"),
        ),
        SuiteCase(
            "rock-arena",
            "High-energy arena rock with distorted guitars, powerful drums, and a bold chorus",
            "rock",
            ("energetic", "bold"),
            ("electric guitar", "drums", "bass"),
            "unspecified",
            ("ambient", "acoustic folk", "orchestral"),
        ),
        SuiteCase(
            "rock-garage",
            "Raw garage rock with crunchy guitar riffs, live drums, and an urgent performance",
            "rock",
            ("raw", "urgent"),
            ("electric guitar", "drums"),
            "unspecified",
            ("ambient", "synthwave", "orchestral"),
        ),
        SuiteCase(
            "blues-delta",
            "Slow Delta blues with slide guitar, weathered vocals, and a sparse groove",
            "blues",
            ("weathered", "sparse"),
            ("slide guitar",),
            "unspecified",
            ("j-pop", "synthwave", "orchestral"),
        ),
        SuiteCase(
            "blues-electric",
            "Electric blues shuffle with expressive lead guitar, bass, drums, and organ",
            "blues",
            ("expressive", "groovy"),
            ("electric guitar", "organ", "bass", "drums"),
            "instrumental",
            ("ambient", "j-pop", "acoustic folk"),
        ),
        SuiteCase(
            "ambient-drone",
            "Slow-evolving ambient drone with deep textures, wide space, and no percussion",
            "ambient",
            ("spacious", "meditative"),
            ("synthesizer",),
            "instrumental",
            ("rock", "j-pop", "blues"),
        ),
        SuiteCase(
            "ambient-piano",
            "Minimal ambient piano surrounded by airy pads and long quiet decay",
            "ambient",
            ("minimal", "quiet"),
            ("piano", "synthesizer"),
            "instrumental",
            ("rock", "j-pop", "synthwave"),
        ),
    ),
)

SUITES = {GENERATIVE_MUSIC_V1.name: GENERATIVE_MUSIC_V1}


def suite_names() -> list[str]:
    return sorted(SUITES)


def write_suite_manifest(
    suite_name: str,
    output: Path,
    *,
    seeds: tuple[int, ...],
    audio_directory: str,
    duration_seconds: float | None,
    sample_rate: int | None,
    channels: int | None,
    overwrite: bool = False,
) -> int:
    if suite_name not in SUITES:
        raise ValueError(f"unknown suite '{suite_name}'; available: {', '.join(suite_names())}")
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if not audio_directory.strip():
        raise ValueError("audio directory must be nonempty")
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing suite manifest: {output}")
    entries = []
    for case in SUITES[suite_name].cases:
        for seed in seeds:
            expectations: dict[str, object] = {}
            if duration_seconds is not None:
                expectations.update(
                    duration_seconds=duration_seconds,
                    duration_tolerance_seconds=0.25,
                )
            if sample_rate is not None:
                expectations["sample_rate"] = sample_rate
            if channels is not None:
                expectations["channels"] = channels
            entries.append(
                {
                    "id": f"{case.id}-seed{seed}",
                    "audio": f"{audio_directory.rstrip('/')}/{case.id}-seed{seed}.wav",
                    "prompt": case.prompt,
                    "negative_prompts": [
                        f"This is {negative_genre} music"
                        for negative_genre in case.negative_genres
                    ],
                    "seed": seed,
                    "labels": {
                        "genre": case.genre,
                        "mood": list(case.mood),
                        "instruments": list(case.instruments),
                        "vocals": case.vocals,
                        "duration_bucket": _duration_bucket(duration_seconds),
                    },
                    "expectations": expectations,
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
        encoding="utf-8",
    )
    return len(entries)


def _duration_bucket(duration_seconds: float | None) -> str:
    if duration_seconds is None:
        return "unspecified"
    if duration_seconds < 30:
        return "short"
    if duration_seconds < 120:
        return "medium"
    return "long"
