from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from music_eval.evaluator import ensure_output_directory, evaluate_manifest
from music_eval.manifest import load_manifest
from music_eval.plugins import AnalysisConfig, available_plugins, resolve_plugins
from music_eval.report import write_reports
from music_eval.suites import SUITES, suite_names, write_suite_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="music-eval", description="Evaluate generated music outputs"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a JSONL manifest")
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("--output", type=Path, default=Path("music-eval-report"))
    evaluate.add_argument("--metrics", default="integrity,pairwise")
    evaluate.add_argument("--silence-dbfs", type=float, default=-50.0)
    evaluate.add_argument("--window-seconds", type=float, default=0.25)
    evaluate.add_argument("--minimum-dropout-seconds", type=float, default=0.5)
    evaluate.add_argument("--clip-threshold", type=float, default=0.999)
    evaluate.add_argument(
        "--fail-on-warning", action="store_true", help="return non-zero for warnings"
    )

    subparsers.add_parser("list-metrics", help="list installed metric plugins")
    subparsers.add_parser("list-suites", help="list built-in benchmark suites")

    init_suite = subparsers.add_parser(
        "init-suite", help="write a benchmark manifest with output placeholders"
    )
    init_suite.add_argument("output", type=Path)
    init_suite.add_argument("--suite", default="generative-music-v1")
    init_suite.add_argument("--seeds", type=int, nargs="+", default=[9, 17])
    init_suite.add_argument("--audio-directory", default="outputs")
    init_suite.add_argument("--duration-seconds", type=float, default=16.0)
    init_suite.add_argument("--sample-rate", type=int)
    init_suite.add_argument("--channels", type=int)
    init_suite.add_argument("--force", action="store_true", help="overwrite output")
    return parser


def _analysis_config(args: argparse.Namespace) -> AnalysisConfig:
    values = (
        args.silence_dbfs,
        args.window_seconds,
        args.minimum_dropout_seconds,
        args.clip_threshold,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("analysis thresholds must be finite")
    if args.window_seconds <= 0:
        raise ValueError("--window-seconds must be positive")
    if args.minimum_dropout_seconds < 0:
        raise ValueError("--minimum-dropout-seconds must be nonnegative")
    if not 0.0 < args.clip_threshold <= 1.0:
        raise ValueError("--clip-threshold must be in the interval (0, 1]")
    return AnalysisConfig(
        silence_dbfs=args.silence_dbfs,
        window_seconds=args.window_seconds,
        minimum_dropout_seconds=args.minimum_dropout_seconds,
        clip_threshold=args.clip_threshold,
    )


def _evaluate(args: argparse.Namespace) -> int:
    config = _analysis_config(args)
    metric_names = [name.strip() for name in args.metrics.split(",") if name.strip()]
    if not metric_names:
        raise ValueError("--metrics must name at least one plugin")
    entries = load_manifest(args.manifest)
    results = evaluate_manifest(
        entries, config=config, plugins=resolve_plugins(metric_names)
    )
    ensure_output_directory(args.output)
    write_reports(results, args.output)
    failed = sum(result.status == "fail" for result in results)
    warnings = sum(result.status == "warning" for result in results)
    print(
        f"evaluated {len(results)} file(s): "
        f"{len(results) - failed - warnings} passed, {warnings} warnings, {failed} failed"
    )
    print(
        f"reports: {args.output / 'report.json'}, {args.output / 'report.md'}, "
        f"and {args.output / 'report.html'}"
    )
    return 1 if failed or (args.fail_on_warning and warnings) else 0


def _init_suite(args: argparse.Namespace) -> int:
    if not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be a positive finite number")
    if args.sample_rate is not None and args.sample_rate <= 0:
        raise ValueError("--sample-rate must be positive")
    if args.channels is not None and args.channels <= 0:
        raise ValueError("--channels must be positive")
    count = write_suite_manifest(
        args.suite,
        args.output,
        seeds=tuple(args.seeds),
        audio_directory=args.audio_directory,
        duration_seconds=args.duration_seconds,
        sample_rate=args.sample_rate,
        channels=args.channels,
        overwrite=args.force,
    )
    print(f"wrote {count} cases to {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "evaluate":
            return _evaluate(args)
        if args.command == "init-suite":
            return _init_suite(args)
        if args.command == "list-metrics":
            print("\n".join(available_plugins()))
            return 0
        if args.command == "list-suites":
            for name in suite_names():
                print(f"{name}\t{SUITES[name].description}")
            return 0
    except (OSError, ValueError) as error:
        print(f"music-eval: {error}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {args.command}")
