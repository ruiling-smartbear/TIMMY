from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from music_eval import __version__
from music_eval.distribution import compare_embedding_distributions, load_embedding_manifest
from music_eval.distribution_report import write_distribution_report
from music_eval.evaluator import ensure_output_directory, evaluate_manifest
from music_eval.listening import (
    DEFAULT_CRITERIA,
    analyze_listening_responses,
    build_listening_study,
    load_comparisons,
)
from music_eval.listening_report import write_listening_report
from music_eval.listening_server import serve_study
from music_eval.listening_web import write_listening_interface
from music_eval.longform import DEFAULT_LONGFORM_DURATIONS, write_longform_manifest
from music_eval.manifest import load_manifest
from music_eval.meta_evaluation import analyze_metric_ordering, load_ordering_manifest
from music_eval.meta_evaluation_report import write_meta_evaluation_report
from music_eval.perturbations import MAD_FIDELITY_SIGMAS, prepare_fidelity_perturbations
from music_eval.plugins import AnalysisConfig, available_plugins, resolve_plugins
from music_eval.report import write_reports
from music_eval.suites import SUITES, suite_names, write_suite_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="music-eval", description="Evaluate generated music outputs"
    )
    parser.add_argument("--version", action="version", version=f"music-eval {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a JSONL manifest")
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("--output", type=Path, default=Path("music-eval-report"))
    evaluate.add_argument("--metrics", default="integrity,pairwise")
    evaluate.add_argument(
        "--silence-dbfs",
        type=float,
        default=-50.0,
        help="mono RMS at or below this level counts as silence (default: %(default)s)",
    )
    evaluate.add_argument(
        "--window-seconds",
        type=float,
        default=0.25,
        help="window length for silent_window_ratio only; dropouts are found on 10 ms "
        "frames and their start and end are refined to the exact sample "
        "(default: %(default)s)",
    )
    evaluate.add_argument(
        "--minimum-dropout-seconds",
        type=float,
        default=0.5,
        help="shortest sample-accurate silent region reported as a dropout "
        "(default: %(default)s)",
    )
    evaluate.add_argument(
        "--clip-threshold",
        type=float,
        default=0.999,
        help="absolute sample value treated as clipped (default: %(default)s)",
    )
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

    longform = subparsers.add_parser(
        "init-longform-study",
        help="write matched prompt/seed cases at increasing target durations",
    )
    longform.add_argument("output", type=Path)
    longform.add_argument("--suite", default="generative-music-v1")
    longform.add_argument(
        "--durations", type=float, nargs="+", default=list(DEFAULT_LONGFORM_DURATIONS)
    )
    longform.add_argument("--seeds", type=int, nargs="+", default=[9, 17])
    longform.add_argument("--case-ids", nargs="+")
    longform.add_argument("--audio-directory", default="outputs")
    longform.add_argument("--sample-rate", type=int)
    longform.add_argument("--channels", type=int)
    longform.add_argument("--force", action="store_true", help="overwrite output")

    init_study = subparsers.add_parser(
        "init-listening-study", help="build an anonymous pairwise listening study"
    )
    init_study.add_argument("comparisons", type=Path)
    init_study.add_argument("--output", type=Path, default=Path("listening-study"))
    init_study.add_argument("--title", default="Generated music study")
    init_study.add_argument("--seed", type=int, default=20260906)
    init_study.add_argument("--repeat-fraction", type=float, default=0.1)
    init_study.add_argument("--criteria", default=",".join(DEFAULT_CRITERIA))
    init_study.add_argument("--force", action="store_true", help="overwrite output")

    serve = subparsers.add_parser("serve-listening-study", help="serve and collect responses")
    serve.add_argument("directory", type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    analyze = subparsers.add_parser(
        "analyze-listening-study", help="rank systems from listening responses"
    )
    analyze.add_argument("organizer_key", type=Path)
    analyze.add_argument("responses", type=Path, nargs="+")
    analyze.add_argument("--output", type=Path, default=Path("listening-report"))
    analyze.add_argument("--bootstrap-samples", type=int, default=1000)
    analyze.add_argument("--bootstrap-seed", type=int, default=20260906)

    distribution = subparsers.add_parser(
        "compare-distributions", help="compare systems in a shared audio embedding space"
    )
    distribution.add_argument("manifest", type=Path)
    distribution.add_argument("--reference-system", required=True)
    distribution.add_argument("--neighbors", type=int, default=3)
    distribution.add_argument("--output", type=Path, default=Path("distribution-report"))

    meta_evaluation = subparsers.add_parser(
        "analyze-metric-ordering",
        help="measure whether metric scores follow known degradation levels",
    )
    meta_evaluation.add_argument("manifest", type=Path)
    meta_evaluation.add_argument(
        "--output", type=Path, default=Path("metric-ordering-report")
    )

    perturbations = subparsers.add_parser(
        "prepare-fidelity-perturbations",
        help="create deterministic Gaussian-noise metric-audit fixtures",
    )
    perturbations.add_argument("sources", type=Path, nargs="+")
    perturbations.add_argument("--output", type=Path, required=True)
    perturbations.add_argument("--seed", type=int, default=20260906)
    perturbations.add_argument(
        "--sigmas", type=float, nargs="+", default=list(MAD_FIDELITY_SIGMAS)
    )
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


def _init_longform_study(args: argparse.Namespace) -> int:
    count = write_longform_manifest(
        args.suite,
        args.output,
        durations_seconds=tuple(args.durations),
        seeds=tuple(args.seeds),
        audio_directory=args.audio_directory,
        case_ids=None if args.case_ids is None else tuple(args.case_ids),
        sample_rate=args.sample_rate,
        channels=args.channels,
        overwrite=args.force,
    )
    print(f"wrote {count} matched long-form cases to {args.output}")
    return 0


def _init_listening_study(args: argparse.Namespace) -> int:
    if not math.isfinite(args.repeat_fraction):
        raise ValueError("--repeat-fraction must be finite")
    criteria = tuple(part.strip() for part in args.criteria.split(",") if part.strip())
    public_path, key_path, trial_count = build_listening_study(
        load_comparisons(args.comparisons),
        args.output,
        title=args.title,
        seed=args.seed,
        repeat_fraction=args.repeat_fraction,
        criteria=criteria,
        overwrite=args.force,
    )
    public = json.loads(public_path.read_text(encoding="utf-8"))
    write_listening_interface(public, args.output / "index.html")
    print(f"wrote {trial_count} anonymous trials to {args.output}")
    print(f"organizer key (keep private): {key_path}")
    return 0


def _analyze_listening_study(args: argparse.Namespace) -> int:
    key = json.loads(args.organizer_key.read_text(encoding="utf-8"))
    responses = [json.loads(path.read_text(encoding="utf-8")) for path in args.responses]
    result = analyze_listening_responses(
        key,
        responses,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_listening_report(result, args.output)
    print(f"analyzed {result['raters']} rater(s) across {len(result['systems'])} systems")
    print(f"reports: {args.output / 'report.json'}, {args.output / 'report.md'}, and "
          f"{args.output / 'report.html'}")
    return 0


def _compare_distributions(args: argparse.Namespace) -> int:
    result = compare_embedding_distributions(
        load_embedding_manifest(args.manifest),
        args.reference_system,
        neighbors=args.neighbors,
    )
    write_distribution_report(result, args.output)
    print(
        f"compared {len(result['comparisons'])} system(s) against {result['reference_system']}"
    )
    print(f"reports: {args.output / 'report.json'}, {args.output / 'report.md'}, and "
          f"{args.output / 'report.html'}")
    return 0


def _analyze_metric_ordering(args: argparse.Namespace) -> int:
    result = analyze_metric_ordering(load_ordering_manifest(args.manifest))
    write_meta_evaluation_report(result, args.output)
    print(f"analyzed {len(result['groups'])} metric-condition group(s)")
    print(
        f"reports: {args.output / 'report.json'}, {args.output / 'report.md'}, "
        f"and {args.output / 'report.html'}"
    )
    return 0


def _prepare_fidelity_perturbations(args: argparse.Namespace) -> int:
    result = prepare_fidelity_perturbations(
        args.sources, args.output, sigmas=tuple(args.sigmas), seed=args.seed
    )
    print(f"wrote {len(result['cases'])} perturbation case(s) to {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "evaluate":
            return _evaluate(args)
        if args.command == "init-suite":
            return _init_suite(args)
        if args.command == "init-longform-study":
            return _init_longform_study(args)
        if args.command == "init-listening-study":
            return _init_listening_study(args)
        if args.command == "serve-listening-study":
            if not 0 <= args.port <= 65535:
                raise ValueError("--port must be between 0 and 65535")
            serve_study(args.directory, args.host, args.port)
            return 0
        if args.command == "analyze-listening-study":
            return _analyze_listening_study(args)
        if args.command == "compare-distributions":
            return _compare_distributions(args)
        if args.command == "analyze-metric-ordering":
            return _analyze_metric_ordering(args)
        if args.command == "prepare-fidelity-perturbations":
            return _prepare_fidelity_perturbations(args)
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
