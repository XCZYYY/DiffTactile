#!/usr/bin/env python
import argparse
import json
from pathlib import Path

from difftactile.utils.headless_recording import DEFAULT_OUTPUT_ROOT
from difftactile.utils.showcase_rendering import (
    SHOWCASE_TASKS,
    ShowcaseConfig,
    render_showcase_suite,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Render advisor-facing 3D showcase videos for DiffTactile.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration_seconds", type=float, default=16.0)
    parser.add_argument("--min_duration_seconds", type=float, default=10.0)
    parser.add_argument("--pyopengl_platform", default="egl")
    parser.add_argument("--task", action="append", choices=SHOWCASE_TASKS)
    parser.add_argument("--no_clean", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    config = ShowcaseConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration_seconds=args.duration_seconds,
        pyopengl_platform=args.pyopengl_platform,
        min_duration_seconds=args.min_duration_seconds,
    )
    summary_request = {
        "repo_root": str(repo_root),
        "output_root": str(args.output_root),
        "tasks": args.task or list(SHOWCASE_TASKS),
        "config": config.__dict__,
        "clean": not args.no_clean,
    }
    if args.dry_run:
        print(json.dumps(summary_request, indent=2, sort_keys=True))
        return 0

    summary = render_showcase_suite(
        output_root=Path(args.output_root),
        repo_root=repo_root,
        config=config,
        tasks=args.task,
        clean=not args.no_clean,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
