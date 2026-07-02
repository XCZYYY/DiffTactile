#!/usr/bin/env python
import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

from difftactile.utils.headless_recording import DEFAULT_OUTPUT_ROOT, ensure_dir
from difftactile.utils import platform_recording as pr


DEMO_TASKS = [
    "box_open",
    "surface_follow",
    "object_repose",
    "cable_straightening",
]


@dataclass(frozen=True)
class DemoPlanItem:
    task: str
    cuda_device: str
    profile: str
    run_name: str
    extra_recorder_args: List[str] = field(default_factory=list)
    extra_task_args: List[str] = field(default_factory=list)


def split_argv(argv):
    if "--" not in argv:
        return argv, []
    idx = argv.index("--")
    return argv[:idx], argv[idx + 1 :]


def parse_devices(value: str) -> List[str]:
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if len(devices) < 2:
        raise ValueError("--devices must provide at least two GPU ids, for example 2,3")
    return devices


def build_demo_plan(devices: Sequence[str], profile: str = "demo") -> List[DemoPlanItem]:
    plan = []
    for idx, task in enumerate(DEMO_TASKS):
        extra_recorder_args = ["--pyopengl_platform", "egl"] if task == "surface_follow" else []
        extra_task_args = ["--disable_3d_window"] if task == "cable_straightening" else []
        plan.append(
            DemoPlanItem(
                task=task,
                cuda_device=devices[idx % len(devices)],
                profile=profile,
                run_name=f"{task}_platform_demo",
                extra_recorder_args=extra_recorder_args,
                extra_task_args=extra_task_args,
            )
        )
    return plan


def retry_plan_item(task: str, cuda_device: str) -> DemoPlanItem:
    extra_task_args = ["--gui_refresh_stride", "1"]
    if task in {"surface_follow", "object_repose"}:
        extra_task_args.extend(["--num_total_steps", "150"])
    if task == "cable_straightening":
        extra_task_args.append("--disable_3d_window")
    return DemoPlanItem(
        task=task,
        cuda_device=cuda_device,
        profile="demo_retry",
        run_name=f"{task}_platform_demo_smooth_retry",
        extra_recorder_args=["--pyopengl_platform", "egl"] if task == "surface_follow" else [],
        extra_task_args=extra_task_args,
    )


def parse_args(argv=None):
    main_argv, passthrough = split_argv(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Record smoother platform demo videos for all DiffTactile tasks.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--devices", default="2,3")
    parser.add_argument("--profile", choices=["demo"], default="demo")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--ffmpeg_loglevel", default="info")
    parser.add_argument("--ffmpeg_preset", default="ultrafast")
    parser.add_argument("--motion_threshold", type=float, default=0.15)
    parser.add_argument("--mean_motion_threshold", type=float, default=0.5)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(main_argv)
    args.devices = parse_devices(args.devices)
    args.passthrough = passthrough
    return args


def command_for_item(repo_root: Path, args, item: DemoPlanItem) -> List[str]:
    return [
        sys.executable,
        str(repo_root / "scripts" / "record_platform_run.py"),
        "--task",
        item.task,
        "--profile",
        item.profile,
        "--run_name",
        item.run_name,
        "--output_root",
        str(args.output_root),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--fps",
        str(args.fps),
        "--ffmpeg_loglevel",
        args.ffmpeg_loglevel,
        "--ffmpeg_preset",
        args.ffmpeg_preset,
        "--cuda_device",
        item.cuda_device,
        *item.extra_recorder_args,
        "--no_auto_extend",
        "--",
        *args.passthrough,
        *item.extra_task_args,
    ]


def run_item(repo_root: Path, args, item: DemoPlanItem) -> dict:
    command = command_for_item(repo_root, args, item)
    if args.dry_run:
        return {
            "task": item.task,
            "run_name": item.run_name,
            "cuda_device": item.cuda_device,
            "profile": item.profile,
            "command": command,
            "returncode": None,
            "dry_run": True,
        }

    completed = subprocess.run(command, cwd=repo_root)
    metadata_path = Path(args.output_root) / "runs" / item.task / item.run_name / "platform_recording_metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    return {
        "task": item.task,
        "run_name": item.run_name,
        "cuda_device": item.cuda_device,
        "profile": item.profile,
        "command": command,
        "returncode": completed.returncode,
        "metadata_path": str(metadata_path),
        "metadata": metadata,
    }


def copy_deliverable(output_root: Path, task: str, metadata: dict) -> Path:
    deliverable_dir = ensure_dir(output_root / "deliverables" / "platform_demos")
    src = Path(metadata["video_path"])
    dst = deliverable_dir / f"{task}_platform_demo.mp4"
    shutil.copy2(src, dst)
    return dst


def main(argv=None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    pr.validate_task_passthrough(args.passthrough)
    output_root = Path(args.output_root)
    records = []

    for item in build_demo_plan(args.devices, profile=args.profile):
        result = run_item(repo_root, args, item)
        if args.dry_run:
            records.append(result)
            continue
        if result["returncode"] != 0:
            records.append({**result, "ok": False, "retried": False})
            continue

        metadata = result["metadata"]
        motion = metadata.get("video_motion") or pr.video_motion_score(Path(metadata["video_path"]))
        min_duration = pr.resolve_profile(item.task, item.profile, item.extra_task_args + args.passthrough).min_duration_seconds
        duration = float(metadata.get("video_duration_seconds", 0.0))
        motion_ok = pr.video_motion_passes(
            motion,
            changed_fraction_threshold=args.motion_threshold,
            mean_absdiff_threshold=args.mean_motion_threshold,
        )
        needs_retry = not motion_ok or duration < min_duration
        final_result = {
            **result,
            "motion": motion,
            "min_duration_seconds": min_duration,
            "retried": False,
            "ok": duration >= min_duration and motion_ok,
        }
        if needs_retry:
            retry_item = retry_plan_item(item.task, item.cuda_device)
            retry_result = run_item(repo_root, args, retry_item)
            retry_metadata = retry_result.get("metadata", {})
            retry_motion = retry_metadata.get("video_motion")
            if retry_metadata.get("video_path"):
                retry_motion = retry_motion or pr.video_motion_score(Path(retry_metadata["video_path"]))
            retry_min_duration = pr.resolve_profile(
                retry_item.task,
                retry_item.profile,
                retry_item.extra_task_args + args.passthrough,
            ).min_duration_seconds
            retry_duration = float(retry_metadata.get("video_duration_seconds", 0.0))
            retry_motion_ok = pr.video_motion_passes(
                retry_motion or {},
                changed_fraction_threshold=args.motion_threshold,
                mean_absdiff_threshold=args.mean_motion_threshold,
            )
            final_result = {
                **retry_result,
                "motion": retry_motion,
                "min_duration_seconds": retry_min_duration,
                "retried": True,
                "ok": (
                    retry_result["returncode"] == 0
                    and retry_duration >= retry_min_duration
                    and retry_motion_ok
                ),
                "original": result,
            }

        if final_result["ok"] and final_result.get("metadata", {}).get("video_path"):
            deliverable = copy_deliverable(output_root, item.task, final_result["metadata"])
            final_result["deliverable_video"] = str(deliverable)
        records.append(final_result)

    index = {
        "output_root": str(output_root),
        "motion_threshold": args.motion_threshold,
        "mean_motion_threshold": args.mean_motion_threshold,
        "tasks": records,
        "ok": all(record.get("ok", True) for record in records),
    }
    index_path = ensure_dir(output_root / "deliverables" / "platform_demos") / "demo_index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(json.dumps(index, indent=2, sort_keys=True))
    return 0 if index["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
