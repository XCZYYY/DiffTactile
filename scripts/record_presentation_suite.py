#!/usr/bin/env python
import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np

from difftactile.utils.headless_recording import DEFAULT_OUTPUT_ROOT, ensure_dir
from difftactile.utils import platform_recording as pr


PRESENTATION_TASKS = [
    "box_open",
    "surface_follow",
    "object_repose",
    "cable_straightening",
]

TASK_GUIDE = {
    "box_open": {
        "title": "Box Open",
        "summary": "A tactile dome presses and pushes a blue box-like object to open or change its angle.",
        "legend": "Blue: object particles. Yellow: tactile dome. Right panels: force and deformation marker maps.",
    },
    "surface_follow": {
        "title": "Surface Follow",
        "summary": "A tactile dome presses onto a curved surface and slides while maintaining contact.",
        "legend": "Blue: surface mesh/particles. Yellow: tactile dome. Right panels: tactile force and marker deformation.",
    },
    "object_repose": {
        "title": "Object Repose",
        "summary": "A tactile dome pushes a rigid object toward a new pose against the scene boundary.",
        "legend": "Blue: object. Yellow: tactile dome. Right panel: tactile marker displacement.",
    },
    "cable_straightening": {
        "title": "Cable Straightening",
        "summary": "A tactile gripper manipulates a rope-like cable and tries to straighten it.",
        "legend": "Blue/red: cable and gripper-side geometry. Yellow/pink: tactile sensor contacts. Right panels: tactile maps.",
    },
}


@dataclass(frozen=True)
class PresentationPlanItem:
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


def _task_title(task: str) -> str:
    return TASK_GUIDE[task]["title"]


def build_presentation_plan(devices: Sequence[str], profile: str = "replay") -> List[PresentationPlanItem]:
    plan = []
    for idx, task in enumerate(PRESENTATION_TASKS):
        recorder_args = ["--pyopengl_platform", "egl"] if task == "surface_follow" else []
        task_args = [
            "--demo_mode",
            "replay",
            "--demo_overlay",
            "--replay_loops",
            "2",
            "--replay_speed",
            "1.0",
            "--demo_warmup_frames",
            "8",
            "--demo_title",
            _task_title(task),
        ]
        if task == "cable_straightening":
            task_args.append("--disable_3d_window")
        plan.append(
            PresentationPlanItem(
                task=task,
                cuda_device=devices[idx % len(devices)],
                profile=profile,
                run_name=f"{task}_platform_presentation",
                extra_recorder_args=recorder_args,
                extra_task_args=task_args,
            )
        )
    return plan


def parse_args(argv=None):
    main_argv, passthrough = split_argv(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Record explanatory replay demos for DiffTactile tasks.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--devices", default="2,3")
    parser.add_argument("--profile", choices=["replay"], default="replay")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--ffmpeg_loglevel", default="info")
    parser.add_argument("--ffmpeg_preset", default="ultrafast")
    parser.add_argument("--active_fraction_threshold", type=float, default=0.6)
    parser.add_argument("--max_static_run_seconds", type=int, default=5)
    parser.add_argument("--max_first_visible_second", type=float, default=2.0)
    parser.add_argument("--max_black_fraction", type=float, default=0.25)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(main_argv)
    args.devices = parse_devices(args.devices)
    args.passthrough = passthrough
    return args


def command_for_item(repo_root: Path, args, item: PresentationPlanItem) -> List[str]:
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


def run_item(repo_root: Path, args, item: PresentationPlanItem) -> dict:
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
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
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


def write_contact_sheet(video_path: Path, output_path: Path, samples: int = 8) -> Path:
    ensure_dir(output_path.parent)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frames / fps if fps else 0.0
    thumbs = []
    for idx, frac in enumerate(np.linspace(0.03, 0.97, samples)):
        cap.set(cv2.CAP_PROP_POS_MSEC, duration * float(frac) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        thumb = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        cv2.putText(
            thumb,
            f"{duration * float(frac):.1f}s",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        thumbs.append(thumb)
    cap.release()
    if not thumbs:
        raise RuntimeError(f"Could not sample video for contact sheet: {video_path}")
    while len(thumbs) % 4:
        thumbs.append(np.zeros_like(thumbs[0]))
    rows = [np.hstack(thumbs[row : row + 4]) for row in range(0, len(thumbs), 4)]
    cv2.imwrite(str(output_path), np.vstack(rows))
    return output_path


def copy_deliverable(output_root: Path, task: str, metadata: dict) -> Path:
    deliverable_dir = ensure_dir(output_root / "deliverables" / "platform_presentation_demos")
    src = Path(metadata["video_path"])
    dst = deliverable_dir / f"{task}_platform_presentation.mp4"
    shutil.copy2(src, dst)
    return dst


def _quality_ok(quality: dict, args) -> bool:
    first_visible = quality.get("first_visible_second")
    return (
        pr.presentation_motion_passes(
            quality,
            active_fraction_threshold=args.active_fraction_threshold,
            max_static_run_seconds=args.max_static_run_seconds,
        )
        and first_visible is not None
        and float(first_visible) <= args.max_first_visible_second
        and float(quality.get("black_fraction", 1.0)) <= args.max_black_fraction
    )


def write_deliverable_docs(output_root: Path, records: Sequence[dict]):
    deliverable_dir = ensure_dir(output_root / "deliverables" / "platform_presentation_demos")
    index = {
        "output_root": str(output_root),
        "tasks": list(records),
        "ok": all(record.get("ok", False) for record in records),
    }
    index_path = deliverable_dir / "demo_index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    lines = [
        "# DiffTactile Platform Presentation Demos",
        "",
        "All videos are real platform screen captures from Xvfb + Taichi GUI recorded with ffmpeg x11grab.",
        "",
    ]
    for record in records:
        guide = TASK_GUIDE.get(record["task"], {"title": record["task"], "summary": "", "legend": ""})
        quality = (
            record.get("metadata", {})
            .get("video_motion", {})
            .get("presentation_quality", {})
        )
        lines.extend(
            [
                f"## {guide['title']}",
                "",
                guide["summary"],
                "",
                guide["legend"],
                "",
                f"- Video: `{record.get('deliverable_video')}`",
                f"- Contact sheet: `{record.get('contact_sheet')}`",
                f"- GPU: `{record.get('cuda_device')}`",
                f"- Run: `{record.get('run_name')}`",
                f"- Duration: `{record.get('metadata', {}).get('video_duration_seconds')}` seconds",
                f"- active_fraction: `{quality.get('active_fraction')}`",
                f"- longest_static_run_seconds: `{quality.get('longest_static_run_seconds')}`",
                "",
            ]
        )
    guide_path = deliverable_dir / "demo_guide.md"
    guide_path.write_text("\n".join(lines) + "\n")
    return index_path, guide_path


def main(argv=None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    pr.validate_task_passthrough(args.passthrough)
    output_root = Path(args.output_root)
    records = []

    for item in build_presentation_plan(args.devices, profile=args.profile):
        result = run_item(repo_root, args, item)
        if args.dry_run:
            records.append({**result, "ok": True})
            continue

        metadata = result.get("metadata", {})
        ok = result["returncode"] == 0 and bool(metadata.get("video_path"))
        quality = {}
        if ok:
            video_path = Path(metadata["video_path"])
            quality = metadata.get("video_motion", {}).get("presentation_quality") or pr.video_presentation_quality(video_path)
            metadata.setdefault("video_motion", {})["presentation_quality"] = quality
            ok = _quality_ok(quality, args)
            deliverable = copy_deliverable(output_root, item.task, metadata)
            contact_sheet = write_contact_sheet(
                deliverable,
                output_root / "deliverables" / "platform_presentation_demos" / f"{item.task}_contact_sheet.jpg",
            )
            result["deliverable_video"] = str(deliverable)
            result["contact_sheet"] = str(contact_sheet)

        records.append(
            {
                **result,
                "metadata": metadata,
                "quality": quality,
                "ok": ok,
            }
        )

    index_path, guide_path = write_deliverable_docs(output_root, records)
    summary = {
        "index": str(index_path),
        "guide": str(guide_path),
        "ok": all(record.get("ok", False) for record in records),
        "tasks": records,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
