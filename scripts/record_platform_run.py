#!/usr/bin/env python
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from difftactile.utils.headless_recording import DEFAULT_OUTPUT_ROOT
from difftactile.utils import platform_recording as pr


def split_argv(argv):
    if "--" not in argv:
        return argv, []
    idx = argv.index("--")
    return argv[:idx], argv[idx + 1 :]


def parse_args(argv):
    main_argv, passthrough = split_argv(argv)
    parser = argparse.ArgumentParser(description="Record a real DiffTactile platform GUI run with ffmpeg x11grab.")
    parser.add_argument("--task", choices=sorted(pr.TASK_SCRIPTS), required=True)
    parser.add_argument("--profile", choices=sorted(pr.PROFILE_STEPS), default="smoke")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--use_existing_display", action="store_true")
    parser.add_argument("--ffmpeg_loglevel", default="info")
    parser.add_argument("--no_auto_extend", action="store_true")
    args = parser.parse_args(main_argv)
    args.passthrough = passthrough
    return args


def run_once(args, profile_name, run_name, passthrough, repo_root: Path):
    config = pr.resolve_profile(args.task, profile_name, passthrough)
    output_root = Path(args.output_root)
    layout = pr.prepare_platform_run_layout(output_root, args.task, run_name)
    cuda_device = pr.choose_visible_gpu()
    started_at = time.time()

    with pr.display_context(args.width, args.height, args.use_existing_display) as display:
        task_env = pr.task_environment(os.environ, display.display, cuda_visible_devices=cuda_device)
        window_manager = pr.start_window_manager(display.display, task_env)
        task_command = pr.build_task_command(
            repo_root=repo_root,
            task=args.task,
            config=config,
            output_root=output_root,
            run_name=layout.run_id,
            extra_args=passthrough,
        )
        ffmpeg_command = pr.build_ffmpeg_command(
            display=display.display,
            output_path=layout.primary_video,
            width=args.width,
            height=args.height,
            fps=args.fps,
            loglevel=args.ffmpeg_loglevel,
        )
        pr.write_metadata(
            layout.metadata_path,
            {
                "task": args.task,
                "run_id": layout.run_id,
                "profile": profile_name,
                "platform_screen_capture": True,
                "display": display.display,
                "started_virtual_display": display.started_virtual_display,
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
                "cuda_visible_devices": cuda_device,
                "task_command": task_command,
                "ffmpeg_command": ffmpeg_command,
                "num_sub_steps": config.num_sub_steps,
                "num_total_steps": config.num_total_steps,
                "num_opt_steps": config.num_opt_steps,
            },
        )

        ffmpeg = pr.start_ffmpeg(ffmpeg_command, layout.ffmpeg_log)
        with open(layout.task_log, "wb") as log_file:
            task = subprocess.Popen(
                task_command,
                cwd=repo_root / "difftactile" / "tasks",
                env=task_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            task_return = task.wait()
        pr.stop_ffmpeg(ffmpeg)
        if window_manager is not None and window_manager.poll() is None:
            window_manager.terminate()
            try:
                window_manager.wait(timeout=5)
            except subprocess.TimeoutExpired:
                window_manager.kill()
                window_manager.wait(timeout=5)

    duration = pr.video_duration_seconds(layout.primary_video)
    readable = pr.video_is_readable_and_nonblank(layout.primary_video)
    mirrored_video = None
    if layout.primary_video.exists():
        pr.mirror_video(layout.primary_video, layout.mirror_video)
        mirrored_video = str(layout.mirror_video)
    pr.write_metadata(
        layout.metadata_path,
        {
            "task": args.task,
            "run_id": layout.run_id,
            "profile": profile_name,
            "platform_screen_capture": True,
            "elapsed_seconds": time.time() - started_at,
            "video_duration_seconds": duration,
            "video_readable_nonblank": readable,
            "video_path": str(layout.primary_video),
            "mirror_video": mirrored_video,
            "task_log": str(layout.task_log),
            "ffmpeg_log": str(layout.ffmpeg_log),
            "task_returncode": task_return,
            "num_sub_steps": config.num_sub_steps,
            "num_total_steps": config.num_total_steps,
            "num_opt_steps": config.num_opt_steps,
        },
    )
    if task_return != 0:
        raise RuntimeError(f"task exited with code {task_return}; see {layout.task_log}")
    if not readable:
        raise RuntimeError(f"recorded video is unreadable or blank: {layout.primary_video}")
    return layout, duration, config


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    repo_root = Path(__file__).resolve().parents[1]
    pr.validate_task_passthrough(args.passthrough)

    run_name = args.run_name or f"{args.task}_platform_{args.profile}"
    layout, duration, config = run_once(args, args.profile, run_name, args.passthrough, repo_root)
    if (
        args.profile == "long"
        and not args.no_auto_extend
        and config.min_duration_seconds
        and duration < config.min_duration_seconds
    ):
        extended_name = f"{run_name}_extended"
        layout, duration, _ = run_once(args, "long_extended", extended_name, args.passthrough, repo_root)

    print(f"Platform recording saved: {layout.primary_video}")
    print(f"Mirrored video saved: {layout.mirror_video}")
    print(f"Duration seconds: {duration:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
