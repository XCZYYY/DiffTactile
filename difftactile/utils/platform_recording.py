import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

import cv2
import numpy as np

from difftactile.utils.headless_recording import DEFAULT_OUTPUT_ROOT, ensure_dir, make_run_dir


TASK_SCRIPTS = {
    "box_open": "box_open.py",
    "surface_follow": "surface_follow.py",
    "object_repose": "object_repose.py",
    "cable_straightening": "cable_straightening.py",
}

PROFILE_STEPS = {
    "smoke": (10, 20, 1, 0, None),
    "demo": (8, 90, 2, 60, 2),
    "demo_retry": (6, 120, 2, 60, 1),
    "long": (20, 300, 10, 600, None),
    "long_extended": (20, 600, 20, 600, None),
}

FORBIDDEN_TASK_FLAGS = {"--headless", "--record_video"}
DEFAULT_XVFB_COMPAT_PREFIX = DEFAULT_OUTPUT_ROOT.parent / "runtime" / "openssl10"


@dataclass(frozen=True)
class TaskProfile:
    task: str
    profile: str
    num_sub_steps: int
    num_total_steps: int
    num_opt_steps: int
    min_duration_seconds: int
    gui_refresh_stride: Optional[int] = None


@dataclass(frozen=True)
class ActiveDisplay:
    display: str
    started_virtual_display: bool


@dataclass(frozen=True)
class PlatformRunLayout:
    output_root: Path
    task_name: str
    run_id: str
    root: Path
    screen_recordings: Path
    primary_video: Path
    mirror_video: Path
    task_log: Path
    ffmpeg_log: Path
    metadata_path: Path


def resolve_profile(task: str, profile: str, passthrough_args: Sequence[str]) -> TaskProfile:
    if task not in TASK_SCRIPTS:
        raise ValueError(f"Unsupported task: {task}")
    if profile not in PROFILE_STEPS:
        raise ValueError(f"Unsupported profile: {profile}")
    sub_steps, total_steps, opt_steps, min_duration, gui_refresh_stride = PROFILE_STEPS[profile]
    return TaskProfile(
        task=task,
        profile=profile,
        num_sub_steps=_arg_int(passthrough_args, "--num_sub_steps", sub_steps),
        num_total_steps=_arg_int(passthrough_args, "--num_total_steps", total_steps),
        num_opt_steps=_arg_int(passthrough_args, "--num_opt_steps", opt_steps),
        min_duration_seconds=min_duration,
        gui_refresh_stride=_arg_int(passthrough_args, "--gui_refresh_stride", gui_refresh_stride)
        if gui_refresh_stride is not None
        else None,
    )


def _arg_int(args: Sequence[str], name: str, default: int) -> int:
    for idx, item in enumerate(args):
        if item == name and idx + 1 < len(args):
            return int(args[idx + 1])
        if item.startswith(name + "="):
            return int(item.split("=", 1)[1])
    return default


def _has_arg(args: Sequence[str], name: str) -> bool:
    return any(item == name or item.startswith(name + "=") for item in args)


def task_passthrough_args(config: TaskProfile, extra_args: Sequence[str]) -> List[str]:
    args = list(extra_args)
    for name, value in [
        ("--num_sub_steps", config.num_sub_steps),
        ("--num_total_steps", config.num_total_steps),
        ("--num_opt_steps", config.num_opt_steps),
    ]:
        if not _has_arg(args, name):
            args = [name, str(value)] + args
    if config.gui_refresh_stride is not None and not _has_arg(args, "--gui_refresh_stride"):
        args = ["--gui_refresh_stride", str(config.gui_refresh_stride)] + args
    return args


def validate_task_passthrough(extra_args: Sequence[str]):
    for item in extra_args:
        flag = item.split("=", 1)[0]
        if flag in FORBIDDEN_TASK_FLAGS:
            raise RuntimeError(f"{flag} is not allowed for platform recording; the recorder controls display and video capture")


def reject_inline_record_video(record_video: Optional[bool], task_name: str):
    if record_video:
        raise RuntimeError(
            f"{task_name} no longer supports inline synthetic --record_video output. "
            "Use scripts/record_platform_run.py to capture the platform screen recorder."
        )


def conda_xvfb_bin_dir(prefix: Optional[str] = None) -> Optional[Path]:
    prefix = prefix or os.environ.get("CONDA_PREFIX")
    if not prefix:
        return None
    candidate = Path(prefix) / "x86_64-conda-linux-gnu" / "sysroot" / "usr" / "bin"
    if (candidate / "Xvfb").exists():
        return candidate
    return None


def xvfb_compat_lib_dir(prefix: Optional[str] = None) -> Optional[Path]:
    prefix = prefix or os.environ.get("DIFFTACTILE_XVFB_COMPAT_PREFIX") or str(DEFAULT_XVFB_COMPAT_PREFIX)
    candidate = Path(prefix) / "lib"
    if (candidate / "libcrypto.so.10").exists():
        return candidate
    return None


def prepare_platform_run_layout(output_root, task_name: str, run_name: Optional[str] = None) -> PlatformRunLayout:
    base = make_run_dir(output_root, task_name, run_name)
    screen_recordings = ensure_dir(base.root / "screen_recordings")
    return PlatformRunLayout(
        output_root=base.output_root,
        task_name=task_name,
        run_id=base.run_id,
        root=base.root,
        screen_recordings=screen_recordings,
        primary_video=screen_recordings / f"{task_name}_platform.mp4",
        mirror_video=base.mirror_videos / f"{base.run_id}_{task_name}_platform.mp4",
        task_log=base.logs / f"{base.run_id}_{task_name}_platform.log",
        ffmpeg_log=screen_recordings / "ffmpeg.log",
        metadata_path=base.root / "platform_recording_metadata.json",
    )


def build_ffmpeg_command(
    display: str,
    output_path: Path,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    loglevel: str = "info",
    preset: str = "ultrafast",
) -> List[str]:
    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        loglevel,
        "-f",
        "x11grab",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        x11grab_input(display),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]


def x11grab_input(display: str) -> str:
    if "." in display.rsplit(":", 1)[-1]:
        return display
    return f"{display}.0"


@contextmanager
def display_context(
    width: int,
    height: int,
    use_existing: bool = False,
    env: Optional[Mapping[str, str]] = None,
):
    env = os.environ if env is None else env
    existing = env.get("DISPLAY")
    if use_existing and existing:
        yield ActiveDisplay(display=existing, started_virtual_display=False)
        return

    old_path = os.environ.get("PATH", "")
    old_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    xvfb_dir = conda_xvfb_bin_dir()
    compat_lib_dir = xvfb_compat_lib_dir()
    if xvfb_dir is not None:
        os.environ["PATH"] = f"{xvfb_dir}{os.pathsep}{old_path}"
    if compat_lib_dir is not None:
        os.environ["LD_LIBRARY_PATH"] = f"{compat_lib_dir}{os.pathsep}{old_ld_library_path}"

    try:
        from pyvirtualdisplay import Display
    except ImportError as exc:
        os.environ["PATH"] = old_path
        os.environ["LD_LIBRARY_PATH"] = old_ld_library_path
        raise RuntimeError(
            "pyvirtualdisplay is required for platform recording. Install it in the difftactile env."
        ) from exc

    display = Display(visible=False, size=(width, height), color_depth=24)
    try:
        display.start()
        os.environ["PATH"] = old_path
        os.environ["LD_LIBRARY_PATH"] = old_ld_library_path
        yield ActiveDisplay(display=os.environ["DISPLAY"], started_virtual_display=True)
    finally:
        try:
            display.stop()
        finally:
            os.environ["PATH"] = old_path
            os.environ["LD_LIBRARY_PATH"] = old_ld_library_path


def task_environment(
    base_env: Mapping[str, str],
    display: str,
    cuda_visible_devices: Optional[str] = None,
    pyopengl_platform: Optional[str] = None,
) -> dict:
    env = dict(base_env)
    env["DISPLAY"] = display
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("DIFFTACTILE_HEADLESS", None)
    if pyopengl_platform:
        env["PYOPENGL_PLATFORM"] = pyopengl_platform
    else:
        env.pop("PYOPENGL_PLATFORM", None)
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    return env


def choose_visible_gpu() -> Optional[str]:
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return os.environ["CUDA_VISIBLE_DEVICES"]
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    candidates = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            candidates.append((int(parts[1]), parts[0]))
        except ValueError:
            continue
    if not candidates:
        return None
    return max(candidates)[1]


def build_task_command(
    repo_root: Path,
    task: str,
    config: TaskProfile,
    output_root: Path,
    run_name: str,
    extra_args: Sequence[str],
    python_executable: Optional[str] = None,
) -> List[str]:
    validate_task_passthrough(extra_args)
    task_args = task_passthrough_args(config, extra_args)
    return [
        python_executable or sys.executable,
        str(repo_root / "difftactile" / "tasks" / TASK_SCRIPTS[task]),
        "--output_root",
        str(output_root),
        "--run_name",
        run_name,
        *task_args,
    ]


def start_window_manager(display: str, env: Mapping[str, str]) -> Optional[subprocess.Popen]:
    metacity = shutil.which("metacity")
    if metacity is None:
        return None
    manager_env = dict(env)
    manager_env["DISPLAY"] = display
    return subprocess.Popen(
        [metacity, "--replace", "--sm-disable"],
        env=manager_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def start_ffmpeg(command: Sequence[str], log_path: Path) -> subprocess.Popen:
    ensure_dir(log_path.parent)
    log_file = open(log_path, "wb")
    process = subprocess.Popen(
        list(command),
        stdin=subprocess.PIPE,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    process._difftactile_log_file = log_file
    time.sleep(1.0)
    if process.poll() is not None:
        log_file.close()
        raise RuntimeError(f"ffmpeg exited before the task started; see {log_path}")
    return process


def stop_ffmpeg(process: subprocess.Popen, timeout: int = 20):
    if process.poll() is None and process.stdin is not None:
        try:
            process.stdin.write(b"q")
            process.stdin.flush()
        except BrokenPipeError:
            pass
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    log_file = getattr(process, "_difftactile_log_file", None)
    if log_file is not None:
        log_file.close()


def video_duration_seconds(path: Path) -> float:
    try:
        output = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            text=True,
        )
        return float(output.strip())
    except Exception:
        return 0.0


def video_is_readable_and_nonblank(path: Path, min_size_bytes: int = 1000) -> bool:
    if not path.exists() or path.stat().st_size <= min_size_bytes:
        return False
    cap = cv2.VideoCapture(str(path))
    variances = []
    for _ in range(5):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        variances.append(float(np.var(frame)))
    cap.release()
    return bool(variances) and max(variances) > 1.0


def video_motion_score(
    path: Path,
    sample_count: int = 24,
    diff_threshold: float = 1.0,
) -> dict:
    if not path.exists() or path.stat().st_size <= 1000:
        return {
            "path": str(path),
            "opened": False,
            "frames_sampled": 0,
            "changed_fraction": 0.0,
            "mean_absdiff": 0.0,
        }
    cap = cv2.VideoCapture(str(path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if not cap.isOpened() or frame_count <= 1:
        cap.release()
        return {
            "path": str(path),
            "opened": False,
            "frames_sampled": 0,
            "changed_fraction": 0.0,
            "mean_absdiff": 0.0,
        }

    indices = np.linspace(0, frame_count - 1, num=min(sample_count, frame_count), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
    cap.release()

    diffs = []
    for prev, cur in zip(frames, frames[1:]):
        diffs.append(float(np.mean(cv2.absdiff(prev, cur))))
    changed = [value for value in diffs if value > diff_threshold]
    return {
        "path": str(path),
        "opened": bool(frames),
        "frames_sampled": len(frames),
        "changed_fraction": 0.0 if not diffs else len(changed) / len(diffs),
        "mean_absdiff": 0.0 if not diffs else float(np.mean(diffs)),
    }


def video_motion_passes(
    motion: Mapping,
    changed_fraction_threshold: float = 0.15,
    mean_absdiff_threshold: float = 0.5,
) -> bool:
    return (
        float(motion.get("changed_fraction", 0.0)) >= changed_fraction_threshold
        or float(motion.get("mean_absdiff", 0.0)) >= mean_absdiff_threshold
    )


def write_metadata(path: Path, data: Mapping):
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def mirror_video(primary: Path, mirror: Path):
    ensure_dir(mirror.parent)
    shutil.copy2(primary, mirror)
