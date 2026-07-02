import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np


DEFAULT_OUTPUT_ROOT = Path("/data1/determined/users/thomas/Dataset/Difftactile/output")
DEFAULT_RECORD_STRIDE = 5
DEFAULT_MAX_VIDEO_FRAMES = 300
SMOKE_SUB_STEPS = 10
SMOKE_TOTAL_STEPS = 20
SMOKE_OPT_STEPS = 1


@dataclass(frozen=True)
class RunLayout:
    output_root: Path
    task_name: str
    run_id: str
    root: Path
    videos: Path
    plots: Path
    trajectories: Path
    frames: Path
    logs: Path
    mirror_videos: Path


@dataclass(frozen=True)
class RunConfig:
    num_sub_steps: int
    num_total_steps: int
    num_opt_steps: int
    record_stride: int


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_run_id(task_name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{task_name}_{stamp}"


def make_run_dir(output_root, task_name: str, run_name: Optional[str] = None) -> RunLayout:
    output_root = ensure_dir(output_root)
    run_id = run_name or timestamp_run_id(task_name)
    root = output_root / "runs" / task_name / run_id
    layout = RunLayout(
        output_root=output_root,
        task_name=task_name,
        run_id=run_id,
        root=ensure_dir(root),
        videos=root / "videos",
        plots=ensure_dir(root / "plots"),
        trajectories=ensure_dir(root / "trajectories"),
        frames=root / "frames",
        logs=ensure_dir(output_root / "logs"),
        mirror_videos=ensure_dir(output_root / "videos"),
    )
    ensure_dir(output_root / "runs")
    ensure_dir(output_root / "patches")
    ensure_dir(output_root / "notes")
    return layout


def save_json(path, data) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return path


def _env_flag(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def resolve_headless(cli_headless: bool = False, env: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if env is None else env
    env_value = _env_flag(env.get("DIFFTACTILE_HEADLESS"))
    if cli_headless or env_value is True:
        return True
    if env_value is False:
        return False
    return env.get("DISPLAY") in (None, "")


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(env.get(name, default))
    except (TypeError, ValueError):
        return default


def resolve_run_config(
    default_sub_steps: int,
    default_total_steps: int,
    default_opt_steps: int,
    smoke: bool = False,
    num_sub_steps: Optional[int] = None,
    num_total_steps: Optional[int] = None,
    num_opt_steps: Optional[int] = None,
    record_stride: Optional[int] = None,
    env: Optional[Mapping[str, str]] = None,
) -> RunConfig:
    env = os.environ if env is None else env
    if smoke:
        return RunConfig(
            num_sub_steps=SMOKE_SUB_STEPS,
            num_total_steps=SMOKE_TOTAL_STEPS,
            num_opt_steps=SMOKE_OPT_STEPS,
            record_stride=1,
        )
    return RunConfig(
        num_sub_steps=num_sub_steps or default_sub_steps,
        num_total_steps=num_total_steps or default_total_steps,
        num_opt_steps=num_opt_steps or default_opt_steps,
        record_stride=record_stride or _env_int(env, "DT_RECORD_STRIDE", DEFAULT_RECORD_STRIDE),
    )


def max_video_frames(env: Optional[Mapping[str, str]] = None) -> int:
    env = os.environ if env is None else env
    return _env_int(env, "DT_MAX_VIDEO_FRAMES", DEFAULT_MAX_VIDEO_FRAMES)


class VideoRecorder:
    def __init__(
        self,
        path,
        fps: int = 30,
        frame_size: Sequence[int] = (960, 720),
        mirror_paths: Optional[Iterable[Path]] = None,
        codec: str = "mp4v",
        fallback_codec: str = "XVID",
    ):
        self.requested_path = Path(path)
        self.path = self.requested_path
        self.fps = fps
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.mirror_paths = [Path(p) for p in (mirror_paths or [])]
        self.codec = codec
        self.fallback_codec = fallback_codec
        self.frames_written = 0
        ensure_dir(self.path.parent)
        self._writer = self._open_writer(self.path, self.codec)
        if not self._writer.isOpened():
            self._writer.release()
            self.path = self.path.with_suffix(".avi")
            self._writer = self._open_writer(self.path, self.fallback_codec)
        if not self._writer.isOpened():
            self._writer.release()
            raise RuntimeError(f"Unable to open video writer for {self.path}")

    def _open_writer(self, path: Path, codec: str):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        return cv2.VideoWriter(str(path), fourcc, self.fps, self.frame_size)

    def write_frame(self, rgb_frame: np.ndarray):
        frame = np.asarray(rgb_frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Video frames must be HxWx3 RGB arrays")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        expected_w, expected_h = self.frame_size
        if frame.shape[1] != expected_w or frame.shape[0] != expected_h:
            frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
        self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.frames_written += 1

    def close(self) -> Path:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self.frames_written == 0:
            raise RuntimeError(f"No frames were written to {self.path}")
        for mirror_path in self.mirror_paths:
            target = mirror_path
            if self.path.suffix.lower() == ".avi" and mirror_path.suffix.lower() != ".avi":
                target = mirror_path.with_suffix(".avi")
            ensure_dir(target.parent)
            shutil.copy2(self.path, target)
        return self.path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        return False


def write_frame_from_rgb_array(recorder: VideoRecorder, rgb_frame: np.ndarray):
    recorder.write_frame(rgb_frame)


def _as_points(points) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return points.reshape(0, 2)
    return points[:, :2]


def draw_scatter_frame(width: int, height: int, points, colors=None, title: Optional[str] = None) -> np.ndarray:
    frame = np.full((height, width, 3), 245, dtype=np.uint8)
    points = _as_points(points)
    if colors is None:
        colors = [(3, 157, 252)] * len(points)
    if len(points):
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        if len(points):
            mins = points.min(axis=0)
            maxs = points.max(axis=0)
            span = np.maximum(maxs - mins, 1e-6)
            normalized = (points - mins) / span
            pixels = np.column_stack(
                (
                    40 + normalized[:, 0] * max(width - 80, 1),
                    height - (40 + normalized[:, 1] * max(height - 80, 1)),
                )
            ).astype(np.int32)
            for idx, pixel in enumerate(pixels):
                color = colors[idx % len(colors)]
                cv2.circle(frame, tuple(pixel), 3, tuple(int(c) for c in color), -1)
    if title:
        cv2.putText(frame, title, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    return frame


def draw_loss_frame(losses, width: int = 960, height: int = 540) -> np.ndarray:
    frame = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(frame, (60, 40), (width - 30, height - 50), (30, 30, 30), 1)
    cv2.putText(frame, "Loss", (60, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    values = np.asarray(list(losses), dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return frame
    if values.size == 1:
        values = np.repeat(values, 2)
    min_v = float(values.min())
    max_v = float(values.max())
    span = max(max_v - min_v, 1e-12)
    xs = np.linspace(70, width - 40, values.size)
    ys = (height - 60) - ((values - min_v) / span) * (height - 120)
    polyline = np.column_stack((xs, ys)).astype(np.int32)
    cv2.polylines(frame, [polyline], False, (10, 120, 210), 2)
    cv2.putText(frame, f"last={values[-1]:.6g}", (70, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
    return frame


def draw_task_frame(
    contact_model,
    task_name: str,
    opt_step: int,
    timestep: int,
    loss: Optional[float] = None,
    width: int = 960,
    height: int = 720,
) -> np.ndarray:
    points = []
    colors = []
    try:
        contact_model.draw_perspective(0)
        if hasattr(contact_model, "draw_pos3"):
            points.append(contact_model.draw_pos3.to_numpy())
            colors.extend([(3, 157, 252)])
        if hasattr(contact_model, "draw_pos"):
            points.append(contact_model.draw_pos.to_numpy())
            colors.extend([(3, 157, 252)])
        if hasattr(contact_model, "draw_pos2"):
            points.append(contact_model.draw_pos2.to_numpy())
            colors.extend([(230, 201, 73)])
        if hasattr(contact_model, "draw_pos1"):
            points.append(contact_model.draw_pos1.to_numpy())
            colors.extend([(245, 66, 161)])
    except Exception:
        points = []
    merged = np.vstack([_as_points(p) for p in points if np.asarray(p).size]) if points else np.empty((0, 2))
    frame = draw_scatter_frame(width, height, merged, colors=colors or None, title=f"{task_name} opt={opt_step} ts={timestep}")
    if loss is not None:
        cv2.putText(frame, f"loss={loss:.6g}", (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    return frame


def validate_videos(paths: Iterable[Path]):
    results = []
    for path in paths:
        path = Path(path)
        cap = cv2.VideoCapture(str(path))
        opened, frame = cap.read()
        cap.release()
        results.append(
            {
                "path": str(path),
                "opened": bool(opened and frame is not None),
                "shape": None if frame is None else list(frame.shape),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return results


def scan_npy_for_nan(root) -> dict:
    root = Path(root)
    checked = 0
    nan_files = []
    for path in sorted(root.rglob("*.npy")):
        checked += 1
        data = np.load(path, allow_pickle=False)
        if np.issubdtype(data.dtype, np.number) and np.isnan(data).any():
            nan_files.append(str(path))
    return {"checked": checked, "nan_files": nan_files}
