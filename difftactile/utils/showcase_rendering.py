import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np
import trimesh

from difftactile.utils.headless_recording import DEFAULT_OUTPUT_ROOT, ensure_dir
from difftactile.utils import platform_recording as pr


SHOWCASE_TASKS = ("box_open", "surface_follow", "object_repose", "cable_straightening")
STALE_NAME_MARKERS = (
    "smoke",
    "probe",
    "medium",
    "legacy",
    "synthetic",
    "platform_demo",
    "smooth_retry",
    "platform_presentation",
)


@dataclass(frozen=True)
class ShowcaseConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    duration_seconds: float = 16.0
    pyopengl_platform: str = "egl"
    min_duration_seconds: float = 10.0


@dataclass(frozen=True)
class ShowcaseSpec:
    task: str
    title: str
    output_name: str
    description: str


def build_showcase_specs(tasks: Optional[Sequence[str]] = None) -> List[ShowcaseSpec]:
    selected = list(tasks or SHOWCASE_TASKS)
    descriptions = {
        "box_open": "A tactile dome pushes a blue box body and opens the lid to show contact-driven manipulation.",
        "surface_follow": "A tactile sensor presses onto a curved surface and follows the height changes smoothly.",
        "object_repose": "A tactile dome pushes a rigid object toward a target pose with visible translation and rotation.",
        "cable_straightening": "A tactile gripper pulls a cable from a wavy shape into a straighter configuration.",
    }
    titles = {
        "box_open": "Box Open",
        "surface_follow": "Surface Follow",
        "object_repose": "Object Repose",
        "cable_straightening": "Cable Straightening",
    }
    specs = []
    for task in selected:
        if task not in SHOWCASE_TASKS:
            raise ValueError(f"Unsupported showcase task: {task}")
        specs.append(
            ShowcaseSpec(
                task=task,
                title=titles[task],
                output_name=f"{task}_showcase.mp4",
                description=descriptions[task],
            )
        )
    return specs


def showcase_output_dir(output_root: Path) -> Path:
    return Path(output_root) / "deliverables" / "showcase_3d"


def cleanup_previous_showcase_outputs(output_root: Path) -> List[Path]:
    output_root = Path(output_root)
    removed: List[Path] = []
    direct_dirs = [
        output_root / "deliverables" / "platform_demos",
        output_root / "deliverables" / "platform_presentation_demos",
        output_root / "deliverables" / "showcase_3d",
        output_root / "videos",
        output_root / "probes",
    ]
    for path in direct_dirs:
        if path.exists():
            shutil.rmtree(path)
            removed.append(path)

    runs_root = output_root / "runs"
    if runs_root.exists():
        for run_dir in sorted(path for path in runs_root.glob("*/*") if path.is_dir()):
            if _is_stale_name(run_dir.name):
                shutil.rmtree(run_dir)
                removed.append(run_dir)
        for task_dir in sorted(path for path in runs_root.glob("*") if path.is_dir()):
            try:
                next(task_dir.iterdir())
            except StopIteration:
                task_dir.rmdir()

    logs_root = output_root / "logs"
    if logs_root.exists():
        for log_path in sorted(logs_root.glob("*")):
            if log_path.is_file() and _is_stale_name(log_path.name):
                log_path.unlink()
                removed.append(log_path)
    return removed


def _is_stale_name(name: str) -> bool:
    lower = name.lower()
    return any(marker in lower for marker in STALE_NAME_MARKERS)


def write_showcase_readme(output_dir: Path, records: Sequence[Mapping]) -> Path:
    ensure_dir(output_dir)
    lines = [
        "# DiffTactile 3D Showcase Videos",
        "",
        "These videos are showcase render outputs generated from DiffTactile task semantics and local mesh assets.",
        "They are not copied from the official GIFs and are not the old black-background debug GUI recordings.",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"## {record['title']}",
                "",
                str(record["description"]),
                "",
                f"- Video: `{Path(record['video']).name}`",
                f"- Duration: `{float(record.get('duration_seconds', 0.0)):.2f}s`",
                "",
            ]
        )
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(lines).strip() + "\n")
    return readme_path


def validate_showcase_video(
    path: Path,
    min_duration_seconds: float = 10.0,
    changed_fraction_threshold: float = 0.12,
    mean_motion_threshold: float = 0.45,
    min_subject_fraction: float = 0.02,
) -> dict:
    path = Path(path)
    failures: List[str] = []
    duration = pr.video_duration_seconds(path)
    readable = pr.video_is_readable_and_nonblank(path)
    motion = pr.video_motion_score(path)
    subject_fraction = estimate_subject_fraction(path)
    if not readable:
        failures.append("unreadable or blank")
    if duration < min_duration_seconds:
        failures.append("too short")
    if not pr.video_motion_passes(
        motion,
        changed_fraction_threshold=changed_fraction_threshold,
        mean_absdiff_threshold=mean_motion_threshold,
    ):
        failures.append("low motion")
    if subject_fraction < min_subject_fraction:
        failures.append("low subject occupancy")
    return {
        "path": str(path),
        "ok": not failures,
        "failures": failures,
        "duration_seconds": duration,
        "readable_nonblank": readable,
        "motion": motion,
        "subject_fraction": subject_fraction,
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def estimate_subject_fraction(path: Path, sample_count: int = 6) -> float:
    if not Path(path).exists():
        return 0.0
    cap = cv2.VideoCapture(str(path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if not cap.isOpened() or frames <= 0:
        cap.release()
        return 0.0
    fractions = []
    for idx in np.linspace(0, frames - 1, num=min(sample_count, frames), dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        background = np.median(lab.reshape(-1, 3), axis=0)
        distance = np.linalg.norm(lab.astype(np.float32) - background.astype(np.float32), axis=2)
        fractions.append(float(np.mean(distance > 10.0)))
    cap.release()
    return 0.0 if not fractions else float(np.mean(fractions))


def truncate_text_to_width(
    text: str,
    max_width: int,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 0.48,
    thickness: int = 1,
) -> str:
    if cv2.getTextSize(text, font, scale, thickness)[0][0] <= max_width:
        return text
    suffix = "..."
    available = max(0, max_width - cv2.getTextSize(suffix, font, scale, thickness)[0][0])
    clipped = ""
    for char in text:
        candidate = clipped + char
        if cv2.getTextSize(candidate, font, scale, thickness)[0][0] > available:
            break
        clipped = candidate
    return clipped.rstrip() + suffix


def render_showcase_suite(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    repo_root: Optional[Path] = None,
    config: Optional[ShowcaseConfig] = None,
    tasks: Optional[Sequence[str]] = None,
    clean: bool = True,
) -> dict:
    output_root = Path(output_root)
    repo_root = Path(repo_root or Path(__file__).resolve().parents[2])
    config = config or ShowcaseConfig()
    if clean:
        cleanup_previous_showcase_outputs(output_root)
    output_dir = ensure_dir(showcase_output_dir(output_root))
    specs = build_showcase_specs(tasks)
    records = []
    for spec in specs:
        video_path = output_dir / spec.output_name
        render_showcase_task(spec, video_path, repo_root, config)
        validation = validate_showcase_video(video_path, min_duration_seconds=config.min_duration_seconds)
        records.append(
            {
                "task": spec.task,
                "title": spec.title,
                "description": spec.description,
                "video": str(video_path),
                "duration_seconds": validation["duration_seconds"],
                "validation": validation,
                "ok": validation["ok"],
            }
        )
    readme_path = write_showcase_readme(output_dir, records)
    summary = {
        "output_dir": str(output_dir),
        "readme": str(readme_path),
        "records": records,
        "ok": all(record["ok"] for record in records),
    }
    return summary


def render_showcase_task(spec: ShowcaseSpec, output_path: Path, repo_root: Path, config: ShowcaseConfig):
    os.environ.setdefault("PYOPENGL_PLATFORM", config.pyopengl_platform)
    pyrender = _import_pyrender()
    ensure_dir(output_path.parent)
    renderer = pyrender.OffscreenRenderer(viewport_width=config.width, viewport_height=config.height)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        config.fps,
        (config.width, config.height),
    )
    if not writer.isOpened():
        renderer.delete()
        raise RuntimeError(f"Could not open video writer for {output_path}")
    try:
        frame_count = max(2, int(round(config.duration_seconds * config.fps)))
        for frame_idx in range(frame_count):
            phase = frame_idx / max(frame_count - 1, 1)
            scene = _make_scene(pyrender)
            _add_camera_and_lights(scene, pyrender, phase)
            _TASK_RENDERERS[spec.task](scene, pyrender, repo_root, phase)
            color, _ = renderer.render(scene)
            frame = _draw_overlay(color, spec, phase)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        renderer.delete()


def _import_pyrender():
    import pyrender

    return pyrender


def _make_scene(pyrender):
    return pyrender.Scene(
        bg_color=np.array([0.94, 0.945, 0.94, 1.0]),
        ambient_light=np.array([0.42, 0.42, 0.42, 1.0]),
    )


def _add_camera_and_lights(scene, pyrender, phase: float):
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.2, aspectRatio=16.0 / 9.0)
    angle = -0.25 + 0.08 * math.sin(2.0 * math.pi * phase)
    eye = np.array([3.5 * math.sin(angle), -4.8, 2.6 + 0.15 * math.cos(2.0 * math.pi * phase)])
    target = np.array([0.0, 0.0, 0.55])
    scene.add(camera, pose=_look_at(eye, target))
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.2), pose=_look_at(np.array([-2.0, -3.0, 5.0]), target))
    scene.add(pyrender.PointLight(color=np.ones(3), intensity=18.0), pose=_translation_matrix([2.4, -2.2, 3.5]))
    table = trimesh.creation.box(extents=(5.0, 4.0, 0.04))
    table.apply_translation([0.0, 0.0, -0.04])
    _add_mesh(scene, pyrender, table, _mat(pyrender, [0.78, 0.80, 0.78, 1.0], roughness=0.85))


def _render_box_open(scene, pyrender, repo_root: Path, phase: float):
    progress = _pulse(phase)
    base = trimesh.creation.box(extents=(1.45, 0.95, 0.24))
    base.apply_translation([0.0, 0.0, 0.12])
    _add_mesh(scene, pyrender, base, _mat(pyrender, [0.12, 0.36, 0.84, 1.0]))

    cavity = trimesh.creation.box(extents=(1.14, 0.62, 0.025))
    cavity.apply_translation([0.0, -0.05, 0.265])
    _add_mesh(scene, pyrender, cavity, _mat(pyrender, [0.04, 0.12, 0.22, 1.0], roughness=0.7))

    lid = trimesh.creation.box(extents=(1.45, 0.82, 0.055))
    hinge_y = 0.49
    angle = math.radians(-4.0 - 82.0 * progress)
    transform = (
        _translation_matrix([0.0, hinge_y, 0.31])
        @ _rotation_matrix(angle, [1, 0, 0])
        @ _translation_matrix([0.0, -0.41, 0.0])
    )
    _add_mesh(scene, pyrender, lid, _mat(pyrender, [0.05, 0.56, 0.92, 1.0]), transform)
    hinge = trimesh.creation.cylinder(radius=0.035, height=1.55, sections=24)
    hinge.apply_transform(_rotation_matrix(math.pi / 2.0, [0, 1, 0]))
    _add_mesh(scene, pyrender, hinge, _mat(pyrender, [0.92, 0.20, 0.12, 1.0]), _translation_matrix([0.0, hinge_y, 0.31]))

    dome = trimesh.creation.uv_sphere(radius=0.24, count=[32, 16])
    dome.apply_scale([1.0, 1.0, 0.62])
    sensor_x = -1.16 + 0.82 * progress
    sensor_z = 0.58 + 0.17 * math.sin(math.pi * progress)
    _add_mesh(
        scene,
        pyrender,
        dome,
        _mat(pyrender, [1.0, 0.76, 0.16, 0.96], metallic=0.0, roughness=0.35),
        _translation_matrix([sensor_x, -0.08, sensor_z]),
    )
    contact = trimesh.creation.cylinder(radius=0.035, height=0.55, sections=24)
    contact.apply_transform(_rotation_matrix(math.pi / 2.0, [0, 1, 0]))
    _add_mesh(scene, pyrender, contact, _mat(pyrender, [1.0, 0.2, 0.1, 1.0]), _translation_matrix([-0.32, -0.08, 0.43]))


def _render_surface_follow(scene, pyrender, repo_root: Path, phase: float):
    surface = _wavy_surface_mesh()
    _add_mesh(scene, pyrender, surface, _mat(pyrender, [0.18, 0.47, 0.72, 1.0], roughness=0.75))
    x = -1.45 + 2.9 * phase
    y = 0.05 * math.sin(2.0 * math.pi * phase)
    z = 0.50 + 0.20 * math.sin(2.0 * math.pi * (phase * 1.15 + 0.1))
    dome = trimesh.creation.uv_sphere(radius=0.23, count=[32, 16])
    dome.apply_scale([1.0, 1.0, 0.66])
    _add_mesh(scene, pyrender, dome, _mat(pyrender, [1.0, 0.75, 0.13, 0.95]), _translation_matrix([x, y, z + 0.17]))
    marker = trimesh.creation.cylinder(radius=0.03, height=0.55, sections=20)
    marker.apply_transform(_rotation_matrix(math.pi / 2.0, [1, 0, 0]))
    _add_mesh(scene, pyrender, marker, _mat(pyrender, [0.95, 0.18, 0.12, 1.0]), _translation_matrix([x, y, z + 0.02]))


def _render_object_repose(scene, pyrender, repo_root: Path, phase: float):
    progress = _pulse(phase)
    target = trimesh.creation.box(extents=(0.95, 0.56, 0.42))
    _add_mesh(
        scene,
        pyrender,
        target,
        _mat(pyrender, [0.2, 0.8, 0.55, 0.28]),
        _translation_matrix([0.65, 0.16, 0.25]) @ _rotation_matrix(math.radians(42), [0, 0, 1]),
    )
    obj = _load_asset_mesh(repo_root / "difftactile" / "meshes" / "objects" / "block-10.stl", fallback="box")
    obj = _normalize_mesh(obj, target_extent=0.92)
    transform = (
        _translation_matrix([-0.55 + 1.2 * progress, -0.18 + 0.34 * progress, 0.32])
        @ _rotation_matrix(math.radians(-22.0 + 64.0 * progress), [0, 0, 1])
    )
    _add_mesh(scene, pyrender, obj, _mat(pyrender, [0.10, 0.38, 0.86, 1.0]), transform)
    dome = trimesh.creation.uv_sphere(radius=0.22, count=[32, 16])
    dome.apply_scale([1.0, 1.0, 0.7])
    _add_mesh(
        scene,
        pyrender,
        dome,
        _mat(pyrender, [1.0, 0.74, 0.12, 0.96]),
        _translation_matrix([-1.12 + 0.85 * progress, -0.34 + 0.22 * progress, 0.38]),
    )


def _render_cable_straightening(scene, pyrender, repo_root: Path, phase: float):
    progress = _pulse(phase)
    xs = np.linspace(-1.35, 1.35, 18)
    amplitude = 0.48 * (1.0 - 0.82 * progress)
    ys = amplitude * np.sin(np.linspace(0.0, 2.4 * np.pi, len(xs)) + 0.4 * math.sin(2.0 * math.pi * phase))
    points = np.column_stack([xs, ys, np.full_like(xs, 0.20)])
    cable = _tube_from_points(points, radius=0.055)
    _add_mesh(scene, pyrender, cable, _mat(pyrender, [0.10, 0.34, 0.82, 1.0], roughness=0.5))

    left_x = -1.35 + 0.55 * progress
    right_x = 1.35 - 0.55 * progress
    _add_gripper(scene, pyrender, left_x, ys[1], 0.40, side=-1)
    _add_gripper(scene, pyrender, right_x, ys[-2], 0.40, side=1)
    tension = trimesh.creation.cylinder(radius=0.018, height=max(0.2, right_x - left_x), sections=16)
    tension.apply_transform(_rotation_matrix(math.pi / 2.0, [0, 1, 0]))
    _add_mesh(scene, pyrender, tension, _mat(pyrender, [0.95, 0.18, 0.13, 1.0]), _translation_matrix([(left_x + right_x) / 2.0, 0.0, 0.45]))


def _add_gripper(scene, pyrender, x: float, y: float, z: float, side: int):
    finger_a = trimesh.creation.box(extents=(0.10, 0.38, 0.36))
    finger_b = trimesh.creation.box(extents=(0.10, 0.38, 0.36))
    spread = 0.18
    yaw = math.radians(8.0 * side)
    base_transform = _translation_matrix([x, y, z]) @ _rotation_matrix(yaw, [0, 0, 1])
    _add_mesh(
        scene,
        pyrender,
        finger_a,
        _mat(pyrender, [1.0, 0.72, 0.12, 0.96]),
        base_transform @ _translation_matrix([0.0, spread, 0.0]),
    )
    _add_mesh(
        scene,
        pyrender,
        finger_b,
        _mat(pyrender, [1.0, 0.47, 0.56, 0.96]),
        base_transform @ _translation_matrix([0.0, -spread, 0.0]),
    )


def _wavy_surface_mesh() -> trimesh.Trimesh:
    xs = np.linspace(-1.65, 1.65, 40)
    ys = np.linspace(-0.9, 0.9, 24)
    vertices = []
    for y in ys:
        for x in xs:
            z = 0.08 + 0.18 * math.sin(2.2 * x) * math.cos(2.4 * y) + 0.08 * math.sin(3.0 * y)
            vertices.append([x, y, z])
    faces = []
    cols = len(xs)
    rows = len(ys)
    for row in range(rows - 1):
        for col in range(cols - 1):
            a = row * cols + col
            b = a + 1
            c = a + cols
            d = c + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    return trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)


def _tube_from_points(points: np.ndarray, radius: float) -> trimesh.Trimesh:
    segments = []
    for start, end in zip(points[:-1], points[1:]):
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length <= 1e-6:
            continue
        cylinder = trimesh.creation.cylinder(radius=radius, height=length, sections=18)
        transform = trimesh.geometry.align_vectors([0, 0, 1], direction / length)
        transform[:3, 3] = (start + end) / 2.0
        cylinder.apply_transform(transform)
        segments.append(cylinder)
    if not segments:
        return trimesh.creation.uv_sphere(radius=radius)
    return trimesh.util.concatenate(segments)


def _load_asset_mesh(path: Path, fallback: str) -> trimesh.Trimesh:
    try:
        loaded = trimesh.load(path, force="mesh")
        if isinstance(loaded, trimesh.Trimesh) and len(loaded.vertices):
            return loaded.copy()
    except Exception:
        pass
    if fallback == "sphere":
        return trimesh.creation.uv_sphere(radius=0.5)
    return trimesh.creation.box(extents=(1.0, 0.7, 0.45))


def _normalize_mesh(mesh: trimesh.Trimesh, target_extent: float) -> trimesh.Trimesh:
    mesh = mesh.copy()
    bounds = mesh.bounds
    center = 0.5 * (bounds[0] + bounds[1])
    extents = bounds[1] - bounds[0]
    scale = target_extent / max(float(np.max(extents)), 1e-6)
    mesh.apply_translation(-center)
    mesh.apply_scale(scale)
    return mesh


def _add_mesh(scene, pyrender, mesh: trimesh.Trimesh, material, pose: Optional[np.ndarray] = None):
    render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
    scene.add(render_mesh, pose=pose if pose is not None else np.eye(4))


def _mat(pyrender, color, metallic: float = 0.0, roughness: float = 0.55):
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=np.asarray(color, dtype=np.float32),
        metallicFactor=metallic,
        roughnessFactor=roughness,
    )


def _draw_overlay(rgb: np.ndarray, spec: ShowcaseSpec, phase: float) -> np.ndarray:
    frame = np.asarray(rgb).copy()
    progress = int(round(100.0 * _pulse(phase)))
    cv2.rectangle(frame, (24, 24), (760, 118), (246, 247, 242), -1)
    cv2.rectangle(frame, (24, 24), (760, 118), (38, 45, 52), 2)
    cv2.putText(frame, spec.title, (46, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (24, 34, 44), 2, cv2.LINE_AA)
    description = truncate_text_to_width(spec.description, max_width=690)
    cv2.putText(frame, description, (46, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 52, 63), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (46, 105), (46 + int(180 * progress / 100.0), 111), (35, 118, 210), -1)
    cv2.putText(frame, f"motion {progress:02d}%", (238, 113), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 52, 63), 1, cv2.LINE_AA)
    cv2.circle(frame, (1030, 44), 10, (30, 90, 210), -1)
    cv2.putText(frame, "object/cable", (1048, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 52, 63), 1, cv2.LINE_AA)
    cv2.circle(frame, (1030, 74), 10, (255, 190, 40), -1)
    cv2.putText(frame, "tactile sensor", (1048, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 52, 63), 1, cv2.LINE_AA)
    return frame


def _pulse(phase: float) -> float:
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * phase)


def _translation_matrix(vector: Sequence[float]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = np.asarray(vector, dtype=np.float64)
    return matrix


def _rotation_matrix(angle: float, axis: Sequence[float]) -> np.ndarray:
    axis_arr = np.asarray(axis, dtype=np.float64)
    axis_arr = axis_arr / max(float(np.linalg.norm(axis_arr)), 1e-12)
    return trimesh.transformations.rotation_matrix(angle, axis_arr)


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    forward = target - eye
    forward = forward / max(float(np.linalg.norm(forward)), 1e-12)
    right = np.cross(forward, up)
    right = right / max(float(np.linalg.norm(right)), 1e-12)
    true_up = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


_TASK_RENDERERS: Dict[str, Callable] = {
    "box_open": _render_box_open,
    "surface_follow": _render_surface_follow,
    "object_repose": _render_object_repose,
    "cable_straightening": _render_cable_straightening,
}
