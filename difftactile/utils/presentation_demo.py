import time
from pathlib import Path

import numpy as np


TASK_OVERLAYS = {
    "box_open": (
        "Tactile dome pushes the blue box body to open/change its angle. "
        "Yellow is the sensor; right panes show tactile force and deformation."
    ),
    "surface_follow": (
        "Tactile dome presses onto a curved surface and slides while tracking "
        "contact force and marker deformation."
    ),
    "object_repose": (
        "Tactile dome pushes a rigid object toward a new pose. "
        "The tactile pane shows marker motion from contact."
    ),
    "cable_straightening": (
        "Tactile gripper manipulates a cable-like rope toward a straighter shape. "
        "Colored particles show cable, gripper, and sensor contacts."
    ),
}


def mark_ready(path):
    if not path:
        return
    ready_path = Path(path)
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    ready_path.write_text("ready\n")


def replay_frame_delay(replay_speed):
    speed = max(float(replay_speed or 1.0), 0.1)
    return 0.05 / speed


def maybe_sleep(seconds):
    if seconds > 0:
        time.sleep(seconds)


def draw_demo_overlay(gui, task, title, stage, step, total_steps, enabled=True):
    if not enabled:
        return
    safe_title = title or task.replace("_", " ").title()
    description = TASK_OVERLAYS.get(task, "")
    total = max(int(total_steps), 1)
    progress = max(0.0, min(1.0, float(step) / total))
    try:
        gui.text(safe_title, pos=(0.03, 0.95), font_size=22, color=0xFFFFFF)
        gui.text(stage, pos=(0.03, 0.90), font_size=16, color=0x36C5F0)
        gui.text(description[:92], pos=(0.03, 0.855), font_size=13, color=0xD8DEE9)
        gui.text("Blue=object/cable  Yellow/Pink=tactile sensor  Right=tactile maps", pos=(0.03, 0.815), font_size=13, color=0xE6C949)
        gui.line(begin=(0.03, 0.78), end=(0.97, 0.78), radius=2, color=0x30363D)
        gui.line(begin=(0.03, 0.78), end=(0.03 + 0.94 * progress, 0.78), radius=4, color=0x36C5F0)
    except Exception:
        # Overlay is explanatory only; never break the physical replay.
        return


def fit_points_to_view(point_arrays, center=(0.50, 0.42), target_span=0.58, fallback_scale=0.1, fallback_offset=(0.0, 0.0)):
    points = []
    for array in point_arrays:
        if array is None:
            continue
        arr = np.asarray(array)
        if arr.ndim != 2 or arr.shape[1] < 2 or arr.size == 0:
            continue
        arr = arr[:, :2]
        arr = arr[np.isfinite(arr).all(axis=1)]
        if len(arr):
            points.append(arr)
    if not points:
        return fallback_scale, list(fallback_offset)

    stacked = np.vstack(points)
    lo = np.min(stacked, axis=0)
    hi = np.max(stacked, axis=0)
    span = float(np.max(hi - lo))
    if not np.isfinite(span) or span < 1e-6:
        return fallback_scale, list(fallback_offset)
    scale = float(target_span) / span
    midpoint = 0.5 * (lo + hi)
    offset = np.asarray(center, dtype=np.float32) - scale * midpoint
    return scale, offset.tolist()


def apply_camera_sweep(contact_model, step, total_steps, base_phi=None, base_theta=None, phi_amplitude=18.0, theta_amplitude=8.0):
    if not hasattr(contact_model, "view_phi") or not hasattr(contact_model, "view_theta"):
        return
    total = max(int(total_steps or 1), 1)
    phase = 2.0 * np.pi * (float(step) % total) / total
    phi0 = float(contact_model.view_phi if base_phi is None else base_phi)
    theta0 = float(contact_model.view_theta if base_theta is None else base_theta)
    contact_model.view_phi = phi0 + phi_amplitude * np.sin(phase)
    contact_model.view_theta = theta0 + theta_amplitude * np.cos(phase)


def presentation_viewport(step, total_steps, center=(0.50, 0.40), span=0.58, x_amplitude=0.08, y_amplitude=0.04, zoom_amplitude=0.06):
    total = max(int(total_steps or 1), 1)
    phase = 2.0 * np.pi * (float(step) % total) / total
    cx = center[0] + x_amplitude * np.sin(phase)
    cy = center[1] + y_amplitude * np.cos(phase)
    target_span = span + zoom_amplitude * np.sin(phase + np.pi / 3.0)
    return (cx, cy), target_span


def _smooth_controls(total_steps, task):
    t = np.linspace(0.0, 1.0, total_steps, dtype=np.float32)
    phase = 2.0 * np.pi * t
    pos = np.zeros((total_steps, 3), dtype=np.float32)
    ori = np.zeros((total_steps, 3), dtype=np.float32)

    if task == "box_open":
        pos[:, 0] = 28.0 + 16.0 * np.sin(phase)
        pos[:, 1] = 22.0 * np.sin(np.pi * t)
        ori[:, 2] = 180.0 * np.sin(phase)
    elif task == "surface_follow":
        pos[:, 0] = -24.0 + 14.0 * np.sin(phase)
        pos[:, 1] = 10.0 * np.sin(np.pi * t)
        pos[:, 2] = 6.0 * np.sin(phase)
        ori[:, 2] = 90.0 * np.sin(phase)
    elif task == "object_repose":
        pos[:, 0] = -24.0 + 30.0 * np.sin(phase)
        pos[:, 1] = 10.0 * np.sin(np.pi * t)
        ori[:, 1] = 160.0 * np.sin(phase)
    else:
        pos[:, 0] = 20.0 * np.sin(phase)
        pos[:, 1] = 10.0 * np.sin(np.pi * t)
    return pos, ori


def apply_single_sensor_replay_controls(contact_model, task):
    pos, ori = _smooth_controls(contact_model.total_steps, task)
    contact_model.p_sensor1.from_numpy(pos)
    contact_model.o_sensor1.from_numpy(ori)
    return pos, ori


def apply_gripper_replay_controls(contact_model):
    total_steps = contact_model.total_steps
    t = np.linspace(0.0, 1.0, total_steps, dtype=np.float32)
    phase = 2.0 * np.pi * t
    pos = np.zeros((total_steps, 3), dtype=np.float32)
    ori = np.zeros((total_steps, 3), dtype=np.float32)
    width = np.zeros((total_steps,), dtype=np.float32)

    pos[:, 0] = 10.0 * np.sin(phase)
    pos[:, 1] = 2.0 * np.sin(np.pi * t)
    pos[:, 2] = 6.0 * np.cos(phase)
    ori[:, 2] = 80.0 * np.sin(phase)
    width[:] = 1.2 * np.sin(phase)

    contact_model.p_gripper.from_numpy(pos)
    contact_model.o_gripper.from_numpy(ori)
    contact_model.w_gripper.from_numpy(width)
    return pos, ori, width
