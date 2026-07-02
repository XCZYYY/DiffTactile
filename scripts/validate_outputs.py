import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_OUTPUT_ROOT = Path("/data1/determined/users/thomas/Dataset/Difftactile/output")


def validate_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    return {
        "path": str(path),
        "opened": bool(ok and frame is not None),
        "shape": None if frame is None else list(frame.shape),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def scan_npy(root: Path) -> dict:
    checked = 0
    nan_files = []
    unreadable = []
    for path in sorted(root.rglob("*.npy")):
        checked += 1
        try:
            data = np.load(path, allow_pickle=False)
        except Exception as exc:
            unreadable.append({"path": str(path), "error": str(exc)})
            continue
        if np.issubdtype(data.dtype, np.number) and np.isnan(data).any():
            nan_files.append(str(path))
    return {"checked": checked, "nan_files": nan_files, "unreadable": unreadable}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    output_root = Path(args.output_root)
    run_root = output_root / "runs"
    run_dirs = [path for path in sorted(run_root.glob("*/*")) if path.is_dir()] if run_root.exists() else []
    completed_runs = [
        path for path in run_dirs
        if any((path / "plots").glob("*.png")) and any((path / "trajectories").glob("*.npy"))
    ]
    run_videos = []
    for path in completed_runs:
        run_videos.extend(sorted((path / "videos").glob("*.mp4")))
        run_videos.extend(sorted((path / "videos").glob("*.avi")))
    mirror_videos = sorted((output_root / "videos").glob("*.mp4")) + sorted((output_root / "videos").glob("*.avi")) if (output_root / "videos").exists() else []
    videos = run_videos + mirror_videos
    all_videos = sorted(output_root.rglob("*.mp4")) + sorted(output_root.rglob("*.avi"))
    ignored_videos = [str(path) for path in all_videos if path not in set(videos)]
    plots = sorted((output_root / "runs").rglob("plots/*.png")) if (output_root / "runs").exists() else []
    logs = sorted((output_root / "logs").glob("*.log")) if (output_root / "logs").exists() else []
    video_results = [validate_video(path) for path in videos]
    npy_result = scan_npy(output_root / "runs") if (output_root / "runs").exists() else {"checked": 0, "nan_files": [], "unreadable": []}
    failures = []
    if not videos:
        failures.append("no videos found")
    if any(not item["opened"] or item["size_bytes"] <= 1000 for item in video_results):
        failures.append("one or more videos are unreadable or too small")
    if not plots:
        failures.append("no loss plots found")
    if not logs:
        failures.append("no logs found")
    if npy_result["nan_files"]:
        failures.append("NaN found in one or more npy files")
    if npy_result["unreadable"]:
        failures.append("one or more npy files could not be read")

    summary = {
        "output_root": str(output_root),
        "videos": video_results,
        "ignored_incomplete_videos": ignored_videos,
        "completed_runs": [str(path) for path in completed_runs],
        "plots": [str(path) for path in plots],
        "logs": [str(path) for path in logs],
        "npy": npy_result,
        "ok": not failures,
        "failures": failures,
    }
    notes_dir = output_root / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    summary_path = notes_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
