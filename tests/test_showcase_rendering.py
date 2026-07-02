import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from difftactile.utils import showcase_rendering as sr


class ShowcaseRenderingTests(unittest.TestCase):
    def test_specs_cover_four_tasks_and_final_file_names(self):
        specs = sr.build_showcase_specs()

        self.assertEqual(
            [spec.task for spec in specs],
            ["box_open", "surface_follow", "object_repose", "cable_straightening"],
        )
        self.assertEqual(
            [spec.output_name for spec in specs],
            [
                "box_open_showcase.mp4",
                "surface_follow_showcase.mp4",
                "object_repose_showcase.mp4",
                "cable_straightening_showcase.mp4",
            ],
        )
        self.assertTrue(all(spec.title for spec in specs))
        self.assertTrue(all("tactile" in spec.description.lower() for spec in specs))

    def test_cleanup_removes_old_demo_outputs_without_touching_notes_or_patches(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            stale_dirs = [
                output_root / "deliverables" / "platform_demos",
                output_root / "deliverables" / "platform_presentation_demos",
                output_root / "runs" / "box_open" / "box_open_platform_demo",
                output_root / "runs" / "box_open" / "box_open_platform_demo_smooth_retry",
                output_root / "runs" / "box_open" / "box_open_platform_presentation",
                output_root / "videos",
            ]
            keep_dirs = [
                output_root / "notes",
                output_root / "patches",
                output_root / "runs" / "box_open" / "keep_experiment",
            ]
            for path in stale_dirs + keep_dirs:
                path.mkdir(parents=True)
                (path / "marker.txt").write_text("x")

            removed = sr.cleanup_previous_showcase_outputs(output_root)

            self.assertEqual({path.name for path in removed}, {path.name for path in stale_dirs})
            for path in stale_dirs:
                self.assertFalse(path.exists(), path)
            for path in keep_dirs:
                self.assertTrue(path.exists(), path)

    def test_readme_is_minimal_and_labels_videos_as_showcase_renders(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            records = [
                {
                    "task": "box_open",
                    "title": "Box Open",
                    "video": str(output_dir / "box_open_showcase.mp4"),
                    "description": "A tactile dome pushes a blue box open.",
                    "duration_seconds": 18.0,
                }
            ]

            readme = sr.write_showcase_readme(output_dir, records)

            text = readme.read_text()
            self.assertIn("showcase render", text)
            self.assertIn("not copied from the official GIFs", text)
            self.assertIn("Box Open", text)
            self.assertLess(len(text.splitlines()), 80)

    def test_validate_showcase_video_requires_motion_and_nonblank_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            moving = Path(tmp) / "moving.mp4"
            static = Path(tmp) / "static.mp4"
            self._write_test_video(moving, moving=True)
            self._write_test_video(static, moving=False)

            moving_result = sr.validate_showcase_video(moving, min_duration_seconds=0.5)
            static_result = sr.validate_showcase_video(static, min_duration_seconds=0.5)

            self.assertTrue(moving_result["ok"], moving_result)
            self.assertFalse(static_result["ok"], static_result)
            self.assertIn("low motion", static_result["failures"])

    def test_overlay_text_is_truncated_to_fit_available_width(self):
        text = "A tactile dome pushes a blue box body and opens the lid to show contact-driven manipulation."

        truncated = sr.truncate_text_to_width(
            text,
            max_width=260,
            font=cv2.FONT_HERSHEY_SIMPLEX,
            scale=0.48,
            thickness=1,
        )

        width = cv2.getTextSize(truncated, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
        self.assertLessEqual(width, 260)
        self.assertTrue(truncated.endswith("..."))

    def _write_test_video(self, path: Path, moving: bool):
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10,
            (96, 64),
        )
        for idx in range(12):
            frame = np.full((64, 96, 3), 235, dtype=np.uint8)
            x0 = 8 + (idx * 4 if moving else 0)
            cv2.circle(frame, (x0, 32), 10, (30, 90, 220), -1)
            writer.write(frame)
        writer.release()


if __name__ == "__main__":
    unittest.main()
