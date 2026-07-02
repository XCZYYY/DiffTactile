import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from difftactile.utils import headless_recording as hr


class HeadlessRecordingTests(unittest.TestCase):
    def test_make_run_dir_creates_canonical_layout_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = hr.make_run_dir(tmp, "box_open", "unit_run")
            self.assertEqual(run.task_name, "box_open")
            self.assertEqual(run.run_id, "unit_run")
            self.assertTrue(run.root.is_dir())
            self.assertTrue(run.videos.is_dir())
            self.assertTrue(run.plots.is_dir())
            self.assertTrue(run.trajectories.is_dir())
            self.assertTrue(run.frames.is_dir())
            self.assertTrue((Path(tmp) / "videos").is_dir())

            metadata_path = hr.save_json(run.root / "metadata.json", {"task": "box_open"})
            self.assertTrue(metadata_path.is_file())
            self.assertIn('"task": "box_open"', metadata_path.read_text())

    def test_resolve_headless_uses_cli_env_and_display(self):
        self.assertTrue(hr.resolve_headless(cli_headless=True, env={"DISPLAY": ":0"}))
        self.assertTrue(
            hr.resolve_headless(
                cli_headless=False,
                env={"DIFFTACTILE_HEADLESS": "1", "DISPLAY": ":0"},
            )
        )
        self.assertTrue(hr.resolve_headless(cli_headless=False, env={"DISPLAY": ""}))
        self.assertTrue(hr.resolve_headless(cli_headless=False, env={}))
        self.assertFalse(hr.resolve_headless(cli_headless=False, env={"DISPLAY": ":0"}))

    def test_resolve_run_config_applies_smoke_defaults_and_env_stride(self):
        config = hr.resolve_run_config(
            default_sub_steps=50,
            default_total_steps=600,
            default_opt_steps=100,
            smoke=True,
            num_sub_steps=None,
            num_total_steps=None,
            num_opt_steps=None,
            record_stride=None,
            env={"DT_RECORD_STRIDE": "7"},
        )
        self.assertEqual(config.num_sub_steps, 10)
        self.assertEqual(config.num_total_steps, 20)
        self.assertEqual(config.num_opt_steps, 1)
        self.assertEqual(config.record_stride, 1)

        config = hr.resolve_run_config(
            default_sub_steps=50,
            default_total_steps=600,
            default_opt_steps=100,
            smoke=False,
            num_sub_steps=20,
            num_total_steps=100,
            num_opt_steps=3,
            record_stride=None,
            env={"DT_RECORD_STRIDE": "7"},
        )
        self.assertEqual(config.num_sub_steps, 20)
        self.assertEqual(config.num_total_steps, 100)
        self.assertEqual(config.num_opt_steps, 3)
        self.assertEqual(config.record_stride, 7)

    def test_video_recorder_writes_readable_video_and_mirror_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "primary.mp4"
            mirror_path = tmp_path / "mirror.mp4"
            recorder = hr.VideoRecorder(
                video_path,
                fps=5,
                frame_size=(96, 64),
                mirror_paths=[mirror_path],
            )
            frame = np.zeros((64, 96, 3), dtype=np.uint8)
            frame[:, :, 1] = 180
            recorder.write_frame(frame)
            recorder.write_frame(255 - frame)
            result_path = recorder.close()

            self.assertTrue(result_path.exists())
            self.assertTrue(mirror_path.exists())
            validation = hr.validate_videos([result_path, mirror_path])
            self.assertEqual(len(validation), 2)
            self.assertTrue(all(item["opened"] for item in validation))
            self.assertTrue(all(item["size_bytes"] > 1000 for item in validation))

            cap = cv2.VideoCapture(str(result_path))
            ok, decoded = cap.read()
            cap.release()
            self.assertTrue(ok)
            self.assertEqual(decoded.shape[:2], (64, 96))

    def test_draw_helpers_return_rgb_frames(self):
        scatter = hr.draw_scatter_frame(
            160,
            120,
            np.array([[0.1, 0.1], [0.9, 0.9]], dtype=np.float32),
            colors=[(255, 0, 0), (0, 255, 0)],
            title="frame",
        )
        losses = hr.draw_loss_frame([3.0, 2.0, 1.0], width=160, height=120)
        self.assertEqual(scatter.shape, (120, 160, 3))
        self.assertEqual(losses.shape, (120, 160, 3))
        self.assertEqual(scatter.dtype, np.uint8)
        self.assertEqual(losses.dtype, np.uint8)

    def test_scan_npy_for_nan_reports_bad_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            good = tmp_path / "good.npy"
            bad = tmp_path / "bad.npy"
            np.save(good, np.ones((2, 2), dtype=np.float32))
            np.save(bad, np.array([1.0, np.nan], dtype=np.float32))
            result = hr.scan_npy_for_nan(tmp_path)
            self.assertEqual(result["checked"], 2)
            self.assertEqual(result["nan_files"], [str(bad)])


if __name__ == "__main__":
    unittest.main()
