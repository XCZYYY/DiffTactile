import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import scripts.record_platform_run as record_platform_run
import scripts.validate_outputs as validate_outputs
from difftactile.utils import platform_recording as pr


class PlatformRecordingTests(unittest.TestCase):
    def test_long_profile_expands_to_required_box_open_steps(self):
        config = pr.resolve_profile("box_open", "long", [])
        self.assertEqual(config.task, "box_open")
        self.assertEqual(config.num_sub_steps, 20)
        self.assertEqual(config.num_total_steps, 300)
        self.assertEqual(config.num_opt_steps, 10)
        self.assertEqual(config.min_duration_seconds, 600)

    def test_demo_profile_expands_to_smoother_short_demo_steps(self):
        config = pr.resolve_profile("box_open", "demo", [])
        self.assertEqual(config.task, "box_open")
        self.assertEqual(config.num_sub_steps, 8)
        self.assertEqual(config.num_total_steps, 90)
        self.assertEqual(config.num_opt_steps, 2)
        self.assertEqual(config.min_duration_seconds, 60)

    def test_replay_profile_expands_to_presentation_defaults(self):
        config = pr.resolve_profile("box_open", "replay", [])
        self.assertEqual(config.task, "box_open")
        self.assertEqual(config.num_sub_steps, 6)
        self.assertEqual(config.num_total_steps, 90)
        self.assertEqual(config.num_opt_steps, 1)
        self.assertEqual(config.min_duration_seconds, 45)
        self.assertEqual(config.gui_refresh_stride, 1)

        args = pr.task_passthrough_args(config, ["--use_state"])
        self.assertIn("--demo_mode", args)
        self.assertIn("replay", args)
        self.assertIn("--replay_loops", args)
        self.assertIn("--demo_overlay", args)

    def test_explicit_passthrough_overrides_profile_steps(self):
        config = pr.resolve_profile(
            "box_open",
            "long",
            ["--num_total_steps", "123", "--num_opt_steps", "4"],
        )
        args = pr.task_passthrough_args(config, ["--use_state"])
        self.assertIn("--num_sub_steps", args)
        self.assertIn("20", args)
        self.assertIn("--num_total_steps", args)
        self.assertIn("123", args)
        self.assertIn("--num_opt_steps", args)
        self.assertIn("4", args)
        self.assertIn("--use_state", args)

    def test_ffmpeg_command_records_x11_display_with_h264(self):
        cmd = pr.build_ffmpeg_command(
            display=":101",
            output_path=Path("/tmp/out.mp4"),
            width=1920,
            height=1080,
            fps=30,
            loglevel="info",
            preset="ultrafast",
        )
        self.assertEqual(cmd[:5], ["ffmpeg", "-y", "-hide_banner", "-loglevel", "info"])
        self.assertIn("-f", cmd)
        self.assertIn("x11grab", cmd)
        self.assertIn("-video_size", cmd)
        self.assertIn("1920x1080", cmd)
        self.assertIn(":101.0", cmd)
        self.assertIn("libx264", cmd)
        self.assertIn("ultrafast", cmd)
        self.assertIn("yuv420p", cmd)

    def test_ffmpeg_command_can_wait_for_ready_file(self):
        cmd = pr.build_ffmpeg_command(
            display=":101",
            output_path=Path("/tmp/out.mp4"),
            width=1280,
            height=720,
            fps=30,
            ready_file=Path("/tmp/ready.flag"),
        )
        self.assertEqual(cmd[:2], ["bash", "-lc"])
        shell_script = cmd[2]
        self.assertIn("/tmp/ready.flag", shell_script)
        self.assertIn("x11grab", shell_script)
        self.assertIn("1280x720", shell_script)

    def test_record_platform_run_accepts_explicit_cuda_device(self):
        args = record_platform_run.parse_args(
            [
                "--task",
                "box_open",
                "--profile",
                "demo",
                "--cuda_device",
                "2",
                "--pyopengl_platform",
                "egl",
                "--",
                "--use_state",
            ]
        )
        self.assertEqual(args.cuda_device, "2")
        self.assertEqual(args.pyopengl_platform, "egl")
        self.assertEqual(args.passthrough, ["--use_state"])

    def test_existing_display_is_reused_when_requested(self):
        with pr.display_context(
            width=800,
            height=600,
            use_existing=True,
            env={"DISPLAY": ":7"},
        ) as display:
            self.assertEqual(display.display, ":7")
            self.assertFalse(display.started_virtual_display)

    def test_inline_record_video_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "platform screen recorder"):
            pr.reject_inline_record_video(True, "box_open")
        pr.reject_inline_record_video(False, "box_open")
        pr.reject_inline_record_video(None, "box_open")

    def test_prepare_run_layout_adds_screen_recording_dir_and_mirror_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = pr.prepare_platform_run_layout(tmp, "box_open", "unit_platform")
            self.assertTrue(run.screen_recordings.is_dir())
            self.assertEqual(
                run.primary_video,
                Path(tmp) / "runs" / "box_open" / "unit_platform" / "screen_recordings" / "box_open_platform.mp4",
            )
            self.assertEqual(
                run.mirror_video,
                Path(tmp) / "videos" / "unit_platform_box_open_platform.mp4",
            )

    def test_motion_score_detects_changed_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "moving.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10,
                (64, 48),
            )
            for idx in range(12):
                frame = np.zeros((48, 64, 3), dtype=np.uint8)
                frame[:, idx : idx + 10, 1] = 255
                writer.write(frame)
            writer.release()

            result = pr.video_motion_score(video_path, sample_count=8)
            self.assertGreater(result["changed_fraction"], 0.5)
            self.assertGreater(result["mean_absdiff"], 1.0)

    def test_motion_score_rejects_static_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "static.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10,
                (64, 48),
            )
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[:, :, 2] = 128
            for _ in range(12):
                writer.write(frame)
            writer.release()

            result = pr.video_motion_score(video_path, sample_count=8)
            self.assertEqual(result["changed_fraction"], 0.0)
            self.assertLess(result["mean_absdiff"], 1.0)

    def test_presentation_quality_detects_active_motion_and_static_runs(self):
        diffs = [0.0, 0.8, 0.9, 0.7, 0.0, 0.6, 0.7, 0.8, 0.0, 0.9]
        quality = pr.motion_quality_from_diffs(diffs, active_threshold=0.5)
        self.assertAlmostEqual(quality["active_fraction"], 0.7)
        self.assertEqual(quality["longest_static_run_seconds"], 1)
        self.assertTrue(
            pr.presentation_motion_passes(
                quality,
                active_fraction_threshold=0.6,
                max_static_run_seconds=5,
            )
        )

        static_quality = pr.motion_quality_from_diffs([0.0, 0.0, 0.7, 0.0, 0.0, 0.0], active_threshold=0.5)
        self.assertFalse(
            pr.presentation_motion_passes(
                static_quality,
                active_fraction_threshold=0.6,
                max_static_run_seconds=2,
            )
        )

    def test_motion_passes_on_fraction_or_mean_difference(self):
        self.assertTrue(
            pr.video_motion_passes(
                {"changed_fraction": 0.08, "mean_absdiff": 0.86},
                changed_fraction_threshold=0.15,
                mean_absdiff_threshold=0.5,
            )
        )
        self.assertFalse(
            pr.video_motion_passes(
                {"changed_fraction": 0.0, "mean_absdiff": 0.01},
                changed_fraction_threshold=0.15,
                mean_absdiff_threshold=0.5,
            )
        )

    def test_validate_platform_video_includes_motion_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "moving.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10,
                (64, 48),
            )
            for idx in range(12):
                frame = np.zeros((48, 64, 3), dtype=np.uint8)
                frame[:, idx : idx + 10, 0] = 255
                writer.write(frame)
            writer.release()

            result = validate_outputs.validate_platform_video(video_path)
            self.assertIn("motion", result)
            self.assertGreater(result["motion"]["changed_fraction"], 0.5)

    def test_task_env_keeps_display_and_removes_headless_flags(self):
        env = pr.task_environment(
            os.environ,
            display=":105",
            cuda_visible_devices="2",
        )
        self.assertEqual(env["DISPLAY"], ":105")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")
        self.assertNotIn("DIFFTACTILE_HEADLESS", env)
        self.assertNotIn("PYOPENGL_PLATFORM", env)

    def test_conda_xvfb_bin_dir_uses_sysroot_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)
            bin_dir = prefix / "x86_64-conda-linux-gnu" / "sysroot" / "usr" / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "Xvfb").write_text("")
            self.assertEqual(pr.conda_xvfb_bin_dir(str(prefix)), bin_dir)

    def test_xvfb_compat_lib_dir_uses_openssl10_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)
            lib_dir = prefix / "lib"
            lib_dir.mkdir(parents=True)
            (lib_dir / "libcrypto.so.10").write_text("")
            self.assertEqual(pr.xvfb_compat_lib_dir(str(prefix)), lib_dir)


if __name__ == "__main__":
    unittest.main()
