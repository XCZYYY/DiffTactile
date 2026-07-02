import os
import tempfile
import unittest
from pathlib import Path

from difftactile.utils import platform_recording as pr


class PlatformRecordingTests(unittest.TestCase):
    def test_long_profile_expands_to_required_box_open_steps(self):
        config = pr.resolve_profile("box_open", "long", [])
        self.assertEqual(config.task, "box_open")
        self.assertEqual(config.num_sub_steps, 20)
        self.assertEqual(config.num_total_steps, 300)
        self.assertEqual(config.num_opt_steps, 10)
        self.assertEqual(config.min_duration_seconds, 600)

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
        )
        self.assertEqual(cmd[:5], ["ffmpeg", "-y", "-hide_banner", "-loglevel", "info"])
        self.assertIn("-f", cmd)
        self.assertIn("x11grab", cmd)
        self.assertIn("-video_size", cmd)
        self.assertIn("1920x1080", cmd)
        self.assertIn(":101.0", cmd)
        self.assertIn("libx264", cmd)
        self.assertIn("yuv420p", cmd)

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
