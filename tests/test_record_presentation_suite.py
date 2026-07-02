import tempfile
import unittest
from pathlib import Path

import scripts.record_presentation_suite as presentation_suite


class RecordPresentationSuiteTests(unittest.TestCase):
    def test_presentation_plan_maps_tasks_to_devices_and_replay_runs(self):
        plan = presentation_suite.build_presentation_plan(["2", "3"], profile="replay")
        self.assertEqual([item.task for item in plan], presentation_suite.PRESENTATION_TASKS)
        self.assertEqual([item.cuda_device for item in plan], ["2", "3", "2", "3"])
        self.assertEqual([item.profile for item in plan], ["replay"] * 4)
        self.assertEqual(plan[0].run_name, "box_open_platform_presentation")
        self.assertIn("--demo_mode", plan[0].extra_task_args)
        self.assertIn("replay", plan[0].extra_task_args)
        self.assertIn("--pyopengl_platform", plan[1].extra_recorder_args)
        self.assertIn("--disable_3d_window", plan[3].extra_task_args)

    def test_command_for_item_uses_compact_capture_defaults(self):
        args = presentation_suite.parse_args(["--output_root", "/tmp/out", "--devices", "2,3"])
        item = presentation_suite.build_presentation_plan(args.devices)[0]
        cmd = presentation_suite.command_for_item(Path("/repo"), args, item)
        self.assertIn("--width", cmd)
        self.assertIn("1280", cmd)
        self.assertIn("--height", cmd)
        self.assertIn("720", cmd)
        self.assertIn("--profile", cmd)
        self.assertIn("replay", cmd)
        self.assertIn("--demo_overlay", cmd)

    def test_demo_guide_and_index_are_written_for_deliverables(self):
        records = [
            {
                "task": "box_open",
                "run_name": "box_open_platform_presentation",
                "cuda_device": "2",
                "deliverable_video": "/tmp/box.mp4",
                "contact_sheet": "/tmp/box_contact.jpg",
                "metadata": {
                    "video_duration_seconds": 48.0,
                    "video_motion": {
                        "presentation_quality": {
                            "active_fraction": 0.72,
                            "longest_static_run_seconds": 2,
                        }
                    },
                },
                "ok": True,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            index_path, guide_path = presentation_suite.write_deliverable_docs(output_root, records)
            self.assertTrue(index_path.exists())
            self.assertTrue(guide_path.exists())
            guide = guide_path.read_text()
            self.assertIn("Box Open", guide)
            self.assertIn("tactile dome", guide)
            self.assertIn("active_fraction", guide)


if __name__ == "__main__":
    unittest.main()
