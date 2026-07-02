import unittest

import scripts.record_demo_suite as demo_suite


class RecordDemoSuiteTests(unittest.TestCase):
    def test_parse_devices_requires_two_devices(self):
        self.assertEqual(demo_suite.parse_devices("2,3"), ["2", "3"])
        with self.assertRaisesRegex(ValueError, "at least two"):
            demo_suite.parse_devices("2")

    def test_demo_plan_maps_tasks_to_devices_in_order(self):
        plan = demo_suite.build_demo_plan(["2", "3"], profile="demo")
        self.assertEqual([item.task for item in plan], demo_suite.DEMO_TASKS)
        self.assertEqual([item.cuda_device for item in plan], ["2", "3", "2", "3"])
        self.assertEqual([item.run_name for item in plan], [
            "box_open_platform_demo",
            "surface_follow_platform_demo",
            "object_repose_platform_demo",
            "cable_straightening_platform_demo",
        ])
        self.assertEqual(plan[1].extra_recorder_args, ["--pyopengl_platform", "egl"])
        self.assertEqual(plan[3].extra_task_args, ["--disable_3d_window"])

    def test_low_motion_retry_uses_smoother_retry_config(self):
        retry = demo_suite.retry_plan_item("box_open", "2")
        self.assertEqual(retry.task, "box_open")
        self.assertEqual(retry.cuda_device, "2")
        self.assertEqual(retry.profile, "demo_retry")
        self.assertEqual(retry.extra_task_args, ["--gui_refresh_stride", "1"])
        self.assertTrue(retry.run_name.endswith("_smooth_retry"))

    def test_fast_tasks_retry_extends_total_steps_for_minimum_duration(self):
        for task in ["surface_follow", "object_repose"]:
            retry = demo_suite.retry_plan_item(task, "2")
            with self.subTest(task=task):
                self.assertIn("--num_total_steps", retry.extra_task_args)
                self.assertIn("150", retry.extra_task_args)


if __name__ == "__main__":
    unittest.main()
