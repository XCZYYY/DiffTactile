import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_FILES = [
    "box_open.py",
    "surface_follow.py",
    "object_repose.py",
    "cable_straightening.py",
]


class PresentationTaskInterfaceTests(unittest.TestCase):
    def test_four_tasks_expose_replay_demo_interface(self):
        for filename in TASK_FILES:
            source = (REPO_ROOT / "difftactile" / "tasks" / filename).read_text()
            with self.subTest(task=filename):
                self.assertIn("--demo_mode", source)
                self.assertIn("--ready_file", source)
                self.assertIn("--replay_loops", source)
                self.assertIn("--replay_speed", source)
                self.assertIn("--demo_overlay", source)
                self.assertIn("run_replay_demo", source)
                self.assertIn("mark_ready", source)


if __name__ == "__main__":
    unittest.main()
