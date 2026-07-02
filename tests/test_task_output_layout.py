from pathlib import Path
import unittest


TASK_FILES = [
    Path("difftactile/tasks/box_open.py"),
    Path("difftactile/tasks/surface_follow.py"),
    Path("difftactile/tasks/object_repose.py"),
    Path("difftactile/tasks/cable_straightening.py"),
]


class TaskOutputLayoutTests(unittest.TestCase):
    def test_patched_tasks_do_not_write_legacy_lr_or_results_dirs(self):
        for path in TASK_FILES:
            source = path.read_text()
            with self.subTest(path=str(path)):
                self.assertNotIn("legacy_dir", source)
                self.assertNotIn('os.mkdir(f"results")', source)
                self.assertNotIn("os.path.join(legacy", source)


if __name__ == "__main__":
    unittest.main()
