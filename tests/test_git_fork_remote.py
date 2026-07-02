import unittest

from difftactile.utils import git_fork_remote as gr


class GitForkRemoteTests(unittest.TestCase):
    def test_origin_target_uses_authenticated_owner_repo(self):
        target = gr.personal_repo_url("XCZYYY", "DiffTactile")
        self.assertEqual(target, "https://github.com/XCZYYY/DiffTactile.git")

    def test_dry_run_commands_create_or_reuse_fork_and_preserve_upstream(self):
        commands = gr.plan_personal_fork_commands(
            official_owner="Genesis-Embodied-AI",
            repo_name="DiffTactile",
            github_owner="XCZYYY",
        )
        joined = "\n".join(commands)
        self.assertIn("gh repo view XCZYYY/DiffTactile", joined)
        self.assertIn("gh repo fork Genesis-Embodied-AI/DiffTactile --clone=false", joined)
        self.assertIn("git remote add origin https://github.com/XCZYYY/DiffTactile.git", joined)
        self.assertIn("git remote set-url upstream https://github.com/Genesis-Embodied-AI/DiffTactile.git", joined)


if __name__ == "__main__":
    unittest.main()
