from typing import List


def personal_repo_url(owner: str, repo_name: str) -> str:
    return f"https://github.com/{owner}/{repo_name}.git"


def official_repo_url(owner: str, repo_name: str) -> str:
    return f"https://github.com/{owner}/{repo_name}.git"


def plan_personal_fork_commands(
    official_owner: str,
    repo_name: str,
    github_owner: str,
) -> List[str]:
    official_slug = f"{official_owner}/{repo_name}"
    personal_slug = f"{github_owner}/{repo_name}"
    personal_url = personal_repo_url(github_owner, repo_name)
    upstream_url = official_repo_url(official_owner, repo_name)
    return [
        f"gh repo view {personal_slug} >/dev/null 2>&1 || gh repo fork {official_slug} --clone=false",
        "git remote get-url upstream >/dev/null 2>&1 || git remote rename origin upstream",
        f"git remote set-url upstream {upstream_url}",
        f"git remote get-url origin >/dev/null 2>&1 && git remote set-url origin {personal_url} || git remote add origin {personal_url}",
    ]
