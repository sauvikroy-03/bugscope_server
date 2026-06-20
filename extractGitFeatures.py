from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pydriller import Repository


def extract_git_features(repo_path, months=12):
    """
    Returns:
        git_features:
        {
            "file.py": [
                num_developers,
                code_churn,
                commit_frequency
            ]
        }
    """

    file_developers = defaultdict(set)
    file_churn = defaultdict(int)
    file_commits = defaultdict(int)

    since_date = datetime.now(timezone.utc) - timedelta(days=months * 30)

    for commit in Repository(repo_path, since=since_date).traverse_commits():
        author = commit.author.email if commit.author else "unknown"

        for modified_file in commit.modified_files:
            path = modified_file.new_path or modified_file.old_path

            if not path:
                continue

            if not path.endswith(".py"):
                continue

            path = path.replace("\\", "/")

            added = modified_file.added_lines or 0
            deleted = modified_file.deleted_lines or 0

            file_developers[path].add(author)
            file_churn[path] += added + deleted
            file_commits[path] += 1

    git_features = {}

    all_files = (
        set(file_developers.keys())
        | set(file_churn.keys())
        | set(file_commits.keys())
    )

    for file_path in all_files:
        total_commits = file_commits[file_path]
        commit_frequency = total_commits / months if months > 0 else 0

        git_features[file_path] = [
            len(file_developers[file_path]),
            file_churn[file_path],
            round(commit_frequency, 4),
        ]

        # print(
        #     f"file: {file_path} | "
        #     f"developers: {len(file_developers[file_path])} | "
        #     f"churn: {file_churn[file_path]} | "
        #     f"commits: {total_commits} | "
        #     f"commit_frequency: {commit_frequency:.4f}"
        # )

    return git_features