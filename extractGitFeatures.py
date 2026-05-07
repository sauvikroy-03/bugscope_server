from pydriller import Repository
from collections import defaultdict
import os


def extract_git_features(repo_path):
    """
    Extract git/process metrics per Python file:
    - number_of_developers
    - code_churn
    - commit_frequency
    """
    file_developers = defaultdict(set)
    file_churn = defaultdict(int)
    file_commits = defaultdict(int)

    for commit in Repository(repo_path).traverse_commits():
        author = commit.author.name if commit.author else "unknown"

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

    all_files = set(file_developers.keys()) | set(file_churn.keys()) | set(file_commits.keys())

    for file_path in all_files:
        git_features[file_path] = [
            len(file_developers[file_path]),   # number_of_developers
            file_churn[file_path],             # code_churn
            file_commits[file_path]            # commit_frequency
        ]

    return git_features