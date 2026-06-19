import os
import ast

import sys


# if len(sys.argv) < 2:
#     print("Usage: python script.py <repo_path>")
#     sys.exit(1)

# repo_path = os.path.abspath(sys.argv[1])

# pattern_for_def=r'^(?![ \t]*#)[ \t]*def\s+'
# pattern_for_class=r'^(?![ \t]*#)[ \t]*class\s+'
# pattern_for_if=r'^(?![ \t]*#)[ \t]*(?:if|elif)\s+'
# pattern_for_loops = r'^(?![ \t]*#)[ \t]*(?:for|while)\s+'

# Function to count effective lines of code (excluding blanks and comments)
def count_effective_lines(lines):
    count = 0
    in_multiline_string = False

    for line in lines:
        stripped = line.strip()

        # skip blank lines
        if not stripped:
            continue

        # handle triple-quoted comment/docstring blocks
        if stripped.startswith(("'''", '"""')):
            in_multiline_string = not in_multiline_string
            continue

        if in_multiline_string:
            continue

        # skip single-line comments
        if stripped.startswith("#"):
            continue

        count += 1

    return count

#Count Comment lines for calculating comment Density

def count_comment_lines(lines):
    comment_count = 0
    in_multiline_comment = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        # multiline comments/docstrings
        if stripped.startswith(("'''", '"""')):
            comment_count += 1

            if not (
                stripped.endswith(("'''", '"""'))
                and len(stripped) > 3
            ):
                in_multiline_comment = not in_multiline_comment

            continue

        if in_multiline_comment:
            comment_count += 1
            continue

        # single-line comments
        if stripped.startswith("#"):
            comment_count += 1

    return comment_count

def extract_code_features(repo_path):
    code_features = {}

    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, repo_path)

                functions = 0
                classes = 0
                if_else = 0
                loops = 0
                loc = 0
                cyclomatic_complexity = 1
                total_function_lines=0
              

                with open(full_path, 'r', encoding='utf-8') as f:
                    try:
                        content = f.read()
                        all_lines = content.splitlines()
                        # Count LOC-----------------------
                        loc = count_effective_lines(all_lines)
                        #--------------------------------------

                        # Count Comment Density-------------------
                        total_lines = len(all_lines)
                        comment_lines = count_comment_lines(all_lines)

                        comment_density = (round(comment_lines / total_lines, 4)
                                           if total_lines > 0 else 0)

                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                functions += 1
                                # Calculate lines of code in the function
                                start_line = node.lineno
                                end_line = getattr(node, 'end_lineno', start_line)
                                function_lines = all_lines[start_line - 1:end_line]
                                effective_func_lines = count_effective_lines(function_lines)

                                total_function_lines += effective_func_lines
                             
                            elif isinstance(node, ast.ClassDef):
                                classes += 1
                              
                            elif isinstance(node, ast.For):
                                loops += 1
                                cyclomatic_complexity += 1
                            elif isinstance(node, ast.While):
                                loops += 1
                                cyclomatic_complexity += 1
                            elif isinstance(node, ast.If):
                                if_else += 1
                                cyclomatic_complexity += 1
                            elif isinstance(node, ast.ExceptHandler):
                                cyclomatic_complexity += 1
                            elif isinstance(node, ast.BoolOp):
                                cyclomatic_complexity += 1
                            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                                cyclomatic_complexity += 1
                        avg_func_len = total_function_lines / functions if functions > 0 else 0


                        code_features[rel_path] = [loc, functions, classes, loops, if_else,cyclomatic_complexity,round(avg_func_len, 2),comment_density]

                    except Exception as e:
                        print("Error processing", full_path, ":", e)

    return code_features
