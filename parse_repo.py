import os
import ast
import json
import sys
import extractFeatures
import extractCodeFeatures
import extractGitFeatures
import predictBugs
repo_path = os.path.abspath(sys.argv[1])

nodes = []
edges = []

# ----------------------------
# Step 1: Collect Python files
# ----------------------------
python_files = []

for root, _, files in os.walk(repo_path):
    for file in files:
        if file.endswith(".py"):
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, repo_path)
            python_files.append(rel_path)

# ----------------------------
# Step 2: Build module map
# ----------------------------
# Example:
# server/models/trainModel.py → server.models.trainModel
module_map = {}

for path in python_files:
    module_name = path.replace(os.sep, ".").replace(".py", "")
    module_map[module_name] = path

# Reverse map for strict lookup
reverse_map = {v: k for k, v in module_map.items()}

# ----------------------------
# Step 3: Resolve import strictly
# ----------------------------
def resolve_import(import_name, current_module):
    """
    Try to resolve import strictly:
    - absolute imports
    - same-project imports
    """

    # Direct match
    if import_name in module_map:
        return module_map[import_name]

    # Try relative to project root
    for mod, path in module_map.items():
        if mod.endswith(import_name):
            return path

    return None

# ----------------------------
# Step 4: Parse AST
# ----------------------------
for rel_path in python_files:
    full_path = os.path.join(repo_path, rel_path)

    nodes.append({
        "id": rel_path,
        "label": os.path.basename(rel_path)
    })

    current_module = rel_path.replace(os.sep, ".").replace(".py", "")

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except:
        continue

    for node in ast.walk(tree):

        # import x
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported = alias.name

                resolved = resolve_import(imported, current_module)
                print("IMPORT:", imported, "->", resolved, "IN", rel_path,file=sys.stderr)
                if resolved:
                    edges.append({
                        "source": resolved,
                        "target": rel_path
                    })

        # from x import y
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported = node.module

                resolved = resolve_import(imported, current_module)

                if resolved:
                    edges.append({
                        "source": resolved,
                        "target": rel_path
                    })

# ----------------------------
# Step 5: Remove duplicates
# ----------------------------
edges = [dict(t) for t in {tuple(d.items()) for d in edges}]

# ----------------------------
# Output
# ----------------------------

graph = {
    "nodes": nodes,
    "edges": edges
}


codeFeatures=extractCodeFeatures.extract_code_features(repo_path)
gitFeatures=extractGitFeatures.extract_git_features(repo_path)
metrics = extractFeatures.build_graph_metrics(graph,codeFeatures,gitFeatures)

# print(metrics)
result={
    "nodes": nodes,
    "edges": edges,
    "metrics": metrics,
    "repo_root": repo_path
    
}

prediction = predictBugs.predict_with_gnn(result)
result["results"] = prediction

print(json.dumps(result))