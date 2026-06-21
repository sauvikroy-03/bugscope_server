# project_context.py
#
# Takes the output of parse_repo.py (nodes, edges, ...), reads every file,
# asks an LLM for a one-line summary of each, and assembles that into a
# compact context block — file summaries + the dependency graph + a
# ready-to-paste natural-language prompt — so you don't have to re-explain
# the project's structure to an LLM every single time.
#
# Can be used two ways:
#   1. Imported:  from project_context import build_project_context
#   2. Standalone: python project_context.py <graph_json_path> [plan] [repo_root]

from dotenv import load_dotenv
load_dotenv()

import os
import sys
import json
import requests
from glob import glob
from typing import Optional

from google import genai
from google.genai import types

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# How much of each file's source to feed the LLM for summarization.
# Keeps prompts small — we only need enough to describe the file's role,
# not its full implementation.
MAX_CHARS = 4000


# ── File reading ──────────────────────────────────────────────────────────────

def locate_absolute_path(relative_path: str, repo_root: Optional[str] = None) -> str:
    """Resolve a graph node id to an absolute path on disk.

    If repo_root is given, this is exact and unambiguous — just join it with
    the relative path. This is the preferred path.

    If repo_root is NOT given, falls back to the same trick as routes/llm.py's
    locate_absolute_path: glob-search the whole 'repos' directory for a file
    matching the trailing relative path. This is a best-effort fallback only —
    if more than one analyzed repo contains a file with the same name (e.g.
    'main.py', 'db.py'), this can silently match the WRONG file. Pass
    repo_root whenever you can to avoid that.
    """
    clean_rel = relative_path.replace("\\", "/")

    if repo_root:
        direct = os.path.join(repo_root, clean_rel.replace("/", os.sep))
        if os.path.exists(direct):
            return os.path.abspath(direct).replace("\\", "/")

    filename = clean_rel.split("/")[-1]

    search_pattern = os.path.join("repos", "**", filename)
    matches = glob(search_pattern, recursive=True)

    for match in matches:
        normalized_match = match.replace("\\", "/")
        if normalized_match.endswith(clean_rel):
            return os.path.abspath(normalized_match).replace("\\", "/")

    # Fallback — best-effort merge if nothing matched
    return os.path.abspath(os.path.join("repos", clean_rel)).replace("\\", "/")


def read_file_safely(absolute_path: str) -> str:
    try:
        with open(absolute_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()[:MAX_CHARS]
    except Exception:
        return ""


# ── LLM callers (same pattern as routes/llm.py) ───────────────────────────────

def analyze_with_gemini(prompt: str) -> str:
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=120
        )
    )
    return response.text or "No response from Gemini."


def analyze_with_ollama(prompt: str) -> str:
    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": "phi4-mini:latest",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 120}
        },
        timeout=120
    )
    return response.json().get("response", "No response from Ollama.")


# ── Per-file summarizer ───────────────────────────────────────────────────────

def _looks_like_echo(summary: str, content: str) -> bool:
    """Detect when the model echoed the prompt/source instead of answering —
    e.g. it starts repeating 'File:' or a code fence, or it's suspiciously
    long for what was supposed to be one sentence."""
    lowered = summary.strip().lower()
    if lowered.startswith("file:") or lowered.startswith("source code"):
        return True
    if "```" in summary:
        return True
    if len(summary) > 350:  # a real one-sentence answer shouldn't be this long
        return True
    # crude overlap check — if a decent chunk of the summary is verbatim from
    # the source, it's almost certainly an echo, not a summary
    if len(content) > 40 and content[:40].strip() and content[:40].strip() in summary:
        return True
    return False


def summarize_file(file_id: str, content: str, plan: str) -> str:
    if not content.strip():
        return "Could not read file contents."

    trimmed = content[:MAX_CHARS]

    prompt = f"""Answer in exactly ONE plain-text sentence. Do not output code. Do not repeat the file content. Do not include the words "File:" or "Source code" or any code fences.

Example:
File content shown below is from app.py.
One-sentence answer: This file is the Flask entrypoint that registers the /login and /signup blueprints and starts the dev server.

Now do the same for this file.

File content shown below is from {file_id}.
---
{trimmed}
---

Remember: ONE sentence only, plain text, no code, no repeating the content above.
One-sentence answer:"""

    try:
        if plan == "pro":
            result = analyze_with_gemini(prompt).strip()
        else:
            result = analyze_with_ollama(prompt).strip()
    except Exception as e:
        return f"Could not generate summary: {e}"

    if _looks_like_echo(result, trimmed):
        return "Could not generate a reliable summary for this file."

    return result


# ── Context prompt renderer ───────────────────────────────────────────────────

def render_context_prompt(all_files: dict, connected_files: list) -> str:
    lines = ["# Project Context", "", "## Files"]
    for file_id, data in all_files.items():
        lines.append(f"- `{file_id}`: {data['summary']}")

    if connected_files:
        lines += ["", "## Dependencies"]
        for edge in connected_files:
            lines.append(f"- `{edge['source']}` is imported by `{edge['target']}`")

    lines += [
        "",
        "Use the file summaries and dependency relationships above to understand "
        "this codebase's structure before answering questions or proposing changes. "
        "You do not need to re-read every file from scratch."
    ]
    return "\n".join(lines)


# ── Main entry point ──────────────────────────────────────────────────────────

def build_project_context(graph_result: dict, plan: str = "free", repo_root: Optional[str] = None) -> dict:
    """
    graph_result : the dict produced by parse_repo.py (must have "nodes" and "edges")
    plan         : "free" (Ollama) or "pro" (Gemini) — same convention as routes/llm.py
    repo_root    : OPTIONAL absolute path to the analyzed repo's root on disk.
                   Strongly recommended if you have it — guarantees correct
                   file resolution. Without it, paths are found via a glob
                   search across ALL of ./repos/, which can grab the wrong
                   file if multiple analyzed repos share a filename.

    Returns:
        {
          "all_files": { "<file_id>": { "summary": "..." }, ... },
          "connected_files": [ { "source": ..., "target": ... }, ... ],
          "context_prompt": "<ready-to-paste natural language context>"
        }
    """
    nodes = graph_result.get("nodes", [])
    edges = graph_result.get("edges", [])

    all_files = {}
    for node in nodes:
        file_id = node.get("id")
        if not file_id:
            continue

        abs_path = locate_absolute_path(file_id, repo_root)
        content = read_file_safely(abs_path)

        if not content.strip():
            print(f"  ⚠️ could not read {file_id} (tried: {abs_path})", file=sys.stderr)

        summary = summarize_file(file_id, content, plan)

        all_files[file_id] = {"summary": summary}
        print(f"  ✅ summarized {file_id}", file=sys.stderr)

    connected_files = [
        {"source": e.get("source"), "target": e.get("target")}
        for e in edges
    ]

    context_prompt = render_context_prompt(all_files, connected_files)

    return {
        "all_files": all_files,
        "connected_files": connected_files,
        "context_prompt": context_prompt,
    }


# ── Standalone CLI usage ──────────────────────────────────────────────────────

if __name__ == "__main__":
    # Usage: python project_context.py <graph_json_path> [plan] [repo_root]
    if len(sys.argv) < 2:
        print("Usage: python project_context.py <graph_json_path> [plan] [repo_root]", file=sys.stderr)
        sys.exit(1)

    graph_json_path = sys.argv[1]
    plan_arg: str = sys.argv[2] if len(sys.argv) > 2 else "free"
    repo_root_arg: Optional[str] = sys.argv[3] if len(sys.argv) > 3 else None

    with open(graph_json_path, "r", encoding="utf-8") as f:
        graph_result_data = json.load(f)

    context = build_project_context(graph_result_data, plan_arg, repo_root_arg)
    print(json.dumps(context, indent=2))