# code_summarizer.py

from dotenv import load_dotenv
load_dotenv()

import os
import ast
import json
import time
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, END
import requests

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

# ── Mistral (now the "pro" tier LLM) ──────────────────────────
from mistralai.client import Mistral

mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

# ── Gemini (kept, commented out, in case you switch back) ───
# from google import genai
# from google.genai import types
#
# gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ══════════════════════════════════════════════════════════════
# 1. CONFIG
# ══════════════════════════════════════════════════════════════

DEFAULT_PLAN = "free"   # 👈 "free" = ollama | "pro" = mistral (used if request omits "plan")


# ══════════════════════════════════════════════════════════════
# 2. STATE
# ══════════════════════════════════════════════════════════════

class SummarizerState(TypedDict):
    file_path          : str
    plan               : str
    raw_code           : str
    module_summary     : str
    functions          : List[dict]
    function_summaries : List[dict]
    final_report       : dict
    error              : Optional[str]


# ══════════════════════════════════════════════════════════════
# 3. LLM CALLERS
# ══════════════════════════════════════════════════════════════

def call_ollama(prompt: str, num_predict: int = 800) -> str:
    """Always available — local model, no rate limits"""
    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model"  : "qwen2.5-coder:3b",
                "prompt" : prompt,
                "stream" : False,
                "options": {"temperature": 0.1, "num_predict": num_predict}
            },
            timeout=180
        )
        return response.json().get("response", "No response from Ollama.")
    except Exception as e:
        return f"Ollama error: {str(e)}"


def call_mistral(prompt: str, retry: bool = True, max_tokens: int = 1024) -> str:
    """Mistral call with rate-limit-aware retry.

    - RPM-style 429s: short backoff (15s), retry once, then fall back to Ollama.
    - RPD-style 429s (daily quota exhausted): retrying won't help — fall
      back to Ollama immediately.
    - Any other error: fall back to Ollama.
    """
    try:
        response = mistral_client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        # pacing delay to stay comfortably under per-minute limits
        time.sleep(4)
        return response.choices[0].message.content or "No response from Mistral."

    except Exception as e:
        err = str(e)
        lowered = err.lower()

        is_rate_limited = "429" in err or "quota" in lowered or "exhausted" in lowered or "rate limit" in lowered
        is_daily_quota = "per day" in lowered or "rpd" in lowered or "daily" in lowered

        if is_rate_limited and is_daily_quota:
            print("  ⚠️ Mistral daily quota exhausted — falling back to Ollama")
            return call_ollama(prompt)

        if is_rate_limited:
            if retry:
                print("  ⚠️ Mistral RPM limit hit — waiting 15s then retrying once...")
                time.sleep(15)
                return call_mistral(prompt, retry=False, max_tokens=max_tokens)
            print("  ⚠️ Mistral still rate limited — falling back to Ollama")
            return call_ollama(prompt)

        print(f"  ⚠️ Mistral failed ({err[:120]}) — falling back to Ollama")
        return call_ollama(prompt)


# ── Original Gemini caller — kept for reference / easy revert ──
# def call_gemini(prompt: str, retry: bool = True, max_tokens: int = 1024) -> str:
#     """Gemini call with rate-limit-aware retry.
#
#     - RPM-style 429s: short backoff (15s), retry once, then fall back to Ollama.
#     - RPD-style 429s (daily quota exhausted): retrying won't help — fall
#       back to Ollama immediately.
#     - Any other error: fall back to Ollama.
#     """
#     try:
#         response = gemini_client.models.generate_content(
#             model="gemini-2.5-flash",
#             contents=prompt,
#             config=types.GenerateContentConfig(
#                 temperature=0.1,
#                 max_output_tokens=max_tokens
#             )
#         )
#         time.sleep(4)
#         return response.text or "No response from Gemini."
#
#     except Exception as e:
#         err = str(e)
#         lowered = err.lower()
#
#         is_rate_limited = "429" in err or "quota" in lowered or "exhausted" in lowered
#         is_daily_quota = "per day" in lowered or "rpd" in lowered or "daily" in lowered
#
#         if is_rate_limited and is_daily_quota:
#             print("  ⚠️ Gemini daily quota exhausted — falling back to Ollama")
#             return call_ollama(prompt)
#
#         if is_rate_limited:
#             if retry:
#                 print("  ⚠️ Gemini RPM limit hit — waiting 15s then retrying once...")
#                 time.sleep(15)
#                 return call_gemini(prompt, retry=False, max_tokens=max_tokens)
#             print("  ⚠️ Gemini still rate limited — falling back to Ollama")
#             return call_ollama(prompt)
#
#         print(f"  ⚠️ Gemini failed ({err[:120]}) — falling back to Ollama")
#         return call_ollama(prompt)


def call_llm(prompt: str, plan: str, max_tokens: int = 1024) -> str:
    if plan == "pro":
        return call_mistral(prompt, max_tokens=max_tokens)
        # return call_gemini(prompt, max_tokens=max_tokens)   # 👈 old Gemini path
    return call_ollama(prompt, num_predict=max_tokens)


# ══════════════════════════════════════════════════════════════
# 4. JSON PARSER — robust, with truncation detection AND
#    per-entry recovery (smaller/local models often emit one
#    malformed entry — e.g. an unescaped quote — inside an
#    otherwise valid array; we shouldn't lose every entry for that)
# ══════════════════════════════════════════════════════════════

def _strip_fences(raw: str) -> str:
    clean = raw.strip()
    for fence in ["```json", "```JSON", "```"]:
        clean = clean.replace(fence, "")
    return clean.strip()


def _split_top_level_objects(array_body: str) -> List[str]:
    """Split the inside of a JSON array into individual {...} object
    strings by tracking brace depth, rather than naive comma-splitting
    (commas can legitimately appear inside string values)."""
    objects = []
    depth = 0
    current = []
    in_string = False
    escape_next = False

    for ch in array_body:
        if escape_next:
            escape_next = False
            if depth > 0:
                current.append(ch)
            continue

        if ch == "\\" and in_string:
            escape_next = True
            if depth > 0:
                current.append(ch)
            continue

        if ch == '"':
            in_string = not in_string

        if ch == "{" and not in_string:
            depth += 1
        if depth > 0:
            current.append(ch)
        if ch == "}" and not in_string:
            depth -= 1
            if depth == 0:
                objects.append("".join(current))
                current = []

    return objects


def parse_llm_json_lenient(raw: str) -> list:
    """Recover as many valid {...} entries as possible from a JSON array,
    even if one or more entries are malformed (e.g. an unescaped quote
    inside a string value — common with smaller local models)."""
    clean = _strip_fences(raw)

    start = clean.find("[")
    end = clean.rfind("]") + 1

    if start == -1:
        raise ValueError(f"No JSON array found. Raw: {clean[:200]}")
    if end == 0:
        raise ValueError(
            f"TRUNCATED: JSON array was cut off (no closing ']'). "
            f"Raw length: {len(clean)} chars"
        )

    array_body = clean[start + 1:end - 1]
    object_strings = _split_top_level_objects(array_body)

    parsed = []
    failed = 0
    for obj_str in object_strings:
        try:
            parsed.append(json.loads(obj_str))
        except json.JSONDecodeError:
            failed += 1

    if failed:
        print(f"  ⚠️ {failed}/{len(object_strings)} entries had malformed JSON and were skipped")

    if not parsed:
        raise ValueError(f"All {len(object_strings)} entries failed to parse. Raw: {clean[:200]}")

    return parsed


def parse_llm_json(raw: str) -> list:
    """Try a clean whole-array parse first (fast path, works for
    well-formed model output). Fall back to per-entry recovery if the
    whole array doesn't parse — this is the common path for Ollama/local
    model output that has one bad entry in an otherwise valid array.
    Truncation (no closing bracket) is still raised distinctly so callers
    can react by retrying with a bigger token budget instead of trying
    to "recover" from an array that was never finished.
    """
    clean = _strip_fences(raw)

    start = clean.find("[")
    end = clean.rfind("]") + 1

    if start == -1:
        raise ValueError(f"No JSON array found. Raw: {clean[:200]}")
    if end == 0:
        raise ValueError(
            f"TRUNCATED: JSON array was cut off (no closing ']'). "
            f"Raw length: {len(clean)} chars"
        )

    try:
        return json.loads(clean[start:end])
    except json.JSONDecodeError:
        return parse_llm_json_lenient(raw)


# ══════════════════════════════════════════════════════════════
# 5. NODES
# ══════════════════════════════════════════════════════════════

# ── Node 1: Read file ─────────────────────────────────────────
def read_file(state: SummarizerState) -> SummarizerState:
    print(f"\n📂 Reading: {state['file_path']}")
    try:
        with open(state["file_path"], "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        return {**state, "raw_code": code, "error": None}
    except Exception as e:
        return {**state, "raw_code": "", "error": f"Could not read file: {e}"}


# ── Node 2: Parse functions via AST ──────────────────────────
def parse_functions(state: SummarizerState) -> SummarizerState:
    print("🔍 Parsing functions with AST...")
    if state.get("error"):
        return state

    try:
        tree = ast.parse(state["raw_code"])
    except SyntaxError as e:
        return {**state, "error": f"Syntax error: {e}"}

    lines     = state["raw_code"].split("\n")
    functions = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args      = [arg.arg for arg in node.args.args]
            docstring = ast.get_docstring(node) or ""
            start     = node.lineno - 1
            full_end  = getattr(node, "end_lineno", start + 30)
            end       = min(start + 30, len(lines))
            body_snip = "\n".join(lines[start:end])
            is_truncated = full_end > end

            returns = ""
            if node.returns:
                try:
                    returns = ast.unparse(node.returns)
                except Exception:
                    returns = ""

            functions.append({
                "name"        : node.name,
                "args"        : args,
                "docstring"   : docstring,
                "body_snippet": body_snip,
                "lineno"      : node.lineno,
                "returns"     : returns,
                "truncated"   : is_truncated,
            })

    print(f"   Found {len(functions)} functions")
    return {**state, "functions": functions}


# ── Node 3: Summarize module ──────────────────────────────────
def summarize_module(state: SummarizerState) -> SummarizerState:
    print("📝 Summarizing module...")
    if state.get("error"):
        return state

    file_name    = os.path.basename(state["file_path"])
    code_preview = state["raw_code"][:4000]

    prompt = f"""You are a senior Python developer doing a code review.

File name: {file_name}

Source code:
```python
{code_preview}
```

Write a module summary in exactly 3 sentences:
Sentence 1: What does this module do? (based on actual code — imports, classes, functions, endpoints)
Sentence 2: What specific problem does it solve or what role does it play?
Sentence 3: How does it fit in the project? (look at imports, FastAPI routers, class names for clues)

Rules:
- Read the code carefully before writing
- Mention actual function names, class names, or endpoints from the code
- Do NOT invent features that are not in the code
- No bullet points, no headers — just 3 plain sentences
- Do NOT start with phrases like "Okay", "Let's dive in", "Sure", or any preamble — begin directly with the first sentence of the summary
- Be concise and developer-friendly"""

    summary = call_llm(prompt, state["plan"], max_tokens=600)
    return {**state, "module_summary": summary.strip()}


# ── Node 4: Summarize ALL functions in ONE call ───────────────
def summarize_functions(state: SummarizerState) -> SummarizerState:
    print("⚙️  Summarizing all functions in ONE LLM call...")
    if state.get("error"):
        return state

    functions = state.get("functions", [])
    if not functions:
        return {**state, "function_summaries": []}

    all_functions_text = ""
    for i, fn in enumerate(functions, 1):
        truncated_note = (
            " (NOTE: this snippet is truncated — the function continues "
            "beyond what's shown here. Do NOT claim it has no return "
            "statement based on this snippet alone.)"
            if fn.get("truncated") else ""
        )
        all_functions_text += f"""
Function {i}:
  Name      : {fn['name']}{truncated_note}
  Arguments : {', '.join(fn['args']) or 'none'}
  Returns   : {fn['returns'] or 'not annotated'}
  Docstring : {fn['docstring'] or 'none'}
  Code:
{fn['body_snippet']}
{"─" * 40}"""

    prompt = f"""You are a senior Python developer doing a detailed code review.

Analyze ALL of these functions from the same Python file.

{all_functions_text}

For EACH function write exactly 2 sentences:
- Sentence 1: What does this function specifically do? (read the code — be specific)
- Sentence 2: What does it return or what is its side effect?

Critical rules you MUST follow:
- If you see a return statement → state exactly what it returns
- If there is no return statement AND the snippet is not marked truncated → say "Returns None."
- If the snippet is marked truncated and no return is visible, say "Return behavior not visible in this snippet."
- If it prints something → mention it
- If it modifies external state → mention it
- NEVER say "does not return" if there is a return statement in the code
- Be specific to the actual code shown — no generic descriptions

CRITICAL JSON FORMATTING RULES:
- You MUST respond with ONLY a valid JSON array — no markdown, no explanation, no fences
- Any double-quote character inside a "summary" string MUST be escaped as \\"
- Do not use unescaped quotes, smart quotes, or line breaks inside string values

[
  {{"name": "exact_function_name", "summary": "Sentence 1. Sentence 2."}},
  {{"name": "exact_function_name", "summary": "Sentence 1. Sentence 2."}}
]"""

    token_budget = min(8192, 300 * len(functions) + 500)

    raw = call_llm(prompt, state["plan"], max_tokens=token_budget)

    parsed = None
    try:
        parsed = parse_llm_json(raw)
    except ValueError as e:
        if "TRUNCATED" in str(e) and state["plan"] == "pro":
            print("  ⚠️ Truncated — retrying once with a larger token budget...")
            retry_budget = min(token_budget * 2, 8192)
            raw = call_mistral(prompt, max_tokens=retry_budget)
            # raw = call_gemini(prompt, max_tokens=retry_budget)   # 👈 old Gemini retry path
            try:
                parsed = parse_llm_json(raw)
            except Exception as e2:
                print(f"  ⚠️ Still failed after retry: {e2}")
        else:
            print(f"  ⚠️ JSON parse failed: {e}")

    if parsed is not None:
        function_summaries = []
        for fn in functions:
            match = next(
                (p for p in parsed if p.get("name") == fn["name"]),
                None
            )
            function_summaries.append({
                "name"   : fn["name"],
                "args"   : fn["args"],
                "lineno" : fn["lineno"],
                "returns": fn["returns"],
                "summary": match["summary"] if match else "Summary not generated.",
            })
            status = "✅" if match else "⚠️ (not found in parsed entries)"
            print(f"   {status} {fn['name']}")
    else:
        print(f"  Raw (first 300 chars): {raw[:300]}")

        function_summaries = []
        for fn in functions:
            function_summaries.append({
                "name"   : fn["name"],
                "args"   : fn["args"],
                "lineno" : fn["lineno"],
                "returns": fn["returns"],
                "summary": "Could not parse summary.",
            })
            print(f"   ⚠️ {fn['name']} (fallback)")

    return {**state, "function_summaries": function_summaries}


# ── Node 5: Build final report ────────────────────────────────
def build_report(state: SummarizerState) -> SummarizerState:
    print("📊 Building final report...")
    if state.get("error"):
        return {**state, "final_report": {"error": state["error"]}}

    report = {
        "file"           : state["file_path"],
        "module_summary" : state.get("module_summary", ""),
        "total_functions": len(state.get("functions", [])),
        "functions"      : state.get("function_summaries", []),
    }
    return {**state, "final_report": report}


def check_error(state: SummarizerState) -> str:
    return "build_report" if state.get("error") else "continue"


# ══════════════════════════════════════════════════════════════
# 6. BUILD GRAPH
# ══════════════════════════════════════════════════════════════

def build_graph():
    graph = StateGraph(SummarizerState)

    graph.add_node("read_file",           read_file)
    graph.add_node("parse_functions",     parse_functions)
    graph.add_node("summarize_module",    summarize_module)
    graph.add_node("summarize_functions", summarize_functions)
    graph.add_node("build_report",        build_report)

    graph.set_entry_point("read_file")

    graph.add_conditional_edges(
        "read_file",
        check_error,
        {
            "continue"    : "parse_functions",
            "build_report": "build_report",
        }
    )

    graph.add_edge("parse_functions",     "summarize_module")
    graph.add_edge("summarize_module",    "summarize_functions")
    graph.add_edge("summarize_functions", "build_report")
    graph.add_edge("build_report",        END)

    return graph.compile()


summarizer_graph = build_graph()


# ══════════════════════════════════════════════════════════════
# 7. PRINT REPORT (kept for local/manual debugging — not used by the API)
# ══════════════════════════════════════════════════════════════

def print_report(report: dict):
    if "error" in report:
        print(f"\n❌ Error: {report['error']}")
        return

    print("\n" + "=" * 60)
    print(f"📄 FILE : {report.get('file')}")
    print("=" * 60)

    print("\n🧩 MODULE SUMMARY:")
    print(report.get("module_summary", "N/A"))

    print(f"\n⚙️  FUNCTIONS ({report.get('total_functions', 0)} found):")
    print("-" * 60)

    for fn in report.get("functions", []):
        args_str = f"({', '.join(fn['args'])})" if fn['args'] else "()"
        ret_str  = f" → {fn['returns']}" if fn.get("returns") else ""
        print(f"\n🔹 {fn['name']}{args_str}{ret_str}")
        print(f"   Line    : {fn['lineno']}")
        print(f"   Summary : {fn['summary']}")

    print("\n" + "=" * 60)


# ══════════════════════════════════════════════════════════════
# 8. ROUTER
# ══════════════════════════════════════════════════════════════

router = APIRouter(
    prefix="/api/summariser",
    tags=["Summarizer"]
)


class SummariseFileRequest(BaseModel):
    file_path: str
    plan: Optional[str] = DEFAULT_PLAN   # "free" (ollama) or "pro" (mistral → ollama fallback)


@router.post("/summariseFile", status_code=status.HTTP_200_OK)
def summarise_file(request: SummariseFileRequest):
    """
    Receives a file path (and optional plan) and runs it through the
    existing LangGraph pipeline: read_file -> parse_functions ->
    summarize_module -> summarize_functions -> build_report.
    Returns the final_report as JSON.
    """
    if request.plan not in ("free", "pro"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="plan must be 'free' or 'pro'")

    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File not found: {request.file_path}")

    print(f"\n🚀 /summariseFile called")
    print(f"   File : {request.file_path}")
    print(f"   Plan : {request.plan} ({'Mistral → Ollama fallback' if request.plan == 'pro' else 'Ollama'})")

    initial_state: SummarizerState = {
        "file_path"          : request.file_path,
        "plan"               : request.plan,
        "raw_code"           : "",
        "module_summary"     : "",
        "functions"          : [],
        "function_summaries" : [],
        "final_report"       : {},
        "error"              : None,
    }

    result = summarizer_graph.invoke(initial_state)
    final_report = result["final_report"]

    if "error" in final_report:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=final_report["error"])

    # Also persist to disk, same as the original script did
    with open("summary_output.json", "w") as f:
        json.dump(final_report, f, indent=2)

    return final_report