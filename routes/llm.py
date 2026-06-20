from dotenv import load_dotenv
load_dotenv()

import os
import json
import requests
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Any, Literal
from google import genai
from google.genai import types

router = APIRouter(prefix="/api/llm", tags=["LLM"])

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ── Models ────────────────────────────────────────────────────────────────────

class LLMRequest(BaseModel):
    results: List[Any]
    plan: Literal["free", "pro"] = "free"


# ── Summarizer ────────────────────────────────────────────────────────────────

def simplify_file_result(r: dict):
    metrics = r.get("important_metrics", {})
    return {
        "file":        r.get("file"),
        "risk_score":  r.get("risk_score"),
        "risk_level":  r.get("risk_level"),
        "main_reasons": r.get("llm_context", []),
        "metrics": {
            "loc":                   metrics.get("loc"),
            "cyclomatic_complexity": metrics.get("cyclomatic_complexity"),
            "avg_function_length":   metrics.get("avg_function_length"),
            
            "code_churn":            metrics.get("code_churn"),
            "commit_frequency":      metrics.get("commit_frequency"),
            "in_degree":             metrics.get("in_degree"),
            "out_degree":            metrics.get("out_degree"),
            "total_degree":          metrics.get("total_degree"),
            "pagerank":              metrics.get("pagerank"),
        },
        "directly_impacted_files": [
            {
                "file":             af.get("file"),
                "impact_score":     af.get("affected_score"),
                "their_risk_score": af.get("risk_score"),
            }
            for af in r.get("affected_files", [])
        ],
    }


def summarize_results(results: list):
    selected = []
    for r in results:
        metrics = r.get("important_metrics", {})
        if (
            r.get("risk_level") in ["HIGH", "MEDIUM"]
            or len(r.get("affected_files", [])) > 0
            or metrics.get("in_degree", 0) >= 2
            or metrics.get("pagerank", 0) >= 0.2
        ):
            selected.append(simplify_file_result(r))

    selected = sorted(
        selected,
        key=lambda r: (
            r.get("risk_level") == "HIGH",
            r.get("risk_level") == "MEDIUM",
            r.get("risk_score") or 0,
            r.get("metrics", {}).get("pagerank") or 0,
        ),
        reverse=True
    )[:8]

    return {
        "total_files":       len(results),
        "high_risk_count":   len([r for r in results if r.get("risk_level") == "HIGH"]),
        "medium_risk_count": len([r for r in results if r.get("risk_level") == "MEDIUM"]),
        "selected_files":    selected,
    }


# ── LLM callers ───────────────────────────────────────────────────────────────

def analyze_with_gemini(prompt: str) -> str:
    print("🚀 Routing to Gemini (Pro user)")
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=512
        )
    )
    print("✅ Gemini responded")
    return response.text or "No response from Gemini."


def analyze_with_ollama(prompt: str) -> str:
    print("🚀 Routing to Ollama (Free user)")
    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": "phi4-mini:latest",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 300}
        },
        timeout=120
    )
    print("✅ Ollama responded")
    return response.json().get("response", "No response from Ollama.")


# ── Per-file fix prompt ───────────────────────────────────────────────────────

def build_fix_prompt(file_data: dict) -> str:
    return f"""
You are BugScope AI. Give a short, developer-friendly fix guide for this file.

File: {file_data["file"]}
Risk Level: {file_data["risk_level"]}
Risk Score: {file_data["risk_score"]}
Main Issues: {", ".join(file_data["main_reasons"])}
Metrics:
- LOC: {file_data["metrics"]["loc"]}
- Cyclomatic Complexity: {file_data["metrics"]["cyclomatic_complexity"]}
- Avg Function Length: {file_data["metrics"]["avg_function_length"]}
- Code Churn: {file_data["metrics"]["code_churn"]}

- In Degree: {file_data["metrics"]["in_degree"]}
- PageRank: {file_data["metrics"]["pagerank"]}

Write exactly 3 fix steps. Each step:
- Start with a bold action verb
- Be specific to the metrics above
- Be 1-2 sentences max

Format:
1. **<action>**: <explanation>
2. **<action>**: <explanation>
3. **<action>**: <explanation>

No intro. No outro. Just the 3 steps.
"""


def get_fix_for_file(file_data: dict, plan: str) -> str:
    prompt = build_fix_prompt(file_data)
    try:
        if plan == "pro":
            return analyze_with_gemini(prompt)
        else:
            return analyze_with_ollama(prompt)
    except Exception as e:
        return f"Could not generate fix: {str(e)}"


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/suggestions")
def get_suggestions(req: LLMRequest):
    print(f"🔥 /api/llm/suggestions HIT — plan: {req.plan}")
    try:
        summary = summarize_results(req.results)

        print("SELECTED FILES:", json.dumps(summary["selected_files"], indent=2))

        # HIGH and MEDIUM files
        risky_files = [
            f for f in summary["selected_files"]
            if f["risk_level"] in ["HIGH", "MEDIUM"]
        ]

        # dependency nodes that are LOW but architecturally important
        dep_nodes = [
            f for f in summary["selected_files"]
            if f not in risky_files and (
                f["metrics"].get("in_degree", 0) >= 2 or
                f["metrics"].get("pagerank", 0) >= 0.2
            )
        ]

        cards = []

        for f in risky_files + dep_nodes:
            print(f"🔧 Generating fix for: {f['file']}")
            fix_text = get_fix_for_file(f, req.plan)

            # enrich affected files with full data from original results
            affected_enriched = []
            for af in f["directly_impacted_files"]:
                full = next(
                    (r for r in req.results if r["file"] == af["file"]),
                    None
                )
                affected_enriched.append({
                    "file":             af["file"],
                    "impact_score":     af["impact_score"],
                    "their_risk_score": af["their_risk_score"],
                    "their_risk_level": full.get("risk_level", "UNKNOWN") if full else "UNKNOWN",
                    "their_reasons":    full.get("llm_context", []) if full else [],
                })

            cards.append({
                "file":          f["file"],
                "risk_level":    f["risk_level"],
                "risk_score":    f["risk_score"],
                "main_reasons":  f["main_reasons"],
                "metrics":       f["metrics"],
                "affected_files": affected_enriched,
                "fix":           fix_text,
            })

        # sort: HIGH first, then MEDIUM, then by risk_score descending
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        cards.sort(key=lambda c: (order.get(c["risk_level"], 3), -c["risk_score"]))

        return {"cards": cards}

    except Exception as e:
        print("❌ ERROR:", e)
        return {"error": str(e)}