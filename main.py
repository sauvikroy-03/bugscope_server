from dotenv import load_dotenv
loaded = load_dotenv()
print("ENV LOADED:", loaded)
import os  
print("KEY:", os.getenv("RAZORPAY_KEY_ID"))
from fastapi import FastAPI

from pydantic import BaseModel

import subprocess
import json
import requests
from fastapi.middleware.cors import CORSMiddleware
from routes import users


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows Next.js (localhost:3000) to communicate without blocks
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(users.router)

@app.get("/")
async def root():
    """
    Root check endpoint to verify that the server is alive and reachable.
    """
    return {"message": "BugScope Core API is online and healthy."}
# ----------------------------
# CORS
# ----------------------------


# ----------------------------
# Setup
# ----------------------------
BASE_DIR = os.path.abspath("repos")
os.makedirs(BASE_DIR, exist_ok=True)

# ----------------------------
# Models
# ----------------------------
class RepoRequest(BaseModel):
    clone_url: str
    name: str

class GraphRequest(BaseModel):
    graph: dict


# ----------------------------
# 🔥 SMART GRAPH SUMMARIZER
# ----------------------------
def summarize_graph(graph):
    metrics = graph.get("metrics", {})

    in_deg = metrics.get("in_degree", {})
    out_deg = metrics.get("out_degree", {})

    # Top important nodes
    top_nodes = sorted(
        in_deg.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    # Most active nodes (outgoing)
    active_nodes = sorted(
        out_deg.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    # Sample edges (limit)
    sample_edges = graph["edges"][:25]

    # Isolated nodes
    isolated = [
        node for node in in_deg
        if in_deg[node] == 0 and out_deg[node] == 0
    ][:5]

    return {
        "top_nodes": top_nodes,
        "active_nodes": active_nodes,
        "sample_edges": sample_edges,
        "isolated": isolated,
        "total_nodes": len(graph["nodes"]),
        "total_edges": len(graph["edges"])
    }


# ----------------------------
# 🔥 LLM FUNCTION (FIXED PROPERLY)
# ----------------------------
def analyze_with_llm(graph):
    try:
        summary = summarize_graph(graph)

        prompt = f"""
You are a STRICT code dependency analyzer.

RULES:
- Do NOT guess
- Use ONLY given data
- Be concise and precise

DATA SUMMARY:

Total files: {summary['total_nodes']}
Total dependencies: {summary['total_edges']}

Top important files (highest in-degree):
{summary['top_nodes']}

Most active files (highest out-degree):
{summary['active_nodes']}

Sample dependencies (source -> target):
{summary['sample_edges']}

Isolated files:
{summary['isolated']}

TASK:
1. Structure (1 line)
2. Most important file (must match highest in-degree)
3. Most active file (must match highest out-degree)
4.Which files are imported by which files
5. One key observation
6. One issue or suggestion

Max 6 lines total.
"""

        print("🚀 Sending request to Ollama...")

        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "phi4-mini:latest",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,   # 🔥 less hallucination
                    "num_predict": 200    # 🔥 shorter output
                }
            },
            timeout=120
        )

        print("✅ Ollama responded")

        data = response.json()

        return data.get("response", "No response from model")

    except Exception as e:
        print("❌ LLM ERROR:", e)
        return f"LLM error: {str(e)}"


# ----------------------------
# ANALYZE (Clone + Parse)
# ----------------------------
@app.post("/analyze")
def analyze_repo(req: RepoRequest):
    repo_path = os.path.join(BASE_DIR, req.name)

    if not os.path.exists(repo_path):
        try:
            subprocess.run(
                ["git", "clone", req.clone_url, repo_path],
                check=True
            )
        except Exception as e:
            return {"error": f"Clone failed: {str(e)}"}

    try:
        script_path = os.path.join(os.path.dirname(__file__), "parse_repo.py")

        result = subprocess.run(
            ["python", script_path, repo_path],
            capture_output=True,
            text=True
        )

        print("----- PARSER OUTPUT -----")
        print("STDOUT:", result.stdout)  # 🔥 avoid huge logs
        print("STDERR:", result.stderr)
        print("-------------------------")

        if result.returncode != 0:
            return {"error": "Parser failed", "details": result.stderr}

        graph = json.loads(result.stdout)

        return {"graph": graph}

    except Exception as e:
        return {"error": str(e)}


# ----------------------------
# 🔥 LLM ENDPOINT
# ----------------------------
@app.post("/llm")
def run_llm(req: GraphRequest):
    print("🔥 /llm endpoint HIT")

    try:
        graph = req.graph

        print(f"📦 Graph received: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")

        result = analyze_with_llm(graph)

        print("🧠 LLM OUTPUT:", result)

        return {"analysis": result}

    except Exception as e:
        print("❌ ERROR:", e)
        return {"error": str(e)}