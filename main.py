from dotenv import load_dotenv
load_dotenv()

import os
import subprocess
import json
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from routes import users, llm  # ← added llm

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users.router)
app.include_router(llm.router)  # ← added


@app.get("/")
async def root():
    return {"message": "BugScope Core API is online and healthy."}


# ── Setup ─────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.abspath("repos")
os.makedirs(BASE_DIR, exist_ok=True)


# ── Models ────────────────────────────────────────────────────────────────────

class RepoRequest(BaseModel):
    clone_url: str
    name: str


# ── Analyze (Clone + Parse) ───────────────────────────────────────────────────

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
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("-------------------------")

        if result.returncode != 0:
            return {"error": "Parser failed", "details": result.stderr}

        graph = json.loads(result.stdout)
        return {"graph": graph}

    except Exception as e:
        return {"error": str(e)}