from dotenv import load_dotenv
load_dotenv()

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any, Literal, Optional

# 👇 plain module, not a router — holds the actual file-reading + LLM-summarizing logic
from project_context import build_project_context

router = APIRouter(
    prefix="/api/context",
    tags=["Context"]
)


# ── Models ────────────────────────────────────────────────────────────────────

class GenerateContextRequest(BaseModel):
    graph: Dict[str, Any]              # the full parse_repo.py output (nodes, edges, ...)
    plan: Literal["free", "pro"] = "free"
    repo_root: Optional[str] = None    # OPTIONAL — absolute path to the repo on disk.
                                        # Strongly recommended: guarantees correct file
                                        # resolution. Without it, falls back to a glob
                                        # search across ALL of ./repos/, which can grab
                                        # the wrong file if filenames collide across repos.


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/generateContext", status_code=status.HTTP_200_OK)
def generate_context(req: GenerateContextRequest):
    """
    Reads every file referenced in req.graph["nodes"], summarizes each with an
    LLM, and returns:
      - all_files: { "<file_id>": { "summary": "..." } }
      - connected_files: the dependency edges, passed through as-is
      - context_prompt: a ready-to-paste natural-language project context block
    """
    if "nodes" not in req.graph:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="graph must include a 'nodes' list."
        )

    try:
        return build_project_context(req.graph, req.plan, req.repo_root)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Context generation failed: {str(e)}"
        )