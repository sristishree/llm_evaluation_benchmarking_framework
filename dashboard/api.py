"""
Phase 6 — FastAPI backend for the React dashboard.

Run from the project root:
    uvicorn dashboard.api:app --reload --port 8000

The React dev server (port 5173) proxies /api/* to this process.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent))
from storage.db import ResultsStore

app = FastAPI(title="LLM Evaluation Benchmark API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

_DB_PATH = Path(__file__).parent.parent / "results.db"


def _store() -> ResultsStore:
    return ResultsStore(_DB_PATH)


# ---------------------------------------------------------------------------
# Run endpoints
# ---------------------------------------------------------------------------

@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    with _store() as s:
        return s.list_runs()


@app.get("/api/runs/{run_id}")
def get_run_summary(run_id: str) -> dict[str, Any]:
    with _store() as s:
        return s.get_run_summary(run_id)


# ---------------------------------------------------------------------------
# Aggregate score endpoints (power the charts)
# ---------------------------------------------------------------------------

@app.get("/api/scores/breakdown")
def score_breakdown() -> list[dict[str, Any]]:
    with _store() as s:
        return s.agg_by_provider_and_type()


@app.get("/api/scores/heatmap")
def score_heatmap() -> list[dict[str, Any]]:
    with _store() as s:
        return s.agg_heatmap()


@app.get("/api/scores/cost-quality")
def cost_vs_quality() -> list[dict[str, Any]]:
    with _store() as s:
        return s.agg_cost_vs_quality()


# ---------------------------------------------------------------------------
# Row-level results (powers the response viewer)
# ---------------------------------------------------------------------------

@app.get("/api/results")
def query_results(
    provider: str | None = Query(None),
    model: str | None = Query(None),
    task_type: str | None = Query(None),
    difficulty: str | None = Query(None),
    domain: str | None = Query(None),
    run_id: str | None = Query(None),
    limit: int = Query(200, le=1000),
) -> list[dict[str, Any]]:
    with _store() as s:
        return s.query(
            provider=provider,
            model=model,
            task_type=task_type,
            difficulty=difficulty,
            domain=domain,
            run_id=run_id,
            dry_run=False,
            limit=limit,
        )
