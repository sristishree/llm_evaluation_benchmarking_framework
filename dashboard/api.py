"""
Phase 6 — FastAPI backend for the React dashboard.

Run from the project root:
    uvicorn dashboard.api:app --reload --port 8000

The React dev server (port 5173) proxies /api/* to this process.
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from storage.db import ResultsStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Show DEBUG logs from our own code but keep third-party libs at INFO
logging.getLogger("evaluators").setLevel(logging.DEBUG)
logging.getLogger("storage").setLevel(logging.DEBUG)

_TAGS_METADATA = [
    {
        "name": "configuration",
        "description": "Provider catalog, dataset task-bank metadata, and rubric defaults.",
    },
    {
        "name": "runs",
        "description": "Start, monitor, retrieve, and delete benchmark runs.",
    },
    {
        "name": "scores",
        "description": "Aggregated score analytics powering charts and comparisons.",
    },
    {
        "name": "results",
        "description": "Row-level generation results with full metric detail.",
    },
    {
        "name": "judge",
        "description": "LLM-as-a-judge prompt preview and calibration against human scores.",
    },
]

app = FastAPI(
    title="LLM Evaluation Benchmark API",
    version="1.0.0",
    description=(
        "REST API for the LLM Evaluation Framework. "
        "Run benchmarks across providers and models, retrieve aggregated scores, "
        "and calibrate an LLM-as-a-judge against human ratings.\n\n"
        "**Interactive docs:** `/docs` (Swagger UI) · `/redoc` (ReDoc)"
    ),
    openapi_tags=_TAGS_METADATA,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

_DB_PATH   = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent.parent))) / "results.db"
_TASK_BANK = Path(__file__).parent.parent / "task_bank"
_CONFIG    = Path(__file__).parent.parent / "config" / "providers.yaml"

# In-memory job registry for in-flight runs
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Dataset metadata — human-readable names and descriptions
# ---------------------------------------------------------------------------

_DATASET_META: dict[str, dict[str, str]] = {
    # Classification
    "clf_amazon_tasks.jsonl":    {"name": "Amazon Reviews",   "description": "Sentiment on product reviews (positive / negative)."},
    "clf_clinc_tasks.jsonl":     {"name": "CLINC150",         "description": "Intent classification for voice assistants across 150 intent classes."},
    "clf_combined_tasks.jsonl":  {"name": "Combined (CLF)",   "description": "Mixed sample across all classification datasets below."},
    "clf_dbpedia_tasks.jsonl":   {"name": "DBpedia",          "description": "Topic classification into 14 ontology categories (Company, Film, Plant …)."},
    "clf_snips_tasks.jsonl":     {"name": "SNIPS NLU",        "description": "Intent detection for smart-home voice commands (7 intents)."},
    "clf_sst5_tasks.jsonl":      {"name": "SST-5",            "description": "Fine-grained 5-class sentiment (very negative → very positive)."},
    "clf_yahoo_tasks.jsonl":     {"name": "Yahoo Answers",    "description": "Topic classification across 10 categories (Society, Science, Health …)."},
    # Q&A
    "qa_combined_tasks.jsonl":   {"name": "Combined (QA)",    "description": "Mixed sample across all QA datasets below."},
    "qa_finqa_tasks.jsonl":      {"name": "FinQA",            "description": "Financial QA requiring numerical reasoning over earnings reports."},
    "qa_musique_tasks.jsonl":    {"name": "MuSiQue",          "description": "Multi-hop reasoning: 2–4 supporting facts needed per answer."},
    "qa_nq_tasks.jsonl":         {"name": "Natural Questions", "description": "Real Google queries answered from Wikipedia passages."},
    "qa_squad2_tasks.jsonl":     {"name": "SQuAD 2.0",        "description": "Reading comprehension with unanswerable questions — tests hallucination resistance."},
    "qa_triviaqa_tasks.jsonl":   {"name": "TriviaQA",         "description": "Trivia questions paired with Wikipedia/web evidence documents."},
    # Summarization
    "summarization_tasks.jsonl": {"name": "CNN / DailyMail",  "description": "News article summarisation. Reference summaries are bullet-point highlights."},
    # NER Extraction
    "ner_combined_tasks.jsonl":  {"name": "Combined (NER)",   "description": "Mixed sample across all NER datasets below."},
    "ner_bc5cdr_tasks.jsonl":    {"name": "BC5CDR",           "description": "Biomedical NER: chemical compounds and diseases from PubMed abstracts."},
    "ner_conll_tasks.jsonl":     {"name": "CoNLL-2003",       "description": "News NER: PER, ORG, LOC, MISC entities from Reuters articles."},
    "ner_few_nerd_tasks.jsonl":  {"name": "Few-NERD",         "description": "Fine-grained NER with 66 entity subtypes across 8 coarse categories."},
    "ner_wnut_tasks.jsonl":      {"name": "WNUT-17",          "description": "Emerging entity NER on social media (Twitter): unusual / novel named entities."},
}

# Files that are combined supersets — shown last and unchecked by default
_COMBINED_FILES = {"clf_combined_tasks.jsonl", "qa_combined_tasks.jsonl", "ner_combined_tasks.jsonl"}

# Glob patterns per task type — combined files are sorted to the end in Python
_TASK_TYPE_GLOBS: dict[str, list[str]] = {
    "classification": ["clf_*.jsonl"],
    "qa":             ["qa_*.jsonl"],
    "summarization":  ["summarization_tasks.jsonl", "summ_*.jsonl"],
    "extraction":     ["ner_*.jsonl"],
}

# Simple in-memory cache so we don't re-scan on every request
_task_bank_cache: dict[str, Any] | None = None
_task_bank_lock = threading.Lock()


def _store() -> ResultsStore:
    return ResultsStore(_DB_PATH)


# ---------------------------------------------------------------------------
# Provider catalog
# ---------------------------------------------------------------------------

@app.get(
    "/api/catalog",
    tags=["configuration"],
    summary="List available providers and models",
    response_description="Parsed contents of providers.yaml — provider names mapped to their model lists.",
)
def get_catalog() -> dict[str, Any]:
    """Return the full provider/model catalog loaded from `config/providers.yaml`."""
    with open(_CONFIG) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Task bank stats (powers the dataset selector panel)
# ---------------------------------------------------------------------------

@app.get(
    "/api/task-bank",
    tags=["configuration"],
    summary="Dataset metadata and difficulty statistics",
    response_description="Map of task_type → list of dataset entries with total count and easy/medium/hard breakdown.",
)
def task_bank_stats(refresh: bool = False) -> dict[str, Any]:
    """Return metadata for every JSONL dataset in the task bank.

    Results are cached in memory; pass `refresh=true` to force a re-scan.
    """
    global _task_bank_cache
    with _task_bank_lock:
        if _task_bank_cache is not None and not refresh:
            return _task_bank_cache
        result = _build_task_bank_stats()
        _task_bank_cache = result
        return result


def _build_task_bank_stats() -> dict[str, Any]:
    out: dict[str, list[dict]] = {}
    for task_type, globs in _TASK_TYPE_GLOBS.items():
        seen: set[Path] = set()
        files: list[Path] = []
        for pat in globs:
            for p in _TASK_BANK.glob(pat):
                if p not in seen:
                    seen.add(p)
                    files.append(p)
        # Individual datasets first, combined supersets last, alphabetical within each group
        files.sort(key=lambda p: (p.name in _COMBINED_FILES, p.name))

        datasets = []
        for path in files:
            if not path.exists():
                continue
            meta = _DATASET_META.get(path.name, {"name": path.stem, "description": ""})
            diff_counts: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
            total = 0
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        diff = d.get("difficulty", "")
                        if diff in diff_counts:
                            diff_counts[diff] += 1
                        total += 1
                    except Exception:
                        continue
            if total > 0:
                datasets.append({
                    "file":        path.name,
                    "name":        meta["name"],
                    "description": meta["description"],
                    "total":       total,
                    "difficulty":  diff_counts,
                    "is_combined": path.name in _COMBINED_FILES,
                })
        if datasets:
            out[task_type] = datasets
    return out


# ---------------------------------------------------------------------------
# Default rubric for a task type (powers the rubric preview in the UI)
# ---------------------------------------------------------------------------

@app.get(
    "/api/rubric",
    tags=["configuration"],
    summary="Get default rubric for a dataset or task type",
    response_description="Rubric string, task_type, and the resolved dataset_file.",
)
def get_default_rubric(
    dataset_file: str | None = Query(None),
    task_type:    str | None = Query(None),
) -> dict[str, Any]:
    """Return the default rubric string for a specific dataset file or task type.

    Prefer dataset_file (exact file) over task_type (first matching file).
    """
    from models.task import ClassificationTask, ExtractionTask, QATask, SummarizationTask

    TYPE_CLS = {
        "classification": ClassificationTask,
        "qa":             QATask,
        "summarization":  SummarizationTask,
        "extraction":     ExtractionTask,
    }

    def _rubric_from_path(path: Path) -> dict | None:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    cls = TYPE_CLS.get(data.get("task_type", ""))
                    if cls:
                        task = cls.model_validate(data)
                        return {"rubric": task.rubric, "task_type": data.get("task_type")}
                except Exception:
                    continue
        return None

    if dataset_file:
        path = _TASK_BANK / dataset_file
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Dataset file not found: {dataset_file}")
        result = _rubric_from_path(path)
        if result:
            return {**result, "dataset_file": dataset_file}
        raise HTTPException(status_code=404, detail=f"No valid tasks in: {dataset_file}")

    if task_type:
        if task_type not in TYPE_CLS:
            raise HTTPException(status_code=404, detail=f"Unknown task type: {task_type}")
        for pat in _TASK_TYPE_GLOBS.get(task_type, []):
            for path in sorted(_TASK_BANK.glob(pat)):
                if path.name in _COMBINED_FILES:
                    continue
                result = _rubric_from_path(path)
                if result:
                    return {**result, "dataset_file": path.name}

    raise HTTPException(status_code=400, detail="Provide dataset_file or task_type")


# ---------------------------------------------------------------------------
# Run list + summary (read-only, from DB)
# ---------------------------------------------------------------------------

@app.get(
    "/api/runs",
    tags=["runs"],
    summary="List all benchmark runs",
    response_description="Array of run metadata records ordered by most-recent first.",
)
def list_runs() -> list[dict[str, Any]]:
    """Return summary metadata for every benchmark run stored in the database."""
    with _store() as s:
        return s.list_runs()


@app.delete(
    "/api/runs/{run_id}",
    tags=["runs"],
    summary="Delete a run and all its results",
    status_code=200,
    response_description="Confirmation with count of deleted result rows.",
)
def delete_run(run_id: str) -> dict[str, Any]:
    """Permanently remove a run batch and every associated result row from the database."""
    with _jobs_lock:
        _jobs.pop(run_id, None)
    with _store() as s:
        deleted = s.delete_run(run_id)
    return {"run_id": run_id, "results_deleted": deleted}


@app.get(
    "/api/runs/{run_id}/summary",
    tags=["runs"],
    summary="Get detailed run summary",
    response_description="Aggregated metrics and metadata for the completed run.",
)
def get_run_summary(run_id: str) -> dict[str, Any]:
    """Return the full summary for a completed benchmark run by its UUID."""
    with _store() as s:
        return s.get_run_summary(run_id)


@app.get(
    "/api/runs/{run_id}",
    tags=["runs"],
    summary="Get run summary (compatibility alias)",
    response_description="Same response as /api/runs/{run_id}/summary.",
    include_in_schema=False,
)
def get_run_summary_compat(run_id: str) -> dict[str, Any]:
    with _store() as s:
        return s.get_run_summary(run_id)


# ---------------------------------------------------------------------------
# Start a benchmark run (async, returns 202 immediately)
# ---------------------------------------------------------------------------

class DatasetSelection(BaseModel):
    file:       str
    limit:      int
    difficulty: Literal["easy", "medium", "hard"] | None = None


class RunRequest(BaseModel):
    provider:              str
    model:                 str
    datasets:              list[DatasetSelection]
    dry_run:               bool = False
    notes:                 str | None = None
    # LLM-as-Judge settings
    use_judge:             bool = False
    judge_provider:        str | None = None
    judge_model:           str | None = None
    mitigate_position_bias: bool = False
    # Per-dataset rubric overrides: dataset_file → rubric string.
    # Applies only for this run — not persisted anywhere.
    rubric_overrides:      dict[str, str] = {}


@app.post(
    "/api/runs",
    tags=["runs"],
    summary="Start a benchmark run",
    status_code=202,
    response_description="Accepted — returns run_id immediately; poll /api/runs/{run_id}/status for progress.",
)
def start_run(req: RunRequest) -> dict[str, Any]:
    """Launch an async benchmark run.

    The run executes in a background thread. The response (HTTP 202) returns immediately
    with a `run_id`. Poll `GET /api/runs/{run_id}/status` to track progress.
    When `state` becomes `"done"`, results are available via `/api/results?run_id=…`.
    """
    if not req.datasets:
        raise HTTPException(status_code=422, detail="Select at least one dataset.")
    run_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[run_id] = {
            "run_id":      run_id,
            "state":       "loading",
            "progress":    0,
            "total":       0,
            "error":       None,
            "started_at":  datetime.now(tz=timezone.utc).isoformat(),
            "finished_at": None,
            "_cancel":     threading.Event(),
        }
    threading.Thread(target=_execute_run, args=(run_id, req), daemon=True).start()
    return {"run_id": run_id}


@app.get(
    "/api/runs/{run_id}/status",
    tags=["runs"],
    summary="Poll run progress",
    response_description="Job state object with `state` (loading | running | done | error), progress counters, and timestamps.",
)
def run_status(run_id: str) -> dict[str, Any]:
    """Return live progress for an in-flight run, or the completed summary for a finished one.

    `state` values: `loading` → `running` → `done` | `error` | `cancelled`.
    """
    with _jobs_lock:
        job = _jobs.get(run_id)
    if job:
        return _public_job(job)
    with _store() as s:
        summary = s.get_run_summary(run_id)
    if summary:
        return {"run_id": run_id, "state": "done", **summary}
    raise HTTPException(status_code=404, detail="Run not found")


@app.post(
    "/api/runs/{run_id}/cancel",
    tags=["runs"],
    summary="Cancel a running benchmark",
    status_code=200,
)
def cancel_run(run_id: str) -> dict[str, Any]:
    """Signal a running benchmark to stop after the current task completes."""
    with _jobs_lock:
        job = _jobs.get(run_id)
    if not job:
        raise HTTPException(status_code=404, detail="Run not found or already completed")
    state = job.get("state")
    if state not in ("loading", "running"):
        raise HTTPException(status_code=409, detail=f"Run is already in state '{state}'")
    cancel_event: threading.Event = job["_cancel"]
    cancel_event.set()
    _set_job(run_id, state="cancelled", finished_at=datetime.now(tz=timezone.utc).isoformat())
    return {"run_id": run_id, "state": "cancelled"}


# ---------------------------------------------------------------------------
# Aggregate score endpoints (power the charts)
# ---------------------------------------------------------------------------

@app.get(
    "/api/scores/breakdown",
    tags=["scores"],
    summary="Score breakdown by provider and task type",
    response_description="Aggregated mean scores grouped by provider × task_type.",
)
def score_breakdown(include_dry_runs: bool = Query(False)) -> list[dict[str, Any]]:
    """Return mean scores aggregated by provider and task type — the primary bar-chart dataset."""
    with _store() as s:
        return s.agg_by_provider_and_type(exclude_dry_runs=not include_dry_runs)


@app.get(
    "/api/scores/heatmap",
    tags=["scores"],
    summary="Provider × model score heatmap",
    response_description="Matrix of mean scores indexed by provider and model.",
)
def score_heatmap(include_dry_runs: bool = Query(False)) -> list[dict[str, Any]]:
    """Return aggregated scores in heatmap format — one row per provider/model combination."""
    with _store() as s:
        return s.agg_heatmap(exclude_dry_runs=not include_dry_runs)


@app.get(
    "/api/scores/cost-quality",
    tags=["scores"],
    summary="Cost vs quality comparison",
    response_description="Per-model data points with estimated_cost_usd and composite quality score.",
)
def cost_vs_quality(include_dry_runs: bool = Query(False)) -> list[dict[str, Any]]:
    """Return cost vs quality data points for the scatter-plot view — one record per provider/model."""
    with _store() as s:
        return s.agg_cost_vs_quality(exclude_dry_runs=not include_dry_runs)


@app.get(
    "/api/scores/by-domain",
    tags=["scores"],
    summary="Score breakdown by domain",
    response_description="Mean scores grouped by domain label.",
)
def score_by_domain(include_dry_runs: bool = Query(False)) -> list[dict[str, Any]]:
    """Return mean scores aggregated by domain — powers the domain breakdown chart."""
    with _store() as s:
        return s.agg_by_domain(exclude_dry_runs=not include_dry_runs)


@app.get(
    "/api/scores/distribution",
    tags=["scores"],
    summary="Individual score values for histogram charts",
    response_description="Flat list of per-result score fields (exact_match, rouge_l, bert_score, etc.).",
)
def score_distribution(include_dry_runs: bool = Query(False)) -> list[dict[str, Any]]:
    """Return one record per result with all metric values — used to draw score-distribution histograms."""
    with _store() as s:
        rows = s.query(dry_run=False if not include_dry_runs else None)
        return [
            {
                "provider":    r["provider"],
                "model":       r["model"],
                "task_type":   r["task_type"],
                "difficulty":  r["difficulty"],
                "domain":      r["domain"],
                "exact_match":      r["exact_match"],
                "rouge_l":          r["rouge_l"],
                "rouge_1":          r["rouge_1"],
                "rouge_2":          r["rouge_2"],
                "token_f1":         r["token_f1"],
                "bert_score":       r["bert_score"],
                "entity_precision": r.get("entity_precision"),
                "entity_recall":    r.get("entity_recall"),
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Row-level results (powers the response viewer)
# ---------------------------------------------------------------------------

@app.get(
    "/api/results",
    tags=["results"],
    summary="Query generation results with filters",
    response_description="Up to `limit` result rows with metrics, raw outputs, and optional judge scores.",
)
def query_results(
    provider:  str | None = Query(None, description="Filter by provider name (e.g. `openai`)."),
    model:     str | None = Query(None, description="Filter by model identifier."),
    task_type: str | None = Query(None, description="One of: `classification`, `qa`, `summarization`, `extraction`."),
    difficulty:str | None = Query(None, description="One of: `easy`, `medium`, `hard`."),
    domain:    str | None = Query(None, description="Domain label stored on the task."),
    run_id:    str | None = Query(None, description="UUID of a specific benchmark run."),
    limit:     int        = Query(200, le=1000, description="Maximum rows to return (hard cap 1000)."),
) -> list[dict[str, Any]]:
    """Query row-level results from the database with optional filters.

    All filter parameters are optional and combinable.
    LLM-judge dimension scores (if present) are surfaced as `judge_dimensions` and `judge_reasoning`.
    """
    with _store() as s:
        rows = s.query(
            provider=provider, model=model, task_type=task_type,
            difficulty=difficulty, domain=domain, run_id=run_id,
            dry_run=False, limit=limit,
        )
    # Parse per-dimension judge scores from scores_json and surface them
    # as a structured dict so the frontend doesn't need to parse JSON blobs.
    for row in rows:
        scores_blob = row.get("scores_json")
        if scores_blob:
            try:
                data = json.loads(scores_blob)
                dims = {
                    k: v for k, v in data.items()
                    if k.startswith("judge_") and isinstance(v, (int, float))
                }
                if dims:
                    row["judge_dimensions"] = dims
                reasoning = data.get("judge_reasoning")
                if reasoning:
                    row["judge_reasoning"] = reasoning
            except Exception:
                pass
    return rows


# ---------------------------------------------------------------------------
# LLM-as-Judge: retroactive judge runs (non-destructive history)
# ---------------------------------------------------------------------------

class RetroJudgeRequest(BaseModel):
    judge_provider:        str | None = None
    judge_model:           str
    rubric_override:       str | None = None
    mitigate_position_bias: bool = False


@app.post(
    "/api/runs/{run_id}/judge",
    tags=["judge"],
    summary="Start a retroactive judge run on an existing benchmark run",
    status_code=202,
)
def start_retro_judge(run_id: str, req: RetroJudgeRequest) -> dict[str, Any]:
    """Launch an async judge pass over all results in an existing benchmark run.

    Results are stored in ``judge_results`` — original run_results are never mutated.
    Poll ``GET /api/judge-runs/{judge_run_id}/status`` for progress.
    """
    with _store() as s:
        rows = s.get_run_results_for_judging(run_id)
    if not rows:
        raise HTTPException(status_code=404, detail="Run not found or has no scoreable results.")

    judge_run_id = str(uuid.uuid4())
    with _store() as s:
        s.create_judge_run(
            judge_run_id=judge_run_id,
            run_id=run_id,
            judge_model=req.judge_model,
            judge_provider=req.judge_provider,
            rubric_override=req.rubric_override,
            mitigate_position_bias=req.mitigate_position_bias,
            source="retroactive",
        )

    with _jobs_lock:
        _jobs[judge_run_id] = {
            "judge_run_id": judge_run_id,
            "state":        "running",
            "progress":     0,
            "total":        len(rows),
            "error":        None,
        }

    threading.Thread(
        target=_execute_retro_judge,
        args=(judge_run_id, run_id, rows, req),
        daemon=True,
    ).start()
    return {"judge_run_id": judge_run_id}


@app.get("/api/judge-runs", tags=["judge"], summary="List all judge runs")
def list_judge_runs(run_id: str | None = Query(None)) -> list[dict[str, Any]]:
    with _store() as s:
        return s.list_judge_runs(run_id=run_id)


@app.get("/api/judge-runs/{judge_run_id}/status", tags=["judge"], summary="Poll judge run progress")
def judge_run_status(judge_run_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(judge_run_id)
    if job:
        return job
    with _store() as s:
        run = s.get_judge_run(judge_run_id)
    if run:
        return {"judge_run_id": judge_run_id, "state": run["state"], **run}
    raise HTTPException(status_code=404, detail="Judge run not found")


@app.get("/api/judge-runs/{judge_run_id}/results", tags=["judge"], summary="Get per-task results of a judge run")
def judge_run_results(judge_run_id: str) -> list[dict[str, Any]]:
    with _store() as s:
        return s.get_judge_run_results(judge_run_id)


def _execute_retro_judge(
    judge_run_id: str,
    run_id: str,
    rows: list[dict],
    req: RetroJudgeRequest,
) -> None:
    import os
    from evaluators.llm_judge import LLMJudge
    from models.result import GenerationConfig, RunResult
    from models.task import ClassificationTask, ExtractionTask, QATask, SummarizationTask

    TYPE_CLS = {
        "classification": ClassificationTask,
        "qa":             QATask,
        "summarization":  SummarizationTask,
        "extraction":     ExtractionTask,
    }

    try:
        judge = LLMJudge(
            model=req.judge_model,
            api_base=os.environ.get("LITELLM_BASE_URL"),
            api_key=os.environ.get("LITELLM_API_KEY"),
            mitigate_position_bias=req.mitigate_position_bias,
        )

        task_ids = {r["task_id"] for r in rows}
        tasks_by_id = _find_tasks_by_ids(task_ids, TYPE_CLS)

        for i, row in enumerate(rows):
            task = tasks_by_id.get(row["task_id"])
            if task is None and not req.rubric_override:
                _set_job(judge_run_id, progress=i + 1)
                continue

            try:
                parsed = json.loads(row["parsed_output"]) if row.get("parsed_output") else None
            except Exception:
                parsed = row.get("parsed_output")

            result = RunResult(
                task_id=row["task_id"],
                task_type=row["task_type"],
                provider=row["provider"],
                model=row["model"],
                generation_config=GenerationConfig(
                    model=row["model"],
                    temperature=row.get("temperature", 0.0),
                    max_tokens=row.get("max_tokens", 1024),
                    top_p=row.get("top_p", 1.0),
                    seed=row.get("seed"),
                ),
                raw_output=row.get("raw_output") or "",
                parsed_output=parsed,
                parse_error=row.get("parse_error"),
                tokens_used={
                    "prompt":     row.get("prompt_tokens", 0),
                    "completion": row.get("completion_tokens", 0),
                    "total":      row.get("total_tokens", 0),
                },
                latency_ms=row.get("latency_ms", 0.0),
            )

            jr = judge.judge(result, task, rubric_override=req.rubric_override)

            with _store() as s:
                s.save_judge_result(
                    judge_run_id=judge_run_id,
                    result_id=row["id"],
                    task_id=row["task_id"],
                    llm_judge_score=jr.overall if jr else None,
                    judge_reasoning=jr.reasoning if jr else None,
                    rubric_overridden=jr.rubric_overridden if jr else False,
                    dimensions=jr.dimensions if jr else {},
                )

            _set_job(judge_run_id, progress=i + 1)

        with _store() as s:
            s.finish_judge_run(judge_run_id, state="done")
        _set_job(judge_run_id, state="done")

    except Exception as exc:
        with _store() as s:
            s.finish_judge_run(judge_run_id, state="error", error=str(exc))
        _set_job(judge_run_id, state="error", error=str(exc))


# ---------------------------------------------------------------------------
# LLM-as-Judge: rubric preview
# ---------------------------------------------------------------------------

class PreviewRequest(BaseModel):
    dataset_files:   list[str]        # one entry per selected dataset
    limit_per_file:  int = 1
    rubric_overrides: dict[str, str] = {}   # dataset_file → rubric string


@app.post(
    "/api/judge/preview",
    tags=["judge"],
    summary="Preview judge prompts (no LLM call)",
    response_description="List of rendered judge prompts — one per sampled task across all dataset files.",
)
def preview_judge(req: PreviewRequest) -> list[dict[str, Any]]:
    """Return judge prompt previews for all selected datasets without any LLM call.

    Useful for inspecting and refining rubrics before spending on a full scored run.
    """
    from evaluators.llm_judge import build_preview
    from models.task import ClassificationTask, ExtractionTask, QATask, SummarizationTask

    TYPE_CLS = {
        "classification": ClassificationTask,
        "qa":             QATask,
        "summarization":  SummarizationTask,
        "extraction":     ExtractionTask,
    }

    previews: list[dict] = []

    for dataset_file in req.dataset_files:
        path = _TASK_BANK / dataset_file
        if not path.exists():
            continue
        sampled: list = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    cls = TYPE_CLS.get(data.get("task_type", ""))
                    if cls:
                        sampled.append(cls.model_validate(data))
                except Exception:
                    continue
                if len(sampled) >= req.limit_per_file:
                    break
        rubric_override = req.rubric_overrides.get(dataset_file)
        for t in sampled:
            p = build_preview(t, rubric_override=rubric_override)
            p["dataset_file"] = dataset_file
            previews.append(p)

    return previews


# ---------------------------------------------------------------------------
# LLM-as-Judge: calibration against human scores
# ---------------------------------------------------------------------------

class HumanScore(BaseModel):
    task_id: str
    score:   float  # 0-10 scale


class CalibrateRequest(BaseModel):
    run_id:                str
    human_scores:          list[HumanScore]
    judge_provider:        str | None = None
    judge_model:           str | None = None
    rubric_override:       str | None = None
    mitigate_position_bias: bool = False


@app.post(
    "/api/judge/calibrate",
    tags=["judge"],
    summary="Calibrate judge against human scores",
    response_description="CalibrationReport with Pearson r, mean bias, and per-task deltas.",
)
def calibrate_judge(req: CalibrateRequest) -> dict[str, Any]:
    """Compare LLM judge scores against human ratings on a small held-out slice.

    If judge_model is provided the judge is re-run on those tasks.
    Otherwise existing llm_judge_score values from the run are used.

    Returns a CalibrationReport and per-task deltas.  A Pearson r < 0.7 or
    |bias| > 0.1 suggests the rubric needs refinement before scaling up.
    """
    from evaluators.calibrate_judge import calibrate, CalibrationReport
    from models.result import GenerationConfig, RunResult
    from models.task import ClassificationTask, ExtractionTask, QATask, SummarizationTask

    human_map: dict[str, float] = {
        hs.task_id: hs.score / 10.0 for hs in req.human_scores
    }
    task_ids = list(human_map.keys())

    with _store() as s:
        db_rows = {
            r["task_id"]: r
            for r in s.query(run_id=req.run_id, limit=10_000)
            if r["task_id"] in human_map
        }

    judge_scores_map: dict[str, float] = {}

    if req.judge_model:
        from evaluators.llm_judge import LLMJudge
        import os

        TYPE_CLS = {
            "classification": ClassificationTask,
            "qa":             QATask,
            "summarization":  SummarizationTask,
            "extraction":     ExtractionTask,
        }

        judge = LLMJudge(
            model=req.judge_model,
            api_base=os.environ.get("LITELLM_BASE_URL"),
            api_key=os.environ.get("LITELLM_API_KEY"),
            mitigate_position_bias=req.mitigate_position_bias,
        )

        # Find tasks from task bank to reconstruct full task objects
        tasks_by_id = _find_tasks_by_ids(set(task_ids), TYPE_CLS)

        for tid, row in db_rows.items():
            task = tasks_by_id.get(tid)
            if task is None or not row.get("parsed_output"):
                continue
            try:
                parsed = json.loads(row["parsed_output"])
            except Exception:
                parsed = row.get("parsed_output")
            result = RunResult(
                task_id=tid,
                task_type=row["task_type"],
                provider=row["provider"],
                model=row["model"],
                generation_config=GenerationConfig(
                    model=row["model"],
                    temperature=row.get("temperature", 0.0),
                    max_tokens=row.get("max_tokens", 1024),
                ),
                raw_output=row.get("raw_output") or "",
                parsed_output=parsed,
                tokens_used={"prompt": 0, "completion": 0, "total": 0},
                latency_ms=0.0,
            )
            jr = judge.judge(result, task, rubric_override=req.rubric_override)
            if jr is not None:
                judge_scores_map[tid] = jr.overall
    else:
        for tid, row in db_rows.items():
            existing = row.get("llm_judge_score")
            if existing is not None:
                judge_scores_map[tid] = float(existing)

    # Only include task_ids where we have both judge and human scores
    common = [tid for tid in task_ids if tid in judge_scores_map]
    if not common:
        raise HTTPException(
            status_code=422,
            detail="No matching scored results found. Provide a run_id with judge scores "
                   "or a judge_model to re-score."
        )

    j_scores = [judge_scores_map[tid] for tid in common]
    h_scores  = [human_map[tid] for tid in common]

    report = calibrate(j_scores, h_scores)

    per_task = [
        {
            "task_id":     tid,
            "judge_score": round(judge_scores_map[tid], 4),
            "human_score": round(human_map[tid], 4),
            "delta":       round(judge_scores_map[tid] - human_map[tid], 4),
        }
        for tid in common
    ]

    return {**report.to_dict(), "per_task": per_task}


def _find_tasks_by_ids(task_ids: set[str], type_cls: dict) -> dict:
    """Scan all task bank JSONL files for the given task_ids."""
    found: dict = {}
    for path in _TASK_BANK.glob("*.jsonl"):
        if not (task_ids - set(found)):
            break
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    tid = data.get("task_id")
                    if tid in task_ids and tid not in found:
                        cls = type_cls.get(data.get("task_type", ""))
                        if cls:
                            found[tid] = cls.model_validate(data)
                except Exception:
                    continue
    return found


# ---------------------------------------------------------------------------
# Internal: async run execution
# ---------------------------------------------------------------------------

def _public_job(job: dict) -> dict:
    """Return a copy of a job dict with internal (underscore-prefixed) fields stripped."""
    return {k: v for k, v in job.items() if not k.startswith("_")}


def _set_job(run_id: str, **kwargs: Any) -> None:
    with _jobs_lock:
        _jobs[run_id].update(kwargs)


def _execute_run(run_id: str, req: RunRequest) -> None:
    try:
        tasks_with_src = _load_tasks(req.datasets)  # list of (task, source_file)
        _set_job(run_id, state="running", total=len(tasks_with_src))

        from models.provider_config import ProviderConfig
        from models.result import Scores, ScoredResult
        from runner.engine import run_task
        from evaluators.scorer import compute_scores

        config = ProviderConfig(provider=req.provider, model=req.model)
        scored: list[ScoredResult] = []

        with _jobs_lock:
            cancel_event: threading.Event = _jobs[run_id]["_cancel"]

        judge = None
        if req.use_judge and req.judge_model and not req.dry_run:
            from evaluators.llm_judge import LLMJudge
            import os
            judge = LLMJudge(
                model=req.judge_model,
                api_base=os.environ.get("LITELLM_BASE_URL"),
                api_key=os.environ.get("LITELLM_API_KEY"),
                mitigate_position_bias=req.mitigate_position_bias,
            )

        for i, (task, source_file) in enumerate(tasks_with_src):
            if cancel_event.is_set():
                break
            result = run_task(task, config, dry_run=req.dry_run)
            scores = compute_scores(result, task)
            if judge is not None:
                rubric_override = req.rubric_overrides.get(source_file) if req.rubric_overrides else None
                jr = judge.judge(result, task, rubric_override=rubric_override)
                if jr is not None:
                    dim_extra = {f"judge_{k}": v for k, v in jr.dimensions.items()}
                    scores = scores.model_copy(update={
                        "llm_judge_score":   jr.overall,
                        "rubric_overridden": jr.rubric_overridden,
                        "judge_reasoning":   jr.reasoning or None,
                        "extra": {**scores.extra, **dim_extra},
                    })
            scored.append(ScoredResult(
                run_id=run_id,
                result=result,
                difficulty=task.difficulty,
                domain=task.domain,
                expected=task.expected,
                task_input=task.input,
                scores=scores,
                estimated_cost_usd=result.estimated_cost_usd,
            ))
            _set_job(run_id, progress=i + 1)

        with _store() as s:
            row_ids = s.save_batch(scored, run_id=run_id)
            if req.notes:
                s._conn.execute(
                    "UPDATE run_batches SET notes = ? WHERE run_id = ?",
                    (req.notes, run_id),
                )
                s._conn.commit()

            # Persist inline judge scores as a judge session so they appear
            # in judge history alongside any retroactive sessions.
            if judge is not None and not cancel_event.is_set():
                jrun_id = str(uuid.uuid4())
                now_iso = datetime.now(tz=timezone.utc).isoformat()
                s.create_judge_run(
                    judge_run_id=jrun_id,
                    run_id=run_id,
                    judge_model=req.judge_model,
                    judge_provider=req.judge_provider,
                    mitigate_position_bias=req.mitigate_position_bias,
                    created_at=now_iso,
                    source="benchmark",
                )
                for sr, row_id in zip(scored, row_ids):
                    if row_id > 0 and sr.scores.llm_judge_score is not None:
                        dims = {
                            k.replace("judge_", ""): v
                            for k, v in sr.scores.extra.items()
                            if k.startswith("judge_")
                        }
                        s.save_judge_result(
                            judge_run_id=jrun_id,
                            result_id=row_id,
                            task_id=sr.result.task_id,
                            llm_judge_score=sr.scores.llm_judge_score,
                            judge_reasoning=sr.scores.judge_reasoning,
                            rubric_overridden=bool(sr.scores.rubric_overridden),
                            dimensions=dims,
                        )
                s.finish_judge_run(jrun_id, state="done")

        if not cancel_event.is_set():
            _set_job(run_id, state="done", finished_at=datetime.now(tz=timezone.utc).isoformat())

    except Exception as exc:
        _set_job(run_id, state="error", error=str(exc),
                 finished_at=datetime.now(tz=timezone.utc).isoformat())


def _load_tasks(selections: list[DatasetSelection]) -> list[tuple]:
    """Return a list of (task, source_file) pairs."""
    from models.task import ClassificationTask, ExtractionTask, QATask, SummarizationTask

    TYPE_CLS = {
        "classification": ClassificationTask,
        "qa":             QATask,
        "summarization":  SummarizationTask,
        "extraction":     ExtractionTask,
    }

    all_tasks: list[tuple] = []
    for sel in selections:
        path = _TASK_BANK / sel.file
        if not path.exists():
            continue
        pool: list = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    cls = TYPE_CLS.get(data.get("task_type", ""))
                    if cls is None:
                        continue
                    task = cls.model_validate(data)
                    if sel.difficulty and task.difficulty != sel.difficulty:
                        continue
                    pool.append(task)
                except Exception:
                    continue
        if len(pool) > sel.limit:
            pool = random.sample(pool, sel.limit)
        all_tasks.extend((task, sel.file) for task in pool)

    if not all_tasks:
        raise ValueError(
            "No tasks loaded — check that the selected datasets exist locally "
            "and the difficulty filter matches available entries."
        )
    return all_tasks
