"""
Results Store — Phase 5.

Persists ScoredResults to SQLite keyed by (task_id, provider, model, timestamp).
Scores are nullable so results can be saved pre-scoring and updated later
without re-running inference.

Usage
-----
    from storage.db import ResultsStore
    from models.result import ScoredResult, Scores

    with ResultsStore("results.db") as store:
        row_id = store.save(scored_result)
        store.update_scores(row_id, new_scores)
        rows   = store.query(provider="anthropic", task_type="summarization")
        stats  = store.get_run_summary(run_id)
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models.result import RunResult, Scores, ScoredResult


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS run_batches (
    run_id     TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    provider   TEXT NOT NULL,
    model      TEXT NOT NULL,
    task_count INTEGER NOT NULL DEFAULT 0,
    notes      TEXT
);

CREATE TABLE IF NOT EXISTS run_results (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id             TEXT    NOT NULL REFERENCES run_batches(run_id),
    task_id            TEXT    NOT NULL,
    task_type          TEXT    NOT NULL,
    difficulty         TEXT    NOT NULL,
    domain             TEXT    NOT NULL,
    provider           TEXT    NOT NULL,
    model              TEXT    NOT NULL,
    temperature        REAL    NOT NULL,
    max_tokens         INTEGER NOT NULL,
    top_p              REAL    NOT NULL DEFAULT 1.0,
    seed               INTEGER,
    raw_output         TEXT,
    parsed_output      TEXT,
    parse_error        TEXT,
    prompt_tokens      INTEGER NOT NULL DEFAULT 0,
    completion_tokens  INTEGER NOT NULL DEFAULT 0,
    total_tokens       INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd REAL,
    latency_ms         REAL    NOT NULL,
    timestamp          TEXT    NOT NULL,
    dry_run            INTEGER NOT NULL DEFAULT 0,
    rouge_1            REAL,
    rouge_2            REAL,
    rouge_l            REAL,
    bert_score         REAL,
    exact_match        REAL,
    token_f1           REAL,
    llm_judge_score    REAL,
    scores_json        TEXT,
    UNIQUE (task_id, provider, model, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_rr_provider_type ON run_results (provider, task_type);
CREATE INDEX IF NOT EXISTS idx_rr_run_id        ON run_results (run_id);
CREATE INDEX IF NOT EXISTS idx_rr_timestamp     ON run_results (timestamp);
"""


# ---------------------------------------------------------------------------
# ResultsStore
# ---------------------------------------------------------------------------

class ResultsStore:
    def __init__(self, db_path: str | Path = "results.db") -> None:
        self._path = Path(db_path)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()

    # -----------------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------------

    def save(self, scored: ScoredResult) -> int:
        """Persist one ScoredResult. Returns the row id (0 if duplicate)."""
        r = scored.result
        s = scored.scores

        self._upsert_batch(scored.run_id, r.provider, r.model)

        row = {
            "run_id":             scored.run_id,
            "task_id":            r.task_id,
            "task_type":          r.task_type,
            "difficulty":         scored.difficulty,
            "domain":             scored.domain,
            "provider":           r.provider,
            "model":              r.model,
            "temperature":        r.generation_config.temperature,
            "max_tokens":         r.generation_config.max_tokens,
            "top_p":              r.generation_config.top_p,
            "seed":               r.generation_config.seed,
            "raw_output":         r.raw_output,
            "parsed_output":      json.dumps(r.parsed_output) if r.parsed_output is not None else None,
            "parse_error":        r.parse_error,
            "prompt_tokens":      r.tokens_used.get("prompt", 0),
            "completion_tokens":  r.tokens_used.get("completion", 0),
            "total_tokens":       r.tokens_used.get("total", 0),
            "estimated_cost_usd": scored.estimated_cost_usd,
            "latency_ms":         r.latency_ms,
            "timestamp":          r.timestamp.isoformat(),
            "dry_run":            int(r.dry_run),
            "rouge_1":            s.rouge_1,
            "rouge_2":            s.rouge_2,
            "rouge_l":            s.rouge_l,
            "bert_score":         s.bert_score,
            "exact_match":        s.exact_match,
            "token_f1":           s.token_f1,
            "llm_judge_score":    s.llm_judge_score,
            "scores_json":        json.dumps(
                {**s.model_dump(exclude={"extra"}), **s.extra}
            ),
        }

        cur = self._conn.execute(
            f"INSERT OR IGNORE INTO run_results ({', '.join(row)}) "
            f"VALUES ({', '.join(':' + k for k in row)})",
            row,
        )
        self._conn.commit()
        # rowcount is 0 when INSERT OR IGNORE silently skips a duplicate
        return cur.lastrowid if cur.rowcount > 0 else 0

    def save_batch(
        self,
        results: list[ScoredResult],
        run_id: str | None = None,
    ) -> list[int]:
        """Persist a list of ScoredResults under a shared run_id.

        Generates a new run_id if one is not supplied. Non-mutating —
        returns a new copy of each result with the run_id patched in.
        """
        rid = run_id or str(uuid.uuid4())
        ids = [self.save(r.model_copy(update={"run_id": rid})) for r in results]
        self._update_task_count(rid, sum(1 for i in ids if i > 0))
        return ids

    def update_scores(
        self,
        row_id: int,
        scores: Scores,
        estimated_cost_usd: float | None = None,
    ) -> None:
        """Overwrite score columns on an existing row (re-score without re-run)."""
        updates: dict[str, Any] = {
            "rouge_1":         scores.rouge_1,
            "rouge_2":         scores.rouge_2,
            "rouge_l":         scores.rouge_l,
            "bert_score":      scores.bert_score,
            "exact_match":     scores.exact_match,
            "token_f1":        scores.token_f1,
            "llm_judge_score": scores.llm_judge_score,
            "scores_json":     json.dumps(
                {**scores.model_dump(exclude={"extra"}), **scores.extra}
            ),
        }
        if estimated_cost_usd is not None:
            updates["estimated_cost_usd"] = estimated_cost_usd

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        self._conn.execute(
            f"UPDATE run_results SET {set_clause} WHERE id = ?",
            [*updates.values(), row_id],
        )
        self._conn.commit()

    # -----------------------------------------------------------------------
    # Read — row-level
    # -----------------------------------------------------------------------

    def query(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        task_type: str | None = None,
        difficulty: str | None = None,
        domain: str | None = None,
        run_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        dry_run: bool | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Return matching rows as plain dicts, newest first."""
        conditions: list[str] = []
        params: list[Any] = []

        def _add(col: str, val: Any) -> None:
            if val is not None:
                conditions.append(f"{col} = ?")
                params.append(val)

        _add("provider",   provider)
        _add("model",      model)
        _add("task_type",  task_type)
        _add("difficulty", difficulty)
        _add("domain",     domain)
        _add("run_id",     run_id)

        if since:
            conditions.append("timestamp >= ?"); params.append(since.isoformat())
        if until:
            conditions.append("timestamp <= ?"); params.append(until.isoformat())
        if dry_run is not None:
            conditions.append("dry_run = ?"); params.append(int(dry_run))

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_sql = f"LIMIT {int(limit)}" if limit else ""

        rows = self._conn.execute(
            f"SELECT * FROM run_results {where} ORDER BY timestamp DESC {limit_sql}",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    # -----------------------------------------------------------------------
    # Read — aggregates (used by the dashboard)
    # -----------------------------------------------------------------------

    def agg_by_provider_and_type(self) -> list[dict]:
        """Per-provider, per-task-type score breakdown (excludes dry runs)."""
        rows = self._conn.execute("""
            SELECT
                provider, model, task_type,
                COUNT(*)                    AS n,
                AVG(rouge_l)                AS avg_rouge_l,
                AVG(bert_score)             AS avg_bert_score,
                AVG(exact_match)            AS avg_exact_match,
                AVG(llm_judge_score)        AS avg_llm_judge,
                SUM(total_tokens)           AS total_tokens,
                SUM(estimated_cost_usd)     AS total_cost_usd,
                AVG(latency_ms)             AS avg_latency_ms
            FROM run_results
            WHERE dry_run = 0
            GROUP BY provider, model, task_type
            ORDER BY provider, task_type
        """).fetchall()
        return [dict(r) for r in rows]

    def agg_heatmap(self) -> list[dict]:
        """Provider × difficulty matrix — powers the task-level heatmap."""
        rows = self._conn.execute("""
            SELECT
                provider, model, task_type, difficulty,
                COUNT(*) AS n,
                AVG(
                    COALESCE(rouge_l, bert_score, exact_match, llm_judge_score)
                ) AS avg_score
            FROM run_results
            WHERE dry_run = 0
            GROUP BY provider, model, task_type, difficulty
            ORDER BY provider, task_type, difficulty
        """).fetchall()
        return [dict(r) for r in rows]

    def agg_cost_vs_quality(self) -> list[dict]:
        """Per-run cost vs quality — powers the scatter plot."""
        rows = self._conn.execute("""
            SELECT
                run_id, provider, model, task_type,
                SUM(estimated_cost_usd)     AS total_cost_usd,
                AVG(
                    COALESCE(rouge_l, bert_score, exact_match, llm_judge_score)
                )                           AS avg_score,
                COUNT(*)                    AS n
            FROM run_results
            WHERE dry_run = 0 AND estimated_cost_usd IS NOT NULL
            GROUP BY run_id, provider, model, task_type
            ORDER BY total_cost_usd
        """).fetchall()
        return [dict(r) for r in rows]

    def get_run_summary(self, run_id: str) -> dict:
        """Aggregate stats for one batch run — used in run history view."""
        row = self._conn.execute("""
            SELECT
                run_id,
                COUNT(*)                    AS task_count,
                SUM(total_tokens)           AS total_tokens,
                SUM(estimated_cost_usd)     AS total_cost_usd,
                AVG(latency_ms)             AS avg_latency_ms,
                AVG(rouge_l)                AS avg_rouge_l,
                AVG(bert_score)             AS avg_bert_score,
                AVG(exact_match)            AS avg_exact_match,
                AVG(llm_judge_score)        AS avg_llm_judge,
                MIN(timestamp)              AS started_at,
                MAX(timestamp)              AS finished_at
            FROM run_results
            WHERE run_id = ?
            GROUP BY run_id
        """, (run_id,)).fetchone()
        return dict(row) if row else {}

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------

    def export_jsonl(self, path: str | Path, **filters) -> int:
        """Write filtered results to a JSONL file. Returns the row count."""
        rows = self.query(**filters)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, default=str) + "\n")
        return len(rows)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _upsert_batch(self, run_id: str, provider: str, model: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO run_batches (run_id, created_at, provider, model) "
            "VALUES (?, ?, ?, ?)",
            (run_id, datetime.now(tz=timezone.utc).isoformat(), provider, model),
        )
        self._conn.commit()

    def _update_task_count(self, run_id: str, count: int) -> None:
        self._conn.execute(
            "UPDATE run_batches SET task_count = task_count + ? WHERE run_id = ?",
            (count, run_id),
        )
        self._conn.commit()

    # -----------------------------------------------------------------------
    # Context manager
    # -----------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ResultsStore:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
