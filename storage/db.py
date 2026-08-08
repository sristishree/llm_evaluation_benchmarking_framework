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
    expected           TEXT,
    task_input         TEXT,
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
    rubric_overridden  INTEGER NOT NULL DEFAULT 0,
    scores_json        TEXT,
    UNIQUE (task_id, provider, model, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_rr_provider_type ON run_results (provider, task_type);
CREATE INDEX IF NOT EXISTS idx_rr_run_id        ON run_results (run_id);
CREATE INDEX IF NOT EXISTS idx_rr_timestamp     ON run_results (timestamp);
"""

_JUDGE_DDL = """
CREATE TABLE IF NOT EXISTS judge_runs (
    judge_run_id           TEXT PRIMARY KEY,
    run_id                 TEXT NOT NULL REFERENCES run_batches(run_id) ON DELETE CASCADE,
    judge_provider         TEXT,
    judge_model            TEXT NOT NULL,
    rubric_override        TEXT,
    mitigate_position_bias INTEGER NOT NULL DEFAULT 0,
    task_count             INTEGER NOT NULL DEFAULT 0,
    created_at             TEXT NOT NULL,
    finished_at            TEXT,
    state                  TEXT NOT NULL DEFAULT 'running',
    error                  TEXT,
    source                 TEXT NOT NULL DEFAULT 'retroactive'
);

CREATE TABLE IF NOT EXISTS judge_results (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    judge_run_id      TEXT NOT NULL REFERENCES judge_runs(judge_run_id) ON DELETE CASCADE,
    result_id         INTEGER NOT NULL REFERENCES run_results(id),
    task_id           TEXT NOT NULL,
    llm_judge_score   REAL,
    judge_reasoning   TEXT,
    rubric_overridden INTEGER NOT NULL DEFAULT 0,
    scores_json       TEXT,
    created_at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_jr_run_id      ON judge_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_jres_judge_run ON judge_results(judge_run_id);
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
        self._conn.executescript(_JUDGE_DDL)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(run_results)")}
        if "expected" not in existing:
            self._conn.execute("ALTER TABLE run_results ADD COLUMN expected TEXT")
        if "rubric_overridden" not in existing:
            self._conn.execute(
                "ALTER TABLE run_results ADD COLUMN rubric_overridden INTEGER NOT NULL DEFAULT 0"
            )
        if "task_input" not in existing:
            self._conn.execute("ALTER TABLE run_results ADD COLUMN task_input TEXT")

        jr_existing = {row[1] for row in self._conn.execute("PRAGMA table_info(judge_runs)")}
        if "source" not in jr_existing:
            self._conn.execute(
                "ALTER TABLE judge_runs ADD COLUMN source TEXT NOT NULL DEFAULT 'retroactive'"
            )
            # Fix pre-existing backfill entries that were created without the source column.
            self._conn.execute(
                "UPDATE judge_runs SET source = 'backfill'"
                " WHERE judge_model = 'inline (pre-history)' AND source = 'retroactive'"
            )

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
            "expected":           json.dumps(scored.expected) if scored.expected is not None else None,
            "task_input":         scored.task_input,
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
            "rubric_overridden":  int(s.rubric_overridden),
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
            "llm_judge_score":   scores.llm_judge_score,
            "rubric_overridden": int(scores.rubric_overridden),
            "scores_json":       json.dumps(
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

    def agg_by_provider_and_type(self, exclude_dry_runs: bool = True) -> list[dict]:
        """Per-provider, per-task-type score breakdown."""
        where = "WHERE dry_run = 0" if exclude_dry_runs else ""
        rows = self._conn.execute(f"""
            SELECT
                provider, model, task_type,
                COUNT(*)                    AS n,
                AVG(rouge_1)                AS avg_rouge_1,
                AVG(rouge_2)                AS avg_rouge_2,
                AVG(rouge_l)                AS avg_rouge_l,
                AVG(bert_score)             AS avg_bert_score,
                AVG(exact_match)            AS avg_exact_match,
                AVG(token_f1)               AS avg_token_f1,
                AVG(llm_judge_score)        AS avg_llm_judge_score,
                SUM(rubric_overridden)      AS n_rubric_overridden,
                SQRT(MAX(0, AVG(rouge_l * rouge_l)       - AVG(rouge_l) * AVG(rouge_l)))           AS std_rouge_l,
                SQRT(MAX(0, AVG(bert_score * bert_score) - AVG(bert_score) * AVG(bert_score)))     AS std_bert_score,
                SQRT(MAX(0, AVG(exact_match * exact_match) - AVG(exact_match) * AVG(exact_match))) AS std_exact_match,
                SQRT(MAX(0, AVG(token_f1 * token_f1)     - AVG(token_f1) * AVG(token_f1)))         AS std_token_f1,
                COUNT(CASE WHEN parse_error IS NOT NULL THEN 1 END) * 100.0 / COUNT(*) AS parse_failure_pct,
                AVG(json_extract(scores_json, '$.entity_precision')) AS avg_entity_precision,
                AVG(json_extract(scores_json, '$.entity_recall'))    AS avg_entity_recall,
                SUM(total_tokens)           AS total_tokens,
                SUM(estimated_cost_usd)     AS total_cost_usd,
                AVG(latency_ms)             AS avg_latency_ms
            FROM run_results
            {where}
            GROUP BY provider, model, task_type
            ORDER BY provider, task_type
        """).fetchall()
        return [dict(r) for r in rows]

    def agg_by_domain(self, exclude_dry_runs: bool = True) -> list[dict]:
        """Per-domain score breakdown — powers the domain performance chart."""
        where = "WHERE dry_run = 0" if exclude_dry_runs else ""
        rows = self._conn.execute(f"""
            SELECT
                provider, model, task_type, domain,
                COUNT(*) AS n,
                AVG(COALESCE(rouge_l, token_f1, exact_match, bert_score)) AS avg_score,
                SQRT(MAX(0,
                    AVG(COALESCE(rouge_l, token_f1, exact_match, bert_score) *
                        COALESCE(rouge_l, token_f1, exact_match, bert_score))
                    - AVG(COALESCE(rouge_l, token_f1, exact_match, bert_score))
                    * AVG(COALESCE(rouge_l, token_f1, exact_match, bert_score))
                )) AS std_score
            FROM run_results
            {where}
            GROUP BY provider, model, task_type, domain
            ORDER BY domain, provider
        """).fetchall()
        return [dict(r) for r in rows]

    def agg_heatmap(self, exclude_dry_runs: bool = True) -> list[dict]:
        """Provider × difficulty matrix — powers the task-level heatmap."""
        where = "WHERE dry_run = 0" if exclude_dry_runs else ""
        rows = self._conn.execute(f"""
            SELECT
                provider, model, task_type, difficulty,
                COUNT(*) AS n,
                AVG(
                    COALESCE(rouge_l, token_f1, exact_match, bert_score, llm_judge_score)
                ) AS avg_score
            FROM run_results
            {where}
            GROUP BY provider, model, task_type, difficulty
            ORDER BY provider, task_type, difficulty
        """).fetchall()
        return [dict(r) for r in rows]

    def agg_cost_vs_quality(self, exclude_dry_runs: bool = True) -> list[dict]:
        """Per-run cost vs quality — powers the scatter plot."""
        dry_clause = "dry_run = 0 AND " if exclude_dry_runs else ""
        rows = self._conn.execute(f"""
            SELECT
                run_id, provider, model, task_type,
                SUM(estimated_cost_usd)     AS total_cost_usd,
                AVG(
                    COALESCE(rouge_l, token_f1, exact_match, bert_score, llm_judge_score)
                )                           AS avg_score,
                COUNT(*)                    AS n
            FROM run_results
            WHERE {dry_clause}estimated_cost_usd IS NOT NULL
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

    def list_runs(self) -> list[dict]:
        """Return all batch runs ordered by creation time, newest first.

        Includes an ``is_dry_run`` flag derived from the run_results rows
        (1 if every result in the run has dry_run=1, 0 otherwise).
        """
        rows = self._conn.execute("""
            SELECT
                rb.*,
                COALESCE(MAX(rr.dry_run), 0) AS is_dry_run
            FROM run_batches rb
            LEFT JOIN run_results rr ON rb.run_id = rr.run_id
            GROUP BY rb.run_id
            ORDER BY rb.created_at DESC
        """).fetchall()
        return [dict(r) for r in rows]

    # -----------------------------------------------------------------------
    # Judge runs
    # -----------------------------------------------------------------------

    def create_judge_run(
        self,
        judge_run_id: str,
        run_id: str,
        judge_model: str,
        judge_provider: str | None = None,
        rubric_override: str | None = None,
        mitigate_position_bias: bool = False,
        created_at: str | None = None,
        source: str = "retroactive",
    ) -> None:
        self._conn.execute(
            """INSERT INTO judge_runs
               (judge_run_id, run_id, judge_model, judge_provider,
                rubric_override, mitigate_position_bias, created_at, state, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?)""",
            (
                judge_run_id, run_id, judge_model, judge_provider,
                rubric_override, int(mitigate_position_bias),
                created_at or datetime.now(tz=timezone.utc).isoformat(),
                source,
            ),
        )
        self._conn.commit()

    def save_judge_result(
        self,
        judge_run_id: str,
        result_id: int,
        task_id: str,
        llm_judge_score: float | None,
        judge_reasoning: str | None,
        rubric_overridden: bool,
        dimensions: dict[str, float],
    ) -> None:
        self._conn.execute(
            """INSERT INTO judge_results
               (judge_run_id, result_id, task_id, llm_judge_score,
                judge_reasoning, rubric_overridden, scores_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                judge_run_id, result_id, task_id, llm_judge_score,
                judge_reasoning, int(rubric_overridden),
                json.dumps({f"judge_{k}": v for k, v in dimensions.items()}),
            ),
        )
        self._conn.execute(
            "UPDATE judge_runs SET task_count = task_count + 1 WHERE judge_run_id = ?",
            (judge_run_id,),
        )
        self._conn.commit()

    def finish_judge_run(
        self,
        judge_run_id: str,
        state: str,
        error: str | None = None,
    ) -> None:
        self._conn.execute(
            """UPDATE judge_runs
               SET state = ?, finished_at = ?, error = ?
               WHERE judge_run_id = ?""",
            (state, datetime.now(tz=timezone.utc).isoformat(), error, judge_run_id),
        )
        self._conn.commit()

    def list_judge_runs(self, run_id: str | None = None) -> list[dict]:
        q = """
            SELECT jr.*, rb.provider, rb.model AS bench_model
            FROM judge_runs jr
            JOIN run_batches rb ON jr.run_id = rb.run_id
        """
        params: list = []
        if run_id:
            q += " WHERE jr.run_id = ?"
            params.append(run_id)
        q += " ORDER BY jr.created_at DESC"
        return [dict(r) for r in self._conn.execute(q, params).fetchall()]

    def get_judge_run(self, judge_run_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM judge_runs WHERE judge_run_id = ?", (judge_run_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_judge_run_results(self, judge_run_id: str) -> list[dict]:
        rows = self._conn.execute(
            """SELECT jr.*, rr.task_type, rr.difficulty, rr.domain,
                      rr.task_input, rr.expected, rr.parsed_output, rr.parse_error
               FROM judge_results jr
               JOIN run_results rr ON jr.result_id = rr.id
               WHERE jr.judge_run_id = ?
               ORDER BY jr.llm_judge_score DESC NULLS LAST""",
            (judge_run_id,),
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            if d.get("scores_json"):
                try:
                    d["judge_dimensions"] = json.loads(d["scores_json"])
                except Exception:
                    pass
            results.append(d)
        return results

    def get_run_results_for_judging(self, run_id: str) -> list[dict]:
        """Return rows needed to reconstruct RunResult objects for the judge."""
        return [
            dict(r) for r in self._conn.execute(
                """SELECT id, task_id, task_type, provider, model,
                          temperature, max_tokens, top_p, seed,
                          parsed_output, raw_output, parse_error,
                          latency_ms, prompt_tokens, completion_tokens, total_tokens
                   FROM run_results
                   WHERE run_id = ? AND dry_run = 0""",
                (run_id,),
            ).fetchall()
        ]

    def delete_run(self, run_id: str) -> int:
        """Delete a run and all its results. Returns the number of result rows deleted."""
        cur = self._conn.execute("DELETE FROM run_results WHERE run_id = ?", (run_id,))
        deleted = cur.rowcount
        self._conn.execute("DELETE FROM run_batches WHERE run_id = ?", (run_id,))
        self._conn.commit()
        return deleted

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
