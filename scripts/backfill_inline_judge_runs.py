"""Backfill judge_runs / judge_results for benchmark runs that had LLM-as-judge
enabled inline but were executed before this history was tracked separately.

Detection: runs where run_results.llm_judge_score IS NOT NULL but there is no
judge_runs entry with source IN ('benchmark', 'backfill').  This correctly
handles the case where a retroactive judge session has already been run on the
same results — those produce source='retroactive' entries, which do NOT satisfy
the check, so the inline scores are still captured as a separate backfill entry.

Idempotent: running the script a second time finds zero uncovered runs.

Usage:
    python scripts/backfill_inline_judge_runs.py
    python scripts/backfill_inline_judge_runs.py --db path/to/results.db [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = Path(__file__).parent.parent / "results.db"
INLINE_JUDGE_MODEL = "inline (pre-history)"


def find_runs_needing_backfill(conn: sqlite3.Connection) -> list[dict]:
    """Return one row per run_id that has inline judge scores but no
    benchmark/backfill judge_run entry yet."""
    rows = conn.execute(
        """
        SELECT DISTINCT
            rr.run_id,
            rb.provider,
            rb.created_at AS run_created_at
        FROM run_results rr
        JOIN run_batches rb ON rr.run_id = rb.run_id
        WHERE rr.llm_judge_score IS NOT NULL
          AND rr.dry_run = 0
          AND NOT EXISTS (
              SELECT 1 FROM judge_runs jr
              WHERE jr.run_id = rr.run_id
                AND jr.source IN ('benchmark', 'backfill')
          )
        ORDER BY rb.created_at ASC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def get_judged_results(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    """Return all run_results with llm_judge_score set for a given run."""
    rows = conn.execute(
        """
        SELECT id AS result_id, task_id, llm_judge_score, rubric_overridden, scores_json
        FROM run_results
        WHERE run_id = ? AND llm_judge_score IS NOT NULL AND dry_run = 0
        ORDER BY id
        """,
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def parse_dims_and_reasoning(scores_json: str | None) -> tuple[dict, str | None]:
    if not scores_json:
        return {}, None
    try:
        data = json.loads(scores_json)
    except (json.JSONDecodeError, TypeError):
        return {}, None
    dims: dict[str, float] = {}
    for k, v in data.items():
        if k.startswith("judge_") and k != "judge_reasoning" and isinstance(v, (int, float)):
            dims[k.replace("judge_", "")] = float(v)
    reasoning = data.get("judge_reasoning")
    return dims, reasoning if isinstance(reasoning, str) else None


def create_judge_run(
    conn: sqlite3.Connection,
    judge_run_id: str,
    run_id: str,
    provider: str,
    created_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO judge_runs
            (judge_run_id, run_id, judge_provider, judge_model,
             mitigate_position_bias, task_count, created_at, finished_at, state, source)
        VALUES (?, ?, ?, ?, 0, 0, ?, ?, 'done', 'backfill')
        """,
        (
            judge_run_id,
            run_id,
            provider,
            INLINE_JUDGE_MODEL,
            created_at,
            created_at,
        ),
    )


def create_judge_result(
    conn: sqlite3.Connection,
    judge_run_id: str,
    row: dict,
    dims: dict[str, float],
    reasoning: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO judge_results
            (judge_run_id, result_id, task_id, llm_judge_score,
             judge_reasoning, rubric_overridden, scores_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            judge_run_id,
            row["result_id"],
            row["task_id"],
            row["llm_judge_score"],
            reasoning,
            row["rubric_overridden"] or 0,
            json.dumps({f"judge_{k}": v for k, v in dims.items()}),
            datetime.now(tz=timezone.utc).isoformat(),
        ),
    )
    conn.execute(
        "UPDATE judge_runs SET task_count = task_count + 1 WHERE judge_run_id = ?",
        (judge_run_id,),
    )


def backfill(db_path: Path, dry_run: bool = False) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Apply source column migration before querying
    jr_cols = {row[1] for row in conn.execute("PRAGMA table_info(judge_runs)")}
    if "source" not in jr_cols:
        conn.execute(
            "ALTER TABLE judge_runs ADD COLUMN source TEXT NOT NULL DEFAULT 'retroactive'"
        )
        conn.execute(
            "UPDATE judge_runs SET source = 'backfill'"
            " WHERE judge_model = 'inline (pre-history)' AND source = 'retroactive'"
        )
        conn.commit()
        print("Applied migration: added source column, fixed existing backfill entries.")
    else:
        # Even if column exists, fix any old entries that slipped through as 'retroactive'
        fixed = conn.execute(
            "UPDATE judge_runs SET source = 'backfill'"
            " WHERE judge_model = 'inline (pre-history)' AND source = 'retroactive'"
        ).rowcount
        if fixed:
            conn.commit()
            print(f"Fixed {fixed} existing backfill entry/entries to source='backfill'.")

    runs = find_runs_needing_backfill(conn)
    if not runs:
        print("Nothing to backfill — all inline judge scores are already tracked.")
        conn.close()
        return

    print(f"Found {len(runs)} run(s) needing backfill.")
    print()

    for run_meta in runs:
        run_id = run_meta["run_id"]
        provider = run_meta["provider"]
        run_created_at = run_meta["run_created_at"]

        results = get_judged_results(conn, run_id)
        judge_run_id = str(uuid.uuid4())

        print(f"  Run {run_id[:12]}…  provider={provider}  results={len(results)}")
        print(f"    → judge_run_id {judge_run_id[:12]}…  model='{INLINE_JUDGE_MODEL}'")

        for row in results:
            dims, reasoning = parse_dims_and_reasoning(row["scores_json"])
            print(
                f"      task {row['task_id'][:20]}…  "
                f"score={row['llm_judge_score']:.2f}  dims={list(dims)}"
            )

        if dry_run:
            print("    [DRY RUN — no changes written]")
            print()
            continue

        create_judge_run(conn, judge_run_id, run_id, provider, run_created_at)
        for row in results:
            dims, reasoning = parse_dims_and_reasoning(row["scores_json"])
            create_judge_result(conn, judge_run_id, row, dims, reasoning)

        conn.commit()
        print(f"    Committed {len(results)} judge_result(s).")
        print()

    conn.close()
    print("Done." if not dry_run else "Dry-run complete — rerun without --dry-run to apply.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to results.db")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without writing anything.",
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"DB not found: {args.db}")
        return

    print(f"DB: {args.db}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()
    backfill(args.db, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
