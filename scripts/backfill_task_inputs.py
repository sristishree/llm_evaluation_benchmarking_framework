"""One-time backfill: populate task_input for existing run_results rows.

Reads every JSONL file in task_bank/, builds a task_id → input map,
then updates rows where task_input IS NULL.

Usage:
    python backfill_task_inputs.py
    python backfill_task_inputs.py --db path/to/results.db
"""
import argparse
import json
import sqlite3
from pathlib import Path

TASK_BANK = Path(__file__).parent / "task_bank"
DEFAULT_DB = Path(__file__).parent / "results.db"


def build_input_map(task_bank: Path) -> dict[str, str]:
    task_inputs: dict[str, str] = {}
    for jsonl in task_bank.glob("*.jsonl"):
        with jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    task_id = obj.get("task_id")
                    inp = obj.get("input")
                    if task_id and inp is not None:
                        task_inputs[task_id] = inp
                except json.JSONDecodeError:
                    continue
    return task_inputs


def backfill(db_path: Path, task_inputs: dict[str, str]) -> int:
    conn = sqlite3.connect(str(db_path))
    updated = 0
    try:
        for task_id, inp in task_inputs.items():
            cur = conn.execute(
                "UPDATE run_results SET task_input = ? WHERE task_id = ? AND task_input IS NULL",
                (inp, task_id),
            )
            updated += cur.rowcount
        conn.commit()
    finally:
        conn.close()
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"DB not found: {args.db}")
        return

    print(f"Reading task bank from {TASK_BANK} ...")
    task_inputs = build_input_map(TASK_BANK)
    print(f"  {len(task_inputs)} tasks loaded")

    print(f"Updating {args.db} ...")
    n = backfill(args.db, task_inputs)
    print(f"  {n} rows updated")


if __name__ == "__main__":
    main()
