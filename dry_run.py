"""Dry-run smoke test for the prompt runner pipeline.

Loads one task from each available task type, builds the prompt,
and returns a placeholder RunResult — no API calls made.

Run from the project root:
    python dry_run.py
"""
import json
import sys
from pathlib import Path
from pydantic import TypeAdapter

sys.path.insert(0, str(Path(__file__).parent))

from models.task import Task
from models.provider_config import ProviderConfig
from models.output_schemas import output_schema_for
from runner.prompt_builder import build_messages
from runner.engine import run_task

TASK_BANK = Path("task_bank")

SAMPLES = {
    "classification": TASK_BANK / "clf_sst5_tasks.jsonl",
    "qa":             TASK_BANK / "qa_squad2_tasks.jsonl",
}

adapter = TypeAdapter(Task)


def _load_first(path: Path) -> Task:
    with path.open() as f:
        return adapter.validate_json(f.readline())


def _print_messages(messages: list[dict]) -> None:
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        if isinstance(content, list):
            # multimodal — show text block only
            text = next((b["text"] for b in content if b.get("type") == "text"), "")
            print(f"  [{role}] {text[:200]}...")
            print(f"  [{role}] + {sum(1 for b in content if b['type'] == 'image_url')} image block(s)")
        else:
            preview = content[:300].replace("\n", " ")
            print(f"  [{role}] {preview}{'...' if len(content) > 300 else ''}")


def main() -> None:
    cfg = ProviderConfig(provider="anthropic", model="claude-sonnet-4-6")

    for task_type, path in SAMPLES.items():
        if not path.exists():
            print(f"\n[{task_type}] SKIP — {path.name} not found in task_bank/\n")
            continue

        task = _load_first(path)
        messages = build_messages(task, vision=True)
        schema_cls = output_schema_for(task)
        result = run_task(task, cfg, dry_run=True)

        print(f"\n{'='*60}")
        print(f"Task type : {task.task_type}")
        print(f"Task ID   : {task.task_id}")
        print(f"Difficulty: {task.difficulty}  |  Domain: {task.domain}")
        print(f"Expected  : {str(task.expected)[:80]}")
        print(f"Schema    : {schema_cls.__name__}")
        print(f"\nPrompt ({len(messages)} messages):")
        _print_messages(messages)
        print(f"\nRunResult (dry run):")
        print(f"  provider       : {result.provider}")
        print(f"  model          : {result.model}")
        print(f"  parsed_output  : {result.parsed_output}")
        print(f"  tokens_used    : {result.tokens_used}")
        print(f"  latency_ms     : {result.latency_ms}")
        print(f"  dry_run        : {result.dry_run}")

    print(f"\n{'='*60}")
    print("Dry run complete — no API calls were made.")


if __name__ == "__main__":
    main()
