"""Dry-run smoke test for the prompt runner and scoring pipelines.

Loads one task from each available task type, builds the prompt,
returns a placeholder RunResult (no API calls), then scores it with
synthetic outputs to verify the full scorer pipeline end-to-end.

BERTScore is disabled (run_bert=False) and the LLM judge is skipped
(no judge_config) so this script runs offline with no model downloads.

Run from the project root:
    python dry_run.py
"""
import sys
from pathlib import Path
from pydantic import TypeAdapter

sys.path.insert(0, str(Path(__file__).parent))

from models.task import Task
from models.provider_config import ProviderConfig
from models.output_schemas import output_schema_for
from models.result import RunResult
from runner.prompt_builder import build_messages
from runner.engine import run_task
from evaluators import Scorer

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
            text = next((b["text"] for b in content if b.get("type") == "text"), "")
            print(f"  [{role}] {text[:200]}...")
            print(f"  [{role}] + {sum(1 for b in content if b['type'] == 'image_url')} image block(s)")
        else:
            preview = content[:300].replace("\n", " ")
            print(f"  [{role}] {preview}{'...' if len(content) > 300 else ''}")


def _synthetic_output(task: Task) -> str | list[dict]:
    """Return a plausible synthetic output for scoring demos (correct answer)."""
    if task.task_type == "classification":
        expected = task.expected
        return expected if isinstance(expected, str) else expected[0]
    if task.task_type == "qa":
        expected = task.expected
        return expected if isinstance(expected, str) else expected[0]
    if task.task_type == "summarization":
        return task.expected
    if task.task_type == "extraction":
        return task.expected
    return ""


def _patch_result(result: RunResult, output) -> RunResult:
    """Return a copy of result with parsed_output replaced by a synthetic value."""
    return result.model_copy(update={"parsed_output": output})


def main() -> None:
    cfg = ProviderConfig(provider="anthropic", model="claude-sonnet-4-6")
    scorer = Scorer(judge_config=None, run_bert=False)

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

        # Score with synthetic correct output to verify the scoring pipeline
        synthetic = _synthetic_output(task)
        scored = scorer.score(task, _patch_result(result, synthetic))
        print(f"\nScoredResult (synthetic correct output, no judge):")
        print(f"  task_id        : {scored.task_id}")
        for metric, value in scored.metrics.items():
            print(f"  {metric:<20}: {value}")
        if scored.judge_reasoning:
            print(f"  judge_reasoning: {scored.judge_reasoning}")

    print(f"\n{'='*60}")
    print("Dry run complete — no API calls were made.")


if __name__ == "__main__":
    main()
