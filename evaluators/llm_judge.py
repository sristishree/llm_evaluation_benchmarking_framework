"""LLM-as-Judge evaluator.

Uses a configurable judge model to score task outputs against the task rubric.
Returns a normalized float in [0, 1] (None on failure or dry-run).

Usage
-----
    from evaluators.llm_judge import LLMJudge

    judge = LLMJudge(model="azure/gpt-4o-prod")
    score = judge.judge(result, task)   # float | None
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from models.result import RunResult
from models.task import Task

load_dotenv(Path(__file__).parent.parent / ".env")

_SYSTEM_PROMPT = (
    "You are an expert evaluator assessing an LLM's output on a benchmark task.\n\n"
    "Score the model response on a scale of 0 to 10:\n"
    "  0  – Completely wrong, irrelevant, or empty\n"
    "  5  – Partially correct but with significant errors or omissions\n"
    " 10  – Perfect: fully correct and meets every requirement in the rubric\n\n"
    "Return ONLY valid JSON with exactly two fields:\n"
    '{"score": <integer 0-10>, "reasoning": "<one or two sentence explanation>"}'
)


class LLMJudge:
    """Calls an LLM to score task outputs using rubric-based evaluation."""

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout_seconds: int = 45,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self._client = OpenAI(
            base_url=api_base or os.environ.get("LITELLM_BASE_URL_REMOTE"),
            api_key=api_key or os.environ.get("LITELLM_API_KEY"),
        )

    def judge(self, result: RunResult, task: Task) -> float | None:
        """Return a normalized score in [0, 1], or None on failure / dry-run."""
        if result.parsed_output is None or result.dry_run:
            return None

        messages = _build_messages(result, task)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=256,
                response_format={"type": "json_object"},
                timeout=self.timeout_seconds,
            )
            raw = response.choices[0].message.content or ""
            return _parse_score(raw)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(val: object, max_len: int = 1000) -> str:
    if isinstance(val, str):
        return val[:max_len]
    try:
        return json.dumps(val, ensure_ascii=False)[:max_len]
    except Exception:
        return str(val)[:max_len]


def _build_messages(result: RunResult, task: Task) -> list[dict]:
    user_content = (
        f"Task type: {task.task_type}\n"
        f"Evaluation rubric: {task.rubric}\n\n"
        f"Input (truncated to 2000 chars):\n{task.input[:2000]}\n\n"
        f"Expected output:\n{_fmt(task.expected)}\n\n"
        f"Model output:\n{_fmt(result.parsed_output)}\n\n"
        "Score the model output according to the rubric. "
        'Return JSON: {"score": <0-10>, "reasoning": "<brief explanation>"}'
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _parse_score(raw: str) -> float | None:
    try:
        data = json.loads(raw)
        score = data.get("score")
        if score is not None:
            return min(1.0, max(0.0, float(score) / 10.0))
    except Exception:
        pass

    match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', raw)
    if match:
        return min(1.0, max(0.0, float(match.group(1)) / 10.0))

    return None
