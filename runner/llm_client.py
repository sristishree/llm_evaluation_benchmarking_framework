from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Any

import litellm
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from models.result import GenerationConfig, RunResult
from models.task import Task
from runner.prompt_builder import build_messages

# Errors worth retrying — rate limits and temporary outages only.
# Auth errors, bad requests, and timeouts are NOT retried.
_RETRYABLE = (litellm.RateLimitError, litellm.ServiceUnavailableError)

_DRY_RUN_OUTPUTS: dict[str, Any] = {
    "summarization": "Dry-run placeholder summary.",
    "extraction": [{"text": "PlaceholderEntity", "label": "ORG"}],
    "classification": "placeholder_label",
    "qa": "Dry-run placeholder answer.",
}


class LLMClient:
    """LiteLLM-backed client for a single provider.

    Handles JSON-mode requests, retries on transient errors, and
    structured output extraction per task type.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        timeout_seconds: int = 60,
        max_retries: int = 3,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def call(self, task: Task, dry_run: bool = False) -> RunResult:
        """Run the task and return a RunResult.

        In dry-run mode no API call is made; a placeholder result is returned
        so callers can verify prompt construction and downstream logic cheaply.
        """
        gen_config = GenerationConfig(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )

        if dry_run:
            return RunResult(
                task_id=task.task_id,
                task_type=task.task_type,
                provider=self.provider,
                model=self.model,
                generation_config=gen_config,
                raw_output="[DRY RUN]",
                parsed_output=_DRY_RUN_OUTPUTS.get(task.task_type),
                tokens_used={"prompt": 0, "completion": 0, "total": 0},
                latency_ms=0.0,
                timestamp=datetime.now(tz=timezone.utc),
                dry_run=True,
            )

        messages = build_messages(task)
        t0 = time.monotonic()
        response = self._call_with_retry(messages)
        latency_ms = round((time.monotonic() - t0) * 1000, 2)

        raw = response.choices[0].message.content or ""
        parsed, parse_error = _parse_output(task.task_type, raw)

        usage = response.usage
        tokens = {
            "prompt": getattr(usage, "prompt_tokens", 0) or 0,
            "completion": getattr(usage, "completion_tokens", 0) or 0,
            "total": getattr(usage, "total_tokens", 0) or 0,
        }

        return RunResult(
            task_id=task.task_id,
            task_type=task.task_type,
            provider=self.provider,
            model=self.model,
            generation_config=gen_config,
            raw_output=raw,
            parsed_output=parsed,
            parse_error=parse_error,
            tokens_used=tokens,
            latency_ms=latency_ms,
            timestamp=datetime.now(tz=timezone.utc),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_with_retry(self, messages: list[dict]) -> Any:
        for attempt in Retrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(_RETRYABLE),
            reraise=True,
        ):
            with attempt:
                return litellm.completion(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    response_format={"type": "json_object"},
                    timeout=self.timeout_seconds,
                )


# ------------------------------------------------------------------
# Module-level helpers (reusable by tests / other modules)
# ------------------------------------------------------------------

def _extract_json(text: str) -> tuple[dict | None, str | None]:
    """Extract the first JSON object from model output.

    Tries three strategies in order:
    1. Direct json.loads on the full text.
    2. Strip markdown code fences and retry.
    3. Regex-extract the first {...} block.

    Returns (parsed_dict, error_message). If all strategies fail,
    parsed_dict is None and error_message describes the failure.
    """
    # 1. Direct parse
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fences
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(stripped), None
    except json.JSONDecodeError:
        pass

    # 3. Find first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0)), None
        except json.JSONDecodeError:
            pass

    return None, f"Cannot parse JSON from output: {text[:300]!r}"


def _parse_output(task_type: str, raw: str) -> tuple[Any, str | None]:
    """Extract the task-relevant value from a JSON response."""
    data, err = _extract_json(raw)
    if err:
        return None, err

    try:
        if task_type == "summarization":
            return data.get("summary"), None
        if task_type == "extraction":
            return data.get("entities", []), None
        if task_type == "classification":
            # model may return "label" (single) or "labels" (multi)
            label = data.get("label") or data.get("labels")
            return label, None
        if task_type == "qa":
            return data.get("answer"), None
        return data, None
    except Exception as exc:
        return data, str(exc)
