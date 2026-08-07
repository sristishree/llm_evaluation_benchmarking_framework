from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Any

import litellm
import openai
from openai import OpenAI

# Pre-warm litellm's cost map once at import time so per-task completion_cost()
# calls don't each trigger a remote fetch (which shows up as 20 proxy hits for 20 tasks).
try:
    litellm.get_model_cost_map()
except Exception:
    pass
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from models.output_schemas import output_schema_for
from models.result import GenerationConfig, RunResult
from models.task import Task
from runner.prompt_builder import build_messages

# Rate-limit and transient server errors are worth retrying.
# Auth errors, bad requests, and timeouts are not.
_RETRYABLE = (openai.RateLimitError, openai.APIStatusError)

_DRY_RUN_OUTPUTS: dict[str, Any] = {
    "summarization": "Dry-run placeholder summary.",
    "extraction": [{"text": "PlaceholderEntity", "label": "ORG"}],
    "classification": "placeholder_label",
    "qa": "Dry-run placeholder answer.",
}


class LLMClient:
    """OpenAI-client-backed LLM client.

    Uses the openai SDK directly against either the official endpoints or a
    LiteLLM proxy (which speaks the OpenAI protocol). Pass api_base and
    api_key to point at a proxy; omit them to use the standard endpoint
    resolved from environment variables (OPENAI_API_KEY, etc.).

    structured_output=True  → response_format=json_schema
    structured_output=False → response_format=json_object + regex extraction
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        timeout_seconds: int = 60,
        max_retries: int = 3,
        structured_output: bool = True,
        vision: bool = True,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.structured_output = structured_output
        self.vision = vision

        self._client = OpenAI(
            base_url=api_base,
            api_key=api_key
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def call(self, task: Task, dry_run: bool = False) -> RunResult:
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

        schema_cls = output_schema_for(task)
        messages = build_messages(task, vision=self.vision)

        t0 = time.monotonic()
        response = self._call_with_retry(messages, schema_cls)
        latency_ms = round((time.monotonic() - t0) * 1000, 2)

        raw = response.choices[0].message.content or ""
        parsed, parse_error = _parse_output(task.task_type, raw, schema_cls)

        usage = response.usage
        tokens = {
            "prompt": getattr(usage, "prompt_tokens", 0) or 0,
            "completion": getattr(usage, "completion_tokens", 0) or 0,
            "total": getattr(usage, "total_tokens", 0) or 0,
        }

        cost = None
        if hasattr(response, "_hidden_params"):
            cost = response._hidden_params.get("response_cost")
        if cost is None:
            try:
                cost = litellm.completion_cost(completion_response=response)
            except Exception:
                cost = None

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
            estimated_cost_usd=cost,
            timestamp=datetime.now(tz=timezone.utc),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_response_format(self, schema_cls: type) -> dict:
        if self.structured_output:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_cls.__name__,
                    "schema": schema_cls.model_json_schema(),
                },
            }
        return {"type": "json_object"}

    def _call_with_retry(self, messages: list[dict], schema_cls: type) -> Any:
        response_format = self._build_response_format(schema_cls)

        for attempt in Retrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(_RETRYABLE),
            reraise=True,
        ):
            with attempt:
                return self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    response_format=response_format,
                    timeout=self.timeout_seconds,
                )


# ------------------------------------------------------------------
# Module-level helpers (reusable by tests / other modules)
# ------------------------------------------------------------------

def _extract_json(text: str) -> tuple[dict | None, str | None]:
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass

    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(stripped), None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0)), None
        except json.JSONDecodeError:
            pass

    return None, f"Cannot parse JSON from output: {text[:300]!r}"


def _parse_output(
    task_type: str, raw: str, schema_cls: type
) -> tuple[Any, str | None]:
    data, err = _extract_json(raw)
    if err:
        return None, err

    try:
        validated = schema_cls(**data)
    except Exception as exc:
        return _best_effort_extract(task_type, data), f"Schema validation failed: {exc}"

    if task_type == "summarization":
        return validated.summary, None
    if task_type == "extraction":
        return [e.model_dump() for e in validated.entities], None
    if task_type == "classification":
        return validated.label, None
    if task_type == "qa":
        return validated.answer, None
    return data, None


def _best_effort_extract(task_type: str, data: dict) -> Any:
    if task_type == "summarization":
        return data.get("summary")
    if task_type == "extraction":
        return data.get("entities", [])
    if task_type == "classification":
        return data.get("label")
    if task_type == "qa":
        return data.get("answer")
    return data
