from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class GenerationConfig(BaseModel):
    model: str
    temperature: float
    max_tokens: int
    top_p: float = 1.0
    seed: int | None = None


class RunResult(BaseModel):
    """Output of a single LLM call against one task.

    parsed_output is task-type specific:
      summarization  -> str  (the summary)
      extraction     -> list[dict]  ([{"text": ..., "label": ...}, ...])
      classification -> str | list[str]  (label or labels)
      qa             -> str  (the answer)
    """
    task_id: str
    task_type: Literal["summarization", "extraction", "classification", "qa"]
    provider: str                        # e.g. "anthropic"
    model: str                           # e.g. "claude-sonnet-4-6"
    generation_config: GenerationConfig
    raw_output: str                      # verbatim text from the LLM
    parsed_output: Any | None            # structured extraction from raw_output
    parse_error: str | None = None       # set if JSON extraction failed
    tokens_used: dict[str, int]          # {"prompt": N, "completion": N, "total": N}
    latency_ms: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    dry_run: bool = False
