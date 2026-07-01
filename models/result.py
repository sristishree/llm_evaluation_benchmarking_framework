from __future__ import annotations

import uuid
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


class Scores(BaseModel):
    """Metric scores produced by the scoring layer (Phase 4).

    All fields are optional — a result can be saved before scoring and
    updated later without re-running inference.
    """
    rouge_1: float | None = None
    rouge_2: float | None = None
    rouge_l: float | None = None
    bert_score: float | None = None
    exact_match: float | None = None
    token_f1: float | None = None
    llm_judge_score: float | None = None
    extra: dict[str, float] = Field(default_factory=dict)


class ScoredResult(BaseModel):
    """A RunResult enriched with task metadata and scores for storage.

    run_id groups results that belong to the same benchmark batch run.
    difficulty and domain are denormalized from the Task so that the
    results store is self-contained and queries don't need a task join.
    """
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    result: RunResult
    difficulty: Literal["easy", "medium", "hard"]
    domain: str
    scores: Scores = Field(default_factory=Scores)
    estimated_cost_usd: float | None = None
