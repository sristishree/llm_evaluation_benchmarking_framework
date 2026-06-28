from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ScoredResult(BaseModel):
    """Scored output for a single LLM run.

    metrics is a flat dict — keys are metric names (e.g. "rouge1", "bert_f1",
    "exact_match", "judge_faithfulness") and values are floats in [0, 1].
    """
    task_id: str
    task_type: Literal["summarization", "extraction", "classification", "qa"]
    provider: str
    model: str
    metrics: dict[str, float]
    judge_reasoning: str | None = None
