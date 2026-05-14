"""
task_id        — unique, stable identifier (e.g. "summ_news_001")
task_type      — summarization | extraction | classification | qa
input          — the raw text the model receives
expected       — ground truth output (format varies by task type)
rubric         — scoring instructions, especially for LLM-as-judge
metadata       — source, domain, difficulty, language, date added
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union
from pydantic import BaseModel, Field


class TaskBase(BaseModel):
    """Common fields shared by every task type."""
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    domain: str
    input: str
    expected: Any          # type varies — see concrete subclasses
    rubric: str            # natural language instructions for LLM-as-judge
    metadata: dict         # source, date_added, notes


class SummarizationTask(TaskBase):
    """
    expected: a reference summary string.
    Optional constraints let task authors cap length or require bullet form.
    """
    task_type: Literal["summarization"] = "summarization"
    expected: str                          # reference summary
    max_words: int | None = None           # if set, model output must be <= this
    bullet_points: bool = False            # True → model must return bullet list


class ExtractionTask(TaskBase):
    """
    expected: list of entity dicts, each with 'text' and 'label'.
    entity_types restricts which labels are in scope for this task.

    Example expected value:
        [{"text": "London", "label": "LOC"}, {"text": "Elon Musk", "label": "PER"}]
    """
    task_type: Literal["extraction"] = "extraction"
    expected: list[dict[str, str]]         # [{"text": ..., "label": ...}, ...]
    entity_types: list[str]                # e.g. ["PER", "ORG", "LOC", "DATE"]
    overlapping_spans: bool = False        # True → entities may overlap


class ClassificationTask(TaskBase):
    """
    expected: the correct label string (or list for multi-label tasks).
    label_set is the closed set of valid outputs.

    Example expected value (single-label):  "positive"
    Example expected value (multi-label):   ["finance", "politics"]
    """
    task_type: Literal["classification"] = "classification"
    expected: str | list[str]             # single label or list for multi-label
    label_set: list[str]                  # exhaustive list of valid labels
    multi_label: bool = False             # True → model may return multiple labels


class QATask(TaskBase):
    """
    expected: the correct answer string (or list of acceptable answer strings).
    context_included signals whether the passage is embedded in `input`.
    answer_extractive=True means the answer must be a verbatim span from input.

    Example expected value:  "42"  or  ["42", "forty-two"]
    """
    task_type: Literal["qa"] = "qa"
    expected: str | list[str]            # one answer or list of acceptable answers
    context_included: bool = True        # False → open-book / retrieval task
    answer_extractive: bool = False      # True → answer must be a span of input


# Discriminated union — use this as the type annotation throughout the codebase.
Task = Annotated[
    Union[SummarizationTask, ExtractionTask, ClassificationTask, QATask],
    Field(discriminator="task_type"),
]
