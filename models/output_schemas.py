"""Pydantic output models for each task type.

These serve two purposes:
1. Build the `json_schema` response_format sent to the LLM — providers that
   support it (OpenAI native, Anthropic via tool use, Gemini via response_schema)
   are constrained to return exactly this structure.
2. Validate and deserialise the raw JSON string returned by the model before it
   is stored in RunResult.parsed_output.
"""
from __future__ import annotations

from pydantic import BaseModel

from models.task import Task


class SummarizationOutput(BaseModel):
    summary: str


class EntitySpan(BaseModel):
    text: str
    label: str


class ExtractionOutput(BaseModel):
    entities: list[EntitySpan]


class ClassificationOutput(BaseModel):
    label: str


class QAOutput(BaseModel):
    answer: str


def output_schema_for(task: Task) -> type[BaseModel]:
    """Return the Pydantic output class that matches this task."""
    if task.task_type == "summarization":
        return SummarizationOutput
    if task.task_type == "extraction":
        return ExtractionOutput
    if task.task_type == "classification":
        return ClassificationOutput
    if task.task_type == "qa":
        return QAOutput
    raise ValueError(f"No output schema for task type: {task.task_type!r}")
