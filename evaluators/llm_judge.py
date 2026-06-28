from __future__ import annotations

import json
import re
from typing import Any

import litellm

from models.provider_config import ProviderConfig
from models.task import ClassificationTask, ExtractionTask, QATask, SummarizationTask


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _call_judge(prompt: str, config: ProviderConfig) -> dict[str, Any]:
    """Call the judge LLM and return a parsed JSON dict."""
    response = litellm.completion(
        model=config.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512,
        response_format={"type": "json_object"},
        timeout=config.timeout_seconds,
    )
    raw = response.choices[0].message.content or ""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Task-specific judge functions
# ---------------------------------------------------------------------------

def judge_summarization(
    task: SummarizationTask,
    output: str,
    config: ProviderConfig,
) -> tuple[dict[str, float], str]:
    """Score a summary on faithfulness, coverage, and conciseness (0–1 each)."""
    prompt = f"""You are an expert text summarization evaluator.

ORIGINAL TEXT:
{task.input}

REFERENCE SUMMARY:
{task.expected}

MODEL SUMMARY:
{output}

RUBRIC:
{task.rubric}

Score the model summary on three dimensions. Return a JSON object with exactly these keys:
- "faithfulness": float 0-1 (every claim in the summary is supported by the source text)
- "coverage": float 0-1 (summary covers the main points of the source)
- "conciseness": float 0-1 (summary is appropriately brief without padding)
- "reasoning": string (1-2 sentences justifying your scores)

Respond with ONLY the JSON object."""

    data = _call_judge(prompt, config)
    scores = {
        "judge_faithfulness": _clamp(data.get("faithfulness")),
        "judge_coverage": _clamp(data.get("coverage")),
        "judge_conciseness": _clamp(data.get("conciseness")),
    }
    return scores, str(data.get("reasoning", ""))


def judge_extraction(
    task: ExtractionTask,
    output: list[dict],
    config: ProviderConfig,
) -> tuple[dict[str, float], str]:
    """Score NER output on hallucination penalty (0–1, higher = fewer hallucinations)."""
    prompt = f"""You are an expert named entity recognition evaluator.

ORIGINAL TEXT:
{task.input}

EXPECTED ENTITIES:
{json.dumps(task.expected, ensure_ascii=False)}

MODEL ENTITIES:
{json.dumps(output, ensure_ascii=False)}

RUBRIC:
{task.rubric}

Score the model output. Return a JSON object with exactly these keys:
- "hallucination_penalty": float 0-1 (1.0 = no invented entities not present in the text, 0.0 = many hallucinated entities)
- "reasoning": string (1-2 sentences justifying your score)

Respond with ONLY the JSON object."""

    data = _call_judge(prompt, config)
    scores = {
        "judge_hallucination_penalty": _clamp(data.get("hallucination_penalty")),
    }
    return scores, str(data.get("reasoning", ""))


def judge_classification(
    task: ClassificationTask,
    output: str,
    config: ProviderConfig,
) -> tuple[dict[str, float], str]:
    """Score a classification label on correctness (0–1)."""
    prompt = f"""You are an expert text classification evaluator.

INPUT TEXT:
{task.input}

VALID LABELS: {", ".join(task.label_set)}

CORRECT LABEL: {task.expected}

MODEL LABEL: {output}

RUBRIC:
{task.rubric}

Score the model output. Return a JSON object with exactly these keys:
- "correctness": float 0-1 (1.0 = exactly correct; 0.5 = adjacent/partially correct on a continuous scale; 0.0 = clearly wrong)
- "reasoning": string (1-2 sentences justifying your score)

Respond with ONLY the JSON object."""

    data = _call_judge(prompt, config)
    scores = {
        "judge_correctness": _clamp(data.get("correctness")),
    }
    return scores, str(data.get("reasoning", ""))


def judge_qa(
    task: QATask,
    output: str,
    config: ProviderConfig,
) -> tuple[dict[str, float], str]:
    """Score a QA answer on grounding and abstention accuracy (0–1 each)."""
    expected_str = (
        task.expected
        if isinstance(task.expected, str)
        else " / ".join(task.expected)
    )
    prompt = f"""You are an expert question answering evaluator.

QUESTION AND CONTEXT:
{task.input}

EXPECTED ANSWER:
{expected_str}

MODEL ANSWER:
{output}

RUBRIC:
{task.rubric}

Score the model answer. Return a JSON object with exactly these keys:
- "grounding": float 0-1 (answer is supported by and consistent with the provided context)
- "abstention_accuracy": float 0-1 (if the question is unanswerable, did the model correctly abstain? If answerable, set 1.0)
- "reasoning": string (1-2 sentences justifying your scores)

Respond with ONLY the JSON object."""

    data = _call_judge(prompt, config)
    scores = {
        "judge_grounding": _clamp(data.get("grounding")),
        "judge_abstention_accuracy": _clamp(data.get("abstention_accuracy")),
    }
    return scores, str(data.get("reasoning", ""))
