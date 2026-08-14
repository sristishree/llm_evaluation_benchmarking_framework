"""LLM-as-Judge evaluator — level 2.

Features
--------
- Multi-dimension scoring  (faithfulness / coverage / conciseness, task-type specific)
- Position-bias mitigation (run judge twice with expected/model swapped, then average)
- Verbosity-bias mitigation (explicit prompt instruction)
- Self-enhancement detection (handled at call site; flagged via rubric_overridden context)
- Rubric override support  (runtime dict or YAML file)
- Preview mode             (build the judge prompt without making an LLM call)

Usage
-----
    from evaluators.llm_judge import LLMJudge, build_preview

    judge = LLMJudge(model="azure/gpt-4o-prod", mitigate_position_bias=True)
    jr = judge.judge(result, task, rubric_overrides={"squad2_001": "Custom..."})
    # jr.overall, jr.dimensions, jr.reasoning, jr.rubric_overridden

    preview = build_preview(task, rubric_overrides={"squad2_001": "Custom..."})
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from evaluators.rubric import Dimension, get_dimensions, get_rubric
from models.result import RunResult
from models.task import Task

load_dotenv(Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — bias mitigation baked in
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a rigorous, impartial evaluator of LLM outputs on benchmark tasks.

SCORING PRINCIPLES
  • Score each dimension from 0 to 10 exactly as specified in the rubric.
  • Base scores only on the rubric criteria and the reference output — never on style.
  • Apply identical standards regardless of which answer appears first (position bias).
  • Do not favour longer or more verbose responses unless length is explicitly valued
    in the rubric (verbosity bias).
  • Be equally critical of all outputs regardless of fluency or writing style.
  • Do not favour outputs that resemble your own training data distribution
    (self-enhancement bias).

OUTPUT FORMAT
  Return ONLY valid JSON matching the schema shown at the end of the user message.
  Do not add prose, markdown fences, or extra keys.\
"""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class JudgeResult:
    """Structured output from a single judge evaluation."""
    overall: float                  # normalised to [0, 1]
    dimensions: dict[str, float]    # dim_key → normalised [0, 1]
    reasoning: str
    rubric_overridden: bool = False


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LLMJudge:
    """LLM-as-judge evaluator with multi-dimension scoring and bias mitigation."""

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout_seconds: int = 60,
        mitigate_position_bias: bool = False,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.mitigate_position_bias = mitigate_position_bias
        self._client = OpenAI(
            base_url=api_base or os.environ.get("LITELLM_BASE_URL"),
            api_key=api_key or os.environ.get("LITELLM_API_KEY"),
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def judge(
        self,
        result: RunResult,
        task: Task,
        rubric_override: str | None = None,
    ) -> JudgeResult | None:
        """Score the result against the task.

        Returns None on dry-run, missing parsed_output, or total LLM failure.
        """
        if result.dry_run:
            logger.debug("judge skip: task=%s reason=dry_run", result.task_id)
            return None
        if result.parsed_output is None:
            logger.warning("judge skip: task=%s reason=no_parsed_output", result.task_id)
            return None

        rubric, overridden = get_rubric(task, rubric_override)
        dimensions = get_dimensions(task.task_type)

        jr = self._call(result, task, rubric, dimensions, swapped=False)
        if jr is None:
            logger.error("judge failed: task=%s model=%s swapped=False — no result returned", result.task_id, self.model)
            return None

        if self.mitigate_position_bias:
            jr_swapped = self._call(result, task, rubric, dimensions, swapped=True)
            if jr_swapped is not None:
                jr = _average(jr, jr_swapped)

        jr.rubric_overridden = overridden
        return jr

    def preview_prompt(
        self,
        result: RunResult | None,
        task: Task,
        rubric_override: str | None = None,
    ) -> dict:
        """Build and return the judge prompt without making an LLM call."""
        return build_preview(task, rubric_override, result)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call(
        self,
        result: RunResult,
        task: Task,
        rubric: str,
        dimensions: list[Dimension],
        swapped: bool,
    ) -> JudgeResult | None:
        messages = _build_messages(result, task, rubric, dimensions, swapped)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                # max_tokens=512,
                response_format={"type": "json_object"},
                timeout=self.timeout_seconds,
            )
            raw = response.choices[0].message.content or ""
            parsed = _parse(raw, dimensions)
            if parsed is None:
                logger.error(
                    "judge parse failed: task=%s swapped=%s raw_response=%r",
                    result.task_id, swapped, raw[:500],
                )
            return parsed
        except Exception as exc:
            logger.error(
                "judge API error: task=%s model=%s swapped=%s error=%s: %s",
                result.task_id, self.model, swapped, type(exc).__name__, exc,
            )
            return None


# ---------------------------------------------------------------------------
# Standalone preview (no LLM call, no client required)
# ---------------------------------------------------------------------------

def build_preview(
    task: Task,
    rubric_override: str | None = None,
    result: RunResult | None = None,
) -> dict:
    """Return the judge prompt structure for a task without calling any LLM.

    Safe to call before a run starts — useful for the rubric preview UI.
    """
    rubric, overridden = get_rubric(task, rubric_override)
    dimensions = get_dimensions(task.task_type)
    messages = _build_messages(result, task, rubric, dimensions, swapped=False)
    return {
        "task_id":           task.task_id,
        "task_type":         task.task_type,
        "rubric":            rubric,
        "rubric_overridden": overridden,
        "dimensions": [
            {"key": d.key, "label": d.label, "prompt": d.prompt}
            for d in dimensions
        ],
        "messages": messages,
        "input_preview": task.input[:400],
    }


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _fmt(val: object, max_len: int = 1000) -> str:
    if isinstance(val, str):
        return val[:max_len]
    try:
        return json.dumps(val, ensure_ascii=False)[:max_len]
    except Exception:
        return str(val)[:max_len]


def _build_messages(
    result: RunResult | None,
    task: Task,
    rubric: str,
    dimensions: list[Dimension],
    swapped: bool,
) -> list[dict]:
    predicted = _fmt(result.parsed_output) if result else "(no output provided)"
    expected  = _fmt(task.expected)

    # Position-bias mitigation: optionally flip which answer is shown first.
    # When swapped the labels are anonymised so the judge can't tell which is
    # the "reference" — averaging the two runs cancels any order preference.
    if swapped:
        a_label, a_val = "Output A", predicted
        b_label, b_val = "Output B", expected
    else:
        a_label, a_val = "Expected output", expected
        b_label, b_val = "Model output",    predicted

    # Dimension scoring block
    dim_lines = "\n".join(f"  • {d.prompt}" for d in dimensions)
    dim_schema: dict = {d.key: "<integer 0-10>" for d in dimensions}

    schema_example = json.dumps(
        {
            "dimensions": dim_schema,
            "overall":    "<integer 0-10 — overall quality; may differ from the mean>",
            "reasoning":  "<one or two sentence explanation of the scores>",
        },
        indent=2,
    )

    user_content = (
        f"Task type: {task.task_type}\n"
        f"Evaluation rubric: {rubric}\n\n"
        f"Score each dimension independently on a 0-10 integer scale:\n{dim_lines}\n\n"
        f"Input (first 2 000 chars):\n{task.input[:2000]}\n\n"
        f"{a_label}:\n{a_val}\n\n"
        f"{b_label}:\n{b_val}\n\n"
        f"Return JSON matching exactly this schema:\n{schema_example}"
    )

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse(raw: str, dimensions: list[Dimension]) -> JudgeResult | None:
    data = _extract_json(raw)
    if data is None:
        logger.error("judge _parse: could not extract JSON from response: %r", raw[:500])
        return None

    dim_scores: dict[str, float] = {}
    dim_data = data.get("dimensions", {})
    for d in dimensions:
        v = dim_data.get(d.key)
        if v is not None:
            try:
                dim_scores[d.key] = min(1.0, max(0.0, float(v) / 10.0))
            except (TypeError, ValueError):
                pass

    raw_overall = data.get("overall")
    if raw_overall is not None:
        try:
            overall = min(1.0, max(0.0, float(raw_overall) / 10.0))
        except (TypeError, ValueError):
            overall = _mean(dim_scores.values())
    else:
        overall = _mean(dim_scores.values())

    if overall is None:
        logger.error("judge _parse: could not compute overall score; dim_scores=%s", dim_scores)
        return None

    return JudgeResult(
        overall=overall,
        dimensions=dim_scores,
        reasoning=str(data.get("reasoning", "")),
    )


def _extract_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except Exception:
        pass
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(stripped)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return None


def _mean(values) -> float | None:
    lst = [v for v in values if v is not None]
    return sum(lst) / len(lst) if lst else None


def _average(a: JudgeResult, b: JudgeResult) -> JudgeResult:
    """Average two JudgeResults — used to cancel position-order preference."""
    overall = (a.overall + b.overall) / 2
    all_keys = set(a.dimensions) | set(b.dimensions)
    dimensions: dict[str, float] = {}
    for k in all_keys:
        vals = [v for v in [a.dimensions.get(k), b.dimensions.get(k)] if v is not None]
        if vals:
            dimensions[k] = sum(vals) / len(vals)
    return JudgeResult(overall=overall, dimensions=dimensions, reasoning=a.reasoning)
